#!/usr/bin/env python3
"""The R-NaD learner's CPU-only checks (learn/rmlearn/rnad.py, rnad_learner.py, rnad_data.py;
docs/rnad_training_plan.md §4 L4). The GPU is hidden from this process.

    learn/.venv/bin/python learn/check_rnad.py                      # the self-contained checks
    learn/.venv/bin/python learn/check_rnad.py --tables DIR --episodes DIR [--omni-episodes DIR]

Checks:
- **oracle**: v-trace (both players, the reward transform), _has_played, _policy_ratio,
  _player_others, the critic loss, NeuRD (and its gradient against finite differences), the legal
  policies and the entropy schedule against a numpy transcription of OpenSpiel's rnad.py
  (8156646cdef0c753162fc6b68a11d57cce642f95) on random batches with irregular seat sequences,
  padded steps and variable legal masks; v-trace scanned in chunks equals the whole scan;
- **composition**: the learner's loss on a fabricated batch through a real ``RMNet`` (recorded and
  recomputed regularisation log π, smoothing with a sideboarding ε, α mixing, fine-tuning on)
  against the oracle's reference loss; chunk-length invariance of the update;
- **games**: R-NaD through a small ``RMNet`` on fabricated tokens: sequential rock-paper-scissors
  with hidden information (from a skewed start, toward the uniform equilibrium) and a two-step game
  with a dominant action (learnt);
- **bookkeeping**: iteration boundaries (α, the regularisation policies, versions), the server
  pushes, the stale-regularisation drop / recompute counts, missing μ refused, a resume from
  ``state_dict`` bit-identical to an uninterrupted run, BC seeding;
- **projection**: ``threshold_policy`` / ``discretize_policy`` against the engine's
  ``src/agents/policy_projection.cpp`` (compiled into a small harness with ``g++``) on random and
  known inputs, and against the reference's plain-ceiling rounding;
- **real data** (with ``--tables`` / ``--episodes``): episodes read with numpy and with
  ``python/rmtraj.py`` agree; learner steps with the default model on CPU are finite (and the
  omniscient critic with ``--omni-episodes``).

Prints ``RNAD-CHECK: ok`` and exits 0, or the failures and exits 1.
"""

import os

os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import copy  # noqa: E402
import glob  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from rmlearn import model as M  # noqa: E402
from rmlearn import rnad as R  # noqa: E402
from rmlearn import rnad_data as RD  # noqa: E402
from rmlearn import rnad_learner as RL  # noqa: E402
from rmlearn import spec as S  # noqa: E402
from rmlearn.batch import Decisions  # noqa: E402

FAILURES = []


def ok(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print("FAIL: " + msg, flush=True)
    return cond


def info(msg):
    print("  " + msg, flush=True)


# ── the numpy oracle: rnad.py transcribed (jnp → np, lax.scan → a reverse loop) ─────────────────

def o_legal_policy(logits, legal):
    l_min = logits.min(axis=-1, keepdims=True)
    logits = np.where(legal, logits, l_min)
    logits = logits - logits.max(axis=-1, keepdims=True)
    logits = logits * legal
    exp_logits = np.where(legal, np.exp(logits), 0)
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def o_legal_log_policy(logits, legal):
    with np.errstate(divide="ignore"):
        logits_masked = logits + np.log(legal)
    max_legal_logit = logits_masked.max(axis=-1, keepdims=True)
    logits_masked = logits_masked - max_legal_logit
    baseline = np.log(np.exp(logits_masked).sum(axis=-1, keepdims=True))
    return legal * (logits - max_legal_logit - baseline)


def o_player_others(player_ids, valid, player):
    res = 2 * (player_ids == player).astype(np.int32) - 1
    return np.expand_dims(res * valid, -1)


def o_policy_ratio(pi, mu, actions_oh, valid):
    def sel(p):
        return (actions_oh * p).sum(-1) * valid + (1 - valid)
    return sel(pi) / sel(mu)


def o_has_played(valid, player_id, player):
    carry = np.zeros_like(player_id[-1], dtype=np.float64)
    out = np.zeros(valid.shape)
    for t in range(valid.shape[0] - 1, -1, -1):
        our_res, opp_res, reset_res = np.ones_like(carry), carry, np.zeros_like(carry)
        our_carry, opp_carry, reset_carry = carry, carry, np.zeros_like(carry)
        ours = player_id[t] == player
        res = np.where(valid[t] > 0, np.where(ours, our_res, opp_res), reset_res)
        carry = np.where(valid[t] > 0, np.where(ours, our_carry, opp_carry), reset_carry)
        out[t] = res
    return out


def o_v_trace(v, valid, player_id, acting_policy, merged_policy, merged_log_policy, player_others, actions_oh,
              reward, player, eta, lambda_, c, rho):
    gamma = 1.0
    has_played = o_has_played(valid, player_id, player)
    policy_ratio = o_policy_ratio(merged_policy, acting_policy, actions_oh, valid)
    inv_mu = o_policy_ratio(np.ones_like(merged_policy), acting_policy, actions_oh, valid)
    eta_reg_entropy = -eta * (merged_policy * merged_log_policy).sum(-1) * np.squeeze(player_others, -1)
    eta_log_policy = -eta * merged_log_policy * player_others
    b = v.shape[1]
    init = dict(reward=np.zeros(b), reward_uncorrected=np.zeros(b), next_value=np.zeros((b, 1)),
                next_v_target=np.zeros((b, 1)), importance_sampling=np.ones(b))
    carry = dict(init)
    v_target = np.zeros_like(v)
    learning_output = np.zeros_like(merged_policy)
    for t in range(v.shape[0] - 1, -1, -1):
        cs, pid, vv, rr, ere, val, im, aoh, elp = (policy_ratio[t], player_id[t], v[t], reward[t],
                                                   eta_reg_entropy[t], valid[t], inv_mu[t], actions_oh[t],
                                                   eta_log_policy[t])
        reward_uncorrected = rr + gamma * carry["reward_uncorrected"] + ere
        discounted_reward = rr + gamma * carry["reward"]
        our_v_target = (vv + np.expand_dims(np.minimum(rho, cs * carry["importance_sampling"]), -1) *
                        (np.expand_dims(reward_uncorrected, -1) + gamma * carry["next_value"] - vv) +
                        lambda_ * np.expand_dims(np.minimum(c, cs * carry["importance_sampling"]), -1) * gamma *
                        (carry["next_v_target"] - carry["next_value"]))
        our_learning_output = (vv + elp + aoh * np.expand_dims(im, -1) *
                               (np.expand_dims(discounted_reward, -1) + gamma *
                                np.expand_dims(carry["importance_sampling"], -1) * carry["next_v_target"] - vv))
        our_carry = dict(reward=np.zeros_like(carry["reward"]), next_value=vv, next_v_target=our_v_target,
                         reward_uncorrected=np.zeros_like(carry["reward_uncorrected"]),
                         importance_sampling=np.ones_like(carry["importance_sampling"]))
        opp_carry = dict(reward=ere + cs * discounted_reward, reward_uncorrected=reward_uncorrected,
                         next_value=gamma * carry["next_value"], next_v_target=gamma * carry["next_v_target"],
                         importance_sampling=cs * carry["importance_sampling"])
        ours = pid == player
        vl = val > 0

        def w(pred, a, b):
            p = pred.reshape(pred.shape + (1,) * (a.ndim - pred.ndim))
            return np.where(p, a, b)

        v_target[t] = w(vl, w(ours, our_v_target, np.zeros_like(our_v_target)), np.zeros_like(our_v_target))
        learning_output[t] = w(vl, w(ours, our_learning_output, np.zeros_like(our_learning_output)),
                               np.zeros_like(our_learning_output))
        carry = {k: w(vl, w(ours, our_carry[k], opp_carry[k]), init[k]) for k in init}
    return v_target, has_played, learning_output


def o_get_loss_v(v_list, v_target_list, mask_list):
    total = 0.0
    for v_n, v_target, mask in zip(v_list, v_target_list, mask_list):
        loss_v = np.expand_dims(mask, -1) * (v_n - v_target) ** 2
        n = mask.sum()
        total += loss_v.sum() / (n + (n == 0.0))
    return total


def o_apply_force(decision_outputs, force, threshold, threshold_center):
    can_decrease = decision_outputs - threshold_center > -threshold
    can_increase = decision_outputs - threshold_center < threshold
    clipped = can_decrease * np.minimum(force, 0.0) + can_increase * np.maximum(force, 0.0)
    return decision_outputs * clipped


def o_renormalize(loss, mask):
    n = mask.sum()
    return (loss * mask).sum() / (n + (n == 0.0))


def o_get_loss_nerd(logit_list, policy_list, q_vr_list, valid, player_ids, legal, is_list, clip, threshold):
    total = 0.0
    num_valid_actions = legal.sum(-1, keepdims=True)
    for k, (logit_pi, pi, q_vr, is_c) in enumerate(zip(logit_list, policy_list, q_vr_list, is_list)):
        adv_pi = q_vr - (pi * q_vr).sum(-1, keepdims=True)
        adv_pi = np.clip(is_c * adv_pi, -clip, clip)
        mean_logit = (logit_pi * legal).sum(-1, keepdims=True) / num_valid_actions
        logits = logit_pi - mean_logit
        nerd = (legal * o_apply_force(logits, adv_pi, threshold, np.zeros_like(logits))).sum(-1)
        total += -o_renormalize(nerd, valid * (player_ids == k))
    return total


class OEntropySchedule:
    def __init__(self, sizes, repeats):
        schedule = [0]
        for size, repeat in zip(sizes, repeats):
            schedule.extend([schedule[-1] + (i + 1) * size for i in range(repeat)])
        self.schedule = np.array(schedule, dtype=np.int32)

    def __call__(self, learner_step):
        s = self.schedule
        last_size = s[-1] - s[-2]
        last_start = s[-1] + (learner_step - s[-1]) // last_size * last_size
        start = np.amax(s * (s <= learner_step))
        finish = np.amin(s * (learner_step < s), initial=s[-1], where=(learner_step < s))
        size = finish - start
        beyond = int(s[-1] <= learner_step)
        iteration_start = last_start * beyond + start * (1 - beyond)
        iteration_size = last_size * beyond + size * (1 - beyond)
        update = learner_step > 0 and np.sum(learner_step == iteration_start + iteration_size - 1) > 0
        alpha = min((2.0 * (learner_step - iteration_start)) / iteration_size, 1.0)
        return alpha, bool(update)


def o_threshold(policy, mask, thr):
    if thr <= 0:
        return policy
    m = mask * ((policy >= thr) + (policy.max(-1, keepdims=True) < thr))
    return m * policy / (m * policy).sum(-1, keepdims=True)


def o_discretize_single(mu, n, slack=0.0):
    """The reference's _discretize_single (slack 0); slack 1e-9 is the engine's rounding."""
    roundup = np.ceil(mu * n - slack).astype(np.int64)
    result = np.zeros_like(mu)
    order = np.argsort(-mu, kind="stable")
    weight_left = n
    for i in range(len(mu)):
        x = min(roundup[order[i]], weight_left)
        if weight_left >= 0:
            result[order[i]] += x
        weight_left -= x
    if weight_left > 0:
        result[order[0]] += weight_left
    return result / n


def o_smooth(lp, legal, sb, eps, eps_sb):
    e = np.where(sb, eps_sb, eps)[..., None]
    n = legal.sum(-1, keepdims=True)
    with np.errstate(divide="ignore"):
        out = np.log((1 - e) * np.exp(lp) + e / n)
    return np.where(legal, out, 0.0)


def o_rnad_loss(logits, v_cur, v_tgt, lp_prev, lp_prev_, alpha, valid, pid, legal, mu, aoh, rewards, eta, c, rho,
                beta, clip, finetune=None):
    """The reference's RNaDSolver.loss on given network outputs (float64)."""
    pi = o_legal_policy(logits, legal)
    log_pi = o_legal_log_policy(logits, legal)
    pp = pi
    if finetune is not None:
        thr, n = finetune
        pp = o_threshold(pi, legal, thr)
        pp = np.stack([np.stack([o_discretize_single(pp[t, b], n, R.CEIL_SLACK) for b in range(pp.shape[1])])
                       for t in range(pp.shape[0])])
    log_policy_reg = log_pi - (alpha * lp_prev + (1 - alpha) * lp_prev_)
    vts, hps, qs = [], [], []
    for p in range(2):
        vt, hp, q = o_v_trace(v_tgt, valid, pid, mu, pp, log_policy_reg, o_player_others(pid, valid, p), aoh,
                              rewards[:, :, p], p, eta, 1.0, c, rho)
        vts.append(vt)
        hps.append(hp)
        qs.append(q)
    loss_v = o_get_loss_v([v_cur] * 2, vts, hps)
    ones = np.ones(valid.shape + (1,))
    loss_nerd = o_get_loss_nerd([logits] * 2, [pi] * 2, qs, valid, pid, legal, [ones] * 2, clip, beta)
    return loss_v, loss_nerd


# ── random batches ───────────────────────────────────────────────────────────────────────────

def random_batch(rng, T=13, B=6, A=7):
    """Games of varied length (padding after), irregular seats, variable menus, μ over each menu,
    a sampled action, the result at each game's last step."""
    lengths = rng.randint(1, T + 1, size=B)
    lengths[0] = T
    valid = (np.arange(T)[:, None] < lengths[None, :]).astype(np.float64)
    pid = rng.randint(0, 2, size=(T, B))
    menu = rng.randint(1, A + 1, size=(T, B))
    legal = (np.arange(A)[None, None, :] < menu[:, :, None]).astype(np.float64)
    perm = rng.rand(T, B, A).argsort(-1)            # menus are any subset of rows, not only a prefix
    legal = np.take_along_axis(legal, perm, -1)
    legal[valid == 0] = 0
    legal[valid == 0, 0] = 1                        # the dummy row of a padded step

    def dist():
        x = rng.rand(T, B, A) * legal
        x[x > 0] += 0.05
        return x / x.sum(-1, keepdims=True)

    mu = dist()
    merged = dist()
    lp = np.where(legal > 0, np.log(dist() + 1e-300), 0.0)
    act = np.array([[rng.choice(A, p=mu[t, b]) for b in range(B)] for t in range(T)])
    aoh = np.eye(A)[act]
    rewards = np.zeros((T, B, 2))
    res = rng.choice([-1.0, 0.0, 1.0], size=B)
    rewards[lengths - 1, np.arange(B), 0] = res
    rewards[lengths - 1, np.arange(B), 1] = -res
    v = rng.randn(T, B, 1)
    logits = rng.randn(T, B, A) * 3
    return dict(valid=valid, pid=pid, legal=legal, mu=mu, merged=merged, lp=lp, aoh=aoh, rewards=rewards, v=v,
                logits=logits)


def t64(a):
    return torch.from_numpy(np.asarray(a, np.float64))


def check_oracle():
    print("oracle: rnad.py's functions against their numpy transcription", flush=True)
    rng = np.random.RandomState(7)
    worst = {"v_target": 0.0, "learning_output": 0.0, "has_played": 0.0, "loss_v": 0.0, "loss_nerd": 0.0,
             "chunked": 0.0, "policies": 0.0, "f32": 0.0}
    for trial in range(40):
        bt = random_batch(rng)
        valid, pid, legal = bt["valid"], bt["pid"], bt["legal"]
        rho = [np.inf, 1.0, 0.7][trial % 3]
        c = [1.0, 1.0, 0.5][trial % 3]
        vts, hps, qs = [], [], []
        for p in range(2):
            o = o_v_trace(bt["v"], valid, pid, bt["mu"], bt["merged"], bt["lp"], o_player_others(pid, valid, p),
                          bt["aoh"], bt["rewards"][:, :, p], p, 0.2, 1.0, c, rho)
            tv, tb, tl = t64(valid).bool(), torch.from_numpy(pid), t64(legal)
            args = (t64(bt["v"]), tv, tb, t64(bt["mu"]), t64(bt["merged"]), t64(bt["lp"]),
                    R._player_others(tb, tv, p).double(), t64(bt["aoh"]), t64(bt["rewards"][:, :, p]), p)
            vt, hp, lo, _ = R.v_trace(*args, eta=0.2, lambda_=1.0, c=c, rho=rho)
            worst["v_target"] = max(worst["v_target"], float(np.abs(vt.numpy() - o[0]).max()))
            worst["learning_output"] = max(worst["learning_output"], float(np.abs(lo.numpy() - o[2]).max()))
            worst["has_played"] = max(worst["has_played"], float(np.abs(hp.numpy() - o[1]).max()),
                                      float(np.abs(R.has_played_scan(tv, tb, p).numpy() - o[1]).max()))
            # The same scan in chunks (last first, the carry passed back).
            cuts = sorted(set([0, valid.shape[0]] + list(rng.randint(1, valid.shape[0], size=3))))
            carry = None
            vt_c = torch.zeros_like(vt)
            lo_c = torch.zeros_like(lo)
            for a, b in reversed(list(zip(cuts[:-1], cuts[1:]))):
                sl = slice(a, b)
                vt_i, _, lo_i, carry = R.v_trace(*[x[sl] if torch.is_tensor(x) else x for x in args[:-1]], p,
                                                 eta=0.2, lambda_=1.0, c=c, rho=rho, carry=carry)
                vt_c[sl], lo_c[sl] = vt_i, lo_i
            worst["chunked"] = max(worst["chunked"], float((vt_c - vt).abs().max()), float((lo_c - lo).abs().max()))
            # float32 inputs, as the learner runs it.
            args32 = [x.float() if torch.is_tensor(x) and x.dtype == torch.float64 else x for x in args]
            vt32, _, lo32, _ = R.v_trace(*args32, eta=0.2, lambda_=1.0, c=c, rho=rho)
            scale = 1.0 + float(np.abs(o[2]).max())
            worst["f32"] = max(worst["f32"], float(np.abs(lo32.double().numpy() - o[2]).max()) / scale)
            vts.append(o[0])
            hps.append(o[1])
            qs.append(o[2])
        lv = o_get_loss_v([bt["v"] + 0.3] * 2, vts, hps)
        tlv = R.get_loss_v([t64(bt["v"] + 0.3)] * 2, [t64(x) for x in vts], [t64(x) for x in hps])
        n = [t64(hps[p]).sum() for p in range(2)]
        tlv2 = R.get_loss_v([t64(bt["v"] + 0.3)] * 2, [t64(x) for x in vts], [t64(x) for x in hps], normalization=n)
        worst["loss_v"] = max(worst["loss_v"], abs(float(tlv) - lv), abs(float(tlv2) - lv))
        pi = o_legal_policy(bt["logits"], legal)
        ones = np.ones(valid.shape + (1,))
        clip = [10_000, 0.5][trial % 2]
        ln = o_get_loss_nerd([bt["logits"]] * 2, [pi] * 2, qs, valid, pid, legal, [ones] * 2, clip, 2.0)
        tln = R.get_loss_nerd([t64(bt["logits"])] * 2, [t64(pi)] * 2, [t64(q) for q in qs], t64(valid).bool(),
                              torch.from_numpy(pid), t64(legal).bool(), [t64(ones)] * 2, clip=clip, threshold=2.0)
        worst["loss_nerd"] = max(worst["loss_nerd"], abs(float(tln) - ln))
        tp = R.legal_policy(t64(bt["logits"]), t64(legal).bool()).numpy()
        tlp = R.legal_log_policy(t64(bt["logits"]), t64(legal).bool()).numpy()
        worst["policies"] = max(worst["policies"], float(np.abs(tp - pi).max()),
                                float(np.abs(tlp - o_legal_log_policy(bt["logits"], legal)).max()))
    for k, wv in worst.items():
        tol = 1e-5 if k == "f32" else 1e-10
        ok(wv <= tol, "oracle: %s differs by %.3g" % (k, wv))
    info("40 random batches (T 13, B 6, A 7; irregular seats, padding, menus any subset of rows; ρ̄ ∞ / 1 / 0.7, "
         "c̄ 1 / 0.5): worst differences " + ", ".join("%s %.1e" % kv for kv in worst.items()))
    # NeuRD's gradient wrt the logits against central differences of the oracle (pi, q fixed).
    bt = random_batch(np.random.RandomState(3), T=5, B=3, A=4)
    valid, pid, legal = bt["valid"], bt["pid"], bt["legal"]
    pi = o_legal_policy(bt["logits"], legal)
    q = [bt["merged"] * 3 - 1, bt["mu"] * 2]
    ones = np.ones(valid.shape + (1,))
    logits = bt["logits"] * 0.4                                  # within the threshold, away from its edges
    x = t64(logits).requires_grad_(True)
    loss = R.get_loss_nerd([x] * 2, [t64(pi)] * 2, [t64(a) for a in q], t64(valid).bool(), torch.from_numpy(pid),
                           t64(legal).bool(), [t64(ones)] * 2, clip=10_000, threshold=2.0)
    loss.backward()
    fd = np.zeros_like(logits)
    h = 1e-6
    for i in np.ndindex(logits.shape):
        e = np.zeros_like(logits)
        e[i] = h
        f = lambda z: o_get_loss_nerd([z] * 2, [pi] * 2, q, valid, pid, legal, [ones] * 2, 10_000, 2.0)  # noqa
        fd[i] = (f(logits + e) - f(logits - e)) / (2 * h)
    gd = float(np.abs(x.grad.numpy() - fd).max())
    ok(gd < 1e-6, "oracle: NeuRD gradient differs from finite differences by %.3g" % gd)
    info("NeuRD's gradient wrt the logits = central differences of the oracle (max diff %.1e)" % gd)
    # The schedule.
    bad = 0
    for sizes, repeats in (([3, 5, 10], [2, 4, 1]), ([1, 4], [3, 1]), ([7], [1]), ([2, 3], [1, 1])):
        ref, mine = OEntropySchedule(sizes, repeats), R.EntropySchedule(sizes=sizes, repeats=repeats)
        for step in range(0, 120):
            a, b = ref(step), mine(step)
            bad += abs(a[0] - b[0]) > 1e-12 or a[1] != b[1]
    ok(bad == 0, "oracle: the entropy schedule differs at %d steps" % bad)
    info("EntropySchedule (α, update) = the reference's at steps 0..119 of four schedules")


# ── the projection against the engine ────────────────────────────────────────────────────────

HARNESS = r"""
#include <cstdio>
#include <vector>
#include "policy_projection.h"
int main() {
    double thr; int n, k;
    while (std::scanf("%lf %d %d", &thr, &n, &k) == 3) {
        std::vector<double> mu(k);
        for (double &x : mu) if (std::scanf("%lf", &x) != 1) return 2;
        agents::threshold_policy(&mu, thr);
        agents::discretize_policy(&mu, n);
        for (size_t i = 0; i < mu.size(); ++i) std::printf("%s%.17g", i ? " " : "", mu[i]);
        std::printf("\n");
    }
    return 0;
}
"""


def projection_cases(rng):
    cases = [([0.5, 0.3, 0.15, 0.05], 0.1, 16), ([0.3, 0.3, 0.4], 0.5, 0), ([0.25, 0.25, 0.25, 0.25], 0, 3),
             ([0.25, 0.75], 0.03, 16), ([0.2, 0.45, 0.35], 0, 1), ([0.3, 0.7], 0, 10), ([0.1] * 10, 0, 10),
             ([0.7, 0.2, 0.1], 0.03, 10), ([1.0], 0.03, 16), ([0.0, 1.0, 0.0], 0.03, 16)]
    for _ in range(3000):
        rows = 1 + rng.randint(12)
        logits = rng.rand(rows) * 10 - 5
        rep = rng.rand(rows) < 0.25
        logits[rep] = logits[0]
        p = np.exp(logits - logits.max())
        p = p / p.sum()
        cases.append((list(p), [0, 0.03, 0.2][rng.randint(3)], [0, 1, 10, 16, 32][rng.randint(5)]))
    return cases


def check_projection(work):
    print("projection: rnad.FineTuning against src/agents/policy_projection.cpp", flush=True)
    rng = np.random.RandomState(20261019)
    cases = projection_cases(rng)
    # Ours, batched per (thresh, disc) over padded rows.
    ours = []
    for mu, thr, n in cases:
        a = torch.tensor([mu + [0.0] * 3], dtype=torch.float64)
        mask = torch.tensor([[True] * len(mu) + [False] * 3])
        t = R.threshold_policy(a, mask, thr)
        d = R.discretize_policy(t, n)
        ours.append(d[0, :len(mu)].numpy())
    # The reference's plain ceiling agrees except where p·n is an integer up to float noise.
    differ = 0
    for (mu, thr, n), o in zip(cases, ours):
        if n > 0:
            t = o_threshold(np.array(mu), np.ones(len(mu)), thr)
            r = o_discretize_single(t, n)
            differ += not np.array_equal(r, o)
    info("%d distributions; the reference's plain ceil(p·n) rounding differs on %d (p·n an integer up to "
         "float noise, e.g. 0.3·10: the engine's 1e-9 slack)" % (len(cases), differ))
    gxx = shutil.which("g++")
    if not gxx:
        ok(False, "projection: no g++ to build the engine harness")
        return
    src_dir = os.path.join(REPO_ROOT, "src", "agents")
    h = os.path.join(work, "harness.cpp")
    exe = os.path.join(work, "projection_harness")
    with open(h, "w") as fp:
        fp.write(HARNESS)
    r = subprocess.run([gxx, "-O2", "-std=c++17", "-I", src_dir, h, os.path.join(src_dir, "policy_projection.cpp"),
                        "-o", exe], capture_output=True, text=True)
    if not ok(r.returncode == 0, "projection: harness build failed: " + r.stderr[-400:]):
        return
    lines = ["%r %d %d %s" % (float(thr), n, len(mu), " ".join(repr(float(x)) for x in mu)) for mu, thr, n in cases]
    r = subprocess.run([exe], input="\n".join(lines) + "\n", capture_output=True, text=True)
    out = [np.array([float(x) for x in ln.split()]) for ln in r.stdout.strip().split("\n")]
    if not ok(len(out) == len(cases), "projection: the harness answered %d of %d" % (len(out), len(cases))):
        return
    exact, worst = 0, 0.0
    for (mu, thr, n), o, c in zip(cases, ours, out):
        if n > 0:
            exact += np.array_equal(o, c)
        else:
            worst = max(worst, float(np.abs(o - c).max()))
    n_disc = sum(1 for _, _, n in cases if n > 0)
    ok(exact == n_disc, "projection: %d of %d discretised distributions differ from the engine's"
       % (n_disc - exact, n_disc))
    ok(worst <= 1e-15, "projection: thresholded distributions differ from the engine's by %.3g" % worst)
    known = [np.array([9, 6, 1, 0]) / 16, None, np.array([1, 1, 1, 0]) / 3, np.array([0.25, 0.75]),
             np.array([0, 1, 0]), np.array([0.3, 0.7])]
    kn = all(np.allclose(ours[i], k, atol=1e-12) for i, k in enumerate(known) if k is not None)
    ok(kn, "projection: a known answer of tools/robomage_check/projection.cpp differs")
    info("engine harness: %d discretised distributions bit-identical, %d thresholded within %.1e; the actor "
         "tier's known answers hold" % (exact, len(cases) - n_disc, worst))


# ── fabricated tables, tokens and games through a real RMNet ─────────────────────────────────

def fake_tables(n_cards=12, seed=0):
    doc = {"vocab_hash": "00000000000000aa", "F": 2, "C": 2, "K": 1, "Q": 1, "categorical_size": 24,
           "token_types": [{"name": "player", "id": 0, "numeric": [{"name": "x"}, {"name": "y"}]},
                           {"name": "menu", "id": 1, "numeric": [{"name": "z"}]}],
           "relation_types": [["menu", "player", "of"]], "menu": "menu", "library_depth": 2,
           "metagame": {"meta_list": 3, "meta_archetype": 2}, "card_tables": {"card": {"rows": n_cards + 1}}}
    spec = S.Spec(json.dumps(doc).encode())
    rng = np.random.RandomState(seed)
    f32 = lambda a: np.ascontiguousarray(a, np.float32)  # noqa: E731
    card = rng.randn(n_cards + 1, 4)
    card[-1] = 0
    text = rng.randn(n_cards + 1, 8)
    text[-1] = 0
    hashes = {"spec_hash": spec.hash, "vocab_hash": doc["vocab_hash"], "text_model": "fake@0"}
    return S.Tables(spec, f32(card), f32(text), f32(rng.randn(5, 3)), f32(rng.randn(6, 2)),
                    np.array([0, 0, 1, 2, 3, 4], np.int64), hashes)


def step_tokens(obs, menu, rng=None):
    """One decision's tokens: a player token (categorical `obs`), then the menu rows' tokens
    (categorical 8 + row), each related to the player token."""
    n = 1 + menu
    tok_type = np.array([0] + [1] * menu, np.int64)
    tok_num = np.zeros((n, 2), np.float32)
    tok_num[0] = [1.0, 0.5]
    tok_num[1:, 0] = np.arange(menu) / 4.0
    tok_cat = np.zeros((n, 2), np.int64)
    tok_cat[0, 0] = obs
    tok_cat[1:, 0] = 8 + np.arange(menu)
    if rng is not None:
        tok_cat[1:, 1] = rng.randint(1, 24, size=menu)
    tok_card = np.full((n, 1), -1, np.int64)
    tok_card[0, 0] = obs % 12
    return {"tok_type": tok_type, "tok_num": tok_num, "tok_cat": tok_cat, "tok_card": tok_card,
            "tok_ability": np.full((n, 1), -1, np.int64), "tok_mask": np.ones(n, np.uint8),
            "rel": np.array([[i, 0, 0] for i in range(1, n)], np.int64).reshape(-1, 3),
            "act_tok": np.arange(1, n, dtype=np.int64), "act_mask": np.ones(menu, np.uint8)}


def make_decisions(steps):
    arrays = {k: np.concatenate([s[k] for s in steps]) for k in RD.TENSOR_NAMES}
    sizes = {"n": np.array([len(s["tok_type"]) for s in steps], np.int64),
             "r": np.array([len(s["rel"]) for s in steps], np.int64),
             "a": np.array([len(s["act_tok"]) for s in steps], np.int64)}
    return Decisions(arrays, sizes)


def make_episode(spec_hash, seats, tokens, mus, chosen, result, sideboarding=None, evals=None, omni_tokens=None,
                 versions=None):
    """An rnad_data.Episode from per-step lists. `evals`: per step a list of (policy id, version, log π)."""
    d = len(seats)
    menus = np.array([len(m) for m in mus], np.int64)
    menu_off = RD._starts(menus)
    p = max((len(e) for e in evals), default=0) if evals else 0
    ev_id = np.full((d, p), -1, np.int64)
    ev_ver = np.full((d, p), -1, np.int64)
    ev_lp = np.zeros((int(menu_off[-1]), p), np.float32)
    for i in range(d):
        for k, (pid, ver, lp) in enumerate(evals[i] if evals else []):
            ev_id[i, k], ev_ver[i, k] = pid, ver
            ev_lp[menu_off[i]:menu_off[i + 1], k] = lp
    fields = {"seat": np.array(seats, np.int64), "game": np.zeros(d, np.int64), "decision": np.arange(1, d + 1),
              "sideboarding": np.array(sideboarding if sideboarding is not None else [False] * d, np.bool_),
              "chosen": np.array(chosen, np.int64), "has_mu": np.ones(d, np.bool_),
              "policy_version": np.array(versions if versions is not None else [0] * d, np.int64),
              "teacher": np.full(d, -1, np.int64)}
    meta = {"spec_hash": spec_hash, "attrs": {"result_a": float(result[0]), "result_b": float(result[1]),
                                              "end_reason": "win"}}
    return RD.Episode(meta, fields, menu_off, np.concatenate([np.asarray(m, np.float32) for m in mus]),
                      (ev_id, ev_ver, ev_lp), make_decisions(tokens),
                      make_decisions(omni_tokens) if omni_tokens is not None else None)


def tiny_model(tables, seed=0, **kw):
    torch.manual_seed(seed)
    cfg = dict(d=16, heads=2, layers=1, ff_mult=2, id_dropout=0.0, aux_heads=False)
    cfg.update(kw)
    return M.RMNet(M.ModelConfig(**cfg), tables)


def policy_of(model, tokens):
    """softmax over each decision's menu (model in eval mode, no gradient)."""
    from rmlearn.batch import pad, to_torch
    decs = make_decisions(tokens)
    was = model.training
    model.eval()
    with torch.no_grad():
        x = pad(decs, {"F": 2, "C": 2, "K": 1, "Q": 1})
        logits = model(*to_torch(x))["logits"]
    model.train(was)
    out = []
    for i, a in enumerate(decs.sizes["a"]):
        out.append(torch.softmax(logits[i, :a].double(), -1).numpy())
    return out


def random_episodes(tables, rng, n_games, reg_model=None, reg_version=0, omni=False, full_games=()):
    """Games of 1..9 steps with irregular seats, menus of 1..6 rows, μ random over the menu,
    sideboarding steps, and (with `reg_model`) the regularisation policy's recorded raw log π at
    about half the steps (some at a wrong version)."""
    eps = []
    for g in range(n_games):
        d = rng.randint(1, 10)
        seats = list(rng.randint(0, 2, size=d))
        menus = list(rng.randint(1, 7, size=d))
        tokens = [step_tokens(1 + rng.randint(7), m, rng) for m in menus]
        mus = []
        for m in menus:
            x = rng.rand(m) + 0.05
            mus.append(x / x.sum())
        chosen = [rng.choice(len(mu), p=mu / mu.sum()) for mu in mus]
        sb = list(rng.rand(d) < 0.3)
        evals = None
        if reg_model is not None:
            lps = [np.log(p) for p in policy_of(reg_model, tokens)]
            evals = []
            for i in range(d):
                u = 0.0 if g in full_games else rng.rand()
                if u < 0.4:
                    evals.append([(0, 9, lps[i] * 0), (1, reg_version, lps[i]), (2, reg_version, lps[i])])
                elif u < 0.55:
                    evals.append([(1, reg_version + 77, lps[i] * 0 - 5.0)])        # a stale version: ignored
                else:
                    evals.append([])
        res = rng.choice([-1.0, 0.0, 1.0])
        omni_tokens = [step_tokens(9 + rng.randint(7), m, rng) for m in menus] if omni else None
        eps.append(make_episode(tables.hashes["spec_hash"], seats, tokens, mus, chosen, (res, -res), sb, evals,
                                omni_tokens))
    return eps


def learner_inputs(learner, batch, eps):
    """The oracle's inputs for `batch` from the learner's networks, computed apart from the learner:
    per valid step the current and target networks' logits and values and both regularisation
    policies' smoothed log π (recomputed), as [T, G, ...] float64 arrays."""
    from rmlearn.batch import pad, to_torch
    T, G, A = batch.T, batch.G, batch.A
    logits = np.zeros((T, G, A))
    v_cur, v_tgt = np.zeros((T, G, 1)), np.zeros((T, G, 1))
    lps = [np.zeros((T, G, A)), np.zeros((T, G, A))]
    for g, ep in enumerate(eps):
        x = to_torch(pad(ep.decs, {"F": 2, "C": 2, "K": 1, "Q": 1}, {"A": A}))
        with torch.no_grad():
            out = learner.model(*x)
            tout = learner.target(*x)
            d = ep.steps
            logits[:d, g] = out["logits"].double().numpy()
            v_cur[:d, g, 0] = out["value"].double().numpy()
            v_tgt[:d, g, 0] = tout["value"].double().numpy()
            for k in range(2):
                reg = learner.regs[k]
                rl = reg.model(*x)["logits"].double()
                lp = torch.log_softmax(rl.masked_fill(~x[9], -float("inf")), -1).numpy()
                legal = batch.legal[:d, g]
                lp = np.where(legal, lp, 0.0)
                lps[k][:d, g] = o_smooth(lp, legal, batch.sideboarding[:d, g], reg.eps, reg.eps_sideboarding) \
                    if reg.smoothed() else lp
    return logits, v_cur, v_tgt, lps


def check_composition(tables):
    print("composition: the learner's loss on a fabricated batch through RMNet = the reference loss", flush=True)
    rng = np.random.RandomState(11)
    net = tiny_model(tables, seed=1)
    cfg = RL.RNaDConfig(entropy_schedule_sizes=(10,), chunk_len=3, finetune_from_learner_steps=0,
                        finetune_threshold=0.05, finetune_discretization=8)
    learner = RL.RNaDLearner(net, cfg)
    # Make the target and the regularisation policies differ from the network.
    with torch.no_grad():
        for p in learner.target.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        for p in learner.regs[0].model.parameters():
            p.add_(torch.randn_like(p) * 0.3)
        r1 = RL.RegPolicy(RL._frozen(learner.regs[0].model), 5, 0.0, 0.0)
        for p in r1.model.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    learner.regs = [RL.RegPolicy(learner.regs[0].model, 6, 0.1, 0.3), r1]
    eps = random_episodes(tables, rng, 9, reg_model=learner.regs[0].model, reg_version=6)
    batch = RD.RNaDBatch(eps, chunk_len=3, policies=(1, 2))
    logits, v_cur, v_tgt, lps = learner_inputs(learner, batch, eps)
    valid = batch.valid.astype(np.float64)
    legal = batch.legal.astype(np.float64).copy()
    legal[batch.valid == 0] = 0
    legal[batch.valid == 0, 0] = 1
    mu = batch.mu.astype(np.float64).copy()
    mu[batch.valid == 0] = legal[batch.valid == 0]
    aoh = np.eye(batch.A)[np.maximum(batch.chosen, 0)]
    logits = np.where(legal > 0, logits, 0.0)
    for alpha in (0.4, 1.0):
        lv, ln = o_rnad_loss(logits, v_cur, v_tgt, lps[0], lps[1], alpha, valid, batch.seat, legal, mu, aoh,
                             batch.reward.astype(np.float64), 0.2, 1.0, np.inf, 2.0, 10_000.0, finetune=(0.05, 8))
        got = learner.accumulate_gradients(batch, alpha, finetune_on=True)
        gv, gn = float(got["sums"]["loss_v"]), float(got["sums"]["loss_nerd"])
        ok(abs(gv - lv) < 1e-5 * (1 + abs(lv)) and abs(gn - ln) < 1e-5 * (1 + abs(ln)),
           "composition: α %.1f loss_v %.8f / NeuRD %.8f, the reference %.8f / %.8f" % (alpha, gv, gn, lv, ln))
        info("α %.1f: loss_v %.6f (reference %.6f), NeuRD %.6f (reference %.6f); %d regularisation log π recorded, "
             "%d recomputed" % (alpha, gv, lv, gn, ln, got["reg_recorded"], got["reg_recomputed"]))
    ok(got["reg_recorded"] > 0 and got["reg_recomputed"] > 0, "composition: both the recorded and the recomputed "
       "regularisation paths should be exercised")
    for p in learner.params:
        p.grad = None
    # Chunk-length invariance of a whole step (no id-dropout, so the forwards are deterministic).
    grads = []
    for L in (2, 5, 32):
        lr = RL.RNaDLearner(tiny_model(tables, seed=2), RL.RNaDConfig(chunk_len=L))
        lr.accumulate_gradients(RD.RNaDBatch(eps, chunk_len=L, policies=(1, 2)), 0.3)
        grads.append(torch.cat([p.grad.flatten() for p in lr.params if p.grad is not None]))
    d = max(float((g - grads[0]).abs().max()) for g in grads) / float(grads[0].abs().max())
    ok(d < 1e-5, "composition: chunk lengths 2 / 5 / 32 give different gradients (relative %.3g)" % d)
    info("chunk lengths 2, 5 and 32 (one game spans up to 5 chunks): the same gradient (relative difference %.1e)"
         % d)


# ── games ────────────────────────────────────────────────────────────────────────────────────

def play_games(learner, n, game, rng):
    """`n` episodes of a two-step game sampled from the learner's current policy (μ = that policy)."""
    obs_tokens = [step_tokens(1, game["menus"][0]), step_tokens(2, game["menus"][1])]
    pol = policy_of(learner.model, obs_tokens)
    eps = []
    for _ in range(n):
        a = rng.choice(len(pol[0]), p=pol[0] / pol[0].sum())
        b = rng.choice(len(pol[1]), p=pol[1] / pol[1].sum())
        r = game["payoff"](a, b)
        eps.append(make_episode(learner.model.hashes["spec_hash"], [0, 1], obs_tokens, [pol[0], pol[1]], [a, b],
                                (r, -r), versions=[learner.learner_steps] * 2))
    return eps, pol


def pretrain(model, targets, steps=300):
    """Behaviour-clones `model` toward fixed distributions at the two observations."""
    from rmlearn.batch import pad, to_torch
    toks = [step_tokens(1, len(targets[0])), step_tokens(2, len(targets[1]))]
    x = to_torch(pad(make_decisions(toks), {"F": 2, "C": 2, "K": 1, "Q": 1}))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        lg = model(*x)["logits"]
        loss = 0
        for i, t in enumerate(targets):
            lp = torch.log_softmax(lg[i, :len(t)], -1)
            loss = loss - (torch.tensor(t, dtype=torch.float32) * lp).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def run_game(tables, game, cfg, iterations, games_per_step, seed, start=None):
    rng = np.random.RandomState(seed)
    net = tiny_model(tables, seed=seed)
    if start is not None:
        pretrain(net, start)
    learner = RL.RNaDLearner(net, cfg)
    total = iterations * cfg.entropy_schedule_sizes[0]
    pols = [policy_of(learner.model, [step_tokens(1, game["menus"][0]), step_tokens(2, game["menus"][1])])]
    for _ in range(total):
        eps, pol = play_games(learner, games_per_step, game, rng)
        learner.step(RD.RNaDBatch(eps, chunk_len=cfg.chunk_len))
        if learner.learner_steps % cfg.entropy_schedule_sizes[0] == 0:
            pols.append(pol)
    return learner, pols


# OpenSpiel's biased rock-paper-scissors (matrix_brps), doubled: its equilibrium is (1, 10, 5) / 16.
BRPS = 2 * np.array([[0, -0.25, 0.5], [0.25, 0, -0.05], [-0.5, 0.05, 0]])
BRPS_NASH = np.array([1, 10, 5]) / 16.0


def nash_distance(pol):
    return max(float(np.abs(pol[i] - BRPS_NASH).max()) for i in range(2))


def check_games(tables, quick=False):
    print("games: R-NaD through RMNet on fabricated tokens", flush=True)
    t0 = time.time()
    brps = {"menus": (3, 3), "payoff": lambda a, b: float(BRPS[a, b])}
    iters = 12 if quick else 16
    runs = {}
    for eta in ((0.2,) if quick else (0.2, 0.0)):
        cfg = RL.RNaDConfig(learning_rate=5e-3, entropy_schedule_sizes=(100,), target_network_avg=0.05, chunk_len=2,
                            reg_eps=0.1, eta=eta)
        learner, pols = run_game(tables, brps, cfg, iters, 128, seed=5, start=[[0.75, 0.15, 0.1], [0.75, 0.15, 0.1]])
        runs[eta] = (learner, pols)
    learner, pols = runs[0.2]
    final = policy_of(learner.target, [step_tokens(1, 3), step_tokens(2, 3)])
    path = [nash_distance(p) for p in pols]
    ok(nash_distance(final) < 0.06 and max(path[-3:]) < 0.08,
       "games: biased rock-paper-scissors: the policy is %.3f from the equilibrium after %d iterations (path %s)"
       % (nash_distance(final), iters, " ".join("%.2f" % x for x in path)))
    info("biased rock-paper-scissors (A then B, B blind to A; equilibrium (1, 10, 5)/16): from π %s, after %d "
         "iterations of Δm 100 the target network's policy is %s / %s (max distance %.3f)"
         % (np.round(pols[0][0], 2), iters, np.round(final[0], 3), np.round(final[1], 3), nash_distance(final)))
    info("η 0.2, the current network's distance per iteration: " + " ".join("%.2f" % x for x in path))
    if 0.0 in runs:
        info("η 0 (no regularisation; for contrast, not asserted): " +
             " ".join("%.2f" % nash_distance(p) for p in runs[0.0][1]))
    dom = {"menus": (3, 2), "payoff": lambda a, b: 1.0 if a == 2 else -1.0}
    cfg = RL.RNaDConfig(learning_rate=3e-3, entropy_schedule_sizes=(40,), target_network_avg=0.05, chunk_len=2)
    learner, pols = run_game(tables, dom, cfg, 6 if quick else 8, 64, seed=9, start=[[0.6, 0.35, 0.05], [0.5, 0.5]])
    final = policy_of(learner.model, [step_tokens(1, 3), step_tokens(2, 2)])
    ok(final[0][2] > 0.9, "games: the dominant action has probability %.3f" % final[0][2])
    info("dominant action (A's row 2 wins whatever B does): π_A %s → %s" % (np.round(pols[0][0], 3),
                                                                          np.round(final[0], 3)))
    info("(%.1f s)" % (time.time() - t0))


# ── bookkeeping ──────────────────────────────────────────────────────────────────────────────

class FakeServer:
    def __init__(self):
        self.calls = []

    def set_policy(self, pid, policy, version):
        self.calls.append((pid, version, policy))


def params_of(m):
    return torch.cat([p.detach().flatten() for p in m.parameters()])


def check_bookkeeping(tables, work):
    print("bookkeeping: boundaries, pushes, stale regularisation, μ, resume, BC seeding", flush=True)
    rng = np.random.RandomState(4)
    eps = random_episodes(tables, rng, 6)
    batch = RD.RNaDBatch(eps, chunk_len=4, policies=(1, 2))
    cfg = RL.RNaDConfig(entropy_schedule_sizes=(3,), chunk_len=4, learning_rate=1e-3)
    learner = RL.RNaDLearner(tiny_model(tables, seed=3, id_dropout=0.3), cfg)
    srv = FakeServer()
    learner.add_hook(RL.ServerLink(srv, push_every=2))
    ok([(c[0], c[1]) for c in srv.calls] == [(1, 0), (2, 0), (0, 0)], "bookkeeping: attach pushes %s"
       % [(c[0], c[1]) for c in srv.calls])
    ref = OEntropySchedule([3], [1])
    alphas, events = [], []
    for i in range(7):
        n_calls = len(srv.calls)
        target_before = None
        m = learner.step(batch)
        alphas.append(m["alpha"])
        events.append([(c[0], c[1]) for c in srv.calls[n_calls:]])
        if m["boundary"]:
            ok(torch.equal(params_of(learner.regs[0].model), params_of(learner.target)),
               "bookkeeping: π_m,reg is not the target network at the boundary")
            ok(learner.regs[0].version == learner.learner_steps and learner.regs[0].eps == 0,
               "bookkeeping: the new regularisation policy's version / smoothing")
        del target_before
    ok(alphas == [ref(i)[0] for i in range(7)], "bookkeeping: α %s" % alphas)
    ok(learner.reg_updates == 2 and [r.version for r in learner.regs] == [6, 3],
       "bookkeeping: after 7 steps of Δm 3: %d updates, versions %s" % (learner.reg_updates,
                                                                         [r.version for r in learner.regs]))
    want = [[], [(0, 2)], [(1, 3), (2, 0), (0, 3)], [(0, 4)], [], [(1, 6), (2, 3), (0, 6)], []]
    ok(events == want, "bookkeeping: pushes per step %s, want %s" % (events, want))
    info("Δm 3 over 7 steps: α %s, boundaries after steps 3 and 6 (π_m,reg = the target network, versions [6, 3]), "
         "pushes %s" % ([round(a, 3) for a in alphas], events))
    # Stale regularisation versions: recompute (counted) or drop the games (counted).
    cfg2 = RL.RNaDConfig(chunk_len=4)
    l2 = RL.RNaDLearner(tiny_model(tables, seed=3), cfg2)
    eps_r = random_episodes(tables, np.random.RandomState(5), 6, reg_model=l2.regs[0].model, reg_version=0,
                            full_games=(1, 4))
    b2 = RD.RNaDBatch(eps_r, chunk_len=4, policies=(1, 2))
    m_rec = l2.step(b2)
    l3 = RL.RNaDLearner(tiny_model(tables, seed=3), RL.RNaDConfig(chunk_len=4, stale_reg="drop"))
    m_drop = l3.step(b2)
    full = sum(int(b2.valid[:, g].sum()) for g in range(b2.G)
               if all(b2.evals[1]["present"][t, g] and b2.evals[1]["version"][t, g] == 0
                      for t in range(int(b2.valid[:, g].sum()))))
    ok(m_rec["reg_recorded"] > 0 and m_rec["reg_recomputed"] > 0 and m_rec["games_dropped_stale"] == 0,
       "bookkeeping: recompute counts %s" % m_rec)
    ok(m_drop["reg_recomputed"] == 0 and m_drop["games_dropped_stale"] > 0 and m_drop["decisions"] == full > 0,
       "bookkeeping: drop mode kept %d decisions, want %d" % (m_drop["decisions"], full))
    info("stale regularisation: recompute %d recorded + %d recomputed log π; drop %d of %d games (%d decisions left)"
         % (m_rec["reg_recorded"], m_rec["reg_recomputed"], m_drop["games_dropped_stale"], b2.G, m_drop["decisions"]))
    # μ missing: refused by the filter (or dropped), and by the step.
    bad = random_episodes(tables, np.random.RandomState(6), 2)
    bad[0].has_mu[0] = False
    try:
        RD.filter_episodes(bad)
        ok(False, "bookkeeping: filter_episodes accepted a step without μ")
    except ValueError:
        pass
    kept, dropped = RD.filter_episodes(bad, missing_mu="drop")
    ok(len(kept) == 1 and dropped == {"no_mu": 1}, "bookkeeping: missing μ drop %s" % dropped)
    try:
        l2.step(RD.RNaDBatch(bad, chunk_len=4))
        ok(False, "bookkeeping: a step without μ reached the update")
    except ValueError:
        pass
    info("a step without μ: filter_episodes raises (missing_mu='error') or drops the game, step() refuses it")
    # Resume: 3 steps, state_dict through torch.save, 4 more == 7 uninterrupted (CPU, bit-identical).
    def fresh(seed):
        return RL.RNaDLearner(tiny_model(tables, seed=seed, id_dropout=0.3),
                              RL.RNaDConfig(entropy_schedule_sizes=(3,), chunk_len=4, learning_rate=1e-3))
    torch.manual_seed(123)
    a = fresh(3)
    for _ in range(7):
        a.step(batch)
    torch.manual_seed(123)
    b = fresh(3)
    for _ in range(3):
        b.step(batch)
    buf = io.BytesIO()
    torch.save(b.state_dict(), buf)
    torch.manual_seed(999)
    torch.rand(100)
    c = fresh(17)
    c.load_state_dict(torch.load(io.BytesIO(buf.getvalue()), weights_only=False))
    for _ in range(4):
        c.step(batch)
    same = (torch.equal(params_of(a.model), params_of(c.model)) and torch.equal(params_of(a.target), params_of(c.target))
            and all(torch.equal(params_of(x.model), params_of(y.model)) for x, y in zip(a.regs, c.regs))
            and [r.version for r in a.regs] == [r.version for r in c.regs]
            and all(torch.equal(x, y) for x, y in zip(a.opt.m + a.opt.v, c.opt.m + c.opt.v))
            and (a.learner_steps, a.reg_updates, a.opt.t) == (c.learner_steps, c.reg_updates, c.opt.t))
    ok(same, "bookkeeping: a resumed run differs from the uninterrupted one")
    d = RL.RNaDLearner.from_state_dict(torch.load(io.BytesIO(buf.getvalue()), weights_only=False), tables)
    for _ in range(4):
        d.step(batch)
    ok(torch.equal(params_of(a.model), params_of(d.model)), "bookkeeping: from_state_dict resume differs")
    info("resume: 3 steps + state_dict (torch.save) + 4 steps = 7 steps bit for bit (network, target, both "
         "regularisation policies, Adam, counters; id-dropout draws included); from_state_dict too")
    # BC seeding.
    bc = tiny_model(tables, seed=8)
    path = os.path.join(work, "bc.pt")
    M.save_checkpoint(bc, path)
    s = RL.RNaDLearner.from_bc_checkpoint(path, tables, RL.RNaDConfig(reg_eps=0.1, reg_eps_sideboarding=0.25))
    ok(torch.equal(params_of(s.model), params_of(bc)) and torch.equal(params_of(s.target), params_of(bc)) and
       s.regs[0] is s.regs[1] and torch.equal(params_of(s.regs[0].model), params_of(bc)) and
       (s.regs[0].eps, s.regs[0].eps_sideboarding, s.regs[0].version) == (0.1, 0.25, 0),
       "bookkeeping: BC seeding")
    lp = torch.log_softmax(torch.tensor([[3.0, 0.0, -2.0, 0.0]]), -1)
    legal = torch.tensor([[True, True, True, False]])
    sm = s._smooth(s.regs[0], lp, legal, torch.tensor([False]))
    sb = s._smooth(s.regs[0], lp, legal, torch.tensor([True]))
    p = torch.softmax(torch.tensor([3.0, 0.0, -2.0]), -1)
    want = torch.log(0.9 * p + 0.1 / 3)
    want_sb = torch.log(0.75 * p + 0.25 / 3)
    ok(torch.allclose(sm[0, :3], want, atol=1e-6) and torch.allclose(sb[0, :3], want_sb, atol=1e-6) and
       float(sm[0, 3]) == 0.0, "bookkeeping: the smoothed regularisation log π")
    info("BC seeding: network = target = π_0,reg = π_−1,reg (one shared frozen copy, version 0, ε 0.1, ε_sb 0.25); "
         "the learner-side smoothing is log((1 − ε)π + ε/n) over the legal rows")
    # The omniscient critic and the other precisions run.
    eps_o = random_episodes(tables, np.random.RandomState(8), 5, omni=True)
    for value in ("tanh", "linear"):
        lo = RL.RNaDLearner(tiny_model(tables, seed=4, omniscient_critic=True, critic_layers=1),
                            RL.RNaDConfig(chunk_len=4, value_activation=value, learning_rate=1e-3))
        before = params_of(lo.model.critic)
        m = lo.step(RD.RNaDBatch(eps_o, chunk_len=4, omniscient=True))
        ok(np.isfinite(m["loss"]) and not torch.equal(before, params_of(lo.model.critic)) and m["loss_value_head"] > 0,
           "bookkeeping: the omniscient critic (%s) step" % value)
    lb = RL.RNaDLearner(tiny_model(tables, seed=4), RL.RNaDConfig(chunk_len=4, precision="bf16"))
    m = lb.step(batch)
    ok(np.isfinite(m["loss"]), "bookkeeping: a bf16 step")
    info("the omniscient critic (tanh and linear values) trains on the omniscient inputs; a bf16 autocast step runs")


# ── real episodes ────────────────────────────────────────────────────────────────────────────

def check_real(args):
    print("real data: engine episodes through the default model on CPU", flush=True)
    tables = S.load_tables(args.tables)
    files = sorted(glob.glob(os.path.join(args.episodes, "*.rmtraj")))
    rmtraj = RD.rmtraj_module()
    eps = []
    for f in files:
        a = RD.read_episode(f)
        b = RD.from_rmtraj(rmtraj.read(f))
        same = all(np.array_equal(a.decs.arrays[k], b.decs.arrays[k]) for k in RD.TENSOR_NAMES) and \
            np.array_equal(a.mu, b.mu) and all(np.array_equal(getattr(a, k), getattr(b, k)) for k in RD.STEP_FIELDS)
        ok(same, "real: %s read with numpy differs from python/rmtraj.py" % f)
        eps.append(a)
    kept, dropped = RD.filter_episodes(eps, spec_hash=tables.hashes["spec_hash"])
    if not ok(len(kept) >= 2, "real: %d usable episodes in %s (%s)" % (len(kept), args.episodes, dropped)):
        return
    info("%d episodes (%d decisions) read with numpy = python/rmtraj.py" % (len(eps), sum(e.steps for e in eps)))
    torch.manual_seed(0)
    net = M.RMNet(M.ModelConfig(), tables)
    learner = RL.RNaDLearner(net, RL.RNaDConfig(chunk_len=32))
    batch = RD.RNaDBatch(kept[:4], chunk_len=32, policies=(1, 2))
    for i in range(2):
        m = learner.step(batch)
        ok(all(np.isfinite(m[k]) for k in ("loss", "loss_v", "loss_nerd", "grad_norm")), "real: step %d %s" % (i, m))
        info("step %d: %d decisions of %d games (T %d, A %d): loss_v %.4f, NeuRD %.4f, |∇| %.3g, %d log π_reg "
             "recomputed, %.1f s" % (i, m["decisions"], m["games"], m["T"], batch.A, m["loss_v"], m["loss_nerd"],
                                     m["grad_norm"], m["reg_recomputed"], m["seconds"]))
    if args.omni_episodes:
        oe = [RD.read_episode(f) for f in sorted(glob.glob(os.path.join(args.omni_episodes, "*.rmtraj")))]
        oe, _ = RD.filter_episodes(oe, spec_hash=tables.hashes["spec_hash"])
        torch.manual_seed(0)
        lo = RL.RNaDLearner(M.RMNet(M.ModelConfig(omniscient_critic=True), tables), RL.RNaDConfig(chunk_len=32))
        m = lo.step(RD.RNaDBatch(oe[:2], chunk_len=32, omniscient=True))
        ok(np.isfinite(m["loss"]), "real: omniscient critic step %s" % m)
        info("omniscient critic: %d decisions, loss_v %.4f, NeuRD %.4f, value head %.4f, %.1f s"
             % (m["decisions"], m["loss_v"], m["loss_nerd"], m["loss_value_head"], m["seconds"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", help="rmfeat --export-tables DIR (with --episodes: the real-data checks)")
    ap.add_argument("--episodes", help="a directory of *.rmtraj episodes with μ (e.g. random agents)")
    ap.add_argument("--omni-episodes", help="a directory of omniscient *.rmtraj episodes (the critic)")
    ap.add_argument("--quick", action="store_true", help="fewer iterations in the games")
    ap.add_argument("--part", default="oracle,projection,composition,games,bookkeeping,real")
    args = ap.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    parts = set(args.part.split(","))
    work = tempfile.mkdtemp(prefix="rnad_check_")
    try:
        tables = fake_tables()
        if "oracle" in parts:
            check_oracle()
        if "projection" in parts:
            check_projection(work)
        if "composition" in parts:
            check_composition(tables)
        if "bookkeeping" in parts:
            check_bookkeeping(tables, work)
        if "games" in parts:
            check_games(tables, args.quick)
        if "real" in parts and args.tables and args.episodes:
            check_real(args)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if FAILURES:
        print("RNAD-CHECK: %d failure(s)" % len(FAILURES))
        sys.exit(1)
    print("RNAD-CHECK: ok")


if __name__ == "__main__":
    main()
