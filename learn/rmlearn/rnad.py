"""R-NaD's learner math (docs/rnad_training_plan.md §4 L4) in PyTorch: a port of OpenSpiel's
``open_spiel/python/algorithms/rnad/rnad.py`` at commit 8156646cdef0c753162fc6b68a11d57cce642f95
(the JAX reference implementation of Perolat et al., arXiv 2206.15378).

Every function keeps the reference's name, arguments and ``[T, B, ...]`` layout (time-major, ``T``
decisions of ``B`` games; ``A`` menu rows), so the two read side by side:

    _player_others, _policy_ratio, _has_played    the per-player masks and the importance ratio
    v_trace                                        two-player v-trace over a mixed seat sequence,
                                                   with the reward transform η·log(π/π_reg)
    get_loss_v, get_loss_nerd                      the critic loss and NeuRD (logit threshold β, clip)
    apply_force_with_threshold, renormalize
    legal_policy, legal_log_policy                 softmax / log-softmax over legal rows (0 elsewhere)
    EntropySchedule                                the Δm schedule: α_n = min(1, 2n/Δm) and the
                                                   iteration boundaries
    threshold_policy, discretize_policy,           the fine-tuning projection (B5), exactly the
    FineTuning                                     engine's src/agents/policy_projection.{h,cpp}

The learner step that drives them is ``rnad_learner.py``.

**Differences from the reference** (each forced by this game or this pipeline; the plan's L4
section lists the same):

1. *Variable menus.* A decision's actions are its menu rows, padded to the batch's largest menu
   with ``legal`` 0 (the reference has one fixed action space with a legal mask, so the math is the
   same). A padded, invalid step gets a dummy one-row menu (row 0 legal, μ one-hot on it), as the
   reference fills invalid steps from a dummy state, so no mean over zero legal rows occurs.
2. *Chunked scan.* ``v_trace`` takes and returns its carry, so a game longer than a chunk is
   scanned chunk by chunk from its last chunk to its first with exactly the reference's result
   (the losses are renormalised by counts over the whole batch, ``get_loss_v`` / ``get_loss_nerd``
   ``normalization=``); the learner forwards and back-propagates one chunk at a time.
3. *Projection rounding.* ``discretize_policy`` is the engine's projection: ``ceil(p·n − 1e-9)``
   where the reference takes ``ceil(p·n)``, so a probability that is k/n up to float noise keeps
   k (the actor's μ and the learner's projected π must be the same function). Computed in float64.
4. *Clamp, not min, for ρ̄ and c̄* (``torch.clamp(max=)``; ρ̄ = ∞ as the reference passes it).

``_has_played`` is ported as the reference computes it: its carry is never set, so it reduces to
``valid · (player_id == player)``; ``has_played_scan`` (the literal reverse scan) is kept for the
checks.
"""

from dataclasses import dataclass

import numpy as np
import torch

REFERENCE = "open_spiel/python/algorithms/rnad/rnad.py @ 8156646cdef0c753162fc6b68a11d57cce642f95"
# How far above an integer a row's scaled probability may lie and still round down to it
# (src/agents/policy_projection.cpp CEIL_SLACK).
CEIL_SLACK = 1e-9


# ── schedule ──────────────────────────────────────────────────────────────────────────────────

class EntropySchedule:
    """The reference's EntropySchedule: an increasing list of learner steps at which the
    regularisation network is updated, built from iteration sizes and their repeats (the last size
    repeats forever). ``EntropySchedule(sizes=[3, 5, 10], repeats=[2, 4, 1])`` → boundaries
    ``[0, 3, 6, 11, 16, 21, 26, 36, 46, ...]``."""

    def __init__(self, *, sizes, repeats):
        sizes, repeats = list(sizes), list(repeats)
        if len(repeats) != len(sizes) or not sizes or any(r <= 0 for r in repeats) or repeats[-1] != 1 or \
                any(s <= 0 for s in sizes):
            raise ValueError("entropy schedule: sizes %s and repeats %s (parallel, non-empty, positive, the "
                             "last repeat 1)" % (sizes, repeats))
        schedule = [0]
        for size, repeat in zip(sizes, repeats):
            schedule.extend([schedule[-1] + (i + 1) * size for i in range(repeat)])
        self.sizes, self.repeats = sizes, repeats
        self.schedule = np.array(schedule, dtype=np.int64)

    def bounds(self, learner_step):
        """(iteration_start, iteration_size) of the iteration `learner_step` is in."""
        s = self.schedule
        last_size = int(s[-1] - s[-2])
        if s[-1] <= learner_step:
            return int(s[-1] + (learner_step - s[-1]) // last_size * last_size), last_size
        start = int(np.max(s * (s <= learner_step)))
        finish = int(np.min(s[learner_step < s]))
        return start, finish - start

    def __call__(self, learner_step):
        """(alpha, update_target_net) at `learner_step`, as the reference returns them."""
        start, size = self.bounds(learner_step)
        update = learner_step > 0 and learner_step == start + size - 1
        alpha = min((2.0 * (learner_step - start)) / size, 1.0)
        return alpha, bool(update)


# ── fine-tuning projection ────────────────────────────────────────────────────────────────────

def threshold_policy(policy, mask, threshold):
    """The reference's FineTuning._threshold (= policy_projection.h threshold_policy per row): rows
    below `threshold` zeroed and the rest renormalised, unless every row is below it."""
    if threshold <= 0:
        return policy
    m = mask.to(policy.dtype) * ((policy >= threshold).to(policy.dtype) +
                                 (policy.amax(-1, keepdim=True) < threshold).to(policy.dtype))
    return m * policy / (m * policy).sum(-1, keepdim=True)


def discretize_policy(policy, n):
    """FineTuning._discretize as the engine computes it (policy_projection.h discretize_policy) over
    the last dimension: rows visited in descending probability (stable: ties in row order) each take
    ceil(p·n − 1e-9) of the n steps, capped at what is left; the remainder goes to the most probable
    row. Float64 arithmetic; returns the input's dtype."""
    if n <= 0:
        return policy
    p = policy.double()
    order = torch.sort(p, dim=-1, descending=True, stable=True).indices
    sp = p.gather(-1, order)
    up = torch.ceil(sp * n - CEIL_SLACK).clamp(min=0, max=n)
    before = torch.cumsum(up, -1) - up
    take = torch.minimum(up, (n - before).clamp(min=0))
    left = n - take.sum(-1, keepdim=True)
    first = torch.zeros_like(take)
    first[..., 0:1] = left
    counts = torch.zeros_like(p).scatter(-1, order, take + first)
    return (counts / n).to(policy.dtype)


@dataclass
class FineTuning:
    """The reference's FineTuning: from learner step `from_learner_steps` on (negative: never), the
    policy the v-trace merges is thresholded then discretised. Defaults are the plan's (B5) and the
    nn: agent's play-time projection, threshold 0.03 and 16 steps (the reference's 32)."""
    from_learner_steps: int = -1
    policy_threshold: float = 0.03
    policy_discretization: int = 16

    def enabled(self, learner_steps):
        return self.from_learner_steps >= 0 and learner_steps > self.from_learner_steps

    def __call__(self, policy, mask, learner_steps):
        return self.post_process_policy(policy, mask) if self.enabled(learner_steps) else policy

    def post_process_policy(self, policy, mask):
        policy = threshold_policy(policy, mask, self.policy_threshold)
        return discretize_policy(policy, self.policy_discretization)


# ── policies ──────────────────────────────────────────────────────────────────────────────────

def legal_policy(logits, legal_actions):
    """The reference's _legal_policy: softmax over legal rows, 0 elsewhere (no NaN or inf)."""
    legal = legal_actions.to(logits.dtype)
    l_min = logits.amin(-1, keepdim=True)
    logits = torch.where(legal_actions, logits, l_min)
    logits = logits - logits.amax(-1, keepdim=True)
    logits = logits * legal
    exp_logits = torch.where(legal_actions, torch.exp(logits), torch.zeros_like(logits))
    return exp_logits / exp_logits.sum(-1, keepdim=True)


def legal_log_policy(logits, legal_actions):
    """The reference's legal_log_policy: log-softmax over legal rows, 0 at illegal rows."""
    legal = legal_actions.to(logits.dtype)
    logits_masked = logits + torch.log(legal)
    max_legal_logit = logits_masked.amax(-1, keepdim=True)
    logits_masked = logits_masked - max_legal_logit
    baseline = torch.log(torch.exp(logits_masked).sum(-1, keepdim=True))
    return legal * torch.where(legal_actions, logits - max_legal_logit - baseline, torch.zeros_like(logits))


# ── masks and ratios ─────────────────────────────────────────────────────────────────────────

def _player_others(player_ids, valid, player):
    """1 at `player`'s valid steps, -1 at the other player's, 0 at invalid steps; [..., 1]."""
    res = 2 * (player_ids == player).to(torch.float32) - 1
    return (res * valid.to(torch.float32)).unsqueeze(-1)


def _policy_ratio(pi, mu, actions_oh, valid):
    """pi(a)/mu(a) at the chosen action on valid steps, 1 on invalid steps."""
    v = valid.to(pi.dtype)

    def select(p):
        return (actions_oh * p).sum(-1) * v + (1 - v)

    return select(pi) / select(mu)


def _has_played(valid, player_id, player):
    """The reference's _has_played: its reverse scan's carry is never set, so the mask is the
    valid steps of `player` (has_played_scan is the literal scan)."""
    return (valid.to(torch.bool) & (player_id == player)).to(torch.float32)


def has_played_scan(valid, player_id, player):
    """The reference's _has_played as written (a reverse scan over T); for the checks."""
    t_len = valid.shape[0]
    carry = torch.zeros_like(player_id[-1], dtype=torch.float32)
    out = torch.zeros(valid.shape, dtype=torch.float32)
    for t in range(t_len - 1, -1, -1):
        val, ours = valid[t].to(torch.bool), player_id[t] == player
        res = torch.where(val, torch.where(ours, torch.ones_like(carry), carry), torch.zeros_like(carry))
        carry = torch.where(val, carry, torch.zeros_like(carry))
        out[t] = res
    return out


# ── v-trace ───────────────────────────────────────────────────────────────────────────────────

@dataclass
class VTraceCarry:
    """The reverse scan's carry (the reference's LoopVTraceCarry), per game."""
    reward: torch.Tensor               # [B]
    reward_uncorrected: torch.Tensor   # [B]
    next_value: torch.Tensor           # [B, 1]
    next_v_target: torch.Tensor        # [B, 1]
    importance_sampling: torch.Tensor  # [B]

    @classmethod
    def init(cls, b, device=None, dtype=torch.float32):
        z = torch.zeros(b, device=device, dtype=dtype)
        return cls(z, z, z.unsqueeze(-1), z.unsqueeze(-1), torch.ones(b, device=device, dtype=dtype))


def v_trace(v, valid, player_id, acting_policy, merged_policy, merged_log_policy, player_others, actions_oh,
            reward, player, eta, lambda_, c, rho, carry=None):
    """The reference's v_trace for `player` over a [T, B] slice: (v_target [T,B,1], has_played [T,B],
    learning_output [T,B,A], carry). `carry` (None: the scan's initial state) is the state after the
    steps that follow the slice, so scanning a game's chunks last to first gives the whole game's
    result. Shapes: v [T,B,1]; valid, player_id, reward [T,B]; policies, actions_oh [T,B,A];
    player_others [T,B,1]."""
    gamma = 1.0
    t_len, b = valid.shape
    dev, dt = v.device, v.dtype
    has_played = _has_played(valid, player_id, player)
    policy_ratio = _policy_ratio(merged_policy, acting_policy, actions_oh, valid)
    inv_mu = _policy_ratio(torch.ones_like(merged_policy), acting_policy, actions_oh, valid)
    eta_reg_entropy = -eta * (merged_policy * merged_log_policy).sum(-1) * player_others.squeeze(-1)
    eta_log_policy = -eta * merged_log_policy * player_others
    init = VTraceCarry.init(b, dev, dt)
    cr = carry if carry is not None else init
    valid_b = valid.to(torch.bool)
    ours_b = player_id == player
    v_target = torch.zeros_like(v)
    learning_output = torch.zeros_like(merged_policy)
    for t in range(t_len - 1, -1, -1):
        cs, vt, r, ere, inv, aoh, elp = (policy_ratio[t], v[t], reward[t], eta_reg_entropy[t], inv_mu[t],
                                         actions_oh[t], eta_log_policy[t])
        reward_uncorrected = r + gamma * cr.reward_uncorrected + ere
        discounted_reward = r + gamma * cr.reward
        our_v_target = (vt + torch.clamp(cs * cr.importance_sampling, max=rho).unsqueeze(-1) *
                        (reward_uncorrected.unsqueeze(-1) + gamma * cr.next_value - vt) +
                        lambda_ * torch.clamp(cs * cr.importance_sampling, max=c).unsqueeze(-1) * gamma *
                        (cr.next_v_target - cr.next_value))
        our_learning_output = (vt + elp + aoh * inv.unsqueeze(-1) *
                               (discounted_reward.unsqueeze(-1) +
                                gamma * cr.importance_sampling.unsqueeze(-1) * cr.next_v_target - vt))
        val, ours = valid_b[t], ours_b[t]
        val1, ours1 = val.unsqueeze(-1), ours.unsqueeze(-1)
        v_target[t] = torch.where(val1 & ours1, our_v_target, torch.zeros_like(our_v_target))
        learning_output[t] = torch.where(val1 & ours1, our_learning_output, torch.zeros_like(our_learning_output))
        # The carry: ours resets the rewards and the ratio, the opponent's accumulates them, an
        # invalid step restores the initial state.
        n_reward = torch.where(ours, torch.zeros_like(r), ere + cs * discounted_reward)
        n_unc = torch.where(ours, torch.zeros_like(r), reward_uncorrected)
        n_value = torch.where(ours1, vt, gamma * cr.next_value)
        n_vt = torch.where(ours1, our_v_target, gamma * cr.next_v_target)
        n_is = torch.where(ours, torch.ones_like(r), cs * cr.importance_sampling)
        cr = VTraceCarry(torch.where(val, n_reward, init.reward),
                         torch.where(val, n_unc, init.reward_uncorrected),
                         torch.where(val1, n_value, init.next_value),
                         torch.where(val1, n_vt, init.next_v_target),
                         torch.where(val, n_is, init.importance_sampling))
    return v_target, has_played, learning_output, cr


# ── losses ────────────────────────────────────────────────────────────────────────────────────

def _norm(n):
    return n + (n == 0).to(n.dtype) if torch.is_tensor(n) else n + (n == 0)


def get_loss_v(v_list, v_target_list, mask_list, normalization=None):
    """The critic loss: per player the masked mean of (v − stop_gradient(v_target))², summed.
    `normalization` (a list parallel to the others) replaces each mask's own sum (a chunk's loss
    with the whole batch's counts)."""
    total = 0.0
    for i, (v_n, v_target, mask) in enumerate(zip(v_list, v_target_list, mask_list)):
        loss_v = mask.unsqueeze(-1) * (v_n - v_target.detach()) ** 2
        n = mask.sum() if normalization is None else normalization[i]
        total = total + loss_v.sum() / _norm(n)
    return total


def apply_force_with_threshold(decision_outputs, force, threshold, threshold_center):
    """The force applied to the logits only where it does not push them past ±threshold of the centre."""
    can_decrease = (decision_outputs - threshold_center > -threshold).to(decision_outputs.dtype)
    can_increase = (decision_outputs - threshold_center < threshold).to(decision_outputs.dtype)
    force_negative = torch.clamp(force, max=0.0)
    force_positive = torch.clamp(force, min=0.0)
    clipped_force = can_decrease * force_negative + can_increase * force_positive
    return decision_outputs * clipped_force.detach()


def renormalize(loss, mask, normalization=None):
    """Σ loss·mask over the steps, divided by the number of steps (or `normalization`)."""
    n = mask.sum() if normalization is None else normalization
    return (loss * mask).sum() / _norm(n)


def get_loss_nerd(logit_list, policy_list, q_vr_list, valid, player_ids, legal_actions,
                  importance_sampling_correction, clip=100, threshold=2, normalization=None):
    """NeuRD: per player k, the advantage q − Σπq (importance-corrected, clipped to ±clip, no
    gradient) pushes the logits, centred on the mean of the legal logits, while they stay within
    ±threshold; renormalised over k's valid steps (or `normalization[k]`)."""
    legal = legal_actions.to(logit_list[0].dtype)
    num_valid_actions = legal.sum(-1, keepdim=True)
    total = 0.0
    for k, (logit_pi, pi, q_vr, is_c) in enumerate(zip(logit_list, policy_list, q_vr_list,
                                                       importance_sampling_correction)):
        adv_pi = q_vr - (pi * q_vr).sum(-1, keepdim=True)
        adv_pi = is_c * adv_pi
        adv_pi = torch.clamp(adv_pi, min=-clip, max=clip).detach()
        valid_logit_sum = (logit_pi * legal).sum(-1, keepdim=True)
        mean_logit = valid_logit_sum / num_valid_actions
        logits = logit_pi - mean_logit
        threshold_center = torch.zeros_like(logits)
        nerd_loss = (legal * apply_force_with_threshold(logits, adv_pi, threshold, threshold_center)).sum(-1)
        mask = valid.to(nerd_loss.dtype) * (player_ids == k).to(nerd_loss.dtype)
        nerd_loss = -renormalize(nerd_loss, mask, None if normalization is None else normalization[k])
        total = total + nerd_loss
    return total
