#!/usr/bin/env python3
"""Phase D / M5 gate: obs bit-parity between bin/az_actor and the Python pipeline.

The in-process C++ actor (bin/az_actor) reconstructs the 6700-float RL observation
engine-side (src/actor/obs_builder.cpp) instead of round-tripping the stdio BQUERY
protocol. This test proves that reconstruction is BIT-EXACT against the Python
env's obs for the identical game:

  1. Build a deterministic AZNet (torch.manual_seed(0)), save its state_dict and
     export the TorchScript module both the C++ actor and this test consume.
  2. Run the C++ actor for one game (--deck league/ur_delver --seed 1) with
     --dump-obs, capturing every decision's (num_choices, chosen action,
     observation) record to a binary file.
  3. Drive the SAME game through the Python env (RoboMageEnv + runner.drive_game),
     both seats piloted by a replay controller that plays the actor's recorded
     action at decision i, so both sides stay on one trajectory for the whole
     game. The controller still evaluates the same TorchScript module over the
     masked logits and records its own argmax and the top-2 logit gap.
  4. Assert identical decision count and, per decision, identical num_choices
     and every observation row bit-exact (np.array_equal). A Python argmax that
     differs from the actor's pick FAILS when the top-2 gap is >= NEAR_TIE_GAP;
     below it the decision is a tolerated near-tie (float noise between the two
     forward passes can flip an exact tie), counted on the PASS line.

Every game (bo1) / match (bo3) is capped at PARITY_MAX_DECISIONS real decisions on
both sides (az_actor --max-decisions; runner.drive_game max_decisions), so a
randomly initialized net that stumbles into a degenerate loop compares a bounded
prefix instead of running unbounded.

Run: train/.venv/bin/python train/test_actor_parity.py
"""

import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from az_net import AZNet, obs_space_from_const, save_torchscript, torchscript_export_path
from env import RoboMageEnv, OBS_SIZE, MAX_ACTIONS
from cli_spec import REPO_ROOT, BIN_DIR, BUILD_DIR
import runner

DECK = "league/ur_delver"
SEED = 1
ACTOR_BIN = os.path.join(BUILD_DIR, "az_actor")
# Per-game (bo1) / per-match (bo3) decision cap shared by every actor parity test
# (test_mcts_parity.py imports it): az_actor --max-decisions on the C++ side,
# runner.drive_game(max_decisions=) on the Python side. Both count every real
# decision of the game/match (both seats, sideboard and single-choice prompts
# included; never search simulation steps), so the two capped streams are the
# same prefix of the same game.
PARITY_MAX_DECISIONS = 3000
# Top-2 masked-logit gap below which a Python argmax that differs from the
# actor's recorded pick is a tolerated near-tie rather than a failure.
NEAR_TIE_GAP = 1e-6


class ReplayController:
    """Replays the actor's recorded actions: decision i plays actor_actions[i]
    while it is a legal index for this menu (else the controller's own argmax,
    whose mismatch _compare then reports). Each decision records (num_choices,
    obs copy, own argmax over the masked logits, top-2 logit gap). One instance
    drives both seats (a single global policy)."""

    def __init__(self, ts_path, actor_actions):
        self.module = torch.jit.load(ts_path)
        self.module.eval()
        self.actor_actions = actor_actions
        self.records = []  # list[(num_choices, obs_copy, argmax, gap)]

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None):
        mask = torch.zeros(1, MAX_ACTIONS, dtype=torch.bool)
        mask[0, :num_choices] = True
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _value = self.module(obs_t, mask)
        own, gap = _top_pick(logits[0, :num_choices])
        i = len(self.records)
        self.records.append((int(num_choices),
                             np.array(obs, dtype=np.float32, copy=True), own, gap))
        if i < len(self.actor_actions) and 0 <= self.actor_actions[i] < num_choices:
            return int(self.actor_actions[i])
        return own


def _top_pick(logits):
    """(argmax index, top-2 gap) of a 1-D logit tensor; the gap is inf for a
    single-choice menu."""
    own = int(torch.argmax(logits))
    if logits.numel() < 2:
        return own, float("inf")
    top2 = torch.topk(logits, 2).values
    return own, float(top2[0] - top2[1])


def _read_dump(path):
    """Read the actor's --dump-obs binary: repeated (int32 num_choices, int32
    chosen action, float32[OBS_SIZE]). Returns list[(num_choices, action,
    np.ndarray)]."""
    out = []
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    rec = 8 + OBS_SIZE * 4
    while off + rec <= len(data):
        nc, action = struct.unpack_from("<ii", data, off)
        vals = np.frombuffer(data, dtype="<f4", count=OBS_SIZE, offset=off + 8)
        out.append((nc, action, np.array(vals, dtype=np.float32, copy=True)))
        off += rec
    if off != len(data):
        raise RuntimeError(f"dump file {path} has {len(data) - off} trailing bytes "
                           "— frame size mismatch")
    return out


def _run_case(td, ts_path, bo3):
    """Run one parity case (bo1 or a bo3 MATCH): the C++ actor with --dump-obs,
    then the Python env driven by a ReplayController playing the actor's recorded
    actions. Returns (actor_recs, py_recs) — _read_dump records and the
    controller's records — or None on an actor error."""
    tag = "bo3" if bo3 else "bo1"
    # 1) C++ actor: one game (bo1) or one best-of-three match (--bo3), dumping obs.
    dump_path = os.path.join(td, f"actor_obs_{tag}.bin")
    cmd = [ACTOR_BIN, "--deck", DECK, "--seed", str(SEED),
           "--model", ts_path, "--dump-obs", dump_path, "--games", "1",
           "--max-decisions", str(PARITY_MAX_DECISIONS)]
    if bo3:
        cmd.append("--bo3")
    proc = subprocess.run(cmd, cwd=BIN_DIR, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print(f"FAIL [{tag}]: az_actor exited nonzero:\n"
              + proc.stderr.decode("utf-8", "replace"), file=sys.stderr)
        return None
    actor_recs = _read_dump(dump_path)

    # 2) Drive the SAME game/match through the Python env, both seats replaying
    #    the actor's actions.
    ctrl = ReplayController(ts_path, [a for _nc, a, _obs in actor_recs])
    env = RoboMageEnv(deck_a=DECK, deck_b=DECK, bo3=bo3)
    # The decision cap is drive_game's max_decisions (the same count the actor's
    # --max-decisions applies); the env's own training-only step truncation is
    # disabled so it can never cut the drive at a different point.
    env.MAX_STEPS = env.MAX_STEPS_BO3 = 1 << 30
    try:
        obs, _ = env.reset(options={"engine_seed": SEED})
        rec = runner.drive_game(env, obs, ctrl, ctrl,
                                max_decisions=PARITY_MAX_DECISIONS)
    finally:
        env.close()
    if rec.capped:
        print(f"NOTE [{tag}]: decision cap {PARITY_MAX_DECISIONS} reached — "
              f"comparing the capped prefix")
    return actor_recs, ctrl.records


def _compare(tag, actor_recs, py_recs):
    """Per decision: num_choices equal and obs bit-exact, then the pick check
    (a differing Python argmax fails unless its top-2 gap is < NEAR_TIE_GAP).
    Returns 0 on success, 1 on any failure."""
    if len(actor_recs) != len(py_recs):
        print(f"FAIL [{tag}]: decision count differs — actor={len(actor_recs)} "
              f"python={len(py_recs)}", file=sys.stderr)
        n = min(len(actor_recs), len(py_recs))
        for i in range(n):
            if actor_recs[i][0] != py_recs[i][0] or not np.array_equal(
                    actor_recs[i][2], py_recs[i][1]):
                print(f"  first divergence at decision {i} "
                      f"(nc actor={actor_recs[i][0]} python={py_recs[i][0]})",
                      file=sys.stderr)
                break
        return 1

    near_ties = 0
    for i, ((a_nc, a_act, a_obs), (p_nc, p_obs, p_pick, gap)) in enumerate(
            zip(actor_recs, py_recs)):
        if a_nc != p_nc:
            print(f"FAIL [{tag}]: num_choices differ at decision {i}: "
                  f"actor={a_nc} python={p_nc}", file=sys.stderr)
            return 1
        if not np.array_equal(a_obs, p_obs):
            diff = np.flatnonzero(a_obs != p_obs)
            print(f"FAIL [{tag}]: obs differ at decision {i} "
                  f"({diff.size} floats differ)", file=sys.stderr)
            for j in diff[:10]:
                print(f"  obs[{j}]: actor={a_obs[j]!r} python={p_obs[j]!r}",
                      file=sys.stderr)
            return 1
        if not 0 <= a_act < a_nc:
            print(f"FAIL [{tag}]: actor action {a_act} out of range at decision "
                  f"{i} (num_choices={a_nc})", file=sys.stderr)
            return 1
        if p_pick != a_act:
            if gap >= NEAR_TIE_GAP:
                print(f"FAIL [{tag}]: pick differs at decision {i}: actor={a_act} "
                      f"python={p_pick} (top-2 gap {gap:.3g} >= {NEAR_TIE_GAP:g})",
                      file=sys.stderr)
                return 1
            near_ties += 1

    print(f"PASS [{tag}]: obs bit-exact over {len(actor_recs)} decisions, "
          f"{near_ties} near-tie pick(s) tolerated "
          f"(deck={DECK} seed={SEED}, OBS_SIZE={OBS_SIZE})")
    return 0


def main():
    if not os.path.exists(ACTOR_BIN):
        print(f"FAIL: {ACTOR_BIN} not found — build it with `make actor`", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as td:
        # Deterministic AZNet -> state_dict ckpt + TorchScript export (shared).
        torch.manual_seed(0)
        net = AZNet(obs_space_from_const()).eval()
        ckpt = os.path.join(td, "parity__azfinal.pt")
        net.save(ckpt)
        ts_path = torchscript_export_path(ckpt)     # parity__azfinal.ts.pt
        save_torchscript(net, ts_path)

        # bo1 game AND a full bo3 match (which exercises the between-games
        # sideboard-phase observation mask — the bo3 context block and the masked
        # stale board must both stay bit-for-bit identical to the Python pipeline).
        rc = 0
        for bo3 in (False, True):
            tag = "bo3" if bo3 else "bo1"
            case = _run_case(td, ts_path, bo3)
            if case is None:
                return 1
            rc |= _compare(tag, *case)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
