#!/usr/bin/env python3
"""az_eval actor-backend gate driver test (rides the opt-in ci tier `actor`).

The search-correctness proof for two-model actor matches lives in
test_mcts_parity.py's gate legs (same-net identity, wiring, EXACT visit parity
vs the two-controller Python reference). This test covers the DRIVER on top:
``az_train.az_eval(use_actor=True)`` with two fresh deterministic nets over a
one-mirror panel must

  * play the panel on the actor (two legs per matchup, seats alternating) and
    return a coherent tally (wins + losses + draws == the panel's match count,
    per-deck/breakdown shapes intact);
  * be DETERMINISTIC: an identical second call returns the identical result;
  * enforce the backend guards loudly: --actor with no incumbent is a
    ValueError (the vs-scripted fallback is Python-only), and the leg/tally
    parser reads MATCH_RESULT/GAME_RESULT lines exactly.

Run: train/.venv/bin/python train/test_az_gate.py
"""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from az_net import AZNet, obs_space_from_const
from az_selfplay import _ACTOR_BIN as ACTOR_BIN  # the driver's own resolution
from az_train import _parse_gate_output, az_eval

DECK = "league/ur_delver"


def _fresh_net_ckpt(td: str, name: str, seed: int) -> str:
    torch.manual_seed(seed)
    net = AZNet(obs_space_from_const()).eval()
    ckpt = os.path.join(td, f"{name}__azfinal.pt")
    net.save(ckpt)
    return ckpt


def _test_parse() -> int:
    bo3_out = ("SELFPLAY-ish noise\nMATCH_RESULT: Player A wins 2-1\n"
               "MATCH_RESULT: Player B wins 2-0\nMATCH_RESULT: Player A wins 2-0\n")
    if _parse_gate_output(bo3_out, bo3=True) != (2, 1, 0):
        print("FAIL: bo3 MATCH_RESULT parse", file=sys.stderr)
        return 1
    bo1_out = ("GAME_RESULT: 1 Player A wins\nGAME_RESULT: 2 draw\n"
               "GAME_RESULT: 3 Player B wins\nMATCH_RESULT: Player A wins 1-0\n")
    # bo1 must count GAME_RESULT lines only (and never MATCH_RESULT ones).
    if _parse_gate_output(bo1_out, bo3=False) != (1, 1, 1):
        print("FAIL: bo1 GAME_RESULT parse", file=sys.stderr)
        return 1
    print("PASS [parse]: MATCH_RESULT/GAME_RESULT tallies")
    return 0


def main() -> int:
    if not os.path.exists(ACTOR_BIN):
        print(f"SKIP: {ACTOR_BIN} not built (make actor)")
        return 0
    rc = _test_parse()
    if rc:
        return rc

    with tempfile.TemporaryDirectory() as td:
        cand = _fresh_net_ckpt(td, "cand", seed=0)
        inc = _fresh_net_ckpt(td, "inc", seed=1)

        # --actor with no incumbent is a loud error (Python-only fallback).
        try:
            az_eval(DECK, candidate=cand,
                    incumbent=os.path.join(td, "missing.pt"),
                    games=2, sims=4, worlds=2, roster=[DECK], cross_pairs=0,
                    bo3=False, use_actor=True, eval_server=False, workers=1)
        except ValueError as exc:
            print(f"PASS [guard]: --actor without incumbent raises ({exc})")
        else:
            print("FAIL: --actor with no incumbent did not raise",
                  file=sys.stderr)
            return 1

        # One-mirror panel (cross_pairs=0), bo1, tiny search budget, no eval
        # server (this box may have no GPU). games=2 -> per=2: one match per
        # seat orientation.
        kw = dict(games=2, sims=4, worlds=2, roster=[DECK], cross_pairs=0,
                  bo3=False, use_actor=True, eval_server=False,
                  cross_world=True, workers=2, seed=7)
        r1 = az_eval(DECK, candidate=cand, incumbent=inc, **kw)
        n = r1["wins"] + r1["losses"] + r1["draws"]
        if n != 2:
            print(f"FAIL: expected 2 tallied games, got {n} ({r1})",
                  file=sys.stderr)
            return 1
        if list(r1["per_deck"].keys()) != [DECK] or len(r1["breakdown"]) != 1:
            print(f"FAIL: panel shape wrong ({r1})", file=sys.stderr)
            return 1
        print(f"PASS [actor-gate]: mirror panel tallied "
              f"{r1['wins']}W-{r1['losses']}L-{r1['draws']}D "
              f"(win_rate={r1['win_rate']:.3f})")

        r2 = az_eval(DECK, candidate=cand, incumbent=inc, **kw)
        for k in ("wins", "losses", "draws", "win_rate", "breakdown",
                  "per_deck", "vetoes", "promoted"):
            if r1[k] != r2[k]:
                print(f"FAIL: actor gate not deterministic ({k}: "
                      f"{r1[k]!r} vs {r2[k]!r})", file=sys.stderr)
                return 1
        print("PASS [determinism]: identical rerun returned the identical "
              "gate result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
