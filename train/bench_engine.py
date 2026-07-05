"""Lean engine throughput benchmark (no narrative decoding / printing).

Drives RoboMageEnv through runner.drive_game (the shared loop) with the
scripted HARD agent on both seats, which exercises the same C++ engine path
training uses (subprocess + --machine BQUERY serialize + full SBA/legal-action
computation). Reports wall time, games/sec and decisions/sec. Deterministic
per --seed.
"""
import sys
import time
import argparse

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.realpath(__file__)))

from env import RoboMageEnv  # noqa: E402
from opponents import ScriptedController  # noqa: E402
from scripted_agent import make_agent  # noqa: E402
import runner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck-a", default="delver")
    ap.add_argument("--deck-b", default="delver")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-decisions", type=int, default=4000)
    args = ap.parse_args()

    ctrl = ScriptedController(make_agent("scripted"))
    total_decisions = 0
    completed = 0
    t0 = time.perf_counter()
    for g in range(args.games):
        env = RoboMageEnv(deck_a=args.deck_a, deck_b=args.deck_b)
        obs, info = env.reset(seed=args.seed + g)
        rec = runner.drive_game(env, obs, ctrl, ctrl,
                                max_decisions=args.max_decisions)
        total_decisions += rec.decisions
        if not rec.capped:
            completed += 1
        try:
            env.close()
        except Exception:
            pass
    dt = time.perf_counter() - t0

    print(f"games={args.games} completed={completed} decisions={total_decisions} "
          f"wall={dt:.3f}s games/s={args.games / dt:.2f} "
          f"decisions/s={total_decisions / dt:.0f} "
          f"ms/decision={1000 * dt / max(1, total_decisions):.3f}")


if __name__ == "__main__":
    main()
