#!/usr/bin/env python3
"""AZ+MCTS baseline sweep: the AZ generalist (with search) vs scripted:hard,
over every league matchup.

`train.py baseline`/`baseline --all` are PPO-only (they call MaskablePPO.load),
so they can't evaluate the AZ net. This mirrors `baseline_all`'s structure — the
full N×N league cross product, mirrors included, with per-game SEAT ALTERNATION
so neither side gets a systematic on-the-play edge (important given the strong
second-player bias in self-play) — but drives the model seat with an AZ+MCTS
controller built from the `az:<...>` spec grammar (opponents.make_controller).

Run from the repo root:
    train/.venv/bin/python train/az_baseline.py --sims 128 --games 20
    train/.venv/bin/python train/az_baseline.py --model "az:gen?sims=192&worlds=4"
"""
import argparse
import datetime
import itertools
import os

import runner
from opponents import make_controller


def _league_roster(decks_dir: str) -> list[str]:
    """('league/<stem>' for every decks/league/*.dk) — the same roster the
    league trains and baseline_all evaluates."""
    ldir = os.path.join(decks_dir, "league")
    if not os.path.isdir(ldir):
        return []
    return sorted("league/" + os.path.splitext(p)[0]
                  for p in os.listdir(ldir) if p.endswith(".dk"))


def _wld_line(w: int, l: int, d: int) -> str:
    total = w + l + d
    pct = 100 * w / total if total else 0.0
    return f"{w}W/{l}L/{d}D  {pct:.1f}% win rate"


def matchup(model_spec: str, deck: str, opp_deck: str, *, binary_path: str,
            n_games: int, seed: int | None):
    """One matchup: the AZ model piloting `deck` vs scripted:hard piloting
    `opp_deck`, seats alternating each game. Returns (w, l, d) from the model's
    perspective. Controllers are rebuilt per game so the search controller's
    per-match clock/stats reset cleanly."""
    wins = losses = draws = 0
    for i in range(n_games):
        model_is_a = (i % 2 == 0)
        ctrl_model = make_controller(model_spec)
        ctrl_scripted = make_controller("scripted")
        ctrl_a, ctrl_b = ((ctrl_model, ctrl_scripted) if model_is_a
                          else (ctrl_scripted, ctrl_model))
        deck_a, deck_b = ((deck, opp_deck) if model_is_a else (opp_deck, deck))
        w, l, d = runner.run_games(
            ctrl_a, ctrl_b,
            label_a="Model" if model_is_a else "Scripted",
            label_b="Scripted" if model_is_a else "Model",
            binary_path=binary_path, deck_a=deck_a, deck_b=deck_b,
            n_games=1, seed=(seed + i) if seed is not None else None,
            transcript="quiet")
        # run_games returns W/L/D from Player A's perspective.
        wins += w if model_is_a else l
        losses += l if model_is_a else w
        draws += d
        print(f"\r  {deck} vs {opp_deck}: game {i+1}/{n_games}  "
              f"W:{wins} L:{losses} D:{draws}", end="", flush=True)
    print()
    return wins, losses, draws


def sweep(model_spec: str, *, binary_path: str, decks_dir: str, n_games: int,
          seed: int | None, log_path: str):
    roster = _league_roster(decks_dir)
    if not roster:
        print(f"No league decks found under {os.path.join(decks_dir, 'league')}")
        return

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = (f"=== AZ+MCTS baseline ({model_spec}) vs scripted:hard "
              f"({len(roster)}x{len(roster)} matchups) — {stamp} — "
              f"{n_games} games/matchup, seed={seed} ===")
    lines = [header]
    print(header, flush=True)

    per_model_deck = {d: [0, 0, 0] for d in roster}
    per_opp_deck = {d: [0, 0, 0] for d in roster}

    for deck, opp in itertools.product(roster, roster):
        w, l, d = matchup(model_spec, deck, opp, binary_path=binary_path,
                          n_games=n_games, seed=seed)
        lines.append(f"az piloting {deck:<22} vs scripted:hard {opp:<22} "
                     + _wld_line(w, l, d))
        for tally in (per_model_deck[deck], per_opp_deck[opp]):
            tally[0] += w
            tally[1] += l
            tally[2] += d

    lines.append("")
    lines.append("per model deck (az piloting it vs the whole scripted field):")
    for deck, (w, l, d) in per_model_deck.items():
        lines.append(f"  {deck:<22} " + _wld_line(w, l, d))
    lines.append("per opponent deck (scripted:hard piloting it vs every az deck; "
                 "win rate is still the model's):")
    for deck, (w, l, d) in per_opp_deck.items():
        lines.append(f"  {deck:<22} " + _wld_line(w, l, d))

    summary = "\n".join(lines)
    with open(log_path, "a") as f:
        f.write(summary + "\n\n")
    print(f"\n{summary}\n\nreport appended to {log_path}", flush=True)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="AZ+MCTS baseline sweep vs scripted:hard")
    ap.add_argument("--model", default=None,
                    help="AZ controller spec (default built from --sims/--worlds, "
                         "e.g. 'az:gen?sims=128&worlds=4'); pass an explicit spec to override")
    ap.add_argument("--sims", type=int, default=128, help="MCTS sims/move (default 128)")
    ap.add_argument("--worlds", type=int, default=4, help="determinized worlds (default 4)")
    ap.add_argument("--games", type=int, default=20, help="games per matchup (default 20)")
    ap.add_argument("--seed", type=int, default=1, help="base seed (game i uses seed+i)")
    ap.add_argument("--binary", default=os.path.join(repo, "bin", "robomage"))
    ap.add_argument("--decks-dir", default=os.path.join(repo, "bin", "resources", "decks"))
    ap.add_argument("--log", default=os.path.join(repo, "train", "checkpoints",
                                                  "az_baseline_report.log"))
    args = ap.parse_args()

    spec = args.model or f"az:gen?sims={args.sims}&worlds={args.worlds}"
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    sweep(spec, binary_path=args.binary, decks_dir=args.decks_dir,
          n_games=args.games, seed=args.seed, log_path=args.log)
