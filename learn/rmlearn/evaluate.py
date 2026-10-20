"""Evaluation games of an exported policy against fixed opponents through engine actors
(``--actor``, docs/actor.md), scored per game (docs/rnad_training_plan.md §6 B1, T8).

    s = run_eval("run/ckpt/p1.pt2", "run/eval/p1", opponents=("scripted", "explore"), matches=50)
    s["scripted"]["win_rate"], s["scripted"]["game1"]["win_rate"], s["scripted"]["games23"]

    python -m rmlearn.evaluate --model run/ckpt/p1.pt2 --out run/eval/p1 [--opponents scripted,explore,random]
        [--pairs ur_delver-burn,uw_control-wrb_energy] [--feature-history N]

The policy plays as ``nn:<model>`` in process (``robomage_nn``, CPU, the GPU hidden; ``--nn-options``
e.g. ``argmax`` or ``temp=1``) against each opponent: ``scripted`` (side ``first`` with the scripted
teacher at ``teacher_mix: 1``, i.e. scripted:hard), ``explore``, ``random`` or any agent spec. Each
opponent is two ``eval`` schedule entries per deck pair, one per order, each with ``swap: sides``
(which moves a side's deck with it, so the second order gives the policy the other deck: it plays
both decks from both seats). ``summarize`` reads the actors' results files: per opponent the games won,
lost, drawn or capped, the win rate (draws and caps count half), game 1 (pre-board) apart from games
2–3 (post-board; game 3 is played only at 1–1, so a selected sample), and per deck pair. A policy
trained on the recent-event history tokens (docs/featurizer.md §3.7) needs ``feature_history``, the
engine's ``--feature-history``, at its training value.
"""

import argparse
import glob
import json
import os
import subprocess
import time

from .corpus import BIN_DIR, CPU_ENV, teacher_command

DEFAULT_PAIRS = (("ur_delver", "burn"), ("bug", "bw_dnt"), ("uw_control", "wrb_energy"), ("gw_maverick", "lands"))


def opponent_side(name):
    """(agent spec, teacher command or None) of an opponent name."""
    if name.startswith("scripted"):
        tier = name.split(":", 1)[1] if ":" in name else "hard"
        return "first", teacher_command(tier)
    return name, None


def make_eval_schedule(model_spec, out_dir, opponents, pairs=DEFAULT_PAIRS, fmt="bo3", version="eval"):
    entries = []
    for opp in opponents:
        agent, teacher = opponent_side(opp)
        # Both orders of each pair: swap: sides moves a side's deck with its agent (side a keeps deck a).
        for a, b in [p for pair in pairs for p in (tuple(pair), tuple(pair)[::-1])]:
            e = {"id": "%s|%s-%s" % (opp, a, b), "purpose": "eval", "deck_a": "league/" + a, "deck_b": "league/" + b,
                 "swap": "sides", "agent_a": model_spec, "agent_b": agent}
            if teacher:
                e.update({"teacher_b": teacher, "teacher_mix": 1})
            entries.append(e)
    return {"version": version, "outputs": {"eval": {"rmlog_dir": os.path.join(out_dir, "rmlog")}},
            "defaults": {"format": fmt}, "entries": entries}


def run_eval(model, out_dir, opponents=("scripted", "explore", "random"), matches=48, actors=12, nn_options="argmax",
             pairs=DEFAULT_PAIRS, seed=11, engine=None, bin_dir=BIN_DIR, log=print, feature_history=0):
    """Plays `matches` matches per opponent (spread over the opponents' entries by weight) over
    `actors` processes; summarize()'s dict."""
    out_dir = os.path.abspath(out_dir)
    engine = os.path.abspath(engine or os.path.join(bin_dir, "release", "robomage_nn"))
    spec = "nn:%s%s" % (os.path.abspath(model), ":" + nn_options if nn_options else "")
    os.makedirs(out_dir, exist_ok=True)
    sched = os.path.join(out_dir, "schedule.json")
    with open(sched + ".tmp", "w") as fp:
        json.dump(make_eval_schedule(spec, out_dir, opponents, pairs), fp, indent=1)
    os.replace(sched + ".tmp", sched)
    total = matches * len(opponents)
    per = [total // actors + (1 if k < total % actors else 0) for k in range(actors)]
    t0 = time.time()
    procs = []
    for k in range(actors):
        if not per[k]:
            continue
        argv = [engine, "--actor", "--actor-schedule", sched, "--actor-out", out_dir, "--actor-id", str(k),
                "--actor-seed", str(seed), "--actor-matches", str(per[k])]
        if feature_history > 0:
            argv += ["--feature-history", str(feature_history)]
        err = open(os.path.join(out_dir, "a%d.stderr.txt" % k), "w")
        procs.append((subprocess.Popen(argv, cwd=bin_dir, stdout=subprocess.DEVNULL, stderr=err, env=CPU_ENV), err))
    failed = 0
    for p, err in procs:
        failed += p.wait() != 0
        err.close()
    s = summarize(out_dir)
    s["seconds"] = round(time.time() - t0, 1)
    s["failed_actors"] = failed
    log("eval: %s" % json.dumps(s))
    return s


def _rate(c):
    n = c["win"] + c["loss"] + c["draw"]
    return (c["win"] + 0.5 * c["draw"]) / n if n else None


def summarize(out_dir, subject=None):
    """Per opponent (the entry id's prefix) the policy's per-game record from the results files.
    The evaluated policy is the schedule's side a: its seat is the line's ``side_a_seat``, whatever
    its agent spec. `subject(agent_spec)`, when given, says instead which seat's agent it is."""
    out = {}
    for path in glob.glob(os.path.join(out_dir, "a*.results.jsonl")):
        with open(path) as fp:
            for line in fp:
                r = json.loads(line)
                opp, pair = r["entry"].split("|", 1)
                if subject is None:
                    seat = r.get("side_a_seat", "A").lower()
                else:
                    seat = "a" if subject(r["agent_a"]) else "b"
                res = r["result_" + seat]
                kind = "win" if res > 0 else "loss" if res < 0 else "draw"
                o = out.setdefault(opp, {"all": _zero(), "game1": _zero(), "games23": _zero(), "pairs": {}})
                for key in ("all", "game1" if r["game"] == 1 else "games23"):
                    o[key][kind] += 1
                    o[key]["caps"] += r["end_reason"] == "cap"
                p = o["pairs"].setdefault(pair, _zero())
                p[kind] += 1
    for o in out.values():
        for key in ("all", "game1", "games23"):
            o[key]["win_rate"] = _rate(o[key])
        for p in o["pairs"].values():
            p["win_rate"] = _rate(p)
        o["games"] = o["all"]["win"] + o["all"]["loss"] + o["all"]["draw"]
        o["win_rate"] = o["all"]["win_rate"]
    return out


def _zero():
    return {"win": 0, "loss": 0, "draw": 0, "caps": 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="an export (.pt2 or .ts.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opponents", default="scripted,explore,random")
    ap.add_argument("--matches", type=int, default=48, help="matches per opponent")
    ap.add_argument("--actors", type=int, default=12)
    ap.add_argument("--nn-options", default="argmax")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--engine", help="robomage_nn (default bin/release/robomage_nn)")
    ap.add_argument("--pairs", help="deck pairs, e.g. ur_delver-burn,uw_control-wrb_energy (default: four league pairs)")
    ap.add_argument("--feature-history", type=int, default=0, help="the policy's recent-event history cap")
    a = ap.parse_args()
    from .matches import parse_pairs
    pairs = parse_pairs(a.pairs) if a.pairs else DEFAULT_PAIRS
    run_eval(a.model, a.out, tuple(a.opponents.split(",")), a.matches, a.actors, a.nn_options, pairs=pairs,
             seed=a.seed, engine=a.engine, feature_history=a.feature_history)


if __name__ == "__main__":
    main()
