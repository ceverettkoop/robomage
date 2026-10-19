"""The teacher corpus (docs/rnad_training_plan.md §6 B0): scripted:hard games as trajectory episodes.

    python -m rmlearn.corpus --out runs/bc1/corpus --actors 16 --matches 2000 [--explore-frac 0.2]

Two routes, both writing ``<out>/episodes/*.rmtraj`` and keeping every decision log in
``<out>/rmlog/``:

- ``actor`` (the default, measured faster): engine actors (``--actor``, ``docs/actor.md``) whose
  seats are ``first`` with the scripted teacher (``python/teacher.py``) at ``teacher_mix: 1``, so the
  teacher plays them and every step carries its label (``teacher``) and a one-hot μ; the episodes
  are written as the games end. Scripted-vs-explore entries (``--explore-frac``) put ``explore`` on
  one side, untaught and unrecorded (``trajectory_seats`` the scripted side), for coverage of bad
  positions. Each actor's teacher gets its own ``--seed`` (the hard tier's tie-breaks).
- ``driver``: ``python/rm_driver.py --player-a scripted --player-b scripted --rmlog-dir`` per
  matchup (bo3), then ``robomage --replay LOG --trajectory-out`` per log; μ and the teacher label
  are absent and the label is the reply (``LabelConfig.chosen_without_mu``).

The league is the 10 ``league/`` decks: every pair (45) with the decks' seats drawn per match
(``swap: decks``); the explore entries every ordered pair. ``make_schedule`` is the schedule the
actor route writes, for the orchestrator to reuse. ``run_actor_corpus`` / ``run_driver_corpus``
return a summary dict (games, decisions, seconds, decisions/s, bytes per decision).
"""

import argparse
import glob
import itertools
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
BIN_DIR = os.path.join(REPO_ROOT, "bin")
LEAGUE = ["bug", "burn", "bw_dnt", "gw_maverick", "lands", "reanimator", "ur_delver", "uw_control", "wrb_energy",
          "wubg_doomsday"]
CPU_ENV = dict(os.environ, HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")


def teacher_command(tier="hard", seed=0):
    """The scripted teacher's command line, relative to the engine's working directory (bin/)."""
    return "python3 ../python/teacher.py --tier %s --seed %d" % (tier, seed)


def make_schedule(out_dir, decks=LEAGUE, explore_frac=0.0, omniscient=True, teacher=None, fmt="bo3",
                  version="b0", knowledge="expert", max_game_decisions=5000):
    """The actor schedule of the teacher corpus (a dict; docs/actor.md §2): scripted vs scripted over
    every pair of `decks` and, with weight share `explore_frac`, scripted vs explore over every ordered
    pair, the explore side unrecorded. Outputs under `out_dir` (absolute)."""
    teacher = teacher or teacher_command()
    pairs = list(itertools.combinations(decks, 2))
    ordered = list(itertools.permutations(decks, 2))
    entries = []
    w_ss = (1.0 - explore_frac) / len(pairs)
    for a, b in pairs:
        entries.append({"id": "ss-%s-%s" % (a, b), "weight": w_ss, "deck_a": "league/" + a, "deck_b": "league/" + b,
                        "swap": "decks", "agent_a": "first", "agent_b": "first", "teacher_a": teacher,
                        "teacher_b": teacher, "teacher_mix": 1})
    if explore_frac > 0:
        w_se = explore_frac / len(ordered)
        for a, b in ordered:
            entries.append({"id": "se-%s-%s" % (a, b), "weight": w_se, "deck_a": "league/" + a,
                            "deck_b": "league/" + b, "swap": "sides", "agent_a": "first", "agent_b": "explore",
                            "teacher_a": teacher, "teacher_mix": 1, "trajectory_seats": "A"})
    return {"version": version,
            "outputs": {"train": {"rmlog_dir": os.path.join(out_dir, "rmlog"),
                                  "trajectory_dir": os.path.join(out_dir, "episodes")}},
            "defaults": {"format": fmt, "trajectory_omniscient": bool(omniscient), "knowledge_a": knowledge,
                         "knowledge_b": knowledge, "max_game_decisions": max_game_decisions},
            "entries": entries}


def _write_json(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w") as fp:
        json.dump(doc, fp, indent=1)
    os.replace(tmp, path)


def _summary(out_dir, t0, games):
    """The corpus's totals: games, recorded decisions (the episodes' steps), rate and size."""
    from .episodes import episode_attrs
    files = glob.glob(os.path.join(out_dir, "episodes", "*.rmtraj"))
    size = sum(os.path.getsize(f) for f in files)
    decisions = sum(int(episode_attrs(f)["decisions"]) for f in files)
    secs = time.time() - t0
    return {"games": games, "episodes": len(files), "decisions": decisions, "seconds": round(secs, 1),
            "decisions_per_s": round(decisions / max(secs, 1e-9), 1),
            "bytes_per_decision": round(size / max(decisions, 1), 1), "bytes": size}


def run_actor_corpus(out_dir, actors=16, matches=2000, explore_frac=0.2, omniscient=True, seed=1,
                     engine=None, bin_dir=BIN_DIR, tier="hard", log=print):
    """Plays `matches` scripted matches over `actors` engine actor processes into `out_dir`; the summary."""
    out_dir = os.path.abspath(out_dir)
    engine = os.path.abspath(engine or os.path.join(bin_dir, "release", "robomage"))
    os.makedirs(os.path.join(out_dir, "actors"), exist_ok=True)
    per = [matches // actors + (1 if k < matches % actors else 0) for k in range(actors)]
    procs = []
    t0 = time.time()
    for k in range(actors):
        if not per[k]:
            continue
        sched = os.path.join(out_dir, "actors", "schedule.%d.json" % k)
        _write_json(sched, make_schedule(out_dir, explore_frac=explore_frac, omniscient=omniscient,
                                         teacher=teacher_command(tier, seed * 1000 + k)))
        argv = [engine, "--actor", "--actor-schedule", sched, "--actor-out", os.path.join(out_dir, "actors"),
                "--actor-id", str(k), "--actor-seed", str(seed), "--actor-matches", str(per[k])]
        err = open(os.path.join(out_dir, "actors", "a%d.stderr.txt" % k), "w")
        procs.append((k, subprocess.Popen(argv, cwd=bin_dir, stdout=subprocess.DEVNULL, stderr=err, env=CPU_ENV), err))
    failed = []
    for k, p, err in procs:
        if p.wait() != 0:
            failed.append(k)
        err.close()
    games = 0
    for k, _, _ in procs:
        with open(os.path.join(out_dir, "actors", "a%d.status.json" % k)) as fp:
            games += json.load(fp)["games"]
    s = _summary(out_dir, t0, games)
    s.update({"route": "actor", "actors": actors, "matches": matches, "failed_actors": failed})
    log("corpus: %s" % json.dumps(s))
    return s


def run_driver_corpus(out_dir, jobs=16, matches_per_pair=4, omniscient=True, seed=1, engine=None, bin_dir=BIN_DIR,
                      log=print):
    """The rm_driver route: scripted vs scripted bo3 matches per league pair (both deck orders), the
    logs replayed into episodes, `jobs` processes at a time; the summary."""
    out_dir = os.path.abspath(out_dir)
    engine = os.path.abspath(engine or os.path.join(bin_dir, "release", "robomage"))
    logs = os.path.join(out_dir, "rmlog")
    eps = os.path.join(out_dir, "episodes")
    os.makedirs(eps, exist_ok=True)
    t0 = time.time()
    cmds = []
    for i, (a, b) in enumerate(itertools.permutations(LEAGUE, 2)):
        cmds.append([sys.executable, os.path.join(REPO_ROOT, "python", "rm_driver.py"), "--player-a", "scripted",
                     "--player-b", "scripted", "--deck-a", "league/" + a, "--deck-b", "league/" + b, "--format",
                     "bo3", "--games", str(matches_per_pair), "--seed", str(seed * 100000 + i * 1000), "--quiet",
                     "--rmlog-dir", logs, "--binary", engine])
    _run_pool(cmds, jobs, REPO_ROOT)
    t1 = time.time()
    flags = ["--trajectory-omniscient"] if omniscient else []
    _run_pool([[engine, "--replay", f, "--trajectory-out", eps] + flags for f in sorted(glob.glob(logs + "/*.rmlog"))],
              jobs, bin_dir)
    s = _summary(out_dir, t0, len(glob.glob(os.path.join(eps, "*.rmtraj"))))
    s.update({"route": "driver", "jobs": jobs, "play_seconds": round(t1 - t0, 1)})
    log("corpus: %s" % json.dumps(s))
    return s


def _run_pool(cmds, jobs, cwd):
    running = []
    failed = 0
    for argv in cmds:
        while len(running) >= jobs:
            for p in list(running):
                if p.poll() is not None:
                    failed += p.returncode != 0
                    running.remove(p)
            time.sleep(0.02)
        running.append(subprocess.Popen(argv, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                        env=CPU_ENV))
    for p in running:
        failed += p.wait() != 0
    if failed:
        raise RuntimeError("%d of %d corpus processes failed" % (failed, len(cmds)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--route", choices=("actor", "driver"), default="actor")
    ap.add_argument("--actors", type=int, default=16, help="actor processes (driver: parallel jobs)")
    ap.add_argument("--matches", type=int, default=2000, help="actor route: bo3 matches in all")
    ap.add_argument("--matches-per-pair", type=int, default=4, help="driver route: matches per ordered pair")
    ap.add_argument("--explore-frac", type=float, default=0.2)
    ap.add_argument("--no-omniscient", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--engine", help="the engine binary (default bin/release/robomage)")
    ap.add_argument("--tier", default="hard")
    a = ap.parse_args()
    if a.route == "actor":
        run_actor_corpus(a.out, a.actors, a.matches, a.explore_frac, not a.no_omniscient, a.seed, a.engine, tier=a.tier)
    else:
        run_driver_corpus(a.out, a.actors, a.matches_per_pair, not a.no_omniscient, a.seed, a.engine)


if __name__ == "__main__":
    main()
