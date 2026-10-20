"""The end-of-run benchmark (docs/rmtrain.md §7.1): a run's policies against scripted:hard over every
league pair, the same matches for each, scored per game and per deck.

    learn/rmtrain bench <run> [--wait] [--policies bc-final,m3,final,final@thresh=0.03:disc=16]
    python -m rmlearn.benchmark --run runs/league12h [--matches 480] [--actors 12] [--label end]

A policy is an export name of the run (``exports/<name>.ts.pt``: ``bc-final``, ``m<k>``, ``final``) or
an export path, optionally with ``@<nn options>`` (default ``argmax``). The default set is the
DAgger policy (``bc-final``, the baseline), every R-NaD fixed point ``m<k>``, and the final network
at argmax and with the fine-tuning projection (the run's ``eval.nn_options_finetune``). Each policy
plays `matches` matches (side a, both decks of each pair and both seats: ``matches.eval_schedule``)
on `actors` actors with the same run seed, actor ids and match counts, so every policy meets the
same deck pairs, seats and game seeds (common random numbers): differences between policies are
the policies'. ``--wait`` waits for the run to stop first (its shutdown finishes the final export).

Writes ``<run>/bench/<label>/``: per policy a directory (the actors' files, ``report.json`` — the
``matches.report`` —, ``report.txt``), then ``summary.json`` and ``summary.md``: per policy the
per-game win rate (Wilson 95%, match-clustered SE), game 1 apart from games 2–3; per deck the
policy's win rate playing it and facing it, each policy's difference from the baseline with its
two-SE half-width; per policy the deck-vs-deck matrix. A policy whose ``report.json`` exists is not
played again (an interrupted benchmark resumes).
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import time

from . import matches as MT
from .corpus import LEAGUE
from .orchestrator.control import alive
from .orchestrator.rundir import RunDir, read_json, resolve_run

DEFAULT_MATCHES = 480
DEFAULT_ACTORS = 12


def default_policies(run):
    """bc-final, every exported fixed point m<k> (in order), final at argmax and projected."""
    names = ["bc-final"]
    fixed = [os.path.basename(p)[:-len(".ts.pt")] for p in glob.glob(os.path.join(run.exports, "m*.ts.pt"))]
    names += sorted((n for n in fixed if re.fullmatch(r"m\d+", n)), key=lambda n: int(n[1:]))
    names.append("final")
    proj = ((read_json(run.config) or {}).get("eval") or {}).get("nn_options_finetune", "thresh=0.03:disc=16")
    if proj:
        names.append("final@" + proj)
    return names


def resolve_policy(run, item):
    """(label, export path, nn options) of a policy item ``name[@options]``."""
    name, _, opts = item.partition("@")
    opts = opts or "argmax"
    path = name if MT.is_export(name) else os.path.join(run.exports, name + ".ts.pt")
    label = os.path.basename(name)[:-len(".ts.pt")] if name.endswith(".ts.pt") else os.path.basename(name)
    if opts != "argmax":
        label += "@" + opts
    return label, os.path.abspath(path), opts


def slug(label):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", label)


def play_policy(label, path, opts, out_dir, matches, actors, seed, engine=None, extra_args=(), log=print):
    """The policy's report (``matches.report``) against scripted:hard over every league pair; reuses
    an existing ``report.json``."""
    done = os.path.join(out_dir, "report.json")
    rep = read_json(done)
    if rep is not None:
        log("%s: already played (%s)" % (label, done))
        return rep
    sched = MT.eval_schedule(path, "scripted", out_dir, pairs=MT.league_pairs(), policy_options=opts,
                             version="bench")
    per = int(math.ceil(matches / float(actors)))
    pool = MT.ActorPool(sched, out_dir, actors=actors, engine=engine, seed=seed, matches_per_actor=per,
                        extra_args=extra_args)
    t0 = time.time()
    log("%s: %d matches on %d actors in %s" % (label, per * actors, actors, out_dir))
    pool.start()
    pool.wait()
    lines = pool.poll()
    failed = pool.failures()
    if failed:
        log("%s: %d actors failed: %s" % (label, len(failed), json.dumps(failed)[:2000]))
    rep = MT.report([MT.game_record(ln) for ln in lines])
    rep["policy"] = {"label": label, "export": path, "options": opts, "seconds": round(time.time() - t0, 1),
                     "failed_actors": len(failed)}
    with open(os.path.join(out_dir, "report.txt"), "w") as fp:
        fp.write(MT.format_report(rep, "%s vs scripted:hard" % label) + "\n")
    MT.write_json(done, rep)
    a = rep["all"]
    log("%s: %s over %d games (%.0f s)" % (label, _pct(a), a["decisive"], time.time() - t0))
    return rep


def by_opponent_deck(rep):
    """{opponent deck: tally} from a report's by_matchup groups (sums of their counts)."""
    out = {}
    for key, t in rep["by_matchup"].items():
        opp = key.split(" vs ", 1)[1]
        o = out.setdefault(opp, {"wins": 0, "losses": 0})
        o["wins"] += t["wins"]
        o["losses"] += t["losses"]
    for o in out.values():
        n = o["wins"] + o["losses"]
        o["decisive"] = n
        o["win_rate"] = o["wins"] / n if n else None
    return out


def _se(t):
    """The match-clustered SE when there is one, else the binomial one."""
    return t.get("se_clustered") or t.get("se")


def _pct(t):
    return "-" if t.get("win_rate") is None else "%.3f" % t["win_rate"]


def _cell(t):
    return "-" if not t or t.get("win_rate") is None else "%.3f (%d)" % (t["win_rate"], t["decisive"])


def summarize(reports, baseline=None):
    """The summary dict of {label: report} (in order; the first is the baseline unless named)."""
    labels = list(reports)
    base = baseline or labels[0]
    seen = {d for r in reports.values() for k in r["by_matchup"] for d in k.split(" vs ", 1)}
    decks = sorted(seen | set(LEAGUE), key=lambda d: LEAGUE.index(d) if d in LEAGUE else len(LEAGUE))
    out = {"baseline": base, "policies": {}, "decks": decks}
    for lab, rep in reports.items():
        p = {"all": rep["all"], "game1": rep["game1"], "games23": rep["games23"], "plays": rep["by_deck"],
             "faces": by_opponent_deck(rep), "matchups": rep["by_matchup"], "findings": rep["findings"],
             "delta_plays": {}}
        if lab != base:
            for d in decks:
                t, b = rep["by_deck"].get(d), reports[base]["by_deck"].get(d)
                if t and b and t["win_rate"] is not None and b["win_rate"] is not None:
                    p["delta_plays"][d] = {"delta": t["win_rate"] - b["win_rate"],
                                           "two_se": 2 * math.hypot(_se(t) or 0, _se(b) or 0)}
            a, b = rep["all"], reports[base]["all"]
            if a["win_rate"] is not None and b["win_rate"] is not None:
                p["delta_all"] = {"delta": a["win_rate"] - b["win_rate"],
                                  "two_se": 2 * math.hypot(_se(a) or 0, _se(b) or 0)}
        out["policies"][lab] = p
    return out


def format_summary(s, title):
    """summary.md: the tables of ``summarize``."""
    labs = list(s["policies"])
    base = s["baseline"]
    L = ["# %s" % title, "",
         "Every policy against scripted:hard over all %d league pairs, both decks and seats, the same "
         "matches for each (common random numbers). Win rates are per decisive game; draws and caps "
         "are excluded and listed under findings. ± is two standard errors (match-clustered)." % len(
             MT.league_pairs()), "",
         "## Overall", "",
         "| policy | games | win rate | 95%% (Wilson) | ± 2 SE | game 1 | games 2–3 | Δ vs %s | draws | caps |" % base,
         "|---|---|---|---|---|---|---|---|---|---|"]
    for lab in labs:
        p = s["policies"][lab]
        a = p["all"]
        lo, hi = a["wilson95"] or (0, 0)
        d = p.get("delta_all")
        L.append("| %s | %d | %s | %.3f–%.3f | %s | %s | %s | %s | %d | %d |" % (
            lab, a["decisive"], _pct(a), lo, hi, "%.3f" % (2 * _se(a)) if _se(a) else "-", _pct(p["game1"]),
            _pct(p["games23"]), "%+.3f ± %.3f" % (d["delta"], d["two_se"]) if d else "baseline", a["draws"],
            a["caps"]))
    for head, key, note in (("The policy playing each deck", "plays", "win rate (decisive games)"),
                            ("The policy facing each deck", "faces", "win rate (decisive games)")):
        L += ["", "## %s" % head, "", "%s; rows are decks, columns policies." % note, "",
              "| deck | " + " | ".join(labs) + " |", "|---|" + "---|" * len(labs)]
        for dk in s["decks"]:
            L.append("| %s | " % dk + " | ".join(_cell(s["policies"][lab][key].get(dk)) for lab in labs) + " |")
    others = [lab for lab in labs if lab != base]
    if others:
        L += ["", "## Difference from %s, playing each deck" % base, "",
              "Δ win rate ± two SEs; a difference larger than its ± is unlikely to be noise.", "",
              "| deck | " + " | ".join(others) + " |", "|---|" + "---|" * len(others)]
        for dk in s["decks"]:
            cells = []
            for lab in others:
                d = s["policies"][lab]["delta_plays"].get(dk)
                cells.append("%+.3f ± %.3f" % (d["delta"], d["two_se"]) if d else "-")
            L.append("| %s | " % dk + " | ".join(cells) + " |")
    ab = {d: d[:6] for d in s["decks"]}
    for lab in labs:
        p = s["policies"][lab]
        L += ["", "## Matchups: %s" % lab, "", "Rows: the policy's deck; columns: scripted:hard's deck.", "",
              "| deck | " + " | ".join(ab[d] for d in s["decks"]) + " |", "|---|" + "---|" * len(s["decks"])]
        for x in s["decks"]:
            row = []
            for y in s["decks"]:
                t = p["matchups"].get("%s vs %s" % (x, y))
                row.append("-" if x == y or not t or t["win_rate"] is None else "%.2f" % t["win_rate"])
            L.append("| %s | " % x + " | ".join(row) + " |")
    fnd = [(lab, f) for lab in labs for f in s["policies"][lab]["findings"]]
    if fnd:
        L += ["", "## Findings (draws and caps)", ""]
        L += ["- %s: %s, game %d after %d decisions: `%s`" % (lab, f["end"], f["game"], f["decisions"], f["rmlog"])
              for lab, f in fnd]
    return "\n".join(L) + "\n"


def wait_for_run(run, poll=60, log=print):
    if alive(run.control_sock):
        log("waiting for run %s to stop" % run.name)
    while alive(run.control_sock):
        time.sleep(poll)


def run_benchmark(run, policies=None, label="end", matches=DEFAULT_MATCHES, actors=DEFAULT_ACTORS, seed=11,
                  engine=None, extra_args=(), wait=False, log=print):
    """Plays every policy (``play_policy``) and writes the summary; the summary dict."""
    if wait:
        wait_for_run(run, log=log)
    elif alive(run.control_sock):
        log("note: run %s is running; the benchmark's actors compete with its actors for the CPU" % run.name)
    items = policies or default_policies(run)
    root = os.path.join(run.path, "bench", label)
    os.makedirs(root, exist_ok=True)
    reports = {}
    for item in items:
        lab, path, opts = resolve_policy(run, item)
        if not os.path.exists(path):
            log("%s: no export %s; skipped" % (lab, path))
            continue
        reports[lab] = play_policy(lab, path, opts, os.path.join(root, slug(lab)), matches, actors, seed,
                                   engine=engine, extra_args=extra_args, log=log)
    if not reports:
        raise SystemExit("benchmark: no policy to play (%s)" % ", ".join(items))
    s = summarize(reports)
    s.update({"run": run.path, "label": label, "matches": matches, "actors": actors, "seed": seed,
              "opponent": "scripted:hard", "at": time.time()})
    MT.write_json(os.path.join(root, "summary.json"), s)
    with open(os.path.join(root, "summary.md"), "w") as fp:
        fp.write(format_summary(s, "Benchmark %s of %s" % (label, run.name)))
    log("summary: %s" % os.path.join(root, "summary.md"))
    return s


def add_args(ap):
    ap.add_argument("--policies", help="comma-separated name[@options] (default: bc-final, every m<k>, final, "
                                       "final@<eval.nn_options_finetune>)")
    ap.add_argument("--label", default="end", help="the benchmark's directory under <run>/bench/")
    ap.add_argument("--matches", type=int, default=DEFAULT_MATCHES, help="matches per policy")
    ap.add_argument("--actors", type=int, default=DEFAULT_ACTORS)
    ap.add_argument("--seed", type=int, default=11, help="the actors' run seed (the same for every policy)")
    ap.add_argument("--engine", help="robomage_nn (default bin/release/robomage_nn)")
    ap.add_argument("--feature-history", type=int, default=0, help="the policies' recent-event history cap")
    ap.add_argument("--wait", action="store_true", help="wait for the run to stop first")


def run_from_args(run, a):
    extra = ["--feature-history", str(a.feature_history)] if a.feature_history > 0 else []
    pol = [p for p in a.policies.split(",") if p] if a.policies else None
    run_benchmark(run, pol, a.label, a.matches, a.actors, a.seed, a.engine, extra, a.wait,
                  log=lambda m: print(time.strftime("%Y-%m-%d %H:%M:%S ") + m, flush=True))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="a run name (runs/<name>) or path")
    add_args(ap)
    a = ap.parse_args(argv)
    return run_from_args(RunDir(resolve_run(a.run)), a)


if __name__ == "__main__":
    sys.exit(main())
