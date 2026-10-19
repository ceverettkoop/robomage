"""The ladder of past fixed points (docs/rnad_training_plan.md B4: "against the previous iterations'
fixed points"): exported policies kept with ids and versions, the current policy played against
each (or the last K) through the SPRT runner's actors, every pairing's per-game record kept, and a
Bradley–Terry rating over all of them.

    lad = Ladder("runs/r1/ladder")                                   # ladder.json there
    lad.add_fixed_point("it3", "runs/r1/ckpt/p30000.pt2", version=30000)
    lad.add_fixed_point("it3-proj", "runs/r1/ckpt/p30000.pt2", options="temp=1:thresh=0.03:disc=16")
    res = lad.evaluate_against_ladder("runs/r1/ckpt/p40000.pt2", policy_id="it4", last_k=5,
                                      cfg=SPRTConfig(max_games=100, sequential=False), add=True)
    lad.ratings()                                                     # {id: {"elo", "games", ...}}

    python -m rmlearn.ladder --dir runs/r1/ladder add --id bc --model runs/bc1/bc/ckpt/p1.pt2
    python -m rmlearn.ladder --dir runs/r1/ladder eval --id it4 --model p40000.pt2 --games 100 [--last-k 5] [--add]
    python -m rmlearn.ladder --dir runs/r1/ladder ratings

**A fixed point** is (``id``, ``model``: an export path or an agent spec, ``options``: the ``nn:``
options it plays with, ``version``, ``meta``); the same export with other options (argmax, sampled,
projected) is another fixed point. **A pairing** is one ``run_sprt`` between two of them (the
first as side a), by default a fixed budget (``sequential=False``: ``max_games`` games whatever the
test says; an SPRT config stops early), recorded with its per-game counts (wins, losses, draws, caps
from the first's view; game 1 and games 2–3 apart), its verdict and its directory.

**Ratings** (``ratings``): a Bradley–Terry fit by minorisation–maximisation (Hunter 2004) over every
recorded pairing, per game (draws and caps count half a win each way; capped games are findings,
not evidence, so they are left out), with a weak prior (``prior_games`` virtual games, half won,
against a virtual player of rating 0) so a player with only wins or only losses keeps a finite
rating; reported on the Elo scale (400·log10 of the strength), shifted so the reference fixed point
(``anchor``, default the first) is 0. A player's ``games`` is its real games; ratings of players
with few games are noisy (the fit gives no interval).

The ladder's directory holds ``ladder.json`` (fixed points and pairings, rewritten whole) and one
directory per pairing (the runner's files).
"""

import argparse
import json
import math
import os
import time

from . import matches as MT
from .sprt import SPRTConfig, add_common_args, config_from_args, run_sprt


class Ladder:
    """Fixed points, pairings and ratings persisted in `directory`/ladder.json."""

    def __init__(self, directory):
        self.dir = os.path.abspath(directory)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "ladder.json")
        self.doc = {"format": 1, "fixed_points": [], "pairings": []}
        if os.path.exists(self.path):
            with open(self.path) as fp:
                self.doc = json.load(fp)

    def save(self):
        MT.write_json(self.path, self.doc)

    # ── fixed points ──

    def fixed_points(self):
        return list(self.doc["fixed_points"])

    def get(self, fid):
        for f in self.doc["fixed_points"]:
            if f["id"] == fid:
                return f
        return None

    def add_fixed_point(self, fid, model, version=None, options="argmax", meta=None):
        """Adds (or replaces, same id) a fixed point; the model path is stored absolute. Returns it."""
        model = os.path.abspath(model) if MT.is_export(model) else model
        fp = {"id": str(fid), "model": model, "options": options or "", "version": version,
              "spec": MT.agent_side(model, options)[0], "meta": meta or {}, "added_at": time.time()}
        self.doc["fixed_points"] = [f for f in self.doc["fixed_points"] if f["id"] != fp["id"]] + [fp]
        self.save()
        return fp

    # ── pairings ──

    def record(self, a, b, result):
        """Records a run_sprt result of `a` (side a) against `b`; the pairing record."""
        rep = result["report"]
        p = {"a": a, "b": b, "at": time.time(), "verdict": result["verdict"], "out_dir": result["out_dir"],
             "seconds": result["seconds"], "failed_actors": len(result["failed_actors"])}
        for key in ("all", "game1", "games23"):
            t = rep[key]
            p[key] = {k: t[k] for k in ("games", "wins", "losses", "draws", "caps", "win_rate", "wilson95")}
        self.doc["pairings"].append(p)
        self.save()
        return p

    def pairings(self, fid=None):
        return [p for p in self.doc["pairings"] if fid is None or fid in (p["a"], p["b"])]

    def evaluate_against_ladder(self, policy, policy_id, options="argmax", last_k=None, opponents=None, cfg=None,
                                add=False, version=None, meta=None, log=print):
        """Plays `policy` (an export path or a spec; the fixed point `policy_id` when it exists and
        `policy` is None) against every fixed point (`opponents`: ids, else the last `last_k`, else
        all), each pairing a run_sprt with `cfg` (default a fixed 100-game budget); records the
        pairings and, with `add`, adds the policy as a fixed point after. Returns {"policy",
        "results": {opponent id: pairing}, "ratings"}."""
        cfg = cfg or SPRTConfig(max_games=100, sequential=False)
        me = self.get(policy_id) if policy is None else None
        if policy is None and me is None:
            raise ValueError("no fixed point %r and no policy given" % policy_id)
        model = me["model"] if me else policy
        opts = me["options"] if me else options
        pool = [f for f in self.doc["fixed_points"] if f["id"] != policy_id]
        if opponents:
            pool = [f for f in pool if f["id"] in opponents]
        elif last_k:
            pool = pool[-last_k:]
        results = {}
        for f in pool:
            out = os.path.join(self.dir, "%s__vs__%s__%d" % (_safe(policy_id), _safe(f["id"]), len(self.doc["pairings"])))
            c = SPRTConfig.from_dict(dict(cfg.to_dict(), policy_options=opts, opponent_options=f["options"]))
            r = run_sprt(model, f["model"], c, out, log=log)
            results[f["id"]] = self.record(policy_id, f["id"], r)
        if add and not me:
            self.add_fixed_point(policy_id, model, version, opts, meta)
        return {"policy": policy_id, "results": results, "ratings": self.ratings()}

    # ── ratings ──

    def ratings(self, anchor=None, prior_games=1.0, iterations=2000):
        """{id: {"elo", "games", "wins", "losses", "draws"}} (module docstring)."""
        ids = [f["id"] for f in self.doc["fixed_points"]]
        for p in self.doc["pairings"]:
            for x in (p["a"], p["b"]):
                if x not in ids:
                    ids.append(x)
        if not ids:
            return {}
        idx = {x: i for i, x in enumerate(ids)}
        n = len(ids)
        w = [[0.0] * n for _ in range(n)]          # w[i][j]: i's (half-)wins over j
        stats = {x: {"games": 0, "wins": 0, "losses": 0, "draws": 0} for x in ids}
        for p in self.doc["pairings"]:
            i, j = idx[p["a"]], idx[p["b"]]
            t = p["all"]
            w[i][j] += t["wins"] + 0.5 * t["draws"]
            w[j][i] += t["losses"] + 0.5 * t["draws"]
            for x, (wi, lo) in ((p["a"], (t["wins"], t["losses"])), (p["b"], (t["losses"], t["wins"]))):
                s = stats[x]
                s["games"] += t["wins"] + t["losses"] + t["draws"]
                s["wins"] += wi
                s["losses"] += lo
                s["draws"] += t["draws"]
        gamma = bradley_terry(w, prior_games, iterations)
        anchor = anchor if anchor in idx else ids[0]
        base = math.log10(gamma[idx[anchor]])
        return {x: dict(stats[x], elo=round(400.0 * (math.log10(gamma[idx[x]]) - base), 1)) for x in ids}


def bradley_terry(w, prior_games=1.0, iterations=2000, tol=1e-10):
    """Strengths γ (a list) maximising the Bradley–Terry likelihood of the win matrix `w` (w[i][j]
    = i's wins over j), each player also having played `prior_games` games (half won) against a
    virtual player of strength 1; MM updates (Hunter 2004)."""
    n = len(w)
    g = [1.0] * n
    half = 0.5 * prior_games
    for _ in range(iterations):
        new = []
        for i in range(n):
            wins = sum(w[i]) + half
            den = sum((w[i][j] + w[j][i]) / (g[i] + g[j]) for j in range(n) if j != i and w[i][j] + w[j][i] > 0)
            den += prior_games / (g[i] + 1.0)
            new.append(wins / den if den > 0 else g[i])
        delta = max(abs(math.log(a / b)) for a, b in zip(new, g))
        g = new
        if delta < tol:
            break
    return g


def _safe(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def format_ratings(r):
    lines = ["%-24s %8s %6s %10s" % ("fixed point", "elo", "games", "W-L-D")]
    for x, s in sorted(r.items(), key=lambda kv: -kv[1]["elo"]):
        lines.append("%-24s %8.1f %6d %4d-%d-%d" % (x, s["elo"], s["games"], s["wins"], s["losses"], s["draws"]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True, help="the ladder's directory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("add", help="add a fixed point")
    a1.add_argument("--id", required=True)
    a1.add_argument("--model", required=True, help="an export or an agent spec")
    a1.add_argument("--options", default="argmax")
    a1.add_argument("--version", type=int)
    a2 = sub.add_parser("eval", help="play a policy against the ladder")
    a2.add_argument("--id", required=True, help="the policy's id (a fixed point's, or a new one)")
    a2.add_argument("--model", help="an export or spec (omit for an existing fixed point)")
    a2.add_argument("--options", default="argmax")
    a2.add_argument("--version", type=int)
    a2.add_argument("--last-k", type=int)
    a2.add_argument("--against", default="", help="fixed point ids a,b,c (default all / --last-k)")
    a2.add_argument("--games", type=int, default=100, help="games per pairing")
    a2.add_argument("--sprt", action="store_true", help="stop a pairing when its SPRT decides")
    a2.add_argument("--add", action="store_true", help="add the policy as a fixed point after")
    add_common_args(a2)
    sub.add_parser("ratings", help="print the ratings")
    a = ap.parse_args(argv)
    lad = Ladder(a.dir)
    if a.cmd == "add":
        print(json.dumps(lad.add_fixed_point(a.id, a.model, a.version, a.options)))
    elif a.cmd == "eval":
        cfg = config_from_args(a, max_games=a.games, sequential=a.sprt)
        res = lad.evaluate_against_ladder(a.model, a.id, a.options, a.last_k, list(filter(None, a.against.split(","))),
                                          cfg, a.add, a.version)
        for opp, p in res["results"].items():
            t = p["all"]
            print("%s vs %s: %s, %d-%d (%d draws, %d caps), game 1 %s, games 2-3 %s"
                  % (a.id, opp, _pct(t["win_rate"]), t["wins"], t["losses"], t["draws"], t["caps"],
                     _pct(p["game1"]["win_rate"]), _pct(p["games23"]["win_rate"])))
    print(format_ratings(lad.ratings()))
    return 0


def _pct(x):
    return "-" if x is None else "%.1f%%" % (100 * x)


if __name__ == "__main__":
    raise SystemExit(main())
