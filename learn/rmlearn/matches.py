"""Evaluation matches through engine actors (``--actor``, docs/actor.md), read game by game as they
finish, and their per-game reports (docs/rnad_training_plan.md T8): the shared machinery of
``sprt.py``, ``ladder.py`` and ``exploiter.py``.

    sched = eval_schedule("nn:/r/p3.pt2:argmax", "scripted", out_dir, pairs=league_pairs())
    pool = ActorPool(sched, out_dir, actors=8, seed=11)
    pool.start()
    for line in pool.poll(): rec = game_record(line)          # results lines as games finish
    pool.stop()                                                # SIGTERM, then wait
    rep = report([...records...])                              # per game, game 1 vs games 2–3, ...
    print(format_report(rep, "p3 vs scripted"))

**Sides.** An evaluation schedule has one entry per deck pair, side a the evaluated policy and side
b the opponent, with ``swap: sides`` (the coin of each match's seed puts each side in either seat,
so the policy plays both decks of the pair and both seats). A results line names seats; the
policy's seat is its ``side_a_seat``, and ``game_record`` turns a line into the side's view: its
result, deck, seat, whether it played first.

**Agents** (``agent_side``): ``scripted`` / ``scripted:<tier>`` is side ``first`` with the scripted
teacher (``python/teacher.py``) at ``teacher_mix: 1`` (the teacher plays the side); ``explore``,
``random`` and any spec are themselves; an export path (``.pt2`` / ``.ts.pt``) becomes
``nn:<absolute path>:<options>`` (``argmax``, ``temp=1``, ``temp=1:thresh=0.03:disc=16``, …);
``nn:…`` and ``nn:remote:<socket>[:policies=…]`` specs are used as they are (a relative model path
is relative to the engine's working directory, ``bin/``).

**Scoring is per game** (plan: bo3 matches, every game scored on its own). ``tally`` counts wins,
losses, draws, caps and concessions; the win rate is over decisive games (wins and losses: draws and
caps are excluded and listed as findings), with a Wilson 95% interval, the binomial standard error
and the **match-clustered** one: the games of one match are not independent (the same decks, the
same opponent read, sideboarding against what was seen, game 3 only at 1–1), so the clustered SE
(a ratio estimator over matches) is the honest one, and their ratio squared is the design effect.
``report`` gives game 1 (pre-board, a pure bo1 measurement) apart from games 2–3 (post-board); game
3 is played only at 1–1, so the games 2–3 figure is a selected sample (``SELECTION_NOTE``).
"""

import ctypes
import itertools
import json
import math
import os
import signal
import subprocess
import time

from .corpus import BIN_DIR, CPU_ENV, LEAGUE, teacher_command

EXPORT_SUFFIXES = (".pt2", ".ts.pt")
SELECTION_NOTE = ("games 2-3 are post-board; game 3 is played only at 1-1, so the games 2-3 figure is a "
                  "selected sample (not comparable to game 1 as a measurement of the same population)")


# ── agents and schedules ─────────────────────────────────────────────────────────────────────

def league_pairs(decks=LEAGUE):
    """Every unordered pair of league decks (45 for the 10 league decks)."""
    return [tuple(p) for p in itertools.combinations(decks, 2)]


def parse_pairs(text):
    """Deck pairs from "a-b,c-d" or "a:b,c:d" (deck names), or "all" (every league pair)."""
    if not text or text == "all":
        return league_pairs()
    return [parse_pair(item) for item in text.split(",")]


def parse_pair(item):
    """'ur_delver-burn' or 'ur_delver:burn' → ('ur_delver', 'burn')."""
    if isinstance(item, (tuple, list)):
        return tuple(item)
    if ":" in item:
        a, b = item.split(":", 1)
        return a, b
    for d in LEAGUE:
        if item.startswith(d + "-") and item[len(d) + 1:]:
            return d, item[len(d) + 1:]
    if item.count("-") == 1:
        a, b = item.split("-")
        return a, b
    raise ValueError("cannot split the deck pair %r (write it a:b)" % item)


def deck_stem(name):
    """A deck name as a schedule stem: league decks get "league/", stems with a '/' stay."""
    return name if "/" in name else "league/" + name


def is_export(name):
    return isinstance(name, str) and name.endswith(EXPORT_SUFFIXES)


def agent_side(name, options="argmax", teacher_seed=0):
    """(agent spec, teacher command or None) of a policy or opponent name (module docstring)."""
    if name.startswith("scripted"):
        tier = name.split(":", 1)[1] if ":" in name else "hard"
        return "first", teacher_command(tier, teacher_seed)
    if name.startswith("nn:"):
        return name, None
    if is_export(name):
        return "nn:%s%s" % (os.path.abspath(name), ":" + options if options else ""), None
    return name, None


def eval_schedule(policy, opponent, out_dir, pairs=None, weights=None, fmt="bo3", knowledge="expert",
                  max_game_decisions=5000, version="eval", policy_options="argmax", opponent_options="argmax",
                  purpose="eval", trajectory_dir=None, trajectory_seats=None, teacher_seed=0):
    """The actor schedule (a dict) of `policy` (side a) against `opponent` (side b): one entry per
    deck pair (`pairs`, default every league pair; `weights` {"a-b" or (a, b): weight}), ``swap:
    sides``. Decision logs under `out_dir`/rmlog; episodes only with `trajectory_dir` (then
    `purpose` should be "train")."""
    pairs = list(pairs or league_pairs())
    weights = {parse_pair(k): float(v) for k, v in (weights or {}).items()}
    pa, ta = agent_side(policy, policy_options, teacher_seed)
    pb, tb = agent_side(opponent, opponent_options, teacher_seed)
    entries = []
    for a, b in pairs:
        e = {"id": "%s-%s" % (a, b), "weight": weights.get((a, b), 1.0), "purpose": purpose,
             "deck_a": deck_stem(a), "deck_b": deck_stem(b), "swap": "sides", "agent_a": pa, "agent_b": pb}
        mix = False
        if ta:
            e["teacher_a"] = ta
            mix = True
        if tb:
            e["teacher_b"] = tb
            mix = True
        if mix:
            e["teacher_mix"] = 1
        if trajectory_seats:
            e["trajectory_seats"] = trajectory_seats
        entries.append(e)
    out = {"rmlog_dir": os.path.join(os.path.abspath(out_dir), "rmlog")}
    if trajectory_dir:
        out["trajectory_dir"] = os.path.abspath(trajectory_dir)
    return {"version": version, "outputs": {purpose: out},
            "defaults": {"format": fmt, "knowledge_a": knowledge, "knowledge_b": knowledge,
                         "max_game_decisions": int(max_game_decisions)},
            "entries": entries}


def write_json(path, doc):
    """Writes `doc` whole (a temporary file renamed over `path`)."""
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w") as fp:
        json.dump(doc, fp, indent=1)
        fp.write("\n")
    os.replace(tmp, path)


# ── the actor processes ──────────────────────────────────────────────────────────────────────

try:                                   # resolved before any fork: the child only calls it
    _PRCTL = ctypes.CDLL("libc.so.6", use_errno=True).prctl
except (OSError, AttributeError):
    _PRCTL = None
_PR_SET_PDEATHSIG = 1
_SIGTERM = int(signal.SIGTERM)


def _die_with_parent():
    """preexec_fn: the child gets SIGTERM when this process dies (Linux PR_SET_PDEATHSIG)."""
    if _PRCTL is not None:
        _PRCTL(_PR_SET_PDEATHSIG, _SIGTERM)


class ActorPool:
    """`actors` engine actor processes playing `schedule` (a dict, written to `out_dir`/schedule.json)
    until stopped (or `matches_per_actor` each). ``poll()`` returns the results lines (dicts) that
    appeared since the last call, in each actor's order; ``stop()`` sends SIGTERM (each actor stops
    before its next decision; its unfinished match is written off) and waits."""

    def __init__(self, schedule, out_dir, actors=8, engine=None, bin_dir=BIN_DIR, seed=11, env=None,
                 matches_per_actor=None, first_id=0, extra_args=()):
        self.out_dir = os.path.abspath(out_dir)
        self.bin_dir = bin_dir
        self.engine = os.path.abspath(engine or os.path.join(bin_dir, "release", "robomage_nn"))
        self.actors = int(actors)
        self.seed = int(seed)
        self.env = env or CPU_ENV
        self.matches_per_actor = matches_per_actor
        self.first_id = first_id
        self.extra_args = list(extra_args)
        os.makedirs(self.out_dir, exist_ok=True)
        self.schedule_path = os.path.join(self.out_dir, "schedule.json")
        write_json(self.schedule_path, schedule)
        self.procs = {}
        self._offsets = {}
        self._partial = {}
        self._logs = {}
        self.started_at = None
        self.exit_codes = {}

    def ids(self):
        return list(range(self.first_id, self.first_id + self.actors))

    def start(self):
        if not os.path.exists(self.engine):
            raise FileNotFoundError("%s does not exist (make BUILD=RELEASE actor)" % self.engine)
        self.started_at = time.time()
        for k in self.ids():
            for suffix in (".results.jsonl", ".status.json"):
                p = os.path.join(self.out_dir, "a%d%s" % (k, suffix))
                if os.path.exists(p):
                    os.unlink(p)
            argv = [self.engine, "--actor", "--actor-schedule", self.schedule_path, "--actor-out", self.out_dir,
                    "--actor-id", str(k), "--actor-seed", str(self.seed)] + self.extra_args
            if self.matches_per_actor:
                argv += ["--actor-matches", str(self.matches_per_actor)]
            log = open(os.path.join(self.out_dir, "a%d.log" % k), "w")
            self._logs[k] = log
            self.procs[k] = subprocess.Popen(argv, cwd=self.bin_dir, stdout=log, stderr=subprocess.STDOUT,
                                             env=self.env, preexec_fn=_die_with_parent)
            self._offsets[k] = 0
            self._partial[k] = b""

    def poll(self):
        """The new results lines (parsed) of every actor."""
        out = []
        for k in self.ids():
            path = os.path.join(self.out_dir, "a%d.results.jsonl" % k)
            try:
                with open(path, "rb") as fp:
                    fp.seek(self._offsets.get(k, 0))
                    data = fp.read()
            except FileNotFoundError:
                continue
            self._offsets[k] = self._offsets.get(k, 0) + len(data)
            data = self._partial.get(k, b"") + data
            lines = data.split(b"\n")
            self._partial[k] = lines.pop()          # an incomplete last line waits for its newline
            for ln in lines:
                if ln.strip():
                    out.append(json.loads(ln))
        for k, p in self.procs.items():
            if k not in self.exit_codes and p.poll() is not None:
                self.exit_codes[k] = p.returncode
        return out

    def alive(self):
        return sum(1 for p in self.procs.values() if p.poll() is None)

    def failures(self):
        """[(actor id, exit code, the log's last lines)] of the actors that exited nonzero."""
        out = []
        for k, p in self.procs.items():
            rc = p.poll()
            if rc not in (None, 0):
                out.append({"actor": k, "exit": rc, "log_tail": _tail(os.path.join(self.out_dir, "a%d.log" % k))})
        return out

    def statuses(self):
        out = {}
        for k in self.ids():
            try:
                with open(os.path.join(self.out_dir, "a%d.status.json" % k)) as fp:
                    out[k] = json.load(fp)
            except (FileNotFoundError, ValueError):
                pass
        return out

    def decisions(self):
        """The decisions the actors have read so far (their status files)."""
        return sum(int(s.get("decisions", 0)) for s in self.statuses().values())

    def stop(self, timeout=120):
        """SIGTERM to every running actor, then wait (SIGKILL after `timeout` s)."""
        for p in self.procs.values():
            if p.poll() is None:
                try:
                    p.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
        end = time.time() + timeout
        for k, p in self.procs.items():
            try:
                p.wait(max(0.1, end - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            self.exit_codes[k] = p.returncode
        for log in self._logs.values():
            log.close()
        self._logs = {}

    def wait(self):
        for k, p in self.procs.items():
            self.exit_codes[k] = p.wait()


def _tail(path, n=6):
    try:
        with open(path, errors="replace") as fp:
            return fp.read().splitlines()[-n:]
    except FileNotFoundError:
        return []


# ── per-game records and reports ─────────────────────────────────────────────────────────────

def game_record(line, side="a"):
    """A results line from `side`'s view ("a": the schedule's side a, the evaluated policy): result
    (+1 / -1 / 0), end reason, game number, seat, whether it played first, its deck and the
    opponent's, the entry (the deck pair), the match (``key``: actor, match), its policy versions."""
    seat_a = line.get("side_a_seat", "A")
    seat = seat_a if side == "a" else ("B" if seat_a == "A" else "A")
    s, o = seat.lower(), ("b" if seat == "A" else "a")
    return {"key": "%s:%s" % (line["actor"], line["match"]), "actor": line["actor"], "match": line["match"],
            "game": int(line["game"]), "result": int(line["result_" + s]), "end": line["end_reason"], "seat": seat,
            "first": line.get("first") == seat, "deck": _short(line["deck_" + s]), "opp_deck": _short(line["deck_" + o]),
            "pair": line["entry"], "decisions": int(line.get("decisions", 0)), "seconds": float(line.get("seconds", 0)),
            "versions": list(line.get("policy_versions_" + s) or []), "rmlog": line.get("rmlog"),
            "episode": line.get("episode")}


def _short(stem):
    return stem[len("league/"):] if stem.startswith("league/") else stem


def wilson(wins, n, z=1.959964):
    """The Wilson score interval of a binomial proportion (lo, hi); (0, 1) with no trials."""
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def clustered_se(records):
    """The match-clustered standard error of the decisive-game win rate (a ratio estimator over
    clusters = matches): sqrt(k/(k-1) · Σ (w_i − p·n_i)²) / Σ n_i; None with fewer than 2 clusters."""
    clusters = {}
    for r in records:
        if r["result"] == 0:
            continue
        c = clusters.setdefault(r["key"], [0, 0])
        c[0] += r["result"] > 0
        c[1] += 1
    k = len(clusters)
    n = sum(c[1] for c in clusters.values())
    if k < 2 or n == 0:
        return None
    p = sum(c[0] for c in clusters.values()) / n
    ss = sum((w - p * m) ** 2 for w, m in clusters.values())
    return math.sqrt(k / (k - 1) * ss) / n


def tally(records):
    """Counts and rates of a list of game records (module docstring)."""
    t = {"games": len(records), "wins": 0, "losses": 0, "draws": 0, "caps": 0, "concessions": 0}
    for r in records:
        if r["end"] == "cap":
            t["caps"] += 1
        elif r["result"] > 0:
            t["wins"] += 1
        elif r["result"] < 0:
            t["losses"] += 1
        else:
            t["draws"] += 1
        t["concessions"] += r["end"] == "concession"
    n = t["wins"] + t["losses"]
    t["decisive"] = n
    t["matches"] = len({r["key"] for r in records})
    t["win_rate"] = t["wins"] / n if n else None
    t["wilson95"] = list(wilson(t["wins"], n)) if n else None
    t["se"] = math.sqrt(t["win_rate"] * (1 - t["win_rate"]) / n) if n else None
    t["se_clustered"] = clustered_se(records)
    t["design_effect"] = (t["se_clustered"] / t["se"]) ** 2 if t["se"] and t["se_clustered"] else None
    return t


def _group(records, key):
    out = {}
    for r in records:
        out.setdefault(key(r), []).append(r)
    return {k: tally(v) for k, v in sorted(out.items(), key=lambda kv: str(kv[0]))}


def report(records):
    """The per-game report of a policy's records: ``all``, ``game1``, ``games23`` (with
    ``selection_note``), ``by_game``, ``by_pair`` (the entry: the deck pair), ``by_deck`` (the
    policy's deck), ``by_matchup`` (policy deck vs opponent deck), ``by_seat``, ``by_order``
    (on the play / on the draw), ``findings`` (drawn and capped games' logs)."""
    g1 = [r for r in records if r["game"] == 1]
    g23 = [r for r in records if r["game"] > 1]
    return {"all": tally(records), "game1": tally(g1), "games23": tally(g23), "selection_note": SELECTION_NOTE,
            "by_game": _group(records, lambda r: str(r["game"])),
            "by_pair": _group(records, lambda r: r["pair"]),
            "by_deck": _group(records, lambda r: r["deck"]),
            "by_matchup": _group(records, lambda r: "%s vs %s" % (r["deck"], r["opp_deck"])),
            "by_seat": _group(records, lambda r: r["seat"]),
            "by_order": _group(records, lambda r: "on_play" if r["first"] else "on_draw"),
            "findings": [{"end": r["end"], "game": r["game"], "rmlog": r["rmlog"], "decisions": r["decisions"]}
                         for r in records if r["result"] == 0]}


def _rate_text(t):
    if not t["decisive"]:
        return "      -      (0 decisive of %d)" % t["games"]
    lo, hi = t["wilson95"]
    s = "%5.1f%% [%4.1f, %4.1f]  %d-%d" % (100 * t["win_rate"], 100 * lo, 100 * hi, t["wins"], t["losses"])
    extra = []
    if t["draws"]:
        extra.append("%d draws" % t["draws"])
    if t["caps"]:
        extra.append("%d caps" % t["caps"])
    return s + ("  (" + ", ".join(extra) + ")" if extra else "")


def format_report(rep, title=""):
    """A human-readable rendering of ``report()``."""
    a = rep["all"]
    lines = []
    if title:
        lines.append(title)
    lines.append("games %d (%d matches), decisive %d, draws %d, caps %d, concessions %d"
                 % (a["games"], a["matches"], a["decisive"], a["draws"], a["caps"], a["concessions"]))
    lines.append("win rate per game  %s" % _rate_text(a))
    if a["se"] is not None:
        lines.append("  SE %.3f binomial, %s match-clustered (design effect %s)"
                     % (a["se"], "%.3f" % a["se_clustered"] if a["se_clustered"] is not None else "-",
                        "%.2f" % a["design_effect"] if a["design_effect"] is not None else "-"))
    lines.append("game 1 (pre-board)  %s" % _rate_text(rep["game1"]))
    lines.append("games 2-3 (post)    %s" % _rate_text(rep["games23"]))
    lines.append("  note: " + rep["selection_note"])
    for name, key in (("by game", "by_game"), ("on the play / draw", "by_order"), ("by seat", "by_seat"),
                      ("by own deck", "by_deck"), ("by matchup (own vs opponent deck)", "by_matchup")):
        lines.append(name + ":")
        for k, t in rep[key].items():
            lines.append("  %-34s %s" % (k, _rate_text(t)))
    if rep["findings"]:
        lines.append("findings (draws and caps, excluded from the rate):")
        for f in rep["findings"]:
            lines.append("  %s game %d after %d decisions: %s" % (f["end"], f["game"], f["decisions"], f["rmlog"]))
    return "\n".join(lines)
