"""Structured sideboard exploration, the orchestrator's side (docs/rmtrain.md §3 ``[sb_explore]``,
§6.1; docs/rnad_training_plan.md §6.2 part 2): ``p_explore``'s schedule, the coverage counts from
the training actors' results lines, and the ``sb_explore`` object each self-play schedule entry
carries (the actor's side: docs/actor.md §2 ``sb_explore``, §4 ``sb``).

    p = p_explore(cfg, "B4", learner_steps)        # 0 outside B4
    cov = Coverage(state.get("sb_explore"))        # the counts kept in state.json
    cov.add(results_line)                          # a training actor's results line
    snap = snapshot(cfg, cov, p)                   # p, count weights, per matchup in / out weights
    entry["sb_explore"] = entry_sb_explore(snap, "league/bug", "league/burn")    # or None
    summary(cov, matchups(cfg), read_decks(bin_dir, ...))     # the status line and metrics

**Counts.** A results line of a self-play game 2 or 3 (purpose ``train``, schedule version ``b4`` /
``b5``) carries ``sb``: per engine seat whether its sideboarding phase explored, its completed swaps
and the cards moved in and out (the frames' names). The deck a seat played is the line's ``deck_a``
(seat A) or ``deck_b`` (seat B): both are per seat after the match's swap, and decks keep their
seats for the whole match. An explored phase adds, for the matchup (own deck, opponent deck), one
phase, its swap count's bucket (0 … 9, 10+) and each distinct card moved in (resp. out) once; a
phase the policy played adds its phase and bucket to the policy's counts.

**Weights.** A card's in- (out-) weight in a matchup is ``(1 + n) ** -coverage_power``, n the explored
phases of that matchup that moved it in (out); a card never moved needs no entry (the actor's
``default_weight`` 1 is the n = 0 weight).

**Cells.** The coverage summary's denominators are the league matchups (both orders of every pair
of positive weight) times each own deck's sideboard cards (in-cells) and maindeck cards (out-cells),
read from the deck files; names are compared by ``card_key``.
"""

import os
import statistics
import unicodedata

BUCKETS = 11                    # swap counts 0 … 9 and 10+
SELFPLAY_VERSIONS = ("b4", "b5")
P_STEP = 0.01                   # a change of p this large rewrites the schedule before rewrite_every_s


def card_key(name):
    """A card name as the frames and the deck files both reach it: the front face (before ``//`` or
    ``/``: ``Dead/Gone`` is the frames' ``Dead``), accents folded (``Lórien Revealed``, the deck
    files' ``Lorien Revealed``), lower case, letters and digits only."""
    front = name.split("//")[0].split("/")[0]
    folded = unicodedata.normalize("NFKD", front)
    return "".join(c for c in folded.lower() if c.isascii() and c.isalnum())


def p_explore(cfg, phase, learner_steps):
    """p_explore at `learner_steps` in `phase`: in B4 ``p_start`` until ``hold_frac`` of
    ``b4.max_learner_steps``, then linear to ``p_end`` at ``b4.max_learner_steps`` (``p_start``
    throughout when that is 0); 0 in every other phase."""
    if phase != "B4":
        return 0.0
    s = cfg["sb_explore"]
    total = cfg["b4"]["max_learner_steps"]
    if total <= 0:
        return float(s["p_start"])
    hold = s["hold_frac"] * total
    if learner_steps >= total:
        return float(s["p_end"])
    if learner_steps <= hold:
        return float(s["p_start"])
    f = (learner_steps - hold) / (total - hold)
    return float(s["p_start"] + f * (s["p_end"] - s["p_start"]))


def p_moved(old, new):
    """Whether p moved enough since the schedule was written to write it again now: by ``P_STEP``,
    or onto 0 or off it."""
    return abs(new - old) >= P_STEP or (new != old and (new == 0 or old == 0))


def matchup(own, opp):
    return "%s|%s" % (own, opp)


def _bucket(swaps):
    return min(max(int(swaps), 0), BUCKETS - 1)


def _new_cell():
    return {"phases": 0, "hist": [0] * BUCKETS, "in": {}, "out": {}}


class Coverage:
    """The explored phases' counts per matchup (``own|opp``: phases, swap-count histogram, phases per
    card moved in and out) and the policy's phases (phases, histogram); ``state()`` round-trips
    through state.json. ``recent`` counts the policy's phases since ``take_recent``."""

    def __init__(self, state=None):
        state = state or {}
        self.explored = state.get("explored", {})
        self.policy = state.get("policy", {})
        self.recent = [0] * BUCKETS

    def state(self):
        return {"explored": self.explored, "policy": self.policy}

    def add(self, line):
        """Counts a results line's sideboarding phases; the number counted (0 for a line that is
        not a self-play game 2–3)."""
        sb = line.get("sb")
        if not isinstance(sb, dict) or line.get("purpose", "train") != "train" or \
                str(line.get("schedule", "")) not in SELFPLAY_VERSIONS:
            return 0
        n = 0
        for seat, own_key, opp_key in (("A", "deck_a", "deck_b"), ("B", "deck_b", "deck_a")):
            t = sb.get(seat)
            own, opp = line.get(own_key), line.get(opp_key)
            if not isinstance(t, dict) or not own or not opp:
                continue
            b = _bucket(t.get("swaps", 0))
            if t.get("explored"):
                c = self.explored.setdefault(matchup(own, opp), _new_cell())
                for kind in ("in", "out"):
                    for name in set(t.get(kind) or []):
                        c[kind][name] = c[kind].get(name, 0) + 1
            else:
                c = self.policy.setdefault(matchup(own, opp), {"phases": 0, "hist": [0] * BUCKETS})
                self.recent[b] += 1
            c["phases"] += 1
            c["hist"][b] += 1
            n += 1
        return n

    def take_recent(self):
        """The policy's swap-count histogram since the last call."""
        out, self.recent = self.recent, [0] * BUCKETS
        return out


def _weights(counts, power):
    return {name: float("%.6g" % (1.0 + n) ** -power) for name, n in sorted(counts.items()) if n > 0}


def snapshot(cfg, cov, p):
    """What the schedule's entries carry: p, the count weights and, with coverage on, per matchup the
    in and out weights of the cards its explored phases moved."""
    s = cfg["sb_explore"]
    weights = {}
    if s["coverage"] and s["coverage_power"] > 0:
        for key, c in cov.explored.items():
            w = {"in": _weights(c["in"], s["coverage_power"]), "out": _weights(c["out"], s["coverage_power"])}
            if w["in"] or w["out"]:
                weights[key] = w
    return {"p": round(float(p), 6), "count_weights": [float(x) for x in s["count_weights"]], "weights": weights}


def entry_sb_explore(snap, deck_a, deck_b):
    """The ``sb_explore`` of a schedule entry between `deck_a` and `deck_b` (stems as the entry names
    them): each deck's weights against the other; None when p is 0 (the entry explores nothing)."""
    if not snap or snap["p"] <= 0:
        return None
    weights = {}
    for own, opp in ((deck_a, deck_b), (deck_b, deck_a)):
        w = snap["weights"].get(matchup(own, opp))
        if w:
            weights[own] = w
    return {"p": snap["p"], "count_weights": list(snap["count_weights"]), "weights": weights, "default_weight": 1.0}


def matchups(pairs):
    """The (own, opponent) matchups of `pairs` ([(deck_a, deck_b, weight)] with full stems): both
    orders of each pair of positive weight."""
    out = []
    for a, b, w in pairs:
        if w <= 0:
            continue
        for m in ((a, b), (b, a)):
            if m not in out:
                out.append(m)
    return out


def parse_deck(path):
    """A ``.dk`` file's (maindeck, sideboard) card keys (sets of ``card_key``)."""
    main, side = set(), set()
    cur = main
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            s = line.strip()
            if not s:
                continue
            if s.upper().startswith("SIDEBOARD"):
                cur = side
                continue
            qty, _, name = s.partition(" ")
            if qty.isdigit() and name.strip():
                cur.add(card_key(name.strip()))
    return main, side


def read_decks(bin_dir, stems):
    """{stem: (maindeck keys, sideboard keys)} of the deck files ``<bin_dir>/resources/decks/<stem>.dk``
    that exist."""
    out = {}
    for stem in stems:
        path = os.path.join(bin_dir, "resources", "decks", stem + ".dk")
        if os.path.exists(path):
            out[stem] = parse_deck(path)
    return out


def summary(cov, mups, decks):
    """The coverage summary over the matchups `mups` and the decks' cards (``read_decks``): explored
    and policy phases, explored phases per matchup (min, median), the shares of (matchup, sideboard
    card) in-cells and (matchup, maindeck card) out-cells tried at least once and at least 10 times,
    the swap-count histograms and the policy's swap rate (phases with a swap)."""
    explored_hist, policy_hist = [0] * BUCKETS, [0] * BUCKETS
    for c in cov.explored.values():
        explored_hist = [x + y for x, y in zip(explored_hist, c["hist"])]
    for c in cov.policy.values():
        policy_hist = [x + y for x, y in zip(policy_hist, c["hist"])]
    per = [cov.explored.get(matchup(o, p), {}).get("phases", 0) for o, p in mups]
    cells = {"in": [0, 0, 0], "out": [0, 0, 0]}        # cells, tried >= 1, tried >= 10
    for own, opp in mups:
        if own not in decks:
            continue
        c = cov.explored.get(matchup(own, opp), {})
        main, side = decks[own]
        for kind, cards in (("in", side), ("out", main)):
            tried = {}
            for name, n in c.get(kind, {}).items():
                k = card_key(name)
                tried[k] = tried.get(k, 0) + n
            for k in cards:
                n = tried.get(k, 0)
                cells[kind][0] += 1
                cells[kind][1] += n >= 1
                cells[kind][2] += n >= 10
    policy_phases = sum(policy_hist)
    out = {"explored_phases": sum(explored_hist), "policy_phases": policy_phases, "matchups": len(mups),
           "phases_min": min(per) if per else 0, "phases_median": statistics.median(per) if per else 0,
           "explored_hist": explored_hist, "policy_hist": policy_hist,
           "policy_swap_rate": (policy_phases - policy_hist[0]) / policy_phases if policy_phases else None}
    for kind in ("in", "out"):
        n, t1, t10 = cells[kind]
        out["%s_cells" % kind] = n
        out["%s_tried_1" % kind] = t1 / n if n else None
        out["%s_tried_10" % kind] = t10 / n if n else None
    return out


def status_line(sx):
    """The status's one line (`sx`: status.json's ``sb_explore``)."""
    def pct(x):
        return "-" if x is None else "%.0f%%" % (100 * x)

    rate = sx.get("policy_swap_rate")
    return ("sb explore: p %.3f (schedule %.3f); %d explored phases, per matchup min %d / median %g over %d; "
            "in-cells %d tried >=1 %s, >=10 %s; out-cells %d >=1 %s, >=10 %s; policy swap rate %s (%d phases)" % (
                sx.get("p", 0.0), sx.get("schedule_p", 0.0), sx.get("explored_phases", 0), sx.get("phases_min", 0),
                sx.get("phases_median", 0), sx.get("matchups", 0), sx.get("in_cells", 0), pct(sx.get("in_tried_1")),
                pct(sx.get("in_tried_10")), sx.get("out_cells", 0), pct(sx.get("out_tried_1")),
                pct(sx.get("out_tried_10")), "-" if rate is None else "%.3f" % rate, sx.get("policy_phases", 0)))
