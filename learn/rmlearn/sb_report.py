"""The benchmark's sideboarding report (docs/rmtrain.md §7.1; docs/rnad_training_plan.md §6.2
Measurements): one side's sideboarding phases in a benchmark's results lines, and their summary.

    recs = phases(lines, side="a")       # side a: the policy; side b: the opponent
    rep = report(lines, side="a")        # overall and per deck: swap rate, histogram, cards, ...
    md = format_section(...)             # summary.md's sideboarding tables

**Phases.** A results line of game 2 or 3 carries ``sb`` (docs/actor.md §4): per engine seat whether
its sideboarding phase explored, its completed swaps and the cards moved in and out, one name per
copy, as the frames name them (a split card's front face: ``Dead`` for ``Dead/Gone``; accents as the
scripts spell them: ``Lórien Revealed``). A side sits in seat ``side_a_seat`` (side a) or the other
(side b); ``deck_a`` / ``deck_b`` are per seat, and decks keep their seats for the whole match.

**Configurations.** A side's maindeck is tracked through the match: it starts at the registered list
(the league deck file's maindeck), and each phase applies its cuts and its cards in, so game 3's
phase starts from game 2's configuration. A seat whose previous game is missing (no line for it, a
line without ``sb``) or whose deck is not a league deck has no configuration for the rest of the match.

**Play / draw** of the game a phase precedes: the line's ``first`` (who played first in that game);
without it the rule: the previous game's loser plays first, after a drawn game the player who
played first before.

**Guide agreement.** A phase agrees when its configuration equals the guide's target for (deck,
opponent deck, play / draw) (``python/sideboard_guides.py``'s ``Guides.target`` over
``python/sideboard_guides.json`` and the league deck files, held-out rows included), cards compared
by ``card_key`` (the front face, accents folded, lower case, letters and digits). Held-in and
held-out matchups (the guides' ``holdout`` list, where ``scripted:hard+sb`` takes Done) are
reported apart: the policy learned the held-in ones from the teacher; agreement on the held-out
ones is the generalisation measurement.

**Next game.** The win rate of the game a phase precedes (decisive games: draws and caps out) after
a phase with swaps and after Done (no swap). Observational: the side chose whether to swap from
what it saw, so the comparison mixes a swap's effect with the matchups and situations it swapped in.

Explored phases (``sb_explore``: none in a benchmark, whose evaluation entries refuse it) are
counted and left out of every figure.
"""

import importlib.util
import os

from .corpus import REPO_ROOT
from .orchestrator.sb_explore import BUCKETS, card_key

TOP = 5
OBSERVATIONAL_NOTE = ("observational: the side chose whether to swap from what it saw, so a swap's "
                      "effect is mixed with the matchups and situations it swapped in")

_GUIDES = None


def guides():
    """The guides (``python/sideboard_guides.py``'s ``Guides``: rows, held-out pairs, league decks),
    loaded once per process."""
    global _GUIDES
    if _GUIDES is None:
        path = os.path.join(REPO_ROOT, "python", "sideboard_guides.py")
        spec = importlib.util.spec_from_file_location("rm_sideboard_guides", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _GUIDES = mod.guides()
    return _GUIDES


def keyed(deck):
    """A deck (card name -> copies) by ``card_key``, empty counts dropped."""
    out = {}
    for name, n in deck.items():
        k = card_key(name)
        out[k] = out.get(k, 0) + n
    return {k: n for k, n in out.items() if n}


def _short(stem):
    return stem[len("league/"):] if stem.startswith("league/") else stem


def _other(seat):
    return "B" if seat == "A" else "A"


def side_seat(line, side):
    """The engine seat of `side` ("a": the schedule's side a, the policy; "b": the opponent)."""
    seat_a = line.get("side_a_seat", "A")
    return seat_a if side == "a" else _other(seat_a)


def first_player(line, prev):
    """Who played first in `line`'s game: its ``first``, else from the previous game's line (its
    loser; after a draw the same player as before); None when neither tells."""
    if line.get("first") in ("A", "B"):
        return line["first"]
    if prev is None:
        return None
    for seat in ("A", "B"):
        if int(prev.get("result_" + seat.lower(), 0)) < 0:
            return seat
    return prev.get("first") if prev.get("first") in ("A", "B") else None


def _apply(config, moves):
    for kind, sign in (("out", -1), ("in", +1)):
        for name in moves.get(kind) or []:
            k = card_key(name)
            config[k] = config.get(k, 0) + sign
    return {k: n for k, n in config.items() if n}


def phases(lines, side="a"):
    """`side`'s sideboarding phases in the results lines (module docstring), match by match in game
    order: deck, opp, game, swaps, in, out, explored, on_play, result, decisive, held_out, has_target,
    agree (None without a configuration or a guide target)."""
    g = guides()
    matches = {}
    for ln in lines:
        matches.setdefault((str(ln["actor"]), str(ln["match"])), []).append(ln)
    out = []
    for games in matches.values():
        config, prev = None, None
        for ln in sorted(games, key=lambda x: int(x["game"])):
            n = int(ln["game"])
            seat = side_seat(ln, side)
            deck, opp = _short(ln["deck_" + seat.lower()]), _short(ln["deck_" + _other(seat).lower()])
            first = first_player(ln, prev)
            if n == 1:
                config = keyed(g.decks[deck][0]) if deck in g.decks else None
                prev = ln
                continue
            if prev is None or int(prev["game"]) != n - 1:
                config = None
            t = (ln.get("sb") or {}).get(seat)
            prev = ln
            if not isinstance(t, dict):
                config = None
                continue
            if config is not None:
                config = _apply(config, t)
            on_play = None if first is None else first == seat
            target = None if on_play is None else g.target(deck, opp, on_play, holdout=False)
            result = int(ln.get("result_" + seat.lower(), 0))
            out.append({"deck": deck, "opp": opp, "game": n, "swaps": int(t.get("swaps", 0)),
                        "in": list(t.get("in") or []), "out": list(t.get("out") or []),
                        "explored": bool(t.get("explored")), "on_play": on_play, "result": result,
                        "decisive": ln.get("end_reason") != "cap" and result != 0,
                        "held_out": (deck, opp) in g.holdout, "has_target": target is not None,
                        "agree": None if target is None or config is None else keyed(target) == config})
    return out


def _rate(wins, n):
    return {"wins": wins, "decisive": n, "win_rate": wins / n if n else None}


def _agreement(recs):
    rated = [r for r in recs if r["agree"] is not None]
    k = sum(r["agree"] for r in rated)
    return {"phases": len(rated), "agree": k, "rate": k / len(rated) if rated else None}


def _top(recs, kind):
    counts = {}
    for r in recs:
        for name in r[kind]:
            counts[name] = counts.get(name, 0) + 1
    return [[name, n] for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP]]


def stats(recs):
    """The summary of phases (none explored): phases, swap rate (phases with a swap), the swap-count
    histogram (0 … 9, 10+), the top cards in and out (copies moved), the next game's win rate after
    swaps and after Done, the guide agreement held-in (all, on the play, on the draw) and held-out."""
    hist = [0] * BUCKETS
    for r in recs:
        hist[min(max(r["swaps"], 0), BUCKETS - 1)] += 1
    swapped = [r for r in recs if r["swaps"] > 0 and r["decisive"]]
    done = [r for r in recs if r["swaps"] == 0 and r["decisive"]]
    held_in = [r for r in recs if not r["held_out"]]
    held_out = [r for r in recs if r["held_out"]]
    return {"phases": len(recs), "swap_rate": (len(recs) - hist[0]) / len(recs) if recs else None, "hist": hist,
            "in": _top(recs, "in"), "out": _top(recs, "out"),
            "after_swap": _rate(sum(r["result"] > 0 for r in swapped), len(swapped)),
            "after_done": _rate(sum(r["result"] > 0 for r in done), len(done)),
            "agreement": {"held_in": dict(_agreement(held_in),
                                          on_play=_agreement([r for r in held_in if r["on_play"]]),
                                          on_draw=_agreement([r for r in held_in if r["on_play"] is False])),
                          "held_out": _agreement(held_out)}}


def report(lines, side="a"):
    """`side`'s sideboarding report of the results lines: ``all`` and ``by_deck`` (``stats``), the
    explored phases left out (``explored``), the games 2–3 whose line carries no ``sb`` for the side
    (``missing``: an engine from before the key)."""
    recs = phases(lines, side)
    missing = sum(int(ln["game"]) > 1 and not isinstance((ln.get("sb") or {}).get(side_seat(ln, side)), dict)
                  for ln in lines)
    kept = [r for r in recs if not r["explored"]]
    decks = {}
    for r in kept:
        decks.setdefault(r["deck"], []).append(r)
    return {"explored": len(recs) - len(kept), "missing": missing, "all": stats(kept),
            "by_deck": {d: stats(v) for d, v in sorted(decks.items())}, "note": OBSERVATIONAL_NOTE}


# ── summary.md ────────────────────────────────────────────────────────────────────────────────

def _r(t):
    return "-" if t["win_rate"] is None else "%.3f (%d)" % (t["win_rate"], t["decisive"])


def _a(t):
    return "-" if t["rate"] is None else "%.2f (%d/%d)" % (t["rate"], t["agree"], t["phases"])


def _hist(h):
    return "/".join(str(x) for x in h)


def _cards(top):
    return ", ".join("%s ×%d" % (n, c) for n, c in top) or "-"


def _row(name, s):
    ag = s["agreement"]
    return "| %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (
        name, s["phases"], "-" if s["swap_rate"] is None else "%.3f" % s["swap_rate"], _hist(s["hist"]),
        _r(s["after_swap"]), _r(s["after_done"]), _a(ag["held_in"]), _a(ag["held_out"]),
        "%s / %s" % (_a(ag["held_in"]["on_play"]), _a(ag["held_in"]["on_draw"])))


_HEAD = ["| %s | phases | swap rate | swaps 0/1/…/9/10+ | next game after swaps | after Done | "
         "guide agreement held-in | held-out | held-in on the play / draw |",
         "|---|---|---|---|---|---|---|---|---|"]


def format_section(policies, decks, opponent):
    """summary.md's sideboarding section: `policies` {label: {"policy": report, "opponent": report or
    absent}} in order, `decks` the row order."""
    L = ["", "## Sideboarding", "",
         "Each side's sideboarding phases before games 2 and 3 (its own seat only). Swap rate: the share "
         "of phases with at least one swap; the histogram counts phases by swaps. Next game: the win rate "
         "(decisive games, count in parentheses) of the game after a phase with swaps and after Done "
         "(%s). Guide agreement: the share of phases whose resulting maindeck (tracked through the "
         "match from the registered list) equals the sideboard guide's target for the matchup and "
         "play / draw; held-out matchups are those the `scripted:hard+sb` teacher leaves at Done, so "
         "agreement there is learned, not taught." % OBSERVATIONAL_NOTE, "",
         _HEAD[0] % "policy", _HEAD[1]]
    for lab, sb in policies.items():
        L.append(_row(lab, sb["policy"]["all"]))
        if sb.get("opponent"):
            L.append(_row("%s (vs %s)" % (opponent, lab), sb["opponent"]["all"]))
    explored = [(lab, sb["policy"]["explored"]) for lab, sb in policies.items() if sb["policy"]["explored"]]
    if explored:
        L += ["", "Explored phases left out (a benchmark should have none): " +
              ", ".join("%s %d" % e for e in explored)]
    missing = [(lab, sb["policy"]["missing"]) for lab, sb in policies.items() if sb["policy"].get("missing")]
    if missing:
        L += ["", "Games 2–3 without sideboarding data (results from an engine before the `sb` key): " +
              ", ".join("%s %d" % e for e in missing)]
    for lab, sb in policies.items():
        sides = [(lab, sb["policy"])] + ([("%s vs %s" % (opponent, lab), sb["opponent"])] if sb.get("opponent") else [])
        for name, rep in sides:
            if not rep["all"]["phases"]:
                continue
            L += ["", "### Sideboarding per deck: %s" % name, "", _HEAD[0] % "deck", _HEAD[1]]
            L += [_row(d, rep["by_deck"][d]) for d in decks if d in rep["by_deck"]]
            L += ["", "| deck | most moved in | most cut |", "|---|---|---|"]
            L += ["| %s | %s | %s |" % (d, _cards(rep["by_deck"][d]["in"]), _cards(rep["by_deck"][d]["out"]))
                  for d in decks if d in rep["by_deck"]]
    return L
