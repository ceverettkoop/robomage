"""Adding decks to a run's league (docs/rmtrain.md §1.1; docs/open_vocab_plan.md S8): ``rmtrain
add-decks <run> STEM ... [--weight W]``.

A new deck enters training through self-play from the current policy (no teacher, no behaviour
cloning). Its cards must be in the run's tables: a deck whose cards were appended to the vocab needs
the engine rebuilt and the run retabled first (stop, ``make BUILD=RELEASE``, ``rmtrain retable``), and
``check_deck`` refuses it until then.

**The new pairs' weight.** With ``league.pairs`` empty every pair of ``league.decks`` has weight 1, so
a deck added there plays as often as any other (11 decks: each in 10 of the 55 pairs). With explicit
pairs, each new pair (the new deck against every deck of the league, and the new decks among
themselves) gets `weight`, by default the **mean weight of the run's pairs** (``default_weight``):
the new deck plays as often as an average deck. A larger weight front-loads it (more of its games
while its cards are new), at the other decks' expense. A weight other than 1 on a league without
explicit pairs writes the pairs out (every old pair at 1).

``plan_add`` computes the new ``league.decks`` / ``league.pairs``; the command applies them together
(one ``set`` of both keys, validated as one config) on a running run — both keys are live: the
orchestrator rewrites the actors' schedule at its next tick and each actor reads it before its next
match — or writes the stopped run's ``config.json`` for its next ``rmtrain start``.
"""

import itertools
import os
import subprocess

from . import config as C
from .actors import cpu_env
from .retable import engine_cards_differ, engine_paths


def pair_weights(cfg):
    """[[deck_a, deck_b, weight]] of the league (league.pairs, or every pair of league.decks at 1)."""
    lg = cfg["league"]
    if lg["pairs"]:
        return [[p[0], p[1], float(p[2]) if len(p) > 2 else 1.0] for p in lg["pairs"]]
    return [[a, b, 1.0] for a, b in itertools.combinations(lg["decks"], 2)]


def default_weight(cfg):
    """The mean weight of the league's pairs (1 with league.pairs empty)."""
    pw = pair_weights(cfg)
    return sum(p[2] for p in pw) / len(pw) if pw else 1.0


def plan_add(cfg, stems, weight=None):
    """(changes, weight, config): the keys whose values change ({"league.pairs": [...], "league.decks":
    [...]}, applied together), the new pairs' weight and the validated config with `stems` added.
    ValueError for a deck already in the league, a held-out deck or no deck."""
    lg = cfg["league"]
    new = []
    for s in stems:
        s = s[len("league/"):] if s.startswith("league/") else s
        if s.startswith("heldout/"):
            raise ValueError("%s is held out (decks/heldout/): never trained on" % s)
        if s in lg["decks"] or s in new:
            raise ValueError("%s is already in league.decks" % s)
        new.append(s)
    if not new:
        raise ValueError("no deck to add")
    w = default_weight(cfg) if weight is None else float(weight)
    if w <= 0:
        raise ValueError("--weight must be > 0")
    decks = list(lg["decks"]) + new
    changes = {}
    if lg["pairs"] or abs(w - 1.0) > 1e-12:
        pairs = pair_weights(cfg) if lg["pairs"] else [[a, b, 1.0] for a, b in itertools.combinations(lg["decks"], 2)]
        pairs += [[a, b, w] for a, b in itertools.combinations(decks, 2) if a in new or b in new]
        changes["league.pairs"] = pairs
    changes["league.decks"] = decks
    out = C.set_keys(cfg, changes)
    return changes, w, out


def check_deck(cfg, stem):
    """"" when `stem` is a league deck whose every card the run's engine loads (so its tables, which
    the run's start check ties to the engine, hold them), else why not."""
    bin_dir, engine, _ = engine_paths(cfg)
    rel = "league/%s.dk" % stem
    if not os.path.exists(os.path.join(bin_dir, "resources", "decks", rel)):
        return "no deck %s under %s" % (rel, os.path.join(bin_dir, "resources", "decks"))
    if not os.path.exists(engine):
        return "no engine %s (make BUILD=%s actor)" % (engine, cfg["run"]["build"].upper())
    r = subprocess.run([engine, "--check-decks", "--check-decks-dir", rel], cwd=bin_dir, env=cpu_env(),
                       capture_output=True, text=True)
    if r.returncode != 0:
        missing = [ln for ln in (r.stdout + r.stderr).splitlines() if "MISSING" in ln or "ERROR" in ln]
        return ("the engine does not load every card of %s (append them to the vocab, rebuild the engine, `rmtrain "
                "retable` the stopped run): %s" % (rel, "; ".join(missing[:5]) or "exit %d" % r.returncode))
    return ""


def check_engine_current(run, cfg):
    """"" when the run's engine still gives the run's cards frame (a running run: no rebuild since it
    started), else why not."""
    bin_dir, engine, _ = engine_paths(cfg)
    if os.path.exists(run.cards) and engine_cards_differ(engine, bin_dir, run.cards):
        return ("the engine's cards frame is not the run's (the engine was rebuilt with other cards since the run "
                "started): stop the run, `rmtrain retable` it, then add the decks")
    return ""
