"""The league sideboard guides and the ``scripted:hard+sb`` tier's sideboarding over them.

``docs/sideboard_guides.md`` is the guides' text; ``sideboard_guides.json`` beside this module is
its machine-readable copy (``pycheck.py``'s ``guides`` part checks that the two agree):

- ``draw_land``: per league deck (its stem under ``bin/resources/decks/league/``) the land it cuts
  on the draw, null for a deck that cuts none;
- ``guides``: per (deck, opponent) the ``in`` and ``out`` lists (card name -> copies) and the
  ``draw`` card (the "Draw +1" card coming in for the draw land; null when the deck has no draw
  land);
- ``holdout``: the (deck, opponent) pairs the tier leaves at Done (the generalisation test,
  ``docs/rnad_training_plan.md`` §6.2); their rows stay in ``guides``.

Card names are spelled as in the league deck files (``Dead/Gone``, ``Cloak and Dagger,
Entwined``); ``card_key`` compares names across spellings (a frame's front face name, a doc's
``Dead // Gone``).

**The tier** (``choose``): at a sideboarding decision the seat identifies its own deck and the
opponent's by matching the registered lists (the DECKLISTS frame; main and sideboard separately)
against the league deck files; an unknown deck on either side, a held-out pair or a pair with no
guide row answers Done. Otherwise the target maindeck for the coming game is the registered main
minus *Out* plus *In*, and on the draw (the seat does not play first in the coming game, the
``match`` table's ``plays_first``) also minus one draw land plus the *Draw +1* card. The seat moves
from its current maindeck (the ``own_deck`` table: game 2's boarded deck when game 3's phase
starts) toward the target on the engine's in-first menu: at a balanced deck the first ``Sideboard
in`` row (menu order) of a card the target has more of than the current main, else Done; with a
move outstanding the first ``Sideboard out`` row of a card the current main has more of than the
target, else the first ``Sideboard out`` row (only reachable from a deck the tier did not build).
Each phase therefore ends after exactly the swaps between the current main and the target, and
never loops: a card moved one way is never offered back the other way in the same phase (the
engine's one-shot locks), and the tier never moves a card away from its target count.

Stdlib only.
"""
from __future__ import annotations

import json
import os
import unicodedata

_HERE = os.path.dirname(os.path.abspath(__file__))
GUIDES_JSON = os.path.join(_HERE, "sideboard_guides.json")
LEAGUE_DIR = os.path.join(os.path.dirname(_HERE), "bin", "resources", "decks", "league")

_CAT_SB_IN = "SIDEBOARD_IN"
_CAT_SB_OUT = "SIDEBOARD_OUT"
_CAT_SB_DONE = "SIDEBOARD_DONE"
SIDEBOARD_CATS = frozenset({_CAT_SB_IN, _CAT_SB_OUT, _CAT_SB_DONE})


def card_key(name: str) -> str:
    """A card name compared across spellings: accents folded, lower case, letters and digits only
    (``Dead/Gone``, ``Dead // Gone`` and a front face ``Dead`` + back ``Gone`` all read
    ``deadgone``)."""
    folded = unicodedata.normalize("NFKD", name)
    return "".join(c for c in folded.lower() if c.isascii() and c.isalnum())


def parse_deck_file(path: str):
    """A ``.dk`` deck file as (main, side): card name -> copies, names as the file spells them."""
    main, side = {}, {}
    cur = main
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.upper().startswith("SIDEBOARD"):
                cur = side
                continue
            qty, _, name = s.partition(" ")
            if not qty.isdigit() or not name:
                continue
            cur[name.strip()] = cur.get(name.strip(), 0) + int(qty)
    return main, side


def read_league_decks(league_dir: str = LEAGUE_DIR) -> dict:
    """Every league deck file as stem -> (main, side)."""
    out = {}
    for fn in sorted(os.listdir(league_dir)):
        if fn.endswith(".dk"):
            out[fn[:-3]] = parse_deck_file(os.path.join(league_dir, fn))
    return out


class Guides:
    """The guides (``sideboard_guides.json``) over the league deck files."""

    def __init__(self, data: dict, decks: dict):
        self.draw_land = data["draw_land"]
        self.guides = data["guides"]
        self.holdout = {tuple(p) for p in data["holdout"]}
        self.decks = decks

    def row(self, own: str, opp: str):
        """The guide row of ``own`` against ``opp`` (``in``, ``out``, ``draw``), or None."""
        return self.guides.get(own, {}).get(opp)

    def target(self, own: str, opp: str, on_play: bool, holdout: bool = True):
        """The target maindeck (deck-file names -> copies) of ``own`` against ``opp`` on the play or
        on the draw; None for an unknown deck, a pair with no row or (``holdout``) a held-out pair."""
        if own not in self.decks or (holdout and (own, opp) in self.holdout):
            return None
        row = self.row(own, opp)
        if row is None:
            return None
        main = dict(self.decks[own][0])
        _move(main, row["out"], -1)
        _move(main, row["in"], +1)
        land = self.draw_land.get(own)
        if not on_play and land is not None and row["draw"] is not None:
            _move(main, {land: 1}, -1)
            _move(main, {row["draw"]: 1}, +1)
        return main

    def identify(self, main: dict, side: dict, resolve):
        """The league deck whose file lists exactly ``main`` and ``side`` (names as ``resolve`` maps a
        deck-file name), or None."""
        for stem, (dmain, dside) in self.decks.items():
            if _resolved(dmain, resolve) == main and _resolved(dside, resolve) == side:
                return stem
        return None


def _move(deck: dict, cards: dict, sign: int):
    for name, n in cards.items():
        left = deck.get(name, 0) + sign * n
        if left:
            deck[name] = left
        else:
            deck.pop(name, None)


def _resolved(deck: dict, resolve) -> dict | None:
    """``deck`` with each name mapped by ``resolve``; None when a name does not resolve."""
    out = {}
    for name, n in deck.items():
        r = resolve(name)
        if r is None:
            return None
        out[r] = out.get(r, 0) + n
    return out


_GUIDES = None


def guides() -> Guides:
    """The guides, read once per process."""
    global _GUIDES
    if _GUIDES is None:
        with open(GUIDES_JSON, encoding="utf-8") as f:
            _GUIDES = Guides(json.load(f), read_league_decks())
    return _GUIDES


def target_main(v):
    """The seat's target maindeck for the coming game (names as the view's cards name them), or
    None when the tier answers Done (module docstring)."""
    own_list, opp_list = v.registered_list(True), v.registered_list(False)
    if own_list is None or opp_list is None:
        return None
    g = guides()
    resolve = v.cards.canonical
    own, opp = g.identify(*own_list, resolve), g.identify(*opp_list, resolve)
    if own is None or opp is None:
        return None
    target = g.target(own, opp, v.plays_first)
    return None if target is None else _resolved(target, resolve)


def choose(v) -> int:
    """The ``scripted:hard+sb`` tier's row at a sideboarding menu (module docstring)."""
    cats = v.cats
    done = next((i for i, c in enumerate(cats) if c == _CAT_SB_DONE), None)
    target = target_main(v)
    if target is None:
        if done is not None:
            return done
        return next(i for i, c in enumerate(cats) if c in (_CAT_SB_IN, _CAT_SB_OUT))
    cur = v.own_current_main
    if done is not None:
        for i, c in enumerate(cats):
            if c == _CAT_SB_IN and target.get(v.card(i), 0) > cur.get(v.card(i), 0):
                return i
        return done
    outs = [i for i, c in enumerate(cats) if c == _CAT_SB_OUT]
    for i in outs:
        if cur.get(v.card(i), 0) > target.get(v.card(i), 0):
            return i
    return outs[0]
