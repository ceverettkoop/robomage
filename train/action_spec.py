"""Semantic action specs: turn a human-written string into the integer index of
the matching legal action, given the *current* decision's decoded menu.

This is the single, stateless "string -> index" resolver shared by every Python
entry point that drives the engine by intent rather than by a fragile positional
index:

  * ``PlayController`` (opponents.py) — consumes a pre-written list of specs for
    machine-driven runs (``test_harness.py --play`` / a ``play:<specs>``
    agent spec, e.g. ``train.py observe --player-a "play:..."``).
  * ``play.py`` — (documented follow-up) lets a human type ``cast bolt`` instead
    of an index at the ``Choose>`` prompt.

The engine itself stays integer-only: a spec is resolved to an index at the
decision point and that index is what gets sent / logged, so deterministic
replay is unaffected. A spec list can even be replayed back as a plain
``--actions`` integer list (see ``PlayController.resolved``).

**One spec does NOT resolve to a menu index: ``concede``.** A concession (CR
104.3a) is an out-of-band decision INPUT, not a legal action the engine offers,
so ``concede`` / ``concede:match`` resolve to the negative sentinels
``env.CONCEDE_GAME`` (-2) / ``env.CONCEDE_MATCH`` (-3). The contract for every
consumer is the same as for a normal index: **pass the resolver's ``index``
straight to ``env.step``** — which sends the sentinels through verbatim (no
confirm-slot remap, no action mask) and logs them for replay. Callers that
merely forward the returned int (``opponents.HumanController``,
``opponents.PlayController``, and ``runner.drive_game`` behind them) therefore
support ``concede`` with no change; only a caller that *validates* the index
against ``0 <= i < num_choices`` would need to special-case it.

Why this survives dynamic menus: a spec is matched against the decoded action
list for *that one decision* (category + card name + controller + the
engine-authored description), so reordering, extra priority passes, or new
triggers never invalidate it the way a hard-coded index does.

Dependency-light: only ``re`` + ``decode`` (numpy + tables; no torch, no engine).

────────────────────────────────────────────────────────────────────────────
Grammar (case-insensitive; one spec resolves to exactly one action):

    pass                      pass priority / take the first choice
    cast:<card>               cast a spell from hand            (CAST_SPELL)
    play:<card> | land:<card> play a land                       (PLAY_LAND)
    activate:<card>[@side]    activate a permanent's ability    (ACTIVATE_ABILITY)
    target:<card>[@side]      choose a card target              (SELECT_TARGET)
    target:<text>[@side]      choose a player/other target by description text
    attack:<card>             declare one attacker              (SELECT_ATTACKER)
    attack:done               confirm attackers                 (CONFIRM_ATTACKERS)
    block:<card>              declare one blocker               (SELECT_BLOCKER)
    block:done                confirm blockers                  (CONFIRM_BLOCKERS)
    mana:<color> | tap:<land> tap for mana (color w/u/b/r/g/c)  (MANA_*)
    search:<card> | search:fail   library/exile/sideboard search or fail to find
                                  (SEARCH_LIBRARY or CHOOSE_CARD, e.g. Karn's -2)
    top:<card>                put a card on top of library      (TOP_LIBRARY)
    bottom:<card>             put a card on library bottom      (BOTTOM_DECK_CARD)
    dig:<card>                pick from the dug cards            (DIG_CHOICE)
    mulligan | keep           mulligan decision                 (MULLIGAN)
    pay:<text>                pay an optional cost               (PAYING_COSTS)
    choice:<text>             a generic/modal choice            (OTHER_CHOICE)
    shuffle:<text>            shuffle decision                  (SHUFFLE)
    sb-in:<card> | sb-out:<card> | sb-done    sideboarding (bo3)
    companion:<card>          pay {3}, companion sideboard->hand (COMPANION)
    desc:<text>               match ANY action by description substring
    concede | concede:game    concede the current game (CR 104.3a); in a bo3 the
                              match continues            -> env.CONCEDE_GAME (-2)
    concede:match             concede the whole match    -> env.CONCEDE_MATCH (-3)
    #<n>  or a bare integer   literal action index (escape hatch)

  Optional leading SEAT KEY (``A:`` / ``B:``, case-insensitive) pins a spec to a
  player seat::

      A:play:Mountain     B:cast:Lightning Bolt     A:target:Grizzly Bears@opp

  This matters only when ONE spec list drives BOTH seats (the test harness's
  dual-seat ``--play`` mode). There, :class:`opponents.PlayController` looks at
  the next spec's seat: if that seat does not currently hold priority, the seat
  that *does* hold priority passes (the spec is left unconsumed) until the keyed
  seat is on the clock — so the author writes the intended line per player and
  never has to hand-interleave the priority-passes between them. A spec with no
  seat key is applied to whoever currently has priority (the legacy behaviour).
  The seat key never affects *which* action a spec resolves to within a single
  decision — only *which decision* it is applied to. ``A:``/``B:`` is the player
  seat; the separate ``@own``/``@opp`` suffix is the target's controller relative
  to the acting seat — the two are independent.

  * ``<card>`` matches case- and apostrophe-insensitively; an exact name wins,
    otherwise a unique substring match is accepted ("bolt" -> "Lightning Bolt").
  * ``@side`` is ``own`` / ``opp`` (also ``me`` / ``self`` / ``you`` -> own,
    ``them`` / ``enemy`` -> opp) to disambiguate which controller's permanent.
  * If a spec matches several legal actions, resolution is reported as ambiguous
    (the caller errors with the candidates) — add ``@side`` or use ``desc:`` /
    ``#<n>`` to pin it.
"""

import re

import decode
from env import CONCEDE_GAME, CONCEDE_MATCH   # (decode already imports env)
from _enums import (CAT_PASS_PRIORITY, CAT_CAST_SPELL, CAT_PLAY_LAND,
                    CAT_ACTIVATE_ABILITY, CAT_SELECT_TARGET, CAT_SELECT_ATTACKER,
                    CAT_CONFIRM_ATTACKERS, CAT_SELECT_BLOCKER, CAT_CONFIRM_BLOCKERS,
                    CAT_MANA_W, CAT_MANA_U, CAT_MANA_B, CAT_MANA_R, CAT_MANA_G,
                    CAT_MANA_C, CAT_SEARCH_LIBRARY, CAT_CHOOSE_CARD, CAT_TOP_LIBRARY,
                    CAT_BOTTOM_DECK_CARD, CAT_DIG_CHOICE, CAT_MULLIGAN, CAT_KEEP_HAND,
                    CAT_PAYING_COSTS, CAT_SHUFFLE, CAT_DONT_SHUFFLE,
                    CAT_SIDEBOARD_IN, CAT_SIDEBOARD_OUT, CAT_SIDEBOARD_DONE,
                    CAT_COMPANION, CAT_OTHER_CHOICE, CAT_SACRIFICE_PERMANENT,
                    CAT_RETURN_PERMANENT, CAT_CHOOSE_X, CAT_DISCARD, CAT_CHOOSE_MODE,
                    CAT_CHOOSE_MANA_COLOR, CAT_PAY_UNLESS, CAT_NAME_CARD,
                    CAT_CHOOSE_TYPE, CAT_KEEP_LEGEND, CAT_ORDER_TRIGGERS,
                    CAT_CHOOSE_REPLACEMENT, CAT_ATTACK_TARGET, CAT_BLOCK_TARGET,
                    CAT_OPTIONAL_YESNO, CAT_SYLVAN_CHOICE, CAT_EXILE_FROM_YARD)

# ── Verb -> the ActionCategory ints it may resolve to ─────────────────────────
# (the generated CAT_* names from train/_enums.py, mirroring src/classes/action.h.)
# `choice` stays broad — it resolves against every dedicated choice category
# (plus the OTHER_CHOICE fallback) — while the verbs below pin a specific kind of
# decision.
_OTHER_CATS = {CAT_OTHER_CHOICE, CAT_SACRIFICE_PERMANENT, CAT_RETURN_PERMANENT,
               CAT_CHOOSE_X, CAT_DISCARD, CAT_CHOOSE_MODE, CAT_CHOOSE_MANA_COLOR,
               CAT_PAY_UNLESS, CAT_NAME_CARD, CAT_CHOOSE_TYPE, CAT_KEEP_LEGEND,
               CAT_ORDER_TRIGGERS, CAT_CHOOSE_REPLACEMENT, CAT_ATTACK_TARGET,
               CAT_BLOCK_TARGET, CAT_OPTIONAL_YESNO, CAT_SYLVAN_CHOICE,
               CAT_CHOOSE_CARD, CAT_DONT_SHUFFLE, CAT_KEEP_HAND,
               CAT_EXILE_FROM_YARD}
_MANA_COLOR_CATS = {CAT_MANA_W, CAT_MANA_U, CAT_MANA_B, CAT_MANA_R, CAT_MANA_G,
                    CAT_MANA_C}
_VERB_CATS = {
    "pass": {CAT_PASS_PRIORITY},
    "cast": {CAT_CAST_SPELL},
    "play": {CAT_PLAY_LAND}, "land": {CAT_PLAY_LAND},
    "activate": {CAT_ACTIVATE_ABILITY},
    "target": {CAT_SELECT_TARGET},
    "attack": {CAT_SELECT_ATTACKER, CAT_CONFIRM_ATTACKERS},
    "block": {CAT_SELECT_BLOCKER, CAT_CONFIRM_BLOCKERS},
    "mana": set(_MANA_COLOR_CATS), "tap": set(_MANA_COLOR_CATS),
    "search": {CAT_SEARCH_LIBRARY, CAT_CHOOSE_CARD}, "top": {CAT_TOP_LIBRARY},
    "bottom": {CAT_BOTTOM_DECK_CARD}, "dig": {CAT_DIG_CHOICE},
    "mulligan": {CAT_MULLIGAN}, "keep": {CAT_KEEP_HAND},
    "pay": {CAT_PAYING_COSTS}, "choice": set(_OTHER_CATS), "shuffle": {CAT_SHUFFLE},
    "noshuffle": {CAT_DONT_SHUFFLE},
    "sb-in": {CAT_SIDEBOARD_IN}, "sb-out": {CAT_SIDEBOARD_OUT},
    "sb-done": {CAT_SIDEBOARD_DONE}, "companion": {CAT_COMPANION},
    # Convenience verbs that pin a specific choice kind.
    "sacrifice": {CAT_SACRIFICE_PERMANENT}, "return": {CAT_RETURN_PERMANENT},
    "x": {CAT_CHOOSE_X}, "discard": {CAT_DISCARD},
    "mode": {CAT_CHOOSE_MODE}, "color": {CAT_CHOOSE_MANA_COLOR}, "name": {CAT_NAME_CARD},
    "exile": {CAT_EXILE_FROM_YARD},   # exile a card from the graveyard to pay an Escape cost
    "desc": set(),   # any category — pure description-substring match
    # Not a menu action at all: resolves to a negative concession sentinel that
    # the caller steps directly (see the module docstring's contract note).
    "concede": set(),
}

# `concede:<arg>` -> sentinel. A bare `concede` (empty arg) is the GAME concede,
# the conservative reading of "I'm done with this one" in a bo3.
_CONCEDE_SENTINEL = {
    "": CONCEDE_GAME, "game": CONCEDE_GAME, "g": CONCEDE_GAME,
    "match": CONCEDE_MATCH, "m": CONCEDE_MATCH,
}

_COLOR_CAT = {"w": CAT_MANA_W, "u": CAT_MANA_U, "b": CAT_MANA_B, "r": CAT_MANA_R,
              "g": CAT_MANA_G, "c": CAT_MANA_C}

# @side aliases -> the decode controller word ("own"/"opp").
_SIDE_ALIASES = {
    "own": "own", "me": "own", "my": "own", "self": "own", "you": "own", "mine": "own",
    "opp": "opp", "them": "opp", "their": "opp", "enemy": "opp", "opponent": "opp",
}


# Leading seat key: a single 'A'/'B' (case-insensitive) followed by ':'. No verb
# is a single letter, so this never collides with the verb grammar.
_SEAT_RE = re.compile(r"\s*([abAB])\s*:\s*(.*)$", re.DOTALL)


def spec_seat(token):
    """Return the player seat ('A' or 'B') a spec is pinned to, or None.

    A cheap front-of-string check (does not validate the rest of the spec) used
    by :class:`opponents.PlayController` to decide, before resolving, whether the
    current priority holder should pass because the next spec belongs to the
    other seat."""
    m = _SEAT_RE.match(token or "")
    return m.group(1).upper() if m else None


def _norm(s):
    """Normalize a name/description for matching: lowercase, drop apostrophes and
    punctuation, collapse whitespace."""
    s = (s or "").lower().replace("'", "").replace("’", "")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class Intent:
    """A parsed spec: what the user wants, independent of any concrete menu."""

    __slots__ = ("verb", "cats", "card", "side", "seat", "color", "keyword",
                 "literal", "sentinel", "raw")

    def __init__(self, raw):
        self.raw = raw
        self.verb = None
        self.cats = set()       # candidate categories (empty => any)
        self.card = None        # normalized card/description text, or None
        self.side = None        # 'own' | 'opp' | None (target's controller)
        self.seat = None        # 'A' | 'B' | None (player seat this spec is for)
        self.color = None       # mana category int, or None
        self.keyword = None     # 'done' | 'fail' | 'keep' | 'mulligan' | None
        self.literal = None     # literal action index for '#N' / bare int
        self.sentinel = None    # out-of-band decision input (concede), or None


class ResolveResult:
    """Outcome of resolving one spec against one decoded menu."""

    __slots__ = ("index", "ok", "reason", "candidates", "kind")

    def __init__(self, index, ok, reason="", candidates=None, kind=""):
        self.index = index
        self.ok = ok
        self.reason = reason
        self.candidates = candidates or []
        # Coarse failure category for callers that branch on *why* resolution failed
        # (the PlayController auto-advances only on "no_match"): one of "ok",
        # "parse_error", "bad_index", "ambiguous", "no_match".
        self.kind = kind or ("ok" if ok else "no_match")


def parse_spec_list(s):
    """Split a comma-separated spec string into a list of tokens (whitespace
    stripped, empties dropped). Card names never contain commas, so comma is a
    safe separator — consistent with ``--actions``."""
    if not s:
        return []
    if isinstance(s, (list, tuple)):
        return [str(t).strip() for t in s if str(t).strip()]
    return [t.strip() for t in s.split(",") if t.strip()]


def parse_spec(token):
    """Parse one spec token into an :class:`Intent`. Raises ValueError on an
    unknown verb so typos fail loudly rather than silently passing."""
    raw = token.strip()
    intent = Intent(raw)

    # Optional leading seat key ('A:' / 'B:') — strip it off and remember the
    # seat; the rest of the token is parsed exactly as before. (Resolution
    # against a menu ignores the seat; PlayController consumes it separately.)
    m = _SEAT_RE.match(raw)
    if m:
        intent.seat = m.group(1).upper()
        raw = m.group(2).strip()
        if not raw:
            raise ValueError(f"seat key with no action in spec {token!r}")

    # Literal index escape hatch: '#7' or a bare integer.
    m = re.fullmatch(r"#?(\d+)", raw)
    if m:
        intent.literal = int(m.group(1))
        return intent

    verb, _, arg = raw.partition(":")
    verb = verb.strip().lower()
    arg = arg.strip()

    if verb not in _VERB_CATS:
        raise ValueError(f"unknown action verb {verb!r} in spec {raw!r}")
    intent.verb = verb
    intent.cats = set(_VERB_CATS[verb])

    # Pull a trailing @side off the argument.
    if "@" in arg:
        arg, _, side = arg.rpartition("@")
        arg = arg.strip()
        side_key = side.strip().lower()
        intent.side = _SIDE_ALIASES.get(side_key)
        if intent.side is None:
            raise ValueError(f"unknown @side {side!r} in spec {raw!r}")

    low = arg.lower()

    if verb == "concede":
        # Resolves to a sentinel, never to a menu index — the argument only
        # picks WHICH concession, so an unknown one is a typo worth erroring on
        # (silently conceding the whole match on a misspelling would be awful).
        if low not in _CONCEDE_SENTINEL:
            raise ValueError(f"unknown concede scope {arg!r} in spec {raw!r} "
                             f"(use 'concede' / 'concede:game' / 'concede:match')")
        intent.sentinel = _CONCEDE_SENTINEL[low]
    elif verb in ("mulligan", "keep"):
        intent.keyword = verb
    elif verb in ("attack", "block") and low in ("done", "confirm", ""):
        intent.keyword = "done"
    elif verb == "search" and low in ("fail", "none", ""):
        intent.keyword = "fail"
    elif verb in ("mana", "tap") and low in _COLOR_CAT:
        intent.color = _COLOR_CAT[low]
    else:
        # Everything else carries a card name / description text (may be empty
        # for verbs like 'pass' / 'shuffle' / 'sb-done').
        intent.card = _norm(arg) if arg else None

    return intent


def _card_matches(intent_card, action, exact):
    """Does ``action`` satisfy the card/description text of the intent?

    For a card-bearing action, match its decoded name (exact equality on the
    exact pass, substring on the loose pass). For a cardless action (a player
    target or a modal choice), match the engine-authored description text by
    substring in both passes — there is no "name" to be exact about."""
    name = _norm(action.get("card")) if action.get("card") else ""
    if name:
        return name == intent_card if exact else intent_card in name
    return intent_card in _norm(action.get("description"))


def _matches(intent, action, exact):
    cat = action["category"]
    if intent.cats and cat not in intent.cats:
        return False
    if intent.side and action.get("controller") != intent.side:
        return False
    if intent.color is not None and cat != intent.color:
        return False

    if intent.keyword == "done":
        return cat in (CAT_CONFIRM_ATTACKERS, CAT_CONFIRM_BLOCKERS)
    if intent.keyword == "fail":
        return action.get("card") is None
    if intent.keyword in ("keep", "mulligan"):
        return intent.keyword in _norm(action.get("description"))

    # `desc:` is a pure description-substring match — match the engine-authored
    # description text directly even when the action also carries a card name
    # (e.g. picking one of a planeswalker's several same-named loyalty abilities:
    # "Activate Karn, the Great Creator (ChangeZone)" vs "(Animate)").
    if intent.verb == "desc" and intent.card:
        return intent.card in _norm(action.get("description"))

    if intent.card:
        return _card_matches(intent.card, action, exact)
    # No card text: category (+ side / color) alone identifies it.
    return True


def resolve(token, actions):
    """Resolve ``token`` against ``actions`` (a decoded menu from
    ``decode.decode_actions_from_obs``). Returns a :class:`ResolveResult`.

    Exact card-name matches are preferred; only if none match is a unique
    substring match accepted. Multiple matches at either level => ambiguous.

    ``index`` is a menu index for every verb except ``concede``, which yields a
    negative out-of-band sentinel; either way the caller steps it directly."""
    try:
        intent = parse_spec(token)
    except ValueError as e:
        return ResolveResult(None, False, reason=str(e), kind="parse_error")

    # A concession is legal at EVERY decision and is not in the menu, so it
    # resolves without looking at `actions` at all. The returned index is the
    # negative sentinel the caller steps verbatim (see the module docstring).
    if intent.sentinel is not None:
        return ResolveResult(intent.sentinel, True)

    if intent.literal is not None:
        if 0 <= intent.literal < len(actions):
            return ResolveResult(intent.literal, True)
        return ResolveResult(None, False,
                             reason=f"literal index {intent.literal} out of range "
                                    f"(0..{len(actions) - 1})", kind="bad_index")

    for exact in (True, False):
        hits = [a for a in actions if _matches(intent, a, exact)]
        if not hits:
            continue
        # Collapse semantically identical choices (e.g. four "Play Mountain" from
        # a hand of duplicates, or two copies of the same card on the stack) —
        # they are interchangeable, so take the first. Distinct matches
        # (different cards/targets/controllers) are genuinely ambiguous and error
        # so the caller can pin them with @side / desc: / #<n>.
        keys = {(a["category"], _norm(a.get("card")), a.get("controller")) for a in hits}
        if len(keys) == 1:
            return ResolveResult(hits[0]["index"], True)
        return ResolveResult(None, False,
                             reason=f"{token!r} is ambiguous ({len(hits)} matches)",
                             candidates=hits, kind="ambiguous")
    return ResolveResult(None, False, reason=f"no legal action matches {token!r}",
                         candidates=actions)


def format_menu(actions):
    """One-line rendering of a decoded menu, for error messages."""
    return "  ".join(f"[{a['index']}] {a['description']}" for a in actions)


class PlayResolveError(Exception):
    """A spec could not be resolved to exactly one legal action."""

    def __init__(self, result, actions):
        self.result = result
        self.actions = actions
        super().__init__(
            f"{result.reason}\n    legal actions: {format_menu(actions)}")


def resolve_to_index(token, actions):
    """Resolve and return the index, raising :class:`PlayResolveError` on
    failure. For callers that prefer exceptions to branching on a result."""
    r = resolve(token, actions)
    if not r.ok:
        raise PlayResolveError(r, actions)
    return r.index
