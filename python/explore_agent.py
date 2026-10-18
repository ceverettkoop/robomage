"""The explore tiers' decision rule: the coverage fuzzer (``explore`` / ``explore:patient``).

``explore_choice(view, seen, patient)`` is the whole tier: ``scripted_core.ScriptedCore`` calls it
for its EXPLORE skill (the ``explore`` / ``fuzz`` and ``explore:patient`` / ``patient`` presets)
with its per-game novelty set. It reads the same view the rest of the agent reads (the
``scripted_core`` docstring), on either protocol, and only these parts of it: ``n``, ``cats``
(category names), ``card(i)`` (the action's card name or None), ``hand`` (own hand card names) and
``cards.mana_value(name)``. No numpy, no layout.

**Decision rules** (in order; what a port, e.g. the C++ explore agent of plan step 3a, implements):

1. A menu of one row (or none) is answered 0 with no draw.
2. A menu holding a ``MULLIGAN`` row starts a game: the novelty set ``seen`` is cleared.
3. Each row ``i`` gets a weight: its category's entry in the profile's table (``_EXPLORE_WEIGHTS``,
   or ``_EXPLORE_PATIENT_WEIGHTS`` for ``patient``), ``_EXPLORE_DEFAULT_WEIGHT`` (2.0) for a
   category not listed. Then, in this order:
   - a ``SEARCH_LIBRARY`` row with no card (the fail-to-find slot): times
     ``_EXPLORE_SEARCH_FAIL_WEIGHT`` (0.3);
   - a ``CAST_SPELL`` / ``ACTIVATE_ABILITY`` row with a card not in ``seen``: times
     ``_EXPLORE_NOVELTY_FACTOR`` (3.0), except an ``ACTIVATE_ABILITY`` row under ``patient``;
   - under ``patient``, a ``CAST_SPELL`` row with a card: times ``1 + mana value``
     (``_PATIENT_CAST_MV_FACTOR``); an ``ACTIVATE_ABILITY`` row whose card is in the own hand:
     divided by ``1 + mana value`` (``_PATIENT_HOLD_MV_FACTOR``). Mana value is the card's printed
     one (X counts 0).
4. A two-row menu whose rows are both ``MULLIGAN`` gets weights ``[3, 1]``. (The engine's
   mulligan menu is ``KEEP_HAND`` + ``MULLIGAN``, so this rule never fires; it stays so the explore
   games on both protocols are unchanged. A port may apply the intended keep bias instead.)
5. Weights summing to 0 or less become uniform.
6. The row is drawn by Python's global ``random``: ``random.choices(range(n), weights)``, i.e. one
   ``random.random()`` ``u`` and the first row whose cumulative weight exceeds ``u * total``. The
   drivers seed ``random`` with the game's seed before each game (``random.seed(seed + i)``), so a
   seed fixes the whole trajectory; a port with its own generator plays other (equally valid) games.
7. The chosen row's card, for a ``CAST_SPELL`` / ``ACTIVATE_ABILITY`` row with a card, joins
   ``seen`` for the rest of the game.

Not a strong opponent: the presets turn the combo lines off, and the tier never reads the board
beyond the hand. Its purpose is coverage: sweeping seeds touches as many engine paths as possible.
"""

import random

_CAT_PASS = "PASS_PRIORITY"
_CAT_SEL_ATK = "SELECT_ATTACKER"
_CAT_CONF_ATK = "CONFIRM_ATTACKERS"
_CAT_SEL_BLK = "SELECT_BLOCKER"
_CAT_CONF_BLK = "CONFIRM_BLOCKERS"
_CAT_ACTIVATE = "ACTIVATE_ABILITY"
_CAT_CAST = "CAST_SPELL"
_CAT_LAND = "PLAY_LAND"
_CAT_MULLIGAN = "MULLIGAN"
_CAT_SEARCH = "SEARCH_LIBRARY"
_CAT_SB_DONE = "SIDEBOARD_DONE"
_CAT_COMPANION = "COMPANION"

# Per-action-category weights for the EXPLORE tier (explore_choice). Higher weight =
# more likely to be chosen. The bias is toward the categories that exercise the most
# engine code (casting, activating, combat), with PASS kept low-but-nonzero so the game
# still advances, and the "finish this phase" confirms (attackers/blockers/sideboard)
# kept below their per-item counterparts so a few items get declared before confirming.
# Any category not listed (targets, X, modes, mana, search, dig, discard, sacrifice,
# name-a-card, ...) uses _EXPLORE_DEFAULT_WEIGHT — a plain uniform pick among that
# decision's options, so which target / X / mode is taken varies freely across seeds
# (one exception: the SEARCH_LIBRARY fail-to-find slot, see
# _EXPLORE_SEARCH_FAIL_WEIGHT).
_EXPLORE_DEFAULT_WEIGHT = 2.0
_EXPLORE_WEIGHTS: dict[str, float] = {
    _CAT_CAST:     8.0,   # cast a spell — stack, resolution, effects, triggers, SBAs
    _CAT_ACTIVATE: 6.0,   # activated abilities (mana, sacrifice, ETB-independent effects)
    _CAT_LAND:     4.0,   # play a land (fuels casts; land ETBs / fetch / man-lands)
    _CAT_SEL_ATK:  3.0,   # declare an attacker — combat triggers, damage, first strike
    _CAT_SEL_BLK:  3.0,   # declare a blocker — blocking, combat damage assignment
    _CAT_PASS:     1.0,   # advance priority / the game (low, but never zero)
    _CAT_CONF_ATK: 0.6,   # finish declaring attackers (declare a few first)
    _CAT_CONF_BLK: 0.6,   # finish declaring blockers
    _CAT_SB_DONE:  0.6,   # finish sideboarding (swap a few cards first, bo3)
    _CAT_COMPANION: 8.0,  # once-per-game and rarely offered — take it when legal so
                          # the companion machinery (and the card itself) gets fuzzed
}

# Novelty bias for the EXPLORE tier: a CAST_SPELL / ACTIVATE_ABILITY option whose
# card this agent has NOT yet chosen this game gets its category weight
# multiplied by this factor. Without it, expensive cards sharing a deck with
# cheaper lines were NEVER cast across a whole seed sweep (e.g. Meteor Sword,
# hard-cast Lorien Revealed) — the weighted draw kept landing on the cheap
# alternatives. First-time options winning ~3x more often fixes that while the
# already-seen cards keep their normal weight. Tracked per game in
# the caller's ``seen`` set (``ScriptedCore._explore_seen``, cleared by ``new_game``).
_EXPLORE_NOVELTY_FACTOR = 3.0

# SEARCH_LIBRARY fail-to-find de-weight (relative to 1.0 per real pick; applies to
# BOTH explore profiles — a category-level tweak, not a patient one). A library
# search menu offers "Fail to find" alongside the real picks at the same uniform
# default weight, so a uniform draw failed ~1,200 tutors across a 441-game fuzz
# campaign — wasting coverage of the ChangeZone search/shuffle/ETB paths behind
# every fetch land and tutor. 0.3 keeps the fail branch reachable (the
# shuffle-without-find path still wants occasional fuzzing) while a real pick wins
# the large majority of searches. The fail slot is identified by its missing card,
# not by index: the engine builds it from Entity(0) — no card — while every real
# pick is a library card. Mandatory searches simply have no such slot, and a
# fail-only menu is a 1-choice decision handled before weighting.
_EXPLORE_SEARCH_FAIL_WEIGHT = 0.3

# ── "Patient" big-mana EXPLORE profile (explore:patient) ────────────────────
# The plain-explore weight draw only picks among LEGAL actions, and empirically
# (130+ explore games) expensive cards never became legal: explore never develops
# enough mana for a 7-drop, and cheap alternate lines with the same card (e.g.
# islandcycling Lorien Revealed) remove the card from hand long before it could be
# hard-cast. No weighting among legal options fixes availability — this profile
# does, by developing mana and holding expensive cards. Shares all the EXPLORE
# machinery (novelty bias on casts, seed-mixed global-random draw); only the
# weights below differ. Plain "explore" reads only _EXPLORE_WEIGHTS and is
# untouched by this table.
_EXPLORE_PATIENT_WEIGHTS: dict[str, float] = {
    _CAT_CAST:     8.0,   # base; scaled up by mana value (_PATIENT_CAST_MV_FACTOR)
                          # so a legal expensive cast dominates the draw
    _CAT_ACTIVATE: 1.0,   # demoted hard: cheap activations (cycling, sac lines)
                          # are what strip expensive cards from hand early
    _CAT_LAND:     100.0, # near-always take the land drop — mana development is
                          # the whole point of this profile
    _CAT_SEL_ATK:  3.0,   # keep combat as in plain explore so games stay decisive
    _CAT_SEL_BLK:  3.0,
    _CAT_PASS:     6.0,   # bank mana-development turns: an early cheap ACTIVATE
                          # now competes with a strong PASS instead of winning
    _CAT_CONF_ATK: 0.6,   # confirms below their per-item counterparts, as in
    _CAT_CONF_BLK: 0.6,   # _EXPLORE_WEIGHTS, so a few get declared first
    _CAT_SB_DONE:  0.6,
    _CAT_COMPANION: 8.0,  # as in _EXPLORE_WEIGHTS: once-per-game, take it when legal
}

# Patient-only CAST scaling: weight *= (1 + mana_value * factor). A legal MV-7
# cast gets 8x the pull of an MV-0 one, so once the mana finally exists the big
# card is cast almost immediately instead of losing the draw to cheap spells.
_PATIENT_CAST_MV_FACTOR = 1.0

# Patient mode gives NO novelty boost to ACTIVATE options. The novelty factor is
# what makes plain explore eagerly try an unseen islandcycle/sac activation the
# first turn it appears — exactly the card-stripping behaviour this profile must
# suppress. Casts keep the boost (a novel expensive cast should win the draw).
_PATIENT_ACTIVATE_NOVELTY = False

# Patient-only "hold" scaling for card-consuming activations: an ACTIVATE whose
# card is still in the acting player's HAND (cycling, Talon Gates-style
# from-hand lines) gets its weight DIVIDED by (1 + mana_value * factor) — the
# mirror image of the CAST scaling. A flat ACTIVATE demotion is not enough: the
# option reappears at every priority window, so even a small per-window pick
# probability compounds into "always cycled before turn 5" over a game. Scaling
# the demotion by MV holds exactly the cards this profile exists to hard-cast
# while leaving cheap from-hand activations barely affected. Battlefield
# activations (Wasteland, equip, ...) are untouched — their card is not in hand.
_PATIENT_HOLD_MV_FACTOR = 1.0


def explore_choice(v, seen: set, patient: bool = False) -> int:
    """The explore tier's action for the decision ``v`` shows (the module docstring's rules).
    ``seen`` is the per-game novelty set (updated in place); ``patient`` selects the big-mana
    profile."""
    num_choices = v.n
    if num_choices <= 1:
        return 0
    cats = v.cats
    # A game always opens with a mulligan query, so seeing one means a new game
    # started: clear the novelty set even when nobody called new_game(). Repeated clears during
    # London mulligans are harmless — nothing is cast pregame.
    if any(c == _CAT_MULLIGAN for c in cats):
        seen.clear()
    # The "patient" profile swaps in the big-mana weight table, scales CAST
    # weight by mana value, and withholds the novelty boost from ACTIVATE
    # (see the _EXPLORE_PATIENT_* constants). The default profile takes the
    # exact same code path with the original table, so plain explore
    # behaviour (and its RNG consumption) is unchanged.
    table = _EXPLORE_PATIENT_WEIGHTS if patient else _EXPLORE_WEIGHTS
    weights = []
    for i, c in enumerate(cats):
        w = table.get(c, _EXPLORE_DEFAULT_WEIGHT)
        # De-weight a search's "Fail to find" (the card-less slot) so the
        # tutor/fetch usually finds — see _EXPLORE_SEARCH_FAIL_WEIGHT.
        if c == _CAT_SEARCH and v.card(i) is None:
            w *= _EXPLORE_SEARCH_FAIL_WEIGHT
        if c in (_CAT_CAST, _CAT_ACTIVATE):
            card = v.card(i)
            if card is not None and card not in seen:
                if c == _CAT_CAST or not patient or _PATIENT_ACTIVATE_NOVELTY:
                    w *= _EXPLORE_NOVELTY_FACTOR
            if patient and card is not None:
                if c == _CAT_CAST:
                    w *= 1.0 + v.cards.mana_value(card) * _PATIENT_CAST_MV_FACTOR
                elif card in v.hand:
                    # Card-consuming from-hand activation (cycling etc.):
                    # hold the card — see _PATIENT_HOLD_MV_FACTOR.
                    w /= 1.0 + v.cards.mana_value(card) * _PATIENT_HOLD_MV_FACTOR
        weights.append(w)
    # Mulligan is a 2-way [keep, mulligan] choice — bias toward keeping (index 0)
    # so most games actually start, but still mulligan ~1/4 of the time to exercise
    # the mulligan + London-bottoming paths.
    if num_choices == 2 and all(c == _CAT_MULLIGAN for c in cats):
        weights = [3.0, 1.0]
    if sum(weights) <= 0:
        weights = [1.0] * num_choices
    choice = int(random.choices(range(num_choices), weights=weights, k=1)[0])
    # Record the chosen card so this game's later draws treat it as non-novel.
    if cats[choice] in (_CAT_CAST, _CAT_ACTIVATE):
        card = v.card(choice)
        if card is not None:
            seen.add(card)
    return choice
