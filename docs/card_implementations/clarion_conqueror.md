# Clarion Conqueror  (vocab index 108)

## Oracle text
Flying

Activated abilities of artifacts, creatures, and planeswalkers can't be activated.

(3/3 Dragon, mana cost {2}{W}.)

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/c/clarion_conqueror.txt`
- Key tags:
  - `K:Flying` — the Flying keyword (rule 702.9); already a generic engine keyword.
  - `S:Mode$ CantBeActivated | AffectedZone$ Battlefield | ValidCard$ Artifact,Creature,Planeswalker | ValidSA$ Activated`
    — static ability: no activated ability of any artifact, creature, or planeswalker on the
    battlefield can be activated (symmetric — affects both players, including Clarion Conqueror's
    own controller and Clarion Conqueror itself).

## Engine work
All changes are general (keyed on the tag's intended meaning), not card-specific.

- **`CantBeActivated` static — generalized from a single type to a type list (extension of an
  existing mechanic).** The engine already supported `CantBeActivated` for the single-type case
  (`ValidCard$ Artifact`, used by Null Rod / Collector Ouphe) and the NamedCard case (Disruptor
  Flute). Clarion Conqueror needs the multi-type filter `Artifact,Creature,Planeswalker`.
  - `src/parse.cpp`: a `CantBeActivated` static now stores the full `ValidCard$` value verbatim
    in `cant_activate_card_filter` (e.g. `"Artifact"` or `"Artifact,Creature,Planeswalker"`),
    instead of hardcoding `"Artifact"`. The NamedCard variant still sets `match_named_card` and
    leaves the type filter empty.
  - `src/systems/rules_modifying.cpp`: added a static helper `permanent_matches_type_filter`
    that splits the comma-separated filter and returns true if the permanent has any of the
    named card types. Both `mana_activation_prohibited` and `activation_prohibited` now use it,
    so the same predicate covers Null Rod's single type and Clarion Conqueror's three types.
- **No new field or new category** — the existing `cant_activate_card_filter` string was
  repurposed from "single type name" to "comma-separated type list", a strict superset of the
  prior behavior (a one-element list is the old case).

## Behavioral decisions (made in lieu of asking a human)
- **The restriction is symmetric (CR 113.6 / the ability has no "you control" qualifier).** The
  static affects every artifact, creature, and planeswalker on the battlefield regardless of
  controller. `g_active_statics` are gathered for all permanents and the predicates do not filter
  by controller, so both players' permanents — and Clarion Conqueror's own controller's mana
  dorks — are shut off. Verified the opponent's Knight of the Reliquary ability disappears, and
  the controller's own Birds of Paradise / Knight abilities are likewise suppressed.
- **Mana abilities of artifacts and creatures ARE activated abilities and are prohibited (CR
  605.1a — a mana ability is a kind of activated ability).** Clarion turns off Llanowar-style
  creatures and mana rocks. Implemented by routing `mana_activation_prohibited` through the same
  type-filter helper, so a Birds of Paradise mana ability no longer appears as a payment source.
  Verified: with Clarion in play, Birds of Paradise produces no mana option.
- **Lands are unaffected.** "Land" is not in the `ValidCard$` filter, so basic-land mana abilities
  (and other land activated abilities) still work; the controller can still produce mana and cast
  spells normally. Verified in full games — the Clarion deck plays out lands and spells normally.
- **Triggered and static abilities are unaffected** — only activated abilities (CR 602) are
  prohibited; the static only gates the activation predicates, never trigger firing or continuous
  effects.
- **Ignored cosmetic tags (documented):** `AffectedZone$ Battlefield` is the engine's default
  scope (the predicates only inspect battlefield permanents), `ValidSA$ Activated` is exactly the
  set the `CantBeActivated` predicates already restrict (activated abilities only), and
  `Description$ ...` is reminder text. None changes behavior; they are silently ignored by the
  `S:` parser without warning.

## Tests
- Isolation (test_harness):
  - Suppression of a non-mana activated ability: A controls Knight of the Reliquary + Clarion
    Conqueror — Knight's "Activate (ChangeZone)" ability is absent from the menu; the control run
    (Knight alone, no Clarion) lists it.
  - Suppression of a creature mana ability: A controls Birds of Paradise + Clarion Conqueror + a
    Forest/Plains — Birds offers no mana; only land-based plays remain.
  - Symmetric / opponent suppression: A controls Clarion, B controls Knight of the Reliquary —
    across a full scripted game the opponent is offered the Knight activation 0 times; the control
    game (no Clarion) offers it 26 times.
- Regression (test_harness --scripted, full games):
  - clarion_test (mav with 4× Scythecat Cub → 4× Clarion Conqueror) vs delver, seeds 1-6: all
    decisive (3 A wins, 3 B wins), no draws, no max-decisions caps, no non-fatal errors. The only
    warnings are the pre-existing cosmetic `Unrecognized ability param` lines for other cards;
    none reference Clarion Conqueror.

## Result
implemented
