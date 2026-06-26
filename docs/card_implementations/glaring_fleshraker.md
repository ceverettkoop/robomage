# Glaring Fleshraker  (vocab index 135)

## Oracle text
Whenever you cast a colorless spell, create a 0/1 colorless Eldrazi Spawn creature token
with "Sacrifice this creature: Add {C}."

Whenever another colorless creature you control enters, Glaring Fleshraker deals 1 damage to
each opponent.

(Creature — Eldrazi Drone, 2/2, mana cost {2}{C}.)

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/g/glaring_fleshraker.txt`
- Key tags:
  - `T:Mode$ SpellCast | ValidCard$ Card.Colorless | ValidActivatingPlayer$ You | TriggerZones$ Battlefield | Execute$ TrigToken`
    with `SVar:TrigToken:DB$ Token | TokenScript$ c_0_1_eldrazi_spawn_sac | TokenOwner$ You`
  - `T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Creature.Other+Colorless+YouCtrl | TriggerZones$ Battlefield | Execute$ TrigDamage`
    with `SVar:TrigDamage:DB$ DealDamage | Defined$ Player.Opponent | NumDmg$ 1`
  - `SVar:BuffedBy:Card.Colorless`, `DeckHints:Color$Colorless` — cosmetic (AI/UI hints), ignored.
- Token resource (new, fetched from Forge@master `res/tokenscripts/`):
  `bin/resources/tokenscripts/c_0_1_eldrazi_spawn_sac.txt`
  — `A:AB$ Mana | Cost$ Sac<1/CARDNAME> | Produced$ C | Amount$ 1` ("Sacrifice this creature: Add {C}.").

## Engine work
The two triggers map onto the existing `SpellCast` and `ChangesZone` trigger infrastructure;
the only genuinely new requirement was a **colorless** card/permanent filter, plus making the
Eldrazi Spawn token a first-class permanent (color + activated mana ability). All changes are
general handlers keyed on the tag's intended meaning, not card-specific shortcuts.

- `src/components/ability.h`: added `trigger_valid_card_colorless` — a general `ValidCard$
  ...+Colorless` filter flag for `SpellCast` and `ChangesZone` triggers.
- `src/parse.cpp`:
  - In the trigger ValidCard parser, set `valid_card_colorless` when the filter contains
    `Colorless`, and carry it onto the ability for both `ChangesZone` and a new plain
    `SpellCast` mapping (`mode_is_spell_cast && valid_card_colorless` → `Events::SPELL_CAST`).
    Previously a plain `SpellCast` line with no noncreature/cmc/spell-count qualifier never
    mapped to any event; now a colorless-spell trigger is honored. Also carried the flag
    through the `Execute$` SVar effect copy.
  - Factored the card `Colors:`-line parsing into a reusable `parse_colors_field()` helper.
  - `parse_token_script()` now also parses the token's `Colors:` line (into
    `Token::explicit_colors`) and its `A:` activated/spell abilities (via `parse_abilities`),
    not just `T:` triggered abilities — so a token can carry an intrinsic activated ability.
- `src/components/token.{h,cpp}`:
  - `Token` gained `explicit_colors` (its color indicator; empty = colorless, CR 105.2c).
  - `bootstrap_token_components()` copies the token's activated/spell abilities onto the
    permanent's `Permanent::abilities` (with `source` set), because a permanent's activatable
    abilities are read from `Permanent::abilities`, not the source component. This is how the
    Eldrazi Spawn token's "Sacrifice this creature: Add {C}." reaches the legal-action / mana
    payment systems. (Its triggered abilities stay on `Token::abilities`, where the trigger
    scan already reads them.)
- `src/game_queries.h`: added `is_colorless_card(CardData)` and `is_colorless_entity(Entity)`
  shared predicates (mirroring the colorless test already in `mana_system.cpp`).
  `is_colorless_entity` handles both real cards and tokens (which have no mana cost — their
  color is the color indicator).
- `src/systems/state_manager_triggers.cpp`: at trigger-match time, when
  `trigger_valid_card_colorless` is set, skip the trigger unless the event card
  (`Params::ENTITY`, carried by both `SPELL_CAST` and `CARD_CHANGED_ZONE`) is colorless via
  `is_colorless_entity`.

Mechanics added (general, not card-specific):
- `ValidCard$ ...+Colorless` filtering for `SpellCast` and `ChangesZone` triggers.
- Plain `SpellCast` triggers (no noncreature/cmc/count qualifier) now map to `Events::SPELL_CAST`.
- Tokens may now carry a color indicator and intrinsic **activated** abilities.

## Behavioral decisions (made in lieu of asking a human)
- Colorlessness (CR 105.2 / 105.2c): an object is colored only by colored mana symbols in its
  cost or a color indicator/CDA. Eldrazi Linebreaker has `K:Devoid` (CR 702.114a), which the
  engine records as `explicit_colors = {COLORLESS}`, so it counts as colorless for both
  triggers even though its printed cost includes a red pip — verified in testing. The Eldrazi
  Spawn token has no `Colors:` line and no mana cost, so it is colorless.
- "Another colorless creature you control enters" (CR 603.2/603.3): when Fleshraker's
  SpellCast trigger creates a Spawn token, that token (a colorless creature you control, not
  Fleshraker itself — `.Other`) entering separately satisfies the ChangesZone trigger, as does
  the cast creature itself when it resolves. So casting one colorless creature spell yields a
  token plus two 1-damage triggers (one for the token's entry, one for the creature's entry).
  Confirmed in isolation testing.
- `TokenOwner$ You` emits a cosmetic `WARNING: Unrecognized ability param`. It is genuinely
  cosmetic here: `effect_token` already creates the token under the controller of the source
  (Fleshraker's controller = you), which is the same as its owner. No behavioral effect.

## Tests
- Isolation (test_harness):
  - Cast colorless creature (Eldrazi Linebreaker, colorless via Devoid) with Fleshraker out →
    SpellCast trigger creates a 0/1 Eldrazi Spawn token; the token entering deals 1 to opp
    (20→19) and the Linebreaker entering deals 1 to opp (19→18). PASS.
  - Cast a colored creature (Grizzly Bears, green) with Fleshraker out → no token, no damage,
    opp stays at 20. PASS (negative case).
  - Cast two colorless spells in a turn → two tokens, four 1-damage triggers (20→16). PASS
    (repeatable).
  - Token's "Sacrifice this creature: Add {C}." → the Eldrazi Spawn token is sacrificed and
    adds {C}, helping pay for a second colorless spell ("Player A sacrifices Eldrazi Spawn
    Token" / "activated Eldrazi Spawn Token for 1(C)"). PASS.
- Regression (test_harness --scripted, 6 seeds, mirror colorless-aggro deck containing 4×
  Glaring Fleshraker + 4× Eldrazi Linebreaker + Chalice/Bolt/Eldrazi Temple/Ancient Tomb/
  Lotus Petal): all 6 games (seeds 1–6) finished decisively with a winner, no draws, no
  fatal/non-fatal errors (only the cosmetic `TokenOwner$ You` warning). PASS.

## Result
implemented
