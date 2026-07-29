# Malevolent Rumble

## Oracle text
Reveal the top four cards of your library. You may put a permanent card from among them into your
hand. Put the rest into your graveyard. Create a 0/1 colorless Eldrazi Spawn creature token with
"Sacrifice this creature: Add {C}."

{1}{G} Sorcery

## Forge script
- **Source:** pre-existing local (`bin/resources/cardsfolder/m/malevolent_rumble.txt`).
- **Key tags:**
  - `A:SP$ Dig | DigNum$ 4 | Reveal$ True | ChangeValid$ Permanent | DestinationZone2$ Graveyard |
    Optional$ True | SubAbility$ DBToken`.
  - `SVar:DBToken:DB$ Token | TokenScript$ c_0_1_eldrazi_spawn_sac | TokenOwner$ You`.
- **Note:** this pinned Forge script creates the Eldrazi Spawn token *unconditionally* (no "then
  you may sacrifice a permanent" clause — that appears in the printed oracle but not in this
  script). Implemented as scripted (do not modify card scripts).

## Engine work
Two general gaps in the Dig handler (`src/effects/effect_dig.cpp`):

1. **`DestinationZone2$` — route the unchosen remainder** (CR 701.20). The Dig handler always
   returned the looked-at-but-unchosen cards to the library (top/bottom via `LibraryPosition2$`).
   Added a `dig_rest_destination` field (`src/components/ability.h`), parsed from
   `DestinationZone2$` (Library/Hand/Graveyard/Battlefield/Exile), and routed the remainder there
   when set — Malevolent Rumble puts the rest into the graveyard. Absent tag keeps the library
   default, so every existing Dig card is unchanged.
   *Mechanics added (general): DestinationZone2$ for the Dig remainder.*

2. **`ChangeValid$ Permanent`** (CR 110.4a). The Dig filter matched printed *type names*, and no
   type is literally named "Permanent", so the filter would have offered nothing. Now the head
   "Permanent" matches any permanent card via the shared `is_permanent_card()` predicate
   (artifact/creature/enchantment/land/planeswalker/battle) — instants/sorceries are excluded.
   *Mechanics added (general): "Permanent" head in the Dig ChangeValid$ filter.*

The `SubAbility$ DBToken` (create Eldrazi Spawn) and its token script
`c_0_1_eldrazi_spawn_sac` were already supported.

## Behavioral decisions
- The put-to-hand is optional (`Optional$ True`): the menu offers "Take nothing".
- Non-permanent cards among the four (Lightning Bolt) are not offered for hand and go to the
  graveyard with the rest.

## Tests
- Isolation (test_harness): stacked A's library top-4 = Grizzly Bears (creature), Lightning Bolt
  (instant), Island, Swamp; cast Malevolent Rumble and took Grizzly Bears.
  - Result: menu offered only the permanent cards (Grizzly Bears / Island / Swamp — Lightning Bolt
    excluded); Grizzly Bears went to hand; Lightning Bolt, Island, Swamp went to the **graveyard**
    (graveyard +3, not the library); a 0/1 Eldrazi Spawn token was created on the battlefield;
    library dropped by 4.
- CI gate: `ci_check.py --tier pygen,vocab,smoke` (run once after all three cards).

## Result: implemented.
