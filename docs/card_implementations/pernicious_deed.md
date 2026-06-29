# Pernicious Deed  (vocab index 256)

## Oracle text
{X}, Sacrifice Pernicious Deed: Destroy each artifact, creature, and enchantment with mana value X
or less.

(Enchantment, mana cost {1}{B}{G}.)

## Forge script
- Source: pre-existing local — `bin/resources/cardsfolder/p/pernicious_deed.txt`
- Key tags:
  - `A:AB$ DestroyAll | Cost$ X Sac<1/CARDNAME> |
    ValidCards$ Artifact.cmcLEX,Creature.cmcLEX,Enchantment.cmcLEX` with `SVar:X:Count$xPaid`.

## Engine work
- none — fully covered by existing handlers:
  - `AB$ DestroyAll`: `src/effects/effect_destroy_all.cpp`, with dynamic `cmcLE<X>` evaluated against
    the X paid at activation.
  - `X` activation cost + `Sac<1/CARDNAME>` cost: `src/parse.cpp` / `src/action_processor.cpp`
    (X chosen at activation into `cur_game.x_paid`; self-sacrifice paid as a cost).

## Behavioral decisions
- The mana-value filter is `<= X` for all three of artifact/creature/enchantment; lands and other
  permanent types are unaffected. Unambiguous from the script.

## Tests (test_harness)
- A: Pernicious Deed + Mountains + Grizzly Bears (mv2); B: Grizzly Bears (mv2) + Knight of Autumn
  (mv3). Activated with **X=2** → both Grizzly Bears destroyed, Knight of Autumn (mv3) survived,
  Pernicious Deed sacrificed, lands (mv0) untouched. PASS.
- Regression (`--scripted`, seeds 1-3): all decisive, no draws, no non-fatal errors.

## Result
implemented
