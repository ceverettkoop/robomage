# Abrupt Decay (vocab index 164)

## Oracle text
This spell can't be countered.
Destroy target nonland permanent with mana value 3 or less.

## Forge script (Source: pre-existing local; key tags)
`bin/resources/cardsfolder/a/abrupt_decay.txt`
- `Types:Instant`, `ManaCost:B G`
- `R:Event$ Counter | ValidCard$ Card.Self | ValidSA$ Spell | Layer$ CantHappen` — the
  "can't be countered" replacement.
- `A:SP$ Destroy | ValidTgts$ Permanent.nonLand+cmcLE3` — destroy target nonland permanent
  with mana value 3 or less.

## Engine work (general non<CardType> target-filter negation)
The `cmcLE3` mana-value bound and the `CantHappen` can't-be-countered replacement were
**already handled** by the engine (cmcLE parsing in `Ability::is_legal_target`, and the
counter-replacement path that prints "Abrupt Decay can't be countered"). The gap was the
battlefield-permanent target branch ignoring the `nonLand` qualifier: `vt.find("Permanent")`
set `inc_permanents`, and `if (inc_permanents) return true;` matched *any* permanent
(including lands).

Fix (general, not card-specific):
- New shared helper `type_set_passes_nontype(spec, types)` in `src/game_queries.h`, next to
  the color negation helpers. It scans a filter/target spec for `non<CardType>` tokens over
  the card types (Land, Creature, Artifact, Enchantment, Planeswalker, Battle, Instant,
  Sorcery, Tribal) and rejects a candidate whose type line (`kind == TYPE`) carries an
  excluded type.
- `Ability::is_legal_target` (`src/components/ability.cpp`) now calls
  `type_set_passes_nontype(vt, tperm.types)` at the top of the battlefield-permanent branch,
  so `nonLand` / `nonCreature` / `nonArtifact` / … negations are honoured for every targeted
  permanent effect, not just Abrupt Decay. Target enumeration in
  `action_processor.cpp` already routes through `is_legal_target`, so the offered menu and the
  resolution-time re-check share the same predicate.

The script's real tags are honoured (no retag): `SP$ Destroy` with
`ValidTgts$ Permanent.nonLand+cmcLE3`.

## Behavioral decisions (CR cites)
- CR 115.1 — target restrictions (including the `nonLand` type restriction and the
  mana-value bound) are checked when targets are chosen and re-checked at resolution.
- CR 110.4a — permanent card types enumerated for the general negation.
- CR 701.x / "can't be countered" — the replacement makes the counter event not happen;
  the spell still resolves and destroys its target.

## Tests
Built `make HEADLESS=TRUE` clean (only the pre-existing cosmetic `AITgts$` warning on Fatal
Push). Verified with `train/test_harness.py` (inline hands/battlefield, semantic `--play`):
- (a) Opponent has a low-MV creature (Birds of Paradise, MV 1) and a Mountain. Casting
  Abrupt Decay offered **only** the creature as a target (the land was excluded); it resolved
  and destroyed the creature.
- (b) Opponent has Murktide Regent (MV 7) and a Mountain. Abrupt Decay was **not** castable
  (no legal target — the high-MV permanent fails cmcLE3 and the land fails nonLand), confirming
  cmcLE3 is still enforced.
- (c) Opponent cast Force of Negation targeting Abrupt Decay; the log printed "Abrupt Decay
  can't be countered" and Abrupt Decay still resolved and destroyed the target.
- Regression: scripted full games, BG-Abrupt-Decay deck vs a low-MV-permanent opponent, seeds
  1/2/3 — all completed with a decisive winner (Player B each), no draws and no
  non-fatal/fatal errors. Temp decks cleaned up.

## Result
Done. General `non<CardType>` permanent target-filter negation implemented and routed through
the shared target predicate; Abrupt Decay targets nonland permanents with MV ≤ 3 only, can't
be countered, and resolves correctly.
