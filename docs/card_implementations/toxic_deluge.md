# Toxic Deluge  (vocab index 241)

## Oracle text
As an additional cost to cast this spell, pay X life.
All creatures get -X/-X until end of turn.

(Sorcery, mana cost {2}{B}.)

## Forge script
- Source: pre-existing local script — `bin/resources/cardsfolder/t/toxic_deluge.txt` (not edited).
- Key tags:
  - `A:SP$ PumpAll | Cost$ 2 B PayLife<X> | ValidCards$ Creature | NumAtt$ -X | NumDef$ -X`
  - `SVar:X:Count$xPaid` — X is the value paid for the spell's X, here paid as **life** (`PayLife<X>`).
- Tags parsed as written; no category was retagged. `SpellDescription$`/`AI:` are cosmetic.

## Engine work (this is the leading commit of the "variable-X parsing/targeting" unit)

Before this work the parser called `std::stoi` unconditionally on numeric fields and crashed
(`std::invalid_argument`) whenever the value was a literal `X`/`Y` resolved via `Count$xPaid`.
This commit makes those spots X-aware and adds the runtime support the three cards in this unit
share. CR 601.2b: X is chosen during announcement, before targets/effects, so `x_paid` is known
when the spell's magnitude (or target count, for the sibling cards) is read.

### Variable life cost `PayLife<X>` (this card)
- `src/parse.cpp` `parse_activation_cost`: `PayLife<N>` with a numeric arg keeps the flat
  `Ability::life_cost`; a non-numeric arg (`PayLife<X>`) now sets the new flag
  `Ability::life_cost_is_x` (`src/components/ability.h`) instead of `stoi("X")`-ing (the old crash).
- `src/game_queries.h` `spell_has_variable_life_cost(cd)`: reads that flag off the SPELL ability
  (mirrors `spell_additional_sac_spec`) so the cast path can find it without retagging.
- `src/action_processor.cpp` (regular-cost cast branch): after the mana payment commits (so a
  cancelled mana payment never loses life), a variable-life spell prompts the caster to choose X
  in `[0, life]` (CR 119.4 — a player may pay up to their whole life total), records
  `cur_game.x_paid = X`, and pays X life. X is still chosen before targets are selected.
- `spell.x_paid` is now also recorded for a variable-life spell (not just `has_x_cost` mana-X
  spells), so `StackManager` restores the right `x_paid` when the spell resolves and the
  `Count$xPaid` magnitude reads the value this cast chose.

### Negative count-SVar pump magnitude `-X` (this card)
- `NumAtt$ -X` / `NumDef$ -X` were already split by `parse_pump_amount` into
  `att_expr/def_expr = "X"` with `att_sign/def_sign = -1`. The resolution of the `"X"` SVar key to
  its runtime `Count$xPaid` expression existed only in the sub-ability parse path
  (`parse_svar_ability`); a **top-level** `SP$ PumpAll` (this card) never had it resolved. Added the
  shared helper `resolve_pump_exprs()` (`src/parse.cpp`) and call it from BOTH `parse_abilities`
  and `parse_svar_ability`, so a primary `PumpAll`/`Pump` whose magnitude scales by X now resolves
  too. At resolution `resolve_pump_amounts` computes `sign * Count$xPaid` = `-X`, applied to every
  creature by `effect_pump_all.cpp` via the existing per-creature `apply_pump_to_creature`. Lethal
  results (toughness ≤ 0) die to the existing state-based check (CR 704.5f). CR 611.3 / 514.2: the
  -X/-X bonus reverts at cleanup.

## Behavioral decisions (made in lieu of asking a human)
- **The life paid IS X** (`Count$xPaid`), chosen as an additional cost while casting; there is no
  mana-X on this card (`ManaCost:2 B`), so the X prompt is driven by the `PayLife<X>` cost.
- **X may be 0** (pay 0 life, all creatures get -0/-0, nothing dies) and up to the caster's whole
  life total (CR 119.4); paying all your life is legal and leaves you at 0 (a subsequent SBA loss).
- The -X/-X applies to **every** creature (both controllers), matching `ValidCards$ Creature` with
  no controller qualifier.

## Tests (test_harness, seed 1)
- **X=3, six creatures (A: 1 Grizzly Bears; B: 2 Grizzly Bears):** caster paid 3 life (20→17), all
  three 2/2s got -3/-3 (→0/0) and died to the SBA; caster ends at 17. PASS.
- **X=0 edge:** caster paid 0 life (stays 20), no creature got a printed pump, nothing died, no
  crash. PASS.
- **Regression (`--scripted` full games, seeds 1-6):** a B/U deck containing 2× Toxic Deluge / 2×
  Candelabra / 2× Hide on the Ceiling vs a R/G creature deck — all six games decisive (no draws),
  no non-fatal errors; the scripted agent actually cast Toxic Deluge in 5 of 6 games.

## Result
implemented
