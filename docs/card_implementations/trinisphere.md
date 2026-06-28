# Trinisphere  (vocab index 270)

## Oracle text
As long as Trinisphere is untapped, each spell that would cost less than three mana to cast
costs three mana to cast. (Additional mana in the cost may be paid with any color of mana or
colorless mana. For example, a spell that would cost {1}{B} to cast costs {2}{B} to cast
instead.)

## Forge script
- Source: pre-existing — `bin/resources/cardsfolder/t/trinisphere.txt`
- Type: `Artifact`, mana cost `3`.
- Key tag:
  - `S:Mode$ SetCost | ValidCard$ Card | Type$ Spell | Amount$ 3 | RaiseTo$ True |
    IsPresent$ Card.Self+untapped` — a continuous static that **sets a minimum total mana
    value of 3** (`Amount$ 3` + `RaiseTo$ True`) on every spell (`ValidCard$ Card`), active
    only while the source itself is on the battlefield untapped (`IsPresent$ Card.Self+untapped`).

No tags were retagged or repurposed; the cost-floor mechanic is keyed on the `SetCost`/`RaiseTo`
tags' intended meaning, reusable by any future cost-floor card.

## Engine work
A general **SetCost cost-floor** static was added to the existing cost-modification path
(`effective_base_cost`, the single shared cast-cost builder used by both affordability in
`determine_legal_actions` and payment in `action_processor`), alongside the RaiseCost surcharge
and ReduceCost reduction.

1. **New `StaticAbility` fields** (`src/components/static_ability.h`): `int set_cost_min`,
   `bool set_cost_raise_to`, `std::string set_cost_filter` — the floor amount, the RaiseTo
   (floor vs. set-exact) flag, and the `ValidCard$` spell filter ("Card"/empty = every spell).

2. **Parser** (`src/parse.cpp`, `parse_static_abilities`): `Mode$ SetCost` maps to
   `category = "SetCost"`; `Amount$` for a `SetCost` static fills `set_cost_min`;
   `RaiseTo$ True` sets `set_cost_raise_to`; `ValidCard$` (other than bare `Card`) fills
   `set_cost_filter`.

3. **Cost-floor query** (`src/systems/state_manager_statics.cpp`): new
   `int active_cost_floor_for(const CardData&)` (declared in `src/systems/state_manager.h`)
   returns the largest `set_cost_min` among active (`condition_met`, non-suppressed) RaiseTo
   floors whose `set_cost_filter` matches the spell, or 0. It checks `ActiveStatic::condition_met`
   so the `IsPresent$` untapped gate is honoured (a RaiseCost/ReduceCost static has no gate, so
   those queries don't check it; a cost-floor with a present condition must).

4. **Applied last in `effective_base_cost`** (CR 601.2f): after every other increase/reduction
   (RaiseCost / ReduceCost / Affinity), if the cost's total mana value (`ManaValue::size()`) is
   below the floor, generic pips are inserted up to the floor. Colored pips are never removed, so
   `{1}{B}` → `{2}{B}` (the oracle example). A cost already at or above the floor, or a spell no
   floor matches, is untouched. The floor is caster-independent (Trinisphere affects *each*
   spell, both players').

5. **`.Self` present-gate support** — `IsPresent$ Card.Self+untapped` requires the present-count
   gate to recognise `Self`:
   - `src/game_queries.cpp`: added the `Self` qualifier (`q == "Self"` → `v.entity == ctx.source`),
     the mirror of the existing `Other`.
   - `src/systems/state_manager_statics.cpp`: `static_present_condition_met` now takes the source
     entity and seeds `MatchCtx::source`, so `.Self`/`.Other` resolve in any present gate (general,
     reused by every future self-referential `IsPresent$`).

## Behavioral decisions (made in lieu of asking a human)
- **Floor applied after all other cost modifications** (CR 601.2f). Trinisphere is worded as a
  cost increase but checks the spell's already-adjusted cost — so a cost reducer that drops a
  spell below {3} still ends at {3}. Applied as the last step of `effective_base_cost`.
- **`Card.Self+untapped` gate ⇒ a tapped Trinisphere imposes no floor.** Modeled through the
  existing present-count condition (now `Self`-aware) rather than a card-specific tap check, so it
  composes generally.
- **Affects both players' spells** (`ValidCard$ Card`, no controller scope) — the floor query is
  caster-independent.
- **Mana value used is `ManaValue::size()`**, consistent with the rest of the engine; the floor
  is computed on the base (pre-X) cost. *Caveat:* a spell with `{X}` in its cost is floored on its
  printed cost with X treated as 0 (`effective_base_cost` excludes the interactive X choice), so an
  `{X}` spell cast for large X is not re-evaluated against the floor. No current vocab interaction
  exercises this; noted for follow-up if an X-spell + Trinisphere case matters.

## Tests
Isolation (`train/test_harness.py`):
- **MV1 spell floored to {3}.** Trinisphere + 3 Mountain on A's battlefield; A casts Lightning
  Bolt (MV1) at a Grizzly Bears → all 3 Mountains tap, bolt resolves for 3, bear destroyed. PASS.
- **Baseline (no Trinisphere).** Same line without Trinisphere → only 1 Mountain taps for
  Lightning Bolt. PASS — proves the floor, not a constant.
- **Colored pip preserved, MV2 → {3}.** Trinisphere + 3 Forest; A casts Grizzly Bears
  (`{1}{G}`, MV2) → all 3 Forests tap (`{2}{G}`), bear enters. PASS (oracle `{1}{B}`→`{2}{B}`
  example).
- **Opponent's spell floored too.** A controls Trinisphere; B casts Lightning Bolt (MV1) from a
  3-Mountain board → all 3 of B's Mountains tap. PASS.

No draws, no non-fatal errors / asserts in any run.

## Result
implemented
