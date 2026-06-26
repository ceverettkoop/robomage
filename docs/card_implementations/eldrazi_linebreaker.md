# Eldrazi Linebreaker  (vocab index 129)

## Oracle text
Devoid (This card has no color.)

Trample

At the beginning of combat on your turn, target creature you control gains haste and gets
+X/+0 until end of turn, where X is the number of Eldrazi you control.

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/e/eldrazi_linebreaker.txt`
- Type: `Creature Eldrazi`, mana cost `1 C R`, P/T `3/3`.
- Key tags:
  - `K:Devoid` — CR 702.114; sets the card's color to colorless.
  - `K:Trample` — CR 702.19; already supported by the combat-damage assignment code.
  - `T:Mode$ Phase | Phase$ BeginCombat | ValidPlayer$ You | TriggerZones$ Battlefield |
    Execute$ TrigPump` — "At the beginning of combat on your turn, ...". `Phase$ BeginCombat`
    maps to `Events::BEGIN_COMBAT_BEGAN`, already parsed (`parse_one_trigger`).
  - `SVar:TrigPump:DB$ Pump | ValidTgts$ Creature.YouCtrl | NumAtt$ +X | KW$ Haste` — the
    targeted pump body: +X power and a Haste grant until end of turn.
  - `SVar:X:Count$Valid Eldrazi.YouCtrl` — X = number of Eldrazi you control.
  - `DeckHints:Type$Eldrazi` / `TgtPrompt$` — informational/cosmetic; ignored.

## Engine work
Three pieces of the `Pump` body needed new general support; everything was implemented keyed on
the tag's intended meaning (no retagging). Trample, Devoid, and the `Phase$ BeginCombat` trigger
were already supported.

1. **`NumAtt$ +X` (count-SVar pump magnitude).** Previously `parse_pump` did `std::stoi(value)`
   directly, which would fault on `+X` (non-numeric, with exceptions disabled). `PumpParams`
   (`src/components/ability_params.h`) gains `att_expr`/`def_expr` (the runtime `Count$`
   expression) and `att_sign`/`def_sign` (the sign of the original `+X`/`-X` token). A literal
   value (`2`, `-1`) still flows into the static `att`/`def`; a non-numeric body (`X`) is stored
   as an SVar key in `*_expr`. `parse_svar_ability` (`src/parse.cpp`) resolves that SVar key to
   its `Count$` expression after the param loop (mirroring the existing `amount_svar` resolution).
   `effect_pump.cpp` evaluates the expression at resolution via `evaluate_dynamic_amount` against
   the ability's controller and applies `sign * magnitude` into the existing EOT-bonus bucket
   (so cleanup reverts it, CR 514.2 / 611.2b).

2. **`KW$ Haste` (until-end-of-turn keyword grant, CR 702.10b).** `PumpParams.grant_keywords`
   holds the granted keyword(s) (` & `-delimited per Forge). At resolution `effect_pump.cpp` adds
   them to a new `Creature::eot_keywords` bucket (`src/components/creature.h`) and to the live
   `cr.keywords`. Because the static pass rebuilds `cr.keywords` from the printed base each pass
   (CR 611.3a, `src/systems/state_manager_statics.cpp`), the EOT grants are **re-merged** onto
   `cr.keywords` after that rebuild so they survive subsequent state passes, and they are cleared
   in the cleanup step (`src/classes/game.cpp`) alongside the EOT P/T bonuses. The haste grant
   lets a summoning-sick creature attack: `declare_attackers` already reads `cr.keywords` for
   `Haste` when `has_summoning_sickness` is set (CR 302.6 / 702.10b).

3. **`Count$Valid Eldrazi.YouCtrl` (generic type/subtype count).** `evaluate_dynamic_amount`
   (`src/components/ability.cpp`) gains a general `Count$Valid <Filter>.YouCtrl` branch that
   counts battlefield permanents you control matching `<Filter>` (any top-level type, supertype,
   or subtype) via `permanent_has_type`. The pre-existing `Count$Valid Creature.YouCtrl`
   special-case is kept above it; this branch handles arbitrary subtype filters (here Eldrazi).

The triggered ability's target is selected by the `Pump` effect at resolution (it presents the
`Creature.YouCtrl` candidates), matching the engine's existing Pump targeting path.

## Behavioral decisions (made in lieu of asking a human)
- **Devoid (CR 702.114) is cosmetic for this engine.** Color is not a modeled gameplay field
  for these effects (no color-matters interaction in the current vocab references it), so the
  keyword is stored but has no rules effect. Documented; no behavior change.
- **`DeckHints` / `TgtPrompt` ignored (cosmetic).** Deck-building/AI/UI hints only; no rules
  effect. `TgtPrompt$` is in the parser's existing ignored-keys set.
- **X is evaluated at resolution.** Eldrazi Linebreaker itself is an Eldrazi, so X ≥ 1 while it
  is on the battlefield; X counts all Eldrazi you control at the moment the ability resolves
  (CR 608.2). Verified scaling to X=2 with two Eldrazi controlled.
- **Haste persists until end of turn, not until combat.** Stored in `eot_keywords` and cleared
  only at the cleanup step (514.2), matching "until end of turn".

## Tests
Isolation (`train/test_harness.py`, pre-set battlefield, seed 1):
- **Self-pump + haste.** `Eldrazi Linebreaker` preplaced; at begin combat the trigger fires,
  targets itself, "gets +1/+0 (now 4/3)" (X=1) and "gains Haste until end of turn", then attacks
  for **4 trample damage**. Next turn the board shows it back at **3/3** (EOT bonus and keyword
  reverted at cleanup). PASS.
- **Haste lets a summoning-sick creature attack.** With `Eldrazi Linebreaker` + 2 Forest in play
  and `Grizzly Bears` cast this turn (shown `(SICK)`), the trigger targets the Bears: "gets
  +1/+0 (now 3/2)" + "gains Haste until end of turn", and the still-`(SICK)` Bears **attacks**
  for 3 damage. PASS.
- **X scales with Eldrazi count.** Two `Eldrazi Linebreaker` controlled → each trigger grants
  **+2/+0** (3/3 → 5/3, X=2). An opponent's Eldrazi Linebreaker does **not** count (YouCtrl).
  PASS.

Regression (`train/test_harness.py --scripted`, seeds 1–6): deck `temp/elb_a`
(4 Eldrazi Linebreaker / 4 Grizzly Bears / 12 Mountain / 4 Forest, padded) vs a Forest deck.
All 6 games finished **decisively** (Player A wins every game), **no draws**, and no non-fatal
errors / asserts / tracebacks / `Unrecognized ability param` warnings. (`train.py observe` could
not be used — `torch` is not installed — so the scripted regression ran directly through the
harness; build was `make HEADLESS=TRUE`, no raylib.)

## Result
implemented
