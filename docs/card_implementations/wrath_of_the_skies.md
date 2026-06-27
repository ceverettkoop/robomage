# Wrath of the Skies  (vocab index 186)

## Oracle text
You get X {E} (energy counters), then you may pay any amount of {E}. Destroy each artifact,
creature, and enchantment with mana value less than or equal to the amount of {E} paid this way.

(Sorcery, mana cost {X}{W}{W}.)

## Forge script
- Source: pre-existing local script — `bin/resources/cardsfolder/w/wrath_of_the_skies.txt` (not edited).
- Key tags:
  - `A:SP$ PutCounter | Defined$ You | CounterType$ ENERGY | CounterNum$ X | SubAbility$ DBChooseNumber`
    → the caster gets X energy on resolution (`SVar:X:Count$xPaid` — the X paid at cast).
  - `DBChooseNumber:DB$ ChooseNumber | Max$ Max | ListTitle$ amount of energy to pay | SubAbility$ DBDestroyAll`
    (`SVar:Max:Count$YourCountersEnergy` — choose 0..current energy).
  - `DBDestroyAll:DB$ DestroyAll | ValidCards$ Artifact.cmcLEY,Creature.cmcLEY,Enchantment.cmcLEY |
    UnlessCost$ PayEnergy<Y> | UnlessPayer$ You | UnlessSwitched$ True`
    (`SVar:Y:Count$ChosenNumber` — pay the chosen energy, then destroy permanents with MV ≤ Y).
- Tags parsed as written; no category was retagged. `ListTitle$`/`AILogic$`/`SpellDescription$`
  and `UnlessPayer$ You` (the controller is the only payer modeled) are cosmetic/informational.

## Engine work (all general, keyed on each tag's intended meaning)

### X spell cast → energy (reused the existing energy foundation, commit 442f8c9)
- **Dynamic `CounterNum$` (`Defined$ You`)** — `src/components/ability_params.h` (`CounterParams::count_expr`),
  `src/effects/effect_put_counter.cpp`. A non-numeric `CounterNum$ X` is stashed as a raw SVar
  token and resolved through the SVar map to its runtime Count$ expression in `parse_abilities`
  / `parse_svar_ability` (`src/parse.cpp`). At resolution the `Defined$ You` PutCounter evaluates
  the expression (`Count$xPaid` → the X paid at cast, restored per-spell in `stack_manager.cpp`)
  and adds that many ENERGY counters to the controller. The numeric path is unchanged. CR 601.2b
  (announce X), 122.1c (energy as a player counter).

### `DB$ ChooseNumber` — new general effect (CR 601.2 / "choose a number up to N")
- `EffectKind::ChooseNumber` registered in `effect_kind.{h,cpp}`, `effect_table.cpp`, `effects.h`.
- Handler `src/effects/effect_choose_number.cpp`: evaluates `Max$` (parsed into
  `Ability::dynamic_amount_expr` and resolved through SVars at parse time) at resolution, prompts
  the resolving controller (priority handed to that seat, like the unless/optional prompts) to
  pick an integer in `[0, Max]` (ActionCategory `CHOOSE_X`), and stores the pick in
  `cur_game.chosen_number` (new field, `src/classes/game.h`). General over any "choose a number up
  to N" card. No new state-vector field.
- **`Count$ChosenNumber`** in `src/svar_eval.cpp` → `cur_game.chosen_number`, so a chained
  sub-ability can read the chosen value (here the DestroyAll's MV bound Y and its PayEnergy<Y>
  unless-cost).
- **`Count$YourCountersEnergy`** in `src/svar_eval.cpp` → the controller's `"ENERGY"` counter
  total (via the same counter map every {E} producer/consumer uses), the `Max$` for the choice.

### DestroyAll with a dynamic MV bound + PayEnergy unless-cost
- `src/components/ability_params.h` (`DestroyAllParams`): added `cmc_expr`/`cmc_op` (dynamic
  `cmcLE<SVar>` bound that is **not** the cast-time X — the legacy `cmcLEX` keys off
  `cur_game.x_paid`, unchanged) and `energy_unless_expr`/`energy_unless_switched`.
- `src/parse.cpp`: `UnlessCost$ PayEnergy<N>` is routed (when it names `PayEnergy`) to
  `DestroyAllParams::energy_unless_expr` instead of the generic-{N}-mana `unless_generic_cost`
  (a numeric `UnlessCost$ N` is still the legacy mana path); `UnlessSwitched$ True` →
  `energy_unless_switched`. The `cmcLE<SVar>` threshold in `ValidCards$` and the energy SVar are
  resolved to their runtime Count$ expressions in `parse_svar_ability`.
- `src/effects/effect_destroy_all.cpp`: at resolution it (1) pays the energy unless-cost from the
  controller via `pay_energy` (`n<=0` is a no-op true; with `UnlessSwitched` a failure to pay
  means "do nothing" — CR's switched unless), then (2) computes the MV bound from `cmc_expr`
  (Y = `Count$ChosenNumber`) and destroys every battlefield Artifact/Creature/Enchantment (the
  comma-separated multi-type `ValidCards$` already ORs the three types) whose mana value ≤ Y,
  respecting indestructibility (CR 702.12b). Net: pay Y energy, then destroy MV ≤ Y.

## Behavioral decisions (made in lieu of asking a human)
- **X is chosen normally at cast** and the energy granted equals that X (`Count$xPaid`); there is
  no special "X = 0" rule for this card.
- **The energy is actually paid in the DestroyAll's `PayEnergy<Y>` unless-cost**, not at the
  ChooseNumber step — ChooseNumber only records the number; the energy leaves the pool when the
  destroy resolves. `UnlessSwitched$ True` is honored as "destroy only if the energy was paid".
- **The pay/choice cap is the current energy total** (`Count$YourCountersEnergy`), so banked
  energy from a prior source (or a prior cast) adds to the X of this cast.
- **Energy stays internal player state** (per the run constraint): `STATE_SIZE`/`OBS_SIZE`/
  `N_CARD_TYPES` and the obs/state-vector layout are unchanged.

## Tests (test_harness, seed 1; opponent board: Birds of Paradise MV1, Sylvan Library MV2,
Null Rod MV2, Leyline of the Void MV4, Murktide Regent MV5; caster has Plains for {X}{W}{W})
- **(a) Cast X=2, pay 2 energy:** caster got 2 ENERGY; ChooseNumber offered [0..2]; chose 2; paid
  2 energy; destroyed the MV ≤ 2 permanents (Birds, Sylvan Library, Null Rod) and left MV4 Leyline
  and MV5 Murktide alive. PASS.
- **(b) Cast X=2, pay 0 energy:** chose 0; no energy spent (retained 2); nothing destroyed (all 5
  permanents remain). PASS.
- **(c) Banked energy + second cast:** first cast X=2 paying 0 banked 2 energy; second cast X=1 →
  energy = 2 + 1 = 3, ChooseNumber capped at [0..3]; chose 3, paid 3 energy, destroyed MV ≤ 3. This
  confirms energy = X granted, banked energy adds, and the choice is bounded by available energy
  (cannot pay more than you have). PASS.
- **Regression (--scripted full games, seeds 1-3):** mono-white Wrath deck (4× Wrath + Ocelot
  Pride / Containment Priest / Voice of Victory / White Orchid Phantom + Plains) vs a mono-green
  creature deck — all three games decisive (Player A wins; no draws), no non-fatal errors (only
  the pre-existing cosmetic `ChangeTypeDesc$ basic land` warning on White Orchid Phantom). Wrath
  cast and its energy/ChooseNumber/DestroyAll chain exercised in real play.

## Result
implemented
