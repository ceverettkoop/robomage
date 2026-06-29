# Static Prison  (vocab index 282)

## Oracle text
When Static Prison enters, exile target nonland permanent an opponent controls until Static
Prison leaves the battlefield. You get {E}{E} (two energy counters).

At the beginning of your first main phase, sacrifice Static Prison unless you pay {E}.

(Enchantment, mana cost {W}.)

## Forge script
- Source: fetched script — `bin/resources/cardsfolder/s/static_prison.txt` (not edited).
- Key tags:
  - `T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self |
    Execute$ TrigExile` — the ETB trigger.
  - `TrigExile:DB$ ChangeZone | Origin$ Battlefield | Destination$ Exile |
    ValidTgts$ Permanent.nonLand+OppCtrl | SubAbility$ DBEnergy | Duration$ UntilHostLeavesPlay`
    — targeted exile that returns when the host leaves play.
  - `DBEnergy:DB$ PutCounter | Defined$ You | CounterType$ ENERGY | CounterNum$ 2` — "you get
    {E}{E}".
  - `T:Mode$ Phase | Phase$ Main1 | ValidPlayer$ You | TriggerZones$ Battlefield |
    Execute$ TrigSac` — "at the beginning of your first main phase".
  - `TrigSac:DB$ Sacrifice | UnlessCost$ PayEnergy<1> | UnlessPayer$ You` — "sacrifice CARDNAME
    unless you pay {E}".
- Tags parsed as written; no category was retagged.

## Engine work (all general, keyed on each tag's intended meaning)

Energy ({E}, CR 122.1c) and the `Duration$ UntilHostLeavesPlay` targeted-exile-with-return were
**already** general in the engine (Guide of Souls / Wrath of the Skies / Amped Raptor for energy;
Cloak and Dagger for the exile). This card reuses both. Three small, general gaps were filled:

### `Phase$ Main1` triggered ability ("at the beginning of your first main phase")
- New `Events::FIRST_MAIN_BEGAN` (id 18, PLAYER = active player) — `src/ecs/events.h`.
- Fired on the DRAW → FIRST_MAIN step transition — `src/classes/game.cpp` (mirrors the existing
  upkeep/draw/begin-combat/end-step phase events).
- Parsed `Phase$ Main1` → `phase_is_first_main` → `trigger_on = FIRST_MAIN_BEGAN`,
  `trigger_valid_player_is_controller = (ValidPlayer$ You)` — `src/parse.cpp`. The generic trigger
  scan already matches any `trigger_on` event and filters on the PLAYER param. Reusable by any
  "at the beginning of your/each first main phase" card.

### Bare `DB$ Sacrifice` ⇒ self-sacrifice (CR 701.16)
- `src/effects/effect_sacrifice.cpp`: a `DB$ Sacrifice` with no `SacValid$` filter and no
  `Defined$ Opponent` now sacrifices its own source ("sacrifice CARDNAME"), via the new
  `sacrifice_self(...)` helper, with no choice presented. (Previously an empty `SacValid$` matched
  *all* the controller's permanents and prompted a choice — wrong for the common "sacrifice this"
  pattern.) Edicts (`Defined$ Opponent`) and filtered sacrifices (`SacValid$ Creature`, etc.) are
  unaffected.

### `UnlessCost$ PayEnergy<N>` on a non-DestroyAll effect ("… unless you pay {E}")
- `UnlessPayKind::ENERGY` added to the shared `run_unless_loop` machinery — `src/effects/effects.h`
  + `src/components/ability.cpp`. The ENERGY branch mirrors the LIFE branch: offer "Pay {E} ×N"
  only when the payer has ≥ N energy (`player_energy`), pay via `pay_energy` on accept (returns
  false ⇒ prevented effect does not happen), else return true (effect happens). Reuses the existing
  `PAY_UNLESS` action category — no observation-layout change.
- `Ability::unless_cost_is_energy` flag — `src/components/ability.h`; the energy count rides on the
  existing `unless_generic_cost`.
- `src/parse.cpp` `UnlessCost$` handler: `PayEnergy<N>` now routes to the new flag for any effect
  *except* `DestroyAll` (which keeps its existing variable-SVar `energy_unless_expr` path for Wrath
  of the Skies). The `<N>` count is parsed as a literal.
- `src/effects/effect_sacrifice.cpp` honors `unless_cost_is_energy`: the payer
  (`UnlessPayer$ You` ⇒ the controller) may pay to prevent the sacrifice before the self-sacrifice
  fires.

## Behavioral decisions (made in lieu of asking a human)
- **Energy stays internal player state, NOT added to the ML observation/state vector** (per the run
  constraint): `STATE_SIZE` / `OBS_SIZE` / `N_CARD_TYPES` and the observation layout are unchanged.
  Energy is read/spent only through `player_energy` / `pay_energy` (CR 122.1c). (`svar_eval`'s
  pre-existing `Count$YourCountersEnergy` reads the same counter, also engine-internal.)
- **"unless you pay {E}" is offered as a pay/decline only when payable** (CR 118/119): with 0 energy
  the pay option is absent and the only choice declines → the sacrifice happens.
- **First-main trigger does not fire the turn Static Prison enters** — it enters during (after the
  start of) that first main phase, so the first pay-or-sacrifice check is the controller's *next*
  first main, matching "at the beginning of your first main phase".

## Tests (test_harness, seed 1)
Setup: Player A casts Static Prison ({W}) with two Plains in play; Player B has a Grizzly Bears.
`--play "A:keep,B:keep,A:cast:Static Prison,A:target:Grizzly Bears@opp"`, auto-advancing.
- **ETB:** "Grizzly Bears is moved to exile" and "Player A gets 2 ENERGY counter(s) (now 2)."
- **Pay to keep:** at each of A's next two first main phases the Sacrifice ability resolves,
  "Player A may pay 1 energy to avoid sacrificing:" → "Player A pays 1 energy." Static Prison
  stays; Grizzly Bears stays exiled (energy 2 → 1 → 0).
- **Fail to pay → sacrifice → return:** the third first main, with 0 energy, "Player A sacrifices
  Static Prison." → "Grizzly Bears is moved to the battlefield" under its owner (Player B), which
  then attacks. The `UntilHostLeavesPlay` return fired on the sacrifice.
- No draws, no non-fatal errors (the unrelated pre-existing parse WARNINGs for Delver/Brainstorm
  are from vocab card loading, not this card).

## Result
implemented
