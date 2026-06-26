# Aether Vial  (vocab index 121)

## Oracle text
At the beginning of your upkeep, you may put a charge counter on Aether Vial.

{T}: You may put a creature card with mana value equal to the number of charge counters on
Aether Vial from your hand onto the battlefield.

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/a/aether_vial.txt`
- Type: `Artifact`, mana cost `1`.
- Key tags:
  - `T:Mode$ Phase | Phase$ Upkeep | ValidPlayer$ You | OptionalDecider$ You | Execute$ TrigPutCounter`
    — "at the beginning of your upkeep, you may …" optional Phase trigger.
  - `SVar:TrigPutCounter:DB$ PutCounter | Defined$ Self | CounterType$ CHARGE | CounterNum$ 1`
    — puts one CHARGE counter on Aether Vial itself.
  - `A:AB$ ChangeZone | Cost$ T | Origin$ Hand | Destination$ Battlefield |
    ChangeType$ Creature.cmcEQX+YouCtrl | Optional$ You` — the tap ability: search your hand
    for a creature whose mana value equals X and put it onto the battlefield.
  - `SVar:X:Count$CardCounters.CHARGE` — X is the number of charge counters on this card.

No tags were retagged or repurposed; every mechanic below is keyed on the tag's intended meaning.

## Engine work
Three pieces, each a general handler keyed on the tag's meaning (CR 122.1 counters,
CR 603.4/OptionalDecider optional triggers, CR 614/zone-change with a dynamic filter):

1. **`Count$CardCounters.<Type>` SVar** (`src/svar_eval.cpp`). A new source-scoped SVar handler
   that returns `get_counters(source, type)` — the number of counters of that kind on the SVar's
   own permanent. `evaluate_sa_svar` already accepted an optional `source` entity for source-scoped
   counts (Keen-Eyed Curator's exiled-with pile), so this slots in alongside.

2. **`PutCounter` on a non-creature permanent** (`src/effects/effect_put_counter.cpp`). The handler
   previously gated on `Creature` and returned early for anything else, so a charge counter on an
   artifact was silently dropped. It now gates on `Permanent` (any permanent can carry counters,
   CR 122.1): a creature still resyncs its +1/+1 / -1/-1 P/T through `add_counters`, while a
   non-creature (Aether Vial) just accrues the typed counter in its counter map. `Defined$ Self`
   already routed the target to the ability's source.

3. **Dynamic mana-value filter on a `ChangeZone` hand search** (`Creature.cmcEQX`, X resolved per
   resolution from the source's charge count):
   - `src/parse.cpp`: after the main A: ability param loop, a `cmcEQ<svar>` / `cmcLE<svar>` (and the
     other comparators) inside `ChangeType$` is detected, its SVar reference resolved to the runtime
     `Count$…` expression, and stored on the ability as `change_type_cmc_expr` + a two-letter
     comparator `change_type_cmc_op` (`Ability`, new fields). This mirrors the existing Fatal-Push
     `cmcLE`-into-`amount_svar` resolution but for a *search filter* rather than an amount.
   - `src/effects/effect_change_zone.cpp`: the search-based `ChangeZone` path evaluates
     `change_type_cmc_expr` against the ability's source via `evaluate_sa_svar(expr, owner, source)`
     and passes the resulting integer bound (and comparator) into `search_zone`.
   - `src/components/ability.cpp`: `search_zone` forwards the bound to `matches_filter_spec`, which
     gained an optional `cmc_bound`/`cmc_op` pair. When `cmc_bound >= 0` it gates each candidate by
     `apply_svar_op(mana_value, op, bound)`. The filter-spec parser was also generalized to (a)
     accept multiple `+`-joined constraints (so `+YouCtrl` is a no-op for an own-zone search) and
     (b) strip a `.cmc…` dot-qualifier from the color slot (it is enforced via `cmc_bound`, not as a
     color). `cmc_bound < 0` preserves the legacy `cmcLEX`-keyed `cur_game.x_paid` path unchanged.

4. **Optional Phase trigger** (`src/parse.cpp`). `trigger_optional` (from `OptionalDecider$ You`) was
   only being assigned inside the zone-change trigger branch; it is now set unconditionally before
   the per-mode branches so a Phase (upkeep) trigger is also optional. At resolution `Ability::resolve`
   already presents the OptionalDecider Accept/Decline (CR 603.4 "you may"), so no further work.

## Behavioral decisions (made in lieu of asking a human)
- **MV equality is exact** (`cmcEQX`): at N charge counters only creatures of mana value exactly N
  are offered; an MV ≠ N creature is never a legal pick. Verified 0→MV0-only, 1→MV1-only (MV0/MV2
  excluded), 2→MV2.
- **At 0 charge counters** Aether Vial puts a mana-value-0 creature (e.g. Dryad Arbor) onto the
  battlefield — the natural reading of "mana value equal to the number of charge counters" when that
  number is 0.
- **Both clauses are optional.** The upkeep trigger is "you may" (Accept/Decline), and the tap
  ability is `Optional$ You` (the search always offers "fail to find"), so the controller can decline
  either. Verified the decline path leaves the counter count unchanged.
- **Counter goes on the artifact, not a creature.** The CHARGE counter is a generic marker on the
  non-creature permanent (CR 122.1); it does not need or use the P/T resync path.

## Tests
Isolation (`train/test_harness.py`, Aether Vial pre-set on A's battlefield):
- **0 counters → MV0 only.** Decline the upkeep trigger (0 counters), activate: the hand search
  offers only `Dryad Arbor` (MV0); `Flying Men` (MV1) is excluded. Picking Dryad Arbor puts it onto
  the battlefield (`Self BF: Dryad Arbor [1/1]`). PASS.
- **Accept adds a charge counter.** Accepting the upkeep trigger logs
  `Put 1 CHARGE counter(s) on Aether Vial (now 1).` PASS.
- **1 counter → MV1 only.** After one accepted upkeep trigger, activate: the search offers only
  `Flying Men` (MV1); `Grizzly Bears` (MV2) and `Dryad Arbor` (MV0) are excluded
  (menu `[0] Fail to find [1] Flying Men`). Flying Men is put onto the battlefield. PASS — proves the
  negative-exclusion case.
- **2 counters → MV2.** Two accepted upkeep triggers (`now 1` then `now 2`), activate: `Grizzly
  Bears` (MV2) is offered and put onto the battlefield (`Self BF: Grizzly Bears [2/2]`). PASS.

Regression (`train/test_harness.py --scripted`, 6 games, seeds 1–6): deck `temp/av_test`
(4 Aether Vial, 4 Flying Men, 4 Grizzly Bears, 4 Birds of Paradise, 4 Dragon's Rage Channeler,
4 Containment Priest, Island/Forest/Mountain) vs `temp/av_opp` (Lightning Bolt / Grizzly Bears /
Flying Men + lands). All 6 games finished decisively (A wins 5, B wins 1), no draws, no
fatal/non-fatal errors, no asserts/tracebacks. Aether Vial was drawn, cast, and its upkeep trigger
and tap ability resolved in real games with the engine stable. (Only the pre-existing cosmetic
`WARNING: Unrecognized ability param` lines from unrelated cards appeared.) Temp decks cleaned up.

## Result
implemented
