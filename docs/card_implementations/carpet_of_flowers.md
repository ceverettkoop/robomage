# Carpet of Flowers  (vocab index 301)

## Oracle text
At the beginning of each of your main phases, if you haven't added mana with this ability this
turn, you may add X mana of any one color, where X is the number of Islands target opponent
controls.

(Enchantment, mana cost {G}.)

## Forge script
- Source: pre-existing local script — `bin/resources/cardsfolder/c/carpet_of_flowers.txt` (not edited).
- Key tags:
  - `T:Mode$ Phase | Phase$ Main1,Main2 | ValidPlayer$ You | CheckSVar$ CarpetX | SVarCompare$ EQ0 |
    OptionalDecider$ You | TriggerZones$ Battlefield | Execute$ TrigMana` — fires at the beginning
    of **each** of the controller's main phases, gated by an intervening-if (`CheckSVar$ CarpetX EQ0`
    = "haven't added mana this turn"), optional (`OptionalDecider$ You`).
  - `SVar:TrigMana:DB$ Pump | ValidTgts$ Opponent | IsCurse$ True | SubAbility$ DBMana` — a
    targeting-only "curse" Pump (no P/T) that establishes the **target opponent** whose Islands are
    counted.
  - `SVar:DBMana:DB$ Mana | Amount$ NumManaMax | Produced$ Any | SubAbility$ CheckPlus` — add X mana
    of one chosen color, X = `SVar:NumManaMax:Count$Valid Island.TargetedPlayerCtrl`.
  - `SVar:CheckPlus:DB$ StoreSVar | SVar$ CarpetX | Type$ Number | Expression$ 1` — latch
    "used this turn".
  - `T:Mode$ Phase | Phase$ Cleanup | Execute$ TrigReset | Static$ True` and
    `T:Mode$ ChangesZone | Origin$ Battlefield | Destination$ Any | ValidCard$ Card.Self |
    Execute$ TrigReset | Static$ True` with `SVar:TrigReset:DB$ StoreSVar | SVar$ CarpetX |
    Type$ Number | Expression$ 0` — re-arm the latch each cleanup / when Carpet leaves play.
- Tags parsed as written; no category was retagged. `TriggerDescription$`/`TgtPrompt$`/`Type$ Number`
  are cosmetic/structural and consumed without behavior change.

## Engine work (all general, keyed on each tag's intended meaning)

### `Phase$ Main2` and the multi-phase trigger (new `SECOND_MAIN_BEGAN` event)
- `src/ecs/events.h`: added `SECOND_MAIN_BEGAN = 20` (mirrors `FIRST_MAIN_BEGAN`) and
  `CLEANUP_BEGAN = 21`. Both carry `PLAYER = active player`.
- `src/classes/game.cpp`: fire `SECOND_MAIN_BEGAN` at the `END_OF_COMBAT → SECOND_MAIN` transition
  and `CLEANUP_BEGAN` at `END_STEP → CLEANUP`, each with the active player (mirrors the existing
  `FIRST_MAIN_BEGAN`/`END_STEP_BEGAN` wiring). CR 505 (main phases), 514 (cleanup).
- `src/parse.cpp`: the `Phase$` value is now comma-split, so a phase trigger may list several
  phases. `Main2` → `SECOND_MAIN_BEGAN`, `Cleanup` → `CLEANUP_BEGAN`. For `Phase$ Main1,Main2`
  the trigger binds `FIRST_MAIN_BEGAN` as `trigger_on` and appends `SECOND_MAIN_BEGAN` to the new
  `Ability::trigger_on_extra` vector.
- `src/systems/state_manager_triggers.cpp`: the general battlefield trigger scan now matches
  `trigger_on` **or** any entry in `trigger_on_extra`, so one trigger fires on every listed phase.

### Per-permanent stored-SVar latch (`DB$ StoreSVar` / `CheckSVar` gate) — new general mechanic
- `src/components/permanent.h`: `std::map<std::string,int> stored_svars` — a per-permanent named
  integer scratch store. Lives on the Permanent so it resets naturally when the source leaves the
  battlefield (a fresh Permanent on re-entry starts empty — this is what the script's
  `ChangesZone Battlefield→Any` reset trigger encodes).
- New effect `EffectKind::StoreSVar` (`effect_kind.{h,cpp}`, `effect_table.cpp`, `effects.h`),
  handler `effects::store_svar` in `src/effects/effect_store_svar.cpp`: writes
  `stored_svar_set_value` into the **source** permanent's `stored_svars[stored_svar_set_name]`
  (no-op if the source is no longer a battlefield permanent — e.g. the leave-battlefield reset).
  Parse hook `parse_store_svar` reads `SVar$`/`Expression$`/`Type$ Number`.
- Trigger gate: `Ability::stored_svar_gate_name`/`stored_svar_gate_compare`. In `src/parse.cpp`,
  `CheckSVar$ <name>` whose SVar resolves to a `Number$...` definition routes to this latch gate
  (instead of the existing CheckSVar→Count$ intervening-if path, e.g. Ocelot Pride); `SVarCompare$`
  feeds whichever gate `CheckSVar` set up. `stored_svar_gate_passes` (`src/svar_eval.{h,cpp}`)
  reads the source's latched int (absent = 0) and tests it (`EQ0`, …). Checked at **both** trigger
  placement (`state_manager_triggers.cpp`) and resolution (`ability.cpp`), like an intervening-if
  (CR 603.4).

### `Static$ True` bookkeeping triggers resolve off-stack
- `Ability::trigger_static_offstack` (set in `src/parse.cpp` for a `Static$ True` phase/ChangesZone
  trigger, excluding the dedicated TapsForMana static path). In `state_manager_triggers.cpp` such a
  trigger resolves its Execute **immediately, off the stack** (a trivial StoreSVar latch write)
  rather than queueing a stack object — both in the battlefield scan (cleanup reset) and the
  self-zone-change scan (leave-battlefield reset). These are Forge bookkeeping triggers, not real
  MTG stack triggers.

### Optional phase trigger + "any one color" opponent-Island mana
- The `OptionalDecider$ You` reuses the existing optional-trigger path (`trigger_optional` →
  `OPTIONAL_YESNO` at resolution, `ability.cpp`).
- `IsCurse$ True` (`Ability::is_curse`, parsed in `effect_pump.cpp`): the Pump is a targeting
  vehicle only — `effects::pump` short-circuits (keeping `ab.target` = the chosen opponent player,
  applying no P/T) instead of re-entering creature target selection.
- `Count$Valid Island.TargetedPlayerCtrl` (`evaluate_dynamic_amount`, `src/components/ability.cpp`):
  a new branch counts battlefield permanents matching the filter controlled by the **targeted
  player** (`ab.target`, inherited by the Mana sub-ability via `bind_sub_target`) — rewrites
  `TargetedPlayerCtrl → YouCtrl` and counts from that player's ownership.
- `Produced$ Any` + dynamic `Amount$` already means "X mana of one chosen color" in
  `effects::add_mana` (`CHOOSE_MANA_COLOR`, like Lion's Eye Diamond). Added a guard: when X
  resolves to 0, skip the color prompt and add nothing (legal no-op).

## Behavioral decisions (made in lieu of asking a human)
- **Two-player auto-target of the lone opponent.** `ValidTgts$ Opponent` enumerates the single
  opponent player as the only legal target; the curse-Pump targets it (forced one-option choice as
  the trigger goes on the stack). The cosmetic Pump applies no P/T (CR scope is two players).
- **"Any one color" = the player picks one color and gets X of it** (not one of each), matching
  the `Produced$ Any` flexible-single-color mana path.
- **This is a triggered mana-producing ability** (CR 603, uses the stack / normal trigger
  resolution), NOT a mana ability — so it is announced, optional at resolution, and re-checks its
  intervening-if/latch on resolution.
- **Once per turn via the latch.** The `CheckSVar CarpetX EQ0` gate blocks the second main-phase
  fire after the first used it; `CheckPlus` sets the latch to 1 only **after** the mana is added
  (so a legitimate fire still reads 0 at the top of resolution); the cleanup reset re-arms it.
  Declining at the first main keeps the latch at 0, so it is offered again at the second main.
- **State-vector unchanged.** `N_CARD_TYPES` (1024) already covers index 301; `STATE_SIZE`/
  `OBS_SIZE` untouched.

## Tests (test_harness, seed 1)
- **Adds opp-Island-count mana, once per turn** (Carpet on A's board, B controls 3 Islands):
  A's first main offers the optional trigger; accept → "Player A adds 3{G}", pool = 3G; the ability
  does **not** fire again at A's second main (gate); it fires again next turn (cleanup re-arm). PASS.
- **Optional decline** (3 Islands): decline at first main → no mana, and it is still offered at the
  second main (latch stayed 0). PASS.
- **X scales / zero:** 0 Islands → accept adds **no mana** (no color prompt, no crash) and still
  latches the gate; 2 Islands → "Player A adds 2{U}". PASS.
- **Reset on leave / cleanup:** the `StoreSVar` reset resolves off-stack exactly once per cleanup
  (verified in the step trace); a re-entered Carpet starts un-latched. PASS.
- **Regression (--scripted full games, seeds 1-3):** green Carpet/Grizzly Bears deck vs a
  blue/Grizzly Bears deck with 18 Islands — all three games decisive (Player A wins; no draws), no
  non-fatal errors and no `TargetedPlayerCtrl` qualifier warnings. Carpet's trigger fires and is
  declined by the scripted agent each main phase, with the opponent target chosen cleanly every time.

## Result
implemented
