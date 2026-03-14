# boomer_mav.dk Implementation Status

## Card Status

| Card | Status | Notes |
|---|---|---|
| Forest | ✅ complete | vocab idx 1 |
| Wasteland | ✅ complete | vocab idx 10 |
| Wooded Foothills | ✅ complete | vocab idx 8 |
| Plains | ✅ vocab only | idx 44; needs card script review |
| Savannah | ✅ vocab only | idx 45 |
| Windswept Heath | ✅ vocab + engine | idx 52; search lands functional |
| Swords to Plowshares | ✅ vocab + engine | idx 48; TargetedController + CardPower SVar implemented |
| Dryad Arbor | ✅ vocab + engine | idx 32; Colors: field override implemented |
| Birds of Paradise | ✅ vocab + engine | idx 30; Produced$ Any implemented |
| Noble Hierarch | ✅ vocab + engine | idx 42; Produced$ Combo W U G; Exalted implemented |
| Ignoble Hierarch | ✅ vocab + engine | idx 38; Produced$ Combo B R G; Exalted implemented |
| Horizon Canopy | ✅ vocab + engine | idx 36; Combo G W + AB$ Draw functional |
| Thalia, Guardian of Thraben | ⬜  vocab + engine | idx 51; RaiseCost static implemented | FIRST STRIKE IS NOT
| Gaea's Cradle | ✅ vocab + engine | idx 34; dynamic Count$Valid Creature.YouCtrl implemented |
| Karakas | ✅ vocab + engine | idx 39; Creature.Legendary filter implemented |
| Knight of the Reliquary | ✅ vocab + engine | idx 41; Sac<1/Forest;Plains/> conditional sac + graveyard count static |
| Scythecat Cub | ✅ vocab + engine | idx 47; landfall trigger + DB$ Pump/PutCounter/MultiplyCounter + Count$ResolvedThisTurn |
| Icetill Explorer | ✅ vocab + engine | idx 37; AdjustLandPlays$ + MayPlay from graveyard + DB$ Mill |
| Scryb Ranger | ✅ vocab + engine | idx 46; Flash + Return<1/Forest> + ActivationLimit + AB$ Untap implemented |
| Green Sun's Zenith | ⬜ vocab only | idx 35; X spells pending (Phase 5a) |
| Once Upon a Time | ⬜ vocab only | idx 43; SP$ Dig + AlternativeCost condition pending |
| Talon Gates of Madara | ⬜ vocab only | idx 50; DB$ Phases pending |
| Endurance | ⬜ vocab only | idx 33; Flash + Evoke + DB$ ChangeZoneAll + DB$ Phases pending |
| Collector Ouphe | ✅ vocab + engine | idx 31; Mode$ CantBeActivated static suppresses artifact activated abilities |
| Keen-Eyed Curator | ⬜ vocab only | idx 40; exile-with-source count static + AB$ ChangeZone (graveyard→exile) |
| Sylvan Library | ⬜ vocab only | idx 49; complex DB$ chain pending (Phase 5f) |

## Completed Phases

### Phase 1a — Vocab entries
All 26 unique cards added to `src/card_vocab.h` (indices 30–52 + existing 1, 8, 10).
`train/card_costs.py` regenerated.

### Phase 1b — Colors: field
`CardData::explicit_colors` field added.
`parse.cpp` reads `Colors:` line (space-separated lowercase color names).
`orderer.cpp` uses `explicit_colors` when building ColorIdentity.

### Phase 1c — Swords to Plowshares SVar
`Ability::defined_targeted_controller` + `dynamic_amount_expr` fields added.
`parse.cpp`: `Defined$ TargetedController` sets field; `LifeAmount$ X` stores SVar key; SVar content preserved if `Targeted$` pattern.
`ability.cpp`: subabilities inherit `target` + `controller` from parent; `GainLife` branch evaluates both.

### Phase 1e — Produced$ Combo/Any
`Ability::mana_choices` vector added.
`parse.cpp` populates it for `Produced$ Any` and `Produced$ Combo W U G` style values.
`state_manager.cpp` emits one MANA_X action per color in choices.
`action_processor.cpp` uses `ability.color` (already set per-choice at emit time).

### Phase 1f — Flash keyword
`state_manager.cpp`: checks `card_data.keywords` for "Flash" before timing gate.

### Phase 1g — RaiseCost static
`static_ability.h`: `raise_cost` + `raise_cost_filter` fields added.
`parse.cpp`: `Mode$ RaiseCost | ValidCard$ ...nonCreature | Amount$ 1` parsed.
`state_manager.cpp`: sums raise_cost from active statics, adds GENERIC to effective_cost before `can_afford`.

### Phase 2c — Dynamic mana amount
`Ability::dynamic_amount_expr` field added.
`parse.cpp`: preserves `Count$Valid`/`Targeted$` SVar content for runtime.
`action_processor.cpp`: evaluates `Count$Valid Creature.YouCtrl` by counting battlefield creatures.

### Phase 2a — Extended ValidTgts (Legendary)
Both `build_valid_targets` (action_processor.cpp) and `is_target_valid` (ability.cpp) check `vt.find("Legendary")`.

### Phase 3a — Conditional sac cost
`Ability::sac_cost_spec` field added (semicolon-separated subtype list).
`parse.cpp`: `Sac<N/Type;Type/label>` parsed; "CARDNAME" → `sac_self`, otherwise → `sac_cost_spec`.
`state_manager.cpp`: checks controller has matching permanent before emitting ability.
`action_processor.cpp`: prompts player to choose which permanent to sacrifice.

### Phase 3b/3c — Return cost + ActivationLimit
`Ability::return_cost_type`, `return_cost_count`, `activation_limit`, `activations_this_turn` fields added.
`parse.cpp`: `Return<N/Type>` in Cost$ parsed; `ActivationLimit$ N` parsed.
`state_manager.cpp`: skips ability if over limit; checks controller has land of return type.
`action_processor.cpp`: prompts to choose land to return; increments `activations_this_turn`.
`game.cpp`: resets `activations_this_turn` at UNTAP for active player.

### Phase 4a — AB$ Untap
`ability.cpp`: new `category == "Untap"` branch clears `is_tapped` on target Permanent.

### Phase 5b — CantBeActivated (Collector Ouphe)
`static_ability.h`: `cant_activate_card_filter` field added.
`parse.cpp`: `Mode$ CantBeActivated | ValidCard$ Artifact` parsed into `cant_activate_card_filter`.
`state_manager.cpp`: before emitting activated abilities for a permanent, checks all CantBeActivated statics on battlefield; suppresses if permanent matches filter.

### Phase 5g — Landfall triggers + DB$ Pump/PutCounter/MultiplyCounter (Scythecat Cub)
`ability.h`: `trigger_valid_card_is_land`, `condition_check_svar`, `condition_svar_compare` fields added.
`parse.cpp`: `ValidCard$ Land.*` sets `trigger_valid_card_is_land`; `.YouCtrl` sets `trigger_valid_player_is_controller`; `ConditionCheckSVar$`/`ConditionSVarCompare$` resolved via svars map.
`state_manager.cpp`: landfall trigger filter checks card type == Land in CARD_CHANGED_ZONE events.
`ability.cpp`: new categories `Mill`, `Pump`, `MultiplyCounter`; conditional execution (body skipped but subabilities still chain); `resolve_put_counter` uses `target` when set.
`game.h`: `ability_resolution_counts` map for `Count$ResolvedThisTurn` tracking.
`stack_manager.cpp`: increments `ability_resolution_counts[source]` when triggered ability resolves.
`game.cpp`: clears `ability_resolution_counts` at CLEANUP.

### Phase 5h — AdjustLandPlays + MayPlay from graveyard + DB$ Mill (Icetill Explorer)
`static_ability.h`: `adjust_land_plays`, `may_play_from_graveyard` fields added.
`parse.cpp`: `AdjustLandPlays$` and `MayPlay$ True` parsed.
`state_manager.cpp`: PLAY_LAND section sums `adjust_land_plays` statics; scans graveyard for lands if `may_play_from_graveyard` active.
`ability.cpp`: `Mill` category moves top N cards to graveyard.

### Phase 6a — Exalted (Noble/Ignoble Hierarch)
`action_processor.cpp`: after attacker tapping, counts attacking creatures; if exactly 1, gives it +1/+1 per battlefield Exalted permanent (via `prowess_bonus`, cleared at CLEANUP).

## Remaining Work

- Phase 5a: X-cost spells (Green Sun's Zenith)
- Phase 5c: AlternativeCost free condition (Once Upon a Time)
- Phase 5d: DB$ Phases (Talon Gates, Endurance)
- Phase 5e: Evoke (Endurance)
- Phase 5f: Sylvan Library complex chain
- Phase 5i: Keen-Eyed Curator exile counting
- Phase 6b: First Strike (Thalia)
- Phase 6c: Protection from blue (Scryb Ranger)
