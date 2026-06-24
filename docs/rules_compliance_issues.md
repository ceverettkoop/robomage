# Rules Compliance Issue Tracker

Audit of the Robomage engine against the **MTG Comprehensive Rules**
(`docs/mtg_comprehensive_rules.txt`). Each issue cites the engine behavior, the
governing rule, a fix sketch, and a complexity/risk estimate.

**Status legend:** `OPEN` (not started) · `IN PROGRESS` · `DONE`

**Priority tiers:**
- **Tier 1** — active bugs affecting cards already in `src/card_vocab.h`; wrong results in playable games today.
- **Tier 2** — foundational/architectural gaps that many cards depend on.
- **Tier 3** — correct-but-incomplete subsystems; real deviations but mostly latent (no current vocab card exercises them).

> "Latent" = the deviation is real but no card currently in the 32-card vocab triggers it. Fix before/with the first card that needs it.

---

## Tier 1 — Active bugs on implemented cards

### T1.1 — "Until end of turn" pumps never expire · `DONE`
- **Rule:** 514.2 / 611.2b
- **Engine:** `effects::pump` writes the bonus into `cr.base_power`/`base_toughness` permanently (`src/effects/effect_pump.cpp:64-67`, comment admits it). Cleanup (`src/classes/game.cpp:227-242`) only zeroes `Damage` and `prowess_bonus`.
- **Rules require:** "until end of turn" continuous effects end during the cleanup step.
- **Live impact:** Giant Growth (in vocab) permanently buffs a creature.
- **Fix:** Store EOT pumps in a dedicated duration-tagged bucket; clear it in the CLEANUP branch beside `prowess_bonus`. Prerequisite scaffolding the layer system (T2.1) will reuse.
- **Complexity/risk:** **Medium** — needs a real temporary-effect store, but the cleanup hook already exists.

### T1.2 — Blocked creature whose blockers all die hits the player · `DONE`
- **Rule:** 509.1h / 510.1c
- **Engine:** `deal_combat_damage` recomputes live blockers each step (`src/systems/state_manager_combat.cpp:82-89`); if a blocked attacker's blockers all die, `blockers.empty()` routes it down the *unblocked* path and it hits the defending player (`:91-115`).
- **Rules require:** a creature stays blocked even if its blockers leave; it then assigns no combat damage (unless it has trample).
- **Fix:** Persist an `is_blocked` flag set at declare-blockers; a once-blocked creature with zero live blockers assigns no damage.
- **Complexity/risk:** **Medium** — new persisted flag distinct from the live-blocker scan, plus trample interaction.

### T1.3 — Equipment keeps a dangling `equipped_to` when its host dies · `DONE`
- **Rule:** 704.5n
- **Engine:** when the equipped creature leaves, lifecycle code removes its `Permanent` but never clears the equipment's `equipped_to` or the creature's `equipped_by` (`src/systems/state_manager_statics.cpp:353-363`). No SBA re-checks attachment legality.
- **Rules require:** Equipment attached to an illegal/absent permanent becomes unattached but stays on the battlefield.
- **Fix:** SBA pass clearing `equipped_to` when the host is no longer a battlefield creature; defensively clear links at the leave site. Link fields already exist (`src/components/permanent.h:24-25`).
- **Complexity/risk:** **Low–Medium** — real dangling-reference bug, small blast radius.

### T1.4 — Lion's Eye Diamond goes on the stack instead of resolving as a mana ability · `DONE`
- **Rule:** 605.1a / 605.3
- **Interpretation (project decision, 2026-06-24):** `InstantSpeed$` is a *timing restriction*, not a
  loss of mana-ability status. It means the ability may be activated only when it would be legal to
  cast an instant — i.e. when its controller has priority. It does **not** make the ability use the
  stack. Concretely:
  - LED's AddMana ability **is a mana ability**: it resolves immediately, off-stack, adding mana
    directly to the pool. It must **not** be pushed onto the stack.
  - Because of the instant-speed timing restriction, it is **not** legal to activate it in the
    middle of paying a cost (inside the mana-payment loop). An ordinary tap-for-mana source can be
    activated mid-payment; an instant-speed mana source can only be activated at priority, like
    casting an instant, to float mana ahead of time (which then pays for a spell cast in that same
    priority window before the pool empties).
- **Engine:** `is_mana_ability = (category == "AddMana" && !instant_speed)` (`src/action_processor.cpp:188`)
  treats LED as a non-mana ability, so it is pushed onto the stack (`:297`) and listed as a stack
  ability (`src/systems/state_manager_actions.cpp:502`). It is also blanket-excluded from payable
  sources (`src/mana_system.cpp:165`).
- **Fix:** Classify `InstantSpeed$` AddMana as a mana ability so it resolves off-stack — drop the
  `!instant_speed` guard at `action_processor.cpp:188` and the stack-listing branch at
  `state_manager_actions.cpp:502`. Make the `instant_speed` exclusion **context-dependent** instead
  of a blanket skip: `collect_available_mana_sources` should still skip instant-speed sources when
  called from inside `prompt_mana_payment` (mid-cost-payment is not instant timing), but **include**
  them when listing actions for a player who currently holds priority (the priority-time
  `collect_mana_legal_actions` path), so LED can be activated to float mana. LED's `Sac<self>` +
  `Discard<0/Hand>` costs are then paid via `pay_secondary_activation_costs` (which already handles
  both) on the off-stack mana-ability activation path.
- **Complexity/risk:** **Medium** — the `instant_speed` skip must become caller-aware (priority vs.
  mid-payment) rather than unconditional, and the off-stack mana-ability path must offer LED's color
  choice (`mana_choices`) the way other multi-color mana sources are presented.
- **Implemented (2026-06-24):** `collect_available_mana_sources`/`collect_mana_legal_actions` gained a
  caller-aware flag (instant-speed sources included only at priority, excluded mid-payment); the
  priority listing at `state_manager_actions.cpp:407` passes it; the stack-listing branch (`:502`) and
  the `!instant_speed` guard at `action_processor.cpp:188` were dropped so LED resolves off-stack.
  Color choice was already handled by the existing per-color `mana_choices` expansion. **Additional
  change beyond the sketch:** the machine-mode gate at `state_manager_actions.cpp:549` previously hid
  *all* priority-time mana abilities from the ML agent (normal mana is auto-paid), which would have
  left LED unusable by the model — it now exposes instant-speed mana abilities to machine mode while
  keeping normal lands hidden, so the agent can float LED mana. Verified via the LED+Street
  Wraith+Doomsday combo (off-stack float persists across the cycling draw) and a 5-game scripted
  doomsday-vs-mav regression (no draws/errors).

### T1.5 — Vigilance ignored: attackers always tap · `DONE`
- **Rule:** 702.21
- **Engine:** every declared attacker is tapped unconditionally (`src/action_processor.cpp:611-614`); `Vigilance` is parsed into `creature.keywords` but never read.
- **Fix:** Skip the tap when the attacker has Vigilance (`creature_has_keyword` helper exists, `src/game_queries.h:69`).
- **Complexity/risk:** **Low** — one guarded statement.

---

## Tier 2 — Architectural / foundational gaps

### T2.1 — No layer system; P/T is a flat additive sum · `OPEN`
- **Rule:** 613 (esp. 613.4 sublayers 7a–7d; 613.7 timestamp; 613.8 dependency)
- **Engine:** `recompute_pt` sums `base_power + plus_one_counters + prowess_bonus + static_power_bonus` (`src/components/creature.cpp:7-10`). CDA "set," "set to N," and +N/+N pumps all collide in `base_*` (`src/systems/state_manager_statics.cpp:629-632`, `src/effects/effect_pump.cpp:66-67`). Only layer-4 type changes are timestamp-sorted (`state_manager_statics.cpp:482-484`); P/T statics are summed in arbitrary `g_active_statics` order (`:671-680`).
- **Rules require:** P/T modifications apply in sublayers 7a CDA → 7b set → 7c +N/+N & counters → 7d switch, each in timestamp order, with dependency (613.8).
- **Fix:** Replace the fixed buckets with an ordered, layer/sublayer-tagged effect list applied in sequence. At minimum split 7b "set" from 7c "modify." Sub-findings that fold into this work:
  - 7b set vs 7a CDA vs 7c pump conflation (613.4a–c) — share `base_*` storage today.
  - Timestamp/dependency ordering across distinct effect sources (613.7/613.8) — absent for P/T statics.
  - P/T switch, sublayer 7d (613.4d) — not represented at all (latent; no vocab card needs it).
- **Complexity/risk:** **High** — touches every reader of `Creature.power/toughness` (~75 sites) and all static/pump/counter writes. Largest single gap; the keystone refactor.

### T2.2 — Replacement effects narrow & mostly dormant; no ordering · `OPEN`
- **Rule:** 614 (replacement effects), 616 (ordering multiple applicable replacements)
- **Engine:** `Effect::Replacement` supports only `ENTERS_TAPPED`, `CANT_BE_COUNTERED`, `EXILE_INSTEAD_OF_GRAVEYARD` (`src/components/effect.h:12-31`); `affected_zones`/`affected_types`/`category`/`amount` are reserved/unused. Enters-tapped/with-counters are hard-coded in `apply_permanent_components` (`src/systems/state_manager_statics.cpp:153-164`, `:219-234`), applied in fixed order with no player choice.
- **Rules require:** 614 — general event-interception class (enters-with-counters, "if it would die, exile instead," damage replacement, "enters as a copy," etc.); 616.1 — affected player/controller orders multiple applicable replacements.
- **Fix:** Generalize into an event-interception layer applied at the relevant events; route enters-tapped/with-counters/finality through it; prompt for order when >1 applies.
- **Complexity/risk:** **High** — architectural; only three hard-coded cases exist today.

### T2.3 — Prevention effects entirely absent · `OPEN`
- **Rule:** 615 (prevention), 122.1c (shield counters)
- **Engine:** `deal_damage`/`deal_damage_to_player` apply damage directly with no prevention path (`src/components/damage.cpp:20-60`); only protection short-circuits. No "prevent" logic anywhere in `src/`.
- **Rules require:** prevention shields ("prevent the next N damage," "if damage would be dealt, prevent it"), including shield counters.
- **Fix:** Consult a prevention layer in `deal_damage`/`deal_damage_to_player` before applying damage, with per-source/per-turn shields.
- **Complexity/risk:** **Medium–High** — new subsystem; no current vocab card exercises it, but it is a complete rules gap.

### T2.4 — Counters: only +1/+1 modeled; loyalty tracked separately · `OPEN`
- **Rule:** 122.1 (counter kinds), 122.3 (+1/+1 vs -1/-1 annihilation SBA), 613.4c (counters in layer 7c), 306.5c (loyalty counters)
- **Engine:** a single `plus_one_counters` int adds symmetrically to P and T (`src/components/creature.h:29`); counter handlers reject anything ≠ `"P1P1"` (`src/effects/effect_put_counter.cpp:21-30`, `state_manager_statics.cpp:219-234`). No -1/-1, no keyword counters, no 122.3 annihilation SBA.
- **Loyalty unification note:** planeswalker loyalty is conceptually loyalty *counters* (122.1b / 306.5c), but the engine tracks it as a standalone int `permanent.loyalty` (`src/components/permanent.h:27`), entirely divorced from the counter subsystem — loyalty-ability costs add/subtract this int directly (`src/components/ability.h:55-60`). **When building the typed counter map, loyalty counters should be unified into that mechanism** (a `LOYALTY` counter type in the map, or at minimum a shared add/remove/query API) so all counters — +1/+1, -1/-1, loyalty, keyword — share one code path. Keep `permanent.loyalty`'s existing semantics (SBA at 0, 306.5c) working through the unified store, and preserve the obs encoding (loyalty float).
- **Rules require:** 122.1a -X/-Y counters; 122.1b keyword counters; 122.3 annihilation as an SBA; counters as timestamped 7c modifiers.
- **Fix:** Store counters in a typed map; add the 122.3 annihilation SBA in `state_based_effects`; fold counters into layer 7c (depends on T2.1); migrate loyalty into the same store.
- **Complexity/risk:** **Medium** — typed counter map + one new SBA + loyalty migration; currently +1/+1 and loyalty are the only live counter users.

---

## Tier 3 — Correct-but-incomplete (real deviations, mostly latent)

### Triggered abilities & priority

#### T3.1 — Triggered abilities not ordered APNAP · `OPEN`
- **Rule:** 603.3b
- **Engine:** triggers are pushed onto the stack in raw entity-ID order (`src/systems/state_manager_triggers.cpp:72-92`), not active-player-first. (Flagged by multiple audit passes.)
- **Fix:** collect pending triggers, partition by controller (active player first), let each player order their own group, then push so the LIFO stack matches APNAP.
- **Complexity/risk:** **Medium** — two-pass collect-then-push + controller ordering prompt; changes resolution order, so seeds/recordings shift.

#### T3.2 — No priority window when triggers fire during cleanup · `OPEN`
- **Rule:** 514.3a
- **Engine:** entering CLEANUP force-sets both pass flags (`src/classes/game.cpp:276-278`); a cleanup-triggered ability then resolves via `resolve_top` before any player is offered priority (`:82-89`).
- **Fix:** when triggers fire / SBAs occur in cleanup, reset the pass flags and give the active player priority instead of pre-passing.
- **Complexity/risk:** **Medium** — must distinguish the no-trigger fast path from the trigger-present path within the untap/cleanup "pretend both passed" hack.

#### T3.3 — "Intervening if" checked once, not twice · `OPEN`
- **Rule:** 603.4
- **Engine:** only the resolution-time check runs (`condition_check_svar`, `src/components/ability.cpp:780-784`); the trigger-time gate is missing, so intervening-if triggers always go on the stack.
- **Fix:** evaluate the condition in `check_triggered_abilities` before pushing, in addition to the resolution-time check; requires flagging intervening-if triggers at parse time.
- **Complexity/risk:** **Medium** — touches trigger plumbing; low rules-risk.

### Stack / resolution / targeting

#### T3.4 — Multi-target partial resolution / "all targets illegal" · `OPEN`
- **Rule:** 608.2b / 608.2c
- **Engine:** `is_target_valid` returns false if *any* of multiple targets is illegal (`src/components/ability.cpp:633-637`), and `resolve()` then fizzles the entire effect (`:761-766`). Single-target effect handlers read only `ab.target`, ignoring `ab.targets` (`src/effects/effect_deal_damage.cpp:32-53`, `effect_counter.cpp`).
- **Rules require:** counter by rules only if *all* targets are illegal; otherwise resolve over the still-legal subset.
- **Fix:** for multi-target abilities, fizzle only when all targets illegal; filter `targets` to the legal subset at resolution; have damage/destroy/counter/pump handlers iterate `ab.targets`.
- **Complexity/risk:** **Medium** — handler rework; no current vocab card multi-targets these effects (verify before relying on it).

#### T3.5 — `fizzle()` is a stub; over-suppresses sub-abilities; stale-target propagation · `OPEN`
- **Rule:** 608.2
- **Engine:** `fizzle()` only logs (`src/components/ability.cpp:441-446`); a fizzle suppresses *all* chained sub-abilities even when only the targeted portion should be skipped (TODO at `:764`). When the resolution-time condition fails, `resolve()` still resolves sub-abilities and copies a possibly-illegal `this->target` into each without re-checking (`:785-793`).
- **Fix:** make fizzle target-scoped (skip only the portion that lost its target); re-run `is_legal_target` on propagated targets before resolving dependent sub-abilities.
- **Complexity/risk:** **Low–Medium** — localized to `resolve()`; mainly relevant to modal/charm-style cards.

### State-based actions

#### T3.6 — Aura attachment legality not checked · `OPEN`
- **Rule:** 704.5m
- **Engine:** no SBA checks whether an Aura is attached to an illegal/absent object or unattached; no aura→host link field exists (only equipment links).
- **Fix:** add an aura attachment-tracking field and an SBA that moves illegally-attached/unattached auras to the owner's graveyard.
- **Complexity/risk:** **Medium** — new attachment state + legality re-check.

#### T3.7 — Empty-library loss applied inline, not as an SBA; no simultaneous-loss draw · `OPEN`
- **Rule:** 704.5b, 104.4a
- **Engine:** loss is set directly inside the draw routine (`src/systems/orderer.cpp:331-342`), not via the SBA loop; Player A's life-loss is checked before B's (`src/systems/state_manager.cpp:95-106`), so simultaneous loss can never be a draw — A always "loses." (Flagged by multiple passes.)
- **Rules require:** loss is an SBA (704.5b) checked at the next SBA check; simultaneous losses → game draw (104.4a).
- **Fix:** set a `drew_from_empty_library` flag and resolve it in the SBA loop; evaluate both players' loss conditions in one pass. Note project policy forbids draws in tests — coordinate before changing the simultaneity behavior.
- **Complexity/risk:** **Low–Medium** — correct for single deck-out today; biased/early-ending otherwise.

#### T3.8 — Poison loss SBA missing (field exists) · `OPEN`
- **Rule:** 704.5c
- **Engine:** `poison_counters` exists on `Player` and is serialized to the obs (`src/components/player.h:13`, `src/machine_io.cpp:59`) but no SBA checks the 10-poison loss.
- **Fix:** add a poison-loss SBA when an infect/poison card is implemented.
- **Complexity/risk:** **Low**, latent.

#### T3.9 — Missing SBAs: +1/+1 vs -1/-1 annihilation, world rule · `OPEN`
- **Rule:** 704.5q (folds into T2.4), 704.5k
- **Engine:** neither exists; no vocab card uses them.
- **Complexity/risk:** **Low**, latent.

### Combat (remaining)

#### T3.10 — No damage-assignment order/division for multiple blockers · `OPEN`
- **Rule:** 509.2 / 510.1c
- **Engine:** damage assigned to blockers in entity-ID order with no player choice (`src/systems/state_manager_combat.cpp:119-142`).
- **Rules require:** attacking player orders blockers and divides damage; with trample, lethal to each blocker before excess.
- **Fix:** prompt the attacking player for blocker order / damage division at the combat-damage step.
- **Complexity/risk:** **High** — new decision point + ActionCategory; ML action-space impact.

#### T3.11 — Trample lethal accounting ignores already-marked damage · `OPEN`
- **Rule:** 510.1c
- **Engine:** trample excess uses `needed = toughness` (or 1 with deathtouch) ignoring damage already marked on the blocker (`src/systems/state_manager_combat.cpp:133-168`).
- **Fix:** per-blocker lethal = `max(0, toughness - already_marked)` (or 1 with deathtouch). Couples to T3.10.
- **Complexity/risk:** **Medium**.

#### T3.12 — First/double strike eligibility re-evaluated live, not snapshotted · `OPEN`
- **Rule:** 510.4 / 702.4c / 702.7c
- **Engine:** `should_deal_damage` reads current keywords each step (`src/systems/state_manager_combat.cpp:36-43`); no snapshot of who had FS/DS at the first step's start.
- **Fix:** record per-creature `had_fs`/`had_ds` at the start of the first combat-damage step and gate the second step on those.
- **Complexity/risk:** **Medium** — transient per-combat state; low-frequency edge case.

#### T3.13 — Menace not enforced · `OPEN`
- **Rule:** 509.1c / 702.111
- **Engine:** no minimum-blocker requirement; `determine_blockable_attackers` has no menace case (`src/action_processor.cpp:666-703`).
- **Fix:** post-declaration validation that any menace attacker is unblocked or blocked by ≥2 creatures.
- **Complexity/risk:** **Medium** — needs a whole-assignment validation pass the per-blocker model lacks.

#### T3.14 — Defender not enforced · `OPEN`
- **Rule:** 702.3b
- **Engine:** attacker eligibility checks only tapped/summoning-sick (`src/action_processor.cpp:489-505`); Defender creatures can attack.
- **Fix:** skip Defender creatures in the eligibility loop.
- **Complexity/risk:** **Low** — one filter line.

### Targeting keywords

#### T3.15 — Hexproof not enforced · `OPEN`
- **Rule:** 702.11b
- **Engine:** `is_legal_target` checks protection but not hexproof (`src/components/ability.cpp:469-626`, protection at `:603`).
- **Fix:** reject a hexproof candidate when `controller != caster` (caster param already available).
- **Complexity/risk:** **Low–Medium**.

#### T3.16 — Shroud not enforced · `OPEN`
- **Rule:** 702.18b
- **Engine:** `is_legal_target` never checks Shroud.
- **Fix:** reject any candidate with Shroud regardless of controller.
- **Complexity/risk:** **Low**.

#### T3.17 — Other evasion (Intimidate/Fear/Skulk/Horsemanship) not enforced · `OPEN`
- **Rule:** 702.13 etc.
- **Engine:** `determine_blockable_attackers` honors only Flying/Reach, Shadow, landwalk, protection (`src/action_processor.cpp:682-699`).
- **Fix:** extend with the corresponding checks per keyword. Latent.
- **Complexity/risk:** **Low** per keyword.

### Triggers / zones

#### T3.18 — Self "dies"/leaves-battlefield triggers skipped · `OPEN`
- **Rule:** 603.6d / 603.10 (last-known information)
- **Engine:** `check_triggered_abilities` only scans battlefield permanents (`src/systems/state_manager_triggers.cpp:72-75`), so a creature's own dies/LTB trigger is skipped once it is in the graveyard.
- **Fix:** for self LTB/dies triggers, consider recently-departed sources using last-known information captured at move time.
- **Complexity/risk:** **Medium**, latent — first self-dies-trigger card added will silently no-op until fixed.

### Mana (remaining)

#### T3.19 — Cost paid before targets chosen for spells · `OPEN`
- **Rule:** 601.2c precedes 601.2f–h
- **Engine:** the `CAST_SPELL` path pays X/Phyrexian/mana before choosing targets (`src/action_processor.cpp:957-1019`); the activated-ability path is correctly ordered (`:240-244`).
- **Fix:** move target selection ahead of the payment block in the spell-cast path.
- **Complexity/risk:** **Medium** — interacts with the payment-failure rewind snapshot; mostly cosmetic for current cards.

#### T3.20 — X stored as a single game global, not per-spell · `OPEN`
- **Rule:** 601.2b
- **Engine:** chosen X is written to `cur_game.x_paid` (`src/action_processor.cpp:970`, `src/classes/game.h:100`); two X spells/abilities on the stack clobber each other. `evaluate_dynamic_amount` also lacks a `Count$xPaid` case (`src/components/ability.cpp:680-757`).
- **Fix:** store locked X on the spell/ability entity; read from the resolving object; add a `Count$xPaid` branch.
- **Complexity/risk:** **Low–Medium** — only Green Sun's Zenith uses X today (as a CMC gate, not a scaling amount).

#### T3.21 — Snow mana and generic cost reductions unimplemented · `OPEN`
- **Rule:** 107.4h / 106.11 (snow); 601.2f (reductions)
- **Engine:** no `{S}` concept; `effective_base_cost` applies cost increases (`src/systems/state_manager_statics.cpp:51-56`) but no reductions.
- **Fix:** add when a snow / cost-reducing card enters the vocab.
- **Complexity/risk:** **Low**, latent.

---

## Suggested order of attack

1. **Quick correctness wins** — T1.2, T1.3, T1.5, T3.14 (Defender), T3.15/T3.16 (Hexproof/Shroud): small guarded checks + one persisted flag. High value, low risk.
2. **Until-end-of-turn effect store** (T1.1): scaffolding the layer system will reuse.
3. **Layer system** (T2.1): keystone refactor; counters/set/modify/timestamp ordering (incl. T2.4) collapse into it.
4. **Replacement/prevention framework** (T2.2/T2.3) and **trigger correctness** (T3.1 APNAP, T3.3 intervening-if, T3.2 cleanup priority) once the above stabilizes.

---

## Verified correct (no action)

Recorded so future audits don't re-investigate:

- Active player gets priority first each step; regains priority after each resolution (117.3a/b).
- No priority in untap step; first player skips first draw (502.4 / 103.8a).
- First-strike combat-damage step created only when a striker is present (510.4).
- LIFO stack resolution; targets chosen at cast and re-verified at resolution (405 / 608.2b); "can't cast with no legal target" gate (601.2c).
- Normal mana abilities never use the stack (605.3b); mana pool empties each step/phase, no mana burn (106.4); generic payable by any color, colored/`{C}` require exact match (107.4b/c); Phyrexian mana (107.4f); payment-failure rewind (733).
- SBAs checked before priority and repeated until stable (704.3); 0-or-less life loss, 0-toughness/lethal/deathtouch death batched simultaneously, planeswalker 0 loyalty, legend rule (APNAP, one conflict per pass), tokens ceasing to exist off-battlefield (704.5a/f-j/d).
- Lifelink life-gain batched simultaneously with combat damage (119.3); deathtouch lethal (702.2c).
- Supported & enforced keywords: Flying, Reach, First/Double Strike, Deathtouch, Trample, Lifelink, Haste, Flash, Shadow, Landwalk, Protection, Prowess, Exalted, Equip (sorcery-speed, you-control), Delve/Cycling/Flashback/Evoke/Dredge.
- Static P/T buffs applied continuously (rebuilt each SBA pass), not baked in once (613.5).
