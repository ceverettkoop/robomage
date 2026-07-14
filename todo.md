TODO:

FEED AZ BACK INTO PPO:
Aistill AZ's policy into PPO (search-improved actions taught to the fast net), and let PPO keep learning its own value via RL. 
Don't try to move the value across — the two critics are answering different questions.

Policy: yes, and it's the principled direction (this is literally Expert Iteration / AlphaZero-as-teacher). AZ's MCTS visit-count distribution π* is a stronger policy target than PPO's on-policy logits. You distill it into PPO's policy head via supervised cross-entropy/KL to π* over self-play states — or, equivalently, behavior-clone AZ self-play trajectories into PPO offline. The reverse tensor map already exists implicitly: from_ppo's mapping (features_extractor↔trunk, mlp_extractor.value_net↔value_body, value_net↔value_head, action_scorer↔action_scorer) is symmetric for the shape-matching per-action flavor, so the shared trunk + policy head graft back cleanly. Caveat: SB3 MaskablePPO has no built-in distillation hook — you'd add an auxiliary CE loss or do an offline BC pre-train.


## Open engine correctness issues

- Dauthi Voidwalker (vocab 88) does not work exactly as written — casts immediately (should be
  the exile-a-card-then-cast-from-exile ability with its "can't be cast unless" restriction).

- rules_modifying::may_play_lands_from_graveyard ignores Affected$ filters (rules_modifying.cpp:136):
  returns true if the player controls ANY `may_play_from_graveyard` static, without consulting
  WHICH cards that static permits. Latent — current vocab only has Crucible-style "play all lands
  from your graveyard", so the boolean is correct today; wrong if a filtered "play only <X> from
  graveyard" card is ever added.

- delve_exiled is a game-GLOBAL vector (cur_game.delve_exiled), written during a spell's cost
  payment and cleared only at a delve creature's ETB. Two delve spells on the stack at once (cast
  one in response to another) would share/miscount the exile set — the later spell over-counts
  (its ETB sees both spells' exiles), the earlier under-counts (its set was already cleared). A
  plain reset-at-cast-start is INSUFFICIENT (it wipes the earlier spell's record); the real fix is
  per-spell scoping (store the exile set keyed by the spell entity). Latent — every current vocab
  delve card is a sorcery-speed creature (Murktide, Barrowgoyf), so two can't be on the stack at
  once.

- When a copied spell targets a permanent with ward, ward will not trigger. Possible similar issues related to
  selecting new targets for copied spells.

## Audit: ability-param keys the parser silently ignores

`apply_param_to_ability` (`src/parse.cpp`) skips a hard-coded `ignored_keys` set
instead of warning. **Task: review every key in that set and confirm the engine
is not missing functionality by ignoring it.** Each key below has an inline
justification comment at its definition; verify each still holds (a card added
later can make a formerly-cosmetic param load-bearing).

**No implementation ever needed** (Forge-AI hints and display/prose only — not
part of our rules model, per project scope):
- AI hints: `AILogic`, `AINoRecursiveCheck`, `AITgts`, `AIXMax`.
- Descriptions / prompts / prose: `SpellDescription`, `StackDescription`,
  `TriggerDescription`, `ConditionDescription`, `TgtPrompt`, `SelectPrompt`,
  `ValidTgtsDesc`, `ValidDescription`, `ChangeTypeDesc`, `ChangeValidDesc`,
  `GiftDescription`, `VoteMessage`, `SacMessage`, `PrecostDesc`, `Name`, `Image`.

**TO CHECK**
`Duration`, `Hidden`, `ForgetOtherTargets`, `ForgetOnMoved`, `Choices`,
`ControlledByPlayer`, `Reveal`, `Ultimate`, `Triggers`, `Stackable`,
`ForgetOtherRemembered`, `DamageMap`, `Announce`, `ValidCards`, `Imprint`,
`ClearImprinted`, `ShuffleNonMandatory`, `ForceRevealToController`.
- **Reorder$ True** -- brainstorm works with out it? maybe reorder false is unimplemented?
- **LockTokenScript$ True** unknown intent
- **ExileOnMoved$ Battlefield**  unknown intent

## Cosmetic / logging

- The One Ring damage-prevention double-logs: a prevented Ancient Tomb self-damage prints BOTH
  "N damage prevented" AND "Dealt N damage (now at X)" for the same event (life is correctly
  unchanged). Also planeswalker loyalty prints negative on lethal damage ("loyalty now -10")
  though the SBA correctly destroys it — floor-0 display only (CR 122.1b). Surfaced in fuzz
  campaign #2 (car_doomsday vs tron).


## Deferred — bigger, needs its own session


### T3.2 cleanup-step trigger priority (rule 514.3a) — DEFERRED
No card in the current vocab has a cleanup-step trigger that uses the stack, so the "no priority
window when triggers fire during cleanup" bug cannot be observed or tested with the present card
pool (the Forge `Phase$ Cleanup` triggers in the DB are mostly `Static$ True` bookkeeping that
bypasses the stack). Deferred until a real cleanup-trigger card is added — Thawing Glaciers
(delayed "return to hand at the next cleanup step") or a discard/madness card are the natural test
vehicles. When implementing: entering CLEANUP force-sets both pass flags (game.cpp), and a
cleanup-triggered ability then resolves before any player gets priority — the fix is to reset the
pass flags and give the active player priority when triggers fire / SBAs occur in cleanup,
distinguishing the no-trigger fast path from the trigger-present path.

### Aura-on-PLAYER state-based action (CR 704.5n / 704.5m) — DEFERRED, latent
The 704.5n Aura fall-off SBA (state_manager.cpp, added with Sheltered by Ghosts) treats an Aura as
illegally-attached and destroys it whenever its `Permanent::equipped_to == 0`. But `equipped_to` is
only written when the enchant target has a Permanent component (apply_permanent_components,
state_manager_statics.cpp), so an Aura that enchants a PLAYER (a Curse, or any "Enchant player"
Aura) resolves with equipped_to == 0 and is immediately put into the graveyard — it can never stay
on the battlefield. No player-enchanting Aura is in the vocab yet, so this is latent. Proper fix
needs player-attachment modeling, which the engine deliberately lacks (see The One Ring notes: "no
player-attach in engine"). When a Curse/Enchant-player Aura is added: record the enchanted PLAYER
(not just a Permanent) as the attach target, and make the 704.5n SBA treat a legally
player-attached Aura as attached. Until then, do NOT add an "Enchant player" Aura to the vocab
without this, or it will self-destruct on resolution.

## Harness / tooling

- test_harness raises PlayResolveError when a seat-keyed `--play` spec lands on a mandatory
  cleanup-discard decision.
- Engine `--battlefield-a/-b` (and other preset list flags) split on commas → assert-crash
  (reported as DRAW) on comma-named cards. Workaround: pass comma-free names (name_to_uid strips
  punctuation, so the same card loads). The underlying assert is unfixed.

## AlphaZero / search — unsafe-root coverage

The search safe window covers exactly ONE call site: the main-loop priority decision
(`game_driver.cpp` `search_set_loop_safe(true)` around the loop-top `get_input`). Every other
prompt is an unsafe root: search cannot START there (SNAPSHOT/DETERMINIZE fatal-error), the
SearchController falls back to raw policy argmax, and AZ self-play stores NO training sample —
the policy head never gets visit-distribution targets at those decisions (~9% of decisions at
the measured ~91% safe fraction). Unsafe decisions ARE still traversed as interior tree nodes
inside simulations; only rooting is blocked.

- **Extend the safe window to mandatory choices — the high-value fix.** Declare attackers,
  declare blockers, cleanup discard, and trigger-ordering prompts run at the LOOP TOP
  (`is_mandatory_choice_pending` → `proc_mandatory_choice`, before the safe window) — unlike
  nested prompts they are re-derived each loop iteration from cur_game + ECS, so a restore
  landing at the loop top plausibly re-derives the same prompt. If so, marking them loop-safe
  puts COMBAT DECLARATIONS under MCTS and into the AZ training data (today the net's combat
  policy is shaped only by PPO warm-start + value backup, never by search CE targets). Must be
  proven by the snapshot round-trip CI test before trusting it (check
  `process_turn_based_actions` idempotence on the restored path).
- Nested mid-resolution prompts (targets/modes at cast, mana payment, search/dig/scry picks,
  discard/sacrifice, unless-costs, etc.) are architecturally unsafe — they live halfway down a
  live C++ call stack that a game-state snapshot cannot rebuild. Making those searchable means
  either resolution-state serialization or moving the choices to the loop top (cf. the deferred
  modal-at-cast refactor); not worth it piecemeal.
- bo3 sideboarding is now a SEARCHABLE root (Phase 2, DONE 2026-07-13): a MATCH-scoped snapshot
  survives the per-game wipe, so the sideboard decision is searched on the next game's horizon
  and stored as an AZ training sample (next-game z) on both the Python and C++ actor backends.
  See docs/alphazero_status.md "Phase 2 — learned sideboarding". Mulligans remain unsafe roots
  (low training value, fine as fallbacks).

## Engine robustness

### MAX_ENTITIES exhaustion / unbounded-loop detection
- MAX_ENTITIES is 1000 (lowered from 5000 to cut per-sim component-array memory; Ability=992B +
  CardData=856B per slot dominate). The cap bounds CONCURRENT living entities (IDs recycle via the
  free queue), not lifetime total. Typical peak ~250-350; token storms maybe 400-500.
- RISK: RELEASE builds compile -DNDEBUG, so the CreateEntity assert (mLivingEntityCount <
  MAX_ENTITIES) is gone. On exhaustion CreateEntity does mAvailableEntities.front() on an empty
  queue -> UB / crash. Need a real runtime guard: detect cap exhaustion and fail the game cleanly
  (loss/draw-abort with a diagnostic) instead of UB.
- Also guard against an unbounded rules-engine loop creating entities without end (token/trigger
  loop, replacement-effect ping-pong): detect runaway creation (living-entity high-water tripwire
  or per-step creation cap) and abort with a logged diagnostic so it surfaces in training.
- Before trusting 1000: instrument a debug build to record peak mLivingEntityCount across the
  engine-sanity-check decks and set the cap from the measured high-water mark (peak x2).