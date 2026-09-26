TODO:

FEED AZ BACK INTO PPO:
Distill AZ's policy into PPO (search-improved actions taught to the fast net), and let PPO keep
learning its own value via RL. Don't try to move the value across — the two critics are answering
different questions.

Policy: yes, and it's the principled direction (this is literally Expert Iteration /
AlphaZero-as-teacher). AZ's MCTS visit-count distribution π* is a stronger policy target than PPO's
on-policy logits. You distill it into PPO's policy head via supervised cross-entropy/KL to π* over
self-play states — or, equivalently, behavior-clone AZ self-play trajectories into PPO offline. The
reverse tensor map already exists implicitly: `az_net.from_ppo`'s mapping
(features_extractor↔trunk, mlp_extractor.value_net↔value_body, value_net↔value_head,
action_scorer↔action_scorer) is symmetric for the shape-matching per-action flavor, so the shared
trunk + policy head graft back cleanly. Caveat: SB3 MaskablePPO has no built-in distillation hook —
you'd add an auxiliary CE loss or do an offline BC pre-train.


## Open engine correctness issues

- **Dauthi Voidwalker (vocab 88)** does not work as written. Its {T}, sacrifice ability should grant
  an optional, turn-long "you may play it without paying its mana cost" permission (the script's
  `DB$ Effect | StaticAbilities$ MayPlay`, `AffectedZone$ Exile`). Instead `effect_choose_card.cpp`
  (~176-217) acts at resolution: a permanent card is put straight onto the battlefield (not cast —
  no cast triggers; a land uses no land drop) and an instant/sorcery is forced onto the stack via
  the blocking mini-cast. Void counters are also a `cur_game.void_countered` set, not real counters.

- `rules_mod::may_play_lands_from_graveyard` (`src/systems/rules_modifying.cpp:197`) ignores
  `Affected$`: it returns true if the player controls ANY `may_play_from_graveyard` static, which
  `parse.cpp:3820` sets for every `MayPlay$ True` static without checking the filter. Latent — the
  vocab's statics (Icetill Explorer, Mole Man) are all `Affected$ Land.YouOwn`; wrong once a
  filtered "play <X> from your graveyard" card is added.

- `cur_game.delve_exiled` is game-GLOBAL: cleared at each delve cast's start
  (`action_processor.cpp:3567`) and when Murktide's etbCounter replacement consumes it
  (`replacement_effects.cpp:463`). Two delve spells on the stack at once would clobber each other
  (the later cast wipes the earlier's record). Real fix: per-spell scoping (exile set keyed by the
  spell entity). Latent — the only vocab delve card is Murktide Regent (sorcery speed).

- Ward doesn't trigger on a copied spell's new targets: `effect_copy_spell.cpp` picks them without
  the Ward/BECAME_TARGET hook (`fire_targeting_hooks`, `action_processor.cpp:1510`) that the cast and
  activation flows run. Same gap for Dauthi's forced mini-cast (`effect_choose_card.cpp:208`). Check
  other "choose new targets" paths too.

## Audit: ability-param keys the parser silently ignores

`apply_param_to_ability` (`src/parse.cpp`, `ignored_keys` at ~1945) skips a hard-coded key set
instead of warning. **Task: review every key and confirm the engine isn't missing functionality by
ignoring it.** Each key has an inline justification at its definition; verify each still holds (a
card added later can make a formerly-cosmetic param load-bearing).

**No implementation ever needed** (Forge-AI hints and display/prose only):
- AI hints: `AILogic`, `AINoRecursiveCheck`, `AITgts`, `AIXMax`.
- Descriptions / prompts / prose: `SpellDescription`, `StackDescription`,
  `TriggerDescription`, `ConditionDescription`, `TgtPrompt`, `SelectPrompt`,
  `ValidTgtsDesc`, `ValidDescription`, `ChangeTypeDesc`, `ChangeValidDesc`,
  `GiftDescription`, `VoteMessage`, `SacMessage`, `PrecostDesc`, `Name`, `Image`.

**TO CHECK**
`Duration`, `Hidden`, `ForgetOtherTargets`, `ForgetOnMoved`, `Choices`,
`ControlledByPlayer`, `Reveal`, `Ultimate`, `Triggers`, `Stackable`,
`ForgetOtherRemembered`, `DamageMap`, `Announce`, `ValidCards`, `Imprint`, `ChoiceZone`,
`ShuffleNonMandatory`, `ForceRevealToController`, `Controller`, `Keywords`, `RemoveKeywords`.

Marked "NOT YET MODELED" at the site (`parse.cpp` ~2053-2061):
- **Reorder$ True** (Brainstorm) -- brainstorm works without it? maybe reorder false is unimplemented?
- **TriggerAmount$ / RememberOriginalTokens$** (Ajani, Nacatl Avenger).
- **LockTokenScript$ True** (Into the Flood Maw) -- unknown intent.
- **ExileOnMoved$ Battlefield** (Manifold Key) -- unknown intent.

## Cosmetic / logging

- Prevented player damage double-logs: `deal_damage_to_player` logs "N damage prevented" and
  returns false, but `effect_deal_damage.cpp` (~60 and ~84) then logs "Dealt N damage (now at X)"
  anyway (life is correctly unchanged). Seen with The One Ring vs Ancient Tomb.
- Planeswalker loyalty goes negative on excess damage ("loyalty now -10"): `damage_planeswalker`
  (`game_queries.h:497`) → `add_counters` doesn't floor at 0, so the stored LOYALTY count (not just
  the log) is negative (CR 609.3: remove only as many as exist). The 0-loyalty SBA still destroys it.


## Deferred — bigger, needs its own session

### T3.2 cleanup-step trigger priority (rule 514.3a) — DEFERRED
Cleanup triggers exist only as bookkeeping: `Phase$ Cleanup` parses to `CLEANUP_BEGAN`
(`parse.cpp` ~3336), used by Carpet of Flowers' `Static$ True` reset, which bypasses the stack.
Missing: if a stack-using trigger fires or an SBA happens during cleanup, players must get priority
and another cleanup step follows (514.3a). Entering CLEANUP force-sets both pass flags
(`game.cpp` ~577), so such a trigger would resolve with no priority window. No vocab card needs it;
Thawing Glaciers (delayed "return at the next cleanup") or a madness/discard card are natural test
vehicles. Fix: when triggers/SBAs occur in cleanup, reset the pass flags and give the active player
priority, keeping the no-trigger fast path.

### Aura-on-PLAYER state-based action (CR 704.5m) — DEFERRED, latent
The Aura fall-off SBA (`state_manager.cpp` ~231, "704.5n" comment) destroys an Aura whenever
`Permanent::equipped_to == 0`. `equipped_to` is only written for a Permanent enchant target
(`state_manager_statics.cpp` ~729), so an Aura that enchants a PLAYER (a Curse, "Enchant player")
resolves unattached and is immediately put into the graveyard. No such Aura is in the vocab. The
engine deliberately has no player-attachment model (see The One Ring). When one is added: record the
enchanted PLAYER as the attach target and make the SBA treat a legally player-attached Aura as
attached. Until then, do NOT add an "Enchant player" Aura to the vocab.

### AZ search: residual `safe=0` prompt sites
Every decision kind is a searchable root, except these residual blocking prompts (classified and
CI-guarded by `train/test_snapshot.py`'s `PROMPT_SITE_WHITELIST`, which fails if a new blocking
`get_input` appears): the 616.1 multi-replacement `choose_one` prompt (needs two replacement effects
on one event, e.g. two Leylines of the Void; counted by `replacement::choose_one_prompt_count`), the
Mox Diamond discard-a-land replacement prompt, `effect_choose_card`'s cast-from-exile mini-cast
(Dauthi — see above), interactive-only prompts machine mode never reaches (mana payment, hybrid
pips), and defensive outside-main-loop fallbacks.

## Harness / tooling

- A preset card name the engine can't load (e.g. half of a comma-split name — `--battlefield-a/-b`
  and the other preset lists split on commas in both the harness and `main.cpp`) assert-crashes
  (reported as a DRAW): `Orderer::place_on_battlefield` / `place_in_graveyard` / `place_in_zone`
  (`orderer.cpp` ~774) call `GetComponent<CardData>` on `load_card`'s failure return. Workaround:
  comma-free names (`name_to_uid` strips punctuation). Fix: fail the preset with a clear error.

## Engine robustness

### MAX_ENTITIES exhaustion / unbounded-loop detection
- `MAX_ENTITIES` is 1000 (`src/ecs/entity.h`; lowered from 5000 to cut per-sim component-array
  memory). It bounds CONCURRENT living entities (IDs recycle), not lifetime total. Typical peak
  ~250-350; token storms maybe 400-500.
- RISK: RELEASE builds use `-DNDEBUG`, so `EntityManager::CreateEntity`'s cap assert
  (`src/ecs/entity_manager.h:25`) is gone; on exhaustion it calls `front()` on an empty queue → UB.
  Need a runtime guard that fails the game cleanly with a diagnostic.
- Also guard against an unbounded rules-engine loop creating entities without end (token/trigger
  loop, replacement ping-pong): a living-entity high-water tripwire or per-step creation cap that
  aborts with a logged diagnostic so it surfaces in training.
- Before trusting 1000: instrument a debug build to record peak living-entity count across the
  league decks and set the cap from the measured high-water mark (peak x2).
