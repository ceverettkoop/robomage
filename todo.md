TODO:

## Open engine correctness issues

- Dauthi Voidwalker (vocab 88) does not work exactly as written — casts immediately (should be
  the exile-a-card-then-cast-from-exile ability with its "can't be cast unless" restriction).

- Keen-Eyed Curator (vocab 40): buff from the types among exiled cards — untested.

- Protection from a color (pro-color) — untested.

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

**Justified as cosmetic/redundant in code — spot-check the reasoning holds:**
`Duration`, `Hidden`, `ForgetOtherTargets`, `ForgetOnMoved`, `Choices`,
`ControlledByPlayer`, `Reveal`, `Ultimate`, `Triggers`, `Stackable`,
`ForgetOtherRemembered`, `DamageMap`, `Announce`, `ValidCards`, `Imprint`,
`ClearImprinted`, `ShuffleNonMandatory`, `ForceRevealToController`.

**Known unimplemented — currently no-ops, need a real handler** (the affected
cards still play via their other tags; drop each from `ignored_keys` when done):
- **Reorder$ True** (Brainstorm): let the player choose the order of the cards put
  back on top of the library. Currently the returned cards keep a default order.
- **TriggerAmount$ Remembered$Amount** and **RememberOriginalTokens$ True**
  (Ajani, Nacatl Avenger): carry the number of tokens created to the transform
  trigger, and remember the original token set it references.
- **LockTokenScript$ True** (Into the Flood Maw): pin the gifted tapped-Fish
  token's script for the gift clause.
- **ExileOnMoved$ Battlefield** (Manifold Key): exile the permanent when it moves
  off the battlefield.

## Cosmetic / logging

- The One Ring damage-prevention double-logs: a prevented Ancient Tomb self-damage prints BOTH
  "N damage prevented" AND "Dealt N damage (now at X)" for the same event (life is correctly
  unchanged). Also planeswalker loyalty prints negative on lethal damage ("loyalty now -10")
  though the SBA correctly destroys it — floor-0 display only (CR 122.1b). Surfaced in fuzz
  campaign #2 (car_doomsday vs tron).


## Deferred — bigger, needs its own session

### Modal spell mode & target selection at CAST (CR 601.2b/c) — FIXED 2026-07-02

Modes + targets are now announced at cast (`announce_spell_targets` in src/action_processor.cpp;
picks recorded in `Ability::charm_chosen`), `effects::charm` is a resolver only (per-mode CR 608.2b
re-verification; choose-at-resolution loop kept as a fallback for unrouted cast paths), copies keep
modes and may re-target (effect_copy_spell.cpp), and `spell_has_castable_targets` requires
CharmNum$ choosable modes. The state vector also grew (STATE_SIZE 2919 → 3183): every stack slot
now serializes its announced targets (all spells/abilities) and chosen-mode multi-hot — a shape
change that invalidates ALL pre-existing checkpoints (retrain from scratch). Engine-internal
ordering note: X is chosen before modes (strict 601.2b says modes first) because mode choosability
can depend on X (Kozilek's Command `Creature.cmcLEX`); not opponent-observable.

Remaining acceptable gaps: mode multi-hot capped at 6, serialized targets capped at 4 per stack
object; an all-modes-untargetable charm COPY is still created and fizzles at resolution
(CR 707.10c pruning not applied per-mode).

### Pre-existing breakage surfaced while testing the modal fix (2026-07-02)

- **meta/ deck loading crashes the engine**: `--deck-a meta/arclight_phoenix` (also boros_aggro)
  dies at startup — parse_card_script assert after trying to open `p/petal.txt` / `p/parlor.txt`
  (something splits multi-word card names: "Lotus Petal" → "Petal", "Elegant Parlor" → "Parlor";
  en route it even parses unrelated fallback matches like charm_peddler.txt). Reproduced on
  UNMODIFIED main (verified via git stash), so unrelated to the modal change. league/ decks load
  fine. Affects any regression pass over meta decks.
- **train/test_revealed_accumulator.py is broken on main**: imports `TestHarness` /
  `get_scripted_action` from train/test_harness.py, which no longer exports them (harness refactor
  moved the loop into runner.py). (train/regression/replay_diff.py had the same breakage and was
  FIXED with the CI work — it now drives the deterministic scripted game via runner.run_games and
  its corpus was re-recorded; test_revealed_accumulator.py still needs the same treatment.)

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

## ML / observation

- ML can only pay for spells AFTER choosing them — precludes some rare optimal lines (e.g. floating
  mana) but reduces noise. LED is the written exception (allows ML to float mana).
- ML does not know what's in exile.

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