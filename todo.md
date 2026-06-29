TODO:

Observation space:
- DONE: Opponent-hand cards are now carried by their specific identity once revealed
  (Duress/Thoughtseize/tutor). Added a per-game Zone::identity_known flag, set inside
  mark_card_revealed when the card is in HAND and cleared on any zone change
  (Orderer::add_to_zone). machine_io serializes a 10-slot known-opponent-hand block
  (STATE_SIZE 2909->2919, layout [2909-2918]); env.py/extractor.py consume it via the
  shared entity encoder (masked mean+max). A slot clears automatically when the known
  card leaves the hand, so the belief tracks the specific card, not just "seen once
  this match". (effect_discard now also records the Duress/Thoughtseize reveal.)
parse and display ability descriptions and targeting prompts

at present a player can pay life to negative life - should not be allowed, they should be allowed to pay to 0 (i.e. kill themselves) but not pay more life than they have

  - DONE (T3.1): when a permanent has multiple simultaneous ETB triggers (e.g. evoked Endurance:
    its graveyard-bottom trigger + the evoke sacrifice trigger), the controller is now prompted
    to choose the order they go on the stack (APNAP, rule 603.3b). See state_manager_triggers.cpp
    place_triggers_apnap.

DEFERRED — T3.2 cleanup-step trigger priority (rule 514.3a):
  No card in the current vocab has a cleanup-step trigger that uses the stack, so the
  "no priority window when triggers fire during cleanup" bug cannot be observed or tested
  with the present card pool (the Forge `Phase$ Cleanup` triggers in the DB are mostly
  `Static$ True` bookkeeping that bypasses the stack). Deferred until a real cleanup-trigger
  card is added to the vocab — Thawing Glaciers (delayed "return to hand at the next cleanup
  step") or a discard/madness card are the natural test vehicles. When implementing: entering
  CLEANUP force-sets both pass flags (game.cpp), and a cleanup-triggered ability then resolves
  before any player gets priority — the fix is to reset the pass flags and give the active
  player priority when triggers fire / SBAs occur in cleanup, distinguishing the no-trigger
  fast path from the trigger-present path.

-dauthi does not work exactly as written - casts immediately

keen eyed curator buff from types exiled untested

Pro color untested

DEFERRED — Aura-on-PLAYER state-based action (CR 704.5n / 704.5m), latent:
  The new 704.5n Aura fall-off SBA (state_manager.cpp, added with Sheltered by Ghosts) treats an
  Aura as illegally-attached and destroys it whenever its `Permanent::equipped_to == 0`. But
  `equipped_to` is only written when the enchant target has a Permanent component
  (apply_permanent_components, state_manager_statics.cpp), so an Aura that enchants a PLAYER (a
  Curse, or any "Enchant player" Aura) resolves with equipped_to == 0 and is immediately put into
  the graveyard by the SBA — it can never stay on the battlefield. No player-enchanting Aura is in
  the vocab yet, so this is latent (surfaced by code-review high over the league branch, 2026-06).
  Proper fix needs player-attachment modeling, which the engine deliberately lacks (see The One
  Ring notes: "no player-attach in engine"). When a Curse/Enchant-player Aura is added: record the
  enchanted PLAYER (not just a Permanent) as the attach target, and make the 704.5n SBA treat a
  legally player-attached Aura as attached. Until then, do NOT add an "Enchant player" Aura to the
  vocab without this, or it will self-destruct on resolution.

Engine stuff:
ML can only pay for spells after choosing them, this does not allow some rare cases of optimal behavior (e.g. floating mana) but should reduce noise. LED is an exception, written so ML can float

MAX_ENTITIES exhaustion / unbounded-loop detection:
- MAX_ENTITIES lowered 5000 -> 1000 to cut per-sim component-array memory (~13MB -> ~2.6MB;
  Ability=992B + CardData=856B per slot dominate). The cap bounds CONCURRENT living entities
  (IDs recycle via the free queue), not lifetime total. Typical peak ~250-350; token storms
  maybe 400-500, so 1000 is ~2-4x headroom.
- RISK: RELEASE builds compile -DNDEBUG, so the CreateEntity assert
  (mLivingEntityCount < MAX_ENTITIES) is gone. On exhaustion CreateEntity does
  mAvailableEntities.front() on an empty queue -> undefined behavior / crash. Need a real
  runtime guard (not just assert): detect cap exhaustion and fail the game cleanly
  (loss/draw-abort with a diagnostic) instead of UB.
- Also need a guard against an unbounded loop in the rules engine creating entities without
  end (e.g. a token/trigger loop, or replacement-effect ping-pong) that would silently march
  toward the cap. Detect runaway entity creation (e.g. living-entity high-water tripwire, or
  per-step creation cap) and abort the game with a logged diagnostic so it surfaces in
  training instead of crashing or hanging.
- Before raising confidence in 1000: instrument a debug build to record peak
  mLivingEntityCount across the engine-sanity-check decks and set the cap from the measured
  high-water mark (peak x 2).

Known ML problems:
Does not know what's in exile

=========================================================================
DEAD-CODE / DUPLICATION CLEANUP (see docs/dead_code_audit.md)
=========================================================================
Tier 1 (working through in order, each committed separately):
[X] 1. Battlefield-permanent scan loop -> is_battlefield_permanent() in
       game_queries.h; refactored 13 open-coded loops across state_manager,
       state_manager_actions, replacement_effects, effect_{sacrifice,amass,pump,
       destroy_all}. effect_choose_card excluded (scans EXILE, not battlefield).
       Behavior-preserving: helper does NOT test is_phased_out; phased checks
       kept explicit at sites that had them. Build + 26-game scripted regression
       across delver/mav/doomsday: 0 draws, no non-fatal errors.
[X] 2. Entity -> display name ("Player A/B" else card name) -> target_display_name()
       in state_manager_internal.h (next to entity_name; defined in
       state_manager_statics.cpp). Replaced the player-or-entity ternary at the
       3 action_processor tgt_name logs, deleted the attack_target_name static
       (3 callers -> helper), folded the select-target life label, and the 2
       combat damage-to-player logs + machine_io stack-target name. Behavior-
       preserving (player_name() returns the same "Player A"/"Player B").
       Verified: build + transcript shows correct names + 18-game regression, 0 draws.
[X] 3. Entity -> card-vocab-index. Added action_card_vocab_idx() (machine_io.h/.cpp)
       implementing the full Permanent(token)->CardData->Ability.source chain. Now
       used by populate_query (BQUERY), record_chosen_action (action-history log,
       FIXES the bug where it omitted the Permanent/token + ability-source-permanent
       cases -> logged vs queried id mismatch), and the stack feature extractor
       (folded in get_stack_card_vocab_idx, kept its Token-source fallback;
       behavior-preserving). get_card_vocab_idx kept separate (slot extractor with
       its own TOKEN_SENTINEL fallback). Build + 18-game regression: 0 draws, no errors.

[X] Phasing follow-up: baked the "phased-out = not on battlefield" rule (CR 702.26e)
       into is_battlefield_permanent(); added battlefield_permanents() list accessor.
       Removed ~13 scattered is_phased_out filter checks across statics (4), state_manager
       (3), replacement_effects (2), state_manager_actions, triggers, mana_system,
       ability target-legality, action_processor attack-targets, machine_io serialization.
       Only intrinsic phasing sites keep the flag (game.cpp untap phase-in/skip,
       effect_phases setter, error.cpp dump). Chose the accessor approach over a new
       PHASED_OUT zone (a zone change would fire CARD_CHANGED_ZONE + strip/re-add the
       Permanent component + re-fire ETB, violating phasing rules). CLAUDE.md updated to
       point future code at these accessors. Build + 32-game regression: 0 draws, no errors.
       (Note: no phasing card is in the vocab yet, so this is behavior-preserving for all
       current games and a latent correctness fix for when one is added.)

Tier 2/3: deferred (safe deletions + localized dup) — see audit doc.



