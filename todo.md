TODO:

Observation space:
- Add opponent's hand to the observation space when it becomes revealed (e.g. by
  Surgical Extraction / Thoughtseize / Duress). There are currently no per-card
  opponent-hand identity slots — only a hand count + the match-scoped revealed
  multi-hot (mark_card_revealed). Add per-card slots for known opponent-hand cards,
  and have those slots update (move/clear) when a known card leaves the hand for
  another public zone, so the belief tracks the specific card rather than just
  "was revealed once this match".
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



