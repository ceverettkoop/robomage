TODO:
parse and display ability descriptions and targeting prompts

can pay life to negative life - should not be allowed

  - TODO: when a permanent has multiple simultaneous ETB triggers (e.g. evoked Endurance:
    its graveyard-bottom trigger + the evoke sacrifice trigger), the controller should
    choose the order they go on the stack (APNAP). Currently they are pushed in a fixed
    order and the player is not prompted.

-dauthi does not work exactly as written - casts immediately

keen eyed curator buff from types exiled untested

Pro color untested

Engine stuff:
ML can only pay for spells after choosing them, this does not allow some rare cases of optimal behavior (e.g. floating mana) but should reduce noise

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

Observation space:
- Add opponent's hand to the observation space when it becomes revealed (e.g. by
  Surgical Extraction / Thoughtseize / Duress). There are currently no per-card
  opponent-hand identity slots — only a hand count + the match-scoped revealed
  multi-hot (mark_card_revealed). Add per-card slots for known opponent-hand cards,
  and have those slots update (move/clear) when a known card leaves the hand for
  another public zone, so the belief tracks the specific card rather than just
  "was revealed once this match".

