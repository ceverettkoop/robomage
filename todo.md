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

Known ML problems:
Does not know what's in exile

Scripted agent fixes (deferred from sanity-check 2026-06-22; minimum changes to reduce noise):
- X spells: agent casts Green Sun's Zenith the moment it has 1 mana, forcing X=0. The
  X-value choice (OTHER_CHOICE) is currently random.choice. Fix: pick the MAX affordable X,
  and/or hold GSZ until it has >=3 mana so X>=2. X=0 only ever finds Dryad Arbor; subsequent
  X=0 casts find nothing and shuffle GSZ back (wasted). scripted_agent.py _greedy_action.
- Keen-Eyed Curator: agent over-activates the graveyard-exile ability, stacking several
  copies that all target the same card (picks target index 0 each time) -> later copies
  ChangeZone-fizzle. Fix: only one Curator ability on the stack at a time (skip ACTIVATE if a
  self Curator ability is already on the stack), so each resolves before the next targets a
  fresh card. scripted_agent.py _greedy_action ACTIVATE loop.

Engine bugs found in scripted-vs-scripted sanity check (2026-06-22, decks delver/doomsday/mav):
- Unholy Heat always deals 0 damage. Script SVar:X:Count$Delirium.6.2 is not parsed (parse.cpp
  delirium/amount_svar resolution only matches a "GE" pattern + Count$Valid/Targeted$, never
  Count$Delirium) -> X stays 0. Dead card. (vocab 29, delver)
- Knight of the Reliquary doesn't grow from lands in graveyard (stays 2/2). Static
  SVar:X:Count$ValidGraveyard Land.YouOwn is unhandled by evaluate_sa_svar in state_manager.cpp
  (only Count$TypeInYourYard.<Type> and Count$ValidGraveyard Card$CardTypes are supported).
  Counter-based growth (Scythecat etc.) works; the land static does not. (vocab 41, mav)
- Deep Analysis (and Draw generally) draws for the spell's CONTROLLER, ignoring the chosen
  target player. Script is SP$ Draw | ValidTgts$ Player ("target player draws two"). Harmless
  here only because the agent targeted the opponent; the Draw effect ignores ab.target.

Observation space:
- Add opponent's hand to the observation space when it becomes revealed (e.g. by
  Surgical Extraction / Thoughtseize / Duress). There are currently no per-card
  opponent-hand identity slots — only a hand count + the match-scoped revealed
  multi-hot (mark_card_revealed). Add per-card slots for known opponent-hand cards,
  and have those slots update (move/clear) when a known card leaves the hand for
  another public zone, so the belief tracks the specific card rather than just
  "was revealed once this match".

