# Goblin Guide  (vocab index 322)
## Oracle text
Haste
Whenever Goblin Guide attacks, defending player reveals the top card of their library. If it's a land card, that player puts it into their hand.
## Forge script
- Source: pre-existing local
- Key tags: `K:Haste`; `T:Mode$ Attacks | ValidCard$ Card.Self | Execute$ TrigDig`; `SVar:TrigDig:DB$ Dig | Defined$ TriggeredDefendingPlayer | DigNum$ 1 | Reveal$ True | ChangeNum$ All | ChangeValid$ Land | LibraryPosition2$ 0`
## Engine work
- **Fix 1 — Defined$ TriggeredDefendingPlayer** (`src/parse.cpp` Defined$ branch, `src/systems/state_manager_triggers.cpp` trigger-fire binding, `src/components/ability.h` new `defined_triggered_defending_player` flag): the `Defined$` parse had no `TriggeredDefendingPlayer` case, so the Dig had no bound player and would have acted on the controller's library. Added the parse flag and, at CREATURE_ATTACKED fire time, bind the defending player as `trigger_ab.target`. The event carries `PLAYER` = the attacker's controller (active player); in a two-player game the defender is that player's opponent (the non-active player). The Dig handler already reads a player `ab.target` as the library owner (the fateseal path).
- **Fix 2 — ChangeNum$ All** (`src/parse.cpp` ChangeNum branch, `src/effects/effect_dig.cpp`, `src/components/ability.h` new `change_num_all` flag): `ChangeNum$ All` was previously misparsed as an SVar amount, leaving the take at the default 1 with an interactive DIG_CHOICE prompt. Added a parse case setting `change_num_all`; in the Dig handler `take_count` becomes `matching.size()` and the pick loop auto-takes every matching card (no player choice / DIG_CHOICE prompt). `LibraryPosition2$ 0` keeps an unchosen (nonland) card on top.
- CR: 508.2 (attack-declaration triggers), 509.1 / two-player defending player, 701 reveal
- Mechanics added (general): `Defined$ TriggeredDefendingPlayer` binding for attack triggers; `ChangeNum$ All` mandatory auto-take for Dig
## Behavioral decisions
- `Reveal$ True` is a cosmetic tag (the card is public to both players); it is ignored — the effect already moves the land to the defender's hand. Because the take is automatic and the resolving "looker" is the ability's controller (attacker), a nonland that stays on top is not additionally surfaced to the defender in the observation, which is acceptable (obs-only, not a rules deviation).
## Tests
- Isolation (test_harness), A attacks with a preset Goblin Guide (haste):
  - Land (Mountain) on top of B's library → "Player B looks at the top 1 card of their library" and "Player B puts Mountain into hand" — automatically, NO DIG_CHOICE prompt. B's hand 7→8 (later 9 after draw), B's library dropped by the drawn card. Damage (2) still dealt.
  - Nonland (Lightning Bolt) on top → "Player B puts 1 card on the top of their library" — revealed, stays on top, not drawn.
  - Confirmed the DEFENDER's (B's) library is affected while A attacks, not the attacker's.
- CI gate: pygen,vocab,smoke — 0 errors, no draws
## Result
implemented
