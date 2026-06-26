# Tormod's Crypt (vocab index 149)

## Oracle text
{T}, Sacrifice Tormod's Crypt: Exile all cards from target player's graveyard.

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/t/tormods_crypt.txt`).
Key tags:
- `A:AB$ ChangeZoneAll | Cost$ T Sac<1/CARDNAME> | Origin$ Graveyard | Destination$ Exile | ValidTgts$ Player | ChangeType$ Card`

## Engine work
None new — covered.
- `Cost$ T Sac<1/CARDNAME>` (tap + sacrifice self): the same activated-ability cost handled
  for Mishra's Bauble / Ghost Quarter.
- `AB$ ChangeZoneAll | Origin$ Graveyard | Destination$ Exile | ValidTgts$ Player |
  ChangeType$ Card` → `EffectKind::ChangeZoneAll` (`src/effects/effect_change_zone_all.cpp`).
  When the ability targets a player, the handler operates on the targeted player's zones
  (lines 35-37), collects every card in their graveyard (lines 56-62), and moves them to
  exile (EXILE destination, line 95). This is the Endurance template (also targets a
  player's graveyard).

## Behavioral decisions
None new. Exiles only the *targeted* player's graveyard (CR 608 / targeting). Because the
ability uses the stack, a graveyard is only exiled with the contents it has at resolution
(verified — a card that hasn't yet died is not exiled).

## Tests
- Scenario: Player A controls Tormod's Crypt; Player B's Birds of Paradise is killed by
  Lightning Bolt (→ B's graveyard). After the Bolt resolves, A activates Tormod's Crypt
  targeting Player B. Observed: "Player A sacrifices Tormod's Crypt", ability on stack
  targeting Player B, "Player B moves Birds of Paradise to exile" / "Player B moves 1
  card(s) to exile". Result: pass — the target player's graveyard is exiled. (Driven with
  positional `--actions` because a seat-keyed `--play` sequencing quirk auto-passed the
  activation; the engine behavior is correct.)

## Result
Implemented (registration only; mechanic pre-existing via the ChangeZoneAll
graveyard→exile player-targeting path, proven by Endurance).
