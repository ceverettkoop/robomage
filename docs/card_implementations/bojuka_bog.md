# Bojuka Bog (vocab index 156)

## Oracle text
Bojuka Bog enters tapped.
When Bojuka Bog enters, exile all cards from target player's graveyard.
{T}: Add {B}.

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/b/bojuka_bog.txt`)

Key tags:
- `R:Event$ Moved | Destination$ Battlefield | ReplaceWith$ ETBTapped` + `SVar:ETBTapped:DB$ Tap | ETB$ True` — enters tapped.
- `A:AB$ Mana | Cost$ T | Produced$ B` — {T}: Add {B}.
- `T:Mode$ ChangesZone | Destination$ Battlefield | ValidCard$ Card.Self | Execute$ TrigExile` + `SVar:TrigExile:DB$ ChangeZoneAll | ValidTgts$ Player | Origin$ Graveyard | Destination$ Exile | ChangeType$ Card | IsCurse$ True` — ETB exile target player's graveyard.

## Engine work
None — covered:
- ETBTapped replacement parsed in `src/parse.cpp` (`replace_with_etb_tapped` →
  `pending_enters_tapped`, applied in replacement_effects.cpp), proven by Undercity Sewers.
- `AB$ Mana | Produced$ B` mana ability.
- ETB `ChangesZone`→Battlefield trigger firing `ChangeZoneAll` with a `ValidTgts$ Player`
  target, Graveyard→Exile (effect_change_zone_all.cpp handles the player-target and
  EXILE destination).
- `IsCurse$ True` is a cosmetic AI hint; the parser emits the expected (pre-existing)
  "Unrecognized ability param: IsCurse$ True" cosmetic warning and ignores it.

## Behavioral decisions
None — exile-all-from-target-graveyard is the standard ChangeZoneAll behavior (CR 614 for
enters-tapped replacement; the ETB trigger uses the targeting framework).

## Tests
Verified via test_harness (A plays Bojuka Bog; B's graveyard seeded with Lightning Bolt +
Brainstorm):
- Enters tapped: board shows `Self BF: Bojuka Bog (T)` after "Bojuka Bog enters tapped." PASS.
- ETB trigger: target menu offered Player A / Player B; targeting Player B →
  "Player B moves Lightning Bolt to exile", "Player B moves Brainstorm to exile",
  "Player B moves 2 card(s) to exile" — B's graveyard fully exiled. PASS.

## Result
Done — registered in vocab, clean build, enters-tapped + ETB graveyard-exile verified.
