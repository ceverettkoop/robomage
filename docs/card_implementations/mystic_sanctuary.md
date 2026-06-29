# Mystic Sanctuary  (vocab index 255)

## Oracle text
({T}: Add {U}.)
Mystic Sanctuary enters tapped unless you control three or more other Islands.
When Mystic Sanctuary enters untapped, you may put target instant or sorcery card from your
graveyard on top of your library.

(Land — Island.)

## Forge script
- Source: pre-existing local — `bin/resources/cardsfolder/m/mystic_sanctuary.txt`
- Key tags:
  - `R:Event$ Moved | ValidCard$ Card.Self | Destination$ Battlefield | ReplaceWith$ LandTapped`
    with `SVar:LandTapped:DB$ Tap | Defined$ Self | ETB$ True | ConditionPresent$
    Island.YouCtrl+Other | ConditionCompare$ LT3` — enters tapped unless you control 3+ other Islands.
  - `T:Mode$ ChangesZone | Destination$ Battlefield | ValidCard$ Card.Self+untapped |
    OptionalDecider$ You | Execute$ TrigChange`.
  - `SVar:TrigChange:DB$ ChangeZone | ValidTgts$ Instant.YouOwn,Sorcery.YouOwn | Origin$ Graveyard |
    Destination$ Library`.

## Engine work
- none — fully covered by existing handlers:
  - Conditional enters-tapped replacement with `ConditionPresent$ ...+Other` / `ConditionCompare$
    LT3`: `src/systems/replacement_effects.cpp` (the entering permanent is excluded from the count,
    giving "+Other"; `compare_svar` evaluates `LT3`).
  - `Card.Self+untapped` enters-untapped trigger + `OptionalDecider$ You`: trigger system.
  - Targeted `DB$ ChangeZone` Graveyard→Library (top): `src/effects/effect_change_zone.cpp`.

## Behavioral decisions
- The trigger is optional (OptionalDecider) — the harness must Accept it to put a card on top.

## Tests (test_harness)
- **<3 other Islands:** with 1 Island in play, play Mystic Sanctuary → "Mystic Sanctuary enters
  tapped." PASS.
- **3+ other Islands:** with 3 Islands in play and a Lightning Bolt in A's graveyard, play Mystic
  Sanctuary → enters untapped; trigger targets Lightning Bolt, Accept → "Lightning Bolt is moved to
  top of library", and A draws Lightning Bolt next turn. PASS.
- Regression (`--scripted`, seeds 1-3): all decisive, no draws, no non-fatal errors.

## Result
implemented
