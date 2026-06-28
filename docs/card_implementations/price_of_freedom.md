# Price of Freedom  (vocab index 259)

## Oracle text
Destroy target artifact or land an opponent controls. Its controller may search their library for a
basic land card, put it onto the battlefield tapped, then shuffle.
Draw a card.

(Sorcery — Lesson, mana cost {1}{R}.)

## Forge script
- Source: pre-existing local — `bin/resources/cardsfolder/p/price_of_freedom.txt`
- Key tags:
  - `A:SP$ Destroy | ValidTgts$ Artifact.OppCtrl,Land.OppCtrl | SubAbility$ DBChangeZone`.
  - `SVar:DBChangeZone:DB$ ChangeZone | Optional$ True | Origin$ Library | Destination$ Battlefield |
    Tapped$ True | ChangeType$ Land.Basic | DefinedPlayer$ TargetedController |
    ShuffleNonMandatory$ True | SubAbility$ DBDraw`.
  - `SVar:DBDraw:DB$ Draw`.

## Engine work
- none — fully covered by existing handlers:
  - `SP$ Destroy` of a targeted opponent permanent: `src/effects/effect_destroy.cpp`.
  - `DefinedPlayer$ TargetedController` routes the search to the destroyed permanent's controller
    (CR 109.5), resolved via `last_known_controller` in `src/effects/effect_change_zone.cpp`.
  - `Tapped$ True` on a Library→Battlefield ChangeZone (the basic enters tapped):
    `src/effects/effect_change_zone.cpp`.
  - SubAbility chain `Destroy → ChangeZone → Draw`: `src/components/ability.cpp`. The unspecified
    `DB$ Draw` is drawn by the spell's controller (the caster).
- `ShuffleNonMandatory$` is a cosmetic param (the search-based ChangeZone already shuffles after a
  library fetch), added to the parser's ignored set (`src/parse.cpp`).

## Behavioral decisions
- "Its controller may search …" → the **target's controller** (opponent) does the optional search
  and gets the tapped basic; the unconditional "Draw a card" is taken by the **caster**. Both
  confirmed in the transcript.

## Tests (test_harness)
- A casts Price of Freedom targeting a Forest B controls → "Forest is destroyed", "Searching Player
  B's library", B put a Forest onto the battlefield, "Forest enters tapped.", "Player B shuffles
  their library", then "Player A draws Mountain" (caster draws). PASS.
- Regression (`--scripted`, seeds 1-3): all decisive, no draws, no non-fatal errors.

## Result
implemented
