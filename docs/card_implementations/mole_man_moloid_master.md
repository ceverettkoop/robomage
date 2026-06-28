# Mole Man, Moloid Master  (vocab index 254)

## Oracle text
You may play lands from your graveyard.
Landfall — Whenever a land you control enters, create a 1/1 green Minion creature token named Moloid
with "Whenever this token attacks, you may mill a card."

(Legendary Creature — Human Villain, mana cost {2}{G}, 1/1.)

## Forge script
- Source: pre-existing local — `bin/resources/cardsfolder/m/mole_man_moloid_master.txt`
- Token script: fetched `bin/resources/tokenscripts/moloid.txt` (Card-Forge).
- Key tags:
  - `S:Mode$ Continuous | Affected$ Land.YouOwn | MayPlay$ True | AffectedZone$ Graveyard` — you may
    play your lands from your graveyard.
  - `T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Land.YouCtrl |
    TriggerZones$ Battlefield | Execute$ TrigToken` — landfall.
  - `SVar:TrigToken:DB$ Token | TokenScript$ moloid | TokenOwner$ You`.

## Engine work
- none — fully covered by existing handlers:
  - `MayPlay$ True` + `AffectedZone$ Graveyard`: `src/parse.cpp` sets `sa.may_play_from_graveyard`;
    enforced at the play-land step.
  - Landfall `Mode$ ChangesZone` → battlefield trigger: `src/parse.cpp` /
    `src/systems/state_manager_triggers.cpp`.
  - `DB$ Token | TokenScript$`: `src/effects/effect_token.cpp`.

## Behavioral decisions
- None novel — both clauses are standard MayPlay-from-zone and landfall-token patterns.

## Tests (test_harness, via stacked temp deck because the name contains a comma)
- **Landfall:** cast Mole Man (3 battlefield Forests), then play a Forest from hand → "Mole Man,
  Moloid Master triggered" → "Token created: 1/1 Moloid". PASS.
- **Play land from graveyard:** with an Island in A's graveyard, after Mole Man is in play the menu
  offered "Play Island (from graveyard)" → "Player A played Island", Island left the graveyard and
  entered the battlefield, and landfall created another Moloid. PASS.
- Regression (`--scripted`, seeds 1-3): all decisive, no draws, no non-fatal errors.

## Result
implemented
