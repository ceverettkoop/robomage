# Baleful Strix  (vocab index 244)

## Oracle text
Flying, deathtouch
When Baleful Strix enters, draw a card.

(Artifact Creature — Bird, mana cost {U}{B}, 1/1.)

## Forge script
- Source: pre-existing local — `bin/resources/cardsfolder/b/baleful_strix.txt`
- Key tags:
  - `K:Flying`, `K:Deathtouch` — vanilla evasion / lethal-damage keywords.
  - `T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self | Execute$ TrigDraw` — ETB trigger.
  - `SVar:TrigDraw:DB$ Draw | Defined$ You | NumCards$ 1` — controller draws one card.

## Engine work
- none — fully covered by existing handlers. The card is a composition of two
  independently-proven mechanics:
  - ETB-self ChangesZone trigger dispatch: `src/systems/state_manager_triggers.cpp:283-290`
    (parsed at `src/parse.cpp:1941-1996`); `Card.Self` enforced via `trigger_only_self`.
  - `DB$ Draw` for the controller: `src/effects/effect_draw.cpp:13-24`.
  - Deathtouch lethal: `src/components/damage.cpp:37-39` + `src/systems/state_manager.cpp:140-141`.
  - Flying block restriction: `src/action_processor.cpp:793-829`.

## Behavioral decisions
- none — behavior unambiguous.

## Tests
- Isolation (test_harness): cast Baleful Strix with U/B available; ETB resolved
  "Resolving ability (category: Draw, amount: 1) / Player A draws Forest" — controller
  drew exactly one card; the 1/1 entered the battlefield. PASS.
- Keywords Flying / Deathtouch are vanilla and already exercised by pre-existing
  shipping cards (Flying Men idx 18, Air Elemental idx 21 for Flying; Barrowgoyf idx 87
  for Deathtouch), so combat-keyword behavior is verify-skipped.

## Result
implemented
