# Pithing Needle  (vocab index 229)
## Oracle text
As Pithing Needle enters, choose a card name.
Activated abilities of sources with the chosen name can't be activated unless they're mana abilities.
## Forge script
- Source: pre-existing local
- Key tags:
  - `K:ETBReplacement:Other:DBNameCard`
  - `SVar:DBNameCard:DB$ NameCard | Defined$ You`
  - `S:Mode$ CantBeActivated | ValidCard$ Card.NamedCard | ValidSA$ Activated.!ManaAbility`
## Engine work
- none — fully covered by existing handlers
- Mechanics:
  - ETB name-a-card: `K:ETBReplacement:Other:DBNameCard` → `DB$ NameCard` (`effects/effect_name_card.cpp`, `src/name_card_choices.cpp`)
  - `CantBeActivated` static gated on `Card.NamedCard`, excluding mana abilities (`ValidSA$ Activated.!ManaAbility`): static handling in `src/parse.cpp` (`match_named_card` / CantBeActivated)
## Behavioral decisions (made in lieu of asking a human)
- none — behavior unambiguous (covered card)
## Tests
- Isolation: skipped — mechanics already proven by Disruptor Flute (ETB NameCard + CantBeActivated on the named card, excluding mana abilities)
- Regression: skipped (verify_skip)
## Result
implemented (verification skipped — proven by Disruptor Flute)
