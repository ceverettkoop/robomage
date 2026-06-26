# Stoneforge Mystic  (vocab index 143)

## Oracle text
When Stoneforge Mystic enters, you may search your library for an Equipment card,
reveal it, put it into your hand, then shuffle.
{1}{W}, {T}: You may put an Equipment card from your hand onto the battlefield.

## Forge script  (Source: pre-existing local; Key tags)
- `T:Mode$ ChangesZone | Destination$ Battlefield | OptionalDecider$ You |
  Execute$ TrigChange` — ETB optional trigger.
- `SVar:TrigChange:DB$ ChangeZone | Origin$ Library | Destination$ Hand |
  ChangeType$ Card.Equipment | ChangeNum$ 1 | ShuffleNonMandatory$ True`.
- `A:AB$ ChangeZone | Cost$ 1 W T | Origin$ Hand | Destination$ Battlefield |
  ChangeType$ Equipment | ChangeNum$ 1` — activated put-onto-battlefield.

## Engine work  (none — covered by existing handlers)
- **ETB optional trigger** (`OptionalDecider$ You`): parsed at `src/parse.cpp:1567`,
  resolved as a "you may" decline/accept prompt in `src/components/ability.cpp:877`.
- **Library tutor with type filter** (`ChangeZone` Library→Hand, `ChangeType$ Card.Equipment`):
  `search_zone` in `src/components/ability.cpp` filters by the Equipment subtype and offers
  fail-to-find; a Library origin auto-shuffles afterward (`src/effects/effect_change_zone.cpp:208`).
  `ShuffleNonMandatory$ True` is the only unrecognized param — cosmetic, since the shuffle is
  already performed (matches the "Adding a New Card" rule that an irrelevant tag may be ignored
  when the behavior is inferred from the others).
- **Hand→battlefield put** (`ChangeZone` Hand→Battlefield, `ChangeType$ Equipment`, `Cost$ 1 W T`):
  same `ChangeZone` handler; the `{1}{W}, {T}` cost is the standard mana+tap activation cost.

## Behavioral decisions  (none / CR)
- ETB is optional ("you may") — modeled with an explicit Decline/Accept prompt.
- The library search reveals only Equipment cards (the type filter excludes lands/others);
  declining/failing to find is offered (CR 701.19 — search can fail to find).

## Tests
- ETB tutor (Cori-Steel Cutter shuffled into the library): accepted the optional trigger,
  "Searching Player A's library: 1: Cori-Steel Cutter" (only the Equipment offered), "Player A
  shuffles their library", Cori placed into hand. Pass.
- Activated ability: with Stoneforge untapped and Cori-Steel Cutter in hand, activated
  {1}{W}, {T} → chose Cori → "Player A puts Cori-Steel Cutter to the battlefield"
  (Stoneforge and two Plains tapped). Pass.

## Result
Done. Covered card; both the Equipment tutor and the put-onto-battlefield ability work
with no engine changes (ShuffleNonMandatory$ is a benign unrecognized-param warning).
