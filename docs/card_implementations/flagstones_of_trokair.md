# Flagstones of Trokair  (vocab index 133)

## Oracle text
{T}: Add {W}.
When Flagstones of Trokair is put into a graveyard from the battlefield, you may search your
library for a Plains card, put it onto the battlefield tapped, then shuffle.

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/f/flagstones_of_trokair.txt`
- Key tags:
  - `A:AB$ Mana | Cost$ T | Produced$ W` — the white mana ability (normalized to `AddMana`).
  - `T:Mode$ ChangesZone | Origin$ Battlefield | Destination$ Graveyard | ValidCard$ Card.Self |
    Execute$ TrigChange | OptionalDecider$ TriggeredCardController` — a leaves-the-battlefield
    self-trigger ("when ~ is put into a graveyard from the battlefield").
  - `SVar:TrigChange:DB$ ChangeZone | Origin$ Library | Destination$ Battlefield | Tapped$ True |
    ChangeType$ Card.Plains | ChangeNum$ 1 | ShuffleNonMandatory$ True` — search the library for a
    Plains card, put it onto the battlefield tapped, then shuffle.

## Engine work
Three general mechanics were added/extended, each keyed on the tag's intended meaning (not a
card-specific shortcut):

1. **Leaves-the-battlefield self-trigger scan** (`src/systems/state_manager_triggers.cpp`).
   The existing trigger scan only inspects battlefield permanents. A `ChangesZone` trigger that
   watches its own source moving *off* the battlefield (Origin$ Battlefield → Destination$
   Graveyard, ValidCard$ Card.Self) can never be caught that way: by the time triggers are checked
   the source has already left the battlefield and lost its `Permanent` component (CR 603.6b — the
   trigger condition is checked against the game state immediately *before* the event; CR 603.10
   leaves-the-battlefield look-back). Added a dedicated pass that, for every `CARD_CHANGED_ZONE`
   event leaving the battlefield, re-scans the changing card's own `CardData` abilities for a
   `trigger_only_self` `ChangesZone` trigger whose origin/destination filter matches, and queues
   it. `CardData` (and the card's last controller, persisted on its `Zone` component) survive the
   move, so the source is fully recoverable. This is general: it will fire any "when ~ is put into
   a graveyard / leaves the battlefield" self-trigger, not just Flagstones.

2. **`OptionalDecider$ TriggeredCardController` → optional trigger** (`src/parse.cpp`). The parser
   previously only recognised `OptionalDecider$ You` as making a trigger optional ("you may ..."),
   leaving `TriggeredCardController` (the source's controller — the same player) mandatory. Broadened
   the check to any named decider containing `You`/`Controller`, so the controller is prompted to
   accept/decline at resolution (handled by the existing `Ability::resolve` `trigger_optional` path).

3. **`Card.<subtype>` and `Card`/`Permanent` wildcard filter heads** (`src/components/ability.cpp`,
   `matches_filter_spec`). The library-search filter `Card.Plains` means "any card with subtype
   Plains". The extended-filter matcher treated the head (`Card`) as a required *type name* (which
   no card has) and the dot-qualifier (`Plains`) only as a color/Basic qualifier, so a Plains card
   never matched. Fixed: `Card`/`Permanent` are now wildcard heads that match any card, and an
   unrecognised dot-qualifier is matched as a subtype name against the card's type list. General —
   benefits any `Card.<subtype>` / `Permanent.<subtype>` search filter.

The rest is covered by existing handlers:
- The `AddMana` ability is handled at activation (never goes on the stack).
- The `ChangeZone` effect already supports a Library→Battlefield search with `Tapped$ True`
  (`enters_tapped` → `pending_enters_tapped`) and shuffles the library after a library search.
- `ChangeNum$ 1` sets the search count to 1; `Mandatory$` is unset, so the search is non-mandatory
  ("you may search" — a Fail-to-find option is offered, CR 701.18).
- `ShuffleNonMandatory$ True` is cosmetic here (the engine always shuffles after a library search,
  matching the card); it produces a harmless `WARNING: Unrecognized ability param` line.

## Behavioral decisions (made in lieu of asking a human)
- The trigger is a *leaves-the-battlefield* trigger: it must look back at the game state just
  before Flagstones left the battlefield to determine that it triggered (CR 603.6b / 603.10).
  Implemented exactly as a look-back over the zone-change event, not by leaving a dead Permanent
  on the battlefield.
- "you may search" → the whole trigger is optional (controller may decline), AND the search itself
  is non-mandatory (may fail to find). Both decline paths verified.
- The fetched Plains enters the battlefield **tapped** (Tapped$ True), unlike a fetchland's
  untapped land. Verified the searched Plains shows `(T)` on the battlefield.
- The library is shuffled after the search regardless of whether a card was found, matching
  "then shuffle".

## Tests
- Mana ability (test_harness): Flagstones in play, cast Swords to Plowshares ({W}) using its
  white mana → Swords resolves, Grizzly Bears exiled, opponent gains 2 life. Confirms `{T}: Add
  {W}`. PASS.
- Isolation — LTB trigger (test_harness): A controls Flagstones, library has a Plains; B's
  Wasteland destroys A's Flagstones. "Flagstones of Trokair is destroyed" → "Flagstones of
  Trokair triggered" (the LTB self-trigger fires from the graveyard) → A accepts the optional
  trigger → search menu offers Plains → "Player A shuffles their library", "Player A puts Plains
  to the battlefield", "Plains enters tapped." Final board: `Plains (T)` on A's battlefield,
  Flagstones in A's graveyard. PASS.
- Optional decline (test_harness): when A auto-passes the trigger's accept/decline, "Player A
  declines the optional triggered ability." and no Plains is fetched. PASS.
- Regression (test_harness `--scripted`, 6 games, seeds 1–6): deck `temp/flagstones_test_a`
  (4 Flagstones + 10 Plains + 4 Swords + 4 Wasteland + 14 Island + 4 Lightning Bolt) vs
  `temp/flagstones_test_b` (Gruul aggro). All 6 games finished with a decisive winner (Player B
  wins all six — deck A is a slow control shell with no threats vs aggro; this is balance, not an
  engine fault), no draws, no fatal/non-fatal errors, no assertions or tracebacks. Only the
  pre-existing cosmetic `WARNING: Unrecognized ability param: ShuffleNonMandatory$` line.

## Result
implemented
