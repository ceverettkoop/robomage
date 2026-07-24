# Triumph of Saint Katherine

Vocab index **350**. Introduces the **Miracle** keyword (CR 702.94) as a general,
reusable mechanic, and completes the self-referential graveyard-recursion (a
"Descend"-style pile) chain used by its dies trigger.

## Oracle text

```
Triumph of Saint Katherine — {4}{W}
Creature — Human Warrior   5/5
Lifelink
Praesidium Protectiva — When Triumph of Saint Katherine is put into your
graveyard from the battlefield, exile it and the top six cards of your library
in a face-down pile. If you do, shuffle that pile and put it back on top of your
library.
Miracle {1}{W} (You may cast this card for its miracle cost when you draw it if
it's the first card you drew this turn.)
```

## Forge script

Source: pre-existing local script `bin/resources/cardsfolder/t/triumph_of_saint_katherine.txt`
(not fetched; DO NOT edit card scripts). Key tags:

- `K:Lifelink` — already supported.
- `K:Miracle:1 W` — the new keyword; the miracle alternative cost `{1}{W}`.
- `T:Mode$ ChangesZone | Origin$ Battlefield | Destination$ Graveyard | ValidCard$ Card.Self+YouOwn | Execute$ TrigExile`
  — the dies trigger, chaining:
  - `TrigExile: DB$ ChangeZone | Defined$ TriggeredNewCardLKICopy | Origin$ Graveyard | Destination$ Exile | RememberChanged$ True | ExileFaceDown$ True`
  - `DBExileTopCard: DB$ Dig | DigNum$ 6 | ChangeNum$ All | DestinationZone$ Exile | RememberChanged$ True | ExileFaceDown$ True`
  - `DBChangeLibrary: DB$ ChangeZoneAll | ConditionCheckSVar$ RememberedSize | ConditionSVarCompare$ GE7 | ChangeType$ Card.IsRemembered | Origin$ Exile | Destination$ Library | LibraryPosition$ 0 | RandomOrder$ True`
  - `DBCleanup: DB$ Cleanup | ClearRemembered$ True`
  - `SVar:RememberedSize:Count$RememberedSize`

`ExileFaceDown$ True` is **ignored as cosmetic** here (per the "ignore an
irrelevant tag" allowance): the pile goes to exile and is immediately shuffled
back onto the library, so face-down status is never engine-visible. No face-down
tracking was built.

## Engine work

### Mechanics added (general): miracle

Miracle is modeled as an alternative casting cost gated by a per-turn "miracle
window", reusing the shared `AltCost` cast path (like Impending/Spectacle):

- `src/components/carddata.h` — `AltCost::is_miracle` flag.
- `src/parse.cpp` — `K:Miracle:<cost>` parses the cost into `card.alt_cost`
  (mana portion) with `is_miracle = true`.
- `src/classes/game.h` — `Game::miracle_window` (set of card entities currently
  castable for their miracle cost).
- `src/systems/orderer.cpp` (`perform_draw`) — the first-draw gate: when a player
  draws a card and `cards_drawn_this_turn.size() == 1` (i.e. it is the first card
  drawn this turn) and the card `is_miracle`, the card is revealed and inserted
  into `miracle_window`.
- `src/systems/state_manager_actions.cpp` (`can_afford_alt`) — the `is_miracle`
  alt cost is offered **only** while the card entity sits in `miracle_window`;
  mana affordability is then checked by the shared alt-cost path. The offered
  action is labeled `Cast <name> (miracle)`.
- `src/classes/game.cpp` (cleanup) — `miracle_window.clear()` each cleanup, so the
  window lapses at end of turn (CR 702.94's one-turn opportunity).

### What was reused / completed for the dies-recursion chain

The recursion was intended to be expressible from existing effects, but three
general pieces were missing and were implemented (not retagged):

- `src/parse.cpp` — `Defined$ TriggeredNewCardLKICopy` / `TriggeredNewCard` /
  `TriggeredCard(LKICopy)` now bind to the ability's own source (`defined_self`).
  For a `Card.Self` ChangesZone trigger the object that changed zones IS the
  source, so the self-exile moves the real card (now in the graveyard) to exile.
  Previously the token was unrecognized and the move fell through to a graveyard
  *search* ("choose a card / fail to find").
- `src/effects/effect_change_zone.cpp` — the `defined_self` mover now honors
  `RememberChanged$ True`, so the self-exiled card is added to
  `remembered_entities` (counted by the GE7 shuffle-back condition).
- `src/effects/effect_change_zone_all.cpp` — (a) `Origin$ Exile` is now a searched
  zone, and (b) a positive `ChangeType$ Card.IsRemembered` filter restricts the
  move to the remembered pile only. Together these let the shuffle-back move the
  7-card pile (self + 6) from exile to the top of the library.
- `src/components/ability.cpp` — `Count$RememberedSize` (and bare
  `RememberedSize`) now evaluates to `remembered_entities.size()`, so the
  `ConditionCheckSVar$ RememberedSize GE7` gate works (previously it evaluated to
  0 and the shuffle-back never fired).
- `src/effects/effect_dig.cpp` — cosmetic: the Dig destination log said "into
  hand" for any non-library/non-battlefield destination; it now names exile /
  graveyard correctly.

`Dig` already honored `DestinationZone$ Exile` and `RememberChanged$ True`; the
Lifelink keyword and the `ConditionCheckSVar` sub-ability gate were already
supported.

CR references: **702.94** (Miracle), 120 (drawing / first card drawn),
601.2f/118.9 (alternative costs), 603.6e (ChangesZone triggers).

## Behavioral decisions

- **First-card-drawn tracking** reuses the existing `Player::cards_drawn_this_turn`
  vector (already reset each turn), so no new per-turn counter was added — the
  first draw of the turn is exactly `size() == 1` at the point of the draw.
- **Reveal / cast-window modeling.** The engine auto-reveals the miracle card
  (a game-log line, and the card becomes a public reveal) and opens the window for
  the remainder of the turn rather than implementing a literal one-shot triggered
  ability that must be cast on the very next priority. The alt cast still respects
  normal timing (a sorcery-speed creature like Triumph is castable in the main
  phase), and if the window is not used the card stays in hand, castable normally
  for `{4}{W}` later. This is a faithful, simpler model of the miracle
  opportunity; it is optional and general over any Miracle card.

## Tests (isolation via `train/test_harness.py`)

1. **Miracle on first draw.** A's opening hand is 7 fillers so Triumph (top of
   library) is drawn on A's draw step as the *first* card drawn that turn. Result:
   `Player A reveals Triumph of Saint Katherine for its miracle cost`, the menu
   offers `Cast Triumph of Saint Katherine (miracle)`, A pays `{1}{W}` (2 Plains),
   and it enters as a 5/5 Lifelink. (Only the miracle cast is offered — with 2
   lands the normal `{4}{W}` is unaffordable.)
2. **Not the first draw → no window; normal cast works.** A casts Brainstorm; its
   *first* draw is a filler and Triumph is the *second* draw, so **no** miracle
   reveal/window appears. With 6 lands A then casts Triumph as the normal
   `Cast Triumph of Saint Katherine` for `{4}{W}` and it enters.
3. **Dies-recursion.** Triumph preset on A's battlefield is killed by B's Dismember
   (-5/-5). The dies trigger runs cleanly: `is moved to exile`, Dig exiles the top
   6, then `moves 7 card(s) to the top of their library` (RememberedSize = 7
   satisfies GE7), Cleanup clears remembered. No non-fatal error.

CI gate: `train/.venv/bin/python train/ci_check.py --tier pygen,vocab,smoke
--smoke-games 1` — 0 errors, no draws.

## Result

Implemented.
