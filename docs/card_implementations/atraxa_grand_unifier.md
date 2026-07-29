# Atraxa, Grand Unifier

## Oracle text

Flying, vigilance, deathtouch, lifelink

When Atraxa, Grand Unifier enters, reveal the top ten cards of your library. For
each card type, you may put a card of that type from among the revealed cards into
your hand. Put the rest on the bottom of your library in a random order. (Artifact,
battle, creature, enchantment, instant, kindred, land, planeswalker, and sorcery are
card types.)

`{3}{G}{W}{U}{B}` — Legendary Creature — Phyrexian Angel — 7/7

Vocab index: **347** (`src/card_vocab.h`).

## Forge script

Source: pre-existing local script `bin/resources/cardsfolder/a/atraxa_grand_unifier.txt`
(not fetched/authored for this task; not modified). Key tags:

- `K:Flying / K:Vigilance / K:Deathtouch / K:Lifelink` — all already covered.
- ETB trigger `T:Mode$ ChangesZone | ... | Execute$ TrigReveal`.
- `TrigReveal: DB$ PeekAndReveal | PeekAmount$ 10 | Reveal$ True | ImprintRevealed$ True | SubAbility$ TrigRepeatTypes`
- `TrigRepeatTypes: DB$ RepeatEach | RepeatTypesFrom$ ValidLibrary Card.IsImprinted | RepeatSubAbility$ ChooseCard | SubAbility$ DBChangeZone`
- `ChooseCard: DB$ ChooseCard | Choices$ Card.ChosenType+YouOwn+IsImprinted | ChoiceZone$ Library | RememberChosen$ True`
- `DBChangeZone: DB$ ChangeZone | Origin$ Library | Destination$ Hand | Defined$ Remembered | SubAbility$ ShuffleRest`
- `ShuffleRest: DB$ ChangeZoneAll | Origin$ Library | Destination$ Library | LibraryPosition$ -1 | RandomOrder$ True | ChangeType$ Card.IsImprinted+!IsRemembered | SubAbility$ DBCleanup`
- `DBCleanup: DB$ Cleanup | ClearRemembered$ True | ClearImprinted$ True`

Note: the local script reveals **ten** cards and each pick is a **"you may"**, matching the
current Oracle text (the task brief quoted an older seven-card/mandatory printing). The
implementation follows the actual script.

## Engine work

### Mechanics added (general): `reveal-pick-one-of-each-type`

Three reusable pieces, keyed on the script's own tags (no retagging):

1. **Imprint the revealed set** — `DB$ PeekAndReveal | ImprintRevealed$ True`.
   `PeekParams::imprint_revealed` (`src/components/ability_params.h`); the handler
   (`src/effects/effect_peek_and_reveal.cpp`) reveals the top *PeekAmount* cards of the
   controller's library to all players (`mark_card_revealed` + log), leaves them in the
   library, and records them in the new `cur_game.imprinted_entities` set
   (`src/classes/game.h`). A chained `Card.IsImprinted` filter reads that set. CR 401
   (library ordering) / 701 (reveal).

2. **Loop per card type** — `DB$ RepeatEach | RepeatTypesFrom$ ...`.
   `Ability::repeat_types_from` gates a new path in `src/effects/effect_repeat_each.cpp`
   (`repeat_each_types`): enumerate the distinct card types (CR 300.1: Artifact, Battle,
   Creature, Enchantment, Instant, Kindred, Land, Planeswalker, Sorcery) present among the
   imprinted cards, in that canonical order, and for each set `cur_game.chosen_type` and
   resolve the `RepeatSubAbility$` body. The leading `Ability::repeat_sub_count` subabilities
   are the per-type body; the remaining are trailing `SubAbility$` links resolved **once**
   after the whole loop (here the `DBChangeZone → ShuffleRest → DBCleanup` chain). Fully
   suspendable — the type index / body-sub index / trailing-sub index persist in `RepeatRt`
   and each child resolves via `FrameCtx::resolve_child`, the same machinery the per-player
   `RepeatEach` (Price of Progress) uses.

3. **Choose one imprinted card of the current type** — `DB$ ChooseCard | Choices$
   Card.ChosenType+YouOwn+IsImprinted | RememberChosen$ True`.
   `Ability::choose_imprinted` gates a new path in `src/effects/effect_choose_card.cpp`:
   offer each imprinted card of `cur_game.chosen_type` owned by the controller (still in the
   library), plus a "put none" option (the Oracle "you may"); `RememberChosen$ True`
   (`Ability::remember_chosen`) appends the pick to `remembered_entities`.

The remaining links reuse existing effects:

- `DBChangeZone | Defined$ Remembered | Library→Hand` moves the chosen cards to hand
  (existing blanket-remembered path in `effect_change_zone.cpp`).
- `ShuffleRest | ChangeZoneAll | ChangeType$ Card.IsImprinted+!IsRemembered | RandomOrder$`
  bottoms the imprinted-but-not-taken cards in random order. A new `IsImprinted` filter in
  `effect_change_zone_all.cpp` restricts the move to `cur_game.imprinted_entities` so it
  bottoms only the revealed leftovers, not the whole library (`RandomOrder$`/bottom already
  existed for Emrakul/Endurance).
- `DBCleanup | ClearImprinted$ True` clears `cur_game.imprinted_entities`
  (`effect_cleanup.cpp` / `Ability::clear_imprinted`).

### Parsing (`src/parse.cpp`)

- `ImprintRevealed$` → `PeekParams::imprint_revealed` (parse hook in
  `effect_peek_and_reveal.cpp`).
- `RepeatTypesFrom$` → `repeat_types_from`; `RememberChosen$` → `remember_chosen`;
  `ClearImprinted$` → `clear_imprinted` (removed from the ignored-keys set; a harmless no-op
  for the Phelia-style redundant imprint whose set is empty).
- `Choices$ ...IsImprinted...` on a `ChooseCard` → `choose_imprinted`. Both parse loops (the
  top-level `parse_abilities` and the SVar loop `parse_svar_ability`) were taught to (a) route
  `RepeatSubAbility$`/`VoteSubAbility$` into `subabilities` and count `RepeatSubAbility$`, and
  (b) let a `ChooseCard`'s `Choices$` fall through to `apply_param_to_ability` instead of the
  charm-modal parser. `ChoiceZone$` is ignored (cosmetic — the imprinted cards stay in the
  library, which the handler already assumes).

## Behavioral decisions

- **"You may" is modeled as optional** — each per-type `ChooseCard` offers a "put no <type>
  card" option, matching the Oracle. Declining leaves that card among the revealed set to be
  bottomed.
- **Type-loop order** is the canonical CR 300.1 card-type order; only types actually present
  among the revealed cards produce a prompt.
- Two-player engine: no multiplayer concerns.

## Tests

Isolation test via `train/test_harness.py` (`--play` with seat keys; Atraxa cast on turn 1
so its ETB fires on entering — a battlefield preset would **not** fire the ETB). Library
stacked so the top ten hold a 6-type mix (Grizzly Bears = Creature, Lightning Bolt = Instant,
Duress = Sorcery, Island = Land, Mishra's Bauble = Artifact, Sylvan Library = Enchantment,
plus duplicate creature/instant/lands). `--hand-a` was padded to 7 cards so `--library-a`
stays the pure library top (a 1-card `--hand-a` otherwise pulls 6 library cards into the
opening hand).

Result: reveal logs all ten; one per-type "you may put" prompt fires for each of the 6
represented types; taking one of each puts exactly 6 cards into hand (Mishra's Bauble,
Grizzly Bears, Sylvan Library, Lightning Bolt, Island, Duress) and bottoms the remaining 4
(Mountain, 2nd Grizzly Bears, Forest, 2nd Lightning Bolt) — "moves 4 card(s) to the bottom".
Counts confirmed: cards to hand = distinct types among the ten; rest bottomed. A decline
variant (`desc:Put no Artifact`) verified the optional path (5 taken, Mishra's Bauble
bottomed) with no crash. No non-fatal errors.

CI gate: `train/ci_check.py --tier pygen,vocab,smoke` — 0 errors, no draws.

## Result

Implemented.
