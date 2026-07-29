# Mox Diamond

## Oracle text
If Mox Diamond would enter the battlefield, you may discard a land card instead. If you do, put Mox Diamond onto the battlefield. If you don't, put it into its owner's graveyard.
{T}: Add one mana of any color. (0-cost artifact)

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/m/mox_diamond.txt`).

Key tags:
- `A:AB$ Mana | Cost$ T | Produced$ Any` — the any-color mana ability (already supported).
- `R:Event$ Moved | Destination$ Battlefield | ValidCard$ Card.Self | ReplaceWith$ PayBeforeETB`
  — the self-replacement.
- `SVar:PayBeforeETB:DB$ Discard | DiscardValid$ Land | Optional$ True | SubAbility$ MoveToGraveyard`
  and its chain (`MoveToGraveyard` -> `Destination$ Graveyard` when nothing was discarded,
  `MoveToBattlefield` when it was). `PlayBeforeLandDrop:TRUE` is an AI hint (ignored).

## Engine work
- **Mechanics added (general): `etb-discard-else-graveyard`** — a self-replacement on the
  "would enter the battlefield" event that offers an OPTIONAL additional cost (discard a matching
  card); if paid the permanent enters, otherwise it is redirected to its owner's graveyard.
- `src/components/effect.h` — new `Effect::Replacement::Kind DISCARD_ELSE_GRAVEYARD` plus a
  `discard_else_filter` field (the type filter of the discardable card, e.g. "Land").
- `src/parse.cpp` (`parse_replacement_effects`) — recognizes the pattern by WALKING the
  `ReplaceWith$` SVar's `SubAbility$` chain (not by retagging): an optional `DB$ Discard` with a
  `DiscardValid$` filter and an "else `Destination$ Graveyard`" move. Honors the script's real tags.
- `src/systems/replacement_effects.cpp` — `collect()` (MOVE_TO_ZONE, destination Battlefield)
  detects the self-replacement on the entering card itself; `apply_one()` (new
  `DISCARD_ELSE_GRAVEYARD` case) offers the affected player a blocking menu of their hand cards
  matching the filter plus a decline option. A chosen card is recorded on
  `ReplacementEvent::pending_discard` (destination stays Battlefield); declining, or holding no
  matching card, redirects `ev.destination` to the graveyard.
- `src/systems/orderer.cpp` (`add_to_zone`) — performs the recorded discard after dispatch (the
  dispatcher has no orderer); the permanent then enters normally.
- The blocking decision is consistent with the documented residual MOVE_TO_ZONE replacement
  prompts (`choose_one`), since the dispatch fires inside `add_to_zone`.

CR: 614.1a (a replacement effect that replaces "would enter" with a different outcome), 614.15
(self-replacement), 616.1 (the affected player applies it).

## Behavioral decisions
- Implemented the discard as an optional additional cost resolved at the replacement, redirecting
  to the graveyard when declined/unpayable — rather than letting the card enter and then bouncing
  it (which would wrongly fire enters-the-battlefield events). This is reusable for the whole
  Chrome Mox / Mox Diamond family (any `DiscardValid$` filter).
- The discard menu lists every matching hand card (all lands), so the player picks WHICH land.

## Tests (isolation)
Harness — A hand Mox Diamond (+ a land):
- Discard a land: A casts Mox Diamond -> offered to discard; picks Forest -> Forest to graveyard,
  Mox Diamond enters the battlefield. A then cast Lightning Bolt using "activated Mox Diamond for
  1(R)" (3 damage, B to 17) -> confirms the any-color mana ability. PASS.
- Decline: A picks "Don't discard" -> Mox Diamond put into its owner's graveyard, does NOT enter.
  PASS.
- No land in hand: A casts Mox Diamond with no land -> no prompt; "has no Land to discard and is
  put into its owner's graveyard." PASS.
- CI gate: `make check` tiers pygen/vocab/smoke.

## Result
Implemented.
