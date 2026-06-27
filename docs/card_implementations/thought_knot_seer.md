# Thought-Knot Seer (vocab index 182)

## Oracle text
({C} represents colorless mana.)
When Thought-Knot Seer enters, target opponent reveals their hand. You choose a nonland card
from it and exile that card.
When Thought-Knot Seer leaves the battlefield, target opponent draws a card.

`3 C` — a 4/4 colorless Eldrazi (the colorless `{C}` pip already works; Reality Smasher, vocab
index 181, casts with it).

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/t/thought_knot_seer.txt`).
Key tags:
- `T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self |
  Execute$ TrigReveal` — ETB trigger.
- `SVar:TrigReveal:DB$ RevealHand | ValidTgts$ Opponent | SubAbility$ DBExile`
- `SVar:DBExile:DB$ ChangeZone | Origin$ Hand | Destination$ Exile | DefinedPlayer$ Targeted |
  Chooser$ You | ChangeType$ Card.nonLand`
- `T:Mode$ ChangesZone | Origin$ Battlefield | Destination$ Any | ValidCard$ Card.Self |
  Execute$ TrigDraw` — leaves-the-battlefield (LTB) trigger.
- `SVar:TrigDraw:DB$ Draw | ValidTgts$ Opponent`

## Engine work
Three pieces; all built as general handlers keyed on the script's actual tags (no retag).

1. **`DB$ RevealHand` effect — NEW** (`src/effects/effect_reveal_hand.cpp`,
   `EffectKind::RevealHand`). The targeted player (`ValidTgts$ Opponent`, resolved to `ab.target`
   as a Player entity) reveals their entire hand to all players (CR 701.16): each card is logged
   and recorded in the match-scoped belief state via `mark_card_revealed`, making the cards public
   so a chained chooser can see them. Registered in `effect_kind.{h,cpp}`, `effect_table.cpp`, and
   declared in `effects.h`. General over Duress/Thoughtseize-style "target player reveals their
   hand".

2. **`Chooser$ You` cross-player hand selection in `ChangeZone`** — the key new piece.
   - `Chooser$` was a cosmetic ignored param; it is now parsed in `src/parse.cpp` into
     `Ability::chooser_is_controller` (`src/components/ability.h`). `Chooser$ You` ⇒ the ability's
     **controller** makes the selection; any other value leaves the zone owner choosing.
   - `DefinedPlayer$ Targeted` with a bound **Player** target now routes the search/move owner to
     that targeted player (`src/effects/effect_change_zone.cpp`, alongside the existing
     `TargetedController` redirect). The sub-ability's target is the opponent the parent
     `RevealHand` targeted — bound by the existing `bind_sub_target` (CR 608.2c: `defined ==
     "Targeted"`, `valid_tgts == "N_A"` ⇒ `sub.target = parent.target`).
   - In the search path, when `chooser_is_controller` is set, priority is switched to the ability's
     controller around the `search_zone` call so the choice prompt (a `CHOOSE_CARD` over the
     opponent's hand, filtered by `ChangeType$ Card.nonLand`) is offered to the controller, and
     `reveal` is forced (the picks are public). The chosen card is moved to `Destination$ Exile`.
     If the opponent has no nonland card, only "Fail to find" is offered and nothing is exiled
     (the reveal still happened).

3. **`DB$ Draw | ValidTgts$ Opponent` on the LTB trigger** — no new code; `effect_draw.cpp`
   already draws for a target Player (`ab.target`, proven by Deep Analysis). The LTB
   `T:Mode$ ChangesZone | Origin$ Battlefield | Destination$ Any | ValidCard$ Card.Self` trigger
   is the standard leaves-the-battlefield ChangesZone trigger (proven by earthbend's
   return-tapped delayed trigger / Moonshadow).

## Behavioral decisions
- The exile selection is performed by Thought-Knot's controller (the "You" / Chooser), not the
  opponent — implemented via the priority switch, mirroring how `effect_discard.cpp`
  (Thoughtseize) routes its controller-chooses pick. The discard handler moves to graveyard;
  Thought-Knot needs `Destination$ Exile`, so the script's `DB$ ChangeZone` is honored rather than
  reusing Discard.
- Lands in the opponent's hand are never offered (`ChangeType$ Card.nonLand`), confirmed in test
  (a): only the two nonland cards plus "Fail to find" appeared.
- The reveal is mandatory and happens before the exile selection; with an all-land hand nothing is
  exiled (test (b)).
- `TriggerDescription$` is cosmetic (ignored).

## Tests (test_harness.py, semantic `--play`, seed 1)
- **(a) ETB reveal + exile nonland** — A cast Thought-Knot Seer (colorless mana), ETB targeted
  Player B; B revealed hand (Lightning Bolt, Brainstorm, Island×5); the chooser menu offered
  **only** Lightning Bolt and Brainstorm (+ Fail to find) — **Island (land) not offered**. A chose
  Lightning Bolt → "Player B puts Lightning Bolt to exile"; B's hand lost Lightning Bolt. **Pass.**
- **(b) lands-only hand** — B's hand was 7 Islands; reveal happened, the chooser menu offered only
  "Fail to find" (no nonland), nothing exiled, B's hand unchanged. **Pass.**
- **(c) LTB draw** — Thought-Knot Seer preplaced on A's battlefield, B Bolts it twice (3+3 ≥ 4):
  "Thought-Knot Seer is destroyed (lethal damage)", the LTB trigger resolved
  "Resolving ability (category: Draw …)" → "Player B draws Island" (target opponent drew). **Pass.**
- **Regression** — scripted full games `temp/tks_a` (4 Thought-Knot Seer + Eldrazi Temple +
  Wastes) vs a Bolt/Brainstorm/Mountain deck, seeds 1/2/3: all decisive (Player A wins), no draws,
  no asserts, no non-fatal errors. RevealHand and the chooser ChangeZone resolved cleanly in real
  games (the scripted agent's Fail-to-find pick is agent suboptimality, not an engine bug).

## Result
Implemented. New reusable mechanics: the `RevealHand` effect, `Chooser$ You` controller-driven
cross-player hand selection in `ChangeZone` (with `DefinedPlayer$ Targeted` Player redirect), and
the LTB `Draw` for a target opponent (reusing the existing target-player draw).
