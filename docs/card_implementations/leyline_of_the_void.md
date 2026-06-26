# Leyline of the Void  (vocab index 104)

## Oracle text
If Leyline of the Void is in your opening hand, you may begin the game with it on the battlefield.

If a card would be put into an opponent's graveyard from anywhere, exile it instead.

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/l/leyline_of_the_void.txt`
- Type: `Enchantment`, mana cost `2 B B`.
- Key tags:
  - `R:Event$ Moved | ActiveZones$ Battlefield | Destination$ Graveyard | ValidCard$ Card.!token+OppOwn | ReplaceWith$ Exile`
    — the graveyard-hate replacement effect.
  - `SVar:Exile:DB$ ChangeZone | Hidden$ True | Origin$ All | Destination$ Exile | Defined$ ReplacedCard`
    — note: **no** `WithCountersType$ VOID`. This is the distinguishing tag versus Dauthi Voidwalker,
    which has the identical R: line but an `Exile` SVar that adds `WithCountersType$ VOID`.
  - `K:MayEffectFromOpeningHand:FromHand` + `SVar:FromHand:DB$ ChangeZone | ... Origin$ Hand | Destination$ Battlefield`
    — the "begin the game with it on the battlefield" clause.

## Engine work
The `EXILE_INSTEAD_OF_GRAVEYARD` replacement-effect kind already existed for **Dauthi Voidwalker**
(parsed in `src/parse.cpp`, dispatched in `src/systems/replacement_effects.cpp`). Leyline's R: line
is byte-for-byte the same shape, so it already parsed to that kind. The one correctness gap was that
the existing code **unconditionally** added a "void counter" to every exiled card (recording it in
`cur_game.void_countered` and logging "exiled with a void counter"). That is a property of Dauthi's
ability (the void counter lets Dauthi's controller later *play* the exiled card via its activated
ability — read in `src/effects/effect_choose_card.cpp`), **not** of the generic
exile-instead-of-graveyard replacement. Leyline plain-exiles; its cards must not become castable.

Mechanics added (general, not card-specific — keyed on the tag's intended meaning):
- `Effect::Replacement::with_void_counter` flag (`src/components/effect.h`).
- `src/parse.cpp` now reads the SVar named by `ReplaceWith$` and sets `with_void_counter` only when
  that zone-change SVar carries a `VOID` counter type (`WithCountersType$ VOID`). The two identical
  R: lines are **not** retagged — they both parse to `EXILE_INSTEAD_OF_GRAVEYARD`; the void-counter
  distinction is read from the actual zone-change effect the script references, per the "parse tags
  as intended" rule.
- `src/systems/replacement_effects.cpp` propagates the flag onto the `Candidate` and, in `apply_one`,
  only inserts into `void_countered` / logs the void-counter message when the flag is set; otherwise
  it plain-exiles with the message "… is exiled instead of being put into a graveyard."

Dauthi Voidwalker is unchanged behaviorally: its SVar still carries `VOID`, so `with_void_counter`
stays true and its existing tests/behavior are preserved.

## Behavioral decisions (made in lieu of asking a human)
- **"Begin the game with it on the battlefield" clause IGNORED.** The
  `K:MayEffectFromOpeningHand` / `SVar:FromHand` opening-hand free-deploy is not supported by the
  engine and is not a game-action the parser models. Per the task's allowance, this clause is safe
  to ignore: it is purely an opening-hand convenience, and the card's *main* functionality (the
  graveyard-hate replacement) works fully. Leyline remains a normally hard-castable `2BB`
  enchantment whose replacement effect fires once it is on the battlefield. The unrecognized
  `MayEffectFromOpeningHand` keyword is retained on the card (the parser keeps unknown keywords
  without error); no warning was observed.
- The replacement applies to a card entering an **opponent's** graveyard from *anywhere* (CR 614 —
  replacement effects; `Origin$ All`). The engine's `MOVE_TO_ZONE` dispatch already keys on the
  destination being a graveyard owned by the opponent of the Leyline controller, regardless of the
  card's origin zone (battlefield, stack, hand, library). Verified for a creature dying (battlefield→GY)
  and a spell resolving (stack→GY).
- Tokens and the controller's own cards are excluded (`Card.!token+OppOwn`): the controller's own
  cards still go to their own graveyard normally. Verified (own creature dying → own graveyard).
- Multiple Leylines: each is an applicable replacement; CR 616.1 lets the affected player choose
  one to apply first (the others then no longer apply since the card is already heading to exile).
  Verified — the scripted opponent was offered the `CHOOSE_REPLACEMENT` menu with no crash.

## Tests
Isolation (`train/test_harness.py`, pre-set battlefields):
- Opponent's creature killed under controller's Leyline: A has `Leyline of the Void` + Mountains,
  B has `Birds of Paradise`; `A:cast:Lightning Bolt, A:target:Birds of Paradise@opp` →
  "Birds of Paradise is destroyed (lethal damage)" then **"Birds of Paradise is exiled instead of
  being put into a graveyard."** (no void-counter message); A's own Lightning Bolt went to A's
  graveyard normally. PASS.
- Own card unaffected + opponent's spell exiled: A has `Leyline of the Void` + `Birds of Paradise`,
  B casts `Lightning Bolt` at A's Birds → A's Birds (controller's own) goes to A's graveyard, while
  **B's** Lightning Bolt (opponent of Leyline's controller, stack→GY) is **"exiled instead of being
  put into a graveyard."** PASS — confirms own-vs-opponent ownership split and origin-agnostic exile.

Regression (`train/test_harness.py --scripted`, 6 games, seeds 1–6): deck `temp/leyline_a`
(4 Leyline of the Void, 4 Lightning Bolt, 10 Swamp, 10 Mountain, 2 Dark Ritual) vs `temp/leyline_b`
(8 Birds of Paradise, 4 Lightning Bolt, 18 Forest). All 6 games finished decisively (A wins each via
decking B), no draws. Leyline was cast and resolved in real games, its replacement fired (opponent's
Lightning Bolts exiled instead of going to graveyard), and multiple Leylines correctly offered the
`CHOOSE_REPLACEMENT` choice. No non-fatal errors / asserts / tracebacks / warnings.

## Result
implemented
