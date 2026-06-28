# Ugin, Eye of the Storms (vocab index 262)

## Oracle text
When you cast this spell, exile up to one target permanent that's one or more colors.
Whenever you cast a colorless spell, exile up to one target permanent that's one or more colors.
[+2]: You gain 3 life and draw a card.
[0]: Add {C}{C}{C}.
[-11]: Search your library for any number of colorless nonland cards, exile them, then shuffle. Until end of turn, you may cast those cards without paying their mana costs.

(Legendary Planeswalker — Ugin, Loyalty 7, `{7}`)

## Forge script
Source: pre-existing local script at `bin/resources/cardsfolder/u/ugin_eye_of_the_storms.txt`.

Key tags: two `T: ... SpellCast` triggers (`TrigChange:DB$ ChangeZone | ValidTgts$
Permanent.nonColorless | Origin$ Battlefield | Destination$ Exile | TargetMin$ 0`); three
loyalty abilities (`AB$ GainLife | Cost$ AddCounter<2/LOYALTY>`, `AB$ Mana | Cost$
AddCounter<0/LOYALTY> | Produced$ C | Amount$ 3`, `AB$ ChangeZone | Cost$
SubCounter<11/LOYALTY> | Origin$ Library | Destination$ Exile | ChangeType$
Card.nonLand+Colorless` + a `DBEffect` free-cast grant).

## Engine work
Planeswalker loyalty abilities, `AddCounter/SubCounter<N/LOYALTY>` costs, and the cast/colorless
`SpellCast` triggers are all already supported (Jace/Ajani precedent). One cosmetic change:

- `src/parse.cpp` — added `ForgetOnMoved` to the parser's `ignored_keys` (the -11's `DBEffect`
  carries `ForgetOnMoved$ Exile`, Forge bookkeeping for dropping a remembered object when it
  changes zones). Purely cosmetic; silences a one-time parse warning.

No retagging; no card-script edits.

## Behavioral decisions (CR cites)
- **Cast trigger** (CR 601.2i / 603): on casting Ugin (and on casting any colorless spell while
  Ugin is on the battlefield), exile up to one colored permanent. `TargetMin$ 0` → may choose no
  target.
- **+2** (CR 606): gain 3 life and draw, loyalty +2. Sorcery-speed, once per turn.
- **0: Add {C}{C}{C}** — the script tags this `AB$ Mana`, so the engine models it as a mana
  ability (CR 605): it is offered during cost payment (like any mana source) and adds three {C}
  with no loyalty change (`AddCounter<0>`). Note this is a deliberate fidelity deviation: by the
  rules a loyalty ability is *not* a mana ability (CR 605.1a), so the printed Ugin "0" should use
  the stack; honoring the `AB$ Mana` tag means it is usable only while paying a cost, not as a
  free stack action at priority. It produces the correct mana and is once-per-payment.
- **-11 ultimate**: gated by the 11-loyalty cost (correctly hidden below loyalty 11). Two parts,
  both now fully implemented:
  - **Search/exile any number + shuffle** (CR 701.19 / 401.6): `ChangeZone` Library→Exile,
    `ChangeType$ Card.nonLand+Colorless`, `ChangeNum$ X` where `X = Count$ValidLibrary
    Card.YouOwn+nonLand+Colorless`. "Any number" is the dynamic count-SVar resolved into the
    search loop's `num_to_move` cap (`effect_change_zone.cpp`, via `dynamic_amount_expr` /
    `evaluate_dynamic_amount`); the player exiles colorless nonland cards one at a time and a
    fail-to-find stops early (the loop already breaks on a 0 pick). Previously the count-SVar was
    ignored and only **one** card could be exiled — fixed. The library is shuffled after the
    search.
  - **Free cast from exile until end of turn** (CR 113.3 / 118.9 / 601.2f): the `DBEffect`
    (`DB$ Effect | StaticAbilities$ STPlay | RememberObjects$ Remembered | ForgetOnMoved$ Exile`,
    where `STPlay` grants `MayPlay$ True + MayPlayWithoutManaCost$ True | AffectedZone$ Exile`)
    grants, for each just-exiled (remembered) card, permission to cast it **from exile without
    paying its mana cost** this turn. Implemented by recording a `FREE` cast-from-exile permission
    per remembered exiled card in `cur_game.impulse_cast_permission` (the same per-turn map the
    alt-cost impulse cast uses, extended with a `FREE` resource), good until cleanup. The casting
    path offers each as `Cast <card> (from exile, no cost)` while it remains in exile
    (`ForgetOnMoved$ Exile` = the permission lapses once a card leaves exile), funnels it onto the
    stack like any cast, and pays nothing. The map is cleared each cleanup, so the permission ends
    at end of turn.

## Tests (`train/test_harness.py`)
- **Cast trigger**: cast Ugin `{7}`, target opponent's Grizzly Bears → "Grizzly Bears is moved to
  exile"; Ugin enters at loyalty 7. With no colored permanents, the trigger offers "No target".
- **+2**: "Activate Ugin (GainLife)" → "Player A gains 3 life (now at 23)" + "category: Draw,
  amount: 1"; loyalty 7 → 9 → 11 across activations (reached 11 manually).
- **0**: with no lands, cast a {3} artifact paying only with Ugin → "activated Ugin, Eye of the
  Storms for 3(C)"; loyalty unchanged.
- **-11 gating**: appears in the menu as "Activate Ugin (ChangeZone)" only once loyalty ≥ 11.
- **-11 full resolution (real Ugin)**: drove Ugin on the battlefield to loyalty 11 (two `+2`
  activations across turns), put two Expedition Maps deep in Player A's library, then activated the
  ultimate. Transcript: `Player A puts Expedition Map to exile` ×2 then `fails to find` (the
  "any number" search), library shuffled, then `Player A may cast Expedition Map from exile without
  paying its mana cost this turn.` ×2. The menu offered `Cast Expedition Map (from exile, no cost)`
  for both; choosing it: `Player A casts Expedition Map without paying its mana cost` →
  `Expedition Map enters the battlefield` (no mana paid). On a follow-up line the permission was
  **gone the next turn** (no `(from exile, no cost)` action after the turn ended), confirming the
  until-end-of-turn expiry.

Regression: scripted full games, `temp/ugin_reg_a` (4 Ugin + 4 Expedition Map + 32 Wastes) vs
`temp/ugin_reg_b` (Grizzly Bears aggro), seeds 1 and 2 — decisive (B wins both), no draws, no
non-fatal errors.

## Result
Fully implemented. Cast triggers, +2, and 0 verified; **-11 ultimate now complete** — it searches
and exiles any number of colorless nonland cards (count-SVar `ChangeNum` resolution fixed),
shuffles, and grants free cast-from-exile of those cards until end of turn (new `FREE` resource on
the impulse cast-from-exile permission). A real Ugin game cast an exiled Expedition Map for free.
Build clean; regression decisive with no errors.
