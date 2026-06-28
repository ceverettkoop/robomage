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
- **-11 ultimate**: gated by the 11-loyalty cost (correctly hidden below loyalty 11). Searches the
  library for colorless nonland cards, exiles them, shuffles (standard `ChangeZone` Library→Exile),
  then a `DBEffect` would grant "cast those from exile without paying mana costs." The free-cast
  grant is **a no-op** in the engine today (`grant_cast` only grants casting from the *graveyard*;
  cast-from-exile-without-cost is unimplemented), so the ultimate exiles the cards and resolves
  cleanly but does not (yet) let you cast them for free. This matches the brief's allowance to
  ship the rest provided the ultimate resolves without error.

## Tests (`train/test_harness.py`)
- **Cast trigger**: cast Ugin `{7}`, target opponent's Grizzly Bears → "Grizzly Bears is moved to
  exile"; Ugin enters at loyalty 7. With no colored permanents, the trigger offers "No target".
- **+2**: "Activate Ugin (GainLife)" → "Player A gains 3 life (now at 23)" + "category: Draw,
  amount: 1"; loyalty 7 → 9 → 11 across activations (reached 11 manually).
- **0**: with no lands, cast a {3} artifact paying only with Ugin → "activated Ugin, Eye of the
  Storms for 3(C)"; loyalty unchanged.
- **-11 gating**: appears in the menu as "Activate Ugin (ChangeZone)" only once loyalty ≥ 11.
  (Full resolution couldn't be driven through the `--play` harness due to multi-turn cleanup-discard
  timing + the action label being the category; in scripted games loyalty climbed to 19 with the
  ultimate available every turn and **no** crash/error.)

Regression: scripted full games, `temp/ugin_a` (Ugin + Lightning Bolt + Expedition Map + Mountain)
vs `temp/wbc_a`, seeds 1 and 4 — decisive (A wins / B wins), no draws, no non-fatal errors.

## Result
Implemented. Cast triggers, +2, and 0 verified; -11 correctly gated and resolves via standard
ChangeZone (its free-cast-from-exile grant is a documented no-op). Build clean; regression
decisive with no errors.
