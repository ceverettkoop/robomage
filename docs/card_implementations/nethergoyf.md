# Nethergoyf (vocab index 177)

## Oracle text
Nethergoyf's power is equal to the number of card types among cards in your graveyard and its
toughness is equal to that number plus 1.

Escape—{2}{B}, Exile any number of other cards from your graveyard with four or more card types
among them. *(You may cast this card from your graveyard for its escape cost.)*

(Creature — Lhurgoyf, `{B}`, `*/1+*`.)

## Forge script (Source: pre-existing local `bin/resources/cardsfolder/n/nethergoyf.txt`)
Key tags (parsed as written — no retagging):
- `S:Mode$ Continuous | CharacteristicDefining$ True | SetPower$ Count$ValidGraveyard Card.YouOwn$CardTypes | SetToughness$ Count$ValidGraveyard Card.YouOwn$CardTypes/Plus.1`
  — a characteristic-defining ability (CR 604.3 / 613.4a) setting base P/T to the number of
  distinct card types among cards in **your** graveyard (power) and that number plus one
  (toughness).
- `K:Escape:2 B ExileFromGrave<X/Card.Other+withTypesGE4/other cards from your graveyard with four or more card types among them>`
  — the Escape keyword (CR 702.139): cast from graveyard for `{2}{B}` plus the additional cost
  of exiling other graveyard cards whose chosen set has ≥4 card types among them.
- `SVar:X:Count$xPaid` — Forge plumbing for "any number"; there is **no** X in the mana cost,
  so the engine does not prompt for an X value (behavior resolved from the Oracle text).

## Engine work
Two general mechanics, both keyed on the script's real tags.

### 1. Controller-scoped CardTypes count (CDA P/T)
The CDA static-P/T path already existed (Barrowgoyf: `Count$ValidGraveyard Card$CardTypes`,
all graveyards). Nethergoyf needs the **controller-scoped** variant
`Count$ValidGraveyard Card.YouOwn$CardTypes` (your graveyard only), with the `/Plus.1` suffix
on toughness already honored by `evaluate_sa_svar`.

`src/svar_eval.cpp` — the all-graveyards `...Card$CardTypes` block now also matches
`Count$ValidGraveyard Card<restriction>$CardTypes`. When the restriction between `Card` and
`$CardTypes` contains `YouOwn`/`YouCtrl`, the distinct-card-type count (CR 205.2) is scoped to
the controller's graveyard (`z.owner == controller`); otherwise it counts all graveyards as
before. The evaluator already receives `controller`, so no signature change. The `/Plus.1`
suffix is stripped and added by the existing suffix handler.

### 2. Escape keyword (CR 702.139)
A new cast-from-graveyard alternate cast, mirroring the existing Flashback enumeration/payment
structure.

- **Parse** (`src/parse.cpp`): `K:Escape:<mana> [ExileFromGrave<...>]`. The leading mana
  symbols (`2 B`) parse into `CardData::escape_mana_cost`; the remainder parses through the
  shared alt-cost grammar into `CardData::escape_alt_cost`. A new `ExileFromGrave<X/<filter>/<label>>`
  token in `parse_alt_cost_tokens` reads `withTypesGE<N>` from the filter into
  `AltCost::exile_grave_min_types` (the group-type constraint). General to any escape card.
- **Data** (`src/components/carddata.h`): `has_escape`, `escape_mana_cost`, `escape_alt_cost`
  on `CardData`; `exile_grave_min_types` on `AltCost`.
- **Enumerate** (`src/systems/state_manager_actions.cpp`): for each card with `has_escape` in
  its owner's graveyard, offer "Cast … (escape)" (`LegalAction::use_escape`) at the timing its
  type allows (sorcery-speed unless instant — same gate Flashback uses), when the controller
  can pay the escape mana **and** enough OTHER graveyard cards exist to reach the required
  distinct card types (`graveyard_card_types(...) >= exile_grave_min_types`). Grafdigger's-style
  cast-from-graveyard prohibitions are respected via the shared `rules_mod::cast_prohibited`.
- **Pay** (`src/action_processor.cpp`): the `use_escape` cast branch pays the escape mana, then
  `pay_exile_from_grave_cost` runs a mandatory choice loop over the caster's other graveyard
  cards, exiling chosen cards until the exiled set covers ≥N distinct card types (CR 601.2f —
  paid as the spell is cast, before it is on the stack). Nethergoyf is a permanent spell, so it
  resolves to the battlefield normally (no on-resolution exile, unlike Flashback); escape grants
  no counters (Nethergoyf has no "enters with counters" clause).

### Reusable helper
`src/game_queries.h` — `graveyard_card_types(owner, entities, except = 0)`: distinct card types
(CR 205.2) among `owner`'s graveyard cards, optionally excluding one entity ("other cards").
`check_delirium` was refactored to call it; the Escape enumeration and the ExileFromGrave
cost-payment loop both use it.

## Behavioral decisions
- **"Four or more card types among them"**: the binding constraint per the Oracle is that the
  exiled set collectively has ≥4 distinct card types; the player may exile any number of cards
  meeting that. Implemented as a mandatory exile loop that ends as soon as the chosen set
  reaches 4 distinct types (the natural minimum). The cast is only offered when ≥4 types are
  available among other graveyard cards, so the cost is always payable when offered.
- **Controller-scoped vs all-graveyards count**: Nethergoyf counts only the controller's
  graveyard (`Card.YouOwn`), in contrast to Barrowgoyf's all-graveyards count — verified by
  test (b) below.
- **No X prompt**: `SVar:X:Count$xPaid` is "any number" plumbing; there is no X mana symbol, so
  the cast does not prompt for an X value.

## Tests (test_harness.py, semantic `--play`, seed 1)
(a) **CDA P/T** — Nethergoyf on battlefield, graveyard with Dragon's Rage Channeler (Creature),
    Lightning Bolt (Instant), Duress (Sorcery), Forest (Land) = 4 types → **Nethergoyf [4/5]**.
    Confirmed it counts only your graveyard: with the 4-type set in the *opponent's* graveyard
    and only 2 types (Instant, Sorcery) in your own, Nethergoyf is **[2/3]** (Barrowgoyf would
    be 4/5). ✓
(b) **Escape cast** — Nethergoyf in graveyard, 3 Swamps in play, graveyard otherwise covering 4
    types. "Cast Nethergoyf (escape)" offered at First Main; paid `{2}{B}`, the exile loop forced
    choosing cards until 4 distinct types were exiled (DRC, Bolt, Duress, Forest), Nethergoyf
    cast and resolved onto the battlefield. Post-resolution it is **[0/1]** — the four cards
    left the graveyard (now empty), correctly lowering its own CDA P/T. ✓
(c) **Escape unavailable** — with only 3 card types among other graveyard cards, no escape cast
    is offered. With ≥4 types available but only Forests in play (no `{B}`), no escape cast is
    offered. ✓
(d) **Regression** — scripted full games, `temp/nethergoyf_a` (Nethergoyf + Swamps + Duress +
    Sheoldred's Edict + Mishra's Bauble + Dauthi Voidwalker — five card types feed the graveyard)
    vs `mav`, seeds 1/2/3: all clean Player A wins, **no draws, no non-fatal errors** attributable
    to Nethergoyf. ✓

## CR references
- 702.139 — Escape.
- 601.2f — additional costs paid as a spell is cast.
- 205.2 — card types.
- 604.3 / 613.4a — characteristic-defining abilities set base P/T.
