# Quantum Riddler

```
Name:Quantum Riddler
ManaCost:3 U U
Types:Creature Sphinx
PT:4/6
K:Flying
T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self | Execute$ TrigDraw | TriggerDescription$ When this creature enters, draw a card.
SVar:TrigDraw:DB$ Draw
R:Event$ DrawCards | ActiveZones$ Battlefield | ValidPlayer$ You | CheckSVar$ X | SVarCompare$ LE1 | ReplaceWith$ DrawPlusOne | Description$ As long as you have one or fewer cards in hand, if you would draw one or more cards, you draw that many cards plus one instead.
SVar:DrawPlusOne:DB$ Draw | NumCards$ ReplaceCount$Number/Plus.1
SVar:X:Count$ValidHand Card.YouOwn
K:Warp:1 U
Oracle:Flying\nWhen this creature enters, draw a card.\nAs long as you have one or fewer cards in hand, if you would draw one or more cards, you draw that many cards plus one instead.\nWarp {1}{U}
```

**Oracle:** Flying. When this creature enters, draw a card. As long as you have one or fewer cards
in hand, if you would draw one or more cards, you draw that many cards plus one instead.
Warp {1}{U}. (You may cast this card from your hand for its warp cost. If you cast it this way,
exile it at the beginning of the next end step. For as long as it remains exiled, you may cast it.)

Vocab index: **346** (`src/card_vocab.h`); cast cost {3}{U}{U} regenerated into
`train/card_costs.py` / `src/gen/card_costs_gen.h` by `make`.

**Source:** pre-existing local script `bin/resources/cardsfolder/q/quantum_riddler.txt`
(unchanged — card scripts are never edited).

**Key tags:** `K:Flying` and the ETB `T:ChangesZone→Battlefield | Execute$ TrigDraw` are already
covered. Two new mechanics were required:
- `R:Event$ DrawCards | ValidPlayer$ You | CheckSVar$ X | SVarCompare$ LE1 | ReplaceWith$
  DrawPlusOne`, with `SVar:DrawPlusOne:DB$ Draw | NumCards$ ReplaceCount$Number/Plus.1` and
  `SVar:X:Count$ValidHand Card.YouOwn`.
- `K:Warp:1 U`.

## Mechanics added (general): `draw-plus-replacement`, `warp`

### 1. `draw-plus-replacement` — an additive draw-count replacement (CR 614.1)

Before this card the draw-replacement subsystem only handled **Dredge** (a *replace-with-a-different-
action* draw replacement). Quantum Riddler needs a **modifying** draw replacement that *increases the
number of cards drawn*, gated by a condition evaluated at replacement time. Built general — any
"draw N + K instead", with an optional `CheckSVar`/`SVarCompare` gate.

- **`src/components/effect.h`** — new `Effect::Replacement::DRAW_ADD` kind with fields `draw_add`
  (the K in "plus K"), `draw_condition_count_expr` (the resolved `Count$` expression) and
  `draw_condition_compare` (the Forge comparator).
- **`src/parse.cpp`** (`parse_replacement_effects`) — recognizes `Event$ DrawCards`,
  `ValidPlayer$ You`, and the `ActiveZones$ Battlefield` scope; reads the additive count `K` out of
  the `ReplaceWith$` SVar's `NumCards$ ReplaceCount$Number/Plus.K`; and resolves `CheckSVar$ X` to
  the named SVar's `Count$` body (here `Count$ValidHand Card.YouOwn` = cards in your hand — an
  expression `evaluate_sa_svar` already supports, from Ensnaring Bridge) with its `SVarCompare$`.
- **`src/systems/replacement_effects.{h,cpp}`** — new pure `replacement::draw_count_bonus(player)`:
  scans the player's battlefield permanents, sums `draw_add` over every `DRAW_ADD` whose (optional)
  count gate holds for the drawing player, and returns the extra-card count. No prompt (unlike the
  dredge menu), so it is safe to evaluate at every draw site.
- **Draw sites** — `Orderer::draw_one` (blocking path) and `effects::draw_n_with_replacements`
  (suspendable path) compute the bonus **before** the base draw (so the "one or fewer cards in
  hand" gate sees the pre-draw hand), then perform `bonus` extra **plain** draws after a real base
  draw. The extra draws are part of the same replaced event (CR 614.5), so they are *not* re-passed
  through the additive replacement (no re-application / infinite loop).

### 2. `warp` — an alternative cast cost with end-step exile + recast-from-exile (a 2025 keyword)

Warp is a very new (2025) keyword and is **not in the checked-in CR snapshot**
(`docs/mtg_comprehensive_rules.txt`), so it is implemented per the script's `K:Warp:<cost>` tag
and the intended semantics, reusing existing infrastructure end to end:

- **`src/components/carddata.h`** — `AltCost::is_warp`. Warp is encoded on the shared `AltCost`
  (mana portion = the warp cost) exactly like Impending/Spectacle/Miracle, so it flows through the
  existing alt-cost cast path with no new payment plumbing.
- **`src/parse.cpp`** — parses `K:Warp:<cost>` into `alt_cost` (`has_alt_cost`, `is_warp`,
  `mana_cost = <cost>`), mirroring Spectacle. `can_afford_alt` gates it purely on affordability;
  `state_manager_actions.cpp` offers "Cast … (warp)" and pays the warp cost via the alt-cost path.
- **`src/components/spell.h`** `cast_with_warp`, set in **`action_processor.cpp`** when the alt cost
  used is warp; **`src/systems/stack_manager.cpp`** carries it into `cur_game.pending_warp` as the
  spell resolves onto the battlefield.
- **End-step exile** — **`src/systems/state_manager_statics.cpp`** consumes `pending_warp` at
  permanent creation and calls the new `mark_warp_permanent`, which registers a one-shot delayed
  triggered ability (CR 603.7b) firing at `END_STEP_BEGAN` — **the same delayed-trigger machinery
  Unearth uses** (`mark_unearthed_permanent`), but with no haste and no leaves-the-battlefield →
  exile redirect. Its fire ability is the new **`WarpExile`** effect.
- **`WarpExile` effect** — **`src/effects/effect_warp.cpp`** (+ `EffectKind::WarpExile` in
  `effect_kind.{h,cpp}` and `effect_table.cpp`, decl in `effects.h`): on resolution it exiles the
  source *if it is still the same battlefield object*, then grants the recast permission.
- **Recast-from-exile** — reuses **`Game::ImpulseCastPermission`** (built for Light Up the Stage /
  Suspend). The grant is `resource = NORMAL` (recast for its **normal** cost) with a new `warp`
  flag. **`src/classes/game.cpp`** cleanup keeps a `warp` permission across turns for as long as the
  card remains in exile (it lapses only once the card leaves exile), and the existing NORMAL "play
  from exile" cast path (`state_manager_actions.cpp` / `action_processor.cpp`) offers it at the
  card's normal (sorcery-speed) timing and enters it as an ordinary permanent.

## Behavioral decisions

- **Draw-count replacement modeled per single-card draw event.** The engine draws one card at a
  time (`draw_one`), so the additive replacement is applied per draw event: with the source on the
  battlefield and the gate satisfied, "draw a card" becomes "draw two". The bonus is computed from
  the hand size *before* the draw (CR 614: replacement applied to the event as it would happen).
  Multiple `DRAW_ADD` sources sum (each applies once per event, 614.5). Dredge + additive
  simultaneity is not jointly modeled (no vocab card combines them): a dredged draw draws no card,
  so the additive bonus, which modifies the *draw*, does not apply.
- **Warp modeled from the script tag + intended semantics** (keyword absent from the CR snapshot):
  cast for the warp cost is an alternative casting cost from hand (sorcery speed for a creature);
  the warp-cast object is exiled at the beginning of the next end step via a delayed triggered
  ability; and a lasting cast-from-exile permission (normal cost) is granted when it is exiled and
  persists while it remains exiled. A warp-cast permanent that leaves before the end step (dies, is
  bounced) simply follows normal rules — `WarpExile` no-ops when the source is no longer the same
  battlefield object. The recast is a normal cast (not warp), so it is not exiled again.
- The card is castable **normally** for {3}{U}{U} (warp is an alternative, never the only option).

## Tests (isolation, `train/test_harness.py`, `--play`)

- **Normal cast + Flying + plain ETB draw** — cast Quantum Riddler for {3}{U}{U} from a full hand:
  enters as a 4/6 Flyer, ETB draws **1** card (hand ≥ 2, so the additive gate is off). ✓
- **Additive draw (draw-2)** — emptied A's hand to 1 card (5 Lightning Bolts), then cast Quantum
  Riddler as the last relevant card: its ETB "draw a card" (base amount 1) drew **2** cards
  (Forest + Swamp) because A's hand was ≤ 1 with Quantum Riddler on the battlefield. ✓
- **Warp lifecycle** — warp-cast for {1}{U} (2 Islands): enters the battlefield; at the next end
  step the `WarpExile` trigger **exiled** it and granted the recast permission; on turn 3 the
  "Cast Quantum Riddler (from exile)" action was offered and A **recast it for {3}{U}{U}**, entering
  it as a permanent that then persisted (attacking on later turns, not re-exiled). ✓
- **CI gate** — `train/ci_check.py --tier pygen,vocab,smoke` (0 errors, no draws); `make` clean
  apart from the pre-existing parse.cpp warnings.

## Result

**Implemented.** Both new general mechanics (`draw-plus-replacement`, `warp`) are in place and
exercised in isolation.
