# Uro, Titan of Nature's Wrath (vocab index 279)

## Oracle text
When Uro enters, sacrifice it unless it escaped.

Whenever Uro enters or attacks, you gain 3 life and draw a card, then you may put a land card
from your hand onto the battlefield.

Escape—{G}{G}{U}{U}, Exile five other cards from your graveyard. *(You may cast this card from
your graveyard for its escape cost.)*

(Legendary Creature — Elder Giant, `{1}{G}{U}`, `6/6`.)

## Forge script (Source: pre-existing local `bin/resources/cardsfolder/u/uro_titan_of_natures_wrath.txt`)
Key tags (parsed as written — no retagging):
- `T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self | Execute$ TrigSac`
  `SVar:TrigSac:DB$ Sacrifice | SacValid$ Self | ConditionNotPresent$ Card.Self+escaped`
  — the ETB "sacrifice it unless it escaped" trigger. The `ConditionNotPresent$` gates the
  sacrifice on the source NOT having escaped.
- `T:... Destination$ Battlefield | Execute$ TrigGainLife` and `T:Mode$ Attacks | ... Execute$ TrigGainLife`
  `SVar:TrigGainLife:DB$ GainLife | LifeAmount$ 3 | SubAbility$ DBDraw`
  `SVar:DBDraw:DB$ Draw | Defined$ You | SubAbility$ DBLand`
  `SVar:DBLand:DB$ ChangeZone | Origin$ Hand | Destination$ Battlefield | ChangeType$ Land | ChangeNum$ 1`
  — the shared enters-or-attacks payoff: gain 3, draw, then (optionally) put a land from hand.
- `K:Escape:G G U U ExileFromGrave<5/Card.Other/other>` — Escape (CR 702.139): cast from the
  graveyard for `{G}{G}{U}{U}` plus exiling **five** other graveyard cards.

## Engine work
The genuinely new mechanic is the **"escaped" flag** carried Spell → Permanent, plus the
`ConditionNotPresent$` condition and the `Card.Self+escaped` qualifier. Two smaller general
extensions (literal-count ExileFromGrave; bare `Self` filter head) were needed to make Uro's
script run end-to-end. Escape itself was already implemented (Nethergoyf).

### 1. The "escaped" flag (Spell → Permanent), the NEW mechanic
A permanent must remember it entered because its spell was cast for the Escape cost, so the ETB
sacrifice can read it. Mirrors the existing evoke (`pending_evoked`/`Permanent::evoked`) and
"cast from hand" carry.
- `Spell::cast_with_escape` (`src/components/spell.h`) — set from `action.use_escape` at cast
  time (`src/action_processor.cpp`).
- When an escape spell resolves to the battlefield, `src/systems/stack_manager.cpp` inserts it
  into `Game::pending_escaped` (`src/classes/game.h`) before the Spell component is removed.
- `src/systems/state_manager_statics.cpp` consumes `pending_escaped` when the Permanent is
  created → `Permanent::cast_with_escape = true` (`src/components/permanent.h`). A normal hand
  cast / any non-escape entry leaves it false. General — reusable by any "if it escaped" card.

### 2. `ConditionNotPresent$` + `Card.Self+escaped` qualifier
- `src/parse.cpp` — `ConditionNotPresent$` parses into `condition_present` plus
  `Ability::condition_negate = true` (`src/components/ability.h`) and `intervening_if = true`
  (re-checked when the trigger goes on the stack and at resolution, CR 603.4-style). The Execute
  SVar's own intervening-if is now OR-ed (not clobbered) by the trigger-line carry.
- `src/systems/state_manager_actions.cpp` — `evaluate_present_condition` is now a thin wrapper
  that calls `present_condition_raw` and inverts the result when `condition_negate` is set
  (general negation for any `ConditionNotPresent$`). A new `Card.Self+escaped` branch reads
  `Permanent::cast_with_escape` off the ability's source.

### 3. Literal-count ExileFromGrave (Escape additional cost)
Nethergoyf's ExileFromGrave is a distinct-card-types constraint (`withTypesGE<N>`); Uro's is a
literal count of five cards (`ExileFromGrave<5/Card.Other/other>`).
- `src/parse.cpp` — the `ExileFromGrave<...>` parser, when there is no `withTypesGE`, reads the
  leading number into `AltCost::exile_grave_count` (`src/components/carddata.h`). Nethergoyf's
  leading `X` is non-numeric and still routes through the `withTypesGE` branch, unchanged.
- `src/systems/state_manager_actions.cpp` — the escape cast is offered only when enough OTHER
  graveyard cards exist (`graveyard_card_count(...) >= exile_grave_count`).
- `src/action_processor.cpp` — `pay_exile_from_grave_count_cost` exiles exactly N other
  graveyard cards as the spell is cast (CR 601.2f), mirroring the distinct-types payer.
- `src/game_queries.h` — new `graveyard_card_count(owner, entities, except)` helper, next to
  `graveyard_card_types`.

### 4. Bare `Self`/`Other` filter head
`SacValid$ Self` uses a bare `Self` as the filter head. `eval_alternative`
(`src/game_queries.cpp`) treated any non-`Card`/`Permanent`/`Spell` head as a type-line name, so
`Self` matched nothing and the sacrifice silently did nothing. Now a bare `Self`/`Other` head is
recognized as an identity qualifier (no type-line requirement; evaluated via `eval_qualifier`),
matching Forge's `SacValid$ Self`. General — any card with a bare-`Self`/`Other` filter benefits.

## Behavioral decisions
- **Sacrifice-unless-escaped as intervening-if**: the TrigSac trigger has no other effect and no
  subabilities, so gating the whole trigger via `intervening_if` (suppress when escaped) is
  observably identical to "resolve but sacrifice nothing". The separate GainLife trigger is
  unaffected and always fires.
- **"You may put a land"**: the `DBLand` ChangeZone offers a "Fail to find" (decline) option, so
  the land-put is correctly optional (existing ChangeZone behavior).

## Tests (test_harness.py, semantic `--play`, seed 1)
(a) **Normal cast from hand** — cast Uro for `{1}{G}{U}`: ETB gains 3 life (20→23), draws a card
    (hand +1), then Uro **is sacrificed** to the graveyard (escaped flag false). Battlefield has
    no Uro; graveyard contains Uro. ✓
(b) **Escape cast from graveyard** — Uro in graveyard with 5 other cards (Mountains) and
    `{G}{G}{U}{U}` available: "Cast Uro (escape)" offered at First Main; the cost forced exiling
    all **5** Mountains from the graveyard, then Uro entered, ETB gained 3 life (23→26) and drew
    a card, and Uro **stayed** on the battlefield (escaped → sacrifice trigger suppressed; only
    the GainLife trigger resolved). ✓
(c) **No non-fatal errors / warnings** in either run. ✓
- The on-**attack** GainLife trigger reuses the identical `TrigGainLife` SVar proven firing twice
  on ETB above; the `Mode$ Attacks` trigger is standard engine machinery.

## CR references
- 702.139 — Escape.
- 601.2f — additional costs paid as a spell is cast.
- 603.4 — intervening-if conditions, re-checked on the stack and at resolution.
- 205.2 — card types.
