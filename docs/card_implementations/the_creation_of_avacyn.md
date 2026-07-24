# The Creation of Avacyn

```
Name:The Creation of Avacyn
ManaCost:1 B B
Types:Enchantment Saga
K:Chapter:3:DBExile,DBSetState,DBReturn
SVar:DBExile:DB$ ChangeZone | ChangeNum$ 1 | ChangeType$ Card | Mandatory$ True | Origin$ Library | Destination$ Exile | ExileFaceDown$ True | SpellDescription$ Search your library for a card, exile it face down, then shuffle.
SVar:DBSetState:DB$ SetState | Defined$ ExiledWith | Mode$ TurnFaceUp | SubAbility$ DBLoseLife | SpellDescription$ Turn the exiled card face up. If it's a creature card, you lose life equal to its mana value.
SVar:DBLoseLife:DB$ LoseLife | ConditionDefined$ ExiledWith | ConditionPresent$ Creature | LifeAmount$ ExiledWith$CardManaCost
SVar:DBReturn:DB$ ChangeZone | Optional$ True | Defined$ ExiledWith | Origin$ Exile | Destination$ Battlefield | ConditionDefined$ ExiledWith | ConditionPresent$ Creature | SubAbility$ DBChangeZone | SpellDescription$ You may put the exiled card onto the battlefield if it's a creature card. If you don't put it onto the battlefield, put it into its owner's hand.
SVar:DBChangeZone:DB$ ChangeZone | Defined$ ExiledWith | Origin$ Exile | Destination$ Hand
```

**Oracle:** (As this Saga enters and after your draw step, add a lore counter. Sacrifice after III.)
- I — Search your library for a card, exile it face down, then shuffle.
- II — Turn the exiled card face up. If it's a creature card, you lose life equal to its mana value.
- III — You may put the exiled card onto the battlefield if it's a creature card. If you don't put
  it onto the battlefield, put it into its owner's hand.

Vocab index: **349** (`src/card_vocab.h`).

Forge script source: pre-existing local script `bin/resources/cardsfolder/t/the_creation_of_avacyn.txt`
(unchanged — card scripts are never edited).

The Saga scaffolding (lore counters on ETB / precombat main, per-chapter trigger firing,
sacrifice-after-final, `K:Chapter` parse) already existed (`src/saga.cpp`,
`state_manager_triggers.cpp`). This card needed three genuinely new **general** mechanics.

## Mechanics added (general): exile-face-down + SetState/TurnFaceUp + Defined$ ExiledWith

### 1. `ExileFaceDown$ True` on a ChangeZone — exile a card face down (CR 708)

- **Storage:** `Zone::is_face_down` (`src/components/zone.h`). Cleared on every zone change (a card
  that leaves exile is no longer that hidden object, CR 708.4) alongside `identity_known`.
- **Move plumbing:** `Orderer::add_to_zone` (`src/systems/orderer.h/.cpp`) gained an
  `exile_face_down` parameter: when true and the destination is EXILE it stamps `Zone::is_face_down`
  and **withholds** the card from the owner's public revealed multi-hot (a face-down exile is not
  public knowledge). `effects::change_zone_move` threads the flag through. Parse:
  `ExileFaceDown$ True` → `Ability::exile_face_down` (`parse_change_zone`).
- The search-based ChangeZone (chapter I) records the exiled card on the source's
  `Permanent::exiled_with` (the "cards exiled with this" association) and logs the face-down exile
  **privately** — the searcher knows the card; the opponent sees only "a card face down".

### 2. `DB$ SetState | Mode$ TurnFaceUp` — a new SetState effect (CR 708.3 / 711.8)

New effect category `SetState` (`src/effects/effect_set_state.cpp`, wired through
`effect_kind.{h,cpp}`, `effect_table.cpp`, `effects.h`). `Mode$ TurnFaceUp` clears the subject's
`Zone::is_face_down` (revealing its real characteristics) and records the now-public identity in the
belief state; the handler is structured so `Mode$ TurnFaceDown` (and other modes) slot in later.
The subject is resolved from `Defined$` (here `ExiledWith`; `Self`/target fall back). There was no
SetState effect before (only the day/night `transform_permanent`).

### 3. `Defined$ ExiledWith` — a cross-chapter reference to the exiled card

The Saga **remembers** the card its chapter I exiled via `Permanent::exiled_with` on the Saga
source; `exiled_with_card(source)` (`src/game_queries.h`) resolves the most-recent still-live entry.
This is wired into every place a later chapter reads it:

- **`Defined$ ExiledWith`** (`Ability::defined_exiled_with`, parsed in `parse.cpp`): the chapter III
  `change_zone` branch moves that card out of exile. It acts only while the card is still in its
  `Origin$` (Exile) zone, so the two-leg "battlefield or hand" chain (DBReturn then DBChangeZone)
  self-arbitrates — once one leg moves the card, the other no-ops.
- **`ExiledWith$CardManaCost`** (chapter II `LifeAmount$`): `evaluate_dynamic_amount` resolves it to
  the exiled card's mana value. The direct-expression form is preserved through
  `parse_svar_ability` (it is not an SVar name, so the normal SVar lookup would have dropped it).
- **`ConditionDefined$ ExiledWith | ConditionPresent$ Creature`**
  (`Ability::condition_on_exiled_with`, parsed in `parse.cpp`, gated in `Ability::resolve` phase 3
  and evaluated in `present_condition_raw`): checks whether the exiled card is a creature, gating the
  life-loss (chapter II) and the battlefield put (chapter III). On failure the body is skipped but
  subabilities still chain — which is exactly the "if it's a creature card … otherwise put it into
  its owner's hand" fallthrough.

### General fix uncovered: `ChangeType$ Card` single-zone search

`search_zone` treated only an *empty* `change_type` as "any card"; `ChangeType$ Card` (a bare
"search your library for a card") matched nothing because no card object has a printed type literally
named "Card", so a mandatory search would "fail to find". Fixed to treat `"Card"` as the catch-all,
mirroring `search_multi_zone` (also benefits Demonic/Vampiric-Tutor-style effects).

## Files touched (engine)

- `src/components/zone.h` — `is_face_down` flag.
- `src/systems/orderer.{h,cpp}` — `add_to_zone` `exile_face_down` param (stamp flag + withhold from
  revealed set); clear `is_face_down` on every zone change.
- `src/components/ability.h` — `defined_exiled_with`, `condition_on_exiled_with`, `set_state_mode`,
  `exile_face_down` fields.
- `src/game_queries.h` — `exiled_with_card(source)` resolver.
- `src/effects/effect_set_state.cpp` (new) + `effect_kind.{h,cpp}`, `effect_table.cpp`, `effects.h`
  — the SetState effect and its parse hook.
- `src/effects/effect_change_zone.cpp` — `ExileFaceDown$` parse, `Defined$ ExiledWith` move branch,
  exiled-with recording + face-down logging on the search path.
- `src/components/ability.cpp` — `ExiledWith$CardManaCost` in `evaluate_dynamic_amount`,
  `condition_on_exiled_with` gate in `resolve`, and the `ChangeType$ Card` search fix.
- `src/systems/state_manager_actions.cpp` — `condition_on_exiled_with` in `present_condition_raw`.
- `src/parse.cpp` — `Defined$/ConditionDefined$ ExiledWith` parse, `ExiledWith$` direct-expr
  preservation for `LifeAmount$`.
- `src/card_vocab.h` (+ regenerated `train/card_costs.py`, `src/gen/card_costs_gen.h`).

Relevant rules: CR 714 (Sagas), 714.2/714.3 (chapter abilities / lore counters), 708 (face-up /
face-down permanents & cards), 701.35 (turn face up/down), 406 (exile), 110.2a (control on entry).

## Behavioral decisions

- Chapter I exiles the card **face down** in exile: its identity is withheld from the opponent's
  belief state until chapter II turns it face up (verified — the opponent only "sees" the card in
  exile after the turn-face-up).
- Chapter II's life loss is gated on the exiled card being a creature and equals its mana value.
- Chapter III presents the "put onto the battlefield" choice **only** when the exiled card is a
  creature; declining (or a noncreature) routes it to its owner's hand. The Saga is sacrificed after
  the final chapter resolves.

## Tests (`train/test_harness.py`)

Preset: The Creation of Avacyn + Swamps in A's hand/battlefield, Grizzly Bears (MV2) buried in A's
library. Lore counters advance on ETB (I) then one per subsequent A turn (II, III).

- **Creature branch, put onto battlefield:** chapter I "exiles Grizzly Bears face down" (opponent
  can't see it); chapter II "Grizzly Bears is turned face up" + "Player A loses 2 life (now at 18)"
  (MV2), and only now does the opponent's revealed set include Grizzly Bears; chapter III → accept →
  "Grizzly Bears is moved to the battlefield" → live **2/2 (SICK)** under A's control, then the Saga
  is sacrificed.
- **Creature branch, declined:** chapter III → decline → "Grizzly Bears is moved to hand".
- **Noncreature branch:** exile a Swamp at I → "Swamp is turned face up" at II with **no** life loss
  (condition gated, LoseLife body skipped) → chapter III offers no battlefield put and "Swamp is
  moved to hand".
- **CI gate:** `ci_check.py --tier pygen,vocab,smoke` (see below).

## Result

Implemented. Three new general mechanics — **exile face down** (`ExileFaceDown$`), a
**`SetState`/`TurnFaceUp`** effect, and the **`Defined$ ExiledWith`** cross-chapter reference (plus
`ExiledWith$CardManaCost` and `ConditionDefined$ ExiledWith`) — on top of the existing Saga
scaffolding, with a general `ChangeType$ Card` single-zone search fix. All three chapters and both
branches verified end-to-end.
