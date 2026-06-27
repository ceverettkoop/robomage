# Wastescape Battlemage

```
Name:Wastescape Battlemage
ManaCost:1 C
Types:Creature Eldrazi Wizard
PT:2/2
K:Kicker:G:1 U
T:Mode$ SpellCast | ValidCard$ Card.Self+kicked 1 | Execute$ TrigExile | ...
SVar:TrigExile:DB$ ChangeZone | ValidTgts$ Artifact.OppCtrl,Enchantment.OppCtrl | Origin$ Battlefield | Destination$ Exile
T:Mode$ SpellCast | ValidCard$ Card.Self+kicked 2 | Execute$ TrigReturn | ...
SVar:TrigReturn:DB$ ChangeZone | ValidTgts$ Creature.OppCtrl | Origin$ Battlefield | Destination$ Hand
```

Vocab index: **205** (`src/card_vocab.h`).

A 2/2 Eldrazi Wizard for {1}{C} with **two independent optional kickers** —
"Kicker {G} and/or {1}{U}" (CR 702.33b: this is "Kicker {G}, kicker {1}{U}"). As
it's cast you may pay either, both, or neither:

- if kicked with its first ({G}) kicker → exile target artifact or enchantment an
  opponent controls;
- if kicked with its second ({1}{U}) kicker → return target creature an opponent
  controls to its owner's hand.

Both linked triggers fire if both kickers were paid (CR 702.33e/f).

## Implementation — a general Kicker mechanic

Kicker is implemented as a **reusable, multikicker-ready optional additional cost**
(CR 702.33 / 601.2b / 601.2f), not bolted onto the single-bool alt-cost path (that
path is for *replacement* alternative costs like Flashback/Evoke; kicker is
*additive* and there can be more than one).

### Data model

- `CardData::kicker_costs` — `std::vector<ManaValue>` (`src/components/carddata.h`).
  One mana cost per kicker; empty for a card with no kicker. Multikicker-ready.
- `Spell::kicked` — `std::vector<bool>` (`src/components/spell.h`). `kicked[i]` is
  true iff the (i+1)th kicker's additional cost was paid as this spell was cast.
  Internal per-Spell state only — **not** in the obs/state vector.
- `Ability::trigger_kicked_index` — `int` (`src/components/ability.h`). 0 = no
  kicker requirement; N ≥ 1 = the linked trigger requires `kicked[N-1]`.

### Parsing (`src/parse.cpp`)

- `K:Kicker:<c1>[:<c2>...]` keyword handler: splits the remaining segments on `:`
  and parses each as a mana cost into `card.kicker_costs` (Forge encodes "and/or"
  as the two colon-separated costs).
- Trigger `ValidCard$ Card.Self+kicked N`: `Card.Self` (prefix-matched, so it still
  matches the bare `Card.Self`) sets `valid_card_self`; `+kicked N` parses the 1-based
  index into `kicked_index`. `Mode$ SpellCast | ValidCard$ Card.Self...` maps to
  `trigger_on = SPELL_CAST`, `trigger_only_self = true`, and
  `trigger_kicked_index = kicked_index`. The index is carried through the `Execute$`
  SVar field-restore so the resolved effect ability keeps it.

### At cast (`src/action_processor.cpp`, regular-cost branch)

After the base cost (and any Offspring) is computed, for each kicker — only when its
extra mana is still affordable on top of everything chosen so far
(`can_pay_mana`) — the caster is asked an `OPTIONAL_YESNO` (`request_optional_yesno`).
An accepted kicker's mana is folded into the single `cost_to_pay` paid by
`prompt_mana_payment`, and the per-kicker flag is recorded in a local `kicked_flags`
vector that is copied onto `Spell::kicked`. So the kicker is genuinely optional and
its cost is paid alongside the rest of the spell's total cost (CR 601.2f–h).

### Trigger matching (`src/systems/state_manager_triggers.cpp`)

A "When you cast this spell" trigger fires while the spell is **on the stack**, not
the battlefield, so the battlefield trigger scan never sees it. A dedicated scan over
`SPELL_CAST` events walks the cast spell's own `CardData` abilities for a
`trigger_only_self` SPELL_CAST trigger and, when `trigger_kicked_index > 0`, gates it
on `Spell::kicked[N-1]`. Matching triggers are queued like any other (APNAP ordering,
target selection at placement). General over any self-cast / `kicked N` SpellCast
trigger.

## Testing (test harness, `--play`)

Battlefield: A controls Ancient Tomb ({C}{C}) + Forest + Islands; B controls Null Rod
(artifact) + Birds of Paradise (creature).

- **Unkicked** (decline both): spell cast, no triggers, a 2/2 enters.
- **Kicker 1 only** ({G}): exile trigger fires — only the artifact/enchantment is a
  legal target; "Null Rod is moved to exile". Bounce trigger does not fire.
- **Kicker 2 only** ({1}{U}): bounce trigger fires — only the creature is a legal
  target; "Birds of Paradise is moved to hand" (opp hand 7→8). Exile trigger does not
  fire.
- **Both kickers**: both triggers fire (an ORDER_TRIGGERS prompt for the two), each
  with its own target; Null Rod exiled and Birds of Paradise bounced.
- Each kicker prompt is offered as Decline/Accept and declining fires nothing.

No non-fatal errors, no draws.
