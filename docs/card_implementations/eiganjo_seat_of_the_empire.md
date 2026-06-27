# Eiganjo, Seat of the Empire

Card UID: `eiganjo_seat_of_the_empire` (vocab index **201**)

```
Name:Eiganjo, Seat of the Empire
ManaCost:no cost
Types:Legendary Land
A:AB$ Mana | Cost$ T | Produced$ W | SpellDescription$ Add {W}.
A:AB$ DealDamage | PrecostDesc$ Channel — | Cost$ 2 W Discard<1/CARDNAME> | ActivationZone$ Hand | ValidTgts$ Creature.attacking,Creature.blocking | TgtPrompt$ Select target attacking or blocking creature | NumDmg$ 4 | ReduceCost$ X | SpellDescription$ It deals 4 damage to target attacking or blocking creature. This ability costs {1} less to activate for each legendary creature you control.
SVar:X:Count$Valid Creature.Legendary+YouCtrl
```

Oracle: `{T}: Add {W}.` plus a **Channel** ability — `{2}{W}, Discard Eiganjo, Seat of
the Empire`: deal 4 damage to target attacking or blocking creature. The ability costs `{1}`
less to activate for each legendary creature you control.

## What already worked

Everything except the cost reduction was already in the engine:

- The `{W}` mana ability (basic activated mana ability).
- The Channel activation **from hand** (`ActivationZone$ Hand`) — the engine already enumerates
  and processes activated abilities whose `activation_zone == Zone::HAND`.
- `DealDamage` (`NumDmg$ 4`), the `Discard<1/CARDNAME>` self-discard activation cost
  (`discard_self_cost`), and the `ValidTgts$ Creature.attacking,Creature.blocking` target
  enumeration (live combat-state qualifiers via the shared filter matcher).

## The gap: `ReduceCost$` on an *activated* ability

The only missing piece was `ReduceCost$ X` on an activated ability, where `X` is the count-SVar
`Count$Valid Creature.Legendary+YouCtrl`. The engine had a continuous **static** `ReduceCost`
(Eye of Ugin / It That Heralds the End), but an *activated ability's* own `ReduceCost$` param was
neither parsed nor applied — only a never-reached literal path existed conceptually.

CR 601.2f (and 118.7): cost reductions reduce the **generic** portion of a cost; colored mana
symbols are never reduced. So Channel goes `{2}{W}` → `{1}{W}` → `{W}` (floored) as you control
0 / 1 / 2+ legendary creatures, with `{W}` always required.

## Implementation — a general, reusable path

Built as a general activated-ability cost-reduction path, not a one-card special case.

1. **New field** `Ability::reduce_cost_expr` (`src/components/ability.h`) — the raw `ReduceCost$`
   value: a literal integer string (`"1"`) or a runtime `Count$`/SVar expression.

2. **Parse** (`src/parse.cpp`):
   - `apply_param_to_ability` stores the `ReduceCost$` value verbatim into `reduce_cost_expr`.
   - The `parse_abilities` post-pass expands a single SVar key (`X`) to its `Count$` expression
     (e.g. `Count$Valid Creature.Legendary+YouCtrl`), exactly like `amount_svar`/`CounterNum$`
     SVar resolution. A literal integer is left as-is.

3. **Evaluation** — `effective_activation_mana_cost(ab, controller, orderer)` in
   `src/mana_system.cpp` (declared in `src/mana_system.h`). When `reduce_cost_expr` is set it
   resolves the amount through the shared `evaluate_dynamic_amount` (literal ints route through
   `evaluate_sa_svar`; `Count$Valid …` counts battlefield permanents), then removes up to that
   many `GENERIC` symbols from `activation_mana_cost` — floored at 0 (the erase loop simply stops
   when generic runs out), colored pips untouched. This is the **single source of truth** used by
   both the legality gate and the payment so the two can never diverge.

4. **Count$Valid generalization** (`src/components/ability.cpp`, `evaluate_dynamic_amount`): the
   generic `Count$Valid <Filter>` branch now routes the **whole** filter spec through the shared
   `permanent_matches_filter` (`game_queries`) instead of matching only the head type. So
   `Creature.Legendary+YouCtrl` honors the `Legendary` supertype and `YouCtrl` control qualifier
   (the old code stripped at the first `.` and counted *all* creatures). Control is enforced by the
   filter's `YouCtrl`/`OppCtrl`, so the loop no longer pre-filters by controller — making it correct
   for OppCtrl filters too. Existing callers (e.g. Eldrazi Linebreaker `Count$Valid Eldrazi.YouCtrl`)
   are unaffected.

5. **Call sites** now compute the effective (post-reduction) cost:
   - Affordability / legality: `src/systems/state_manager_actions.cpp` — both the battlefield
     activated-ability path and the `ActivationZone$ Hand` path.
   - Payment: `src/action_processor.cpp` — the from-hand activation, the equip path, and the
     generic activated-ability path (the latter two are no-ops without a `ReduceCost$`).

   Mana-ability paths in `mana_system.cpp` are untouched: mana abilities never carry a
   `ReduceCost$`, so routing them through the helper would be a no-op.

No script was edited or retagged. `PrecostDesc$`/`TgtPrompt$` are cosmetic and ignored.

## Test evidence (`train/test_harness.py` scenarios)

Eiganjo in hand; A controls Plains + legendary creatures; A attacks with its own creature and
Channels it (any attacking/blocking creature is a legal target).

| Legendary creatures you control | Cost paid | Plains tapped |
|---|---|---|
| 0 (Scythecat Cub attacker) | `{2}{W}` | **3** |
| 1 (Super Shredder) | `{1}{W}` | **2** |

In every case: Eiganjo is discarded as the cost, 4 damage is dealt to the target, the creature
dies, and the `{W}` pip is always required (a Plains is always tapped for white).

**Affordability gate (same reduced cost as payment):** with only 2 Plains available —
- 0 legendaries → cost `{2}{W}` (3) > 2 mana → Channel is **not offered** (correctly gated out).
- 1 legendary → cost `{1}{W}` (2) = 2 mana → Channel **is offered**, paid with 2 Plains, resolves.

This confirms legality and payment use the identical reduced cost.

The 2+/floor case (`{W}`, count ≥ 2 → generic floors to 0) is guaranteed structurally by the
clamping erase loop (`while (reduction > 0 && it != cost.end())`) and by the linear 0→1 results;
a runtime 2-legendary scenario could not be staged because the only comma-free legendary creature
with a present script in the vocab is Super Shredder (the harness comma-splits battlefield names,
and a second copy of the same legend is removed by the legend rule).

Regression: scripted delver-vs-mav full games (seeds 1–3) complete with wins, no draws, no
fatal/non-fatal errors, no new filter warnings.
