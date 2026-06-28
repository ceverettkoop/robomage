# Urza's Workshop (vocab index 261)

## Oracle text
{T}: Add {C}.
Metalcraft — {T}: Add {C} for each Urza's land you control. Activate this ability only if you control three or more artifacts.

(Land — Urza's)

## Forge script
Source: pre-existing local script at `bin/resources/cardsfolder/u/urzas_workshop.txt`.

```
A:AB$ Mana | Cost$ T | Produced$ C | SpellDescription$ Add {C}.
A:AB$ Mana | Cost$ T | Activation$ Metalcraft | Produced$ C | Amount$ X | SpellDescription$ ...
SVar:X:Count$Valid Urza's.Land+YouCtrl
```

Key tags: two `AB$ Mana` abilities sharing a `{T}` cost; the second is gated by
`Activation$ Metalcraft` and scales by `Amount$ X` where `X = Count$Valid Urza's.Land+YouCtrl`.

## Engine work
The Metalcraft activation gate already exists (`game_queries.h` `activation_condition_met`
→ `controller_has_metalcraft`, shared with Mox Opal). Two general gaps had to be closed:

- `src/components/ability.cpp` — `identical_activated_ability()` did not compare
  `activation_condition` or `dynamic_amount_expr`, so when the permanent's abilities are
  copied from `CardData` (`state_manager_statics.cpp`), the Metalcraft ability was deduped
  away as "identical" to the plain `{T}: Add {C}` (same category/cost/color, both amount 0
  pre-default). Added both comparisons: a condition-gated ability and a differently-scaled
  ability are now correctly distinct. (General — any card with two same-shape mana abilities
  differing only by gate or dynamic amount.)
- `src/mana_system.cpp` — `eval_mana_amount()` only special-cased the literal string
  `Count$Valid Creature.YouCtrl` (Gaea's Cradle). Generalized it to evaluate any
  `Count$Valid <filter>` by counting the controller's battlefield permanents that match the
  filter through the shared `permanent_matches_filter()` (full qualifier grammar: subtype
  head `Urza's`, type qualifier `Land`, `YouCtrl`). Covers Gaea's Cradle and Urza's
  Workshop with one path.

No retagging; no card-script edits.

## Behavioral decisions (CR cites)
- Metalcraft (CR 702.46): the gated ability is only offered when the controller has 3+
  artifacts. Urza's Workshop itself is a Land (not an artifact), so it does not count toward
  its own gate.
- The Metalcraft ability's yield equals the number of Urza's lands the controller has on
  the battlefield (tapped or not — count is over all matching permanents, CR 109.5).

## Tests (`train/test_harness.py`)
Preset battlefield, single payment, using a `{3}` generic artifact (Icon of Ancestry) to
force the gated ability (basic {C} alone is insufficient):
- **2 Urza's Workshop + 3 Expedition Map (3 artifacts → Metalcraft met):** paying {3} logs
  "activated Urza's Workshop for 2(C)" — the Metalcraft ability produces 2 (count of Urza's
  lands), and the {3} spell resolves.
- **2 Urza's Workshop + 2 Expedition Map (2 artifacts → Metalcraft NOT met):** the {3}
  spell is never offered/castable — only the basic {C} (2 total) is available, confirming the
  gate blocks the scaled ability.
- Basic ability: separately observed paying generic costs "for 1(C)".

Regression: scripted full games, `temp/urza_a` (Urza's Workshop + Expedition Map + Grim
Monolith + Lightning Bolt + Grizzly Bears + Mountain) vs `temp/wbc_b`, seeds 2 and 5 — both
end decisively (Player A wins on opponent deck-out), no draws, no non-fatal errors.

## Result
Implemented. Both mana abilities work; Metalcraft gate and the per-Urza's-land scaling are
correct. Fixed two general engine gaps (ability dedup over-collapsing gated abilities;
`Count$Valid` mana-amount generalization). Build clean; regression clean.
