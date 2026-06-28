# Lavaspur Boots

```
Name:Lavaspur Boots
ManaCost:1
Types:Artifact Equipment
K:Equip:1
S:Mode$ Continuous | Affected$ Creature.EquippedBy | AddKeyword$ Haste & Ward:1 | AddPower$ 1 | Description$ Equipped creature gets +1/+0 and has haste and ward {1}.
Oracle:Equipped creature gets +1/+0 and has haste and ward {1}.\nEquip {1}
```

**Oracle:** Equipped creature gets +1/+0 and has haste and ward {1}. Equip {1}.

Vocab index: **224** (`src/card_vocab.h`).

Forge script source: pre-existing local script
`bin/resources/cardsfolder/l/lavaspur_boots.txt` (unchanged).

## Engine work

Most of the card's clauses are covered by existing handlers (proven by Cori-Steel Cutter, which
also uses an `Affected$ Creature.EquippedBy` static with `AddPower$` + `AddKeyword$ Trample &
Haste`):

- **Equip {1}** — the `K:Equip:1` keyword activated ability (`src/action_processor.cpp` equip
  flow + `effects::attach`).
- **EquippedBy +1/+0** — `S: Affected$ Creature.EquippedBy | AddPower$ 1`, applied by the
  EquippedBy static-layer appliers (`src/systems/state_manager_statics.cpp` / layers).
- **Granted Haste** — `AddKeyword$ … Haste …`, merged into the equipped creature's effective
  `Creature::keywords` by `add_keywords_from_spec`; the declare-attackers eligibility check
  already honors a granted Haste keyword.

### The gap fixed: granted Ward (CR 702.21)

Ward granted via a continuous effect was **not honored**. `trigger_ward_for_targets`
(`src/action_processor.cpp`) read **only** the target's printed `CardData::ward_cost`, so a Ward
granted by an equipment/aura static (or a Pump grant / keyword counter) never fired — an
opponent could target the equipped creature for free.

Storage of the two ward forms (kept distinct so they are not double-counted):

- **Printed ward** (`K:Ward:N`): `parse.cpp` stores the numeric cost in `CardData::ward_cost`
  (with `ward_is_life` for Ward—Pay N life) and pushes the **bare** string `"Ward"` onto
  `CardData::keywords`.
- **Granted ward** (`AddKeyword$ … Ward:1 …`): `add_keywords_from_spec` pushes the raw spec part
  `"Ward:1"` (colon + number) onto the effective keyword list — never the bare `"Ward"`.

Fix — a new shared helper `collect_ward_instances(Entity)` (`src/action_processor.cpp`) returns
every Ward ability a permanent currently has:

- the printed instance from `ward_cost` (carrying `ward_is_life`), and
- every granted instance parsed out of a `"Ward:N"` string in the effective keyword list (the
  same effective-keyword view as `permanent_has_keyword`: a creature's rebuilt
  `Creature::keywords`, else printed `CardData`/`Token` keywords). `N` is parsed as the generic
  mana cost; a bare granted `"Ward"` with no number defaults to {1}.

`trigger_ward_for_targets` now loops over these instances, pushing one Ward "counter unless pay"
trigger per instance — **reusing the same `unless_generic_cost` / `unless_cost_is_life`
pay-unless plumbing the printed ward already used** (no parallel payment path).

Dedup: the bare `"Ward"` keyword (the printed marker) is skipped in the keyword scan, so a
printed ward is counted exactly once (via `ward_cost`); and identical granted copies collapse,
so a single granted `Ward:1` fires exactly once. Distinct ward costs (e.g. a hypothetical
printed Ward 2 plus a granted Ward 1) each yield their own instance and each trigger, consistent
with CR 702.21h (a permanent can have multiple instances of Ward).

## Behavioral decisions

- A granted `Ward:N` is treated as **{N} generic mana** (the script form for granted ward),
  matching Lavaspur Boots' "ward {1}". Granted Ward—Pay-life is not produced by any current
  script form, so the granted path is mana-only; the printed path still carries `ward_is_life`.
- Ward fires only when the targeting spell/ability is controlled by an opponent of the warded
  permanent's controller (CR 702.21b) — the existing `is_battlefield_permanent(tgt, opp)` guard
  is unchanged.

## Tests (`train/test_harness.py`)

- **Equip grants +1/+0, haste, ward:** Lavaspur Boots equipped to Birds of Paradise →
  "Birds of Paradise gains 1/0 Haste & Ward:1"; Birds shows **[1/1]** (printed 0/1 + boots).
- **Granted Ward {1} — opponent declines:** opponent casts Lightning Bolt at the equipped Birds
  → "Ward {1}: … may pay to counter"; opponent picks "Don't pay" → **"Lightning Bolt is
  countered"** (Birds survives).
- **Granted Ward {1} — opponent pays:** opponent taps a Mountain and chooses "Pay {1}" →
  "Player A pays {1} — spell is not countered" → Bolt resolves, **"Birds of Paradise is
  destroyed (lethal damage)."**
- **Haste lets a turn-1 equipped creature attack:** Dragon's Rage Channeler (no native haste)
  equipped with Lavaspur Boots is offered in Declare Attackers on its controller's turn and
  **"deals 2 damage to Player B"** (1/1 + boots = 2/1).
- **Printed-ward regression (dedup):** opponent bolts a Kappa Cannoneer (`K:Ward:4`) → exactly
  one **"Ward {4}"** trigger fires (printed ward unaffected; not double-counted from the bare
  `"Ward"` keyword).
- **Real-game scripted regression:** `temp/lavaspur_delver` (delver list with 3 Lavaspur Boots)
  vs `mav`, scripted seeds 1/2/3 → Player A wins all three, no non-fatal errors, no draws;
  Lavaspur Boots casts and equips in real games (seed 2 equips Murktide Regent).

## Result

Implemented. Equip / EquippedBy +power / granted Haste reuse existing handlers; granted Ward is
now honored generally (CR 702.21) via `collect_ward_instances`, reusing the printed-ward
pay-unless plumbing. Printed ward (Kappa Cannoneer Ward {4}) verified unaffected.
