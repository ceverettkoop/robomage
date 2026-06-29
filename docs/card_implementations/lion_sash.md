# Lion Sash

**Vocab index:** 286
**Types:** Artifact Creature Equipment Cat — base P/T 1/1, mana cost {1}{W}
**Script:** `bin/resources/cardsfolder/l/lion_sash.txt`

## Oracle
- {W}: Exile target card from a graveyard. If it was a permanent card, put a +1/+1 counter on Lion Sash.
- Equipped creature gets +1/+1 for each +1/+1 counter on Lion Sash.
- Reconfigure {2} ({2}: Attach to target creature you control; or unattach from a creature. Reconfigure only as a sorcery. While attached, this isn't a creature.)

## Rules references
- **CR 702.151 — Reconfigure.** An Equipment keyword on a creature card. The reconfigure cost
  gives (a) an activated ability "attach to target creature you control" and (b) an activated
  ability "unattach", both usable only any time you could cast a sorcery (your main phase, empty
  stack). 702.151b: while a reconfigure permanent is attached, it is an Equipment and is **not**
  a creature (it loses its creature-ness).
- **CR 301 — Equipment / 301.5** attach mechanics; an Equipment whose host leaves the
  battlefield becomes unattached and stays in play (a reconfigure Equipment then becomes a
  creature again).
- **CR 110.4a** — permanent card types (artifact, battle, creature, enchantment, land,
  planeswalker), the test behind "if it was a permanent card".
- **CR 122 / 613 layer 7** — +1/+1 counters set P/T; Lion Sash (base 1/1) is (1+N)/(1+N) with N
  counters.

## Implementation
Reconfigure is parsed and driven entirely through the **existing Equip attach machinery**
(`equipped_to` / `equipped_by` links), with a small set of reconfigure-specific additions.

### Parsing — `src/parse.cpp`
`K:Reconfigure:<cost>` sets `CardData::is_equipment = true`, `CardData::equip_cost`, and the new
`CardData::is_reconfigure = true` flag (`src/components/carddata.h`). Being `is_equipment`, it
reuses the entire equip attach flow; `is_reconfigure` selects the extra behaviour below.

### Attach + Unattach actions — `src/systems/state_manager_actions.cpp`
The same sorcery-speed-gated `is_equipment` block that synthesises the Equip action now:
- labels the attach action "Reconfigure …" for a reconfigure card (still the `"Equip"` category
  internally, so the existing attach handler runs unchanged);
- excludes the source itself from the "is there a creature to attach to" check (a reconfigure
  permanent is itself a creature while unattached — `e2 == entity` skip; CR 301.5c);
- when the reconfigure permanent is already attached (`equipped_to != 0`), offers an
  `"Unattach"` activated ability at the same cost.

`src/action_processor.cpp` excludes the source from the equip target list (`e == permanent_entity`)
and adds the `"Unattach"` category handler: pay the cost, clear the `equipped_to`/`equipped_by`
link.

### "Isn't a creature while attached" — `src/systems/state_manager_statics.cpp`
In `apply_permanent_components` (the per-SBA permanent lifecycle), a reconfigure permanent with
`equipped_to != 0` is treated as **not a creature**: `is_creature` is forced false and its
`Creature` / `Damage` components are stripped. They are re-created on the next pass once it
unattaches (`equipped_to == 0`). This removes it from combat, creature targeting, and
creature-based state-based actions while attached, and restores it on unattach — no separate
flag is consulted at call sites.

### {W} graveyard-exile ability + conditional counter — script tags, reused effects
`A:AB$ ChangeZone | Origin$ Graveyard | Destination$ Exile | ValidTgts$ Card | RememberChanged$ True`
moves the targeted graveyard card to exile (existing ChangeZone effect) and remembers it; the
chained `DB$ PutCounter | ConditionDefined$ Remembered | ConditionPresent$ Permanent` puts a
+1/+1 counter on Lion Sash only if the remembered (now-exiled) card is a permanent card.

**General fix made for the condition:** the filter head `"Permanent"` was previously a pure
wildcard (no type-line requirement), so "if it was a permanent card" would have passed for an
instant/sorcery. `src/game_queries.cpp` now requires a permanent card type for the `"Permanent"`
head (`view_is_permanent`). This is a no-op for battlefield objects (always permanents) and
correctly excludes instants/sorceries when the filter matches a card in another zone.

## Tests (test harness)
- **{W} on a permanent card:** bolt a Grizzly Bears into the graveyard, activate {W} targeting
  it → exiled, +1/+1 counter, Lion Sash 1/1 → 2/2.
- **{W} on a nonpermanent:** exile Lightning Bolt (instant) from the graveyard → exiled, **no**
  counter, Lion Sash stays 2/2.
- **Reconfigure attach:** pay {2} in main phase to attach Lion Sash to your Grizzly Bears →
  Lion Sash displays with no P/T (not a creature) and cannot be declared as an attacker; only
  the host attacks.
- **Unattach:** pay {2} → Lion Sash returns to a 1/1 creature and can attack again.
- **Sorcery-speed gate:** while a spell is on the stack, the menu offers only Pass and the
  instant-speed {W} ability — Reconfigure is absent; it reappears in the main phase with an
  empty stack.
