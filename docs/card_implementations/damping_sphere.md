# Damping Sphere  (vocab index 277)

## Oracle text
If a land is tapped for two or more mana, it produces {C} instead of any other type and amount.

Each spell a player casts costs {1} more to cast for each other spell that player has cast this
turn.

## Forge script
- Source: `bin/resources/cardsfolder/d/damping_sphere.txt`
- Type: `Artifact`, mana cost `2`.
- Key tags:
  - `R:Event$ ProduceMana | ActiveZones$ Battlefield | ValidCard$ Land | ManaAmount$ GE2 |
    ReplaceWith$ ProduceC` with `SVar:ProduceC:DB$ ReplaceMana | ReplaceMana$ C` — a
    **replacement effect** (CR 614.1): when a land is tapped for ≥2 mana, replace its
    production with the same total amount of `{C}`.
  - `S:Mode$ RaiseCost | Activator$ Player | Type$ Spell | Amount$ X | Relative$ True` with
    `SVar:X:Count$ThisTurnCast_Card.YouCtrl` — a **relative cost-increase** static (CR 601.2f):
    each spell costs {1} more per other spell its caster has cast this turn.

No tags were retagged or repurposed. The `ProduceMana` replacement is keyed on that event's
intended meaning and the relative `RaiseCost` reuses the existing cost-modification path.

## Engine work

### 1. General `ProduceMana` replacement effect (the new mechanic)
A new replacement-event type was added to the existing replacement subsystem (CR 614/616),
mirroring the structure of the SKIP_UNTAP / ENTERS_TAPPED replacements.

- **`Effect::Replacement::PRODUCE_MANA`** (`src/components/effect.h`) with fields
  `produce_valid_type` (the `ValidCard$` type filter, e.g. "Land"), `produce_min_amount` (the
  `ManaAmount$ GEN` threshold, e.g. 2), and `produce_replacement_color` (the `ReplaceMana$`
  color the production is converted to, e.g. COLORLESS).
- **Parser** (`src/parse.cpp`, `parse_replacement_effects`): `Event$ ProduceMana` +
  `ManaAmount$ GEN` + `ValidCard$ <type>` + a `ReplaceWith$` SVar whose body is `DB$ ReplaceMana`
  (the `ReplaceMana$` letter mapped to a `Colors`) builds the replacement. The SVar body is
  inspected rather than the `R:` line being retagged (the same pattern Dauthi/Ba Sing Se use).
- **`ReplacementEvent::PRODUCE_MANA`** (`src/systems/replacement_effects.h`): input
  `produced_color` / `produced_amount` (+ the producing `entity` and its controller), output
  the (possibly rewritten) `produced_color` and a `mana_replaced` idempotency flag.
- **Dispatcher** (`src/systems/replacement_effects.cpp`): `collect()` gains a PRODUCE_MANA
  branch — when the producing permanent's type matches `produce_valid_type` and the amount is
  ≥ `produce_min_amount`, it returns one candidate (and only one — identical replacements yield
  the same {C} production, so they collapse, keeping the dispatch choice-free so no input is read
  during mana-payment simulation). `apply_one()` sets `produced_color` and `mana_replaced`.
- **Mana production hook** (`src/mana_system.cpp`, `activate_mana_source`): before the produced
  mana enters the pool, when a source yields ≥2 mana it dispatches a PRODUCE_MANA event and adds
  the (possibly replaced) color instead of the ability's native color. This runs in BOTH commit
  and simulate mode, so affordability (`can_pay_mana`) and the real payment agree on the colors.
  The activation narrative now prints the *produced* (post-replacement) color.

### 2. Relative `RaiseCost` surcharge (the spell tax)
The fixed-amount `RaiseCost` was already supported; the **relative** form (`Amount$ X |
Relative$ True`, `X = Count$ThisTurnCast_Card.YouCtrl`) was not and was added generally.
- **`StaticAbility::raise_cost_per_spell_cast`** (`src/components/static_ability.h`).
- **Parser** (`src/parse.cpp`, `parse_static_abilities`): a non-numeric `Amount$` SVar on a
  `RaiseCost` static whose body references `ThisTurnCast` sets the flag.
- **Cost query** (`src/systems/state_manager_statics.cpp`, `active_raise_cost_for`): now takes the
  casting player and, for a `raise_cost_per_spell_cast` static, adds that player's
  `spells_cast_this_turn` (the "other" spells already cast — the new spell isn't counted yet).
  `effective_base_cost` forwards the caster, so affordability and payment share the surcharge.

## Behavioral decisions (made in lieu of asking a human)
- **`ManaAmount$ GE2` → applies at ≥2 mana; `activate_mana_source` only dispatches when a source
  yields ≥2.** No printed `ProduceMana` replacement triggers below 2 mana, so single-mana taps
  (every basic land) skip the battlefield scan entirely — basics keep their normal colored output.
- **Amount preserved, only the color/type replaced.** The script's `ReplaceMana$ C` converts the
  produced mana to colorless of the same total amount (a Tron land making 3 → `{C}{C}{C}`; a
  Gaea's Cradle making `{G}{G}` → `{C}{C}`). This matches the oracle "that much {C}".
- **Relative surcharge counts spells already cast this turn** (the "other" spells), evaluated at
  cost-computation time before the current spell increments the counter — so the Nth spell of a
  turn costs {N-1} more. Applies to *each* player's spells (`Activator$ Player`).
- **Multiple Damping Spheres collapse** to one application (idempotent {C} conversion), avoiding
  a 616.1 choice prompt during mana simulation.

## Tests
Isolation (`train/test_harness.py`), seed 1:
- **Color conversion (visible).** Damping Sphere + Gaea's Cradle + 2 creatures on A's board;
  A casts a second Damping Sphere ({2}) → `activated Gaea's Cradle for 2(C)` (normally green),
  spell cast. PASS.
- **Control (no Sphere).** Same Gaea's Cradle board without Damping Sphere → `activated Gaea's
  Cradle for 2(G)`. PASS — proves the replacement, not a constant.
- **Tron lands (task case).** Damping Sphere + assembled Tron; A casts Ensnaring Bridge ({3}) →
  Urza's Mine / Power Plant each `activated … for 2(C)` (colorless preserved), bridge cast. PASS.
- **Basic land unaffected.** Damping Sphere + Mountain; A casts Lightning Bolt → `activated
  Mountain for 1(R)` (single-mana producer untouched). PASS.
- **Spell tax.** Damping Sphere + 5 Mountain; A casts two Lightning Bolts at Player B → first
  taps 1 Mountain ({R}), second taps 2 ({R}{1}, +1 surcharge). PASS.
- **Regression.** `delver` vs `mav` scripted full game completes normally (no non-fatal errors).

No draws, no non-fatal errors / asserts in any run.

## Result
implemented
