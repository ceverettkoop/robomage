# The Tabernacle at Pendrell Vale

**Vocab index:** 337

## Oracle text

> All creatures have "At the beginning of your upkeep, destroy this creature unless you pay {1}."

(The paper card reads "sacrifice"; the Forge script models it as `DB$ Destroy`, which is what the
engine implements.)

## Forge script

Source: pre-existing local (`bin/resources/cardsfolder/t/the_tabernacle_at_pendrell_vale.txt`).
Key tags:

- `S:Mode$ Continuous | Affected$ Creature | AddTrigger$ TabernacleTrig | AddSVar$ TabernacleDestroy`
  — the continuous static that grants a triggered ability to every creature (the new mechanic).
- `SVar:TabernacleTrig:Mode$ Phase | Phase$ Upkeep | ValidPlayer$ You | TriggerZones$ Battlefield |
  Execute$ TabernacleDestroy` — the granted trigger (each controller's upkeep).
- `SVar:TabernacleDestroy:DB$ Destroy | Defined$ Self | UnlessPayer$ You | UnlessCost$ 1` — destroy
  the creature unless its controller pays {1}.
- `NeedsToPlayVar` / `CountOpps` / `CountMe` are AI hints (ignored).

## Engine work

**Mechanics added (general): `grant-triggered-ability-to-all` (a Continuous static's
AddTrigger$/AddSVar$, CR 613.1f layer 6 / 603), plus destroy-unless-pay and Defined$ Self on the
Destroy effect.**

- `src/components/static_ability.h` — new `add_trigger` / `add_trigger_svar_name` /
  `add_trigger_svar` fields holding the resolved granted-trigger line and its Execute$ SVar.
- `src/parse.cpp` — `parse_one_static_ability` now parses `AddTrigger$` / `AddSVar$`, resolving
  each named SVar to its body.
- `src/parse.h` / `src/parse.cpp` — new `parse_granted_trigger(trigger_line, svar_name, svar_body)`
  helper: builds the minimal SVar table the trigger's Execute$ needs and runs the shared
  `parse_one_trigger`, so the granted trigger honors the full trigger grammar
  (Mode$/Phase$/ValidPlayer$/Execute$ + the DB$ effect body).
- `src/systems/state_manager_statics.cpp` — new Phase 3 in `apply_layer6_ability_grants`: for each
  active Continuous static with a non-empty `add_trigger`, parse the granted trigger once and
  attach a per-recipient copy (tagged `granted_by_static`, `source = recipient`) to every creature
  the `Affected$` filter designates. Rebuilt from scratch each pass (Phase 1 strips all
  `granted_by_static` abilities), so the grant appears/disappears as creatures and the source
  enter/leave. The trigger scan already reads `perm.abilities`, so the granted trigger fires like
  any innate one.
- `src/effects/effect_destroy.cpp` — the Destroy effect now:
  - resolves `Defined$ Self` (binds the effect's own source as the target — the creature that has
    the granted trigger), and
  - runs the unless-cost payment loop when `unless_generic_cost > 0` (`run_unless_loop`, MANA kind
    for {1}, also honoring the life/discard/energy variants): the payer (UnlessPayer$ You ⇒ the
    ability's controller) may pay to prevent the destruction; suspendable on the payment decision.

## Behavioral decisions

- The grant reuses the existing layer-6 ability-grant infrastructure
  (`apply_layer6_ability_grants` / `ability_grant_targets` / `affected_permanents_for_static`),
  which already fans an `Affected$ Creature` filter across every creature of BOTH players. The
  granted trigger is controller-relative: `ValidPlayer$ You` gates it to the recipient's
  controller's upkeep, and `Defined$ Self` makes the tax act on that recipient — so each player
  pays for (or loses) their own creatures on their own upkeep.
- Modeled as `DB$ Destroy` per the Forge script (the paper "sacrifice" wording), reusing the
  general destroy + unless-pay primitives rather than a card-specific path.

## Tests

Isolation (test_harness `--play` with seat keys): preset The Tabernacle + a Grizzly Bears on A,
a Grizzly Bears on B, plus lands to pay {1}.

- **A's upkeep:** the granted trigger fires only for **A's** Grizzly Bears (not B's). Paying {1}
  (tap Island) → "Player A pays {1}" and the bear survives.
- **Decline:** choosing "Don't pay" at A's upkeep → "Grizzly Bears is destroyed" (to A's
  graveyard); B's Grizzly Bears is untouched.
- **B's upkeep (next turn):** the granted trigger fires for **B's** Grizzly Bears, and B gets its
  own "may pay {1} to prevent destruction" decision — confirming the tax fires for every creature
  during its own controller's upkeep.

CI gate: `ci_check.py --tier pygen,vocab,smoke` after all three cards.

**Result: implemented.**
