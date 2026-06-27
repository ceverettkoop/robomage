# Smash to Smithereens (vocab index 168)

## Oracle text
Destroy target artifact. Smash to Smithereens deals 3 damage to that artifact's controller.

## Forge script (Source: pre-existing local; key tags)
`bin/resources/cardsfolder/s/smash_to_smithereens.txt`

- `A:SP$ Destroy | ValidTgts$ Artifact | SubAbility$ DBDealDamage` — destroy the targeted artifact.
- `SVar:DBDealDamage:DB$ DealDamage | Defined$ TargetedController | NumDmg$ 3` — chained
  sub-ability deals 3 damage to the controller of the parent spell's target (the artifact).

No retagging was done; the engine was extended to honor the script's real tags.

## Engine work (general)
The `Destroy` effect already worked, and the parser already parsed `Defined$ TargetedController`
into `Ability::defined_targeted_controller` (used by GainLife / ChangeZone). The gap was that
`DealDamage` did not honor that flag, so the 3-damage rider misfired.

- `src/effects/effect_deal_damage.cpp`, `effects::deal_damage()`: added a general
  `Defined$ TargetedController` branch. The DealDamage sub-ability inherits the parent's target
  (the artifact) as `ab.target` via the sub-ability chaining in `Ability::resolve()`
  (`src/components/ability.cpp`, the `if (sub_ab.valid_tgts == "N_A") sub_ab.target = this->target;`
  rule). The branch resolves the target's controller — `Zone::controller` while on the
  battlefield, falling back to `Permanent::controller` — and deals the damage to that player.
  This mirrors the existing `defined_targeted_controller` handling in `effect_gain_life.cpp`
  (Swords to Plowshares) and the `defined_each_opponent` / `defined_you` plumbing already in
  this file. It is general: any `DealDamage` with `Defined$ TargetedController` is covered.

### Last-known-information for the controller
The `SP$ Destroy` main effect moves the artifact to the graveyard before the chained
`DB$ DealDamage` runs. Both run synchronously inside one `Ability::resolve()`, so no
state-based-action pass intervenes to strip the `Permanent` component between them — the
target's controller is still readable when the damage is dealt. `Orderer::add_to_zone` does
not clear `Permanent::controller`, so reading the controller off the just-destroyed artifact
yields its last-known controller (CR 608.2g/h): the effect's instructions are followed in the
written order, and "that artifact's controller" refers to the artifact's controller as it last
existed.

## Behavioral decisions (CR cites)
- CR 608.2g/h: a spell's instructions are carried out in order; an object referenced after it
  has left the zone uses last-known information. Here the destroy and the damage are one
  resolution, and the controller is read from the (now-destroyed) target's last-known state.
- CR 109.5 / object-controller semantics: the damage is dealt to the controller of the target,
  which may be the caster (when targeting one's own artifact) — not always an opponent.

## Tests
Verified with `train/test_harness.py` (semantic `--play`):
- (a) Opponent controls Aether Spellbomb; Player A casts Smash to Smithereens targeting it:
  the artifact is destroyed AND the opponent drops from 20 to 17 (exactly 3), Player A unchanged.
- (b) Player A targets its OWN Aether Spellbomb: the artifact is destroyed AND Player A drops
  from 20 to 17, opponent unchanged — proving the damage follows the target's controller.
- Regression: scripted vs scripted full games, seeds 1/2/3, red deck (4x Smash to Smithereens +
  Lightning Bolt + Abrade) vs an artifact deck (Aether Spellbomb / Mishra's Bauble / Null Rod /
  Aether Vial). All games resolved with a winner (Player A), no draws, zero non-fatal errors.

## Result
Implemented. `Defined$ TargetedController` is now honored generally by the DealDamage effect.
