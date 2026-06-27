# Option B — Explicit sub-ability target resolution

## The old sentinel

`Ability::resolve()` (`src/components/ability.cpp`) chains a spell/ability's sub-abilities
(`subabilities`, parsed from `SubAbility$`/`Execute$` SVar references). Before resolving each
sub it bound the sub's target with a blanket sentinel, at two sites (the condition-failed
fall-through and the normal chain):

```cpp
if (sub_ab.valid_tgts == "N_A") sub_ab.target = this->target;
```

i.e. **every** untargeted sub inherited the parent's target. That worked only because the
downstream effect handlers that reference an *independent* object (`Defined$ You`, `Opponent`,
`Remembered`, `Self`, ...) read their own `defined_*` flag and return before ever touching
`ab.target` — so overwriting `ab.target` was harmless for them but never *correct* by intent.
It was fragile: a future handler that consulted `ab.target` as a fallback would silently act on
the wrong object.

## The new explicit rule (CR 608.2c)

CR 608.2c: as a spell/ability resolves it follows its instructions in order, and each
instruction references the objects it acts on **by its own definition**. The Forge scripts
already encode that definition in the sub's `Defined$`. The two sites now call one shared
file-local helper, `bind_sub_target(const Ability& parent, Ability& sub)`:

| Sub declares | Binding |
|---|---|
| its own `ValidTgts$` (`valid_tgts != "N_A"`) — a target was chosen for it at cast/activation | keep `sub.target` untouched |
| `Defined$ {Targeted, ParentTarget, Parent, TargetedController}`, **or** no `Defined$` at all (legacy implicit inherit) | `sub.target = parent.target` (it references the parent's chosen target — or that target's controller/power) |
| any other explicit `Defined$` (`You` / `Opponent`(`Player.Opponent`) / `Remembered` / `Self` / `TriggeredActivator` / ...) | leave `sub.target` alone; the effect resolves its own `Defined` reference |

`TargetedController` is deliberately in the **inherit** bucket: the `DealDamage` / `GainLife` /
`ChangeZone` handlers for `defined_targeted_controller` read `ab.target` to find the targeted
permanent's controller (and, for `Targeted$CardPower`, its power), so the parent target must be
inherited for them. Putting it elsewhere would break Swords to Plowshares / Solitude / Smash to
Smithereens.

This is **behavior-preserving**: the inherit bucket reproduces exactly what the old sentinel
set, and the third bucket only *stops* writing a value those handlers never read.

## Parser change

`Ability` gained a general `std::string defined` (default empty), populated verbatim in
`apply_param_to_ability` (`src/parse.cpp`) from the `Defined$`/`DefinedPlayer$` param — the same
function sub-ability SVars route through (`parse_svar_ability` → `apply_param_to_ability`), so
sub-abilities carry their own `defined` token. The pre-existing specific `defined_*` bools are
kept and remain authoritative for their effects; `defined` is read only by `bind_sub_target`.

## Cards audited (test harness, scripted regression)

- **Cabal Therapy** (vocab 159) — regular and **flashback**: `DBDiscard` has its own
  `ValidTgts$ Player`, so it keeps its independently-chosen player target. Confirmed the chosen
  player reveals and discards all copies of the named card (both casts).
- **Smash to Smithereens** (168) — `DB$ DealDamage Defined$ TargetedController`: deals 3 to the
  destroyed artifact's controller. Confirmed both opponent's artifact (opponent takes 3) and own
  artifact (controller takes 3).
- **Swords to Plowshares** (48) — `DB$ GainLife Defined$ TargetedController` + `Targeted$CardPower`:
  the exiled creature's controller gains life equal to its power. Confirmed.
- **Scythecat Cub** (47) — landfall `Pump` (own `ValidTgts$`) → `DBPutCounter Defined$ Targeted`
  → `DBMultiplyCounter Defined$ Targeted`: the +1/+1 counter lands on the parent's chosen
  creature (2/2 → 3/3). Confirmed the explicit `Defined$ Targeted` inherit path.
- **Sheoldred's Edict** (163) — Charm modes `DB$ Sacrifice Defined$ Opponent`: each opponent
  sacrifices their own matching permanent. Confirmed the opponent (not the caster) is prompted.
- **Solitude** (141) — ETB exile + `GainLife Defined$ TargetedController`: behavior is identical
  to baseline (the "up to one target" ETB trigger path is unchanged by this refactor).
- **Scripted regression**: `mav` vs `delver` seeds 1/2/3 and `delver` vs `mav` seed 5 — all
  decisive, no draws, no non-fatal errors.

## Files

- `src/components/ability.h` — new `std::string defined` field.
- `src/parse.cpp` — populate `ability.defined` in the `Defined`/`DefinedPlayer` branch.
- `src/components/ability.cpp` — `bind_sub_target` helper + forward declaration; both former
  sentinel sites now call it.
