# Sylvan Safekeeper

Vocab index: 238 (uid `sylvan_safekeeper`)

## Oracle

Sylvan Safekeeper — {G} — Creature — Human Wizard — 1/1

"Sacrifice a land: Target creature you control gains shroud until end of turn."

## Forge script source + key tags

`bin/resources/cardsfolder/s/sylvan_safekeeper.txt`:

```
A:AB$ Pump | Cost$ Sac<1/Land> | ValidTgts$ Creature.YouCtrl | KW$ Shroud | ...
```

Key tags:
- `AB$ Pump` — activated ability; the Pump effect grants a keyword (no P/T change here).
- `Cost$ Sac<1/Land>` — activation cost is sacrificing one land.
- `ValidTgts$ Creature.YouCtrl` — targets a creature you control.
- `KW$ Shroud` — the keyword granted to the target until end of turn.

## Engine work

Triage found the activation cost (`Sac<1/Land>`), the Pump targeting, and the `KW$`
keyword-grant plumbing (`effect_pump.cpp` → `grant_keywords`, which merges the keyword
into `Creature::keywords` and the `eot_keywords` cleanup bucket) were ALL already
supported. The gap was that **shroud was never enforced as a targeting restriction** —
`Ability::is_legal_target` (`src/components/ability.cpp`) consulted only protection and the
"Other" self-target restriction, so granting shroud was a functional no-op.

Implemented a general targeting restriction for **Shroud (CR 702.18)** and
**Hexproof (CR 702.11)**, wired into the single target-legality predicate so it covers
both target enumeration (`build_valid_targets`) and re-verification at resolution
(`is_target_valid`):

- **Shroud (CR 702.18b/702.18e):** "This permanent/player can't be the target of spells or
  abilities." Blocks ALL targeting — the controller's own spells/abilities and opponents'
  alike.
- **Hexproof (CR 702.11b):** "This permanent can't be the target of spells or abilities your
  opponents control." Blocks targeting only when the targeting source's controller is an
  opponent of the candidate's controller; the candidate's own controller may still target it.

The check reads the candidate's **effective** keyword set, not the printed list, so a keyword
granted at runtime (Pump grant, continuous effect, or keyword counter) is honored. To avoid
duplicating the effective-keyword logic, a shared accessor `permanent_has_keyword(Entity,
const char*)` was added to `src/game_queries.h` next to the existing `is_indestructible`
helper (which already reads the same effective keyword set: a creature's
`Creature::keywords`, else the printed `CardData`/`Token` keywords). The new check in
`is_legal_target` sits immediately after the protection check:

```cpp
if (permanent_has_keyword(cand, "Shroud")) return false;
if (permanent_has_keyword(cand, "Hexproof") &&
    global_coordinator.entity_has_component<Permanent>(cand) &&
    global_coordinator.GetComponent<Permanent>(cand).controller != caster)
    return false;
```

`caster` is the controller of the targeting spell/ability; the candidate's controller is its
`Permanent::controller`.

## Behavioral decisions

- The restriction is placed in `Ability::is_legal_target`, the single source of truth for
  target legality, so enumeration and re-verification cannot drift. This means a creature that
  gains shroud while a spell/ability targeting it is already on the stack will cause that
  spell/ability to be countered as illegal on resolution (CR 608.2b), matching the rules.
- Shroud/hexproof are read from the live (effective) keyword set via the shared
  `permanent_has_keyword` accessor, so the restriction applies whether the keyword is printed,
  granted by Pump (this card), granted by a continuous effect, or carried by a keyword counter.
- Hexproof is implemented generally even though Sylvan Safekeeper grants only Shroud, since the
  predicate is the natural home for both and hexproof is a printed/grantable keyword in vocab
  (keyword counters include `Hexproof`).
- Scope: the keyword check covers battlefield permanents (where granted keyword sets live).
  Player-shroud/hexproof via static keyword grants on players is not in current vocab and is
  out of scope.

## Tests

Built `make HEADLESS=TRUE` clean. Isolation tests via `train/test_harness.py --play`
(Birds of Paradise used as the creature, since `grizzly_bears.txt` is not present in the
on-disk cardsfolder):

- **Shroud blocks opponent's targeted removal →** Sylvan Safekeeper sacrifices a Forest,
  ability resolves granting Birds of Paradise shroud ("Birds of Paradise gains Shroud until
  end of turn."). Opponent then casts Lightning Bolt; its target menu offers
  `Player A`, `Player B`, `Sylvan Safekeeper` but **NOT Birds of Paradise** (correctly
  excluded). PASS.
- **Shroud blocks the controller's own spell →** with Birds shrouded, Player A's own Lightning
  Bolt target menu offers `Player B`, `Player A`, `Sylvan Safekeeper` but **NOT Birds of
  Paradise** (shroud blocks all, including the controller). PASS.
- **Control (no shroud) →** Birds of Paradise with no shroud granted IS offered as a Lightning
  Bolt target. Existing targeting unaffected. PASS.

Regression (torch-free scripted full games):
- Existing decks `delver` vs `mav`, seeds 1 & 2 → decisive winner, no draws, no non-fatal
  errors (normal targeting/removal still works). PASS.
- Temp green deck with 4× Sylvan Safekeeper vs `mav`, seeds 1, 2, 3 → decisive winner each,
  no draws, no non-fatal errors; Sylvan Safekeeper observed cast and resolved in a real game.
  PASS. (Temp deck cleaned up.)

## Result

Implemented. Shroud is now a real, general targeting restriction (CR 702.18), as is Hexproof
(CR 702.11), enforced in the shared `is_legal_target` predicate via the new
`permanent_has_keyword` effective-keyword accessor. Sylvan Safekeeper's granted shroud now
correctly makes the target untargetable by all players until end of turn.
