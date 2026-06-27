# Whipflare (vocab index 169)

## Oracle text
Whipflare deals 2 damage to each nonartifact creature.

(Sorcery — `{1}{R}`)

## Forge script
Source: pre-existing local script at `bin/resources/cardsfolder/w/whipflare.txt`.

```
A:SP$ DamageAll | NumDmg$ 2 | ValidCards$ Creature.nonArtifact | ValidDescription$ each nonartifact creature. | SpellDescription$ CARDNAME deals 2 damage to each nonartifact creature.
```

Key tags:
- `SP$ DamageAll` — spell ability of the new `DamageAll` effect kind.
- `NumDmg$ 2` — amount of damage dealt to each matching permanent (consumed by the
  existing `parse_deal_damage` hook → `Ability::amount`).
- `ValidCards$ Creature.nonArtifact` — the filter spec (consumed by the existing
  `parse_destroy_all` hook → `Ability::valid_cards_filter`).
- `ValidDescription$` / `SpellDescription$` — cosmetic prose, ignored (in the parser's
  `ignored_keys`).

There is no `ValidTgts$`, so the spell has no chosen target; it resolves over the whole
battlefield like `DestroyAll`/`SacrificeAll`.

## Engine work
A new **general, reusable `DamageAll` effect**, modeled on the existing `DestroyAll`
effect:

- `src/effects/effect_damage_all.cpp` — new handler `effects::damage_all`. Scans
  `orderer->mEntities`, and for every battlefield permanent matching the `ValidCards$`
  filter via the shared `permanent_matches_cards_filter()`, marks `NumDmg$` damage on it
  with the global `::deal_damage(source, e, amount)`. Damage is marked on every matching
  creature first; the lethal-damage state-based action then destroys any creature that
  took lethal damage (and respects Indestructible, since that lives in the SBA / does not
  apply to damage marking — only to the destruction). Players are never targeted, so no
  life loss occurs.
- Registrations mirroring `DestroyAll`:
  - `src/effects/effect_kind.h` — added `DamageAll` to the `EffectKind` enum.
  - `src/effects/effect_kind.cpp` — added `{"DamageAll", EffectKind::DamageAll}`.
  - `src/effects/effect_table.cpp` — added `case EffectKind::DamageAll: return &damage_all;`.
  - `src/effects/effects.h` — declared `bool damage_all(...)`.
- The new effect file is picked up automatically by the Makefile's `$(wildcard
  $(SRCDIR)/*/*.cpp)` glob; no Makefile edit was needed.
- Param parsing reuses existing hooks: `NumDmg$` via `parse_deal_damage`, `ValidCards$`
  via `parse_destroy_all`. No retagging — the script's real `DamageAll`/`NumDmg`/
  `ValidCards` tags are honored as written.

### Shared filter: generic `non<Type>` negation
`permanent_matches_cards_filter()` (in `src/effects/effect_put_counter_all.cpp`, shared by
PutCounterAll/SacrificeAll/ImmediateTrigger and now DamageAll) previously failed-closed on
any qualifier it didn't recognize, so `Creature.nonArtifact` would have matched nothing.
Added a **general** `non<Type>` qualifier: any qualifier of the form `non<TypeName>`
(other than the already-special-cased `nonLand`/`nonToken`) excludes a permanent that
carries that type/subtype (`permanent_has_type`). This covers `nonArtifact`,
`nonCreature`, etc. for every mass effect, not just Whipflare.

## Behavioral decisions (CR cites)
- The damage to each matching creature is dealt **simultaneously** (CR 119; a single
  "deals N damage to each ..." event). The handler marks damage on all matching creatures
  before any SBA pass, so the simultaneity is preserved and lethal is checked once for all.
- Lethal damage destroys a creature via the state-based action (CR 704.5g: a creature with
  lethal damage marked is destroyed), respecting Indestructible (CR 702.12) in the SBA, not
  in the damage step.
- "each nonartifact creature" affects creatures on **both** sides (CR 109.5: an unscoped
  object reference is not limited to a single player's permanents). Artifact creatures are
  excluded by the `nonArtifact` filter qualifier.

## Tests (`train/test_harness.py`)
Scenario (single cast, preset battlefield), Whipflare `{1}{R}` cast off two Mountains:
- Player A battlefield: Mountain, Mountain, Birds of Paradise (0/1), Murktide Regent (3/3)
- Player B battlefield: Orcish Bowmasters (1/1), Kappa Cannoneer (4/4 Artifact Creature)
- `--play "A:keep,B:keep,A:cast:Whipflare,B:pass"`

Result:
- `Resolving ability (category: DamageAll, amount: 2)`.
- 2 damage dealt to Birds of Paradise, Murktide Regent, Orcish Bowmasters (both sides).
- Kappa Cannoneer (Artifact Creature) **unaffected** — not in the damage list.
- Birds of Paradise (t1) and Orcish Bowmasters (t1) destroyed by lethal-damage SBA.
- Murktide Regent (3/3) **survives** with 2 damage marked (`[3/3, 2dmg]`).
- Both players remain at 20 life (no player damage).

Regression (scripted full games, `--deck-a` mono-red w/ 2 Whipflare + 4 Lightning Bolt,
`--deck-b` Birds of Paradise + Dragon's Rage Channeler + Forest), seeds 1/2/3:
- Seed 1 → Player B wins; seeds 2/3 → Player A wins. Every game has a decisive winner
  (no draws), Whipflare's DamageAll resolves, and there are **zero** non-fatal errors.
- (Initial regression decks used Orcish Bowmasters as the small creature; its missing
  Orc Army token script produced pre-existing non-fatal errors unrelated to Whipflare, so
  the opponent deck was switched to token-free creatures.)

## Result
Implemented. New general `DamageAll` effect + generic `non<Type>` filter negation; build
clean (`make HEADLESS=TRUE`); card behaves per Oracle text; regression clean across three
seeds with no draws and no non-fatal errors.
