# Toxicrene (vocab index 265)

## Oracle text
Reach, deathtouch
All lands have "{T}: Add one mana of any color" and lose all other abilities.

(Creature — Tyranid, 2/4, `{3}{G}`)

## Forge script
Source: pre-existing local script at `bin/resources/cardsfolder/t/toxicrene.txt`.

```
K:Reach
K:Deathtouch
S:Mode$ Continuous | Affected$ Land | AddAbility$ Mana | RemoveAllAbilities$ True | Description$ ...
SVar:Mana:AB$ Mana | Cost$ T | Produced$ Any | Amount$ 1
```

Key tags: one `Continuous` static over `Affected$ Land` that BOTH `AddAbility$ Mana`
("{T}: Add any color") AND `RemoveAllAbilities$ True` (lose all OTHER abilities).

## Engine work
The `AddAbility$` grant already fans out to a general `Affected$ Land` filter via
`affected_permanents_for_static` (so lands got the any-color mana ability already). Two changes
in `src/systems/state_manager_statics.cpp` made `RemoveAllAbilities$` work for a typed filter
and coexist with the same static's grant (general — benefits any "all X have '…' and lose all
other abilities" card, not just Toxicrene):

- `removal_affects()` previously only recognized `Affected$ Creature` (Humility) or an empty
  filter. Generalized it to run the Affected$ filter through the shared `permanent_matches_filter`,
  so `Affected$ Land` (and any grammar) is honored.
- `recompute_abilities()` full-removal previously did `abilities.clear()`, which would also wipe
  the any-color ability the same static just granted (layer-6 grants run before this post-layer-7
  pass). Changed it to drop every ability **except** one granted by a remover that affects this
  permanent — i.e. the remover's own "…and gains X" clause survives its own removal (CR 613:
  "have [this ability] and lose all OTHER abilities"). A pure remover that grants nothing
  (Humility) preserves nothing, so its behavior is unchanged.

No retagging; no card-script edits. Reach/Deathtouch are printed keywords (already supported).

## Behavioral decisions (CR cites)
- The static affects **all** lands, both players' (CR 109.5 unscoped reference). Under Toxicrene
  every land taps for any one color and loses its intrinsic/printed abilities (including a basic
  land's subtype-derived mana and a nonbasic's activated abilities).
- "Lose all OTHER abilities" + "have '{T}: Add any color'" from one static: the granted mana
  ability is kept; everything else on the land is removed (CR 613 layer 6, same-static
  grant-survives-removal). The removal is reversible — once Toxicrene leaves, lands re-derive
  their normal abilities on the next SBA pass.

## Tests (`train/test_harness.py`)
- **Grant works / survives removal**: with Toxicrene + a Forest in play, casting Lightning Bolt
  `{R}` taps the Forest for `R` ("activated Forest for 1(R)") — the Forest produces a color it
  never normally could, proving the any-color grant is present after the removal pass.
- **Removal works**: a Prismatic Vista shows "Activate Prismatic Vista (ChangeZone)" (its fetch)
  with no Toxicrene; with Toxicrene on the battlefield that activated ability is **gone** from the
  menu (only the granted any-color mana remains).

Regression: scripted full games, `temp/toxi_a` (Toxicrene + Grizzly Bears + Lightning Bolt +
Forest/Mountain) vs `temp/witch_a`, seeds 1 and 2 — decisive (Player A wins), Toxicrene resolves
and attacks, no draws, no non-fatal errors.

## Result
Implemented. Lands lose all other abilities and gain "{T}: Add any color"; the grant survives the
same static's removal. Generalized `removal_affects` (typed Affected$ filter) and made
full-removal preserve a same-static grant. Build clean; regression decisive with no errors.
