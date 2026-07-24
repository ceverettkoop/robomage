# Thespian's Stage

**Vocab index:** 335

## Oracle text

> {T}: Add {C}.
> {2}, {T}: Thespian's Stage becomes a copy of target land, except it has this ability.

## Forge script

Source: pre-existing local (`bin/resources/cardsfolder/t/thespians_stage.txt`). Key tags:

- `A:AB$ Mana | Cost$ T | Produced$ C` — the {C} mana ability (already handled).
- `A:AB$ Clone | Cost$ 2 T | ValidTgts$ Land | GainThisAbility$ True` — the in-place copy
  ability (the new mechanic).

## Engine work

**Mechanics added (general): `becomes-copy-in-place` (AB$ Clone, CR 706.2, layer-1 copy applied
to an existing permanent) + `GainThisAbility$`.**

- `src/effects/effect_kind.h` / `effect_kind.cpp` — new `EffectKind::Clone`, mapped from category
  `Clone` (distinct from `CopyPermanent`, which spawns a token copy).
- `src/effects/effect_table.cpp` — dispatch `Clone` → `clone`.
- `src/effects/effects.h` — `clone` declaration.
- `src/effects/effect_copy_permanent.cpp` — the `clone` handler
  (`effect_copy_permanent.cpp:clone`): the SOURCE permanent (Thespian's Stage) becomes a copy of
  the target land IN PLACE — the same entity acquires the target's copiable characteristics. It
  overwrites the source's per-entity `CardData` with a value copy of the target's `CardData`
  (name, types, abilities, keywords, static abilities) and resets the permanent's derived state
  (`Permanent::name`/`types`, and clears `abilities`/`static_abilities`) so the next state-based
  pass re-derives the permanent's activated / mana / triggered / static abilities from the new
  characteristics (`apply_permanent_components` + `apply_land_abilities`). Non-copiable state —
  counters, tapped state, attachments — is deliberately left untouched (CR 706.2).
- `src/components/ability.h` / `src/parse.cpp` — new `bool gain_this_ability` field, parsed from
  `GainThisAbility$ True`. The clone handler captures the source's own `Clone` ability(ies) BEFORE
  the overwrite and appends them onto the copied characteristics, so the copy retains
  "…except it has this ability."

## Behavioral decisions

- The copy is modeled by overwriting the permanent's per-entity `CardData` (which is already a
  value copy — each card entity owns its own `CardData` component) rather than layering a copy
  effect each pass. Thespian's Stage's Clone is a one-shot activated ability whose result persists
  until the permanent is re-cloned or leaves, so a persistent overwrite is the correct model and
  requires no continuous re-application.
- Because copiable values do NOT include counters (706.2) and Dark Depths' ice counters come only
  from its `etbCounter` (an ETB-only effect), a Stage cloning Dark Depths becomes a Dark Depths
  with ZERO ice counters — enabling the Marit Lage combo.
- After cloning a land with subtype-derived mana (e.g. Forest) the copy gains that mana ability
  through `apply_land_abilities`; after cloning a land with no mana ability (Dark Depths) the copy
  has none (its scripted {C} was cleared with the rest of Thespian's Stage's characteristics). The
  retained Clone ability is the only thing carried across.

## Tests

Isolation / combo (test_harness `--play` with seat keys):

- **Marit Lage combo (end-to-end, the key test):** preset `Thespians Stage, Dark Depths` on A's
  battlefield + Islands. Activate the Stage's `{2}{T}` Clone targeting Dark Depths →
  "Thespian's Stage becomes a copy of Dark Depths." The cloned Stage is a Dark Depths with 0 ice
  counters → the Mode$ Always state trigger (see `dark_depths.md`) fires → "Player A sacrifices
  Dark Depths" → **Marit Lage [20/20] [Flying, Indestructible]** token created, attacks for 20.
  The ORIGINAL Dark Depths `[ice:10]` is unaffected (only the copy triggered).
- **GainThisAbility / mana change:** preset `Thespians Stage, Forest` + Islands. Clone the Forest →
  the Stage becomes a Forest and the menu still offers "Activate Forest (Clone)" (the Clone ability
  is retained), while its printed {C} mana ability is replaced by the Forest's subtype-derived {G}.

CI gate: `ci_check.py --tier pygen,vocab,smoke` after all three cards.

**Result: implemented.**
