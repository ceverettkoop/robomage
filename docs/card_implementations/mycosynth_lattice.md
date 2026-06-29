# Mycosynth Lattice

**Vocab index:** 290
**Types:** Artifact — mana cost {6}
**Script:** `bin/resources/cardsfolder/m/mycosynth_lattice.txt` (pre-existing local Forge script)

## Oracle
- All permanents are artifacts in addition to their other types.
- All cards that aren't on the battlefield, spells, and permanents are colorless.
- Players may spend mana as though it were mana of any color.

## Forge script (key tags)
```
S:Mode$ Continuous | Affected$ Permanent | AddType$ Artifact
S:Mode$ Continuous | Affected$ Card | SetColor$ Colorless | AffectedZone$ All
S:Mode$ ManaConvert | ManaConversion$ AnyType->AnyColor
```
Three global continuous statics, none controller-restricted. The first two are CR 613 layer
effects (type-changing / color-changing); the third is a mana-spending permission (CR 609.4 /
106.6). Every handler added here is **general** over the tag's meaning, not Mycosynth-specific.

## Rules references
- **CR 613.1d — Layer 4 (type-changing).** "All permanents are artifacts in addition to their
  other types." adds a card type to every permanent additively.
- **CR 613.1e / 612 — Layer 5 (color-changing).** A color-changing effect overrides an object's
  color; "are colorless" sets the color set to empty (CR 105.2c).
- **CR 609.4 / 106.6 — Spending restrictions/permissions.** "Players may spend mana as though it
  were mana of any color." lets any mana pay any colored pip.

## Implementation

### Parse (`src/parse.cpp`, `parse_one_static_ability`)
New tags read into `StaticAbility` (`src/components/static_ability.h`):
- `SetColor$` → `set_color` (the colorset spec, e.g. `Colorless`).
- `AffectedZone$` → `affected_zone` (`All`, or a specific zone list).
- `ManaConversion$` → `mana_conversion` (e.g. `AnyType->AnyColor`); paired with the new
  `Mode$ ManaConvert` category.
No tag is repurposed; each new key is honoured as written.

### Mechanic 1 — global additive type static (general, reusable)
**`apply_global_addtype_statics(entities)` — `src/systems/state_manager_statics.cpp`** (called from
`apply_type_changing_effects`, layer 4). For each active `Continuous` static whose `AddType$` names
card/super/sub types and whose `Affected$` is a **general permanent filter** — explicitly **not**
the `Self` self-animate (`apply_self_animate_statics`), **not** the `Land.nonBasic` land-subtype
setter (Blood Moon's `RemoveLandTypes$` form), **not** `AllNonBasicLandType` — the named types are
added to **every** matching battlefield permanent in addition to its existing types
(`permanent_matches_filter` with the static's controller/source as the `YouCtrl`/`.Other`
references). `Affected$ Permanent` matches every permanent.
- **Self-cleaning.** Each pass first strips the types this subsystem recorded on each permanent
  (`Permanent::static_added_types`, new) and clears the record, then re-adds for the currently-
  active statics. So the grant lapses the instant Mycosynth leaves the battlefield, with no
  per-source bookkeeping leaking. Only **genuinely-new** types are recorded (a permanent that is
  already an artifact is left untracked and never has its printed Artifact type erased).
- **Ordering.** Runs **after** the existing land-subtype changer so a land whose subtypes were just
  reset (Blood Moon) still gains its global Artifact type. Runs in both the early-return
  (no land changers) and normal paths.
- Reuses `parse_self_added_types` (TYPE/SUBTYPE/SUPERTYPE classifier) and the shared
  `affected_is_general_filter` predicate.

### Mechanic 2 — global SetColor color override (general, reusable)
**`setcolor_override_for(e, out)` — `src/systems/state_manager_statics.cpp`** (declared in
`src/game_queries.h`). Scans `g_active_statics` for an active `Continuous` `SetColor$` static whose
`AffectedZone$` + `Affected$` filter designate `e`; on a match it writes the override color set
into `out` and returns true. `parse_setcolor_spec` maps the spec to a color set: `Colorless`
(or empty) → empty set (CR 105.2c); an explicit `White`/`Blue`/… list → those colors (so the
handler is general, not Colorless-only). `setcolor_zone_matches` honours `AffectedZone$`
(`All` = every zone; empty = battlefield default; otherwise a zone list).
- **Integration seam.** `effective_colors(e)` (`src/game_queries.cpp`) — the single live-color
  accessor that all "read a target's color" sites already route through (protection-from-color
  targeting, the `.Blue`/`Colorless` filter qualifiers) — now consults the override first, in
  every zone. `is_colorless_entity(e)` (`src/game_queries.h`) likewise consults it. So a global
  `SetColor$ Colorless` makes every color-dependent query see the affected card as colorless.
- **Recursion-safe.** The filter is matched against the object's **printed** characteristics
  (`card_matches_filter`), never `permanent_matches_filter` (which would recurse
  `effective_colors → setcolor_override_for → effective_colors`). Tokens (no `CardData`) are not
  matched by an `Affected$ Card` static and are skipped (documented limitation below).

### Mechanic 3 — spend-mana-as-any-color (general, reusable)
**`any_mana_as_any_color_active()` — `src/systems/state_manager_statics.cpp`** (declared in
`src/systems/state_manager.h`). True while an active `ManaConvert` static carries
`ManaConversion$ AnyType->AnyColor`. Consulted in the mana-payment path (`src/mana_system.cpp`):
- **`pay_from_pool`** (the single pool-spend primitive behind `can_afford_pool` / `spend_mana` /
  `pay_partial`): when active, colored pips are paid exact-color first (on-color mana is never
  wasted), then any still-unpaid colored pip is paid from any remaining mana — the generic-pip
  rule extended to colored pips.
- **`auto_pay_mana`** (the single algorithm behind both `can_pay_mana` legality and the machine
  payer): the colored-pip source-matching drops its exact-color gate, so any mana source can be
  tapped to pay a colored pip. Pips are kept **colored** (not rewritten to generic) so Delve /
  Improvise — which reduce only generic costs — never touch them.
- **`can_afford_with_sources`** (activation-cost affordability): under the effect, colored
  obligations are counted against the total available mana like generic.

The total pip count (and thus mana value) is never changed; only *what mana may pay each colored
pip* changes, exactly as CR 609.4 / 106.6 describes.

## Behavioral decisions / limitations (documented)
- **Color override scope.** The override applies to real cards (`CardData`) in any zone the static
  reaches, including spells on the stack and battlefield permanents. **Tokens** are not made
  colorless by the `Affected$ Card` SetColor clause (a token is not a card, CR 111.1, and the
  recursion-safe matcher needs `CardData`). Mycosynth's other clause still makes tokens artifacts.
  No current-vocab interaction depends on a token's color under Mycosynth.
- **CardData-only color overloads unchanged.** `card_colors(cd)` / `is_colorless_card(cd)` take a
  bare `CardData` with no entity, so they cannot consult an entity-keyed static and continue to
  report printed color. All entity-aware seams (`effective_colors`, `is_colorless_entity`) honour
  the override; the targeting/protection checks route through those.
- **First-match wins** for `setcolor_override_for` / single global type pass — no current vocab
  stacks two competing global SetColor or AddType statics, so the layer's full 613.7 timestamp
  ordering across two such statics is not exercised.

## Tests (test harness, `train/test_harness.py`)
- **AddType (positive):** `Mycosynth Lattice` preset on A's battlefield; A casts **Abrade** and
  picks "Destroy target artifact" — the menu offers the opponent's **Forest** (now an artifact),
  and casting destroys it. **Control (no Mycosynth):** the same line yields "No valid modes —
  charm fizzles" (the Forest is not an artifact, and Abrade's other mode targets a creature).
- **SetColor (positive):** with `Mycosynth Lattice` on A's battlefield, A casts **Red Elemental
  Blast** targeting the opponent's blue **Flying Men** → "No valid modes — charm fizzles" (Flying
  Men is now colorless, so the "destroy target blue permanent" mode has no legal target and there
  is no blue spell to counter). **Control (no Mycosynth):** REB's destroy mode targets Flying Men
  and destroys it.
- **ManaConvert (positive):** A controls only Mountains plus `Mycosynth Lattice`; **Flying Men**
  ({U}) is offered as "Cast Flying Men" and is cast by tapping a Mountain for {R}. **Control (no
  Mycosynth):** "Cast Flying Men" is never offered (0 occurrences) — {U} is unpayable with red.
- **Regression:** scripted full games, two decks each running Mycosynth Lattice + Abrade / Red
  Elemental Blast / Flying Men / Grizzly Bears / lands, seeds 1, 2, 3, 7 — every game finished
  decisively (no draws), with Mycosynth actually cast and entering, players tapping off-color
  lands for spells, and **no** non-fatal errors (only pre-existing cosmetic
  `Unrecognized ability param` lines).

## Result
**Implemented at full fidelity.** Three reusable handlers added — global additive AddType
(`apply_global_addtype_statics`), global SetColor color override (`setcolor_override_for`,
integrated into `effective_colors` / `is_colorless_entity`), and spend-mana-as-any-color
(`any_mana_as_any_color_active`, integrated into the mana-payment path). Build clean; all three
statics demonstrated positively and against a no-Mycosynth control; scripted regression green.
