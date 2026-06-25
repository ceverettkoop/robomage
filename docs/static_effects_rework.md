# Static Effects Rework

Status: **PLANNING** · Created 2026-06-24 · Scope: continuous effects (CR 611/613), static
abilities (CR 604), replacement effects (CR 614/616)

This document is the design backlog for closing the gap between the engine's current
static-ability / replacement-effect handling and the Comprehensive Rules. It collects:

1. [Miscategorized / ad-hoc effects and where each belongs](#1-miscategorized--ad-hoc-effects)
2. [Replacing ability re-derivation with a CR-compliant recompute](#2-ability-re-derivation-rework)
3. [Test scenarios](#3-test-scenarios)
4. [Humility implementation plan](#4-humility-implementation-plan)
5. [Dependency system (613.8) outline](#5-dependency-system-6138-outline)

Companion context: the T2.1 layer engine ([`layer-system`] memory) and the T2.2 replacement
dispatcher (`src/systems/replacement_effects.{h,cpp}`, `docs/rules_compliance_issues.md`).

---

## Background: the three pipelines (and the seam)

The engine routes continuous-ish rules text through **two** real subsystems, plus a scatter of
inline special cases that belong to neither:

| Pipeline | Rule | Driver | Owns |
|---|---|---|---|
| Replacement dispatcher | 614/616 | `replacement::dispatch(ReplacementEvent&)` | enters-tapped, enters-with-counters, exile-instead, dredge |
| Layer engine | 611/613 | `StateManager::apply_continuous_effects()` → per-layer appliers | type change (L4), keyword grant (L6, add-only), P/T (L7a/7c), MustAttack (613.11) |
| Inline special cases | 113.6g / 601 / 602 / 603 | checks at point-of-use | can't-be-countered, CantBeCast, CantBeActivated, DisableTriggers, RaiseCost |

Two structural facts drive everything below:

- **Characteristics are rebuilt from base each SBA pass for P/T but *accumulated* for abilities.**
  `gather_active_statics` resets `static_*_bonus` to 0 and `recompute_pt` rebuilds P/T from
  `base_power/base_toughness` every pass (`creature.cpp:13-35`) — so P/T effects are correctly
  "not locked in" (611.3a). Abilities are the opposite: `apply_permanent_components` copies them
  from `CardData` onto the `Permanent` **once** and they stick forever
  (`state_manager_statics.cpp:180-201`). This asymmetry is why "loses all abilities" is
  unimplementable today — see §2.
- **Layer 6 only adds; it never removes.** `apply_layer6_ability_effects`
  (`state_manager_statics.cpp:611-666`) grants keywords on a condition transition and removes only
  the keywords *it itself* granted. There is no general ability-removal path.

---

## 1. Miscategorized / ad-hoc effects

> **Re-scoped after code inspection (2026-06-24).** The original "dead field" premise here was
> largely **inaccurate** (it came from an over-eager survey). On inspection, most listed effects
> are already wired and working:
> - `adjust_land_plays` — consumed at `state_manager_actions.cpp:207` (Icetill Explorer). **Works.**
> - `may_play_from_graveyard` — consumed at `state_manager_actions.cpp:204-224`. **Works.**
> - untap-prevention (`hidden_keyword "doesn't untap"`) — the `game.cpp:107-135` path exists but
>   is **dead code**: no vocab card sets `hidden_keyword`. **Choke** actually uses
>   `R:Event$ Untap | Layer$ CantHappen` (a *replacement* effect), which the engine does **not**
>   handle — so Choke's untap-prevention is genuinely **unimplemented** (its Islands untap normally).
>   This is a real gap (a feature, not consolidation): properly fixing it means routing untap through
>   the replacement dispatcher (`ReplacementEvent::UNTAP`) and parsing the `R:Event$ Untap` line.
>   Deferred/offered separately — the §1 consolidation only relocated the (dead) `hidden_keyword`
>   query into `rules_mod::untap_prevented` (behavior-preserving).
> - can't-be-countered — cast-time flag at `action_processor.cpp:1063` → `effect_counter.cpp:61`. **Works.**
>
> So §1a/§1c are **not** bug fixes — they would be pure architectural consolidation (move scattered
> special cases into one `rules_modifying` query surface; route untap through the dispatcher; model
> can't-be-countered as a `CantBeCountered` static). No behavioral gain, real regression risk.
> **Deferred as optional cleanup.**
>
> **Implemented (2026-06-24)** — the two genuine fidelity items:
> - **§1d Blood Moon 305.7** — a land set to a basic type now loses its rules-text abilities
>   (keeping the regenerated intrinsic mana). Layer 4 records type-set lands + suppresses their
>   statics; `recompute_abilities` erases the rest. Verified: Magus of the Moon turns Gaea's Cradle
>   into a Mountain (Lightning Bolt castable off it; scripted "Add G for each creature" gone).
>   Corpus byte-identical (Magus is sideboard-only; the type-set path is dormant in corpus games).
> - **§1b honor-suppressed** — all 7 non-layer `g_active_statics` consumers (RaiseCost,
>   CantBeActivated ×2, CantBeCast, DisableTriggers, untap-prevention, land-plays) now skip
>   `suppressed` statics, so Humility (§4) removes prohibition/permission/cost statics from creature
>   sources (Collector Ouphe, Icetill Explorer, Doorkeeper Thrull) too. Inert for current vocab.
>
> **Consolidation (2026-06-24)** — Part A done; Part B reduced; Part C N/A:
> - **Part A** — new `src/systems/rules_modifying.{h,cpp}` (`rules_mod::` namespace) is the single
>   home for the prohibition/permission/cost static queries: `activation_prohibited` /
>   `mana_activation_prohibited` (CantBeActivated — dedups the two intentionally-different variants,
>   replacing `permanent_cant_activate`), `cast_prohibited` (CantBeCast), `land_play_bonus` +
>   `may_play_lands_from_graveyard`, `etb_triggers_suppressed` (DisableTriggers), `untap_prevented`.
>   Call sites in mana_system / state_manager_actions / state_manager_triggers / game.cpp now call
>   these. Behavior-preserving (corpus byte-identical). RaiseCost (`active_raise_cost_for`) left in
>   place (already a clean free function).
> - **Part B** — untap query relocated into `rules_mod::untap_prevented` (the `hidden_keyword`
>   path), but that path is dead (see the untap note above). Routing it through the replacement
>   dispatcher is the real fix and is deferred as a feature.
> - **Part C** — not applicable (can't-be-countered is correctly a cast-time `Spell` flag).
>
> Original plan (kept for context):

Each row is an effect that is currently mishandled, dead, or implemented off-pipeline, with its
correct CR classification and a target pipeline. "Target" is where it *should* live.

### 1a. Parsed-but-never-applied static fields (dead code)

| Effect (card) | Field | CR class | Target | Plan |
|---|---|---|---|---|
| Additional land plays (Icetill Explorer 37) | `adjust_land_plays` | rules-modifying static (305.2 / 613.11) | **rules-modifying static query** | Add `active_land_play_bonus(player)` that sums `adjust_land_plays` over `g_active_statics` controlled by `player`; consume it where the land-per-turn limit is enforced (search `lands_played` / `PLAY_LAND` legality in `state_manager_actions.cpp`). No characteristic layer involved. |
| Play lands from graveyard (Icetill Explorer 37) | `may_play_from_graveyard` | permission static (611.3d) | **permission static query** | Add `may_play_from(zone, player)` consulted by `PLAY_LAND`/cast legality so a matching graveyard card becomes playable. Mirror the existing legal-action enumeration; the card stays in the graveyard until actually played. |
| Doesn't untap (Choke 82 / `S:` `hidden_keyword`) | `hidden_keyword`, `affected_subtype` | **replacement effect** on the untap event (614.1d) | **replacement dispatcher** | This is the only *replacement* effect among the dead fields. Add an `UNTAP` event type (§1c). The `S:`-based `hidden_keyword` carrier is dead; route untap-skipping through the dispatcher and delete the unused fields if no other consumer appears. Confirm whether Choke is actually a parsed `R:` line or relies on this dead `S:` path first. |

### 1b. Inline static prohibitions / cost modifiers (off-pipeline but functionally working)

These are **not** characteristic effects, so they correctly stay out of the 7-layer pipeline —
but they are scattered, each with its own ad-hoc loop over `g_active_statics` or its own flag. The
rework unifies their *dispatch*, not their location in the layer order. They are "rules-modifying
continuous effects" (613.11) and cost/permission rules (601/602).

| Effect (card) | Category | Current site | Plan |
|---|---|---|---|
| Raise cost (Disruptor Flute 91) | `RaiseCost` | `active_raise_cost_for()` (`state_manager_statics.cpp:37-51`) | Already centralized as a query — keep, but move under a single `rules_modifying.{h,cpp}` namespace alongside the others so all 613.11/cost statics share one home and one condition-evaluation pass. |
| Can't activate (Null Rod 71, Collector Ouphe 31, Disruptor Flute 91) | `CantBeActivated` | `state_manager_actions.cpp:433-450`, mana exemption `mana_system.cpp:100-110` | Wrap in `is_activation_prohibited(entity, ability)` query reading `g_active_statics`; keep the mana-ability exemption (605.1a). |
| Can't cast (filtered, per-turn limit) | `CantBeCast` | `state_manager_actions.cpp:334-349` | Wrap in `is_cast_prohibited(card, player)`; keep `NumLimitEachTurn` per-player bookkeeping. |
| Disable ETB triggers (Doorkeeper Thrull) | `DisableTriggers` | `state_manager_triggers.cpp:147-165` | Wrap in `triggers_suppressed_for(entity, cause, mode)`. |
| Must attack | `MustAttack` | `apply_rules_modifying_effects()` (`:738-746`) | Already in the layer driver as a 613.11 step — this is the model the others should follow. Leave as-is; it anchors the new `rules_modifying` grouping. |

**Net of 1b:** introduce one `rules_modifying` query surface (a thin header of free functions over
`g_active_statics`) so prohibitions/cost statics stop being copy-pasted scans. No behavior change —
this is a consolidation that makes adding the next prohibition a one-liner and gives can't-be-countered
(1c) a home.

### 1c. True outliers — not modeled as `StaticAbility` / `ReplacementEvent` at all

| Effect (card) | What it is now | CR class | Target | Plan |
|---|---|---|---|---|
| Can't be countered (Long Goodbye, Cavern of Souls) | cast-time `Spell.cant_be_countered` bool (`action_processor.cpp` → `effect_counter.cpp:59-67`); misnamed `Effect::Replacement::CANT_BE_COUNTERED` enum carrier | static prohibition (113.6g/701.5) **or** a cast-time granted ability (611.2f for Cavern) | **rules-modifying static query** (1b) for the printed-on-permanent case; keep the cast-time grant for Cavern | Correctly **not** a replacement effect — do **not** route to the dispatcher. Model the permanent-sourced prohibition as a `CantBeCountered` rules-modifying static queried at the counter site, replacing the ad-hoc bool. Cavern's "the spell you cast can't be countered" is a 611.2f cast-time grant — keep a per-`Spell` flag for that, but set it via the unified query, not a bespoke path. |
| Untap skipping (Choke) | dead `hidden_keyword` `S:` field (1a) | replacement effect (614.1d) | **replacement dispatcher** | Add `ReplacementEvent::UNTAP`. The untap step turn-based action (search the UNTAP step handler in `main.cpp` / step advance) builds one `UNTAP` event per permanent-about-to-untap and calls `dispatch`; an applicable "doesn't untap" effect sets `ev.skip_untap = true`. This makes Choke, Winter Orb-style effects, and any future "doesn't untap" share the 614/616 choose-one machinery. |

### 1d. Incomplete layer-4 (Blood Moon / 305.7)

Not miscategorized — correctly in the layer engine — but **incomplete**, and it is the headline
correctness bug, so it's tracked here. `apply_type_changing_effects` (`:468-547`) resets land
subtypes and regenerates *mana* abilities, but per **305.7** a land whose subtype is set to a basic
type must also **lose all abilities generated from its rules text** (non-mana activated abilities
included). Because abilities are re-derived from `CardData` every pass (§2), a Blood-Mooned Gaea's
Cradle / Karakas / Wasteland keeps its printed activated ability — wrong. **The fix is §2** (once
abilities are recomputed from base each pass, layer 4 strips printed abilities the same way it
strips subtype-derived mana abilities). Also note **Blood Moon itself is not in `card_vocab.h`** —
only Magus of the Moon (86), which has identical text; add Blood Moon to vocab when convenient, but
Magus is the testable stand-in today.

---

## 2. Ability re-derivation rework

> **Implemented (2026-06-24).** Landed behavior-preserving: corpus **byte-identical** (0/108
> games differ) and clean build. The implementation refined the plan below — it did **not** add
> `base_abilities` to `Permanent`; instead it uses the immutable `CardData`/`Token` as the base
> and makes removal *reversible by re-derivation*. Concretely:
> - **Keywords rebuild-from-base each pass:** `gather_active_statics` resets `cr.keywords` to the
>   printed base (`CardData.keywords`, or `Token.keywords`; transformed DFCs keep their back-face
>   set), mirroring the existing `static_*_bonus` reset. `apply_layer6_ability_effects` re-grants
>   every pass (keywords are no longer sticky) with transition-only logging. Dead
>   `remove_keywords_from_spec` removed.
> - **Static-ability removal = suppression flag, not vector erase:** `g_active_statics` holds raw
>   pointers into `perm.static_abilities`, so removal can't erase that vector. New
>   `ActiveStatic::suppressed`; `suppress_removed_statics()` (driver phase 1, right after gather)
>   marks statics whose source loses its abilities; layers 4/6/7 all `if (a.suppressed) continue`.
> - **Activated/triggered/keyword erase = `recompute_abilities()`** (driver phase 2, after layer 7
>   so layer-4 mana regen can't undo it): clears `perm.abilities` + `cr.keywords` for affected
>   permanents. Re-derived next pass by `apply_permanent_components`/`apply_land_abilities`/the
>   keyword reset, so removal ends cleanly when the effect leaves.
> - **New carrier:** `StaticAbility::remove_all_abilities` + `RemoveAllAbilities$ True` parsing.
>   Inert until §4 (Humility) / §1d (Blood Moon 305.7) set it.
> - **Known limitations (deferred):** (a) same-layer grant-vs-removal *timestamp* ordering
>   (Humility + anthem, 613.7) — removal currently always wins, runs after grants → §5; (b)
>   `activations_this_turn` resets for an ability erased and re-derived while a remover is active
>   (corner case: ActivationLimit + a remover flickering mid-turn); (c) non-layer consumers of
>   `g_active_statics` (RaiseCost/CantBeActivated/CantBeCast/DisableTriggers) don't yet honor
>   `suppressed` — fold into §1b (no current-vocab creature sources such a static, so inert).
>
> Original plan (kept for context):

**The problem.** Abilities are *accumulated and sticky*: `apply_permanent_components` copies
activated/static abilities from `CardData` onto the `Permanent` exactly once (guarded by
`already_present` / `static_abilities.empty()`), and `apply_land_abilities` /
`apply_keyword_abilities` add idempotently. Nothing ever rebuilds the ability list from scratch, so:

- "Loses all abilities" (Humility) cannot suppress anything — removed abilities are re-added next pass.
- Blood Moon can't strip printed non-mana abilities (305.7) for the same reason.
- A granted ability that should end when its source leaves (611.3a "not locked in") lingers.

**The principle (611.3a / 613).** Effective characteristics — *including abilities* — must be
**recomputed from base every SBA pass**, exactly as P/T already is. P/T is the working model:
`gather_active_statics` zeroes the deltas, layer appliers re-add, `recompute_pt` rebuilds from base.
Abilities need the identical lifecycle.

### Design

Split each permanent's abilities into **base** (printed + granted-at-creation) and **effective**
(what the rest of the engine reads), and rebuild effective each pass through the layers:

1. **`Permanent` gains `base_abilities` and `base_static_abilities`** (the printed set, populated
   once at ETB from `CardData`, plus genuinely permanent grants like a resolved Aura's static).
   `Permanent::abilities` / `static_abilities` become the *computed* output, rebuilt each pass.
   (Equivalently: keep `abilities` as base and add `effective_abilities` — pick whichever touches
   fewer read sites; most readers want effective.)

2. **New driver step `recompute_abilities()`** run inside `apply_continuous_effects`, positioned at
   **layer 6** (after layer 4 type changes, before layer 7 P/T — abilities can change what a CDA in
   7a reads). Per permanent:
   1. Start from `base_abilities` + subtype-derived mana abilities for its *current* (post-layer-4)
      land subtypes + keyword-derived triggered abilities for its *current* keywords.
   2. Apply **layer-6 add** effects (existing keyword grants; future ability grants).
   3. Apply **layer-6 remove** effects (Humility "loses all abilities"; Blood Moon 305.7 printed-
      ability strip for type-set lands; "can't have abilities").
   4. Write the result to the computed ability list.

3. **Mana / keyword / land re-derivation moves *inside* this recompute** so it is ordered correctly
   relative to removal. Today `apply_land_abilities` (`:378`) and `apply_keyword_abilities` (`:419`)
   run in `apply_permanent_components` (before the layer engine) and in layer 4 — they must instead
   feed step 2.1 so that step 2.3 removal wins (Humility removing Dryad Arbor's Forest mana ability;
   Blood Moon removing a non-mana printed ability). This is the crux: **derivation must precede
   removal within layer 6.**

4. **Idempotence stops mattering** — because the list is rebuilt from base, the `already_present`
   guards and the "never cleared" accumulation go away. Removal is just "don't re-add."

### Touch list

- `src/components/permanent.h` — add `base_abilities`, `base_static_abilities`; document
  `abilities`/`static_abilities` as computed.
- `src/systems/state_manager_statics.cpp` — populate base sets once at ETB in
  `apply_permanent_components`; add `recompute_abilities()`; relocate `apply_land_abilities` /
  `apply_keyword_abilities` calls into it; have `apply_type_changing_effects` (305.7) and the new
  layer-6 removal feed it.
- `src/systems/state_manager_layers.cpp` — call `recompute_abilities()` in the driver at the layer-6
  slot (the existing keyword-grant applier folds into it).
- Audit read sites of `Permanent::abilities` (activation legality, stack resolution, mana) — they
  should read the computed list, which is what they already read; only the *source of truth* changes.

### Risks / guards

- **Performance:** rebuilding ability vectors every SBA pass for every permanent. P/T already does
  this; ability lists are small. Acceptable; revisit only if profiling shows it.
- **Granted abilities with duration** (611.3d — "may play / gains an ability until end of turn"):
  those are *not* from a currently-present static, so they belong in `base_abilities` (or a
  duration-tracked grant list), not re-derived. Keep them out of the per-pass rebuild's removable set
  unless a layer-6 remove explicitly targets them.
- **Determinism / corpus:** this is a behavior-preserving refactor for all *current* vocab (no card
  removes abilities yet), so the regression corpus must stay **byte-identical**. Verify with
  `train/gen_corpus.sh` after the refactor and before Humility (§4) adds the first real remover.

---

## 3. Test scenarios

Two buckets: **(A) confirm current behavior / current bugs** (runnable now), and
**(B) post-implementation acceptance** (gated on §2/§4). All via `train/test_harness.py --play`
(see CLAUDE.md). Card availability confirmed in `card_vocab.h`: Magus of the Moon 86, Gaea's Cradle
34, Karakas 39, Wasteland 10, Dryad Arbor 32, Knight of the Reliquary 41, Barrowgoyf 87,
Grizzly Bears 3.

### A. Runnable now — characterize current behavior (expect some to FAIL → that's the bug)

1. **Blood-Moon strips printed land abilities (305.7) — expect CURRENT FAIL.**
   Magus of the Moon in play + a Gaea's Cradle on the same side. Gaea's Cradle should become a
   Mountain that taps only for `{R}` and *lose* "{T}: Add {G} for each creature you control".
   ```bash
   train/.venv/bin/python train/test_harness.py \
     --battlefield-a "Magus of the Moon,Gaea's Cradle,Grizzly Bears" \
     --hand-a "Mountain" --library-a "Mountain,Mountain,Mountain,Mountain,Mountain,Mountain,Mountain,Mountain" \
     --hand-b "Forest" --library-b "Forest,Forest,Forest,Forest,Forest,Forest,Forest,Forest" \
     --play "keep,keep" --max-decisions 12
   ```
   Inspect Gaea's Cradle's available mana abilities. **Today it likely still offers the G-for-each
   ability (bug). Target: only `{R}`.** Repeat with Karakas (its activated bounce should vanish) and
   Wasteland (its sacrifice ability should vanish).

2. **Knight's sacrifice cost respects live types under Blood Moon.**
   Magus of the Moon in play + Knight of the Reliquary + a *nonbasic* Forest/Plains dual. Under
   Magus the dual is a Mountain and must **not** be a legal sacrifice for Knight; only a basic
   Forest/Plains should be. Verify the `Sac<…/Forest;Plains/…>` filter reads post-layer-4 subtypes,
   not `CardData`.

3. **Knight P/T tracks graveyard lands (baseline, should PASS).**
   Knight of the Reliquary with N land cards milled to its controller's graveyard → Knight is
   (2+N)/(2+N). Confirms the layer-7c SVar path is intact before §2 touches recompute.

### B. Post-implementation acceptance (gated on §2 + §4)

4. **Humility makes a vanilla creature 1/1 (7b set).** Humility enchantment + Grizzly Bears (2/2) →
   Bears is 1/1.

5. **Humility + external pump survives (7c after 7b).** Humility + Grizzly Bears + a `+1/+1`
   counter → 2/2 (set to 1/1, then counter adds in 7c, not removed as an "ability").

6. **Humility + Knight of the Reliquary → 1/1, no graveyard bonus.** Knight's self-pump is its own
   ability, removed in layer 6 (before 7c), so the bonus disappears regardless of timestamps. With
   5 lands in graveyard Knight is still **1/1** under Humility (contrast: a +1/+1 counter on it would
   make it 2/2). This is the key cross-layer test — the current architecture would wrongly give
   1+N/1+N.

7. **Humility + Barrowgoyf (CDA) → 1/1.** Barrowgoyf's CDA (7a) is overridden by Humility's set
   (7b); ability removal (L6) also strips the CDA. Either way → 1/1.

8. **Humility + Dryad Arbor → 1/1 Land Creature that taps for nothing.** Dryad Arbor's only ability
   is its intrinsic Forest mana ability; Humility removes it (the recompute must order derivation
   *before* removal — §2 step 3). It stays a land, stays summoning-sick, but produces no mana.

9. **Choke/untap replacement.** A permanent under a "doesn't untap" effect skips its untap step;
   removing the effect lets it untap next turn. Drives `ReplacementEvent::UNTAP` (§1c).

10. **Dependency: Humility vs Opalescence (or a constructed pair).** Gated on §5 — see there.

Capture 4–9 as JSON scenarios under `train/` (and add the cards to a corpus deck's main list if
they should also be covered by `gen_corpus.sh`, per the T2.2 coverage-prep pattern).

---

## 4. Humility implementation plan

> **Implemented (2026-06-24).** Humility added to vocab (idx 99, `{2}{W}{W}`); card-costs regenerated.
> Ability removal was already scaffolded in §2 (suppression + `recompute_abilities`); this step added
> the missing **layer 7b** "set P/T to N" path and a general-SVar fix:
> - **Layer 7b applier** (`apply_layer7_pt_effects`, between 7a and 7c): collects active non-CDA
>   setters (`set_power_svar`/`set_toughness_svar`, not `characteristic_defining`), then for each
>   battlefield creature applies the latest-timestamp setter whose `Affected$` matches it — including
>   `Affected$ Creature` = **every creature** (the all-creatures application pattern, new vs the
>   self/EquippedBy appliers). `gather_active_statics` resets `has_set_pt`/`set_power`/`set_toughness`
>   each pass. The 7c additive loop now excludes *all* setters (was: only CDA setters).
> - **`evaluate_sa_svar` integer-literal fix** — a plain numeric SVar (Humility's `SetPower$ 1`) used
>   to fall through to the `Count$` handlers and return 0, making creatures 0/0 (they died). Now a
>   numeric literal evaluates to itself. General fix, not Humility-specific.
> - **Verified via harness:** Grizzly Bears → **1/1 (alive)**; Knight of the Reliquary with a land in
>   its graveyard → **3/3 without Humility but 1/1 with** (definitive proof the self-pump is *removed*
>   in layer 6, not merely base-overridden — the §4d headline); Dryad Arbor → **1/1** and its `{T}: Add
>   G` mana ability **removed** (Ignoble Hierarch no longer castable off it). Corpus byte-identical
>   (Humility/7b dormant for the current corpus vocab).
>
> Original plan (kept for context):

Humility (`bin/resources/cardsfolder/h/humility.txt`, **not yet in `card_vocab.h`**):
`S:Mode$ Continuous | Affected$ Creature | SetPower$ 1 | SetToughness$ 1 | RemoveAllAbilities$ True`.

Today it would be a **complete no-op**: `SetPower/SetToughness` parse into `set_power_svar` but the
only consumer is the 7a branch gated on `characteristic_defining` (false here), layer 7b is inert,
and `RemoveAllAbilities` has no field and no layer-6 removal path. Implementing it requires three
pieces, in order:

### 4a. Wire layer 7b ("set P/T to N") for non-CDA statics
- **Parse:** in `parse.cpp` (~1585) keep `set_power_svar`/`set_toughness_svar`, but mark the static
  as a 7b setter when `CharacteristicDefining$ True` is **absent** (add `bool set_pt = true` or
  reuse a flag). Do not retag (CLAUDE.md): `SetPower` already means "set", the gap is the consumer.
- **Apply:** add a **7b applier** between 7a and 7c in `apply_layer7_pt_effects` that, for each
  active non-CDA setter affecting a creature, sets `cr.has_set_pt = true; cr.set_power = eval; …`.
  `recompute_pt` already honors `has_set_pt` (`creature.cpp:20-23`) — 7b becomes live.
- **Reset:** `gather_active_statics` must clear `has_set_pt`/`set_power`/`set_toughness` each pass
  (alongside the `static_*_bonus` reset) so it's rebuilt, not sticky.
- **Timestamp:** multiple 7b setters order by source timestamp (613.7); last wins. Reuse
  `order_continuous_effects`.

### 4b. Layer-6 ability removal (`RemoveAllAbilities`)
- **Parse:** add `bool remove_all_abilities` to `StaticAbility`; set it from `RemoveAllAbilities$ True`.
- **Apply:** depends on **§2** (ability recompute). In `recompute_abilities` step 2.3, for each
  active `remove_all_abilities` static, clear the computed ability set of every affected creature
  (`Affected$ Creature`) — *after* derivation (2.1) and grants (2.2). This is what makes Dryad
  Arbor's mana ability and Knight's self-pump actually disappear.
- **Scope:** "all abilities" removes activated, triggered, static (including the creature's own
  P/T-modifying static and CDAs), and keyword abilities. The CDA removal interacts with 7a — see 4d.

### 4c. Add to vocab + costs
- Append `{"Humility", N}` to `card_vocab.h`; bump `N_CARD_TYPES` in `machine_io.h` if needed; regen
  `train/gen_card_costs.py` (CLAUDE.md "Adding a New Card").

### 4d. Correctness notes (the layer reasoning to encode)
- **Knight under Humility = 1/1, unambiguously.** Removal (L6) precedes P/T mods (L7c); Knight's
  pump source is gone before 7c runs. No timestamp dependency because the two effects are in
  *different layers* (613.8a dependency is intra-layer only). The naive risk is that
  `apply_layer7_pt_effects` reads Knight's `StaticAbility` directly without consulting whether L6
  removed it — §2's recompute is what prevents that (a removed static is not in the computed set the
  7c applier iterates). **Ensure the 7c/7a appliers iterate the *computed* statics, not
  `Permanent::static_abilities` raw**, or Humility's removal won't take.
- **Barrowgoyf/CDA under Humility = 1/1:** 7a CDA either removed (L6) or overridden by 7b set; both
  give 1/1.
- **External pumps survive:** counters (7c) and resolved-spell continuous effects are not "abilities"
  and are not removed — Humility + counter = 2/2.

### Sequencing
4a (7b) and the §2 recompute are prerequisites for 4b. Land both behind a byte-identical corpus run
*before* adding Humility to vocab (4c), so the refactor is provably behavior-preserving; then 4c/4d
introduce the first real behavior change, covered by scenarios 4–8.

---

## 5. Dependency system (613.8) outline

Today `resolve_dependencies()` (`state_manager_layers.cpp:39-47`) is a documented no-op; ordering is
pure timestamp (613.7). No current vocab forms a dependency, so this is the lowest-priority item —
but it is the last real 613 conformance gap and several famous interactions need it.

### What a dependency is (613.8a)
Within one (sub)layer, effect A **depends on** B iff: (a) same layer/sublayer; (b) applying B would
change A's text/existence, what A applies to, or what A does; and (c) not exactly one of them is a
CDA. Dependent effects wait until their dependees are applied (613.8b); loops fall back to timestamp;
after each application the remaining order is re-evaluated (613.8c).

### Minimal design
1. **Represent effects uniformly per layer** as a list of `{source, affected-set-fn, apply-fn,
   timestamp, is_cda}` — the `ContinuousEffect` struct (`continuous_effects.h:44-55`) is most of
   this already; extend layers 4/6 to produce them instead of bespoke loops.
2. **Dependency check** `depends_on(A, B)`: apply B to a *trial* copy of game state, recompute A's
   affected-set and effect, and compare to A without B. Concretely, for layer 4: does applying B
   change which permanents A applies to, or A's resulting types? This needs the affected-set to be a
   recomputable function, not a precomputed list — the main refactor cost.
3. **Ordering:** within a layer, build the dependency graph over the (non-CDA-mismatched) effects;
   topologically order, breaking ties and cycles by timestamp (613.8b). Replace the plain
   `std::sort` by-timestamp with this.
4. **Re-evaluation (613.8c):** after each apply, recompute the graph for unapplied effects. Start
   simple (recompute the whole graph each step; layers are tiny) and optimize later.

### Canonical cases to target (and a test)
- **Layer 4: Blood Moon vs. Urborg, Tomb of Yawgmoth** — actually *independent* (each still applies
  to the same lands; timestamp decides) — good negative test that the dependency check does **not**
  fire.
- **Layer 4: Blood Moon vs. an effect granting an ability to lands of a type** — dependent.
- **Cross-effect: Humility + Opalescence** — the textbook loop; 613.8b's loop clause (fall back to
  timestamp) is what resolves it. Use as the headline acceptance test once layers 6/7 produce
  uniform effects.

### Phasing
Dependencies are **deferred** until at least one vocab pair needs them. Build §2 (ability recompute)
and §4 (Humility) first; they don't need dependencies (Knight/Dryad Arbor interactions are
cross-layer). Revisit 613.8 when a same-layer dependent pair enters the vocab.

---

## Suggested order of work

1. **§2 ability recompute** (behavior-preserving; unblocks everything; corpus byte-identical).
2. **§1d Blood Moon 305.7** (falls out of §2 — strip printed land abilities on type-set; flips
   scenarios A1/A2 from fail to pass).
3. **§1a/§1b/§1c consolidation** (dead-field consumers + `rules_modifying` query surface + untap
   `ReplacementEvent`; mostly independent, low risk).
4. **§4 Humility** (7b wiring → ability removal → vocab; first real behavior change; scenarios 4–8).
5. **§5 dependencies** (deferred until a vocab pair demands it).

Track these as new tiered items in `docs/rules_compliance_issues.md` as each is picked up.
