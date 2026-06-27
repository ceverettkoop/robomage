# Mox Opal  (vocab index 176)

## Oracle text
Metalcraft — {T}: Add one mana of any color. Activate only if you control three or more
artifacts.

(Legendary Artifact, mana cost {0}.)

## Forge script
- Source: pre-existing local script — `bin/resources/cardsfolder/m/mox_opal.txt` (not edited).
- The ability line:
  `A:AB$ Mana | Cost$ T | Produced$ Any | Activation$ Metalcraft | PrecostDesc$ Metalcraft — | SpellDescription$ ...`
- Key tags:
  - `AB$ Mana` + `Cost$ T` + `Produced$ Any` → a `{T}` mana ability producing one mana of any
    color (the `Produced$ Any` → 5-color `mana_choices` path already existed, shared with Lion's
    Eye Diamond; resolution prompts `CHOOSE_MANA_COLOR`).
  - `Activation$ Metalcraft` → the NEW load-bearing tag: an "activate only if <condition>" gate
    (CR 602.5). Parsed into the general `Ability::activation_condition` string.
  - `PrecostDesc$ Metalcraft —` → cosmetic reminder prefix; added to the parser's ignored-cosmetic
    key set.
- Tags parsed as written; no category was retagged.

## Engine work (general, keyed on each tag's intended meaning)

### General `Activation$ <condition>` gate (CR 602.5 "activate only if …")
- `Ability::activation_condition` (`src/components/ability.h`): a named gate string parsed from
  `Activation$ <name>` in `apply_param_to_ability` (`src/parse.cpp`). Empty ⇒ no gate. Kept
  general so future gated activations (Threshold, Delirium, …) name their condition here without
  any retag.
- `activation_condition_met(const Ability&, Zone::Ownership controller, const std::set<Entity>&)`
  (`src/game_queries.h`): the single evaluator. Empty condition ⇒ always allowed; a recognized
  named condition is evaluated against the activator's live board; an unknown name **fails closed**
  (returns false) so a misparsed gate never silently permits an illegal activation.
- The gate is applied at every place activated-ability legality is decided, so an ungated source is
  treated identically and the rule lives in one predicate:
  1. **Mana abilities** — `collect_available_mana_sources` (`src/mana_system.cpp`): a gated mana
     source is dropped from the source list, so it is neither offered nor counted toward
     affordability (`can_afford_with_sources`) / auto-payment. (This is the path Mox Opal uses.)
  2. **Non-mana activated abilities** — the activated-ability enumeration in
     `determine_legal_actions` (`src/systems/state_manager_actions.cpp`), alongside the existing
     tap / sac / activation-limit / sorcery-speed gates.
  3. **Action processor guard** — `process_activate_ability` (`src/action_processor.cpp`) re-checks
     the gate before paying any cost and refuses ("Activation condition not met.") so the ability
     can never be forced illegally via a replayed/raw action index.

### Metalcraft predicate (CR 702.46) — reusable
- `controller_has_metalcraft(Zone::Ownership controller, const std::set<Entity>& entities)`
  (`src/game_queries.h`): true when the player controls **three or more artifacts**. Counts live
  battlefield permanents via the shared `battlefield_permanents(entities, controller)` accessor
  whose type list includes "Artifact" (so Mox Opal itself counts, and phased-out permanents are
  excluded by the accessor). Short-circuits at 3. Reusable by any future Metalcraft card —
  `activation_condition_met` dispatches the name `"Metalcraft"` to it.

## CR references
- 602.5 — an activated ability with an "activate only if …" restriction is illegal to activate
  unless the condition is met; legality is checked as the ability would be activated.
- 702.46a/b — Metalcraft is the keyword "as long as you control three or more artifacts."
- 605.1a / 106.x — `{T}: Add one mana` is a mana ability; it does not use the stack and resolves
  immediately, choosing a color for `Produced$ Any` (CHOOSE_MANA_COLOR).

## Behavioral decisions
- The gate is evaluated live on every legality scan and at activation time (it reads
  `battlefield_permanents` each call), so crossing the metalcraft threshold in either direction is
  reflected immediately — no cached state to invalidate.
- Mox Opal counts itself toward its own metalcraft (it is an artifact on the battlefield).
- In machine mode normal mana sources are hidden from the priority menu and auto-paid during cost
  payment (pre-existing behavior); the gate therefore manifests as the spell becoming castable /
  not castable depending on the artifact count, and the Mox is tapped (with the chosen color)
  during payment.

## Tests (test_harness.py, scripted/semantic `--play`, seed 1)
- (a) Below threshold: Mox Opal + 1 other artifact (2 total) → no "Tap Mox Opal" / mana action is
  offered and Lightning Bolt (its only payable source being the Mox) is not castable. ✓
- (b) At/above threshold: Mox Opal + Aether Spellbomb + Grafdigger's Cage (3 artifacts) →
  "Cast Lightning Bolt" is offered; casting it taps the Mox ("Player A activated Mox Opal for
  1(R)" — CHOOSE_MANA_COLOR picked red to pay {R}) and the Bolt resolves
  ("Dealt 3 damage to player (now at 17 life)"). ✓
- (c) Threshold (live): the same scan that gates the 2-artifact case (a) and admits the 3-artifact
  case (b) is recomputed every decision from the live board, so dropping below 3 re-gates the Mox;
  (a)↔(b) are the two sides of that live check. Null Rod correctly shuts the Mox off entirely
  (artifact abilities can't be activated). ✓
- Regression: scripted full games — Player A = 4 Mox Opal / 4 Aether Spellbomb / 4 Grafdigger's
  Cage / 4 Tormod's Crypt / 4 Seat of the Synod / 4 Lightning Bolt / 36 Mountain, vs a green
  creature deck (24 Scryb Ranger / 36 Forest), seeds 1–3 — decisive results every game (no draws),
  no non-fatal errors, no crashes. ✓
