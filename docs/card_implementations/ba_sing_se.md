# Ba Sing Se  (vocab index 197)

## Oracle text
This land enters tapped unless you control a basic land.

{T}: Add {G}.

{2}{G}, {T}: Earthbend 2. Activate only as a sorcery. (Target land you control becomes a 0/0
creature with haste that's still a land. Put two +1/+1 counters on it. When it dies or is exiled,
return it to the battlefield tapped.)

(Land, no mana cost.)

## Forge script
- Source: pre-existing local script — `bin/resources/cardsfolder/b/ba_sing_se.txt` (not edited).
- Key tags:
  - `R:Event$ Moved | ValidCard$ Card.Self | Destination$ Battlefield | ReplaceWith$ LandTapped |
    ReplacementResult$ Updated` → `LandTapped:DB$ Tap | Defined$ Self | ETB$ True |
    ConditionPresent$ Land.Basic+YouCtrl | ConditionCompare$ EQ0`.
  - `A:AB$ Mana | Cost$ T | Produced$ G` (a basic mana ability).
  - `A:AB$ Earthbend | Cost$ 2 G T | SorcerySpeed$ True | Num$ 2`.
- Tags parsed as written; no category was retagged. The cosmetic `SpellDescription$` /
  `ReplacementResult$ Updated` are ignored.

## Engine work (all general, keyed on each tag's intended meaning)

### Conditional "enters tapped" replacement (`ReplaceWith$ <SVar>` = DB$ Tap | ETB$ True | …)
- The replacement parser (`parse_replacement_effects`, `src/parse.cpp`) previously only handled
  the literal `ReplaceWith$ ETBTapped` token. It now also recognises `ReplaceWith$ <SVar>` whose
  body is a `DB$ Tap | ETB$ True` with a `ConditionPresent$`/`ConditionCompare$` gate, producing a
  **conditional** `ENTERS_TAPPED` replacement. The condition spec (filter + compare, e.g.
  `Land.Basic` / `EQ0`) is stored on `Effect::Replacement` (`src/components/effect.h`); the
  `+YouCtrl` qualifier is dropped because the count is always evaluated controller-relative.
- `replacement::collect` (`src/systems/replacement_effects.cpp`) evaluates the gate
  (`tapped_condition_met`) when building the SELF_TAPPED candidate: it counts the entering
  permanent's controller's battlefield permanents matching the filter (honouring a `.Basic` /
  `.nonBasic` supertype qualifier) and compares via `compare_svar`. An empty filter = the plain
  unconditional "enters tapped". So Ba Sing Se enters tapped iff the controller controls zero
  basic lands; with a basic land it enters untapped.
- A resolve-time `DB$ Tap` handler (`EffectKind::Tap`, `src/effects/effect_tap.cpp`) covers the
  generic case of a Tap that reaches the stack (taps `Defined$ Self` or the target); Ba Sing Se's
  ETB$ True path is realized entirely by the replacement, so it never reaches resolution.

### Earthbend keyword action — activated, sorcery-speed (shared with Badgermole Cub)
- See `badgermole_cub.md` for the full Earthbend handler (`EffectKind::Earthbend`,
  `src/effects/effect_earthbend.cpp`) and the Animate land-animation wiring + the
  leaves-the-battlefield return-tapped delayed trigger. Ba Sing Se uses `Num$ 2` (two +1/+1
  counters → a 2/2 after the 0/0 base).
- **Sorcery-speed gating**: new `Ability::sorcery_speed_only` flag, parsed from
  `SorcerySpeed$ True` (`src/parse.cpp`). The activated-ability legal-action enumeration
  (`src/systems/state_manager_actions.cpp`) skips a `sorcery_speed_only` ability unless the
  controller is in the sorcery-speed window (their main phase, empty stack) — CR 605.x.
- The `2 G T` activation cost is paid by the existing activation-cost machinery (mana + tap-self).
- `Cost$ T` mana ability is a plain subtype-independent AddMana(G) — handled by the existing mana
  system (it is not a basic-land subtype-derived ability, it is scripted).

## CR references
- 614.1d / 616.1 — "enters tapped" replacement; the controller is the chooser/affected player.
- 605.x — an activated ability that can be activated only as a sorcery.
- 613.4 (layer 7) / 603.6e / 614.1d — see `badgermole_cub.md` for the earthbend land-animation,
  the return-tapped delayed trigger, and the returned land entering tapped.
- 122.1 — +1/+1 counters.

## Behavioral decisions
- "Unless you control a basic land" is evaluated as the permanent enters (616.1) against the
  controller's current basic-land count; a nonbasic land (e.g. Ghost Quarter) does not satisfy it.
- The earthbend activated ability is sorcery-speed only and is hidden from the action menu outside
  the controller's main phases / when the stack is non-empty.

## Tests (test_harness.py, scripted/semantic `--play`)
- ETB tapped gate: with only a nonbasic land (Ghost Quarter) → "Ba Sing Se enters tapped",
  `Ba Sing Se (T)`; with a basic Mountain → enters untapped (no tapped line, board shows
  `Ba Sing Se` with no `(T)`). ✓
- `{T}: Add {G}`: tapping Ba Sing Se ("Player A activated Ba Sing Se for 1(G)") pays toward a
  {1}{G} spell. ✓
- Earthbend 2 (`2 G T`, sorcery speed): targets a Forest you control → "Forest becomes a 0/0
  creature with haste ... put 2 +1/+1 counter(s)"; board shows `Forest [2/2]`, still a land, and
  it attacked the same turn (haste). ✓
- Sorcery-speed: "Activate Ba Sing Se (Earthbend)" is offered ONLY in First/Second Main of the
  controller's turn — never in untap/combat/cleanup or the opponent's turn. ✓
- Regression: scripted full games (decks featuring Ba Sing Se + Badgermole Cub), seeds 1–6 —
  decisive both ways, no draws, no non-fatal errors; Ba Sing Se entered tapped and earthbent a
  land in real games. ✓
