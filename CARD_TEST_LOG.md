# Card Functionality Test Log

Testing every card in `src/card_vocab.h` (indices 0–89) one-by-one with
`train/test_harness.py`. No corpus / replay-diff tools used.

**Methodology notes**
- The harness opens with two mulligan decisions (D1 = Player A keep/mull, D2 = Player B keep/mull). `0` = keep.
- Only the active player stops for priority when the stack is empty; the non-active player auto-passes unless it has a response.
- TURN 0 is Player A on the play (no draw step).
- Casting auto-taps available lands for the cost; a separate target decision follows when the spell/ability needs a target. Target index `0` is typically the opponent (player/permanent), `1` self.
- Action-index sequences below are passed via `--actions`; once the list is exhausted the harness auto-passes (`0`) for the remainder.

**Status legend:** ✅ pass · ❌ fail (paused for instruction) · ❓ needs clarification (paused) · ↔︎ deferred to predecessor

---

## 0. Mountain ✅
Basic land, taps for {R}.
- Test: `--hand-a "Mountain,Lightning Bolt"` … `--actions "0,0,1,1,0"` (keep, keep, play Mountain, cast Bolt, target opp).
- Result: Mountain entered, tapped for {R} to pay for Lightning Bolt (auto-tap shown as `Mountain (T)`), Bolt resolved. Land mana ability works.

## 2. Lightning Bolt ✅
Instant — deal 3 damage to any target.
- Test: same run as Mountain above. `--actions "0,0,1,1,0"`.
- Result: `Player A casts Lightning Bolt targeting Player B / Dealt 3 damage to player (now at 17 life)`. Correct.

## 1. Forest ✅
Basic land, taps for {G}.
- Test: `--hand-a "Grizzly Bears" --battlefield-a "Forest,Forest"` … `--actions "0,0,7"` (cast Grizzly Bears for 1G).
- Result: Both Forests tapped (`Forest (T)`) to pay 1G; `Grizzly Bears [2/2] (SICK)` entered. Land mana ability works.

## 3. Grizzly Bears ✅
Vanilla 2/2 creature for {1}{G}.
- Test: same run as Forest above.
- Result: Entered battlefield as `[2/2]` with summoning sickness. Correct. (Reference card for vanilla creatures.)

## 19. Island ✅
Basic land, taps for {U}.
- Test: `--hand-a "Brainstorm" --battlefield-a "Island"` … `--actions "0,0,1"` (cast Brainstorm for U).
- Result: Island tapped (`Island (T)`) to pay {U} for Brainstorm. Works.

## 24. Brainstorm ✅
Instant — draw three cards, then put two cards from your hand on top of your library.
- Test: same run as Island above, `--actions "0,0,1,0,0"`.
- Result: Hand 6→9 (drew 3), then two "Put on top" choices returned hand to 7; Brainstorm to graveyard. Correct.

## 11. Ponder ✅
Sorcery — look at top 3, put back in any order, may shuffle, draw a card.
- Test: `--hand-a "Ponder" --battlefield-a "Island"`, library = mixed lands, `--actions "0,0,7,0,0,0,0"`.
- Result: Offered to reorder the top 3 (two "Put on top" choices to order them), then a SHUFFLE choice, then drew 1 (hand 6→7), Ponder to graveyard. Correct.

## 14. Soul Warden ✅
1/1 — whenever another creature enters, you gain 1 life.
- Test: `--battlefield-a "Soul Warden,Forest,Forest" --hand-a "Grizzly Bears"`, `--actions "0,0,7"`.
- Result: On Grizzly Bears entering, Player A life 20→21. Triggers on *another* creature (did not gain from itself). Correct.

## 44. Plains ✅
Basic land, taps for {W}.
- Test: `--hand-a "Swords to Plowshares" --battlefield-a "Plains" --battlefield-b "Grizzly Bears"`, `--actions "0,0,1,0"`.
- Result: Plains tapped for {W} to pay for Swords to Plowshares. Works.

## 48. Swords to Plowshares ✅
Instant — exile target creature; its controller gains life equal to its power.
- Test: same run as Plains above.
- Result: `Grizzly Bears is moved to exile / Player B gains 2 life (now at 22)` (power 2). Correct.

## 62. Swamp ✅
Basic land, taps for {B}.
- Test: `--hand-a "Thoughtseize" --battlefield-a "Swamp"`, `--actions "0,0,7,0,0"`.
- Result: Swamp tapped for {B} to pay for Thoughtseize. Works.

## 54. Thoughtseize ✅
Sorcery — target player reveals hand, you choose a nonland card, they discard it, you lose 2 life.
- Test: same run as Swamp; opponent hand contained Lightning Bolt + Forests.
- Result: `Player B reveals their hand … Player B discards Lightning Bolt … Player A loses 2 life (now at 18)`. Only nonland (Lightning Bolt) was offered. Correct.

## 55. Dark Ritual ✅
Instant — add {B}{B}{B}.
- Test: `--hand-a "Dark Ritual" --battlefield-a "Swamp"`, cast in first main `--actions "0,0,0,0,0,7"`.
- Result: After resolution, same-step pool shows `mana: 3B` (Swamp tapped to pay {B}, then +BBB). Correct. (Mana empties at step transitions, so the pool must be read in the same step.)

## 23. Lightning Strike ✅ (≈ Lightning Bolt)
Instant — deal 3 damage to any target, cost {1}{R}. Differs from Lightning Bolt (#2) only in mana cost.
- Test: `--hand-a "Lightning Strike" --battlefield-a "Mountain,Mountain"`, `--actions "0,0,1,0"`.
- Result: `Player A casts Lightning Strike targeting Player B` (2 Mountains tapped for 1R). Damage effect identical to Lightning Bolt. Correct.

## 68. Consider ✅
Instant — Surveil 1, then draw a card. Cost {U}.
- Test: `--hand-a "Consider" --battlefield-a "Island"`, `--actions "0,0,1,0,0"`.
- Result: Surveil decision offered for the top card (keep on top / to graveyard), then `Resolving ability (category: Draw, amount: 1)`; hand 6→7. Correct.

## 69. Duress ✅ (≈ Thoughtseize, narrower filter)
Sorcery — target opponent reveals hand; you choose a noncreature, nonland card; they discard it. Cost {B}.
- Test: `--hand-a "Duress" --battlefield-a "Swamp"`, opp hand = Lightning Bolt + Grizzly Bears + Forest, `--actions "0,0,7,0,0"`.
- Result: Only `Lightning Bolt` offered as discard choice (Grizzly Bears = creature, Forest = land correctly excluded); `Player B discards Lightning Bolt`. No life loss (unlike Thoughtseize). Correct.

## 30. Birds of Paradise ✅ (mana)
0/1 flier — {T}: add one mana of any color.
- Test: `--battlefield-a "Birds of Paradise" --hand-a "Lightning Bolt"`, `--actions "0,0,1,0"`.
- Result: `Player A activated Birds of Paradise for 1(R)` — produced {R} (any color) to cast Lightning Bolt; opp 20→17. Mana ability works. (Flying keyword present in script; not separately combat-tested.)

## 42. Noble Hierarch ✅ (mana + exalted)  — exalted verified below
0/1 — Exalted; {T}: add {W}, {U}, or {G}.
- Test: `--battlefield-a "Noble Hierarch" --hand-a "Brainstorm"`, `--actions "0,0,1,…"`.
- Result: `Player A activated Noble Hierarch for 1(U)` — produced {U} to cast Brainstorm. Mana works. NOTE: Exalted (+1/+1 when a creature attacks alone) tested separately — see exalted entry below.

## 16. Delver of Secrets ✅  /  ## 17. Insectile Aberration ✅
DFC. Front: 1/1; at your upkeep look at top card, may reveal; if instant/sorcery revealed, transform. Back: Insectile Aberration 3/2 flying.
- IMPORTANT test-setup note: this DFC's script file is `delver_of_secrets_insectile_aberration.txt`; the card must be referenced by the **combined name** "Delver of Secrets Insectile Aberration" (as the deck files do). Using the bare front name "Delver of Secrets" before the combined file is loaded crashes the loader (`parse_card_script` asserts on missing file) — a loader naming quirk, not a gameplay bug.
- Test: `--battlefield-a "Delver of Secrets Insectile Aberration"`, top of library stacked to Lightning Bolt (instant), `--actions "0,0,1"` (the `1` = "Reveal" at the upkeep peek prompt).
- Result: `Top card of library: Lightning Bolt / Revealed: Lightning Bolt / Delver of Secrets transforms into Insectile Aberration!` → permanent becomes `[3/2]`. Choosing "Don't reveal" (0) correctly leaves it a 1/1. Transform works. (Benign warnings: `RevealOptional`/`RememberRevealed` params unrecognized, but behavior is implemented via the explicit reveal prompt + inline instant/sorcery check.)

---
## Lands — dual lands (subtype-based mana injection)

## 4. Volcanic Island ✅
Dual land (types Land Island Mountain), taps for {U} or {R}.
- Test: `--battlefield-a "Volcanic Island,Volcanic Island" --hand-a "Lightning Bolt,Brainstorm"`, `--actions "0,0,1,0,1,0,0"`.
- Result: `activated Volcanic Island for 1(R)` (for Bolt) and `activated Volcanic Island for 1(U)` (for Brainstorm). Produces both colors. Reference card for ABUR duals. ✅

## 15. Tundra ✅ (≈ Volcanic Island, W/U)
- Test: `--battlefield-a "Tundra" --hand-a "Swords to Plowshares"` vs opp Grizzly Bears, `--actions "0,0,1,0"`.
- Result: `activated Tundra for 1(W)`; StP resolved. {U} via Island subtype (same mechanism as Volcanic Island). ✅

## 45. Savannah ✅ (≈ Volcanic Island, G/W)
- Test: `--battlefield-a "Savannah,Savannah" --hand-a "Grizzly Bears"`, `--actions "0,0,7"`.
- Result: `activated Savannah for 1(G)` ×2 to cast Grizzly Bears. {W} via Plains subtype. ✅

## 64. Underground Sea ✅ (≈ Volcanic Island, U/B)
- Test: `--battlefield-a "Underground Sea" --hand-a "Thoughtseize"`, `--actions "0,0,7,0,0"`.
- Result: `activated Underground Sea for 1(B)`; Thoughtseize resolved. {U} via Island subtype. ✅

---
## Lands — fetchlands ({T}, pay 1 life, sac: search two land types, shuffle)

## 5. Scalding Tarn ✅
Fetches Island or Mountain.
- Test: `--battlefield-a "Scalding Tarn"`, library stacked with Island+Mountain (7-card hand so they stay in library), `--actions "0,0,1,1"` (activate, then Search: Island).
- Result: `Player A pays 1 life` (20→19), search list = {Fail to find, Island, Mountain}, picked Island, `Player A shuffles their library`, `Self BF: Island`, Scalding Tarn sacrificed. Reference card for fetchlands. ✅
- Note: search index `0` = "Fail to find"; pick `1`+ to actually fetch.

## 6. Flooded Strand ✅ · ## 7. Polluted Delta ✅ · ## 8. Wooded Foothills ✅ · ## 9. Misty Rainforest ✅ · ## 65. Bloodstained Mire ✅ · ## 66. Verdant Catacombs ✅ · ## 52. Windswept Heath ✅
All ≈ Scalding Tarn, differ only in the two fetchable basic types.
- Test: each `--battlefield-a "<fetch>"` with both fetchable types stacked in library, `--actions "0,0,1,1"`.
- Result: each offered exactly its correct two search types and fetched: Flooded Strand→Plains/Island, Polluted Delta→Island/Swamp, Wooded Foothills→Mountain/Forest, Misty Rainforest→Forest/Island, Bloodstained Mire→Swamp/Mountain, Verdant Catacombs→Swamp/Forest, Windswept Heath→Plains/Forest. ✅

---
## Lands — utility

## 10. Wasteland ✅
{T}: Add {C}. / {T}, Sac: destroy target nonbasic land.
- Test: `--battlefield-a "Wasteland" --battlefield-b "Volcanic Island"`, `--actions "0,0,1,0"` (activate destroy, target Volcanic Island).
- Result: `Player A sacrifices Wasteland … Resolving ability (category: Destroy) … Volcanic Island is destroyed`. Correctly only targeted the nonbasic land. ✅ (Mana {C} ability also present in script.)

## 25. Thundering Falls ✅
Dual (Island Mountain), enters tapped, surveil 1 on ETB, taps U/R.
- Test: play from hand `--hand-a "Thundering Falls,…"`, `--actions "0,0,1,0,0"`.
- Result: `Thundering Falls enters tapped. / Thundering Falls triggered / Resolving ability (category: Surveil, amount: 1)`; permanent shown as `Thundering Falls (T)`. ETB-tapped + surveil work. ✅

## 63. Undercity Sewers ✅ (≈ Thundering Falls, U/B)
Same as Thundering Falls (enters tapped, surveil 1) but Island Swamp (U/B). Not separately re-run; identical script structure — referred to #25.

## 34. Gaea's Cradle ✅
{T}: Add {G} for each creature you control.
- Test: `--battlefield-a "Gaea's Cradle,Grizzly Bears,Grizzly Bears" --hand-a "Grizzly Bears"`, `--actions "0,0,7"`.
- Result: `Player A activated Gaea's Cradle for 2(G)` (2 creatures → GG) and cast Grizzly Bears. Amount scales with creature count. ✅

## 39. Karakas ✅
{T}: Add {W}. / {T}: Return target legendary creature to its owner's hand.
- Test: `--battlefield-a "Karakas" --battlefield-b "Thalia Guardian of Thraben"` (comma-free!), `--actions "0,0,1,0"`.
- Result: targeted Thalia (legendary), `Thalia, Guardian of Thraben is moved to hand`, opp hand 7→8. ✅
- HARNESS NOTE: `--battlefield-*` splits on commas, so cards whose names contain a comma (e.g. "Thalia, Guardian of Thraben") must be passed comma-free as "Thalia Guardian of Thraben".

## 32. Dryad Arbor ✅
Land Creature Forest Dryad 1/1; {T}: Add {G}; affected by summoning sickness.
- Test: `--battlefield-a "Dryad Arbor" --hand-a "Birds of Paradise"`, `--actions "0,0,7"`.
- Result: `Player A activated Dryad Arbor for 1(G)` (pre-placed → not sick), cast Birds. Shows as `Dryad Arbor [1/1]`. ✅

## 36. Horizon Canopy ✅
{T}, Pay 1 life: Add {G} or {W}. / {1},{T},Sac: Draw a card.
- Mana test (green): cast Birds of Paradise off Canopy → `Player A pays 1 life` then `activated Horizon Canopy for 1(G)`.
- Mana test (white): cast Swords to Plowshares off Canopy → `Player A pays 1 life` then `activated Horizon Canopy for 1(W)`, life 20→19, Grizzly exiled.
- Draw test: `--actions "0,0,1,…"` → `activated Forest for 1(G)` (pays {1}), `Player A sacrifices Horizon Canopy`, `Player A draws Forest`; Canopy to graveyard.
- Result: taps for both {G} and {W}, each costing 1 life; sac-to-draw works. ✅ (Confirmed colored-mana production AND the per-activation 1-life cost.)

## 50. Talon Gates of Madara ✅ (FIXED)
Land. ETB: up to one target creature phases out. Also {T}:{C}; {1}{T}: any color; {4} put from hand.
- Original bug: `Talon Gates of Madara triggered / Resolving ability (category: Phases, amount: 0)` with **no target prompt** — the creature never phased. Root cause: when a *triggered* ability with `ValidTgts$` was pushed onto the stack (`StateManager::check_triggered_abilities`), `select_target` was never called, so the trigger always had target 0. (Cast/activated abilities already called `select_target`; triggered ones did not.)
- Fix: in `src/systems/state_manager.cpp`, before `push_ability_onto_stack`, call `select_target(trigger_ab, orderer, perm.controller)` for triggers where `valid_tgts != "N_A" && target == 0 && has_legal_targets(...)`. Guard skips already-targeted triggers (e.g. ExaltedBonus) and no-target triggers.
- Retest: play from hand vs opp Grizzly Bears, `--actions "0,0,1,1"` (the `1` = Target Grizzly Bears at the new prompt). Result: `Grizzly Bears phases out` → could not attack while phased → `Grizzly Bears phases in` on its controller's untap. Correct phasing. ✅
- Regression check: Soul Warden (no-target trigger) still gains life; `train.py --diag` 10 games, 0 draws, no crashes.

## 67. Cavern of Souls ✅ (works — earlier diagnosis was a test-setup error)
Land. As it enters, choose a creature type. {T}:{C}. {T}: any color, only for a creature spell of the chosen type (uncounterable).
- My first attempt used an **all-Forest deck**, so there were no creature subtypes to offer → the ETB choose-type correctly did nothing, and the harness mislabels OTHER_CHOICE actions as `Choice: Cavern of Souls`. That looked broken but was a bad test.
- Code review: `creature_subtypes` is built from the player's whole deck at game start (`orderer.cpp`); the ETB handler (`state_manager.cpp`) prompts with one option per creature subtype the player owns, sorted most-prominent first.
- Proper test: deck with creatures of several types (Grizzly Bears=Bear, Birds=Bird, Soul Warden=Human/Cleric); play Cavern from hand → `Choose a creature type for Cavern of Souls:` offered **4 options** (Bear/Bird/Cleric/Human); chose Bear (`Player A chose creature type: Bear`); then `activated Cavern of Souls for 1(G)` (its chosen-type restricted mana) + Forest to `cast Grizzly Bears` (a Bear). ✅
- No code change needed.

---
## Counters / responses
(Setup pattern: Player A holds the counter + untapped lands; Player B casts a spell on its turn; A gets priority with the spell on the stack and responds.)

## 22. Counterspell ✅
Instant {U}{U} — counter target spell.
- Test: A `--battlefield-a "Island,Island" --hand-a "Counterspell"`; B `--battlefield-b "Forest,Forest" --hand-b "Grizzly Bears"`; `--actions "0,0,0,0,8,1,0"` (B casts Grizzly at D5 idx 8; A casts Counterspell at D6 idx 1; target the spell).
- Result: `Player A casts Counterspell targeting Grizzly Bears / Resolving ability (category: Counter) / Grizzly Bears is countered`. ✅

## 12. Force of Will ✅
Instant {3}{U}{U} — counter target spell. Alt cost: pay 1 life + exile a blue card from hand.
- Test: A `--hand-a "Force of Will,Brainstorm"` (no lands → forces alt cost); B casts Grizzly; `--actions "0,0,0,0,8,1,…"`.
- Result: `Player A pays 1 life` (20→19), `Player A exiles Brainstorm` (blue card), `casts Force of Will targeting Grizzly Bears`, `Grizzly Bears is countered`. Alt cost + counter both work. ✅

## 13. Daze ✅
Instant {1}{U} — counter target spell unless its controller pays {1}. Alt cost: return an Island.
- Test: A `--battlefield-a "Island" --hand-a "Daze"`; B casts Grizzly while tapped out; `--actions "0,0,0,0,8,1,…"`.
- Result: `Player A returns Island to hand` (alt cost), targeted Grizzly, `Resolving ability (category: Counter) / Grizzly Bears is countered` (B had no mana to pay {1}). ✅ (The "pays {1}" branch where the controller has mana was not separately exercised.)

## 72. Force of Negation ✅ (thoroughly verified)
Instant {1}{U}{U} — counter target noncreature spell; if countered this way, exile it. Alt cost (only if it's NOT your turn): exile a blue card from hand.
- Noncreature-only: B cast Grizzly Bears (creature) with A holding FoN + 3 Islands → A was **never offered "Cast Force of Negation"** (no legal target); Grizzly resolved. ✅
- Counters + EXILES: A hardcast FoN on Lightning Bolt → `Lightning Bolt is countered`; Bolt is **absent from every graveyard** (only FoN itself ends in A's GY) → exiled, not graveyarded. Same for Thoughtseize via pitch. ✅
- Pitch on opponent's turn: A with **no lands** + FoN + Brainstorm; on B's turn B cast Thoughtseize → A offered "Cast Force of Negation", `Player A exiles Brainstorm` (no life cost), `Thoughtseize is countered`. ✅
- No pitch on your own turn: A with no lands, on A's own turn, B cast Lightning Bolt at instant speed → A's only option was "Pass priority" (FoN **not** castable, pitch unavailable on your turn). ✅
- All four requested properties hold.

## 77. Pyroblast ✅ (FIXED — color is a conditional effect, not a target filter)  /  ## 76. Hydroblast ✅
Pyroblast {R} — Charm: counter target spell if blue / destroy target permanent if blue. Hydroblast {U} — mirror (red).
- Spec (per user): they may TARGET any spell/permanent (target always legal); the counter/destroy only happens if the target is the appropriate color; the modes are offered whenever any spell is on the stack or any permanent is in play.
- Original bug: `ConditionPresent$ <type>.<Color>` was applied as a **target-legality filter** in `Ability::is_legal_target`, so they could only target matching-color objects and otherwise fizzled (`No valid modes — charm fizzles`).
- Fix:
  1. `src/components/ability.cpp` — removed the ConditionPresent color block from `is_legal_target` (targeting is now color-agnostic; ValidTgts color qualifiers like `Card.Red` are still honored — see note below).
  2. `src/effects/effect_counter.cpp` / `effect_destroy.cpp` — new helper `target_color_condition_met(ab, target)` (declared in `effects.h`) gates the EFFECT: counter/destroy only fires if the target has the required color, else logs "… is not the required color — not countered/destroyed" and the spell resolves doing nothing.
  3. `src/action_processor.cpp` `build_valid_targets` — a spell/ability can no longer target itself on the stack (a modal spell that picks its target at resolution is still on the stack); without this the auto-picked target was the blast itself.
- Verified matrix:
  - Pyroblast vs blue Brainstorm → `Brainstorm is countered`. Pyroblast vs red Bolt → offered & targets the Bolt, `Lightning Bolt is not the required color — not countered`, Bolt resolves.
  - Hydroblast vs red Bolt → `Lightning Bolt is countered`.
  - Pyroblast destroy vs blue Flying Men → `Flying Men is destroyed`; vs non-blue Grizzly → targetable, `Grizzly Bears is not the required color — not destroyed`.
- Regression: Counterspell still counters, Wasteland still destroys, `train.py --diag` 10 games 0 draws/no crashes.
- CONSISTENCY NOTE (for future Blue/Red Elemental Blast): those cards use `ValidTgts$ Card.Red` / `Permanent.Blue` (color in the *target spec*) → color-restricted targeting via the untouched `color_qualifier` path. Pyroblast/Hydroblast use `ConditionPresent$` → conditional effect, any target. The two mechanisms are now cleanly separated.

## 89. Stifle ✅
Instant {U} — counter target activated or triggered ability.
- Test: B activated Scalding Tarn (fetch); A responded with Stifle. `--battlefield-b "Scalding Tarn"` (fetchables in B's library), A `--battlefield-a "Island" --hand-a "Stifle"`, `--actions "0,0,1,1,0,0"`.
- Result: `Player B pays 1 life / Player B sacrifices Scalding Tarn` (costs paid), fetch ability on stack (`Scalding Tarn (ability, opponent)`); `Player A casts Stifle targeting <unknown> / <unknown> is countered`; the fetch never resolved (no land found). Correct — Stifle counters the ability, B loses the land+life for nothing. (`<unknown>` is cosmetic: a standalone ability has no card name.) ✅

---
## Burn / removal

## 29. Unholy Heat ✅
Instant {R} — 2 damage to creature/PW (6 with delirium).
- Test: `--battlefield-a "Mountain" --hand-a "Unholy Heat"` vs opp Grizzly Bears, `--actions "0,0,1,0"`.
- Result: `Dealt 2 damage to creature / Grizzly Bears is destroyed (lethal damage)`. Base 2-damage mode works. (Delirium 6-damage branch not separately set up; amount is SVar-computed `GE4.6.2`.) ✅

## 75. Abrade ✅
Charm {1}{R} — 3 damage to creature OR destroy artifact.
- Damage mode: vs Grizzly → `Dealt 3 damage to creature / Grizzly Bears is destroyed`.
- Destroy mode: vs Null Rod (artifact) → `Null Rod is destroyed` (→ Opp GY). Both Charm modes work. ✅

## 79. Dismember ✅ (incl. Phyrexian life payment)
Instant {1}{B/P}{B/P} — target creature gets -5/-5. ({B/P} = pay {B} or 2 life.)
- Hardcast (3 Swamps) vs Grizzly → `Grizzly Bears gets -5/-5 (now 0/0)` → graveyard.
- Phyrexian payment (1 Swamp only): casting prompts a `Pay cost` choice per {B/P}; chose life for both → `Player A pays 2 life` ×2 (20→18→16) and `activated Swamp for 1(B)` for the {1}. Final life 16, creature gets -5/-5. Phyrexian-as-life works. ✅

## 84. Fatal Push ✅ (revolt FIXED; conditional target; machine masking kept)
Instant {B} — destroy target creature with mana value ≤2; Revolt (a permanent you controlled left the battlefield this turn) → ≤4 instead.
- Original bug: revolt never worked — the destroy ability had no `Amount/NumDmg`, so its `X` SVar (`Count$Revolt.4.2`) was never wired to `dynamic_amount_expr`; `effects::destroy` always used the default threshold 2. (Also `ConditionDefined$ Targeted` was unparsed, and a machine-mode mask mis-gated cast legality.)
- Fixes:
  1. `src/parse.cpp` — when `ConditionPresent` contains `cmcLE<SVar>` with no explicit amount, wire that SVar into `amount_svar` so it resolves into `dynamic_amount_expr` (→ `Count$Revolt.4.2`).
  2. `src/parse.cpp` + `src/components/ability.h` — parse `ConditionDefined$ Targeted` into `condition_on_target` (the condition is checked on the chosen target at resolution, not board state at cast).
  3. `src/systems/state_manager.cpp` — skip `check_condition_present` cast-time gating for `condition_on_target` abilities (Fatal Push can legally target any creature). The machine-mode action-mask (don't offer to the RL agent when no creature is within the current revolt-aware threshold) is KEPT.
- Verified (effect): vs Grizzly (mv2) → destroyed; vs Air Elemental (mv5) → `mana value 5 (threshold 2) — not destroyed`; vs Knight of Autumn (mv3) with NO revolt → `threshold 2 — not destroyed`; vs Knight (mv3) WITH revolt (Lotus Petal sacrificed / fetch cracked first) → `Knight of Autumn is destroyed` (threshold 4).
- Verified (machine mask): vs Grizzly (mv2) → "Cast Fatal Push" offered; vs Knight (mv3) no revolt → NOT offered (masked); after cracking Scalding Tarn (revolt) → "Cast Fatal Push" offered and Knight destroyed.
- Regression: `train.py --diag` 0 draws / no crashes; Pyroblast/Counterspell still counter; Wasteland destroys.

## 83. Long Goodbye ✅
Instant {1}{B} — destroy creature/PW with mana value ≤3; can't be countered.
- Test: vs Grizzly Bears (mv2) → `Grizzly Bears is destroyed`. ✅ (can't-be-countered and the >3 exclusion not separately exercised.)

## 80. Meltdown ✅
Sorcery {X}{R} — destroy each artifact with mana value ≤X.
- Test: `--battlefield-a "Mountain,Mountain,Mountain" --hand-a "Meltdown"` vs opp Lotus Petal (mv0) + Null Rod (mv2), `--actions "0,0,7,2,…"`.
- Result: `Choose X value (0-2): / Player A chooses X = 2 / Lotus Petal is destroyed / Null Rod is destroyed`. X selection + cmc-gated mass destroy work. ✅

---
## Mana dorks / artifacts / static-ability permanents

## 38. Ignoble Hierarch ✅ (≈ Noble Hierarch)
0/1, Exalted, {T}: add {B}/{R}/{G}. Mana: `activated Ignoble Hierarch for 1(R)` cast Lightning Bolt (opp→17). Exalted = same keyword verified on Noble Hierarch below. ✅

## Exalted (Noble Hierarch #42 / Ignoble Hierarch #38) ✅
- Test: `--battlefield-a "Noble Hierarch,Grizzly Bears"`, attack with Grizzly **alone**.
- Result: `Resolving ability (category: ExaltedBonus) / Exalted (Noble Hierarch): Grizzly Bears gets +1/+1 until end of turn` → Grizzly `[3/3]`, dealt 3 to Player B. Triggers only when a creature attacks alone. ✅

## 51. Thalia, Guardian of Thraben ✅
2/1 first strike; noncreature spells cost {1} more.
- Test: with Thalia out, Lightning Bolt (normally {R}) needs {1}{R}: with 1 Mountain → NOT castable; with 2 Mountains → castable and dealt 3. Tax of exactly +1 confirmed. ✅ (First strike keyword present, not separately combat-tested. Name passed comma-free as "Thalia Guardian of Thraben".)

## 86. Magus of the Moon ✅
2/2; nonbasic lands are Mountains.
- Test: `--battlefield-a "Magus of the Moon,Volcanic Island"`. Lightning Bolt (R) → castable off the now-Mountain Volcanic Island (dealt 3). Brainstorm (U) → NOT castable (no blue, Volcanic Island only taps R). ✅

## 71. Null Rod ✅
Artifacts' activated abilities can't be activated.
- Test: `--battlefield-a "Null Rod,Lotus Petal" --hand-a "Lightning Bolt"`. At priority, Lotus Petal's mana ability is NOT offered and Lightning Bolt is NOT castable (no mana available). ✅

## 81. Deafening Silence ✅
Each player can't cast more than one noncreature spell each turn.
- Test: `--battlefield-a "Deafening Silence,Mountain×4" --hand-a "Lightning Bolt,Lightning Bolt"`. First Bolt cast (opp→17); afterwards the second Bolt is NOT offered as castable that turn. ✅

## 56. Lotus Petal ✅
Artifact, {T}, Sac: add one mana of any color.
- Verified during Fatal Push revolt test: `Player A activated Lotus Petal for 1(B)` (sacrificed for mana). ✅

## 57. Lion's Eye Diamond ✅
Artifact, {Sac, Discard your hand}: add three mana of one color (instant speed).
- Test: `--battlefield-a "Lion's Eye Diamond,Mountain"` + a 7-card hand; activate (index 2).
- Result: `Player A sacrifices Lion's Eye Diamond / Player A discards [entire hand]` then `Resolving ability (category: AddMana, amount: 3) / Player A adds 3{W}`. Sac + full-hand discard + 3 mono-color mana. ✅

## 27. Mishra's Bauble ✅
Artifact, {T}, Sac: look at top of target player's library; draw a card at the next turn's upkeep.
- Test: `--battlefield-a "Mishra's Bauble"`, activate (index 1).
- Result: `Player A sacrifices Mishra's Bauble / Player A looks at top of Player B's library: Forest / Delayed trigger registered: Draw at next Upkeep` → later `Delayed trigger fires / Player A draws`. Peek + slowtrip delayed draw work. ✅

## 82. Choke ✅
Enchantment — Islands don't untap during their controllers' untap steps.
- Test: `--battlefield-a "Choke,Island"`, cast Brainstorm (taps Island), advance turns.
- Result: Island stays `Island (T)` across subsequent untap steps (never untaps while Choke is in play). ✅

---
## Tutors / card advantage / combo

## 35. Green Sun's Zenith ✅
Sorcery {X}{G} — search a green creature with mana value ≤X to the battlefield, shuffle GSZ into library.
- Test: `--battlefield-a "Forest,Forest" --hand-a "Green Sun's Zenith,…"`, library has Birds of Paradise; `--actions "0,0,7,1,1"` (X=1, search Birds).
- Result: `Choose X value (0-1)` → `Search: Birds of Paradise` → `shuffles library` → `Birds of Paradise [0/1]` on battlefield; GSZ shuffled into library (not graveyard). cmc≤X gating works. ✅

## 43. Once Upon a Time ✅
Instant {1}{G} (free if first spell of the game) — look at top 5, reveal a creature/land to hand, rest to bottom.
- Test: `--hand-a "Once Upon a Time,…"`, top of library = Grizzly Bears; `--actions "0,0,1,1"`.
- Result: `Player A casts for free (alternate cost)` → `Resolving Dig` → dig offers `Grizzly Bears` (creature) to take to hand. Free-first-spell alt cost + dig work. ✅

## 31. Collector Ouphe ✅ (≈ Null Rod, on a creature)
2/2; artifacts' activated abilities can't be activated.
- Test: `--battlefield-a "Collector Ouphe,Lotus Petal"` → Lotus Petal's mana ability not offered, Lightning Bolt not castable (no mana). ✅

## 41. Knight of the Reliquary ✅
2/2; +1/+1 per land in your graveyard; {T}, Sac a Forest/Plains: search any land to battlefield.
- Test: `--battlefield-a "Knight of the Reliquary,Forest,Forest"`, library has lands; activate, sac a Forest.
- Result: `Player A sacrifices Forest` → Knight becomes `[3/3]` (1 land in GY); search offered any land (`Island`/`Forest`…) to battlefield. P/T scaling + land tutor work. ✅

## 49. Sylvan Library ✅
At your draw step, may draw 2 extra; for each of 2 cards drawn this turn, pay 4 life or put on top.
- Test: `--battlefield-a "Sylvan Library"`, reach draw step.
- Result: `Sylvan Library triggered / … draws Island / draws Mountain / draws 2 cards` (hand 7→9), then `Choose a card drawn this turn` ×2 and `For Forest: pay 4 life or put on top of library?` → paid → life 20→16. Full mechanic works. ✅

## 58. Thassa's Oracle ✅
1/3; ETB look at top X (devotion to blue); if X ≥ library size, you win.
- Test: manual small deck (`decks/temp/thassa_test.dk`: Thassa's Oracle + 7 Islands → 1-card library), `--battlefield-a "Island,Island"`, cast Thassa's Oracle.
- Result: `Resolving Dig` (X=2 devotion) then `Resolving WinsGame / Player A wins the game! (Thassa's Oracle)` (devotion 2 ≥ library 1). Win condition works. ✅

## 59. Personal Tutor ✅
Sorcery {U} — search library for a sorcery, reveal, shuffle, put on top.
- Test: library has Ponder; cast → `Put on top: Ponder / shuffles library`. ✅

## 60. Street Wraith ✅
3/4 swampwalk; Cycling—Pay 2 life.
- Test: activate cycling from hand → `Player A pays 2 life / Player A draws Island` (life 20→18, Street Wraith to GY). ✅ (Swampwalk keyword present, not separately combat-tested.)

## 61. Edge of Autumn ✅
Sorcery {1}{G} — if you control ≤4 lands, fetch a basic land tapped; Cycling—Sacrifice a land.
- Test: `--battlefield-a "Forest,Forest"` (2 lands), cast at first main → search basics → `shuffles library / Plains (T)` (entered tapped). Condition + tapped-fetch work; cycling ability also present (`Activate Edge of Autumn`). ✅

## 47. Scythecat Cub ✅
2/2 trample; Landfall — put a +1/+1 counter on target creature (double if 2nd landfall this turn).
- Test: `--battlefield-a "Scythecat Cub"`, play a Forest → landfall targets Scythecat → `Put 1 +1/+1 counter(s) on creature (now 3/3)`. Base landfall counter works. ✅ (Doubling on the 2nd landfall not separately exercised.)

---
## Creatures with abilities (set 2)

## 26. Murktide Regent ✅ (delve + etbCounter replacement)
{5}{U}{U} 3/3 flying, Delve; enters with a +1/+1 counter per instant/sorcery exiled with it; triggered +1/+1 when an instant/sorcery leaves your graveyard.
- Hardcast (empty GY) → `Murktide Regent [3/3]` (0 counters).
- Delve: cast 2 Lightning Bolts first (→ GY), then cast Murktide → `Player A exiles Lightning Bolt via Delve` ×2 → `Murktide Regent enters with 2 +1/+1 counter(s) (5/5)`. ✅
- Doorkeeper Thrull interaction (per user): with Doorkeeper Thrull in play, Murktide STILL `enters with 2 +1/+1 counter(s) (5/5)` — the enter-with-counters is a REPLACEMENT effect (`apply_permanent_components`, independent of the DisableTriggers trigger-suppression path), so Doorkeeper correctly does not prevent it. ✅

## 85. Doorkeeper Thrull ✅
1/2 flash flying; artifacts/creatures entering don't cause abilities to trigger.
- Test: with Doorkeeper + Soul Warden in play, casting Grizzly Bears → Soul Warden's ETB lifegain is SUPPRESSED (life stays 20, vs 21 without Doorkeeper). ✅ Correctly suppresses triggers but not replacements (see Murktide above).

## 20. Dragon's Rage Channeler ✅
{R} 1/1; surveil 1 whenever you cast a noncreature spell; Delirium → +2/+2 flying.
- Test: cast Lightning Bolt (noncreature) → `Dragon's Rage Channeler triggered / Resolving ability (category: Surveil, amount: 1)` with a surveil choice. ✅ (Delirium +2/+2 flying not separately set up — needs 4 card types in GY.)

## 73. Force of Vigor ✅
Instant {2}{G}{G} — destroy up to 2 target artifacts/enchantments; alt cost (not your turn): exile a green card.
- Test: vs opp Null Rod + Lotus Petal, select both → `Null Rod is destroyed / Lotus Petal is destroyed`. Multi-target (up to 2) works. ✅ (Pitch alt cost is the same mechanism verified on Force of Negation.)

## 88. Dauthi Voidwalker ✅ replacement / ⚠️ play-from-exile unimplemented
{B}{B} 3/2 shadow; if a card would go to an opponent's graveyard, exile it with a void counter instead; {T},Sac: play an exiled void-counter card for free.
- Replacement: Bolt opp's Grizzly → `Grizzly Bears is destroyed (lethal damage) / Grizzly Bears is exiled with a void counter` (not put in opp's graveyard). ✅
- Activated "play it for free" ability: relies on an **Effect-granted** `MayPlay` (`DB$ Effect | StaticAbilities$ MayPlay | RememberObjects$ ChosenCard`), which is **unimplemented** (those params unrecognized at parse). Note the *permanent static* MayPlay form works (see Icetill Explorer); only this one-shot Effect form is missing. Marked in `todo.md`. ⚠️ (Shadow keyword present, not separately combat-tested.)

## 87. Barrowgoyf ✅
{2}{B} */1+* deathtouch lifelink; power = number of card types among cards in all graveyards, toughness = that +1.
- Test: starts `[0/1]` (empty graveyards); after casting Lightning Bolt (1 card type, Instant, in GY) → `Barrowgoyf [1/2]`. Characteristic-defining P/T tracks graveyard card types. ✅ (Combat-damage mill trigger not separately exercised.)

## 37. Icetill Explorer ✅ (landfall + extra land + play-lands-from-GY all work)
{2}{G}{G} 2/4; extra land each turn; may play lands from graveyard; Landfall — mill a card.
- Landfall mill: playing a land → `Resolving Mill / Player A mills Island`. ✅
- Extra land play: after playing one land, more "Play land" actions remain available (additional land plays). ✅
- Play lands from graveyard: the milled Island in the graveyard is offered as `Play land: Island` and resolves (`Player A played Island` → onto battlefield). The permanent static `MayPlay` form works. ✅

## 18. Flying Men ✅ (vanilla flyer)
{U} 1/1 flying. Cast → `Flying Men [1/1]` enters. (Like Grizzly Bears #3 + Flying keyword.) ✅

## 21. Air Elemental ✅ (vanilla flyer)
{3}{U}{U} 4/4 flying. Cast off 5 Islands → `Air Elemental [4/4]` enters. (≈ Flying Men, bigger body.) ✅
