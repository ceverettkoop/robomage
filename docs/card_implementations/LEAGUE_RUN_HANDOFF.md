# League missing-cards run — HANDOFF / RESUME LEDGER

**Purpose:** durable, self-contained state for the multi-wave run implementing the 57 cards
referenced by `bin/resources/decks/league/*.dk` but missing from `src/card_vocab.h`.
A fresh session can resume from THIS file (the original was in the ephemeral session scratchpad).

## HOW TO RESUME (read first)
1. This is an `implement-missing-cards` run (interactive skill). To continue, invoke the
   **implement-missing-cards** skill and point it at this file and the "REMAINING" section below,
   OR just continue manually following the same per-card cycle.
2. Branch: **league**. Pre-branch reference HEAD (for the final review diff): **d1e5e7e**.
3. Build with **plain `make`** (GUI). `make HEADLESS=TRUE` FAILS to link on this machine
   (stale gui.o + raylib dropped); the user's standing preference is plain `make` and raylib IS
   present. Incremental `make` picks up header changes.
4. Test with `train/.venv/bin/python train/test_harness.py` + semantic `--play` specs.
   NEVER `--interactive` (no TTY). No commas inside card names in `--play`/`--hand` args
   (deck FILES are line-based and fine). Non-fatal errors and DRAWS are UNACCEPTABLE.
5. Commit rule: explicit `git add <paths>` only — NEVER `git add -A`, and NEVER stage
   `.claude/skills/implement-missing-cards/SKILL.md` (it has a pre-existing unrelated edit).
   One commit per card (or per shared-mechanic batch). End EVERY commit message with:
   ```
   Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
   Claude-Session: https://claude.ai/code/session_01M7847mUsvXYw7YLCWY32xL
   ```
   No model id in the commit subject. Do NOT push (unless the user asks).
6. **Next free vocab index = 269** (N_CARD_TYPES=1024 — NEVER change it, nor STATE_SIZE/OBS_SIZE).
   (241–266 = Wave 1; 267 The Fantasticar; 268 Cloak and Dagger, Entwined.)

## CURRENT STATUS (most recent first)
- 27 league cards implemented + committed (idx 241–268), all the originally-incomplete/broken cards fixed, plus 3 follow-up engine fixes. Tree clean, build GREEN. HEAD ≈ `329bfca`.
- NEXT: Wave 2 (small mechanics) — see "REMAINING TO IMPLEMENT" below. Sheltered by Ghosts / Static Prison are cheap now (reuse the UntilHostLeavesPlay mechanic).

## LOCKED USER DECISIONS (Phase 2)
1. Sequencing = WAVES, easiest-first, CHECK IN with the user between waves (let context reset).
2. FULL implementation of EVERY keyword — incl. Companion deckbuilding gate, full Daybound/Nightbound. No "out of scope" skips.
3. All 6 large mechanics at FULL rules fidelity (build reusable general engines).

## ====== WAVE 1 COMPLETE — 25 cards committed (idx 241–266), build GREEN, tree clean ======
Commits d1e5e7e..3587bd2 (25 "Implement" commits). Reusable mechanics added are listed per unit below.

- 241 Toxic Deluge, 242 Candelabra of Tawnos, 243 Hide on the Ceiling
  * VARIABLE-X mechanic: PayLife<X> (ability.life_cost_is_x, parse.cpp:212; spell_has_variable_life_cost game_queries.h:519; cast prompts X, sets spell.x_paid ~action_processor.cpp 1442/1533); {X} on activated ability (ability.activation_has_x, process_activate_ability ~action_processor.cpp 334); EXACTLY-X / UP-TO-X targeting (non-numeric TargetMin/TargetMax stashed parse.cpp:1093 → resolve_xpaid_target_counts(); select_target loop ~action_processor.cpp 1040). effect_untap.cpp untaps all ab.targets.
  * CAVEAT: Candelabra x_paid NOT persisted past activation; an ability whose RESOLUTION reads Count$xPaid would need x_paid persistence added.
- 244 Baleful Strix, 245 Blue Elemental Blast, 246 Expedition Map, 247 Gaddock Teeg, 248 Grim Monolith, 249 Stony Silence, 250 Voltaic Key, 251 Manifold Key
  * CantBeCast honors ANY ValidCard$ characteristic filter (cast_prohibited(spell CardData&) rules_modifying.{h,cpp}; hasXCost filter game_queries.cpp; 5 cast sites in state_manager_actions.cpp).
  * effect_untap.cpp: untargeted AB$ Untap falls back to ab.source.
  * SKIP_UNTAP self replacement (parse.cpp + replacement_effects.cpp; ValidCard$ Card.Self + Event$ Untap + Layer$ CantHappen → applies_to_self_only).
- 252 Boomerang Basics, 253 Liquimetal Coating, 254 Mole Man Moloid Master, 255 Mystic Sanctuary, 256 Pernicious Deed, 257 Pick Your Poison, 258 Prismatic Vista, 259 Price of Freedom
  * RememberLKI$ + ConditionDefined$ RememberedLKI: ability.remember_lki; effect_change_zone.cpp pushes moved entity to remembered set; evaluate_present_condition resolves off-battlefield remembered card controller from cur_game.last_known_info[e].controller (CR 608.2g).
  * with<Keyword> filter qualifier (game_queries.cpp eval_qualifier: Creature.withFlying etc.).
  * Ignored cosmetic params (no warnings): SacMessage, ChangeTypeDesc, ShuffleNonMandatory. Fetched moloid token script.
- 260 Witherbloom Command, 261 Urza's Workshop, 262 Ugin Eye of the Storms, 263 Witch Enchanter (front), 264 Witch-Blessed Meadow (back — REGISTERED but DEFERRED, see INCOMPLETE), 265 Toxicrene, 266 Planar Nexus
  * effect_mill.cpp / effect_lose_life.cpp honor a Player-entity target.
  * ability.cpp identical_activated_ability compares activation_condition + dynamic_amount_expr.
  * mana_system.cpp eval_mana_amount: Count$Valid <filter> evaluates any battlefield filter.
  * state_manager_statics.cpp: removal_affects generalized to any Affected$ filter (RemoveAllAbilities on Land); self-CDA AddType$ AllNonBasicLandType.
  * parse.cpp: ForgetOnMoved ignored.

## ====== FORMERLY-INCOMPLETE CARDS — ALL FIXED ======
These four were triaged-covered but turned out to need mechanics (or were partial); all now DONE:

| Card | State | What it needs | Target wave |
|---|---|---|---|
| Cloak and Dagger, Entwined | ✅ FIXED — idx 268 (fb5ff06) | Built GENERAL `Duration$ UntilHostLeavesPlay` exile-return (register_exile_until_host_leaves, effect_change_zone.cpp:78). Sheltered/Static Prison reuse this. | DONE |
| The Fantasticar | ✅ FIXED — idx 267 (9e151b6) | Built until-EOT Animate (animate_added_types_eot, revert at CLEANUP game.cpp). | DONE |
| Witch-Blessed Meadow (idx 264) | ✅ FIXED — back face functional (14e2e2f) | Built general MODAL-DFC play-from-hand (CardData::is_modal_dfc; LegalAction::play_back_face) + pay-life-or-enter-tapped replacement (Effect::Replacement::tapped_unless_life). | DONE |
| Ugin, Eye of the Storms (idx 262) | ✅ FIXED — ultimate complete (93844c1) | Built FREE cast-from-exile grant (Game::ImpulseCastPermission::FREE; ability.effect_grant_free_cast_from_exile). Verified driving real Ugin to 11 loyalty. | DONE |

### Mechanics now available from the fix-four sub-task (reuse these):
- `Duration$ UntilHostLeavesPlay` exile-return — DONE. **Sheltered by Ghosts / Static Prison** now only need their scripts' `ChangeZone|Destination$ Exile|Duration$ UntilHostLeavesPlay` targeting a permanent (origin battlefield); the targeted ChangeZone path auto-registers the return. No new engine code.
- until-EOT Animate (Permanent.animate_added_types_eot/animate_make_creature_eot) — DONE.
- MODAL-DFC play-from-hand + pay-life-or-tapped replacement — DONE (future MDFCs: just `AlternateMode:Modal` on front script; land back needs no extra work).
- FREE cast-from-exile grant via Effect (MayPlayWithoutManaCost on remembered exiled cards until EOT) — DONE.

## ====== POST-WAVE-1 ENGINE FIXES (modal-DFC follow-ups, no new vocab) ======
- **Back-face action card-id (fdd9368):** `action_card_vocab_idx(const LegalAction&)` overload (machine_io.cpp:71) resolves the BACK face's vocab id for a `play_back_face`/`cast_back_face` action (front-face source entity would mis-report it). BQUERY + action log both route through it. Verified: Play Witch-Blessed Meadow → 264; Cast Witch Enchanter → 263. (Supersedes the old "KNOWN MINOR" note.)
- **Modal-DFC nonland back faces (a0e66b7):** generalized MDFC play-from-hand to NONLAND backs — `LegalAction::cast_back_face` (action.h:88); `offer_modal_back_face_casts()` (state_manager_actions.cpp ~344); CAST_SPELL path (action_processor.cpp ~1244) rebinds card_data to `*backside` so cost/SP$/targeting/events/log source from the back; a permanent back enters via `pending_enters_transformed`. Verified with Halvar // Sword of the Realms (creature // equipment). USER ARCH DECISIONS: face model LEFT AS-IS (transform-flag reuse, no anti-transform guard); scope GENERALIZED to nonland backs.
- **DFC flicker — non-permanent front stays exiled (329bfca):** CR 110.4a/712.10. `change_zone_move` (effect_change_zone.cpp:42) refuses to move a non-permanent-front card onto the battlefield; it stays in its current zone (like a Grafdigger's Cage divert). Fixes flicker (Flickerwisp/Phelia) of a modal DFC in play on its back face whose FRONT is instant/sorcery (e.g. Fell the Profane // Fell Mire) — now stays in EXILE instead of limbo-on-battlefield. A permanent-front DFC (Witch Enchanter // Witch-Blessed Meadow) still returns as its untransformed front face with ETBs intact. General (covers Phelia, reanimation, etc.).
- KNOWN PRE-EXISTING (not in worklist, flag for later): parser std::stoi crash on `UnlessCost$ Sac<...>` at parse.cpp:1016 (surfaced via Tergrid's Lantern back face).

DFC back-face registration convention: existing engine registers BOTH faces (Ajani 100/101, Delver 16/17). So when implementing the remaining DFCs, register the back face too: Outland Liberator → back "Frenzied Trapbreaker"; Tamiyo Inquisitive Student → back "Tamiyo, Seasoned Scholar"; (Witch-Blessed Meadow already at 264).

## ====== REMAINING TO IMPLEMENT (32 cards), grouped by mechanic ======
(Verify each triage claim — the Wave-1 "covered" list had 3 wrong calls. Group cards sharing a mechanic into ONE batch unit, mechanic written once.)

### Wave 2 — SMALL mechanics
- mechanic:UntapAll — **Paradox Engine** (on your cast, untap all nonland permanents you control). Add an UntapAll effect mirroring the other *All effects.
- mechanic:CantAttack static — **Ensnaring Bridge** (creatures with power > cards-in-your-hand can't attack; X = Count$ValidHand). New static mode like CantBeCast.
- mechanic:SetCostMin — **Trinisphere** (each spell costs at least {3}; distinct from RaiseCost surcharge). New static.
- mechanic:Threshold count — **Cabal Ritual** (Count$Threshold.5.3: add {B}{B}{B}, or {B}{B}{B}{B}{B} if 7+ cards in your graveyard). Add Count$Threshold.<hi>.<lo> to the dynamic-amount evaluator.
- mechanic:TypeCycling — **Lorien Revealed** (Islandcycling {1}; also "draw 3" main spell). Add TypeCycling:<Subtype>:<cost> as a cycling variant that tutors a land of that subtype.
- mechanic:CountUrzaLands conditional — **Urza's Mine**, **Urza's Power Plant**, **Urza's Tower** (Count$UrzaLands.<hi>.<lo>: produce <lo> normally, <hi> if you control all three Tron pieces). One batch.
- mechanic:dynamic-animate-PT — **Karn, the Great Creator** (planeswalker; +1 Animate target noncreature artifact into an X/X where X = its mana value, Power$ X/Toughness$ X = Targeted$CardManaCost; −2 fetch artifact from exile/sideboard to hand; static: opp artifacts' activated abilities can't be activated [CantBeActivated already supported]). The literal-X parse fix is DONE; needs Power/Toughness$ X handling in Animate + dynamic P/T on the animated permanent.
- mechanic:ProduceManaReplacement — **Damping Sphere** (lands produce only {C} & only one at a time [R:Event$ ProduceMana]; RaiseCost +{1} per other spell already supported). New replacement event ProduceMana.
- mechanic:CantBeCountered-static — **Hexing Squelcher** (spells you control can't be countered [R:Event$ Counter static]; Ward already supported). New can't-be-countered replacement (also unblocks reuse).
- mechanic:escaped-flag-condition — **Uro, Titan of Nature's Wrath** (Escape covered by Nethergoyf; ETB gain 3 + draw + play a land; sacrifice unless it escaped → needs ConditionNotPresent$ Card.Self+escaped, i.e. an "escaped" flag on the permanent).

### DurationUntilHostLeavesPlay batch (Wave 2/3)
- **Sheltered by Ghosts** (Aura: ETB exile target until this leaves; enchanted creature +1/+0, lifelink, ward 2), **Static Prison** (ETB exile target until this leaves; energy upkeep tax), **Cloak and Dagger** (Wave-1 blocked). Build the until-source-leaves continuous/exile subsystem once.
- AnimateUntilEOT: **The Fantasticar** (Wave-1 blocked).

### Wave 3 — MEDIUM mechanics
- mechanic:Gift — **Into the Flood Maw** (parser X/Y already fixed; needs Gift keyword: promise opponent a gift as you cast → they make a tapped 1/1 Fish, then the spell bounces a nonland permanent instead of just a creature; Count$PromisedGift.x.y).
- mechanic:Reconfigure — **Lion Sash** (equip-like attach to a creature; {W}: exile a card from a graveyard, put +1/+1 counter; static AddPower = counter count).
- mechanic:Ninjutsu + mechanic:StunCounter — **Kaito, Bane of Nightmares** (planeswalker; Ninjutsu 1UB; becomes 3/4 hexproof on your turn; emblem via Effect; Surveil; tap+stun counters). Stun counter = skip-next-untap counter.
- mechanic:Unearth — **Cityscape Leveler** (Unearth {8}: return from GY, gains haste, exile at EOT/leaves; trample; destroy triggers on cast/attack; powerstone token).
- mechanic:Companion (FULL per decision #2) — **Yorion, Sky Nomad** (flying; ETB blink any number of your permanents, return EOT; Companion deckbuilding gate — implement the from-sideboard companion mechanic too).
- mechanic:in-ability-ReplacementEffects + conditional-hexproof — **Veil of Summer** (draw if opp cast blue/black this turn; your spells/abilities can't be countered by opp + you/permanents gain hexproof from blue & black this turn).
- mechanic:protection-from-everything + wasCastByYou + burden — **The One Ring** (indestructible; ETB protection from everything until your next turn; upkeep lose life = burden counters; {T}: add a burden counter, draw that many).

### Wave 4 — LARGE mechanics (full rules fidelity)
- mechanic:Saga (Chapter/lore counters, CR 714) — **Urza's Saga** (land Saga: I gives "{T}: add {C}", II gives "{T}: add {C}{C}, spend only on artifacts/abilities" + becomes a 0/0 construct? no — makes a Construct token */* = artifacts you control; III tutor a 0-or-1-mana artifact) AND **Summon: Bahamut** (enchantment creature Saga, FF set; chapters I/II destroy up to one target, III draw 2, IV Mega Flare damage = total CMC of other permanents). Build a general Saga engine: lore counter on ETB + each of your draw steps, chapter triggers, sacrifice after final chapter (Sagas that are also creatures stay). 
- mechanic:Storm (CR 702.40) — **Flusterstorm** (counter target instant/sorcery unless pay {1}; Storm = copy for each spell cast before it this turn, each copy may choose new targets).
- mechanic:DayNight (Daybound/Nightbound, CR 726, FULL) — **Outland Liberator // Frenzied Trapbreaker** (DFC werewolf; day/night state, transforms; {1},Sac: destroy artifact/enchantment).
- mechanic:Impending (time counters) — **Overlord of the Balemurk** (Impending 5—{1}{B}: enters with 5 time counters, not a creature until last removed; remove one each of your upkeeps; ETB/attack mill 4 then return a creature/PW to hand).
- mechanic:extra-turn (AddTurn) + mechanic:Annihilator (CR 702.85) — **Emrakul, the Aeons Torn** (can't be countered; when you cast it, take an extra turn; flying, protection from colored spells, Annihilator 6; when put into GY from anywhere, shuffle GY into library).
- mechanic:Mycosynth (global static, full) — **Mycosynth Lattice** (all permanents are artifacts [AddType supported]; all cards/permanents are colorless [SetColor static — NEW]; all mana spent as any color [ManaConvert static — NEW]).
- **Tamiyo, Inquisitive Student // Tamiyo, Seasoned Scholar** (DFC planeswalker): front creature — Attacks → Investigate (Clue token, NEW effect); "when you draw your 3rd card each turn" (Drawn-count trigger Number$ 3, NEW) → transform; back PW — loyalty abilities incl. emblem-via-Effect, ultimate SetMaxHandSize Unlimited (NEW). Register back "Tamiyo, Seasoned Scholar" too.

## FINAL (Phase 4, after all cards): run /code-review at medium over d1e5e7e..HEAD; fix low-risk findings (each its own commit); report tally; confirm green + clean; push only if user asks.
