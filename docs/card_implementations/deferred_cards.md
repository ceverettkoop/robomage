# Deferred cards — for user-directed implementation

Cards the autonomous runs triaged but did NOT implement. Each needs a human decision or a larger
change than an unattended run should make. Pick one up by running the interactive
`implement-missing-cards` skill (or directing a fresh autonomous run at it once unblocked).

## Run 2026-06-26  (cap N=30, inspected 60 cards)

### Deferred at triage

Each card below has a local Forge script available; the blocker is a missing engine subsystem
(grouped by the mechanic that needs to be built first). Implementing one card in a group usually
unblocks the others in that group.

**Energy** — all three energy cards (Amped Raptor, Guide of Souls, Wrath of the Skies) were
implemented in the 2026-06-27 run below; the energy subsystem now exists.

**Adventure / split-mode cards** (cast one half, exile, recast other half later):
- **Brazen Borrower Petty Theft** — Adventure (`AlternateMode:Adventure`) (Forge script: available)

**Modal double-faced cards** (choose which face to play from hand; pay-life-or-enters-tapped land):
- **Witch Enchanter Witch-Blessed Meadow** — MDFC + `R:Moved ReplaceWith DBTap UnlessCost PayLife<3>` (Forge script: available)

**Planeswalker / loyalty-adjacent multi-mechanic cards** (loyalty framework exists, but each stacks several other new mechanics):
- **Tamiyo, Inquisitive Student Tamiyo, Seasoned Scholar** — `DB$ Investigate` (Clue tokens) + `Mode$ Drawn Number$ N` (Nth-card-this-turn) + `SetMaxHandSize` emblem (Forge script: available)
- **Kaito, Bane of Nightmares** — Ninjutsu + planeswalker-becomes-creature animation + emblem static + tap/stun counters (Forge script: available)

**Other missing subsystems:**
- ~~**Consign to Memory** — Replicate keyword + countering a triggered ability on the stack (Forge script: available)~~ **IMPLEMENTED** (general Replicate keyword: `K:Replicate:<cost>` parsed into `CardData::replicate_cost`/`has_replicate`; repeatable optional `OPTIONAL_YESNO` payment at cast folds mana into the cast cost and records `Spell::replicate_count`; general `copy_spell_on_stack` routine creates N copies on the stack, each re-running target selection, marked `Spell::is_copy` so they cease to exist on resolve/counter; `TargetType$ Spell.Colorless,Triggered` now matched via a comma-split OR matcher with Colorless support; CR 702.x Replicate, 707 copies; see `docs/card_implementations/consign_to_memory.md`)
- ~~**Petrified Hamlet** — `DB$ NameCard` effect + Continuous `AddAbility$` granting a mana ability to other named lands (Forge script: available)~~ **IMPLEMENTED** (general layer-6 `AddAbility$` continuous ability-grant static, reusing the `Affected$` resolver; per-source named-card via `Permanent::chosen_name`; see `docs/card_implementations/petrified_hamlet.md`)
- **Pinnacle Emissary** — Warp alternative cost (cast for warp cost, exile at end step, recast from exile) (Forge script: available)
- ~~**Wastescape Battlemage** — Kicker (multi-kicker cost choice at cast + `kicked N` SpellCast condition) (Forge script: available)~~ **IMPLEMENTED** (general Kicker mechanic: `K:Kicker:<c1>[:<c2>...]` parsed into `CardData::kicker_costs`; per-kicker optional `OPTIONAL_YESNO` payment folded into the cast's total cost; per-`Spell::kicked` flags; `ValidCard$ Card.Self+kicked N` SpellCast trigger condition matched in a dedicated self-cast SPELL_CAST trigger scan; CR 702.33; see `docs/card_implementations/wastescape_battlemage.md`)
- ~~**Eiganjo, Seat of the Empire** — Channel-from-hand works, but `ReduceCost$ X` on an activated ability (cost reduced per legendary creature) is unsupported (Forge script: available)~~ **IMPLEMENTED** (uid `eiganjo_seat_of_the_empire`; general SVar/`Count$Valid` `ReduceCost$` on an activated ability, reduces generic mana only per CR 601.2f, applied to both the affordability gate and payment via `effective_activation_mana_cost`; see `docs/card_implementations/eiganjo_seat_of_the_empire.md`)
- ~~**It That Heralds the End** — anthem-style broadcast `+1/+1` to an `Affected$` class + spell cost-reduction static (Forge script: available)~~ **IMPLEMENTED** (general `Affected$`-filter resolver for continuous P/T statics + `cmcGE7` static mana-value bound on the `ReduceCost` filter; see `docs/card_implementations/it_that_heralds_the_end.md`)
- ~~**Lorehold Charm** — `DB$ PumpAll` (no PumpAll handler) + targeted graveyard→battlefield reanimation (Forge script: available)~~ **IMPLEMENTED** (general `PumpAll` mass-pump handler + targeted graveyard→battlefield reanimation targeting/control; see `docs/card_implementations/lorehold_charm.md`)
- ~~**Mox Amber** — `ManaReflected` (produce one mana of any color found among legendary permanents you control) (Forge script: available)~~ **IMPLEMENTED** (general `AB$ ManaReflected` mana ability: a CR 605 mana ability resolved off-stack whose producible colors are the union of `effective_colors` over the `Valid$`-matching permanents you control, expanded into per-color choices like `mana_choices`; empty set produces nothing; see `docs/card_implementations/mox_amber.md`)
- ~~**Alpha Deathclaw** — Monstrosity (`Monstrosity$` on PutCounter + `BecomeMonstrous` trigger) (Forge script: available)~~ **IMPLEMENTED** (general Monstrosity per CR 701.37: `Permanent::is_monstrous` internal state, `Monstrosity$ N` on `PutCounter` → N +1/+1 counters + monstrous designation, a `NotMonstrous` activation gate, and a new `BECAME_MONSTROUS` event firing `Mode$ BecomeMonstrous` triggers; see `docs/card_implementations/alpha_deathclaw.md`)
- ~~**Canoptek Scarab Swarm** — `ChangeZoneAll RememberChanged` + `Remembered$Valid` count expr to size a token (Forge script: available)~~ **IMPLEMENTED** (`ChangeZoneAll` now populates `remembered_entities` when `RememberChanged$ True`, mirroring single-target `ChangeZone`; new general `Remembered$Valid <comma-OR-filter>` dynamic-amount handler counts remembered cards via `card_matches_filter`; see `docs/card_implementations/canoptek_scarab_swarm.md`)
- **Urza's Saga** — Saga/Chapter/lore-counter system + `DB$ Animate` (grant abilities to self) + `ManaCost0/1` search filter (three subsystems) (Forge script: available)
- **Atraxa, Grand Unifier** — per-card-type reveal/tutor ETB (`RepeatTypesFrom`, `ImprintRevealed`, `ChosenType`) (Forge script: available)
- **Council's Judgment** — Vote / will-of-the-council subsystem (Forge script: available)
- **Damping Sphere** — `ProduceMana`/`ReplaceMana` replacement + dynamic per-spell-this-turn `RaiseCost` (`Relative$ True`) (Forge script: available)
- ~~**Forth Eorlingas!** — Monarch subsystem (`BecomeMonarch` + monarch tracking) + `DB$ Effect` with floating `Triggers$` (Forge script: available)~~ **IMPLEMENTED** (general Monarch subsystem: `Game::monarch_entity`/`set_monarch` + `DB$ BecomeMonarch` effect + sourceless inherent monarch triggers — end-step extra draw and steal-on-combat-damage — fired from `check_triggered_abilities` per CR 725.2; general until-end-of-turn floating triggered ability via `DB$ Effect | Triggers$ <SVar>` registered in `Game::floating_triggers`; new `Mode$ DamageAll | ValidSource$ Creature.YouCtrl` combat-damage trigger; see `docs/card_implementations/forth_eorlingas.md`)

### Deferred during implementation

- **Phelia, Exuberant Shepherd** — triaged "covered" but misclassified: its exile→return chain
  uses `RememberObjects$ RememberedLKI` / `Defined$ DelayTriggerRememberedLKI` / `Imprint$`, none
  handled, and the `DB$ DelayedTrigger` path maps `Phase$ "End of Turn"` to the wrong step. Needs
  real new mechanic code (remembered-LKI passing from a ChangeZone exile into a delayed trigger and
  back, + an end-of-turn phase alias). Flickerwisp shares this template. (Forge script: available)

### Not reached (cap)

None. The N=30 cap was met exactly: 31 cards were triaged implementable; Phelia deferred during
implementation (above), and the other 30 were all implemented — so no implementable card was left
unstarted by the cap.

### Review findings (post-implementation code review, for human follow-up)

The Phase-3.5 `code-review` (medium) found no shipped-card bugs beyond two low-risk issues already
fixed this run (ward `PayLife` `std::stoi` guard; `.Other` self-exclusion tokenization, commit
`b6873c1`). The following are **latent fragilities / altitude items** left for a human — none
breaks a currently-implemented card, but each is worth a deliberate fix:

- ✅ **RESOLVED** (branch `claude/filter-matcher-unify`) — **Four parallel permanent/card filter
  matchers** (`matches_filter_spec`, `permanent_matches_cards_filter`,
  `permanent_matches_subtype_spec`, `card_matches_reduce_filter`) carried **diverging qualifier
  coverage**. Consolidated onto one shared evaluator with two entry points
  (`card_matches_filter` / `permanent_matches_filter`) in `game_queries`; coverage is now the union
  across all sites, and the live `non<Color>` bug (e.g. `nonWhite` never excluding) is fixed.
- ✅ **RESOLVED** (same branch) — **Fail-closed on unknown qualifiers:** the unified evaluator still
  fails closed on a qualifier it can't interpret, but now **warns once** so unrecognized specs
  surface during testing instead of silently matching nothing. (`non<Color>` mass filters are now
  handled, not just failed-closed.)
- **NameCard candidate set** (`effect_name_card.cpp`) is restricted to nonland vocab cards in the
  named player's zones — a Cabal-Therapy-shaped approximation of CR 201.4's "name any card."
  *(Still open — unrelated to the filter-matcher refactor.)*
- ✅ **RESOLVED** (same branch) — **`Defined$ TargetedController` / `TriggeredActivator` /
  last-known-controller** resolution was open-coded across a few effects. Centralized into
  `source_controller` / `last_known_controller` / `resolve_defined_player` in `game_queries`, and
  the leaving permanent's controller is now captured in `LastKnownInfo` so "that permanent's
  controller" resolves even after it has left the battlefield (fixed the previously-dead fallback
  that could deal 0).

## Run 2026-06-27  (implement-deferred, N=10 — no deferrals)

User-directed `implement-deferred-cards` run: 10 cards pulled from the deferred queue above and
**all implemented** (one commit per card / shared-mechanic batch on branch
`claude/autonomous-card-batch-8072th`):

| Card | vocab | commit | new mechanic(s) |
|---|---|---|---|
| Guide of Souls | 171 | `442f8c9` | energy foundation (player counter map, `PayEnergy`, Animate type-add, keyword counters, immediate-trigger optional cost, AttackersDeclared) |
| Wrath of the Skies | 186 | `b1bc865` | `ChooseNumber`, dynamic-CMC `DestroyAll` + PayEnergy-unless |
| Amped Raptor | 170 | `d152cf3` | `DigUntil`, general resource-parameterized alt-cost cast (`DB$ Play`) |
| Badgermole Cub + Ba Sing Se | 175, 197 | `3185ab4` | Earthbend (land→creature Animate), leaves-battlefield return-tapped delayed trigger, `TapsForMana`, conditional ETB-tapped |
| Mox Opal | 176 | `ed24922` | general `Activation$` gate + Metalcraft predicate |
| Nethergoyf | 177 | `5dc0a6d` | controller-scoped `CardTypes` CDA P/T, Escape keyword + `ExileFromGrave` group cost |
| Reality Smasher | 181 | `529cac2` | general `BecomesTarget` trigger (Ward left untouched), `TriggeredSourceSA`, discard-unless-cost |
| Thought-Knot Seer | 182 | `396ec38` | `RevealHand` effect, `Chooser$ You` cross-player hand exile |
| Skyclave Apparition | 188 | `7650318` | `TokenOwner$ RememberedOwner`, SVar-sized token P/T, ExiledWithSource condition |

### Review findings (post-implementation code review 2026-06-27, for human follow-up)

A medium `code-review` over the full run diff (3 finder passes) found **no bug in any of the 10
shipped cards** — each was verified behaving correctly and the scripted regression is green. The
following were **latent fragilities in the *generalized* mechanisms** (they would bite a *future*
card, not a current one). **All seven were addressed on branch `claude/deferred-cards-review-3c5jsj`**
(one commit each; each verified with the existing vocab cards that reach the path, with code review
for the specific bug branches no shipping card can reach):

- ✅ **RESOLVED** — **`is_legal_target` controller/token filter matched substrings over the whole
  `ValidTgts$` string, not per-clause** (`src/components/ability.cpp` ~640–660). (a) A comma-joined
  `…YouCtrl,…OppCtrl` spec set both flags and rejected every candidate; (b) a filter whose text
  contained `token` without the `!token`/`nonToken` negation was wrongly treated as token-only.
  Fix: the battlefield-permanent branch now routes through the shared `permanent_matches_filter`
  evaluator (`game_queries`) per OR-clause, bridging the `ValidTgts` grammar (',' OR, `YouDontCtrl`,
  `Any`) to it; protection (CR 702.16e) stays a separate check. Verified: Abrupt Decay
  (`nonLand+cmcLE3`), Skyclave Apparition (`nonLand+!token+YouDontCtrl+cmcLE4` — own permanents
  correctly excluded).
- ✅ **RESOLVED** — **Activated `Cost$ PayEnergy<N>` was parsed into `Ability::energy_cost` but
  never paid or gated** (`src/systems/state_manager_actions.cpp` legality + `src/action_processor.cpp`
  `pay_secondary_activation_costs`). Now gated for affordability (both battlefield- and hand-activated
  paths) and paid alongside the life cost. Inert for all current cards (no activated ability carries
  `energy_cost`; Guide of Souls' is on an ImmediateTrigger).
- ✅ **RESOLVED** — **Single permanent-target `PutCounter` ignored dynamic `CounterNum$`
  (`count_expr`) and the second counter (`CounterType2$`/`CounterNum2$`)**
  (`src/effects/effect_put_counter.cpp` ~51). The targeted branch now evaluates `count_expr` (as the
  `Defined$ You` branch does) and applies the second counter (as `put_counter_all` does); the
  static-count path is unchanged. Latent (Wrath uses `Defined$ You`; Guide uses `PutCounterAll`).
- ✅ **RESOLVED (guarded)** — **Non-switched energy `UnlessCost` in `DestroyAll` paid energy
  unconditionally and destroyed regardless** (`src/effects/effect_destroy_all.cpp`). Investigation
  confirmed **Wrath of the Skies is correct** — it is switched (`UnlessSwitched$ True`): you get X
  {E}, choose/pay an amount, and destroy each artifact/creature/enchantment with MV ≤ the **energy
  paid** (verified by board test: pay 2 → MV≤2 destroyed; gain 3/pay 1 → only MV≤1 destroyed). The
  switched single-payment path is kept; the non-switched genuine "destroy each X unless **its**
  controller pays {E}" (a per-permanent payment, paying *prevents* destruction) is unsupported and no
  shipping card uses it, so it now fails closed with a one-time warning instead of charging the
  spell's controller and destroying regardless.
- ✅ **RESOLVED** — **The generic `ValidPlayer$ You` trigger gate was not exempted for
  `BECAME_TARGET`** (`src/systems/state_manager_triggers.cpp` ~180), where the event's PLAYER param is
  the *targeting* spell's controller. Added the same `BECAME_TARGET` exemption `trigger_only_self`
  already had. Latent: Reality Smasher uses `ValidSource$` (verified still firing its
  counter-unless-discard).
- ✅ **RESOLVED** — **`Count$ValidGraveyard Card.YouOwn$CardTypes` honored only the ownership
  restriction; any extra type/CMC subfilter was ignored** (`src/svar_eval.cpp` ~153). Ownership stays
  manual (a graveyard card has no live controller); any remaining qualifiers now route through
  `card_matches_filter`. Verified: Nethergoyf (unchanged 3/4 from a 3-type graveyard).
- ✅ **RESOLVED** — **`can_afford_with_sources()` omitted the `TapsForMana` additional-mana bonus**
  (`src/mana_system.cpp`), diverging from the real payment path for *nested* mana-source activation
  costs. Now fires the same `fire_taps_for_mana_triggers` helper per counted source in both passes.
  Narrow and latent (no shipping mana source has a mana activation cost); spell-cast affordability
  already used the correct `can_pay_mana` path (Badgermole's bonus verified still letting a single
  Birds of Paradise pay a `{1}{G}` cost).

## Run 2026-06-28  (cap N=30, inspected 34 cards — full remaining worklist)

The full missing-card worklist held only 34 cards, all with local Forge scripts. Triage classified
11 as already-covered, 13 as bounded new-mechanic work (implemented this run), and the 10 below as
deferred — each needs a sizeable engine subsystem an unattended run should not build blind.

### Deferred at triage
- **Tamiyo, Inquisitive Student / Tamiyo, Seasoned Scholar** — front/back DFC where a *creature*
  transforms into a *planeswalker*, plus `Investigate` (Clue tokens), a "third card drawn this turn"
  draw-count threshold trigger, and an emblem; several new subsystems at once (Forge script: available)
- **Brazen Borrower / Petty Theft** — Adventure (`AlternateMode:Adventure`): cast the instant half,
  exile it, later cast the creature half from exile (Forge script: available)
- **Pinnacle Emissary** — `K:Warp` (suspend-like alternative cast that exiles then recasts from
  exile) + a missing Drone token script (Forge script: available)
- **Urza's Saga** — Saga subsystem (`K:Chapter:N:...` lore counters / chapter abilities / final-chapter
  sacrifice) + a missing Construct token + a `ManaCost0/1` library-search filter (Forge script: available)
- **Kaito, Bane of Nightmares** — `Ninjutsu` keyword + `Stun` counters + a planeswalker that becomes a
  creature on your turn (Forge script: available)
- **Atraxa, Grand Unifier** — `RepeatEach` iterating over *card types* (`RepeatTypesFrom$`) with a
  per-type `ChosenType` ChooseCard; a new repeat-over-types iterator (Forge script: available)
- **Eladamri, Korvecdal** — casting creatures off the top of your library (`MayPlay$` top-of-library
  + `MayLookAt$`), a tap-two-other-creatures activation cost, and a Hand+top-of-library ChooseCard
  (Forge script: available)
- **Grist, the Hunger Tide** — `AB$ Repeat` loop primitive (`RepeatCheckSVar`), a reflexive `Sac`
  unless-cost, and an off-battlefield characteristic-defining "1/1 Insect everywhere but the
  battlefield" static (Forge script: available)
- **Springheart Nantuko** — `Bestow` keyword + a Landfall ("a land you control enters") triggered
  ability + copy-the-enchanted-creature (Forge script: available)
- **Yorion, Sky Nomad** — `Companion` deck-building keyword (cast from outside the game) + a
  player-chosen "any number" mass self-blink with delayed end-of-turn return (Forge script: available)

### Not reached (cap)
None — the N=30 cap exceeded the 24 implementable cards in the worklist, so every covered and
bounded-mechanic card was attempted this run.
