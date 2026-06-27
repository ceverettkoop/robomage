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
- **Consign to Memory** — Replicate keyword + countering a triggered ability on the stack (Forge script: available)
- **Petrified Hamlet** — `DB$ NameCard` effect + Continuous `AddAbility$` granting a mana ability to other named lands (Forge script: available)
- **Pinnacle Emissary** — Warp alternative cost (cast for warp cost, exile at end step, recast from exile) (Forge script: available)
- **Wastescape Battlemage** — Kicker (multi-kicker cost choice at cast + `kicked N` SpellCast condition) (Forge script: available)
- **Eiganjo, Seat of the Empire** — Channel-from-hand works, but `ReduceCost$ X` on an activated ability (cost reduced per legendary creature) is unsupported (Forge script: available)
- **It That Heralds the End** — anthem-style broadcast `+1/+1` to an `Affected$` class + spell cost-reduction static (Forge script: available)
- **Lorehold Charm** — `DB$ PumpAll` (no PumpAll handler) + targeted graveyard→battlefield reanimation (Forge script: available)
- **Mox Amber** — `ManaReflected` (produce one mana of any color found among legendary permanents you control) (Forge script: available)
- **Alpha Deathclaw** — Monstrosity (`Monstrosity$` on PutCounter + `BecomeMonstrous` trigger) (Forge script: available)
- **Canoptek Scarab Swarm** — `ChangeZoneAll RememberChanged` + `Remembered$Valid` count expr to size a token (Forge script: available)
- **Urza's Saga** — Saga/Chapter/lore-counter system + `DB$ Animate` (grant abilities to self) + `ManaCost0/1` search filter (three subsystems) (Forge script: available)
- **Atraxa, Grand Unifier** — per-card-type reveal/tutor ETB (`RepeatTypesFrom`, `ImprintRevealed`, `ChosenType`) (Forge script: available)
- **Council's Judgment** — Vote / will-of-the-council subsystem (Forge script: available)
- **Damping Sphere** — `ProduceMana`/`ReplaceMana` replacement + dynamic per-spell-this-turn `RaiseCost` (`Relative$ True`) (Forge script: available)
- **Forth Eorlingas!** — Monarch subsystem (`BecomeMonarch` + monarch tracking) + `DB$ Effect` with floating `Triggers$` (Forge script: available)

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
following are **latent fragilities in the *generalized* mechanisms** (they would bite a *future*
card, not a current one); left for a human because each touches a hot/cross-cutting path with no
shipping card to verify a fix against, and the run branch was kept green:

- **`is_legal_target` controller/token filter matches substrings over the whole `ValidTgts$`
  string, not per-clause** (`src/components/ability.cpp` ~640–660). (a) A comma-joined
  `…YouCtrl,…OppCtrl` spec sets both flags and rejects every candidate; (b) a filter whose text
  contains `token` without the `!token`/`nonToken` negation is wrongly treated as token-only. No
  current card has such a spec. Fix: evaluate controller/token qualifiers per clause.
- **Activated `Cost$ PayEnergy<N>` is parsed into `Ability::energy_cost` but never paid or gated**
  on a normal `AB$` ability (`src/action_processor.cpp` activation-cost path; only
  `effect_immediate_trigger` consumes `energy_cost`). An activated ability with a PayEnergy cost
  would activate for free. Latent (Guide of Souls' PayEnergy is on an ImmediateTrigger).
- **Single permanent-target `PutCounter` ignores dynamic `CounterNum$` (`count_expr`) and the
  second counter (`CounterType2$`/`CounterNum2$`)** (`src/effects/effect_put_counter.cpp` ~51) —
  only the `Defined$ You` branch and `put_counter_all` honor them. A targeted dynamic-count counter
  would place zero. Latent (Wrath uses `Defined$ You`; Guide uses `PutCounterAll`).
- **Non-switched energy `UnlessCost` in `DestroyAll` pays energy unconditionally and destroys
  regardless of payment** (`src/effects/effect_destroy_all.cpp` ~39) — only `UnlessSwitched$ True`
  consults the paid result. A normal "destroy … unless its controller pays {E}" would be inverted.
  Latent (Wrath of the Skies is switched).
- **The generic `ValidPlayer$ You` trigger gate is not exempted for `BECAME_TARGET`**
  (`src/systems/state_manager_triggers.cpp` ~180), where the event's PLAYER param is the *targeting*
  spell's controller. `trigger_only_self` was exempted but `ValidPlayer` was not, so a future
  becomes-target trigger authored with `ValidPlayer$ You` would be mis-gated. Latent (Reality
  Smasher uses `ValidSource$`).
- **`Count$ValidGraveyard Card.YouOwn$CardTypes` honors only the ownership restriction; any extra
  type/CMC subfilter in the Card clause is ignored** (`src/svar_eval.cpp` ~153). Correct for
  Nethergoyf (plain `.YouOwn`); a future card with a real subfilter would mis-count.
- **`can_afford_with_sources()` omits the `TapsForMana` additional-mana bonus**
  (`src/mana_system.cpp` ~273), diverging from the real payment path for *nested* mana-source
  activation costs (a mana source whose own activation cost could be covered by a creature's mana +
  Badgermole's extra {G} is undercounted and not offered). Narrow and latent; spell-cast
  affordability uses the correct `can_pay_mana` path.
