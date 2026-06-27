# Deferred cards — for user-directed implementation

Cards the autonomous runs triaged but did NOT implement. Each needs a human decision or a larger
change than an unattended run should make. Pick one up by running the interactive
`implement-missing-cards` skill (or directing a fresh autonomous run at it once unblocked).

## Run 2026-06-26  (cap N=30, inspected 60 cards)

### Deferred at triage

Each card below has a local Forge script available; the blocker is a missing engine subsystem
(grouped by the mechanic that needs to be built first). Implementing one card in a group usually
unblocks the others in that group.

**Energy** (energy counters, `PayEnergy` cost, energy-gated effects):
- **Guide of Souls** — energy counters + `PayEnergy<3>` immediate-trigger cost (Forge script: available)
- ~~**Wrath of the Skies** — `PutCounter ENERGY` on player + `ChooseNumber` + `DestroyAll` bounded by `PayEnergy<Y>` (Forge script: available)~~ — implemented this run (see `wrath_of_the_skies.md`)

**Adventure / split-mode cards** (cast one half, exile, recast other half later):
- **Brazen Borrower Petty Theft** — Adventure (`AlternateMode:Adventure`) (Forge script: available)

**Modal double-faced cards** (choose which face to play from hand; pay-life-or-enters-tapped land):
- **Witch Enchanter Witch-Blessed Meadow** — MDFC + `R:Moved ReplaceWith DBTap UnlessCost PayLife<3>` (Forge script: available)

**Planeswalker / loyalty-adjacent multi-mechanic cards** (loyalty framework exists, but each stacks several other new mechanics):
- **Tamiyo, Inquisitive Student Tamiyo, Seasoned Scholar** — `DB$ Investigate` (Clue tokens) + `Mode$ Drawn Number$ N` (Nth-card-this-turn) + `SetMaxHandSize` emblem (Forge script: available)
- **Kaito, Bane of Nightmares** — Ninjutsu + planeswalker-becomes-creature animation + emblem static + tap/stun counters (Forge script: available)

**Other missing subsystems:**
- **Consign to Memory** — Replicate keyword + countering a triggered ability on the stack (Forge script: available)
- **Nethergoyf** — Escape keyword (cast from graveyard, exile cards as cost, with X) + CDA P/T counting card types in graveyard (Forge script: available)
- **Petrified Hamlet** — `DB$ NameCard` effect + Continuous `AddAbility$` granting a mana ability to other named lands (Forge script: available)
- **Pinnacle Emissary** — Warp alternative cost (cast for warp cost, exile at end step, recast from exile) (Forge script: available)
- **Reality Smasher** — general `T: Mode$ BecomesTarget` triggered ability + `UnlessCost$ Discard<1/Card>` / `UnlessPayer$` (Forge script: available)
- **Thought-Knot Seer** — `RevealHand` effect + opponent-hand exile with `Chooser$ You` / `DefinedPlayer$ Targeted` (Forge script: available)
- **Wastescape Battlemage** — Kicker (multi-kicker cost choice at cast + `kicked N` SpellCast condition) (Forge script: available)
- **Skyclave Apparition** — exile-on-ETB + LTB token sized by exiled card's MV (`TokenOwner$ RememberedOwner`, `Remembered$CardManaCost`) (Forge script: available)
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
