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
- **Amped Raptor** — energy counters + `DigUntil` + `DB$ Play` paying energy (Forge script: available)
- **Guide of Souls** — energy counters + `PayEnergy<3>` immediate-trigger cost (Forge script: available)
- **Wrath of the Skies** — `PutCounter ENERGY` on player + `ChooseNumber` + `DestroyAll` bounded by `PayEnergy<Y>` (Forge script: available)

**Adventure / split-mode cards** (cast one half, exile, recast other half later):
- **Brazen Borrower Petty Theft** — Adventure (`AlternateMode:Adventure`) (Forge script: available)

**Modal double-faced cards** (choose which face to play from hand; pay-life-or-enters-tapped land):
- **Witch Enchanter Witch-Blessed Meadow** — MDFC + `R:Moved ReplaceWith DBTap UnlessCost PayLife<3>` (Forge script: available)

**Planeswalker / loyalty-adjacent multi-mechanic cards** (loyalty framework exists, but each stacks several other new mechanics):
- **Tamiyo, Inquisitive Student Tamiyo, Seasoned Scholar** — `DB$ Investigate` (Clue tokens) + `Mode$ Drawn Number$ N` (Nth-card-this-turn) + `SetMaxHandSize` emblem (Forge script: available)
- **Kaito, Bane of Nightmares** — Ninjutsu + planeswalker-becomes-creature animation + emblem static + tap/stun counters (Forge script: available)

**Other missing subsystems:**
- **Consign to Memory** — Replicate keyword + countering a triggered ability on the stack (Forge script: available)
- **Badgermole Cub** — Earthbend (animate land + counter + dies-return delayed trigger) + `TapsForMana` triggered mana (Forge script: available)
- **Ba Sing Se** — Earthbend + conditional ETB-tapped (`ConditionPresent` gating on a land-tapped replacement) (Forge script: available)
- **Mox Opal** — Metalcraft activation gating (control 3+ artifacts) on a mana ability (Forge script: available)
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

### Not reached (cap)

Implementable this run but left for a future run because the N=30 cap was reached first
(highest-frequency first). Finalized at end of run — see the run's final report.
