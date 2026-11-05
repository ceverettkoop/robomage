# Plan: every candidate mechanic, built on shared engine primitives

Goal: implement every mechanic that every deck in `bin/resources/decks/candidates/` needs, ordered
by what makes the engine most reusable and extensible, not by which deck unlocks first. A deck
becoming playable is an output of this plan (Appendix C), never a reason to reorder it.

`candidates/REPORT.md` remains the source of *what* is missing. It ranks by deck unlock and sizes
each mechanic alone, so it is the wrong source for *how* to build them. This plan groups the same
173 parse-report findings (76 mechanics, 71 cards) under 11 engine primitives. Most findings are
one more row in a table once the primitive exists; built one at a time, each would be one more
special case.

## 0. Executing this plan (read first)

Written to run in a session with no context beyond the repo.

1. **Starting state.** Branch `refactor/open-vocab`, Forge pin `f62f8048` (2026-11-05), M0 done and
   `make check` green. Card scripts are untracked: on a fresh clone, run
   `python3 tools/forge_fetch/provision_decks.py`.
2. **File and line references** come from a survey at `502f8b07` and will drift. Treat them as
   where to start reading, and re-verify before relying on one.
3. **No training run may be active** while engine binaries, the vocab or the state format change:
   `systemctl --user list-units 'rmtrain*' 'rmpilot*' 'rmbench*' --all`. Ask the user before
   stopping one.
4. **One subagent per stage or per mechanic**, told not to spawn agents (CLAUDE.md). One mechanic
   per commit. `make check` green locally before each commit, run in the background with the
   output path stated.
5. **Standing rules from CLAUDE.md apply**, the sharpest here: do not modify card scripts; honour
   the scripts' tags and never retag; two-player scope only (implement "each opponent" for one
   opponent); all decision state lives in `cur_game`/ECS and every prompt suspends through
   `FrameCtx`; update `docs/rules_deviations.md` for any known gap; consult the CR by rule number.

## 1. Principles

These decide every design question below.

- **Migrate, then extend.** Each primitive lands as at least two commits. The first moves *existing*
  behaviour onto the primitive and must be provably inert: `replay` byte-identical, `make registry`
  a no-op, `--parse-report @vocab` unchanged. Only later commits add mechanics. A refactor and a
  behaviour change never share a commit, so any corpus drift points at one commit.
- **A new mechanic should be a row, not a branch.** Today the trigger binder, filter matcher, SVar
  evaluator, cost parser and cast menu are if-chains, and `Defined$` is a set of booleans. The
  target for each is one table: a name, a parser, an evaluator. Extensibility is measured by how
  few files the next keyword touches.
- **One recogniser, one implementation.** What `--parse-report` calls known must be derived from
  the same table the engine executes. Today the SVar evaluator keeps a second, hand-written
  recognised list, and static params can be accepted without being read. Those gaps are how a
  card reports clean and plays wrong (§3).
- **Parse once.** Filters, `Defined$` specs and costs become structured values on the immutable
  `AbilityDef`, not strings re-read at each use.
- **Clean is not correct.** A clean parse report means nothing was left unread. It does not mean
  the card works. Every new card gets a scenario (§4).

## 2. M0: the pin regression (blocks everything)

**Done** (2026-11-05). `make check` green, the opt-in `actor` tier skipped, corpus not re-recorded: the fold
reproduces the old `Affected$` strings, so nothing else moved. New scenarios
`cr122_1d_stun_counter_replaces_untap` and `cr301_5_equipment_static_affects_only_equipped_creature`.
All 458 local scripts were compared byte for byte with the pin's blobs, which replaced the `make
refetch` run. The rest of this section is the record of what the bump broke.

The bump to `f62f8048` changed 35 local scripts. It broke the gate: **6 of 126 scenarios and 23 of
66 corpus replays fail**, plus `vocab` (15 vocab cards with a new `static_param:AffectedDefined`
finding, and the card digest). The 21 cards with new findings are in Appendix A.

**Cause.** Forge rewrote `Affected$ Creature.EquippedBy` as `AffectedDefined$ Equipped | Affected$
Creature`, and likewise `Enchanted` and `Self`. The engine does not read `AffectedDefined$`, so:

| script form | scripts | effect today |
|---|---|---|
| `Equipped` / `Enchanted` with `Affected$ Creature` | 10 | an anthem on every creature (Shadowspear, Animate Dead's -1/-0) |
| `Equipped` / `Enchanted` with no `Affected$` | 2 | buffs the source instead (Lion Sash, Springheart Nantuko) |
| `Self` with `Affected$ <filter>` | 1 | fans out to everything matching (Kaito) |
| `Self` with no `Affected$` | 8 | correct by accident |

**Work, one commit each:**

1. **`AffectedDefined$`.** Fold it into `sa.affected` after the static's param loop in `parse.cpp`,
   since every consumer reads that string: `Equipped` → `<Affected or Card>.EquippedBy`, `Enchanted`
   → `.EnchantedBy`, `Self` → `.Self` merged with any `Affected$` qualifiers. M2 replaces this fold.
2. **Counter keys.** Kaito's script now says `CounterType$ STUN`; the untap step in
   `classes/game.cpp` read the exact key `"Stun"`. Silent: no finding. Every counter type in the
   scripts is now uppercase, so the reader changes and no canonicaliser is needed.
3. **Data.** `make registry` for Voice of Victory (now a Human Bard: the digest's only drift) and
   `make text-table` for its row and Urza's Saga's.
4. **Review the other script changes** (Appendix D lists them). All looked inert on reading, because
   the engine ignores the changed keys. Confirm with the replay tier.
5. `tools/onboard/mechanic_sizes.json`: retire the `FORGE_PIN bump` group, add the new findings.

**Exit.** `make check` green with the corpus unchanged. If replays still drift, explain each one
before any `make regen`. Commit the pin, status files and report with this stage. A separate
`make refetch` run on a clean tree confirms the provisioned scripts reproduce the result.

## 3. M1: truthful reporting and the test kit

Do this before any new mechanic; every later stage leans on it.

**Silent failures: done** (2026-11-05). `--parse-report` now reports what it used to pass over. The
rule the audit left behind: a key a parser branch accepts is either honoured for that value or
a finding, per ability category, static mode and trigger mode (`param_form_read`, `static_mode_reads`,
`trigger_mode_reads`), and a cost token the engine does not pay adds nothing. What it found:

- Costs: an unmodelled token no longer becomes mana. `Sac<N>`, `Return<N>` and `Discard<N>` beyond
  the count their path pays, a variable `PayLife`/`PayEnergy`, hybrid and Phyrexian pips outside
  `ManaCost:`, a `Cost$` on a spell or trigger that nothing pays, `UnlessCost$` on an effect that
  never runs it, alternative-cost conditions.
- Statics: `Activator$` (Defense Grid), `Caster$`, unread gates, filters the form ignores.
- Triggers: `ValidCard$`, `ValidSource$`, `ValidTarget$`, `ValidSA$` values the mode ignores.
- Abilities: the ignored-param list now holds only prose, AI hints and keys another param states;
  `Imprint`, `ForgetOnMoved`, `ForgetOtherRemembered` and the rest are limited to their safe forms,
  `ForgetOtherTargets` is off it. `Defined$` names with no reader, the forms of an `Effect`.
- `cmc` qualifiers: a literal bound is compared in the matcher; a bound naming an SVar with no reader
  to supply it is a `filter` finding.
- Three vocab bugs fixed: Brotherhood's End destroyed every artifact, a declined "may search"
  shuffled, and a Powerstone's mana cast anything. One vocab finding recorded: Ajani's
  `TriggerAmount$` (`docs/rules_deviations.md` §2.8).

Not audited: `R:` lines, which still match whole shapes (`replacement` when none fits), and whether a
known `Defined$` name is read by the particular effect it sits on, which M2 settles.

**Mechanic-first onboarding.** `onboard.py vocab` appends a whole deck and only once that deck needs
nothing. CLAUDE.md also forbids testing cards outside the vocab. A mechanic-first plan cannot work
under both, so add `onboard.py vocab --cards A,B`: append named cards whose own findings are gone,
then `make registry` and `make text-table`. Each stage's exit appends the cards it finished
(Appendix B). Ids stay permanent and append-only, so nothing existing moves.

**The test kit** (§4 defines how it is used):

- A gate check that every card in the vocab added under this plan has a scenario naming it.
- `decks/tests/mech_<primitive>.dk`: dense fixture decks, so fuzz reaches a primitive in most games
  rather than waiting for a rare line.
- `--coverage FILE`: per run, hit counts per `EffectKind`, cost part, cast option, trigger mode,
  replacement kind and action category, summed by the `fuzz` tier. A fixture matchup fails if a
  listed path has zero hits, which proves fuzz reached the code rather than assuming it.

## 4. Testing model

Every stage uses the same evidence; the stage sections list only what is specific to them.

| Change | Required evidence |
|---|---|
| Migration commit | `replay` byte-identical; `make registry` no-op; `--parse-report @vocab` unchanged; `bench` tier compared with the prior commit (the hot paths move) |
| New rules behaviour | One `tests/scenarios/<cr-rule>_<what>.json` per rule, asserting the outcome and the illegal case (`expect_menu_not`), failing without the fix |
| New card | A scenario using the card as written; for a card with several abilities, each ability |
| New prompt | Suspends and resumes: the `state` tier's snapshot check over the scenario; `--play` verb and `play_spec` entry; explore (both `python/explore_agent.py` and `src/agents/explore.h`) makes the choice rather than defaulting; `--trace-decisions` and CLI render it (`cli` golden if it changes) |
| New persistent state | Added to `schema.cpp` and `encode.cpp` together; `make registry`; `docs/state_format.md`; `--selfcheck-state`; snapshot deep-copy covers it with no edit, so assert that with a scenario resumed across it |
| Hidden information | `--selfcheck-knowledge`, scramble and permutation checks; a scenario proving the opponent's frame withholds it |
| New primitive path | Fixture deck fuzz with `--coverage` non-zero, then `--fuzz-games 100` for the fixture; any draw or error is a finding |
| Deliberate behaviour change | Its own commit, `make regen`, the corpus diff read and explained |

Agent behaviour: keep `scripted_core.py` changes minimal (new decisions get a safe default,
confirmed by test, since the default for unlisted categories in the scripted agent is unverified).
Teaching the agent to pilot new decks well is out of scope.

Learned-model safety: a change to how a known card parses moves its table rows. The `vocab` tier
reports it through the card digest, and each such diff is read before `make registry` records it.
New enum values and columns append through the registry; capacity (`action_category` 128, 48 used;
`abilities_per_card` 16, no candidate near it; card ids to 899, 366 used; token band to 1022, 917
used) is ample, so no layout bump is expected. Existing runs need `rmtrain retable` once at the next
training start, not per stage.

## 5. Stages

Order is by dependency, so each primitive exists before the mechanics that would otherwise hard-code
it. Sizes use `mechanic_sizes.json`'s S/M/L/XL and are relative; they include the migration, which
the report's 240 points do not. The findings each stage closes are in Appendix A.

```
M0 pin → M1 reporting + kit → M2 object references → M3 filters + SVars ─┬→ M4 memory + flow
                                                                         ├→ M5 costs → M6 mana → M7 cast options ─┐
                                                                         ├→ M8 characteristics ───────────────────┼→ M11 face-down
                                                                         └→ M9 events → M10 activation + targets ─┴→ M12 designations → M13 sweep
```

After M3 the three lanes (M4; M5–M7; M8–M10) touch mostly separate files and may run in separate
worktrees if time matters. M7 and M8 must both land before M11.

### M2. Object references (L)

**Today.** `Defined$` becomes booleans at parse (`parse.cpp`), with some names compared raw in effect
files, and at least 10 effect files read the flags directly beside `affected_objects` /
`affected_player`. Statics use no resolver. The SVar evaluator's signature has no ability context.

**Build.** An `ObjectSpec` value parsed once onto `AbilityDef` (Self, Remembered, Imprinted, Targeted,
ParentTarget, Triggered*, Sacrificed, Equipped, Enchanted, `Valid <filter>`, `ValidHand <filter>`,
and player forms), and `resolve_objects` / `resolve_players(spec, ResolveCtx)`, where `ResolveCtx`
carries controller, source, ability instance and trigger bindings. Effects, statics, SVars and
filters all use it. Anything kept past a zone change stays an `ObjectRef`.

**Migrate.** Effect by effect, deleting each boolean once unread, then M0's fold.

**Lands here.** `Reveal$ RevealDefined`; `DamageSource$`, where damage comes from another object
(Pyrogoyf, Hawkeye's Bow, with `TriggeredCardLKICopy` reading `departed_lki_for`); the `*.Defined`
findings, names no reader acts on (`Player` on Draw and Discard, `ChosenCard`, `Imprinted`,
`Player.IsRemembered`, `Targeted.<filter>`).

**Test.** Scenarios for damage from a departed source: lifelink, deathtouch and protection follow the
LKI source, CR 120.

### M3. Filter and SVar tables (L)

**Today.** `eval_qualifier` and the SVar evaluator are if-chains; parametrised forms are prefix-parsed
by hand; `<Set>$<Property>` exists per name; sets reach the matcher by reading
`cur_game.resolution.memory` directly; `MatchCtx` holds single values only.

**Build.** A qualifier table (`name` or prefix, parser, evaluator) with filters compiled once, and
`filter_unknown_qualifiers` / `svar_expr_recognized` derived from the same tables. Three evaluator
shapes:

- *Object-local:* `with<Kw>` / `without<Kw>`, `TopLibrary`, `!wasCastFromYourHand`.
- *Set-relative,* through `ResolveCtx` and M2's `ObjectSpec`: `sharesControllerWith`,
  `SharesColorWithOther`, `ControlledBy Remembered`, `AttachedTo`, `TargetedPlayerOwn`,
  `ExiledWithSourceLKI`, `Defined.Imprinted`.
- *Population-relative,* evaluated over the candidate set: `leastPower`, `lowestRememberedCMC`,
  `TopGraveyardCreature`.

SVars become set spec + property table + aggregate, with a suffix table (add `/Twice`,
`/DivideEvenlyDown.N`), so `TargetedPlayer$LifeTotal/HalfUp`, `Imprinted$Amount` and
`TriggeredCard$CardPower` need no code of their own. Move `cmc` enforcement into its qualifier.

**Test.** Table-driven unit coverage through scenarios: for each new qualifier one match and one
miss, tie cases for `leastPower`. A `bench` comparison, since compiled filters sit on the hottest path.

### M4. Resolution memory and effect control flow (L)

**Today.** 14 effect files write memory by hand; only `RememberTargets` is generic; imprint and
forget params are ignored; per-permanent memory is `chosen_type`, `chosen_name`, `exiled_with`.
`RepeatEach` exists; `Repeat` and `Branch` do not (unknown categories resolve as no-ops), and no
`AtRandom` exists.

**Build.**

- `MemoryOps` on `AbilityDef`, applied by one step of `resolve_ability()` after any effect
  (`RememberChanged`, `RememberDiscarded`, `ImprintCards`, `ForgetObjects`, `ForgetImprinted`...),
  retiring the per-effect writes.
- Imprint held on the source permanent, as `ObjectSet`, so it outlives the resolution (Chrome Mox,
  The Stone Brain, Goblin Welder).
- `Repeat`: a persisted iteration index in `EffectRuntime`, re-deriving the same `ChildKind` on
  resume. `Branch`: the evaluated side persisted the same way, since a resume that re-derives a
  different child is fatal. A `MaxRepeat` bound guards each loop (CR 732 for anything unbounded).
- Generic `UnlessCost` on any effect through `unless_payment` (`UnlessResolveSubs`).
- Small categories on these rails: `Shuffle`, `TapOrUntap`, `AtRandom` (through `stable_rand_below`,
  never another generator), `Discard Mode$ Hand`, `UseAllOriginZones`, `NoShuffle`, `ClearNamedCard`.

**Test.** A suspension inside each loop iteration and on each `Branch` side. Grindstone against a
library that stops the repeat, and one that empties. `AtRandom` replays identically.

### M5. Typed costs and one payer (XL)

**Today.** No cost type: ad-hoc `AbilityDef` fields from `parse_activation_cost`, `AltCost` from a second
parser, and per-keyword reductions of both (Flashback mana/life/sac, Escape exile only, Kicker mana
only). Payment is spread across `run_cast_flow`, `run_activation_flow`, `activate_mana_source`, resolution
and `run_unless_loop`. Only some parts suspend: discards in the pay step cannot. Flashback, escape and
alt casts skip the X, phyrexian and delve steps.

**Build.**

- `CostPart {kind, count, filter, ...}` lists from one parser, for activations, alternative costs and
  keyword costs.
- `can_pay(parts)` and one suspendable `run_cost_parts(FrameCtx)` used by every path, with any part
  able to ask for a choice. Extend the state format's `cost_part` enum, the pending-cost table's view.
- `ChosenCost` recorded on the `Spell`, replacing the flags and `cast_with_*` fields.
- `TapToPay`: Improvise, Convoke and Waterbend as one payment modifier with an eligibility filter and
  a pip rule. Today eligibility is hard-coded to artifacts, only generic is paid, activations pass
  no improvise, and `mana_spent` counts tapped pips. Fix that count: Sunburst and Converge depend on it.
- `ActivationZone$` for any zone. A mana ability activated from hand must work without a permanent
  (Simian Spirit Guide): today the zone path sends the card to the graveyard and the mana branch
  needs a `Permanent`.

**Migrate.** Field by field; the `cost` text column is the script's string, so card digests should not
move. Improvise is greedy with no prompt in machine mode; giving Convoke a real choice changes
decisions, so that is a deliberate-change commit with `make regen`.

**Lands here.** `Discard<N/...>`, `Exile<1/CARDNAME>`, `ExileFromHand`, `ExileFromGrave`, `Exert`
(CR 701.43), Waterbend (CR 701.67), `AlternateAdditionalCost`, Bargain (CR 702.166), Convoke (CR
702.51), Crew and Vehicles (CR 301.7); hybrid pips in any cost (Fiend Artisan, Vibrance), a mana cost
on a trigger (Nihil Spellbomb), `Activator$` on a cost static (Defense Grid).

**Test.** Each part paid from each path (cast, activate, mana ability, trigger, unless); cost-payment
order CR 601.2h; a mid-payment suspension for every chooser; can't-pay hides the option.

### M6. Mana with provenance (L)

**Today.** The pool is `multiset<Colors>`, so mana carries nothing. Spells record a pip count and colours
only. Restrictions are three bools on the source; the only rider is a global,
`pending_cant_be_countered`.

**Build.** `ManaUnit {color, source, rider, restriction}` in the pool and a `paid` list on the `Spell`.
`colors_spent`, Converge, Sunburst (CR 702.44), `Count$CastTotalManaSpent` and `ManaFromCard` become
views of it. Riders and restrictions are data on the unit; the global goes. `mana_snap` rollback must
carry units.

**Lands here.** `AddsKeywords` riders, `S:Mode$ ManaRestriction` (Hogaak's new script), `ManaConvert`
(Agatha's Soul Cauldron), Sunburst.

**Test.** Rollback restores units; a rider survives payment across a suspension; the pool's frame stays
identical for unrestricted mana, and restricted mana is distinguishable in the seat's frame.

### M7. Cast options (XL)

**Today.** Separate loops for hand, faces, flashback, escape, graveyard and exile grants; `LegalAction`
carries booleans and no cast mode. `card_play_permission` covers Graveyard and Exile. `DB$ Play` is
exile-only and reads two params. Post-resolution destination is an if-chain in `stack_manager.cpp`,
repeated in `orderer.cpp`. Spells cannot override their characteristics beyond choosing a face.

**Build.**

- `enumerate_cast_options(card)` → `CastOption {face, zone, cost basis, permission, overlay,
  destination}`, the only source of cast rows, and `LegalAction` holding the chosen option. Row order
  must stay identical so the migration is inert.
- `Permission`: one registry for any zone and any source (static, effect, emblem, the card itself),
  including Library (top card).
- `spell_destination(spell)`: one hook for flashback, Adventure, Gaea's Will and `ReplaceGraveyard`.
- `SpellOverlay`: what the option changes on the spell. Bestow (an Aura while cast, CR 702.103),
  Overload (no targets, "each", CR 702.96), Fuse (both halves, CR 702.102), Adventure (CR 715), Prepare
  (CR 722).
- `DB$ Play` over any zone, with `Valid`, `WithoutManaCost`, `ReplaceGraveyard`.

**Lands here.** Adventure, Prepare, Overload, Bestow, Fuse, Hogaak's graveyard cast, Mystic Forge's cast,
Bilbo, Beseech the Mirror.

**Test.** Every existing cast mode through the corpus. New: an Adventure exiled and cast later, and
countered (goes to the graveyard); Bestow falling off into a creature (CR 702.103f); Overload with hexproof
objects; a permission lost mid-cast; `ReplaceGraveyard` when the spell is countered. Top-of-library casting
across a shuffle. The state format's how-cast enum extended.

### M8. A characteristics view across zones (L)

**Today.** Continuous effects come from battlefield permanents and emblems and mostly reach the
battlefield. Only `SetColor` reaches other zones. No `effective_types` exists; layer 4 writes
`Permanent.types`. `static_targets` ignores `AffectedZone$`, and five readers bypass it. The filter's
`card_view` reads printed colours. Blocking is hard-coded and nothing models `CantBlockBy`.

**Build.**

- One reader for an object's types, colours, P/T and abilities in any zone, used by filters and
  targeting. Layers keep owning the battlefield; other zones get the small subset rules need.
- Static scope: `static_targets` honours `AffectedZone$` / `ExcludeZone$`, all readers go through it,
  and characteristic-defining abilities work in every zone (CR 604.3), so a static can have an
  off-battlefield source.
- `CopyOverlay` (CR 707.9) shared by `CopyPermanent`, `Clone`, `ChangeZoneAll StaticEffect` and token
  copies. `AnimateAll` shares `Animate`.
- Dynamic ability grants, `GainsAbilitiesOf`, in layer 6, and SVars available to granted bodies
  (today parsed with none).
- `rules_mod::block_prohibited` absorbing the hard-coded checks; `CantPreventDamage` in `damage_prevented`.

**Lands here.** Painter's Servant, Grist off the battlefield, Agatha's grants, `CantBlockBy`, Lazotep
Quarry, Fable's copies, The Everflowing Well, Allosaurus Shepherd.

**Test.** Painter's Servant: colour in hand, library and on the stack, and protection or Blue Elemental
Blast style targeting reading it; that colour in a hidden zone does not leak through a frame. Grist as a
creature card in the graveyard for reanimation filters. Copy of a copy.

### M9. Effect objects, triggers and replacements (L)

**Today.** Triggers bind in a 14-mode if-chain. `Effect`s hold no replacements: two shapes of
`ReplacementEffects$` are substring-matched into flags, and `Emblem` holds statics only. `Event$ Draw`
yields a replacement only for win-the-game text.

**Build.**

- `EffectObject {statics, replacements, triggers, duration, remembered}`, unifying `Emblem` and resolved
  effects, with the substring cases migrated onto it.
- A replacement table whose `ReplaceWith$` runs any SVar body, and new events `Counter` and `Moved` beside
  `DRAW_CARD`.
- A trigger-mode table (name, event, binder, matcher), with Origin$ for every zone, `Revolt$` on the
  existing player flag, per-trigger `ActivationLimit$`, and `Static$ True` no-stack triggers in general.
- One tap choke point, so `Taps` needs one `SendEvent`.

**Lands here.** Gaea's Will, Theorist's Proxy's can't-be-countered, Hullbreacher (with Treasure tokens),
Hawkeye's Bow, Renegade Rallier, Sand Scout, Chrome Mox's forget trigger; Tezzeret's emblem with a
trigger, Sundering Eruption's Effect with a continuous static, an `Effect` of several forms, a
`ValidCard$` with `Self` in one alternative (Pyrogoyf) or any filter on a `SpellCast`.

**Test.** CR 614 and 616 with two applicable replacements; replacement with a suspended choice; Hullbreacher
on the first draw of the draw step; trigger order with `Static$` mixed in; an `EffectObject` across the
game boundary in bo3.

### M10. Activation rules and announcement constraints (M)

**Today.** Limits use two per-turn mechanisms (`activation_limit`, a loyalty bool) with no per-game limit.
No per-creature attacked-this-turn. `DividedAsYouChoose` and cross-target constraints do not exist;
`TargetPickGate` is the only per-candidate hook.

**Build.** An `ActivationRule` list (per turn, per object's lifetime, named condition, timing) with loyalty
migrated onto it. Target-group constraints in `TargetPickGate`. A `DIVIDE_DAMAGE` decision, at announcement
and at trigger placement (CR 601.2d, CR 603.3d). Targets across zones (`PumpZone`).

**Lands here.** Boast (CR 702.142), Exhaust (CR 702.177), Goblin Welder's two targets, Unlicensed Hearse,
Fury, Shatterskull Smashing, ChangeTargets (CR 115.7, Untimely Malfunction).

**Test.** Division with one target, and with a target illegal at resolution (CR 608.2); Exhaust across a flicker
(a new object may activate again); Boast with and without attacking; Welder with one target gone.

### M11. Face-down objects and standing looks (XL)

**Today.** `Zone::is_face_down` exists for exile, and visibility, the state format's FACE_DOWN pool and
`Loc::FACE_DOWN` in the knowledge sidecar are partly there. No characteristics query reads the flag.
There is no standing top-of-library look.

**Build.** Face-down permanents and spells as an M8 overlay (2/2, no text, CR 708) with an M7 cast option;
turn-up as a special action; Morph (CR 702.37), Cloak (CR 701.58); the reveal when a face-down object leaves
(CR 708.9). A `STANDING_LOOK` reveal window for `MayLookAt`, following the top card as it changes.

**Lands here.** Aphetto Alchemist, Cryptic Coat, Mystic Forge's look.

**Test.** This is the plan's main information-leak risk. Scramble and permutation checks over fixture games;
the knowledge sidecar's exactness on a look that outlives a shuffle; a face-down object copied, bounced, and
blinked; identity withheld in the opponent's frames through every transition.

### M12. Designations and subsystems (L, with Dungeons XL)

**Today.** Each per-player or per-object field is hand-added across component, schema and encoder.

**Build.** A designation store, per player and per permanent, with the schema columns declared once, so that
state plumbing is one row. Then Speed (CR 702.178–179), Dungeons and venture (CR 309, 701.49: dungeon scripts
provisioned, command-zone objects, a room-choice decision), Meld (CR 701.42, 712) and Empower.

**Notes.** Perilous Snare's `AddAbility$` on itself needs `ability_grant_targets` to handle the
single-target forms: it returns nothing for them today (the old `Affected$ Card.Self` script had the
same gap, and no vocab card uses the form, so M0 left it). Dungeons exist for one card (Acererak) and are large; they are included because the goal is every
mechanic, and scheduled last so nothing waits on them. "Empower Jace" is not in the checked-in CR (effective
2026-04-17): implement from the oracle text and record it in `docs/rules_deviations.md`, and add its token to
`provision_decks.py`'s `keyword_tokens`.

**Test.** Speed raised once per turn and on opponent life loss only; venture with an already-complete dungeon;
Meld with one half missing; each new field across a snapshot.

### M13. The sweep (L)

- Verify the 45 candidate cards that report clean, as scenarios, including the ones M1 reclassified. Defense
  Grid shows why this cannot be skipped.
- Finish the vocab, then run `onboard.py` stages 8–11 for every deck: checks, the fuzz campaign, designation,
  baseline. Run `death_and_taxes` and `uwx_control` here too; they need no engine work and can go earlier.
- Designation needs the user: which decks go to `league/` and which to `heldout/`. At least one fair and one combo
  deck stay held out.
- Fuzz all playable decks, `--matchups all --fuzz-games 100`, bo1 and `explore:patient`.
- Update CLAUDE.md's architecture notes for what changed, and `docs/rules_deviations.md`.

## 6. Risks

- **Migration size.** M2, M5 and M7 rewrite core paths with no user-visible gain until later commits. The
  inert-migration rule is the safety net; if a migration cannot be made byte-identical, stop and find out why
  rather than re-recording.
- **Hot paths.** Compiled filters and mana units change throughput. Compare the `bench` tier per stage; training
  throughput depends on it.
- **Decision-schedule changes.** Convoke choices and any new prompts change what is asked, which invalidates old
  decision logs and trajectories. Group these where possible and call them out in the commit.
- **Upstream drift.** Forge rewrote 3,109 scripts in under four months. Keep the pin fixed during this plan; a
  mid-plan bump means repeating M0.
- **Unknown unknowns.** M1 will find more silent failures, and clean cards may need work this plan does not list.
  Add them to the stage that owns the primitive.

## 7. Decisions needed

None block M0 to M12. Before M13: the `league/` and `heldout/` split. Before M12's Dungeons: whether Acererak alone
justifies the subsystem, or whether Aluren should swap the card out. This plan assumes it is built.
## Appendix A. Every finding, by stage

Generated from the status files at pin `f62f8048`: 173 findings, none unmapped. `onboard.py`'s mechanics are the report's grouping; this is the plan's.

| stage | findings |
|---|---|
| M0 | `static_param:AffectedDefined` |
| M2 | `ability_param:Animate.Defined`, `ability_param:ChooseCard.Defined`, `ability_param:CopyPermanent.Defined`, `ability_param:DealDamage.DamageSource`, `ability_param:Destroy.Defined`, `ability_param:Discard.Defined`, `ability_param:Draw.Defined`, `ability_param:PutCounter.Defined`, `ability_param:Reveal.Defined`, `ability_param:Reveal.RevealDefined` |
| M3 | `amount_svar:All`, `filter:!wasCastFromYourHand`, `filter:AttachedTo Creature`, `filter:ControlledBy Remembered`, `filter:Defined`, `filter:ExiledWithSourceLKI`, `filter:Imprinted`, `filter:SharesColorWithOther`, `filter:TargetedPlayerOwn`, `filter:TopGraveyardCreature`, `filter:TopLibrary`, `filter:cmcEQX`, `filter:leastPower`, `filter:lowestRememberedCMC`, `filter:sharesControllerWith`, `filter:withoutFlying`, `filter:withoutIndestructible`, `svar_expr:Count$ValidExile`, `svar_expr:Count$ValidGraveyard`, `svar_expr:Count$xPaid/Twice`, `svar_expr:Imprinted$Amount`, `svar_expr:Imprinted$Valid`, `svar_expr:PlayerCountOpponents$HighestValid`, `svar_expr:Sacrificed$CardManaCost/Plus.2`, `svar_expr:TargetedPlayer$CardsInLibrary/HalfUp`, `svar_expr:TargetedPlayer$CardsInLibrary`, `svar_expr:TargetedPlayer$LifeTotal/HalfUp`, `svar_expr:TriggeredCard$CardPower` |
| M4 | `ability_param:Branch.BranchConditionSVarCompare`, `ability_param:Branch.BranchConditionSVar`, `ability_param:Branch.FalseSubAbility`, `ability_param:Branch.TrueSubAbility`, `ability_param:ChangeZone.Imprint`, `ability_param:ChangeZone.NoShuffle`, `ability_param:ChangeZoneAll.UseAllOriginZones`, `ability_param:ChooseCard.AtRandom`, `ability_param:Cleanup.ClearNamedCard`, `ability_param:CopyPermanent.UnlessCost`, `ability_param:CopyPermanent.UnlessCost`, `ability_param:CopyPermanent.UnlessResolveSubs`, `ability_param:DealDamage.UnlessCost`, `ability_param:DealDamage.UnlessCost`, `ability_param:Discard.Mode`, `ability_param:Discard.RememberDiscarded`, `ability_param:ImmediateTrigger.ConditionDefined`, `ability_param:ImmediateTrigger.TriggerAmount`, `ability_param:ImmediateTrigger.UnlessCost`, `ability_param:ImmediateTrigger.UnlessCost`, `ability_param:Pump.ForgetImprinted`, `ability_param:Pump.ForgetObjects`, `ability_param:Pump.ImprintCards`, `ability_param:Repeat.MaxRepeat`, `ability_param:Repeat.RepeatCheckSVar`, `ability_param:Repeat.RepeatSVarCompare`, `ability_param:Token.UnlessCost`, `ability_param:Token.UnlessCost`, `category:Branch`, `category:Repeat`, `category:Shuffle`, `category:TapOrUntap` |
| M5 | `ability_param:Discard.ActivationZone`, `ability_param:Play.Condition`, `cost:Discard`, `cost:Exert`, `cost:ExileFromGrave`, `cost:ExileFromHand`, `cost:Exile`, `cost:Hybrid`, `cost:Mana`, `cost:Waterbend`, `keyword:AlternateAdditionalCost`, `keyword:Bargain`, `keyword:Convoke`, `keyword:Crew`, `static_param:Activator` |
| M6 | `ability_param:AddMana.AddsKeywordsUntil`, `ability_param:AddMana.AddsKeywordsValid`, `ability_param:AddMana.AddsKeywords`, `ability_param:AddMana.Produced`, `keyword:Sunburst`, `static_mode:ManaRestriction`, `static_param:InvertValidSource`, `static_param:ValidCard`, `static_param:ValidPlayer`, `static_param:ValidSA`, `static_param:ValidSource`, `svar_expr:Count$CastTotalManaSpent`, `trigger_param:ValidSA` |
| M7 | `ability_param:Play.ImprintPlayed`, `ability_param:Play.ReplaceGraveyardValid`, `ability_param:Play.ReplaceGraveyard`, `ability_param:Play.ValidZone`, `ability_param:Play.Valid`, `ability_param:Play.WithoutManaCost`, `keyword:Bestow`, `keyword:ETBReplacement:DBPrepare`, `keyword:Fuse`, `keyword:Overload`, `layout:Adventure`, `layout:Prepare`, `static_form:MayPlay.Graveyard`, `static_form:MayPlay.Library`, `static_param:CheckSecondSVar`, `static_param:EffectZone` |
| M8 | `ability_param:Animate.AtEOT`, `ability_param:AnimateAll.Power`, `ability_param:AnimateAll.Toughness`, `ability_param:AnimateAll.Types`, `ability_param:ChangeZoneAll.GainControl`, `ability_param:ChangeZoneAll.StaticEffect`, `ability_param:ChangeZoneAll.WithCountersType`, `ability_param:Clone.CloneTarget`, `ability_param:Clone.Duration`, `ability_param:CopyPermanent.AddKeywords`, `ability_param:CopyPermanent.AtEOT`, `ability_param:CopyPermanent.SetColor`, `ability_param:CopyPermanent.SetCreatureTypes`, `ability_param:CopyPermanent.SetPower`, `ability_param:CopyPermanent.SetToughness`, `keyword:ETBReplacement:ChooseColor`, `keyword:ETBReplacement:DBCopy`, `keyword:HIDDEN CARDNAME can't block.`, `static_form:Affected.Counters`, `static_mode:CantBlockBy`, `static_mode:CantPreventDamage`, `static_param:AddColor`, `static_param:ExcludeZone`, `static_param:GainsAbilitiesOfZones`, `static_param:GainsAbilitiesOf`, `static_param:IsCombat`, `static_param:ValidAttacker`, `static_param:ValidBlocker` |
| M9 | `ability_param:DelayedTrigger.Static`, `ability_param:DelayedTrigger.ValidActivatingPlayer`, `ability_param:Effect.Duration`, `ability_param:Effect.ForgetOnMoved`, `ability_param:Effect.RememberObjects`, `replacement:Draw`, `replacement:Effect:Counter`, `replacement:Effect:Moved`, `static_form:Effect.Continuous`, `trigger_form:ChangesZone.ValidCard`, `trigger_form:Effect.Phase`, `trigger_form:SpellCast.ValidCard`, `trigger_mode:Taps`, `trigger_param:ActivationLimit`, `trigger_param:Revolt`, `trigger_zone:ChangesZone:Exile` |
| M10 | `ability_param:Animate.Exhaust`, `ability_param:ChangeZone.TargetsWithSameController`, `ability_param:ChangeZone.TgtZone`, `ability_param:DealDamage.Boast`, `ability_param:DealDamage.DividedAsYouChoose`, `ability_param:Pump.PumpZone`, `ability_param:Pump.TargetsWithDefinedController`, `category:ChangeTargets` |
| M11 | `ability_param:Cloak.Defined`, `ability_param:Cloak.RememberCloaked`, `category:Cloak`, `keyword:Morph`, `static_param:MayLookAt` |
| M12 | `ability_param:Empower.Type`, `category:Empower`, `category:Venture`, `keyword:Start your engines`, `layout:Meld`, `static_param:Condition`, `svar_expr:DungeonsCompleted$Valid` |

The 21 cards M0 repairs: the 15 vocab cards (Animate Dead, Cori-Steel Cutter, Dragon's Rage Channeler, Elvish Reclaimer, Kaito Bane of Nightmares, Keen-Eyed Curator, Knight of the Reliquary, Lavaspur Boots, Lion Sash, Meteor Sword, Pre-War Formalwear, Shadowspear, Sheltered by Ghosts, Skateboard, Wight of the Reliquary) and 6 candidates (Cryptic Coat, Dark Matter Manipulator, Fiend Artisan, Hawkeye's Bow, Perilous Snare, Springheart Nantuko).

## Appendix B. When each card can enter the vocab

A card joins the vocab at the exit of the last stage it needs (M1's `onboard.py vocab --cards`). The 37 cards reporting clean need no stage but still need a scenario (M13).

| last stage | cards (all stages needed) |
|---|---|
| M3 | Dark Matter Manipulator; Drop of Honey (M2 M3); Peer into the Abyss; Powder Keg; Rough/Tumble; Strategic Betrayal |
| M4 | Chaos Defiler (M2 M3 M4); Echo of Eons (M2 M4); Galvanic Discharge; Gamble; Grindstone (M3 M4); Sewer-veillance Cam; Song of Creation; Stronghold Gambit (M2 M3 M4) |
| M5 | Bitter Triumph; Defense Grid; Fiend Artisan; Formidable Speaker; Melded Moxite; Nihil Spellbomb; Runehorn Hellkite (M2 M4 M5); Simian Spirit Guide; The Echoverse Fulcrum; The Stone Brain (M3 M4 M5); Vibrance |
| M6 | Arena of Glory (M5 M6); Engineered Explosives |
| M7 | Beseech the Mirror (M3 M5 M7); Bilbo, Thief in the Night (M3 M7); Carnivorous Cultivator; Cyclonic Rift; Hogaak, Arisen Necropolis (M5 M6 M7); Massacre; Springheart Nantuko (M2 M3 M4 M7); Wear/Tear |
| M8 | Agatha's Soul Cauldron (M4 M6 M8); Allosaurus Shepherd; Brazen Borrower (M3 M7 M8); Fable of the Mirror-Breaker (M4 M8); Frenzied Baloth; Ghost Vacuum; Grist, the Hunger Tide (M4 M8); Lazotep Quarry (M3 M8); Mockingbird (M6 M8); Painter's Servant; Shallow Grave (M3 M8) |
| M9 | Chrome Mox (M3 M4 M9); Gaea's Will (M7 M9); Hawkeye's Bow (M2 M9); Hullbreacher; Pyrogoyf (M2 M3 M9); Renegade Rallier; Sand Scout (M3 M9); Sundering Eruption (M3 M9); Tezzeret, Cruel Captain (M2 M9); The Everflowing Well (M6 M8 M9) |
| M10 | Broadside Bombardiers (M3 M10); Fury; Goblin Welder (M3 M4 M10); Invasion Submersible (M5 M10); Shatterskull Smashing (M3 M10); Sink into Stupor; Unlicensed Hearse (M3 M5 M10); Untimely Malfunction (M8 M10) |
| M11 | Aphetto Alchemist; Cryptic Coat (M8 M11); Mystic Forge (M3 M7 M11) |
| M12 | Acererak the Archlich (M4 M12); Perilous Snare; Phyrexian Dragon Engine (M5 M12); Theorist's Proxy (M9 M12) |

## Appendix C. Deck readiness (an output, not a goal)

The stage after which each deck has every mechanic. Decks with no entry in the table's left column need no engine work.

| ready after | decks (meta share %) |
|---|---|
| M3 | lands_top8 (5.0) |
| M8 | cradle_control (1.0), dimir_tempo (10.0), elves (2.0), hogaak (2.0), reanimator_top8 (3.0), ur_tempo (4.0) |
| M9 | boros_aggro (6.0), maverick (1.0), storm (3.0) |
| M10 | artifacts_blue (2.0), cloudpost_tron_ramp (5.0), dragon_stompy (1.0) |
| M11 | stoneblade (1.0), welder_cam (2.0) |
| M12 | aluren (4.0), artifacts_prison (2.0), doomsday (4.0), mystic_forge (2.0), painter (1.0) |

## Appendix D. Script changes at the bump with no parse finding

Reviewed by reading the diff; M0 confirms them through the replay tier.

| change | cards | status |
|---|---|---|
| `CounterType$ Stun` → `STUN` | Kaito | **a regression**: the engine reads `"Stun"` exactly (M0) |
| `Types` Human Soldier → Human Bard | Voice of Victory | digest drift, `make registry` (M0) |
| Oracle wording | Urza's Saga, Fable of the Mirror-Breaker, Arena of Glory | text rows for vocab cards (`make text-table`) |
| `Reveal$` removed | Atraxa Grand Unifier, Endurance | key ignored by the parser |
| `NoReveal$` removed | Sylvan Library | key not read |
| `Reorder$ True` added | Jace the Mind Sculptor | the picks are the order (M1 checked the put-back loop) |
| AI-only keys | Cabal Therapy, Natural Order, Fiend Artisan | no rules effect |
| `Activator$ Player.NonActive` → `Player.!Active` | Defense Grid | not read either way: `static_param:Activator` (M5) |
| `SVar:SacMe:1` added | Clue token | AI hint |
