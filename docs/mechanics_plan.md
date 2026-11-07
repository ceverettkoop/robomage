# Plan: every candidate mechanic, built on shared engine primitives

Goal: implement every mechanic that every deck in `bin/resources/decks/candidates/` needs, ordered
by what makes the engine most reusable and extensible, not by which deck unlocks first. A deck
becoming playable is an output of this plan (Appendix C), never a reason to reorder it.

`candidates/REPORT.md` remains the source of *what* is missing. It ranks by deck unlock and sizes
each mechanic alone, so it is the wrong source for *how* to build them. This plan groups the same
174 parse-report findings (77 mechanics, 71 cards) under 11 engine primitives. Most findings are
one more row in a table once the primitive exists; built one at a time, each would be one more
special case.

## 0. Executing this plan (read first)

Written to run in a session with no context beyond the repo.

1. **Starting state.** Branch `refactor/open-vocab`, Forge pin `f62f8048` (2026-11-05), M0 to M7 done
   and `make check` green. Card scripts are untracked: on a fresh clone, run
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

**Done** (2026-11-05). `make check` green, the opt-in `actor` tier skipped, corpus not re-recorded.

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

The per-param form checks M1 added to `src/parse.cpp` (`param_form_read` and similar) are a second
recogniser, against §1's "one recogniser, one implementation". They are temporary: retire each as M2
and M3 make the engine's own tables the source of what is known. M2 retired the `Defined$` one: its
names are read from the table (`defined_name_read`).

**Mechanic-first onboarding.** `onboard.py vocab` appends a whole deck and only once that deck needs
nothing. CLAUDE.md also forbids testing cards outside the vocab. A mechanic-first plan cannot work
under both, so add `onboard.py vocab --card A --card B`: append named cards whose own findings are gone,
then `make registry` and `make text-table`. Each stage's exit appends the cards it finished
(Appendix B). Ids stay permanent and append-only, so nothing existing moves. **Done** (2026-11-05):
`onboard.py vocab --card NAME` (repeatable, as names hold commas), and the first test-kit bullet below.

**The test kit: done** (2026-11-05; §4 defines how it is used):

- A gate check that every card in the vocab added under this plan has a scenario naming it.
- `decks/tests/mech_<primitive>.dk`: dense fixture decks, so fuzz reaches a primitive in most games
  rather than waiting for a rare line. Each has a sidecar `.expect`; the first is `mech_cost_paths`.
- `--coverage FILE`: per run, hit counts per `EffectKind`, cost part, cast option, trigger mode,
  replacement kind and action category (`docs/game_running.md`), summed by the `fuzz` tier. A fixture
  fails if a listed key has zero hits, which proves fuzz reached the code rather than assuming it
  (`docs/ci.md`). The `cost` and `cast` names are today's fields, simply named at the pay step; M5 and
  M7 replace them with their typed parts and options, and rename `mech_cost_paths.expect`'s keys.

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

**Adding a fixture.** A stage with a new primitive path adds `decks/tests/mech_<primitive>.dk` (vocab
cards only, so its cards enter the vocab first, Appendix B; dense enough to reach the path in most
games, and legal for the engine) and beside it `mech_<primitive>.expect`, the coverage keys that prove
the path ran, one per line. To write the list, run
`bin/debug/robomage_check --tier fuzz --matchups tests/mech_<primitive>` and read the summed counts in
`ci_out/coverage/mech_<primitive>.txt`. A path with no key of its own gets one: a hit in `src/coverage.h`'s
kinds at the stage's choke point. Nothing else is wired up: `make check` picks the fixture up, and
the stage's evidence is that same command with `--fuzz-games 100`.

Agent behaviour: keep `scripted_core.py` changes minimal (new decisions get a safe default,
confirmed by test, since the default for unlisted categories in the scripted agent is unverified).
Teaching the agent to pilot new decks well is out of scope.

Learned-model safety: a change to how a known card parses moves its table rows. The `vocab` tier
reports it through the card digest, and each such diff is read before `make registry` records it.
New enum values and columns append through the registry; capacity (`action_category` 128, 48 used;
`abilities_per_card` 16, no candidate near it; card ids to 899, 372 used; token band to 1022, 917
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

**Progress.** The primitive and the inert migration are done (2026-11-05): `ObjectSpec`
(`src/components/object_spec.h`), the `Defined$` table, `ResolveCtx` and `resolve_objects` /
`resolve_players` (`src/queries/affected.{h,cpp}`). Every `defined_*` boolean and raw `Defined$`
compare is gone, M0's fold is replaced by `StaticAbility::affected_defined`, and the SVar evaluator has
its `ResolveCtx` overloads (signature only; the five `evaluate_amount` callers that pass exactly
those values use them, the rest move with M3). `replay` stayed byte-identical at each commit.
Effects that behave differently per form still compare `def->defined`; a new name only they read
needs a branch as well as its row.

**Today.** `Defined$` becomes booleans at parse (`parse.cpp`), with some names compared raw in effect
files, and at least 10 effect files read the flags directly beside `affected_objects` /
`affected_player`. Statics use no resolver. The SVar evaluator's signature has no ability context.

**Build.** An `ObjectSpec` value parsed once onto `AbilityDef` (Self, Remembered, Imprinted, Targeted,
ParentTarget, Triggered*, Sacrificed, Equipped, Enchanted, `Valid <filter>`, `ValidHand <filter>`,
and player forms), and `resolve_objects` / `resolve_players(spec, ResolveCtx)`, where `ResolveCtx`
carries controller, source, ability instance and trigger bindings. Effects, statics, SVars and
filters all use it. Anything kept past a zone change stays an `ObjectRef`.

**Migrate.** Effect by effect, deleting each boolean once unread, then M0's fold.

**Lands with its first card, not here.** No card is finished by M2 alone (Appendix B), and a card
outside the vocab can't be tested, so nothing below is built until a stage can prove it. Each is a
row in the `Defined$` table plus, where the effect needs it, a branch:

| item | first testable card | stage |
|---|---|---|
| `Destroy Defined$ ChosenCard` | Drop of Honey | M3 |
| `DamageSource$ TriggeredCard`, with `TriggeredCard` bound to the triggering object | Pyrogoyf | M9 |
| `Draw Defined$ Player` (`Discard`: Runehorn Hellkite, M5) | Echo of Eons | M4 |
| `Defined$ Player.IsRemembered`, `Reveal$ RevealDefined` | Stronghold Gambit | M4 |
| `Defined$ Targeted.<filter>` | Tezzeret, Cruel Captain | M9 |
| `CopyPermanent Defined$ Enchanted` | Springheart Nantuko | M7 |
| `DamageSource$ TriggeredCardLKICopy` | Hawkeye's Bow | M9 |

`TriggeredCard`, `TriggeredCardLKICopy` and `TriggeredNewCard` resolve to the trigger's source today.
That is right for the two vocab scripts using them, which trigger on themselves; Pyrogoyf is the first
that needs the real binding, so its unit fixes it. Damage from a departed source (CR 120: lifelink,
deathtouch and protection follow the LKI source) is tested with Hawkeye's Bow.

**Done** (2026-11-05). `make registry` a no-op, parse reports unchanged over all 430 scripts, 1,560
explore games byte-identical, `bench` flat (139.7 to 140.6 us). Appendix A and B still count these
findings under M2; the table above says when each lands.

### M3. Filter and SVar tables (L)

**Progress.** The two tables and the inert migration are done (2026-11-05). Every qualifier is a row of
the table in `src/queries/filters.cpp`, and a spec compiles once to a `CompiledFilter`, interned by text
behind the `FilterSpec` a definition holds (`src/components/filter_spec.h`); no matcher takes a string.
`IsRemembered`, `IsImprinted`, `nonChosenCard`, `targetedBy` and `ControlledBy ParentTarget` are
set-relative through an `ObjectSpec` and `MatchCtx::resolve`. `cmc<OP><SVar>` enforces its own bound.
Every expression is a row of the tables in `src/svar_eval.cpp`: its own value, or a set, a property and
an aggregate, with `<Set>$<Property>` a row of neither, plus the suffix table. `filter_unknown_qualifiers`
and `svar_expr_recognized` read those tables; the tokens only a filter's reader reads are rows too, and
`EquippedBy`, which nothing reads, is now a finding (Cryptic Coat, Hawkeye's Bow).

No qualifier is population-relative, so that shape is defined and unused: `select_matching` and
`population_row` have no caller (see Done below for why `leastPower` is not one). Still hand-kept, because none is a question either table
answers: `is_runtime_amount_expr` and `svar_expr_read_at_parse` (the forms the parser resolves),
`cmc_bound_supplied` (which reader supplies a bound), `trigger_valid_card_read` (M9's binder), and
TargetType$'s own reader (`stack_spell_alt_matches`). The legacy `evaluate_svar` callers that pass source
0 stay as they are: moving them to `ResolveCtx` gives them a source, which is a behaviour change.

`replay` stayed byte-identical at each commit, `make registry` a no-op, parse reports unchanged over
all 430 scripts but for that finding; 6,144 explore games byte-identical; `bench` faster (137.4 to
128.6 us, median of 11 alternating release runs).

**Lands with its first card, not here.** A card outside the vocab can't be tested, so as in M2 these
wait for the first stage that can prove them. The stage is when the card's own last need is met:

| finding | first testable card | stage |
|---|---|---|
| `filter:ControlledBy Remembered` | Chaos Defiler | M4 |
| `filter:SharesColorWithOther` | Grindstone | M4 |
| `filter:lowestRememberedCMC` | Stronghold Gambit | M4 |
| `svar_expr:Imprinted$Amount` | The Stone Brain | M5 |
| `amount_svar:All` | Beseech the Mirror | M7 |
| `filter:!wasCastFromYourHand` | Bilbo, Thief in the Night | M7 |
| `filter:AttachedTo Creature` | Springheart Nantuko | M7 |
| `filter:TopGraveyardCreature` | Shallow Grave | M8 |
| `filter:cmcEQX` in `ValidTgts$` (the X paid, against a card in a graveyard) | Lazotep Quarry | M8 |
| `filter:Defined`, `filter:ExiledWithSourceLKI`, `filter:Imprinted` | Chrome Mox | M9 |
| `filter:EquippedBy` | Hawkeye's Bow (then Cryptic Coat, M11) | M9 |
| `svar_expr:PlayerCountOpponents$HighestValid` | Sand Scout | M9 |
| `svar_expr:TriggeredCard$CardPower` | Pyrogoyf | M9 |
| `filter:sharesControllerWith Imprinted`, `svar_expr:Imprinted$Valid` | Goblin Welder | M10 |
| `svar_expr:Count$ValidExile` | Unlicensed Hearse | M10 |
| `svar_expr:Count$xPaid/Twice` | Shatterskull Smashing | M10 |
| `svar_expr:Sacrificed$CardManaCost/Plus.2` | Broadside Bombardiers | M10 |
| `filter:TopLibrary` | Mystic Forge | M11 |

`Remembered$CardManaCost` still reads the first object and `Count$Valid …$CardManaCost` sums printed
values: no M3 card needs either to do more. `lowestRememberedCMC` and `TopGraveyardCreature` are the
remaining candidates for the population shape, and each should be checked against its card as
`leastPower` was before it is built as one.

**Done** (2026-11-05). In the vocab, ids 367 to 372: Rough/Tumble, Peer into the Abyss, Dark Matter
Manipulator, Powder Keg, Drop of Honey, Strategic Betrayal, each with per-rule scenarios that fail with
its row or fix removed. New rows: `without<Keyword>`, `leastPower`, `ChosenCard`, `TargetedPlayerCtrl`
and `TargetedPlayerOwn` (`ChosenCard` and `TargetedPlayerCtrl` were read-elsewhere tokens that no
ChooseCard or ChangeZoneAll read), the `TargetedPlayer` set, the `CardsInLibrary` and `LifeTotal` properties, the
`/DivideEvenlyDown.N` suffix, and `Defined$ ChosenCard` on Destroy. A `cmc<OP>X` other than `cmcLEX` on
DestroyAll is a named bound. A static's SVar opening with `SVar$<name>` expands at parse.

`leastPower` is a plain row, not population-relative. "The creature with the least power" compares with
every creature on the battlefield, so select_matching's narrowing would be wrong: with an indestructible
creature least, `Creature.leastPower+withoutIndestructible` must match nothing, not promote the next
creature up. Both population-shape helpers therefore still have no caller.

ChooseCard's player is now the one it names or targets (`affected_player`), its battlefield candidates
include tokens, and with no candidate it leaves nothing chosen. A qualifier with no reading fails
closed when negated too, which changed no vocab card. `replay` stayed byte-identical at each commit:
no vocab card changed behaviour and the corpus was not re-recorded. `decks/tests/mech_filters.dk` is the
fixture, with a new coverage kind, `row`.

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

**Done** (2026-11-05). In the vocab, ids 373 to 380: Sewer-veillance Cam, Song of Creation, Gamble, Echo of
Eons, Galvanic Discharge, Grindstone, Chaos Defiler, Stronghold Gambit, each with per-rule scenarios that
fail with their row or fix removed, but for Song of Creation's second land and its draw, which ran on existing
code. `decks/tests/mech_memory.dk` is the fixture: 100 games each of `explore` and `explore:patient` with no
error or draw, every key hit, and 48 games clean under `--selfcheck-knowledge`.

The memory table. A definition holds `memory_ops`, rows of the one table in `src/resolution_memory.cpp`
(types in `src/components/memory_ops.h`): 15 by param and 5 by effect kind. A handler only reports what it
did (`report_effect`, `report_object`), and `apply_memory_ops`, one step of `resolve_ability()`, writes it
once the handler returns anything but `SUSPENDED`. The reports live on the resolving `Ability`, so a suspended
effect's persist in its `FrameLevel`. `REPLACE` reads whether the effect reported at all; `FORGET` runs a row's
own `ObjectSpec` as any effect finishes (`ForgetObjects$`, on a Pump with nothing to pump). The inert unit
moved 38 write lines in 13 effect files off the handlers with `replay` byte-identical at each commit,
6,144 explore games byte-identical, `bench` flat.

Not in the after-effect step, and why:

- `RememberTargets` and `RememberObjects$ …Targeted…` also run as the ability starts to resolve: a
  DelayedTrigger's watched object and a prevention Effect read the target during their own effect.
- A multi-zone search reads its own earlier picks through `ZoneSearch::picking`.
- `chosen_number`, `named_card` and `chosen_type` stay hand-written: each is an effect's own answer, a value
  with no param, not objects a row could move.
- `restore_remembered_exiled_with` and the frame's save and restore scope the memory. RepeatEach no longer
  swaps it: an iteration adds its player and forgets them after, so what the body remembers reaches the later
  iterations and the trailing links (Chaos Defiler's chosen permanent, Stronghold Gambit's chosen cards).

New categories `TapOrUntap` (a `CHOOSE_MODE` menu of two rows; `mode:<text>` now names a mode by its text),
`Shuffle` and `Repeat` (`RepeatLoopRt` persists the iteration, the body's index and the bound read once, each
child a `REPEAT_SUB` level). Trailing links of a repeating effect chain after it: `next_sub` starts past
`repeat_sub_count`. New rows: the filters `SharesColorWithOther Remembered`, `ControlledBy Remembered`,
`RememberedPlayerCtrl`, `lowestRememberedCMC`; the `Defined$` names `Player.IsRemembered` and `ValidHand <filter>`;
`Draw Defined$ Player`, `Discard Mode$ Hand`, `NoShuffle$`, `ChooseCard AtRandom$`, `Reveal RevealDefined$`,
`ForgetObjects$`. `RepeatPlayers$` is an `ObjectSpec`. `UseAllOriginZones$` is an ignored key: a move here
always takes every zone of `Origin$`.

`lowestRememberedCMC` is a set-relative row, not population-relative: "revealed this way" compares with the
remembered cards, never with the candidates, so an unrevealed creature of lower or equal mana value is no match
and a filter's other qualifiers can't change the minimum. Tied cards all match; with none remembered nothing
does. `select_matching` and `population_row` still have no caller.

Galvanic Discharge reported one finding (`DealDamage.UnlessCost`), and that was the least of what it needed.
Its target sits two links down its script, and nothing looked past one: `Ability::chained_subs` now walks the
chain for announcing, the cast gate and the frame's target rows, leaving out what a later trigger runs. A stack
object with no target of its own and all its chained targets illegal no longer resolves (CR 608.2b), which
is also Cabal Therapy under Veil of Summer; the corpus doesn't reach it. `NumDmg$ X` with `Count$ChosenNumber`
was silently 0: `is_runtime_amount_expr` drops any expression it doesn't list, with no finding. The same drop hits
Broadside Bombardiers, Pyrogoyf, Shatterskull Smashing and The Stone Brain, whose units must add theirs, and
Surgical Extraction's `TargetedController$…` (it reads `ChangeNum$` otherwise). The switched energy unless-cost
is the definition's now, `pay_switched_energy`, with no second prompt: the chosen number is the decision.

A ChangeZoneAll naming no player and targeting none reaches every player's zones, each owner's cards a batch,
and shuffles each library, moved into or not (CR 701.24d). No vocab card moved: every such filter names
remembered or imprinted cards, or the move is Terminus's.

What these scenarios cannot show:

- **A suspension inside a `Repeat` iteration.** No card prompts there: Grindstone's body is Cleanup and Mill, and
  Grist's is Cleanup, Token, Mill and PutCounter. The path exists (`resolve_child`, `RepeatLoopRt`) and is
  unexercised. The suspended loop that is exercised is RepeatEach's: Chaos Defiler's and Stronghold
  Gambit's prompts, through the `state` tier's snapshot check.
- **`AtRandom` differing by seed.** In a two-player game Chaos Defiler's random pick is among at most one card,
  so `stable_rand_below` draws nothing and no seed can differ. A replay matches, and removing the row
  shows as a prompt, but nothing proves the draw itself. It needs a card with several candidates.
- **`MaxRepeat$` binding.** Grindstone's bound is the library's size and a pair per iteration empties it first, so
  the bound never ends a loop. `Repeat` with no `MaxRepeat$` (Grist) runs until its check fails.
- **`Reveal$` on `AtRandom`.** Chaos Defiler's pick is a permanent, already public (`param_form_read` said so).

`replay` stayed byte-identical at every commit and the corpus was not re-recorded. `--selfcheck-knowledge` is
clean over Stronghold Gambit, and a scenario holds that Player B's frame at B's own choice has Air Elemental
at 0-1 in A's hand, known after the reveal.

**Lands with its first card, not here.** As in M2 and M3, nothing below has a card in the vocab or in this unit,
so none landed. The stage is when the card's own last need is met:

| item | first testable card | stage | what it must build |
|---|---|---|---|
| `ClearNamedCard`, `Imprint$`, `Imprinted$Amount` | The Stone Brain | M5 | landed in (c1), in the resolution's memory: the source has exiled itself, and what outlives a resolution is Chrome Mox's |
| `Discard Defined$ Player` (`Mode$ Hand`) | Runehorn Hellkite | M5 | landed in (c1); Echo of Eons has no Discard, despite M2's table |
| `UnlessCost$` on CopyPermanent, `UnlessResolveSubs$` | Springheart Nantuko | M7 | landed in (b), through `run_unless_loop` in `copy_permanent`; a step of `resolve_ability` waits for a second card |
| `RememberDiscarded$` | Fable of the Mirror-Breaker | M8 | a row reported from `discard`. Mox Diamond's script names it but never reaches `discard`: the replacement dispatcher takes its R: line whole, so it can't test the row |
| `ImmediateTrigger ConditionDefined$` | Agatha's Soul Cauldron | M8 | the condition read through the `Defined$` table, not the battlefield scan |
| `ImmediateTrigger UnlessCost$ Sac<…>`, `Repeat` with no `MaxRepeat$` | Grist, the Hunger Tide | M8 | the cost through `unless_payment`; a bound for CR 732 if a loop can go on with no library to run out |
| `ForgetImprinted$`, imprint on the source | Chrome Mox | M9 | a `FORGET` row on the imprinted slot; `forget_remembered` covers only the remembered set |
| `Branch` (`BranchConditionSVar$`, `True` / `FalseSubAbility$`), `ImprintCards$` | Goblin Welder | M10 | the evaluated side persisted in `EffectRuntime`, since a resume that re-derives a different child is fatal |
| `UnlessCost$` on Token, `UnlessPayer$ Player.IsRemembered` | none: Acererak the Archlich left the plan | | through `unless_payment`, with a payer per RepeatEach iteration |
| a suspension inside a `Repeat` iteration | none in `candidates/` | | the first card that prompts in one |
| an `AtRandom` draw among several | none in `candidates/` | | the first card with more than one candidate |

Appendix A and B still count these findings under M4; the table above says when each lands.

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

**Progress.** Units (a), typed parts from one parser and `can_pay`, and (b), one suspendable `run_cost_parts` and
`ChosenCost`, are done (2026-11-05, 06) and inert. Unit (c1), the new cost kinds and the cards they finish, is done (2026-11-06): the eleven M5 cards are in the vocab, ids 418 to
428. Unit (c2), `TapToPay` and the oddities, is done (2026-11-06), which closes M5: see **Done**.

A definition holds `CostParts`, a list of `CostPart {kind, count, count_is_x, filter, object, ...}`
(`src/components/cost_part.h`): `MANA`, `X_MANA`, `TAP`, `LIFE`, `ENERGY`, `LOYALTY`, `SACRIFICE`, `DISCARD`, `RETURN`,
`EXILE_FROM_HAND`, `EXILE_FROM_GRAVEYARD`, `EXILE`, and `MANDATORY`, which pays nothing. Filters are `FilterSpec`s, CARDNAME an
`ObjectSpec` SELF. `parse_cost` is the one parser: `read_cost_token` says what a token is, `cost_form_pays` whether a
`CostForm` pays it (activated, spell, triggered, ImmediateTrigger, alternative, flashback, escape, mana-only keyword,
Escalate, Ward, unless), and what no form pays is a `cost` finding, as before. Activations, `AltCost`, Flashback, Escape,
Kicker, Offspring, Replicate, Suspend, Impending, Escalate, Ward and `UnlessCost$` all read through it.
`parse_alt_cost_tokens`, `read_alt_cost_token`, `read_activation_cost_token`, `cost_token_paid`, Escalate's token loop,
Escape's split at `ExileFromGrave`, Ward's digit scan and the replacement line's `stoi` are gone, with every cost field
they filled: 15 on `AbilityDef`, 5 `unless_*` ones, 11 on `AltCost`, and on `CardData` the mana and `AltCost` pairs of
Flashback and Escape, Escalate's three and Ward's two.

`can_pay(parts, CostCheck)` (`src/queries/costs.h`) is the affordability check of activations, loyalty, mana abilities,
alternative, flashback and escape casts, a regular cast's additional sacrifice, free casts, suspend, miracle and an
ImmediateTrigger's energy. The caller supplies the mana after reductions and floors and the payee; the {T} exclusion and
the life reserve come off the parts. A path's own leniency is named at its call site in `CostCheck::unchecked`.

What stayed of (a), and why:

- **`unless_energy_expr`.** Galvanic Discharge's and Wrath of the Skies' amount is an SVar read as the effect resolves, not
  a count a part can hold. `Effect::Replacement::tapped_unless_life` is an int, read from its token, because its payment
  sits in `replacement_effects.cpp`. `AbilityDef::mandatory` is not a cost: `Mandatory$` writes it too.
- **`PlayCost$`.** `parse_play` still reads `PayEnergy<ConvertedManaCost>` by hand; the amount is no count. M7's.
- **Offers that don't call `can_pay`.** `run_unless_loop` tests the pool, life or energy inside its live menu;
  `pay_mana_cost` and that loop filter sources by the floating pool. A play permission's resource and the companion's {3}
  are no parsed cost.

Leniency kept, and in `docs/rules_deviations.md` §2.5: a mana ability's Sac, Return and energy are unchecked (Lazotep
Quarry, a candidate: `activate_mana_source` pays only a self-sacrifice, so the auto-payer never charges it); a spell's
additional sacrifice is checked only from hand and on a free cast; `NotPlayerTurn` isn't read on `Cost$ 0`.

(a) inert: `replay` byte-identical and `make check` green at each of the four commits, `make registry` a no-op, parse reports
(all 430 provisioned cards) and `--cards-frame` unchanged. Every provisioned card's parsed cost, each of its 838 interned
definitions and each keyword cost, dumped before and after: identical. 9,600 explore games over the 20 top-level, `league/`,
`heldout/` and `tests/mech_*` decks, 3,200 each of bo1, `explore:patient` bo3 and `--no-auto-mana`, under `--trace-decisions`
so that every menu is compared: 5.4 GB byte-identical. 240 games under `--selfcheck-state` clean. Over 675 games every kind
`can_pay` can refuse was seen both ways. Release `bench`, median of 9 alternating runs:
129.8 us per decision before, 131.0 after, inside the runs' 126.7 to 133.1 spread.

The parsed form differs from the old readers in cases no script has: a mana token after `ExileFromGrave` now counts; `Sac`,
`Return` and `ExileFromHand` on Escape and `ExileFromGrave` on an alternative cost are findings, where they were accepted
and dropped; several bad tokens in one `UnlessCost$` give a finding each and keep the rest; a Counter's `PayEnergy<N>`
would pay energy, not {N}.

**Unit (b).** `run_cost_parts(CostPayer, FrameCtx, orderer)` (`src/cost_payment.{h,cpp}`) pays a cost through its path's
table of steps, in the order that path pays: CAST, ACTIVATION, MANA_ABILITY, TRIGGER, IMMEDIATE_TRIGGER, UNLESS. A step pays
a part or chooses what a part moves (a `PickRule` row: part, cost slot, wording, how many); it may ask through its
`FrameCtx`, and `CostRun`, on `PendingCast` and `PendingActivation`, holds all the progress. `FrameCtx::flow` gives those two
flows, and a special action, a context that parks with their tag. A path of one step keeps no `CostRun` and re-derives its
question on resume, as it did. `pay_mana_cost` replaces `prompt_mana_payment`: its `ManaPaymentRT` persists on the flow and the
Converge sink is armed for each entry, so nothing spans a suspension in a static or on the stack.

Newly suspending: a row of the interactive mana payment (cast, activation, suspend, companion) and a hybrid pip, both under
`--no-auto-mana`. They were the only blocking choosers of a cost; there is no discard in an activation's pay step yet, since
`Discard<N/...>` is (c)'s. Such a prompt carries `PendingQuery::mid_payment`: no Cancel row of `--offer-cancel` (the payment's
own last row cancels), and its frame reports tag NONE, as a blocking prompt's did, because the `pending` table's `tag` would
otherwise change. A concession read at one is answered by the unwind as before, so the payment runs out on first choices and
what follows it is discarded. `pending_cost` shows what it showed; the `cost_part` enum is unchanged.

`ChosenCost` (`src/components/chosen_cost.h`) is built on `PendingCast` and recorded on the `Spell` at CR 601.2i: mode
(`REGULAR`, `FLASHBACK`, `ESCAPE`, `ALTERNATIVE`, `EVOKE`, `IMPENDING`, `WARP`), Offspring, impulse, without-mana-cost, back
face, X, kicks, replicate and escalate counts, the gift, `mana_spent`, `colors_spent` and `paid`, what the cost moved. Gone:
the `Spell`'s six `cast_with_*` flags, `cast_back_face`, `x_paid`, `kicked`, `replicate_count`, `gift_promised`, `mana_spent`,
`colors_spent`; on `PendingCast` the `use_*` flags, `impulse_cast`, `without_mana_cost`, `cast_back_face`, `x_paid`,
`kicked_flags`, `replicate_count`, `gift_promised`, `mana_spent`, `mana_spent_colors`, `escalate_extra`, every `deferred_*`
field, `life_x_announced`, `cost_removals`, the pick counters and the delve and escape progress; on `PendingActivation`
`sac_choice`, `return_choice` and the snapshot pair; and the steps `COST_SAC`, `COST_RETURN`, `PAY`, `PAY_APPLY`, `ALT_*`,
`SPELL_SAC`, `ESCALATE_DISCARD`, `DELVE_*`, `DEF_*`, `MANA_PAY`, each flow's now being one `PAY_COSTS`. Kicker, Replicate and
Escalate's offers ask `can_pay`, whose `CostCheck` takes the hybrid pips, a repeat count and the hand and life the rest of the
cost takes; `Discard<N/...>` is a part it holds a player to.

`cost:` keys are `<KIND>[.<form>]` from `CostPart::Kind` (`docs/game_running.md`), counted at the pay steps: `Sac` is
`SACRIFICE`, `SacSelf` `SACRIFICE.self`, `DiscardSelf` and `DiscardHand` `DISCARD.self` and `DISCARD.hand`, `PayLife` `LIFE`,
`PayLifeX` `LIFE.x`, `PhyrexianLife` `LIFE.phyrexian`, `PayEnergy` `ENERGY`, `Tap` `TAP`, `Mana` `MANA`, `Loyalty` `LOYALTY`,
`Return` `RETURN`, `Discard` `DISCARD`, `ExileFromHand` `EXILE_FROM_HAND`, `ExileFromGrave` `EXILE_FROM_GRAVEYARD`, `Delve`
`EXILE_FROM_GRAVEYARD.delve`. Parse findings keep their `cost:<token>` names.

What stayed of (b), and why:

- **What determines a cost.** Kicker, Replicate, X, hybrid and Phyrexian pips, life X and the gift stay steps of `run_cast_flow`:
  they announce or determine the cost before the targets (CR 601.2b, 601.2f). `run_cost_parts` starts after the targets.
- **Suspend and the companion.** They call `pay_mana_cost` directly from `PendingSpecialAction` (tag `SPECIAL_ACTION`, past the
  end of SCHEMA's `pending_tag` table, which no frame can show). Mana is all they pay, and neither is a parsed cost.
- **`tapped_unless_life`** stays in `replacement_effects.cpp`: a latched `SBE_LATCHED` prompt, except an entering token's
  `ask_inline`, which blocks. **`pay_switched_energy`** asks nothing.
- **`EntryInfo::cast_from_hand`** is set before any cost, belongs to the card's next entry and has its own lifetime rules
  (cleared on a move off the stack elsewhere, dropped by a rewind); moving it would change a permanent spell put onto the
  battlefield by an effect, so it isn't part of `ChosenCost`.
- **A mana ability's `SubAbility$`** resolves through a blocking context. No vocab rider asks.

A mid-payment prompt reporting no paused flow, and the concession running a payment out, are kept only so the frames hold. A commit
that drops them changes frames. The other oddities (b) kept are judged under **Done**.

(b) inert: `replay` byte-identical and `make check` green at every commit, `make registry` a no-op, parse reports (all 430
provisioned cards) and `--cards-frame` unchanged, `onboard.py check` 0 stale, `tests/corpus`, `tests/cli` and `src/featurize`
untouched. 9,600 explore games over the 20 top-level, `league/`, `heldout/` and `tests/mech_*` decks, 3,200 each of bo1,
`explore:patient` bo3 and `--no-auto-mana`, under `--trace-decisions`: 5.5 GB byte-identical. The `--rmstate-dump` frames of 738
games, 246 of each, byte-identical. 120 random-agent games under `--offer-cancel`, 2,143 cancellations and so rollbacks among
them, byte-identical. 240 games under `--selfcheck-state`: no error, and byte-identical but for the summary's `snapshots=` in 78
`--no-auto-mana` games (8,850 to 9,254 in all), a mid-payment prompt now being loop-safe. The snapshot check, counted with a
throwaway instrumented build at `--selfcheck-state-sample 1` over 300 games (22,462 snapshots) and the five new scenarios, landed on
a mana row 703 + 12 times in a cast, 276 in an activation, 5 in a suspend, once for a companion, on a hybrid pip 3 times; on the
pitch 106, alternative return 20, alternative sacrifice 27, additional sacrifice 13, Escalate discard 3, delve count 7 and pick
97, flashback sacrifice 3, Escape by count 22 and by types 6, activation sacrifice 36 and return 10; on the unless offers of mana
101, energy 6, life 2 and discard 2 with its pick 2; on The Fantasticar's sacrifice 3 and an ImmediateTrigger's energy 5. No error.
Release `bench`, median of 9 alternating runs: 130.0 us per decision before, 130.5 after, inside the runs' 127.5 to 133.5 spread.

**Unit (c1).** Every new kind is a part in `read_cost_token` and a line of `cost_form_pays`, paid by a step or a `PickRule` row, with
no branch for one card:

- **`TRIGGER`'s table.** `pay_trigger_cost` runs on a `CostRun` in the resolving level's runtime (an `EffectRuntime` alternative), so
  the path now parks. A triggered ability's offered cost has an offer (asked only if `can_pay` holds), a `kTriggerDiscard` pick, the
  mana and the moves. That offer is the ability's one "may": `OptionalDecider$` doesn't ask again. A payment given up by its last row
  restores its mana abilities and declines. The state format's `cost_part` gains `TRIGGER`. `Discard<N/...>` and a mana cost landed
  here, not on an activation: Formidable Speaker's, Melded Moxite's and Nihil Spellbomb's are `AB$ ... | Cost$` under an `Execute$`.
- **`EXILE_SELF`.** One step for `Exile<1/CARDNAME>`, `ExileFromHand<1/CARDNAME>` and `ExileFromGrave<1/CARDNAME>`, on the activation
  and mana ability paths.
- **The pitch.** A filter's card view now takes the layer-5 override (a bug fix, below), so `EXILE_FROM_HAND` reads its compiled
  filter, `CostPart::color` is gone and `can_pay` asks for all N.
- **`ActivationZone$`.** Read through `Zone::from_script_name`, offered by `offer_zone_activations`, one call per zone. Hand and
  Graveyard are the zones any script names; another stays a finding until it has a call and a card.
- **A mana ability from the hand.** Listed with the battlefield's rows at priority, in a payment and at an unless offer
  (`append_hand_mana_actions`); the flow's FINISH takes mana first, and production needs no `Permanent`. The auto-payer doesn't spend
  it and `can_pay_mana` doesn't count it (`docs/rules_deviations.md` §1.4), so a spell it alone would pay for is offered once the
  mana floats. A permanent's copy of the ability is no source (CR 113.6b).
- **`AlternateAdditionalCost`.** `CardData::additional_choices`. `can_cast_now` needs one form payable, the cast's `ADDITIONAL` step
  asks which before the gift and the targets (CR 601.2b), and it pays with the rest, by a `PickRule` row for the discard.
- **Hybrid pips.** A `MANA` part holds them; `announce_hybrid_pips` is the cast's step, an activation's `HYBRID` and an alternative
  cost's `ALT_HYBRID`. `can_pay` reads a cost's own pips, X is bounded with them, the floor counts them. Forms other than `ACTIVATED`
  and `ALTERNATIVE` still refuse a hybrid token, since nothing else announces one.
- **Cost statics.** `StaticAbility::CostActivator`, one predicate for `Player`, `You`, `Player.!Active`.
- **Rows.** `Imprint$`, `ClearNamedCard$`, `Imprinted$Amount`, `<Set>$Amount`, `/NMinus.N`, `Discard Defined$ Player`,
  `Count$Adamant_<N>.<Color>.<high>.<low>`, and the `CastSA>` prefix on a trigger's `CheckSVar$`.

Clean was not correct three times. Vibrance's `CheckSVar$ CastSA>...` was read as a permanent filter, so neither trigger could fire.
The Stone Brain's `ChangeNum$ Y`, `SVar$X/NMinus.4`, silently read 1, and its `NumCards$ Z` the same. A ChangeZone that targets the
player it searches (`ValidTgts$ Opponent`, `DefinedPlayer$ Targeted`) was taken for a target move, and with `Origin$ Graveyard` for a
target in a graveyard: a FATAL on The Stone Brain.

Bugs fixed in shared code, each with a scenario failing without it: a filter's color on a card off the battlefield ignored Mycosynth
Lattice (the vocab cards that read one are named in that commit, and change only under Lattice); a search with `ChangeNum$` of 0 didn't
shuffle (Ugin, Eye of the Storms' -11, The Stone Brain).

What differs from the plan as written, and why:

- **Imprint is the resolution's, not the source permanent's.** The Stone Brain exiles itself as a cost, so it has no `Permanent` at
  resolution, and its imprint ends with the resolution (`DBCleanup`). The row writes the resolution's `imprinted` slot. An `ObjectSet`
  on the permanent has its first test in Chrome Mox (M9), which keeps its other findings until then; its `Imprint$` finding is gone,
  so the table below carries it.
- **`Discard<N/...>` on an activation has no row.** No provisioned script has one, and a row with no test is not done.
- **`spent_w` to `spent_g` are filled only for a card that reads them.** Filled for every cast permanent they moved 467 of the open
  vocabulary fixtures' decisions, a feature change for a fact no other card reads.

What stayed of (c1), and why:

- **The cast's hybrid prompts are unfiltered under `--no-auto-mana`**, and `pending_cost` lists a cast's pips whole at each prompt. Both
  are frames, kept so `--no-auto-mana` games don't move. Under auto-mana a card that reads spent colors filters its rows.
- **Which sources pay Vibrance's generic mana** is the auto-payer's, so {R}{R} can't be aimed through a generic pip (§1.4).
- **A triggered ability's sacrifice shows no `pending_cost`**: The Fantasticar's frames would move.

**Lands with its first card, not here.** Checked over every provisioned script: `Exert<`, `K:Bargain`, `K:Convoke`, `Waterbend<` and
`K:Crew` appear in one script each, so no other card can test them. Of those, none can enter before the stage below.

| item | first testable card | stage | what it must build |
|---|---|---|---|
| `Exert` (CR 701.43) | Arena of Glory | M6 | done (M6's Done) |
| Convoke (CR 702.51) | Hogaak, Arisen Necropolis | M7 | done (M7's Done) |
| Bargain (CR 702.166) | Beseech the Mirror | M7 | done (M7's Done) |
| Waterbend (CR 701.67) | Invasion Submersible | M10 | a `TapToPay` row and a cost token calling `set_tap_to_pay`, with `Exhaust$`. CR 701.67b limits a tap to the generic in the waterbend cost, which the rule table doesn't hold: this card's whole cost is it |
| Crew (CR 301.7) | Unlicensed Hearse | M10 | an activation whose cost taps other creatures |
| `Imprint$` that outlives the resolution, an `ObjectSet` on the permanent | Chrome Mox | M9 | the slot on the source, with `ForgetImprinted$` |
| `Discard<1/Hand>` on a trigger | Phyrexian Dragon Engine | M12 | `whole_hand` in `TRIGGER`'s table |
| `Discard<N/...>` on an activation | none provisioned | | one `PickRule` row in `kActivationSteps` and its `cost_form_pays` line |
| `Exile<N/filter>`, `ActivationZone$` past Hand and Graveyard, hybrid pips on any other cost | none provisioned | | a row and a call of the function already there |

(c1)'s evidence: `replay` byte-identical at every commit and `tests/corpus`, `tests/cli` untouched, so no `make regen`. 51 scenario
files name each card, each ability and each can't-pay case. Every new chooser parks in a scenario the `state` tier snapshots at
sample 1: the offer, the trigger's pick and mana row, the form of an additional cost and its pick, a hand card's row, a hybrid pip on an
activation and on an alternative cost. `decks/tests/mech_cost_kinds` is the fixture: 100 games each of `explore` and `explore:patient`,
every key hit, no error. Its one draw, and 33 before the fixture went to one Hellkite a side, is both players decking on Runehorn
Hellkite's draw seven (CR 104.4a), read in each log. 96 games under `--selfcheck-knowledge` at sample 1, half `--no-auto-mana` (73 NameCard
resolutions among them), and 40 random-agent games under `--offer-cancel`: no error. `scripted` and Python `explore` each play six
games of it. No candidate deck holding one of these cards is wholly in the vocab yet, so the fixture is the only deck that plays them.

**Done (2026-11-06).** Unit (c2), and with it M5.

- **`TapToPay`** (`src/components/tap_to_pay.h`, the rule table in `mana_system.cpp`). A kind on the card or the cost, and a row: `keyword`,
  `eligible` (a `FilterSpec`) and `pips`. `CostRun::tap_to_pay` carries it, so `PAY_MANA` and `pay_mana_cost` serve a cast, an activation and
  a triggered ability alike; `CostParts::set_tap_to_pay` has no caller until `Waterbend<` is a token. A row is the kind, the table line and
  the keyword (or cost token). `GENERIC_OR_OWN_COLOR` is Convoke's and no row uses it: proved by a throwaway row, Improvise switched to it,
  a blue artifact paying {U} under both payers and colorless ones refused. A real Convoke choice is still the deliberate change of M7, since
  the payer taps only for what mana can't pay (`docs/rules_deviations.md` §1.4).
- **`mana_spent`** is what left the pool. Vocab cards that move: Kappa Cannoneer and Metallic Rebuke. Roiling Vortex's and Lavinia's outcome
  moves only for a cost of generic mana alone (cast without paying its mana cost, Thalia or Trinisphere adding it).
- **`delve_exiled`** is gone. `ChosenCost::paid` rides `EntryInfo::paid` to the entry, so it has its reader, and an object that wasn't cast has
  none. `pay_mana_cost` was only ever called with `has_delve` false, so its Delve rows, `delve_exile_one` and the snapshot's undo went too.
  `EntryInfo::paid` is not in the frames, as `delve_exiled` wasn't: nothing decides on it between a spell resolving and it entering.

The oddities, against the rules:

| oddity | verdict | outcome |
|---|---|---|
| Phyrexian life paid as the pip is announced | Bug by order (CR 601.2c, 601.2g-h), though neither Phyrexian card (Dismember, Surgical Extraction) reads life while announcing, so no game outcome moved | Fixed: owed in `CostRun::life`, paid at `LIFE`; §1.1 and `rewind_cast`'s refund gone |
| Activation's picks log after they move, a cast's before | Not a rule: a log | `docs/rules_deviations.md` §1.1 |
| `repeats_chosen` | Bug (CR 118.3): one object can't pay two costs | Fixed, the row removed. No script reaches it |
| Pitch and return not stopping on an empty pool | Bug (CR 118.3, 601.2h), as was `stops_when_empty` paying short | Fixed: a pick with no candidate fails the payment. Every offer now holds the whole cost, so nothing reaches it |
| Flashback, escape, alt casts skip X, Phyrexian, Delve | Half right. An alternative cost replaces the mana cost, so its {X} is 0 and its pips aren't owed (CR 118.9, 107.3b); additional costs, Delve and TapToPay act on the total (CR 118.9d, 702.66a) | Fixed: one flow, `PendingCast::mana_cost_replaced` skipping only the mana cost's own steps. Reachable by the exile grants only. Offspring stays a row of two casts, §2.5 |
| A spell's additional sacrifice checked only from hand and on a free cast | Bug (CR 118.3, 601.2f): Natural Order from exile cost nothing | Fixed at `can_cast_now` |
| `NotPlayerTurn` not read on `Cost$ 0` | Bug, no script | Fixed |
| A mana ability's Sac, Return and energy unchecked | Gap | Stays for Lazotep Quarry, M8 |

Found on the way, and fixed: The Stone Brain's target sits on a sub-ability, which `can_activate_now` didn't read, so an opponent with protection
from everything made it FATAL (`explore:patient` bo3 seed 210, which a comparison caught and the gate did not).

**Test.** Each (path, part) a provisioned script has, with a scenario naming a card that has it, was listed from the scripts. What was unproved
got a scenario, 14 in all; hybrid, trigger and unless parts were (c1)'s. Payment order, CR 601.2h: a cast's life after its mana, an activation's
sacrifice after its mana, delve's exile after the mana, a land tapped then sacrificed. Can't pay hides it: cast, activation, alternative, pitch evoke,
flashback, escape, kicker, replicate, suspend, Improvise, a trigger's offer, an unless offer. A mid-payment suspension for every chooser: (b) and (c1)
counted theirs; a throwaway instrumented build at `--selfcheck-state-sample 1` over 240 fixture and `ur_delver` games, half `--no-auto-mana`, 24,088
snapshots and no error, landed on an Improvise row 140 times, delve 44, a Phyrexian pip 12. A kicker asked on a free cast is the same step as any
kicker's, and its scenario passes the `state` tier; no snapshot was seen to land on it.

No script has these, so each stays out of the scenarios:

| combination | what it would need |
|---|---|
| Two sacrifices on one cast, or a Sac and a Return on an activation | a card; a second pick already finds the first gone |
| A pick reaching an empty pool | a gate that lies; the payment fails |
| `Condition$ NotPlayerTurn` on `Cost$ 0` | a card |
| X, Phyrexian pips, Delve or Improvise with a card's own flashback, escape or alternative cost; kicker or replicate with one | a card; the one flow carries it |
| {X} in an alternative cost | `cost_form_pays` refuses it, a finding |
| `TapToPay` on an activation or a trigger | Waterbend (M10) |
| A colored pip paid by a tap | done: Convoke (M7) |
| A mana ability's Sac, Return, energy | Lazotep Quarry (M8) |
| Sacrifice as an unless cost | Grist (a candidate, its own stage) |

Evidence. Migration: 3,300 explore games over `mech_tap_to_pay` and `mech_cost_kinds`, a third each of bo1, `explore:patient` bo3 and
`--no-auto-mana`, under `--trace-decisions`: 2.4 GB byte-identical with 12,927 Improvise taps; `make registry` a no-op. Debug `bench`, engine alone,
three alternating pairs: 1,577 us per decision before, 1,595 after, 1.1% in all three; release was not measured. `mana_spent`: `--rmstate-dump`
frames of 450 games with no Improvise card, 3.9 GB byte-identical. Pick cleanup and one flow: 900 games each, identical. `tests/corpus` and `tests/cli`
never moved, so no `make regen`: no corpus game pays Phyrexian life or casts an Improvise card. After the last behaviour commit: 126 games
under `--selfcheck-state`, 4,947 snapshots, no error, and each of the six `mech_*` fixtures at `--fuzz-games 100`, every key hit, no error. One draw, `mech_cost_kinds`
`--no-auto-mana`: Runehorn Hellkite's draw seven decking both players, read in the log (CR 104.4a).

No candidate entered: the parse report over the 46 candidate cards outside the vocab finds none at zero. Ten stand at one finding, each another
stage's: Morph, Overload, Sunburst, Fuse, `DividedAsYouChoose`, a `Draw` replacement, `CheckSecondSVar` (Massacre), `Revolt`, `TgtZone`, `Effect.Continuous`.

### M6. Mana with provenance (L)

**Today.** The pool is `multiset<Colors>`, so mana carries nothing. Spells record a pip count and colours
only. Restrictions are three bools on the source; the only rider is a global,
`pending_cant_be_countered`.

**Left to this stage by M5.** `spent_w` to `spent_g` are filled only for a card that reads them (`spell_reads_colors_spent`): generalise them here, as `paid`
makes the fact free, and expect the frames to move. A mana ability from the hand (Simian Spirit Guide) is one the auto-payer doesn't spend and
`can_pay_mana` doesn't count (`docs/rules_deviations.md` §1.4): a `ManaUnit` source should close that too.

**Starting point.** `ManaPayee` (`src/mana_system.h`): every payment already says what it pays for and
why (`ManaPurpose`: cast, activate, other), and `mana_source_usable_for` reads it. Restrictions still
act on the source, when it is tapped, so mana already floating carries none.

**Build.** `ManaUnit {color, source, rider, restriction}` in the pool and a `paid` list on the `Spell`.
`colors_spent`, Converge, Sunburst (CR 702.44), `Count$CastTotalManaSpent` and `ManaFromCard` become
views of it. Riders and restrictions are data on the unit; the global goes. `mana_snap` rollback must
carry units.

**Lands here.** `AddsKeywords` riders, `S:Mode$ ManaRestriction` (Hogaak's new script), `ManaConvert`
(Agatha's Soul Cauldron), Sunburst.

**Test.** Rollback restores units; a rider survives payment across a suspension; the pool's frame stays
identical for unrestricted mana, and restricted mana is distinguishable in the seat's frame.

**Progress (2026-11-06).** Unit (a), the inert migration, and the two rules fixes it uncovered. Unit (b) closes the stage: see **Done**.

What moved. `Player::mana` is a `ManaPool` (`src/components/mana_unit.h`): `ManaUnit`s sorted by color, oldest first within one, so with nothing restricted it
spends as the multiset did. A unit has its `color`, `source`, `riders` and `restriction`. `AbilityDef::mana_restriction` replaces the six `restrict_*` bools, and
a chosen type is read off the source as the mana is made, then interned (`intern_mana_restriction`, append-only like the ability definitions), so it outlives the
source and a snapshot holds a pointer. `ChosenCost::paid` is now `moved`; `ChosenCost::mana_paid` is the units that left the pool, in order. `mana_spent()`,
`colors_spent()` (Converge), `mana_of_color` (Vibrance, through `EntryInfo` and `Permanent::cast_mana_paid` to last-known information) and `mana_from` are views of it.
`s_mana_spent_sink`, `CostRun::spent_colors` and the `RECORD_MANA_SPENT` step are gone; the list is a parameter. `mana_snap` and the snapshot carry units with no
further edit.

Carrying the data alone changes nothing; reading it does, for vocab cards, so each read is a commit after the inert ones:

| commit | fix | vocab cards that move | scenario, failing on the commit before |
|---|---|---|---|
| `Mana: floating mana keeps its restriction` | CR 106.6: a unit is spendable only on what its restriction allows, everywhere the pool pays or is counted; a mana ability's own cost is paid as an activation | Eldrazi Temple's surplus {C}; Cityscape Leveler's Powerstone floated at priority; Cavern of Souls and Secluded Courtyard under `--no-auto-mana` | `cr106_6_surplus_restricted_mana_keeps_its_restriction`, `cr106_6_floating_powerstone_mana_keeps_its_restriction`, `cr106_6_cancelled_payment_restores_restricted_mana` (the rollback) |
| `Mana: a rider goes with the mana that carries it` | CR 106.6: the spell the mana is spent on, not the next one cast; `pending_cant_be_countered` is gone | Cavern of Souls under `--no-auto-mana` | `cr106_6_cavern_rider_goes_with_the_mana` (the rider across parked rows) |

The pool has no choice of unit: `spendable_unit` takes restricted or rider mana first, else the pool's order (`docs/rules_deviations.md` §1.4).

What stayed, and why. `Permanent::cast_mana_paid` is still filled only for a card that reads it (`spell_reads_colors_spent`), and the pool's frame is colors only: both move
frames. Simian Spirit Guide's mana is still unseen by the payer: a behaviour change. `adds_no_counter` stays on the definition, for the auto-payer's preference. An activation
or a trigger records no list: nothing reads one. `max_available_mana` still counts a restricted unit, a coarse bound whose overestimate the X menu's trial absorbs.
`mana_source_affordable`'s estimate still asks as "no particular payment", so a mana ability's cost can't count mana restricted to activations of it; no card has both.
`identical_activated_ability` now compares the Eldrazi restriction too, which it skipped; no two abilities differ only there.

(b) must:

- Let `collect_available_mana_sources`, the auto-payer and `can_pay_mana` see a hand mana ability (the unit's source already can be one), and close `docs/rules_deviations.md` §1.4's last sentence. Menus move.
- Drop `spell_reads_colors_spent` from `record_cast_entry`, so `spent_w` to `spent_g` fill for every cast, and expect the frames and `make regen`.
- Make restricted mana distinguishable in the seat's frame, the pool's frame unchanged for unrestricted mana: today a seat sees mana it can't spend as ordinary, and `pl.mana.size()` in the player row counts it. Schema, encoder, `make registry`, `docs/state_format.md`.
- Give `ManaFromCard` a reader: `mana_from` has none yet, and `ValidSA$ Spell.ManaFromCard.StrictlySelf` is unparsed. `Count$CastTotalManaSpent` (Mockingbird) is `mana_spent()` and needs its row.
- Riders are a bit each, which `AddsKeywords` outgrows: it needs a keyword and a duration, so the field becomes a pointer to interned data, as `restriction` is.
- `S:Mode$ ManaRestriction` (Hogaak) restricts the spell, not the mana: a check of each unit's `source` against the spell in `unit_spendable_on`, so `source` must then be stamped in a simulated payment too (it is empty there).
- Sunburst reads `colors_of` at entry, so it needs the entry's list whatever `spell_reads_colors_spent` says. `ManaConvert` (Agatha's Soul Cauldron) goes through `spendable_unit`'s `NO_COLOR` read as Mycosynth Lattice does.

Evidence. Inert tip `bd862230` against the starting commit, release builds, 22 decks over all 484 ordered pairs: 3,200 games each of bo1, `explore:patient` bo3 and
`--no-auto-mana`, under `--narrative --trace-decisions`, 5.5 GB byte-identical (Cavern of Souls tapped 9,247 times, Eldrazi Temple 4,330, Secluded Courtyard 4,046, Vibrance cast 789, Prismatic
Ending 905); `--rmstate-dump` frames of 300 games, 2.1 GB identical; 120 random-agent games under `--offer-cancel` identical; 240 games under `--selfcheck-state`, no error; `make registry`
a no-op, `--parse-report @vocab` and `--cards-frame` identical, `tests/corpus`, `tests/cli` and `src/featurize` untouched. No game has a Powerstone or Abundant Countryside: the token is proved by scenario.
Release bench, median of 9 alternating runs, the actor loop: 2,476 decisions/s before, 2,480 at the inert tip; 2,481 before, 2,474 after the fixes (0.3%, inside the runs' spread); engine alone
136 us either way.

The fixes against the inert tip, the same 9,600: floating restriction moves 24 games (4 bo1, 16 `explore:patient`, 4 `--no-auto-mana`), 23 with `eldrazi_aggro`. Every first difference read: 22 follow a Temple's {C}{C} that turn, the other two a Cavern's or Courtyard's mana floating under `--no-auto-mana`. The rider moves none, so it rests on its scenario. After both, 240 games under `--selfcheck-state`, no error. `tests/corpus` and `tests/cli` never moved, so no `make regen`.
The `actor` tier's torch part was skipped, as it is without `$ROBOMAGE_TORCH_PYTHON`: `robomage_nn` was not rebuilt.

**Done (2026-11-06).** Unit (b), and with it M6.

- **The hand's mana.** `collect_available_mana_sources` lists the hand's mana abilities after the battlefield's, so the auto-payer and `can_pay_mana` see them. A card spent is a card given up: it goes last, after a self-sacrificing source. Three holds keep one card from paying two things (CR 118.3): it is no source of its own cast or activation, `ManaPayee::chosen` (the run's picks, set in `CostPayer`) keeps a chosen discard out, and `ManaPayee::hand_reserved` holds an offer to the cards the rest of the cost takes, `fewest_hand_cards` for a form of an additional cost. The row at priority stays, since the player may want the mana for nothing that asks. `docs/rules_deviations.md` §1.4's last sentence is closed. `tests/corpus` and `tests/cli` did not move: only `mech_cost_kinds` holds Simian Spirit Guide.
- **Riders.** `ManaUnit::rider` and `AbilityDef::mana_rider` are one pointer to an interned `ManaRider`: `cant_be_countered`, and a `keyword` with `valid`, the spells it goes to, and `until`. `adds_no_counter` went into it. A unit whose rider matches the spell gives `Spell::keywords_until_eot`, which `EntryInfo` hands to `Creature::eot_keywords` (CR 400.7a). `spendable_unit` takes such a unit first, and one whose rider would be wasted after plain mana.
- **`pool_mana` (49).** A row per color, restriction and rider of floating mana that has one, for both players, all three viewers. The player row is untouched and counts it all, so plain mana adds no row. Token type 35. `chosen_type` and `keyword_valid` are text and dropped by the featurizer: a subtype column costs 1,025 categorical ids, and the plan's "no layout bump" needs them. 5,090 of 6,144 are used. The CLI notes it beside the pool; `--selfcheck-state`'s complete check holds its rows to the pools; `expect_dump` reads `mana` and `mana_unit`.
- **Sunburst (CR 702.44).** An `EtbCounter` whose count is `colors_of` the entry's list and whose kind follows the printed types, so what wasn't cast has no list and gets none. `spell_reads_colors_spent` holds it. A creature with Sunburst takes the same line with +1/+1 counters; no provisioned script is one, so that arm has no scenario.
- **Exert (CR 701.43).** `CostPart::EXERT`, `CostStep::EXERT` on the activation and mana ability paths, `Permanent::exerted_by`, read in the untap step and restored by `ManaPaymentSnapshot`. `exerted` is a bool column of `object`, numeric so that its 0 leaves the recorded inputs alone: a categorical slot's none is not 0. A control change is the one thing a bool can't say, and no card does both.
- **`Produced$ R R`** is two mana. A payment compares a permanent's abilities by what they add net of their own mana cost, or Arena of Glory's second ability would always replace its first. So the payer never chooses that ability, and it is offered at priority, the mana floats and the rider goes with the first spell it matches (§1.4).

Against the (b) list: `ManaFromCard`, `Count$CastTotalManaSpent`, `S:Mode$ ManaRestriction` and `ManaConvert`'s scoped form are the table below; `spell_reads_colors_spent` stays, under **What stops `spent_w` to `spent_g`**.

**Rows with no card now.** Checked over all the provisioned scripts: each name appears in the one script given, none of them in the vocab or this unit's cards, so none was built.

| item | first testable card | stage | what it must build |
|---|---|---|---|
| `S:Mode$ ManaRestriction` | Hogaak, Arisen Necropolis | M7 | done (M7's Done) |
| `ManaConvert` with `ValidPlayer$`, `ValidCard$`, `ValidSA$` | Agatha's Soul Cauldron | M8 | `any_mana_as_any_color_active` taking the payee, read through `spendable_unit`'s `NO_COLOR` as Mycosynth Lattice is; it also needs `GainsAbilitiesOf$` |
| `Count$CastTotalManaSpent` | Mockingbird | M8 | a row reading the entry's `mana_paid`, which `spell_reads_colors_spent` must then hold while it stands; it also needs `K:ETBReplacement:Copy` |
| `ManaFromCard`, `ValidSA$ Spell.ManaFromCard.StrictlySelf` | The Myriad Pools (The Everflowing Well's back) | M9 | `mana_from` as the reader, on a `SpellCast` trigger's `ValidSA$`; it also needs `Clone`'s `CloneTarget$` and `Duration$` and `ValidCard$ Permanent` |

**What stops `spent_w` to `spent_g`.** `spell_reads_colors_spent` stays in `record_cast_entry`, and `tests/open_vocab/` was left alone, because the acceptance did not hold, and did not before this unit. Since settled, in §7: the numbers below are from before `s0_rows.json`.

`learn/.venv/bin/python` runs both here (torch 2.10; the S0 originals are in `ckpt.v1/`). `open_vocab_acceptance.py gate` **fails at `e71d7e71`, where this unit started, and at this tip, with the same numbers**: 8 failures, max |Δ logit| 0.145 against `TOL` 1e-4 over 614 decisions; `full`, 72 failures over 35 networks and 8,735 decisions. So nothing here broke it, and the check is red with or without the flag.

The cause is the Forge pin bump `ec57222a`. For one network over all 8,735 decisions, 1,813 fail, every one holding Voice of Victory (`card_subtype` Soldier to Bard: 1,050) or Urza's Saga (its oracle-text row: 876 of 1,062); the 6,736 with neither stay within 9.8e-06. `inputs.json` can't see that: it digests a decision's tensors, and a card's table rows aren't in them, so the `featurize` tier stays green. The `actor` tier, which would see it, has been skipped since, for want of `$ROBOMAGE_TORCH_PYTHON`.

The flag itself turned out harmless to the networks. In a throwaway build with it dropped, 467 of the gate's 614 digests moved, and the acceptance's numbers and its 150 failing decisions didn't change at all, the 462 without those two cards staying within 5.7e-06. `spent_w` to `spent_g` arrived in `d52df694`, after every S0 checkpoint, so `rmlearn.migrate` left them at zero: `num_proj.weight` reads exactly 0 at those five columns in the latest checkpoint of all ten runs read. A network trained from now on learns them from 0.

To decide, in order. (1) What the acceptance means after a card's table rows move on purpose: re-baseline S0's outputs for those two cards, or have `gate` and `full` leave out decisions holding them, as `--skip-cards` does for `--expanded`. Either changes the check, so neither was done. (2) Then the flag can go, in its own commit, and the digests can be re-recorded. `inputs --drop-moved` would drop 467 decisions the networks don't feel, which shrinks the check; plain `inputs` would bring back the 96 dropped earlier, which moved for real. Keeping the file's decisions with today's digests needs a small mode of `write_inputs`.

What stayed, and why:

- **Which sources pay a generic pip** stays the auto-payer's, so Engineered Explosives' count can't be aimed under auto-mana; `--no-auto-mana` aims it, and a scenario proves that (§1.4). A card that reads colors spent paying by rows under auto-mana would close this, and Vibrance's half with it, but moves Vibrance's frames.
- **The form of an additional cost** is offered with its mana unchecked: with one card and one form, a discard the mana needs fails the payment and reverses the cast (CR 733.1). The offer of the spell holds the whole of it.
- **A card exiled for mana** isn't given back by a reversed payment, as a sacrificed source isn't (§2.5).
- **`mana_potential`** counts the battlefield alone, so the frame's source counts don't show a hand's mana. It is hidden from the opponent, and the column is every frame's.
- **A rider's keyword on a spell** shows in `object_keyword`, with no `object_provenance` row: that table is read for permanents.

Evidence. Every new behaviour's scenario failed on removal: own payment, chosen discard, reserved hand, the card last, the entry's list, net yield, the rider's filter, the plain-before-wasted order, the exert restored. 31 scenario files, naming each card and each ability. `tests/corpus` and `tests/cli` never moved but for the new `tests/cli/restricted_mana`, so no `make regen`. `decks/tests/mech_mana` is the fixture: `--fuzz-games 100` in `explore` and `explore:patient`, every key hit, no error, no draw. 120 games at `--selfcheck-knowledge --selfcheck-state-sample 1`, a third each of `explore`, `--no-auto-mana` and random under `--offer-cancel`: no error. `scripted` and Python `explore` each six games of it. No new prompt: the rows are existing kinds. The rider's scenarios run in the `state` tier; a payment parked with rider mana is among the 40 `--no-auto-mana` games, but no snapshot was counted landing on one. Release `bench` against `e71d7e71`, median of 7 alternating runs: engine alone 137.6 us per decision before, 135.6 after; the actor loop 2,483 decisions/s before, 2,481 after, inside the runs' spread. All seven `mech_*` fixtures at `--fuzz-games 100`: every key hit, no error; `mech_cost_kinds`' three draws are Runehorn Hellkite's draw seven decking both players (CR 104.4a), each read in its log.

Unproved. The `actor` tier's torch part was skipped, as it is without `$ROBOMAGE_TORCH_PYTHON`, and `robomage_nn` was not rebuilt; the acceptance was run by hand, as above. No game of the fixture was read for how often a rider's mana reaches a creature: 3 to 6 hits a hundred games, by scenario otherwise. A Sunburst creature and a rider with another duration have no card. No candidate entered but these two: the parse report over the 330 candidate cards leaves 44 outside the vocab, none at zero.

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

Fuse is also where split cards get audited against CR 709, since Dead/Gone and Rough/Tumble entered
without it: combined characteristics outside the stack (709.4, mana value and colors from the combined
cost, 709.4b), each half's name for name-a-card (709.4a), and the cast half's own name on the stack
(709.3b: Tumble's damage logs as "Rough deals" today). Unchecked so far, and not in
`docs/rules_deviations.md`.

**Test.** Every existing cast mode through the corpus. New: an Adventure exiled and cast later, and
countered (goes to the graveyard); Bestow falling off into a creature (CR 702.103f); Overload with hexproof
objects; a permission lost mid-cast; `ReplaceGraveyard` when the spell is countered. Top-of-library casting
across a shuffle. The state format's how-cast enum extended.

**Progress (2026-11-06).** Unit (a), the inert migration. Nothing about play changed.

- **`spell_destination(spell, proposed, exit)`** (`src/queries/spells.h`) is the one answer to where a spell goes off the stack: the resolve path's if-chain and `remove_from_stack`'s repeat read it, with `SpellExit` saying which it is.
- **`Permission`** (`src/queries/permissions.{h,cpp}`) is `kPermissionRows`: a kind, the zone it reaches, whether only the owner has it, what covers the card, when it lapses, what using it spends. `play_permissions` walks the rows of the card's zone and looks at no other zone, so a Library row is one line. Its values are the state format's `permission_kind`.
- **`CastOption`** (`src/components/cast_option.h`): face, zone, `CastCost`, permission, Offspring, destination and its row's name. `LegalAction::option` and `ChosenCost::option` replace eight booleans of the one and the mode and four booleans of the other. `CastCost` takes in `ChosenCost::Mode`, so evoke, impending and warp come from the row and nothing works them out as the spell becomes cast; the NORMAL grant is `MANA_COST` on an exile row, and nothing looks the grant up to tell. A PLAY_LAND row holds its face there too.
- **How the cast proceeds:** the COST step is `kCastCostRows`, a row per `CastCost`: what it owes and the next step. `mana_cost_replaced` is `cost != MANA_COST`.
- **The cast table** (`src/cast_options.{h,cpp}`): `kCastSections`, each a zone, permission and face with its rows, walked by `offer_cast_options` a section at a time and by `enumerate_cast_options` a card at a time, which is what keeps the menu's order. A row is a name, a cost, Offspring, ordinal, label, destination and the gate that offers it; the alternative kinds are rows. No cast row is built anywhere else: a miracle's and a resolution's cast from exile are rows too (`kMiracleCast`, `kResolutionGrantCast`), offered by `cast_during_resolution`, not walked. Rows share `can_cast_now` and the timing through `CastOffer`, worked out once per card.
- **Coverage:** `cast_offer` and `cast_row` count the rows. The `cast` keys, and every `.expect`, stay.

What stayed, and why:

- **`SpellOverlay` isn't defined.** An empty type is dead weight here: nothing could read it, and (b) decides what it holds. It is a column of `CastRow` and a field of `CastOption` when it has a kind.
- **Land rows from the hand and the graveyard** stay in `determine_legal_actions`: no land is cast, and they come before the companion's row. A land from exile, and suspend, sit among the casts in the menu, so they are their sections' `beside` hooks.
- **`free_cast_grant`** stays in `rules_mod`: it says what a cast costs, not where it may be cast from. **`may_cast_this_turn` and `impulse_cast_permission`** stay as they are, read and spent through the rows.
- **`PAY_COSTS`** still fills `CostPayer::alternative`, `flashback` and `escape` one by one: `cost_payment.cpp`'s pick tables name those slots.
- **`exile_resource` and the `*_offspring` rows nothing reaches** stay, as the loops offered them. Removing one would be a change.
- **`CLAUDE.md`'s key-file index** doesn't list the two new files: this unit was not told to edit it.

**(b) must:**

- Define `SpellOverlay` on `CastRow` and `CastOption`, with `operator==` (`interchangeable_choices` compares options) and its reader through `active_face`.
- Library: a row of `kPermissionRows` and a section. A new `Permission` goes before `NONE` and appends to `permission_kind`, and `encode.cpp`'s loop ends at `EXILE_GRANT`. `Orderer::begin_cast_move` is fatal on a library cast.
- `destination` applies at every exit. Adventure's holds only at `SpellExit::RESOLVED` (countered, it goes to the graveyard), and `ReplaceGraveyard` belongs to a resolution, so both need `spell_destination` to read more than the field. `resolve_top` logs " (flashback)" for any `by_cast`.
- `spell_destination` tries `shuffle_into_library` first; flashback says exile instead of anywhere else (CR 702.34a). A bug by order, with no script having both. Swap them with the first card that does.
- `DB$ Play` over any zone: `kResolutionGrantCast` is exile's, `effects::play` stops anywhere else, and `impulse_cast_permission` is the only store a cast reads its resource from.
- A new `CastCost` is a row of `kCastCostRows`, in order: its `static_assert` names `WARP` as the last, and `alternative()` is `>= ALTERNATIVE`. Fuse needs a third `CastFace` past `back_face()`.
- `cast_variant_mask` and the stack row's variant read the option field by field: the how-cast enum extends there, with `make registry`.
- A permission lost mid-cast: the row is decided at the offer, and `determine_grant_cost` reads the grant after the card has moved.
- Row gates must stay free of side effects: a walk skips them by short circuit and shares `CastOffer`'s answers. Convoke, Bargain and `ManaRestriction` (M5's and M6's tables) are `TapToPay` and cost rows beside this.

Evidence. Against `ab4c992d`, built apart, the tip's debug binary: 9,600 explore games under `--narrative --trace-decisions` over the 23 top-level, `league/`, `heldout/` and `tests/mech_*` decks, 3,200 each of bo1, `explore:patient` bo3 and `--no-auto-mana`, 5.4 GB byte-identical; `--rmstate-dump` frames of 300 games, 2.1 GB, identical; 120 random games under `--offer-cancel` and 240 under `--selfcheck-state`, identical, no error. Each run's old `--coverage` keys identical too. All 410 scenarios run through both binaries: output identical. `--parse-report @vocab`, over 430 cards with the candidates, and `--cards-frame`: identical. `make registry` a no-op; these commits touch nothing in `tests/` or `src/featurize`, so no `make regen`. Each commit: `make check` green and 560 games identical. Release `bench`, median of 9 alternating runs: engine alone 137.0 us per decision before, 132.8 after; the actor loop 2,445 decisions/s before, 2,491 after, and neither pair's ranges overlap. Likely why, not measured: the offer no longer builds a row it won't offer. The `actor` tier's torch part, run by hand on `robomage_nn` rebuilt at the tip: green.

How often each mode was reached, over those 10,260 games:

| row | offered | cast | row | offered | cast |
|---|---|---|---|---|---|
| `hand` | 676,985 | 278,110 | `flashback` | 8,944 | 3,890 |
| `alternative` | 39,135 | 27,475 | `escape` | 1,676 | 594 |
| `evoke` | 6,403 | 3,711 | `exile_mana_cost` | 1,922 | 698 |
| `warp` | 3,627 | 1,548 | `exile_no_cost` | 86 | 65 |
| `impending` | 1,410 | 616 | `exile_free` | 22 | 18 |
| `spectacle` | 756 | 374 | `exile_free_offspring` | 2 | 1 |
| `offspring` | 109 | 17 | `exile_resolution` (free 428, energy 334) | at resolution | 697 |
| `hand_free` | 2,541 | 1,313 | `miracle` | at resolution | 78 |
| `hand_free_offspring` | 36 | 14 | `exile_land` | 486 | 254 |
| `back_face` (split halves only) | 1,099 | 439 | `graveyard_land` | | 1,150 |

Beside them: 704 suspends, 437 companions, 560 land back faces.

No game reached these. `graveyard_cast`: Emry is in no deck, and a `--play` line through both binaries is identical. `back_face_free`: the same, Gone under Omniscience. With no vocab card or script to reach them, so unproved beyond the code: a modal DFC's nonland back face (`back_face`'s other half; Witch Enchanter's back is a land), `back_face_free_offspring`, `graveyard_free`, `graveyard_free_offspring` (no free-cast static covers the graveyard, no Offspring card is an artifact), `exile_resource` and a grant's life (nothing grants energy or life at priority, and no `PlayCost$` pays life).

**Progress (2026-11-06).** Unit (b), spells that become something else when cast.

- **`SpellOverlay`** (`src/components/cast_option.h`) is a column of `CastRow` and a field of `CastOption`, compared by `operator==`. A kind is a row of `kSpellOverlayRows` (`src/queries/overlays.{h,cpp}`): its name, its `cast_variant` bit, where it goes as it finishes resolving, the face it builds at parse (`has`, `change`), what follows its resolution, and whether the permanent keeps it or the object is a copy. `CardData::overlay_faces` holds the built faces, and **`cast_face(card, option)`** is the one reader, through `active_face`: a new overlay is a kind, a row, and a row of the cast table. A derived definition names its printed one (`AbilityDef::printed_as`), so a frame's ability reference survives.
- **Overload** (`overload_spell`): `ValidTgts$` reads as `Defined$ Valid`, no target, so hexproof doesn't matter and ward never triggers. **Fuse:** `CastFace::BOTH` reads `CardData::combined`, a split card's halves as one face, built at parse for every split card.
- **Split cards against CR 709.** 709.3b: `entity_name` and the stack's logs read the active face. 709.4, 709.4b: `card_characteristics` is the filter's and `effective_colors`' reader, the cast face on the stack and both halves combined anywhere else; Wear/Tear is white and red in a hand, Tear white on the stack, and Blue Elemental Blast, Solitude's pitch and the object rows agree. 709.4a: `card_names`, a row per name, and an object has either. 709.3: a grant's rows cover the back face from exile.
- **Adventure:** its own layout, `SpellOverlay::ADVENTURE` on a back-face section. `spell_destination` reads the overlay's `resolved_to` only at `SpellExit::RESOLVED`, and a stack object that fizzled exits as `REMOVED` now (`ResolutionFrame::fizzled`), so a countered or targetless Adventure goes to the graveyard. `on_an_adventure` leaves the exile grant, which lasts while it stays exiled. Another effect's mana-cost grant may cast it as an Adventure again. `S:Mode$ CantBlockBy` is a general static, `rules_mod::block_prohibited` over `ValidAttacker$` and `ValidBlocker$`: M8's blocking still has its other hard-coded checks to absorb.
- **Bestow:** an alternative cost whose face is an Aura, kept on `Permanent::overlay` through `EntryInfo`. `aura_spell_target_illegal` reads the active face; an illegal target ends the overlay and it resolves as a creature (CR 702.103e); `ends_overlay` is the state-based exception to 704.5m (702.103f, g), summoning sick the turn it entered. `CopyPermanent` reads `Defined$ Enchanted`, `UnlessCost$`, `UnlessSwitched$`, `UnlessResolveSubs$ WhenNotPaid`, the offer `run_unless_loop` in its own handler; a filter's `AttachedTo <filter>` runs to the next `+`.
- **Prepare** (`src/queries/prepared.{h,cpp}`): `K:ETBReplacement` reads the `AlterAttribute` SVar; `sync_prepared_copies`, with the state-based pass, makes the copy in exile (a `PreparedCopy` component, the prepare spell's `CardData` alone) and removes it with its permanent (phased out too, and made again as it phases in). `Permission::PREPARED` and a section with `SpellOverlay::PREPARE`: a copy, not a card, so it ceases to exist off the stack; `prepared_copy_cast` takes the designation as the spell becomes cast. To a frame it is the preparation card's back face, token-flagged, so the sidecar holds no unit for it.
- **Also:** a stack object fizzles only with every target illegal (CR 608.2b): a shared fix, for Fuse. Massacre's `CheckSecondSVar$`. `cast_variant` is 32 bits, with OVERLOAD, FUSED, ADVENTURE, BESTOW, PREPARED; `layout` ADVENTURE, PREPARE; `permission_kind` PREPARED; object columns `bestowed`, `prepared`. `cast` keys `overload`, `fused`, `adventure`, `bestow`, `prepared`.

What stayed, and why:

- **`CastCost` has no new row.** Overload and Bestow ride `ALTERNATIVE` and the card's one `AltCost`: a card with both, or with either and another alternative cost, has none provisioned.
- **`spell_destination`'s order:** no card has both a shuffle and a flashback, so the swap waits with its note.
- **A resolution's cast** (`kResolutionGrantCast`, Amped Raptor, suspend) offers the front: it is `cast_during_resolution`'s, and (c) reworks it. A free or resource grant offers no Adventure.
- **Overload and Bestow from another zone:** rows of the hand, as every alternative cost is.
- **`CantBlockBy`** arrived ahead of M8 only for Brazen Borrower. **`UnlessCost$`** isn't a step of `resolve_ability`: `CopyPermanent` is one more handler with its own offer, and `Token`'s (Acererak, M12) would be the next.

**(c) must:**

- Library: a row of `kPermissionRows` and a section, before `NONE` and appended to `permission_kind` after `PREPARED`; `encode.cpp`'s loop ends at `PREPARED`; `Orderer::begin_cast_move` is fatal on a library cast.
- `DB$ Play` over any zone: `kResolutionGrantCast` is exile's and the front's, `effects::play` stops anywhere else, `impulse_cast_permission` is the only store a cast reads its resource from. A split card or an Adventure cast there needs the back rows the priority sections have.
- `ReplaceGraveyard` in `spell_destination`: it belongs to a resolution, and the hook reads the overlay and the option now, at both exits.
- A permission lost mid-cast: the row is decided at the offer, and `determine_grant_cost` reads the grant after the card has moved.
- Convoke, Bargain, `ManaRestriction`: `TapToPay` and cost rows, as (a) left them.

Evidence. `c825d039`, the inert commit: `replay` byte-identical, `make registry` a no-op, `--parse-report @vocab` unchanged. Every fix fails with it removed, by reverting the file or mutating the line: the half's name, the combined colors, the half's name for name-a-card, the second half from exile, the second condition, the chained target, `SpellExit` at a counter and a fizzle, `block_prohibited`, the overlay's end at resolution and in the state-based pass, the paid `UnlessResolveSubs$`, `AttachedTo`'s filter, the prepared designation and the copy leaving with its permanent. 46 scenario files, each card and each ability; a scenario that could only look at the log was given a line that tells the two apart. The corpus moved once, `eb1a31cc`: one `.replay` menu of `lg_burn__wrb_energy` gains `Name card: Gone`, its decision log unchanged. `tests/cli` never moved. `decks/tests/mech_cast` is the fixture, every key hit at 16 games under three seeds: `--fuzz-games 100` in `explore` and `explore:patient`, no error, no draw, `cast:fused` the rarest key at 19 and 20. Its name put it first among the fixtures, and fixture k's seeds moved every other's: `mech_mana` lost its rare key's one hit and `mech_clean_cards` met a 35,000-decision Basalt Monolith game. A fixture's seeds come from its name now. 120 games at `--selfcheck-knowledge --selfcheck-state-sample 1`, a third each of `explore`, `--no-auto-mana` and random under `--offer-cancel`: no error; twelve explore games of them, counted, reached every overlay (adventure 39, prepared 38, bestow 16, overload 11, fused 2). `scripted` and Python `explore`: six bo1 games each, `scripted` bo3 three matches. No new prompt: the rows are `CAST_SPELL`, the target `SELECT_TARGET`, the offer `PAY_UNLESS`, and `state`'s snapshots run through each scenario. `play_spec`'s `name:` reads the row, since a split card's names ground to one card. The debug `bench` tier: engine alone 1605.6 us per decision before the unit, 1686.5 after, the actor loop 245 decisions/s and 241.

Unproved. Release `bench` wasn't compared; the debug numbers are one run each. A token copy of a preparation card (`Token::prepare_spell`, Nantuko copying a Cultivator) and a prepared permanent phasing out and in are code with no scenario. The `actor` tier's torch part was skipped, for want of `$ROBOMAGE_TORCH_PYTHON`. Reading a modal back face's characteristics on the stack through `card_characteristics` has no vocab card. A copy of an overloaded or fused spell has no card to make one.

**Done (2026-11-06).** Unit (c), and with it M7.

- **A resolution's cast is rows.** `CastOption::grant` holds what an effect puts in place of the mana cost, set as the row is offered, so `determine_grant_cost` reads the row and `during_resolution` goes. `kResolutionCasts`, walked by `resolution_cast_rows` on `ResolutionCastTerms`, has for exile and the graveyard each the front, the back and the Adventure, plus a free-cast static's row from the graveyard; `cast_during_resolution` takes a list of offers, so one menu holds a card's faces and a zone's cards. `offer_timing_ok` answers true for a resolution. A menu row carries only the face and overlay: the frames of a front-face offer don't move.
- **`DB$ Play`** reads `PlayParams`: `ValidZone$`, `Valid$`, `ValidSA$` (a filter of the face), `PlayCost$`, `WithoutManaCost$`, `Condition$ Bargain`, `ReplaceGraveyard$`, `ReplaceGraveyardValid$`, `Amount$ All`. `ImprintPlayed$` is a memory row reading `EffectObjects::PLAYED`. `play_cost` amounts are the mana value of the face cast (`amount_is_mana_value`).
- **`ReplaceGraveyard$`:** `CastOption::cast_by` names the effect, and `spell_destination` reads it at both exits. `cast_variant` gains `GRAVEYARD_REPLACED`, for the frames. A spell it exiles when countered is logged by `remove_from_stack`.
- **Convoke:** `TapToPay::CONVOKE`, `GENERIC_OR_OWN_COLOR`, and `asks` on the rule. The cast's `TAP_TO_PAY` step comes after the snapshot and asks a creature at a time, each row leaving the rest payable; then mana pays what is left, with nothing more tapped. Improvise stays the payer's, so nothing moved. Rows are `PAYING_COSTS`: explore draws among them, `scripted` takes the first, the CLI and `--trace-decisions` render them, and the `--play` verb is `convoke:<card>` / `convoke:done`. In the simulation a colored pip takes the creature of fewest colors, which is exact for the two colors of the one card.
- **Bargain:** a cast step for the intention, a pick for the sacrifice. `spell_bargained` reads `ChosenCost::moved` for a spell with the keyword; `cast_variant` gains `BARGAINED`.
- **`S:Mode$ ManaRestriction`:** `spell_refuses_mana` checks each unit's `source` against `ValidSource$` in `unit_spendable_on` and `mana_source_usable_for`, and a simulated payment stamps `source`. A source gone from the game counts as matching: `Card,Emblem` is every object that makes mana.
- **`Permission::GRAVEYARD_SELF`,** a row of `kPermissionRows` and a section, appended to `permission_kind`: a card whose own static lets its owner cast it from the graveyard. `static_form:MayPlay.Graveyard` for `Card.Self` and `EffectZone$ Graveyard` close with it; Gaea's Will's Effect form stays.
- **`wasCastFromYourHand`** reads `MatchCtx::cast_from`: `effective_base_cost` and `floored_alt_mana_cost` take the zone.

**A permission lost mid-cast.** Decided: the row stands. CR 601.3 checks the permission as casting begins, and CR 601.5 again after the proposal (601.2a-d), but no player acts between them, mana abilities and costs come at 601.2f-h, and "it doesn't matter if a rule or effect would make the casting of the spell illegal while determining and paying that spell's costs". So nothing the engine models can lose a permission where it counts. Omniscience sacrificed to Beseech the Mirror's bargain leaves that cast free (`cr601_5_...`); a prepared permanent sacrificed while its copy is paid for is `prepared_copy_cast`'s null case, which reads the same. Nothing to fix but the read after the move, which the row replaced.

Bugs, each with a scenario:

| bug | rule | vocab cards it changes |
|---|---|---|
| A reduction missed an alternative cost | CR 118.9d | none but under Bilbo: Eye of Ugin and It That Heralds the End reach no card with one |
| `ConditionDefined$ Imprinted` read the remembered set, and the `Defined$ Remembered` move reported nothing, so `Imprint$ True` wrote nothing | CR 608.2c | Phelia, Exuberant Shepherd: same outcome, her scenarios hold before and after |
| Delve in a simulated payment counted the spell itself in its graveyard | CR 601.2a | none but Hogaak |
| Amped Raptor's energy was the card's mana value, a split card's two halves together | CR 709.3 | Amped Raptor over Dead/Gone, Wear/Tear, Rough/Tumble, Brazen Borrower |

Found by the gate before any commit: `spell_bargained` read every sacrifice, and `open_vocab` caught Natural Order's frames moving.

What stayed, and why:

- **An alternative cost at a resolution, Overload and Bestow from another zone:** `docs/rules_deviations.md` §2.5. Bilbo casts Cyclonic Rift for its mana cost.
- **Improvise** taps only for what mana can't pay, as (c2) left it: asking would move `ur_delver`'s frames for a choice only `--no-auto-mana` needs.
- **`Tap nothing more`,** Convoke's stop row, has no card: Hogaak takes no mana, so the row never shows. A throwaway build with the restriction off showed it: two Bears, seven Forests, one tap, the row, six Forests paying the rest.
- **`spell_destination`'s order:** still no card with both a shuffle and a flashback.
- **`Amount$ All`** reads one card, and `Condition$` only `Bargain`.
- **`CheckSecondSVar$`** closed in (b). Nothing left needs it.

**Lands with its first card, not here.** Mystic Forge is the only provisioned script that casts from a library, and it needs M11's `MayLookAt$` and the `TopLibrary` qualifier, so none of this was built.

| item | first testable card | stage | what it must build |
|---|---|---|---|
| Library: a row of `kPermissionRows` and a section | Mystic Forge | M11 | before `NONE`, appended to `permission_kind` after `GRAVEYARD_SELF`; `encode.cpp`'s loop ends at `GRAVEYARD_SELF` |
| `Orderer::begin_cast_move` from a library | Mystic Forge | M11 | it is fatal today; `rewind_cast_move` reopens only graveyard and exile slots, and the card's handle is hidden: what each seat sees on a rewind |
| `static_form:MayPlay.Library`, `TopLibrary` | Mystic Forge | M11 | the top card read live, so the permission moves with a shuffle: the plan's top-of-library test |
| An Effect's `MayPlay$` over a graveyard | Gaea's Will | M9 | beside its `Moved` replacement |

Evidence. `40d7d5a7` and `5ee8495c`, the inert commits: 494 and 152 explore games under `--narrative --trace-decisions` identical to the prior binary, the parse report unchanged. Every fix failed with its line removed, by mutating the file and rerunning its scenarios: the half's mana value, the back rows, `wasCastFromYourHand`, `ReplaceGraveyard$` and its filter, `ValidSA$`, `Valid$`, a resolution's timing, the reduction on an alternative cost, the imprinted read, `Condition$ Bargain`, the bargain's filter, the unit's refusal, Delve's own card, each pip rule, `owner_only`, `asks`. The source check in `mana_source_usable_for` alone isn't proved by a scenario: the unit's check catches the same mana. 36 scenario files, naming each card and each ability; a floating {G} and a cancelled Convoke cast among them, and the `state` tier runs every one at sample 1, so each parks on the new rows. `tests/corpus` and `tests/cli` never moved, so no `make regen`. `decks/tests/mech_cast_zones` is the fixture: every key hit at 16 games under four seeds, and `--fuzz-games 100` in `explore` and `explore:patient`, no error, no draw, `cast_row:graveyard_resolution_back` the rarest at 7 and 2, so it stays out of `.expect`. 120 games at `--selfcheck-knowledge --selfcheck-state-sample 1`, a third each of `explore`, `--no-auto-mana` and random under `--offer-cancel`: no error; twelve of the explore games, counted, reached every row but `graveyard_self` (bargained 13, exile 3, graveyard 4, Adventure 2, back 2, Convoke taps 10, replaced 5). `scripted` six bo1 games and three bo3 matches, Python `explore` six games. `make registry` appended the two keywords, `cast_variant`'s two bits and `permission_kind`'s value, and the digest added three cards; no known card's rows moved. The debug `bench` tier, one run each: 225 decisions/s engine alone before the unit, 229 after, the actor loop 243 and 251. The candidates' status files are refreshed.

Unproved. The `actor` tier's torch part was skipped and `robomage_nn` not rebuilt. `Tap nothing more` has no card, only the throwaway. A prepared permanent sacrificed while its copy is paid for has no scenario. A Treasure's mana on Hogaak (a source gone from the game) has no scenario. A copy of a bargained spell has no card to make one. A Beseech cast rewound under `--offer-cancel` went through random games with no error, but it was not read that the card, public for the cast, comes back face down with its owner's look lost (§2.6a's reveal note).

**Stage gate (2026-11-06), at `173630f2`.** `make check` 0 errors, 0 warnings. Release `bench`, seven alternating rounds against M6's `81f3b905`, medians: the actor loop 2,453 to 2,418 decisions/s (-1.4%), engine alone 137.7 to 140.7 us per decision (+2.2%). Acceptance `gate`: 4 networks, 614 decisions. `full`, its first pass since the pin bump: 35 networks, 8,725 decisions, 18 cut-over files equal, worst |Δ value| 1.2e-5, |Δ logit| 3.1e-5 against a tolerance of 1e-4.

`full` had failed every network since `ec57222a`, on ten decisions `gate`'s sources don't hold, and the failing log went unread. Both causes moved on purpose, so the ten left `tests/open_vocab/inputs.json`, and `full`'s menu check reads only the decisions the file holds:

- Rows 2584 and 2597, `lg_burn__wrb_energy`: the menu gains `Name card: Gone`, 17 rows to 18 (`eb1a31cc`).
- Rows 5161, 5162, 6568, 6593, 7292, 7299, 7345, 7346, each with a Voice of Victory trigger on the stack. Frames against S0's dumps differ in `object_subtype` alone, Soldier to Bard. An ability on the stack holds no subtype rows, so the featurizer's object delta reads the card's printed subtypes as lost, and a `Bard` token reaches the network from the frame, which `s0_rows.json` cannot patch. Putting `Soldier` back in those tokens took three networks from |Δ value| 0.13 and |Δ logit| 0.04 to under 6e-6.

That delta on a stack ability is noise: an ability has no subtypes to lose. Every checkpoint trained with it, so it stays.

**League consolidation (2026-11-06), after the stage gate.** The newer list of each near copy replaces the older: `ur_tempo`, `lands_top8` and `death_and_taxes` are designated into `league/` (`onboard.py` stages 7 to 10; the campaign 1,443 games each, repeated over the final league, no error and one draw, reviewed: Archon of Cruelty's trigger takes `lands_top8` to -1 life and draws from reanimator's empty library in one resolution, a true CR 104.4a draw; stage 11's baseline waits for a policy run), and `ur_delver`, `lands`, `bw_dnt` and `gw_maverick` are deleted. `maverick` joins after M9. The league is ten decks: bug, burn, death_and_taxes, eldrazi_aggro, lands_top8, reanimator, ur_tempo, uw_control, wrb_energy, wubg_doomsday, which `LEAGUE` now names, so `eldrazi_aggro` is a default deck and gw_maverick's references went to it. The corpus has 56 games, its league part nine decks without `eldrazi_aggro` as before. The guides cover nine decks, the three replacements' sections rewritten and not yet reviewed (`docs/sideboard_guides.md`).

### M8. A characteristics view across zones (L)

**Today.** Continuous effects come from battlefield permanents and emblems and mostly reach the
battlefield. Only `SetColor` reaches other zones. No `effective_types` exists; layer 4 writes
`Permanent.types`. `static_targets` ignores `AffectedZone$`, and five readers bypass it. The filter's
`card_view` reads printed colours. Blocking is hard-coded but for `CantBlockBy`, which M7 (b) brought for Brazen Borrower.

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

**Lands here.** Painter's Servant, Grist off the battlefield, Agatha's grants, Lazotep
Quarry, Fable's copies, The Everflowing Well, Allosaurus Shepherd.

**Test.** Painter's Servant: colour in hand, library and on the stack, and protection or Blue Elemental
Blast style targeting reading it; that colour in a hidden zone does not leak through a frame. Grist as a
creature card in the graveyard for reanimation filters. Copy of a copy.

**Progress (2026-11-06).** Unit (a), the inert migration. Nothing here changes what the engine does; (b) extends these seams.

- **`ObjectCharacteristics`** (`src/queries/characteristics.h`): types, colors, P/T. `permanent_characteristics` reads what the layers left, `card_object_characteristics` a card where it is, `printed_characteristics` the printed face for a static's own filter; the filter's views read them. **`effective_types`** and **`object_abilities`** answer in any zone. None reads last-known information, unlike `effective_colors`: a card in a new zone is a new object (CR 400.7), and a graveyard count taken in the resolution that killed an animated land must not read Creature.
- **Static scope.** `static_reaches` (`StaticRead::LIVE`, `CARD`, `PRINTED`) and `static_reaches_face` ask about one object; `static_targets` enumerates. The plan's five readers were `static_defined_covers`' five callers: `setcolor_override_for`, `land_type_setter_affects`, `removal_affects`, `entering_object_loses_abilities`, `free_cast_grant`. A sixth bypassed it too, `ability_grant_targets`: its `NamedCard` reading is `affected_permanents_for_static`'s now. Every read of `Affected$`, `AffectedDefined$` and `AffectedZone$` outside the parser sits in one block of `state_manager_statics.cpp`.
- **`CopyOverlay`** (`src/components/copy_overlay.h`, on `AbilityDef`): Offspring's 1/1 and `GainThisAbility$`. `src/queries/copies.h` has `copied_token`, `become_copy`, `end_copy_effect`; a new exception is a field and a line of `apply_copy_overlay`.
- **`AnimateAll`** goes through `animate_permanent`. `changes_keywords_alone` stops it there while it names no type or P/T, so Shadowspear takes no timestamp.
- **`kBlockRestrictions`,** a row each: can't be blocked this turn, landwalk, shadow, flying and reach, protection, `CantBlockBy`. Menace is `minimum_blockers`: it restricts the declaration, and no pair of blocker and attacker shows it.

What stayed, and why:

- **`AffectedZone$` on a `LIVE` read** stays unread, as today. Vocab statics show the change is safe, but it is (b)'s.
- **`Affected$` under `AffectedDefined$`** restricts the object in `static_reaches` and not in `static_targets`, which reads the named object alone: Kaito's `Permanent.counters_GE1_LOYALTY`, equipment's `Creature`. Unifying them is a behaviour question.
- **`card_view` of a battlefield permanent** reads it as a card: `card_matches_filter` on one is printed, as it was.
- **`effective_power` and `effective_toughness`** read the front face off the battlefield, not `card_characteristics`: Petty Theft on the stack is 3/1 there.
- **A spell's front-face readers:** `trigger_valid_card_non_creature` and `src/mana_system.cpp`'s restrictions. `effective_types` answers differently for an Adventure or a modal back face (`docs/rules_deviations.md` §2.5), so that is a fix of its own.
- **`ChangeZoneAll StaticEffect$`** has no implementation, and is no copy: Ghost Vacuum's static adds a type and sets P/T on the cards moved, so it belongs with `Animate`, not `CopyOverlay`.
- **Clone** copies a nontoken permanent's whole `CardData`, front face up, and `copied_token` the face it shows. Each kept its reading.
- **Cards scanned as a pool** (`Orderer`'s and Cavern's subtype lists) read printed types on purpose.

**(b) must:**

- `AddColor$`: beside `setcolor_override_for`, read by `card_object_characteristics` and `effective_colors`. `encode.cpp` writes `card_colors(face)` in the `CARDS` frame and `effective_colors` per object, so a color in a hidden zone is the leak to test.
- A characteristic-defining static off the battlefield: `g_active_statics` and `ActiveStatic::sa()` hold battlefield and emblem sources alone. Grist can be a face built at parse that `card_characteristics` returns off the battlefield, which every reader here takes up; `effective_types` returns a reference, so anything computed per object needs another shape.
- `ExcludeZone$`, and `AffectedZone$` on `LIVE`: `object_in_affected_zone`, one line each.
- `GainsAbilitiesOf$`: `object_abilities` over the exiled cards, in `apply_layer6_ability_grants`, whose bodies have no SVars today.
- `CopyOverlay` fields: `AddKeywords$`, `SetPower$`, `SetToughness$`, `SetColor$`, `SetCreatureTypes$`, `AddTypes$`; `CopyPermanent`'s `ValidTgts$` and `Defined$ Remembered` (`copied_token` takes a card in any zone), `AtEOT$`.
- `Clone`'s `Duration$`: `end_copy_effect` at cleanup, and `Defined$` / `CloneTarget$` the other way round from Thespian's Stage. Copying a token: `become_copy` needs a `CardData`, which a token hasn't got.
- `K:ETBReplacement:Copy`: `become_copy` as it enters, its replacement picking the original.
- An `Animate` of end-of-turn duration: `animate_permanent` skips `animate_has_pt` and `Keywords$` for one, which Allosaurus Shepherd and Shallow Grave need, and the parser reads no `AnimateAll` `Power$`, `Toughness$`, `Types$`.
- `HIDDEN CARDNAME can't block.`: a row of `kBlockRestrictions`.

Evidence. Each of the five commits: 509 explore games (bo1, 15 bo3, `--no-auto-mana`, every league pair, the held-out decks, each `mech_` fixture; 3.1 million lines of `--narrative --trace-decisions`, capped at 2,500 decisions) hash for hash the same as `92ab1ff1`'s binary, which reran identically. `replay` byte-identical, `make registry` a no-op, `--parse-report @vocab` the same bytes, `tests/corpus` and `tests/cli` unmoved. Neither landwalk nor menace has a scenario, so each was compared by hand on both binaries: Street Wraith against Swamp and Forest, and Alpha Deathclaw and Sire of Seven Deaths over three Bears, 80 games and 24 menace releases, no difference.

Unproved. Release `bench` wasn't compared. `static_reaches` adds a call per permanent per general static, and `affected_permanents_for_static` builds a `FilterSpec` per call for Petrified Hamlet. Several unifications hold for the vocab and token scripts alone, by reading them: an empty `Affected$`, `AffectedZone$` read for an entering card, a Self `AddAbility$`, `effective_types` of split halves that differ. The `actor` tier's torch part was skipped.

**Progress (2026-11-06).** Unit (b1), characteristics across zones. Frenzied Baloth 442, Painter's Servant 443, Grist, the Hunger Tide 444 with the Insect token 922, Agatha's Soul Cauldron 445.

- **Static scope.** `object_in_affected_zone` is asked on every read, `LIVE` too, and of the objects a static names; `ExcludeZone$` alone reaches every other zone.
- **A spell's types.** The Fantasticar's fourth-noncreature trigger and the spending restrictions of `src/mana_system.cpp` read `effective_types`, and an offer the face it offers (`ManaPayee::face`). `effective_power` and `effective_toughness` read `card_characteristics` off the battlefield.
- **Layer 5.** `apply_color_statics` replaces `setcolor_override_for`: `SetColor$` and `AddColor$` alike, in timestamp order, over what the object has before the layer, in any zone; last-known colors read the same. `CHOOSE_COLOR` is a prompt of its own, parked as the choose-type site is. `Permanent::chosen_color` is the `object` table's one new column, a count (1 + the color): an enum column moved all 620 fixture decisions' tensors, a scalar none.
- **Characteristic-defining abilities.** `CardData::off_battlefield` is a face built at parse (`build_overlay_faces`), which `card_characteristics` returns off the battlefield and `off_battlefield_face` for a bare face being cast or named. A counted P/T can't sit on a face: `card_power_toughness` reads it, and the filter only when a qualifier asks (`view_pt`), since the count filters the card itself. No static needed an off-battlefield source in `g_active_statics`: a CDA affects its own object, so its card is read directly.
- **`ImmediateTrigger`.** `UnlessCost$ Sac<1/…>` is `UnlessPayKind::SACRIFICE` through `run_unless_loop`: one menu of what can go and the decline, nothing asked with none. Its battlefield scan is gone: `resolve_ability` reads `ConditionPresent$`, over what `ConditionDefined$` names, before the handler runs.
- **`Repeat` with no `MaxRepeat$`** needs no bound for CR 732 on Grist: each pass mills a card, and an empty library mills none. `cr732_grist_stops_at_an_empty_library`.
- **`GainsAbilitiesOf$`.** `grant_abilities_of`, in `apply_layer6_ability_grants`: `object_abilities` over the cards the filter names, each the definition its own script gave it, so its SVars come with it, through the path `AddAbility$` uses (`grant_ability`). A frame's `object_ability` row names the card lent from (`Ability::granted_from`). `ExiledWithSource` is a filter row now. A creature can hold Grist's loyalty abilities, and pay +1.
- **`ManaConvert`.** `any_mana_as_any_color_active` takes the payee; `ValidPlayer$ You`, `ValidCard$`, `ValidSA$ Activated` scope it. A `counters_` qualifier in `Affected$` gates the source only when the static names the source.
- **`CantPreventDamage`:** `rules_mod::damage_cant_be_prevented`, ahead of every prevention in `damage_prevented`, shields and protection alike.
- **Fixture:** `decks/tests/mech_characteristics.dk`, keys `category:CHOOSE_COLOR`, `row:add_color`, `row:off_battlefield_face`, `cost:SACRIFICE.unless`, `trigger:Reflexive`, `effect:Repeat`, `row:gains_abilities_of`, `row:mana_convert_scoped`, `row:cant_prevent_damage`.

Bugs. CR 709.3, 712.8f, 715.3b: a spell's front face read by the two readers above, so Petty Theft was a creature spell for The Fantasticar and Abundant Countryside, Cavern of Souls, Eldrazi Temple and a Powerstone's mana. CR 608.2h: a permanent's last-known colors left out layer 5, which only Mycosynth Lattice leaving at once could show; fixed in the Servant's commit, with no scenario of its own. No known card's table rows drifted, and the corpus didn't move.

What stayed, and why:

- **A frame's P/T of a counted CDA off the battlefield** stays printed (`docs/rules_deviations.md` §2.5): every rule reads the count, and showing it moves 165 of the fixture decisions, which is an acceptance run's to drop. `encode.cpp` would call `card_power_toughness`.
- **`AddAbility$` bodies** still parse with no SVars: no provisioned script's body names one, so nothing can test it. A grant's first card with one takes it (M13 at the latest).
- **A token copy** makes no "as this enters" choice (`docs/rules_deviations.md` §2.5).
- **`ConditionDefined$`** stays a set of booleans, not `Defined$` table rows: the row's aim, a read of the remembered cards and no battlefield scan, needed no more.
- **A sacrifice as an unless-cost** is one permanent, on `ImmediateTrigger` alone: Acererak's `DB$ Token` stays a finding for M12.

**(b2) still must:** the `CopyOverlay` fields, `Clone`'s `Duration$` and copying a token, `K:ETBReplacement:Copy`, an end-of-turn `Animate`'s P/T and keywords, `AnimateAll`'s `Power$`, `Toughness$`, `Types$`, `HIDDEN CARDNAME can't block.`, `RememberDiscarded$`, `filter:TopGraveyardCreature`, `filter:cmcEQX`, `Count$CastTotalManaSpent`, a mana ability's Sac, Return and energy.

Evidence. Each card by removal: the unit's scenarios fail without `damage_cant_be_prevented`, without `AddColor$`'s add (all ten), without the face or the count, without `grant_abilities_of`, `mana_convert_reaches`, `ev_exiled_with_source`. Hidden information: `cr400_2_a_color_added_in_a_hidden_zone_stays_hidden` (the dump's new `object` lines: the owner's hand shows the color, the opponent's frame no hidden object with one), every scenario at `--selfcheck-state-sample 1`, and 48 games of the fixture under `--selfcheck-knowledge --selfcheck-state-sample 1` (explore, `--no-auto-mana`, `explore:patient` bo3, random under `--offer-cancel`), no error. `--fuzz-games 100` on the fixture clean, no draw. `scripted` and Python `explore` over it, 32 games and matches, none drawn. `make registry`'s digest adds the four cards and the Insect and changes no other; `action_category` 50 values, `object` slot 89.

Unproved. Release `bench` wasn't compared. The `actor` tier's torch part was skipped. A granted mana ability has run under the fixture's fuzz but has no scenario of its own. Painter's Servant on a spell copy reads the copy's card; nothing tested it. `Spell.Creature+YouCtrl` on an opponent's spell (Frenzied Baloth's second line) rests on the caster check, not a scenario.

**Progress (2026-11-06).** Unit (b2), copies and end-of-turn animation. Allosaurus Shepherd 446, Shallow Grave 447, Fable of the Mirror-Breaker 448 with Reflection of Kiki-Jiki 449 and the Goblin Shaman token 923, Lazotep Quarry 450, Mockingbird 451, The Everflowing Well 452 with The Myriad Pools 453, Ghost Vacuum 454. **M8 is ready for its stage gate.**

- **`AtEOT$`.** `AtEotMove` on the definition, `register_at_eot_move` the one registration: Animate's `Sacrifice` and `Exile`, CopyPermanent's. The inert commit moved Sneak Attack's onto it.
- **An end-of-turn Animate.** `animate_set_pt_eot` beside `animate_set_pt`; `set_pt_from_animate` reads the later of the two into `Creature`'s 7b slot at each pass's rebuild, so a set P/T now sits over a characteristic-defining ability's 7a (Ghost Vacuum's Nethergoyf). `Keywords$` lands in the creature's until-end-of-turn keywords; `EntryInfo::animates` replays once the Creature exists. `AnimateAll` reads `Power$`, `Toughness$`, `Types$` through the same rows.
- **`CopyOverlay`:** `add_keywords`, `add_types`, `set_creature_types`, `set_colors`, and `SetPower$` / `SetToughness$` on the fields Offspring has: `read_copy_overlay_param` and a line each of `apply_copy_overlay`. `is_creature_subtype` is CR 205.3g-i inverted, so a Zombie copy keeps an Equipment's or a land's subtypes.
- **`copied_card`** is `copied_token`'s other half: the card values of a copy of a card or a token, the face up with its front's mana value (CR 707.8, 712.8e), a token's values as a card (`CardData::token_script` keeps which). `become_copy` and `enter_as_copy` both read it. `become_copy` takes a token, and a permanent's Creature follows at once: base P/T, or none.
- **`K:ETBReplacement:Copy`.** `CardData::enters_as_copy`, a Clone definition whose `Choices$` is the filter. `choose_copy_on_entry` parks before the Permanent exists (`SbeSite::ETB_COPY`, `CHOOSE_CARD`, the tapped-unless-life site's contract), so the Permanent is built from the copied values and "enters with" and enters triggers apply (CR 707.5). `EntryInfo::printed_card` carries the card to `Permanent::printed_card`.
- **`Clone`.** `CloneTarget$ Targeted` the other way round from the Stage; `Ability::cast_spell` holds the spell, read through `lki_entity()`, so a countered spell still gives its values. `end_copy_until_eot` at cleanup: `copy_eot_card` / `copy_eot_token` put back what was there, a Stage's lasting copy included.
- **A SpellCast's `ValidCard$ Permanent` and `ValidSA$ Spell.ManaFromCard.StrictlySelf`** came forward from M9 as two gates, the second on M6's `mana_from`. The Well's other parts needed nothing of M9, so it enters here and leaves that row. M9's table still owns any filter on a `SpellCast`.
- **`StaticEffect$`:** no copy, as (a) found. `read_static_effect` reads the `Mode$ Continuous` body as the `Duration$ Permanent` Animate it amounts to, `AbilityDef::static_effect`, and `record_battlefield_entry` puts it on each moved card's entry beside `GainControl$` and `WithCountersType$`. Any other body is a finding. `ExiledWithSource` reads last-known information once its source is gone, so a Vacuum sacrificed as a cost still knows.
- **A mana ability's Sac, Return, energy.** `mana_cost_paid_at_priority`: such an ability is listed at priority under auto-mana too, goes through the activation flow, which suspends for the pick, and no payment taps it; a payment's path is blocking and pays neither. `mana_source_affordable` now holds it to those parts. `docs/rules_deviations.md` §2.5. Only the sacrifice has a card.
- **`cmc<OP>X` in `ValidTgts$`** reads the announced X, and passes before the announcement. `x_leaves_a_target` offers only an X with a target; with none, the activation is reversed. The X ladder takes the X from the row, since a row's index no longer is X.
- **`RememberDiscarded$`** a row of the memory table; **`Discard`'s `Optional$`** a stop row at each pick; **`TopGraveyardCreature`** a filter row; **`Count$CastTotalManaSpent`** a value row.
- **Frames.** No `object` column added, the registry unchanged. An in-place copy has `card` its printed card, `copy_of` the card copied, and its `object_ability` rows the copied card's; an overlay shows as what differs from `copy_of`'s rows: Mockingbird as a Visionary has `Bird` in `object_subtype`, `Flying` in `object_keyword`, `COPY` in `object_provenance` (`END_OF_TURN` under The Myriad Pools). A token copy of a card was the token sentinel with no card at all: its `copy_of` now names the card, `Token::copied_card`, and its abilities reference that card's rows; the fixture tensors didn't move. `DecisionView` and `rm_agent.py` name such a token by it, and `--play` reaches it. Nothing face down exists to copy until M11; a copy reads only public objects, and `visible_card_id` holds through the entry (`EntryInfo::printed_card`).
- **Fixtures:** `decks/tests/mech_copy.dk`, keys `row:animate_eot_base_pt`, `row:TopGraveyardCreature`, `row:copy_add_keywords`, `row:copy_set_creature_types`, `row:cmc_bound_x_target`, `row:mana_sacrifice_at_priority`, `row:enters_as_copy`, `effect:CopyPermanent`, `trigger:Delayed`, `category:DISCARD`, `category:CHOOSE_CARD`; `decks/tests/mech_copy_well.dk`, which mills itself to reach Descend 8: `effect:SetState`, `effect:Clone`, `row:copy_until_eot`, `row:static_effect`.

Bugs. CR 714.2: a Saga's back face read as a Saga, so Reflection of Kiki-Jiki took lore counters, ran its chapters again and was sacrificed: `card_is_saga` asked of the front card at entry, at the precombat lore step and in the sacrifice check. Reported clean and played wrong: `SetState`'s `Mode$ Transform` did nothing, Delver transforming only inside `PeekAndReveal`; CR 701.27f keeps the two from flipping it back. A graveyard or exile preset gave every card position 0, its own commit. Caught by the fixture's fuzz before commit: `Count$CastTotalManaSpent` read a reanimated Mockingbird's earlier life. `cr202_3_mockingbird_reanimated_after_a_cast_spent_no_mana`. No known card's table rows drifted, and the corpus didn't move.

What stayed, and why:

- **`HIDDEN CARDNAME can't block.`** stays out: its one script is Untimely Malfunction, an M10 card, and no vocab card can test a `kBlockRestrictions` row for it. Shepherd's script doesn't say it.
- **A scenario for the CR 608.2h last-known colors fix (5ef694eb):** none. `lki.colors` is read only by `effective_colors` of an object that left during the running resolution, while the color's source left too or is the object: no vocab card moves an object and then reads its color in one resolution (filters read a card where it is, and a source sacrificed as a cost is superseded at once). Painter's Servant and Mycosynth Lattice reach every zone, so what they add survives the move anyway.
- **A token copy of a double-faced permanent** has one face (CR 707.8a), and a spell copied after it left the stack is read as its whole card: `docs/rules_deviations.md` §2.5.
- **Cards put into a graveyard together** stay in moved order (CR 404.3), which Shallow Grave is the first card to feel: `docs/rules_deviations.md` §2.6.
- **A Mockingbird that copies a Mockingbird** asks once: the replacement has applied to the event (CR 614.5). One that entered as a copy gives what it copied.
- **`Return` and energy on a mana ability** are held and paid at priority with no card to show it.

Evidence. Each card by removal: without `register_at_eot_move`'s exile, `set_pt_from_animate`, Animate's end-of-turn keywords, the AnimateAll rows, `RememberDiscarded$`, `apply_copy_overlay`'s lines, `card_is_saga`'s face, the X bound, the payer's gate, the Mana Spent row, `end_copy_until_eot`, the two SpellCast gates, `SetState`'s `Transform`, `ExiledWithSource`'s last-known read, `record_battlefield_entry`: each named scenario fails. The can't-be-countered scenarios pass on code Hexing Squelcher already has. Hidden information: every scenario at `--selfcheck-state-sample 1`, and 32 games of the two fixtures under `--selfcheck-knowledge --selfcheck-state-sample 1` (`explore:patient` against `explore`, and random under `--no-auto-mana`), no error. `--fuzz-games 100` on each fixture clean, no draw. `scripted` and Python `explore` over them, 27 games and matches, none drawn. `make registry`'s digest adds the nine faces and the Goblin Shaman and changes no other; `registry.json`, `tests/corpus`, `tests/cli` and `tests/parse_report/vocab.json` unchanged, `object` slot 89.

Unproved. Release `bench` wasn't compared. The `actor` tier's torch part was skipped. A Bird type on a copy is shown by the dump's rows alone: no vocab filter reads it. `Return` and energy on a mana ability have no card. The Myriad Pools on a spell with X, a bestowed spell or an Adventure's creature face hasn't been run.

### M9. Effect objects, triggers and replacements (L)

**Also here.** CR 610.3: the return of an "until" exile is a one-shot effect immediately after the event, but `register_exile_until_host_leaves` fires it as a delayed trigger, so players get priority first and a host that re-enters at once orders it among that entry's triggers (Portable Hole under Displacer Kitten; in `docs/rules_deviations.md`). Scenario that re-blink in both orderings.

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

**Also here.** CR 608.2b's check is incomplete for a spell with targets of its own and more on its sub-abilities: `resolve_ability()` fizzles it when its own are illegal, whatever the chain's are, and a spell with no target of its own is checked against the chain alone. It should count all of them, and prune per instruction. Test with the first vocab card of that shape.

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

### M12. Designations and subsystems (L)

**Today.** Each per-player or per-object field is hand-added across component, schema and encoder.

**Build.** A designation store, per player and per permanent, with the schema columns declared once, so that
state plumbing is one row. Then Speed (CR 702.178–179), Meld (CR 701.42, 712) and Empower. Dungeons and venture are out of
this plan (2026-11-06): `aluren` left the candidates, and Acererak was the only card using them.

**Notes.** Perilous Snare's `AddAbility$` on itself needs `ability_grant_targets` to handle the
single-target forms: it returns nothing for them today (the old `Affected$ Card.Self` script had the
same gap, and no vocab card uses the form, so M0 left it). "Empower Jace" is not in the checked-in CR (effective
2026-04-17): implement from the oracle text and record it in `docs/rules_deviations.md`, and add its token to
`provision_decks.py`'s `keyword_tokens`.

**Test.** Speed raised once per turn and on opponent life loss only;
Meld with one half missing; each new field across a snapshot.

### M13. The sweep (L)

- Verify any candidate card that reports clean and is still outside the vocab, as scenarios, including the ones
  M1 reclassified. Most enter earlier (Appendix B). Defense Grid shows why this cannot be skipped.
  Temur Sabertooth is one to check first: reading the ChangeZone branch it reaches, `ChangeNum$ 1` looks
  unenforced, so it may return more than one creature. Not run, since the card is outside the vocab.
- Finish the vocab, then run `onboard.py` stages 8–11 for every deck: checks, the fuzz campaign, designation,
  baseline. Run `death_and_taxes` and `uwx_control` here too; they need no engine work and can go earlier.
- Designation needs the user: which decks go to `league/` and which to `heldout/`. At least one fair and one combo
  deck stay held out.
- Fuzz all playable decks, `--matchups all --fuzz-games 100`, bo1 and `explore:patient`.
- Update CLAUDE.md's architecture notes for what changed, and `docs/rules_deviations.md`. Known so far: its
  key files omit `src/resolution_memory.{h,cpp}`, `src/cost_payment.{h,cpp}`, `src/components/chosen_cost.h`, `src/cast_options.{h,cpp}`, `src/components/cast_option.h` and
  `src/queries/permissions.{h,cpp}`, its `src/coverage`
  line omits the `row` kind, and its machine protocol section still names `prompt_mana_payment` (now `pay_mana_cost`).
- Make an amount SVar `is_runtime_amount_expr` doesn't list a finding, or read it from the SVar table: today it
  is silently 0 (M4's Done note).

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

None block M0 to M12. Before M13: the `league/` and `heldout/` split.

Settled 2026-11-06: `learn/open_vocab_acceptance.py` runs on a test-only copy of the tables holding the rows S0 recorded (`tests/open_vocab/s0_rows.json`), so a card whose rows move on purpose is one more entry there and every decision stays in the check. `gate` passes; `full` passed at M7's stage gate (below). `spell_reads_colors_spent` no longer gates the record: `spent_w` to `spent_g` fill for every cast, and `inputs --keep-decisions` kept the file's 8,735 decisions at today's digests (6,572 moved; S0's networks hold those columns at zero, so `gate` passes unchanged). It stays only for a hybrid pip's choice under auto-mana.

## Appendix A. Every finding, by stage

Generated from the status files at pin `f62f8048`: 174 findings, none unmapped. `onboard.py`'s mechanics are the report's grouping; this is the plan's.

| stage | findings |
|---|---|
| M0 | `static_param:AffectedDefined` |
| M2 | `ability_param:Animate.Defined`, `ability_param:ChooseCard.Defined`, `ability_param:CopyPermanent.Defined`, `ability_param:DealDamage.DamageSource`, `ability_param:Destroy.Defined`, `ability_param:Discard.Defined`, `ability_param:Draw.Defined`, `ability_param:PutCounter.Defined`, `ability_param:Reveal.Defined`, `ability_param:Reveal.RevealDefined` |
| M3 | `amount_svar:All`, `filter:!wasCastFromYourHand`, `filter:AttachedTo Creature`, `filter:ControlledBy Remembered`, `filter:Defined`, `filter:EquippedBy`, `filter:ExiledWithSourceLKI`, `filter:Imprinted`, `filter:SharesColorWithOther`, `filter:TargetedPlayerOwn`, `filter:TopGraveyardCreature`, `filter:TopLibrary`, `filter:cmcEQX`, `filter:leastPower`, `filter:lowestRememberedCMC`, `filter:sharesControllerWith`, `filter:withoutFlying`, `filter:withoutIndestructible`, `svar_expr:Count$ValidExile`, `svar_expr:Count$ValidGraveyard`, `svar_expr:Count$xPaid/Twice`, `svar_expr:Imprinted$Amount`, `svar_expr:Imprinted$Valid`, `svar_expr:PlayerCountOpponents$HighestValid`, `svar_expr:Sacrificed$CardManaCost/Plus.2`, `svar_expr:TargetedPlayer$CardsInLibrary/HalfUp`, `svar_expr:TargetedPlayer$CardsInLibrary`, `svar_expr:TargetedPlayer$LifeTotal/HalfUp`, `svar_expr:TriggeredCard$CardPower` |
| M4 | `ability_param:Branch.BranchConditionSVarCompare`, `ability_param:Branch.BranchConditionSVar`, `ability_param:Branch.FalseSubAbility`, `ability_param:Branch.TrueSubAbility`, `ability_param:ChangeZone.Imprint`, `ability_param:ChangeZone.NoShuffle`, `ability_param:ChangeZoneAll.UseAllOriginZones`, `ability_param:ChooseCard.AtRandom`, `ability_param:Cleanup.ClearNamedCard`, `ability_param:CopyPermanent.UnlessCost`, `ability_param:CopyPermanent.UnlessCost`, `ability_param:CopyPermanent.UnlessResolveSubs`, `ability_param:DealDamage.UnlessCost`, `ability_param:DealDamage.UnlessCost`, `ability_param:Discard.Mode`, `ability_param:Discard.RememberDiscarded`, `ability_param:ImmediateTrigger.ConditionDefined`, `ability_param:ImmediateTrigger.TriggerAmount`, `ability_param:ImmediateTrigger.UnlessCost`, `ability_param:ImmediateTrigger.UnlessCost`, `ability_param:Pump.ForgetImprinted`, `ability_param:Pump.ForgetObjects`, `ability_param:Pump.ImprintCards`, `ability_param:Repeat.MaxRepeat`, `ability_param:Repeat.RepeatCheckSVar`, `ability_param:Repeat.RepeatSVarCompare`, `ability_param:Token.UnlessCost`, `ability_param:Token.UnlessCost`, `category:Branch`, `category:Repeat`, `category:Shuffle`, `category:TapOrUntap` |
| M5 | `ability_param:Discard.ActivationZone`, `ability_param:Play.Condition`, `cost:Discard`, `cost:Exert`, `cost:ExileFromGrave`, `cost:ExileFromHand`, `cost:Exile`, `cost:Hybrid`, `cost:Mana`, `cost:Waterbend`, `keyword:AlternateAdditionalCost`, `keyword:Bargain`, `keyword:Convoke`, `keyword:Crew`, `static_param:Activator` |
| M6 | `ability_param:AddMana.AddsKeywordsUntil`, `ability_param:AddMana.AddsKeywordsValid`, `ability_param:AddMana.AddsKeywords`, `ability_param:AddMana.Produced`, `keyword:Sunburst`, `static_mode:ManaRestriction`, `static_param:InvertValidSource`, `static_param:ValidCard`, `static_param:ValidPlayer`, `static_param:ValidSA`, `static_param:ValidSource`, `svar_expr:Count$CastTotalManaSpent`, `trigger_param:ValidSA` |
| M7 | `ability_param:Play.ImprintPlayed`, `ability_param:Play.ReplaceGraveyardValid`, `ability_param:Play.ReplaceGraveyard`, `ability_param:Play.ValidZone`, `ability_param:Play.Valid`, `ability_param:Play.WithoutManaCost`, `keyword:Bestow`, `keyword:ETBReplacement:DBPrepare`, `keyword:Fuse`, `keyword:Overload`, `layout:Adventure`, `layout:Prepare`, `static_form:MayPlay.Graveyard`, `static_form:MayPlay.Library`, `static_param:CheckSecondSVar`, `static_param:EffectZone` |
| M8 | `ability_param:Animate.AtEOT`, `ability_param:AnimateAll.Power`, `ability_param:AnimateAll.Toughness`, `ability_param:AnimateAll.Types`, `ability_param:ChangeZoneAll.GainControl`, `ability_param:ChangeZoneAll.StaticEffect`, `ability_param:ChangeZoneAll.WithCountersType`, `ability_param:Clone.CloneTarget`, `ability_param:Clone.Duration`, `ability_param:CopyPermanent.AddKeywords`, `ability_param:CopyPermanent.AtEOT`, `ability_param:CopyPermanent.SetColor`, `ability_param:CopyPermanent.SetCreatureTypes`, `ability_param:CopyPermanent.SetPower`, `ability_param:CopyPermanent.SetToughness`, `keyword:ETBReplacement:ChooseColor`, `keyword:ETBReplacement:DBCopy`, `keyword:HIDDEN CARDNAME can't block.`, `static_form:Affected.Counters`, `static_mode:CantBlockBy`, `static_mode:CantPreventDamage`, `static_param:AddColor`, `static_param:ExcludeZone`, `static_param:GainsAbilitiesOfZones`, `static_param:GainsAbilitiesOf`, `static_param:IsCombat`, `static_param:ValidAttacker`, `static_param:ValidBlocker` |
| M9 | `ability_param:DelayedTrigger.Static`, `ability_param:DelayedTrigger.ValidActivatingPlayer`, `ability_param:Effect.Duration`, `ability_param:Effect.ForgetOnMoved`, `ability_param:Effect.RememberObjects`, `replacement:Draw`, `replacement:Effect:Counter`, `replacement:Effect:Moved`, `static_form:Effect.Continuous`, `trigger_form:ChangesZone.ValidCard`, `trigger_form:Effect.Phase`, `trigger_form:SpellCast.ValidCard`, `trigger_mode:Taps`, `trigger_param:ActivationLimit`, `trigger_param:Revolt`, `trigger_zone:ChangesZone:Exile` |
| M10 | `ability_param:Animate.Exhaust`, `ability_param:ChangeZone.TargetsWithSameController`, `ability_param:ChangeZone.TgtZone`, `ability_param:DealDamage.Boast`, `ability_param:DealDamage.DividedAsYouChoose`, `ability_param:Pump.PumpZone`, `ability_param:Pump.TargetsWithDefinedController`, `category:ChangeTargets` |
| M11 | `ability_param:Cloak.Defined`, `ability_param:Cloak.RememberCloaked`, `category:Cloak`, `keyword:Morph`, `static_param:MayLookAt` |
| M12 | `ability_param:Empower.Type`, `category:Empower`, `keyword:Start your engines`, `layout:Meld`, `static_param:Condition` |

The 21 cards M0 repairs: the 15 vocab cards (Animate Dead, Cori-Steel Cutter, Dragon's Rage Channeler, Elvish Reclaimer, Kaito Bane of Nightmares, Keen-Eyed Curator, Knight of the Reliquary, Lavaspur Boots, Lion Sash, Meteor Sword, Pre-War Formalwear, Shadowspear, Sheltered by Ghosts, Skateboard, Wight of the Reliquary) and 6 candidates (Cryptic Coat, Dark Matter Manipulator, Fiend Artisan, Hawkeye's Bow, Perilous Snare, Springheart Nantuko).

## Appendix B. When each card can enter the vocab

A card joins the vocab as soon as every mechanic it needs is implemented (M1's `onboard.py vocab --card`), at the end of the unit that finishes the last of them, whatever stage the table below gives: the table is the survey's estimate, and a card outside the vocab cannot be run. So each unit ends by entering every card it finished, and the cards reporting clean do not wait for M13. Each still needs a scenario and a reading of its script and oracle text, since clean is not correct; a clean card that turns out to need a later primitive stays out and is placed here.

**Clean-card batch (2026-11-05).** Nineteen cards entered with no later stage needed, ids 381 to 399: Temur Sabertooth; Burning Wish; Elvish Mystic, Fyndhorn Elves, Llanowar Elves, Elvish Visionary, Quirion Ranger, Wirewood Symbiote, Stitcher's Supplier; Taiga, Watery Grave, Great Furnace, Lush Portico, Nurturing Peatland, Raucous Theater; Faithless Looting, Tendrils of Agony, Empty the Warrens, Unmask. None was placed. Three reported clean and played wrong until fixed: a numeric `ChangeNum$` on the battlefield multi-select, an untargeted Pump's `Defined$`, and a bare `Origin$ Stack` move.

**Second clean-card batch (2026-11-05).** Eighteen cards entered, ids 400 to 417: Aluren, Archive Arbiter, Basalt Monolith, Chromatic Star, Collective Brutality, Crucible of Worlds, Cursed Totem, Displacer Kitten, Fiery Confluence, Giant's Boulder, Goblin Engineer, Haywire Mite, Legion Extruder, Mishra's Research Desk, Portable Hole, Soul-Guide Lantern, Thought Monitor, Thundertrap Trainer. None was placed: Aluren needed nothing from M7, its free cast and flash already read through `free_cast_grant`. Two reported clean and played wrong until fixed: an unearth offered on the battlefield (Mishra's Research Desk, and Cityscape Leveler before it) and a token returning from a same-resolution exile (Displacer Kitten, and Eldrazi Confluence before it). One gap stays, in `docs/rules_deviations.md`: the return of an "until" exile uses the stack, against CR 610.3.

**M5 (c1) (2026-11-06).** All eleven of the M5 row entered, ids 418 to 428, the Robot token 920: Defense Grid, Formidable Speaker, Melded Moxite, Nihil Spellbomb, The Echoverse Fulcrum, The Stone Brain, Runehorn Hellkite, Simian Spirit Guide, Bitter Triumph, Fiend Artisan, Vibrance. None was placed elsewhere. No other candidate card's findings reached zero: the nearest have one finding each, of a later stage (Aphetto Alchemist, Cyclonic Rift, Engineered Explosives, Fury, Hullbreacher, Massacre, Renegade Rallier, Sink into Stupor, Sundering Eruption, Wear/Tear). Two had parts that reported clean and played wrong until fixed: Vibrance and The Stone Brain (M5's Unit (c1)).

**M6 (b) (2026-11-06).** Both of the M6 row entered: Engineered Explosives 429, Arena of Glory 430. None was placed elsewhere. No other candidate card's findings reached zero: the 44 left outside the vocab of the 330 candidate cards hold one finding at best, each another stage's (Aphetto Alchemist, Cyclonic Rift, Fury, Hullbreacher, Massacre, Renegade Rallier, Sink into Stupor, Sundering Eruption, Wear/Tear). Next free card id 431, token 921.

**M7 (b) (2026-11-06).** Six of the M7 row and Brazen Borrower entered: Massacre 431, Cyclonic Rift 432, Wear/Tear 433 (three names, one id), Brazen Borrower 434 with Petty Theft 435, Springheart Nantuko 436 with the Insect token 921, Carnivorous Cultivator 437 with Enroot 438. Brazen Borrower is the only Adventure card among the provisioned scripts, so `S:Mode$ CantBlockBy` came with it from M8; the only other scripts with that mode are Kappa Cannoneer, Manifold Key's effect and Cryptic Coat, which needs M11. None of the six was placed elsewhere. No other candidate card's findings reached zero: the 38 names left outside the vocab of the 330 candidate cards hold one finding at best, each another stage's (Aphetto Alchemist, Fury, Hullbreacher, Renegade Rallier, Sink into Stupor, Sundering Eruption). Unit (c) has Hogaak, Bilbo and Beseech the Mirror left of the M7 row. Next free card id 439, token 922.

**M7 (c) (2026-11-06).** The rest of the M7 row entered: Bilbo, Thief in the Night 439, Beseech the Mirror 440, Hogaak, Arisen Necropolis 441. None was placed elsewhere. Gaea's Will (M9) and Mystic Forge (M11) stay out, their findings open. No other candidate card's findings reached zero: the 34 names left outside the vocab of the 321 candidate cards hold one finding at best, each another stage's (Aphetto Alchemist, Fury, Hullbreacher, Renegade Rallier, Sink into Stupor, Sundering Eruption). All three had parts that reported clean and played wrong until fixed: Bilbo's flashback cost, Beseech's "if it wasn't cast this way", Hogaak's Delve of itself. Next free card id 442, token 922.

**M8 (b1) (2026-11-06).** Four of the M8 row entered: Frenzied Baloth 442, Painter's Servant 443, Grist, the Hunger Tide 444 with the Insect token 922, Agatha's Soul Cauldron 445. None was placed elsewhere. No other candidate card's findings reached zero: the 30 names left outside the vocab hold one finding at best, each another stage's (Aphetto Alchemist, Fury, Hullbreacher, Renegade Rallier, Sink into Stupor, Sundering Eruption). Unit (b2) has Fable, Lazotep Quarry, The Everflowing Well, Mockingbird, Allosaurus Shepherd, Shallow Grave and Ghost Vacuum left. All four had parts that reported clean and needed reading: the Baloth's protection, the Servant's last-known colors, Grist's own Insect in the graveyard, the Cauldron's loyalty abilities on a creature. Next free card id 446, token 923.

**M8 (b2) (2026-11-06).** The rest of the M8 row and The Everflowing Well entered: Allosaurus Shepherd 446, Shallow Grave 447, Fable of the Mirror-Breaker 448 with Reflection of Kiki-Jiki 449 and the Goblin Shaman token 923, Lazotep Quarry 450, Mockingbird 451, The Everflowing Well 452 with The Myriad Pools 453, Ghost Vacuum 454. The Well's table entry said M9 for a `SpellCast` filter; two gates brought it here, and it leaves that row. None was placed elsewhere. No other candidate card's findings reached zero: the 23 names left outside the vocab hold one finding at best, each another stage's (Aphetto Alchemist, Fury, Hullbreacher, Renegade Rallier, Sink into Stupor, Sundering Eruption). All reported clean at some part and needed reading: Fable's back face a Saga, the Well's `SetState`, Mockingbird's reanimated bound, Lazotep Quarry's X row. Next free card id 455, token 924.

| last stage | cards (all stages needed) |
|---|---|
| M3 | Dark Matter Manipulator; Drop of Honey (M2 M3); Peer into the Abyss; Powder Keg; Rough/Tumble; Strategic Betrayal |
| M4 | Chaos Defiler (M2 M3 M4); Echo of Eons (M2 M4); Galvanic Discharge; Gamble; Grindstone (M3 M4); Sewer-veillance Cam; Song of Creation; Stronghold Gambit (M2 M3 M4) |
| M5 | Bitter Triumph; Defense Grid; Fiend Artisan; Formidable Speaker; Melded Moxite; Nihil Spellbomb; Runehorn Hellkite (M2 M4 M5); Simian Spirit Guide; The Echoverse Fulcrum; The Stone Brain (M3 M4 M5); Vibrance |
| M6 | Arena of Glory (M5 M6); Engineered Explosives |
| M7 | Beseech the Mirror (M3 M5 M7); Bilbo, Thief in the Night (M3 M7); Carnivorous Cultivator; Cyclonic Rift; Hogaak, Arisen Necropolis (M5 M6 M7); Massacre; Springheart Nantuko (M2 M3 M4 M7); Wear/Tear |
| M8 | Agatha's Soul Cauldron (M4 M6 M8); Allosaurus Shepherd; Brazen Borrower (M3 M7 M8); Fable of the Mirror-Breaker (M4 M8); Frenzied Baloth; Ghost Vacuum; Grist, the Hunger Tide (M4 M8); Lazotep Quarry (M3 M8); Mockingbird (M6 M8); Painter's Servant; Shallow Grave (M3 M8); The Everflowing Well (M6 M8) |
| M9 | Chrome Mox (M3 M4 M9); Gaea's Will (M7 M9); Hawkeye's Bow (M2 M3 M9); Hullbreacher; Pyrogoyf (M2 M3 M9); Renegade Rallier; Sand Scout (M3 M9); Sundering Eruption (M3 M9); Tezzeret, Cruel Captain (M2 M9) |
| M10 | Broadside Bombardiers (M3 M10); Fury; Goblin Welder (M3 M4 M10); Invasion Submersible (M5 M10); Shatterskull Smashing (M3 M10); Sink into Stupor; Unlicensed Hearse (M3 M5 M10); Untimely Malfunction (M8 M10) |
| M11 | Aphetto Alchemist; Cryptic Coat (M3 M8 M11); Mystic Forge (M3 M7 M11) |
| M12 | Perilous Snare; Phyrexian Dragon Engine (M5 M12); Theorist's Proxy (M9 M12) |

## Appendix C. Deck readiness (an output, not a goal)

The stage after which each deck has every mechanic. Decks with no entry in the table's left column need no engine work.

| ready after | decks (meta share %) |
|---|---|
| M3 | lands_top8 (5.0) |
| M8 | cradle_control (1.0), dimir_tempo (10.0), elves (2.0), hogaak (2.0), reanimator_top8 (3.0), ur_tempo (4.0) |
| M9 | boros_aggro (6.0), maverick (1.0), storm (3.0) |
| M10 | artifacts_blue (2.0), cloudpost_tron_ramp (5.0), dragon_stompy (1.0) |
| M11 | stoneblade (1.0), welder_cam (2.0) |
| M12 | artifacts_prison (2.0), doomsday (4.0), mystic_forge (2.0), painter (1.0) |

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

## After the plan: an MCP server for LLM play

Design an MCP server so an LLM can drive one seat of a game against the model, and use those games as a benchmark. Not designed yet, and not part of M0 to M13.

What it can build on: `--machine` gives a seat its own frames with hidden information withheld and takes a menu row as its reply; `src/cli/board.h` already renders those frames as text; `nn:` plays the other seat; a decision log replays each game.

The design has to settle how a game's state and menu reach the LLM as tools, how the benchmark is scored (decks, seeds, formats, games per pairing, and how it sits beside `rmtrain bench`), and how a long game fits an LLM's context and budget.
