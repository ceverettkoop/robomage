# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code Style

- Don't put new functions in main.cpp
- Don't edit my comments for spelling or punctuation. Only change them if something substantive changed.
- Do not include in comments explanations of how things were previously prior to a change. Document only how code currently behaves.
- Avoid inline logic for anything that will be repeated; write new functions that are reusable
- Declare local functions as private in the class, if the header contains a single class/struct, if header does not contain a class, write them as static functions in global namespace C-style.
- Iterate through mEntities when possible (working within a system class), rather than iterating through all entities
  Never scan entity ids. Free code with no system's set in reach (SVar evaluation, replacement
  dispatch, the observation serializer) walks `zoned_entities()` (`src/queries/zones.h`), the
  systems' shared `{Zone}` set; players are reached through `cur_game`'s player entities.
- Try to consolidate iterations through entities within a function, rather than iterating through many times
- Shared entity queries live in `src/queries/`, one header per concern (list under Key files);
  include the specific header you use.
- To find battlefield permanents, use the shared accessors in `src/queries/battlefield.h` — the
  `is_battlefield_permanent(entity, ctrl)` predicate as a loop guard / single-entity check,
  or `battlefield_permanents(mEntities, ctrl)` for the whole list — instead of open-coding
  the `Permanent` + `Zone` + `BATTLEFIELD` (+ controller) scan inline. These bake in the rule
  that a phased-out permanent is treated as though it doesn't exist (CR 702.26b), so do **not**
  add your own `is_phased_out` check at the call site. The only code that reads
  `Permanent::is_phased_out` directly is the phasing subsystem itself (the untap-step
  phase-in/skip in `classes/game.cpp`), the phase-out setter in `effects/effect_phases.cpp`,
  and the observation serializer — the state-format encoder `state_format/encode.cpp` — which
  deliberately includes phased-out permanents, flagged.
  Add similar shared accessors (in the `src/queries/` header for their concern) when a new
  entity-scan pattern starts repeating.
- Static (local) functions should be forward declared at top of source file for clarity
- C++17 with exceptions disabled (`-fno-exceptions`; only the libtorch actor TUs in `src/actor/` enable them)
- The kept Python (`python/`: the RMSTATE agent driver, the scripted agent, the decoder, the corpus
  recorder and their checks) is **stdlib only**; run it with `python3`, no venv. Flags use one
  vocabulary: `--format bo1|bo3` (driver), a seat's deck `--deck-a` / `--deck-b`, a seat's agent
  `--player-a` / `--player-b` (driver) or `--agent-a` / `--agent-b` (the engine's in-process
  agents).
- Uses clang-format configuration in `.clang-format`
- DO NOT MODIFY CARD SCRIPTS
- When given a long list of tasks or bugs to fix, do them one at a time (unless it is sensible to batch some)
  and use subagents for each one. Instruct the subagents to not spawn additional agents. 

## Project Overview

Robomage is a C++ implementation of a Magic: The Gathering game engine using an Entity Component System (ECS) architecture.

**Scope: two-player games only.** The engine simulates exactly one 1-v-1 matchup (Player A vs
Player B). Multiplayer rules (CR 800+: 3+ players, range of influence, "each opponent" over
several opponents, …) are out of scope — implement card effects for the two-player case and add
no multiplayer-only machinery.

Every game decision logged as an integer. Games can be replayed deterministically when provided with the correct seed.

Players and agents drive the engine through its in-process agents (`--agent-a/-b`, `--play`), its
C++ CLI (human seats: `--player A --agent-b explore`, or hotseat by default; it renders only the
seat's decoded frames, `src/cli/`), or the machine protocol (`--machine`): **RMSTATE**, an exhaustive
binary game state per viewer with hidden information withheld (`docs/state_format.md`), each
seat's frames carrying the C++ knowledge sidecar's tables (perfect recall and sound deduction,
`docs/knowledge_store.md`). The small Python side (`python/`) is the rule-based scripted agent
and its driver over that protocol, used for agent games and to record the replay corpus.

The plan this repo follows is [`docs/engine_fork_plan.md`](docs/engine_fork_plan.md) (steps 1–5
done — 4: one decision schedule and a C++ CLI on the decoder; 5: a C++ featurizer, the in-process
actor and remote batched evaluation; next: `docs/rnad_training_plan.md`). The RL stack (PPO/AlphaZero training, the float state vector, the libtorch actor, the
TUI/GUI) is archived in place: branch `archive/rl-stack` (tag `rl-stack-final`, if created), whose
tip `cf7f97f` is the last commit with both stacks.

## Rules Reference (authoritative)

The official **MTG Comprehensive Rules** are checked in at
[`docs/mtg_comprehensive_rules.txt`](docs/mtg_comprehensive_rules.txt) (navigation guide:
[`docs/mtg_comprehensive_rules.md`](docs/mtg_comprehensive_rules.md)). **Consult them whenever you
implement or test a mechanic** — they define correct behavior independent of the engine's current
state. `grep` the numbered rule rather than reading the 9k-line file (e.g.
`grep -nE "^509\." docs/mtg_comprehensive_rules.txt`; 702 keywords, 704 SBAs, 613 layers).
Known deviations and gaps are listed in [`docs/rules_deviations.md`](docs/rules_deviations.md);
update it when you knowingly deviate or leave a rules gap.

## Build Commands

Card scripts are not tracked; provision them first on a fresh clone (the SessionStart hook does
it in cloud sessions): `python3 tools/forge_fetch/provision_decks.py`.

- `make` — debug build (`-ggdb`, warnings, `-D_GLIBCXX_ASSERTIONS`, no `-O2`) of the four
  binaries: `bin/debug/robomage` (the engine), `bin/debug/rmknow` (the knowledge sidecar over a
  recorded stream), `bin/debug/rmfeat` (the featurizer over a recorded stream: FEATURE_SPEC, the
  card tables, tensors, learner targets) and `bin/debug/robomage_check` (the test gate)
- `make BUILD=RELEASE` — the same into `bin/release/` (`-O2 -flto -march=native`; separate obj/bin
  trees per config, so switching `BUILD` never needs `make clean`)
- `make pgo` — `bin/pgo/robomage` alone: the release flags with a profile (`-fprofile-use`), from an
  instrumented build playing explore bo3 games with the featurizer on (~10–15% faster; `docs/ci.md`
  throughput reference)
- `make check` — build, then `bin/debug/robomage_check` (every tier; below and `docs/ci.md`)
- `make regen` — provision, build, then `python3 python/record_corpus.py`: re-record the replay
  corpus after an intentional behavior change
- `make refetch` — delete the local card/token scripts, re-provision them at the pinned Forge
  commit and run `make check` without re-recording (detects upstream script drift)
- `make rmembed` — `bin/<config>/rmembed`, the oracle-text table's generator (libtorch, C++20; found
  like `make actor`, `LIBTORCH_DIR=…`); not part of plain `make`
- `make text-table` — check `bin/resources/embeddings/oracle_text.rmtb` against the vocab and
  `tools/embed/MODEL_PIN` (`robomage --text-table-check`); when rows are stale, fetch the pinned
  model (`tools/embed/fetch_model.py`, network) and regenerate only those rows with `rmembed`
  (`docs/featurizer.md` §5). Part of `all`: a current committed table needs no network or libtorch
- `make actor` — `bin/<config>/robomage_nn`: the engine plus the libtorch TUs of `src/actor/`
  (exceptions on, C++20), which add the `nn:<model>` and `nn:remote:<socket>` agents
  (`docs/featurizer.md` §6), and `bin/<config>/rmserve`, the reference inference server
  (`docs/inference_protocol.md`). Needs a torch importable by `python3`, or `LIBTORCH_DIR=…`; not
  part of plain `make`
- `make clean`

**Which binary the tooling runs is independent of which one you just built.** The gate
(`robomage_check`) runs the engine beside it (`bin/<config>/robomage`), so `make check` tests the
debug build and `bin/release/robomage_check` the release one; `python/rm_driver.py` and
`python/record_corpus.py` default to `bin/debug/robomage`; `ROBOMAGE_BUILD=release` moves the
driver onto `bin/release/robomage`, and `--binary <path>` overrides one invocation of any of them.

**There is no codegen.** `bin/resources/decks/archetypes.json` is read directly (its `_archetypes`
key is the canonical label list; `src/state_format/metagame.cpp`, the `METAGAME` frame).

## Long-running commands (Claude)

When Claude runs a long command (build, gate, fuzz, bench) it must run it **in the
background** and make the live output inspectable by the user:

- Always state the task's output-file path in chat when launching it, so the user can
  `tail -f` it while it runs. (No symlinks or other indirection — just the path.)

## Testing guidelines

- **The standard gate is `make check`**: the build, then every tier of the C++ gate
  `bin/debug/robomage_check` (it runs engine subprocesses in parallel and links no engine TU). It
  must pass before pushing, and CI runs exactly it. Tiers, in order (`--tier a,b` runs a subset;
  details, runtimes and reproducing a CI failure in [`docs/ci.md`](docs/ci.md)):

  | Tier | Checks |
  |---|---|
  | `vocab` | `--check-decks`: every top-level / `league/` deck card loads and is in the vocab; the text tokenizer against `tests/embed/tokens_golden.jsonl` (when the model's vocab is cached) |
  | `concede` | the `-2` / `-3` concession sentinels over RMSTATE sessions and at parked queries; results and `--replay` |
  | `scenarios` | every rules-regression scenario (`tests/scenarios/*.json`) |
  | `cli` | the human CLI's golden transcripts: each scripted stdin session (`tests/cli/<name>.json`), stdout byte-identical to `<name>.out` |
  | `replay` | each corpus decision log (`tests/corpus/<name>.rmlog`) replayed, output byte-identical to `<name>.replay` |
  | `state` | the state format's self-checks over the replayed corpus, explore fuzz games and the scenarios; re-encoding, decoder refusals, stream sessions vs dumps, name-a-card menus, include isolation |
  | `featurize` | the card tables and FEATURE_SPEC from `--cards-frame`, exported and read back; the corpus's and explore games' dumps featurized (deterministic, shapes and indices against the spec, row-order invariance, the actor entry point's refusal, learner targets); `--selfcheck-featurize`; `rmfeat` vs in-process; `python/pycheck.py` on the files |
  | `traj` | the trajectory writer (`--trajectory-out`, `docs/trajectory_format.md`): in-process runs (bo1 with the omniscient part, bo3, concessions, a capped seat-A-only run) and their `--replay`: identical episodes, the replies the decision log's, tensors those of the dump's frames featurized, μ a distribution with mass at the reply, one episode per game with the `GAME_RESULT` footers; `python/rmtraj.py` vs the C++ reader; a `rm_driver.py --rmlog-dir` scripted match re-featurized; the teacher channel (`--teacher-a/-b`, `python/teacher.py`): at `--teacher-mix 1` the decisions are `rm_driver.py` scripted:hard's and the teacher's stream is the `--machine` stream byte for byte, at mix 0 the decision log is unchanged, at mix 0.5 the coin, the μ mixture and the replay; the labels `teacher.py`'s on the recorded streams; broken teachers fatal |
  | `actorloop` | the continuous actor (`--actor`, `docs/actor.md`): six matches of a two-entry schedule (bo1 swapping sides, bo3 swapping decks), the status and results files against the episodes' footers, every match's decision log replayed alone to the same episodes, a rerun with the same run seed and actor id writing the same logs; the per-game cap (end reason `cap`, the actor goes on); a taught entry over three matches, labels a fresh `teacher.py`'s on each replay; a schedule rewritten mid-run picked up at the next match, then SIGTERM mid-match: exit 0, `QUIT` in the log, no partial episode |
  | `actor` | the policy projection's (`thresh=` / `disc=`) unit checks; then opt-in (`bin/<config>/robomage_nn` and `$ROBOMAGE_TORCH_PYTHON`, else skipped): the reference policy (`tools/nn/`) built; `nn:` vs explore bo3 matches (one under `temp=0.02:thresh=0.03:disc=16`, its behaviour distributions checked), `.pt2` and TorchScript playing the same argmax match, exact re-featurization of a `--replay`, the three exports' agreement, `nn:` trajectories against `dump=` / `evals=`; `nn:remote:` through `rmserve` (concurrent actors, batches across clients) and through `tools/nn/serve_check.py` playing the in-process games, evaluations within 1e-5, handshake refusals, a killed server failing the actor; `robomage_nn --actor` reloading a replaced `nn:<model>` between matches; a continuous `--actor` smoke run (the random-initialised reference policy on both seats, one taught at mix 0.5, bo3: μ, labels, status and results checked, every match replayed to the same episodes) |
  | `knowledge` | `rmknow --selftest` / `--exactness` / `--belief-exactness`, soundness under `--selfcheck-knowledge`, isolation and truth, archetype inference, belief calibration |
  | `fuzz` | C++ explore vs explore over league matchups (bo1, plus bo3 `explore:patient` and bo1 `--no-auto-mana`); draws are findings |
  | `bench` | per-decision cost of a fixed explore workload by stage: engine alone, + frames, + knowledge, + featurizer, and (opt-in, as `actor`) evaluation in process and through `rmserve` (no threshold) |
  | `smoke` | scripted hard vs hard through `python/rm_driver.py`, and `python/pycheck.py` (incl. `python/rmtb.py` on C++-written RMTB samples) |

- Non fatal errors are not acceptable
- Draws are not acceptable, outside of exceedingly rare cases, and require review
- Do not attempt to test cards that are not already in `src/card_vocab.h`. Cards absent from the card vocab are considered unimplemented.
- **Agent games** (the quick check of a new build): `python3 python/rm_driver.py --player-a
  scripted --player-b explore --deck-a league/bug --deck-b league/burn --format bo1 --games N`
  (bo3 by default; game i uses seed `--seed`+i; `--verbose` per-decision narrative + menu, `--out
  FILE`, `--quiet`, `--max-decisions N`, `--timing`, `--rmlog-dir DIR` (keep every game's decision
  log); `--help` lists all). Agents: `scripted` (= the
  **hard** tier: smart mulligan, combat simulation, combo lines), `scripted:easy` / `greedy` /
  `random`, `explore` / `explore:patient` (coverage fuzzer; big-mana mode), `first` (= `auto`),
  `actions:<ints>`. Choose `--deck-a/-b` that exercise what you changed (decks relative to
  `bin/resources/decks/`). Details: [`docs/game_running.md`](docs/game_running.md).
  - **Fuzz campaign** (any draw is a finding): `bin/debug/robomage_check --tier fuzz --matchups
    all --fuzz-games 100 [--mode explore:patient] [--seed N]` (the engine's C++ explore agents;
    transcripts and draw logs under `ci_out/`)
  - **Engine throughput:** `bin/release/robomage_check --tier bench`

### Testing card behavior: the engine's own harness

The engine itself sculpts and scripts a game. Run it from `bin/` (`cd bin && ./debug/robomage
…`) with `--narrative --trace-decisions` (the perfect-information game log, plus every decision's
board, pending decision, numbered menu and the choice made, as the CLI renders the deciding seat's
frame: `| ` lines),
the test-setup presets and `--play`; `--help` lists every flag (zone presets
`--battlefield/graveyard/exile/sideboard-a/-b`, `--life-a/-b`, `--game-seed`, `--max-decisions`,
`--decision-log`, …).

- **Format:** one game by default; `--bo3` plays a match with sideboarding. Cap the run with
  `--max-decisions N` (it stops before decision N+1 with a `MAX_DECISIONS:` line, exit 0).
- **Seed:** `--game-seed N` is the game the driver and the scenario files call seed N (the engine
  seed derived from it, and the in-process agents' generator seeded with N); `--seed` sets the
  raw engine seed instead.
- **Setting up cards:** `--hand-a/-b` + `--library-a/-b` build a stacked deck (hand first; padded
  to 15 cards with its last card). `--battlefield-a/-b` permanents start in play without summoning
  sickness. Otherwise `--deck-a/-b <stem>` (relative to `bin/resources/decks/`; default `delver`).
- **Shuffling:** libraries shuffle with the seeded RNG unless `--no-shuffle` (deck-file order =
  draw order; first 7 = opening hand). Any `--hand-a/-b` implies `--no-shuffle` for both seats.
- **No commas in card names.** List flags (`--play`, `--hand-*`, `--library-*`, zone presets)
  split on commas; card lookup strips punctuation, so write `Thalia Guardian of Thraben` (in
  `--play`, a unique comma-free substring such as `cast:thalia`). Apostrophes are optional.
- **Who decides:** `--play` (semantic specs; preferred) or `--actions` (positional indices;
  fragile) is ONE global script for both seats. It makes every decision until it runs out; then
  each seat's `--agent-a/-b` agent decides (default `first` = action 0, so the game
  auto-advances; e.g. `explore`). Agent specs: `first` (`auto`), `random`, `explore`,
  `explore:patient`, `actions:<i,i,…>`, `play:<specs>` (`src/agents/agent.h`), and in
  `robomage_nn` only `nn:<model>[:temp=<t>][:thresh=<p>][:disc=<n>][:argmax]` (a learned policy, `src/actor/nn_agent.h`)
  and `nn:remote:<socket>[:…]` (the same, evaluated by an inference server such as `rmserve`).
  **Claude cannot play a seat interactively** (no TTY; a seat without `--play`, `--actions` or an
  agent reads the RMSTATE stream on stdin) — use `--play`. A `--play` run prints `resolved --actions: …` for exact
  replay (and `SCRIPT INCOMPLETE: …` when specs were left unapplied); keep `--game-seed` fixed while
  extending a line.

**`--play` grammar** (the C++ resolver `src/agents/play_spec.{h,cpp}`; full list in its header).
Each spec is resolved against *that* decision's live menu by intent, so it survives menu
reordering; an unmatched or ambiguous spec **fails loudly with the legal menu** (`FATAL: --play: …`).

| Spec | Meaning |
|---|---|
| `A:<spec>` / `B:<spec>` | seat key (see below) |
| `pass` | pass priority / take the first choice |
| `keep` / `mulligan` | opening mulligan decision |
| `cast:<card>` · `play:<card>` / `land:<card>` | cast a spell / play a land |
| `activate:<card>[@own/opp]` | activate a permanent's ability |
| `target:<card>[@own/opp]` / `target:<text>` | choose a target (card, or player/mode by description) |
| `attack:<card>` / `attack:done` · `block:<card>` / `block:done` | declare / confirm attackers, blockers |
| `mana:<w/u/b/r/g/c>` / `tap:<land>` | tap for mana |
| `search:<card>` / `search:fail` · `top:` · `bottom:` · `dig:<card>` | library search / top / bottom / dig pick |
| `pay:<text>` · `choice:<text>` · `sb-in:` / `sb-out:<card>` / `sb-done` | optional cost · generic/modal choice · sideboarding |
| `sacrifice:` `return:` `x:` `discard:` `mode:` `color:` `name:` `exile:` `shuffle:` `companion:` … | the matching choice category, by text / card |
| `concede` / `concede:game` / `concede:match` | the `-2` / `-3` sentinels |
| `desc:<text>` · `#<n>` or bare integer | any action by description substring · literal index |

Card names match case- and apostrophe-insensitively; an exact name wins, else a unique substring
(`cast:bolt`). Identical duplicate choices collapse to the first; genuinely distinct matches are
ambiguous — pin them with `@own`/`@opp`, `desc:`, or `#<n>`. Start a sculpted-hand line with one
`keep` per seat.

**Seat keys** (`A:`/`B:`, case-insensitive; independent of `@own`/`@opp`) let you write each
player's line without hand-sequencing priority passes. If the next spec is keyed to the seat
lacking priority, the priority holder auto-passes (its mandatory choices, e.g. cleanup discard, go
to its `--agent-a/-b` agent) until the keyed seat is on the clock. A keyed spec also passes its
own seat forward through priority windows until the action is legal (e.g. `A:attack:X` written in
main phase waits for declare-attackers); a typo'd/ambiguous spec, or one that hits a mandatory
choice it doesn't match, still fails loudly. Unkeyed specs apply to whoever has priority, with no
auto-advance. **The engine never asks a decision with one choice** (a lone pass, or any menu left
with one row: a single legal target, a confirm with nothing left to declare, a fail-only search; no
frame, not logged), so write `pass` only where that seat could do something else. A spec that names
such a one-row decision (not `pass`, `#n` or a concession) is consumed when the engine takes it, so
a script may name it or leave it out; `--trace-decisions` prints it as `| auto: Player X: <row>`,
and scenario `expect_menu` assertions see it.

```bash
cd bin
# Sculpted hands, script the opening, then the explore agents play on
./debug/robomage --narrative --trace-decisions --game-seed 1 --max-decisions 30 \
  --hand-a "Mountain,Lightning Bolt" \
  --library-a "Island,Island,Mountain,Mountain,Mountain,Mountain,Mountain,Mountain" \
  --hand-b "Forest,Grizzly Bears" --library-b "Forest,Forest,Forest,Forest,Forest,Forest,Forest,Forest" \
  --play "keep,keep" --agent-a explore --agent-b explore

# Bolt a bear already in play, seat-keyed (B has no response, so it gets no decision)
./debug/robomage --narrative --trace-decisions --game-seed 1 --max-decisions 10 \
  --hand-a "Lightning Bolt" --battlefield-a "Mountain" --battlefield-b "Grizzly Bears" \
  --play "A:keep,B:keep,A:cast:Lightning Bolt,A:target:Grizzly Bears@opp"
```

`--agent-a "play:<specs>"` drives one seat by specs against the other seat's agent (each list
drives only its seat, so leave specs unkeyed there).

**Rules-regression scenarios.** When you fix a rules bug, save the harness line that shows it as
`tests/scenarios/<cr-rule>_<what>.json` (a JSON scenario plus a `"why"` line citing
the CR rule and the commit, and assertions on the rules outcome: `expect` / `expect_not` /
`expect_order` regexes over the narrative, `expect_menu` / `expect_menu_not` over offered actions,
`expect_result`; every key in `tools/robomage_check/scenarios.h`). Keep `max_decisions` just past the
asserted event, check it passes (`bin/debug/robomage_check --tier scenarios --scenario <name>`,
`--show narrative` to see what the assertions match) and fails without the fix; the
`scenarios` tier of `make check` runs them all.

**JSON scenarios** (`tests/scenarios/*.json`, run by `robomage_check --tier scenarios --scenario
<glob>`) capture a reproducible setup: keys `name`, `hand_a/b`, `library_a/b`, `deck_a/b`,
`battlefield_a/b`, `graveyard_a/b`, `exile_a/b`, `sideboard_a/b`, `life_a/b`, `play` (spec list)
or `actions`, `seed` (the `--game-seed`, default 1), `max_decisions` (default 60), `format`,
`offer_cancel`, `engine_flags`, `variants`, plus the assertions (incl. `expect_count`,
`expect_unique`, `expect_same` and the knowledge-dump keys `expect_dump*`). Each runs as one engine
process with `--narrative --trace-decisions`; once its script runs out a seat takes action 0.
```json
{"name": "bolt_kills_bear", "why": "CR 120.6: …", "hand_a": ["Mountain", "Lightning Bolt"],
 "battlefield_b": ["Grizzly Bears"],
 "play": ["keep", "keep", "play:Mountain", "cast:Lightning Bolt", "target:Grizzly Bears@opp"], "seed": 1,
 "expect": ["Grizzly Bears is destroyed"]}
```

**Decision logs and replay.** `--decision-log FILE` (or `--log-decisions`: `resources/logs/game_<seed>.log`)
writes the RMLOG v3 log (seed, setup flags, both decklists, every asked decision); `--replay FILE`
replays it deterministically (the header alone fixes the setup; output flags such as
`--narrative`, `--trace-decisions`, `--rmstate-dump`, `--selfcheck-*` still apply).

## Architecture

### ECS

Based on [Austin Morlan's ECS tutorial](https://austinmorlan.com/posts/entity_component_system/):
entities are `uint32_t` IDs capped at `MAX_ENTITIES` (`src/ecs/entity.h`), components are data
structs (`src/components/`), systems iterate their signature's `mEntities` (an `EntitySet`, a bitset
iterated in id order like the `std::set` it replaced: `src/ecs/entity_set.h`; `src/systems/`), all
reached through the `global_coordinator` singleton (`src/ecs/coordinator.h`). The registered
components and systems are listed in `init_ecs()` (`src/game_driver.cpp`).
Globals: `global_coordinator` and `cur_game` (the current `Game`; both declared in
`src/game_driver.h`), `card_db` (uid → loaded card entity, `src/card_db.h`), and `RESOURCE_DIR`
(`getcwd() + "/resources"`, so the engine runs with cwd `bin/`, as the driver and the gate do).
**Per-object state lives on components; `Game` holds only game-scoped state and cross-object
registries keyed by `ObjectRef`** (`ObjectSet` / `ObjectMap`, e.g. a permission granted to a card).

**Components:** `CardData` (printed card: name, types, cost, P/T, ability definitions, statics,
replacements), `Zone` (location, owner, controller, `distance_from_top`), `Permanent` (on the
battlefield: controller, tapped, summoning sickness, abilities, statics, typed `counters`),
`Creature` (P/T, combat state), `Damage`, `Spell` (card entity on the stack), `Ability` (an
ability instance — see below; also a standalone stack entity for activated/triggered abilities,
`Orderer::push_ability_onto_stack`),
`Token`, `Player` (life, mana, per-turn counters), `EntryInfo` (what a cast or effect recorded
about a card's next battlefield entry — tapped, transformed, attacking, how it was cast, an Aura's
chosen object; consumed as its Permanent is built, dropped if it goes elsewhere); an object's color is read through
`effective_colors` (`src/queries/characteristics.h`). **Tokens have no `CardData`**
(Zone + Permanent + Creature + Damage + Token) — guard `CardData` reads on battlefield objects.
`Effect` is never instantiated; only its nested `Effect::Replacement` (parsed from `R:` lines into
`CardData::replacement_effects`) is live.

**Systems:** `Orderer` (zone moves, draw, shuffle, stack placement), `StateManager` (SBAs, turn-based
actions, triggers, statics, `determine_legal_actions`; split across `state_manager_*.cpp`),
`StackManager` (`resolve_top`). Non-System subsystems alongside them: the CR 613 layer engine
(`StateManager::apply_continuous_effects`, `state_manager_layers.cpp`, data model in
`continuous_effects.h`), replacement effects (`replacement::dispatch`, `replacement_effects.*`),
and rules-modifying prohibitions (`rules_mod::`, `rules_modifying.*`).

### Game flow

`Game` (`src/classes/game.h`; the `Step` enum UNTAP … CLEANUP) groups its state by concern:
`turn_state` (turn, step, active player, extra turns, cleanup progress), `priority` (holder, pass
flags), `combat`, `pending` (every suspended-decision state), `resolved_effects` (effects that
belong to no permanent, emblems), `rng`, `identity` (object stamps), plus the timestamp, delayed
triggers and last-known information; `reset_turn_counters` is the one turn-end reset of "this
turn" counts. `engine_main.cpp` parses flags and runs the no-game modes or the match; the loop is
`play_single_game` in `src/game_driver.cpp` (bo3 sequencing is `play_bo3_match`). Each
iteration: emit a parked `pending.query` → pregame stage
→ turn-based actions → mandatory choice (`proc_mandatory_choice`) → SBEs → `advance_step` (resolve
top of stack / next step once both players pass) → SBEs → `determine_legal_actions` (a lone pass is
auto-taken) → `InputLogger::get_input` (takes any other one-row menu itself, unasked) → `process_action`.

**Decisions must be suspendable.** With `-fno-exceptions` a mid-resolution prompt can't unwind, so
effect handlers ask through `FrameCtx` (`src/resolution_frame.h`): the query parks as a
`PendingQuery` (`src/pending_query.h`), `SUSPENDED` propagates to the loop, which re-emits it and
re-enters with the answer. All state lives in `cur_game`/ECS, never in statics, so a snapshot
(`src/snapshot.cpp`: the `--selfcheck-state` snapshot check, and search agents to come) covers it.

**Engine result lines:** `GAME_RESULT: N Player A|B wins` — or `GAME_RESULT: N draw`
when both players lose at once (CR 104.4a) — after every game, bo1 and bo3 (`print_game_result`);
`MATCH_RESULT: Player A|B wins X-Y` ends a bo3 match. A drawn bo3 game counts for neither player:
the match plays on until someone has won two games (fatal past `MAX_MATCH_GAMES`), and the player
who went first in the drawn game goes first again. Every game end goes through `Game::end_game`
(the per-game decision cap's, which is no rules outcome, through `Game::end_game_at_cap`).

### Ability resolution

**Definition vs instance** (`src/components/ability.h`). `AbilityDef` is what a script (or a
keyword / engine rule) says about an ability — category, costs, targeting spec, trigger
conditions, effect params (`params` variant, `ability_params.h`), `SubAbility$` / mode chains —
built by the parser and never changed afterwards. It lives in a process-wide append-only store
(`intern_ability_def` / `keyed_ability_def` / `derived_ability_def`), so a `const AbilityDef *`
stays valid across snapshot restores and per-game ECS resets; `CardData`/`Token` hold such
pointers, and card/token scripts and granted-ability bodies are parsed once per process. `Ability`
is one instance — `def` plus source, controller, targets, X, trigger-fire bindings, and instance
sub-ability/mode children mirroring `def`'s chains — so it is cheap to copy into a `LegalAction`,
`DelayedTrigger`, `FrameLevel` or trigger record. Never write through `def`: an engine-built
ability gets a keyed definition, a variant of an existing one a `derived_ability_def`, and
anything fixed per instance (target bounds, a mana ability's chosen color) is an instance field.
What a resolution remembers/chooses/names for its later instructions (remembered, imprinted,
chosen cards, named card, chosen number/type) is `cur_game.resolution.memory`; X and Converge
are read through `current_x_paid()` / `current_converge()` (`src/queries/spells.h`) — the cast or
activation in flight, else the resolving object's own values — never a global.

`resolve_ability()` (`src/resolution.cpp`) is a phased state machine that dispatches on the
definition's `EffectKind` (bound from its category string when the definition is interned) through
`effects::handler_for` (`effect_table.cpp`) to a per-effect handler in
`src/effects/effect_<name>.cpp`, then chains `SubAbility$`. Every category is one line in
`src/effects/effect_kinds.def`, which generates the enum, the category map, the handler
declarations and the dispatch table: a new category is that line plus its handler. The parser
warns on a script category with no entry (it would resolve as a no-op). Mana abilities
(`AddMana`) resolve at activation, off the stack.
Targets are chosen before costs are paid and re-checked at resolution.

**Last-known information (CR 400.7 / 608.2h).** An effect that reads a departed object's
characteristics AFTER the resolution that moved it (its own leaves/dies triggers, an ability whose
source it was resolving later) must use `departed_lki_for` (`src/queries/lki.h`), not
`effective_power`/`effective_*`: those read `lki_for`, whose snapshot is superseded when that
resolution ends and at the object's next zone change (tokens excepted, CR 111.7).

**Object identity (CR 400.7).** An entity id survives zone changes and is eventually reissued, so
any reference kept across a zone change, a resolution boundary or a turn is an `ObjectRef`
(`src/object_ref.h`: the entity plus its `Zone::obj_gen` stamp and the id's issue count, made with
`ObjectRef::of`), never a bare `Entity`; keyed state uses `ObjectSet` / `ObjectMap`. Read it with
`get()` and handle 0 (the object is gone or became a new object); `lki_entity()` is only for
last-known-information lookups, logging and link bookkeeping (0 once the id is issued to another
entity; last-known information is keyed the same way, `LastKnownInfo::issue`). Within one resolution the follow window keeps an object the
effect moved findable (CR 400.7j). Plain `Entity` is for values that live within one step.

**Name-a-card candidate set (deviation from CR 201.4; see `docs/rules_deviations.md`).** "Name a card" effects (Cabal Therapy,
Disruptor Flute, Petrified Hamlet) do **not** offer every card. `build_name_card_choices()`
(`src/name_card_choices.{h,cpp}`) returns a LIMITED set — the distinct vocab cards in the
relevant deck(s), filtered by `ValidCards$`. `NameCardScope` selects the source: `CHOOSER_ONLY`
(one player's whole deck) or `BOTH_PLAYERS` (either player's cards, de-duped — used for
land-naming), so a land-naming card can still name a land only in the opponent's deck.
Under `--closed-decklist` (register 11a) the set is instead every vocab card passing `ValidCards$`
(read from the parse cache, no entities), uncapped.

### Card scripts and parsing

Cards are Forge `.txt` scripts in `bin/resources/cardsfolder/<first letter>/<uid>.txt`, loaded on
demand by `load_card` (`src/card_db.cpp`) into the `card_db` map and parsed by
`parse_card_script` (`src/parse.cpp`): top-level `Name`/`ManaCost`/`Types`/`PT`/`Oracle`/
`Loyalty`, `K:` keywords, `A:` spell/activated abilities (`SP$`/`AB$ <category>`; `Mana` →
`AddMana`), `T:` triggers, `S:` statics, `R:` replacements, `SVar:` bodies (`DB$` sub-abilities
chained by `SubAbility$`). Per-ability `Key$ Value` params land in `apply_param_to_ability`
(fields on `AbilityDef` plus typed structs in `src/components/ability_params.h`). `name_to_uid`
lowercases, maps space/`-`/`/` to `_`, drops other characters, collapses `__`. Basic land types
get their mana ability from `StateManager::apply_land_abilities`, not the script.
Decks (`.dk`, `bin/resources/decks/`) are `<quantity> <card name>` lines, then an optional
`SIDEBOARD:` line and sideboard entries.

**Double-faced cards live under ONE combined `<front>_<back>.txt` script — never author a
front-name-only duplicate.** `load_card` resolves exact `<uid>.txt` first, else scans for a
combined `<uid>_*.txt` (e.g. `tamiyo_inquisitive_student_tamiyo_seasoned_scholar.txt`), and
aliases both face names. A front-name `<uid>.txt` alongside a combined script adds the card twice
and shadows the combined one.

**Fetching missing scripts — `tools/forge_fetch/fetch_script.py`** (the single correct way to
provision a script) pulls from Card-Forge/forge, add-only (`--force` to overwrite), **pinned** to
`tools/forge_fetch/FORGE_PIN` so every clone gets byte-identical scripts (bumping can break tests;
the file header holds the bump recipe; `--ref master` previews live Forge).
- `fetch_script.py "Brainstorm" "Tamiyo, Inquisitive Student"` — DFC-aware: a front-name miss
  discovers and fetches the combined filename. Tokens: `--token b_0_0_orc_army`.
- Accented names aren't transliterated for the filename/uid (`name_to_uid` drops non-ASCII) —
  fetch those by ASCII stem by hand. Vocab matching IS accent-insensitive (`ascii_fold_card_name`).
- `tools/forge_fetch/provision_decks.py` (run by CI and by the SessionStart hook, remote sessions
  only) fetches every vocab card plus the top-level/`meta/`/`league/` deck cards and their tokens,
  including engine-synthesized ones (Amass → `b_0_0_<subtype>_army`, Investigate →
  `c_a_clue_draw`, Mobilize → `r_1_1_warrior`). A card that synthesizes a NEW token kind needs its
  stem added to that script's `keyword_tokens`.

### Adding a New Card

1. Append `{"Card Name", N}` (next free index) to `card_vocab_entries` in `src/card_vocab.h`;
   `N_CARD_TYPES` (`src/card_vocab.h`) must exceed the highest index. Tokens are a separate band
   (`token_vocab_entries`, from index 900).
2. Provision the script, then `make`. `bin/debug/robomage_check --tier vocab` (the engine's
   `--check-decks`) lists any deck card that does not load or is missing from the vocab.

**Parse script tags as intended — do not retag them.** When a card needs a mechanic the
engine lacks, implement the mechanic so the parser honors the script's actual tags
(`SP$`/`AB$`/`DB$` category, `Origin$`/`Destination$`/`ChangeType$`/`DefinedPlayer$`, etc.).
Do NOT rewrite one category into another or force a different Origin/Destination to shortcut
a single card's behavior — a retag that happens to satisfy one card silently corrupts every
other card that shares the tag. Add a real, general handler keyed on the tag's intended
meaning. In limited cases it is acceptable to *ignore* an irrelevant tag when the card's full
functionality is inferable from the others (a cosmetic `StackDescription$`/`TgtPrompt$`, or a
`ChangeNum$` count-SVar when the effect already moves all matching cards); repurposing a tag is not.

### Machine protocol, state format and knowledge sidecar

`--machine` (also spelled `--machine-format rmstate`) selects, for each seat with no in-process
agent that is not a human seat, the RMSTATE stream (`src/state_format/stream.h`,
`docs/state_format.md` §9); it changes I/O only, never which decisions are asked or what a menu
offers. Mana payment is the match-wide **auto-mana** option, on by default for every consumer
(ordinary mana abilities hidden at priority, costs auto-paid, hybrid pips auto-resolved);
`--no-auto-mana` (recorded in the RMLOG header) lists every mana ability and pays costs through the
interactive payment loop (`prompt_mana_payment`):

- stdout carries only `RMSTATE <nbytes>\n` + payload records; everything else (narrative under
  `--narrative`, `GAME_RESULT` / `MATCH_RESULT`, diagnostics) goes to stderr.
- The deciding seat gets, the first time it decides in the match, its `SCHEMA`, `CARDS`,
  `DECKLISTS` and `METAGAME` frames, then a `STATE` frame (snapshot tables, the events since its
  last frame, the menu, its `k_*` knowledge tables) at each of its decisions.
- The reply is one integer line: a menu row, `-1` (the last row: an attacker/blocker
  declaration's confirm), `-2` / `-3` (concede). End of input or an unparseable line reads as `-1`.
- At match end each seat, A then B, gets a terminal `STATE` frame (`header.terminal = 1`, no menu,
  no reply); the engine then exits. Under `--replay` the log supplies every decision and no stream
  is written.

Every menu goes through `InputLogger::get_input`, which leaves out a choice interchangeable with
an earlier one (two identical cards in the same graveyard, or in the same library when each seat
sees both or neither right now — `interchangeable_cards`, `src/queries/zones.cpp`); menu row *i*
is reply *i*.

**Concession (CR 104.3a):** `-2` (`CONCEDE_GAME`) / `-3` (`CONCEDE_MATCH`) are accepted wherever a
decision is read (machine, `--replay`, in-process agents, CLI); the pending-decision seat loses via
the ordinary `Game::player_loses` path, and the sentinel is logged so it replays. `-3` also ends
the match (even from the sideboard phase, where `-2` is a no-op). Python constants in
`python/rm_driver.py` (`CONCEDE_GAME` / `CONCEDE_MATCH`); gate tier `concede`.

**Other frame writers:** `--rmstate-dump FILE` (every viewer's frames — A, B, omniscient — at each
decision, beside any protocol; `--state-dump json` adds `FILE.jsonl`), `--rmstate-read FILE`
(decode → re-encode byte check), `--cards-frame FILE` (the SCHEMA, whole-vocab CARDS and METAGAME
frames, the featurizer's card-table source; plays no game), the in-process agents' feed and
`--trace-decisions`. The format
is defined once in `src/state_format/schema.cpp`; decoders build themselves from the SCHEMA frame
(C++ `src/state_format/decode.h`, Python `python/rmstate.py`), so no constant is copied.

**Knowledge sidecar** (`src/knowledge/`, `docs/knowledge_store.md`): one instance per frame writer
and seat, fed only that seat's frames, appends the `k_*` tables (and at `expert+belief` the
`b_*` belief tables) to each of the seat's STATE frames; `--knowledge-a/-b off|expert|expert+belief`
(default `expert`) changes those tables only, never play. Self-checks: `--selfcheck-state` (the
format: roundtrip, leak scramble, complete, permutation, snapshot) and `--selfcheck-knowledge`
(every `k_*` fact against the true world); both print `ERROR:` lines and a summary on stderr and
change no other output. `bin/<config>/rmknow` runs the sidecar over a recorded stream.

### Key files

A where-is-X index. Module docstrings / header comments hold the detail; files described in the
sections above are not repeated here.

**Engine (`src/`)**
- `src/main.cpp` — `engine_main` alone; `src/engine_main.{h,cpp}` — the process entry (flags, the
  no-game modes, the match), shared with `robomage_nn`'s main; `src/game_driver.cpp` — the
  per-game decision loop (`play_single_game`, `play_bo3_match`), ECS init, test-setup presets,
  concession
- `src/action_processor.cpp` — executes a chosen action (`process_action`), cast/activation/targeting
  flows, `proc_mandatory_choice`
- `src/systems/state_manager*.cpp` — SBAs + `determine_legal_actions`, split by concern (`_actions`,
  `_combat`, `_layers` = CR 613 layer driver, `_statics`, `_triggers` = APNAP placement)
- `src/systems/replacement_effects.{h,cpp}` — CR 614/616 dispatcher; `rules_modifying.{h,cpp}` —
  cast/activate/land-play prohibitions
- `src/effects/` — one TU per resolution effect; `effect_table.cpp` dispatches `resolve_ability()` to them
- `src/resolution.cpp` — the resolution state machine; `src/targeting.cpp` — target legality and
  candidate lists; `src/zone_search.cpp` — library/graveyard/exile searches; `src/unless_payment.cpp`
  — unless-cost payment; `src/svar_eval.cpp` — the one SVar (`Count$` …) evaluator and comparator;
  `src/mana_system.cpp` — mana pools, payment and auto-pay
- `src/queries/` — shared entity queries, a header (+ `.cpp`) per concern: `battlefield` (live-permanent
  accessors, phasing rule), `characteristics` (face up, colors, mana value, `effective_*`, `entity_name`,
  `source_vocab_idx`), `types`, `keywords`, `counters`, `filters` (`MatchCtx`, the one filter matcher), `players` (seats,
  controller, `Defined$` player), `affected` (the objects / player an effect acts on:
  targets or `Defined$`), `player_resources` (life, energy), `player_effects`, `combat`, `damage`,
  `attachments` (equip), `activation`, `spells`, `zones` (graveyard/exile, play permissions, linked
  exile, `interchangeable_cards`), `lki` (object identity stamp, last-known info), `entry`
  (`EntryInfo`), `delayed_triggers`, `visibility` (what a viewer sees: `identity_visible`, reveal
  windows, `individually_identifiable`, `canonical_object_order`, and `visible_card_id` /
  `card_identity`, the state format's one identity read), `pending_cost` (the cost being paid at a
  decision, `CostPaymentScope`)
- `src/resolution_frame.h`, `src/pending_query.h` — suspension protocol that parks mid-resolution
  decisions (makes every prompt a loop-safe snapshot point)
- `src/classes/match_context.h`, `deck_state.h` — the bo3 state kept across the per-game ECS reset
- `src/file_util.{h,cpp}` — `make_dirs` (mkdir -p) and `write_file_atomic` (a temporary file renamed
  into place)
- `src/stable_rng.h` — every gameplay random draw (platform-portable; never `std::shuffle` /
  `std::uniform_int_distribution` / `rand()` on the game RNG)
- `src/state_format/handles.{h,cpp}` — per-viewer opaque handle tables (A, B, omniscient) on
  `Game`: issue, old → new links at zone changes, retirement; `--trace-handles`
- `src/state_format/schema.{h,cpp}` — the RMSTATE format's single definition (std-only): table/column
  definitions, the format's enums, the frame writer (debug-checked rows), SCHEMA frame and hash
- `src/state_format/enum_names.{h,cpp}` — the engine's enums as SCHEMA string tables, type-mask and
  keyword indices
- `src/state_format/encode.{h,cpp}` — engine → SCHEMA / CARDS / DECKLISTS / STATE frames per viewer
- `src/state_format/dump.{h,cpp}` — `--rmstate-dump FILE`: every viewer's frames at each decision
  (and `--state-dump json`'s `FILE.jsonl`)
- `src/state_format/stream.{h,cpp}` — the `--machine` RMSTATE stream (the deciding seat's frames on
  stdout) and the per-decision / match-end bookkeeping both frame writers share
- `src/state_format/decode.{h,cpp}` — RMSTATE bytes → plain structs (std + `schema.h` only): schema
  from the SCHEMA frame, generic table decode, re-encode, stream record reader
- `src/state_format/json.{h,cpp}` — `--state-dump json`: a decoded frame as one JSON line;
  `recode.{h,cpp}` — `--rmstate-read FILE` (decode → re-encode byte check, JSON to stdout)
- `src/state_format/events.{h,cpp}` — observation events: the `obs_*` hooks at the zone-move, draw,
  shuffle, reveal-window and game-boundary choke points, per-viewer projected queues on `Game`
  (match-long `seq`), drained into each STATE frame; `--trace-events`
- `src/state_format/metagame.{h,cpp}` — the reference metagame (`bin/resources/decks/archetypes.json`:
  labels and tagged reference lists) the `METAGAME` frame carries, read once per process
- `src/state_format/knowledge_feed.{h,cpp}` — the knowledge sidecar in the engine: one `Knowledge`
  per frame writer (dump, stream, self-check, agent feed) and seat, fed that writer's frames,
  appending `k_*` to each seat STATE frame; `--knowledge-a/-b`; snapshot sidecar hooks
  (`snapshot_set_sidecar_hooks`)
- `src/state_format/agent_feed.{h,cpp}` — the in-process frame writer: for each seat an in-process
  agent decides (or both under `--trace-decisions`) the bytes the RMSTATE stream would send it,
  `k_*` included, handed over in memory
- `src/state_format/selfcheck.{h,cpp}` — `--selfcheck-state`: per-decision roundtrip, scramble
  (frames and events rebuilt with hidden identities rotated), complete and, sampled, permutation
  (hidden zones physically permuted) and snapshot (save → K decisions → exact restore) checks;
  `ERROR:` lines, `SELFCHECK-STATE:` summary; `selfcheck_knowledge.{h,cpp}` — `--selfcheck-knowledge`
  (implies `--selfcheck-state`): every `k_*` fact of both seats against the true world built from
  the engine state, and the knowledge state across the snapshot check's restore;
  `SELFCHECK-KNOWLEDGE:` summary
- `src/snapshot.{h,cpp}` — in-process state snapshot/restore and the loop-safe decision marker
- `src/cli_output.cpp` — `game_log` (the narrative) and `cli_print_help`
- `src/cli/` — the C++ CLI: `board.h` renders one seat's decoded frames as text (board, pending
  decision and cost, menu, knowledge panel, zones, card details, events; also `--trace-decisions`),
  `human.h` the human seat (an in-process agent reading stdin: menu rows and the `board` / `zone` /
  `card` / `know` / `log` / `z` (auto-pass) / `concede` / `help` commands); std + the agents' frame
  reader only
- `src/cli_args.{h,cpp}` — the engine's flag table (parsing, bad-flag warnings, removed-flag
  errors, replay-header flags, which seats are human); keep `cli_print_help` in
  `src/cli_output.cpp` in sync
- `src/input_logger.cpp` — decision input (machine stream, `--replay`, the in-process provider of
  the agents and human seats), the interchangeable-choice collapse, RMLOG, concede sentinels,
  `--max-decisions`
- `src/card_vocab.h` — card name → vocab index (the implemented-card registry; the card id of the
  state format's `card` columns), `N_CARD_TYPES`, the token band
- `src/name_card_choices.{h,cpp}` — the name-a-card candidate set
- `src/text_table.{h,cpp}` — `--text-table-check` / `--text-table-inputs`: the oracle-text table's
  entries (per face vocab id, from the parse cache) and its staleness
- `src/featurize/` — std only (include-isolated like `src/knowledge/`): `json` (the shared JSON
  reader), `rmtb` (RMTB named-array files, `docs/featurizer.md` §4.4), `oracle_text` (text
  normalisation, text / vocab hashes, the model pin, staleness, the inputs file), `wordpiece`
  (the BERT WordPiece tokenizer); the featurizer (`docs/featurizer.md`): `columns` (the tag rules),
  `card_tables` (the card, ability and part tables from a whole-vocab CARDS frame, their RMTB
  files and the ability index), `feature_spec` (FEATURE_SPEC: token types, slots, relations,
  per-table column plans, `spec_json` / `spec_hash`), `featurizer` (`featurize_seat`, the actor
  entry point, refusing a non-seat frame; `featurize_viewer`, learner-side; the tensors file),
  `targets` (learner targets from the omniscient frame), `frame_stream` (a recorded stream frame by
  frame), `setup` (`Featurization`: the cards file and text table loaded, the tables exported),
  `batch` (the padding batcher: decisions → the network contract's `[B, …]` arrays), `episode`
  (the trajectory episode format: `EpisodeWriter`, `read_episode`; `docs/trajectory_format.md`)
- `src/state_format/cards_file.{h,cpp}` — `--cards-frame FILE`: the SCHEMA, whole-vocab CARDS and
  METAGAME frames for the card tables; plays no game. `load_engine_featurization`: the same
  records and the text table as the featurizer's inputs in process (`nn:`, `--featurize-seats`)
- `src/state_format/selfcheck_featurize.{h,cpp}` — `--selfcheck-featurize`: each seat's tensors
  from the actor entry point equal under the scramble and permutation checks' frames, the
  omniscient frame refused; `SELFCHECK-FEATURIZE:` summary
- `src/deck_check.{h,cpp}` — `--check-decks` (`--check-decks-dir`): every top-level / `league/` deck
  card loads and is in the vocab; plays no game
- `src/knowledge/` — the knowledge sidecar (`Knowledge`: one seat's recall and deduction from its RMSTATE
  frames, `k_*` rows; exact solver `solver.h`, compaction `compact.h`; archetype inference over the
  METAGAME frame's reference lists `templates.h` + `archetype.cpp` (`k_template` / `k_archetype`,
  §10); the shared soundness check `truth.h`; std + `state_format/decode.h`/`schema.h` only;
  `docs/knowledge_store.md` §9-10)
- `src/knowledge/belief.{h,cpp}` — the belief layer (level `expert+belief`, `b_list` / `b_archetype`):
  each reference list's likelihood as the weighted world count of its hypothesis network, the
  posterior in fixed point (`docs/knowledge_store.md` §10.6)
- `src/agents/` — in-process agents on the decision hook, reading only the seat's decoded frames
  (`agent.h`: the interface, specs `first`/`random`/`actions:`/`play:`/`explore`/`explore:patient`;
  `seat_frames.h`, `decision_view.h`: the decoded stream and decision; `py_random.h`: Python's
  `random.Random` bit for bit; `game_seed.h`: the driver's engine seed of a game seed;
  `explore.h`: `python/explore_agent.py` ported, same games; `policy_projection.h`: DeepNash's
  play-time thresholding and discretisation, the `nn:` agent's `thresh=` / `disc=`; `play_spec.h`: the `--play` resolver
  and its global script; `trace.h`: `--trace-decisions`; `register_agent_kind`: a kind a binary
  adds, `nn:` in `robomage_nn`). `driver.h` is the engine side:
  `--agent-a/-b`, `--play`, `--actions` (imply `--machine`), the human seats, the provider, the trace;
  `trajectory.h` too: the trajectory writer (`--trajectory-out DIR`, `--trajectory-seats`,
  `--trajectory-omniscient`), one episode file per game from the recorded seats' decoded frames, the
  deciding agent's `last_behaviour` (μ, versions, evaluations) and the game boundaries;
  `teacher.h`: the teacher channel (`--teacher-a/-b <command>`, `--teacher-mix p`,
  `--teacher-timeout s`): a seat's `--machine` stream to a subprocess whose replies are the
  trajectory's `teacher` labels, played with probability p on the teacher's own seeded coin;
  `actor.h`: the continuous actor (`--actor`, `--actor-schedule`, `--actor-out`, `--actor-id`,
  `--actor-seed`, `--actor-start`, `--actor-matches`; `docs/actor.md`): match after match drawn
  from a schedule (`actor_schedule.h`: entries, weights, seat swaps, purposes, outputs), the
  match-scoped resets, per-match seeds from (run seed, actor, match), the status and results files,
  SIGTERM / SIGHUP; the per-game cap `--max-game-decisions` (`cap_current_game`, game_driver.h)

**The actor (`src/actor/`, `make actor`; libtorch, exceptions on)** — `evaluator.{h,cpp}` (the
`Evaluator` interface: a batch of featurized decisions → per decision each policy's logits, `log π`,
value and version; the in-process one loads TorchScript or an AOTInductor `.pt2`, checks the
model's spec / vocab / text-model metadata), `remote_evaluator.{h,cpp}` (torch-free: the inference
protocol's client behind `nn:remote:`), `nn_agent.{h,cpp}` (the `nn:` agent: options, sampling
on the shared generator, the behaviour distribution, `dump=` / `evals=` / `stats`, the remote
options), `robomage_nn.cpp` (the binary's main: registers `nn:`, then `engine_main`).
`src/inference/protocol.{h,cpp}` (std + POSIX, not an engine TU; linked by `robomage_nn`, `rmserve`
and the gate) — the inference protocol's wire code (`docs/inference_protocol.md`): framing,
handshake JSON, the shared region, request / response layouts.

**Tools (`tools/`)**
- `tools/robomage_check/` — `bin/<config>/robomage_check` (part of `make`; `make check` runs it):
  the gate, engine subprocesses only (links the std-only decoder and frame reader, no engine TU);
  one `tier_*.cpp` per tier group, `scenarios.{h,cpp}` (the scenario keys and runner), `corpus.*`,
  `traj.*` (episode reading and comparisons, the `traj`, `actorloop` and `actor` tiers),
  `scan.cpp` (error/warning classification); flags `--tier`, `-j`, `--seed`, `--binary`,
  `--out-dir` (default `ci_out/`), `--fuzz-games`, `--smoke-games`, `--matchups`, `--mode`,
  `--scenario GLOB --show output|narrative|dump`, `--record`
- `tools/rmknow/` — `bin/<config>/rmknow` (part of `make`): the sidecar over a recorded stream
  (`--viewer A|B`, `--level`, `--text`, `--truth` vs the omniscient frames, `--isolation` vs the
  engine's appended `k_*`, `--stats`, `--selftest`, `--exactness N` vs an independent world enumerator,
  `--belief` / `--hide-lists` calibration lines, `--belief-exactness N` vs the weighted enumerator and
  `generative.h`, the belief model played out directly)
- `tools/rmfeat/` — `bin/<config>/rmfeat` (part of `make`; std + the decoder + `src/featurize/`, no
  engine TU): `--spec --cards CARDS`, `--export-tables CARDS OUTDIR`, `--featurize DUMP --viewer
  A|B|omniscient --out FILE`, `--targets DUMP --seat A|B --out FILE`, `--stats` (token counts,
  timing); `--tables DIR`, `--text-table FILE`
- `tools/rmembed/` — `bin/<config>/rmembed` (`make rmembed`, libtorch): the BERT encoder in the
  libtorch C++ API and its safetensors reader; writes the oracle-text table
- `tools/embed/` — `MODEL_PIN` (the text model's repo, commit, pooling, prefix), `fetch_model.py`
  (stdlib download into `.cache/embed/`), `make_goldens.py` (needs torch + transformers, never run
  by `make`: the tokenizer golden, the encoder and table checks against transformers)
- `tools/nn/make_reference_policy.py` — needs torch (+ onnx, onnxscript, onnxruntime), never run by
  `make`: `build` the network contract's reference policy from the exported tables (seeded) as
  TorchScript, `.pt2` and ONNX with the hashes; `check` the three exports agree on a tensors file
- `tools/rmserve/` — `bin/<config>/rmserve` (`make actor`, libtorch): the reference inference server
  (`--socket PATH --policy ID=MODEL … [--version ID=N] [--max-batch N] [--deadline-us N]
  [--threads N]`): many clients, batches across them, the in-process evaluator per policy,
  `RMSERVE-STATS:` on SIGTERM; `tools/nn/serve_check.py` — a minimal Python server of the same
  protocol (torch + numpy, never run by `make`; the `actor` tier plays a game against it)
- `tools/forge_fetch/` — pinned Forge script provisioning (above); `tools/meta_scraper/` — MTGTop8 →
  `.dk` scraper (stdlib only, its own README)

**Learner (`learn/`, torch, its own venv; `learn/README.md`)**
- `learn/rmtrain` — the training program (`rmlearn/rmtrain.py`, `rmlearn/orchestrator/`): `start` /
  `status` / `tail` / `pause` / `resume` / `stop` / `set` / `eval` / `phase` / `bench` over a run directory
  (`runs/<name>`), the phases B0–B5, one GPU process (learner + inference server) and the actors
  ([`docs/rmtrain.md`](docs/rmtrain.md)); checks `learn/check_rmtrain.py` (`--smoke`: a tiny run)

**Python (`python/`, stdlib only)**
- `scripted_core.py` — the rule-based scripted agent's decisions (`ScriptedCore`, all `scripted:*`
  tiers) over an abstract view of a decision (card names, engine enum names)
- `rm_agent.py` — the agent on RMSTATE: one seat's decoded frames and `k_*` tables as the core's
  view (`SeatFrames`, `FrameView`), card knowledge from `CARDS`' structured columns (`FrameCards`)
- `explore_agent.py` — the `explore` / `explore:patient` coverage fuzzer's decision rule
  (`explore_choice`, called by `ScriptedCore`; ported to C++ as `src/agents/explore.h`)
- `rm_driver.py` — the agent-game driver on `--machine`: engine subprocess, frame read, reply
  write; scripted tiers, explore, `first`, `actions:`; `run_games` / `play_game`; its own CLI
  (`CLI_FLAGS`); runs the `smoke` tier's games and the corpus recording
- `rmtb.py` — RMTB reader (`docs/featurizer.md` §4.4), stdlib `struct`
- `teacher.py` — the scripted agent (`--tier`, default hard) as a labeller on the machine protocol
  (stdin frames, stdout replies; the engine's `--teacher-a/-b`); reseeds `random` with `--seed` and
  resets at each match of the stream, `--record FILE` keeps the bytes read
- `rmtraj.py` — trajectory episode reader (`docs/trajectory_format.md`): each step's tensors
  expanded to the featurizer's layout
- `rmstate.py` — RMSTATE decoder built from the SCHEMA frame (stream reader, row dicts, the
  `--state-dump json` renderer); `rmstate.py dump FILE` prints JSON lines
- `record_corpus.py` — re-records the replay corpus with the live scripted agent, plus explore
  `--no-auto-mana` games (`make regen`)
- `pycheck.py` — the kept Python's checks (Python decoder vs the engine's JSON, refusals, the
  self-lethal guard, the driver's seeds and imports), run by the `smoke` tier (its `featurize` /
  `traj` parts by those tiers)

**Tests (`tests/`)**
- `tests/scenarios/*.json` — the rules-regression and state-format / knowledge sharpness scenarios
- `tests/cli/<name>.json` + `<name>.out` — the `cli` tier's sessions (engine flags and stdin lines)
  and their golden stdout (`robomage_check --tier cli --record` rewrites them)
- `tests/embed/tokens_golden.jsonl` — the text tokenizer's golden ids (`make_goldens.py tokens`)
- `tests/corpus/<name>.rmlog` + `<name>.replay` — the replay corpus (decision log and its replay
  baseline; `docs/ci.md`)

**Docs (`docs/`)** — `ci.md` (the gate), `game_running.md` (running games), `state_format.md`
(RMSTATE), `knowledge_store.md` (the sidecar), `featurizer.md` (the featurizer and the network
contract), `trajectory_format.md` (trajectory episodes), `actor.md` (the continuous actor, its
schedule, status and results files), `inference_protocol.md` (remote evaluation),
`rules_deviations.md`, `engine_fork_plan.md` (the plan), `rnad_training_plan.md` (the training plan); `step1_handoff.md` and `state_format_agent_map.md` are historical records of steps 1–2.
