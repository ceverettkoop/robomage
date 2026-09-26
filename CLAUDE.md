# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code Style

- Don't put new functions in main.cpp
- Don't edit my comments for spelling or punctuation. Only change them if something substantive changed.
- Do not include in comments explanations of how things were previously prior to a change. Document only how code currently behaves.
- Avoid inline logic for anything that will be repeated; write new functions that are reusable
- Declare local functions as private in the class, if the header contains a single class/struct, if header does not contain a class, write them as static functions in global namespace C-style.
- Iterate through mEntities when possible (working within a system class), rather than iterating through all entities
- Try to consolidate iterations through entities within a function, rather than iterating through many times
- To find battlefield permanents, use the shared accessors in `src/game_queries.h` — the
  `is_battlefield_permanent(entity, ctrl)` predicate as a loop guard / single-entity check,
  or `battlefield_permanents(mEntities, ctrl)` for the whole list — instead of open-coding
  the `Permanent` + `Zone` + `BATTLEFIELD` (+ controller) scan inline. These bake in the rule
  that a phased-out permanent is treated as though it doesn't exist (CR 702.26b), so do **not**
  add your own `is_phased_out` check at the call site. The only code that reads
  `Permanent::is_phased_out` directly is the phasing subsystem itself (the untap-step
  phase-in/skip in `classes/game.cpp`), the phase-out setter in `effects/effect_phases.cpp`,
  and the observation serializer in `machine_io.cpp` (which deliberately includes phased-out
  permanents, flagged).
  Add similar shared accessors (next to these) when a new entity-scan pattern starts repeating.
- Static (local) functions should be forward declared at top of source file for clarity
- C++17 with exceptions disabled (`-fno-exceptions`; only the libtorch actor TUs in `src/actor/` enable them)
- Two interactive front ends, both Python, both sitting on the shared driver in
  `train/game_driver.py`: the Textual TUI (`train/tui_game.py`) and the PySide6 GUI
  (`train/gui_main.py` app shell over the `train/gui_game.py` board, with its analysis window in
  `train/gui_analysis.py`), plus a plain text board (`play.py --board text`). There is NO C++
  front end (the old raylib GUI was removed); the engine is always a `--machine` subprocess.
- Uses clang-format configuration in `.clang-format`
- Every Python CLI flag lives in `train/cli_spec.py` (the TUI/GUI forms are built from it). The
  match format is one flag everywhere, `--format bo1|bo3` (default bo3, PPO training included).
  Seats use one vocabulary everywhere: a single seat's deck is `--deck-a` / `--deck-b` (multi-deck
  pools stay `--decks` / `--opponents`) and a seat's agent (an `opponents.make_controller` spec)
  is `--player-a` / `--player-b` — the interactive human is just the spec `human`. Where one side
  is "under test" (analysis, baseline, training) it is player A.
  When a flag is renamed or removed, add the old spelling to
  `cli_spec.REMOVED_FLAGS` (global or scoped to a tool / `tool/sub`) so it errors with
  "`--old` was removed; use …" instead of vanishing — parsers built with `apply_to_parser` pick the
  table up automatically, a standalone argparse calls `add_removed_flags(parser, scope)`. A removed
  environment variable goes in `cli_spec.REMOVED_ENV_VARS` (enforced by `check_removed_env`).
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

The python side of the project enables machine learning of the game and analysis.

## Rules Reference (authoritative)

The official **MTG Comprehensive Rules** are checked in at
[`docs/mtg_comprehensive_rules.txt`](docs/mtg_comprehensive_rules.txt) (navigation guide:
[`docs/mtg_comprehensive_rules.md`](docs/mtg_comprehensive_rules.md)). **Consult them whenever you
implement or test a mechanic** — they define correct behavior independent of the engine's current
state. `grep` the numbered rule rather than reading the 9k-line file (e.g.
`grep -nE "^509\." docs/mtg_comprehensive_rules.txt`; 702 keywords, 704 SBAs, 613 layers).

## Build Commands

- `make && make actor` — debug build: `bin/debug/robomage` + `bin/debug/az_actor` (the libtorch
  actor; `make actor` needs torch in `train/.venv` and is not part of plain `make`)
- `make BUILD=RELEASE && make actor BUILD=RELEASE` — same into `bin/release/` (separate obj/bin
  trees per config, so switching `BUILD` never needs `make clean`)
- `make clean` · `make check` (the test gate, below) · `make regen` (re-record the replay corpus
  after an intentional behavior change; see `docs/ci.md`)

**Which binary the Python tooling runs is independent of which one you just built.** Each tool
has a default tier; `ROBOMAGE_BUILD=debug|release` forces every tool onto one config, and
`--binary <path>` overrides one invocation:

- **Debug** (`bin/debug/robomage`; assertions, `-D_GLIBCXX_ASSERTIONS`, no `-O2`): the test
  harness, `ci_check`/`make check`, `train.py observe`, and `train.py baseline`'s Python fallback.
- **Release** (`bin/release/robomage`, `bin/release/az_actor`): `play.py` (every board),
  `analysis.py` (browse/report), every PPO/AZ training and `bench-*` subcommand, and `baseline`'s
  default C++-actor path.

**Codegen is part of the build.** `make` first runs `pygen` (`train/gen_enums.py`,
`gen_card_costs.py`, `gen_card_props.py`, `gen_archetypes.py`, `gen_sb_rules.py`; write-if-changed
via `train/gen_util.py`, so unchanged outputs force no recompile).
- **Committed** (derived from tracked sources; the `pygen` CI tier guards only these):
  `train/_enums.py` (C++ headers) and `src/gen/archetypes_gen.h` (`train/archetypes.py` +
  `bin/resources/decks/archetypes.json`).
- **Untracked**, rebuilt locally every build: `train/card_costs.py`, `train/card_props.py`,
  `src/gen/card_costs_gen.h` (from gitignored, Forge-fetched card-script content) and
  `train/sb_rules.py` + `src/gen/sb_rules_gen.h` (from `train/sb_dead_rules.json`). A vocab
  change + `make` keeps them in sync with nothing to commit.

## Long-running commands (Claude)

When Claude runs a long command (build, training, gate, fuzz, bench) it must run it **in the
background** and make the live output inspectable by the user:

- Always state the task's output-file path in chat when launching it, so the user can
  `tail -f` it while it runs. (No symlinks or other indirection — just the path.)

## Testing guidelines

- **The standard gate is `make check`** (build + every default tier of `train/ci_check.py`:
  codegen sync, vocab coverage, obs invariants, byte-identical replay corpus, deterministic league
  smoke, short fuzz, …). It must pass before pushing, and CI runs exactly it. Tiers, reproducing a
  CI failure, re-recording the corpus / regenerating codegen: [`docs/ci.md`](docs/ci.md).
- Non fatal errors are not acceptable
- Draws are not acceptable, outside of exceedingly rare cases, and require review
- Do not attempt to test cards that are not already in `src/card_vocab.h`. Cards absent from the card vocab are considered unimplemented.
- **Every game-running tool sits on `runner.drive_game` — do not hand-roll decision loops.**
  From Python use `runner.run_match(agent_a, agent_b, deck_a=…, deck_b=…, games=, bo3=True,
  seed=1, transcript="compact|verbose|narrative|quiet", out=…)`. Agent specs
  (`opponents.make_controller`, one grammar everywhere): `scripted` (= the **hard** tier: smart
  mulligan, combat simulation, combo lines), `scripted:easy`/`greedy`, `scripted:random`,
  `explore` / `explore:patient` (coverage fuzzer; big-mana mode), `auto`, `human`,
  `play:<specs>`, `actions:<ints>`, `gen` / `az:gen` / `azraw:gen` / `mcts:gen`, or a checkpoint
  path. Details: [`docs/game_running.md`](docs/game_running.md).
- **`train.py observe`** runs N games of one matchup between any two agents — the quick check of a
  new build. bo3 by default (`--format bo1` for single games); game i uses seed `--seed`+i.
  `--games N` (per-game results + W/L/D), `--verbose` (per-decision board + menu + narrative),
  `--out FILE` (transcript to FILE, one-line summary to stdout), `--quiet`, `--max-decisions N`.
  Choose `--deck-a/-b` that exercise what you changed (decks relative to `bin/resources/decks/`).
  - **Fuzz campaign** (any draw is a finding; also saved to `draw_<stamp>.txt`):
    `observe --player-a explore --player-b explore --deck-a league/ur_delver --deck-b league/gw_maverick --games 100 --verbose --out out.txt`
  - **Engine throughput:** `observe --format bo1 --games 40 --max-decisions 4000 --quiet --timing`

### Test harness for card behavior verification

`train/test_harness.py` runs the engine with `--machine --narrative` (narrative = the
perfect-information game log). At every decision it prints the narrative, the decoded board
state, and the numbered **Available actions** menu. Flags live in `cli_spec.HARNESS_TOOL`;
`--help` lists them all (zone presets `--battlefield/graveyard/exile/sideboard-a/-b`,
`--life-a/-b`, `--scenario`, `--merge-sideboard`, `--coverage-json`, `--log-decisions`, …).

- **Format:** bo3 by default — once the script runs out play auto-continues through games 2–3
  and sideboarding up to `--max-decisions` (default 1500). Sculpted scenarios usually want
  `--format bo1` (cap 500; required by `--merge-sideboard`).
- **Setting up cards:** `--hand-a/-b` + `--library-a/-b` build a stacked temp deck in
  `bin/resources/decks/temp/` (hand first; padded to 15 cards with the last library card; deleted
  at exit). `--battlefield-a/-b` permanents start in play without summoning sickness. Otherwise
  `--deck-a/-b <stem>` (relative to `bin/resources/decks/`; default `delver`). For an exact small
  library (e.g. Thassa's Oracle) write your own `1 CardName`-per-line `.dk` in `decks/temp/`
  (hand cards first, plus `--no-shuffle`); hand-made files there are not auto-deleted.
- **Shuffling:** libraries shuffle with the seeded RNG unless `--no-shuffle` (deck-file order =
  draw order; first 7 = opening hand). Any `--hand-a/-b` implies `--no-shuffle` for both seats.
- **No commas in card names.** List flags (`--play`, `--hand-*`, `--library-*`, zone presets)
  split on commas; card lookup strips punctuation, so write `Thalia Guardian of Thraben` (in
  `--play`, a unique comma-free substring such as `cast:thalia`). Apostrophes are optional.
- **Who decides:** `--play` (semantic specs; preferred) or `--actions` (positional indices;
  fragile) is ONE global script for both seats. It makes every decision until it runs out; then
  each seat's `--player-a/-b` agent decides (default `auto` = action 0, so the game
  auto-advances; e.g. `scripted` to hand over to the hard tier). **Claude cannot use `human`**
  (no TTY) — use `--play`. A `--play` run prints `resolved --actions: …` for exact replay; keep
  `--seed` fixed (default 1) while extending a line.

**`--play` grammar** (full list in the `train/action_spec.py` docstring). Each spec is resolved
against *that* decision's live menu by intent, so it survives menu reordering; an unmatched or
ambiguous spec **fails loudly with the legal menu**.

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
| `desc:<text>` · `#<n>` or bare integer | any action by description substring · literal index |

Card names match case- and apostrophe-insensitively; an exact name wins, else a unique substring
(`cast:bolt`). Identical duplicate choices collapse to the first; genuinely distinct matches are
ambiguous — pin them with `@own`/`@opp`, `desc:`, or `#<n>`. Start a sculpted-hand line with one
`keep` per seat.

**Seat keys** (`A:`/`B:`, case-insensitive; independent of `@own`/`@opp`) let you write each
player's line without hand-sequencing priority passes. If the next spec is keyed to the seat
lacking priority, the priority holder auto-passes (its mandatory choices, e.g. cleanup discard, go
to its `--player-a/-b` agent) until the keyed seat is on the clock. A keyed spec also passes its
own seat forward through priority windows until the action is legal (e.g. `A:attack:X` written in
main phase waits for declare-attackers); a typo'd/ambiguous spec, or one that hits a mandatory
choice it doesn't match, still fails loudly. Unkeyed specs apply to whoever has priority, with no
auto-advance. **The engine never asks a seat whose only legal action is a pass**, so write `pass`
only where that seat could do something else.

```bash
# Sculpted hands, script the opening, then scripted agents play on
train/.venv/bin/python train/test_harness.py --format bo1 \
  --hand-a "Mountain,Lightning Bolt" \
  --library-a "Island,Island,Mountain,Mountain,Mountain,Mountain,Mountain,Mountain" \
  --hand-b "Forest,Grizzly Bears" --library-b "Forest,Forest,Forest,Forest,Forest,Forest,Forest,Forest" \
  --play "keep,keep" --player-a scripted --player-b scripted --max-decisions 30

# Bolt a bear already in play, seat-keyed (B has no response, so it gets no decision)
train/.venv/bin/python train/test_harness.py --format bo1 \
  --hand-a "Lightning Bolt" --battlefield-a "Mountain" --battlefield-b "Grizzly Bears" \
  --play "A:keep,B:keep,A:cast:Lightning Bolt,A:target:Grizzly Bears@opp"
```

`train.py observe --player-a "play:<specs>"` drives one seat by specs against any agent (each list
drives only its seat, so leave specs unkeyed there).

**JSON scenarios** (`--scenario FILE`) capture a reproducible setup: keys `name`, `hand_a/b`,
`library_a/b`, `battlefield_a/b`, `graveyard_a/b`, `exile_a/b`, `sideboard_a/b`, `life_a/b`,
`play` (spec list) or `actions`, `seed`, `max_decisions`. Command-line flags override scenario
values; format and players come only from the command line.
```json
{"name": "bolt_kills_bear", "hand_a": ["Mountain", "Lightning Bolt"],
 "battlefield_b": ["Grizzly Bears"],
 "play": ["keep", "keep", "play:Mountain", "cast:Lightning Bolt", "target:Grizzly Bears@opp"], "seed": 1}
```

## Architecture

### ECS

Based on [Austin Morlan's ECS tutorial](https://austinmorlan.com/posts/entity_component_system/):
entities are `uint32_t` IDs capped at `MAX_ENTITIES` (`src/ecs/entity.h`), components are data
structs (`src/components/`), systems iterate their signature's `mEntities` (`src/systems/`), all
reached through the `global_coordinator` singleton (`src/ecs/coordinator.h`). The registered
components and systems are listed in `init_ecs()` (`src/game_driver.cpp`).
Globals: `global_coordinator` and `cur_game` (the current `Game`; both declared in
`src/game_driver.h`), `card_db` (uid → loaded card entity, `src/card_db.h`), and `RESOURCE_DIR`
(`getcwd() + "/resources"`, so the engine runs with cwd `bin/`, as the Python drivers do).

**Components:** `CardData` (printed card: name, types, cost, P/T, ability/static/replacement
templates), `Zone` (location, owner, controller, `distance_from_top`), `Permanent` (on the
battlefield: controller, tapped, summoning sickness, abilities, statics, typed `counters`),
`Creature` (P/T, combat state), `Damage`, `Spell` (card entity on the stack), `Ability` (also a
standalone stack entity for activated/triggered abilities, `Orderer::push_ability_onto_stack`),
`Token`, `Player` (life, mana, per-turn counters), `ColorIdentity`. **Tokens have no `CardData`**
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

`Game` (`src/classes/game.h`; the `Step` enum UNTAP … CLEANUP) holds turn/step, active player,
timestamp, seed + `mt19937`, delayed triggers, and every suspended-decision state. `main.cpp` only
parses flags; the loop is `play_single_game` in `src/game_driver.cpp` (shared with `az_actor`;
bo3 sequencing is `play_bo3_match`). Each iteration: emit a parked `pending_query` → pregame stage
→ turn-based actions → mandatory choice (`proc_mandatory_choice`) → SBEs → `advance_step` (resolve
top of stack / next step once both players pass) → SBEs → `determine_legal_actions` (a lone pass is
auto-taken) → `InputLogger::get_input` → `process_action`.

**Decisions must be suspendable.** With `-fno-exceptions` a mid-resolution prompt can't unwind, so
effect handlers ask through `FrameCtx` (`src/resolution_frame.h`): the query parks as a
`PendingQuery` (`src/pending_query.h`), `SUSPENDED` propagates to the loop, which re-emits it and
re-enters with the answer. All state lives in `cur_game`/ECS, never in statics, so a snapshot
(MCTS search, `src/snapshot.cpp`) covers it.

### Ability resolution

`Ability::resolve()` (`src/components/ability.cpp`) is a phased state machine that maps the
category string to an `EffectKind` (`src/effects/effect_kind.{h,cpp}`), dispatches through
`effects::handler_for` (`effect_table.cpp`) to a per-effect handler in
`src/effects/effect_<name>.cpp` (declared in `effects.h`), then chains `SubAbility$`. An unmapped
category silently resolves as a no-op, so a new category needs all four: enum member, string
mapping, handler, table case. Mana abilities (`AddMana`) resolve at activation, off the stack.
Targets are chosen before costs are paid and re-checked at resolution.

**Last-known information (CR 400.7 / 608.2h).** An effect that reads a departed object's
characteristics AFTER the resolution that moved it (its own leaves/dies triggers, an ability whose
source it was resolving later) must use `departed_lki_for` (`src/game_queries.h`), not
`effective_power`/`effective_*`: those read `lki_for`, whose snapshot is superseded when that
resolution ends and at the object's next zone change (tokens excepted, CR 111.7).

**Name-a-card candidate set (deviation from CR 201.4).** "Name a card" effects (Cabal Therapy,
Disruptor Flute, Petrified Hamlet) do **not** offer every card. `build_name_card_choices()`
(`src/name_card_choices.{h,cpp}`) returns a LIMITED set — the distinct vocab cards in the
relevant deck(s), filtered by `ValidCards$`. `NameCardScope` selects the source: `CHOOSER_ONLY`
(one player's whole deck) or `BOTH_PLAYERS` (either player's cards, de-duped — used for
land-naming), so a land-naming card can still name a land only in the opponent's deck.

### Card scripts and parsing

Cards are Forge `.txt` scripts in `bin/resources/cardsfolder/<first letter>/<uid>.txt`, loaded on
demand by `load_card` (`src/card_db.cpp`) into the `card_db` map and parsed by
`parse_card_script` (`src/parse.cpp`): top-level `Name`/`ManaCost`/`Types`/`PT`/`Oracle`/
`Loyalty`, `K:` keywords, `A:` spell/activated abilities (`SP$`/`AB$ <category>`; `Mana` →
`AddMana`), `T:` triggers, `S:` statics, `R:` replacements, `SVar:` bodies (`DB$` sub-abilities
chained by `SubAbility$`). Per-ability `Key$ Value` params land in `apply_param_to_ability`
(fields on `Ability` plus typed structs in `src/components/ability_params.h`). `name_to_uid`
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
   `N_CARD_TYPES` (`src/machine_io.h`) must exceed the highest index. Tokens are a separate band
   (`token_vocab_entries`, from index 900).
2. Provision the script, then `make` (its `pygen` step regenerates the untracked
   `train/card_costs.py`, `train/card_props.py`, `src/gen/card_costs_gen.h`; without a build run
   `train/gen_card_costs.py` / `train/gen_card_props.py`). Nothing to commit; a new card only adds
   a property row, so the network shape and checkpoints are unaffected.

**Parse script tags as intended — do not retag them.** When a card needs a mechanic the
engine lacks, implement the mechanic so the parser honors the script's actual tags
(`SP$`/`AB$`/`DB$` category, `Origin$`/`Destination$`/`ChangeType$`/`DefinedPlayer$`, etc.).
Do NOT rewrite one category into another or force a different Origin/Destination to shortcut
a single card's behavior — a retag that happens to satisfy one card silently corrupts every
other card that shares the tag. Add a real, general handler keyed on the tag's intended
meaning. In limited cases it is acceptable to *ignore* an irrelevant tag when the card's full
functionality is inferable from the others (a cosmetic `StackDescription$`/`TgtPrompt$`, or a
`ChangeNum$` count-SVar when the effect already moves all matching cards); repurposing a tag is not.

## Reinforcement Learning

`train/` holds the Python side: gymnasium env, PPO (`MaskablePPO`, sb3-contrib) and AlphaZero
training, analysis and front ends. Venv: `train/.venv/` (invoke `train/.venv/bin/python`).

**Engine result lines (machine mode):** `GAME_RESULT: N Player A|B wins` after each game;
`MATCH_RESULT: Player A|B wins X-Y` ends the match.

**Reward (Player A's perspective, per game):** ±1.0 per game (`GAME_WIN_REWARD`/`GAME_LOSS_REWARD`,
`train/env.py`); in bo3 it lands at every `GAME_RESULT`. The match reward (`MATCH_WIN_REWARD`/
`MATCH_LOSS_REWARD`) is 0.0 — `MATCH_RESULT` only ends the episode. Per-game rewards match the
AlphaZero outcome target, so a PPO checkpoint warm-starting an AZ net (`az_net.from_ppo`) hands
over a calibrated critic; AZ trains value on `(1 - q_mix) * z + q_mix * td_q` (n-step TD target;
`--td-n` / `--q-mix`, see `cli_spec.py`). Shaping is capped per game (`SHAPING_EPISODE_CAP`).
PopArt (`train/popart.py`) normalizes each archetype bucket's value targets, on by default;
`--no-popart` disables it and `--stock-head` implies `--no-popart`. Its stats live in policy
buffers and updates are output-preserving, so checkpoints resume under either setting.

**Bo3 state-vector fields** (indices in the `src/machine_io.h` layout block): match context
(`game_number`, match wins, `is_sideboard_phase`; all 0.0 in bo1), library counts, known top-5
library cards, and a match-scoped `revealed` bit per opponent registered-decklist slot (the
deterministic belief state, persisted across the per-game ECS reset; `src/classes/match_state.{h,cpp}`).

### Machine mode protocol

`--machine`: at each decision the engine prints a `BQUERY` header + binary payload on stdout and
reads one integer on stdin; other stdout lines are narrative. Emitter: `emit_machine_frame` in
`src/cli_output.cpp` (layout comment at the top of `src/machine_io.h`).
```
BQUERY: <N> <STATE_SIZE> <MAX_ACTIONS>\n
float32[STATE_SIZE] state   (priority player's = "self" perspective)
int32  [MAX_ACTIONS] cats   (ActionCategory)
float32[MAX_ACTIONS] ids    (vocab_idx / N_CARD_TYPES; -1/N_CARD_TYPES = null)
float32[MAX_ACTIONS] ctrl   (1 self / 0 opp / null for non-entity)
float32[MAX_ACTIONS] pub    (card identity public; side-channel, NOT in the obs)
int32  [MAX_ACTIONS] zone   (ActionRefZone)
int32  [MAX_ACTIONS] refs   (entity-reference slot, -1 = none)
int32  [MAX_ACTIONS] ords   (mode/X/color/ability index, -1 = n/a)
```
Arrays are padded to `MAX_ACTIONS`. The header's sizes are a layout handshake the Python driver
asserts. Under `--narrative` the frame also carries per-action description strings and
per-permanent counter / token-name strings (display only). `--broadcast-steps` emits the same
payload as `BSTATE:` frames that take no reply.

**Source of truth (never duplicate values here or re-spell them as literals):**
- Layout, `STATE_SIZE`, `N_CARD_TYPES`, per-slot field order: `src/machine_io.h`. Card identity
  is one normalized id float (`norm_card_id`), not a one-hot.
- Block widths: `machine_io.h`'s "State-vector block widths", mirrored to `train/_enums.py` by
  `gen_enums.py` — import them in `env.py`/`extractor.py`/`obs_builder.cpp`.
- Block offsets: `machine_io.h`'s `OFFSET_CHAIN`; `env.py` derives the same chain, the `actorobs`
  ci tier compares them block-by-block, and `extractor.py` asserts vs `env.py` at import.
- Per-action obs positions: `env.py`'s `ACT_*_START` constants — never `STATE_SIZE + k*MAX_ACTIONS`.
- `ActionCategory`: `src/classes/action.h` (mirrored to `_enums.py`, incl. `ACTION_CATEGORY_MAX`).
- `OBS_SIZE` = `STATE_SIZE + N_ACTION_OBS_BLOCKS*MAX_ACTIONS` + matchup tail (`train/env.py`);
  `N_ACTION_OBS_BLOCKS` (`machine_io.h`) alone says how many per-action arrays fold in (`pub` is not).
  Per-card cost facts live in the network's frozen `card_props`, not the obs.

**Confirm slot:** mandatory attacker/blocker menus end with a confirm action; the env remaps
`num_choices - 1` to `-1` before sending.

**Concession (CR 104.3a):** `-2` (`CONCEDE_GAME`) / `-3` (`CONCEDE_MATCH`) are accepted wherever a
decision is read (machine, `--replay`, actor, CLI); the pending-decision seat loses via the
ordinary `Game::player_loses` path, and the sentinel is logged so it replays. `-3` also ends the
match (even from the sideboard phase, where `-2` is a no-op). Python constants in `train/env.py`;
`GameDriver.concede(match=…)`; regression tier `concede`.

**Perspective flags:** the header has "priority player is active" (relative) and "self is Player A"
(absolute); their XNOR is `active_is_a`. The seat flag is zeroed out of both network trunks
(`extractor.network_global_ctx`) so the net cannot learn seat-specific play.

**No action history in the obs** (it leaked hidden picks). What it carried is now explicit fields:
pass flags / `is_priority_window` / mulligan state, per-turn counters, per-permanent turn statuses,
pending delayed triggers, player effects, zone play permissions, and the pending-decision source.

### Interactive front ends: TUI, GUI, and the analysis window

`./tui.sh` (`train/tui.py`) is the Textual control panel (decks, training, league, observe, play).
`train/play.py` is the one play entry point (`--help` lists flags; defaults: `--board gui`
(falls back to tui without PySide6), human on A vs `az:gen` on B, `league/bug` vs
`league/ur_delver`, `--on-the-play a|b|random` via `cli_spec.resolve_on_the_play`/`seat_play_sides`).
Exactly one seat is `human`. A flag a board cannot honour errors. The gui/tui boards share
`train/game_driver.py` (`GameDriver`, `build_session`); `--board text` uses `runner.run_games` with
a `HumanController`. `./gui.sh` opens the GUI app's welcome pane.

- **The GUI mirrors the CLI.** Play/Analysis dialog fields are `play.py` / `analysis.py browse`
  flags with the same dests and cli_spec defaults, tabled in `train/launcher_config.py`
  (`test_cli_spec.py` asserts field ↔ flag parity); settings in `~/.robomage/gui_launcher.json`.
  A new GUI knob gets a cli_spec flag first, then a field. The search-knob → `az:`/`mcts:` spec
  fold is `cli_spec.search_knob_pairs` / `apply_search_knobs`, shared by every caller.
- **Analysis browser** (`analysis.py browse`, `--board tui|gui`): `--source` (`cli_spec.browse_source_kind`)
  is `simulate` (default), a `shard_*.npz` dir, or a `.rmtrace`; flags invalid for the kind error
  (`cli_spec.BROWSE_SOURCE_DESTS`). `--games 0` on shards is refused past
  `shard_replay.MAX_UNBOUNDED_SHARD_BYTES` — training pools like `az_data/gen` are ~100 GB.
  Both boards (`tui_analysis.py`, `gui_browser.py`) sit on `train/browse_session.py`: **a new
  browser feature goes there first**, boards add only rendering. Tier `browser`.
- **Headless smokes:** `ROBOMAGE_SMOKE=<legs>` — see `docs/ci.md` (`gui` tier). Never point
  `browser`/`tree` legs at a training pool.
- **Analysis window** (GUI play board, live MCTS on a detached engine): `train/gui_analysis.py` over
  the Qt-free `train/analysis_session.py`; opt-in tier `analysis`.
- **Shard recording** (`play.py --record-shards`): `train/shard_record.py` (exactly replayable via
  `.rmplay` sidecars); exact search-tree rebuild `tree_rebuild.py` / `tree_cache.py`; net probes
  `shard_probes.py`. Search-vs-net KL / top-1 numbers all come from `decode.search_net_divergence`.

### Key files

A where-is-X index. Module docstrings / header comments hold the detail; files described in the
sections above are not repeated here.

**Engine (`src/`)**
- `src/game_driver.cpp` — the per-game decision loop (`play_single_game`, `play_bo3_match`), ECS init,
  concession; shared by `main.cpp` and `az_actor`
- `src/action_processor.cpp` — executes a chosen action (`process_action`), cast/activation/targeting
  flows, `proc_mandatory_choice`
- `src/systems/state_manager*.cpp` — SBAs + `determine_legal_actions`, split by concern (`_actions`,
  `_combat`, `_layers` = CR 613 layer driver, `_statics`, `_triggers` = APNAP placement)
- `src/systems/replacement_effects.{h,cpp}` — CR 614/616 dispatcher; `rules_modifying.{h,cpp}` —
  cast/activate/land-play prohibitions
- `src/effects/` — one TU per resolution effect; `effect_table.cpp` dispatches `Ability::resolve()` to them
- `src/game_queries.h` — shared entity queries (battlefield accessors, filters, `effective_*`, LKI)
- `src/resolution_frame.h`, `src/pending_query.h` — suspension protocol that parks mid-resolution
  decisions (makes every prompt a search root)
- `src/snapshot.cpp`, `src/search_server.cpp` — state snapshot/restore and the `--search-server` MCTS
  protocol
- `src/machine_io.{h,cpp}` — state-vector layout/constants and `serialize_state`/`populate_query`;
  `src/cli_output.cpp` writes the BQUERY frame
- `src/cli_args.{h,cpp}` — the engine's flag table (parsing, bad-flag warnings, replay-header
  flags); keep `cli_print_help` in `src/cli_output.cpp` in sync
- `src/input_logger.cpp` — decision input (machine, `--replay`, CLI, actor provider), RMLOG, concede
  sentinels
- `src/card_vocab.h` — card name → vocab index (emitted as a normalized id float, not a one-hot)

**AlphaZero C++ actor (`src/actor/`, `make actor` → `bin/<config>/az_actor`)** — each file is a twin of a
Python function and MUST change in lockstep with it (bit-parity: opt-in tier `actor`; layout: `actorobs`):
- `obs_builder.cpp` ↔ `env.py` obs (+ `extractor.py`, `az_net.ScriptTrunk`); `az_mcts.cpp` ↔
  `mcts.run_search`; `menu_merge.h` ↔ `decode.menu_merge_reps`; `sb_rules.h` ↔ `mcts.sb_dead_mask`;
  `td_targets.cpp` ↔ `az_selfplay.compute_td_targets`; `root_diag.h` ↔ `shard_record` diag
- `az_evaluator.cpp` (TorchScript AZNet), `oracle_client.cpp` (↔ `train/scripted_oracle.py`),
  `npz_writer.cpp` (shards)

**Python core (`train/`)**
- `env.py` — `RoboMageEnv` / `ModelVsScriptedEnv` / `SelfPlayEnv`; obs offsets; re-exports
  `scripted_action` (back-compat)
- `scripted_agent.py` — the rule-based `ScriptedAgent` (all `scripted:*` tiers)
- `runner.py` — `drive_game` / `run_games` / `run_match` (see `docs/game_running.md`)
- `opponents.py` — `Controller`, `make_controller` spec grammar, training pools, and THE model-spec
  resolver (`parse_model_spec` / `strip_spec_knobs` → `load_spec_evaluator` / `load_spec_net` /
  `load_spec_value_model`: bare / `mcts:` = the PPO checkpoint, `az:`/`azraw:`/`.pt` = the AZ
  warm-start ladder); use it instead of stripping prefixes/knobs by hand
- `decode.py` — obs/action decoders shared by every front end (torch-free)
- `cli_spec.py` — every CLI flag and `Tool`/subcommand definition (stdlib-only)
- `extractor.py` — `CardGameExtractor` policy trunk; `popart.py` — per-bucket value normalization
- `archetypes.py` — deck → archetype / value-bucket metadata (`bin/resources/decks/archetypes.json`)
- `action_spec.py` — `--play` semantic spec resolver (see Test harness)

**Training (`train/`)**
- `train.py` — CLI for PPO (`train`, `league`, `exploiter`, …), `observe`, `baseline`, `curriculum`,
  the `az-*` drivers and `bench-*`
- `curriculum.py` — multi-phase plan schema + runner (stdlib-only); `progress_io.py` — atomic JSON
  resume sidecars
- `mcts.py` — determinized PUCT (`run_search`, `run_search_parallel`, `IncrementalSearch`);
  duplicate-edge merge on by default (`merge_dupes`), dead-sideboard-IN pruning (`sb_dead_mask`)
- `sb_dead_rules.json` — hand-edited dead-sideboard-card rules, compiled by `gen_sb_rules.py`;
  self-play boards by net prior (`--sb-selfplay-mode prior`), gates/eval/play by plan search
- `az_net.py` (AZNet, `from_ppo` warm start), `az_selfplay.py` (self-play shards), `az_train.py`
  (`train_az`, `az_eval` gate, `az_league`), `gate_sprt.py` (gate SPRT)
- `az_eval_server.py` — batched GPU inference server for the actor fleet; `scripted_oracle.py` —
  scripted:hard over a socket for the actor
- `az_baseline.py` — engine of `train.py baseline`
- `search_env.py` — `SearchRoboMageEnv` (snapshot-protocol client, mirror pool)

**Analysis / inspection (`train/`)**
- `analysis.py` — `browse` (analysis browser) and `report` (HTML battery); `viz.py` — its chart helpers
- `az_inspect.py` — static AZ-checkpoint views, a subcommand each (list in its docstring); `tui` →
  `tui_az_inspect.py`
- `shard_replay.py` — rebuilds browsable matches (and `.rmplay` replays) from shards

**Codegen (`train/`, run by `make`)**
- `gen_enums.py` → `_enums.py`; `gen_archetypes.py` → `src/gen/archetypes_gen.h`; `gen_sb_rules.py` →
  `sb_rules.py` + `src/gen/sb_rules_gen.h`
- `gen_card_costs.py` → `card_costs.py` + `src/gen/card_costs_gen.h`; `gen_card_props.py` →
  `card_props.py` (103 frozen property columns, concatenated with the 32-dim learned embedding).
  Never hand-edit the outputs.
- `missing_cards.py` — deck cards absent from `src/card_vocab.h`

**Tests / CI (`train/`)**
- `ci_check.py` — `make check` driver; default `ALL_TIERS` vs `OPT_IN_TIERS` (`actor`, `analysis`,
  `treerebuild`, `azinspect`, `gui`); its module docstring maps each tier to its `test_*.py`
- `test_harness.py` — card-behavior harness (see Test harness)
- `test_obs_invariants.py` — structural invariants on the raw state vector (tier `obsinv`)
- `test_model_spec.py` (`modelspec`), `test_curriculum.py` (`curriculum`), `test_shard_record.py`
  (`shardrec`), `test_analysis_session.py` + `test_browse_session.py` (opt-in `analysis`),
  `test_az_inspect.py` (opt-in `azinspect`)
- Fuzz campaigns are `train.py observe --player-a explore --player-b explore` (see Testing guidelines)
