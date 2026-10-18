# robomage

A Magic: The Gathering rules engine in C++17 (an Entity Component System over Forge card
scripts), for two-player games. Games are deterministic given a seed, every decision is a choice
from a finite menu and is logged as an integer, and a decision log replays the game exactly.

Agents drive the engine three ways:

- **in-process agents** (`--agent-a/-b`: `first`, `random`, `explore`, `explore:patient`,
  `actions:…`, `play:…`) and the semantic `--play` script used for sculpted tests;
- the **terminal prompt**: a human deciding at the terminal (`--player A|B` picks whose hidden
  information is shown);
- the **machine protocol** (`--machine`): **RMSTATE**, an exhaustive binary game state sent to the
  deciding seat with everything hidden from it withheld, answered with one integer per decision.
  Each seat's frames carry the **knowledge sidecar**'s tables: what that seat can remember and
  soundly deduce from its own frames (perfect recall, exact inference).

The rule-based **scripted agent** (Python, standard library only) plays over that protocol; it is
the realistic opponent, and its games are the replay corpus the test gate checks byte for byte.

The decks under `bin/resources/decks/` (top-level and `league/`) are fully implemented. Cards
outside `src/card_vocab.h` are considered unimplemented; rules fidelity is not guaranteed for
them. `todo.md` lists open engine correctness items, `docs/rules_deviations.md` the known
deviations from the Comprehensive Rules.

The reinforcement-learning stack this engine was first built for (PPO / AlphaZero training, the
float observation vector, the libtorch self-play actor, the TUI and GUI) has been archived: it
lives on branch `archive/rl-stack` (tag `rl-stack-final`, if created; commit `cf7f97f` is the last
one with both stacks). `docs/engine_fork_plan.md` is the plan this repo follows from here (a single
decision schedule and a C++ CLI on the state decoder, then a C++ featurizer and in-process actor
for learned models).

## Prereqs

- A C++17 compiler and GNU make (tested on Linux and macOS; the engine is C++17 + the standard
  library).
- `python3` (3.10+, standard library only; CI uses 3.12) for card-script provisioning, the agent
  driver, the corpus recorder and the gate's `smoke` tier. No venv, no packages.

## Building

Card scripts live in `bin/resources/cardsfolder/` and `bin/resources/tokenscripts/` (Forge
format, not tracked in git, pinned to a Forge commit in `tools/forge_fetch/FORGE_PIN`). Fetch them
first:

```bash
python3 tools/forge_fetch/provision_decks.py   # every vocab card + the league/meta/top-level deck cards + their tokens
make                  # debug build   -> bin/debug/{robomage,rmknow,robomage_check}
make BUILD=RELEASE    # optimized     -> bin/release/{robomage,rmknow,robomage_check}
make check            # the test gate (below)
```

Each configuration has its own `bin/<config>/` and `obj/<config>/` tree, so switching needs no
`make clean`. `bin/<config>/rmknow` runs the knowledge sidecar over a recorded RMSTATE stream.
`make actor` compiles the libtorch TorchScript evaluator kept for the future in-process actor (no
binary yet; needs a torch importable by `python3` or `LIBTORCH_DIR=…`).

## Running games

The engine must be run from `bin/` (it finds `resources/` relative to the working directory).
`./debug/robomage --help` lists every flag.

```bash
cd bin
./debug/robomage --deck-a delver --deck-b mav                    # interactive: you decide for both seats
./debug/robomage --deck-a league/bug --deck-b league/burn \
  --agent-a explore --agent-b explore --game-seed 5 --decision-log /tmp/g5.rmlog    # an explore game, logged
./debug/robomage --replay /tmp/g5.rmlog --machine --narrative     # replay it
```

At the terminal prompt a number selects a choice (every choice is logged); `--player A|B` shows
hidden information from that seat's view only. Deck names are paths under
`bin/resources/decks/` without `.dk` (`delver`, `league/ur_delver`). A human seat against an
agent is plan step 4b's C++ CLI: today a `--player` seat under `--machine` (which any
`--agent-a/-b` implies) is prompted without its menu being printed.

**Agent games with the scripted agent** go through the stdlib-only driver, from the repo root:

```bash
python3 python/rm_driver.py --player-a scripted --player-b explore \
  --deck-a league/bug --deck-b league/burn --format bo1 --games 10
python3 python/rm_driver.py --player-a scripted --player-b scripted \
  --deck-a league/ur_delver --deck-b league/gw_maverick --verbose --out match.txt   # a bo3 match, full transcript
```

Agents: `scripted` (hard), `scripted:easy` / `greedy`, `scripted:random`, `explore`,
`explore:patient`, `first`, `actions:<i,i,…>`. `--help` lists the flags;
[`docs/game_running.md`](docs/game_running.md) covers the driver, the in-process agents, `--play`
scripts and decision logs.

**Sculpted positions** (testing a card or a rule): the engine's test-setup presets and `--play`
specs, e.g.

```bash
cd bin
./debug/robomage --narrative --trace-decisions --game-seed 1 --max-decisions 10 \
  --hand-a "Lightning Bolt" --battlefield-a "Mountain" --battlefield-b "Grizzly Bears" \
  --play "A:keep,B:keep,A:cast:Lightning Bolt,A:target:Grizzly Bears@opp"
```

## The state format and the knowledge sidecar

- [`docs/state_format.md`](docs/state_format.md) — RMSTATE: visibility, handles, the snapshot
  tables, observation events, the encoding, the `--machine` protocol, the self-checks, and its
  decision register. The format describes itself in a SCHEMA frame; the decoders (C++
  `src/state_format/decode.h`, Python `python/rmstate.py`) are built from it.
- [`docs/knowledge_store.md`](docs/knowledge_store.md) — the knowledge sidecar
  (`src/knowledge/`): levels `off` / `expert` / `expert+belief`, the model, the update rules, the
  `k_*` / `b_*` tables, the verification.

`--rmstate-dump FILE` writes every viewer's frames beside any game (`--state-dump json` adds a
JSON line per frame); `python3 python/rmstate.py dump FILE` prints them.

## The test gate

`make check` builds and runs every tier of `bin/debug/robomage_check`: deck/vocab check,
concessions, the rules-regression scenarios (`tests/scenarios/`), the byte-identical replay corpus
(`tests/corpus/`), the state format's and the knowledge sidecar's self-checks, an explore fuzz, a
throughput bench and scripted-agent smoke games (about 5 minutes on 4 cores). CI runs exactly
it; a nightly workflow runs a deep fuzz over every league matchup. After an intentional behavior
change, `make regen` re-records the corpus. Details: [`docs/ci.md`](docs/ci.md).

Developer notes (architecture, conventions, how to test card behavior, the key-file index):
[`CLAUDE.md`](CLAUDE.md).
