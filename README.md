# robomage

A Magic: The Gathering rules engine in C++17 for two-player games. It is an Entity Component System
built over Forge card scripts. Games are deterministic given a seed. Every decision is a choice from a
finite menu and is logged as an integer, so a decision log replays its game exactly.

Agents drive the engine three ways:

- **In-process agents** (`--agent-a/-b`): `first`, `random`, `explore`, `explore:patient`,
  `actions:…` and `play:…`. There is also the semantic `--play` script for sculpted tests. In the
  `robomage_nn` build there are also `nn:<model>` and `nn:remote:<socket>`, a learned policy
  evaluated in process or by an inference server.
- **The C++ CLI**: a human at the terminal. `--player A|B` picks a human seat against an agent, and a
  plain run is hotseat. It renders only that seat's view of the game.
- **The machine protocol** (`--machine`): **RMSTATE**, an exhaustive binary game state sent to the
  deciding seat with everything hidden from it withheld. The seat answers with one integer per
  decision. Each seat's frames carry the **knowledge sidecar**'s tables: what that seat can remember
  and soundly deduce from its own frames (perfect recall, exact inference).

The rule-based **scripted agent** (Python, standard library only) plays over that protocol. It is the
realistic opponent and the training teacher, and its games make up the replay corpus that the test
gate checks byte for byte.

On top of the engine sit a **featurizer** (C++), which turns a seat's frames into a network's input
tensors, and the **continuous actor**, which plays match after match and writes trajectory episodes.
The **R-NaD learner** (`learn/`, PyTorch) trains a policy from those episodes.

**Decks.** The playable decks are under `bin/resources/decks/`:

- the top-level decks;
- `league/`: the 26 training decks;
- `heldout/`: four decks never trained on.

Every card in them is implemented. `candidates/` holds onboarding status files, and `tests/` holds
the mechanic fixture decks. Cards outside `src/card_vocab.h` are considered unimplemented, and rules
fidelity is not guaranteed for them. `todo.md` lists open engine correctness items, and
`docs/rules_deviations.md` lists the known deviations from the Comprehensive Rules (checked in at
`docs/mtg_comprehensive_rules.txt`).

The earlier reinforcement-learning stack (PPO and AlphaZero training, a float observation vector, a
TUI and a GUI) is archived on branch `archive/rl-stack`.

## Prereqs

- A C++17 compiler and GNU make (tested on Linux and macOS). The engine needs only the standard
  library.
- `python3` (3.10+, standard library only) for card-script provisioning, the agent driver, the
  scripted agent, the corpus recorder and the gate's `smoke` tier. It needs no venv and no packages.
- Optional, for learned policies and training:
  - libtorch, from a torch importable by `python3` or from `LIBTORCH_DIR=…`, for `make actor`;
  - the learner's own venv: `python3 -m venv learn/.venv && learn/.venv/bin/pip install -r
    learn/requirements.txt` (torch, numpy and onnx; the pinned torch is a ROCm build).

## Building

Card scripts live in `bin/resources/cardsfolder/` and `bin/resources/tokenscripts/`. They are in
Forge format, not tracked in git, and pinned to a Forge commit in `tools/forge_fetch/FORGE_PIN`.
Fetch them first:

```bash
python3 tools/forge_fetch/provision_decks.py   # every vocab card + the top-level/league/heldout/candidates deck cards + their tokens
make                  # debug build   -> bin/debug/{robomage,rmknow,rmfeat,robomage_check}
make BUILD=RELEASE    # optimized     -> bin/release/…
make check            # the test gate (below)
```

Each configuration has its own `bin/<config>/` and `obj/<config>/` tree, so switching needs no
`make clean`. The four binaries:

- `robomage`: the engine.
- `rmknow`: runs the knowledge sidecar over a recorded RMSTATE stream.
- `rmfeat`: runs the featurizer over a recorded stream (the feature spec, card tables, tensors and
  learner targets).
- `robomage_check`: the test gate.

Other targets:

- `make actor`: `robomage_nn` (the engine plus the libtorch `nn:` agents) and `rmserve` (the
  reference inference server).
- `make pgo`: a profile-guided release engine.
- `make registry`: after adding a card, keyword or mechanic, appends to the featurizer's layout
  registry and card digest.
- `make regen`: re-records the replay corpus after an intentional behaviour change.

`CLAUDE.md` lists them all.

## Running games

The engine must be run from `bin/`, because it finds `resources/` relative to the working
directory. `./debug/robomage --help` lists every flag.

```bash
cd bin
./debug/robomage --player A --agent-b explore --deck-a league/burn --deck-b league/bug   # you (A) vs an agent
./debug/robomage --deck-a delver --deck-b mav                                           # hotseat
./debug/robomage --deck-a league/bug --deck-b league/burn \
  --agent-a explore --agent-b explore --game-seed 5 --decision-log /tmp/g5.rmlog        # an explore game, logged
./debug/robomage --replay /tmp/g5.rmlog --machine --narrative                           # replay it
```

At the CLI, a number picks a menu row. `board`, `zone`, `card`, `know`, `log`, `z` (auto-pass),
`concede` and `help` inspect the game. Deck names are paths under `bin/resources/decks/` without
`.dk` (`delver`, `league/ur_tempo`). `--bo3` plays a match with sideboarding.

**Agent games with the scripted agent** go through the stdlib-only driver, run from the repo root:

```bash
python3 python/rm_driver.py --player-a scripted --player-b explore \
  --deck-a league/bug --deck-b league/burn --format bo1 --games 10
python3 python/rm_driver.py --player-a scripted:hard+sb --player-b scripted:hard+sb \
  --deck-a league/ur_tempo --deck-b league/eldrazi_aggro --verbose --out match.txt   # a bo3 match, full transcript
```

The driver's agents:

- `scripted` (the hard tier);
- `scripted:hard+sb`, which sideboards per the league guides in `docs/sideboard_guides.md`;
- `scripted:easy`, `greedy` and `scripted:random`;
- `explore` and `explore:patient`;
- `first` and `actions:<i,i,…>`.

`--help` lists the flags. [`docs/game_running.md`](docs/game_running.md) covers the driver, the
in-process agents, `--play` scripts and decision logs.

**Sculpted positions** (for testing a card or a rule) use the engine's test-setup presets and
`--play` specs, for example:

```bash
cd bin
./debug/robomage --narrative --trace-decisions --game-seed 1 --max-decisions 10 \
  --hand-a "Lightning Bolt" --battlefield-a "Mountain" --battlefield-b "Grizzly Bears" \
  --play "A:keep,B:keep,A:cast:Lightning Bolt,A:target:Grizzly Bears@opp"
```

## The state format and the knowledge sidecar

- [`docs/state_format.md`](docs/state_format.md): RMSTATE. Covers visibility, handles, the snapshot
  tables, observation events, the encoding, the `--machine` protocol and the self-checks. The format
  describes itself in a SCHEMA frame, and the decoders (C++ `src/state_format/decode.h`, Python
  `python/rmstate.py`) are built from it.
- [`docs/knowledge_store.md`](docs/knowledge_store.md): the knowledge sidecar (`src/knowledge/`).
  Covers the levels `off` and `expert`, the model, the update rules, the `k_*` tables and the
  verification.

`--rmstate-dump FILE` writes every viewer's frames beside any game, and `--state-dump json` adds a
JSON line per frame. `python3 python/rmstate.py dump FILE` prints them.

## Learned policies and training

- [`docs/featurizer.md`](docs/featurizer.md): the featurizer (`src/featurize/`, `rmfeat`). Covers
  the feature spec, the card, ability and text tables, the network contract, and the layout registry
  that lets checkpoints load across vocab growth.
- [`docs/trajectory_format.md`](docs/trajectory_format.md): the episodes `--trajectory-out` writes.
- [`docs/actor.md`](docs/actor.md): the continuous actor (`--actor`), its schedule, and its status
  and results files.
- [`docs/inference_protocol.md`](docs/inference_protocol.md): remote evaluation (`nn:remote:`,
  `rmserve`, the learner's server).
- [`docs/rmtrain.md`](docs/rmtrain.md) and [`learn/README.md`](learn/README.md): `learn/rmtrain`,
  the R-NaD training program. It covers runs under `runs/<name>`, the phases B0–B5, evaluation, the
  end-of-run benchmark, and adding decks to a run.

New decks enter through the onboarding pipeline (`tools/onboard/`, its README): scrape, parse
report, engine work, vocab, fuzz, then designation to `league/` or `heldout/`.

## The test gate

`make check` builds, then runs every tier of `bin/debug/robomage_check`:

- the vocab, layout-registry and parse-report checks;
- concessions;
- the rules-regression scenarios (`tests/scenarios/`);
- the CLI's golden transcripts (`tests/cli/`);
- the byte-identical replay corpus (`tests/corpus/`);
- the state format's self-checks;
- the featurizer;
- the trajectory writer and teacher channel;
- the continuous actor;
- the knowledge sidecar;
- an explore fuzz over the playable decks, plus the mechanic fixtures' coverage;
- a throughput bench;
- scripted-agent smoke games.

It takes about 6 minutes on 4 cores, and CI runs exactly it. A nightly workflow runs a deep fuzz.

The `actor` tier (`nn:` play, exports, `rmserve`, the checkpoint migration's acceptance) is opt-in.
It needs `make actor` and `$ROBOMAGE_TORCH_PYTHON`; otherwise it is skipped. Details are in
[`docs/ci.md`](docs/ci.md).

Developer notes (architecture, conventions, how to test card behaviour, the key-file index) are in
[`CLAUDE.md`](CLAUDE.md).
