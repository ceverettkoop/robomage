# robomage

Magic the Gathering rules engine built for reinforcement learning. Games run are deterministic, based on random seed. Agent (human, scripted or ML model) is presented with finite options for every decision point.

Current agent training workflow is maskable PPO training with SB3 to train a base policy and then refines that model with MCTS training. The latest release will include a training model to play any
of the decks in bin/resources/decks/league. It may be able to handle other decks with the same or similar cards but note that the engine will require rebuilding for any cards not currently listed in the card-vocab
and rules engine fidelity is not guaranteed for cards outside of the vocab. todo.md includes open engine correctness items.

Agent training workflow and rules engine are under active development so things may break frequently.
Developer docs: `CLAUDE.md` (architecture, test harness, full build/tooling notes), `docs/ci.md`
(the `make check` gate), `docs/game_running.md` (running games programmatically).

## Prereqs

**C++ toolchain** — to build the engine binary (`make`). Engine is C++17 + standard library. Tested on Mac and Linux
would probably work on Windows with mingw but have not tested.

**Python toolchain** — for the RL training stack, the TUI, and the test harness.
Python 3.10+ (CI is tested on 3.12).
```bash
python3 -m venv train/.venv
train/.venv/bin/pip install --upgrade pip
train/.venv/bin/pip install numpy gymnasium                     # minimum: test harness / env only
train/.venv/bin/pip install stable-baselines3 sb3-contrib shap  # full RL training + analysis (pulls in PyTorch)
train/.venv/bin/pip install -r train/requirements-tui.txt       # textual, needed by ./tui.sh
train/.venv/bin/pip install -r train/requirements-gui.txt       # PySide6, optional: desktop GUI board
```
PyTorch is not needed to play against or test with the scripted agents.

## Building

Card scripts live in `bin/resources/cardsfolder/` and `bin/resources/tokenscripts/` (Forge format,
not tracked in git). Fetch them first — the build's codegen reads them:

```bash
python3 tools/forge_fetch/provision_decks.py   # every vocab card + the league/meta/top-level deck cards + their tokens
make                  # debug build   -> bin/debug/robomage
make BUILD=RELEASE    # optimized     -> bin/release/robomage
make actor BUILD=RELEASE   # optional: C++ AlphaZero self-play actor -> bin/release/az_actor
```

Each configuration has its own `bin/<config>/` and `obj/<config>/` tree, so switching needs no
`make clean`. `make actor` links the venv's PyTorch (pin `torch==2.10.*`; newer headers need
C++20). Which binary the Python tools run is chosen per tool: test/CI tools (harness, `observe`,
`make check`) default to the debug build, while the GUI, analysis browser, and training default to
release. `ROBOMAGE_BUILD=debug|release` forces one, and every command takes `--binary PATH`.

## Running

**`./tui.sh`, run from the repo root, exposes all features.** It launches a Textual TUI
(`train/tui.py`) that wraps deck management, training, league runs, observing games, analysis, and
interactive play against a model; every form is built from the same flags as the CLIs below.

The raw `robomage` binary underlies all of it and can also be driven directly. It must be run
from the `bin` directory (it finds `resources/` relative to the working directory):

```bash
cd bin
./debug/robomage --deck-a delver --deck-b mav            # interactive (you play both sides)
./debug/robomage --replay resources/logs/game_12345.log  # replay a saved game (logs are named by seed)
./debug/robomage --machine                               # machine mode for RL training
```

In interactive mode, numbers select a choice (every choice is logged), z passes priority, q quits.
`--help` lists the other engine flags.

## Cards

The ML agent only understands cards listed in `src/card_vocab.h`. The decks in this repo's `bin/resources/decks` are fully implemented. Other cards may or may not work but will in any case require being added to the vocab.
`train/missing_cards.py --decks-dir DIR` lists the cards a set of decks uses that are not in the vocab.

## Training

All commands run from the repo root with a venv that has the appropriate prereqs. `./tui.sh`
wraps all of these; `train/train.py --help` lists every subcommand. Deck names are paths under
`bin/resources/decks/` without `.dk` (`delver`, `league/ur_delver`).

There is **one generalist model** that pilots any deck, saved as `train/checkpoints/gen__final.zip`
(plus periodic `gen__v{steps}.zip` snapshots); the AZ net is `train/checkpoints/az/gen__azfinal.pt`.

The standard end-to-end run (PPO league, then AZ self-play) is the tracked curriculum plan
`train/checkpoints/curricula/default.plan.json`:

```bash
train/.venv/bin/python train/train.py curriculum --plan default --dry-run   # print each phase's command
train/.venv/bin/python train/train.py curriculum --plan default             # run it (--resume / --status)
```

### League (PFSP) — train the PPO generalist

`league` is the primary way to train `gen__final.zip`: it rotates a single learner across a
roster of decks, training each against a shared pool of frozen snapshots of past versions
(prioritised fictitious self-play). The single-matchup `train`/`sweep` subcommands are for
scripting one deck-vs-deck session, not the primary training loop.

```bash
train/.venv/bin/python train/train.py league                                          # all decks in decks/league/
train/.venv/bin/python train/train.py league --decks delver,mav --total-timesteps 5000000
train/.venv/bin/python train/train.py league --resume                                 # resume an interrupted run
train/.venv/bin/python train/train.py --deck-a delver --deck-b mav                    # one matchup (`train` is the default subcommand; --fresh restarts gen)
train/.venv/bin/python train/train.py observe --deck-a delver --deck-b mav --games 10 # sanity-check a build (scripted vs scripted + summary)
train/.venv/bin/python train/train.py baseline --player-a gen --deck-a delver         # gen's win rate vs the scripted agent
```

### AlphaZero (AZ) training — warm-started from PPO

The AZ loop trains a policy/value net on MCTS output instead of PPO gradients: self-play games
use determinized search over engine snapshots, each decision's visit counts become a policy
target, and outcomes train the value head. Run `league` first: when no AZ checkpoint exists yet,
the first `az`/`az-train` cycle warm-starts the AZ net from `gen__final.zip` (falling back to a
random init only if there is no PPO checkpoint either).

```bash
train/.venv/bin/python train/train.py az --decks delver               # one full cycle: self-play -> train -> gate
train/.venv/bin/python train/train.py az-league                       # rotate AZ cycles across decks/league/
```

Self-play uses the C++ actor automatically when `bin/<config>/az_actor` is built (release by
default, see Building), else a pure-Python backend — same shards either way. The AZ net plays
anywhere a controller spec is accepted: `az:gen?sims=128&worlds=4` (with search) or `azraw:gen`
(net only); `mcts:gen` runs search over the PPO checkpoint instead.

## Play against model

```bash
./gui.sh                                                                              # GUI app on its welcome pane — File ▸ New Session opens the play/analysis dialogs
train/.venv/bin/python train/play.py --deck-a (deck) --deck-b (deck)                  # straight into a game on the GUI board; you are player A vs az:gen on --deck-b
train/.venv/bin/python train/play.py --deck-a (deck) --deck-b (deck) --board tui      # Textual terminal board
train/.venv/bin/python train/play.py --board text --player-b scripted                 # plain-text board (type an action number or 'cast:bolt')
```

`play.py --board gui|tui|text` picks the front end (default `gui`, which falls back to the TUI
with a notice when PySide6 is missing). The GUI's File ▸ New Session ▸ Play… dialog is the
`play.py` command line as a form — same fields, same defaults (`train/.venv/bin/python
train/play.py --help` lists them), remembered in `~/.robomage/gui_launcher.json`. Defaults:
you (`--player-a human`) on `league/bug`, on the play, vs `az:gen` on `league/ur_delver`, best of
three; the search opponent uses 8 worlds on a 25-minute match clock (`--sims`, `--worlds`,
`--think-time`, `--match-clock`, … tune it); `--on-the-play b|random` changes who starts.

The GUI's **analysis window** (F9 toggles it in-game; `--no-analysis` to start without it) is
live MCTS evaluation of your current decision on a separate, detached engine copy that never
blocks the live game. It is GUI-only; `--board tui`/`text` reject the `--analysis*` flags.

## Analyze games

`analysis.py browse` is a full-screen browser: pick a game from the sidebar and page through its
board states one decision at a time, with the model's policy at each step; seek by clicking the
V(s) histogram; run any analysis view (summary, cardvalue, targeting, swings, regret, entropy,
calibration, shap, …, the game transcript, and `chart …` views that save PNGs under
`train/analysis_out/`); and branch a counterfactual `whatif` at the current step (`w` key).

`--source` picks what it browses: `simulate` (the default — `--games` games of `--player-a` vs
`--player-b` on `--deck-a`/`--deck-b`), a directory of recorded shards (AZ self-play such as
`train/az_data/gen`, or a play recording under `train/az_data/recorded/`; `--games N` loads the
first N matches, and `--games 0` is refused for a directory over 2 GiB), or a saved `.rmtrace`
session. `--board tui` (the default) is the Textual browser; `--board gui` opens the PySide6 app.
The TUI's **analysis → browse** form and the GUI's New Session ▸ Analysis… dialog carry the same
flags.

```bash
train/.venv/bin/python train/analysis.py browse --player-a (gen, or a checkpoint path) --player-b (model, or 'scripted') --deck-a (model's deck) --deck-b (opponent's deck) --games 20
train/.venv/bin/python train/analysis.py browse --source train/az_data/gen --player-a gen                     # first 20 matches (--games)
train/.venv/bin/python train/analysis.py browse --source session.rmtrace --board gui
```

`analysis.py report` runs the standard battery headless and writes one self-contained HTML
report. With a search `--player-a` (`az:gen`, `mcts:gen`) it adds search-vs-net sections
(KL(search‖net), top-1 agreement, net V vs the search root value); `--workers N` splits the games
across processes.
