# robomage

Magic the Gathering rules engine built for reinforcement learning. Games run are deterministic, based on random seed. Agent (human, scripted or ML model) is presented with finite options for every decision point.

Current agent training workflow is maskable PPO training with SB3 to train a base policy and then refines that model with MCTS training. The latest release will include a training model to play any
of the decks in bin/resources/decks/league. It may be able to handle other decks with the same or similar cards but note that the engine will require rebuilding for any cards not currently listed in the card-vocab
and rules engine fidelity is not guaranteed for cards outside of the vocab. todo.md includes open engine correctnes items.

Agent training workflow and rules engine are under active development so things may break frequently.

## Prereqs

**C++ toolchain** — to build the engine binary (`make`). Engine is C++17 + standard library. Tested on Mac and Linux 
would probably work on Windows with mingw but have not tested.

**Python toolchain** — for the RL training stack, the TUI, and the test harness.
Python 3.10+ (CI is tested on 3.12).
```bash
python3 -m venv train/.venv
train/.venv/bin/pip install --upgrade pip
train/.venv/bin/pip install numpy gymnasium                     # minimum: test harness / env only
train/.venv/bin/pip install stable-baselines3 sb3-contrib shap  # full RL training + analysis
train/.venv/bin/pip install -r train/requirements-tui.txt       # textual, needed by ./tui.sh
train/.venv/bin/pip install -r train/requirements-gui.txt       # PySide6, optional: desktop GUI board
```
`stable-baselines3` pulls in PyTorch, technically not required for play vs scripted (dumb) agents built into the engine, or for self-play.

## Building

```bash
make                  # default build is headless; with debug symbols -> bin/debug/robomage
make BUILD=RELEASE    # optimized                                  -> bin/release/robomage
```

Card scripts live in `bin/resources/cardsfolder/` and `bin/resources/tokenscripts`. See the card-forge repository for compatible scripts. `tools/forge_fetch/provision_decks.py` will automatically download any missing scripts based on cards listed in card_vocab.h.

Each configuration writes to its own tree — `bin/debug/robomage` (and `obj/debug/`) from a plain
`make`, `bin/release/robomage` (and `obj/release/`) from `make BUILD=RELEASE` — so the two builds
never clobber each other and no `make clean` is needed when switching between them. Python tooling
runs the debug binary by default; set `ROBOMAGE_BUILD=release` to reach the optimized one (or pass a
command's `--binary` flag explicitly).

## Quick start — play the v0.2 release

The release tarball is the source tree **plus** the trained models (`gen__final.zip`, the PPO
generalist, and `az/gen__azfinal.pt`, the AlphaZero net) already in place under
`train/checkpoints/` — nothing to move after extracting.

```bash
# grab robomage-0.2.tar.gz from https://github.com/ceverettkoop/robomage/releases/latest
tar xzf robomage-0.2.tar.gz && cd robomage-0.2

make BUILD=RELEASE                                              # optimized engine -> bin/release/robomage
export ROBOMAGE_BUILD=release                                   # point tooling at the optimized binary
python3 -m venv train/.venv
train/.venv/bin/pip install --upgrade pip
train/.venv/bin/pip install numpy gymnasium stable-baselines3 sb3-contrib
train/.venv/bin/pip install -r train/requirements-gui.txt       # PySide6 desktop board

./gui.sh                                                        # launcher dialog -> Start game
```

`./gui.sh` opens the launcher with the following defaults:

- **Your deck** `league/bug`, **Opponent deck** `league/ur_delver`, random seat, best-of-three
- **Opponent** `MCTS search (az:gen)` — the AlphaZero net with MCTS on top, 8 determinized
  worlds, no per-decision simulation cap, paced off a 1500 s (25 min) whole-match thinking clock
- **Analysis window** on, evaluating each of your decisions with the same AZ net (4 worlds,
  2000 sims), which you can toggle in-game with F9

**Use the decks in `bin/resources/decks/league/`** (`bug`, `ur_delver`, `gw_maverick`, `bw_dnt`,
`wrb_energy`) for both seats.

`azraw:gen` plays the AZ net with no search (fast, weaker); `mcts:gen` runs search over the PPO
checkpoint instead; `scripted:hard` is the rule-based agent and needs no model at all.


## Running

**`./tui.sh`, run from the repo root, exposes all features** It launches a Textual TUI (`train/tui.py`) that wraps deck management,
training, league runs, observing games, and interactive play against a model.

```bash
./tui.sh    # from the repo root — training, play, observe, league, etc.
```

The raw `robomage` binary underlies all of it and can also be driven directly. It must be run
from the `bin` directory:

```bash
cd bin
./robomage                                         # interactive (you play both sides)
./robomage --replay resources/logs/game_12345.log  # replay a saved game
./robomage --machine                               # machine mode for RL training
```

In interactive mode, numbers select a choice (every choice is logged), z passes priority, q quits.

## Cards

The ML agent only understands cards listed in `src/card_vocab.h`. The decks in this repos `bin/resources/decks` are fully implemented. Other cards may or may not work but will in any case require being added to the vocab.

## Training

All commands run from the repo root with a venv that has the appropriate prereqs. `./tui.sh`
wraps all of these; use the raw commands below for scripting or options the TUI doesn't expose.

There is **one generalist model** that pilots any deck, saved as `checkpoints/gen__final.zip`
(plus periodic `gen__v{steps}.zip` snapshots). 

**Use `league` (below) to actually train the PPO generalist** — it rotates the roster and pool
of past snapshots for you. The single-matchup `train`/`sweep` subcommands below are for
scripting one deck-vs-deck session or inspecting a checkpoint, not the primary training loop.

```bash
train/.venv/bin/python train/train.py --deck delver --opponent mav                 # continue gen on one matchup
train/.venv/bin/python train/train.py --deck delver --opponent mav --fresh         # start gen over from scratch
train/.venv/bin/python train/train.py observe --deck delver --opponent mav --games 10  # sanity-check a build (scripted vs scripted + summary)
train/.venv/bin/python train/train.py baseline gen --deck delver                   # gen's win rate vs the scripted agent
```

### League (PFSP) — train the PPO generalist

`league` is the primary way to train `gen__final.zip`: it rotates a single learner across a
roster of decks, training each against a shared pool of frozen snapshots of past versions
(prioritised fictitious self-play). Snapshots feed the pool over time, so later rotations face
stronger opponents.

```bash
train/.venv/bin/python train/train.py league                                          # all decks in decks/league/
train/.venv/bin/python train/train.py league --decks delver,mav --total-timesteps 5000000
train/.venv/bin/python train/train.py league --resume                                 # resume an interrupted run
```

### AlphaZero (AZ) training — warm-started from PPO

The AZ loop trains a policy/value net on MCTS output instead of PPO gradients: self-play games
use determinized search over engine snapshots, each decision's visit counts become a policy
target, and outcomes train the value head. **AZ always starts from a PPO warm start** — run
`league` first to build `gen__final.zip`, and the first `az`/`az-train` cycle initializes the
AZ net (`checkpoints/az/gen__azfinal.pt`) from it automatically; there's no from-scratch AZ path.
See `docs/ppo_az_training.md` for when to switch from PPO to AZ and `docs/alphazero_status.md`
for the machinery.

```bash
train/.venv/bin/python train/train.py az --deck delver                # one full cycle: self-play -> train -> gate
train/.venv/bin/python train/train.py az-league                       # rotate AZ cycles across decks/league/
```

Self-play uses the C++ actor (`make actor`, needs libtorch) automatically when `bin/az_actor` is
built, else a pure-Python backend — same shards either way. The AZ net plays anywhere a
controller spec is accepted: `az:gen?sims=128&worlds=4` (with search) or `azraw:gen` (net only);
`mcts:gen` runs search over the PPO checkpoint instead.

## Play against model

```bash
train/.venv/bin/python train/play.py --human-deck (deck) --model-deck (deck)          # TUI game board (default); the generalist (gen) pilots --model-deck
train/.venv/bin/python train/play.py --human-deck (deck) --model-deck (deck) --gui    # PySide6 desktop board (needs requirements-gui.txt)
train/.venv/bin/python train/gui_game.py                                              # GUI launcher — pick decks/opponent/format/analysis in a dialog
./gui.sh                                                                              # shortcut for the line above, run from the repo root
```

**`./gui.sh`**, run from the repo root, is the GUI-board equivalent of `./tui.sh` — it launches
`train/gui_game.py` with no arguments, which opens the launcher dialog (needs
`requirements-gui.txt`; falls back to the TUI if PySide6 is missing). The launcher's "Game
setup" group picks your deck, the opponent's deck, the opponent controller (`gen`, a scripted
tier, or an `az:`/`azraw:`/`mcts:` search spec), which seat you play, and bo3-vs-single-game
format; choices persist to `~/.robomage/gui_launcher.json` for next time. When the opponent is
a search spec (`az:`/`mcts:`), a "Search opponent settings" group appears with tuning knobs
(each defaults to "omit the knob" unless set): simulations per decision, determinized worlds,
think-time-per-decision (overrides the sims cap with a wall-clock budget), search procs
(engine processes to fan worlds across), a whole-match thinking clock, and whether to pace
responses to mask which decisions were easy/hard.

The GUI also has an always-available **analysis window** (launcher's "Analysis window" group,
or `--gui --analysis`; F9 toggles it in-game): live MCTS evaluation of your current decision on
a separate, detached engine copy that never blocks the live game. 

## Run N games and analyze them (interactive)

```bash
train/.venv/bin/python train/analysis.py interactive (gen, or a checkpoint path) --opponent (model, or 'scripted') --deck-a (model's deck) --deck-b (opponent's deck)
```

`analysis.py` also has a non-interactive subcommands for a report each — `report`,

### TUI analysis browser

`tui_analysis.py` is a full-screen Textual front end for the same simulated-game analysis:
pick a game from the sidebar and page through its board states (rendered like the TUI game
board) one decision at a time, with the model's full policy distribution shown at each step;
seek by clicking the V(s) histogram docked at the bottom; run any REPL analysis view (summary,
cardvalue, targeting, swings, regret, entropy, calibration, shap, …) from the sidebar menu; and
branch a counterfactual `whatif` at the current step (`w` key) to simulate an alternative line.

From `./tui.sh`, pick the **analysis-tui** tool and its **browse** subcommand to fill in the
same options through the form. To invoke it directly:

```bash
train/.venv/bin/python train/tui_analysis.py (gen, or a checkpoint path) --opponent (model, or 'scripted') --deck-b (opponent's deck) --n-games 20
```
