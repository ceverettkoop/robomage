# robomage

Card game rules engine built for reinforcement learning. Games run are deterministic, based on random seed. Agent (human, scripted or ML model) is presented with finite options for every decision point.

## Prereqs

**C++ toolchain** — to build the engine binary (`make`). Engine is C++17 + standard library.

**Python toolchain** — for the RL training stack, the TUI, and the test harness.
Python 3.10+ (CI is tested on 3.12).
```bash
python3 -m venv train/.venv
train/.venv/bin/pip install --upgrade pip
train/.venv/bin/pip install numpy gymnasium                     # minimum: test harness / env only
train/.venv/bin/pip install stable-baselines3 sb3-contrib shap  # full RL training + analysis
train/.venv/bin/pip install -r train/requirements-tui.txt       # textual, needed by ./tui.sh
```
`stable-baselines3` pulls in PyTorch, technically not required for play vs scripted (dumb) agents built into the engine, or for self-play.

## Building

```bash
make                  # default build is headless; with debug symbols
make BUILD=RELEASE    # optimized
```

Card scripts live in `bin/resources/cardsfolder/` and `bin/resources/tokenscripts`. See the card-forge repository for compatible scripts. `tools/forge_fetch/provision_decks.py` will automatically download any missing scripts based on cards listed in card_vocab.h.

The binary is written to `bin/robomage`.

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

### Training commands

All commands are run from the repo root, these commands assume a venv with appropriate
prereqs. `./tui.sh` wraps all of these; use the raw commands below for scripting or options
the TUI doesn't expose.

`train.py` uses subcommands. Run `train/.venv/bin/python train/train.py -h` to
list them, or `train.py <subcommand> -h` for a subcommand's options. The `train`
subcommand is assumed when none is given, so one-liner training works without
typing `train`.

Models are **per-deck generalists**, saved as `checkpoints/{deck}__final.zip` (plus periodic
`{deck}__v{steps}.zip` snapshots). Training against any opponent continues that same
checkpoint rather than forging a matchup-specific model, so a bare deck stem (`delver`) works
as shorthand wherever a checkpoint is expected.

```bash
# Training (the 'train' subcommand is implied when omitted)
train/.venv/bin/python train/train.py --deck delver --opponent mav                 # train/continue delver's generalist vs the mav deck
train/.venv/bin/python train/train.py train --deck delver --opponent mav --load checkpoints/delver__v500000.zip  # resume a specific snapshot
train/.venv/bin/python train/train.py --self-play --deck delver --opponent mav     # self-play against frozen checkpoints
train/.venv/bin/python train/train.py --deck delver --opponent mav --fresh         # start delver's generalist over from scratch
train/.venv/bin/python train/train.py sweep                                           # train all matchup pairs from decks in bin/resources/decks
train/.venv/bin/python train/train.py sweep --deck delver                             # only matchups featuring delver

# Verify a build / inspect a model
# observe replaces the old diag/watch commands: one command for any {scripted|model} vs {scripted|model} matchup
train/.venv/bin/python train/train.py observe --deck delver --opponent mav                          # scripted vs scripted, one game
train/.venv/bin/python train/train.py observe --deck delver --opponent mav --games 10                # 10 games + W/L/D summary (env sanity check)
train/.venv/bin/python train/train.py observe --player-a delver --player-b scripted --deck delver --opponent mav  # watch delver's model play
train/.venv/bin/python train/train.py baseline delver                                 # win rate vs the scripted agent
```

### League (PFSP)

`league` rotates a single learner across a roster of decks, training each against
a shared pool of frozen snapshots of past versions (prioritised fictitious
self-play). Snapshots feed the pool over time, so later rotations face stronger
opponents.

```bash
train/.venv/bin/python train/train.py league                                          # all decks in decks/league/
train/.venv/bin/python train/train.py league --decks delver,mav --total-timesteps 5000000
train/.venv/bin/python train/train.py league --resume                                 # resume an interrupted run
```

### AlphaZero (AZ) training

The AZ loop trains a policy/value net on MCTS output instead of PPO gradients:
self-play games are played with determinized search over engine snapshots
(`--search-server`), each searched decision's visit counts become a policy
target, and game outcomes train the value head. AZ models are per-deck like PPO
ones, saved as `checkpoints/az/{deck}__azfinal.pt` (the gate-promoted incumbent)
plus `{deck}__azv{steps}.pt` candidate snapshots, and warm-start automatically
from the deck's PPO checkpoint. The usual pattern is PPO first, AZ after it
plateaus — see `docs/ppo_az_training.md` for when to switch and
`docs/alphazero_status.md` for the machinery.

```bash
train/.venv/bin/python train/train.py az --deck delver                # one full cycle: self-play -> train -> gate
train/.venv/bin/python train/train.py az-selfplay --deck delver --games 50 --sims 128 --worlds 4  # generate shards only
train/.venv/bin/python train/train.py az-train --deck delver          # train a candidate on the shard window
train/.venv/bin/python train/train.py az-eval --deck delver --candidate delver --promote  # gate candidate vs incumbent
train/.venv/bin/python train/train.py az-league                       # rotate AZ cycles across decks/league/
train/.venv/bin/python train/eval_search_gate.py --checkpoint delver --deck delver  # search-vs-raw strength gate
```

Self-play uses the C++ actor (`make actor`, needs libtorch) automatically when
`bin/az_actor` is built, else a pure-Python backend — same shards either way.
AZ models play anywhere a controller spec is accepted: `az:delver?sims=128&worlds=4`
(with search) or `azraw:delver` (net only); `mcts:delver` runs search over a PPO
checkpoint instead.

## Play against model

```bash
train/.venv/bin/python train/play.py --human-deck (deck) --model-deck (deck)   # TUI game board (default); loads the deck's checkpoint if present
```

## Run N games and analyze them (interactive)

```bash
train/.venv/bin/python train/analysis.py interactive (model, or deck shorthand) --opponent (model, or 'scripted') --deck-a (model's deck) --deck-b (opponent's deck)
```

`analysis.py` also has a non-interactive subcommands for a report each — `report`,
