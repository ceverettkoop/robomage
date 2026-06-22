# robomage

Card game rules engine built for reinforcement learning.

The ML agent only understands cards listed in `src/card_vocab.h`. The decks in this repos `bin/resources/decks` are fully implemented. Other cards may or may not work.

Card scripts live in `bin/resources/cardsfolder/`. See the card-forge repository for compatible scripts.


## Building

```bash
make                  # debug build
make BUILD=RELEASE    # optimized
make HEADLESS=TRUE    # no GUI, no raylib dependancy
```

The binary is written to `bin/robomage`. Game must be run from the bin directory.

## Running

```bash
cd bin
./robomage                                         # interactive (you play both sides)
./robomage --replay resources/logs/game_12345.log  # replay a saved game
./robomage --machine                               # machine mode for RL training
./robomage --gui                                   # gui 
```

In interactive mode, numbers select a choice (every choice is logged), z passes priority, q quits.

## Reinforcement Learning

### Prereqs:
Game engine and CLI:

C++17

GUI:

Raylib

Machine learning:

Python venv and packages per below commands:
```bash
python -m venv train/.venv
train/.venv/bin/pip install gymnasium stable-baselines3 sb3-contrib shap
```

### Training commands

All commands are run from the repo root, these commands assume a venv with appropriate prereqs.

#### Training

`train.py` uses subcommands. Run `train/.venv/bin/python train/train.py -h` to
list them, or `train.py <subcommand> -h` for a subcommand's options. The `train`
subcommand is assumed when none is given, so one-liner training works without
typing `train`.

```bash
# Training (the 'train' subcommand is implied when omitted)
train/.venv/bin/python train/train.py --deck delver --opponent mav                 # train delver vs the mav deck
train/.venv/bin/python train/train.py train --opponent mav --load checkpoints/robomage_final.zip  # resume
train/.venv/bin/python train/train.py --self-play --deck delver --opponent mav     # self-play against frozen checkpoints
train/.venv/bin/python train/train.py sweep                                           # train all matchup pairs from decks in bin/resources/decks
train/.venv/bin/python train/train.py sweep --deck delver                             # only matchups featuring delver

# Verify a build / inspect a model
train/.venv/bin/python train/train.py diag                                            # 10 quick games to verify the env
train/.venv/bin/python train/train.py watch                                           # watch one scripted-vs-scripted game
train/.venv/bin/python train/train.py baseline checkpoints/robomage_final.zip         # win rate vs the scripted agent
train/.venv/bin/python train/train.py observe checkpoints/robomage_final.zip          # watch the model play one game
```

#### League (PFSP)

`league` rotates a single learner across a roster of decks, training each against
a shared pool of frozen snapshots of past versions (prioritised fictitious
self-play). Snapshots feed the pool over time, so later rotations face stronger
opponents.

```bash
train/.venv/bin/python train/train.py league                                          # all decks in bin/resources/decks
train/.venv/bin/python train/train.py league --decks delver,mav --total-timesteps 5000000
```

**Snapshot promotion gate (`--promote-margin`):** a frozen `{deck}__v{steps}.zip`
is only added to the pool when the learner's win-rate is at least `0.5 + margin`,
so the pool collects genuinely stronger snapshots instead of near-duplicates.

- The default is **`-0.1`** (gate at a 40% win-rate). A *negative* margin gates
  below 50% — useful for decks that take extensive training to break even, so
  they still promote instead of starving the pool. `0` disables the gate
  entirely; the first snapshot of each deck is always exempt so self-play can
  bootstrap.
- The win-rate is measured over a **recent sliding window**, not the cumulative
  average, so a deck that started weak can promote once it is *currently* strong.
- Two related knobs are intentionally not CLI flags (edit in `train/train.py` if
  you need to tune them): `PFSPCallback(recent_window=200)` — number of recent
  decisive games the gate averages over — and `SnapshotCallback(min_gate_samples=30)`
  — the window must hold at least this many games before the gate can block a
  snapshot (otherwise it errs toward keeping the pool fed).
- The active opponent pool is capped per env process (`--opp-ckpt-ratio`) and
  keeps the **newest** snapshots in that cap.

#### Play against model

```bash
train/.venv/bin/python train/play.py --human-deck (deck) --model-deck (deck) --gui #will load appropriate model if present, otherwise you specify 
```

### Run N games and analyze them

```bash
train/.venv/bin/python train/analysis.py interactive --opponent (model for player B, or 'scripted' for scripted) (model for player A) 
```
