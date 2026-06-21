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

#### Play against model

```bash
train/.venv/bin/python train/play.py --human-deck (deck) --model-deck (deck) --gui #will load appropriate model if present, otherwise you specify 
```

### Run N games and analyze them

```bash
train/.venv/bin/python train/analysis.py interactive --opponent (model for player B, or 'scripted' for scripted) (model for player A) 
```
