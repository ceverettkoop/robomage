# robomage

Card game rules engine built for reinforcement learning.

The ML agent only understands cards listed in `src/card_vocab.h`. The decks in this repos bin/resources/decks are fully implemented. Other cards may or may not work.

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

### Prereqs

```bash
python -m venv train/.venv
train/.venv/bin/pip install gymnasium stable-baselines3 sb3-contrib shap
```

### Training commands

All commands are run from the repo root, these commands assume a venv with appropriate prereqs.

#### Training

```bash
train/.venv/bin/python train/train.py #trains against a scripted agent
train/.venv/bin/python train/train.py --load checkpoints/robomage_final.zip  # resume
train/.venv/bin/python train/train.py --train-all #trains all possible matchup pairs from decks in bin/resources/decks
train/.venv/bin/python train/train.py --self-play #trains against models for the corresponding match
```

#### Play against model

```bash
train/.venv/bin/python train/play.py --human-deck (deck) --model-deck (deck) --gui #will load appropriate model if present, otherwise you specify 
```

### Run N games and analyze them

```bash
train/.venv/bin/python train/analysis.py interactive --model (model for player A) --opponent (model for player B, or 'scripted' for scripted) #will load appropriate model if present, otherwise you specify 
```
