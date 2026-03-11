# robomage

Card game rules engine built for reinforcement learning.

Both players currently use `delver.dk`. The ML agent only understands cards listed in `src/card_vocab.h`.

Card scripts live in `bin/resources/cardsfolder/`. See the card-forge repository for compatible scripts.

Python mostly LLM written so likely a mess, will be revisited.

## Building

```bash
make                  # debug build
make BUILD=RELEASE    # optimized
make HEADLESS=TRUE    # no GUI, no raylib dependancy
```

The binary is written to `bin/robomage`. Game must be run from the bin directory at present.

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

### Setup

```bash
python -m venv train/.venv
train/.venv/bin/pip install gymnasium stable-baselines3 sb3-contrib
```

`sb3-contrib` provides `MaskablePPO`, which prevents the agent from sampling illegal actions via `action_masks()`.

### Training commands

All commands are run from the repo root, this commands assume a venv with appropriate prereqs.

#### Phase 1 — train against scripted agent

```bash
train/.venv/bin/python train/train.py
train/.venv/bin/python train/train.py --load checkpoints/robomage_final.zip  # resume
```

Trains for 1,000,000 steps across 7 parallel game processes. Player B is a rule-based scripted agent (always attacks, never blocks, casts every affordable spell). Checkpoints are saved to `train/checkpoints/` every 25,000 steps.

#### Phase 2 — self-play against frozen checkpoints

```bash
train/.venv/bin/python train/train.py --self-play --load checkpoints/robomage_final.zip
```