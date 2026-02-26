# robomage

Card game rules engine built for reinforcement learning.

Currently only a few cards tested.

## Building

Requires clang (macOS) or gcc (Linux).

```bash
make program          # debug build
BUILD=RELEASE make program  # optimized build
```

The binary is written to `bin/robomage`. The game expects to be run from the `bin/`
directory so it can find `bin/resources/`.

## Running

```bash
cd bin
./robomage                        # interactive
./robomage --replay resources/logs/game_12345.log  # replay a saved game
./robomage --machine              # machine mode for RL training
```

### Machine mode protocol

In `--machine` mode the game communicates over stdio. Each time a decision is
required it emits a single line to stdout:

```
QUERY: <num_choices> <f0> <f1> ... <f192>
```

- `num_choices` — how many valid actions exist at this step (action indices `0..num_choices-1`)
- `f0..f192` — 193-float observation vector (see `src/machine_io.h` for the full layout)

The driver writes a single integer back on stdin and the game continues.

All other stdout lines (game narrative, debug output) are not QUERY lines and can
be ignored or discarded.

## Reinforcement Learning

### Dependencies

```bash
pip install gymnasium stable-baselines3 sb3-contrib
```

`sb3-contrib` provides `MaskablePPO`, which prevents the agent from ever sampling
illegal actions via the `action_masks()` method on the environment.

### Training

```bash
cd train
python train.py
```

This launches 4 parallel game subprocesses and trains with PPO for 1,000,000
timesteps (~5 hours on CPU). Checkpoints are saved to `train/checkpoints/` every
25,000 steps.

```bash
# Resume from a checkpoint
python train.py --load checkpoints/robomage_100000_steps.zip

# Quick smoke test (~15 minutes)
# Edit TOTAL_TIMESTEPS in train/train.py to 50_000
```

### Evaluating a trained model

```bash
# Win rate vs a random agent (the meaningful benchmark)
python train.py --baseline checkpoints/robomage_final.zip --baseline-games 100

# Watch the model play one game against a random opponent
python train.py --observe checkpoints/robomage_final.zip
```

The `--baseline` mode pits the model (Player A) against an agent that picks random
legal actions (Player B). A well-trained model should win well above 50% of these
games. `ep_rew_mean` during self-play training stays near 0 by definition — baseline
win rate is the real measure of progress.

`--observe` prints the full game narrative alongside each decision so you can read
what the model is actually doing.

### What to watch during training

| Metric | What it means |
|---|---|
| `ep_rew_mean` | Average reward per episode. Stays near 0 in self-play (A winning = B losing). Not the primary signal. |
| `explained_variance` | How well the value network predicts outcomes. Should climb toward 1.0. |
| `ep_len_mean` | Average decisions per game. Should decrease as the agent learns to play decisively. |
| `entropy_loss` | Policy entropy. Starts high (random), decreases as the agent develops preferences. |

### Architecture

`train/env.py` — `RoboMageEnv` gymnasium wrapper. Spawns the game as a subprocess,
parses QUERY lines into observations, and provides `action_masks()` for legal move
filtering.

`train/train.py` — Self-play training loop using `MaskablePPO` with `SubprocVecEnv`
for parallel data collection.

## Project structure

```
src/
  components/     pure data structs (CardData, Zone, Creature, Ability, ...)
  systems/        logic operating on components (Orderer, StateManager, StackManager)
  classes/        game state and supporting types
  ecs/            entity-component-system core (Coordinator, System, ...)
  main.cpp        game loop
  machine_io.cpp  state serialization for --machine mode
  input_logger.cpp  decision logging and replay
bin/
  resources/
    decks/        .dk deck files
    cardsfolder/  card script files
    logs/         saved game logs
train/
  env.py          gymnasium environment
  train.py        PPO training script
```

## Reproducibility

Every game is seeded. The seed is written as the first line of the log file.
Replaying with `--replay` and the same log exactly reproduces the game including
all random events (shuffles, etc.).
