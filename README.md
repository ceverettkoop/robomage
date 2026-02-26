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
QUERY: <num_choices> <f0> <f1> ... <f1152> <cat0> ... <catN-1>
```

- `num_choices` — how many valid actions exist at this step (action indices `0..num_choices-1`)
- `f0..f1152` — 1153-float observation vector (see `src/machine_io.h` for the full layout)
- `cat0..catN-1` — one `ActionCategory` integer per legal action

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
# From the repo root:
train/.venv/bin/python train/train.py                                        # by default this trains against a simple model
train/.venv/bin/python train/train.py --load checkpoints/robomage_final.zip  # resume
train/.venv/bin/python train/train.py --tally                                # print A/B win counts each rollout
```

This launches 7 parallel game subprocesses and trains with MaskablePPO for
1,000,000 timesteps. Checkpoints are saved to `train/checkpoints/` every
25,000 steps. Player B is controlled by a rule-based scripted agent (always
attacks, never blocks, casts every spell).

### Architecture

`train/env.py` — `RoboMageEnv` gymnasium wrapper. Spawns the game as a subprocess,
parses QUERY lines into observations, and provides `action_masks()` for legal move
filtering. `ModelVsScriptedEnv` wraps it so the model always plays as Player A
with the scripted agent handling Player B.

`train/extractor.py` — `CardGameExtractor`: per-entity shared-weight MLP encoder
with mean+max pooling over battlefield and hand slots.

`train/train.py` — `MaskablePPO` training loop using `SubprocVecEnv` for parallel
data collection.

## Reproducibility

Every game is seeded. The seed is written as the first line of the log file.
Replaying with `--replay` and the same log exactly reproduces the game including
all random events (shuffles, etc.).
