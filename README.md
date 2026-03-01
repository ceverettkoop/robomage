# robomage

Card game rules engine built for reinforcement learning.

Both players currently use `test_minimal.dk` (a blue/red fetch-land deck). The ML agent only understands cards listed in `src/card_vocab.h`. Near-term goal: train on the UR Delver mirror match.

Card scripts live in `bin/resources/cardsfolder/`. See the card-forge repository for compatible scripts.

## Building

```bash
make                  # debug build
make BUILD=RELEASE    # optimized
```

The binary is written to `bin/robomage`. The game must be run from the `bin/` directory so it can find `bin/resources/`.

## Running

```bash
cd bin
./robomage                                         # interactive (you play both sides)
./robomage --replay resources/logs/game_12345.log  # replay a saved game
./robomage --machine                               # machine mode for RL training
```

In interactive mode, numbers select a choice (every choice is logged), z passes priority, q quits.

### Machine mode protocol

In `--machine` mode the game communicates over stdio. Each decision emits one line:

```
QUERY: <num_choices> <f0>...<f2757> <cat0>...<catN-1> <id0>...<idN-1>
```

- `num_choices` — number of legal actions at this step
- `f0..f2757` — 2758-float state vector (see `src/machine_io.h` for layout)
- `cat0..catN-1` — `ActionCategory` integer per legal action (0–21)
- `id0..idN-1` — card vocab index float per action (`vocab_idx / N_CARD_TYPES`, or `-0.03125` for non-card entities)

The driver writes a single integer back on stdin; the game continues. All non-QUERY stdout is narrative and can be ignored.

## Reinforcement Learning

### Setup

```bash
python -m venv train/.venv
train/.venv/bin/pip install gymnasium stable-baselines3 sb3-contrib
```

`sb3-contrib` provides `MaskablePPO`, which prevents the agent from sampling illegal actions via `action_masks()`.

### Training commands

All commands are run from the repo root.

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

Each episode the model is randomly assigned to Player A or B. The opponent is a randomly sampled frozen checkpoint from `train/checkpoints/`. Observations are symmetry-normalised (`mirror_obs`) so the model always sees itself as Player A regardless of which side it actually controls. The opponent is reloaded every 10 episodes. Falls back to random play until the first checkpoint exists.

#### Mixed training (self-play + scripted anchor)

```bash
train/.venv/bin/python train/train.py --self-play --scripted-fraction 0.3 --load checkpoints/robomage_final.zip
```

Mixes scripted-agent envs into the self-play pool. With `N_ENVS=7` and `--scripted-fraction 0.3`, 2 envs use the scripted opponent and 5 use self-play. The scripted envs act as a permanent anchor that prevents the policy from drifting to strategies that beat itself but lose to general play.

### All flags

| Flag | Default | Description |
|---|---|---|
| `--load <path.zip>` | — | Resume training from a checkpoint |
| `--total-timesteps N` | 1,000,000 | Total training steps |
| `--self-play` | off | Train via self-play against frozen checkpoints |
| `--scripted-fraction F` | 0.0 | Fraction of envs using scripted opponent during self-play (e.g. `0.3` = 2 of 7 envs) |
| `--tally` | off | Print A/B win counts after each rollout |
| `--binary <path>` | `bin/robomage` | Path to the game binary |
| `--baseline <path.zip>` | — | Evaluate a checkpoint's win rate vs random opponent |
| `--baseline-games N` | 100 | Number of games for `--baseline` |
| `--observe <path.zip>` | — | Watch one game: model (A) vs random (B), with commentary |
| `--watch-scripted` | off | Watch one game: scripted A vs scripted B (engine sanity check) |
| `--eval <path.zip>` | — | Self-play evaluation (win rate is always ~50%; useful for other diagnostics) |
| `--eval-games N` | 100 | Number of games for `--eval` |

### Architecture

**`train/env.py`**
- `RoboMageEnv` — gymnasium wrapper; spawns the game as a subprocess, parses QUERY lines into 3032-float observations, provides `action_masks()`.
- `ModelVsScriptedEnv` — wraps `RoboMageEnv`; model always plays Player A, scripted agent handles Player B.
- `SelfPlayEnv` — wraps `RoboMageEnv`; randomly assigns the model to A or B each episode, loads a frozen opponent from the checkpoint pool, and symmetry-normalises observations via `mirror_obs()`.
- `mirror_obs(obs)` — flips an observation A↔B: swaps player stats, slot blocks (creatures/lands/graveyard 0–9 ↔ 10–19), `controller_is_A` flags, and the bf_ability_costs block, so the model always perceives itself as Player A.
- `scripted_action(obs, n)` — rule-based agent used as Player B during phase 1 training and as an optional anchor during self-play.

**`train/extractor.py`** — `CardGameExtractor`: per-entity shared-weight MLP encoder (permanent slots, stack, graveyard, hand) with mean+max pooling. Produces a fixed-size representation independent of card ordering.

**`train/train.py`** — `MaskablePPO` training loop with `SubprocVecEnv` for parallel data collection. Supports scripted, self-play, and mixed environments.

## Reproducibility

Every game is seeded. The seed is written as the first line of the log file. Replaying with `--replay` and the same log exactly reproduces the game, including all random events (shuffles, etc.).
