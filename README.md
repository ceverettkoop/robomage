# robomage

Card game rules engine built for reinforcement learning.

The ML agent only understands cards listed in `src/card_vocab.h`. The decks in this repos `bin/resources/decks` are fully implemented. Other cards may or may not work.

Card scripts live in `bin/resources/cardsfolder/`. See the card-forge repository for compatible scripts.


## Building

```bash
make                  # default build is headless; with debug symbols
make BUILD=RELEASE    # optimized
make GUI=TRUE         # with raylib gui, deprecated
```

The binary is written to `bin/robomage`.

> **Note:** The GUI (raylib) front end is **deprecated** and no longer actively maintained. The
> text/CLI (TUI) interface is the actively maintained front end.

## Running

**`./tui.sh`, run from the repo root, is the primary and intended way to drive the
application.** It launches a Textual TUI (`train/tui.py`) that wraps deck management,
training, league runs, observing games, and interactive play against a model — the commands
documented below are the raw building blocks it composes, useful for scripting or when you
need an option the TUI doesn't expose.

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
./robomage --gui                                   # gui (deprecated, not actively maintained)
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
train/.venv/bin/pip install -r train/requirements-tui.txt   # textual, needed by ./tui.sh
```

### Training commands

All commands are run from the repo root, these commands assume a venv with appropriate
prereqs. `./tui.sh` wraps all of these; use the raw commands below for scripting or options
the TUI doesn't expose.

#### Training

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

#### League (PFSP)

`league` rotates a single learner across a roster of decks, training each against
a shared pool of frozen snapshots of past versions (prioritised fictitious
self-play). Snapshots feed the pool over time, so later rotations face stronger
opponents.

```bash
train/.venv/bin/python train/train.py league                                          # all decks in decks/league/
train/.venv/bin/python train/train.py league --decks delver,mav --total-timesteps 5000000
train/.venv/bin/python train/train.py league --resume                                 # resume an interrupted run
```

**Snapshot promotion gate (`--promote-margin`):** a frozen `{deck}__v{steps}.zip`
is only added to the pool when the learner's win-rate is at least `0.5 + margin`,
so the pool collects genuinely stronger snapshots instead of near-duplicates.

- The default is **`0.05`** (gate at a 55% win-rate). A *negative* margin gates
  below 50% — useful for decks that take extensive training to break even, so
  they still promote instead of starving the pool. `0` disables the gate
  entirely; the first snapshot of each deck is always exempt so self-play can
  bootstrap.
- The win-rate is measured over a **recent sliding window**, not the cumulative
  average, so a deck that started weak can promote once it is *currently* strong.
- The active opponent pool is capped per env process (`--opponent-ckpt-ratio`), but
  isn't just a recency slice: it keeps a **guaranteed** per-deck anchor
  (`{deck}__final`, or the newest snapshot if none yet) that is never evicted,
  plus discretionary `__v*` intermediates filled newest-first round-robin across
  decks — so every roster deck stays represented as an opponent even if it's a
  perennial loser.

#### Play against model

```bash
train/.venv/bin/python train/play.py --human-deck (deck) --model-deck (deck)         # TUI game board (default); loads the deck's checkpoint if present
train/.venv/bin/python train/play.py --human-deck (deck) --model-deck (deck) --gui   # raylib GUI instead (deprecated)
```

### Run N games and analyze them

```bash
train/.venv/bin/python train/analysis.py interactive (model, or deck shorthand) --opponent (model, or 'scripted') --deck-a (model's deck) --deck-b (opponent's deck)
```

`analysis.py` also has non-interactive subcommands for a single chart/report each — `report`,
`cardvalue`, `shap`, `value-swings`, `regret`, `entropy`, `consistency` — see
`train/.venv/bin/python train/analysis.py -h`.

### View logs in tensorboard:
```
train/.venv/bin/tensorboard --logdir logs
```