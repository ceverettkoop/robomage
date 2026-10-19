# rmtrain: the training program

`learn/rmtrain` runs one training run of `docs/rnad_training_plan.md` from the command line: the
teacher corpus, behaviour cloning, DAgger, R-NaD self-play and fine-tuning (§6 B0–B5), detached,
for days, resuming from its last checkpoint after a crash, and steerable while it runs (plan §4
L6). The code is `learn/rmlearn/rmtrain.py` (the CLI) and `learn/rmlearn/orchestrator/`.

```sh
make BUILD=RELEASE && make BUILD=RELEASE actor          # bin/release/robomage_nn, rmfeat
learn/rmtrain defaults > my.toml                         # every key, its default and what it does
learn/rmtrain start r1 --config my.toml                  # runs/r1; detaches and returns
learn/rmtrain status r1                                  # also when it is not running
learn/rmtrain tail r1 gpu                                # follow a log (tail -F)
learn/rmtrain set r1 rnad.eta 0.15                       # a live parameter
learn/rmtrain stop r1                                    # checkpoint, stop; `start r1` resumes
```

## 1. Starting and resuming

`rmtrain start <run> [--config FILE] [--set section.key=value …] [--foreground]`

- `<run>` is a name (the run directory `runs/<name>`, git-ignored) or a path.
- A new run: the config file (TOML, or JSON by its `.json` extension) over the defaults, then the
  `--set` overrides, validated (an unknown key, a wrong type or a bad value is refused naming the
  key), written as `config.json` (the live, resolved config) and `config.toml` (as created, with
  the documentation).
- An existing run is **resumed**: its `config.json` is used (`--set` changes keys of it, also
  non-live ones such as `model.*` — at your own risk; `--config` is refused). The phase and stage
  come from `state.json`, the GPU process reloads the last checkpoint, the actors start after their
  last match. A finished run (phase `done`) is refused.
- Detached (the default): `systemd-run --user --unit=rmtrain-<name>-<hash>` with
  `MemoryMax=run.memory_max` (the whole run: orchestrator, GPU process, actors, teachers),
  `KillMode=mixed` (a `systemctl stop` signals only the orchestrator, which stops the rest in
  order), `Restart=on-failure` (`run.restart_on_failure`, after `run.restart_delay_s`): a run that
  fails — the GPU process crashed, the orchestrator was killed — is restarted by systemd and
  resumes from its checkpoint. `start` returns once the control socket answers and prints the
  status. Without a usable `systemd-run --user` it double-forks a plain daemon and warns: no
  memory limit, no restart. **After a reboot** the transient unit is gone: `rmtrain start <run>`
  resumes it.
- `--foreground` runs the orchestrator in the terminal (its log echoed; Ctrl-C stops the run cleanly).

## 2. Inspecting and steering

Every command but `start`, `defaults` and the file-only cases talks to the orchestrator over its
control socket (`/tmp/rmtrain-<hash>/control.sock`, recorded in `sockets.json`: a socket path must
stay under 108 bytes). One JSON request per connection, one JSON reply (`orchestrator/control.py`).

| Command | What |
|---|---|
| `status <run> [--json]` | phase and stage, hold / pause, the phase's progress, games and decisions (and per second, over the last minute), actor health (running, restarts, why one is down), the GPU process (mode, inference server stats from `InferenceServer.stats()`: rows/s, batch mean, queue p99, clients, served policies and versions; GPU memory; the replay window), the latest metrics of each kind, the evaluation running / queued / last results, the last checkpoint, recent transitions. When the run is not running: the same from `status.json` and `state.json` (marked NOT RUNNING) |
| `tail <run> [component] [-n N] [--no-follow]` | `orchestrator` (default), `gpu`, `export`, `metrics`, `evals`, `a<k>` (an actor), `actors` (all), `all` |
| `pause <run>` | the actors are stopped (SIGTERM: a match in progress is written off) and the learner stops stepping; evaluations stop (queued again). Kept across restarts |
| `resume <run>` | undoes `pause` |
| `stop <run>` | clean shutdown: evaluations cancelled, actors SIGTERM, queued exports finished, a checkpoint, the GPU process stopped; waits until the run is gone |
| `set <run> <section.key> <value>` | a live parameter (below), validated, applied at once and written to `config.json` (a resume keeps it); a `set` metrics line |
| `eval <run> [--evaluators a,b]` | queue an evaluation of the current policy now |
| `phase <run> hold\|release\|next\|status` | `hold` blocks the next transition (the exit test still runs; the run stays in the phase and says why in `status`); `release` lifts it; `next` forces the current phase's (or stage's) exit, also when held |
| `config <run>` | the resolved config as TOML |
| `defaults` | the default config as TOML, documented |

**Live parameters** (`config.LIVE`): `bc.lr`, `bc.report_every`, `bc.eval_every`; `rnad.eta`,
`rnad.learning_rate`, `rnad.entropy_schedule_sizes` (Δm; an integer is a one-size schedule),
`rnad.entropy_schedule_repeats`, `rnad.reg_eps`, `rnad.reg_eps_sideboarding` (also applied to the
smoothed π_0,reg while it is still a regularisation policy), `rnad.push_every`,
`rnad.report_every`, `rnad.games_per_step`; `actors.count`; the phase limits (`b0.decisions`,
`b1.*` but `play_eval`, `b2.rounds_*`, `b2.decisions_per_round`, `b2.min_steps_per_round`,
`b2.min_improvement`, `b4.iterations`, `b4.max_learner_steps`, `b4.nn_options`,
`b5.max_learner_steps`); `eval.matches`, `eval.ladder`, `eval.ladder_matches`, `eval.actors`,
`eval.every_minutes`, `eval.per_iteration`; `checkpoint.every_minutes`. The schedule's Δm is the
reference's function of the learner step, so a changed Δm can move α and the next boundary.

## 3. The config

`learn/rmtrain defaults` prints every key. The main ones:

| Key | Default | |
|---|---|---|
| `run.seed` | 1 | actors' `--actor-seed`, model init, held-out split |
| `run.build` | `release` | `bin/<build>/robomage_nn`, `rmfeat` |
| `run.tables` | `""` | reuse an `rmfeat --export-tables` directory (else exported into `tables/`) |
| `run.memory_max` | `100G` | systemd `MemoryMax` of the run |
| `run.device` | `cuda` | the GPU process's device (`cpu` for tests) |
| `run.hsa_override_gfx_version` | `10.3.0` | the GPU process's `HSA_OVERRIDE_GFX_VERSION` |
| `run.gpu_lock`, `run.gpu_lock_wait` | `/tmp/rm-gpu.lock`, false | the flock the GPU process holds; fail at once or wait |
| `run.restart_on_failure`, `run.restart_delay_s` | true, 60 | systemd's restart of a failed run |
| `league.decks` / `league.pairs` | the 10 league decks / every pair | matchups and weights (`[[a, b, w], …]`) |
| `league.format`, `league.knowledge`, `league.max_game_decisions` | `bo3`, `expert`, 5000 | |
| `league.omniscient` | true | B0/B2 episodes carry the omniscient part (aux targets) |
| `actors.count` | 26 | training actors |
| `actors.restart_delay_s`, `crash_loop_restarts`, `crash_loop_window_s`, `max_backoff_s` | 2, 5, 300, 600 | crash handling |
| `model.d`, `heads`, `layers`, `id_dropout`, `omniscient_critic` | 192, 4, 4, 0.1, false | `ModelConfig` |
| `replay.max_decisions`, `replay.max_gb` | 4M, 40 | the window (BC's and R-NaD's) |
| `replay.heldout_fraction` | 0.05 | BC's held-out games |
| `replay.delete_evicted_selfplay` | true | evicted self-play `.rmtraj` deleted (RMLOGs kept) |
| `server.max_batch`, `deadline_us`, `precision` | 512, 2000, fp32 | the RMIP server |
| `b0.decisions`, `b0.explore_frac`, `b0.corpus_dir` | 5M, 0.2, `""` | the corpus; a corpus directory skips B0 |
| `bc.*` | `BCConfig` | batch 128, lr 3e-4, warm-up 300, smoothing 0.05, fp16; `eval_every` 2000, `report_every` 250 |
| `b1.min_steps`, `max_steps`, `patience`, `min_delta`, `play_eval` | 2000, 30000, 3, 0.002, true | B1's exit |
| `b2.rounds_min`, `rounds_max` | 3, 5 | DAgger rounds |
| `b2.teacher_mix_start`, `teacher_mix_end` | 0.5, 0 | linear over the rounds |
| `b2.decisions_per_round`, `min_steps_per_round`, `min_improvement` | 1M, 2000, 0.01 | |
| `rnad.eta`, `learning_rate`, `entropy_schedule_sizes` | 0.2, 5e-5, [10000] | `RNaDConfig` |
| `rnad.games_per_step`, `chunk_len`, `precision` | 32, 32, fp16 | G × chunk (fits a d 192 model in 12 GB) |
| `rnad.reg_eps`, `reg_eps_sideboarding` | 0.1, 0.25 | π_0,reg's smoothing (B3) |
| `rnad.push_every` | 50 | learner steps between pushes of policy 0 |
| `b4.iterations`, `b4.max_learner_steps`, `b4.nn_options` | 0, 0, `""` | B4's exit (0: until `phase next`); self-play sampling options |
| `b5.max_learner_steps` | 0 | the run's end (0: never) |
| `eval.evaluators` | `["fixed", "ladder"]` | |
| `eval.opponents`, `eval.pairs`, `eval.matches` | scripted, explore, random; 4 pairs; 48 | |
| `eval.ladder`, `eval.ladder_matches` | 3, 24 | |
| `eval.actors`, `eval.every_minutes`, `eval.per_iteration` | 4, 60, true | |
| `eval.nn_options_bc` / `_rnad` / `_finetune` | `argmax` / `""` / `thresh=0.03:disc=16` | the evaluated policy's options per phase |
| `checkpoint.every_minutes`, `keep`, `export_formats` | 15, 3, ts + pt2 | |

## 4. The run directory

```
runs/<name>/
  config.json  config.toml       the resolved config (live) / as created
  state.json                     phase, stage, hold, pause, counters + results offsets, last checkpoint,
                                 ladder, evaluation queue and history, transitions (atomic, every ~10 s)
  status.json                    the last status (every run.status_every_s)
  sockets.json  daemon.json      the socket paths; how it was detached (unit name or pid)
  metrics.jsonl                  one line per learner report, held-out evaluation, evaluation, checkpoint,
                                 export, transition, set / pause / hold, start / stop: time, phase, stage, step
  evals.jsonl                    every evaluation's full result (per opponent, per pair, game 1 vs 2–3)
  schedule.json                  the actors' schedule (docs/actor.md §2)
  cards.rmstate  tables/         the cards frame and the exported card tables
  logs/orchestrator.log gpu.log export.log actors/a<k>.log
  actors/                        a<k>.status.json, a<k>.results.jsonl (the actors' --actor-out)
  episodes/b0 dagger selfplay    trajectory episodes (.rmtraj)
  rmlog/b0 dagger selfplay       every match's decision log (always kept)
  ckpt/                          bc-p<step>.pt (+ .opt), rnad-p<step>.pt: periodic, the last checkpoint.keep;
                                 bc-b1.pt, bc-r<k>.pt, bc-final.pt, rnad-seed.pt, rnad-final.pt: phase ends
  ckpt/fixed/m<k>.pt             the R-NaD fixed points (π_m,reg at each boundary)
  exports/<name>.ts.pt .pt2      exports of the phase-end checkpoints and fixed points (nn:<model>)
  eval/<seq>-<label>/            an evaluation: schedule, actors' status / results, rmlog, logs
```

## 5. Processes

| Process | What |
|---|---|
| orchestrator | `python -m rmlearn.rmtrain _run RUN`: the phase loop (every 0.5 s), the control socket, `state.json` / `status.json` / `metrics.jsonl`. No GPU, no torch |
| GPU process | `python -m rmlearn.orchestrator.gpu RUN` (`orchestrator/gpu.py`): **the only process of the run that touches the GPU**: the learner (BC: `BCTrainer` and its replay window; R-NaD: `RNaDLearner`, `ServerLink` and the self-play window) and the RMIP `InferenceServer` at `/tmp/rmtrain-<hash>/rmip.sock`. Holds the flock `run.gpu_lock` while it lives (fails at once naming the lock, or with `run.gpu_lock_wait` waits for it and the status says so): two runs, or a run and another GPU job using the same lock, never share the GPU. Commands as JSON lines on its stdin, replies and events on its stdout (fd 1 redirected to `gpu.log` so a library cannot corrupt the protocol); it ignores SIGTERM / SIGINT and exits on end of input |
| actors | `bin/<build>/robomage_nn --actor` (docs/actor.md) with the run's `schedule.json`, the GPU hidden (`HIP_VISIBLE_DEVICES=` …), one process group each (their teachers, `python/teacher.py`, with them) |
| evaluation actors | the same with `--actor-matches`, in `eval/<seq>-<label>/` |
| exporter | `python -m rmlearn.orchestrator.exporter` (CPU only), one at a time: `.ts.pt` / `.pt2` of a checkpoint |

**Served policies:** 0 the current policy (BC: pushed at each DAgger round's start; R-NaD:
`ServerLink`, every `rnad.push_every` steps), 1 and 2 the regularisation policies (π_m,reg,
π_m−1,reg; raw networks, the learner smooths π_0,reg itself), 3 the evaluation snapshot, 10+ the
ladder rungs of an evaluation.

**Actor crashes.** An actor that exits unasked is restarted after `actors.restart_delay_s` at its
next match index: the status file's `next_match`, + 1 when a match was in progress (written off),
and never below the results file's last match + 1. More than `crash_loop_restarts` exits within
`crash_loop_window_s` back off exponentially up to `max_backoff_s`; `status` lists restarts and the
last exit's reason. **The GPU process dying** (or failing a command) stops the run: actors stopped,
state saved, exit status 2 — systemd restarts it (`Restart=on-failure`), or `start` does.

**Memory.** The replay window holds the episodes' bytes (~13 KB per decision with the omniscient
part, ~5 KB without): `replay.max_gb` bounds it (B0's 5M decisions with the omniscient part
are ~65 GB on disk; the BC window keeps the newest 40 GB). Each actor is ~30 MB plus ~40 MB per
teacher. `run.memory_max` caps all of it; leave room for the page cache.

## 6. The phases

The orchestrator brings the processes to what the phase and stage want each tick (`sync`: the
schedule, the actor count, the GPU process's mode). Every transition is logged (`orchestrator.log`,
a `transition` metrics line with its reason) and blocked by `phase hold` (a transition decided by an
evaluation's result waits as *pending* and happens on `release`).

| Phase | Stage | What | Exit |
|---|---|---|---|
| B0 | collect | the teacher corpus: both seats `first` + the scripted teacher at mix 1 over the league (swap decks), `b0.explore_frac` scripted vs explore (the scripted side recorded), episodes with the omniscient part | `b0.decisions` decisions played (counted from the actors' results lines of schedule version `b0`). Skipped with `b0.corpus_dir` |
| B1 | train | BC on the window over `episodes/b0` (or the corpus dir) and `episodes/dagger`; a held-out evaluation every `bc.eval_every` steps | held-out agreement not improving by `b1.min_delta` for `b1.patience` evaluations after `b1.min_steps`, or `b1.max_steps` → `bc-b1.pt`, export |
| B1 | eval | evaluation `b1` (fixed opponents, argmax), the learner idle | its result (the DAgger baseline) |
| B2 | collect | round r: the BC policy (server policy 0, frozen for the round) on both seats via `nn:remote`, sampling, the teacher on both seats at mix `teacher_mix_start → end`; BC goes on training on the growing window | `b2.decisions_per_round` decisions of version `b2-r<r>` and `b2.min_steps_per_round` BC steps → actors stopped, `bc-r<r>.pt`, export |
| B2 | eval | evaluation `b2-r<r>` (fixed opponents) | after `b2.rounds_min` rounds, a win rate vs scripted:hard not `b2.min_improvement` above the best so far; or `b2.rounds_max` → B3; else round r + 1 |
| B3 | | `bc-final.pt`; `RNaDLearner.from_bc_checkpoint` (π_0,reg smoothed by `rnad.reg_eps`, `reg_eps_sideboarding`); `rnad-seed.pt` | at once |
| B4 | selfplay | both seats `nn:remote:…:policies=0,1,2` (+ `b4.nn_options`), train, the league; the learner takes each fresh game once (`take_fresh`, `trainable_episodes`: capped / quit games dropped); at each boundary the fixed point `ckpt/fixed/m<k>.pt`, its export, an evaluation `it<k>` (`eval.per_iteration`); evaluations every `eval.every_minutes` | `b4.iterations` iterations or `b4.max_learner_steps` steps (0: never) |
| B5 | selfplay | the learner's fine-tuning projection from the current step; evaluations with `eval.nn_options_finetune` | `b5.max_learner_steps` (0: never) → `done`: `rnad-final.pt`, exports, exit 0 |

**Why DAgger plays through the server** (`nn:remote`, not `nn:<export>` in process): the actors
spend no CPU on the network's forward (in process a d 192 forward per decision costs several times
the engine's decision), the GPU is otherwise idle during DAgger's play, no `.pt2` export (tens of
seconds) is needed before a round, and it is B4's path, exercised early.

**Pause** stops the actors rather than freezing them (SIGSTOP): the engine's teacher and server
timeouts are wall-clock deadlines, so a long freeze would fail them. The price is one written-off
match per actor per pause.

## 7. Evaluation

`orchestrator/evaluation.py`: a registry of **evaluators** (`@evaluator("name")`, a function of the
context returning groups: named schedule entries, a match count, server policies to load). An
evaluation snapshots the current network as server policy 3, loads its groups' policies, plays
the matches with `eval.actors` evaluation actors (`--actor-matches`), scores every game
(`evaluate.summarize`: per opponent and pair, game 1 apart from games 2–3), unloads, and appends to
`evals.jsonl`, `metrics.jsonl` (`eval`) and the status.

- `fixed`: against `eval.opponents` (`scripted` = scripted:hard: `first` + the teacher at mix 1;
  `explore`, `random`, any agent spec) over `eval.pairs`, both decks and seats (`swap: sides`).
- `ladder`: against the BC policy and the most recent `eval.ladder` R-NaD fixed points, each a server
  policy (10 + j) played with `eval.opponent_nn_options`.

Evaluations run one at a time, queued: at B1's end, after each DAgger round, at each R-NaD
iteration, on the `eval.every_minutes` cadence in B4–B5, and on `rmtrain eval`. The T8 SPRT runner
and the exploiter are to be added as evaluators.

## 8. Durability

- **Checkpoints** every `checkpoint.every_minutes` (once the GPU process holds a model) and at
  every phase end, written by the GPU process (a temporary file renamed): BC — the model checkpoint
  and the optimiser state beside it (`BCTrainer.save`); R-NaD — `learner.state_dict()` (networks,
  target, regularisation policies, Adam, counters, generators) and the replay window's file lists
  (the episodes in the window, those not yet taken), not their contents. `state.json` records the
  checkpoint after it is written.
- **Resume** (after `stop`, a crash or `kill -9` of everything): the phase and stage from
  `state.json`; BC reloads its checkpoint and re-reads the newest episodes up to the window's
  limit; R-NaD reloads the learner and re-reads the checkpoint's window (its untaken games fresh
  again, files older than its oldest left out, newer ones polled as fresh); the results files are
  read on from the recorded offsets; actors start after their last match. At most one checkpoint
  interval of learning is lost (and the games played in it are trained on again or skipped).
- A transient failure of a command to the GPU process is fatal for the run (exit 2) by design:
  the restart resumes from the checkpoint.

## 9. Checks

```sh
learn/.venv/bin/python learn/check_rmtrain.py            # CPU, no engine, ~4 s: RMTRAIN-CHECK: ok
learn/.venv/bin/python learn/check_rmtrain.py --smoke    # GPU + bin/release: RMTRAIN-SMOKE: ok
```

The CPU checks: config parsing and validation, the control protocol, an orchestrator over a fake
run driven through its socket and its status from the files, and the resume bookkeeping (actor
start indices, the results tail, the actor pool's crash restarts and back-off against a fake engine,
the window restore, checkpoint pruning, the schedules and evaluators). The smoke runs a tiny run
(3 actors, d 64 / 2 layers, B0 5,000 decisions, BC ended by `phase next`, one DAgger round of
3,000 decisions, R-NaD with Δm 30 and G 2) detached under systemd, exercises every command, kills
the whole unit with SIGKILL in B4 (systemd restarts it; it resumes at the checkpoint), stops and
starts it again, and runs it to `done`.

Measured (the training box, after T7; other agents' jobs on the GPU and CPU): B0 ~2 s (5,300
decisions on 3 actors), B1 ~20 s (the commands exercised, then `phase next`), B2 ~14 s (one round:
3,000 decisions, its evaluation), B3 < 1 s, B4 ~4.8 min (250 learner steps, 8 iterations and their
evaluations, a `kill -9` and systemd's restart 46 s later, a stop / start), the whole smoke ~6.5 min,
of which the GPU process spent ~1.7 min waiting for the GPU lock other jobs held.
