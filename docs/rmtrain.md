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
learn/rmtrain bench r1 --wait                            # after the run: its policies vs scripted:hard (§7.1)
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
  resumes from its checkpoint; after `run.crash_loop_fatals` fatal stops in a row without progress
  it exits with status 4, which systemd does not restart (`RestartPreventExitStatus=3 4`; 3: the
  run directory has no `config.json`), and `status` shows it FAILED with the last fatal message. An
  explicit `start` resets that count. `start` returns once the control socket answers and prints the
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
| `status <run> [--json]` | phase and stage, hold / pause, the phase's progress, games and decisions (and per second, over the last minute), actor health (running, restarts, why one is down), the GPU process (mode, inference server stats from `InferenceServer.stats()`: rows/s, batch mean, queue p99, clients, served policies and versions; GPU memory; the replay window), the latest metrics of each kind, the evaluation running / queued / last results (its subject, version and options; an error when it failed), structured sideboard exploration's line in B4–B5 (§6.1), the findings (drawn games, §7), the last checkpoint, recent transitions, the fatal stops in a row without progress (§5). When the run is not running: the same from `status.json` and `state.json` (marked NOT RUNNING; FAILED after a crash loop, with the last fatal message and the command that resumes it) |
| `tail <run> [component] [-n N] [--no-follow]` | `orchestrator` (default), `gpu`, `export`, `metrics`, `evals`, `a<k>` (an actor), `actors` (all), `all` |
| `pause <run>` | the actors are stopped (SIGTERM: a match in progress is written off) and the learner stops stepping; evaluations stop (queued again). Kept across restarts |
| `resume <run>` | undoes `pause` |
| `stop <run>` | clean shutdown: evaluations cancelled, actors SIGTERM, queued exports finished, a checkpoint, the GPU process stopped; waits until the run is gone |
| `set <run> <section.key> <value>` | a live parameter (below), validated, applied at once and written to `config.json` (a resume keeps it); a `set` metrics line |
| `eval <run> [--evaluators a,b]` | queue an evaluation of the current policy (the network when the evaluation starts) |
| `phase <run> hold\|release\|next\|status` | `hold` blocks the next transition (the exit test still runs; the run stays in the phase and says why in `status`); `release` lifts it; `next` forces the current phase's (or stage's) exit, also when held |
| `bench <run> [--wait] [--policies …]` | the end-of-run benchmark (§7.1): the run's policies against scripted:hard over every league pair, per deck; runs in the terminal, not through the socket |
| `config <run>` | the resolved config as TOML |
| `defaults` | the default config as TOML, documented |

**Live parameters** (`config.LIVE`): `bc.lr`, `bc.report_every`, `bc.eval_every`; `rnad.eta`,
`rnad.learning_rate`, `rnad.entropy_schedule_sizes` (Δm; an integer is a one-size schedule),
`rnad.entropy_schedule_repeats`, `rnad.reg_eps`, `rnad.reg_eps_sideboarding` (also applied to the
smoothed π_0,reg while it is still a regularisation policy), `rnad.push_every`,
`rnad.report_every`, `rnad.games_per_step`, `rnad.value_after_every`; `actors.count`; the phase limits (`b0.decisions`,
`b1.*` but `play_eval`, `b2.rounds_*`, `b2.decisions_per_round`, `b2.min_steps_per_round`,
`b2.min_improvement`, `b4.iterations`, `b4.max_learner_steps`, `b4.nn_options`,
`b5.max_learner_steps`); `eval.matches`, `eval.ladder`, `eval.ladder_matches`, `eval.actors`,
`eval.every_minutes`, `eval.per_iteration`, `eval.also_sampled`, `eval.sprt_max_games`, `eval.exploiter_every_minutes`,
`eval.exploiter_minutes`; `checkpoint.every_minutes`; every `sb_explore.*` key (§6.1: the next
self-play schedule carries the change). The schedule's Δm is the
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
| `run.crash_loop_fatals` | 3 | fatal stops in a row without progress after which the daemon exits with status 4, which systemd does not restart (0: never; §5) |
| `run.start_from` | `""` | a BC checkpoint (another run's `ckpt/bc-final.pt`): the run starts at B3, R-NaD seeded from it (B0–B2 skipped; the model is the checkpoint's) |
| `league.decks` / `league.pairs` | the 10 league decks / every pair | matchups and weights (`[[a, b, w], …]`) |
| `league.format`, `league.knowledge`, `league.max_game_decisions` | `bo3`, `expert`, 5000 | |
| `league.omniscient` | true | B0/B2 episodes carry the omniscient part (aux targets) |
| `actors.count` | 26 | training actors |
| `actors.teacher_tier` | `hard` | the scripted teacher's tier (`python/teacher.py --tier`): both seats of B0's scripted-vs-scripted corpus, the scripted seat of its explore matches and every DAgger label (B2). `hard+sb` sideboards per the league guides, so BC learns to sideboard (plan §6.2). Self-play has no teacher, and every evaluation opponent stays `scripted:hard` whatever this is (`eval.opponents`, `sprt`, `rmtrain bench`), so results compare across runs |
| `actors.policy_eval` | `local` | how the training actors (DAgger, self-play) evaluate the policy: `local` (in process, §5.1) or `remote` (the GPU process's server) |
| `actors.local_count` | 16 | `local`: the training actors while they play the network (DAgger, self-play; 0: `actors.count`; live) — more play fewer decisions in all on this box (§5.1) |
| `actors.export_format`, `export_every_s`, `export_every_steps` | `ts`, 30, 0 | `local`: the actors' export of the current policy and its cadence (seconds, and learner steps when > 0; both live) |
| `actors.restart_delay_s`, `crash_loop_restarts`, `crash_loop_window_s`, `max_backoff_s` | 2, 5, 300, 600 | crash handling |
| `model.d`, `heads`, `layers`, `id_dropout`, `omniscient_critic` | 192, 4, 4, 0.1, false | `ModelConfig` |
| `replay.max_decisions`, `replay.max_gb` | 4M, 40 | the window (BC's and R-NaD's) |
| `replay.heldout_fraction` | 0.05 | BC's held-out games |
| `replay.max_decision_tokens` | 640 | the window (BC's and R-NaD's) leaves out a whole episode with a decision of more seat tokens (menu tokens included; 0: no cap; live: a change applies to the episodes ingested from then on, and a lowered cap drops the window's episodes over it). One such decision pads its whole batch to it, and attention is quadratic in the padded length (§5 Memory). Each one is logged once in `gpu.log` with its path and largest decision and counted in the window's stats (`skipped_oversize`, `oversize_max_tokens`: the `window` of `bc_eval`, `rnad_step` and `status`, `bc_train`'s `window_skipped_oversize`, the status's window line). The actors still play such games, and their results lines count as any other's |
| `replay.delete_evicted_selfplay` | true | evicted self-play `.rmtraj` deleted (RMLOGs kept) |
| `server.max_batch`, `deadline_us`, `precision` | 512, 2000, fp32 | the RMIP server |
| `server.attention_budget` | 2^26 | the most attention cells (rows × padded tokens², per head and layer) of one forward: a batch runs sorted by token count in forwards of at most 64 decisions and this many cells, so a few long decisions run in forwards of their own (§5 Memory) |
| `b0.decisions`, `b0.explore_frac`, `b0.corpus_dir` | 5M, 0.2, `""` | the corpus; a corpus directory skips B0 |
| `bc.*` | `BCConfig` | batch 128, lr 3e-4, warm-up 300, smoothing 0.05, fp16; `eval_every` 2000, `report_every` 250 |
| `bc.sideboard_weight` | 1.0 | the policy loss's weight of a sideboarding decision (others 1): the batch's loss is `sum(w·loss) / sum(w)`, in B1 and B2 (one trainer); value and aux losses unweighted. `bc_train` reports the sideboarding decisions' own `train_policy_sideboarding`, `train_agree_sideboarding`, `train_frac_sideboarding`, `bc_eval` their `eval_agree_group:sideboarding` and `eval_policy_loss_group:sideboarding` |
| `b1.min_steps`, `max_steps`, `patience`, `min_delta`, `play_eval` | 2000, 30000, 3, 0.002, true | B1's exit |
| `b2.rounds_min`, `rounds_max` | 3, 5 | DAgger rounds |
| `b2.teacher_mix_start`, `teacher_mix_end` | 0.5, 0 | linear over the rounds |
| `b2.decisions_per_round`, `min_steps_per_round`, `min_improvement` | 1M, 2000, 0.01 | |
| `rnad.eta`, `learning_rate`, `entropy_schedule_sizes` | 0.2, 5e-5, [10000] | `RNaDConfig` |
| `rnad.games_per_step`, `chunk_len`, `precision` | 32, 32, fp16 | G × chunk (fits a d 192 model in 12 GB) |
| `rnad.forward_tokens`, `attention_budget`, `checkpoint_tokens` | 65536, 2^26, 2^17 | a chunk's forwards take rows of similar token counts, at most `forward_tokens` padded tokens and `attention_budget` attention cells (rows × padded tokens²) each; a chunk whose forwards hold more than `checkpoint_tokens` padded tokens or `attention_budget` cells together checkpoints each forward (its activations recomputed in the backward, one forward at a time: the same gradients for one more forward, ~15% of a step; §5 Memory) |
| `rnad.reg_eps`, `reg_eps_sideboarding` | 0.1, 0.25 | π_0,reg's smoothing (B3), `reg_eps_sideboarding` at sideboarding steps. With a sideboarding teacher (`actors.teacher_tier = "hard+sb"`) set `reg_eps_sideboarding` to about 0: π_0,reg is then the taught sideboarding, and exploration belongs to the actors (plan §6.2). 0 is valid (π_0,reg's raw log π at those steps) |
| `rnad.push_every` | 50 | learner steps between pushes of policy 0 |
| `rnad.max_policy_lag` | 0 | drop a self-play game whose oldest step was played by a policy more than this many learner steps old (0: off; live) |
| `rnad.fresh_order` | `newest` | which fresh self-play games a learner step takes first: `newest` (the least policy lag when the actors play more than the learner takes) or `oldest` (live) |
| `rnad.value_after_every` | 10 | learner steps between the value diagnostic's after-update forwards (0: never): `value_mse_first_sight` (every step), `value_mse_after`, `value_mse_gap` in `rnad_step` and `status` |
| `b4.iterations`, `b4.max_learner_steps`, `b4.nn_options` | 0, 0, `""` | B4's exit (0: until `phase next`); self-play sampling options |
| `b5.max_learner_steps` | 0 | the run's end (0: never) |
| `sb_explore.p_start`, `p_end`, `hold_frac` | 0, 0, 0.5 | `p_explore`, structured sideboard exploration's probability per self-play seat and sideboarding phase (§6.1): `p_start` until `hold_frac` × `b4.max_learner_steps` learner steps, then linear to `p_end` at `b4.max_learner_steps` (`p_start` throughout B4 when that is 0); 0 in B5, in every other phase and in every evaluation. The default 0 explores nothing (live) |
| `sb_explore.count_weights` | 11 × 1 | the weights of an exploring phase's swap count k = 0 … 10 (live) |
| `sb_explore.coverage`, `coverage_power` | true, 1 | a card's in- (out-) weight in a matchup is `(1 + n) ** -coverage_power`, n the explored phases of that (own deck, opponent deck) that moved it in (out); `false` or power 0: uniform (live) |
| `sb_explore.rewrite_every_s` | 600 | seconds between rewrites of the self-play schedule with the current counts; also at once when `p_explore` moves by 0.01 (or onto or off 0), at the B4 → B5 transition or when an `sb_explore` key is set (live) |
| `eval.evaluators` | `["sprt", "fixed", "ladder"]` | the evaluators of a per-iteration / cadence evaluation (`exploiter` too; §7) |
| `eval.opponents`, `eval.pairs`, `eval.matches` | scripted, explore, random; 4 pairs; 48 | `scripted` is `scripted:hard` whatever `actors.teacher_tier` is; `scripted:<tier>` names another tier |
| `eval.ladder`, `eval.ladder_matches` | 3, 24 | |
| `eval.actors`, `eval.every_minutes`, `eval.per_iteration` | 4, 60, true | |
| `eval.policy_eval` | `local` | the evaluation actors: `local` (the evaluated policy and the ladder rungs exported, then played in process; the exploiter stays remote) or `remote` (server policies) |
| `eval.nn_options_bc` / `_rnad` / `_finetune` | `argmax` / `argmax` / `thresh=0.03:disc=16` | the evaluated policy's options per phase (the primary result; §7) |
| `eval.also_sampled` | false | the `fixed` evaluator also plays the policy with the actors' sampling options (groups `<opponent>@sampled`) |
| `eval.sprt_opponents`, `sprt_pairs` | `["scripted"]`, every league pair | the `sprt` evaluator's opponents and deck pairs |
| `eval.sprt_p0`, `sprt_p1`, `sprt_alpha`, `sprt_beta`, `sprt_max_games` | 0.45, 0.55, 0.05, 0.05, 400 | its test on the per-game win rate; the game cap (undecided there) |
| `eval.exploiter_every_minutes`, `exploiter_minutes` | 0 (off), 20 | the exploiter's B4/B5 cadence and training time |
| `eval.exploiter_actors`, `exploiter_pause_learner` | 8, true | its actors (beside the training actors); the learner idles while it trains |
| `eval.exploiter_baseline_games`, `exploiter_final_games` | 100, 100 | games of its initial policy (= the victim) and of its frozen final version |
| `eval.exploiter_pairs`, `exploiter_victim_options`, `exploiter_lr`, `exploiter_rollout_decisions` | `eval.pairs`, `""`, 3e-5, 4096 | |
| `checkpoint.every_minutes`, `keep`, `export_formats` | 15, 3, ts + pt2 | |

## 4. The run directory

```
runs/<name>/
  config.json  config.toml       the resolved config (live) / as created
  state.json                     phase, stage, hold, pause, counters + results offsets, sideboard exploration's
                                 counts (sb_explore), last checkpoint, ladder, evaluation queue and history,
                                 transitions, findings (atomic, every ~10 s)
  status.json                    the last status (every run.status_every_s)
  sockets.json  daemon.json      the socket paths; how it was detached (unit name or pid)
  metrics.jsonl                  one line per learner report, held-out evaluation, evaluation (eval, or
                                 eval_failed with its error), checkpoint, export, transition, set / pause /
                                 hold, start / restore / stop, sb_explore (§6.1): time, phase, stage, step
  evals.jsonl                    every evaluation's full result (per opponent, per pair, game 1 vs 2–3)
  schedule.json                  the actors' schedule (docs/actor.md §2)
  cards.rmstate  tables/         the cards frame and the exported card tables
  logs/orchestrator.log gpu.log export.log actors/a<k>.log
  actors/                        a<k>.status.json, a<k>.results.jsonl (the actors' --actor-out)
  episodes/b0 dagger selfplay    trajectory episodes (.rmtraj)
  rmlog/b0 dagger selfplay       every match's decision log (always kept)
  ckpt/                          bc-p<step>.pt (+ .opt), rnad-p<step>.pt: periodic (named by the GPU
                                 process's step when it writes them), the last checkpoint.keep;
                                 bc-b1.pt, bc-r<k>.pt, bc-final.pt, rnad-seed.pt, rnad-final.pt: phase ends
  ckpt/fixed/m<k>.pt             the R-NaD fixed points (π_m,reg at each boundary)
  exports/<name>.ts.pt .pt2      exports of the phase-end checkpoints and fixed points (nn:<model>)
  exports/live/current.ts.pt     local actors: the current policy's export (replaced atomically on the cadence)
  exports/eval/<name>-v<n>.ts.pt local evaluations: the evaluated policy's and the ladder rungs' exports
  eval/<seq>-<label>/            an evaluation: schedule, actors' status / results, rmlog, logs
```

## 5. Processes

| Process | What |
|---|---|
| orchestrator | `python -m rmlearn.rmtrain _run RUN`: the phase loop (every 0.5 s), the control socket, `state.json` / `status.json` / `metrics.jsonl`. No GPU, no torch |
| GPU process | `python -m rmlearn.orchestrator.gpu RUN` (`orchestrator/gpu.py`): **the only process of the run that touches the GPU**: the learner (BC: `BCTrainer` and its replay window; R-NaD: `RNaDLearner`, `ServerLink` and the self-play window) and the RMIP `InferenceServer` at `/tmp/rmtrain-<hash>/rmip.sock`. Holds the flock `run.gpu_lock` while it lives (fails at once naming the lock, or with `run.gpu_lock_wait` waits for it and the status says so): two runs, or a run and another GPU job using the same lock, never share the GPU. Commands as JSON lines on its stdin, replies and events on its stdout (fd 1 redirected to `gpu.log` so a library cannot corrupt the protocol); it ignores SIGTERM / SIGINT and exits on end of input |
| actors | `bin/<build>/robomage_nn --actor` (docs/actor.md) with the run's `schedule.json`, the GPU hidden (`HIP_VISIBLE_DEVICES=` …) and `OMP_NUM_THREADS=1` (one thread each), one process group each (their teachers, `python/teacher.py`, with them) |
| evaluation actors | the same with `--actor-matches`, in `eval/<seq>-<label>/` |
| exporter | `python -m rmlearn.orchestrator.exporter` (CPU only), one at a time: `.ts.pt` / `.pt2` of a checkpoint |
| live exporter | `python -m rmlearn.orchestrator.exporter --serve` (CPU only, one process kept running): local actors' and local evaluations' exports (`orchestrator/live.py`) |

### 5.1 Local evaluation (`actors.policy_eval = "local"`)

With `remote` the actors evaluate every decision through the GPU process's inference server, which
shares the GPU with the learner; with `local` each actor evaluates in process (`nn:<export>`, one
CPU thread: `src/actor/evaluator.cpp` sets libtorch to one intra-op and one inter-op thread, and
the actors run with `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1`), the GPU only learns, and the
learner computes the regularisation policies' log π itself.

- **The weights.** In R-NaD the GPU process saves the current network (`exports/live/src-v<step>.pt`)
  when B4 starts and then every `actors.export_every_s` seconds (or `export_every_steps` learner
  steps) when the learner stepped (a `policy_saved` event); in DAgger once at each round's start
  (the round's frozen policy). The **live exporter** (`exporter --serve`, one CPU process kept
  running, `orchestrator/live.py`) exports it to `exports/live/current.ts.pt` through a temporary
  file renamed over it; a newer save replaces one still queued. The actors' schedule names that file
  (`nn:<run>/exports/live/current.ts.pt`), and each actor's agents reload it before a match when it
  changed (docs/actor.md §7). After the GPU process loads a model (a start, a resume, R-NaD's
  seeding) or a DAgger round starts, the actors wait for that model's first export (`status`: actors
  "waiting for the live export"), so a resume never plays an export of lost steps. Every export is a
  `live_export` metrics line (version, seconds, latency); `status` shows the last one.
- **The format** is TorchScript (`actors.export_format = "ts"`): measured on this box (d 192 /
  4 layers, one thread, idle machine), TorchScript exports in ~0.07 s and evaluates in 6.7 ms per
  decision, an AOTInductor `.pt2` takes ~16 s to compile and evaluates in 6.2 ms (−7 %); at a 30 s
  cadence the `.pt2` compile would occupy a core half the time and lag the policy by its compile.
  In the run the live export (load, fold, script, save) takes ~0.43 s, ~0.5 s from the save.
- **Reload between games?** The actor reloads between matches only. Reloading before each game
  of a bo3 would be easy and safe (the agent keeps no game-scoped state beside its evaluator; each
  step records its policy version), but it would bound a lag that is already small: with 26 local
  actors a game takes 3.8 s on average (114 decisions; ~30 decisions/s per actor), a bo3 match ~9 s,
  against the 30 s export cadence, so it is not done.
- **Each record's `policy_version`** is the export's (its metadata: the learner step it was
  saved at). The learner's step metrics report the lag (learner step − the step's policy version)
  per report window: `policy_lag_p50`, `_p95`, `_max`, `_mean`. `rnad.max_policy_lag` (0: off)
  drops games older than that, `rnad.fresh_order = "newest"` takes the newest fresh games first
  (the least lag when the actors play more than the learner takes).
- **Regularisation log π.** Local trajectories record only the played policy's evaluation, so
  every step's π_m,reg and π_m−1,reg log π is recomputed by the learner (`stale_reg = "recompute"`,
  `reg_recomputed` in the metrics): no-grad forwards of the frozen networks in the step's precision
  (fp16), on the same row groups and input tensors as the step's forward (no second gather), one
  network while π_0,reg sits in both slots, two after the first boundary.
- **Evaluations** (`eval.policy_eval = "local"`, separate since evaluation actors are separate):
  the evaluated policy (a fixed point, or the current network saved by `save_model`) and each
  ladder rung are exported first (`exports/eval/<name>-v<version>.ts.pt`, reused while newer than
  their checkpoint), then played in process; the exploiter trains in the GPU process and stays
  remote.

**Measured** (the option-2 benchmark; the full table and the actor-count scaling in
`docs/rnad_training_plan.md` §5): val1's two matchups and BC policy (d 192 / 4 layers), Δm 100, B4
only, ~12–22 min each, one after the other:

| | remote, 26 actors | local, 26 actors | local, 16 actors (the default) |
|---|---|---|---|
| self-play decisions/s (games/hour) | 500 (15.9k) | 754 (23.7k) | 1,045 (33.7k) |
| learner steps/min; decisions/s (wall) | 8.2; 496 | 12.0; 730 | 14.9; 879 |
| learner waiting for games | 41 % | 5 % | 3 % |
| GPU use (mean), VRAM (max) | 93 %, 12.0 GB | 78 %, 11.8 GB | 89 %, 11.9 GB |
| actors' CPU (cores) | 0.44 (waiting for the server) | 25.3 | 15.7 |
| policy lag p50 / p95 / max (learner steps) | 28 / 38 / 52 | 6 / 10 / 20 | 6 / 10 / 18 |
| exports | — | every 32.5 s, 0.43 s each | every 32.2 s, 0.25 s each |
| loss, KL to the regularisation policy (means; finite) | 0.46, 0.0102 | 0.95, 0.0097 | 0.99, 0.0108 |

More than 16 local actors play fewer decisions in all (16 physical cores; 26 also starve the GPU
process's CPU-side batch building), so `actors.local_count` is 16. At d 128 the same 16 actors play
2,061 decisions/s and the learner takes 1,180.

**Served policies:** 0 the current policy (BC: pushed at each DAgger round's start; R-NaD:
`ServerLink`, every `rnad.push_every` steps), 1 and 2 the regularisation policies (π_m,reg,
π_m−1,reg; raw networks, the learner smooths π_0,reg itself), 3 the evaluation snapshot, 10+ the
ladder rungs of an evaluation, 20 and 21 an exploiter in training and its victim (§7).

**Actor crashes.** An actor that exits unasked is restarted after `actors.restart_delay_s` at its
next match index: the status file's `next_match`, + 1 when a match was in progress (written off),
and never below the results file's last match + 1. More than `crash_loop_restarts` exits within
`crash_loop_window_s` back off exponentially up to `max_backoff_s`; `status` lists restarts and the
last exit's reason. **The GPU process dying** (or failing a command) stops the run: actors stopped,
state saved, exit status 2 — systemd restarts it (`Restart=on-failure`), or `start` does.
**A crash loop** — the same failure at every restart, such as a batch that runs out of GPU memory
at the same step — ends after `run.crash_loop_fatals` (3) fatal stops in a row without progress:
the same phase and stage, and neither its step (BC's or the learner's) nor the decisions its
actors collected (B0, the DAgger round) past the previous fatal stop's. The daemon then exits with
status 4, which systemd does not restart. `state.json`'s `crash_loop` records the count, the
progress mark, the last fatal message and whether the run failed; `status` shows a failed run as
`FAILED (a crash loop)` with that message and the command that resumes it, and the `fatal` and
`crash_loop` events are in `metrics.jsonl`. Fix the cause, then `rmtrain start <run>`: an explicit
start resets the count. A unit keeps the `RestartPreventExitStatus` of the `start` that launched it.

**Memory.** The replay window holds the episodes' bytes (~13 KB per decision with the omniscient
part, ~5 KB without): `replay.max_gb` bounds it (B0's 5M decisions with the omniscient part
are ~65 GB on disk; the BC window keeps the newest 40 GB). Each actor is ~30 MB plus ~40 MB per
teacher. `run.memory_max` caps all of it; leave room for the page cache.

On the GPU, attention is quadratic in a batch's padded token count, and almost every decision
has at most ~600 seat tokens; a rare legal game (a board of hundreds of tokens) has decisions of
2000 and more. `replay.max_decision_tokens` (640) leaves such games out of the window, so a BC
batch of `bc.batch_size` decisions pads to at most the cap. The actors still play such games, and
the inference server still evaluates them: `server.attention_budget` puts a batch's long decisions
in forwards of their own. An R-NaD chunk (G × `chunk_len` decisions) back-propagates as one, so
its forwards' activations add up; past `rnad.checkpoint_tokens` or `rnad.attention_budget` they
are checkpointed and the backward holds one forward's at a time. Peaks measured on the RX 6700 XT
with the default d 192 / 4-layer model (the learner and the server share the process):

| Work | Peak |
|---|---|
| BC step, 128 decisions padded to 640 tokens (the cap) | 4.7 GiB |
| BC held-out evaluation, 256 decisions at 640 tokens | 2.2 GiB |
| R-NaD step, a chunk of 1024 decisions at 640 tokens | 4.0 GiB (out of memory without checkpointing) |
| R-NaD step, a chunk of 1024 decisions of ~150 tokens | 2.8 GiB (7.6 GiB without checkpointing) |
| Server batch of 512 with four decisions of ~2070 tokens (fp32) | 0.6 GiB (out of memory in chunks of 64 alone) |

## 6. The phases

The orchestrator brings the processes to what the phase and stage want each tick (`sync`: the
schedule, the actor count, the GPU process's mode). Every transition is logged (`orchestrator.log`,
a `transition` metrics line with its reason) and blocked by `phase hold` (a transition decided by an
evaluation's result waits as *pending* and happens on `release`).

| Phase | Stage | What | Exit |
|---|---|---|---|
| B0 | collect | the teacher corpus: both seats `first` + the scripted teacher (`actors.teacher_tier`) at mix 1 over the league (swap decks), `b0.explore_frac` the teacher vs explore (the scripted side recorded), episodes with the omniscient part | `b0.decisions` decisions played (counted from the actors' results lines of schedule version `b0`). Skipped with `b0.corpus_dir` |
| B1 | train | BC on the window over `episodes/b0` (or the corpus dir) and `episodes/dagger`; a held-out evaluation every `bc.eval_every` steps | held-out agreement not improving by `b1.min_delta` for `b1.patience` evaluations after `b1.min_steps`, or `b1.max_steps` → `bc-b1.pt`, export |
| B1 | eval | evaluation `b1` (fixed opponents, argmax), the learner idle | its result (the DAgger baseline) |
| B2 | collect | round r: the BC policy (frozen for the round: its export in process, or server policy 0 via `nn:remote`) on both seats, sampling, the teacher on both seats at mix `teacher_mix_start → end`; BC goes on training on the growing window | `b2.decisions_per_round` decisions of version `b2-r<r>` and `b2.min_steps_per_round` BC steps → actors stopped, `bc-r<r>.pt`, export |
| B2 | eval | evaluation `b2-r<r>` (fixed opponents) | after `b2.rounds_min` rounds, a win rate vs scripted:hard not `b2.min_improvement` above the best so far; or `b2.rounds_max` → B3; else round r + 1 |
| B3 | | `bc-final.pt` (or `run.start_from`); `RNaDLearner.from_bc_checkpoint` (π_0,reg smoothed by `rnad.reg_eps`, `reg_eps_sideboarding`); `rnad-seed.pt` | at once |
| B4 | selfplay | both seats the current policy (local: `nn:<run>/exports/live/current.ts.pt`; remote: `nn:remote:…:policies=0,1,2`; + `b4.nn_options`), train, the league; each entry's `sb_explore` while `p_explore` > 0 (§6.1); the learner takes each fresh game once (`take_fresh`, `trainable_episodes`: capped / quit games dropped); at each boundary the fixed point `ckpt/fixed/m<k>.pt`, its export, an evaluation `it<k>` of that fixed point (`eval.per_iteration`); evaluations of the current policy every `eval.every_minutes` | `b4.iterations` iterations or `b4.max_learner_steps` steps (0: never) |
| B5 | selfplay | the learner's fine-tuning projection from the current step (no sideboard exploration); evaluations with `eval.nn_options_finetune` | `b5.max_learner_steps` (0: never) → `done`: `rnad-final.pt`, exports, exit 0 |

**DAgger plays in process too** (`actors.policy_eval = "local"`): the round's frozen BC policy is
exported once at the round's start (TorchScript, ~0.1 s), and the GPU only trains BC. Measured (val1's
two matchups and corpus, teacher mix 0.5, ~5 min of a round each): 1,050 decisions/s with 16 local
actors against 751 through the server with 26 (the server shared the GPU with BC's training).

### 6.1 Structured sideboard exploration (B4)

The actors' side is the schedule entry's `sb_explore` (`docs/actor.md` §2: a coin `p` per seat and
sideboarding phase; an exploring phase plays the proposal `q` with μ = `q`); the orchestrator's
(`orchestrator/sb_explore.py`; `docs/rnad_training_plan.md` §6.2) is `p_explore`'s schedule and the
coverage weights.

- **Counts.** Every training actor's results line of a self-play game 2 or 3 (purpose `train`,
  schedule `b4` / `b5`) carries `sb` per engine seat. A seat's deck is the line's `deck_a` (seat A) or
  `deck_b` (seat B): both name the seat's deck after the match's swap coin, and decks keep their
  seats for the match, so the matchup of seat A is (`deck_a`, `deck_b`) and of seat B (`deck_b`,
  `deck_a`). An explored phase adds to its matchup one phase, its swap count's bucket (0 … 9, 10+)
  and each distinct card it moved in, and out, once; a phase the policy played (its coin did not
  land, or p 0) adds a phase and a bucket to the policy's counts. The counts live in `state.json`
  (`sb_explore`), saved together with the results files' offsets, so a restarted orchestrator goes
  on from both without counting a line twice; the results files are append-only and never pruned.
- **Weights.** In a matchup, a card's in-weight is `(1 + n) ** -coverage_power` with n the explored
  phases that moved it in (out-weight: moved out); an untried card needs no entry (the entry's
  `default_weight` 1 is the n = 0 weight). Names are the frames' (`Dead` for Dead/Gone), as the
  actor matches them.
- **The schedule.** Each B4 entry (deck a vs deck b) gets `sb_explore` = `{p, count_weights, weights:
  {deck a: its weights against deck b, deck b: against deck a}, default_weight: 1}` when `p` > 0.
  The snapshot behind it is retaken every `sb_explore.rewrite_every_s`, when `p_explore` moves by 0.01
  or onto / off 0, when the phase changes and when an `sb_explore` key is set; the schedule file is
  rewritten (atomically) when its text changes, and each actor picks it up before its next match.
  B5's schedule, B0's, DAgger's and every evaluation's carry no `sb_explore` (the engine refuses an
  eval entry with p > 0).
- **Status** (`rmtrain status`, B4–B5, or once phases were counted), one line, for example:
  `sb explore: p 0.250 (schedule 0.250); 5210 explored phases, per matchup min 41 / median 58 over 90;
  in-cells 1260 tried >=1 87%, >=10 31%; out-cells 2950 >=1 64%, >=10 9%; policy swap rate 0.083
  (48120 phases)` — `p` now and the `p` the schedule carries; explored phases in all and per
  matchup; the shares of (matchup, sideboard card) in-cells and (matchup, maindeck card) out-cells
  tried at least once and at least 10 times (cells from the league deck files, names compared by
  front face, accents folded and letters and digits only, so `Dead/Gone` is the frames' `Dead`); the
  share of the policy's phases with at least one swap. `status --json` has the same as
  `sb_explore`.
- **Metrics:** each new snapshot in B4–B5 writes an `sb_explore` line: `p`, `explored_phases`,
  `policy_phases`, `matchups`, `phases_min`, `phases_median`, `explored_hist` and `policy_hist`
  (swap counts 0 … 9, 10+, cumulative), `policy_swap_rate`, `policy_hist_recent` and
  `policy_swap_rate_recent` (the policy's phases since the previous line), `in_cells`,
  `in_tried_1`, `in_tried_10`, `out_cells`, `out_tried_1`, `out_tried_10`, `weighted_matchups`.

**Pause** stops the actors rather than freezing them (SIGSTOP): the engine's teacher and server
timeouts are wall-clock deadlines, so a long freeze would fail them. The price is one written-off
match per actor per pause.

## 7. Evaluation

`orchestrator/evaluation.py`: a registry of **evaluators** (`@evaluator("name")`, a function of the
context returning groups: named schedule entries, a match count, server policies to load). An
evaluation loads its subject as server policy 3, loads its groups' policies, plays the matches with
`eval.actors` evaluation actors (`--actor-matches`), scores every game (`evaluate.summarize`: per
opponent and pair, game 1 apart from games 2–3), unloads, and appends to `evals.jsonl`,
`metrics.jsonl` (`eval`) and the status.

**The subject.** An R-NaD iteration's evaluation (`it<k>`, `it<k>-sprt`) evaluates that
iteration's fixed point, `ckpt/fixed/m<k>.pt` (π_m,reg at the boundary), loaded from the file when
the evaluation starts — also when it was queued behind others, stopped by a pause or re-queued after
a restart; it is named `m<k>` in the results and the ladder, and its own rung is not one of its
ladder opponents. Every other evaluation (B1, DAgger rounds, the `eval.every_minutes` cadence,
`rmtrain eval`) evaluates the current network as it is when the evaluation starts (a snapshot,
named `v<version>`), not when it was queued. An exploiter always trains against the current network.

**Options.** The evaluated policy plays at `argmax` by default in every phase but B5
(`eval.nn_options_bc`, `eval.nn_options_rnad`), so results are comparable from B1 to B4; B5 plays the
fine-tuning projection (`eval.nn_options_finetune`, `thresh=0.03:disc=16`). That is the **primary**
result. With `eval.also_sampled` the `fixed` evaluator also plays the policy with the options the
phase's actors sample with (`b2.nn_options` in B1–B2, `b4.nn_options` after; `""` = temperature 1)
as groups `<opponent>@sampled`: a secondary result, left out of the ladder's ratings. Every result
carries its `options` (the evaluation's and each group's) in `evals.jsonl` and `metrics.jsonl`; the
status labels them (`argmax`, `sampled`, or the options).

**Failures.** An evaluation whose groups would collide (a repeated group, schedule entry id or server
policy) or whose fixed point is gone is not played; one whose evaluation actors all fail plays no
game. Either is an error, never an empty result: an `ERROR:` line in the orchestrator log, an
`eval_failed` metrics line, the error in `evals.jsonl` and in the status's last evaluation, and no
ladder record; the run goes on. Some actors failing is a warning (partial results).

- `fixed`: against `eval.opponents` (`scripted` = scripted:hard: `first` + the teacher at mix 1;
  `explore`, `random`, any agent spec) over `eval.pairs`, both decks and seats (`swap: sides`).
- `ladder`: against the BC policy and the most recent `eval.ladder` R-NaD fixed points, each a server
  policy (10 + j) played with `eval.opponent_nn_options`.

- `sprt` (plan T8; `rmlearn/sprt.py`'s test): against each of `eval.sprt_opponents` (default
  scripted:hard) over `eval.sprt_pairs` (default every league pair), a Wald test on the **per-game**
  win rate, H1 p = `sprt_p1` (0.55) against H0 p = `sprt_p0` (0.45), α = β = 0.05, fed game by game
  as the evaluation actors finish them; the evaluation stops its actors as soon as every test
  decides, or at `sprt_max_games` games (verdict `undecided`). Draws and caps are excluded and
  listed. The result group `sprt-<opponent>` carries the verdict, the llr and its bounds, and the
  full per-game report (`matches.report`: game 1 apart from games 2–3 — game 3 is played only at
  1–1, a selected sample —, per matchup, deck, seat, play / draw, the Wilson interval and the
  match-clustered standard error). Games of one match are correlated, so the test's error rates are
  somewhat optimistic; the clustered SE and design effect show by how much.
- `exploiter` (plan T8; `rmlearn/exploiter.py`): a short best response against the current policy,
  trained by PPO **inside the GPU process** (one GPU process for the run): `exploit_start` freezes a
  copy of the current network as the victim (server policy 21) and trains another copy as the
  exploiter (policy 20, a new version per PPO update) between the learner's units; with
  `eval.exploiter_pause_learner` (default) the learner idles meanwhile, so the two never compete for
  the GPU (the training actors play on). `eval.exploiter_actors` evaluation actors play exploiter
  (side a, recorded) against victim (side b) over `eval.exploiter_pairs` (both decks and seats);
  first `exploiter_baseline_games` games of the initial exploiter (the policy itself: ~50% by
  symmetry), then `exploiter_minutes` of training, then `exploiter_final_games` games of the frozen
  final version: its win rate is the measured exploitability (groups `exploiter-baseline`,
  `exploiter-final`; the full result, with the win-rate curve per update and PPO's statistics,
  under `exploiter` in `evals.jsonl`, files in `eval/<seq>-<label>/`). Off by default
  (`eval.exploiter_every_minutes` 0): at ~20 minutes of the GPU per run it costs the learner that
  time; run one with `rmtrain eval <run> --evaluators exploiter`, or set the cadence. Its PPO is not yet tuned: against
  the BC policy a 20-minute exploiter stayed below 50% (plan T8, measured), so read its figure as a
  lower bound on exploitability until it is.
- `ladder` also feeds the **ladder record** (`rmlearn/ladder.py`, `<run>/ladder/ladder.json`): after
  every evaluation (exploiters and failed evaluations aside) each primary group's games are recorded
  as a pairing of the evaluated policy (`m<k>` or `v<version>`) against the group's opponent (a rung by its label, `bc`, `m<k>`; a fixed
  opponent by name, `scripted`, `explore`, `random`), and a Bradley–Terry fit over every pairing so
  far (per game, a weak prior, the Elo scale, `scripted` at 0) is written into `state.json`,
  `status` (`ratings`) and the evaluation's line in `evals.jsonl`.

`sprt` and `exploiter` stop on their own terms, so each runs as an evaluation of its own: a request
naming them with others is split (`it3` → `it3`, `it3-sprt`). Evaluations run one at a time,
queued: at B1's end, after each DAgger round, at each R-NaD iteration, on the `eval.every_minutes`
cadence in B4–B5 (the exploiter on `eval.exploiter_every_minutes`), and on `rmtrain eval`.

Evaluation schedules give the evaluated policy both decks of each pair: `swap: sides` moves a
side's deck with its agent (docs/actor.md §2), so every pair is two entries, one per order.

**Findings: draws.** A drawn game (CR 104.4a: both players lost at once — e.g. a burn player's
Price of Progress or Eidolon damage killing both) is legitimate, but rare enough to review. Every
draw in the training actors' or an evaluation's results is noted in `state.json`'s `findings` (the
count and the latest ten: source, actor, match, game, decks, the results file and the match's
decision log) and the orchestrator log, and `status` shows a `findings:` line with the latest three;
they are not errors. Replay one with `cd bin && ./release/robomage_nn --replay <rmlog> --narrative`.

### 7.1 The end-of-run benchmark

The run's own evaluations are small and spread over its life: the `fixed` evaluator plays four deck
pairs, an `sprt` evaluation stops as soon as its test decides (often after a few dozen games, so its
per-deck counts are tiny), and nothing evaluates the final network: the run ends at
`b5.max_learner_steps` with `rnad-final.pt` and its export. **Run the benchmark after every run**
(`rmlearn/benchmark.py`):

```sh
learn/rmtrain bench r1 --wait            # waits for r1 to stop, then plays (CPU only, ~3 min per policy)
learn/rmtrain bench r1 --policies bc-final,m3,final --label m3-check --matches 960
learn/rmtrain bench r1 --opponent scripted:hard+sb       # the same against the sideboarding tier: bench/end-vs-hard_sb/
```

- **Policies.** Export names of the run (`exports/<name>.ts.pt`) or export paths, each optionally
  `name@<nn options>` (default `argmax`) and `<label>=` in front, to tell apart policies of several
  runs (`ctl-m3=runs/r1/exports/m3.ts.pt`). A run's name with a checkpoint but no export (a fixed
  point `ckpt/fixed/m<k>.pt` whose evaluation never ran, `final` from `ckpt/rnad-final.pt`) is
  exported first (TorchScript, CPU, a few seconds). The default set: `bc-final` (the DAgger policy: the
  **baseline**), every R-NaD fixed point `m<k>`, `final` at argmax, and `final@<eval.nn_options_finetune>`
  (the fine-tuning projection B5 trained for, sampled). A missing export is skipped with a note.
- **Opponent.** `--opponent` (default `scripted:hard`, the opponent every run's evaluations use)
  takes any `scripted:<tier>`; `scripted:hard+sb` sideboards per the guides
  (`python/sideboard_guides.json`) except in its ten held-out matchups. Against another opponent
  than `scripted:hard` the directory is `bench/<label>-vs-<tier>/` (`end-vs-hard_sb`;
  `benchmark.bench_dir`), so results against different opponents never mix; each `report.json`
  records its opponent (one without the key was played against `scripted:hard`), and one played
  against another opponent is refused rather than reused.
- **Matches.** Each policy plays `--matches` (default 480) bo3 matches against the opponent on
  `--actors` (default 12) engine actors, as side a of `matches.eval_schedule` over all 45 league
  pairs, both orders (so both decks), seats by `swap: sides`. Every policy uses the same run seed,
  actor ids and per-actor match count, so each meets the same deck pairs, seats and game seeds
  (common random numbers): the policies' differences are not the draw of the matches. At 480
  matches each deck is the policy's in ~48 matches (~110 games) and the opponent's in as many:
  about ±0.09 (two match-clustered SEs) per deck, ±0.03 overall.
- **When.** After the run: `--wait` returns once the run is finished (phase `done`, its shutdown
  having finished the queued exports, `final` included), or has stayed down for 10 minutes (a failed
  run is restarted by systemd after `run.restart_delay_s`, so a short outage is not an end). On a running run it warns and plays anyway, competing with
  the training actors for the CPU.
- **Output** `<run>/bench/<label>/` (default label `end`): per policy a directory (the actors'
  results, status and logs, the decision logs, `report.json` — `matches.report`: per game, game 1
  apart from games 2–3, by deck, matchup, seat, play / draw, Wilson interval, match-clustered SE —
  and `report.txt`), then `summary.json` and **`summary.md`**: the overall table (win rate per
  decisive game, Wilson 95%, ± two SEs, game 1, games 2–3, the difference from the baseline, draws,
  caps); per deck the policy's win rate **playing** it and **facing** it; each policy's per-deck
  difference from the baseline ± two SEs, **paired by match** (`benchmark.paired_delta`: the same
  seeds, decks and seats, so the shared luck cancels; with policies a few percent of decisions apart
  this is several times tighter than the win rates' own ±, and `same` gives the share of matches
  played exactly as the baseline did); each policy's 10 × 10 deck-vs-deck matrix; the
  **sideboarding** section (below); the findings (draws and caps, with their decision logs). A
  policy whose `report.json` exists (against the same opponent) is not played again: an interrupted
  benchmark resumes, and `--policies` with the same `--label` adds policies to an existing summary.
- **Sideboarding** (`rmlearn/sb_report.py`, from the results lines' `sb`, `docs/actor.md` §4; in
  `summary.json` each policy's `sideboarding`: `policy`, and `opponent` when the opponent is not
  `scripted:hard`, which always takes Done). Per policy, from its own seat's phases before games 2
  and 3 (the opponent's in separate rows), overall and per deck: the **swap rate** (the share of
  phases with at least one swap) and the swap-count histogram (0 … 9, 10+); the five cards most
  moved in and most cut (copies, as the frames name them); the **next game's win rate** after a
  phase with swaps and after Done (decisive games, with counts; observational: the policy chose
  when to swap, so this mixes a swap's effect with where it swapped); and **agreement with the
  guides**: the share of phases whose resulting maindeck equals the guide's target for (deck,
  opponent deck, play / draw) (`python/sideboard_guides.py`'s `Guides.target`, the held-out rows
  included), held-in and held-out matchups apart (the held-out ones, where the teacher took Done,
  measure generalisation), held-in also on the play and on the draw. The maindeck is tracked through
  the match: the registered list (the league deck file), then each phase's cuts and cards in, so game
  3's phase starts from game 2's deck. Play / draw is the line's `first` (who played first in the
  game the phase precedes; without it, the previous game's loser, the same player after a draw).
  Card names are compared by their front face with accents folded (`Dead` for `Dead/Gone`,
  `Lórien Revealed` for `Lorien Revealed`). A benchmark's policies play argmax and its entries carry
  no `sb_explore`, so no phase is explored; any that were would be left out and counted, as are
  games 2–3 whose lines lack `sb` (an engine from before the key).

Read the per-deck tables before the overall rate: in league self-play the decks are far from even
(the fast decks win most games with any pilot), so an overall rate near 0.5 can hide a policy that
pilots the simple decks well and the complex ones (doomsday, control, reanimator) badly.

## 8. Durability

- **Checkpoints** every `checkpoint.every_minutes` (once the GPU process holds a model) and at
  every phase end, written by the GPU process (a temporary file renamed): BC — the model checkpoint
  and the optimiser state beside it (`BCTrainer.save`); R-NaD — `learner.state_dict()` (networks,
  target, regularisation policies, Adam, counters, generators) and the replay window's file lists
  (the episodes in the window, those not yet taken), not their contents. `state.json` records the
  checkpoint after it is written.
- **Resume** (after `stop`, a crash or `kill -9` of everything): the phase and stage from
  `state.json`; BC reloads its checkpoint and re-reads the newest episodes up to the window's
  limit; R-NaD reloads the learner and re-reads the checkpoint's window; the results files are read
  on from the recorded offsets (sideboard exploration's counts with them, §6.1); actors start after
  their last match. An evaluation that was running
  (`state.json`'s `eval.running`) is queued again and played from its start. At most one checkpoint
  interval of learning is lost.
- **Which self-play games are fresh after a resume.** A checkpoint polls the window first, so it
  lists every episode file on disk at that moment: those in the window and those the learner had
  not taken yet (fresh). The restore offers only the latter as fresh. Every other episode already
  on disk is left out (kept on disk, never trained on): those the learner took before the checkpoint,
  and those written *after* the checkpoint but before the crash. The latter were taken by the lost
  steps; they are not trained on again because they were played by policies the restored learner
  never had, under versions it issues again for other weights (the policy and regularisation
  versions are learner steps), so their recorded evaluations would be taken for its own. The cost
  is at most one checkpoint interval of games; games written after the restore are fresh.
- **Progress after a resume** is the restored learner's: its step and iteration from the
  checkpoint (a `restore` metrics line names them and the run's previous ones; the pre-crash
  learner metrics leave the status), then the GPU process's live step. What the lost steps made is
  rolled back: the ladder rungs of later iterations, their queued evaluations, and the ladder
  record's players of the lost steps (`m<k>` past the restored iteration, R-NaD `v<version>` past its
  step — names the restored learner issues again for other weights). Crossing a boundary again
  writes the fixed point anew, adds its rung once and queues its evaluation.
- **Config keys added since a run was created** take their defaults when it resumes (`start` names
  them; `config.json` is rewritten).
- A transient failure of a command to the GPU process is fatal for the run (exit 2) by design:
  the restart resumes from the checkpoint.

## 9. Checks

```sh
learn/.venv/bin/python learn/check_rmtrain.py            # CPU, no engine, ~4 s: RMTRAIN-CHECK: ok
learn/.venv/bin/python learn/check_rmtrain.py --smoke    # GPU + bin/release: RMTRAIN-SMOKE: ok
learn/.venv/bin/python learn/check_rmtrain.py --smoke --policy-eval remote   # the server path
```

The CPU checks: config parsing and validation, local evaluation (the schedules' in-process specs,
the GPU process's save cadence, the live exporter's coalescing against a fake exporter, the actors
waiting for the first export and `actors.local_count`, local evaluations exporting their subject
and rungs first — reusing an export newer than its checkpoint, re-exporting an older one, a failed
export an `eval_failed` —, `rnad.fresh_order`, the policy-lag cap and percentiles), the control protocol, an orchestrator over a fake
run driven through its socket and its status from the files, the crash-loop guard (three fatal
stops without progress ending in exit 4, progress starting the count over, the FAILED status with
the last message and the resume command, `start` resetting the count, the unit's
`RestartPreventExitStatus`), and the resume bookkeeping (actor
start indices, the results tail, the actor pool's crash restarts and back-off against a fake engine,
the window restore, the window's decision-size cap (an episode over `replay.max_decision_tokens` left
out whole, counted and logged; a lowered cap live; the default on a run created before the key; the
status line), the attention budget (`forward_groups`' limits; a server batch cut by
`server.attention_budget` and an R-NaD chunk cut by `rnad.attention_budget` or checkpointed give the
plain forwards' values, losses and gradients, the generators' stream unchanged), checkpoint pruning, the schedules and evaluators (with `actors.teacher_tier = "hard+sb"`:
the teacher command, B0's scripted seats and DAgger's labels at that tier, every evaluation opponent
`scripted:hard`), `bc.sideboard_weight` on a synthetic batch (1: the plain mean exactly; the weighted
mean otherwise), `rnad.reg_eps_sideboarding` at 0 and 0.01 (finite log π and gradients), structured sideboard exploration
(`p_explore`'s hold, ramp and end, 0 outside B4, live keys and refusals; the counts from synthetic
results lines — the seat's deck with and without the swap, explored vs policy phases, a card once per
phase —; the weights `(1 + n)^-power`; the self-play schedule's `sb_explore` with both decks' weights,
none at p 0, in B5, B0, DAgger and evaluations; the orchestrator's snapshot cadence, its metrics line,
the status line, the counts in `state.json`; the league deck files' card keys; and, when a built
engine knows `sb_explore`, the engine's `--actor --actor-matches 0` reading the written schedule), an SPRT evaluation stopping
when its test decides, the request split, the ladder's ratings), and the first validation run's
findings: a B4 resume at an earlier step (progress, metrics and checkpoint names follow the restored
learner; the lost iteration's rung, evaluation and ladder players dropped; a re-crossed boundary
adds one rung), episodes the learner took before the checkpoint or that were written after it not
fresh again, an iteration's evaluation loading its fixed point (also after a pause) while a cadence
one snapshots the current network, an evaluation whose actors all fail reported as an error,
colliding groups refused, argmax by default with the sampled groups optional and every result
labelled with its options, and draws as findings in the status; the benchmark's default policy set
and its summary (per-deck playing and facing rates, the difference from the baseline, the tables), its
sideboarding report (from synthetic results lines: the swap rate and histogram, the top cards, the
next game after swaps vs Done, guide agreement on the play and on the draw with a game-3 flip carried
from game 2's maindeck, held-in vs held-out, explored and missing phases left out, the opponent's side
apart) and `--opponent` (its directory, report reuse against the same opponent only). The smoke runs a tiny run
(3 actors, d 64 / 2 layers, B0 5,000 decisions, BC ended by `phase next`, one DAgger round of
3,000 decisions, R-NaD with Δm 30 and G 2) detached under systemd, exercises every command, kills
the whole unit with SIGKILL in B4 (systemd restarts it; it resumes at the checkpoint), stops and
starts it again, and runs it to `done`; per-iteration SPRT evaluations and one exploiter
(`rmtrain eval --evaluators exploiter`) run on the way. It checks that no evaluation failed, that
each iteration's evaluations evaluated its fixed point, that every result names its options (an
argmax and a sampled group with `eval.also_sampled`), that the ladder's rungs stay unique across the
kill, and the resumes' `restore` lines. In local mode (the default; `--policy-eval remote` runs the
server path) it also checks that the live exports followed the learner, that DAgger played one,
that every self-play game's policy versions are exported versions, that the learner recorded no
regularisation log π and recomputed them, that the steps report the lag percentiles, and that the
local evaluations ran.

Measured (the training box, after T7; other agents' jobs on the GPU and CPU): B0 ~2 s (5,300
decisions on 3 actors), B1 ~20 s (the commands exercised, then `phase next`), B2 ~14 s (one round:
3,000 decisions, its evaluation), B3 < 1 s, B4 ~4.8 min (250 learner steps, 8 iterations and their
evaluations, a `kill -9` and systemd's restart 46 s later, a stop / start), the whole smoke ~6.5 min,
of which the GPU process spent ~1.7 min waiting for the GPU lock other jobs held.
