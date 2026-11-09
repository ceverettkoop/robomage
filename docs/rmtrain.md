# rmtrain: the training program

`learn/rmtrain` runs one training run from the command line: the teacher corpus, behaviour cloning,
DAgger, R-NaD self-play and fine-tuning (phases B0–B5, §6), detached, for days, resuming from its
last checkpoint after a crash, and steerable while it runs. The code is `learn/rmlearn/rmtrain.py`
(the CLI) and `learn/rmlearn/orchestrator/`.

```sh
make BUILD=RELEASE && make BUILD=RELEASE actor          # bin/release/robomage_nn, rmfeat
learn/rmtrain defaults > my.toml                         # every key, its default and what it does
learn/rmtrain start r1 --config my.toml                  # runs/r1; detaches and returns
learn/rmtrain status r1                                  # also when it is not running
learn/rmtrain tail r1 gpu                                # follow a log (tail -F)
learn/rmtrain set r1 rnad.eta 0.15                       # a live parameter
learn/rmtrain stop r1                                    # checkpoint, stop; `start r1` resumes
learn/rmtrain bench r1 --wait                            # after the run: its policies vs scripted:hard (§7.1)
learn/rmtrain retable r1                                 # a stopped run onto the engine's tables (§1.1)
learn/rmtrain add-decks r1 boros_aggro                   # a deck into the run's league, live or stopped (§1.1)
learn/rmtrain bench r1 --decks league,heldout --label heldout   # held-out decks too (§7.1)
```

A layout change (not a vocab expansion) migrates the checkpoints first: §1.2.

## 1. Starting and resuming

`rmtrain start <run> [--config FILE] [--set section.key=value …] [--foreground]`

- `<run>` is a name (`runs/<name>`, git-ignored) or a path.
- A new run: the config file (TOML, or JSON by its `.json` extension) over the defaults, then
  `--set`, validated (an unknown key, wrong type or bad value is refused by name), written as
  `config.json` (the live, resolved config) and `config.toml` (as created, documented).
- An existing run is **resumed** from its `config.json` (`--config` refused; `--set` may change any
  key, also non-live ones such as `model.*`, at your own risk — but not `run.feature_history`). Phase
  and stage come from `state.json`, the GPU process reloads the last checkpoint, the actors start
  after their last match. A finished run (phase `done`) is refused.
- Detached (default): `systemd-run --user --unit=rmtrain-<name>-<hash>` with
  `MemoryMax=run.memory_max` (the whole run: orchestrator, GPU process, actors, teachers),
  `KillMode=mixed` (a `systemctl stop` signals only the orchestrator, which stops the rest in order)
  and `Restart=on-failure` (`run.restart_on_failure`, after `run.restart_delay_s`): a failed run is
  restarted and resumes from its checkpoint. After `run.crash_loop_fatals` fatal stops in a row
  without progress it exits with status 4 (`RestartPreventExitStatus=3 4`; 3: no `config.json`) and
  `status` shows FAILED with the last fatal message (§5). An explicit `start` resets the count.
  `start` returns once the control socket answers, printing the status. Without a usable
  `systemd-run --user` it double-forks a plain daemon and warns: no memory limit, no restart. After
  a reboot the transient unit is gone: `rmtrain start <run>` resumes it.
- `--foreground` runs the orchestrator in the terminal (log echoed; Ctrl-C stops cleanly).
- **At every start** the run's tables are checked against the engine's (`orchestrator/retable.py`
  `check_start`): the engine's whole-vocab cards frame must equal the run's `cards.rmstate` byte for
  byte and the tables must carry a model manifest (`docs/featurizer.md` §6.2), else a fatal error
  names `rmtrain retable <run>`. When the tables differ from those `state.json` records (`tables`),
  every export of `state.json` is queued again from its checkpoint, the evaluations' and live exports
  removed (remade on demand), and a `tables_changed` metrics line written. An external `run.tables`
  directory is exempt from both checks. Both kinds of tables must be of the run's
  `run.feature_history` (their `spec.json`'s `history`), else the start stops.

### 1.1 Adding decks: `rmtrain retable <run>` and `rmtrain add-decks <run>`

A deck onboarded into `decks/league/` (`tools/onboard/README.md`) joins a run through
`league.decks`. It enters training through self-play from the current policy (no teacher, no BC),
its cards on the unknown-card path until a training batch holds them (`CardRows.card_seen`: the
first batch with one of its games marks all its cards, each id row starting from the unknown-card
vector).

```sh
learn/rmtrain stop r1                                    # the vocab is compiled into the engine
make BUILD=RELEASE && make BUILD=RELEASE actor PYTHON=learn/.venv/bin/python
learn/rmtrain retable r1                                 # the new cards' tables, every export rebuilt
learn/rmtrain add-decks r1 boros_aggro                   # league.decks (and league.pairs) in config.json
learn/rmtrain start r1                                   # resumes
```

When the deck's cards are already in the run's tables, `add-decks` on a **running** run does the
same live: `league.decks` and `league.pairs` are live keys, set together and validated as one
config; the orchestrator writes the new self-play schedule at its next tick and each actor plays it
from its next match. `rmtrain add-decks <run> STEM … [--weight W]` (`orchestrator/league.py`):

- refuses a deck already in the league, a held-out deck (`heldout/`; the config refuses one in
  `league.decks` / `league.pairs` too), a deck file not under `decks/league/`, a deck with a card
  the run's engine does not load (`--check-decks`: append it to the vocab, rebuild, retable first),
  and, on a running run, an engine whose cards frame is no longer the run's;
- **weights**: with `league.pairs` empty every pair weighs 1; with explicit pairs the new deck's
  pairs (against every league deck and the other new decks) get the mean weight of the run's pairs.
  `--weight W` overrides (on a league without explicit pairs it writes them out, old pairs at 1).
  With explicit pairs both keys must name the same decks, which `add-decks` keeps;
- a running run: one `set` of both keys (two `set` metrics lines); a stopped one: `config.json`
  rewritten and an `add_decks` metrics line.

Carried over unchanged: the replay window (card and ability ids are stable; the retable's lineage
accepts the old spec hashes), the learner and its regularisation policies (a fixed point keeps the
seen-card mask it was frozen with, so π_m,reg reads new cards as unknown until the next boundary),
the exports, the sideboard exploration counts (coverage cells re-read from the new league's deck
files), the evaluations (`eval.sprt_pairs` empty: the league's pairs, new ones included).

**Retable.** Appending cards (or changing how the engine parses a known card) needs a rebuild and a
stopped run. `retable` (refused while the run runs; `orchestrator/retable.py`):

1. writes the engine's cards frame and exports its tables into `<run>/tables.new/` (`--history N`
   for a run with `run.feature_history` N); if frame and spec hash are unchanged it says so, exit 0;
2. checks the old tables' manifest against the new one and every checkpoint (`ckpt/**/*.pt`)
   against the new tables (`spec.check_compatible`: same layout and text model, every card id of the
   model's vocab present under its name; new ids pass). Any refusal (a removed or renamed card, a
   layout bump: §1.2) changes nothing and is printed;
3. prints the **table drift** (`rmfeat --table-drift`): known cards whose card, ability or part rows
   changed (each column, old and new), new and removed cards — a changed known card reads other
   inputs than it was trained on, so the change must be intended;
4. keeps the old tables as `tables.<old spec_hash>/` (with their cards frame), moves the new ones to
   `tables/` and the frame to `cards.rmstate`, and writes `tables/lineage.json` (earlier tables,
   oldest first): the window and R-NaD's episode filter accept the lineage's spec hashes
   (`spec.Tables.accepted_spec_hashes`);
5. rebuilds every export of `state.json` (the latest per name: phase ends, fixed points, `final`)
   in `checkpoint.export_formats`, and removes the evaluations' and live exports;
6. records the change in `state.json` (`tables`; `retables`: from, to, kept directory, checkpoints
   checked, drift, exports rebuilt) and a `retable` metrics line.

### 1.2 After a layout change: `python -m rmlearn.migrate`

A change of the layout itself (a capacity exceeded, a `layout_version` bump — `rmfeat
--registry-raise`, `docs/featurizer.md` §6.1 —, or a checkpoint from before the layout registry) is
not a retable: the checkpoints' parameters are indexed by the old positions.
`learn/rmlearn/migrate.py` moves them by name (token types, numeric and categorical slots, enum
values, relations, table columns, card and ability slots), refusing any old name the current layout
lacks (except the removed archetype inputs); an old reserved position has no name and is not
carried. A network with the seen-card mask keeps it; one without gets `--seen` (a JSON list of the
card ids its training inputs held). An R-NaD learner state's replay window is emptied (self-play
refills it). A file whose manifest names another layout than the old tables' is refused. The old
tables are the run's (or `--old-tables` / `--old-cards`), exported before the engine was rebuilt:

```sh
cd learn && .venv/bin/python -m rmlearn.migrate --run ../runs/r1 --tables NEW_TABLES --keep layout1 \
    [--seen IDS.json] [--old-tables OLD_TABLES --old-cards OLD_CARDS]
learn/rmtrain retable r1                                  # the current tables, exports rebuilt
```

`--run` migrates every `ckpt/**/*.pt` (BC `.opt` files too) in place, keeping the originals in
`ckpt.<keep>/` and the old exports in `exports.<keep>/` (nothing is deleted; `ckpt/migration.json`
records each file's source, both sha256s, the seen set and any earlier migration as `previous`), so
`state.json`'s paths name the migrated files. `retable` then accepts the run across the bump (vocab
and text model still checked, the layout not: only migrated checkpoints pass) and keeps the old
tables as `tables.<old spec_hash>/` outside the lineage, so the window takes no episode of the old
layout. A single file: `python -m rmlearn.migrate OLD.pt --tables DIR [--seen IDS] --out NEW.pt`.
Its acceptance check is `learn/open_vocab_acceptance.py` (the gate's `actor` tier runs a subset).

## 2. Inspecting and steering

Every command but `start`, `defaults` and the file-only cases talks to the orchestrator over its
control socket (`/tmp/rmtrain-<hash>/control.sock`, recorded in `sockets.json`; a socket path must
stay under 108 bytes): one JSON request per connection, one JSON reply (`orchestrator/control.py`).

| Command | What |
|---|---|
| `status <run> [--json]` | phase and stage, hold / pause, phase progress, games and decisions (and per second over the last minute), actor health, the GPU process (mode, inference server stats: rows/s, batch mean, queue p99, clients, served policies and versions; GPU memory; the replay window), the latest metrics of each kind, evaluations running / queued / last (subject, version, options; an error when failed), sideboard exploration in B4–B5 (§6.1), findings (§7), the last checkpoint, recent transitions, the fatal-stop count (§5). When not running: the same from `status.json` and `state.json`, marked NOT RUNNING (FAILED after a crash loop, with the last fatal message and the resume command) |
| `tail <run> [component] [-n N] [--no-follow]` | `orchestrator` (default), `gpu`, `export`, `metrics`, `evals`, `a<k>` (an actor), `actors`, `all` |
| `pause <run>` | actors stopped (SIGTERM: a match in progress is written off; not SIGSTOP, since the engine's teacher and server timeouts are wall-clock), learner stops stepping, evaluations stopped and queued again. Kept across restarts |
| `resume <run>` | undoes `pause` |
| `stop <run>` | clean shutdown: evaluations cancelled, actors SIGTERM, queued exports finished, a checkpoint, the GPU process stopped; waits until the run is gone |
| `set <run> <section.key> <value>` | a live parameter (below), validated, applied at once and written to `config.json`; a `set` metrics line |
| `eval <run> [--evaluators a,b]` | queue an evaluation of the current policy (as it is when the evaluation starts) |
| `phase <run> hold\|release\|next\|status` | `hold` blocks the next transition (the exit test still runs; `status` says why); `release` lifts it; `next` forces the current phase's (or stage's) exit, also when held |
| `bench <run> [--wait] [--policies …] [--decks …] [--focus …]` | the end-of-run benchmark (§7.1); runs in the terminal |
| `retable <run>` | a stopped run onto the engine's tables (§1.1); runs in the terminal |
| `add-decks <run> STEM … [--weight W]` | league decks into the run (§1.1), live or into `config.json` |
| `config <run>` | the resolved config as TOML |
| `defaults` | the default config as TOML, documented |

**Live parameters** (`config.LIVE`): `bc.lr`, `bc.report_every`, `bc.eval_every`; `rnad.eta`,
`rnad.learning_rate`, `rnad.entropy_schedule_sizes` (Δm; an integer is a one-size schedule; a
change can move α and the next boundary), `rnad.entropy_schedule_repeats`, `rnad.reg_eps`,
`rnad.reg_eps_sideboarding` (also applied to the smoothed π_0,reg while it is a regularisation
policy), `rnad.push_every`, `rnad.report_every`, `rnad.games_per_step`, `rnad.value_after_every`,
`rnad.max_policy_lag`, `rnad.fresh_order`; `replay.max_decision_tokens`; `actors.count`,
`actors.local_count`, `actors.export_every_s`, `actors.export_every_steps`; the phase limits
(`b0.decisions`, `b1.*` but `play_eval`, `b2.rounds_*`, `b2.decisions_per_round`,
`b2.min_steps_per_round`, `b2.min_improvement`, `b4.iterations`, `b4.max_learner_steps`,
`b4.nn_options`, `b5.max_learner_steps`); `league.decks`, `league.pairs` (set together, §1.1);
`eval.matches`, `eval.ladder`, `eval.ladder_matches`, `eval.actors`, `eval.every_minutes`,
`eval.per_iteration`, `eval.also_sampled`, `eval.sprt_max_games`, `eval.exploiter_every_minutes`,
`eval.exploiter_minutes`; `checkpoint.every_minutes`; every `sb_explore.*` key (§6.1).

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
| `run.crash_loop_fatals` | 3 | fatal stops in a row without progress before exit status 4 (not restarted; 0: never; §5) |
| `run.start_from` | `""` | a BC checkpoint (e.g. another run's `ckpt/bc-final.pt`): start at B3 from it, B0–B2 skipped, its model config |
| `run.feature_history` | 0 | recent-event history tokens (`docs/featurizer.md` §3.7), at most N events (0: off). Fixed at creation (not live; a resume changing it is refused). Passed to the tables export (`--history N`, at start and in `retable`) and every engine the run starts (`--feature-history N`: training, evaluation, ladder and exploiter actors; `rmtrain bench`'s default). Changes the spec hash, not the layout hash: a checkpoint of either setting can start a run (`run.start_from`); the window keeps only the tables' spec; exports are rebuilt per setting |
| `league.decks` / `league.pairs` | the eleven decks of `config.py`'s `LEAGUE` (not read from `decks/league/`; add others with `add-decks`) / every pair | matchups and weights (`[[a, b, w], …]`, naming exactly `league.decks`); never a `heldout/` deck; live |
| `league.format`, `league.knowledge`, `league.max_game_decisions` | `bo3`, `expert`, 5000 | |
| `league.omniscient` | true | B0/B2 episodes carry the omniscient part (aux targets) |
| `actors.count` | 26 | training actors |
| `actors.teacher_tier` | `hard` | the scripted teacher's tier (`python/teacher.py --tier`): both seats of B0's scripted corpus, the scripted seat of its explore matches, every DAgger label. `hard+sb` sideboards per the league guides, so BC learns to sideboard. Self-play has no teacher; evaluation opponents stay `scripted:hard` regardless |
| `actors.policy_eval` | `local` | how training actors (DAgger, self-play) evaluate the policy: `local` (in process, §5.1) or `remote` (the GPU process's server) |
| `actors.local_count` | 16 | `local`: training actors while they play the network (0: `actors.count`; live); more play fewer decisions in all on a 16-core box |
| `actors.export_format`, `export_every_s`, `export_every_steps` | `ts`, 30, 0 | `local`: the live export's format and cadence (seconds; learner steps when > 0; live) |
| `actors.restart_delay_s`, `crash_loop_restarts`, `crash_loop_window_s`, `max_backoff_s` | 2, 5, 300, 600 | actor crash handling (§5) |
| `model.d`, `heads`, `layers`, `id_dropout`, `omniscient_critic` | 192, 4, 4, 0.1, false | `ModelConfig`. `id_dropout`: probability a card slot reads the unknown-card vector in training (the path an unseen card plays; 0.2–0.25 for runs meant to generalise to unseen decks); a training lever, not part of the contract; applies to a resumed model and a `run.start_from` checkpoint |
| `replay.max_decisions`, `replay.max_gb` | 4M, 40 | the window (BC's and R-NaD's) |
| `replay.heldout_fraction` | 0.05 | BC's held-out games |
| `replay.max_decision_tokens` | 640 | the window leaves out a whole episode with a decision of more seat tokens (menu included; 0: no cap; live: applies to episodes ingested from then on, a lowered cap drops window episodes over it). Each is logged once in `gpu.log` and counted (`skipped_oversize`, `oversize_max_tokens` in `bc_eval`, `rnad_step`, `status`; `bc_train`'s `window_skipped_oversize`). Actors still play such games (§5 Memory) |
| `replay.delete_evicted_selfplay` | true | evicted self-play `.rmtraj` deleted (RMLOGs kept) |
| `server.max_batch`, `deadline_us`, `precision` | 512, 2000, fp32 | the RMIP server |
| `server.attention_budget` | 2^26 | most attention cells (rows × padded tokens², per head and layer) per forward: a batch runs sorted by token count in forwards of at most 64 decisions and this many cells (§5 Memory) |
| `b0.decisions`, `b0.explore_frac`, `b0.corpus_dir` | 5M, 0.2, `""` | the corpus; a corpus directory skips B0 |
| `bc.*` | `BCConfig` | batch 128, lr 3e-4, warm-up 300, smoothing 0.05, fp16; `eval_every` 2000, `report_every` 250 |
| `bc.sideboard_weight` | 1.0 | policy-loss weight of a sideboarding decision (others 1; batch loss `sum(w·loss) / sum(w)`; B1 and B2; value and aux unweighted). Reported as `train_policy_sideboarding`, `train_agree_sideboarding`, `train_frac_sideboarding` (`bc_train`), `eval_agree_group:sideboarding`, `eval_policy_loss_group:sideboarding` (`bc_eval`) |
| `b1.min_steps`, `max_steps`, `patience`, `min_delta`, `play_eval` | 2000, 30000, 3, 0.002, true | B1's exit |
| `b2.rounds_min`, `rounds_max` | 3, 5 | DAgger rounds |
| `b2.teacher_mix_start`, `teacher_mix_end` | 0.5, 0 | linear over the rounds |
| `b2.decisions_per_round`, `min_steps_per_round`, `min_improvement` | 1M, 2000, 0.01 | |
| `rnad.eta`, `learning_rate`, `entropy_schedule_sizes` | 0.2, 5e-5, [10000] | `RNaDConfig` |
| `rnad.games_per_step`, `chunk_len`, `precision` | 32, 32, fp16 | G × chunk (fits d 192 in 12 GB) |
| `rnad.forward_tokens`, `attention_budget`, `checkpoint_tokens` | 65536, 2^26, 2^17 | a chunk's forwards group rows of similar token counts, each at most `forward_tokens` padded tokens and `attention_budget` cells; a chunk over `checkpoint_tokens` tokens or `attention_budget` cells in all checkpoints each forward (recomputed in the backward; same gradients, ~15% slower step; §5 Memory) |
| `rnad.reg_eps`, `reg_eps_sideboarding` | 0.1, 0.25 | π_0,reg's smoothing (B3), the latter at sideboarding steps. With `actors.teacher_tier = "hard+sb"` set `reg_eps_sideboarding` to about 0 (π_0,reg is then the taught sideboarding; exploration is the actors'). 0 is valid |
| `rnad.push_every` | 50 | learner steps between pushes of policy 0 |
| `rnad.max_policy_lag` | 0 | drop a self-play game whose oldest step's policy is more than this many learner steps old (0: off; live) |
| `rnad.fresh_order` | `newest` | which fresh games a step takes first: `newest` (least lag) or `oldest` (live) |
| `rnad.value_after_every` | 10 | steps between the value diagnostic's after-update forwards (0: never): `value_mse_first_sight` (every step), `value_mse_after`, `value_mse_gap` in `rnad_step` and `status` |
| `b4.iterations`, `b4.max_learner_steps`, `b4.nn_options` | 0, 0, `""` | B4's exit (0: until `phase next`); self-play sampling options |
| `b5.max_learner_steps` | 0 | the run's end (0: never) |
| `sb_explore.p_start`, `p_end`, `hold_frac` | 0, 0, 0.5 | `p_explore` per self-play seat and sideboarding phase (§6.1): `p_start` until `hold_frac` × `b4.max_learner_steps`, then linear to `p_end` at `b4.max_learner_steps` (`p_start` throughout when that is 0); 0 outside B4 and in every evaluation (live) |
| `sb_explore.count_weights` | 11 × 1 | weights of an exploring phase's swap count k = 0 … 10 (live) |
| `sb_explore.coverage`, `coverage_power` | true, 1 | a card's in- (out-) weight in a matchup is `(1 + n) ** -coverage_power`, n the explored phases that moved it in (out); `false` or 0: uniform (live) |
| `sb_explore.rewrite_every_s` | 600 | seconds between schedule rewrites with the current counts; also when `p_explore` moves by 0.01 (or onto / off 0), at B4 → B5, or when an `sb_explore` key is set (live) |
| `eval.evaluators` | `["sprt", "fixed", "ladder"]` | evaluators of a per-iteration / cadence evaluation (`exploiter` too; §7) |
| `eval.opponents`, `eval.pairs`, `eval.matches` | scripted, explore, random; 4 pairs; 48 | `scripted` is `scripted:hard`; `scripted:<tier>` names another tier |
| `eval.ladder`, `eval.ladder_matches` | 3, 24 | |
| `eval.actors`, `eval.every_minutes`, `eval.per_iteration` | 4, 60, true | |
| `eval.policy_eval` | `local` | evaluation actors: `local` (subject and ladder rungs exported, played in process; the exploiter stays remote) or `remote` |
| `eval.nn_options_bc` / `_rnad` / `_finetune` | `argmax` / `argmax` / `thresh=0.03:disc=16` | the evaluated policy's options per phase (the primary result; §7) |
| `eval.also_sampled` | false | `fixed` also plays the policy with the actors' sampling options (groups `<opponent>@sampled`) |
| `eval.sprt_opponents`, `sprt_pairs` | `["scripted"]`, every league pair | `sprt`'s opponents and deck pairs |
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
  state.json                     phase, stage, hold, pause, counters + results offsets, sb_explore counts,
                                 last checkpoint, ladder, evaluation queue and history, transitions,
                                 findings, crash_loop, tables identity and retables (atomic, every ~10 s)
  status.json                    the last status (every run.status_every_s)
  sockets.json  daemon.json      the socket paths; how it was detached (unit name or pid)
  metrics.jsonl                  one line per learner report, held-out evaluation, evaluation (eval /
                                 eval_failed), checkpoint, export, transition, set / pause / hold,
                                 start / restore / stop, sb_explore, retable / tables_changed, fatal /
                                 crash_loop: time, phase, stage, step
  evals.jsonl                    every evaluation's full result (per opponent, per pair, game 1 vs 2–3)
  schedule.json                  the actors' schedule (docs/actor.md §2)
  cards.rmstate  tables/         the cards frame and card tables (manifest.json, registry.json, lineage.json)
  tables.<spec_hash>/            earlier tables a retable replaced (with their cards.rmstate)
  logs/orchestrator.log gpu.log export.log actors/a<k>.log
  actors/                        a<k>.status.json, a<k>.results.jsonl (the actors' --actor-out)
  episodes/b0 dagger selfplay    trajectory episodes (.rmtraj)
  rmlog/b0 dagger selfplay       every match's decision log (always kept)
  ckpt/                          bc-p<step>.pt (+ .opt), rnad-p<step>.pt: periodic, the last checkpoint.keep;
                                 bc-b1.pt, bc-r<k>.pt, bc-final.pt, rnad-seed.pt, rnad-final.pt: phase ends
  ckpt/fixed/m<k>.pt             the R-NaD fixed points (π_m,reg at each boundary)
  exports/<name>.ts.pt .pt2      exports of phase-end checkpoints and fixed points (nn:<model>)
  exports/live/current.ts.pt     local actors: the current policy's export
  exports/eval/<name>-v<n>.ts.pt local evaluations: the subject's and ladder rungs' exports
  eval/<seq>-<label>/            an evaluation: schedule, actors' status / results, rmlog, logs
  ladder/ladder.json             the ladder record (§7)
  bench/<label>[-vs-<tier>]/     benchmarks (§7.1)
```

## 5. Processes

| Process | What |
|---|---|
| orchestrator | `python -m rmlearn.rmtrain _run RUN`: the phase loop (every 0.5 s), the control socket, `state.json` / `status.json` / `metrics.jsonl`. No GPU, no torch |
| GPU process | `python -m rmlearn.orchestrator.gpu RUN`: **the only process that touches the GPU**: the learner (BC: `BCTrainer` and its window; R-NaD: `RNaDLearner`, `ServerLink`, the self-play window) and the RMIP `InferenceServer` at `/tmp/rmtrain-<hash>/rmip.sock`. Holds the flock `run.gpu_lock` while it lives (fails at once naming the lock, or waits with `run.gpu_lock_wait`). Commands as JSON lines on stdin, replies and events on stdout (fd 1 redirected to `gpu.log`); ignores SIGTERM / SIGINT, exits on end of input |
| actors | `bin/<build>/robomage_nn --actor` (docs/actor.md) on `schedule.json`, GPU hidden (`HIP_VISIBLE_DEVICES=` …), `OMP_NUM_THREADS=1`, one process group each (with their `python/teacher.py`) |
| evaluation actors | the same with `--actor-matches`, in `eval/<seq>-<label>/` |
| exporter | `python -m rmlearn.orchestrator.exporter` (CPU), one at a time: `.ts.pt` / `.pt2` of a checkpoint |
| live exporter | `python -m rmlearn.orchestrator.exporter --serve` (CPU, kept running): local actors' and local evaluations' exports (`orchestrator/live.py`) |

### 5.1 Local evaluation (`actors.policy_eval = "local"`)

With `remote` the actors evaluate every decision through the GPU process's server, sharing the GPU
with the learner. With `local` each actor evaluates in process (`nn:<export>`, one CPU thread:
libtorch at one intra- and inter-op thread, `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1`), the GPU only
learns, and the learner computes the regularisation policies' log π itself.

- **Weights.** In R-NaD the GPU process saves the current network (`exports/live/src-v<step>.pt`)
  when B4 starts and then every `actors.export_every_s` (or `export_every_steps`) when the learner
  stepped (a `policy_saved` event); in DAgger once at each round's start (the round's frozen
  policy). The live exporter writes it to `exports/live/current.ts.pt` (temporary file renamed over
  it; a newer save replaces one still queued). The schedule names that file
  (`nn:<run>/exports/live/current.ts.pt`) and each actor reloads it before a match when it changed
  (docs/actor.md §7), not between games of a match. After the GPU process loads a model (start,
  resume, R-NaD seeding) or a DAgger round starts, the actors wait for that model's first export
  (`status`: "waiting for the live export"). Each export is a `live_export` metrics line (version,
  seconds, latency).
- **Format:** TorchScript (`actors.export_format = "ts"`); an AOTInductor `.pt2` takes ~16 s to
  compile, too slow for the cadence.
- **Policy lag.** Each record's `policy_version` is the export's learner step. The learner's step
  metrics report the lag per report window (`policy_lag_p50`, `_p95`, `_max`, `_mean`);
  `rnad.max_policy_lag` and `rnad.fresh_order` bound it.
- **Regularisation log π.** Local trajectories record only the played policy's evaluation, so the
  learner recomputes every step's π_m,reg and π_m−1,reg log π (`stale_reg = "recompute"`,
  `reg_recomputed` in the metrics): no-grad forwards of the frozen networks in the step's precision,
  on the step's row groups and inputs; one network while π_0,reg is in both slots, two after.
- **Evaluations** (`eval.policy_eval = "local"`): the subject (a fixed point, or the current network
  saved by `save_model`) and each ladder rung are exported first (`exports/eval/<name>-v<version>.ts.pt`,
  reused while newer than their checkpoint), then played in process; the exploiter stays remote.

**Served policies:** 0 the current policy (BC: pushed at each DAgger round's start; R-NaD:
`ServerLink`, every `rnad.push_every` steps), 1 and 2 the regularisation policies (π_m,reg,
π_m−1,reg; raw networks, the learner smooths π_0,reg itself), 3 the evaluation subject, 10+ the
ladder rungs of an evaluation, 20 and 21 an exploiter in training and its victim (§7).

**Actor crashes.** An actor that exits unasked is restarted after `actors.restart_delay_s` at its
next match index (the status file's `next_match`, + 1 when a match was in progress, never below the
results file's last match + 1). More than `crash_loop_restarts` exits within `crash_loop_window_s`
back off exponentially up to `max_backoff_s`; `status` lists restarts and the last exit's reason.

**GPU process failure** (dying, or failing a command, even transiently) stops the run: actors
stopped, state saved, exit status 2; systemd (or `start`) resumes it from the checkpoint. **A crash
loop** ends after `run.crash_loop_fatals` fatal stops in a row without progress (same phase and
stage, and neither the step nor the decisions collected past the previous fatal stop's): exit status
4, not restarted. `state.json`'s `crash_loop` records the count, progress mark, last fatal message
and whether the run failed; `status` shows `FAILED (a crash loop)` with that message and the resume
command; `fatal` and `crash_loop` events are in `metrics.jsonl`. Fix the cause, then `rmtrain start
<run>` (resets the count). A unit keeps the `RestartPreventExitStatus` of the `start` that launched it.

**Memory.** The replay window holds the episodes' bytes (~13 KB per decision with the omniscient
part, ~5 KB without), bounded by `replay.max_gb`. Each actor is ~30 MB plus ~40 MB per teacher.
`run.memory_max` caps all of it; leave room for the page cache.

On the GPU attention is quadratic in a batch's padded token count. Almost every decision has at most
~600 seat tokens, but a rare legal game (hundreds of tokens on the board) has decisions of 2000+.
`replay.max_decision_tokens` keeps such games out of the window, so a BC batch pads to at most the
cap; the inference server still evaluates them, `server.attention_budget` putting long decisions in
forwards of their own. An R-NaD chunk (G × `chunk_len` decisions) back-propagates as one; past
`rnad.checkpoint_tokens` or `rnad.attention_budget` its forwards are checkpointed so the backward
holds one forward's activations at a time.

## 6. The phases

The orchestrator brings the processes to what the phase and stage want each tick (`sync`: the
schedule, the actor count, the GPU process's mode). Every transition is logged (`orchestrator.log`,
a `transition` metrics line with its reason) and blocked by `phase hold` (a transition decided by an
evaluation waits as *pending* until `release`).

| Phase | Stage | What | Exit |
|---|---|---|---|
| B0 | collect | the teacher corpus: both seats `first` + the scripted teacher (`actors.teacher_tier`) at mix 1 over the league (swap decks); a `b0.explore_frac` share teacher vs explore (scripted side recorded); episodes with the omniscient part | `b0.decisions` decisions (from results lines of schedule version `b0`). Skipped with `b0.corpus_dir` |
| B1 | train | BC on the window over `episodes/b0` (or the corpus dir) and `episodes/dagger`; a held-out evaluation every `bc.eval_every` steps | held-out agreement not improving by `b1.min_delta` for `b1.patience` evaluations after `b1.min_steps`, or `b1.max_steps` → `bc-b1.pt`, export |
| B1 | eval | evaluation `b1` (fixed opponents, argmax), learner idle | its result (the DAgger baseline) |
| B2 | collect | round r: the BC policy frozen for the round (in process, or server policy 0 via `nn:remote`) on both seats, sampling, the teacher on both seats at mix `teacher_mix_start → end`; BC keeps training on the growing window | `b2.decisions_per_round` decisions of version `b2-r<r>` and `b2.min_steps_per_round` BC steps → actors stopped, `bc-r<r>.pt`, export |
| B2 | eval | evaluation `b2-r<r>` (fixed opponents) | after `b2.rounds_min` rounds, a win rate vs scripted:hard not `b2.min_improvement` above the best so far, or `b2.rounds_max` → B3; else round r + 1 |
| B3 | | `bc-final.pt` (or `run.start_from`) → `RNaDLearner.from_bc_checkpoint` (π_0,reg smoothed by `rnad.reg_eps`, `reg_eps_sideboarding`) → `rnad-seed.pt` | at once |
| B4 | selfplay | both seats the current policy (local: `nn:<run>/exports/live/current.ts.pt`; remote: `nn:remote:…:policies=0,1,2`; + `b4.nn_options`) over the league; `sb_explore` while `p_explore` > 0 (§6.1); the learner takes each fresh game once (capped / quit games dropped); at each boundary the fixed point `ckpt/fixed/m<k>.pt`, its export and an evaluation `it<k>` (`eval.per_iteration`); evaluations of the current policy every `eval.every_minutes` | `b4.iterations` iterations or `b4.max_learner_steps` steps (0: never) |
| B5 | selfplay | the learner's fine-tuning projection from the current step (no sideboard exploration); evaluations with `eval.nn_options_finetune` | `b5.max_learner_steps` (0: never) → `done`: `rnad-final.pt`, exports, exit 0 |

### 6.1 Structured sideboard exploration (B4)

The actors' side is the schedule entry's `sb_explore` (`docs/actor.md` §2: a coin `p` per seat and
sideboarding phase; an exploring phase plays the proposal `q` with μ = `q`); the orchestrator's
(`orchestrator/sb_explore.py`) is `p_explore`'s schedule and the coverage weights.

- **Counts.** Every training actor's results line of a self-play game 2 or 3 (purpose `train`,
  schedule `b4` / `b5`) carries `sb` per engine seat. Seat A's matchup is (`deck_a`, `deck_b`), seat
  B's (`deck_b`, `deck_a`) (decks named after the swap coin). An explored phase adds to its matchup
  one phase, its swap count's bucket (0 … 9, 10+) and each distinct card moved in, and out, once; a
  phase the policy played adds a phase and a bucket to the policy's counts. Counts live in
  `state.json` (`sb_explore`), saved with the results files' offsets, so a restart counts no line
  twice; the results files are append-only.
- **Weights.** A card's in-weight in a matchup is `(1 + n) ** -coverage_power`, n the explored phases
  that moved it in (out-weight: out); an untried card needs no entry (`default_weight` 1). Names are
  the frames' (`Dead` for Dead/Gone).
- **The schedule.** Each B4 entry (deck a vs deck b) gets `sb_explore` = `{p, count_weights, weights:
  {deck a: its weights against deck b, deck b: against deck a}, default_weight: 1}` when `p` > 0. The
  snapshot is retaken on the triggers of `sb_explore.rewrite_every_s` (§3) and on a phase change; the
  schedule file is rewritten atomically when its text changes, and actors pick it up before their
  next match. B5, B0, DAgger and evaluation schedules carry no `sb_explore` (the engine refuses an
  eval entry with p > 0).
- **Status** (B4–B5), one line, e.g.
  `sb explore: p 0.250 (schedule 0.250); 5210 explored phases, per matchup min 41 / median 58 over 90;
  in-cells 1260 tried >=1 87%, >=10 31%; out-cells 2950 >=1 64%, >=10 9%; policy swap rate 0.083
  (48120 phases)`: `p` now and in the schedule; explored phases overall and per matchup; the shares of
  (matchup, sideboard card) in-cells and (matchup, maindeck card) out-cells tried ≥ 1 and ≥ 10 times
  (cells from the league deck files, names by front face, accents folded); the share of policy phases
  with a swap. `status --json`: `sb_explore`.
- **Metrics:** each new snapshot writes an `sb_explore` line: `p`, `explored_phases`,
  `policy_phases`, `matchups`, `phases_min`, `phases_median`, `explored_hist` and `policy_hist` (swap
  counts 0 … 9, 10+, cumulative), `policy_swap_rate`, `policy_hist_recent`,
  `policy_swap_rate_recent` (since the previous line), `in_cells`, `in_tried_1`, `in_tried_10`,
  `out_cells`, `out_tried_1`, `out_tried_10`, `weighted_matchups`.

## 7. Evaluation

`orchestrator/evaluation.py`: a registry of **evaluators** (`@evaluator("name")`, a function of the
context returning groups: named schedule entries, a match count, server policies to load). An
evaluation loads its subject as server policy 3 and its groups' policies, plays the matches with
`eval.actors` evaluation actors (`--actor-matches`), scores every game (`evaluate.summarize`: per
opponent and pair, game 1 apart from games 2–3), unloads, and appends to `evals.jsonl`,
`metrics.jsonl` (`eval`) and the status. Evaluations run one at a time, queued: at B1's end, after
each DAgger round, at each R-NaD iteration, every `eval.every_minutes` in B4–B5 (the exploiter on
`eval.exploiter_every_minutes`), and on `rmtrain eval`. `sprt` and `exploiter` stop on their own
terms, so a request naming them with others is split (`it3` → `it3`, `it3-sprt`).

**The subject.** An R-NaD iteration's evaluation (`it<k>`, `it<k>-sprt`) evaluates that iteration's
fixed point `ckpt/fixed/m<k>.pt`, loaded when the evaluation starts (also after queueing, a pause or
a restart), named `m<k>`; its own rung is not one of its ladder opponents. Every other evaluation
snapshots the current network when it starts, named `v<version>`. An exploiter always trains against
the current network.

**Options.** The subject plays `argmax` in every phase but B5 (`eval.nn_options_bc`, `_rnad`), so
results compare from B1 to B4; B5 plays the fine-tuning projection (`eval.nn_options_finetune`).
That is the **primary** result. With `eval.also_sampled` the `fixed` evaluator also plays the
sampling options of the phase's actors (`b2.nn_options` in B1–B2, `b4.nn_options` after; `""` =
temperature 1) as groups `<opponent>@sampled`: secondary, left out of the ladder's ratings. Every
result carries its `options` in `evals.jsonl` and `metrics.jsonl`; the status labels them.

**Failures.** An evaluation whose groups collide (a repeated group, schedule entry id or server
policy) or whose fixed point is gone is not played; one whose actors all fail plays no game. Either
is an error: an `ERROR:` line in the orchestrator log, an `eval_failed` metrics line, the error in
`evals.jsonl` and the status, no ladder record; the run goes on. Some actors failing is a warning.

Evaluation schedules give the subject both decks of each pair: `swap: sides` moves a side's deck
with its agent (docs/actor.md §2), so every pair is two entries.

- `fixed`: against `eval.opponents` (`scripted` = scripted:hard: `first` + the teacher at mix 1;
  `explore`, `random`, any agent spec) over `eval.pairs`.
- `ladder`: against the BC policy and the most recent `eval.ladder` R-NaD fixed points, each a server
  policy (10 + j) played with `eval.opponent_nn_options` (default `argmax`). Every evaluation but
  exploiters and failures also feeds the **ladder record** (`rmlearn/ladder.py`,
  `<run>/ladder/ladder.json`): each primary group's games as a pairing of the subject (`m<k>` /
  `v<version>`) with its opponent (a rung's label `bc` / `m<k>`, or `scripted`, `explore`, `random`),
  and a Bradley–Terry fit over all pairings (per game, weak prior, Elo scale, `scripted` at 0) written
  into `state.json`, `status` (`ratings`) and the evaluation's `evals.jsonl` line.
- `sprt` (`rmlearn/sprt.py`): against each of `eval.sprt_opponents` over `eval.sprt_pairs`, a Wald
  test on the **per-game** win rate, H1 p = `sprt_p1` against H0 p = `sprt_p0`, α `sprt_alpha`, β
  `sprt_beta`, fed game by game; the evaluation stops its actors once every test decides, or at
  `sprt_max_games` (verdict `undecided`). Draws and caps are excluded and listed. Group
  `sprt-<opponent>` carries the verdict, the llr and its bounds, and the per-game report
  (`matches.report`: game 1 apart from games 2–3, per matchup, deck, seat, play / draw, Wilson
  interval, match-clustered SE). Games of a match are correlated, so the error rates are somewhat
  optimistic; the clustered SE and design effect show by how much.
- `exploiter` (`rmlearn/exploiter.py`): a short PPO best response, trained **inside the GPU
  process**: `exploit_start` freezes a copy of the current network as the victim (policy 21) and
  trains another as the exploiter (policy 20, a new version per update) between the learner's
  units; with `eval.exploiter_pause_learner` the learner idles meanwhile. `eval.exploiter_actors`
  actors play exploiter (side a, recorded) vs victim over `eval.exploiter_pairs`: first
  `exploiter_baseline_games` of the initial exploiter (~50% by symmetry), then `exploiter_minutes` of
  training, then `exploiter_final_games` of the frozen final version, whose win rate is the measured
  exploitability (groups `exploiter-baseline`, `exploiter-final`; the full result with the per-update
  win-rate curve and PPO statistics under `exploiter` in `evals.jsonl`). Off by default; run one with
  `rmtrain eval <run> --evaluators exploiter` or set the cadence. Its PPO is untuned: read the figure
  as a lower bound on exploitability.

**Findings: draws.** A drawn game (CR 104.4a) is legitimate but rare enough to review. Every draw in
the training or evaluation actors' results is noted in `state.json`'s `findings` (count and the
latest ten: source, actor, match, game, decks, results file, decision log) and the orchestrator log;
`status` shows the latest three. They are not errors. Replay one with `cd bin &&
./release/robomage_nn --replay <rmlog> --narrative`.

### 7.1 The end-of-run benchmark

The run's own evaluations are small (four `fixed` pairs; `sprt` stops early) and none evaluates the
final network, so **run the benchmark after every run** (`rmlearn/benchmark.py`):

```sh
learn/rmtrain bench r1 --wait            # waits for r1 to stop, then plays (CPU only)
learn/rmtrain bench r1 --policies bc-final,m3,final --label m3-check --matches 960
learn/rmtrain bench r1 --opponent scripted:hard+sb       # against the sideboarding tier: bench/end-vs-hard_sb/
```

- **Policies.** Export names of the run (`exports/<name>.ts.pt`) or export paths, each optionally
  `name@<nn options>` (default `argmax`) and prefixed `<label>=` (`ctl-m3=runs/r1/exports/m3.ts.pt`).
  A name with a checkpoint but no export (a fixed point `m<k>`, `final` from `ckpt/rnad-final.pt`) is
  exported first (TorchScript, CPU). Default set: `bc-final` (the **baseline**), every `m<k>`, `final`
  at argmax and `final@<eval.nn_options_finetune>`. A missing export is skipped with a note.
- **Opponent.** `--opponent` (default `scripted:hard`): any `scripted:<tier>` (`scripted:hard+sb`
  sideboards per `python/sideboard_guides.json`, except in its held-out matchups) or an export path
  (a fixed policy at argmax). For another opponent than `scripted:hard` the directory is
  `bench/<label>-vs-<tier>/` (`benchmark.bench_dir`). Each `report.json` records its opponent (none:
  `scripted:hard`); one of another opponent is refused rather than reused.
- **Matches.** Each policy plays `--matches` (480) bo3 matches on `--actors` (12) engine actors with
  `--seed` (11), as side a of `matches.eval_schedule` over every pair, both orders, seats by `swap:
  sides`. Every policy uses the same seed, actor ids and per-actor match count, so all meet the same
  deck pairs, seats and game seeds (common random numbers). At 480 matches over the 55 league pairs
  a deck is the policy's in ~48 matches: about ±0.09 (two match-clustered SEs) per deck, ±0.03
  overall. `--engine` (default `bin/release/robomage_nn`), `--feature-history` (default the run's).
- **Decks** (`--decks`, `--focus`). Every pair of the listed decks: comma-separated `league` (every
  deck of `decks/league/`, `LEAGUE`'s first), `heldout` (every deck of `decks/heldout/`) or single
  decks (`x` = `league/x`, `heldout/x`); `--focus D[,…]` keeps only pairs holding one of them
  (`tools/onboard/onboard.py baseline`). Default: `LEAGUE`'s eleven decks. A `report.json` records its
  pairs; one of other pairs is refused (give it its own `--label`). With `--decks` the summary adds,
  per deck, the policy **facing** it with each policy's paired difference from the baseline, and the
  **deck groups** `league`, `league:original` (`LEAGUE`'s decks, when `league/` holds more) and
  `heldout`, each as `plays:G`, `faces:G`, `within:G`. `scripted:hard` pilots an unseen deck with
  generic heuristics only, so compare policies by their paired differences, not by raw rates on new
  decks. More decks mean fewer matches per deck: raise `--matches`.
- **When.** `--wait` returns once the run is finished (phase `done`, queued exports including
  `final` done) or has stayed down for 10 minutes (a failed run restarts after
  `run.restart_delay_s`). On a running run it warns and plays anyway, competing for the CPU.
- **Output** `<run>/bench/<label>/` (default `end`): per policy a directory (actors' results, status,
  logs, decision logs, `report.json` — `matches.report` — and `report.txt`), then `summary.json` and
  **`summary.md`**: the overall table (win rate per decisive game, Wilson 95%, ± two SEs, game 1,
  games 2–3, difference from the baseline, draws, caps); per deck the win rate **playing** and
  **facing** it; each policy's per-deck difference from the baseline ± two SEs, **paired by match**
  (`benchmark.paired_delta`: shared luck cancels; `same` is the share of matches played exactly as
  the baseline); each policy's deck-vs-deck matrix; the sideboarding section; the findings (draws
  and caps, with decision logs). A policy whose `report.json` exists (same opponent) is not played
  again: an interrupted benchmark resumes, and `--policies` with the same `--label` adds policies.
- **Sideboarding** (`rmlearn/sb_report.py`, from the results lines' `sb`, `docs/actor.md` §4;
  `summary.json`'s `sideboarding`: `policy`, and `opponent` unless it is `scripted:hard`, which
  always takes Done). Per policy, from its own seat's phases before games 2 and 3 (the opponent's in
  separate rows), overall and per deck: the **swap rate** and swap-count histogram (0 … 9, 10+); the
  five cards most moved in and most cut; the **next game's win rate** after a phase with swaps vs
  after Done (observational); and **agreement with the guides**: the share of phases whose
  resulting maindeck equals `Guides.target` (`python/sideboard_guides.py`) for (deck, opponent deck,
  play / draw), held-in and held-out matchups apart, held-in also on the play and on the draw. The
  maindeck is tracked through the match (the deck file, then each phase's swaps). Play / draw is the
  line's `first` (without it, the previous game's loser; the same player after a draw). Card names
  compare by front face, accents folded. Explored phases and games 2–3 without `sb` are left out and
  counted.

Read the per-deck tables before the overall rate: the league decks are far from even, so an overall
rate near 0.5 can hide a policy that pilots the simple decks well and the complex ones badly.

## 8. Durability

- **Checkpoints** every `checkpoint.every_minutes` (once the GPU process holds a model) and at every
  phase end, written by the GPU process (temporary file renamed): BC — the model and the optimiser
  state beside it (`BCTrainer.save`); R-NaD — `learner.state_dict()` (networks, target,
  regularisation policies, Adam, counters, generators) and the window's file lists (episodes in the
  window and those not yet taken), not their contents. `state.json` records it after it is written.
- **Resume** (after `stop`, a crash or `kill -9`): phase and stage from `state.json`; BC reloads its
  checkpoint and re-reads the newest episodes up to the window's limit; R-NaD reloads the learner
  and the checkpoint's window; results files are read on from the recorded offsets (sideboard
  counts with them); actors start after their last match; a running evaluation (`eval.running`) is
  queued again from its start. At most one checkpoint interval of learning is lost.
- **Fresh games after a resume** are only those the checkpoint listed as not yet taken. Other
  episodes on disk are never trained on (kept on disk): those taken before the checkpoint, and those
  written after it, whose recorded policy versions the restored learner would reissue for other
  weights.
- **Progress after a resume** is the restored learner's (a `restore` metrics line names its step and
  iteration and the previous ones). What the lost steps made is rolled back: later iterations'
  ladder rungs, their queued evaluations, and ladder players past the restored point (`m<k>` past
  the iteration, `v<version>` past the step). Crossing a boundary again writes the fixed point anew,
  adds its rung once and queues its evaluation.
- **Config keys added since a run was created** take their defaults on resume (`start` names them;
  `config.json` is rewritten).

## 9. Checks

```sh
learn/.venv/bin/python learn/check_rmtrain.py            # CPU, no engine, seconds: RMTRAIN-CHECK: ok
learn/.venv/bin/python learn/check_rmtrain.py --smoke    # GPU + bin/release: RMTRAIN-SMOKE: ok
learn/.venv/bin/python learn/check_rmtrain.py --smoke --policy-eval remote   # the server path
learn/.venv/bin/python learn/check_rmtrain.py --smoke --feature-history 32   # with run.feature_history
```

The CPU checks cover: config parsing and validation; local evaluation (in-process specs, the save
cadence, the live exporter's coalescing, actors waiting for the first export, `actors.local_count`,
evaluation exports reused or redone, a failed export as `eval_failed`); `rnad.fresh_order`, the
policy-lag cap and percentiles; the control protocol; an orchestrator over a fake run through its
socket and its status from files; the crash-loop guard (exit 4, progress resetting the count, the
FAILED status, `start` resetting, `RestartPreventExitStatus`); resume bookkeeping (actor start
indices, results tail, actor crash restarts and back-off against a fake engine, window restore);
`replay.max_decision_tokens`; the attention budgets (cut or checkpointed forwards equal to plain ones
in values, losses and gradients); checkpoint pruning; schedules and evaluators (including
`actors.teacher_tier = "hard+sb"`); `bc.sideboard_weight`; `rnad.reg_eps_sideboarding` at 0;
structured sideboard exploration (the `p_explore` schedule, counts, weights, schedule entries,
snapshot cadence, metrics, status, and the engine reading the schedule when built); SPRT stopping,
the request split, ladder ratings; resume rollback (lost rungs, evaluations and ladder players;
non-fresh episodes); evaluation subjects, failures, collisions and options; draws as findings; the
benchmark (default policies, summary, sideboarding report, `--opponent`, `--decks`, `--focus`);
`add-decks` and live `league.*`; and `run.feature_history` (not live, resume refusal, the engines'
and `rmfeat`'s flags, the start check, `bench`'s default).

The smoke runs a tiny run (3 actors, d 64 / 2 layers, B0 5,000 decisions, BC ended by `phase next`,
one DAgger round of 3,000 decisions, R-NaD with Δm 30 and G 2) detached under systemd, exercises
every command, SIGKILLs the whole unit in B4 (systemd restarts it from the checkpoint), stops and
starts it, and runs it to `done`, with per-iteration SPRT evaluations and one exploiter on the way.
It checks: no evaluation failed, each iteration's evaluations evaluated its fixed point, every
result names its options (argmax and a sampled group), ladder rungs unique across the kill, the
`restore` lines. In local mode also: live exports followed the learner, DAgger played one, every
self-play game's policy versions are exported versions, the learner recomputed the regularisation
log π, the steps report lag percentiles, local evaluations ran. Every smoke checks that `set` and a
resume refuse `run.feature_history`, that the tables match the run's history setting and every
episode left is of their spec. `--feature-history N` runs it with history N, then a short second run
(`<run>-from`: three decks, R-NaD to step 20) started (`run.start_from`) from a checkpoint without
history (`--start-from CKPT`, default a d 64 network over tables exported without history): the
checkpoint accepted, an export of it refused (`spec_hash`), B0 → B3 → B4 with no fatal, its
self-play episodes carrying history tokens. The smoke takes several minutes.
