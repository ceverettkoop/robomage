# Training plan: scripted-agent bootstrap into R-NaD (after step 5)

Goal: train one network that plays from RMSTATE, starting from imitation of the scripted agent
(`scripted:hard`) and moving into R-NaD self-play (Regularised Nash Dynamics, the DeepNash
algorithm), with the whole loop running at high throughput on one workstation-class box.

This plan starts where `docs/engine_fork_plan.md` ends. It assumes step 5 is done: the
`FEATURE_SPEC` document, the C++ featurizer (`src/featurize/`), the whole-vocab card / text /
ability tables, and an `nn:<model>` agent in `src/actor/`. Where step 5 has to leave a door open
for this plan, section 2 says so.

Reference box (the one this was measured on): Ryzen 9 5900X (12 cores / 24 threads), 62 GB RAM,
one Radeon RX 6700 XT (12 GB, RDNA2 gfx1031, ROCm with `HSA_OVERRIDE_GFX_VERSION=10.3.0`). The
training box is a Ryzen 9 7950X (16 cores / 32 threads) with 124 GB RAM and the same GPU; section 5
is sized for it.

**The end product** is one training program, `learn/rmtrain`, run from the command line. It runs
for days without intervention (detached, resuming from its last checkpoint after a crash or a
reboot), walks the phases of section 6 by itself, and can be queried, inspected and steered from
the command line while it runs (section 4, L6).

**Format: bo3 matches, scored per game.** Actors and evaluation play bo3 matches with
sideboarding by default. Every game of a match is scored on its own (+1 / −1, as a bo1 result):
the reward an episode trains on, the SPRT and every reported win rate count games, never match
wins. Sideboarding enters training as the decisions before a game (section 6.1).

---

## 1. Background

### 1.1 What R-NaD needs

R-NaD (Perolat et al., *Mastering the game of Stratego with model-free multiagent reinforcement
learning*, Science 2022, arXiv 2206.15378; supplement Table 2) is model-free self-play with no
search:

- **Reward transform.** Each step's reward for the player to move is shaped toward a
  *regularisation policy*: `r − η·log(π(a|o) / π_reg(a|o))`, with η = 0.2. Over an iteration of
  Δm learner steps, `π_reg` is interpolated from the previous regularisation policy to the current
  one (`α_n = min(1, 2n/Δm)`).
- **Dynamics.** The learner trains toward the fixed point of that shaped game. It uses two-player
  v-trace for the value and Q estimates (c̄ = 1; OpenSpiel's reference passes ρ̄ = ∞) and NeuRD
  for the policy (an update on the logits rather than the probabilities, logit threshold β = 2,
  clip 10,000).
- **Update.** At the end of the iteration, the averaged target network becomes the next
  `π_reg`. Δm in DeepNash was 10k steps for iterations 1–100, 100k up to 165, then 35k.
- **Fine-tuning.** Late in training, the policy is projected through thresholding (drop actions
  below 3%) and discretisation (round to multiples of 1/16 in play), and the value and Q
  estimates are computed under that projection.
- **Other settings:** Adam (b1 = 0, b2 = 0.999, ε = 1e-8), lr 5e-5, target-network averaging
  0.001, 768 full games per batch, games up to 3,600 steps. One network plays both seats on
  player-centric observations. The board deployment (piece setup) has its own head inside the
  same game.

DeepNash was trained from scratch: 7.21M learner steps × 768 games per batch, about 5.5 billion
games, on 768 TPU nodes for learners and 256 for actors. This box will manage roughly 2M games a
day at best (section 6), so a run from scratch is out of reach. **Option A** replaces the
uniform start with the scripted agent:

1. imitate `scripted:hard` (behaviour cloning, then DAgger on the network's own states);
2. initialise both the network and the first regularisation policy `π_0,reg` from that
   imitation policy, smoothed so it gives every legal action some probability;
3. run R-NaD. Each `π_reg` update walks the regularisation target away from the teacher, so its
   pull fades by design rather than being a fixed KL penalty that must be annealed by hand.

### 1.2 Measured throughput of the current engine (release build, one core)

Six bo3 matches (uw_control vs bw_dnt, explore vs explore):

| Path | Decisions/s per core |
|---|---|
| Engine only (`--replay` of the same logs) | ~5,600 |
| + frames encoded and decoded, in-process agents, knowledge `off` | ~2,700 |
| + knowledge sidecar `expert` on both seats | ~1,850 |
| `bench` tier (process start included) | ~1,290 |

The rules engine is about a third of the per-decision cost; the knowledge sidecar is about a
third (32%), and the frame round-trip most of the rest. The scripted agent over RMSTATE runs at
roughly 310–530 decisions/s per process (Python).

### 1.3 Lessons from the archived AlphaZero stack (`cf7f97f`)

The archive's AlphaZero attempt is the reason this plan is model-free:

- About 800 games/hour on 8 actors (128 simulations, 8 worlds). The value head memorised its own
  games (train MSE 0.14 vs 0.67 held out), which flattened the search posteriors. That is a
  shortage of games, not of search.
- The GPU inference server pattern worked: batches of 256–512 over a Unix socket, about 31k
  rows/s on the RX 6700 for the archived network. This plan reuses that pattern.
- One game per process is a hard constraint while `global_coordinator`, `cur_game`, `card_db`
  and the file-level statics are globals. Parallelism comes from many processes.
- Keeping two implementations (Python and C++) of one algorithm bit-identical was a constant
  cost. Here each piece has one implementation: actor-side code in C++, learner-side in PyTorch.

---

## 2. What step 5 must leave open

These belong in step 5's design (`docs/featurizer.md`) so nothing has to be reopened later.

1. **Export-friendly network contract.** Padded tensors and masks only, with no control flow
   that depends on the data. The same model must also export through `torch.export` / AOTInductor
   or ONNX, not only TorchScript, which is in maintenance mode upstream.
2. **Batchable evaluation.** The evaluator interface takes a batch of featurized decisions and
   returns logits and values per row, so the same contract serves an in-process CPU evaluator and
   a remote GPU server (step T2).
3. **The omniscient featurization exists.** The featurizer runs on the omniscient viewer's frame
   with the same `FEATURE_SPEC` (hidden cards revealed), for the learner's critic and auxiliary
   targets. The `nn:` agent must never receive it; the isolation checks cover that.
4. **Featurize from decoded structs.** The featurizer takes the decoded frame the agent feed
   already holds, so no extra encode/decode pass is added per decision.
5. **`bench` breaks down per stage:** engine, frame encode/decode, knowledge sidecar, featurizer,
   evaluation.

---

## 3. Engine work (steps T1–T8)

All of this is C++ in the engine or in `src/actor/`, under the existing rules: no functions in
`main.cpp`, shared queries in `src/queries/`, `-fno-exceptions` outside `src/actor/`, and
`make check` green. None of it changes rules behaviour, so the replay corpus stays byte-identical.

### T1. Trajectory writer (`--trajectory-out DIR`)

For each game an actor plays, write one episode record:

- **Header:** the episode format version, feature-spec version, vocab hash, model version or
  versions of each seat's policy, both decklists, the seeds, the format (bo1 / bo3), and the
  RMLOG path (or the RMLOG embedded).
- **Per decision:**
  - the deciding seat;
  - the featurized tensors of that seat's frame;
  - the legal-action mask;
  - the **behaviour distribution μ over every menu row**: the exact distribution that was
    sampled from, after temperature and any thresholding (v-trace needs the full distribution,
    not only the chosen row's probability);
  - the chosen row;
  - optional: the teacher's label (T4) and the regularisation policies' log-probabilities (T2);
  - the omniscient featurization (section 2 item 3), only when the learner asks for it.
- **Footer:** the result per seat (+1 / −1 / 0), the end reason (win, concession, draw, decision
  cap) and the decision count.

**One episode per game.** In a bo3 match, an episode is one game together with the sideboarding
phase before it: a sideboarding frame's `header.game` is already the upcoming game, so its
decisions belong to that game's episode and take that game's result (section 6.1). Game 1's
episode has no sideboarding decisions. The header records the game number and the match score at
its start.

Decisions the engine auto-takes (a menu with one row: a lone pass, a single legal target, …) are
never asked, so they never appear. The seat sequence is therefore irregular, and every record
carries its seat explicitly.

**Format:** framework-neutral, like the step-5 card tables. A small versioned header, then
little-endian fixed-width records, readable with the standard library; fp16 for scaled features,
integers for ids and indices. Variable lengths (entity count, menu size) are stored with offsets,
not padded on disk.

**Storage policy:** the RMLOG of every game is always kept. It is about 1 KB per game, replays
deterministically, and lets the learner re-featurize old games after a feature-spec change. The
tensor episodes are working data: the learner keeps a window of recent episodes in RAM (section 5)
and old tensor files are deleted.

**Tests** (a new gate tier, e.g. `traj`):
- re-featurizing an episode from its RMLOG reproduces its tensors byte for byte;
- μ sums to 1 over the mask, and the chosen row has μ > 0;
- the seat ids match the RMLOG decisions;
- the omniscient tensors are absent unless they were requested.

Done: `--trajectory-out DIR` (`--trajectory-seats`, `--trajectory-omniscient`; `src/agents/trajectory.h`,
the format `src/featurize/episode.h`, the stdlib reader `python/rmtraj.py`), documented in
[`docs/trajectory_format.md`](trajectory_format.md). One `<log stem>.g<n>.rmtraj` per game, written
atomically; the decision log is always kept (default `DIR/s<seed>.rmlog`) and `--replay LOG
--trajectory-out` re-featurizes it byte for byte (μ absent: the RMLOG carries none), which is also
B0's path (`rm_driver.py --rmlog-dir` keeps the scripted games' logs). Per decision: seat, decision
key, compact tensors (f16 sparse numerics, per-type-width categorical / card / ability slots, about
5 KB per decision against the featurizer's ~100 KB), μ (f32; the random agent's uniform one too),
the reply, the policy version, every `policies=` policy's value and `log π`, a reserved teacher
label; footers with the result per seat and the end reason (win, concession, draw, cap, quit).
The `traj` gate tier checks the list above (and bo3 sideboarding, concessions, the cap, the Python
reader against the C++ one); the `actor` tier checks `nn:` episodes against `dump=` / `evals=`.

### T2. Remote batched evaluator (`nn:remote:<endpoint>`)

The actor-side client of an inference server process:

- sends one featurized decision (or the decisions of several games, see T3) per request over
  shared memory, with a Unix socket for signalling;
- receives logits over menu rows and a value, plus a policy version number;
- falls back to the in-process CPU evaluator when no server is configured (tests, small runs).

**Regularisation policies on the server.** The two regularisation networks (`π_m,reg` and
`π_m−1,reg`) are frozen for a whole R-NaD iteration. The server can evaluate them in the same
batch and return `log π_reg` for every menu row, which T1 writes into the trajectory. The learner
then needs no forward passes for them. This removes about half of the learner's compute.
Trajectories that straddle an iteration boundary are tagged with the iteration and either
recomputed at the learner or dropped.

**The server** is PyTorch (section 4, L5). The engine side owns only the client and the wire
format, documented in `docs/inference_protocol.md`. Both exist since plan step 5d: `nn:remote:`
(`src/actor/remote_evaluator.h`, `policies=` naming the current and regularisation policies, each
response carrying every policy's version and `log π`) and the C++ reference server `rmserve`; what
remains here is T3's several games per request and T1 recording the regularisation policies'
`log π` (`tools/nn/serve_check.py` is a minimal Python server to start L5 from).

**Tests:**
- the remote and in-process evaluators return the same logits for the same model and batch,
  within float tolerance;
- a killed server makes the actor fail loudly; it never falls back to a default action silently.

### T3. Continuous actor mode

Today an engine process plays one game or one bo3 match and exits. An actor needs to play
indefinitely:

- `--actor` mode: loop over games. Each game samples its matchup and seats from a schedule (T5),
  runs the normal per-game `init_ecs` reset, and writes its episode (T1).
- **Weight hot-reload:** the actor (or the server) picks up a new policy version between
  decisions, and each record carries the version it was evaluated with.
- **Seeding:** game *i* of actor *k* uses a seed derived from (run seed, *k*, *i*), so any game
  can be replayed from its RMLOG alone.
- **Caps:** `--max-decisions` per game, with the cap recorded as the end reason. Under self-play
  these are findings to inspect, like the fuzz tier's draws.
- **Optional, later: several games in flight per process.** With the globals, a process can keep
  several games by snapshotting one, restoring another, and alternating (`src/snapshot.cpp`, at
  loop-safe decisions only), so one process can batch requests from several games. Measure the
  snapshot cost first; plain one-game-per-process actors may be enough, given ~20 processes.

Done: `--actor` (`--actor-schedule FILE`, `--actor-out DIR`, `--actor-id k`, `--actor-seed S`,
`--actor-start i`, `--actor-matches N`; `src/agents/actor.h`), documented in
[`docs/actor.md`](actor.md). The loop is over **matches** (a bo3 match is the unit a schedule entry
draws; every game of it is scored and recorded on its own). Match *i* of actor *k* has game seed
`actor_game_seed(S, k, i)` (SplitMix64 rounds over S, k, i; 62 bits), the engine seed the drivers
derive from it (so `--game-seed g` replays the same game: 50 of 50 actor matches had the decisions
of the one-process-per-match runs), its own decision log `a<k>-m<i>.rmlog` (replays alone) and its
episodes `a<k>-m<i>.g<n>.rmtraj`. Between matches the process resets what is match-scoped
(`reset_match_state`, the agents' generator and decoded streams, the in-process feed's
once-per-match frames, the knowledge instances, the trajectory writer, the teacher coin); agents are
reused when a seat's spec repeats (`Agent::begin_match`). The per-game cap is
`--max-game-decisions N` (a recorded setup option, default 5000 under `--actor`): the game stops
with end reason `cap` and the match with it, and the actor goes on. Hot reload: `nn:remote:`
versions come from the server; an in-process `nn:<model>` is reloaded before a match when its file
changed. The status file (`a<k>.status.json`, every 2 s and per match: counts, rates, caps, draws,
the current match, the schedule's version and error, the policy versions seen, the exit reason) and
the results file (`a<k>.results.jsonl`, one line per game: entry, purpose, seats, decks, agents,
teachers, first player, score, results, end reason, decisions, policy versions, log and episode)
are the orchestrator's interface. SIGTERM / SIGINT stop it before the next decision (the log ends
with `QUIT`, the unfinished game writes no episode, exit 0); SIGHUP re-reads the schedule; a fatal
error exits 1 with the reason in the status file. Teachers keep running across matches while a
seat's command is unchanged; `teacher.py` reseeds with its constant `--seed` at each match of its
stream, so its labels in a match are a fresh `teacher.py`'s on that match's stream (the match seed is
not passed to it: no frame carries the seed). Measured (release build, explore vs explore, five
league pairings, bo3, knowledge `expert` on both seats, one core, other work on the box): 50 matches
(115 games, 18,315 decisions) in one actor process at ~2,180 decisions/s (~6 matches/s), against
~1,940 decisions/s for the same matches one process each; ~1,810 decisions/s with trajectories
written. RSS does not grow: 17.6 MB after 10 matches and 18.0–19.2 MB after 150–200 (bo3, no
trajectories), 24–33 MB with no trend from match 10 to match 600 with trajectories written. The gate's `actorloop` tier checks
the loop (replays, a byte-identical rerun, the cap, a taught entry, a schedule change, SIGTERM) and
the `actor` tier the hot reload.

### T4. Teacher channel (labels from the scripted agent)

DAgger needs the teacher's choice at states the *network* reaches. The scripted agent is Python
and reads a seat's RMSTATE stream, so:

- `--teacher-a / --teacher-b <command>`: alongside the in-process agent that decides the seat,
  the engine streams the same seat's frames to a teacher subprocess (the exact `--machine`
  protocol, `docs/state_format.md` §9) and reads back one integer per decision.
- The teacher's reply is a **label**: it is written into the trajectory (T1) and is not played,
  except under the DAgger mixing option `--teacher-mix p`, where the teacher's choice is played
  with probability p (p = 1 is pure teacher play; DAgger decays p over rounds).
- The teacher sees exactly the stream it would see if it played the seat, in order from the
  match's first frame. Its per-game state (mulligan counts, the fruitless-activation stall guard)
  is a deterministic function of that stream, so its labels are well defined even when its advice
  is not followed.
- Python-side: a small `python/teacher.py` wraps `ScriptedCore` as a stdin/stdout labeller. It
  stays stdlib-only, like the rest of `python/`.

**Tests:**
- with `--teacher-mix 1` the game is decision-for-decision identical to `scripted:hard` playing
  that seat through `rm_driver.py`, for the same seed;
- with `--teacher-mix 0` the play is unchanged by the teacher's presence (the RMLOG is identical
  with and without `--teacher-*`).

Done: `--teacher-a / --teacher-b <command>`, `--teacher-mix p`, `--teacher-timeout s`
(`src/agents/teacher.h`) and `python/teacher.py` (`--tier`, default hard; `--seed`; `--record`),
documented in [`docs/trajectory_format.md`](trajectory_format.md) §1.1. The teacher gets the seat's
in-process feed frames, which are the `--machine` stream's bytes (terminal frames included), over a
Unix socket on its stdin/stdout, and thinks while the engine decodes the frame and the agent
decides. Its reply is the step's `teacher` label (−1 the last row; a concession sentinel −1, never
played). Mixing draws one coin per taught in-process decision from `random.Random(seed + 2**32)`
after the agent has decided, so μ = p·onehot(label) + (1 − p)·μ_agent is exactly the distribution
the played reply came from; the decision log records the reply played. A teacher that exits,
hangs past the timeout, replies garbage or exits nonzero is fatal, naming the seat. `--replay LOG
--teacher-a …` relabels a logged game. `teacher.py` reseeds `random` with `--seed` and resets at
each match boundary of its stream (a `SCHEMA` frame after a terminal frame), so one process labels
successive matches. Measured (release, explore on both seats, league bo3): ~1,790 decisions/s
untaught, ~1,400 with one seat taught, ~1,180 with both (teacher start ~30 ms per process
included); `teacher.py` alone labels ~4,800 decisions/s per core. The `traj` gate tier checks the
tests above (bo3 with sideboarding, against `rm_driver.py` and byte for byte against the
`--machine` stream), the labels against `teacher.py` rerun on the recorded streams, an intermediate
mix's coin, μ and replay, and the failure modes.

### T5. Matchup schedule and opponent pool

The actor reads a schedule file (written by the orchestrator, section 4 L6):

- the deck pairs and their weights (league decks first; randomised decks later, section 7);
- the format, bo3 by default (bo1 per schedule entry);
- the seat assignments: self-play (`nn:remote` on both seats, the main R-NaD stream), current
  policy vs a past checkpoint, or vs `scripted:hard` through the teacher channel at p = 1;
- which games are training data and which are evaluation only. R-NaD trains on self-play
  trajectories; games against the scripted agent and past checkpoints are evaluation, or
  imitation data in phases B0–B2.

Done: the schedule file (`src/agents/actor_schedule.h`, [`docs/actor.md`](actor.md) §2): JSON
written whole by the orchestrator and re-read between matches when it changes (a file that does not
parse leaves the last good one in use, its error in the status file). Entries carry a weight, a
purpose (`train` writes episodes, `eval` by default only decision logs; output directories per
purpose or per entry), both decks, the format (bo3 default), both sides' agent specs, teachers,
`teacher_mix` and knowledge levels, the per-game cap and the trajectory options; `swap: decks`
exchanges the decks and `swap: sides` everything of the two sides, on a coin drawn from the match's
game seed. The opponent pool is entries: self-play, a past checkpoint (`nn:<export>` in process or a
second server policy), `scripted:hard` (`first` plus the scripted teacher at `teacher_mix: 1`),
`explore`.

### T6. Policy post-processing in the `nn:` agent

- Sampling temperature, and argmax for evaluation.
- DeepNash's play-time projection: threshold (default 0.03) and discretisation (default 16),
  as agent options (`nn:<model>:thresh=0.03:disc=16`).
- Whatever is applied, the exact distribution sampled from is what T1 records as μ.

Done: `nn:…:thresh=<p>:disc=<n>` (also `nn:remote:`), OpenSpiel's `_threshold` / `_discretize` in
`src/agents/policy_projection.h` (std only, so the learner side can link or mirror it), applied
after temperature and before the draw; `last_behaviour` holds the projected μ, and `evals=` writes
it (`mu`, `chosen`). The `actor` tier checks the projection and a projected bo3 match.

### T7. Throughput work

Set targets in the `bench` tier and work toward them. Measured today: ~1,850 decisions/s per core
with knowledge on.

- Profile the knowledge sidecar (about 32% of the per-decision cost) and the frame encode/decode.
  The agent feed encodes bytes and decodes them again in the same process; the featurizer should
  read the decoded structs (section 2 item 4).
- The engine already skips menus with only a pass; check that the remaining trivial decisions
  (forced single choices in mandatory prompts) are also auto-taken where the rules allow.
- Release build with `-O2 -flto -march=native`. Keep a profile-guided build (`-fprofile-use`) as
  an option once the actor loop is stable.
- **Target:** ≥ 1,500 decisions/s per core for the full actor loop (engine, knowledge,
  featurizer, remote-evaluation round-trip excluded).

Done:
- **One-row menus are taken, not asked** (`InputLogger::choose`; `docs/state_format.md` §9): every
  menu left with one row after the interchangeable-choice collapse, for every consumer alike; no
  frame, no decision number, no decision-log line (RMLOG v3; `--replay` refuses a v2 log), its
  events in the seat's next frame; a seat concedes only at an asked decision. `--play` specs that
  name such a decision are consumed when it is taken; `--trace-decisions` prints it as `| auto:`.
  Removed: 11.4% of the asked decisions under explore (13,809 → 12,237 in 40 league bo3 matches:
  the confirm with no attacker left, single targets, fail-only searches, pay-or-decline with no
  payment, …) and 5.0% under `scripted:hard` (`first` + `teacher.py` at mix 1 on both seats,
  7,853 → 7,459 in 20 matches). The replay corpus and CLI goldens were re-recorded; B0's corpus
  (`runs/bc1/corpus`, RMLOG v2) is regenerated by the pipeline.
- **Per-decision path** (perf over the actor loop; no output byte changes: the same decision logs
  and episode arrays): a system's `mEntities` is a bitset (`src/ecs/entity_set.h`: the same order
  and mutation-during-iteration outcome as the `std::set` it replaces, whose node walks were 11% of
  the loop); the agent feed takes the frame the knowledge sidecar decoded (plus its decoded `k_*`
  tables) instead of decoding the frame again; the sidecar skips the exact solver at a frame with
  no event (its answer depends on the model alone); the frame writer looks tables up by id and
  writes each row in place; the decoder reads blocks in place; `physical_card_of` is memoised.
- **`make pgo`**: the release build with `-fprofile-use` from an instrumented build playing explore
  bo3 games with the featurizer (bin/pgo/robomage), ~10–15% faster again.
- Measured (7950X, one process): the actor loop, 40 explore bo3 matches with knowledge expert and
  `--featurize-seats`, 7.18 s → 5.78 s (2,116 decisions/s; 5.20 s and 2,355 decisions/s with
  `make pgo`), from 1,924 decisions/s before; the `bench` tier's stages and the reference numbers
  are in `docs/ci.md`, whose `bench` tier now warns (never fails) when an optimised build's actor
  loop falls below 1,500 decisions/s. The knowledge sidecar is now the largest stage (~40% of the
  loop: its exact solver and per-event propagation); the engine's continuous-effects rebuild in
  every SBA pass is the next engine-side cost.

### T8. Evaluation tooling

- An **SPRT match runner** for a checkpoint against `scripted:hard` over the league matchups
  (the archive's `gate_sprt.py` design: H1 0.55 vs H0 0.45), run by the orchestrator, not by
  `make check`. It plays bo3 matches and scores games, not matches. Reports give the win rate of
  game 1 (pre-board, a pure bo1 measurement) apart from games 2–3 (post-board); the difference
  shows what sideboarding gains or loses. Game 3 is played only at 1–1, so the games 2–3 figure is
  a selected sample, and the reports say so.
- **Exploiter runs:** periodically train a short best-response policy (PPO on the same pipeline)
  against a frozen checkpoint and report its win rate. Low exploitability is R-NaD's claim, so
  this measures it directly.
- A `traj` gate tier (T1 tests) and an `actor` smoke tier (a short continuous run with a
  random-initialised model and a CPU evaluator, no GPU) join `make check`.

Done:
- **The runner** (`learn/rmlearn/matches.py`, `sprt.py`): `run_sprt(policy, opponent, SPRTConfig,
  out_dir)` → a result dict (`sprt.json`, `summary.txt`), `python -m rmlearn.sprt`. The policy is an
  export (`nn:<path>:<options>` in process, CPU: argmax, sampled, projected) or any `nn:` /
  `nn:remote:` spec; the opponent `scripted[:tier]` (`first` + `teacher.py` at mix 1), `explore`,
  `random`, an export or a spec. N actors (default 8) over the league pairs (configurable, weighted),
  bo3, every pair in **both orders** with `swap: sides` (a side's deck moves with it, so one order
  alone gives the policy one deck of each pair — the B1 evaluation above saw half the decks); the
  results files are read as games finish; Wald's test on the per-game win rate (H1 0.55 / H0 0.45,
  α = β = 0.05, no verdict before 30 decisive games, since the first games in are the short ones),
  draws and caps excluded and listed as findings; SIGTERM to the actors when it decides, `undecided`
  at the game cap. Reports: verdict, llr, games, per-game win rate with a Wilson interval, the
  binomial and the match-clustered standard error (design effect), game 1 apart from games 2–3
  (the selection note), per game number, matchup, own deck, seat, on the play / draw.
- **The ladder** (`ladder.py`): fixed points (an export or spec with its options and version),
  pairings played by the runner (a fixed budget, or SPRT), a Bradley–Terry fit (MM, a weak prior,
  the Elo scale) over every pairing; `Ladder(dir)` (`ladder.json`), `add_fixed_point`,
  `evaluate_against_ladder`, `record_counts`, `ratings`; `python -m rmlearn.ladder add|eval|ratings`.
- **The exploiter** (`exploiter.py`): `run_exploiter(victim, init, ExploiterConfig, out_dir,
  tables)` (standalone: its own `InferenceServer`) and `ExploiterSession` (the same, ticked by a
  host: rmtrain's GPU process). PPO (clipped surrogate on μ recorded by the actors, GAE(λ) over the
  exploiter's decisions with γ = 1, value head, Adam, fp16 autocast) on a copy of the victim (or
  BC); exploiter and victim are two policy ids of one server, each seat `nn:remote:…:policies=<id>`
  (the agent samples from its first id; no engine change needed); only the exploiter's side
  recorded. Phases: baseline games at version 0, training for the budget, then games of the frozen
  final version; reports both records, a win-rate curve per update, PPO's statistics.
- **rmtrain** (`orchestrator/evaluation.py`, docs/rmtrain.md §7): evaluators `sprt` (per R-NaD
  iteration by default: the evaluation stops when the test decides), `exploiter` (inside the GPU
  process, the learner idle meanwhile; off by default, `eval.exploiter_every_minutes` or `rmtrain
  eval --evaluators exploiter`), and the ladder record behind every evaluation (ratings in `status`
  and `evals.jsonl`).
- **The actor smoke run** joins the `actor` tier (docs/ci.md): `robomage_nn --actor`, four bo3
  matches of the reference policy (random-initialised, CPU) on both seats, one taught at mix 0.5,
  the other projected: status / results / footers, μ, labels and versions, replays (~3 s).
- **Found on the way:** `.pt2` exports ran their compiled kernels on every core (inductor's
  `cpp.threads` defaults to the machine's): 12 in-process actors were ~30× slower than with the
  single-threaded export (`export.py`, `make_reference_policy.py` now compile with `cpp.threads` 1;
  `runs/bc1/bc/ckpt/p1.pt2` predates it).
- **Measured** (7950X, other jobs on the box; BC = `runs/bc1/bc`, re-exported single-threaded, argmax):
  SPRT against `scripted:hard` over all 45 league pairs: **H0** after 153 decisive games (73–91,
  44.5% [37.1, 52.2]; game 1 46.4%, games 2–3 43.2%), 164 games in 10 s on 12 actors (~2,100
  decisions/s; the old multi-threaded export: H0 after 203 games, 45.9%, in 302 s); against
  `explore`: **H1** after 30 decisive games (28–3, 90.3%) in 3 s. Ladder, 100 games a pairing (~13 s
  each): BC beat the random-initialised policy 94–6, BC sampled with the projection
  (`temp=1:thresh=0.03:disc=16`) beat it 92–11 and BC argmax 55–46; ratings random-init 0,
  BC +406, BC-projected +418. Exploiter against BC (both sampled at temperature 1, init BC, all league
  pairs, 12 actors, 20 min, ~700 decisions/s in the actors, ~90 PPO updates, ~385k trained
  decisions, ~6,100 games): the baseline (version 0, the symmetric pairing) 48.4% [39.7, 57.1]
  over 122 games, as it should be; the final frozen version **did not rise above 50%**: 36.2%
  [27.6, 45.7] (entropy bonus 0.01, GAE λ 0.95, lr 1e-4), 36.0% (no entropy bonus), 39.2% [30.3,
  48.9] (no bonus, full-game returns, lr 3e-5, now the defaults). In every run the policy's entropy
  rose steadily (0.23 → 0.6–1.0 nats) with no rise in the win rate: from a nearly deterministic BC
  start, ~60 games per update of a ±1 terminal reward is mostly noise, and noise on saturated
  logits raises entropy (the update's first-minibatch KL is ~0: the served and trained networks
  agree). So the tooling works end to end, but a 20-minute PPO exploiter is not yet a measure of
  BC's exploitability; it needs larger rollouts per update, a KL anchor to the initial policy, or a
  much longer budget — to be tuned before it is read as R-NaD's exploitability.

---

## 4. Learner side (new: `learn/`, PyTorch)

The kept Python in `python/` is stdlib-only by rule. The learner needs PyTorch (ROCm build), so it
lives in its own directory with its own requirements file and virtual environment, and nothing in
the engine, the gate or `python/` imports it.

- **L1. Episode reader.** Reads T1 episodes into a RAM replay window. Batches are full games,
  padded to the longest game in the batch and split into fixed-length chunks, as DeepNash does,
  so returns are exact rather than bootstrapped.

  Done: `learn/rmlearn/episodes.py` (`read_episode`: the file's bytes kept, every array a view, plus
  per-step offsets; `expand`: steps of many episodes → the contract's padded inputs in one scatter
  per tensor, equal to `batch.pad`; nothing names a token type, the widths come from each file) and
  `learn/rmlearn/replay.py`: `ReplayWindow` (polls trajectory directories, ingests renamed-in
  `*.rmtraj` oldest first, evicts oldest past `max_decisions` / `max_bytes`, deletes evicted
  `.rmtraj` only with `delete_evicted`, never an `.rmlog`; skips other specs and, whole, episodes with
  a decision over `max_decision_tokens` seat tokens (`replay.max_decision_tokens`); labels per step: the
  teacher's, else the reply at steps without μ unless the seat is `explore` / `random`; held-out
  split by a hash of the game), `DecisionBatch` (uniform over labelled decisions; inputs, label,
  outcome, aux targets, optional omniscient inputs), `TrajectoryBatch` (G games time-major
  `[T, G]`, T rounded up to `chunk_len`; `valid`, `seat`, `chosen`, `has_mu`, `mu`, `legal`,
  `teacher`, `policy_version`, each requested policy's recorded value / `log π`, `reward` with each
  seat's result on its game's last step; network inputs per chunk), `Prefetcher` (background
  threads). Measured on the B0 corpus (spec version 2, one core): `expand` ~42–48k decisions/s,
  decision batches with aux targets ~33–36k/s, trajectory chunks ~36k/s; the window holds ~13 KB
  per decision (seat 4.5 KB, omniscient part and targets 8.5 KB). The loader is far ahead of the
  learner (below). `learn/check.py` checks it (against `rmtraj.py`, the window, both batch kinds).
- **L2. Model.** The step-5 contract: an entity transformer with relation biases, a pointer
  action head over menu rows, a value head, the card / text / ability tables as frozen buffers,
  and an id embedding with id-dropout. Auxiliary heads predict hidden information (the
  opponent's hand, the libraries' tops) from omniscient targets. Export to the actor's format at every
  checkpoint.
- **L3. Behaviour cloning.**
  - Cross-entropy to the teacher's label, with label smoothing.
  - Value regression to the game outcome, optionally through an omniscient critic (below).

  Done: `learn/rmlearn/bc.py`: `BCTrainer(model, window, BCConfig)` with `train(n)`, `evaluate()`,
  `save` / `restore`, `export` (metric dicts for `metrics.jsonl`), and `python -m rmlearn.bc`.
  Label-smoothed (0.05) cross-entropy over the menu rows; the value head (tanh) regressed by MSE to
  the deciding seat's game result (kept scalar so the exported value is what the actor and R-NaD
  read); the aux heads weighted 0.05 on omniscient steps; Adam, warm-up then cosine decay, clipping
  at 1, fp16 autocast with a gradient scaler; batches drawn four at a time and cut by token count.
  Evaluation on held-out games: agreement overall, per group (in-game, mulligan, sideboarding) and
  per action category of the label (the menu token's `category` slot, from the spec), value MSE and
  sign, aux accuracies. `learn/rmlearn/evaluate.py` plays an export (`nn:<model>`, CPU, in the
  actors) against `scripted` / `explore` / `random`, scored per game, game 1 apart from games 2–3.
- **L4. R-NaD.** A PyTorch port of OpenSpiel's `rnad.py` (JAX reference implementation):
  - the reward transform with the server-supplied `log π_reg` (T2);
  - two-player v-trace over the irregular seat sequence;
  - NeuRD with logit thresholding;
  - target-network averaging;
  - the Δm schedule and the fine-tuning projection.

  Done: `learn/rmlearn/rnad.py` (the math, a port of OpenSpiel's `rnad.py` at
  [`8156646`](https://github.com/google-deepmind/open_spiel/blob/8156646cdef0c753162fc6b68a11d57cce642f95/open_spiel/python/algorithms/rnad/rnad.py):
  `v_trace`, `_has_played`, `_policy_ratio`, `_player_others`, `get_loss_v`, `get_loss_nerd`,
  `legal_policy` / `legal_log_policy`, `EntropySchedule`, `FineTuning`) and
  `learn/rmlearn/rnad_learner.py` (`RNaDConfig`, `RNaDLearner`, `ServerLink`, `trainable_episodes`)
  over L1's `replay.TrajectoryBatch`. The learner holds the network, the target network (averaged
  at 0.001) and the two regularisation networks (π_m,reg, π_m−1,reg, frozen, each with a version and
  a smoothing ε), Adam (b1 0, b2 0.999, ε 1e-8, lr 5e-5, each update clipped to ±10,000 as the
  reference's optax chain), the learner step and the iteration index. `step(batch)` scans the
  batch's chunks last to first: the network's forward with gradient on the chunk's valid steps
  (rows grouped by token count, each group padded only to its own largest decision), the target
  network's value, the reward transform `log π − (α·log π_m,reg + (1 − α)·log π_m−1,reg)`, per
  player v-trace with its carry passed back to the earlier chunk, the critic loss and NeuRD
  (β 2, clip 10,000) normalised over the whole batch, backward; then the update, the target
  average and, at a boundary of the Δm schedule (default 10k steps, a list of sizes and repeats),
  π_m−1,reg ← π_m,reg ← the target network. The regularisation log π comes from the trajectory's
  evaluations when the recorded version is the learner's (matched by version, so a step recorded
  before a boundary still serves as π_m−1,reg), else it is recomputed from the frozen network
  (`stale_reg="recompute"`) or the game dropped (`"drop"`); both are counted. Seeding (B3,
  `from_bc_checkpoint`): network = target = π_0,reg = the BC policy; π_0,reg's smoothing
  (ε 0.1, 0.25 at sideboarding steps) is applied **learner-side** to its raw log π (the server
  serves raw networks; the step's `sideboarding` flag picks ε). `ServerLink` pushes the network
  (id 0, version = learner step) on attach and every K steps, the regularisation networks (ids 1,
  2) on attach and at each boundary. `state_dict()` / `from_state_dict()` hold everything (networks,
  Adam, counters, loss scale, the torch generators) for an exact resume; `configure()` changes
  settings mid-run. The omniscient critic is `ModelConfig.omniscient_critic` (its value replaces
  the seat value head's in v-trace; the head is still regressed to the same targets). Steps without
  μ are refused (`trainable_episodes` raises, or drops the game); capped and quit games are dropped.
  B5's projection is the engine's (`src/agents/policy_projection.cpp`, bit for bit). Differences
  from the reference (each listed in the modules' docstrings): menus padded with a legal mask and
  a dummy row at padded steps; the chunked scan; regularisation log π from the trajectory; the
  projection's 1e-9 rounding slack; the model's tanh value head (`value_activation="linear"` reads
  it before the tanh); Adam ε 1e-8 (the reference writes 10e-8); discretisation 16 by default (the
  reference's 32). Checks (`learn/check_rnad.py`, CPU): every function against a numpy
  transcription of the reference on random batches with irregular seats, padding and arbitrary
  legal subsets (≤ 1e-14), NeuRD's gradient against finite differences, the learner's whole loss
  through a real `RMNet` against the reference loss, chunk-length invariance, the projection against
  the C++ (3,010 distributions), biased rock-paper-scissors from a skewed start to its equilibrium
  (1, 10, 5)/16 within 0.02 (η 0 keeps oscillating) and a dominant action learnt, the boundary and
  push bookkeeping, a resumed run bit-identical to an uninterrupted one, real episodes and a BC
  checkpoint seeding. Measured (`learn/bench_rnad.py`, RX 6700 XT, d 192, 4 layers, random-agent
  league bo3 episodes, 227 decisions per game and 166 tokens per decision on average, the
  regularisation log π recorded): fp32 G 32 × chunk 16 ~600 decisions/s (8.4 GB peak); fp16 G 32 ×
  chunk 32 ~930 (9.7 GB), G 64 × chunk 16 ~960 (10.0 GB); G 64 × chunk 32 does not fit. The step
  is GPU-bound in the model's forward and backward (about 1.33 of BC's step per decision: one more
  forward for the target), so the learner, not the actors, binds at ~1k decisions/s (about 15k
  games/hour) until the model is made cheaper.
- **The omniscient critic is an experiment, not DeepNash's design.** DeepNash's critic saw only
  the player's observation. A critic that sees the omniscient featurization reduces value variance
  in a hidden-information game, but in R-NaD the critic also produces the Q estimates the policy
  is updated from, so test it against the observation-only critic before committing to it.
- **L5. Inference server.** GPU, ROCm. It batches requests from all actors (target batch
  256–512, with a micro-timeout), evaluates the current policy and both regularisation policies,
  and versions the weights it serves. It speaks RMIP (`docs/inference_protocol.md`) and runs
  **inside the learner's process**: one process owns the GPU, and new weights reach the server in
  memory. (Several processes on this GPU at once have reset it before.)
- **L6. Orchestrator: `learn/rmtrain`.** Starts and watches N actors and the GPU process
  (learner + server). It writes the schedule (T5), handles checkpoints and the R-NaD iteration
  boundary (freezing `π_m,reg`, pushing it to the server), and runs evaluation (T8) on a fixed
  cadence. Its command line:
  - `rmtrain start <run> [--config FILE]` detaches the run (under `systemd-run --user` with a
    `MemoryMax`) and returns; `rmtrain start` on an existing run resumes it from its last
    checkpoint, which is also what happens after a crash or a reboot.
  - `rmtrain status`, `tail`, `pause`, `resume`, `stop`, `set <param> <value>` (the
    hyperparameters that may change mid-run: learning rate, η, Δm, ε, actor count, evaluation
    cadence), `eval` (run an evaluation now), `phase hold|next` (hold or force the next phase
    transition). These talk to the running orchestrator over a Unix control socket in the run
    directory.
  - The run directory holds everything durable: the config as resolved, `metrics.jsonl`, the
    checkpoints and exported models, the evaluation results, the RMLOGs and per-component logs
    in plain files that can be followed with `tail -f`. `status` and the metrics are readable
    from the files alone when the run is not running.
  - Actors that crash are restarted, and their last game is written off; the GPU process
    crashing stops the run, which resumes from the last checkpoint.

  Done (as built): `learn/rmtrain` (`learn/rmlearn/rmtrain.py`, `learn/rmlearn/orchestrator/`),
  documented in [`docs/rmtrain.md`](rmtrain.md). A TOML / JSON config over defaults for every key
  (`rmtrain defaults`), `--set` overrides, the resolved `config.json` in the run directory; `start`
  detaches under `systemd-run --user` (`MemoryMax`, `KillMode=mixed`, `Restart=on-failure`, so a
  crashed or killed run resumes by itself; a double-fork daemon with a warning without systemd) and
  returns once the control socket answers. Processes: the orchestrator (no GPU), one GPU process
  (the learner — BC or R-NaD — and the RMIP server, holding the flock `run.gpu_lock`), N actors
  (`robomage_nn --actor`, GPU hidden; teachers started by them), evaluation actors and a CPU-only
  exporter. The phases run by themselves: B0 (teacher corpus through the actors, or a reused
  corpus), B1 (BC until held-out agreement plateaus or a step cap, then an evaluation), B2 (DAgger
  rounds: the BC policy through `nn:remote` with the teacher labelling both seats at a decaying
  mix, BC continuing on the growing window; exit when the win rate against scripted:hard stops
  improving or at the round cap), B3 (`from_bc_checkpoint`), B4 (self-play on policies 0, 1, 2,
  fixed points saved and exported at each boundary, an evaluation of each iteration's fixed point
  (argmax) against the fixed opponents and a ladder of past fixed points), B5 (the learner's projection, evaluation with
  `thresh=0.03:disc=16`); `phase hold|release|next`. Evaluation is a registry of evaluators (the
  extension point for T8). Checkpoints on a cadence and at phase ends, `state.json` written
  atomically, `metrics.jsonl` / `evals.jsonl`; a resume re-reads the replay window's checkpointed
  episodes (only the untaken ones fresh; games of the lost steps left out) and restarts actors after
  their last match. Since the option-2 benchmark (§5) the actors and evaluations play the policy **in
process** by default (`actors.policy_eval` / `eval.policy_eval = "local"`, 16 actors): the GPU process
saves the current network every ~30 s, a CPU live exporter writes the actors' TorchScript export
(replaced atomically, reloaded between matches), and the learner recomputes both regularisation
policies' log π; `remote` keeps the server path. `run.start_from` starts a run at B3 from a BC
checkpoint. `learn/check_rmtrain.py`: CPU checks and a
  `--smoke` tiny end-to-end run (B0 → done in ~6.5 min including a `kill -9` in B4 and a stop/start).
- **Environment.** `learn/` lives in this repository with `learn/requirements.txt` (torch
  2.10.0+rocm7.0, the build that runs on this GPU) and its own `learn/.venv`.

---

## 5. Running it on this box

**Process layout:**

| Component | Resources |
|---|---|
| Actors | ~26 engine processes in `--actor` mode, one per hardware thread, leaving about 6 threads for the rest (the count is a run setting) |
| Learner + inference server | 1 process, the only one on the GPU; 1–2 threads for data |
| Teacher labellers (phases B1–B2 only) | Python scripted agents, which displace actors while running |

**Sharing the GPU.** The server and the learner share one 12 GB card. Keep the model small (about
3–10M parameters, d ≈ 192–256, 4 layers): that size is enough to start with, and the VRAM holds
the current, target and two regularisation networks plus activations. Prefer large server
batches; the archive measured about 3 ms per forward at batch 1 and about 8 ms at batch 256.
Attention is quadratic in the padded token count: the replay window leaves out games with
decisions over 640 tokens, the server and the learner bound each forward's attention cells, and an
R-NaD chunk past its budget is checkpointed (the measured peaks: `docs/rmtrain.md` §5).

**Replay in RAM.** Store tensors in fp16 with variable lengths and no padding, at roughly
10 KB per decision. A window of up to ~4M decisions (~40 GB) fits in 124 GB with room for the
processes.

**Rough capacity** (to confirm with T7 and L5 measurements):

| Stage | Capacity |
|---|---|
| CPU, all actors | ~25–30k decisions/s at the T7 target |
| GPU inference, small model | well above that |
| Learner | the GPU is the binding limit; a sustained ~5–10k decisions/s |
| Result | about 25–50 games/s, i.e. **100–180k games/hour**, about 2–4M games/day |

**Measured: where the policy is evaluated** (the option-2 benchmark, after the first validation run
`runs/val1`; `docs/rmtrain.md` §5.1). In val1's B4 the actors evaluated through the GPU process's
inference server (`nn:remote`), which shares the GPU with the learner: under the learner's load it
served ~500 rows/s at batch ~13, the 26 actors waited on it (0.44 cores busy in all) and the learner
waited for games 41 % of the time. Evaluating in process on the actors' cores (`actors.policy_eval =
"local"`: TorchScript exports of the current policy every ~30 s, the learner recomputing the
regularisation log π) on val1's two matchups and BC policy (d 192 / 4 layers, 3.97M parameters),
B4, Δm 100, each mode on its own:

| | remote, 26 actors | local, 26 actors | local, 16 actors | local, 16 actors, d 128 |
|---|---|---|---|---|
| self-play decisions/s (games/hour) | 500 (15.9k) | 754 (23.7k) | 1,045 (33.7k) | 2,061 (65.7k) |
| learner steps/min (G 32, ~3.6k decisions) | 8.2 | 12.0 | 14.9 | 19.5 |
| learner decisions/s: wall; while stepping | 496; 839 | 730; 772 | 879; 907 | 1,180; 1,217 |
| learner waiting for games | 41 % | 5 % | 3 % | 3 % |
| GPU use (mean), VRAM (max) | 93 %, 12.0 GB | 78 %, 11.8 GB | 89 %, 11.9 GB | 89 %, 11.8 GB |
| actors' CPU | 0.44 cores | 25.3 cores | 15.7 cores | 15.6 cores |
| policy lag p50 / p95 (learner steps) | 28 / 38 | 6 / 10 | 6 / 10 | 7 / 11 |

- **The actors saturate the cores, not the threads.** 16 physical cores (32 threads): self-play
  decisions/s of N local actors alone (no learner, d 192): 1 → 170, 8 → 923, 12 → 1,310,
  16 → 1,342, 20 → 1,169, 26 → 949, 32 → 823 (SMT siblings and the two model copies per actor,
  one per seat, contend for the cores and caches; in the run the 26 actors also starved the GPU
  process's CPU-side batch building: GPU use p10 14 %). Hence `actors.local_count` 16.
- **The CPU forward** (one thread, idle box): d 192 6.7 ms per decision (TorchScript; `.pt2` 6.2 ms
  but ~16 s to compile), d 128 3.9 ms; the engine's own decision is ~0.5 ms.
- **The learner's recompute** (`learn/bench_rnad.py`, G 32 × chunk 32, fp16, alone on the GPU):
  1,366 decisions/s with the regularisation log π recorded, 1,112 recomputing one network (π_0,reg
  in both slots, before the first boundary), 935 recomputing two (−32 %). Local mode still wins
  because the learner no longer shares the GPU with the server and no longer waits for games.
- **DAgger (B2)** the same way: 1,050 decisions/s with 16 local actors (the round's BC policy
  exported once), 751 through the server with 26 (sharing the GPU with BC's training).
- **The binding limit is now the learner** (~0.9k decisions/s at d 192, ~1.2k at d 128, ≈ 28k /
  37k games/hour of ~112 decisions); 16 local actors play ~1.2x (d 192) and ~1.7x (d 128) what it
  takes, so `rnad.fresh_order = "newest"` keeps the lag at the export cadence and the surplus games
  wait in the window (`rnad.max_policy_lag` drops the stale ones when set). The plan's rough figures
  above (5–10k decisions/s learner) assumed a cheaper model or step; at these sizes the learner, not
  the actors, sets the pace.

---

## 6. The training cycle (phases)

### B0. Teacher corpus

`scripted:hard` vs `scripted:hard` over every league matchup, bo3, through `rm_driver.py` with
RMLOGs kept. Add `scripted:hard` vs `explore` games for coverage of bad positions.

At ~400 decisions/s per Python process × 20, about 1M decisions takes around 2–3 minutes; aim for
5–10M. The tensors are produced by re-featurizing the RMLOGs (T1's replay path) in parallel.

Done (tooling): `learn/rmlearn/corpus.py`. The actor route (`--actor`, both seats `first` with the
scripted teacher at mix 1, writing labelled episodes directly) beat `rm_driver.py` + `--replay`
(~6.7k against ~3.7k decisions/s on 8 processes) and is the default; `--refeaturize RMLOG_DIR`
replays the kept logs into new episodes after a spec change. A first corpus: 2,200 bo3 matches,
5,216 games (80% scripted vs scripted over the 45 league pairs, 20% scripted vs explore with only
the scripted seat recorded), 792,904 decisions in 72 s on 16 actors (~11k decisions/s), no caps or
draws; re-featurized at spec version 2 from its logs in 39 s on 24 processes (~20k decisions/s),
~13 KB per decision with the omniscient part.

### B1. Behaviour cloning

Train L3 on B0.
- **Exit:** top-1 agreement with the teacher on held-out games, and an SPRT result for the BC
  policy against `scripted:hard` (expected to be below 50%; it is a starting point, not a goal).

A first run (tooling done, the run itself a sanity check): the B0 corpus above at spec version 2,
5% of games held out, the default model (d 192, 4 layers, ~4M parameters), batch 128, fp16, 26 min
on the RX 6700 XT at ~1,200 decisions/s (14,500 steps, ~2.3 passes; the loader never waited).
Held-out agreement with the teacher 97.6% (in-game 97.7%, mulligan 95.0%, sideboarding 100%; the
weakest categories `NAME_CARD` 52%, `DIG_CHOICE` 72%, `KEEP_LEGEND` 72%, `SELECT_BLOCKER` 76%);
the value memorises (MSE 0.14 train against 0.72 held out, sign 78.5%), so BC's value is a weak
start. Played argmax as `nn:` in the actors over four league pairs (both seats, but only the first
deck of each pair: `swap: sides` moves a side's deck with it, see T8), per game: 24% against
`scripted:hard` (game 1 26%, games 2–3 23%), 69% against `explore`, 76% against `random`; over
every league pair with both decks (T8's SPRT) it is ~45% against `scripted:hard`. The losses to `random` show the compounding error DAgger (B2) is for: e.g. bug dredging
Life from the Loam at every draw instead of drawing, a state the teacher's games rarely reach.

### B2. DAgger, 3–5 rounds

The network plays with sampling (T6) while the teacher labels every decision (T4), with
`--teacher-mix` decaying from about 0.5 to 0. Aggregate the labels with B0 and retrain.
- **Exit:** the BC policy's win rate against `scripted:hard` stops improving between rounds.

### B3. Seeding R-NaD

- The network = the final DAgger policy.
- `π_0,reg` = that policy **smoothed**: mixed with uniform-over-legal, ε ≈ 0.1. A nearly
  deterministic teacher gives untaken actions almost zero probability, and `log(π/π_reg)` would
  then forbid them outright.
- Use larger ε for decks the scripted agent was not built for.
- Use a larger ε at sideboarding decisions (about 0.2–0.3): the teacher never sideboards, so its
  prior there is Done with certainty (section 6.1).
- The value head starts from the BC value.

### B4. R-NaD

Self-play on the league matchups, bo3, with DeepNash's settings (η = 0.2, lr 5e-5, the v-trace
and NeuRD parameters). Use short Δm early (about 10k learner steps), so the regularisation target
leaves the teacher quickly, and lengthen it once evaluation shows steady gains.

**Evaluate every iteration:**
- an SPRT against `scripted:hard`;
- against the previous iterations' fixed points (a ladder);
- exploiter win rate (T8), on a slower cadence.

### B5. Fine-tuning

Turn on the thresholding and discretisation projection in the learner. Play with the same
projection at evaluation time (T6).

### B6. Extensions, each with its own plan

- **Deck building:** sideboarding (section 6.1) is the first step toward it.
- **Deck randomisation:** colour-coherent, curve-sane generated decks from the vocab, for
  breadth.
- **Held-out-card evaluation:** train without one card's decks and test on them.
- **New cards:** engine work, gated by the fuzz tier as now.

### 6.1 Sideboarding

RMSTATE already carries everything a sideboarding decision needs, and the featurizer already
tokenises it, so sideboarding needs no format change and no head of its own:

- between games the sideboarding seat gets `STATE` frames with `header.sideboarding = 1`, whose
  `game` is the upcoming game;
- the `match` token (game number, wins, who plays first next, `sideboard_phase`, `sb_swaps`,
  `sb_drift`) and the ledger tokens (`docs/featurizer.md` §3.6: per card of the seat's current 75
  its `own_deck` main and side counts);
  the opponent's sideboard is a count and its post-board configuration stays hidden;
- the menu's `SIDEBOARD_IN` / `SIDEBOARD_OUT` / `SIDEBOARD_DONE` rows, each with `source_card`
  (the card's vocab id) and `ordinal` (copies). The engine offers a swap in-first (a card in,
  then the cut that balances it), withholds Done while the deck is off-size, and caps a phase at
  15 swaps;
- the knowledge sidecar's memory of the opponent across games (revealed cards, `k_seen`).

The pointer head picks sideboarding rows like any other menu rows. A sideboarding decision is part
of the next game's episode (T1) and is credited with that game's result. A sideboarding row moves
one copy, so a phase of k swaps is 2k decisions (each in, then its cut) and a Done.

**Measured in league12h (2026-10-20), with the teacher answering Done and `reg_eps_sideboarding`
0.25:** under argmax no policy ever swapped (0 of ~690 phases per policy in the end-of-run
benchmark); in self-play the policy's probability off Done at sideboarding decisions climbed with
every fixed point (0.05 BC → 0.10 m1 → 0.27 final, still rising), its swaps were close to random
(doomsday boarding in Plains for Brainstorm), and a swapping seat won its next game 2.4 points less
often than a non-swapping one in the same matchup. The smoothing in `π_0,reg` pulled the policy
itself toward random swaps instead of exploring them. Section 6.2 replaces it.

### 6.2 Sideboarding: a heuristic warm start and structured exploration

The goal is a policy that sideboards from what it sees (its own 75, the opponent's registered list
under open decklists, what the opponent showed in earlier games), in a way that carries to
matchups and cards it was not taught. Two parts:

**1. A teacher that sideboards (the warm start).** A scripted tier with basic sideboarding
heuristics, `scripted:hard+sb`, labels B0–B2; `scripted:hard` stays unchanged as an evaluation
opponent, so results stay comparable across runs. The heuristics may be hard-coded per matchup
(which cards each league deck brings in and cuts against each opponent); they are a prior, not the
target. Behaviour cloning learns them; R-NaD starts from them (`π_0,reg` is that policy, with
`reg_eps_sideboarding` about 0: exploration moves to the actors, part 2).

**2. Structured exploration in the actors.** In self-play (B4–B5), at each sideboarding phase a
seat explores with probability `p_explore` (a run setting, scheduled from high to low over the
run, live). An exploring phase follows a proposal distribution `q` instead of the policy for the
whole phase:

- **How many swaps:** a target count k from 0 to 10 (0 is the control: Done at once), drawn from a
  configurable distribution over 0–10 (default uniform). It is realised as a stopping hazard over
  the phase's swap count, the match token's `sb_swaps`: at a balanced deck with s swaps made,
  `q(Done) = P(k = s) / P(k ≥ s)`. So `q` at every decision is a function of what the seat sees,
  and the distribution of k is the configured one.
- **Which card in:** among the offered `SIDEBOARD_IN` rows, by per-card weights; **which cut:**
  among the offered `SIDEBOARD_OUT` rows, by per-card weights. No masking: every card the engine
  offers stays possible, the engine's balance rules and its 15-swap cap apply as for any agent.
- **Coverage:** the orchestrator keeps per matchup (own deck, opponent deck) counts of explored
  (card in) and (card out) and writes weights favouring the least explored into each schedule
  entry, so over a run every sideboard card is tried in and every maindeck card tried out in every
  matchup, at every swap count. Uniform weights are the fallback.

The actor records `q` of the chosen row as μ at every step of an exploring phase (the coin that
chose exploration is taken per phase; each branch is a valid behaviour policy, so the importance
ratios π/μ of v-trace and NeuRD stay unbiased). The learner therefore sees outcome-credited
evidence for swaps the policy would not make, without being pulled toward them: the policy changes
only when the outcomes favour a swap. Exploration is training-only: evaluations play the policy.

**What the learner can resolve, and when.** A day is about 750k sideboarding phases (both seats),
some 8k per deck ordering; at `p_explore` 0.1 with ten candidate swaps per matchup, a single
(matchup, swap) cell sees about 80 tries a day, and a 5-point effect needs about 1,500. Per cell,
outcome-only learning takes weeks; the policy shares what it learns across matchups through the
card tokens (a graveyard-hate card against any graveyard deck), and that sharing is what
generalisation means here. A game's result is mostly unpredictable at its start (section 7, open
issue 10): that, not the value head, sets the noise of every sideboarding decision's credit.

**Measurements** (`rmtrain bench` and the run's status):

- per policy: the swap rate, the swap-count distribution, cards in and out per deck, and the next
  game's win rate after a swap vs Done, per matchup (the paired benchmark against `scripted:hard`
  and `scripted:hard+sb`);
- in self-play: the same over the explored phases per (matchup, card), the evidence the learner sees;
- **generalisation:** hold some matchups' heuristics out of the teacher (it answers Done there);
  the policy's sideboarding in those matchups after R-NaD, against the held-in ones, shows whether
  it learned more than the hard-coded table.

**Work.** Done: the `scripted:hard+sb` tier (`python/scripted_core.py`,
`python/sideboard_guides.{py,json}`; card scripts untouched) and the warm start's wiring in
`rmtrain`: `actors.teacher_tier = "hard+sb"` makes it B0's scripted players (both seats) and
DAgger's labeller while every evaluation opponent stays `scripted:hard`, and `bc.sideboard_weight`
weights the sideboarding decisions in BC's policy loss (`docs/rmtrain.md` §3). Done too: the
actor's exploring phase and its μ (the schedule entry's `sb_explore`: `p`, `count_weights`, per-deck
`in` / `out` weights, `default_weight`; `docs/actor.md` §2; `src/agents/sb_explore.{h,cpp}` and the
driver's decision hook: the coin per seat and phase from the agents' generator, the `nn:`
evaluation kept at every exploring step, μ = `q`), each results line's `sb` (per seat: explored,
swaps, cards in and out; `docs/actor.md` §4), and the gate (`actorloop`: μ = `q` recomputed from
replayed frames, the counts cover 0–10 in proportion, the card weights move the frequencies, p 0
changes nothing, replays and reruns identical, the schedule's refusals; `actor`: the same on `nn:`
self-play at p 0.5, the other phases' μ the policy's). Done too: the orchestrator's side
(`learn/rmlearn/orchestrator/sb_explore.py`; `docs/rmtrain.md` §6.1): `p_explore`'s schedule
(`sb_explore.p_start` held for `hold_frac` of `b4.max_learner_steps`, then linear to `p_end`; 0 in
B5 and every evaluation; live, default 0), the coverage counts per (own deck, opponent deck, card)
in and out from the training actors' results lines (kept in `state.json` with the results offsets),
the weights `(1 + n)^-coverage_power` written into each self-play entry's `sb_explore` (rewritten every
`sb_explore.rewrite_every_s` and when p moves), and the status line and `sb_explore` metrics (explored
phases per matchup, the share of (matchup, card) cells tried, the explored and policy swap-count
histograms, the policy's swap rate). Done too: the benchmark's sideboarding report (`rmtrain bench`,
`learn/rmlearn/sb_report.py`; `docs/rmtrain.md` §7.1): per policy and per deck the swap rate, the
swap-count histogram, the cards most moved in and cut, the next game's win rate after swaps vs Done,
and agreement with the guides' targets (the maindeck tracked through the match, play / draw per
game) held-in vs held-out; `--opponent scripted:hard+sb` plays the same benchmark against the
sideboarding tier (its own directory), whose phases are reported beside the policy's. Left:
`reg_eps_sideboarding` about 0 (a run setting; the default stays 0.25).

---

## 7. Decisions and open issues

Decided (2026-10-19):

1. **Where the learner lives:** `learn/` inside this repository, with its own requirements and
   virtual environment.
2. **Storage:** actors write the tensor episodes and every RMLOG is kept (T1's storage policy).
3. **GPU layout:** one process owns the GPU, the learner with the inference server inside it.
4. **Several games per process (T3, optional):** not built in this pass.
5. **Format:** bo3 matches by default, scored per game (section 6.1, T8).

Open:

1. **Omniscient critic:** adopt it after the A/B test in L4, or keep the observation-only critic.
2. **Teacher smoothing ε and the first Δm.** Tune them on a small run (two matchups) before the
   full league.
3. **Sideboarding** (decided 2026-10-20, section 6.2): a teacher tier with basic sideboarding
   heuristics (per matchup allowed) for the warm start, and structured exploration in the actors
   (0–10 swaps, coverage-weighted cards, no masking) instead of smoothing in `π_0,reg`. The
   archive's dead-card table (`sb_dead_rules.json` on `archive/rl-stack`) is not used as a mask;
   it can inform the heuristics.
4. **Memory / history.** The network sees the current state only. Candidate: recent-event tokens
   over a short window (this turn and the previous one), without the fields that fingerprint a game
   (handles, sequence numbers, exact turn counts), with dropout on them. Behaviour cloning's
   multi-pass training memorises values (BC1 measured value MSE 0.14 on training data vs 0.72
   held out, with no history at all), so keep BC's value-loss weight low or freeze the value head
   during BC. Run it as an A/B judged by the R-NaD learner's first-sight vs after-update value MSE
   gap (`value_mse_gap`), held-out value MSE and per-game win rate vs `scripted:hard`. A recurrent
   core is the heavier alternative.

   **Stage 1 (hist1, 2026-10-19): negative; stage 2 (a paired R-NaD run) not warranted.** Built
   as an opt-in featurizer option (`--feature-history N`, `rmfeat --history N`;
   `docs/featurizer.md` §3.7; off by default and then byte-identical): the seat's events of this
   turn and the previous one, newest first, at most 32, each with up to 8 card tokens, game
   fingerprints dropped, handles turned into relations to the objects holding them now; BC's
   `event_dropout`. Offline test: val1's B0 scripted:hard games (ur_delver / burn / uw_control /
   wrb_energy) replayed both ways from the same decision logs, 4,878 games, 562k decisions, 10%
   of the matches held out (501 games, 57.7k decisions); BC at the defaults (d192 / L4, value
   weight 0.5), 6,000 steps (~1.5 passes), three seeds per arm. Mean ± sd over seeds at step 6000
   (paired differences against off):

   | | off | history | history + dropout 0.25 |
   |---|---|---|---|
   | held-out agreement | 96.66 ± 0.16% | 95.92 ± 0.10% (−0.73 ± 0.23 pp, every seed) | 96.06 ± 0.26% (−0.59 ± 0.10 pp) |
   | held-out value MSE (mean of steps 4500–6000) | 0.390 ± 0.008 | 0.393 ± 0.018 (+0.004 ± 0.024) | 0.388 ± 0.007 (−0.001 ± 0.006) |
   | train − held-out value gap (same window) | 0.120 ± 0.022 | 0.115 ± 0.001 | 0.114 ± 0.008 |
   | held-out opponent-hand aux accuracy | 0.491 ± 0.022 | 0.489 ± 0.005 | 0.484 ± 0.016 |
   | win rate vs scripted:hard (argmax, ~425 games per arm) | 0.447 ± 0.024 | 0.447 ± 0.024 | 0.460 ± 0.024 |

   History cost agreement on every seed (PLAY_LAND 99.0 → 96.0%, TOP_LIBRARY 89.4 → 86.4%,
   KEEP_HAND 98.9 → 94.8%; dropout recovered little), learned the value faster early (held-out
   MSE 0.395 vs 0.407 at step 2000) but not better by the end, and changed neither the value gap,
   the aux heads nor play. Memorisation is real but not history's doing: one seed of each run on
   to 18,000 steps (~4.5 passes) opened the train / held-out value gap to 0.33 without history and
   0.34 with it (held-out MSE 0.466 vs 0.471, best 0.368 at 4,000 steps vs 0.366 at 2,000), and
   history's agreement deficit closed (98.1% vs 97.9%). So BC's value memorisation comes from
   multi-pass training on few games (keep BC's value weight low or stop by held-out value MSE),
   and the recent-event tokens add cost (+~30 tokens, +25% time per step) without a gain the
   knowledge sidecar's tables do not already give. Kept in the code, off. Outputs:
   `runs/hist1-*` (data, BC runs, evaluations, `runs/hist1-logs/analysis.md`).
5. **Engine-computed tactical features.** BC's weakest categories are the computation-heavy ones
   (SELECT_BLOCKER 76% agreement, DIG_CHOICE 72%, NAME_CARD 52%). Candidates: combat previews per
   block (damage, deaths), lethal on board, mana available next turn.
6. **Decision-time search.** If the R-NaD policy plateaus mainly on combat and combo prompts,
   search them (or use search for evaluation) over worlds sampled from the knowledge sidecar's
   model, with the snapshot/restore that exists, and this network as prior and value.
7. **Model size for the long run.** R-NaD decisions per second by size (the model-perf step):
   d192/L4 1,328, d128/L4 1,844, d128/L3 2,249. Decide after a short BC comparison at d128.
8. **The exploiter gives no meaningful number yet.** Three 20-minute PPO runs against the BC policy
   ended at 36–39% win rate, below the 48% they started at. It needs larger rollouts, a KL anchor
   to the start policy, or a longer budget.
9. **Knowledge error with test-harness presets in bo3 game 2.** "A card is created from zone none
   with nothing set aside" (found during T7, pre-existing): the knowledge sidecar reports it in
   game 2 of a bo3 started with test-harness zone presets. It does not occur in training games.
10. **The value head and the matchup** (league12h, 2026-10-20; corrected 2026-10-21). R-NaD made
    the value head calibrated (DAgger predicted +0.95 for positions that ended +0.80; the final
    network +0.93 → +0.95) and cut its self-play error from 0.66 to 0.51. Early in a game it is
    about as good as a lookup of each matchup's mean result: 0.82 at the start of game 1 and 0.73
    at the start of games 2–3, against the lookup's 0.79 when the lookup is fitted on the run's
    other games (some 7,000 per matchup). An earlier comparison fitted the lookup on the ~400
    scored games themselves (about 9 per matchup), which flattered it (0.67–0.69) and suggested the
    head ignored the matchup; it does not. What remains is that a game's result is mostly
    unpredictable at its start (squared error near 0.8 against 1.0 for predicting 0), which is the
    noise of every sideboarding decision's credit, not a defect of the head. Its gradient on a
    32-game batch is about 2% signal (break-even near 1,650 games), the trunk's about 1%.
11. **Learning pace** (league12h, 2026-10-20). Against `scripted:hard`, paired by match, the final
    network gained +0.017 ± 0.013 over the DAgger policy, rising steadily with steps (about +0.23
    points per 1,000 learner steps; 15k steps a day). The policy moved little (its top move differs
    from DAgger's at 1% of decisions) because each step is mostly noise: a 32-game gradient is about
    7% signal overall and 1.4% per coordinate (Adam's view), with a break-even batch near 420 games
    (DeepNash's 768 is above it). Candidates, to pilot before a multi-day run: learning rate 2–4×
    5e-5, larger batches, and the value-head pass above.
