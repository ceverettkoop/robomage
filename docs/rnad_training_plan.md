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
  v-trace for the value and Q estimates (ρ̄ = c̄ = 1) and NeuRD for the policy (an update on the
  logits rather than the probabilities, logit threshold β = 2, clip 10,000).
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

Decisions the engine auto-takes (a lone pass) are never asked, so they never appear. The seat
sequence is therefore irregular, and every record carries its seat explicitly.

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

---

## 4. Learner side (new: `learn/`, PyTorch)

The kept Python in `python/` is stdlib-only by rule. The learner needs PyTorch (ROCm build), so it
lives in its own directory with its own requirements file and virtual environment, and nothing in
the engine, the gate or `python/` imports it.

- **L1. Episode reader.** Reads T1 episodes into a RAM replay window. Batches are full games,
  padded to the longest game in the batch and split into fixed-length chunks, as DeepNash does,
  so returns are exact rather than bootstrapped.
- **L2. Model.** The step-5 contract: an entity transformer with relation biases, a pointer
  action head over menu rows, a value head, the card / text / ability tables as frozen buffers,
  and an id embedding with id-dropout. Auxiliary heads predict hidden information (the
  opponent's hand, its archetype) from omniscient targets. Export to the actor's format at every
  checkpoint.
- **L3. Behaviour cloning.**
  - Cross-entropy to the teacher's label, with label smoothing.
  - Value regression to the game outcome, optionally through an omniscient critic (below).
- **L4. R-NaD.** A PyTorch port of OpenSpiel's `rnad.py` (JAX reference implementation):
  - the reward transform with the server-supplied `log π_reg` (T2);
  - two-player v-trace over the irregular seat sequence;
  - NeuRD with logit thresholding;
  - target-network averaging;
  - the Δm schedule and the fine-tuning projection.
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

---

## 6. The training cycle (phases)

### B0. Teacher corpus

`scripted:hard` vs `scripted:hard` over every league matchup, bo3, through `rm_driver.py` with
RMLOGs kept. Add `scripted:hard` vs `explore` games for coverage of bad positions.

At ~400 decisions/s per Python process × 20, about 1M decisions takes around 2–3 minutes; aim for
5–10M. The tensors are produced by re-featurizing the RMLOGs (T1's replay path) in parallel.

### B1. Behaviour cloning

Train L3 on B0.
- **Exit:** top-1 agreement with the teacher on held-out games, and an SPRT result for the BC
  policy against `scripted:hard` (expected to be below 50%; it is a starting point, not a goal).

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
  `sb_drift`) and the `own_deck` tokens (the seat's current 75, main and side counts per card);
  the opponent's sideboard is a count and its post-board configuration stays hidden;
- the menu's `SIDEBOARD_IN` / `SIDEBOARD_OUT` / `SIDEBOARD_DONE` rows, each with `source_card`
  (the card's vocab id) and `ordinal` (copies). The engine offers a swap in-first (a card in,
  then the cut that balances it), withholds Done while the deck is off-size, and caps a phase at
  15 swaps;
- the knowledge sidecar's memory of the opponent across games (revealed cards, `k_archetype`).

The pointer head picks sideboarding rows like any other menu rows. A sideboarding decision is part
of the next game's episode (T1) and is credited with that game's result. Exploration comes from the
larger sideboarding ε in `π_0,reg` (B3); B1–B2 teach Done there, which is all the teacher does.

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
3. **No teacher for sideboarding.** The scripted agent never sideboards (it always answers Done),
   so imitation teaches nothing about sideboarding, and sideboarding skill comes only from R-NaD's
   exploration, credited through one game's result per sideboarding phase: a weak, noisy signal.
   To refine later: a sideboarding teacher (rules in the scripted agent, or the archive's
   dead-card rules table, `sb_dead_rules.json` on `archive/rl-stack`), or the user's own guidance
   per matchup as labels or as a prior in `π_0,reg`.
