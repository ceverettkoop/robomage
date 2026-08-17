# Plan: cross-world leaf batching + GPU self-play inference

Status: DESIGN (nothing below is built). Companion measurement: the
`bench_actor.py --batch` sweep (branch `claude/leaf-batching-mcts-assessment-*`)
quantifies the CPU-side win of batching leaf evals; this doc is the follow-on
plan. Scope is the **self-play generation** side of AZ training only — `az_train`
SGD is fast enough on CPU and is deliberately out of scope.

The four stages are strictly ordered: each one is useful on its own, and each
later stage consumes the previous one's machinery.

```
Stage 0  cross-world batching      (CPU, no quality cost, K = worlds)
Stage A  CUDA in the actor         (same K, GPU forward, validates the path)
Stage B  multi-game actor process  (K = games x worlds, one process)
Stage C  central inference server  (K = fleet-wide, the KataGo shape)
```

Throughout: `K` = rows per net forward. The measured CPU sweep tells us what
amortization K buys; GPU economics need K >= ~32–64 to beat the CPU forward at
this net size (CardGameExtractor trunk + [256,256] heads).

## Measured: CPU batch sweep (2026-08-17)

`bench_actor.py --games 2 --sims 128 --worlds 4 --batch 1 4 8 16 --no-python`,
release actor, fresh deterministic net, `league/ur_delver` mirror, single
thread, 4-core cloud container. ms/dec = wall per SEARCHED root (128 sims
each), the batch-comparable metric (game trajectories diverge across batch
values, so decision counts differ; per-decision cost does not care).

| batch | ms/dec | speedup |
|------:|-------:|--------:|
|     1 | 1008.5 |   1.00x |
|     4 |  600.6 |   1.68x |
|     8 |  486.0 |   2.08x |
|    16 |  360.8 |   2.79x |

Readings:
- **Net share of per-sim cost is ~2/3 or more**: b=16 removes >64% of the
  per-decision cost and the curve has not flattened, so the engine floor
  (restore + path replay) is below 360 ms/dec — i.e. batching-only ceiling on
  CPU is ~3–3.5x, and post-Stage-C the same floor bounds the per-actor gain.
- **Cross-world batching (Stage 0) is worth ~1.7x at worlds=4 and ~2.1x at
  worlds=8** (K = worlds, read off the b=4/b=8 rows) with zero quality cost.
- The b=16 row runs HALF of each 32-sim world tree in flight under virtual
  loss — quality-suspect at this budget; treat it as an upper bound on the
  amortization curve, not a recommended config.
- Single-thread numbers = CPU-work reductions, so they translate ~directly to
  fleet throughput at fixed cores.

## Why not just use the existing `--batch K` (virtual loss)?

The actor's implemented batching collects K leaves **within one world's tree**
under virtual loss. At the production budget (256 sims / 4 worlds = 64 sims per
tree) a useful K is a large fraction of the tree, so selection quality — and
therefore the π visit targets the trainer learns from — degrades, and the
config leaves the `test_mcts_parity` bit-parity envelope. Cross-world batching
gets K = worlds with **zero** search-quality cost, because the worlds are
already independent trees.

## Stage 0 — cross-world batching (CPU)

**Idea.** Run the per-world sims round-robin instead of world-at-a-time:
`w0.s0, w1.s0, …, w(W-1).s0, flush, w0.s1, …`. Each round, every world with
remaining budget contributes at most one PendingLeaf; the round ends with ONE
batched forward (`evaluate_double_batch`, K <= worlds). No virtual loss: a
world's own next descent never starts before its previous leaf is backed up, so
every tree evolves exactly as it does today.

**Equivalence argument.** Round-robin vs sequential world order does not change
any single world's sim sequence or the state each of its descents sees
(IncrementalSearch already relies on exactly this to be bit-identical to
`run_search`). Deferring the leaf eval to the end of the round backs it up
before the world's next descent, so per-world trees are *arithmetically
identical* to today's search. The only divergence is last-ulp: a batched GEMM's
logits differ from single-row forwards. Under a UniformEvaluator there is no
net, so visit counts must match **bit-exactly** — that is the CI gate for the
scheduler (see Testing).

**Changes (all in `src/actor/az_mcts.{h,cpp}` unless noted):**

1. `MCTSConfig`: add `bool cross_world = false` (actor flag `--cross-world`,
   plumbed through `actor_selfplay_cmd` like `--batch`). Mutually exclusive
   with `batch > 1`.
2. **Roots up front.** Move the `begin_world(w)` calls (root adoption + noise
   draw + merge fold) for all worlds to search start, mirroring
   `mcts.py::_init_worlds`. RNG stream is unchanged: sims draw nothing from the
   noise rng, so hoisting the per-world gamma draws preserves the sequence.
3. **Scheduler.** Replace the `cur_world`/`cur_sim` block loop in
   `advance_after_restore`/`start_next_world_sim` with: `next_world` pointer +
   per-world `remaining` budgets; after each `finish_sim`, advance to the next
   world with budget; on wrap (a full round), `flush_pending()` first.
   Worlds with zero top-up budget (boundary persistence) fold in as today.
4. **PendingLeaf without virtual loss**, plus a `value_only` flag so the
   depth-cap eval (today an immediate `eval_one`) can also be deferred — it
   backs up a value but creates no node.
5. **Terminals / memo hits** back up immediately as today (they need no eval);
   the round's batch just comes up short.
6. **Rollout roots stay on the immediate path** (`cur_rollout_turns > 0`
   forces per-leaf eval, exactly like `batch > 1` today) — i.e. bo3 sideboard
   roots see no change and no gain. Documented, accepted.
7. **Knob interaction:** recommend `--worlds 8` at fixed `--sims` once this
   lands — it doubles K *and* determinization coverage; quality moves up, not
   down. Sweep 4 vs 8 through the az-eval gate before making it the default.

**Testing.**
- `test_mcts_parity.py` gains a cross-world line: UniformEvaluator (torch-free)
  cross-world vs plain `run_search` — **exact** visit parity required (this
  pins the scheduler). With the real net: argmax-agreement line only (GEMM ulp),
  like today's batch=16 line.
- `eval_search_gate.py` A/B same checkpoint, cross-world on vs off — strength
  must be flat within noise.

## Stage A — CUDA in the actor

Smallest possible GPU step: same search, same K, the forward moves to the GPU.
Its purpose is to validate the CUDA build/link/runtime path and get a real
measurement of small-K GPU economics before any architecture work.

1. **`AZEvaluator` device support** (`src/actor/az_evaluator.cpp`): `load()`
   takes a device string; `module_.to(device)`. `evaluate_double_batch` stages
   the `[k, OBS]` batch in a pinned CPU tensor, `.to(device, /*non_blocking=*/
   true)`, forwards, copies logits/value back to CPU. The double-precision
   prior math (float32 softmax → float64 renormalize) stays on CPU, unchanged —
   `AZEvalResultD` semantics identical.
2. **Makefile**: when `$(LIBTORCH_DIR)/lib/libtorch_cuda.so` exists, append
   `-ltorch_cuda -lc10_cuda` wrapped in `-Wl,--no-as-needed` (without it the
   linker drops `libtorch_cuda` — no direct symbol refs — and CUDA silently
   isn't registered at runtime). CPU-only venvs build exactly as today.
3. **Flags**: `az_actor --device cpu|cuda` (default cpu);
   `az_selfplay`/`az`/`az-league` pass-through `--actor-device`.
4. **Fleet shape**: N processes sharing one GPU each own a CUDA context
   (~hundreds of MB) and serialize on the device. Run fewer, fatter workers
   (e.g. 8 x worlds=8) and/or enable CUDA MPS. This stage is expected to be
   only a modest win — K = worlds is below the GPU break-even; that is fine,
   it is the stepping stone, and the measurement decides whether B/C are
   worth it.

## Stage B — multiple concurrent games per actor process

Goal: K = `games_in_flight x 1` leaves per flush (x worlds if the scheduler
interleaves at world granularity), from ONE process — GPU batches in the 32–256
range without any IPC.

**The blocker and the key enabler.** The engine is global-state
(`global_coordinator`, `cur_game`), so M games cannot coexist as live engine
states. But `snapshot.h` already deep-copies the COMPLETE per-game state —
`cur_game + ECS + card_db + match revealed arrays`, and MATCH scope even
round-trips an `init_ecs()` teardown. So M games can coexist as M snapshot
blobs, with exactly one hydrated in the engine at a time.

**Design: fibers + snapshot context-switch.**

- M worker fibers (ucontext/boost::context — or OS threads with a baton mutex,
  same structure), each running its own `play_single_game`/match loop with its
  own `AZMcts` instance and its own snapshot slot(s). Exactly one fiber runs at
  a time; there is no engine-level concurrency.
- **Yield point** = the natural pause the state machine already has: after a
  sim latches its restore and the provider is back at AWAITING_ROOT, the
  engine state equals the fiber's own search-root snapshot. Park there: no
  extra capture needed. To resume fiber j: `snapshot_restore(slot_j)` and
  continue its state machine. Context-switch cost ≈ one restore (~0.3–1 ms),
  paid once per collected leaf — measure against the ~5–10 ms/sim baseline;
  if it dominates, switch at coarser granularity (one full ROUND of a fiber's
  worlds per turn) to amortize.
- A shared `PendingLeaf` queue across fibers; the scheduler flushes one big
  forward when every runnable fiber has contributed its round (or a row cap is
  hit), then resumes fibers to back up their results. Cross-game rows are
  independent — still no virtual loss anywhere.
- `N_SNAPSHOT_SLOTS` (currently 4, `src/snapshot.h`) becomes M + spare; each
  slot is a full deep copy, so M is bounded by memory — M = 8–16 is the
  target range, which with worlds=8 already yields K = 64–128.
- Shard writing: per-fiber sample buffers, flushed under the existing
  per-match discipline; seeds partition exactly as the process-level fleet
  does today (fiber i owns seed range i).
- **Non-goals**: no engine de-globalization, no engine-thread parallelism.
  The engine stays single-threaded; only the net forward is batched.

This is the most invasive stage. It should only be built if Stage A's
measurement shows small-K GPU forwards leaving most of the device idle AND
Stage C's IPC route is unattractive (see below) — otherwise skip straight to C.

## Stage C — central inference server (the KataGo/ELF shape)

One GPU-owning server process; the existing one-game-per-process actor fleet
submits leaf batches over IPC and blocks for results. Achieves fleet-wide K
(28 actors x worlds rows in flight) with NO change to the actor's game/search
structure — C does not require B.

- **Transport**: Unix domain socket, length-prefixed binary frames.
  Request: `k, k x OBS_SIZE float32, k x int32 num_choices`. Reply:
  `k x MAX_ACTIONS float32 logits, k x float32 value`. The server returns RAW
  logits; the client keeps the exact double-precision prior computation
  locally, so `AZEvalResultD` numerics are identical whether the forward ran
  locally or remotely.
- **Server** (`train/az_eval_server.py`, ~150 lines): loads the `.ts.pt`
  (or checkpoint) once, accumulates requests up to `max_batch` (256–512) or a
  micro-timeout (~0.5–2 ms), one forward, dispatches replies. Torch already
  lives in the venv; no new C++.
- **Client**: a second `AZEvaluator` backend (`--eval-server <socket>`)
  implementing `evaluate_double_batch`/`evaluate_double` over the socket.
  Everything above the evaluator interface is untouched — Stage 0's
  cross-world flush is what makes each actor's requests K-row instead of
  1-row, keeping request rate (and per-row IPC overhead) low.
- **Lifecycle**: `az_selfplay` starts the server before the fleet and passes
  the socket path; simplest checkpoint-rotation story is a fresh server per
  az-league slot (startup is seconds). Any transport/server error in the
  actor is FATAL AND LOUD (`exit(1)`), matching the evaluator's existing
  error discipline — no silent local fallback.
- **Sizing sanity check**: ~28 actors x ~100–200 evals/s ≈ 3–6 k rows/s
  fleet-wide; one GPU at batch 256 on this net does an order of magnitude
  more. The engine (restore + path replay, ~0.5–1.5 ms/sim floor) becomes
  the bottleneck again — which bounds the whole program at roughly
  1/(engine share) per actor. That engine share is exactly what the CPU
  batch sweep measures; read the ceiling off that number before promising
  more.

## Cross-cutting: what guards quality at every stage

- The bit-parity harness (batch=1, CPU) stays green and untouched — it remains
  the machinery gate. Cross-world adds the uniform-evaluator EXACT gate.
- Every stage lands behind a flag, default off, and must pass: (1) the
  cross-world uniform parity gate, (2) an `eval_search_gate` A/B at equal sims
  (strength flat), (3) an az-league slot where the candidate still clears the
  az-eval promotion gate. The promotion gate bounds the blast radius of any
  mistake to wasted compute — `gen__azfinal` cannot regress silently.
- π/z/td_q sample semantics are unchanged at every stage: Stage 0 produces
  arithmetically identical visits; A/B/C only change WHERE the same forward
  runs and how many rows share it.
