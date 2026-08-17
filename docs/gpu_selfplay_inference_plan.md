# Plan: cross-world leaf batching + GPU self-play inference (AMD RX 6700)

Status: Stage 0 is IMPLEMENTED (actor `--cross-world`; gates in
`test_mcts_parity.py`); the sanity check is PASSED — verdict **GO** (2026-08-17,
measured results in the sanity section below, working torch pair pinned in the
Install bullet); Stages A and C are DESIGN. Scope is the **self-play
generation** side of AZ training only — `az_train` SGD is fast enough on CPU
and is deliberately out of scope.

Revision note: an earlier draft had a Stage B (multi-game actor processes via
snapshot-multiplexed fibers). It is DROPPED: Stage C achieves cross-process
batching without touching the actor's one-game-per-process structure, so B's
complexity (fiber scheduling, snapshot context switches, `N_SNAPSHOT_SLOTS`
growth) buys nothing C doesn't. The remaining ladder:

```
Stage 0  cross-world batching       (CPU, no quality cost, K = worlds)  [BUILT]
Sanity   time the net on the 6700   (10 minutes; gates all GPU work)   [PASSED — GO]
Stage A  HIP/ROCm in the actor      (same K, GPU forward, validates the path)  [BUILT]
Stage C  central inference server   (K = fleet-wide, the KataGo shape)  [BUILT]
```

Throughout: `K` = rows per net forward.

## Target hardware: AMD Radeon RX 6700 (ROCm, not CUDA)

The GPU is an RX 6700 — Navi 22, **`gfx1031`**, RDNA2, ~11–13 TFLOPS FP32,
10 GB VRAM (ample; the net is tiny). Everything GPU-side goes through
**PyTorch's ROCm build**, which deliberately masquerades as CUDA:
`torch.cuda.is_available()` is True and `device="cuda"` targets the Radeon, so
Python-side code needs no AMD-specific branches. Constraints to plan around:

- **Officially unsupported card.** ROCm's support list does not include RDNA2
  consumer parts. The standard community workaround is
  `HSA_OVERRIDE_GFX_VERSION=10.3.0` (run `gfx1030` kernels — Navi 21/22 are
  ISA-compatible), widely reported reliable for torch. Treat ROCm upgrades as
  potentially breaking; pin the working ROCm/torch pair once found.
- **Linux only.** No Windows torch-ROCm; WSL2 ROCm does not cover RDNA2.
- **No matrix cores** (WMMA is RDNA3+): plain shader FP32/FP16 throughput.
  Irrelevant at our scale — the fleet needs only ~3–6k leaf evals/s and even
  a few effective TFLOPS is orders of magnitude past that at batch 128–256.
- **Install — the pinned working pair (verified 2026-08-17)**:
  `train/.venv/bin/pip install torch==2.10.0+rocm7.0 --index-url
  https://download.pytorch.org/whl/rocm7.0` (libtorch ships inside the same
  wheel — Stage A links against it). Why this exact pair and not the ROCm 6.x
  this bullet originally named: the venv is Python 3.14, and the rocm6.x
  indexes top out at torch 2.9.1 for cp314 — whose `torch.jit.script` is
  broken on 3.14 (`__annotations__` → `__annotate_func__`), so it can run but
  not CREATE `.ts.pt` exports, which az training writes per candidate.
  2.10.0+rocm7.0 passes the full sanity check on the 6700 (HIP 7.0 + the gfx
  override, timings equivalent to 2.9.1+rocm6.4) and matches the torch
  version the venv ran before the CUDA→ROCm swap.
- **Swapping the venv's torch invalidates the actor binaries.**
  `bin/{debug,release}/az_actor` link libtorch out of the venv by rpath but
  are compiled against its headers; a version swap under existing binaries
  crashes at runtime (observed: c10 `set_stride` abort from 2.9.1-era inlined
  header code calling 2.10 libs). After ANY venv torch change: `make actor`,
  `make actor BUILD=RELEASE`, then `train/.venv/bin/python train/ci_check.py
  --tier actor` (full tier green on 2.10.0+rocm7.0, 2026-08-17: obs bit-exact,
  MCTS visit parity exact incl. the cross-world gates, shard + trainer legs).

## Sanity check — run BEFORE any GPU engineering

Ten minutes on the 6700 box decides whether Stages A/C are live. With the
ROCm wheel installed and `HSA_OVERRIDE_GFX_VERSION=10.3.0` exported:

1. `torch.cuda.is_available()` → must be True;
   `torch.cuda.get_device_name(0)` names the Radeon.
2. Export the net once (`train/az_net.py --export gen`, or a fresh
   `AZNet(obs_space_from_const())` under `torch.manual_seed(0)` where no
   checkpoint exists) and `torch.jit.load` it.
3. Time batched forwards, CPU vs GPU, e.g.:
   ```python
   import time, torch
   m = torch.jit.load("gen__azfinal.ts.pt")
   for dev in ("cpu", "cuda"):
       net = m.to(dev)
       for k in (1, 8, 64, 256):
           obs = torch.randn(k, OBS_SIZE, device=dev)
           mask = torch.ones(k, MAX_ACTIONS, dtype=torch.bool, device=dev)
           for _ in range(10): net(obs, mask)          # warmup
           torch.cuda.synchronize() if dev == "cuda" else None
           t0 = time.perf_counter()
           for _ in range(100): net(obs, mask)
           torch.cuda.synchronize() if dev == "cuda" else None
           print(dev, k, (time.perf_counter() - t0) / 100 * 1e3, "ms")
   ```
4. **Go/no-go**: GPU at k=256 should beat CPU-per-row by a wide margin
   (order of magnitude expected); GPU at k=1 will likely LOSE to CPU —
   that's the launch-latency economics the whole plan is built around, not a
   failure. If the override crashes or numbers look broken, stop: stay on
   the CPU path (Stage 0 alone) until the ROCm situation changes.

### Measured 2026-08-17 — PASSED, verdict GO

torch `2.10.0+rocm7.0`, `HSA_OVERRIDE_GFX_VERSION=10.3.0`,
`HIP_VISIBLE_DEVICES=0` (device 0 is the 6700; device 1 is the Raphael iGPU —
pin it out), net `gen__azfinal.ts.pt` (OBS_SIZE 7161, MAX_ACTIONS 64),
100 timed forwards after 10 warmups, no crashes:

| k   | CPU ms/fwd | GPU ms/fwd | GPU rows/s | GPU/CPU |
|-----|-----------|-----------|------------|---------|
| 1   | 1.88      | 3.34      | 299        | 0.56x   |
| 8   | 3.50      | 3.71      | 2 154      | 0.94x   |
| 64  | 17.8      | 4.50      | 14 223     | 3.95x   |
| 256 | 66.0      | 8.18      | 31 312     | 8.07x   |

GPU@256 beats CPU-per-row (k=1) by **59x**; k=1 loses exactly as predicted,
crossover sits near k≈8–16. CPU saturates ~3.9–4.1k rows/s regardless of
batch; GPU@256 delivers ~31k rows/s — several times the fleet's ~3–6k leaf
evals/s need. Implication for the ladder: at Stage 0/A's per-process
K = worlds (default 4) the GPU does NOT pay (below crossover); the win is
Stage C's fleet-wide K. (2.9.1+rocm6.4 measured equivalent — 8.54 ms @
k=256 — so the ROCm major version is not the bottleneck; 2.10/rocm7.0 is
pinned for the py3.14 `jit.script` fix.)

## Why not the existing `--batch K` (virtual loss)?

The actor's older batching collects K leaves **within one world's tree**
under virtual loss. At the production budget (256 sims / 4 worlds = 64 sims
per tree) a useful K is a large fraction of the tree, so selection quality —
and the π visit targets the trainer learns from — degrades, and the config
leaves the bit-parity envelope. Cross-world batching gets K = worlds with
**zero** search-quality cost, because the worlds are independent trees.

## Stage 0 — cross-world batching (CPU) — BUILT

Implemented as `MCTSConfig::cross_world` / `az_actor --cross-world`
(mutually exclusive with `--batch K>1`), plumbed through
`actor_selfplay_cmd(cross_world=)` and `bench_actor.py --cross`.

Mechanism: all world roots are built up front (ascending world order — the
noise-rng stream is unchanged), then sims run round-robin across worlds. A
freshly expanded (or depth-capped) leaf defers into the shared PendingLeaf
batch; the batch is flushed in ONE forward before any world's own next
descent starts (per-world in-flight flags; finalize flushes the tail), so
**no virtual loss is applied** and every per-world tree is arithmetically
identical to the sequential search. Only the batched GEMM's last-ulp logits
can differ. Searches whose budget has leaf rollouts on (bo3 sideboard roots
by default) keep the unchanged sequential path — a deferred leaf cannot
drive a playout.

Gates (`test_mcts_parity.py`), all green 2026-08-17 (release actor, plus
`make check` fully green on the same tree):

- Every pre-existing batch=1 gate unchanged and exact (bo1 387 roots, bo3
  2319, no-merge 931, both sb-budget modes).
- `xw-uniform-bo1`: cross-world visits **bit-exact** vs sequential over
  1968 searched roots (31488 root visits).
- `xw-uniform-bo3-sb-persist`: **bit-exact** over 5337 roots (85360 visits)
  — the sequential/cross mode split and boundary persistence hold together.
- Real-net report: **387/387 = 1.000** argmax agreement over the full game —
  the batched forward's ulp differences never flipped a decision (contrast
  virtual-loss batch=16: 0.960 over only 25 roots before the games diverge).

Measured (same bench protocol as the sweep below; note both legs play
IDENTICAL games — equal decision counts — which is the arithmetic-identity
property showing up live):

| worlds | b=1 ms/dec | cross ms/dec | speedup |
|-------:|-----------:|-------------:|--------:|
|      4 |     1108.6 |        643.4 | **1.72x** |
|      8 |     1022.7 |        471.2 | **2.17x** |

Knob guidance: prefer `--worlds 8` at fixed `--sims` — doubles K *and*
determinization coverage; sweep 4 vs 8 through the az-eval gate before
changing the default.

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

## Stage A — HIP/ROCm in the actor — BUILT (2026-08-17)

Smallest possible GPU step: same search, same K (= worlds), the forward moves
to the Radeon. Its purpose is validating the ROCm build/link/runtime path in
C++ and measuring small-K GPU economics on this exact card before the server
work.

Implemented as designed (items 1–3 below: `AZEvaluator::load(path, device)`
with pinned staging + CPU prior math, the conditional `--no-as-needed`
HIP link, `az_actor --device cpu|cuda` + the `--actor-device` pass-through on
az-selfplay/az/az-league with the HSA override + `HIP_VISIBLE_DEVICES=0`
exported by `_generate_actor`). Verified: a full cross-world searched game on
the 6700 (1056 sims, clean GAME_RESULT). One landmine found and fixed:
**TorchScript's NNC/TensorExpr fuser** effectively hangs (gdb-verified: the
IRSimplifier grinds >4 min at 100% CPU on the first cuda forward) recompiling
this gather-heavy module for the GPU — `AZEvaluator` now calls
`torch::jit::setTensorExprFuserEnabled(false)` on non-cpu loads (the net is
GEMM-dominated; fusion is worthless here). The CPU path never executes that
branch, so the bit-parity envelope is untouched — `ci_check --tier actor`
stays the machinery gate.

1. **`AZEvaluator` device support** (`src/actor/az_evaluator.cpp`): `load()`
   takes a device string; `module_.to(device)`. `evaluate_double_batch`
   stages the `[k, OBS]` batch in a pinned CPU tensor,
   `.to(device, /*non_blocking=*/true)`, forwards, copies logits/value back.
   The double-precision prior math (float32 softmax → float64 renormalize)
   stays on CPU, unchanged — `AZEvalResultD` semantics identical whether the
   forward ran on CPU or GPU. (torch's HIP backend registers as `cuda`, so
   the device string stays `"cuda"` even on the Radeon.)
2. **Makefile**: when the venv libtorch has `lib/libtorch_hip.so`, append
   `-ltorch_hip -lc10_hip` wrapped in `-Wl,--no-as-needed` (without it the
   linker drops the lib — no direct symbol refs — and the backend silently
   never registers; the same gotcha as CUDA's `torch_cuda`). CPU-only venvs
   build exactly as today.
3. **Flags**: `az_actor --device cpu|cuda` (default cpu; `"cuda"` = the
   Radeon under ROCm); `az_selfplay`/`az`/`az-league` pass-through
   `--actor-device`. Runtime env: `HSA_OVERRIDE_GFX_VERSION=10.3.0` must be
   set for every actor process — export it in the launcher
   (`_generate_actor`), not per-shell.
4. **Fleet shape**: N processes sharing the one Radeon serialize on the
   device and each hold a context. Run fewer, fatter workers (e.g. 8 x
   `--worlds 8`). There is no AMD equivalent of CUDA MPS to lean on, which
   is fine — Stage A is a stepping stone, and K = worlds is below GPU
   break-even by design. The measurement (bench `--cross` leg with
   `--actor-device cuda`) decides how quickly to move to C.

## Stage C — central inference server (the KataGo/ELF shape) — BUILT (2026-08-17)

One Radeon-owning server process; the existing one-game-per-process actor
fleet submits leaf batches over IPC and blocks for results. Achieves
fleet-wide K (workers x worlds rows in flight) with NO change to the actor's
game/search structure.

Implemented as designed below: `train/az_eval_server.py` (UDS, hello layout
handshake, micro-timeout batching, raw logits + value replies, fresh server
per generation pass) and the `AZEvaluator::connect_server` socket backend
behind `az_actor --eval-server <socket>` (exactly one of --model/--uniform/
--eval-server; the prior math stays local). Driver: the `--eval-server` flag
on az-selfplay/az/az-league (persisted in the az-league resume sidecar) makes
`_generate_actor` start the server (on `--actor-device`, default cuda, with
the ROCm env exports), wait for READY, hand every actor the socket, and tear
it down in the same `finally` as the fleet. The socket lives in a short
`mkdtemp` dir — AF_UNIX paths cap at ~107 chars, so `out_dir` is not a safe
home for it. Verified: single actor game over the socket (GPU server, clean
GAME_RESULT, same outcome as the Stage A local-GPU game on the same seed) and
an end-to-end `train.py az-selfplay --actor --eval-server` pass.

Test/bench coverage (2026-08-17): `test_mcts_parity.py` gained a
**`server-cpu` EXACT gate** (a CPU server must be bit-transparent through the
wire protocol — PASS, 387/387 roots identical to the local forward; runs in
CI with no GPU, rides the existing `actor` tier) and a **`server-gpu` drift
REPORT** (one bo1 game, report-only, self-skips without a GPU — measured
387/387 argmax agreement with visit-count L1 drift max=0: on this net the
GPU's ulp differences flipped nothing). `--legs server` runs just these
against a fresh local reference (~minutes) for protocol iteration; the
default `--legs full` keeps every slow Python-reference gate.
`bench_actor.py` gained `--device cpu|cuda`, `--eval-server DEV`, and
`--fleet N` (N concurrent actors, per-leg evals/s column) for fleet
throughput measurement; production-budget numbers not yet recorded. A
strength gate (az-eval A/B, promotion-gated az-league slot) remains the
deliberately-un-built track before --eval-server becomes a default.

- **Transport**: Unix domain socket, length-prefixed binary frames.
  Request: `k, k x OBS_SIZE float32, k x int32 num_choices`. Reply:
  `k x MAX_ACTIONS float32 logits, k x float32 value`. The server returns
  RAW logits; the client keeps the exact double-precision prior computation
  locally, so `AZEvalResultD` numerics are identical whether the forward ran
  locally or remotely.
- **Server** (`train/az_eval_server.py`, ~150 lines): venv ROCm torch, loads
  the `.ts.pt` once, `.to("cuda")` (the Radeon), accumulates requests up to
  `max_batch` (256–512) or a micro-timeout (~0.5–2 ms), one forward,
  dispatches replies. Being a plain venv-python process, the whole
  ROCm/HSA-override surface lives HERE and nowhere else — with Stage C in
  place the actors can stay CPU-built (Stage A's HIP link becomes optional
  validation tooling, not a production dependency).
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
- **Sizing sanity check**: the fleet generates ~3–6k leaf evals/s; the 6700
  at batch 256 on this small net does an order of magnitude more (confirm
  with the sanity-check timing above). The engine (restore + path replay,
  measured floor < 360 ms per 128-sim root) becomes the bottleneck again —
  bounding the per-actor gain at roughly 3x over today's unbatched CPU
  baseline, per the sweep table.

## Cross-cutting: what guards quality at every stage

- The bit-parity harness (batch=1, CPU) stays green and untouched — it
  remains the machinery gate. Stage 0 adds the uniform-evaluator EXACT
  gates for the cross scheduler.
- Every stage lands behind a flag, default off, and must pass: (1) the
  cross-world uniform parity gates, (2) an `eval_search_gate` A/B at equal
  sims (strength flat), (3) an az-league slot where the candidate still
  clears the az-eval promotion gate. The promotion gate bounds the blast
  radius of any mistake to wasted compute — `gen__azfinal` cannot regress
  silently.
- π/z/td_q sample semantics are unchanged at every stage: Stage 0 produces
  arithmetically identical visits; A and C only change WHERE the same
  forward runs and how many rows share it. GPU forwards differ from CPU in
  last-ulp logits only — same class of difference the cross-world gates
  already scope out via the uniform-evaluator legs.
