# AlphaZero MCTS — implementation status & handoff

Branch: `claude/alphazero-mcts-plan-dwcubj`. This documents what has landed, how to
drive it, and exactly what to run on real hardware. The full design plan (four
phases, decisions, risks) is summarized at the bottom.

## What is DONE and pushed

### Phase A — engine snapshot/rollback + search protocol (commits `c6c0a33`, `4937a0e`)

The engine can deep-copy and roll back its complete game state in-process, and
exposes it over machine mode behind a new **`--search-server`** flag (implies
`--machine`; incompatible with `--replay` / `--log-decisions` / `--player`).
With the flag off, engine behavior is byte-identical to before (CI-proven).

Protocol, alongside the usual integer action reply:

| Command | Effect | Reply |
|---|---|---|
| `SNAPSHOT <slot>` (0-3) | deep copy state into slot (only at `safe=1` decisions) | `SNAPSHOT_OK <slot>` + re-emitted query |
| `RESTORE <slot>` | roll back (works from ANY decision via cooperative unwind) | restored decision's query |
| `DETERMINIZE <seed>` | reshuffle hidden zones with a world-local RNG (only at `safe=1`) | `DETERMINIZE_OK` + re-emitted query |
| `RELEASE` | drop all snapshots / leave the terminal intercept | `RELEASE_OK` (+ query at a live decision) |

Every query is preceded by `SEARCHINFO safe=<0|1>`. Loop-safe (`safe=1`) =
priority decisions, attacker/blocker SELECT prompts, cleanup discard; the
attack/block TARGET sub-prompts, combat damage assignment, mid-resolution
prompts (targets/digs) and mulligans are `safe=0` (snapshot there is a fatal
error — but simulations traverse them freely; safety only gates the root).
If a game ends while a snapshot is live the engine emits `SIM_RESULT: <A|B|DRAW>`
and blocks for RESTORE/RELEASE instead of exiting.

Key invariants (all enforced by the CI `snapshot` tier, `train/test_snapshot.py`,
run by `make check`): snapshot→diverge→restore is byte-identical; DETERMINIZE
never touches `cur_game.gen`, so the real line replays identically after any
amount of search; same world seed reproduces the same simulated future,
different seeds diverge.

Implementation map: `src/snapshot.{h,cpp}` (state copy: `cur_game` + all four
ECS managers + `card_db` + revealed arrays), `src/search_server.{h,cpp}`
(protocol, cooperative unwind, terminal intercept, determinize),
`src/input_logger.cpp` (line-based machine read + command dispatch),
`src/main.cpp` (loop-top restore, intercept call sites, flag),
`src/action_processor.cpp` (loop-safe markers).

### Phase B — Python MCTS at inference (commit `df6fa6f`)

- `train/mcts.py` — determinized PUCT `run_search(env, evaluator, sims=, worlds=,
  c_puct=, root_noise_eps=, root_noise_alpha=, rng=)`: one root snapshot, K
  worlds (per-world trees; root visit counts summed), each simulation =
  RESTORE + DETERMINIZE(world seed) + descend. Values are backed up in each
  node's own mover perspective (obs is always priority-relative); terminal
  `SIM_RESULT`s back up exact ±1. Pluggable `Evaluator` protocol:
  `UniformEvaluator` (torch-free) and `PPOEvaluator` (MaskablePPO policy/value
  heads; value squashed `tanh(v/v_scale)`).
- `train/search_env.py` — `SearchRoboMageEnv` / `SearchNarrativeEnv` with
  `snapshot()/restore()/determinize()/release()/sim_step()`. Sim reader trusts
  only `SIM_RESULT` for simulated outcomes. **Driver rule:** snapshots are
  released before every real `step()` (run_search does this) — a real game end
  with a live snapshot would park the engine in the terminal intercept.
- `train/opponents.py` — `SearchController` (advertises `wants_search_env`,
  receives the env via `bind_env`; falls back to the evaluator's raw policy at
  `safe=0` decisions; tallies `stats` = searched/fallback/sims/sim_steps).
  Spec grammar: **`mcts:<ckpt-or-deck>?sims=128&worlds=4&c=1.5&temp=0&vscale=1&seed=0`**
  (`mcts:uniform` = torch-free). Works everywhere controller specs work
  (`observe`, `run_match`, harness).
- `train/env.py` — BQUERY parsing extracted to `_parse_bquery_payload` (shared
  by both readers), `SEARCHINFO` capture (`env.last_search_safe`),
  `_extra_engine_flags` hook. Training path behavior-preserving.

**Verified in-container** (4-CPU cloud box, no GPU): all protocol smoke tests;
full `make check` including the new `snapshot` CI tier; search runs end-to-end
via `run_match` with both evaluators; PPOEvaluator exercised against a real
MaskablePPO. A tiny 60k-step PPO checkpoint was trained here purely to prove
the model path (`train/checkpoints/delver__final.zip` — NOT committed, and far
too weak for strength conclusions).

### Phase C — AlphaZero training loop (Python) — DONE and VERIFIED

All Phase C deliverables are on the branch (landed across the WIP commits
`d055313`/`66ac222`/`1af2cfd` + the gating fix `80f5dbe`):

- **`train/az_net.py`** — `AZNet(obs, mask) -> (logits, value)`, a plain torch
  module (no SB3 classes): shared `CardGameExtractor(per_action_head=True)`
  feature stack + `_ActionScorer` per-action policy head + tanh value head,
  with in-graph masking. TorchScript-scriptable; `--export-check` compares
  scripted vs eager. Checkpoints `train/checkpoints/az/{deck}__azfinal.pt`
  (the gate-promoted incumbent) / `{deck}__azv{steps}.pt` (candidate
  snapshots) + JSON layout-handshake meta; `from_ppo` warm-start handles both
  the per-action and stock MlpPolicy flavors (shape-checked, transfer
  reported). One necessary deviation from the plan's literal wording:
  `torch.jit.script` cannot compile `CardGameExtractor` (its forward slices
  with module-level int globals), so `AZNet` adopts a real extractor's
  submodules into a scriptable twin (`ScriptTrunk`, offsets baked as
  `__constants__`) with identical parameter names/shapes; drift is guarded by
  `verify_trunk_matches_extractor` (bit-identical assert, run in
  `--export-check`).
- **`train/az_selfplay.py`** — multiprocess mirror self-play: per searched
  (loop-safe, >1 choice) decision stores `(obs copy, visit-count pi, legal
  mask, mover seat)`; root Dirichlet (eps .25 / alpha 1.0), tau=1 for the
  first 20 moves then argmax; z backfilled ±1 per-mover (0 on draws); unsafe
  roots fall back to the net's argmax and are NOT stored. Shards
  `train/az_data/{deck}/shard_*.npz` (obs/pi/z/mask) — the exact format the
  Phase D C++ writer must reproduce.
- **`train/az_train.py`** — Adam (wd 1e-4) on
  `CE(pi, masked_log_softmax) + c_v*MSE(v,z)` over a last-M-shards window;
  saves CANDIDATE snapshots only. **`__azfinal` advances exclusively through
  the `az-eval` promotion gate** (candidate vs incumbent, seats alternating —
  half the games each way — promote at >= 0.55; vs `scripted` when no
  incumbent exists yet). The trainer resumes the candidate line (newest
  `__azv`, `resolve_az_checkpoint(prefer="snapshot")`); self-play and the
  `az:` opponent spec default to the incumbent.
- **Integration** — `az:<ckpt-or-deck>?sims=&worlds=&c=&temp=&seed=` and
  `azraw:` controller specs in `make_controller`; `train.py` subcommands
  `az-selfplay` / `az-train` / `az-eval` / `az` (one full
  generate→train→gate cycle).

**Verification record (this container, 4-core CPU):**
1. `make check` — full pass, 0 errors/warnings (snapshot + obsinv tiers green).
2. `az_net.py --export-check` — scripted==eager Δ=0, trunk-vs-extractor Δ=0.
3. `from_ppo` on a stock checkpoint: 16 trunk tensors transferred, heads fresh.
4. Gating semantics (post-fix `80f5dbe`): az-train resumed the candidate line
   (azv1920→azv3200→azv3840) writing no `__azfinal`; promotion produced a
   byte-identical incumbent copy (`cmp` verified); a failed gate (candidate
   azv3840 vs incumbent azv3200: 1W-1L @ 0.500 < 0.55, seats alternated) left
   the incumbent untouched (`cmp` verified again).
5. End-to-end `train.py az` cycle: self-play 2 games → 239 samples/2 shards
   (239 searched / 131 fallback, 1-1 A/B, no draws); train loss 1.88 → 1.52;
   gate ran az-vs-az (both seats searching) to completion.

Phase C has NOT yet been shown to *learn* (that needs real-hardware cycles at
real sims/games budgets) — the container verification is functional, not a
strength result.

## Phase B strength gate — PASSED (2026-07-11, real hardware)

Run on a 32-core Linux box, `make BUILD=RELEASE` (full `make check` pass first,
0 errors/warnings; snapshot perf probe 1.06 ms/pair). Checkpoint:
`train/checkpoints/league/bw_dnt__final.zip` (the league PPO generalist trained
2026-07-11; gitignored, lives on this machine only). Command:

```
train/.venv/bin/python train/eval_search_gate.py \
    --checkpoint league/bw_dnt --deck league/bw_dnt \
    --games 192 --sims 128 --worlds 4 --workers 31 --batch-size 8
```

(`--batch-size 8` only splits the 192 games into 24 batches so all 31 workers
are used; seeds stay disjoint per batch, seats split evenly. Default seed 1.)

- **GATE (mcts vs raw, same ckpt): 128W 64L 0D of 192 — winrate 66.7% → PASS
  (bar 55%)**, at the DEFAULT knobs `sims=128 worlds=4 c=1.5 vscale=1.0` —
  no `--vscale`/`--c`/`--sims` sweep was needed.
- Seat split: 62W-34L (64.6%) with MCTS in seat A, 66W-30L (68.8%) in seat B.
- **Safe-fraction 57.8%** for bw_dnt (12,594 searched roots / 9,193 fallbacks;
  1.61M sims, 6.07M sim steps). Zero draws, zero protocol errors.
- **scripted:hard references** (absolute anchor, same deck mirror):
  raw checkpoint 7W-25L (**21.9%**); mcts 6W-10L (**37.5%**) — search also
  lifts play vs the external opponent, but the bw_dnt PPO checkpoint itself
  is still well below scripted:hard.
- Throughput: ~13-14 min per 8-game batch per worker (~1.7 min/game at
  sims=128/worlds=4, 1 torch thread); the whole 192-game gate + reference
  batches finished in ~16 min wall on 31 workers.

Earlier in-container plumbing runs (4-core cloud box, debug build), kept for
history: throwaway 60k-step checkpoint 6W-6L over 12 games at sims=48/worlds=3;
`league/ur_delver__final.zip` (branch `delver_checkpoint_temp`) first 24-game
batch at sims=128/worlds=4 was 11W-13L (45.8%), 63% safe fraction — stopped at
n=24 (statistically meaningless; box too slow for 200 games).

## What is IN FLIGHT / handed off

- **Phase D (C++ libtorch actor)**: not started; outline below.

## What to run on real hardware

Environment: python venv per `train/` docs + `pip install torch stable_baselines3
sb3-contrib tensorboard`. Build: `make BUILD=RELEASE` for speed (the snapshot
CI perf probe expects release-build timing). Gate everything with `make check`.

1. **Checkpoint**: a trained `league/ur_delver` MaskablePPO checkpoint lives on
   branch **`delver_checkpoint_temp`** (`train/checkpoints/league/
   ur_delver__final.zip`, ~28 MB — checkpoints are gitignored on the feature
   branch itself). Pull it with
   `git fetch origin delver_checkpoint_temp && git checkout
   origin/delver_checkpoint_temp -- train/checkpoints/league/ur_delver__final.zip`
   (then `git restore --staged` it). Or train a fresh baseline:
   `train.py --deck <deck> --opponent <deck>` (consider
   `ROBOMAGE_PER_ACTION_HEAD=1` — content-based priors search better than the
   positional head, and it's what AZNet mirrors).
2. **The Phase B gate** — DONE 2026-07-11 (see "Phase B strength gate —
   PASSED" above; run with `league/bw_dnt` instead of ur_delver). Kept as a
   recipe for gating other decks/checkpoints — one command, committed on the
   branch:
   ```
   train/.venv/bin/python train/eval_search_gate.py \
       --checkpoint league/ur_delver --deck league/ur_delver \
       --games 192 --sims 128 --worlds 4 --workers <cores-1>
   ```
   Seats alternate by batch with disjoint seed ranges; prints per-batch
   results, a running total, `scripted:hard` reference batches, and a final
   PASS/FAIL against the **≥55%** bar. Knobs if it underperforms: `--vscale`
   first (the PPO value head is trained on shaped returns — miscalibration is
   the classic cause of low-sim search underperforming; try 1–3), then `--c`
   (prior-heavier 0.8–1.0), then `--sims` 64→256 / `--worlds` 2→8. Check the
   printed **safe-fraction** per deck (ur_delver measured 63%): a low value
   caps attainable lift because unsafe roots fall back to the raw policy.
3. **Phase C**: `train.py az --deck <deck>` cycles (selfplay → train → gate)
   on a mirror, sims 100-200, worlds 3-5; watch the `az-eval` promotion rate
   and the losses in `checkpoints/az/{deck}_az_train.log`. First cycle
   warm-starts from the deck's PPO checkpoint automatically; `__azfinal` only
   moves when a candidate clears the gate.
4. **Throughput expectation**: search cost ≈ sims × (restore ~0.3-1 ms +
   path-replay ~0.24 ms/decision + net eval). The container measured ~24
   ms/sim (debug build, contended 4-core CPU) → ~2.4 min/game at
   sims=128/worlds=4; release + real cores should be several× faster, but
   Python self-play is still ~10-50× slower per game than PPO rollouts —
   that's the Phase D motivation.

### Suggested prompt for the Phase B verification session

> On branch `claude/alphazero-mcts-plan-dwcubj`, complete the Phase B
> verification per docs/alphazero_status.md. Build with `make BUILD=RELEASE`
> and confirm `make check` passes. Pull the ur_delver checkpoint from branch
> `delver_checkpoint_temp` (do not commit it). Run the gate:
> `train/eval_search_gate.py --checkpoint league/ur_delver --deck
> league/ur_delver --games 192 --sims 128 --worlds 4` with `--workers` set to
> about one less than the machine's cores. If the final win rate is below
> 55%, sweep `--vscale 1 2 3` at 48 games each, take the best, and rerun the
> full 192-game gate with it (then `--c 1.0` and `--sims 256` if still
> short). Report: final W/L/D + win rate, the PASS/FAIL line, safe-fraction,
> the scripted:hard reference win rates for both sides, and the winning knob
> settings. Record the results in the "Phase B strength gate" section of
> docs/alphazero_status.md, commit, and push to the same branch. Do not
> start Phase D.

## Phase D outline (C++ libtorch self-play actor) — NOT STARTED

Design pinned by the approved plan; build only after Phase C shows learning
on real hardware:

- **Build**: optional `make LIBTORCH=1 actor` target linking the existing
  engine objects + new `src/actor/az_actor_main.cpp`. `LIBTORCH_DIR` make
  variable; when absent the target is skipped and default `make`/`make check`
  are untouched (libtorch is ~1-2 GB, stays optional). Actor TUs are compiled
  **with** exceptions (libtorch requires them); engine objects stay
  `-fno-exceptions`; all torch calls wrapped catch-at-boundary.
- **Model**: TorchScript export from `az_net.py` (`forward(obs, mask) ->
  (logits, value)`, masking in-graph — already implemented and export-checked)
  loaded via `torch::jit::load`. The network stays defined once, in Python.
- **Search**: in-process MCTS reusing the Phase A `snapshot.h` API directly
  (no stdio) — `snapshot_save/restore`, `determinize_hidden_state`, direct
  action stepping. Virtual loss + batched leaf evaluation (collect 8-32
  leaves per forward call — the main net-side throughput win). One game per
  process (the engine has global state); parallelism = N actor processes.
- **Output**: shard writer emitting the exact Phase C `.npz` format
  (obs/pi/z/mask) into `train/az_data/{deck}/` — the Python trainer must not
  be able to tell C++ shards from Python ones. (If .npz writing from C++ is
  awkward, a raw `.bin` + tiny Python converter is acceptable; byte-identical
  training inputs is the invariant.)
- **Verification**: (a) export unit test scripted==eager (exists:
  `az_net.py --export-check`); (b) actor-generated shards train identically
  to Python-generated shards (same loss trajectory on a fixed seed);
  (c) throughput benchmark vs the Phase C Python generator (expected
  10-100× more games/hour/core from removing stdio + Python overhead and
  batching leaf evals).

## Notes, quirks, known limitations

- The engine's `SEARCHINFO safe=` split means combat SELECT prompts are
  searchable roots but target sub-prompts fall back to the raw policy at the
  ROOT only (inside simulations they're ordinary tree nodes). `SearchController.stats`
  reports the searched/fallback ratio — check it per deck.
- DETERMINIZE pins both players' known-top-library prefixes and known
  (revealed) opponent hand cards; the opponent's unknown hand + unpinned
  library form one exchange pool. **Post-board (bo3 game 2+) the opponent's
  SIDEBOARD cards join that pool** (their sideboard swaps are hidden — P only
  knows the 75, so sampled worlds re-deal which cards are in the deck; a
  declared companion is public and stays pinned; game 1 models the known
  pre-board 60). Covered by the `sideboard_determinize` CI test. Mild accepted
  approximations are documented in `src/search_server.cpp`.
- Snapshot slots are wiped at each game start (`snapshot_release_all` in
  `play_single_game`) — no cross-game reuse; bo3 sideboard decisions are unsafe
  roots (searchable later if wanted).
- `train/checkpoints/` and `train/az_data/` are gitignored artifacts; move
  checkpoints between machines out-of-band.
- The stock `delver__final.zip` here was trained WITHOUT the per-action head;
  PPOEvaluator works with both flavors.

## Plan summary (agreed decisions)

Determinized MCTS (K-world root sampling, no oracle search) · phased rollout
A→B→C→D with the PPO path kept working throughout · in-process C++ snapshots
over stdio protocol (done) · Python search first, C++ libtorch actor (Phase D)
only after Phase C shows learning; the network stays defined once in Python
and reaches C++ as TorchScript. Full risk table and phase details live in the
session plan; this file is the working summary.
