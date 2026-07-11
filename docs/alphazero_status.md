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

## What is IN FLIGHT (not yet committed)

- **Phase B strength eval**: `mcts:delver?sims=48&worlds=3` vs the same
  checkpoint raw, 6 games per seat + a scripted:easy reference, was running
  when this doc was written. Whatever it shows on this box is a plumbing
  result, not a strength result — the checkpoint is undertrained.
- **Phase C (AlphaZero training loop)**: being implemented (az_net.py partially
  written at time of writing). Deliverables per plan: `train/az_net.py`
  (`AZNet` plain torch module: CardGameExtractor(per_action_head) trunk +
  per-action policy head + tanh value head; TorchScript `forward(obs, mask) ->
  (logits, value)` with an `--export-check`; checkpoint format
  `train/checkpoints/az/{deck}__azfinal.pt` + JSON layout-handshake meta;
  `from_ppo` warm start), `train/az_selfplay.py` (search-driven mirror
  self-play → `train/az_data/{deck}/shard_*.npz` with obs/pi/z/mask arrays),
  `train/az_train.py` (CE-on-visit-counts + MSE-on-outcome trainer, `az-eval`
  ≥55% promotion gate), `az:`/`azraw:` controller specs, `train.py`
  subcommands `az-selfplay`/`az-train`/`az-eval`/`az`.
- **Phase D (C++ libtorch actor)**: not started. Design pinned in the plan:
  optional `make LIBTORCH=1 actor` target, actor TUs compiled WITH exceptions,
  TorchScript net loaded via `torch::jit::load`, in-process MCTS on the
  `snapshot.h` API, shard writer matching Phase C's format.

## What to run on real hardware

Environment: python venv per `train/` docs + `pip install torch stable_baselines3
sb3-contrib tensorboard`. Build: `make BUILD=RELEASE` for speed (the snapshot
CI perf probe expects release-build timing). Gate everything with `make check`.

1. **Train a real PPO baseline** (if none exists):
   `train/.venv/bin/python train/train.py --deck delver --opponent delver`
   (2M steps default; consider `ROBOMAGE_PER_ACTION_HEAD=1` — the per-action
   policy head gives content-based priors that should search better than the
   positional head, and is what AZNet mirrors).
2. **The Phase B gate** — same checkpoint, search vs raw, seats alternating:
   - `run_match("mcts:delver?sims=128&worlds=4", "delver", deck_a="delver",
     deck_b="delver", games=100, seed=1)` and the seat-swapped mirror batch.
   - Success: ≥55% for the search side over ~200 games. Knobs if it
     underperforms: `sims` (64→256), `worlds` (2→8), `c` (0.8–2.5),
     `vscale` (match the reward scale the value head was trained on; try 1–3).
   - Also run vs `scripted:hard` for an absolute reference.
3. **Phase C once committed**: `train.py az` cycles (selfplay → train → gate)
   on delver mirror, sims 100-200, worlds 3-5; watch `az-eval` promotion rate
   and the policy/value losses in `checkpoints/az/delver_az_train.log`.
4. **Throughput expectation**: search cost ≈ sims × (restore ~0.3-1 ms +
   path-replay ~0.24 ms/decision + net eval). On this 4-core box, 48 sims × 3
   worlds ≈ 1-2 s/decision debug-build; release + real cores will be several×
   faster, but Python self-play is still ~10-50× slower per game than PPO
   rollouts — that's the Phase D motivation.

## Notes, quirks, known limitations

- The engine's `SEARCHINFO safe=` split means combat SELECT prompts are
  searchable roots but target sub-prompts fall back to the raw policy at the
  ROOT only (inside simulations they're ordinary tree nodes). `SearchController.stats`
  reports the searched/fallback ratio — check it per deck.
- DETERMINIZE pins both players' known-top-library prefixes and known
  (revealed) opponent hand cards; the opponent's unknown hand + unpinned
  library form one exchange pool. Mild accepted approximations are documented
  in `src/search_server.cpp`.
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
