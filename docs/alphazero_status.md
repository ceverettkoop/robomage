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

- **Phase D (C++ libtorch actor)**: DONE and verified (see "Phase D — DONE"
  below). The actor, its self-play + shard writer, the opt-in `actor` CI tier,
  the throughput bench, and the `--actor`/AUTO backend wiring in
  `az-selfplay`/`az`/`az-league` all landed on the branch.
- **Analysis-tool integration (M10)**: DONE — `analysis.py` accepts AZ checkpoint
  specs and gained a `search` (search-vs-raw) subcommand; see
  "Analysis-tool integration" below.
- **Still open (real-hardware, not code):**
  - Phase C **learning at scale** — the container/32-core runs proved the loop is
    functional and that search beats the raw policy (Phase B gate), but no run has
    yet demonstrated the AZ loop *improving* a checkpoint across many
    generate→train→gate cycles at real sims/games budgets. The Phase D actor now
    removes the stdio + Python self-play bottleneck that made this expensive.
  - **Multi-process actor fleets** — one game per process (the engine has global
    state), so throughput scales by running N `bin/az_actor` processes; a fleet
    launcher / shard-dir coordinator across machines is not written.
  - M10 future-work items in "Analysis-tool integration".

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

## Phase D — DONE (2026-07-11/12): C++ libtorch self-play actor

The in-process AlphaZero actor is built and verified. It runs full games and
determinized MCTS entirely inside one process — no stdio BQUERY round-trip, no
Python search loop — and writes the exact Phase C shard format the trainer
consumes. Default `make` / `make check` are untouched: the actor is a separate,
opt-in target and its CI tier self-skips when the binary isn't built.

**What was built:**

- **`src/game_driver.{h,cpp}` refactor** — the globals, ECS setup (`init_ecs`),
  and per-game loop (`play_single_game` / `run_sideboard_phase`) were lifted out
  of `main.cpp` so a second binary can link every engine object **except**
  `obj/main.o` and still drive games in-process. `main.cpp` now just calls into
  it; engine behavior is unchanged.
- **Default-null engine hooks** — two hooks let the actor intercept the engine
  loop without a protocol, both no-ops when unset (so `main.cpp` / stdio machine
  mode are byte-identical when they're absent):
  `InputLogger::set_input_provider(std::function<int(const std::vector<LegalAction>&)>)`
  (machine-mode decision dispatch takes the provider instead of reading stdin
  when one is installed), and `search_set_game_end_hook` /
  `search_clear_game_end_hook` (a simulated line's game-over is handed to the hook
  — which records the result and latches a RESTORE — instead of the stdio
  `SIM_RESULT`/stdin branch).
- **`bin/az_actor` via `make actor`** — links the engine objects (minus
  `main.o`) + `src/actor/*.cpp` + libtorch. `LIBTORCH_DIR` **auto-detects** the
  venv's torch (`train/.venv/.../site-packages/torch`); override with
  `make actor LIBTORCH_DIR=/path`. Actor TUs compile **with** exceptions
  (libtorch needs them) while engine objects stay `-fno-exceptions`; the target
  errors early with a clear message if torch can't be found, and is never part of
  the default build (libtorch is ~1-2 GB).
- **Bit-exact obs builder** (`src/actor/obs_builder.{h,cpp}`) — reconstructs the
  machine-mode observation vector in C++, proven bit-identical to the engine's
  own serializer.
- **MCTS state machine** (`src/actor/az_mcts.{h,cpp}`) — in-process determinized
  PUCT reusing the Phase A `snapshot` API directly (snapshot/restore/determinize
  + direct action stepping), with **exact parity to `mcts.py`** (same world
  seeding, same per-mover backup, same selection). **Batched leaf evaluation +
  virtual loss**: leaves are collected and evaluated in one TorchScript forward
  (the net-side throughput win), with virtual loss keeping parallel descents
  diverse.
- **TorchScript net** — the model stays defined once in `az_net.py`; the actor
  loads the `.ts.pt` export (`forward(obs, mask) -> (logits, value)`, masking
  in-graph) via `torch::jit::load` and checks the `OBS_SIZE`/`MAX_ACTIONS`/
  `N_CARD_TYPES` handshake meta before running.
- **Self-play + uncompressed-npz shards** (`src/actor/npz_writer.{h,cpp}`) —
  `bin/az_actor --selfplay` plays games, stores `(obs, pi, z, mask)` per searched
  root, and writes `train/az_data/{deck}/shard_*.npz` in the **exact** Phase C
  format — the Python trainer's `load_window` cannot tell a C++ shard from a
  Python one.
- **Opt-in `ci_check` actor tier** — `train/ci_check.py --tier actor` runs
  `test_actor_parity.py` (obs bit-parity), `test_mcts_parity.py` (visit-count
  parity), and `test_actor_shards.py` (trainer ingest). It is **not** in the
  default `make check` run and self-skips with a message when `bin/az_actor`
  isn't built or torch is unavailable.
- **Backend AUTO wiring** — `az-selfplay` / `az` / `az-league` take
  `--actor` / `--no-actor`; the default (neither) is **AUTO**: use `bin/az_actor`
  iff it is built, else the pure-Python multiprocess backend. `az-league`
  rotates `az` cycles over the `decks/league/` roster and resumes from
  `checkpoints/_az_league_progress.json`.
- **Throughput bench** — `train/bench_actor.py` times the C++ actor
  (`bin/az_actor --selfplay`) against the single-worker Python `az_selfplay` on
  the **same** deterministic net (torch seed 0, exported to `.ts.pt`), same deck,
  same sims/worlds, both single-thread (`set_num_threads(1)`).

**Verification chain (all green on this branch):**

1. **Obs bit-parity** — `test_actor_parity.py`: the C++ obs builder reproduces
   the engine's observation vector **bit-exact over 226 decisions**
   (`league/ur_delver`, seed 1, `OBS_SIZE=6700`).
2. **MCTS visit parity** — `test_mcts_parity.py`: C++ vs `mcts.py` visit counts
   **exact over 271 searched roots** (4336 total root visits; sims=16 worlds=2
   c=1.5, batch=1). Batched search (batch=16) is a separate line — it agrees on
   argmax at 11/12 comparable roots before the two RNG streams diverge, as
   expected once batching reorders leaf evaluation.
3. **Shards trainer-interchangeable** — `test_actor_shards.py`: a C++
   `--selfplay` run's shard has the exact schema, sample count matches the
   per-game tallies, and the Python trainer's `load_window` ingests it.
4. **`make check` green throughout** — the C++ refactor + hooks kept every
   default tier passing (`make check BUILD=RELEASE`: 0 errors/0 warnings,
   snapshot + obsinv tiers included).
5. **Bench** — the C++ actor is ~**1.2-1.5× faster per game single-thread** than
   the Python reference. The honest read: at real sims/worlds **both legs are
   dominated by the identical TorchScript forward passes**, so the actor's win is
   the engine/search overhead it removes (stdio framing, Python tree bookkeeping),
   not a network speedup — the large multiplier comes from **running many actor
   processes**, not from a single process being an order of magnitude faster.

**How to run everything:**

```bash
# Build the actor (auto-detects venv libtorch; default make/make check untouched).
make actor
# Export a net for the actor (once per checkpoint) and gate the actor path.
train/.venv/bin/python train/az_net.py --export <deck>       # writes {deck}__azfinal.ts.pt
train/.venv/bin/python train/ci_check.py --tier actor        # obs/MCTS/shard parity
# Self-play with the AUTO backend (uses bin/az_actor iff built):
train/.venv/bin/python train/train.py az-selfplay --deck <deck> --games 50 --sims 128 --worlds 4
# Full AZ league (rotate self-play -> train -> gate over decks/league/):
train/.venv/bin/python train/train.py az-league                 # --resume to continue
# Throughput bench (C++ actor vs single-worker Python):
train/.venv/bin/python train/bench_actor.py --games 2 --sims 128 --worlds 4
```

## Analysis-tool integration (M10)

The model-analysis tools (`train/analysis.py` + its shared CLI in
`train/cli_spec.py`) now understand AZ checkpoints and search play. Two additive
changes; no surgery on the analysis internals.

**AZ checkpoints as the model argument.** `analysis.py report` / `interactive`
accept an `az:<deck-or-.pt>` spec (or a bare `.pt` path, or a deck shorthand that
resolves to an AZ checkpoint via `resolve_az_checkpoint`). A thin adapter
(`_AZModelAdapter`) wraps the `AZNet` and exposes exactly the three surfaces the
analysis code touches — `policy.predict_values`, `policy.get_distribution`, and
`model.predict` (for `ModelController`) — so the **entire existing battery**
runs unchanged on an AZ checkpoint: card importance, targeting, value
calibration, turning points, trajectory archetypes, value swings, policy regret,
policy entropy, decision consistency, the SHAP surrogate, and the interactive
`whatif` / `run` counterfactuals.

*What transfers, and why:* analysis only ever reads two model outputs — the
state value V(s) and the masked action distribution — and an `AZNet` natively
provides both (a tanh value head and a masked-softmax policy). SHAP fits a
surrogate regressor on the collected traces, so it needs no PPO internals either.
**No analysis fundamentally requires PPO-only machinery, so none had to be
disabled** — instead the report prints a one-line banner noting the one real
difference: the AZ V(s) is a **bounded game-outcome estimate in [-1, 1]** (tanh),
not the PPO shaped-return critic, so absolute value magnitudes aren't directly
comparable across the two kinds of report (the shapes/rankings/calibration all
still hold). If a future AZ checkpoint truly couldn't supply one of those
outputs the adapter degrades rather than crashing.

**`analysis.py search` — search-vs-raw comparison.** A new subcommand drives N
games with an MCTS controller (AZ **or** PPO evaluator) and, per searched
(loop-safe) root, records the net's priors + leaf value against MCTS's visit
distribution + root value, then reports:

- **mean KL(priors ‖ visits)** (median + max) — how far search moved off the
  prior;
- **argmax agreement** — how often the net's greedy move equals search's pick;
- **value MAE + correlation** — how well the net's leaf value tracks the search
  root value;
- the **biggest-disagreement** roots, decoded (turn/step/life, the net-greedy
  action vs the search pick with their prior/visit mass and both values).

It reuses the Phase B/C `SearchController` + `run_search` and the existing decode
helpers; output is a terminal summary (headless-safe, matching the tool's
conventions). Example:

```bash
train/.venv/bin/python train/analysis.py search az:league/ur_delver \
    --deck-a league/ur_delver --deck-b league/ur_delver \
    --n-games 4 --sims 64 --worlds 4 --top 8
```

Both are surfaced in the TUI for free: the `search` subcommand is declared in
`cli_spec.ANALYSIS_TOOL`, which `tui.py` renders generically, so it appears as a
capture-mode form with no extra UI code. (`tui_analysis.py` — the full-screen
board-state browser — was left alone: its per-decision replay pager is a
different surface from a batch comparison report, so wiring `search` into it
would be real UI work, deliberately out of M10 scope.)

*Future work (documented, not built):* (a) the search-compare tool fixes the
model to seat A each game — alternating seats would remove any first-player skew
in the aggregate stats; (b) it drives real games rather than replaying a fixed
trace, so two evaluators aren't compared on the *same* states — a "compare two
checkpoints' priors/values on one recorded game" mode would need the analysis
collector to store per-decision masks (a small trace-schema addition); (c)
optional charts (KL histogram, value scatter) via `viz.py` were skipped since the
terminal summary suffices.

## Original Phase D outline (for reference)

Design pinned by the approved plan; **implemented** as recorded above. Kept here
as the design intent it was built against:

- **Build**: optional `make actor` target linking the existing engine objects +
  `src/actor/az_actor_main.cpp`. `LIBTORCH_DIR` make variable; when absent the
  target is skipped and default `make`/`make check` are untouched. Actor TUs
  compiled **with** exceptions; engine objects stay `-fno-exceptions`.
- **Model**: TorchScript export from `az_net.py` (`forward(obs, mask) ->
  (logits, value)`, masking in-graph) loaded via `torch::jit::load`. The network
  stays defined once, in Python.
- **Search**: in-process MCTS reusing the Phase A `snapshot.h` API directly (no
  stdio). Virtual loss + batched leaf evaluation (collect 8-32 leaves per forward
  call). One game per process; parallelism = N actor processes.
- **Output**: shard writer emitting the exact Phase C `.npz` format
  (obs/pi/z/mask) into `train/az_data/{deck}/` — byte-identical training inputs
  is the invariant.
- **Verification**: (a) export unit test scripted==eager (`az_net.py
  --export-check`); (b) actor-generated shards train identically to
  Python-generated shards; (c) throughput benchmark vs the Phase C Python
  generator.

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
