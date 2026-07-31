# AlphaZero MCTS — implementation status & handoff

Branch: `claude/alphazero-mcts-plan-dwcubj`. This documents what has landed, how to
drive it, and exactly what to run on real hardware. The full design plan (four
phases, decisions, risks) is summarized at the bottom.

> **Update (generalist collapse, 2026-07).** AZ has since collapsed to **one
> generalist net** (`checkpoints/az/gen__azfinal.pt` + `gen__azv{steps}.pt`), the
> counterpart of the single PPO `gen__final.zip`. The deck the net pilots is now an
> explicit observation input (self live-library + opponent static-decklist blocks;
> for the current `STATE_SIZE`/`OBS_SIZE` read `src/machine_io.h` and
> `train/env.py` — they drift with every layout change), not a per-checkpoint
> identity — so self-play
> is **mirrors + cross-deck** (a focus deck vs a mirror with `--mirror-frac`,
> default 0.25, else a uniform roster opponent), shards pool into **`az_data/gen/`**,
> and the promotion gate is an **aggregate win-rate over a ROSTER-WIDE panel**
> (a candidate-vs-incumbent mirror for every roster deck + direction-balanced
> cross pairs, per-matchup and per-piloted-deck breakdowns printed) with a
> **per-deck floor veto** (`--gate-floor`, default 0.2 at >=4 matches — a
> candidate that collapsed at piloting any one deck is rejected even when its
> aggregate clears 0.55). `az-league --matrix` replaces the one-deck-per-slot
> focus rotation with the whole-roster focus matrix every slot (stationary
> training window; no per-deck forgetting sweep). `--expert-decks <decks>`
> (az / az-league; standalone via `az-selfplay --expert`) additionally writes
> **expert demonstration shards** each cycle — scripted:hard piloting both
> seats, `pi` = one-hot on the expert's action, same shard schema — so the
> trainer behavior-clones hand-coded combo lines (the doomsday fix: the
> warm-started value net scores mid-combo states as lost, so search prunes the
> line before sampling it; demonstrations put prior mass on the line and price
> those states by games the combo wins). Controller specs are `az:gen` /
> `azraw:gen` / `mcts:gen` (a bare deck shorthand is rejected; the deck travels as a
> separate explicit parameter). The per-deck `{deck}__az*` naming, mirror-only
> self-play, and `az_data/{deck}/` sharding described below are historical; commit
> hashes / run logs are kept as records.

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

Every query is preceded by `SEARCHINFO safe=<0|1>`. Since the snapshot-safe
conversion project (branch `snapshot_safe`, Batches 0-14, completed 2026-07-18)
**every decision kind is loop-safe**: priority, combat declarations and their
target sub-prompts, damage assignment, every resolution-time effect prompt
(targets/digs/scrys/searches/unless-costs/...), trigger placement, SBE prompts
(legend rule, ETB choices, enters-tapped-unless-life), cast/activation flows
(modes, targets, X, delve, alt costs), mulligans/pregame, the turn-draw dredge
replacement, and bo3 sideboarding. Measured scripted-vs-scripted over the nine
replay-corpus pairings (delver/doomsday/mav, seed 1): **2844/2844 decisions
safe=1 (100.0%)**; the delver control line in `test_snapshot.py` opens safe=1
at decision 0 and stays safe for all 148 decisions; a league probe
(wrb_energy/bw_dnt/ur_delver/gw_maverick/bug pairings, 3 seeds each, 1550
decisions) is also 100% safe. The documented residual `safe=0` surface (all
blocking-by-design, quantified in `test_snapshot.py`'s `PROMPT_SITE_WHITELIST`
audit):

- the 616.1 multi-replacement `choose_one` prompt (needs >= 2 simultaneous
  replacement effects on one event — e.g. two Leylines of the Void in play;
  0 occurrences in every measured run; counted at runtime by
  `replacement::choose_one_prompt_count`);
- interactive-only prompts machine mode never reaches (interactive mana
  payment, interactive hybrid pips);
- the blocking-shim fallbacks for prompts fired outside the main loop
  (defensive; believed unreachable since the pregame gate) and
  `effect_choose_card`'s cast-from-exile mini-cast announce path.

(snapshot at a `safe=0` decision is a fatal error — but simulations traverse
them freely; safety only gates the root).
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
  scripted vs eager. Checkpoints `train/checkpoints/az/gen__azfinal.pt`
  (the one generalist, gate-promoted incumbent) / `gen__azv{steps}.pt` (candidate
  snapshots) + JSON layout-handshake meta; `from_ppo` warm-start handles both
  the per-action and stock MlpPolicy flavors (shape-checked, transfer
  reported). One necessary deviation from the plan's literal wording:
  `torch.jit.script` cannot compile `CardGameExtractor` (its forward slices
  with module-level int globals), so `AZNet` adopts a real extractor's
  submodules into a scriptable twin (`ScriptTrunk`, offsets baked as
  `__constants__`) with identical parameter names/shapes; drift is guarded by
  `verify_trunk_matches_extractor` (bit-identical assert, run in
  `--export-check`).
- **`train/az_selfplay.py`** — multiprocess self-play, mirrors + cross-deck (a
  focus deck vs a mirror with `--mirror-frac`, else a uniform roster opponent):
  per searched (loop-safe, >1 choice) decision stores `(obs copy, visit-count pi,
  legal mask, mover seat)`; root Dirichlet (eps .25 / alpha 1.0), tau=1 for the
  first 20 moves then argmax; z backfilled ±1 per-mover (0 on draws); unsafe
  roots fall back to the net's argmax and are NOT stored (post snapshot_safe
  effectively never — every decision kind is a safe root). Shards pool into
  `train/az_data/gen/shard_*.npz` (obs/pi/z/mask) — the exact format the
  Phase D C++ writer reproduces.
- **`train/az_train.py`** — Adam (wd 1e-4) on
  `CE(pi, masked_log_softmax) + c_v*MSE(v,z)` over a last-M-shards window from the
  pooled `az_data/gen/` dir; saves CANDIDATE snapshots only. **`gen__azfinal`
  advances exclusively through the `az-eval` promotion gate** (candidate vs
  incumbent over a ROSTER-WIDE panel — a mirror per roster deck plus
  direction-balanced cross pairs, seats alternating — promote on aggregate
  win-rate >= 0.55 AND no per-deck floor veto (`--gate-floor`), with per-matchup
  and per-piloted-deck breakdowns persisted to the az-league sidecar;
  vs `scripted` when no incumbent exists yet). The trainer resumes the candidate
  line (newest `gen__azv`, `resolve_az_checkpoint(prefer="snapshot")`); self-play
  and the `az:` opponent spec default to the incumbent.
- **Integration** — `az:<gen-or-ckpt>?sims=&worlds=&c=&temp=&seed=` and
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
   `train.py --deck <deck> --opponent <deck>` (fresh models default to the
   per-action head — content-based priors search better than the positional
   head, and it's what AZNet mirrors; `--stock-head` opts out).
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
   printed **safe-fraction** per deck (ur_delver measured 63% BEFORE the
   snapshot-safe conversion; ~100% since — see the safe-window section): a low
   value caps attainable lift because unsafe roots fall back to the raw policy.
3. **Phase C**: `train.py az --deck <deck>` cycles (selfplay → train → gate),
   mirrors + cross-deck (`--mirror-frac`), sims 100-200, worlds 3-5; watch the
   `az-eval` promotion rate and the losses in `checkpoints/az/gen_az_train.log`.
   First cycle warm-starts from the generalist PPO checkpoint (`gen__final.zip`)
   automatically; `gen__azfinal` only moves when a candidate clears the aggregate
   gate.
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
  root, and writes shards into the pooled `train/az_data/gen/shard_*.npz` in the
  **exact** Phase C format — the Python trainer's `load_window` cannot tell a C++
  shard from a Python one.
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
   (`league/ur_delver`, seed 1, at the `OBS_SIZE` of that run — 6922; the obs has
   widened since, and the test reads the size from `env.py` rather than pinning it).
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
train/.venv/bin/python train/az_net.py --export gen         # writes gen__azfinal.ts.pt
train/.venv/bin/python train/ci_check.py --tier actor        # obs/MCTS/shard parity
# Self-play with the AUTO backend (uses bin/az_actor iff built):
train/.venv/bin/python train/train.py az-selfplay --deck <deck> --games 50 --sims 128 --worlds 4
# Full AZ league (rotate self-play -> train -> gate over decks/league/):
train/.venv/bin/python train/train.py az-league                 # --resume to continue
# Throughput bench (C++ actor vs single-worker Python):
train/.venv/bin/python train/bench_actor.py --games 2 --sims 128 --worlds 4
```

## Phase 2 — learned sideboarding — DONE (2026-07-13)

**What it is.** In a best-of-three match the between-game sideboard decision (which
cards to swap between main deck and sideboard) is no longer a blind heuristic: it is
a **searched MCTS root** evaluated on the **next game's** horizon, and the choice is
persisted as a training sample whose value target is the result of that next game
(next-game `z`). The generalist therefore *learns to sideboard* from the same
self-play loop that teaches it to play — both backends (pure-Python and the C++
`bin/az_actor`) generate these samples, bit-for-bit compatibly.

**Mechanism in brief.**

- **MATCH-scoped snapshot across `init_ecs()`** — a sideboard prompt sits *between*
  two games, i.e. across a fresh ECS. A match-scoped snapshot slot survives the
  per-game `snapshot_release_all`, so the sideboard root can be snapshotted, searched
  (with simulations that roll into the *next* game), and restored, all without
  corrupting the real match state. The bo3 match loop is resumable and the sideboard
  prompt is a **loop-safe** searchable root (a sim's unwind re-enters the dispatcher
  top rather than double-counting the game).
- **Seed-salt determinization** — a sideboard root's determinized worlds are salted
  by the *next* game's seed, so the K sampled worlds vary correctly game-to-game
  (covered by the snapshot-tier world-variation test).
- **Sideboard-root search budget** — the sideboard root gets its own heavier, deeper
  budget: `sb_sims=256` / `sb_worlds=4` / `sb_max_depth=200` / `sb_rollout_turns=12`
  / `sb_persist=1` (the `DEFAULT_SB_*` constants in `cli_spec.py`; the horizon is
  game-long, so it is deeper than an in-game decision, leaf rollouts — see the
  2026-07-31 update below — carry each sim to turn 12 of the sampled next game, and
  boundary persistence shares one set of trees + a rollout memo across a boundary's
  picks). Inert for bo1.
- **Actor parity + next-game flush** — the C++ actor mirrors the Python match loop
  exactly: `--bo3` treats each `--games` unit as a match (match seeds spaced by 3),
  it searches the sideboard root with the same `--sb-sims`/`--sb-worlds`/
  `--sb-max-depth`/`--sb-rollout-turns` budget, and it **flushes** each game's samples priced by that
  game's winner while sideboard samples recorded between a game's backfill and the
  next game's start stay buffered and are priced by the NEXT game (matching Python's
  `game_idx == k+1` sideboard samples). All obs + MCTS visits stay bit-exact with the
  Python reference.

**New CLI knobs** (on `az-selfplay` where bo3 applies, and `az` / `az-league` /
`az-eval`, all bo3 by default with `--bo1` to opt out):

```
--sb-sims N          PUCT sims at a bo3 sideboard root (default 256; was 32, then 128)
--sb-worlds N        determinized worlds at a bo3 sideboard root (default 4)
--sb-max-depth N     descent depth cap at a bo3 sideboard root (default 200)
--sb-rollout-turns N leaf-rollout horizon in player turns of the next game
                     (default 12; 0 = off — see the 2026-07-31 update below)
--sb-persist 0|1     persist trees + rollout memo across a sideboard boundary
                     (default 1 — see the boundary-persistence update below)
```

> **Update (2026-07-31, later): sideboard-boundary persistence + rollout
> memo.** A boundary is one seat's contiguous run of ~10-20 pick decisions,
> and leaf rollouts multiplied their cost — each pick was a fresh full-budget
> search re-simulating futures the previous pick already explored. With
> `--sb-persist 1` (default) the per-world trees now SURVIVE across a
> boundary's picks: world seeds are pinned to the boundary's first searched
> root (mandatory — at a sb root the seed IS the sampled next-game deal), each
> next pick re-roots every world at the actually-played action's child
> (walking searched picks, the finalize-chosen action, and nc==1 forced
> actions alike; a failed walk re-roots that world fresh, never killing the
> boundary), the re-rooted node's P is overwritten with the fresh clean/noised
> root priors, and the search only tops up to `sb_sims` CUMULATIVE root visits
> (`sims_run` counts new sims; the training pi target keeps the inherited
> mass — deliberately, it is real signal). A per-boundary **rollout memo**
> additionally keys finished rollouts by (pinned world seed, leaf seat, sorted
> multiset of (cat, card id, seat) picks since the boundary root), so permuted
> pick orders reaching the same net configuration reuse one playout. The memo
> is a deliberate approximation: engine deck-vector order is mutation-history
> dependent, so a hit substitutes the first-seen ordering's value (a
> different-but-equally-random deal / completion of the same cards); paths
> containing a takeback are never memoized (direction locks genuinely
> diverge). Boundary identity = `mcts.sb_root_key` (seat + upcoming game
> number); all shared rules live in `train/mcts.py` (`world_seeds_for`,
> `sb_root_key`, `sb_pick_descriptor`, `walk_reuse_root`, `rollout_memo_*`)
> and are mirrored bit-exactly in `az_mcts.cpp`. `test_mcts_parity.py` runs
> the sb gate both ways (`bo3-sb-persist` + the persist-off baseline), all
> bit-exact. Out of scope (fresh searches): the analysis window /
> `IncrementalSearch`, `procs>1` mirror-pool search, `run_search_parallel`.
>
> Measured on one full boundary (ur_delver vs gw_maverick, 256/4/200/12,
> PPO-warm-start evaluator, Python path): fresh per-pick searches 19 picks at
> **79.3 s/pick** (4864 sims, 563k engine steps); persistence **44.0 s/pick**
> (1.8x, 46% fewer new sims, 1728 inherited visits; the memo added 25 hits
> and left the pick sequence identical). The C++ actor (production self-play)
> is far faster per step; the ratios carry over.

> **Update (2026-07-31): leaf rollouts at sideboard roots.** Depth benchmarking
> showed the PUCT tree alone never reaches the next game: at `sb_sims=256` the
> most-visited line is ~3 decisions deep (max 8 — still inside the swap picks),
> and even 4096 sims (~40 s/root) reach only ~turn 1, because the ~33-child swap
> menu with diffuse priors spreads visits sideways (~+3 PV depth per sims
> doubling). So a freshly expanded leaf now **plays the determinized world
> forward** with the argmax-greedy raw policy (both seats, no rng) to the end of
> player-turn `--sb-rollout-turns` of the sampled next game — through the
> remaining sideboard picks and mulligans — and backs up THAT state's net value
> (or the true terminal ±1). Pure rollout-end value, no leaf mixing; rolled-out
> states are never added to the tree; a hard cap of `40 * turns` rollout steps
> (`mcts.ROLLOUT_STEPS_PER_TURN`, mirrored in `az_mcts.cpp`) bounds pathological
> lines. Measured at a real sideboard root (ur_delver vs gw_maverick, 256 sims,
> PPO-warm-start evaluator): steps/sim 2.0 → 100.0 — every sim now sees ~12
> player turns of the sampled game — at ~75 s/root in Python (the C++ actor is
> substantially faster). The mechanism is a general `run_search(...,
> rollout_turns=)` option (in-game roots keep 0 by default, anchor = the root's
> own turn), mirrored bit-exactly in the C++ actor as a new ROLLOUT phase
> (`--rollout-turns`/`--sb-rollout-turns`, -1 = inherit; rollouts force the
> immediate leaf-eval path even under `--batch K>1`). `test_mcts_parity.py`
> gained a rolled bo3-sb gate at `sb_rollout_turns=3`; the bo1 and
> inherited-budget bo3 gates stay rollout-free as the unrolled baseline.

> **Update (2026-07-26).** `--sb-sims` default raised 32 → **128**. The 32 was
> chosen when a sideboard decision was the old paired IN→OUT menu; the balanced
> delta menu offers every sideboard card *and* every maindeck card at once (~33
> children on a league deck, up to ~39), so 32 sims was about ONE visit per child
> and the visit distribution the policy trains on could not rank cards at all.
> 128 gives ~3-4 visits per child.
>
> Measured per-root cost on a real league sideboard root (ur_delver vs
> gw_maverick, 33 legal choices, uniform evaluator, `worlds=4`,
> `max_depth=200`, identical world seeds, only `sims` varied, interleaved and
> repeated): **~116 ms at 32 sims → ~399 ms at 128 sims, 3.4x**. Slightly
> sublinear against the 4x sim ratio because each root pays a fixed
> snapshot/determinize overhead. Affects the AZ / search paths only — PPO
> training does no search, so its cost is unchanged.
>
> This also fixed real drift — `az_train.py` hardcoded `32`/`4`/`200` in six
> places instead of importing the `cli_spec` constants, so the az / az-league
> paths would have silently kept the old budget.

Both self-play backends now run bo3 — `--actor` / `--no-actor` (default AUTO) picks
the backend independently of bo3 (the actor is no longer bo1-only). The actor argv
carries `--bo3` plus the three `--sb-*` flags; `--games` means MATCHES on both
backends.

**CI coverage.**

- `snapshot` tier (`test_snapshot.py`) — the match-scoped snapshot roundtrip across
  `init_ecs()` and the seed-salt world-variation at a sideboard root.
- `sbselfplay` tier — the Python bo3 self-play persists searched sideboard samples
  with next-game `z` (in-game samples priced by their own game's winner).
- `actor` tier — the bo3 sideboard parity case (`test_actor_parity.py` /
  `test_mcts_parity.py` bo3 MATCH cases, obs + visit parity) and
  `test_actor_shards.py`'s bo3 case (64 sideboard samples, `z` verified against the
  next game's winner, trainer-ingestible shard schema).

### User-facing tools at sideboard roots (Stage 7)

Every consumer that drives a searching (`az:`/`azraw:`/`mcts:`) seat through the
shared `opponents.SearchController` — `play.py` (via `tui_game`), `analysis.py`
(`search` + `report`/`interactive`), `train.py observe`, and the
`az_eval`/`eval_search_gate.py` promotion gate — now handles the bo3 **sideboard
root** correctly:

- **`SearchController` selects the sideboard budget at sideboard roots.** When the
  current decision is a bo3 sideboard prompt (`obs[_IS_SIDEBOARD_IDX] > 0.5`) the
  controller searches it with `sb_sims` / `sb_worlds` / `sb_max_depth` /
  `sb_rollout_turns` (the `DEFAULT_SB_*` constants, currently `256`/`4`/`200`/`12`)
  instead of the in-game budget. This mirrors `az_selfplay`:
  in-game roots keep `run_search`'s default `max_depth=60`, which at a sideboard
  root would be swallowed by the remaining sideboard/mulligan prompts and land the
  leaf value on a masked between-game observation (weak signal). The sideboard vs
  in-game searched split is tallied separately in `stats["sb_searched"]`.
- **Shared constants, no duplication.** `DEFAULT_SB_SIMS/WORLDS/MAX_DEPTH/
  ROLLOUT_TURNS` live in `cli_spec.py` (the single home, imported by
  `az_selfplay`, `SearchController`, and the CLI flag defaults); the
  sideboard-flag index `_IS_SIDEBOARD_IDX` is a named constant in `env.py`.
- **Spec query knobs.** `az:`/`azraw:`/`mcts:` specs accept
  `?sb_sims=&sb_worlds=&sb_max_depth=&sb_rollout_turns=` (alongside
  `sims=`/`worlds=`), so any tool taking a controller spec (observe
  `--player-a`, play `--model`) can tune the sideboard budget inline.
- **`analysis.py search`** gained `--sb-sims`/`--sb-worlds`/`--sb-max-depth`/
  `--sb-rollout-turns` and reports how many searched roots were sideboard roots
  (`N searched (K at bo3 sideboard roots)`); its recording sub-controller
  applies the same budget split.
- **`az_eval` / `eval_search_gate.py`** thread the same four `--sb-*` flags
  into the `az:`/`mcts:` spec they build, so the gate prices
  sideboard roots explicitly rather than inheriting them silently.

Verified end-to-end (`league/ur_delver` vs `league/gw_maverick`, `az:gen`): observe
runs a bo3 to `MATCH_RESULT` with the model sideboarding and search firing at each
sideboard swap; `analysis.py search` decodes SIDEBOARD roots; play's engine path
(`runner.run_match`, human-like seat vs searched model) coexists cleanly; the gate
runs without crashing on the masked sideboard obs. Observed per-decision latency at
the default sideboard budget (`sb_sims=32`, `sb_worlds=4`, `sb_max_depth=200`):
~200ms mean (166–312ms) per sideboard decision, vs ~68ms in-game at `sims=8`.

### Wall-clock per-decision search budget (Stage 8)

A search seat's effort can be set by **wall-clock time** instead of a fixed sim
count: the `time=<seconds>` spec knob (`az:gen?time=5&worlds=4`, `mcts:…?time=…`) —
and `play.py --think-time <seconds>` which appends it — give the search a
per-decision deadline, within which it runs as many simulations as fit (more time =
stronger play). `mcts.run_search` gained `time_budget_s`: when set it builds the
`worlds` roots up front and runs sims **round-robin** across worlds until the
deadline (so a timed cutoff never biases visits toward the first world), with a
floor of one sim per world; `sims`/`sb_sims` become an optional hard cap. When
`time=` is absent the original per-world sequential loop is **byte-for-byte
unchanged** (actor visit-parity holds). The one budget applies to both in-game and
sideboard roots (the `sb_max_depth`/`sb_rollout_turns` split still applies; the
deadline is checked between sims, so a rolled sideboard sim can overshoot it by
up to one rollout). `runner.run_games` now
prints a per-side search-effort line (roots searched, sideboard split, total and
mean sims/decision) so the effort spent is visible. Measured with `az:gen` on
`league/ur_delver` vs `league/gw_maverick`: `time=2&worlds=4` ≈ 190 mean
sims/decision (vs the 128 fixed default); `time=1` bo3 ≈ 103 mean over 471 roots
incl. 28 timed sideboard roots. Standalone check: `train/test_search_time_budget.py`
(not wired into a ci tier — wall-clock bounds can be flaky under CI load).

### World-parallel mirror-pool search (Stage 10)

The engine is a single-threaded global-singleton ECS, so one process runs one
simulation at a time — but a determinized search's `worlds` are fully independent
(own root node, own seed, per-sim `restore(0)`+`determinize(seed)`). For
**interactive** consumers (play/TUI/observe/analysis) the `procs=<n>` spec knob
(and `play.py --search-procs <n>`) opts into a **mirror pool**: `SearchRoboMageEnv`
records its real-action history and `ensure_mirrors(k)` spins up `k` extra engine
processes, each reset to the primary's seed and fast-forwarded by replaying that
history to a byte-identical state (asserted via obs equality). At a searchable
decision the worlds split contiguously across `[primary] + mirrors` and
`mcts.run_search_parallel` runs each env's slice in its own thread (each thread
mostly blocks on its engine pipe, so the GIL is not the bottleneck); the shared
evaluator is serialized under a lock (engine stepping at ~6ms/sim dominates the net
eval). World seeds are pre-drawn with the exact `run_search` derivation, so with one
env the parallel path is **bit-identical** to plain `run_search` and with N it just
fans the same worlds out — visit counts sum across envs. Any mirror drift/spawn
failure disables the pool with one stderr warning and the primary plays on alone
(interactive robustness over strictness). `run_search` itself is untouched and
`procs` defaults to 1, so **self-play and the parity corpus never touch this path**.
Regression: `train/test_mirror_search.py`, wired into `ci_check.py` as the default
`mirror` tier (torch-free: bit-exact merge, bo3 lockstep across the sideboard
boundary, and drift-fallback).

### Analysis trace games honor search specs (Stage 11)

`analysis.py` (and the `tui_analysis.py` browser, which shares its simulation
layer) used to play its trace games with the RAW net policy even for an
`az:`/`mcts:` model — the prefix only chose which net loaded. Now the **spec
prefix is the lever**: when the model (or model-opponent) spec is a search spec
(`az:` / `mcts:`), the simulated trace games are PLAYED by the real
`opponents.SearchController` (built via `make_controller`, bound to a
search-capable `SearchRoboMageEnv`), so the browser inspects states arising from
search-quality play. `azraw:gen` and bare PPO specs keep the raw-policy
`ModelController` — exactly what those specs mean everywhere else, so no new CLI
flag. The **inspection net** (SHAP / value / policy-probs displays) is still
loaded exactly as before via `_load_az_analysis_model` / `MaskablePPO.load` and
evaluated on the recorded obs — only how games are *driven* changed. The
SearchController is cached per model object (env is reused across games; its
`stats` accumulate) and its searched/fallback/sims tally is printed after the run.
Whatif/replay is preserved: `_replay_to_step` feeds recorded `full_actions` (no
live search during a replay); the whatif counterfactual `_rollout_from` re-drives
the branch live, which now legitimately searches on a search spec (works with the
search env; snapshots released between decisions as usual).

## Analysis-tool integration (M10)

The model-analysis tools (`train/analysis.py` + its shared CLI in
`train/cli_spec.py`) now understand AZ checkpoints and search play. Two additive
changes; no surgery on the analysis internals.

**AZ checkpoints as the model argument.** `analysis.py report` / `interactive`
accept an `az:gen` spec (or a bare `.pt` path) resolved to the generalist AZ
checkpoint via `resolve_az_checkpoint`; the deck it pilots is supplied separately
via `--deck-a`/`--deck-b`. A thin adapter
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
    --n-games 4 --sims 64 --worlds 4 --top 8 \
    --bo3 --sb-sims 32 --sb-worlds 4 --sb-rollout-turns 4   # sb-* apply at bo3 sideboard roots
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

- The `SEARCHINFO safe=` split is effectively closed since the snapshot-safe
  conversion (branch `snapshot_safe`, 2026-07-18): every decision kind is a
  searchable root (measured 100% over the corpus matchups + a league probe;
  residual `safe=0` surface documented in the safe-window section above).
  `SearchController.stats` still reports the searched/fallback ratio — a
  nonzero fallback count now indicates one of the documented residuals (or a
  regression; the `prompt_site_guard` CI test in `test_snapshot.py` walls off
  new blocking prompts).
- DETERMINIZE pins both players' known-top-library prefixes and known
  (revealed) opponent hand cards; the opponent's unknown hand + unpinned
  library form one exchange pool. **Post-board (bo3 game 2+) the opponent's
  SIDEBOARD cards join that pool** (their sideboard swaps are hidden — P only
  knows the 75, so sampled worlds re-deal which cards are in the deck; a
  declared companion is public and stays pinned; game 1 models the known
  pre-board 60). Covered by the `sideboard_determinize` CI test. Mild accepted
  approximations are documented in `src/search_server.cpp`.
- Per-GAME snapshot slots are wiped at each game start (`snapshot_release_all` in
  `play_single_game`) — no cross-game reuse. The bo3 **sideboard** decision is the
  exception: it is a searchable root backed by a MATCH-scoped snapshot that survives
  the per-game wipe (Phase 2, above), so a sideboard prompt can be searched on the
  next game's horizon.
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
