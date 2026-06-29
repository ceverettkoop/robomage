# Engine optimization pass — training throughput (2026-06-29)

Goal: reduce per-decision engine cost so RL training (and scripted regression) runs
more games/second. Scope of this pass: profile a **release** build, then land the
contained, high-confidence ECS hot-path fixes and re-measure. All changes are pure
performance — game logic and decision trajectories are unchanged (verified, below).

## TL;DR

| Build | wall (40 games) | decisions/s | ms/decision |
|---|---|---|---|
| Release, baseline | ~1.47 s | ~2,850 | 0.348 |
| Release, after this pass | ~1.07 s | ~3,900 | 0.256 |
| **Delta** | **−27% wall** | **+36%** | **−26%** |

Determinism fingerprint: the delver-mirror benchmark plays **4,186 decisions / 40
games** before *and* after every change — identical, i.e. trajectories are byte-for-byte
unchanged. Correctness re-checked on three non-mirror matchups (delver/mav, mav/doomsday,
doomsday/delver): all games complete, no crashes, no non-fatal errors.

The engine's share of total benchmark wall fell from **~71% → ~55%** (perf); the rest is
Python + per-game subprocess spawn + card-script parsing, which this pass did not touch —
so the **engine-only** CPU reduction is larger than the 27% end-to-end number.

## Methodology

- Build: `make clean && make BUILD=RELEASE HEADLESS=TRUE` (`-O2 -flto -march=native -DNDEBUG`).
- Benchmark: `train/bench_engine.py` — drives `RoboMageEnv` (the real `--machine` training
  path) with the scripted agent for both seats, no narrative decoding/printing. Reports
  wall / games-per-sec / decisions-per-sec. Deterministic per `--seed`.
  - Run from `bin/` with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`
    (otherwise idle OpenBLAS worker threads spin and pollute both the wall-clock and the
    profile).
  - Baseline: `../train/.venv/bin/python ../train/bench_engine.py --games 40 --seed 1000`
- Profiling: `perf record -g -F 3000 -- <bench>` then `perf report --dsos=robomage`
  (perf follows the forked engine subprocesses).

## What the profiler showed (baseline, release)

robomage = **71%** of samples (so the engine, not Python, is the bottleneck). Top engine
self-symbols:

| % self | Symbol | What it is |
|---|---|---|
| 14.35% | `_Map_base<const char*,…>::operator[]` | `mComponentTypes[typeid(T).name()]` lookup inside **every** `entity_has_component` / `GetComponent` |
| ~8.9% | `state_based_effects` | SBA + continuous-effects rebuild loop |
| 8.33% | `_Sp_counted_base::_M_release` | `shared_ptr` atomic refcount (from `GetComponentArray` returning a `shared_ptr` **by value**) |
| 6.22% | `~_Sp_counted_ptr_inplace<Orderer>` | more `shared_ptr` churn |

So ~28% of engine self-time was string-hash component lookups + `shared_ptr` refcounting —
overhead `-O2`/LTO cannot remove (a runtime hash lookup can't be constant-folded; the
atomic refcount is real). Those were the targets.

## Changes landed

All in the ECS core + one main-loop guard. No card scripts touched.

1. **Component type ids are a process-global integer, not a `typeid()`+hash lookup.**
   `src/ecs/component_manager.h`: added `component_type_id<T>()` (a memoized function-local
   `static`, assigned once on first use). `GetComponentType<T>()` returns it directly;
   component arrays are stored in a `std::array<…, MAX_COMPONENTS>` indexed by that id.
   Safe across the per-game `Coordinator::Init()` because `init_ecs()` registers the same
   component set in the same order every game, so the ids are stable.
   - Removes the 14.35% `unordered_map<const char*>` lookup entirely.

2. **`GetComponentArray<T>()` returns a raw `ComponentArray<T>*`, not a `shared_ptr` copy.**
   `src/ecs/component_manager.h`: `.get()` on the stored array, re-fetched each call (stays
   valid across `Init()`). No atomic refcount on the hot path.
   - Removes the `shared_ptr` `_M_release` / inplace-dtor churn from component access.

3. **`entity_has_component` reads the signature by reference + tests the memoized bit.**
   `src/ecs/coordinator.h` + new `EntityManager::GetSignatureRef()` (`entity_manager.h`):
   no `Signature` (bitset) copy, no type-lookup.

4. **`ComponentArray` entity↔index maps are flat arrays, not `unordered_map`s.**
   `src/ecs/component_array.h`: entity ids are dense and bounded by `MAX_ENTITIES`, so
   `entity → dense index` (with a `NO_INDEX` sentinel) and `index → entity` are
   `std::array`s. Every `GetData` / has-component access is now a single array index
   instead of a hash `find`.
   - Removes the `mEntityToIndexMap.find` symbol, which became the new #1 (10.88%) once
     the type lookup was gone.

5. **Skip the redundant machine-mode `populate_gamestate`.**
   `src/main.cpp`: the per-decision `populate_gamestate` + `print_game_state` at the input
   site is dead work in `--machine` mode — `print_game_state` early-returns there, and
   `InputLogger::get_input` re-populates into its own buffer. Guarded behind
   `if (!machine_mode)`, eliminating one full entity scan + large `memset` per decision.

## Measured effect of each step (delver mirror, 40 games, BLAS=1)

| Step | wall | decisions/s |
|---|---|---|
| Baseline | ~1.47 s | ~2,850 |
| + type-id memoize, raw-ptr array, redundant-populate (1–3, 5) | ~1.16 s | ~3,600 |
| + flat-array entity index (4) | ~1.07 s | ~3,900 |

After: robomage drops to ~55% of samples; the string-map and `shared_ptr` symbols are gone
from the profile. New top engine symbol is `state_based_effects` (12% self), whose
`_Rb_tree_increment` (3.4%) is `Permanent::counters` — a `std::map` — being iterated by
`get_counters` each SBA pass.

## Remaining opportunities (ranked, with profiler evidence)

1. **Continuous-effects / SBA rebuild — now the #1 engine cost (`state_based_effects`
   ~12%).** `apply_continuous_effects` + `gather_active_statics` rebuild every creature's
   P/T and keyword state from scratch on every SBA stability iteration, and SBA runs 2–3×
   per decision. Two angles: (a) a dirty-flag to skip the rebuild when no permanent
   entered/left/changed since the last pass (high impact, **higher risk** — correctness
   currently leans on the unconditional rebuild); (b) replace `Permanent::counters`
   (`std::map`, the 3.4% `_Rb_tree_increment`) with a small flat/sorted vector so
   `get_counters` stops walking a tree. (b) is lower risk and independently useful.
2. **`std::shared_ptr<Orderer>` passed by value through ~40 signatures** (`state_based_effects`,
   the `effect_*` handlers, `mana_system`, …). The `GetComponentArray` churn is gone, but
   these by-value params still copy a `shared_ptr` (atomic inc/dec) per call. Pass by
   `const&` (or raw pointer). Low complexity per site, but a wide mechanical sweep — worth
   its own commit.
3. **String / `Ability` copies in the statics rebuild** (`basic_string::_M_construct` ~5%,
   `Ability(const Ability&)` ~1.5%): `gather_active_statics` reallocates each creature's
   `keywords` vector from `CardData` every pass, and `entity_name`/log helpers build
   temporaries. Reuse buffers / avoid the per-pass keyword copy. Low–med.
4. **`serialize_state` allocates a fresh ~135 KB float vector per decision**
   (`machine_io.cpp`). Not a top symbol in this engine-bound benchmark, but it is a
   per-decision `malloc` on the training path — reuse a `static`/`thread_local` buffer.
   (Do **not** resize it — that changes `OBS_SIZE` and invalidates checkpoints.) Low.
5. **Per-game subprocess respawn — measured to be LOW value; do not prioritize.**
   `env.reset()` spawns a new `bin/robomage` per episode. Measuring the two suspected
   per-process costs:
   - **Deck/card parsing: negligible (~0.1 ms/game).** The parse symbols are visible in
     perf (`parse_card_face`, `parse_types`, `parse_mana_cost`, `parse_abilities`,
     `parse_static_abilities`, `load_card`, `Orderer::generate_libraries`, `Deck` ctor) but
     sum to **<1% of samples** (`generate_libraries` inclusive = 0.28%, the `parse_*`
     leaves 0.01–0.02% each). Warm page cache; a *cold* container pays a one-time disk read
     on first access only.
   - **Spawn + dynamic link + static init: ~1 ms/game.** Measured directly:
     `300× ./robomage --help` (no game thread, no parse) = 0.296 s → **0.99 ms/spawn**;
     corroborated by `ld-linux` at 2.6% in the profile.

   So a persistent process could recover at most **~4% of wall**. The real non-engine cost
   is the **Python interpreter + numpy building/parsing the 34k-float observation every
   decision** (libpython ~18% + numpy ~7% of self samples ≈ ~25% of wall) — which a
   persistent C++ process does **not** remove, since each decision still crosses the pipe
   and rebuilds the obs array. Higher-leverage non-engine wins live on the Python side
   (e.g. shrink/skip per-decision obs work, vectorize envs), not in process lifetime.

## Reproduce

```bash
make clean && make BUILD=RELEASE HEADLESS=TRUE
cd bin
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
../train/.venv/bin/python ../train/bench_engine.py --games 40 --seed 1000
# profile:
perf record -g -F 3000 -- ../train/.venv/bin/python ../train/bench_engine.py --games 80 --seed 1000
perf report --dsos=robomage --sort symbol
```

Files changed this pass: `src/ecs/component_manager.h`, `src/ecs/component_array.h`,
`src/ecs/coordinator.h`, `src/ecs/entity_manager.h`, `src/main.cpp`; benchmark added at
`train/bench_engine.py`.
