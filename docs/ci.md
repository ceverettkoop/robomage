# Continuous integration & the test standard

The standard for changes to this repo: **`make check` must pass.** CI runs exactly it; run it
locally before pushing.

```bash
make check    # `make` (debug build + codegen), then train/ci_check.py (every default tier)
```

It needs the Forge card scripts provisioned first (the SessionStart hook does this in cloud sessions only;
run it yourself on a fresh local clone):

```bash
train/.venv/bin/python tools/forge_fetch/provision_decks.py
```

`provision_decks.py` fetches the script for **every card in `src/card_vocab.h`** plus every card
named by the top-level, `meta/` and `league/` decks, and the tokens they create. Fetching the
whole vocab keeps the script-derived codegen (`card_costs.py`, `card_props.py`,
`src/gen/card_costs_gen.h`) complete and independent of which decks happen to be provisioned.
`ci_check.py` exits 2 without running anything when a tier needs a missing engine binary or
card scripts, or when `--tier` names an unknown tier.

## The tiers (`train/ci_check.py`)

Tiers run in the order below. Every requested tier runs even if an earlier one fails, so one
invocation reports every finding. Exit 1 on any **error**; warnings alone pass. Most tiers wrap
one `train/test_*.py` script, which can be run standalone to reproduce.

| Tier | Checks (script) | Notes |
|---|---|---|
| `pygen` | Runs every generator; fails if the **tracked** outputs `train/_enums.py` / `src/gen/archetypes_gen.h` differ from the committed copies (restored afterward). The untracked script-derived outputs are only crash-checked | stale = a C++/JSON input changed without regenerating |
| `vocab` | every top-level and `league/` deck card resolves to a `card_vocab.h` entry (DFCs via their script's front face) | |
| `curriculum` | curriculum plan schema, each phase kind's composed `train.py` argv, resume forms, plan-hash prefix check (`test_curriculum.py`) | stdlib-only |
| `clispec` | the shared CLI vocabulary (`test_cli_spec.py`): removed flags error with their hint and stay out of `--help`/TUI forms, `--format` defaults to bo3, GUI launcher fields mirror play.py/browser flags, search-knob fold, per-board option errors, `analysis.py browse --source` dispatch | legs whose script needs torch/SB3 (`train.py`, `bench_actor.py`) or textual (TUI stubs) print `[skip]` without them |
| `gatesprt` | the AZ promotion gate's SPRT (`gate_sprt.py` via `test_gate_sprt.py`): symmetric hypotheses, draws half/half, Wald bounds, monotone/mirror-symmetric verdict, round-cap and floor-lock rules, simulated gates accept a stronger candidate, reject a weaker one, coin-flip an equal one | a silent change here changes which nets get promoted. Stdlib-only |
| `shardrec` | play-session shard recorder (`test_shard_record.py`): trainer schema, mid-game flush validity (`z=0` rows), no duplicate rows on rewrite, per-mover z backfill, both readers round-trip, π from the search posterior only, shared search-vs-net divergence | torch-free, engine-free |
| `treecache` | rebuilt-search-tree cache (`test_tree_cache.py`): per-world trees round-trip node-for-node through the npz layout; a bad format version is refused | torch-free, engine-free |
| `browser` | Textual analysis browser (`test_tui_browser.py`) driven headlessly: `.rmtrace` load/save round trip, a net probe, Tree tab rebuild/expand/walk, live-stream rows | needs the engine, textual and matplotlib; `[skip]` without the packages |
| `modelspec` | model-spec resolver (`test_model_spec.py`): the spec-family table, knob stripping, and one spec → one checkpoint across every consumer | torch-free, engine-free |
| `concede` | CR 104.3a concession sentinels (`test_concede.py`): bo1 / bo3-game / match concede outcomes, and the sentinel replays via `--replay` | needs the engine |
| `obsinv` | per-decision structural invariants on the raw observation over seeded scripted games (`test_obs_invariants.py`) — see its docstring for the list | prints the violated invariant (decision, seat, block, slot, value) |
| `actorobs` | `make actor-syntax` (`-fsyntax-only` over the actor's libtorch-free TUs, firing the obs-layout `static_assert`s in `src/actor/obs_builder.{h,cpp}`), plus a generated TU asserting `machine_io.h`'s `OFFSET_CHAIN` equals `env.py`'s offsets block by block | compiler only |
| `pergame` | bo3 per-game-index win-rate split: seat-flip consistency, tally merging, a live bo3 (`test_per_game_split.py`) | needs the engine |
| `snapshot` | `--search-server` snapshot/restore/determinize protocol: round-trip identity, RNG isolation, terminal intercept (`test_snapshot.py`) | needs the engine |
| `sbrules` | dead-sideboard-card rules table and `mcts.sb_dead_mask` (`test_sb_rules.py`) | engine- and torch-free; the C++ twin is covered by `actor` |
| `sbselfplay` | AZ self-play writes bo3 sideboard samples priced by the next game's result (`test_sideboard_selfplay.py`) | needs the engine |
| `plansearch` | sideboard plan search (`mcts.run_plan_search` + `IncrementalPlanSearch`): coverage, accounting, determinism, memo (`test_plan_search.py`) | needs the engine |
| `mirror` | world-parallel mirror-pool search: bit-identical visits vs serial, bo3 lockstep, pool disable on drift (`test_mirror_search.py`) | needs the engine |
| `xwsearch` | cross-world batched leaf evaluation in the Python search: exact parity vs sequential (`test_xw_search.py`) | torch legs skip without torch |
| `replay` | the byte-identical replay corpus (`train/regression/replay_diff.py check`): 9 curated delver/doomsday/mav scenarios plus every league pairing and mirror (`lg_<a>__<b>`), 64 transcripts, one frozen seed | any transcript drift |
| `smoke` | scripted **hard** vs itself over the league mirrors + a ring of crosses (`--smoke-games`, default 2 per matchup) | see classification below |
| `fuzz` | the `explore` fuzzer (`--mode`) over the league mirrors (`--fuzz-games`, default 8 per matchup) | see classification below |

`smoke`/`fuzz` play through `runner.run_games` with seed `--seed + 1000·k` for matchup `k`, and
write `ci_out/<tier>_<a>__<b>.txt` (Python output plus engine stderr).

### Opt-in tiers (valid for `--tier`, not in the default run)

| Tier | Checks | Needs |
|---|---|---|
| `actor` | `test_actor_parity.py` (obs bit-parity), `test_mcts_parity.py` (MCTS visit parity incl. two-model legs), `test_actor_shards.py` (shard schema/ingest), `test_actor_trains.py` (C++ vs Python shards train alike), `test_az_gate.py` (actor-backend gate driver) | `bin/<config>/az_actor` (`make actor`) + torch, else `[skip]`. Errors (instead of running) if any `src/` engine source (excluding `src/gen/`) is newer than the actor binary. The actor links the venv's libtorch but compiles against its headers, so after any change of the venv's torch rebuild both configs (`make actor` and `make actor BUILD=RELEASE`): stale actors abort with c10 errors (e.g. `set_stride`) or fail parity spuriously |
| `analysis` | `test_analysis_session.py` (chunked `IncrementalSearch` bit-identical to `run_search`, detached-mirror lockstep, rewind, cancellation), `test_shard_replay.py` (`analysis.py browse --source DIR` reconstruction), `test_browse_session.py` (browser core), `test_gui_session_io.py` (`.rmplay`/`.rmtrace` save/load) | the engine; torch-free |
| `treerebuild` | `test_tree_rebuild.py`: a recorded match's searches (fixed-sims and timed) re-run from their recorded world seeds + sim counts reproduce the recorded root visits exactly; cached tree reopens identical; followed rows resolve to their origin | the engine; torch-free |
| `azinspect` | `test_az_inspect.py`: every `az_inspect` view against a fresh net + synthetic shards | torch (else `[skip]`); no engine |
| `gui` | headless PySide6 legs (offscreen): play board, record-shards, analysis window, session save/reopen, `.rmtrace` open, shard browser, and a torch-free `mcts:uniform` search recording whose tree is rebuilt — each browsing only the scratch recordings the earlier legs wrote | PySide6 (else `[skip]`) |

```bash
train/.venv/bin/python train/ci_check.py --tier actor
```

Each `gui` leg is one process selected by `ROBOMAGE_SMOKE` (a comma list of `play[:N]`,
`analysis`, `session`, `trace`, `browser[:DIR]`, `tree:DIR`; parsed by `cli_spec.smoke_legs`).
Reproduce a leg by hand under a memory cap, against a small recording only (never a training
pool):

```bash
QT_QPA_PLATFORM=offscreen ROBOMAGE_SMOKE=play:8 ROBOMAGE_RECORD_DIR=/tmp/rec \
  train/.venv/bin/python train/play.py --no-analysis --deck-a league/ur_delver \
  --deck-b league/gw_maverick --player-b scripted --format bo1 --record-shards
QT_QPA_PLATFORM=offscreen ROBOMAGE_SMOKE=browser:/tmp/rec \
  systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 \
  train/.venv/bin/python train/gui_main.py
```

Flags: `--tier a,b` (subset), `--smoke-games N`, `--fuzz-games N`, `--seed N` (default 1),
`--mode explore|explore:patient` (fuzz agent), `--matchups mirrors|ring|mirrors+ring|all|"a:b,c:d"`
(default: smoke `mirrors+ring`, fuzz `mirrors`), `--out-dir DIR` (default `ci_out/`). The run
ends with a `Reproduce:` line naming its tiers, seed, mode and matchups.

### Error and warning classification

Draws are not acceptable, but the causes differ in severity:

- **Error** (fails the gate): an engine crash (nonzero exit / EOF mid-game), fewer games
  completed than requested, or any transcript line matching `^ERROR:` / `^FATAL:`, a Python
  traceback, `Segmentation fault`, a failed assertion, `Aborted` or `core dumped` (a non-fatal
  engine `ERROR:` is not acceptable per `CLAUDE.md`).
- **Warning** (flagged for review, passes): a game that ends with no winner because the engine
  hit its step cap (a **stall** draw; its log is moved into `--out-dir`), or an engine
  `WARNING:` line.

`WARNING: Unrecognized ability param` lines stay in the transcript (greppable) but are not
surfaced: they fire in bulk for cosmetic / AI-hint params. Params the parser drops without
warning are the `ignored_keys` set in `src/parse.cpp`; their review is tracked in `todo.md`
("Audit: ability-param keys the parser silently ignores").

## Workflows

- **`.github/workflows/ci.yml`** — push / PR to `main`, and `workflow_dispatch`. Runs the default
  `ci_check.py` with the fixed seed, so a red run is the diff's fault, not RNG. Transcripts are
  uploaded as `ci-transcripts-<run_id>` even on success, so stall-draw logs are reviewable. The
  image has only numpy + gymnasium: the `browser` tier, the `clispec` legs that need
  torch/SB3/textual, and `xwsearch`'s torch legs print `[skip]` here.
- **`.github/workflows/nightly-fuzz.yml`** — daily 08:00 UTC, plus manual (`games`, `seed`
  inputs). Three jobs:
  - `fuzz` — the `fuzz` tier over **all** league matchups, once with `explore` and once with
    `explore:patient`, 40 games per matchup, with a **rotating** seed (run number × 100000
    unless given). The job summary prints the exact local reproduce command; artifact
    `nightly-fuzz-explore[-patient]-<run>`.
  - `actor-parity` — installs CPU torch (pinned `torch==2.10.*`, matching the dev venv; later
    headers need C++20) and the SB3 stack, builds `make actor`, runs `--tier actor`.
  - `full-check` — same torch pin plus SB3, matplotlib and `train/requirements-tui.txt`, then the
    default `ci_check.py`, so every leg skipped per-push runs at least nightly.

All jobs share `.github/actions/setup-robomage` (Python 3.12 venv with numpy + gymnasium,
card-script provisioning, ccache, debug build). The card-script cache is keyed on the
`FORGE_PIN` hash (a pin bump restores nothing and re-fetches everything) and saved right after
provisioning, so even a failing first run leaves a warm cache; the fetch tool is add-only, so a
stale cache within a pin only re-fetches the delta.

## Reproducing a CI failure locally

1. **PR gate:** run `make check` (same command, same fixed seed). The failing tier names its
   transcript under `ci_out/`; the run's `ci-transcripts-*` artifact has CI's copies. Re-run one
   tier with `--tier <name>` or its `test_*.py` script directly (e.g.
   `train/.venv/bin/python train/test_obs_invariants.py`; fixed internal seeds).
2. **Nightly fuzz:** run the `ci_check.py --tier fuzz --matchups all --mode … --fuzz-games …
   --seed …` line from the job summary; the `nightly-fuzz-*` artifact has the transcripts.

## Intentionally changing behavior

Some tiers pin current behavior, so an **intended** change updates them in the same commit (the
diff is part of the review). The one-command path is **`make regen`**: provision the card set,
`make` (which reruns every generator), then re-record the replay corpus. Review and commit the
changed files. What it covers:

- **`replay` drift** from a deliberate engine / agent / card-data change: re-record and commit.
  ```bash
  train/.venv/bin/python train/regression/replay_diff.py record
  ```
  The corpus is **platform-portable**: all gameplay randomness goes through `src/stable_rng.h`
  (hand-rolled Fisher–Yates + rejection sampling over the raw mt19937 stream), so Mac and Linux
  produce byte-identical transcripts from a seed. Never call `std::shuffle`,
  `std::uniform_int_distribution` or `rand()` with `cur_game.gen` (or for any gameplay
  decision): their output is implementation-defined and differs between libstdc++ (CI) and
  libc++ (macOS), which would lock the corpus, RMLOG replays and bug-repro seeds to one platform.
- **`pygen` staleness** after editing a C++/JSON codegen input: `make pygen` (every generator,
  unconditionally, write-if-changed), then commit `train/_enums.py` and
  `src/gen/archetypes_gen.h`. The other generated files are untracked; nothing to commit.

`make refetch` deletes the local card/token scripts, re-provisions them at the pinned Forge
commit and runs `make check` **without** re-recording, to detect whether fresh scripts drift
from the committed corpus; if the drift is intended, follow with `make regen`.

`train.py observe --player-a explore --player-b explore --verbose --out FILE` is the manual,
exploratory fuzz campaign (dumps a transcript for review; always exits 0). `ci_check.py` is the
gating wrapper.
