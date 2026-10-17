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
| `pygen` | Runs every generator; fails if the **tracked** outputs `train/_enums.py` / `src/gen/archetypes_gen.h` differ from the committed copies (restored afterward). The untracked script-derived outputs are crash-checked, and the C++ cost tables (`src/gen/card_costs_gen.h`) must equal `train/card_costs.py`'s matrices float32-exactly (`test_cost_tables.py`) | stale = a C++/JSON input changed without regenerating |
| `vocab` | every top-level and `league/` deck card resolves to a `card_vocab.h` entry (DFCs via their script's front face) | |
| `curriculum` | curriculum plan schema, each phase kind's composed `train.py` argv, resume forms, plan-hash prefix check (`test_curriculum.py`) | stdlib-only |
| `clispec` | the shared CLI vocabulary (`test_cli_spec.py`): removed flags error with their hint and stay out of `--help`/TUI forms, `--format` defaults to bo3, GUI launcher fields mirror play.py/browser flags, search-knob fold, per-board option errors, `analysis.py browse --source` dispatch | legs whose script needs torch/SB3 (`train.py`, `bench_actor.py`) or textual (TUI stubs) print `[skip]` without them |
| `gatesprt` | the AZ promotion gate's SPRT (`gate_sprt.py` via `test_gate_sprt.py`): symmetric hypotheses, draws half/half, Wald bounds, monotone/mirror-symmetric verdict, round-cap and floor-lock rules, simulated gates accept a stronger candidate, reject a weaker one, coin-flip an equal one | a silent change here changes which nets get promoted. Stdlib-only |
| `shardrec` | play-session shard recorder (`test_shard_record.py`): trainer schema, mid-game flush validity (`z=0` rows), no duplicate rows on rewrite, per-mover z backfill, both readers round-trip, π from the search posterior only, shared search-vs-net divergence | torch-free, engine-free |
| `treecache` | rebuilt-search-tree cache (`test_tree_cache.py`): per-world trees round-trip node-for-node through the npz layout; a bad format version is refused | torch-free, engine-free |
| `menus` | the Python search's menu semantics on synthetic observations: duplicate-edge merging (`decode.menu_merge_reps`, the `mcts._Node` fold/select/walk; `test_menu_merge.py`) and the interchangeable-menu collapse behind the `SearchController` trivial fast path (`test_trivial_menu.py`) | torch-free, engine-free; the C++ merge twin is covered by `actor` |
| `browser` | Textual analysis browser (`test_tui_browser.py`) driven headlessly: `.rmtrace` load/save round trip, a net probe, Tree tab rebuild/expand/walk, live-stream rows | needs the engine, textual and matplotlib; `[skip]` without the packages |
| `modelspec` | model-spec resolver (`test_model_spec.py`): the spec-family table, knob stripping, and one spec → one checkpoint across every consumer | torch-free, engine-free |
| `concede` | CR 104.3a concession sentinels (`test_concede.py`): bo1 / bo3-game / match concede outcomes, the sentinel replays via `--replay`, and a concession at a parked query (Brainstorm's put-back, Cavern of Souls' ETB type; BQUERY and `--machine-format rmstate`) ends the game with nothing parked | needs the engine |
| `scenarios` | rules-regression scenarios (`test_scenarios.py`): every `train/regression/scenarios/*.json` sculpted harness game (bo1 unless its `format` says bo3, parallel) still shows its asserted rules outcome — narrative lines present / absent / in order, actions offered or withheld, the game result, and (`expect_dump*`, the `knowledge_*` sharpness scenarios) the deductions on the seats' knowledge tables, rendered as text lines from a `--rmstate-dump --state-dump json` run | needs the engine; ~2 s. A failure names the missing / forbidden line; `test_scenarios.py --narrative <name>` (`--dump <name>` for the knowledge lines) prints what the assertions match against |
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
| `rmstate` | the RMSTATE state format (`test_selfcheck_state.py`, below): the engine's `--selfcheck-state` checks over the replay corpus, a short explore fuzz and every scenario, with each seat's knowledge sidecar (`k_*` tables) on at level expert; the Python-vs-C++ decode comparison, decoder refusals and `--machine-format rmstate` sessions (`test_rmstate.py`'s parts); name-a-card candidates with open and closed decklists; the knowledge-side include isolation | needs the engine; ~3 min on 4 cores (2.5 min before the knowledge sidecar). Engine stderr in `ci_out/rmstate/` |
| `knowledge` | the knowledge sidecar (`test_knowledge.py`, below): `rmknow --selftest` / `--exactness`; an explore fuzz under `--selfcheck-knowledge` (soundness against the engine state, knowledge state across snapshot restores); `--machine-format rmstate` sessions each with a `--rmstate-dump`, stream frames equal to the dump's `k_*` tables included, `rmknow --isolation` on each writer's frames; `rmknow --truth` on every dump; archetype inference against the true lists and the cards seen; the belief layer's structure, calibration and exactness. The replay corpus runs under `--selfcheck-knowledge` in the `rmstate` tier | needs the engine and `rmknow`; ~100 s on 4 cores. Engine stderr in `ci_out/knowledge/` |
| `agentparity` | the scripted agent on RMSTATE (`test_agent_parity.py`, below): its card knowledge from a `CARDS` frame equals the generated tables'; replay-corpus games over `--machine-format rmstate` equal the committed transcripts; smoke games (two bo3) equal on both protocols; listed divergences against their re-recorded baselines | needs the engine; ~20 s on 4 cores (`--full`: every corpus and smoke-tier game and a bo3 match per mirror, ~75 s) |
| `smoke` | scripted **hard** vs itself over the league mirrors + a ring of crosses (`--smoke-games`, default 2 per matchup) | see classification below |
| `fuzz` | the `explore` fuzzer (`--mode`) over the league mirrors (`--fuzz-games`, default 8 per matchup) | see classification below |

`smoke`/`fuzz` play through `runner.run_games`; game `i` of matchup `k` uses seed
`--seed + 1000·k + i`. Every game is its own process-pool task (games are independent: own
engine process, seed and fresh controllers), and each matchup's games are reassembled in game
order into `ci_out/<tier>_<a>__<b>.txt` (per game: a `--- Game i/N [mode] seed S ---` header,
Python output plus that game's engine stderr). A draw's saved log lands next to it as
`<tier>_<a>__<b>_g<i>_draw_<stamp>.txt`.

### The `rmstate` tier

`train/test_selfcheck_state.py` (`run()` is the tier; `--part` runs one part) passes its engine
flags through `ROBOMAGE_ENGINE_FLAGS` and keeps each engine's stderr in `ci_out/rmstate/` (the
`--rmstate-dump` files it writes are deleted once checked). A self-check failure is an `ERROR:
selfcheck-state <check>: viewer …, decision … (game, turn, step): <first differing cell>` line on
an engine's stderr; each engine prints one `SELFCHECK-STATE: decisions=… errors=N` line when its
match ends (`docs/state_format.md` §10). Every seat frame carries its writer's knowledge tables
(`docs/knowledge_store.md` §9.6): the self-check's own instance's consistency errors are
`selfcheck-state knowledge` failures, a dump's or stream's instance prints `ERROR: knowledge[…] …
(dump frames)` / `(stream frames)`, and the roundtrip, references and snapshot checks cover the
`k_*` tables.

| Part | Runs | Fails on |
|---|---|---|
| `corpus` | `replay_diff.py check` with `--selfcheck-knowledge` (which implies `--selfcheck-state`; the `knowledge` tier shares these runs) | corpus drift, an `ERROR:` line, a missing summary (one `SELFCHECK-STATE`, one `SELFCHECK-KNOWLEDGE` and one `SHADOW-KNOWLEDGE` per corpus game), `errors≠0`, or a `SHADOW:` line no signature of `train/regression/knowledge_divergences.json` classifies (the `knowledge` tier's `shadow` part) |
| `fuzz` | 16 explore-vs-explore games (`FUZZ_GAMES`: bo1 and bo3; reanimator for face-down exile, wrb_energy and bw_dnt for Cabal Therapy and Disruptor Flute; four with `--selfcheck-state-sample 1`); two write `--rmstate-dump` files | a crash, an `ERROR:` line; a finished game also needs its `errors=0` summary (one stopped at the 2500-decision cap is scanned only) |
| `scenarios` | `test_scenarios.py` with `--selfcheck-state --selfcheck-state-sample 1` | any failing scenario (the runner fails on an `ERROR:` line; a scenario stops at its decision cap, so it prints no summary) |
| `decode` | `test_rmstate.check_dump` / `check_refusals` on the fuzz dumps, `stream_checks` on two `--machine-format rmstate` sessions with the self-checks on (one bo3 under `--closed-decklist`), `check_name_card_candidates` (a Disruptor Flute preset's opening menu: the opponent's registered cards with open decklists, every vocab card under `--closed-decklist`) | any check's failure; a session's `ERROR:` line or missing summary |
| `isolation` | the includes of `src/state_format/decode.*`, `schema.*`, `json.*`, `recode.*` and `src/knowledge/*` | an include outside `STD_HEADERS`, `ALLOWED_PROJECT_HEADERS` (`decode.h`, `schema.h`, `json.h`), the file's own header and (for `src/knowledge/`) its own headers |

Reproduce a failure:

```bash
train/.venv/bin/python train/test_selfcheck_state.py --part fuzz      # one part
# one fuzz game (its seed is in the failure's label); its engine stderr is ci_out/rmstate/fuzz_<seed>.err
train/.venv/bin/python train/test_selfcheck_state.py --part fuzz --fuzz-seed 102
# one scenario with the checks on
ROBOMAGE_ENGINE_FLAGS="--selfcheck-state --selfcheck-state-sample 1" \
  train/.venv/bin/python train/test_scenarios.py --show <name>
```

### The `knowledge` tier

`train/test_knowledge.py` (`run()` is the tier; `--part` runs one part) keeps each engine's stderr in
`ci_out/knowledge/` and deletes its dumps and saved streams once checked. Every seat frame of every
writer carries that writer's knowledge instance's `k_*` tables (`docs/knowledge_store.md` §9.6).
`--selfcheck-knowledge` (implies `--selfcheck-state`; `src/state_format/selfcheck_knowledge.h`)
checks, at every decision, every `k_*` fact of both seats against the true world built from the
engine state, and every knowledge instance's state across the self-check's snapshot restores; a
failure is an `ERROR: selfcheck-knowledge soundness|snapshot: viewer …, decision …: …` line and each
engine prints one `SELFCHECK-KNOWLEDGE: decisions=… facts=… errors=N` line at match end. It also
runs the shadow comparison (`src/state_format/shadow_knowledge.h`): wherever the `k_*` tables and the
old BQUERY vector's memory (known library tops, known opponent hand cards, the decklist revealed
bits) disagree, a `SHADOW: <kind> <direction> viewer=… owner=… card=… … events=…` line when the
mismatch appears, and one `SHADOW-KNOWLEDGE: decisions=… compared=… mismatches=… shown=…` line at
match end; a mismatch is no engine error, the tier classifies it. The replay corpus runs under it in
the `rmstate` tier's corpus part (shared engine runs).

| Part | Runs | Fails on |
|---|---|---|
| `selftest` | `rmknow --selftest`; `rmknow --exactness 40` (`EXACTNESS_N`) | a failed test or check |
| `fuzz` | 6 explore-vs-explore games (`FUZZ_GAMES`: bo1 and bo3; ur_delver, bug and uw_control put-backs, wubg_doomsday and reanimator tutors, bug, wrb_energy and reanimator discard, mav's Sylvan Library, reanimator's face-down exile; one with `--selfcheck-state-sample 1`) under `--selfcheck-knowledge`; two write `--rmstate-dump` files | a crash, an `ERROR:` line; a finished game also needs its `errors=0` `SELFCHECK-STATE` and `SELFCHECK-KNOWLEDGE` summaries |
| `isolation` | three `--machine-format rmstate` sessions (`STREAM_GAMES`: ur_delver vs bug bo1, reanimator vs uw_control bo3, bug vs wubg_doomsday bo3 under `--closed-decklist`), each with a `--rmstate-dump` and `--selfcheck-knowledge`; `test_rmstate.check_stream` (every stream frame equals the dump's for that seat and decision, `k_*` tables included; decision numbers unique), then `rmknow --isolation` for both seats on every dump (the fuzz games' too) and each saved stream (`test_rmstate.check_knowledge_isolation`) | a frame or event difference, a duplicate decision number, a bo3 session without a sideboarding frame, an isolation difference, an `ERROR:` line or a missing summary |
| `truth` | `rmknow --truth` for both seats on every dump (fuzz and stream sessions): the soundness check against the dump's omniscient frames | a violated fact, a model error, no frame checked |
| `shadow` | the `SHADOW:` lines of the fuzz and stream sessions' engines, matched against the signatures of `train/regression/knowledge_divergences.json` (each a kind, a direction, a pattern over the line, a classification, a why and a regression scenario; the first that matches classifies it); then every signature's scenario run with `--selfcheck-knowledge` | an unclassified mismatch, a malformed signature file, a scenario that fails or prints no `SHADOW:` line its signature classifies |
| `archetype` | archetype inference (`docs/knowledge_store.md` §10): four `--machine-format rmstate` sessions (`ARCHETYPE_GAMES`: three under `--closed-decklist`, one bo3, one against bw_dnt's companion; one with open decklists); the METAGAME frame against `archetypes.json`; per seat frame the `k_template` / `k_archetype` facts against the true lists and the cards seen | a tagged deck missing from the METAGAME frame, a label out of order, the true list reported inconsistent, a wrong exact archetype, an open list not classified as itself from the first frame, a consistent hidden list lacking a card seen, an archetype the cards seen pin down not reported exact (or never becoming exact), an `ERROR:` line |
| `belief` | the belief layer (`docs/knowledge_store.md` §10.6) at `--knowledge-a/-b expert+belief`: two `--machine-format rmstate` sessions each with a dump (`BELIEF_STREAM_GAMES`: reanimator vs ur_delver under `--closed-decklist` and `--selfcheck-knowledge`, burn vs mav with open decklists): stream frames equal the dump's (`b_*` included), `rmknow --isolation --level expert+belief` on each writer's frames, `rmknow --truth` on the dump (probability zero exactly for the lists `k_template` excludes, positive for the true list, rows summing to the full scale); 13 calibration sessions under `--closed-decklist` (every reference list against the one five places on, `calibration_games`): log-loss, Brier, reliability bins and first-vs-last-frame figures printed; `rmknow --belief-exactness 10` (`BELIEF_EXACTNESS_N`) | a structural or isolation difference, a truth error, a mean list or archetype log-loss not below the uniform prior's, the true archetype's mean probability not higher at the games' last frames than at their first, an exactness failure, an `ERROR:` line |

```bash
train/.venv/bin/python train/test_knowledge.py --part fuzz --fuzz-seed 203   # one fuzz game
train/.venv/bin/python train/test_knowledge.py --part isolation
train/.venv/bin/python train/test_knowledge.py --part shadow
train/.venv/bin/python train/test_knowledge.py --part belief
# the calibration report over more sessions (every list against the lists 1..3 places on, bo1 and bo3)
train/.venv/bin/python train/test_knowledge.py --calibration 3 --binary "$PWD/bin/release/robomage"
# classify the SHADOW lines of any engine stderr (counts per signature; unclassified lines fail)
train/.venv/bin/python train/test_knowledge.py --classify ci_out/rmstate/corpus.err
# the soundness check on any game: one errors=0 SELFCHECK-KNOWLEDGE line per engine
ROBOMAGE_ENGINE_FLAGS=--selfcheck-knowledge train/.venv/bin/python train/regression/replay_diff.py check
```

### The `agentparity` tier

`train/test_agent_parity.py` (plan step 2, pass 1; `run()` is the tier, `--part` runs one part) plays
the scripted agent (hard tier, both seats) over `--machine-format rmstate`: `runner.run_games`
runs `rmstate_env.RMStateNarrativeEnv`, which rebuilds the old observation from each seat's frames
and `k_*` tables (`rmstate_adapter.SeatAdapter`), and the agent reads its card knowledge from
`CARDS` (the engine adds the raw scripts under `--rmstate-card-scripts`) and its deck names and the
opponent's list from `DECKLISTS`. A game is compared through `project`: the RMSTATE transcript has no
board block (the adapter rebuilds only what the agent reads), so the lines between `--- Decision N
---` and `Actions:` are left out of both, and the `PA:` / `PB:` hand lines are compared sorted;
the narrative, every menu and every choice must be identical.

| Part | Runs | Fails on |
|---|---|---|
| `cards` | a `--closed-decklist` CARDS frame (every vocab card): `FrameCardKnowledge` against `scripted_agent.VocabCardKnowledge` | any card whose cost row, land flag, property column, script lines or type line differs |
| `corpus` | `CORPUS_GAMES` of the replay corpus (`--full`: all 64) over RMSTATE | a projection that differs from the committed corpus transcript's |
| `smoke` | `SMOKE_GAMES`: smoke-tier games, two bo3 matches and the listed divergence (`--full`: every smoke-tier game and a bo3 match per league mirror), on both protocols | projections that differ |

A game whose old vector leaked or had a quirk that changed what the agent saw is listed in
`train/regression/agent_parity_divergences.json` (the game, its first differing decision, the seat,
the read that differed, a classification naming a signature of `knowledge_divergences.json`, why)
and re-recorded: its RMSTATE projection is committed as `train/regression/agent_parity/<game>.txt`,
which the check compares against, and the BQUERY game must still first differ at the listed
decision. `--record` rewrites those baselines. `--lockstep GAME` is the diagnostic that attributes a
difference: the BQUERY game is played with `--rmstate-dump` and `--selfcheck-knowledge`; at every
decision the adapter's observation is compared with the real vector block by block and a second agent
decides on it; the first differing choice is printed with the differing blocks, the menu and the
classified `SHADOW:` line of the deciding seat for each differing memory block.

```bash
train/.venv/bin/python train/test_agent_parity.py --full
train/.venv/bin/python train/test_agent_parity.py --games smoke/burn__bw_dnt/11001
train/.venv/bin/python train/test_agent_parity.py --lockstep match/bw_dnt__wrb_energy/298
```

### Opt-in tiers (valid for `--tier`, not in the default run)

| Tier | Checks | Needs |
|---|---|---|
| `actor` | `test_actor_parity.py` (obs bit-parity), `test_mcts_parity.py` (MCTS visit parity incl. two-model legs), `test_actor_shards.py` (shard schema/ingest), `test_actor_trains.py` (C++ vs Python shards train alike), `test_az_gate.py` (actor-backend gate driver) | `bin/<config>/az_actor` (`make actor`) + torch, else `[skip]`. Errors (instead of running) if any `src/` engine source (excluding `src/gen/`) is newer than the actor binary. The actor links the venv's libtorch but compiles against its headers, so after any change of the venv's torch rebuild both configs (`make actor` and `make actor BUILD=RELEASE`): stale actors abort with c10 errors (e.g. `set_stride`) or fail parity spuriously |
| `analysis` | `test_analysis_session.py` (chunked `IncrementalSearch` bit-identical to `run_search`, detached-mirror lockstep, rewind, cancellation), `test_shard_replay.py` (`analysis.py browse --source DIR` reconstruction), `test_browse_session.py` (browser core), `test_gui_session_io.py` (`.rmplay`/`.rmtrace` save/load), `test_tree_follow.py` (SearchController tree-reuse gates, hidden-info fingerprint) | the engine; torch-free |
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
ends with a per-tier wall-time table and a `Reproduce:` line naming its tiers, seed, mode and
matchups.

### Error and warning classification

Draws are not acceptable, but the causes differ in severity:

- **Error** (fails the gate): an engine crash (nonzero exit / EOF mid-game), fewer games
  completed than requested, or any transcript line matching `^ERROR:` / `^FATAL:`, a Python
  traceback, `Segmentation fault`, a failed assertion, `Aborted` or `core dumped` (a non-fatal
  engine `ERROR:` is not acceptable per `CLAUDE.md`).
- **Warning** (flagged for review, passes): a game that ends with no winner — a drawn game (both
  players lost at once, CR 104.4a) or a **stall** where the engine hit its step cap (either way
  its log is moved into `--out-dir`) — or an engine `WARNING:` line.

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

**Extra engine flags.** `ROBOMAGE_ENGINE_FLAGS` (whitespace-separated) is appended to the command
line of every engine a Python `RoboMageEnv` starts (`train/env.py`: the harness, `observe`, the
scenario runner), e.g. `ROBOMAGE_ENGINE_FLAGS=--trace-visibility` to narrate reveal windows
(`VISIBILITY:` lines). A scenario passes its own through its `engine_flags` key
(`test_scenarios.py`). Leave it unset for `make check`: a flag that changes output changes
transcripts. `ROBOMAGE_ENGINE_FLAGS="--rmstate-dump /abs/dir/d_%p.rms --state-dump json"` writes
every engine's RMSTATE frames (and their JSON lines) beside an unchanged run; check such dumps
with `train/.venv/bin/python train/test_rmstate.py --dumps /abs/dir/d_*.rms` (engine re-encode byte
identity, Python vs C++ JSON). `ROBOMAGE_ENGINE_FLAGS=--selfcheck-state` runs the state format's
self-checks in every engine (`src/state_format/selfcheck.h`, `docs/state_format.md` §10): it
changes no transcript (the replay corpus stays byte-identical under it), failures are `ERROR:`
lines on the engine's stderr, and each engine prints a `SELFCHECK-STATE: … errors=N` summary
line on stderr at match end; `--selfcheck-state-sample N` (default 8) sets how often the sampled
permutation and snapshot checks run. `ROBOMAGE_ENGINE_FLAGS=--selfcheck-knowledge` adds the knowledge
sidecar's soundness check (and implies `--selfcheck-state`): a `SELFCHECK-KNOWLEDGE: … errors=N`
summary per engine. `train/test_rmstate.py` alone plays its own fuzz games and
`--machine-format rmstate` sessions (`run()` is the entry point for a gate tier), and checks the
knowledge isolation on them; `train/test_rmstate.py --isolation FILE… [--knowledge-a off]` runs
just that on given dumps or saved streams: `bin/<config>/rmknow --isolation --viewer A|B --level
off|expert --in FILE` recomputes every seat frame's `k_*` tables from the frames and compares them
with the engine's byte for byte. `--knowledge-a off` / `--knowledge-b off` (engine flags) turn a
seat's sidecar down to the `k_meta` row alone; play is the same.

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
  `std::uniform_int_distribution` or `rand()` with `cur_game.rng.engine` (or for any gameplay
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
