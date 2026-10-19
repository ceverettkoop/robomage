# Engine-only project: staged plan (steps 1–5)

Goal: turn robomage into a project that is **only the C++ rules engine**. Players and agents
drive it through in-process agents, a C++ CLI, or a machine protocol that emits an **exhaustive
binary game state** with hidden information withheld from each viewer. The RL stack is
discarded: PPO/AZ training, the libtorch actor, the float state vector, embeddings, codegen and
the Python front ends. It is archived in place, not forked: branch `archive/rl-stack` (tag
`rl-stack-final`, if created) ends at `cf7f97f`, the last commit with both stacks, and
development continues on `main` in this repository.

**Status: steps 1–4 are done** (each step's status paragraphs below are its history); step 5
is the plan forward. Paths under `train/` in those paragraphs name files deleted in step
3b (they are on `archive/rl-stack`), except the seven kept Python files, which moved to `python/`
and are written here with their current paths.

**The Python scripted agent is kept.** It is the only realistic opponent and a deterministic
consumer of the state. It is adapted to read the new format, and its games become the test that
the format carries everything it needs.

**Memory and inference are actor-side, in C++.** The engine sends only what each viewer
explicitly knows right now. A C++ knowledge sidecar (`docs/knowledge_store.md`) turns each
viewer's frame stream into perfect recall and sound logical inference. It is attached to the
same frames for every kind of agent, with levels `off` / `expert`.

**A C++ path to learned models.** Step 5 adds a schema-driven C++ featurizer and an in-process
C++ actor that runs a TorchScript policy on it. Card text reaches the model as oracle-text
embeddings computed in C++, exported with the structured card and ability tables as
framework-neutral whole-vocab files that any ML framework can load. This replaces the discarded RL stack's actor with one built on the
new format; a learner (training loop) is not in scope.

Scope: steps 1–5. The TUI and GUI (later steps) are deferred entirely; steps 1–5 only keep the
state decoder, knowledge tables and featurizer reusable for them. The eventual machine agent
is unspecified; the format just has to give it a complete, leak-free state and a structured
menu.

## Starting point (before step 1)

What made this feasible:

- `bin/robomage` has no torch and needs no Python at run time. The engine core (ECS, rules,
  parser, effects, `src/queries/`, replacement/layer systems, bo3, RMLOG replay) is ~47k lines
  of C++ outside `src/actor/`.
- The libtorch actor is already a separate binary (`src/actor/`, ~5.1k lines, `make actor`).
- No `pygen` output is used by the engine; `src/gen/*.h` are included only by `src/actor/`.
- `card_db` loads any script without the vocab; `card_vocab.h` is the implemented-card registry
  and the card id used by knowledge bookkeeping.
- There is an in-process decision hook: `InputLogger::set_input_provider`
  (`src/input_logger.h:86`), consulted on the machine path (`src/input_logger.cpp:394-424`).
- A C++ human CLI exists (`print_game_state` / `print_query`, `src/cli_output.cpp`;
  `--player A|B`), rendered from the RL-shaped `GameState` struct (`src/classes/gamestate.h`).
- RMLOG v2 decision logs (`--log-decisions`) replay deterministically under `--replay`, with a
  divergence guard (`src/input_logger.cpp:380-388`).

What made it costly:

- **The float state vector is the only interface between the engine and all Python.** The
  scripted agent, `explore` fuzzer, test harness, `--play` resolver, rules scenarios, TUI and
  GUI all decode it, and the test gate (`make check` → `train/ci_check.py`) is Python.
- "Machine mode" changes the **decision schedule**, not just the output: an RL masking
  heuristic, hidden mana sources plus auto-pay, and RL shaping signals (step 4a).

What the scripted agent depends on (`train/scripted_agent.py`, ~3.8k lines):

1. **Vector positions:** ~100 layout constants imported from `env.py`, ~150 lines reading
   `obs[...]`, plus `decode.decode_game_state`.
2. **Per-action arrays:** category, card id, controller, zone, entity-ref slot (used to tell
   duplicate legends apart) and option ordinal.
3. **Card knowledge from outside the engine:**
   - the generated `card_props` (103 columns) and `card_costs` tables;
   - raw Forge script lines (`_card_script_lines`);
   - ~260 references to card-specific vocab-id constants.
4. **The opponent's registered decklist** (`decode_opp_decklist`, used by `_name_card_pick`,
   `scripted_agent.py:2344`). Decklists are open by default in the new format, so this needs no
   special option.
5. **Mana auto-pay:** the agent never chooses which sources pay a cost.
6. **Engine-side memory:** the known top of each library (`Game::KnownLibraryTop`), known
   opponent hand cards (`Zone::identity_known`) and the match revealed bits. Under the new
   format these move to the C++ knowledge sidecar (`docs/knowledge_store.md`).

## Ground rules

1. **Change behaviour only on purpose.** Each step says whether it may change engine
   behaviour. Steps that must not end with the replay corpus byte-identical; an intentional
   change re-records the corpus in its own commit, saying why.
2. **Build the new beside the old.** RMSTATE (with its knowledge tables) was emitted *alongside*
   BQUERY until every consumer of the old vector was ported or deleted, and the Python gate kept
   running until the C++ gate replaced its engine tiers (both done in step 3).
3. **Where the work happens.** Steps 1–2 only added code to robomage as it stood. At the end of
   step 2 the RL stack was archived in place (branch `archive/rl-stack`, ending at `cf7f97f`)
   instead of forking; steps 3–5 happen in this repository, on `main`, keeping its history.
4. The existing conventions still apply: `-fno-exceptions` in engine TUs, shared queries in
   `src/queries/`, `ObjectRef` across zone changes, two-player only, CR as authority.

---

## Step 1 — State format, emitted, plus the knowledge store (additive)

Step 1 delivers the complete data path from engine to agent:

- the engine emits RMSTATE frames (memoryless snapshot plus observation events) for each
  viewer;
- a C++ knowledge sidecar turns each viewer's frame stream into knowledge tables with perfect
  recall and sound logical inference, attached to the same frames;
- everything is verified.

Later steps only consume this. It is all emitted next to BQUERY, so nothing existing changes.

**Handoff brief:** [`step1_handoff.md`](step1_handoff.md) has the ordered task list (T0–T12),
file layout, code locations, checks and exit criteria for a session executing step 1.

**Status: done.** 1a, 1b and 1c are implemented and every check below runs in `make check` (the
`rmstate` and `knowledge` tiers, `docs/ci.md`), with the replay corpus byte-identical and `obsinv`
green. Deviations from the plan as written:

- the shadow comparison was part of `--selfcheck-knowledge` (no flag of its own) and ran on the
  same engines as the soundness check; its mismatches were printed as `SHADOW:` lines and
  classified against `train/regression/knowledge_divergences.json` (`knowledge_store.md` §9.7;
  deleted with the old vector's memory in 3b, part B item 6). Three old-vector leaks were found (the known Brainstorm put-back
  one, its per-copy variant, and library positions followed through ranges the viewer cannot tell
  apart); the other classes are quirks (§9.7);
- it found and fixed, in the engine: a card put on the bottom of an empty library got position 1
  (`Orderer::move_object`); Doomsday's one-at-a-time search looked again at the cards it had
  already chosen, so the other player could not tell a pick from them (`zone_search.cpp`); in the
  sidecar, two library runs from one junction did not rejoin once compaction had folded the
  history behind the junction into it (`PlayerModel::rejoin`);
- the sharpness scenarios assert on the knowledge dump through the scenario runner's
  `expect_dump*` keys (then `train/test_scenarios.py`, now `robomage_check`'s `scenarios` tier), which render each seat frame's `k_*` tables as text lines; scenarios may
  now also be bo3 (`format`).

### 1a. Design (docs)

- [`state_format.md`](state_format.md): visibility, handles, snapshot tables, events, encoding,
  protocol, decision register. Drafted and decided.
- [`state_format_agent_map.md`](state_format_agent_map.md): every scripted-agent read mapped to
  its RMSTATE source. Drafted.
- [`knowledge_store.md`](knowledge_store.md): the sidecar. Levels `off` / `expert` (default
  `expert`), the pool/flow model, per-event update rules, output tables, the solver, the
  verification. Drafted.

**Exit:** all three reviewed. Every update rule in `knowledge_store.md` §5 is computable from
the fields of the event it consumes, which is the completeness check on the event schema.

### 1b. Engine emission

- **Visibility.** `src/queries/visibility.{h,cpp}` with per-viewer reveal windows
  (generalising `Game::revealed_in_library`). The old knowledge bookkeeping stays, still
  feeding BQUERY, so today's vector must not change; the corpus and `obsinv` tiers check that.
- **Observation events:**
  - typed, viewer-projected constructors;
  - emitted at the choke points where the engine's knowledge bookkeeping changes today:
    `Orderer::add_to_zone`, `shuffle_library`, `put_in_library_at_depth`, draw,
    `note_library_card_known`, `mark_card_revealed` (~33 call sites in 12 files), the
    resolution-end clear in `stack_manager.cpp:103`, face-down transitions, cast/play from
    hidden zones;
  - a per-viewer queue on `Game`, drained into each frame.
- **Handles.** A per-viewer handle table on `Game`, so snapshots cover it.
- **Encoder and decoders.**
  - Encoder `src/state_format/encode.*`, emitted with `--machine-format rmstate`.
  - A schema frame plus `CARDS` / `DECKLISTS`.
  - C++ decoder `src/state_format/decode.*`, and `--state-dump json` rendered from the decoded
    binary.
  - Python decoder for the agent, built from the schema frame.
- **Engine self-checks** (`--selfcheck-state`, run as new tiers of the existing Python gate):
  - **roundtrip:** encode → decode → re-encode is byte-identical at every decision;
  - **snapshot leak:** permute everything hidden from the viewer at that instant (both hidden
    zones' contents and both libraries' order, keeping live reveal windows fixed); the viewer's
    snapshot must be byte-identical;
  - **event leak:** each event's bytes, rebuilt in a permuted world, are identical;
  - **complete:** every permanent present, counts match zone sizes, every handle reference
    resolves, every menu handle exists;
  - **snapshot:** save → K actions → restore; both viewers' frames match the pre-save bytes.

### 1c. Knowledge sidecar (C++)

- **Core and tools.** `src/knowledge/`: the model, update rules, bound propagation and the
  budgeted exact solver, per `knowledge_store.md`. It includes only the state-format decoder
  and the standard library (a gate check enforces this). Also the standalone `rmknow` tool.
- **Engine integration.** Each seat has an instance fed from that seat's outbound frame bytes.
  `--knowledge-a/-b off|expert` (default `expert`). It appends the `k_*` tables to the
  viewer's RMSTATE frame, and its state is saved with engine snapshots.
- **Verification** (new gate tiers):
  - **soundness:** zero violations against the omniscient truth over the corpus, smoke and
    fuzz games;
  - **exactness:** with `exact` set, results equal a brute-force enumeration in small
    sculpted games;
  - **isolation:** `rmknow` output equals the in-process output;
  - **snapshot/restore:** knowledge state survives a restore unchanged;
  - **shadow comparison:** `expert`'s implied view is compared with the engine's
    `known_top_library_*`, `opp_known_hand` and revealed bits, and every mismatch is
    classified as a sidecar/event bug (fixed) or an old-vector leak or quirk (documented, with
    a regression scenario). One leak is already known: per-physical-card `identity_known`
    reveals which cards an opponent's Brainstorm put back (`knowledge_store.md` §7).

**Exit:**

- every self-check and knowledge tier is green;
- shadow-comparison mismatches are all classified;
- the old vector is unchanged.

## Step 2 — Port the scripted agent to RMSTATE + knowledge (additive)

- **Pass 1, strict parity.** An adapter on the Python decoder answers every old read with the
  value the old vector gave. Its emulation list is the read map's §5: the old normalisers and
  caps (48 permanent slots, 12 stack, 10 hand, 64 graveyard/exile), empty slots as zeros, and
  the token and keyword quirks. The two knowledge reads (own known top; `opp_known_hand`) come
  from the `expert` tables (`k_position`, `k_known_hand`). Card knowledge (`card_props`,
  `card_costs`, script lines) comes from `CARDS`; vocab constants become name lookups; deck
  names and the opponent's list come from `DECKLISTS`.
  **Exit:**
  - the replay corpus and the scripted-hard smoke games are byte-identical with the agent on
    RMSTATE versus BQUERY;
  - the only exceptions are games where a classified old-vector leak changed what the agent
    saw. Those are re-recorded, each with its classification as the reason.
  **Status: done.** The adapter is `train/rmstate_adapter.py` (the old observation rebuilt from a
  seat's frames and `k_*` tables; the card knowledge from `CARDS`, whose raw scripts the engine adds
  under `--rmstate-card-scripts`, read through the generators' own parsing), driven by
  `train/rmstate_env.py` under `runner.run_games(machine_format="rmstate")` (`--machine-format
  rmstate` on observe and the harness) for every `rmstate_capable` controller: the scripted tiers,
  `auto`, `actions:`, `play:`. The emulation as built is the read map's section 7. The vocab
  constants are name lookups, checked against each match's `CARDS`. The gate tier `agentparity`
  (`train/test_agent_parity.py`, `docs/ci.md`) holds the agent's games byte-identical on both
  protocols (the transcript without its board block, which the RMSTATE run cannot print). Results:
  all 64 corpus games, every smoke-tier game and a bo3 match per league mirror are identical; in a
  sweep of 300 bo3 matches six differ, all Cabal Therapy name picks reading the opponent's known
  hand (five `put_back_leak`, one `public_entry_untracked`), listed in
  `train/regression/agent_parity_divergences.json` with their re-recorded RMSTATE baselines. The
  replay corpus itself is unchanged: `make check` still plays it on BQUERY. Format fixes it took:
  `object.types` reports Creature only while the permanent is one (impending, attached reconfigure),
  a phased-out creature's own P/T, the narrative flushed ahead of each frame. The RMSTATE path runs
  at about half the BQUERY decision rate (release: ~310 vs ~630 decisions/s for scripted hard vs
  hard: the frames and the knowledge sidecar in the engine, and the Python decode and adapter).
- **Pass 2, simplify.** Replace offset arithmetic with decoded objects, drop the caps, and drop
  numpy if possible. The agent may now use the richer knowledge tables. Re-record the corpus
  with the live agent, saying why.
  **Status: done** (`state_format_agent_map.md` §8). The decisions are one numpy-free core over a
  view (`python/scripted_core.py`): the decoded-frame view (`python/rm_agent.py`) is the default on
  RMSTATE, the vector view keeps the BQUERY consumers' games unchanged (checked in lockstep;
  `agentparity` pins them). The replay corpus holds the frame view's RMSTATE games (4 of 64 games
  changed: rules-correct sickness and mana values, the opponent's possible hand cards); head to
  head over 2,600 bo3 matches the two passes are level (pass 2 0.502, 95% CI 0.483-0.522). Format gap
  fixed: a departed stack target's `lki` row (register 40).
- **Explore.** The `explore` tiers in the same file move with the rest; step 3 decides whether
  they also get a C++ port.
  **Status: done** (`state_format_agent_map.md` §9). The rule is `python/explore_agent.py` on the
  shared core's view, its rules listed for the C++ port; the `fuzz` tier (and the nightly fuzz)
  plays it on RMSTATE through the driver: the same games as on BQUERY.
- **Driver.** A minimal driver (engine subprocess, frame read, reply write) replaces `env.py`
  / `runner.py` / `opponents.py` for agent games. **Archive point** (originally "fork here": the
  RL stack was archived in place instead, ground rule 3).
  **Status: done** (`state_format_agent_map.md` §9). `python/rm_driver.py`, stdlib only: the
  scripted tiers, explore, `first`, `actions:`; corpus-format transcripts (the replay corpus is
  unchanged); the `replay`, `smoke` and `fuzz` tiers and the `rmagent` tier's games run on it.
  `play:` stays on `runner.py` until 3a. Release, scripted hard vs hard: ~530 decisions/s (BQUERY
  ~1,100).

## Step 3 — C++ gate, then strip the RL stack

**Status: done** (3a parts 1–2, 3b parts A–C below).

### 3a. C++ agents, the `--play` resolver and the gate

- **Agents.** `src/agents/`: an Agent interface on the provider hook (extended to pass the
  deciding seat), receiving the decoded frame including knowledge. Flags `--agent-a/-b <spec>`.
  Specs: `first`, `random`, `actions:<ints>`, `play:<specs>`, and optionally a C++ `explore`
  port for fast in-process fuzzing.
- **`--play` resolver.** Port `train/action_spec.py` and the seat-key logic of
  `PlayController` (`train/opponents.py:1158`) to `src/agents/play_spec.*`. Parity: every
  scenario resolves to the same `resolved --actions` list as the Python harness, and the same
  assertions pass.
- **Engine flags** replacing harness tricks:
  - `--hand-a/-b` and `--library-a/-b`, replacing the temp deck in `decks/temp/`;
  - `--max-decisions`;
  - `--check-decks`, replacing the `vocab` tier and `missing_cards.py`;
  - `--trace-decisions`: board and menu at every decision.
  **Status: part 1 done** (agents, resolver, flags; the gate driver is part 2). `src/agents/`
  on the decision hook (now passed the deciding seat; it covers chosen seats, the rest keep their
  input path), fed by an in-process frame writer (`state_format/agent_feed.h`: the stream's bytes
  for the seat, `k_*` included); specs `first`, `random`, `actions:`, `play:`, `explore`,
  `explore:patient`, plus global `--play` / `--actions` scripts. The agents draw from a port of
  Python's `random.Random` (MT19937, `random()`, `randrange`, `choices`) seeded with the game seed
  (`--game-seed N`: also the drivers' engine seed of N), so C++ explore plays exactly the Python
  tier's games: 660 seeded bo1 games and bo3 matches (explore and explore:patient, 20 league
  matchups) identical. The C++ resolver gives every one of the 98 scenarios the harness's
  `resolved --actions` list, with their assertions passing on the engine's own output. New flags:
  `--hand-a/-b`, `--library-a/-b`, `--game-seed`, `--max-decisions` (recorded in the log),
  `--check-decks` (`--check-decks-dir`), `--trace-decisions`; the zone presets were already engine
  flags. Gate tier `cppagents` (`train/compare_cpp_agents.py`). Release: C++ explore in-process
  ~750 decisions/s (4.6 games/s; ~1,090 with `--knowledge-a/-b off`) vs Python explore over the
  driver ~470 (2.9 games/s); the frame encoding and the knowledge sidecar are most of the cost.
- **Gate driver `robomage_check`.** A separate C++ binary that links no engine TUs (built with
  `-fno-exceptions` like the engine: decided, no exemption). It runs engine
  subprocesses in parallel. It reads the existing 70 scenario JSONs unchanged: their schema is
  flat, and all 271 regexes use no Python-only syntax, so `std::regex` applies. It also ports
  the transcript scan (`train/ci_check.py:265`).

| Tier | Needs Python? | What it does |
|---|---|---|
| `vocab` | no | `--check-decks` over top-level + `league/` decks |
| `concede` | no | pipes `-2`/`-3` at bo1/bo3 decisions; checks result lines and replay |
| `scenarios` | no | the 70 rules scenarios |
| `replay` | no | replays the corpus RMLOGs; diffs output against baselines |
| `state` | no | the step-1b self-checks over replayed and fuzz games |
| `knowledge` | no | the step-1c soundness / exactness / isolation / snapshot checks |
| `fuzz` | no (C++ explore) or yes (Python explore) | live explore-vs-explore league games; draws are findings |
| `smoke` | **yes** | live scripted-hard league games through the Python driver |
| `bench` | no | throughput of a fixed fuzz workload |

- **Corpus format change.** The corpus becomes RMLOG decision logs recorded from scripted-agent
  games, with the engine's `--replay` output as baseline. Checking it needs no Python;
  re-recording it uses the live agent.

  **Status: part 2 done** (the gate driver; `docs/ci.md`, "The C++ gate"). `tools/robomage_check/`
  → `bin/<config>/robomage_check`, part of `make`, built like the engine with `-fno-exceptions` (no
  exemption: a scenario regex is compiled in a forked child first, so a malformed one is reported,
  not an abort); it links only the std-only decoder, schema, frame reader and `PyRandom`. All nine
  tiers are implemented; every one but `smoke` (the Python driver, skipped without an interpreter)
  is Python-free. Parity with the Python tiers: all 98 scenarios pass on both runners, and of 21
  mutated scenarios (every assertion key, variants, a `--play` typo, malformed files, controls) both
  give the same verdict on 20; the other is a difference of the engine runs, not of the runners (a
  seat's own hand handles follow the event order when an event consumer is on, the in-process
  agents being one; `docs/state_format.md` §6). The C++ explore and stream sessions play the Python
  tiers' games: the `knowledge` and `state` parts report the same decisions, facts, records,
  isolation frames, truth facts, shadow tallies, name-a-card candidates, archetype summaries and
  calibration figures as the Python ones, and the `fuzz` tier the same results. The
  corpus is now `train/regression/corpus/<name>.rmlog` (recorded by `replay_diff.py record` through
  the new output flag `--decision-log FILE`) and `<name>.replay` (`--replay --machine --narrative
  --trace-decisions`, the log path and the trace's board lines left out; ~3.4 MB for 64 games),
  identical from the debug and release builds; the Python transcripts stay beside them until 3b.
  `make check` runs the C++ gate as the Python gate's tier `cppgate` (every tier but `smoke`, fuzz at
  2 games per mirror; ~4.5 min of `make check`'s ~13.5); `make check-cpp` runs it alone. Not
  ported, as they test Python code: the Python decoder's own checks (`rmstate`'s Python-vs-C++ JSON
  and refusals), `test_concede.py`'s `GameDriver` checks and the tiers comparing Python with C++
  (`cppagents`, `agentparity`, `rmagent`).

### 3b. Strip

The corpus must stay byte-identical across 3b.

- Switch the gate: `make check` runs `robomage_check` (every tier; `smoke` through the Python
  driver), and `train/ci_check.py`, `regression/replay_diff.py`'s `check` mode and the corpus
  transcripts `<name>.txt` go (`record` stays: it writes the decision logs and baselines with the
  live agent). The `.rmlog` / `.replay` files move with the corpus (the baselines name only the
  log's file name, so the move changes no byte).

- Delete `src/actor/`'s old-observation code (`obs_builder`, `az_mcts`, `menu_merge`,
  `sb_rules`, `td_targets`, `root_diag`, `npz_writer`, `oracle_client`, `az_actor_main`).
  **Keep the libtorch plumbing step 5 reuses:** the Makefile's `ACTOR_*` rules
  (exceptions-enabled TUs, libtorch detection and linking) and `az_evaluator.cpp`'s TorchScript
  loading, which becomes the base of the step-5 evaluator. `az_mcts.cpp` stays in git history
  as a reference for a future search agent.
- Delete everything in `train/` except the kept agent, driver and decoder (list below). Move
  `train/regression/{scenarios,corpus}` to `tests/`.
- Delete the `pygen` target and every `gen_*.py`; `src/gen/`; `train/sb_dead_rules.json`. **Keep
  `bin/resources/decks/archetypes.json`**: it is the reference metagame's source (its `_archetypes`
  label list and deck tags, read by `src/state_format/metagame.cpp` for the `METAGAME` frame and the
  sidecar's archetype inference, `knowledge_store.md` §10); only the codegen around it goes
  (`gen_archetypes.py`, `src/gen/archetypes_gen.h`, the BQUERY observation's matchup tail).
- Delete `src/search_server.{h,cpp}` and its hooks (cooperative unwind and `SEARCHINFO` in
  `input_logger.cpp`, `search_intercept_game_end` / restore handling in `game_driver.cpp`, the
  flag in `cli_args.cpp`). **Keep `src/snapshot.{h,cpp}`** (search agents, undo, analysis;
  tested by the `snapshot` self-check).
- Delete the BQUERY payload: float packing, the `N_CARD_TYPES` normalisation, `OFFSET_CHAIN` /
  block widths, `BSTATE` and `--broadcast-steps`. `--machine` now means RMSTATE. Keep
  `populate_gamestate` / `GameState` only as the CLI printer's source until step 4b.
- Delete the RL shaping signals `MANA_WASTED` (`src/mana_system.cpp:138`) and
  `MULLIGAN_PENALTY` (`src/game_driver.cpp:796`).
- Delete the engine-side knowledge bookkeeping, now replaced by the sidecar:
  `Game::KnownLibraryTop` and its `known_top_library_*` methods, `Zone::identity_known`, the
  match revealed bits in `src/classes/match_state.cpp`, and the `opp_known_hand` pass. The one
  schedule-affecting reader, the menu collapse in `src/queries/zones.cpp:123-128`, keeps its
  current behaviour until 4a redefines it.

  This is the one 3b item that touches the decision schedule. If that reader can't be kept
  exact without the bookkeeping, move this deletion into 4a.
- Delete `tui.sh`, `gui.sh` and `train/gen_corpus.sh`.
- Infrastructure:
  - CI (`ci.yml`, `nightly-fuzz.yml`, `.github/actions/setup-robomage`): drop the
    numpy/gymnasium venv; `python3` is needed only for provisioning and the `smoke` tier.
  - SessionStart hook: `python3 tools/forge_fetch/provision_decks.py`.
  - `.claude/settings.json`: drop the `train/` permissions.
- Docs: rewrite CLAUDE.md, README, `docs/ci.md` and `docs/game_running.md`. Keep
  `docs/rules_deviations.md` and the CR files.

**Status: part A done** (the gate switch). `make check` is the build plus every `robomage_check`
tier (~5.5 min on 4 cores, debug, of which the gate ~5.2; was ~13.5); `make check-cpp` is gone, and
`make` no longer runs `pygen` (only `make actor` does: `src/gen/` is read only by `src/actor/`, the
other outputs only by the Python RL stack). `train/ci_check.py` is deleted with the engine-test
scripts only it ran (`test_concede`, `test_scenarios`, `test_selfcheck_state`, `test_rmstate`,
`test_knowledge`, `test_rm_agent`, `test_agent_parity` with its `regression/agent_parity*`,
`compare_cpp_agents`); the other `test_*.py` (RL code) are unrun and go with `train/` in part C.
The kept Python stays tested by `python/pycheck.py` (stdlib only), run by the `smoke` tier beside its
games: `rmstate.py`'s decode of three dumps (an in-process bo1 game and bo3 match, a driver bo3
session under `--closed-decklist`) equals the engine's `--state-dump json` and the live `.jsonl`,
the two decoders' refusals of mutated frames agree, the driver's engine seeds are the pinned ones
and a driven game imports nothing outside the kept Python (numpy included). The scenarios, the
corpus and `knowledge_divergences.json` moved to `tests/` (`tests/scenarios/`, `tests/corpus/`,
`tests/knowledge_divergences.json`, deleted in part B); the corpus transcripts `<name>.txt` are deleted and
`replay_diff.py` is `python/record_corpus.py` (`make regen`), which records only the decision logs
and, through `robomage_check --tier replay --record`, their baselines (re-recording reproduces
every byte). CI: `ci.yml` runs `make check`; the nightly fuzz runs `robomage_check --tier fuzz
--matchups all` (40 games, explore and explore:patient); its actor-parity and full-check jobs went
with their runner; `setup-robomage` installs no venv (python3, stdlib only: provisioning and
`smoke`). **Decided: the kept Python lives in `python/`** after 3b (`rm_driver.py`,
`scripted_core.py`, `rm_agent.py`, `explore_agent.py`, `rmstate.py`, `record_corpus.py`,
`pycheck.py`; the move is part C's, with the smoke tier's paths, `find_python`'s venv path and
`make regen`; `rmstate.py`'s CLI drops its `cli_spec` import). Part B note: the replay
baselines hold no BQUERY bytes, but their command is `--replay LOG --machine --narrative
--trace-decisions`, so `--machine`'s output and decision-schedule semantics in replay must survive
the BQUERY deletion unchanged (or the corpus command change must be byte-neutral); the `concede`
tier's two BQUERY legs are marked in `tier_concede.cpp`.

**Status: part B done** (the engine-side strip; one commit per item, each built debug and release
without warnings and passing `make check` with the corpus byte-identical; ~11,700 lines deleted).
(1) `src/actor/` keeps only `az_evaluator.{h,cpp}` (its layout includes now name
`src/actor/az_layout.h`, which freezes the 6922-float input width the existing checkpoints were
exported with, and `card_vocab.h` for `N_CARD_TYPES`) and the Makefile's `ACTOR_*` plumbing;
`make actor` compiles the actor objects (the step-5 binary links `$(ENGINE_OBJ_NO_MAIN)
$(ACTOR_OBJ) $(ACTOR_LDFLAGS)`), `actor-syntax` is gone (no torch-free actor TU is left).
(2) `pygen`, `train/gen_*.py`, `gen_util.py`, `src/gen/`, `sb_dead_rules.json`, `train/_enums.py`
and the untracked outputs' ignore lines are gone; `archetypes.json` stays.
(3) `src/search_server.*` and its hooks are gone, with the determinizer's helpers
(`collect_pending_pins`, `deck_state_resplit_live`); the loop-safe marker moved to `snapshot.h`
(the snapshot self-check saves only at a loop-safe decision).
(4) **`--machine` now means RMSTATE**: outside `--replay`, every seat with no in-process agent
that is not the `--player` seat decides over the RMSTATE stream (stdout the frames, all other
output on stderr); `--replay … --machine` reads every decision from the log, writes no stream and
prints exactly what it did. The decision schedule `--machine` selects (`is_machine_mode`,
`is_machine_schedule`, the RMLOG `machine` marker) is untouched — 4a's to change.
`--machine-format rmstate` stays as a spelling of `--machine`; `--machine-format bquery`,
`--broadcast-steps` and `--search-server` fail with "was removed" (a removed-option table in
`cli_args.cpp`). The BQUERY payload, `populate_query` / `Query`, `serialize_state`, the layout
constants and `OFFSET_CHAIN` are gone; `machine_io.{h,cpp}` is the CLI board's slimmed
`populate_gamestate` / `GameState` (until 4b) and `action_card_vocab_idx`; `N_CARD_TYPES` lives in
`card_vocab.h`. (5) `MANA_WASTED` / `MULLIGAN_PENALTY` are gone (no baseline held them).
(6) **Done exactly, not moved to 4a.** The menu collapse read only whether the library owner's
opponent knew the card at each of the top five positions (`by_opponent[p] != -1`); it now reads
`Game::OppKnownLibraryTop`, one bool per position, kept by the same push / remove / note / move /
clear operations at the same sites with each written value mapped to `v != -1` — the same
predicate by construction. Deleted: `KnownLibraryTop`, `Zone::identity_known`, the match revealed
bits, the `opp_known_hand` pass and the shadow comparison (`shadow_knowledge.*`, `SHADOW:` lines,
the gate's classifier and shadow part, `knowledge_divergences.json`; the five
`knowledge_divergence_*` scenarios stay, their assertions being `k_*` sharpness facts).
`mark_card_revealed` moved to `classes/reveals.{h,cpp}` with its library part only. Evidence
besides the corpus: 3,030 C++ explore games (bo1 `explore` and bo3 `explore:patient`, the ten
league mirrors and 20 crosses, 100 seeds each; 448,878 decisions) give byte-identical decision
logs, and identical stdout and stderr but for the deleted shaping lines, before part B and after
items 6 and 4. (7) The `concede` tier's BQUERY leg pipes the answered session's replies, unread,
into plain `--machine` and checks the frames are byte-identical (140 checks, was 134).

**Status: part C done** (Python and infrastructure). `train/` is gone: 97 files (~75,000 lines)
deleted with `tui.sh` / `gui.sh` (all on branch `archive/rl-stack`), and the seven kept files moved
to `python/` (stdlib only, run with `python3`; `rmstate.py` builds its own CLI parser and lost its
numpy `Schema.arrays`, `rm_driver.py` runs plain `--machine`, `pycheck.py`'s driver check fails on
any module from outside the standard library and `python/`). `robomage_check` runs
`python/rm_driver.py` / `python/pycheck.py` with the first `python3` on `PATH` (no venv); `make
regen` runs `python/record_corpus.py`, `PYTHON ?= python3`, and `make actor` finds libtorch through
`$(PYTHON) -c 'import torch'`. The SessionStart hook only runs `python3
tools/forge_fetch/provision_decks.py`; `.claude/settings.json` lost its two `train/` permissions;
`.gitignore` its `train/` entries. The docs rewrite followed: CLAUDE.md, README,
`docs/ci.md` and `docs/game_running.md` describe the engine-only project, and the state-format,
knowledge and history docs had their statements about BQUERY, the shadow comparison and `train/`
paths corrected.

## Step 4 — One decision schedule, then the C++ CLI

### 4a. One rules-faithful decision schedule (intentional; re-record)

| Site | What it does | Fate |
|---|---|---|
| `state_manager_actions.cpp:104` `machine_masks_conditional_destroy` | hides Fatal Push-style casts with no currently passing target ("NOT a rules gate") | **delete** |
| `state_manager_actions.cpp:1085`, `mana_system.cpp:1547` | hides ordinary mana abilities at priority; auto-pays costs | → explicit option `auto-mana`, **on by default for every consumer**; `--no-auto-mana` turns it off |
| `action_processor.cpp:2201` | auto-resolves hybrid pips | part of `auto-mana` |
| `game_driver.cpp:1306` | caps sideboard swaps per player | keep as an always-on safety limit |
| `game_driver.cpp:607`, `cli_output.cpp` guards (`print_query`) | output only | fold into the output mode |
| `name_card_choices.cpp:62` | truncates open-decklist name-a-card menus to `MAX_ACTIONS` (64, `classes/gamestate.h`) | lift (menus are unbounded) |
| `queries/zones.cpp` (`interchangeable_cards`) | keeps two identical library cards apart when one sits at a position the other player knows (engine memory: `Game::OppKnownLibraryTop`, kept by 3b for this reader alone) | redefine on currently explicit information: a card in a live reveal window is distinguishable, a remembered one is not; `OppKnownLibraryTop` then goes |

- **`auto-mana` is on by default for everyone:** CLI, scripted agent and machine agents.
  - With it on, ordinary mana abilities stay hidden at priority and the greedy payer
    (`auto_pay_mana`) pays costs and resolves hybrid pips. Two kinds stay offered so mana can
    still be floated: instant-speed sources, and sources targeted by removal on the stack.
  - `--no-auto-mana` turns it off for the match (match-wide; no per-seat variant).
    With it off, every mana ability is listed at priority and costs are paid through the
    interactive payment loop, for any consumer.
  - The setting is recorded in the RMLOG header (`record_header_flags`, `src/cli_args.h`), so
    the existing corpus (recorded with auto-pay) keeps replaying.
- **The frame shows what is owed during every cost-payment decision:** the remaining cost by
  pip type, and what it is paying for (`state_format.md` §4, `pending_cost`). This covers
  manual payment with `--no-auto-mana`, and also decisions that happen with auto-mana on:
  pitch and alternative costs, unless-payments, X.
- **Not adopted for now:** a "pay automatically" first menu choice inside the payment loop.
  It could be revisited later as another option.
- After 4a, machine mode means only "speak RMSTATE".
- The agent's positional menu assumptions (read map §2b) are kept, or changed together with
  the agent.
- Re-record the corpus with the live agent, and add explore games with `--no-auto-mana` so
  manual mana payment is covered.

**Status: done** (part 1 the schedule, part 2 `interchangeable_cards`, part 3 the corpus
re-record and the manual-payment coverage). `machine_masks_conditional_destroy` is deleted: a Fatal Push-style spell is
offered whenever it has a legal target (CR 601.2c) and its condition is checked on resolution.
`auto-mana` is the match-wide launch option `auto_mana` (`game_driver.h`), on unless
`--no-auto-mana` (a setup option in the `cli_args.cpp` table, so the RMLOG `FLAGS` line carries
`no-auto-mana` and a header without it replays with auto-mana on); it gates the priority-menu
mana abilities, `auto_pay_mana` in `prompt_mana_payment` and the hybrid pips. With it off, the
payment loop's last row "Cancel casting" is offered to every consumer (it reverses the spell or
ability and counts toward the `payment_blocked` guard, now also for suspend and the companion
special action), and a cost-bearing source is offered there only while the pool covers its
activation cost, so every other row makes progress. The sideboard-swap cap is always on.
`InputLogger::is_machine_schedule` and the replay's `machine` reading are gone; the RMLOG
`machine` marker is still written and is informational only, and `is_machine_mode` gates only
I/O (the stream, the agents' provider, the CLI board and prompt output). The open-decklist
name-a-card menu is no longer truncated. Scenarios `cr608_2c_fatal_push_*`,
`cr605_3a_no_auto_mana_manual_payment`, `cr605_3a_auto_mana_hides_land_mana_ability`. Evidence:
C++ explore games with `--no-auto-mana --selfcheck-knowledge` over the ten league mirrors and
crosses (100 bo1 `explore`, 60 bo1 and 60 bo3 `explore:patient`) end with no error and no draw.
Of the corpus only `lg_bug__reanimator` drifts (Fatal Push is now offered against a lone
Atraxa); the re-record follows. Part 2: `interchangeable_cards` keeps two identical library cards
apart only while some seat sees one of them and not the other (`identity_visible` per seat: a live
reveal window); what a seat remembers of a closed window is its sidecar's. `Game::OppKnownLibraryTop`
and its orderer bookkeeping (`LibraryTopView`, `note_library_card_known`), `Game::revealed_in_library`
and `mark_card_revealed` (`classes/reveals.*`), which only fed the collapse, are deleted. Scenarios
`menu_collapse_remembered_library_copy` (a Bauble look that has ended no longer splits two Forests)
and `menu_collapse_revealed_library_copies` (two copies both players see in a reveal merge). With
the current card pool no library menu meets a one-sided live window (library windows last one
resolution), so the "kept apart" branch has no scenario. Corpus drift from part 2:
`doomsday_mirror` (Doomsday's rearrange: a Street Wraith that came from the graveyard and one from
the library merge) and `lg_gw_maverick__reanimator`, `lg_reanimator__reanimator` (Atraxa's reveal:
two revealed copies merge). Evidence: 240 C++ explore games with `--selfcheck-knowledge` (eight matchups
of the Doomsday, reanimator, BUG, Lands, Delver, control and Maverick decks; 10 seeds each, bo1
`explore`, bo3 `explore:patient` and bo1 `--no-auto-mana`; 95,604 decisions) end with no error and
no draw; the 43 whose decision log differs from part 1's first diverge at a menu that lost a
duplicate row. Part 3: the corpus re-recorded with the live agent (`make regen`) changes exactly
the four drifting games, no result: the baselines of `lg_bug__reanimator` (the two Fatal Pushes in
the menu), `lg_gw_maverick__reanimator` and `lg_reanimator__reanimator` (the merged reveal rows),
and one logged decision of `doomsday_mirror` (Thassa's Oracle one row up). Manual payment is
covered by two corpus games recorded explore vs explore with `--no-auto-mana`
(`manual_doomsday_vs_delver`, `manual_mav_mirror`; the scripted agent pays with the payment loop's
first row, often the wrong colour of a dual, so it records none) and by the `fuzz` tier's
`--no-auto-mana` leg: `--manual-mana-games` (default 2) bo1 `--mode` games per ring matchup, 20
games and about 13 s of the debug gate on 4 cores; the nightly sweep plays 40 per ring matchup.

### 4b. C++ CLI on the decoder

- **The CLI renders only the decoded frame of the human's seat**, never engine internals, so
  any display bug is a format bug. It shows players, battlefield, hand, stack with targets,
  graveyards and exile, mana pool, the pending decision, the numbered menu, and a
  **knowledge panel** from the `k_*` tables ("you know / you can deduce"; empty under
  `--knowledge off`).
- **Commands:** an index, `concede` / `concede match`, `board`, `zone <name>`, `card <n>`,
  `know`, `log` (viewer-redacted narrative), `help`.
- **Seating:** human vs in-process agent, and hotseat. Human vs the Python scripted agent
  (external) needs a second channel (e.g. `--machine-fd`); decide here whether it is wanted
  now. Today a `--player` seat under `--machine` (which any `--agent-a/-b` implies) is prompted
  without its menu (`print_query` returns early in machine mode), so human vs agent is not
  playable until this step.
- **Delete:** `GameState`, `populate_gamestate`, `print_game_state` and the remains of
  `src/machine_io.*`. `--trace-decisions` switches to the new renderer.
- **Test (tier `cli`):** golden transcripts of scripted stdin sessions.

**Status: part 1 done** (the CLI on the decoder). A human seat is an in-process agent
(`cli::HumanAgent`, `src/cli/human.h`) handed its seat's frames by the in-process frame writer
(`state_format/agent_feed.h`) exactly as the C++ agents are; it renders them, decoded, with
`src/cli/board.h` (standard library and the agents' frame reader only) and reads its commands from
stdin. Seating (`cli_args.cpp`): the `--player` seat is human, and so is every seat without an agent
when `--machine` is not given, so a plain run is hotseat; a human seat may take over a `--play` /
`--actions` script's seat once it runs out. A human seat turns machine mode on (no narrative
without `--narrative`) and always writes the decision log; no seat may be left to the RMSTATE
stream beside it (decided: no second channel, so no human against the external Python agent). The
board shows the players (life, zone sizes, pool, counters, designations), both battlefields (P/T,
statuses, counters, attachments, combat; identical permanents merged), the stack with targets,
the seat's hand (cost, type line), graveyards, exile, open reveals, effects and delayed triggers,
a "You know" summary from the `k_*` tables (cards known in the opponent's hand, known library
positions, the opponent's archetype; nothing under `--knowledge off`), the pending decision with
`pending_cost`'s remaining pips, and the menu (row *i* = reply *i*, a confirm last row marked as
`-1`'s). Objects are labelled by their frame handle. Commands: a row, `concede` / `concede match`,
`board`, `zone <name> [opp]`, `card <#handle|name>` (the `CARDS` frame), `know` (every `k_*` and
`b_*` fact), `log [n|all]` (the seat's own events since match start, rendered from its frames, and
life changes between its frames), `help`, `quit`; end of input quits. The terminal path of
`InputLogger` (`print_query`, `get_int_input`, the `z` auto-pass and `q` keys) is gone: every
decision outside `--replay` is the provider's or the stream's. `--trace-decisions` prints the same
board and pending lines; its header, `| menu:`, menu-row and choice lines keep their shape, and the
corpus baseline filter recognises a menu row exactly (`|   <i>: `), so the corpus replays
byte-identical and the scenarios pass unchanged. Left for part 2: `GameState` /
`populate_gamestate` / `print_game_state` now serve only the board a `--replay` without
`--machine` prints; the `cli` tier.

**Status: part 2 done; 4b and step 4 are done.** `GameState`, `populate_gamestate`,
`print_game_state` and `src/machine_io.*` are deleted. A plain `--replay` prints the narrative and
the replayed inputs, no board (its output is the output flags'); the corpus replays under
`--machine`, so its baselines are unchanged. What they still provided moved to its users:
`source_vocab_idx` (`queries/characteristics.h`), `SIDEBOARD_MENU_LIMIT` (`game_driver.cpp`) and
`ACTOR_MAX_ACTIONS` (`actor/az_layout.h`). Auto-pass is back as a frame-only command: `z` (`pass
turn`) answers the pass row until the next turn begins, `z mine` until the seat's own next turn;
it starts only at a plain priority decision and stops at the first decision of that turn, at any
other decision (mandatory choices, attackers, blockers, targets, payments, mulligans) and when the
opponent puts a new object on the stack. The gate's `cli` tier (`tools/robomage_check/tier_cli.cpp`)
plays the scripted stdin sessions of `tests/cli/<name>.json` and compares each stdout with
`<name>.out` byte for byte (`--record` rewrites them): human vs explore, hotseat, manual payment,
auto-pass and a bo3 match conceded.

**Exit:** the C++ engine, RMSTATE with knowledge tables, the C++ CLI, the C++ gate, and the
kept Python below.

## Step 5 — C++ featurizer and in-process actor

Target consumer: a **C++ in-process actor**. In one process it runs engine → frame →
knowledge → featurizer → TorchScript policy → action, with no Python per step.

**The card tables are for any ML framework, not only this actor.** The oracle-text vectors (5c)
and the card and ability feature tables (5a) are plain data files, not part of a model:
whole-vocab matrices indexed by vocab id, readable with no ML library. Plausible consumers are
the step-5 TorchScript policy, a learner in PyTorch, JAX or TensorFlow, non-neural models
(gradient-boosted trees, linear probes), analysis tools (nearest neighbours, clustering cards
into roles) and the stdlib-only Python. libtorch is how 5c computes the vectors and how 5d runs a
policy; neither the files nor their consumers depend on it. A model may copy the tables into its
weights as frozen buffers, but the exported file stays the interface and the source of truth.

### 5a. Design (`docs/featurizer.md`)

- **`FEATURE_SPEC`.** Announces every tensor's shape and field index, like `SCHEMA` does for
  tables: a JSON document (`rmfeat --spec`), so consumers outside C++ read it with any JSON
  parser; its hash is in every exported table and model. The design is `docs/featurizer.md`.
- **Tensors per decision:**
  - entity tokens: objects, players, globals and knowledge groups / unseen cards; numeric
    features scaled by their schema tag, plus categorical embedding ids;
  - entity mask;
  - relations `(src, dst, type)` from handle columns;
  - action tokens with gather indices to their source/target entities, for a pointer head;
  - action mask;
  - card references as vocab ids: an object's `card`, `copy_of` and `chosen_name`, a menu row's
    `source_card` and the like are row indices into the card tables, and an `object_ability`
    row's `abref` is a row index into the ability table. No card vector travels per decision.
- **Card tables: the whole vocab, indexed by vocab id** (not per match). Every card and
  registered token of `src/card_vocab.h` has a row whatever the match's decklists, so a `card`
  column is the row index with no per-match remap. Face down (`0xFFFF`) and the token sentinel
  get rows of their own.
  - **Card table** `[N_CARD_TYPES × printed features]`, from `CARDS`' `card` rows: cost pips,
    mana value, colours, types, supertypes, subtypes, P/T, loyalty, keywords, layout, token. A
    double-faced card's back face has its own vocab id (`face_card`), so each face has its own row.
  - **Text table** `[N_CARD_TYPES × 384]`: the oracle-text vectors (5c).
  - **Ability table** `[abilities × ability features]`: one row per `card_ability` row of every
    vocab card (kind, category, timing, tap, limit, and the structured parameters below), with
    an index `(card, ab_index) → row` matching `abref`, so a permanent's current abilities point
    at their rows. A card has a variable number of abilities, so they are a table of their own,
    not card-table columns; the model gathers a card's or permanent's ability rows and pools or
    attends over them. Abilities with no card row (`ab_card` none: granted or engine-built) get
    rows keyed by `ab_key`, from a fixed key list in the spec.
  - **Exported** by a tool as versioned files: a small header (table kind, vocab hash,
    feature-spec version, encoder model id and text-normalisation version for the text table,
    row count, width), then little-endian rows in vocab-id order (float32 for vectors and scaled
    features, integer columns for categorical ids), readable with the standard library alone.
    Recomputed only when an input changes (a `FORGE_PIN` bump, a vocab change, a feature-spec or
    encoder change).
- **State-format change: structured ability parameters in `CARDS`** (a new register in
  `docs/state_format.md` §11). Today an ability's effect parameters exist only in
  `card_ability.line` (raw script text) and in the oracle text: for Lightning Bolt the row says
  spell, instant, `DealDamage`, but the 3 and "any target" are not columns. `card_ability`
  gains them, read from the parsed `AbilityDef` and its typed params (`ability_params.h`),
  generic per effect category and never per card:
  - **amount:** the effect's number (`NumDmg$`, `NumCards$`, `CounterNum$`, `NumAtt$`/`NumDef$`,
    `PeekAmount$` …) with a kind: fixed, variable (an X or count-SVar) or none;
  - **targeting:** what may be targeted (a mask over creature, other permanent kinds, player,
    spell, ability, from `ValidTgts$`) and the target min/max;
  - **structure:** `SubAbility$` chain links and `Charm` modes (with `CharmNum$`) as rows of a
    child table of `card_ability` (each part with its parent, order, category and the same
    parameters), so a chained or modal ability's parts are each described and `abref.ab_index`
    keeps its meaning.

  The exact columns are fixed in 5a and Appendix A of `docs/state_format.md`; the encoder, both
  decoders, the JSON renderer and the self-checks move with them.
- **Network contract:** the TorchScript `forward` signature (policy logits over menu rows,
  value), with the card tables loaded from the exported files as frozen buffers.
- **Padding:** runtime options, not format caps.
- **Recommended model family**, for whoever builds a learner: entity transformer with relation
  biases, pointer action head, `expert` knowledge as tokens.

### 5b. C++ featurizer (`src/featurize/`)

- A pure function of one viewer's decoded frame, knowledge tables and card table, driven by
  the schema's semantic tags. It includes only the state-format decoder and the standard
  library (gate-checked).
- **Tests:**
  - deterministic;
  - shapes and indices agree with `FEATURE_SPEC`;
  - invariant to row order where the format says order is meaningless;
  - every relation and gather index resolves.

### 5c. Oracle-text embeddings computed in C++ (option 2)

- **What it is:** a sentence embedding, one fixed-length vector per text, placed so that texts
  with related meaning lie close together (after normalisation, cosine similarity is the dot
  product). It carries what a card broadly does (removal, card draw, a counterspell, a dies
  trigger); exact numbers and fine rules distinctions are weak in it ("deals 3" ≈ "deals 2"),
  which is why the card and ability tables' structured columns (5a) sit beside it.
- **Text: oracle text only**, per vocab entry: the type line + `oracle` (`CARDS`). Each face of
  a double-faced card is its own text and row (the back face under its `face_card` id); tokens
  use their own oracle text; the card's own name is replaced by a placeholder. Script text
  (`--rmstate-card-scripts`) is not embedded: the Forge DSL tokenises poorly with an English
  vocabulary, scripts run past the encoders' windows, and what they add over the oracle text is
  carried by the structured ability parameters. Oracle texts fit (the longest is ~470
  characters, ~120 tokens); the tokenizer test asserts that none is truncated.
- **Model:** a small pretrained sentence-embedding encoder (BERT-family, ~22–33M parameters,
  384-d), e.g. bge-small-en-v1.5 or all-MiniLM-L6-v2. Final pick and licence check in 5a.
  Pooling follows how the chosen model was trained (all-MiniLM-L6-v2: the mean over real
  tokens; bge-small-en-v1.5: the `[CLS]` vector).
  - It runs only to generate missing or stale rows (below), never in the actor, which reads the
    committed table. The generator `rmembed` implements the BERT encoder in the libtorch C++
    API over the model's safetensors weights (no TorchScript export, no Python), so libtorch
    stays the only ML dependency; golden token ids and vectors from the reference
    implementation check it (`docs/featurizer.md` §5).
  - The weights (~25–130 MB) are fetched on demand at a pinned revision (a pin file beside
    `tools/forge_fetch/FORGE_PIN`'s, naming the model repository and its commit), never committed.
- **Tokenizer:** C++ WordPiece reading the model's vocabulary, checked against a golden file of
  reference token ids.
- **Pipeline:** tokenize → encode → pool → L2-normalise, giving one 384-float vector per vocab
  entry, for the **whole vocab every time** (~330 entries, ~0.5 MB), written as the text table's
  exported file (5a).
- **Cache and determinism:** the exported table is the source of truth. Each row is keyed by
  the model id plus a hash of its normalised text (the normalisation version included); only a
  stale row is recomputed. A frozen encoder is deterministic in principle, but the last float
  bits can vary with CPU instruction set, BLAS, thread count or libtorch version, so consumers
  read the file and never recompute it; that makes runs bit-exact across machines.
- **The text table is committed** (`bin/resources/embeddings/oracle_text.rmtb`, ~1.5 MB dense
  over the vocab ids; layout in `docs/featurizer.md` §4.4), and **`make` keeps it complete**:
  - Every `make` runs a check (std only, no ML library, no network; in the engine binary beside
    `--check-decks`) that the table has a row for every vocab entry with oracle text, each row's
    text hash matching that entry's current normalised text, and that the header's model id
    matches the model pin.
  - When rows are missing or stale (a card added to the vocab, a `FORGE_PIN` bump that changed
    an oracle text), `make` downloads the pinned model and generates **only those rows**,
    rewriting the table in place; the rows already in it are kept bit for bit. The rewritten
    file shows in `git status` and is committed with the change that caused it. A model-pin
    change makes every row stale, so it regenerates the whole table.
  - Generation needs libtorch (found as `make actor` finds it) and network access. When either
    is missing, `make` fails naming the stale rows and the command to run; it never builds on
    with an incomplete table. A clean checkout with a current committed table never downloads
    the model or needs libtorch, so CI and plain `make` stay as they are; a stale table in CI
    fails there, which is what catches a change committed without its regenerated rows.
- **Sanity report:** nearest neighbours per card (Bolt ↔ Chain Lightning, Brainstorm ↔
  Ponder) to eyeball quality.

### 5d. In-process actor (`src/actor/`, exceptions-enabled TUs)

- **Evaluator:** loads a TorchScript policy (from the kept `az_evaluator.{h,cpp}` base, whose
  comments and frozen input width `az_layout.h` still describe the archived observation; both are
  rewritten here) and checks that its card tables' vocab hash and feature-spec version match
  the engine's (one check per build, since the tables cover the whole vocab). The binary links
  the Makefile's kept plumbing: `$(ENGINE_OBJ_NO_MAIN) $(ACTOR_OBJ) $(ACTOR_LDFLAGS)`, with the
  exceptions-enabled `ACTOR_CXXFLAGS` for `src/actor/` only.
- **Agent:** `nn:<model.pt>` on the step-3 Agent interface: featurize, evaluate, choose
  (argmax or sample) over menu rows.
- **Batched-evaluation hook** for a later search agent.
- **End-to-end test** with a randomly initialised model of the contract's shape.
- **Exit:**
  - `--agent-a nn:<model.pt>` plays full bo3 matches without errors;
  - the featurizer, tokenizer and embedding tests pass;
  - the committed text table passes `make`'s completeness check on a clean checkout, and
    generating a deleted row reproduces the same rows elsewhere in the file untouched;
  - the `CARDS` ability parameters pass the state tier's self-checks, and the exported card
    tables load with a standard-library reader (no ML library) to the same values the actor uses;
  - throughput is measured in the `bench` tier.

### Possible later addition: learned text encoder (option 3)

A learner could make the text encoder part of the policy network, initialised from the same
pretrained encoder and fine-tuned by the RL loss. Gradients from the policy and value losses
flow into its weights. The mitigations for having only ~330 distinct texts are:

- a low learning rate and staged unfreezing;
- text plus id embedding with id dropout;
- auxiliary losses predicting `CARDS` features;
- held-out-card checks.

The actor side is unchanged: the learner exports the recomputed whole-vocab
`[N_CARD_TYPES × D]` text table per checkpoint, in the same file format (its model id names the
checkpoint), and actors and every other consumer still only look it up. The encoder runs inside
the learner's framework while it trains; the exported table does not depend on that framework.
This needs a learner, which is outside this plan.

---

## Python inventory: what remains

**Status: done** (3b part C). Everything kept is stdlib only and run with `python3` (no venv).

| File (location since 3b) | Role | Notes |
|---|---|---|
| `tools/forge_fetch/fetch_script.py`, `fetch_deck_scripts.py`, `provision_decks.py` | pinned Forge script provisioning (CI, SessionStart hook) | stdlib only; no `train/` reference left |
| `tools/meta_scraper/scrape_meta.py` | MTGTop8 → `.dk` scraper | stdlib only; kept (open decision 3) |
| `python/scripted_core.py`, `python/rm_agent.py`, `python/explore_agent.py` (was `train/scripted_agent.py`) | the realistic opponent and the Python explore tier (also ported to C++, `src/agents/explore.*`) | on the Python RMSTATE decoder; knowledge comes from the C++ sidecar's tables; no numpy |
| `python/rm_driver.py` (replaces `env.py` / `runner.py` / `opponents.py` for agent games) | engine subprocess, frame I/O, match loop, its own CLI | runs the `smoke` tier's games |
| `python/rmstate.py` (replaces `decode.py` for the agent) | bytes → objects, knowledge tables included | no copied constants (schema frame) |
| `python/record_corpus.py` (was `train/regression/replay_diff.py record`) | records the replay corpus with the live agent | `make regen` |
| `python/pycheck.py` | the kept Python's own checks | run by the `smoke` tier |

There is no Python knowledge tracker (memory and inference live in the C++ sidecar) and no
Python in the embedding path (step 5 computes it in C++). The engine builds, runs and passes
every gate tier except `smoke` without Python (explore is in C++). Recording the corpus needs the
agent.

**Ported to C++, then deleted:**

| Python | C++ replacement (step) |
|---|---|
| `train/ci_check.py` engine tiers, `train/regression/replay_diff.py`, `train/test_concede.py` | `robomage_check` tiers (3a) |
| `train/missing_cards.py` | `--check-decks` + `vocab` tier (3a) |
| `train/action_spec.py`, `PlayController` / `ActionListController`, `train/test_harness.py`, `train/test_scenarios.py` | `src/agents/play_spec.*`, engine presets, `scenarios` tier (3a) |
| `train/test_snapshot.py`, `train/test_obs_invariants.py` | `--selfcheck-state` snapshot / complete (1b) |
| `train/bench_engine.py` | `bench` tier (3a) |

**Deleted without replacement (3b):** everything else in `train/`:

- training and AlphaZero: `train.py`, `env.py`, `extractor.py`, `az_*`, `mcts.py`, `popart.py`,
  `curriculum.py`, `gate_sprt.py`, `search_env.py`, the `bench_*` actor/worker benches;
- shards, trees and inspection: `shard_*`, `tree_*`, `az_inspect.py`, `analysis*.py`, `viz.py`;
- front ends, deferred to steps 6–7 on the C++ decoder: `tui*.py`, `gui*.py`,
  `game_driver.py`, `play.py`, `launcher_config.py`, `browse_session.py`, `scryfall_cache.py`;
- plumbing and codegen: `decode.py`, `runner.py`, `opponents.py`, `cli_spec.py`,
  `progress_io.py`, `gen_*.py` with their outputs (`_enums.py`, `card_costs.py`,
  `card_props.py`, `sb_rules.py`);
- `coverage_report.py`, `fuzz_campaign.py`, the RL test files, and `requirements-*.txt`.

## Open decisions (outside the format and knowledge design)

The format's decisions are in `docs/state_format.md` §11; the knowledge store's choices are in
`docs/knowledge_store.md`.

1. ~~`auto-mana` defaults~~ **Decided:** on by default for everyone; `--no-auto-mana` turns
   it off.
2. ~~**`snapshot.cpp`**~~ **Decided:** kept (3b), tested by the `--selfcheck-state` snapshot
   check; the base for search agents.
3. **`tools/meta_scraper`**: kept through 3b; keep or drop is still open.
4. ~~**C++ `explore` port**~~ **Decided and done** (3a): `fuzz` is Python-free and in-process,
   playing the Python tier's games for the same seeds.
5. ~~**`robomage_check` with exceptions on**~~ **Decided:** built with `-fno-exceptions` (3a).
6. ~~**Human vs Python agent in the CLI** (4b)~~ **Decided:** no second channel; a human plays
   the in-process agents or hotseat.
7. **Embedding model** (5a/5c): the pick and its licence check. Candidates: all-MiniLM-L6-v2 /
   L12-v2, bge-small-en-v1.5, e5-small-v2, gte-small, snowflake-arctic-embed-xs / -s (384-d,
   22–33M parameters), and, since the committed table makes encoder size cost nothing in the
   actor, larger ones such as bge-base-en-v1.5, all-mpnet-base-v2 or nomic-embed-text-v1.5
   (768-d; Matryoshka-truncatable). Pooling and any required text prefix follow the pick.
   **Decided:** oracle text only (no script text), the whole vocab every time, exported as a
   framework-neutral file, committed, kept complete by `make` (5c), weights fetched at a pinned
   revision only to generate rows.
8. **Learner** (after step 5): language (PyTorch vs libtorch) and algorithm. A learner is
   what would enable option 3.

## Rough size

| Step | Work (approx. lines) | Risk |
|---|---|---|
| 1a | design docs (three drafted) | medium: the decisions shape everything after |
| 1b | visibility + reveal windows 500–800; events 800–1,200; handles 300; encoder 2,000–3,000; C++ decoder 800–1,200; Python decoder 400–700; self-checks 800–1,200 | **high**: the leak audit, and finding every event choke point |
| 1c | knowledge model + update rules 1,200–1,800; solver 600–1,000; `rmknow` + tests 800–1,200 | **high**: exact inference, and classifying every shadow mismatch |
| 2 | adapter 600–1,000; agent pass 2 touches 1,000–2,000 | medium: strict parity is slow but mechanical |
| 3 | agents + resolver 1,000–1,500; `robomage_check` 1,500–2,000; then mostly deletion | low–medium |
| 4 | 4a ~300 changed; CLI 1,000–1,500 | low–medium |
| 5 | `CARDS` ability parameters 500–800; featurizer + table export 1,700–2,800; tokenizer + embedding 600–900; actor 800–1,200; tests 800–1,200 | medium: tokenizer parity and the network contract |
