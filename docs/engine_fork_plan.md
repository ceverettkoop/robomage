# Engine-only fork: staged plan (steps 1–5)

Goal: fork robomage into a project that is **only the C++ rules engine**. Players and agents
drive it through in-process agents, a C++ CLI, or a machine protocol that emits an **exhaustive
binary game state** with hidden information withheld from each viewer. The RL stack is
discarded: PPO/AZ training, the libtorch actor, the float state vector, embeddings, codegen and
the Python front ends.

**The Python scripted agent is kept.** It is the only realistic opponent and a deterministic
consumer of the state. It is adapted to read the new format, and its games become the test that
the format carries everything it needs.

Scope: steps 1–5. The TUI (step 6) and GUI (step 7) are deferred entirely; steps 1–5 only keep
the state decoder reusable for them. The eventual machine agent is unspecified; the format just
has to give it a complete, leak-free state and a structured menu.

## Starting point

What makes this feasible:

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

What makes it costly:

- **The float state vector is the only interface between the engine and all Python.** The
  scripted agent, `explore` fuzzer, test harness, `--play` resolver, rules scenarios, TUI and
  GUI all decode it, and the test gate (`make check` → `train/ci_check.py`) is Python.
- "Machine mode" changes the **decision schedule**, not just the output: an RL masking
  heuristic, hidden mana sources plus auto-pay, and RL shaping signals (step 5a).

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
   `scripted_agent.py:2344`), which assumes open decklists.
5. **Mana auto-pay:** the agent never chooses which sources pay a cost.

## Ground rules

1. **Change behaviour only on purpose.** Each step says whether it may change engine
   behaviour. Steps that must not end with the replay corpus byte-identical; an intentional
   change re-records the corpus in its own commit, saying why.
2. **Build the new beside the old.** The new format is emitted *alongside* BQUERY until every
   consumer of the old vector is ported or deleted. The Python gate keeps running until the C++
   gate replaces its engine tiers.
3. **Where the work happens.** Steps 1–3 only add code and can run on robomage as it stands.
   Fork (keeping git history) at the end of step 3; steps 4–5 happen in the fork.
4. The existing conventions still apply: `-fno-exceptions` in engine TUs, shared queries in
   `src/queries/`, `ObjectRef` across zone changes, two-player only, CR as authority.

---

## Step 1 — Design the format (spec only; no code)

Outputs: `docs/state_format.md` (the spec) and an **agent read map** (below).

- **Visibility model.** One set of rules for what each viewer knows, based on CR 400.2
  (public/hidden zones), 701.20 (reveal), 708 / 406.3 (face down) and the "look at" effects.
  It replaces the ad-hoc sources:
  - `Zone::identity_known` (`src/components/zone.h:38`);
  - `face_down_hidden_from` (`src/machine_io.cpp:404`);
  - `KnownLibraryTop` (`src/classes/game.h:843`);
  - the `opp_known_hand` pass (`src/machine_io.cpp:954`);
  - match revealed bits (`src/classes/match_state.cpp`);
  - narrative redaction (`src/cli_output.cpp:89-110`).
- **Object handles.** Per-viewer opaque handles, never raw entity ids. Entity ids are allocated
  in deck-list order (`src/systems/orderer.cpp:611-615`) and a shuffle only rewrites
  `distance_from_top`, so a raw id of a library or hand card reveals its identity. Define when
  a handle stays stable and when it is reissued (the viewer-side analogue of CR 400.7).
- **Records.** Header, card table, turn/priority/combat, players, objects in every zone,
  stack, rules state (continuous effects, replacement/prevention shields, prohibitions, delayed
  triggers, last-known info), pending decision, match, and a structured menu with no length cap.
- **Encoding and protocol.** Byte layout, versioning, how the schema reaches consumers, the
  `--machine` frame, reply sentinels (`-1` confirm, `-2`/`-3` concede).
- **Hidden-information options:**
  - `--open-decklists`, which the agent needs (dependency 4);
  - name-a-card menus, which today list only cards from the opponent's deck
    (`build_name_card_choices`) and so leak that list;
  - the narrative channel;
  - description strings;
  - hand order.
- **Agent read map.** Every agent dependency 1–5 mapped to the field that will serve it. This
  is the completeness checklist for the agent; the exhaustive inventory is the superset.

**Exit:** the spec and read map are reviewed, and every open decision is closed or explicitly
deferred.

## Step 2 — Implement the encoder alongside BQUERY (additive)

- `src/queries/visibility.{h,cpp}` from the step-1 model; the old ad-hoc sources rewired
  through it. This must not change today's vector, which is checked by the corpus and `obsinv`
  tiers.
- Handle table on `Game`, so snapshots cover it.
- Encoder `src/state_format/encode.*`, emitted with `--machine-format rmstate`, plus a C++
  decoder (`src/state_format/decode.*`) and `--state-dump json`, rendered from the decoded
  binary rather than by a second encoder.
- Python decoder for the agent. Per the step-1 decision on schema delivery, it carries no
  copied constants.
- **Engine self-checks** that run while any driver plays (`--selfcheck-state`):
  - **roundtrip:** encode → decode → re-encode is byte-identical at every decision;
  - **leak:** permute every hidden zone from the viewer's perspective, keeping known
    positions; the viewer's encoding must be byte-identical across permutations;
  - **complete:** structural invariants (every permanent present, counts match zone sizes,
    every handle reference resolves, every menu record's handles exist);
  - **snapshot:** save → K actions → restore; both viewers' encodings match the pre-save
    bytes.

  They run as new tiers of the existing Python gate.

**Exit:** the self-checks are green over the corpus, smoke and fuzz games, and the old vector
is unchanged.

## Step 3 — Port the scripted agent to the new format (additive)

- **Pass 1, strict parity.** An adapter on the Python decoder answers every old read with the
  value the old vector gave. That includes rounding through the old normalisers and mimicking
  the old caps (48 permanent slots, 64 graveyard slots, 16 delayed triggers, 10 hand slots) so
  overflow cases decide the same way. Card knowledge (`card_props`, `card_costs`, script lines)
  comes from the format's card table; vocab constants become name lookups.
  **Exit:** the replay corpus and the scripted-hard smoke games are byte-identical with the
  agent on `rmstate` versus BQUERY. This is the proof that the format serves the agent.
- **Pass 2, simplify.** Replace offset arithmetic with decoded objects, drop the caps, and drop
  numpy if possible. The corpus may change here; re-record it with the live agent, saying why.
- The `explore` tiers in the same file move to the new format with the rest; step 4 decides
  whether they also get a C++ port.
- The new minimal driver (engine subprocess, frame read, reply write) replaces `env.py` /
  `runner.py` / `opponents.py` for agent games. **Fork here.**

## Step 4 — C++ gate, then strip the RL stack

### 4a. C++ agents, the `--play` resolver and the gate

- `src/agents/`: an Agent interface on the provider hook (extended to pass the deciding seat),
  with flags `--agent-a/-b <spec>`. Specs: `first`, `random`, `actions:<ints>`, `play:<specs>`,
  and optionally a C++ `explore` port for fast in-process fuzzing.
- Port `train/action_spec.py` and the seat-key logic of `PlayController`
  (`train/opponents.py:1158`) to `src/agents/play_spec.*`. Parity: every scenario resolves to
  the same `resolved --actions` list as the Python harness, and the same assertions pass.
- Engine flags replacing harness tricks:
  - `--hand-a/-b` and `--library-a/-b`, replacing the temp deck in `decks/temp/`;
  - `--max-decisions`;
  - `--check-decks`, replacing the `vocab` tier and `missing_cards.py`;
  - `--trace-decisions`: board and menu at every decision.
- **Gate driver `robomage_check`.** A separate C++ binary that links no engine TUs and so may
  enable exceptions (needs a CLAUDE.md exemption beside the actor's). It runs engine
  subprocesses in parallel. It reads the existing 70 scenario JSONs unchanged: their schema is
  flat, and all 271 regexes use no Python-only syntax, so `std::regex` applies. It also ports
  the transcript scan (`train/ci_check.py:265`).

| Tier | Needs Python? | What it does |
|---|---|---|
| `vocab` | no | `--check-decks` over top-level + `league/` decks |
| `concede` | no | pipes `-2`/`-3` at bo1/bo3 decisions; checks result lines and replay |
| `scenarios` | no | the 70 rules scenarios |
| `replay` | no | replays the corpus RMLOGs; diffs output against baselines |
| `state` | no | the step-2 self-checks over replayed and fuzz games |
| `fuzz` | no (C++ explore) or yes (Python explore) | live explore-vs-explore league games; draws are findings |
| `smoke` | **yes** | live scripted-hard league games through the Python driver |
| `bench` | no | throughput of a fixed fuzz workload |

- **Corpus format change.** The corpus becomes RMLOG decision logs recorded from scripted-agent
  games, with the engine's `--replay` output as baseline. Checking it needs no Python;
  re-recording it uses the live agent.

### 4b. Strip

The corpus must stay byte-identical across 4b.

- Delete `src/actor/`, `make actor` and the libtorch flags.
- Delete everything in `train/` except the kept agent, driver and decoder (list below). Move
  `train/regression/{scenarios,corpus}` to `tests/`.
- Delete the `pygen` target and every `gen_*.py`; `src/gen/`; `bin/resources/decks/archetypes.json`;
  `train/sb_dead_rules.json`.
- Delete `src/search_server.{h,cpp}` and its hooks (cooperative unwind and `SEARCHINFO` in
  `input_logger.cpp`, `search_intercept_game_end` / restore handling in `game_driver.cpp`, the
  flag in `cli_args.cpp`). **Keep `src/snapshot.{h,cpp}`** (search agents, undo, analysis;
  tested by the `snapshot` self-check).
- Delete the BQUERY payload: float packing, the `N_CARD_TYPES` normalisation, `OFFSET_CHAIN` /
  block widths, `BSTATE` and `--broadcast-steps`. `--machine` now means RMSTATE. Keep
  `populate_gamestate` / `GameState` only as the CLI printer's source until step 5b.
- Delete the RL shaping signals `MANA_WASTED` (`src/mana_system.cpp:138`) and
  `MULLIGAN_PENALTY` (`src/game_driver.cpp:796`).
- Delete `tui.sh`, `gui.sh` and `train/gen_corpus.sh`.
- Infrastructure:
  - CI (`ci.yml`, `nightly-fuzz.yml`, `.github/actions/setup-robomage`): drop the
    numpy/gymnasium venv; `python3` is needed only for provisioning and the `smoke` tier.
  - SessionStart hook: `python3 tools/forge_fetch/provision_decks.py`.
  - `.claude/settings.json`: drop the `train/` permissions.
- Docs: rewrite CLAUDE.md, README, `docs/ci.md` and `docs/game_running.md`. Keep
  `docs/rules_deviations.md` and the CR files.

## Step 5 — One decision schedule, then the C++ CLI

### 5a. One rules-faithful decision schedule (intentional; re-record)

| Site | What it does | Fate |
|---|---|---|
| `state_manager_actions.cpp:104` `machine_masks_conditional_destroy` | hides Fatal Push-style casts with no currently passing target ("NOT a rules gate") | **delete** |
| `state_manager_actions.cpp:1085`, `mana_system.cpp:1553` | hides ordinary mana abilities at priority; auto-pays costs | → explicit option `auto-mana` |
| `action_processor.cpp:2201` | auto-resolves hybrid pips | part of `auto-mana` |
| `game_driver.cpp:1212` | caps sideboard swaps per player | keep as an always-on safety limit |
| `game_driver.cpp:538`, `cli_output.cpp` guards | output only | fold into the output mode |
| `name_card_choices.cpp:54` | truncates name-a-card menus to `MAX_ACTIONS` | lift (menus are unbounded) |

- `auto-mana` is recorded in the RMLOG header (`record_header_flags`, `src/cli_args.h`); the
  scripted agent plays with it on.
- After 5a, machine mode means only "speak RMSTATE".
- Re-record the corpus with the live agent, and add explore games with `auto-mana` off so
  manual mana payment is covered.

### 5b. C++ CLI on the decoder

- **The CLI renders only the decoded state of the human's seat**, never engine internals, so
  any display bug is a format bug. It shows players, battlefield, hand, stack with targets,
  graveyards and exile, mana pool, the pending decision and the numbered menu.
- **Commands:** an index, `concede` / `concede match`, `board`, `zone <name>`, `card <n>`,
  `log` (viewer-redacted narrative), `help`.
- **Seating:** human vs in-process agent, and hotseat. Human vs the Python scripted agent
  (external) needs a second channel (e.g. `--machine-fd`); decide here whether it is wanted
  now.
- **Delete:** `GameState`, `populate_gamestate`, `print_game_state` and the remains of
  `src/machine_io.*`. `--trace-decisions` switches to the new renderer.
- **Test (tier `cli`):** golden transcripts of scripted stdin sessions.

**Exit:** the C++ engine, RMSTATE, the C++ CLI, the C++ gate, and the kept Python below.

---

## Python inventory: what remains

**Remains after step 5:**

| File (target location decided at the fork) | Role | Notes |
|---|---|---|
| `tools/forge_fetch/fetch_script.py`, `fetch_deck_scripts.py`, `provision_decks.py` | pinned Forge script provisioning (CI, SessionStart hook) | stdlib only. Trim `provision_decks.py`'s references to `train/gen_card_costs.py` |
| `tools/meta_scraper/scrape_meta.py` | MTGTop8 → `.dk` scraper | stdlib only; **keep or drop** (recommend keep) |
| scripted agent (today `train/scripted_agent.py`) | the realistic opponent; `explore` tiers unless ported to C++ | on the Python RMSTATE decoder; numpy dropped if possible |
| agent driver (new, replaces `env.py` / `runner.py` / `opponents.py` for agent games) | engine subprocess, frame I/O, match loop | small |
| RMSTATE decoder (new, replaces `decode.py` for the agent) | bytes → objects | no copied constants (schema delivery per step 1) |

The engine builds, runs and passes every gate tier except `smoke` (and Python `fuzz`, if
explore stays Python) without Python. Recording the corpus needs the agent.

**Ported to C++, then deleted:**

| Python | C++ replacement (step) |
|---|---|
| `train/ci_check.py` engine tiers, `train/regression/replay_diff.py`, `train/test_concede.py` | `robomage_check` tiers (4a) |
| `train/missing_cards.py` | `--check-decks` + `vocab` tier (4a) |
| `train/action_spec.py`, `PlayController` / `ActionListController`, `train/test_harness.py`, `train/test_scenarios.py` | `src/agents/play_spec.*`, engine presets, `scenarios` tier (4a) |
| `train/test_snapshot.py`, `train/test_obs_invariants.py` | `--selfcheck-state` snapshot / complete (2) |
| `train/bench_engine.py` | `bench` tier (4a) |

**Deleted without replacement (4b):** everything else in `train/`:

- training and AlphaZero: `train.py`, `env.py`, `extractor.py`, `az_*`, `mcts.py`, `popart.py`,
  `curriculum.py`, `gate_sprt.py`, `search_env.py`, the `bench_*` actor/worker benches;
- shards, trees and inspection: `shard_*`, `tree_*`, `az_inspect.py`, `analysis*.py`, `viz.py`;
- front ends, deferred to steps 6–7 on the C++ decoder: `tui*.py`, `gui*.py`,
  `game_driver.py`, `play.py`, `launcher_config.py`, `browse_session.py`, `scryfall_cache.py`;
- plumbing and codegen: `decode.py`, `runner.py`, `opponents.py`, `cli_spec.py`,
  `progress_io.py`, `gen_*.py` with their outputs (`_enums.py`, `card_costs.py`,
  `card_props.py`, `sb_rules.py`);
- `coverage_report.py`, `fuzz_campaign.py`, the RL test files, and `requirements-*.txt`.

## Open decisions (outside the step-1 format design)

1. **`auto-mana` defaults** (5a): proposed off for `--machine`, on for the scripted agent and
   the CLI.
2. **`snapshot.cpp`**: proposed kept, tested by the `snapshot` self-check.
3. **`tools/meta_scraper`**: keep or drop.
4. **C++ `explore` port** (4a): makes `fuzz` Python-free and fast in-process; otherwise fuzz
   runs through the Python driver.
5. **`robomage_check` with exceptions on** (4a): needs a CLAUDE.md exemption.
6. **Human vs Python agent in the CLI** (5b): whether to add the second channel now.

## Rough size

| Step | Work (approx. lines) | Risk |
|---|---|---|
| 1 | spec + read map (docs) | medium: the decisions shape everything after |
| 2 | visibility 500–800; handles 300; encoder 2,000–3,000; C++ decoder 800–1,200; Python decoder 400–700; self-checks 800–1,200 | **high**: the leak audit |
| 3 | adapter 600–1,000; agent pass 2 touches 1,000–2,000 | medium: strict parity is slow but mechanical |
| 4 | agents + resolver 1,000–1,500; `robomage_check` 1,500–2,000; then mostly deletion | low–medium |
| 5 | 5a ~300 changed; CLI 1,000–1,500 | low–medium |
