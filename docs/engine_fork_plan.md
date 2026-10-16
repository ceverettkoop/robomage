# Engine-only fork: staged plan (steps 1–5)

Goal: fork robomage into a project that is **only the C++ rules engine**. Players and agents
drive it through in-process agents, a C++ CLI, or a machine protocol that emits an **exhaustive
binary game state** with hidden information withheld from each viewer. The RL stack is
discarded: PPO/AZ training, the libtorch actor, the float state vector, embeddings, codegen and
the Python front ends.

**The Python scripted agent is kept.** It is the only realistic opponent and a deterministic
consumer of the state. It is adapted to read the new format, and its games become the test that
the format carries everything it needs.

**Memory and inference are actor-side, in C++.** The engine sends only what each viewer
explicitly knows right now. A C++ knowledge sidecar (`docs/knowledge_store.md`) turns each
viewer's frame stream into perfect recall and sound logical inference. It is attached to the
same frames for every kind of agent, with levels `off` / `expert`.

**A C++ path to learned models.** Step 5 adds a schema-driven C++ featurizer and an in-process
C++ actor that runs a TorchScript policy on it. Card text reaches the model as oracle-text
embeddings computed in C++. This replaces the discarded RL stack's actor with one built on the
new format; a learner (training loop) is not in scope.

Scope: steps 1–5. The TUI and GUI (later steps) are deferred entirely; steps 1–5 only keep the
state decoder, knowledge tables and featurizer reusable for them. The eventual machine agent
is unspecified; the format just has to give it a complete, leak-free state and a structured
menu. ## Starting point

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
2. **Build the new beside the old.** RMSTATE (with its knowledge tables) is emitted *alongside*
   BQUERY until every consumer of the old vector is ported or deleted. The Python gate keeps
   running until the C++ gate replaces its engine tiers.
3. **Where the work happens.** Steps 1–2 only add code and run on robomage as it stands. Fork
   (keeping git history) at the end of step 2; steps 3–4 happen in the fork.
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
- **Pass 2, simplify.** Replace offset arithmetic with decoded objects, drop the caps, and drop
  numpy if possible. The agent may now use the richer knowledge tables. Re-record the corpus
  with the live agent, saying why.
- **Explore.** The `explore` tiers in the same file move with the rest; step 3 decides whether
  they also get a C++ port.
- **Driver.** A minimal driver (engine subprocess, frame read, reply write) replaces `env.py`
  / `runner.py` / `opponents.py` for agent games. **Fork here.**

## Step 3 — C++ gate, then strip the RL stack

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
| `state` | no | the step-1b self-checks over replayed and fuzz games |
| `knowledge` | no | the step-1c soundness / exactness / isolation / snapshot checks |
| `fuzz` | no (C++ explore) or yes (Python explore) | live explore-vs-explore league games; draws are findings |
| `smoke` | **yes** | live scripted-hard league games through the Python driver |
| `bench` | no | throughput of a fixed fuzz workload |

- **Corpus format change.** The corpus becomes RMLOG decision logs recorded from scripted-agent
  games, with the engine's `--replay` output as baseline. Checking it needs no Python;
  re-recording it uses the live agent.

### 3b. Strip

The corpus must stay byte-identical across 3b.

- Delete `src/actor/`'s old-observation code (`obs_builder`, `az_mcts`, `menu_merge`,
  `sb_rules`, `td_targets`, `root_diag`, `npz_writer`, `oracle_client`, `az_actor_main`).
  **Keep the libtorch plumbing step 5 reuses:** the Makefile's `ACTOR_*` rules
  (exceptions-enabled TUs, libtorch detection and linking) and `az_evaluator.cpp`'s TorchScript
  loading, which becomes the base of the step-5 evaluator. `az_mcts.cpp` stays in git history
  as a reference for a future search agent.
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

## Step 4 — One decision schedule, then the C++ CLI

### 4a. One rules-faithful decision schedule (intentional; re-record)

| Site | What it does | Fate |
|---|---|---|
| `state_manager_actions.cpp:104` `machine_masks_conditional_destroy` | hides Fatal Push-style casts with no currently passing target ("NOT a rules gate") | **delete** |
| `state_manager_actions.cpp:1085`, `mana_system.cpp:1553` | hides ordinary mana abilities at priority; auto-pays costs | → explicit option `auto-mana` |
| `action_processor.cpp:2201` | auto-resolves hybrid pips | part of `auto-mana` |
| `game_driver.cpp:1212` | caps sideboard swaps per player | keep as an always-on safety limit |
| `game_driver.cpp:538`, `cli_output.cpp` guards | output only | fold into the output mode |
| `name_card_choices.cpp:54` | truncates name-a-card menus to `MAX_ACTIONS` | lift (menus are unbounded) |
| `queries/zones.cpp:123-128` (`interchangeable_cards`) | keeps two identical library cards apart when one sits at a position the other player knows (engine memory) | redefine on currently explicit information: a card in a live reveal window is distinguishable, a remembered one is not |

- `auto-mana` is recorded in the RMLOG header (`record_header_flags`, `src/cli_args.h`); the
  scripted agent plays with it on.
- After 4a, machine mode means only "speak RMSTATE".
- The agent's positional menu assumptions (read map §2b) are kept, or changed together with
  the agent.
- Re-record the corpus with the live agent, and add explore games with `auto-mana` off so
  manual mana payment is covered.

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
  now.
- **Delete:** `GameState`, `populate_gamestate`, `print_game_state` and the remains of
  `src/machine_io.*`. `--trace-decisions` switches to the new renderer.
- **Test (tier `cli`):** golden transcripts of scripted stdin sessions.

**Exit:** the C++ engine, RMSTATE with knowledge tables, the C++ CLI, the C++ gate, and the
kept Python below.

## Step 5 — C++ featurizer and in-process actor

Target consumer: a **C++ in-process actor**. In one process it runs engine → frame →
knowledge → featurizer → TorchScript policy → action, with no Python per step.

### 5a. Design (`docs/featurizer.md`)

- **`FEATURE_SPEC` frame.** Announces every tensor's shape and field index, like `SCHEMA` does
  for tables.
- **Tensors per decision:**
  - entity tokens: objects, players, globals and knowledge groups / unseen cards; numeric
    features scaled by their schema tag, plus categorical embedding ids;
  - entity mask;
  - relations `(src, dst, type)` from handle columns;
  - action tokens with gather indices to their source/target entities, for a pointer head;
  - action mask.
- **Per-match card table** `[cards × (printed features + text embedding)]`, held by the model
  as a frozen buffer.
- **Network contract:** the TorchScript `forward` signature (policy logits over menu rows,
  value).
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

- **Text per card:** type line + oracle text (`CARDS`); both faces for double-faced cards;
  tokens use their own oracle text; the card's own name is replaced by a placeholder.
- **Model:** a small pretrained sentence-embedding encoder (BERT-family, ~22–33M parameters,
  384-d), e.g. bge-small-en-v1.5 or all-MiniLM-L6-v2. Final pick and licence check in 5a.
  - It runs through libtorch as a TorchScript export, so no new runtime dependency; prefer a
    pre-converted artifact.
  - The weights (~25–130 MB) are provisioned like card scripts: fetched and pinned, not
    committed.
- **Tokenizer:** C++ WordPiece reading the model's vocabulary, checked against a golden file of
  reference token ids.
- **Pipeline:** tokenize → encode → mean-pool → L2-normalise, giving one 384-float vector per
  card. Cached as a `[cards × 384]` table keyed by model id plus a per-card text hash; a stale
  row is recomputed; the cache makes runs bit-exact.
- **Sanity report:** nearest neighbours per card (Bolt ↔ Chain Lightning, Brainstorm ↔
  Ponder) to eyeball quality.

### 5d. In-process actor (`src/actor/`, exceptions-enabled TUs)

- **Evaluator:** loads a TorchScript policy (from the kept `az_evaluator.cpp` base) and checks
  that its card-set hash matches the match's `CARDS`.
- **Agent:** `nn:<model.pt>` on the step-3 Agent interface: featurize, evaluate, choose
  (argmax or sample) over menu rows.
- **Batched-evaluation hook** for a later search agent.
- **End-to-end test** with a randomly initialised model of the contract's shape.
- **Exit:**
  - `--agent-a nn:<model.pt>` plays full bo3 matches without errors;
  - the featurizer, tokenizer and embedding tests pass;
  - throughput is measured in the `bench` tier.

### Possible later addition: learned text encoder (option 3)

A learner could make the text encoder part of the policy network, initialised from the same
pretrained encoder and fine-tuned by the RL loss. Gradients from the policy and value losses
flow into its weights. The mitigations for having only ~400 distinct texts are:

- a low learning rate and staged unfreezing;
- text plus id embedding with id dropout;
- auxiliary losses predicting `CARDS` features;
- held-out-card checks.

The actor side is unchanged: the learner exports the recomputed `[cards × D]` table with the
weights, and actors still only look it up. This needs a learner, which is outside this plan.

---

## Python inventory: what remains

**Remains after step 4:**

| File (target location decided at the fork) | Role | Notes |
|---|---|---|
| `tools/forge_fetch/fetch_script.py`, `fetch_deck_scripts.py`, `provision_decks.py` | pinned Forge script provisioning (CI, SessionStart hook) | stdlib only. Trim `provision_decks.py`'s references to `train/gen_card_costs.py` |
| `tools/meta_scraper/scrape_meta.py` | MTGTop8 → `.dk` scraper | stdlib only; **keep or drop** (recommend keep) |
| scripted agent (today `train/scripted_agent.py`) | the realistic opponent; `explore` tiers unless ported to C++ | on the Python RMSTATE decoder; knowledge comes from the C++ sidecar's tables; numpy dropped if possible |
| agent driver (new, replaces `env.py` / `runner.py` / `opponents.py` for agent games) | engine subprocess, frame I/O, match loop | small |
| RMSTATE decoder (new, replaces `decode.py` for the agent) | bytes → objects, knowledge tables included | no copied constants (schema frame) |

There is no Python knowledge tracker (memory and inference live in the C++ sidecar) and no
Python in the embedding path (step 5 computes it in C++). The engine
builds, runs and passes every gate tier except `smoke` (and Python `fuzz`, if explore stays
Python) without Python. Recording the corpus needs the agent.

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

1. **`auto-mana` defaults** (4a): proposed off for `--machine`, on for the scripted agent and
   the CLI.
2. **`snapshot.cpp`**: proposed kept, tested by the `snapshot` self-check.
3. **`tools/meta_scraper`**: keep or drop.
4. **C++ `explore` port** (3a): makes `fuzz` Python-free and fast in-process; otherwise fuzz
   runs through the Python driver.
5. **`robomage_check` with exceptions on** (3a): needs a CLAUDE.md exemption.
6. **Human vs Python agent in the CLI** (4b): whether to add the second channel now.
7. **Embedding model** (5a/5c): bge-small-en-v1.5 vs all-MiniLM-L6-v2 (or another small
   384-d encoder), and how its weights are provisioned and pinned.
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
| 5 | featurizer 1,500–2,500; tokenizer + embedding 600–900; actor 800–1,200; tests 800–1,200 | medium: tokenizer parity and the network contract |
