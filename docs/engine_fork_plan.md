# Engine-only fork: staged plan (steps 1–5)

Goal: fork robomage into a project that is **only the C++ rules engine**. Players and agents
drive it through an in-process agent interface, a C++ CLI, or a machine protocol that emits an
**exhaustive binary game state** with hidden information withheld from each viewer. The RL stack
is discarded: the PPO/AZ training code, the libtorch actor, the float state vector, embeddings,
codegen and the Python front ends.

Scope of this document: steps 1–5. The TUI (step 6) and GUI (step 7) are deferred entirely and
are not planned here; nothing in steps 1–5 should be shaped around them beyond keeping the
state decoder reusable (step 5). The eventual machine agent is unspecified; step 4 only has to
give it a complete, leak-free state and a structured menu.

## Starting point

What already makes this feasible:

- `bin/robomage` has no torch and needs no Python at run time. The engine core (ECS, rules,
  parser, effects, `src/queries/`, replacement/layer systems, bo3, RMLOG replay) is ~47k lines
  of C++ outside `src/actor/`.
- The libtorch actor is already a separate binary (`src/actor/`, ~5.1k lines, `make actor`).
- None of the `pygen` outputs are used by the engine. `src/gen/*.h` are included only by
  `src/actor/`, so the Python build step can be deleted outright.
- `card_db` loads any script without consulting the vocab. `card_vocab.h` is just the
  implemented-card registry and the card id used by knowledge bookkeeping.
- There is an in-process decision hook: `InputLogger::set_input_provider`
  (`src/input_logger.h:86`), consulted on the machine path (`src/input_logger.cpp:394-424`).
- A C++ human CLI exists (`print_game_state` / `print_query` in `src/cli_output.cpp`,
  `--player A|B`), rendered from the RL-shaped `GameState` struct (`src/classes/gamestate.h`).
- RMLOG v2 decision logs (`--log-decisions`) replay deterministically with `--replay`, with a
  divergence guard (`src/input_logger.cpp:380-388`).

What makes it costly:

- **The float state vector is the only interface between the engine and all Python.** The test
  harness, `--play` resolver, rules scenarios, scripted "hard" agent, `explore` fuzzer, TUI and
  GUI all decode it. The whole test gate (`make check` → `train/ci_check.py`) is Python, so
  stripping Python first would remove every check that the engine still behaves the same.
- "Machine mode" changes the **decision schedule**, not just the output: an RL masking
  heuristic, hidden mana sources plus auto-pay, and RL shaping signals (inventory in step 3b).

## Ground rules

1. **Change behaviour only on purpose.** Each step says whether it may change engine behaviour.
   A step that must not change it ends with the replay corpus byte-identical. An intentional
   change re-records the corpus in its own commit, saying why.
2. **Python is the oracle while it exists.** Steps 1–2 build the C++ replacement *alongside*
   the Python gate. Both run in CI and must agree before step 3 deletes anything.
3. **Where the work happens.** Steps 1–2 only add code and are useful to robomage as it stands,
   so do them on robomage. Fork (keeping git history) at the end of step 2; steps 3–5 happen in
   the fork.
4. The project's existing conventions still apply: `-fno-exceptions` in engine TUs, shared
   queries in `src/queries/`, `ObjectRef` across zone changes, two-player only, CR as authority.

---

## Step 1 — C++ agents and gate foundation (additive; no engine behaviour change)

**1a. Agent interface.** Add `src/agents/` with a small `Agent` interface: given the live
`std::vector<LegalAction>` and the deciding seat, return an index or a concession sentinel. Then:

- Extend the provider hook to pass the seat (today it takes only the action list).
- Add engine flags `--agent-a <spec>` / `--agent-b <spec>` that install in-process agents. They
  run under the current machine decision schedule, so games match today's Python-driven ones,
  but emit no BQUERY frames and read no stdin.
- Initial specs: `first` (today's `auto`, action 0), `random`, `explore`, `explore:patient`,
  `actions:<ints>`.

**1b. Port the explore fuzzer.** Port `ScriptedAgent._explore_action` (`train/scripted_agent.py:3674`)
with its two weight tables, its novelty set and its seeded RNG. It reads `LegalAction` and
`CardData` directly, where Python used decoded obs and `card_props`. It does not need to match
Python's choices bit for bit; it needs the same coverage character (the nightly fuzz matrix).

**1c. Freeze realistic games before the scripted agent goes.** The scripted "hard" agent
(~3.8k lines reading obs offsets) is the only source of realistic play and is not being ported.
Before step 3, record RMLOG decision logs (`--log-decisions`) of scripted-hard games:

- the 64 matchups behind `train/regression/corpus/`;
- the smoke matchups: league mirrors + ring, several seeds, bo3.

Each log plus its `--replay` output becomes a baseline. This also changes what the replay
corpus means: today it is a Python-agent + engine transcript; afterwards it is engine-only
replay output. Caveat: a frozen log stays valid only while the decision schedule of the game
it records is unchanged (see step 3b).

**1d. Engine-side test flags** (replacing Python tricks):

- `--hand-a/-b` and `--library-a/-b` zone presets, replacing the temp-deck file the Python
  harness writes into `decks/temp/`;
- `--max-decisions N`;
- `--check-decks <dirs>`: every deck card resolves to a vocab entry and a script, with DFC
  front-face resolution. This replaces the `vocab` tier and `train/missing_cards.py`.

**1e. Gate driver `robomage_check`.** A separate C++ binary (`src/check/`, built by `make check`)
that runs engine subprocesses in parallel, so one crashing game cannot take down the gate.
It never links engine TUs, so it may be built with exceptions enabled (like the actor TUs
today); that keeps `std::regex` and I/O failures reportable per test. Tiers:

| Tier | Replaces | What it does |
|---|---|---|
| `vocab` | `ci_check` vocab, `missing_cards.py` | `--check-decks` over top-level + `league/` decks |
| `concede` | `test_concede.py` | pipes `-2`/`-3` at bo1/bo3 decisions; checks result lines and replay |
| `replay` | `replay_diff.py` | replays every frozen RMLOG; diffs output against its baseline |
| `smoke` | scripted-hard smoke | the frozen scripted-hard league logs from 1c, replayed + scanned |
| `fuzz` | explore fuzz | live `--agent-a explore --agent-b explore` over league matchups; draws are findings |
| `bench` | `bench_engine.py`, `observe --timing` | throughput of a fixed fuzz workload |

It also ports the transcript scan (`_ERROR_RE` / `_WARNING_RE` in `train/ci_check.py:265`) and
`draw_<stamp>.txt` reporting.

**Exit criteria:** `robomage_check` tiers are green next to an unchanged, green Python gate,
and both run in CI.

## Step 2 — `--play` resolver and rules scenarios in C++ (additive)

- Port `train/action_spec.py` (spec grammar, intent matching, exact-vs-substring, `@own`/`@opp`,
  ambiguity and failure messages, `format_menu`). Also port the seat-key auto-advance and
  hand-over-to-agent logic of `PlayController` (`train/opponents.py:1158`) as an Agent
  (`src/agents/play_spec.*`, spec `play:<specs>`), installed by `--play`. It matches against
  `LegalAction` fields (category, source card, controller, description) instead of decoded obs.
  The engine prints `resolved --actions: …` as the harness does today.
- Add the scenario runner to `robomage_check` (tier `scenarios`). It reads the existing 70
  `train/regression/scenarios/*.json` files unchanged. Their schema is flat (strings, string
  lists, ints, plus `expect_count`'s map), so a small JSON reader is enough; no dependency is
  needed. It implements every assertion key in the `test_scenarios.py` docstring. All 271
  regexes in the current set use no Python-only syntax, so `std::regex` (ECMAScript) applies
  as-is. `expect_menu` matches `LegalAction::description`.
- Interactive debugging: `--trace-decisions` prints the narrative, board and numbered menu at
  every decision (today's harness output), using the existing CLI printers for now. Step 5
  upgrades it.

**Parity (the exit criterion):** for every scenario, the C++ resolver yields exactly the
`resolved --actions` list the Python harness prints, and the same assertions pass. Rewrite
CLAUDE.md's "Test harness" and "Rules-regression scenarios" sections for the C++ commands.
**Fork here.**

## Step 3 — Strip (3a: no behaviour change; 3b: intentional schedule change)

### 3a. Delete the RL stack

The corpus must stay byte-identical across 3a.

- `src/actor/`, `make actor`, the libtorch/`ACTOR_*` Makefile flags.
- `train/` entirely, after moving `train/regression/{scenarios,corpus}` and the frozen logs to
  `tests/`. Python fates are listed below.
- The `pygen` target and every `gen_*.py`; `src/gen/`; `bin/resources/decks/archetypes.json`;
  `train/sb_dead_rules.json`.
- `src/search_server.{h,cpp}` and its hooks: cooperative unwind and `SEARCHINFO` in
  `input_logger.cpp`, `search_intercept_game_end` / restore handling in `game_driver.cpp`, the
  flag in `cli_args.cpp`.
  **Keep `src/snapshot.{h,cpp}`**: any future search agent needs it, and it gives undo and
  analysis. It gets a test in step 4.
- The BQUERY payload: the float packing (`serialize_state`, the `N_CARD_TYPES` normalisation,
  `OFFSET_CHAIN` / block widths, per-action float arrays), `cli_emit_machine_query` / `BSTATE`
  and `--broadcast-steps`. Keep `populate_gamestate` / `GameState` only as the CLI printer's
  source until step 5. With no consumer left, the external `--machine` flag goes away until
  step 4 brings it back; the internal schedule switch stays until 3b.
- RL shaping signals: `MANA_WASTED` (`src/mana_system.cpp:138`) and `MULLIGAN_PENALTY`
  (`src/game_driver.cpp:796`). Both print only in machine mode, so replays are unaffected.
- `tui.sh`, `gui.sh`, `train/gen_corpus.sh`.
- Infrastructure:
  - CI (`.github/workflows/ci.yml`, `nightly-fuzz.yml`, `.github/actions/setup-robomage`):
    drop the venv and pip cache; run `make check` and `robomage_check fuzz`. Keep `python3`
    only for the provisioning step.
  - The SessionStart hook (`.claude/hooks/session-start.sh`): no venv, numpy or gymnasium;
    just `python3 tools/forge_fetch/provision_decks.py`.
  - `.claude/settings.json`: drop the `train/` permissions.
- Docs: rewrite CLAUDE.md, README, `docs/ci.md` and `docs/game_running.md`. Keep
  `docs/rules_deviations.md` and the CR files.

Leave in place (step 4 decides their fate): knowledge bookkeeping keyed by vocab index. This
covers `KnownLibraryTop`, the match revealed bits in `src/classes/match_state.cpp`,
`Zone::identity_known`, and `source_vocab_idx` / `creator_vocab_idx` on emblems and delayed
triggers. These are real knowledge state, not RL artefacts.

### 3b. One rules-faithful decision schedule

Every place where machine mode changes the menu or the play:

| Site | What it does | Fate |
|---|---|---|
| `state_manager_actions.cpp:104` `machine_masks_conditional_destroy` | hides Fatal Push-style casts with no currently passing target ("NOT a rules gate") | **delete** |
| `state_manager_actions.cpp:1085`, `mana_system.cpp:1553` | hides ordinary mana abilities at priority; auto-pays costs | → explicit option `auto-mana` |
| `action_processor.cpp:2201` | auto-resolves hybrid pips | part of `auto-mana` |
| `game_driver.cpp:1212` | caps sideboard swaps per player | keep as an always-on safety limit |
| `game_driver.cpp:538`, `cli_output.cpp` guards | output only | fold into the output mode |
| `name_card_choices.cpp:54` | truncates name-a-card menus to `MAX_ACTIONS` | lifted in step 4 (unbounded menus) |

`auto-mana` is recorded in the RMLOG header (`record_header_flags`, `src/cli_args.h`), so the
frozen scripted-hard logs, all recorded with today's auto-pay, keep replaying with it on.
A frozen log that now diverges, i.e. one that touched the Fatal Push mask, is reported by the
replay divergence guard. Drop it and note it in the commit. Then:

- add explore-agent replay logs with `auto-mana` off, so manual mana payment is covered;
- after 3b, `machine_mode` means only "speak the binary protocol".

Default for `auto-mana`: off for the machine protocol (the exhaustive interface offers the real
choice of which sources to tap); the CLI may default it on.

**Exit criteria:** corpus re-recorded once with a written reason; the scenario, fuzz and smoke
tiers are green.

## Step 4 — Exhaustive binary state (RMSTATE)

### 4a. One visibility model before any encoder

Today hidden information is enforced ad hoc:

- `Zone::identity_known` (`src/components/zone.h:38`);
- `face_down_hidden_from` (`src/machine_io.cpp:404`);
- `KnownLibraryTop` (`src/classes/game.h:843`, 5 positions);
- the `opp_known_hand` pass (`src/machine_io.cpp:954`);
- match-scoped revealed bits (`src/classes/match_state.cpp`);
- narrative redaction (`game_log_private` / `game_log_redacted`, `src/cli_output.cpp:89-110`).

Replace these with `src/queries/visibility.{h,cpp}`: `identity_visible(obj, viewer)`,
`order_visible(zone, viewer)`, and the known-position library model. Base it on CR 400.2
(public/hidden zones), 701.20 (reveal), 708 / 406.3 (face down), and the "look at" effects.
Every existing source, and the CLI, reads only this. Extend `KnownLibraryTop` beyond 5 if
an implemented card needs it.

### 4b. Per-viewer opaque handles, never raw entity ids

Entity ids are allocated in deck-list order at game setup (`src/systems/orderer.cpp:611-615`),
and a shuffle only rewrites `distance_from_top`. A raw id of a library or hand card therefore
reveals its identity. Each viewer instead gets handles from a table keyed by `ObjectRef`:

- stable while that viewer can track the object (battlefield → graveyard, a revealed card
  staying in hand);
- reissued when the object becomes untrackable for that viewer (shuffled into a library, drawn
  unseen, put into a hand face down). This is the viewer-side analogue of CR 400.7.

The table lives in `Game`, so snapshots cover it.

### 4c. Content: everything the engine knows, filtered by 4a

- **Header:** format version, schema hash, viewer seat, decision number, game/match index.
- **Card registry** (once per match): vocab id → name, plus per-card printed data a consumer
  needs to interpret ids.
- **Turn:** turn, step, active player, priority holder and pass flags, extra turns.
- **Combat:** attackers and their targets, blockers, damage-assignment order.
- **Players:** life, poison, energy, mana pool with restrictions and sources, designations
  (monarch, city's blessing, day/night), per-turn counters, mulligan state, player effects,
  emblems.
- **Objects in every zone.** Visible ones are full records; hidden ones are counts or
  identity-less handles. A record holds owner and controller, face / transformed / face-down
  state, and base *and* post-layer characteristics: types, subtypes, colours, P/T, every
  keyword (not a 16-wide multi-hot), abilities including granted ones. It also holds all
  counters by kind, damage, tapped / sick / phased, attachments, chosen name / colour / type,
  linked exiled cards, and play permissions.
- **Stack:** spell or ability, source, controller, modes, X, targets, kicked / alternative
  costs paid, copies.
- **Rules state:** active continuous effects (layer, source, affected set, expiry), replacement
  and prevention shields, rules-modifying prohibitions, every pending delayed trigger (no
  16-slot cap), last-known info for referenced departed objects.
- **Pending decision:** the deciding seat, the query kind, its source, and its constraints
  (choose N, min/max, ordering).
- **Match:** game number, wins, sideboard phase, the viewer's own deck and sideboard.
- **Menu:** one structured record per legal action (category, source handle, target handles,
  mode, X, cost summary, description), with no length cap.

Hidden-information decisions this forces:

- **Opponent decklist.** The current obs exposes the opponent's registered maindeck (the
  `DECKLIST_MAIN_SLOTS` block), which assumes open decklists. Make that a match option
  (`--open-decklists`), off by default.
- **Name-a-card menus.** `build_name_card_choices` restricts candidates to the opponent's deck
  (a documented deviation from CR 201.4), so the menu leaks that decklist. With unbounded
  menus, offer every vocab card that passes `ValidCards$` and retire the deviation, or keep the
  restriction only under `--open-decklists`.
- **Narrative.** The perfect-information narrative never goes on the machine channel; at most
  the viewer-redacted log does.
- **Descriptions.** Audit `LegalAction::description` and choice labels
  (`src/choice_labels.cpp`, `src/announce.cpp`) for hidden names.
- **Hand order.** Emit hands in a canonical order (e.g. by handle) rather than entity order.

### 4d. Encoding and protocol

- **Encoding:** a hand-rolled, versioned, little-endian tag–length–value stream. Each record
  carries numbered fields, so readers skip unknown ones and the format can grow without
  breaking consumers. The source of truth is one header (`src/state_format/schema.h`) plus
  `docs/state_format.md`. Alternatives considered:
  - FlatBuffers: workable, but adds a dependency and a codegen step;
  - protobuf: its runtime is heavy under `-fno-exceptions`;
  - JSON: not binary. Do provide `--state-dump json`, rendered from the decoded binary and not
    a second encoder, for tests and humans.
- **Protocol (`--machine`):** at each decision write `RMSTATE <bytes>\n` + payload, then read
  one line. The reply is an action index, `-1` (confirm), or `-2`/`-3` (concede game/match),
  with today's sentinels kept. An agent seat in-process (`--agent-x`) receives the same encoded
  state through the Agent interface, so in-process and external agents see identical data.

### 4e. Tests (new `robomage_check` tiers)

- **roundtrip:** at every decision of fuzz games, encode → decode → re-encode is byte-identical.
- **leak:** at sampled decisions, permute every hidden zone *from the viewer's perspective*
  (opponent's unknown hand and library contents, own unknown library order), keeping known
  positions. The viewer's encoded state must be byte-identical across the permutations. This is
  the generic guarantee that hidden information is hidden; it relies on 4b.
- **complete:** structural invariants, the successor to `obsinv`. Every battlefield permanent
  appears, counts match zone sizes, every handle reference resolves, every menu record's
  handles exist.
- **snapshot:** save → K random actions → restore; both viewers' encodings match the
  pre-save bytes. This is the test promised for keeping `snapshot.cpp`.

**Exit criteria:** `--machine` speaks RMSTATE; the four tiers are green; `docs/state_format.md`
is published.

## Step 5 — C++ CLI on the new format

- **Decoder:** `src/state_format/decode.*` turns the byte stream into plain structs. It is
  shared by the CLI, `--state-dump json`, the tests and, later, steps 6–7.
- **The CLI renders only the decoded state of the human's seat**, never engine internals, so
  any display bug is a format bug. It shows players, battlefield (grouped and annotated),
  hand, stack with targets, graveyards and exile, mana pool, pending-decision context and the
  numbered menu.
- **Commands:** an index, `concede` / `concede match`, `board`, `zone <name>`,
  `card <n>` (oracle text from the script), `log` (the viewer-redacted narrative), `help`.
- **Seating:** human vs in-process agent (`--player A --agent-b explore`) and hotseat.
  Human vs *external* machine agent needs a second channel (e.g. `--machine-fd`); leave it to
  whoever specifies that agent.
- **Delete:** `GameState`, `populate_gamestate`, `print_game_state` and what remains of
  `src/machine_io.*`. `--trace-decisions` (step 2) switches to the new renderer.
- **Test (tier `cli`):** golden transcripts of scripted stdin sessions (a few scenarios with a
  human seat), diffed byte for byte.

**Exit criteria:** the fork is the C++ engine, the RMSTATE protocol, the C++ CLI and the C++
gate. The only Python left is listed below.

---

## Python inventory: what remains

**Remains after step 5** (stdlib only; never needed to build, run or test the engine):

| File | Role | Notes |
|---|---|---|
| `tools/forge_fetch/fetch_script.py` | fetches pinned Forge card/token scripts | kept, per decision |
| `tools/forge_fetch/fetch_deck_scripts.py` | fetches every card in given decks | kept |
| `tools/forge_fetch/provision_decks.py` | provisioning entry point for CI and the SessionStart hook | kept. Reads `src/card_vocab.h` and decks; trim its docstring's references to `train/gen_card_costs.py` |
| `tools/meta_scraper/scrape_meta.py` | MTGTop8 → `.dk` deck scraper | **decision needed**: same category as forge_fetch (data provisioning, standalone). Recommend keeping it |

CI and the SessionStart hook need only a system `python3`, and only for provisioning. No venv.

**Ported to C++, then deleted (step in brackets):**

| Python | C++ replacement |
|---|---|
| `train/ci_check.py` (engine tiers), `train/regression/replay_diff.py`, `train/test_concede.py` | `robomage_check` tiers [1] |
| `train/missing_cards.py` | `--check-decks` + `vocab` tier [1] |
| explore/random tiers of `train/scripted_agent.py` | `src/agents/` [1] |
| `train/bench_engine.py` | `bench` tier [1] |
| `train/action_spec.py`, `PlayController` / `ActionListController` (`train/opponents.py`), `train/test_harness.py`, `train/test_scenarios.py` | `src/agents/play_spec.*`, engine presets, `scenarios` tier [2] |
| `train/test_snapshot.py` | `snapshot` tier [4] |
| `train/test_obs_invariants.py` | `complete` tier [4] (new invariants, same idea) |

**Deleted without replacement [3]:** everything else in `train/`:

- training and AlphaZero: `train.py`, `env.py`, `extractor.py`, `az_*`, `mcts.py`, `popart.py`,
  `curriculum.py`, `gate_sprt.py`, `search_env.py`, the `bench_*` actor/worker benches;
- shards, trees and inspection: `shard_*`, `tree_*`, `az_inspect.py`, `analysis*.py`, `viz.py`;
- front ends, deferred to steps 6–7 and rebuilt on the step-5 decoder: `tui*.py`, `gui*.py`,
  `game_driver.py`, `play.py`, `launcher_config.py`, `browse_session.py`, `scryfall_cache.py`;
- plumbing and codegen: `decode.py`, `runner.py`, `opponents.py`, `cli_spec.py`,
  `progress_io.py`, `gen_*.py` with their outputs;
- the scripted "hard" agent (`scripted_agent.py`); its play survives only as the frozen logs
  from 1c;
- `coverage_report.py`, `fuzz_campaign.py`, the RL test files, and `requirements-*.txt`.

## Open decisions

1. **`auto-mana` defaults** (3b): off for `--machine`, on for the CLI, as proposed.
2. **Open decklists** (4c): off by default, as proposed; affects name-a-card menus too.
3. **`snapshot.cpp`**: proposed kept (3a), tested in 4e. Dropping it later is easy; rebuilding
   it is not.
4. **`tools/meta_scraper`**: keep or drop.
5. **Coverage after 3b:** frozen scripted-hard logs remain valid only while their schedule is
   untouched. After a later menu-shape change, realistic-play coverage falls back to explore
   fuzz until a competent C++ agent exists. Porting the scripted agent is out of scope here.
6. **`robomage_check` with exceptions on** (1e): it is a separate binary that links no engine
   TUs; CLAUDE.md's exceptions rule would need this second exemption written in.

## Rough size

| Step | New / changed C++ (approx. lines) | Risk |
|---|---|---|
| 1 | agents + explore port 600–900; engine flags 200; `robomage_check` 1,000–1,500 | low: additive, checked against the Python gate |
| 2 | spec resolver + play agent 700–900; scenario runner + JSON reader 500–700 | low–medium: parity is mechanical but the seat-key semantics are fiddly |
| 3 | mostly deletion; 3b ~300 changed | medium: 3b is the one intentional schedule change |
| 4 | visibility 500–800; handles 300; encoder 2,000–3,000; tests 800–1,200 | **high**: the design work and the leak audit |
| 5 | decoder 800–1,200; CLI 1,000–1,500 | low–medium |
