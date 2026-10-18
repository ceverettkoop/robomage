# Step 1 handoff: state format emitted + knowledge sidecar

> **Historical record.** This brief describes the repository as it was at step 1, with the RL
> stack, BQUERY and the Python gate (`train/ci_check.py`) still in place. Plan step 3b deleted
> them: `train/` paths name files now on branch `archive/rl-stack` (the kept Python is in
> `python/`, the scenarios and corpus in `tests/`), the gate is `robomage_check` (`docs/ci.md`),
> `--machine` speaks RMSTATE, and the shadow comparison is gone (its findings:
> `knowledge_store.md` §9.7). For the current project read `CLAUDE.md`.

**Status: done.** Every task (T0–T12) is implemented and the exit criteria of section 6 hold:
the `rmstate` and `knowledge` checks run in `make check`, the replay corpus is byte-identical,
`obsinv` is green, and every shadow-comparison mismatch is classified in
`train/regression/knowledge_divergences.json` (`knowledge_store.md` §9.7; deviations from this
brief in `engine_fork_plan.md`, step 1). The brief below is kept as written for the record.

This is a self-contained brief for a session picking up **step 1** of
[`engine_fork_plan.md`](engine_fork_plan.md). Read it top to bottom before writing code.

## 0. Read first, in this order

1. `CLAUDE.md`: repo conventions (code style, testing, long-running commands, subagents).
   They apply in full.
2. [`engine_fork_plan.md`](engine_fork_plan.md), the **Step 1** section and the ground rules.
3. [`state_format.md`](state_format.md): the format spec. Its decision register (§11) is
   closed; do not reopen a decided entry without asking the user.
4. [`knowledge_store.md`](knowledge_store.md): the knowledge sidecar design.
5. [`state_format_agent_map.md`](state_format_agent_map.md): what the scripted agent reads
   today. Not needed to build step 1, but it is the consumer step 2 will plug in, so every
   field it names must exist.

## 1. Status at handoff

- **Branch:** `claude/nifty-cerf-16733w` on `ceverettkoop/robomage`. Docs only so far; no code
  has changed.
- **1a (design):** the three docs above are drafted and their decisions are closed. One check
  remains (task T0).
- **1b (engine emission)** and **1c (knowledge sidecar):** not started. This brief covers both.
- **Main decisions already made** (see the register for all 31):
  - The engine sends each viewer only what that viewer **explicitly knows right now**: a
    memoryless snapshot plus ordered **observation events**. Recall and inference are
    **actor-side**, in a C++ knowledge sidecar.
  - Per-viewer **opaque handles**, never entity ids. Entity ids follow deck-list order and
    survive shuffles, so they leak.
  - **Relational fixed-width tables**, a **`SCHEMA` frame** with semantic column tags and enum
    string tables, and `RMSTATE <nbytes>\n` framing.
  - **Open decklists** by default (`--closed-decklist` turns them off), with deck names.
  - `CARDS` carries oracle text; rows carry owner/controller both absolute and
    viewer-relative.
  - Knowledge levels: `off` and `expert` (the default), per seat. Knowledge tables (`k_*`)
    travel inside the viewer's frame.

## 2. Scope and hard constraints

- **Additive only.** BQUERY and the Python stack keep working unchanged. Today's state vector
  must not change: the `replay` corpus tier stays byte-identical and the `obsinv` tier stays
  green. `make check` must pass before every push.
- **Do not** delete the engine's knowledge bookkeeping (`KnownLibraryTop`,
  `Zone::identity_known`, the match revealed bits). That is step 3b. In step 1 they keep
  feeding BQUERY and are the reference for the shadow comparison.
- **Do not** change the decision schedule or menus. `auto-mana` default-on (register 32)
  lands in step 4a, not here. The `interchangeable_cards` menu collapse
  (`src/queries/zones.cpp:123-128`) reads engine memory; leave it alone (step 4a).
- **Do not** port the scripted agent (step 2), build a C++ gate (step 3), or touch the
  featurizer (step 5).
- **Code rules:**
  - `-fno-exceptions` everywhere you touch;
  - shared entity queries go in `src/queries/`;
  - `ObjectRef` for anything kept across a zone change;
  - two-player only;
  - no new functions in `main.cpp`;
  - static helpers forward-declared at the top of the file;
  - never modify card scripts.
- **Long runs** (build, `make check`, fuzz) go in the background, with the output path
  stated in chat (CLAUDE.md).

## 3. Code layout to create

| Path | Contents | May include |
|---|---|---|
| `src/queries/visibility.{h,cpp}` | the visibility rules + per-viewer reveal windows | engine |
| `src/state_format/schema.{h,cpp}` | table/column definitions, semantic tags, enum string tables, schema hash: **the single source of truth** for the format | std only |
| `src/state_format/decode.{h,cpp}` | bytes → plain structs | `schema.h` + std only |
| `src/state_format/encode.{h,cpp}` | engine → bytes (snapshot, events, `CARDS`, `DECKLISTS`, `SCHEMA`) | engine |
| `src/state_format/events.{h,cpp}` | typed, viewer-projected event constructors + per-viewer queues | engine |
| `src/state_format/handles.{h,cpp}` | per-viewer handle tables | engine |
| `src/state_format/selfcheck.{h,cpp}` | `--selfcheck-state` / `--selfcheck-knowledge` | engine + decode + knowledge |
| `src/knowledge/*.{h,cpp}` | model, update rules, solver, `k_*` output | `state_format/decode.h`, `schema.h`, std only |
| `tools/rmknow/rmknow.cpp` | standalone frame-stream → knowledge tool | decode + knowledge only |
| `python/rmstate.py` | Python decoder, built from the `SCHEMA` frame | stdlib (+ numpy optional) |

**Build notes:**
- The Makefile compiles `src/*.cpp` and `src/*/*.cpp` into `bin/<config>/robomage`, so
  `src/state_format/` and `src/knowledge/` are linked into the engine automatically.
- `rmknow` lives outside `src/` so its `main` is not swept into the engine. Add a Makefile
  target that links only the decode, schema and knowledge objects.
- Add a gate check that greps the includes of `src/state_format/decode.*`, `schema.*` and
  `src/knowledge/*` and fails on any engine header. This is what makes "the sidecar cannot
  leak" structural.

## 4. Tasks, in order

Commit after each task, with `make check` green. Use one subagent per task (CLAUDE.md), and
tell subagents not to spawn further agents.

### T0. Close 1a: event-completeness check (docs)

- Walk `knowledge_store.md` §5. For every update rule, confirm the fields it uses exist on its
  event in `state_format.md` §5. Fix the event table where they don't.
- Turn `state_format.md` §4's table list into **concrete columns** (name, storage type,
  semantic tag, range) in a short appendix. T3 codes them in `schema.h`.
- **Done when:** both docs agree. Ask the user to confirm before coding if anything
  substantive changed.

### T1. Visibility and reveal windows

- `identity_visible(entity, viewer)`, `order_visible(zone, owner, viewer)`, and the reveal
  window set (per viewer: entity → why, until when).
- Generalise `Game::revealed_in_library` (`src/classes/game.h:751-755`). It is cleared at
  resolution end (`src/systems/stack_manager.cpp:103`), on zone change
  (`src/systems/orderer.cpp:332`) and on shuffle (`orderer.cpp:561`). Extend the same
  lifetime rules to hand reveals and looks (CR 701.20a, 701.20d, 701.20e).
- Rules sources: CR 400.2, 401.2–3, 402.3, 406.3, 701.20, 708.5–6 (grep
  `docs/mtg_comprehensive_rules.txt`).
- **The open reveal and look sites to hook** (the same sites that feed today's bookkeeping):
  - `mark_card_revealed`: `effect_reveal.cpp:26`, `effect_set_state.cpp:41`,
    `effect_discard.cpp:74,96`, `effect_peek_and_reveal.cpp:50,73,107,125`,
    `effect_dig.cpp:155,236`, `effect_dig_until.cpp:57`,
    `effect_change_zone.cpp:635,883,961,1014,1024,1025`, `effect_reveal_hand.cpp:25,52`,
    `orderer.cpp:102,359,441`, `companion.cpp:100`, `action_processor.cpp:3399`,
    `unless_payment.cpp:88`;
  - library-top views: `LibraryTopView` at `effect_dig.cpp:96`, `effect_change_zone.cpp:142`,
    `effect_change_zone_all.cpp:161`;
  - `Orderer::note_library_card_known` (`orderer.cpp:516`).
- **Done when:** the old bookkeeping is untouched (corpus and `obsinv` green), and unit checks
  show windows opening and closing at the right moments in a few sculpted harness lines
  (Thoughtseize, Brainstorm, Ponder, a tutor that reveals, a face-down exile). Face-down
  exile follows CR 406.3 (register 35): whoever looked at the card before it was exiled face
  down, or may look at it, keeps a look window until it leaves exile.

### T2. Handles

- A per-viewer table on `Game` (covered by `snapshot.cpp`), reset in `init_ecs` per game.
- **Rules** (`state_format.md` §6):
  - a handle exists only while the viewer can individually identify the object;
  - a public→public move keeps the old→new link;
  - becoming unidentifiable retires the handle;
  - new handles are issued in canonical order (table order, then zone/position), never entity
    order.
- **Done when:** a unit check shows no handle survives a hidden→hidden move without a reveal,
  and issue order is independent of entity ids. Test it by building the same position with
  two different deck orders.

### T3. Schema and encoder (snapshot tables + once-per-match frames)

- `schema.h` from T0's column appendix. Include the schema hash, the version, and enum string
  tables generated from the C++ enums (step names, `ActionCategory`, zones, keywords, counter
  kinds, card types, event kinds). Do not copy them by hand.
- **Encoder for every snapshot table** in `state_format.md` §4. Main sources:
  - battlefield via the `src/queries/battlefield.h` accessors (phased-out permanents included
    and flagged, as the serializer does today);
  - characteristics via `src/queries/characteristics.h` (`effective_*`);
  - counters (`src/queries/counters.h`), keywords (`src/queries/keywords.h`), the stack,
    delayed triggers (`src/queries/delayed_triggers.h`), player effects
    (`src/queries/player_effects.h`), play permissions (`src/queries/zones.h`), LKI
    (`src/queries/lki.h`);
  - the menu from the live `std::vector<LegalAction>`, with handles for source/target;
  - pending context from `PendingQuery` (`src/pending_query.h`), plus `pending_cost` during
    every cost-payment decision (register 33). The remaining cost lives in the payment
    flows: the interactive payment loop in `src/mana_system.cpp` from about line 1553, the
    pending-cast state in `src/action_processor.cpp`, and `src/unless_payment.cpp`. Expose it
    through a query rather than reading those flows' locals.
- **Once-per-match frames:**
  - `CARDS`: vocab id, name, uid, printed characteristics, `CardData::oracle_text`,
    abilities by reference with a keyword key;
  - `DECKLISTS`: names, closed-decklist aware;
  - `SCHEMA`.
- **Watch for:**
  - tokens have no `CardData` (guard every read);
  - sideboard phase: the viewer is `sideboard_phase_player` (see `input_logger.cpp`, around
    the BQUERY path);
  - deterministic output only: no `unordered_*` iteration in emitted order.
- **Done when:** frames encode at every decision of corpus games, with no crash and no
  `ERROR:`.

### T4. Observation events

- Typed constructors that take already-filtered data (viewer + visibility answer), never a raw
  entity. Per-viewer queues on `Game`, drained into the viewer's next frame. Game/match
  boundary events. Event kinds are listed in `state_format.md` §5.
- **Emission points:**
  - `Orderer::add_to_zone` (`orderer.cpp:105`, the main zone-move choke point; see the
    known-top push at :265-276 and the hand-identity logic at :325-346);
  - `close_zone_gap` (:364);
  - `begin_cast_move` / `complete_cast_move` / `rewind_cast_move` (:415-475);
  - `put_in_library_at_depth` (:523);
  - `shuffle_library` (:542);
  - `draw` / `draw_one` / `perform_draw` (:686-734);
  - `mill` (:838);
  - the reveal/look sites from T1;
  - mulligan (pregame in `game_driver.cpp`);
  - face-down transitions.
  - draw provenance (register 36): `turn` on every event, the opening-hand flag on `draw`,
    `hand_subset = drawn_this_turn` on Sylvan Library's put-back (`effect_sylvan_library.cpp`,
    which reads `Player::cards_drawn_this_turn`), and the draw link on the Miracle reveal
    (`effect_miracle.cpp`, the first-draw check in `orderer.cpp` around :804).
- **Done when:** every place today's bookkeeping changes has a matching event. Verify by
  grepping `known_top_library_*`, `identity_known =`, `revealed_in_library.` and
  `mark_card_revealed`.

### T5. Decoder, dumps, machine format

- C++ `decode.*` (std-only).
- **Flags:**
  - `--state-dump json` renders decoded frames as JSON;
  - `--rmstate-dump FILE` writes every viewer's frames (`A`, `B` and `omniscient`) to a file
    while the normal protocol (BQUERY) runs unchanged;
  - `--machine-format rmstate` replaces BQUERY on stdout for RMSTATE consumers.
- Add each flag to the flag table in `src/cli_args.cpp` and to `cli_print_help`
  (`src/cli_output.cpp`).
- **Done when:** decode → re-encode of dumped frames is byte-identical.

### T6. Python decoder

- `python/rmstate.py`, built from the `SCHEMA` frame. Decode with `struct.iter_unpack` /
  `numpy.frombuffer` structured dtypes; no copied constants.
- **Done when:** for dumped frames, the Python decode equals the C++ `--state-dump json`.

### T7. Engine self-checks (`--selfcheck-state`)

Failures print `ERROR:` lines (the gate's transcript scan fails on them).

- **roundtrip:** encode → decode → re-encode, every decision.
- **scramble leak:** encode each viewer's frame twice, once normally and once with a test-mode
  scrambler in the identity accessors that substitutes a different card id for every entity
  not visible to that viewer. The bytes must match. Apply the same to every event when it is
  constructed. This catches any read of a hidden identity, so all identity reads in the
  encoder and events must go through the visibility accessors.
- **permutation leak (stronger, sampled):**
  1. snapshot (`snapshot.cpp`);
  2. physically permute both hidden zones' contents and both libraries' order, keeping live
     reveal windows fixed;
  3. encode, compare, restore.

  The current search server's DETERMINIZE (`src/search_server.cpp`, around :330-390) is a
  reference for shuffling hidden zones safely.
- **complete:**
  - every public object present;
  - counts match zone sizes;
  - every handle reference resolves;
  - every menu handle exists.
- **snapshot:** save → K actions → restore, and both viewers' frames match the pre-save
  bytes.

Self-checks are off under `--search-server`, because the cooperative-unwind path returns early
from `get_input`.

### T8. Gate tier `rmstate` (Python gate, `train/ci_check.py`)

- **Flag injection:** add an environment variable (e.g. `ROBOMAGE_ENGINE_FLAGS`) read by the
  base `RoboMageEnv._extra_engine_flags` (`train/env.py:1115`). The tier then runs corpus-style
  scripted games and a short explore fuzz with `--selfcheck-state --rmstate-dump`. Document the
  variable in `docs/ci.md`.
- **The tier runs:**
  - the self-checks;
  - the Python-vs-C++ decode comparison (T6);
  - the include-isolation grep (section 3).
- Add `rmstate` to `ALL_TIERS`.
- **1b exit:** `make check` green with `rmstate`, corpus byte-identical, `obsinv` green.

### T9. Knowledge model and update rules (`src/knowledge/`)

- **Model:** pools, units with known identity, and the flow-over-time network
  (`knowledge_store.md` §3). Roots: game 1 = registered main deck; games 2–3 = registered 75
  split into deck + sideboard; `--closed-decklist` = unconstrained.
- **Update rules:** one function per event kind, exactly the rules of `knowledge_store.md` §5.
- **Per-frame consistency check:** pool sizes equal snapshot zone counts, and snapshot-visible
  identities agree with the model. A mismatch prints `ERROR:`.
- **Serialisable state:** the engine snapshot captures it through a registered hook. The
  knowledge code does not include engine headers; the engine calls its serialize/deserialize.

### T10. Solver

- Bound propagation, then a memoised exact feasibility search per (pool, card, count) query
  under a per-frame step budget, setting `k_meta.exact`.
- Network compaction after shuffles, so size stays bounded.
- **Done when:** unit tests pass against an independently written naive enumerator on small
  synthetic event streams (`knowledge_store.md` §7, exactness).

### T11. Integration

- `--knowledge-a/-b off|expert` (default `expert`).
- A per-seat instance fed from that seat's encoded frame bytes, appending `k_*` tables before
  emission.
- `rmknow` tool (section 3).

### T12. Knowledge verification (`--selfcheck-knowledge`) and gate tier `knowledge`

- **soundness:** compare every `k_*` fact against the omniscient frame's truth. Zero
  tolerance.
- **isolation:** `rmknow` on the `--rmstate-dump` stream equals the in-process `k_*` tables
  byte for byte.
- **snapshot:** knowledge state survives restore unchanged.
- **shadow comparison:** at each decision, compare `expert`'s implied view (own exact top
  positions 0–4; definitely-in-hand opponent cards; cards seen this match) with the engine's
  `known_top_library_*`, `opp_known_hand` and revealed bits.
  - Every mismatch must match a signature in a new `train/regression/knowledge_divergences.json`,
    each with a classification (old-vector leak or quirk) and a regression scenario.
  - An unclassified mismatch fails the tier.
  - **Expected entry:** the Brainstorm put-back leak, where per-card `identity_known` reveals
    which cards went back (`knowledge_store.md` §7).
- **sharpness scenarios:** new `train/regression/scenarios/*.json` asserting expected
  deductions on the knowledge dump. This adds one assertion key to `train/test_scenarios.py`,
  e.g. `expect_dump` regexes over `--state-dump json` lines.
- Add `knowledge` to `ALL_TIERS`.
- **1c exit:** every tier green; every shadow mismatch classified; the old vector unchanged.

## 5. Gotchas

- **Mid-resolution decisions** (targets, dig picks, scry) are frames too; reveal windows are
  typically live during them. The engine never asks a seat whose only legal action is a pass,
  so a viewer can miss whole reveals; that is why events exist.
- **bo3:** `init_ecs` resets the ECS per game. Handles, queues and windows reset per game; match
  memory persists (`MatchContext`, `src/game_driver.cpp`). Sideboarding frames have the
  sideboarding player as viewer.
- **Last-known info:** use `departed_lki_for` (`src/queries/lki.h`) for departed objects.
- **Identity:** `ObjectRef` across zone changes, never a bare `Entity`.
- **Phasing:** do not add your own phasing checks; the battlefield accessors handle it.
- **The in-process actor** (`src/actor/`, `InputLogger::input_provider`) must keep working.
  Do not break the machine path it shares.
- **Description strings** go into frames, so the leak checks cover them. Watch `choice_labels`
  and `announce` text for hidden names.
- **New rules gaps or deviations:** record them in `docs/rules_deviations.md`.

## 6. Exit for step 1 (from the plan)

- Every `rmstate` and `knowledge` check green in `make check`, with the corpus byte-identical
  and `obsinv` green.
- Every shadow-comparison mismatch classified in `knowledge_divergences.json`.
- The docs updated wherever implementation changed a detail: the schema appendix, event
  fields, the register.

Report to the user at the end of 1b and again at the end of 1c, before starting step 2.
