# Text-Output Surfaces Audit

A review of every code path that produces human-readable text describing game
state, available actions, combat, casts, zone changes, and similar — across all
run modes — with a plan to unify the duplicated logic.

The guiding constraint: output **must** legitimately vary by mode (binary vs
graphical vs console; perspective-relative for ML vs absolute for debugging).
The goal is therefore **one source for each semantic primitive, with
mode-specific presentation layered on top** — not one identical string
everywhere.

## 1. The output modes (inventory)

| # | Surface | Language | Entry point | Selected by |
|---|---------|----------|-------------|-------------|
| 1 | CLI / interactive console | C++ | `cli_output.cpp` (`print_game_state`, `print_query`, `game_log`) | default (no flag) |
| 2 | GUI (raylib) | C99 | `gui.c` reading shared buffers / `GameState` | `--gui` |
| 3 | Machine / BQUERY (binary) | C++ | `cli_emit_machine_query` | `--machine` |
| 4 | Narrative log (overlays 1–3) | C++ | `game_log()` dispatch in `cli_output.cpp` | `--narrative` (implicit in CLI/GUI) |
| 5 | Python decoders — `test_harness.py`, `tui_game.py` | Python | `train/decode.py` | running those tools |
| 6 | Python ad-hoc renderers — `train.py` (observe/watch/diag/replay), `analysis.py`, `play.py` | Python | private tables in each file | running those tools |

There is **no separate TUI in C++** — "TUI" means either the CLI console
(surface 1) or the Python `tui_game.py` Textual app (a surface-5 consumer). The
raylib GUI is the only graphical mode. `make HEADLESS=TRUE` drops `-DGUI` and
`-lraylib`; `gui.c`'s body is entirely behind `#ifdef GUI`.

## 2. What is already well-factored

- **Narrative is single-sourced.** Every "casts X", "deals N damage",
  zone-change, and combat line routes through one variadic `game_log(...)`
  (`cli_output.cpp:98`), dispatched to GUI ring-buffer / stdout / no-op. The GUI
  is a dumb renderer of these engine-authored lines (`gui.c:783-834`).
- **Action labels are single-sourced within C++.** `LegalAction::description`
  is built once in `state_manager_actions.cpp` (+ target labels in
  `action_processor.cpp`) and consumed verbatim by CLI (`print_query`) and GUI
  (`render_choices`).

## 3. The variations (where logic forks)

### A. Action descriptions reimplemented 3+ times, two languages, divergent wording

`LegalAction::description` is intentionally **not** serialized into the BQUERY
payload (only category int + card-vocab id + controller/public flags cross the
boundary). Each Python tool reconstructs labels from scratch:

| Implementation | Style example | Used by |
|---|---|---|
| C++ inline (`state_manager_actions.cpp`) | `Cast Lightning Bolt`, `Activate Tarn (ChangeZone)` | CLI, GUI |
| `decode.describe_action` (`decode.py:248`) | `Cast Lightning Bolt`, `Target Grizzly Bears (opp)` | test_harness, tui_game |
| `train.py._describe_action` (`train.py:1319`) | `CAST Lightning Bolt` | observe, watch, diag, replay, analysis.py |
| `play.py._action_label` (`play.py:67`) | `cast X`, `attack with X` | play.py CLI + GUI |

### B. Card-name lookup — one data source, 5 lookup implementations

All read `card_costs._VOCAB_NAMES`, but only `decode.card_index_to_name`
handles the Token sentinel and out-of-range indices. Inline copies in `train.py`
(×2), `analysis.py` (×4 sites), and `play.py._card_name` lack Token handling and
use inconsistent fallbacks (`?(idx)` vs `?{idx}` vs none).

### C. Category/step name tables duplicated

`_CAT_NAMES` and `_STEP_NAMES` exist identically in both `decode.py` and
`train.py` (analysis.py imports train.py's copy). The C++ `ActionCategory` enum
is the real source of truth; a change must be mirrored in two Python files.

### D. Battlefield/state formatting duplicated CLI vs GUI

`print_game_state` (`cli_output.cpp:247`) and the GUI's
`draw_perm_card`/info-bars (`gui.c:249-622`) independently format the same
`GameState` fields. Card-identity strings are shared via `gui_card_info.cpp`;
layout/label strings are not.

### E. Three framings for "whose is it"

- Perspective-relative (`self`/`opp`/`own`): decode.py, test_harness, tui_game.
- Absolute (`Player A`/`Player B`): all C++, train.py watch/diag/observe, analysis.py.
- `[You]`/`[Model]`: play.py.

### F. Minor

`gui_card_info.cpp` mixes uid-keyed (`name_to_uid`) and display-name-keyed
(`card_db.find`) lookups across fields — latent bug for cards whose db key is
the uid.

## 4. Unification plan (ordered by leverage)

1. **Generate the enum/step tables from C++.** Emit a generated
   `train/_enums.py` (mirroring the `gen_card_costs.py` pattern) so
   `_CAT_NAMES`/`_STEP_NAMES` and category integers derive from the C++ headers;
   delete the hand-maintained copies in `decode.py` and `train.py`.
2. **Collapse all Python rendering onto `decode.py`.** Delete `train.py`'s
   `_CAT_NAMES`/`_STEP_NAMES`/`_describe_action`/`_decode_hand`, `analysis.py`'s
   inline vocab indexing, and `play.py`'s `_card_name`/`_action_label`/
   `_format_state`; route through `decode`'s functions.
3. **Parameterize framing** (`perspective=absolute|relative|namemap`) on
   `decode`'s formatters rather than forking strings.
4. **Decide whether `LegalAction::description` should cross the BQUERY
   boundary** (status quo keeps Python authoritative; alternative emits the
   string behind `--narrative` for byte-identical labels).
5. **Extract a shared C++ state-line formatter** for CLI and GUI.
6. **Fix the `gui_card_info.cpp` uid-vs-display-name inconsistency.**

The two highest-value, lowest-risk moves are #1 and #2.

## 5. Implementation status

### Step 1 — generate enum/step tables from C++ — **done**

- Added `train/gen_enums.py` (mirrors the `gen_card_costs.py` pattern). It parses
  the `ActionCategory` enum from `src/classes/action.h` and the `Step` enum from
  `src/classes/game.h`, and emits `train/_enums.py` with `_CAT_NAMES`
  (category int → short display name), `_STEP_NAMES` (ordered step names), and
  `ACTION_CATEGORY_MAX` / `N_ACTION_CATEGORIES`.
- The **integer values and ordering** come from the C++ enums (the source of
  truth). The cosmetic short display strings live in exactly one place —
  `_CAT_DISPLAY` / `_STEP_DISPLAY` inside `gen_enums.py`. The generator's
  `_check_coverage` **fails loudly** if the C++ enum and the display map drift
  apart (a new/renamed C++ entry with no display string, or a stale display
  entry), so an enum change forces a one-file update.
- `decode.py` now imports `_CAT_NAMES` / `_STEP_NAMES` from `_enums` instead of
  defining them inline.
- Generated tables were verified byte-identical to the previous hand-maintained
  copies in `decode.py` and `train.py`; regeneration is idempotent.

**Workflow:** codegen is wired into the Makefile. `make` (default `all` goal)
runs the `pygen` target before building, regenerating `train/_enums.py` and
`train/card_costs.py` as proper file targets — i.e. only when their C++ sources
change (`_enums.py` ← `src/classes/action.h`, `src/classes/game.h`;
`card_costs.py` ← `src/card_vocab.h`, `src/machine_io.h`), not on every build.
The `PYTHON` make-variable prefers `train/.venv/bin/python` and falls back to
`python3` (both generators are stdlib-only). To regenerate by hand:
`make pygen`, or run a single generator directly
(`train/.venv/bin/python train/gen_enums.py`). Note `card_costs.py` also depends
on the card script files, which are not Make prerequisites (too many); re-run
`make pygen` or `gen_card_costs.py` manually after editing card scripts.

### Step 2 — collapse Python rendering onto decode.py — **done**

- `decode.py` gained `decode_hand(obs)` so the priority-player hand decoder lives
  in one place (Token-sentinel and out-of-range aware via `card_index_to_name`).
- `train.py`: deleted its duplicate `_CAT_NAMES`/`_STEP_NAMES` (now imported from
  `_enums`) and rewrote `_decode_hand` → `decode.decode_hand` and
  `_describe_action`'s vocab lookup → `decode.card_from_id`. The CAT-token debug
  wording (`CAST Lightning Bolt`) for watch/diag/replay is preserved — only the
  duplicated lookup logic was removed. Dropped now-unused `_VOCAB_NAMES` /
  `_HAND_START` imports.
- `analysis.py`: repointed `_CAT_NAMES`/`_STEP_NAMES` to `_enums` (was importing
  train's re-export).
- `play.py`: deleted `_card_name` (a verbatim duplicate of `decode.card_from_id`)
  and routed its call sites through the already-imported `_card_from_id`.

**Verification:** all of `decode`, `train`, `analysis`, `play` import cleanly;
`_CAT_NAMES`/`_STEP_NAMES` are the same object (`is`) across them. `test_harness.py`
renders state/actions correctly; `train.py watch` shows correct abbreviations,
steps, and apostrophe-bearing card names (`Mishra's Bauble`, `Gaea's Cradle`);
`train.py diag` reported 7W/3L/0D with no draws or non-fatal errors.

### Item 1 — `gui_card_info.cpp` uid-vs-display-name fix (variation F) — **done**

- `gui_card_type_line`, `gui_card_base_power`, `gui_card_base_toughness` looked up
  via `card_db.find(<display name>)` while `gui_card_oracle`/`gui_card_mana_cost`/
  `gui_card_color_identity` (which render correctly) use
  `name_to_uid(card_index_to_name(idx))`. `card_db` is **only ever keyed by uid**
  (`card_db.cpp` `load_card` emplaces `name_to_uid(...)` keys exclusively), so the
  display-name lookups missed for any card whose uid ≠ display name (apostrophes,
  spaces, punctuation) — blank type line and 0/0 base P/T in the GUI.
- Fixed the three accessors to use the identical `name_to_uid(...)` + `uid[0]=='?'`
  guard + `card_db.find(uid)` pattern as the working three. All six now resolve a
  card the same way.
- **Verify:** `make` (GUI) recompiles `gui_card_info.o` and links cleanly. Code
  walkthrough confirmed e.g. "Lightning Bolt" → `lightning_bolt` resolves in all
  six accessors. (No graphical run in this environment.)

### Item 2 — collapse `analysis.py` inline vocab lookups (variation B) — **done**

- Added `import decode`; converted the **clean** name-resolution sites (including
  `_resolve_card_name`) to `decode.card_index_to_name`, keeping each site's own
  guards (e.g. `cid<0/NULL → "Player"`).
- **Deliberately kept** the empty-string-sensitive / different-fallback sites
  (`analysis.py:484`, `:1406`, `:1649`, `:1966`, `:1974`) verbatim: they rely on
  `_VOCAB_NAMES[cid]` being `""` for unused slots or use a `card#<n>` fallback,
  which `card_index_to_name` ("Token"/"?(idx)") would change. `_VOCAB_NAMES`
  import retained for those.
- **Verify:** `import analysis` OK; report output (`summary`/`cards`/`cast-timing`/
  `targeting`/`choice-rates`/`replay`) byte-identical before/after on a fresh
  recording, including a SELECT_TARGET-at-Token decision; unit-compared old vs new
  `_resolve_card_name` across valid/Token/null/out-of-range cids — 0 mismatches on
  values that occur in real data.

### Item 3 — parameterize framing on `decode.py` formatters (variation E) — **done**

- Added `SELF_OPP_LABELS` (the exact current wording: action suffix `own`/`opp`;
  stack-entry controller `self`/`opponent`) and an optional `labels=SELF_OPP_LABELS`
  parameter threaded through `_ctrl_str`, `describe_action`, `_decode_stack`,
  `decode_game_state`, `decode_actions`, `decode_actions_from_obs`. Default output
  is byte-for-byte unchanged.
- `play.py`: removed the duplicated raw-offset permanent decoding — `_perm_str`/
  `_split_bf`/`_format_state` now consume `decode._decode_permanents`/`_decode_hand`,
  with a `_PLAY_LABELS` map for its "You"/"Model" vocabulary. Kept play.py's thin
  layout assembly and its lowercase-verb `_action_label` (reproducing them through
  decode would distort decode's general API). Token-hiding behavior preserved.
- **Data-limitation note:** the observation loop (`runner.run_games`, used by
  `train.py observe` and the harness) uses **absolute** "Player A/B" / side framing
  derived from the loop's seat tracking (`obs[31]`/`obs[32]`),
  not from per-action data — the binary's per-action controller flag is relative
  (`controller_is_self`). decode therefore cannot recover the absolute seat, so
  train.py's absolute framing was correctly **left as-is** (it already shares
  card-name/hand decoding via `decode.card_from_id`/`decode.decode_hand`).
- **Verify:** `test_harness.py` 20-decision run byte-identical before/after;
  play.py formatters byte-identical over captured obs + synthetic edge cases;
  label param smoke-tested (default `(own)`, custom `(you)`); all touched modules
  import.

### Single observation loop — `observe` + `runner.py` (done)

The three overlapping `train.py` subcommands (`watch`, `diag`, `observe`) and the
separate observation loop inside `test_harness.py` were collapsed into **one**
game-driving loop:

- **`train/runner.py`** (new, lightweight — numpy + env + decode + the generated
  enum tables; torch only if a model controller is passed) owns the single
  `run_games(controller_a, controller_b, …)` loop and the unified transcript
  rendering. Default output is the compact per-decision line; `--verbose` emits the
  test-harness transcript format (`--- Decision N ---` block + `format_state_lines`
  + `format_action_lines` menu + `format_chosen_action` + `--- Narrative ---`
  blocks). Multi-game runs print per-game results + a `W/L/D` summary; a draw/stall
  dumps the full log.
- **`train.py observe`** is now a thin wrapper: it builds per-side controllers via
  `opponents.make_controller` and calls `runner.run_games`. It subsumes `watch`
  (scripted vs scripted, one game) and `diag` (`--games N` regression pass), and
  adds `--seed`, `--bo3`, and `--verbose`. `watch`/`diag` were removed.
- **`test_harness.py`** keeps only its real job — *seeding the state* (stacked
  decks, inline hands, `--battlefield-a/-b`, scenarios, `--no-shuffle`) — selects a
  controller for its play mode (`ActionListController` / `InteractiveController` /
  scripted / `AutoPassController`, all added to `opponents.py`), and calls
  `runner.run_games`. Its bespoke subprocess loop (`TestHarness` class:
  start/read_until_query/run_game/_send/kill) and confirm-slot remapping were
  deleted — the loop and the env's confirm handling are now shared.
- **`env.py`**: `RoboMageEnv` gained `narrative` / `no_shuffle` / `battlefield_a` /
  `battlefield_b` flags (appended to the engine command); `NarrativeEnv` defaults
  `narrative=True` so the shared loop captures the full engine game log the way the
  harness used to via its direct `--narrative` subprocess.
- **TUI**: `cli_spec` drops the `watch`/`diag` Subs, expands `observe`, and adds a
  `HARNESS_TOOL` so the `tui.py` launcher can compose/run `test_harness` (a `flat`
  Tool, like `play`). `CLAUDE.md` and the `engine-sanity-check` skill now use
  `observe --verbose`.
- **Verify:** `observe` compact/verbose/`--games 10`/model-or-scripted controllers
  all work; 10 scripted games → 0 draws (old `diag` parity); `test_harness` inline
  hands + `--battlefield-b` + `--scripted`/`--actions` drive through the shared loop
  and reach a winner; all modules import; `watch`/`diag` subcommands are gone.

### Remaining unification opportunities (not done)

- **Variation A — BQUERY `description`:** still Python-authoritative (decode is the
  one Python labeler). Emitting the C++ `LegalAction::description` over the wire
  behind `--narrative` for byte-identical CLI/GUI/harness labels remains a
  decision (see §4 step 4).
- **Variation D — shared C++ CLI/GUI state-line formatter:** `print_game_state`
  (`cli_output.cpp`) and `gui.c`'s `draw_perm_card`/info-bars still format the same
  `GameState` fields independently.
- **Residual framing:** `analysis.py`'s absolute "Player A/B" rendering and the
  guarded vocab sites listed above, and train.py's absolute-seat framing, are not
  routed through decode (by design — see the data-limitation note).
