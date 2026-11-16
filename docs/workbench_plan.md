# The workbench: a GUI for analysis and play

A desktop application that is analysis software first and a way to play against the model second. Its
look and behaviour are fixed by the mock-up at <https://claude.ai/artifact/3CpAVcRnNEpzjBjroc6gqZ>
(the "workbench"; the earlier 90s-style analyzer and play-table mock-ups are superseded). This plan
lists what the real application needs that the mock-up only pretends to do (§2), analysis tools that
fit the same frame (§3), and the screens still to be mocked up before building (§4).

## 1. Decisions taken

| | Decision |
|---|---|
| Framework | Dear ImGui (docking branch) + ImPlot, on SDL2 or GLFW with OpenGL 3. Plain C++, no exceptions needed, so it builds with the engine's flags |
| Look | ImHex-like dark theme: blue tab strips, monospace UI font (JetBrains Mono, embedded), tables with alternating rows, a status bar |
| Layout | Dockable, tileable tool panels; drag a tab to re-dock, drag splitters to resize, close and reopen from **View**; preset layouts per mode (Replay (ImHex), Play, Analysis, Table only). ImGui's docking provides this; `imgui.ini` persists it and the DockBuilder API builds the presets |
| Modes | **Play** (a live game against a model) and **Replay** (a recorded game), each with its own layout |
| Platform | Desktop only; no phone layout |
| Cards | Full-colour Scryfall images, oldest paper printing by default (non-promo, non-digital), per-card override under Edit → Settings only. Magnify by holding Q over a card or right-clicking it; no other enlargement |
| Table | MTGO-style: opponent's lands and permanents on top, the phase and the stack (only when not empty) across the middle, your permanents, lands and hand below. Opponent's hand is a count only. Cards size themselves to fill the Table panel. Background: colour, texture (default wood) or image |
| Passing | F2: pass to the end of the turn unless something goes on the stack. F6: pass to the end of the turn whatever happens, taking the first choice at every decision |

## 2. Unimplemented features the mock-up needs

Each item names what in the engine or its tools it builds on. Sizes are rough: S (days), M (a week
or two), L (more).

### 2.1 The application shell

**Status: done** (`tools/rmwork/`, `make workbench`, `bin/<config>/rmwork`). Play and Replay, each with
its own dockspace and layout (one `imgui.ini` under `~/.config/robomage/workbench/`); the nine tools as
placeholders, the Value Graph's empty axes in ImPlot; the four presets (DockBuilder, the mock-up's split
ratios) and the user's named layouts per mode (View → Save Layout As…, asking before it overwrites; the
saved names under the presets; Manage Layouts… to show, rename or delete), each saved as its tree of
splits and tab groups in `settings.json` and rebuilt like a preset; View's tool toggles, Reset Layout and
mode switch; the menus, toolbar and status bar (the configured engine binary found or missing, the
inference server's socket present or not connected); the key table (← → Home End, F2 / F6, Q reserved,
the menu shortcuts), rebound from Settings → Keys (capture the next chord, Esc cancels; a chord another
action answers to where both work is refused; a reset per row and Reset All); eight colour themes, each a
full set of the mock-up's colour tokens and the only source of colour (ImHex Dark, the default and the
mock-up's look; ImHex Light, ImGui Dark, ImGui Classic, Solarized Dark and Light, Nord, Windows 95), and
eight embedded UI font families (JetBrains Mono, Fira Code, IBM Plex Mono, Source Code Pro, Cascadia
Mono, Hack, and the proportional Inter and IBM Plex Sans) in two weights at 11–20 px (default JetBrains
Mono Medium 14), chosen in Settings → Appearance and applied live; Settings' other tabs: table
background, the printings placeholder and the image cache directory, Engine (binary, working
directory), Models (server socket, the `id = model` policies, run directory), Stops (Play's default
stops, your turn and the opponent's, per step), each path marked found or missing, stored and validated
in `settings.json` for the stages that use them; the error dialog (`App::report_error`). Checked by
`rmwork --selftest` (the gate's opt-in `workbench` tier); `rmwork --screenshot` renders a PNG. Waiting
for later stages by design: Enter / arrows in Choices once Choices has rows (§2.3); the data links
followed in §2.2.

| Feature | Notes | Size |
|---|---|---|
| Binary and build | `bin/<config>/rmwork`, sources under `tools/rmwork/`, `make workbench` (not part of plain `make`, like `make actor`). Vendors Dear ImGui (docking), ImPlot and stb_image; links the std-only decoder (`src/state_format/decode.h`), the frame reader, `src/featurize/` and `src/knowledge/` like `rmfeat` / `rmknow`, no engine TU, no libtorch | M |
| Theme and fonts | An ImHex-like ImGui style (rounding 0–2, padding) under eight colour themes; the UI fonts embedded (eight families, two weights each) | S |
| Docking, presets, persistence | Dockspace over the window; one saved layout per mode; presets built with DockBuilder; View → tool toggles, presets, Reset Layout; user-named layouts (Save Layout As…) | S |
| Settings file | `~/.config/robomage/settings.json`: table background, engine binary path, models and the inference server, default stops, key bindings, card-image cache location | S |
| Menus, toolbar, status bar | As mocked; status bar shows mode, decision, turn/step/decider, V, and engine and inference-server health | S |
| Keyboard | ← → Home End (Replay), F2 / F6 (Play), Q (magnify), Enter / arrows in Choices, Esc; rebindable | S |
| Error handling | Engine exits, a frame that fails to decode, rmserve disconnects: a modal with the stderr tail and a "copy diagnostics" button | S |

### 2.2 Data: recorded games

**Status: done** (`tools/rmwork/`: `loader`, `match`, `episodes`, `replay_cache`, `replay_session`,
`game_log`, `file_picker`, `run_browser`). File → Open Replay (`rmwork --open LOG`) replays a decision
log on a background thread, `<engine> --replay LOG --rmstate-dump DUMP --narrative` in the engine's
working directory (`posix_spawn`, stdout and stderr into files), with its progress (decisions replayed,
then indexed) and a Cancel in the status bar; the engine is Settings → Engine's binary, except that a
`robomage_nn` with a `robomage` beside it replays with that `robomage` (a replay needs no learned policy,
and plain `make` keeps it current). The dump and the narrative are cached under
`$XDG_CACHE_HOME/robomage/workbench/replays/<log hash>-<engine hash>/` (the engine's resolved path, size
and modification time, so a rebuild replays again; written to a `.tmp-<pid>` directory and renamed into
place once the engine exits 0; the oldest entries pruned past 4 GB), not beside the log, since run and
corpus directories are not ours to write. The dump is indexed once: per decision the three viewers'
`STATE` frame offsets (A, B, omniscient, in the dump's order, their headers' decision keys checked to
agree), the deciding seat (the seat frame that carries the menu), game and sideboarding, turn, step,
life, the reply; frames are decoded again on demand through a small LRU cache, never a whole match. The
decision count must be the log's. Best-of-three: the toolbar's game list holds the games and their
sideboarding phases ("Game 1", "Sideboarding 2", "Game 2"), each game's winner and the match result (from
the omniscient frames' `game_end` / `match_end` events); choosing one goes to its first decision. The
viewer toggle (A / B / Omni, seat A at start) selects the frames the tools read. The Game Log shows, in a
seat view, that seat's own log built only from its own frames' event tables (the CLI's `event_lines`, with
turn and game headers; decision i's lines are its frame at decision i, the terminal frame's at the last),
and in Omni the narrative cut at the replay's per-decision lines (`(REPLAY) [… ] Input: n`, written as each
logged decision is taken: decision i's lines run to its marker, which becomes "» Player X: <the chosen
row>"); the current decision's lines highlighted with a rule after them, earlier ones normal, later ones
dimmed, a click going to a line's decision, the view following the cursor. Navigation: first / previous /
play-pause (0.6 s a decision) / next / last and ← → Home End, the status bar's decision `n / N`, game,
turn, step and deciding seat; next swing stays disabled until §2.7. Episodes: File → Open Trajectory
Episode (`--episode`) opens its decision log (its `rmlog` attribute, else by name) with it, joining each
step to the decision of its (game, sideboarding, decision) key with the same seat and reply (μ, versions,
evaluations and teacher label kept per decision for §2.6–2.7); a log's own episodes (`<stem>.g<n>.rmtraj`
beside it, in `<run>/episodes/<purpose>/` for `<run>/rmlog/<purpose>/`, or an actor's `traj/`) join
whenever it is opened. File → Open Run Directory (`--run`) lists a run's logs by purpose, with the games
their episodes cover, a filter and "with episodes only" (read on a thread: a run holds 180 000 logs);
choosing one opens it with its episodes. The file picker (a path field, places, recent files kept in
`settings.json`, the listing in natural order with size and time read only for the rows drawn, filters for
decision logs, episodes, both, directories, all) serves every Open. Engine exits (the `FATAL` line, the
stderr and stdout tails), dumps that do not decode, malformed logs, a missing engine and episodes of
another match reach the error dialog. Checked by `rmwork --selftest-data` (the gate's `workbench` tier).
Waiting for later stages: every other tool's contents (they say which game and decision they will show:
the Table, Players, Choices, Inspector and State Tree from the frames, §2.5–2.6), the episodes'
evaluations in the Value Graph, Swings and next swing (§2.7), the full Episode Browser and the run's other
files (`bench/`, `metrics.jsonl`, §3).

| Feature | Notes | Size |
|---|---|---|
| Open an `.rmlog` | Run the engine `--replay FILE --rmstate-dump DUMP` (every viewer's frames at each decision) and `--narrative` (the omniscient log, stderr) in a subprocess; decode the dump; index decisions. Cache the dump beside the log | M |
| Open a trajectory episode | `read_episode` (`src/featurize/episode.h`): μ, policy versions, `eval_value` / `eval_log_pi` per policy, `teacher` labels, game boundaries; join to the replay's decisions by index | M |
| Open a run directory | `runs/<name>/`: `rmlog/`, `episodes/`, `bench/`, `ladder/`, `ablation/`, `metrics.jsonl`, `evals.jsonl`; feeds the episode combo box and the run tools (§3) | M |
| Best-of-three | Game tabs or a game strip; sideboarding phases as decisions (`sb` frames); the match result | S |
| Viewer frames | Seat A, seat B and omniscient frames for every decision (`--rmstate-dump` writes all three), so the View toggle shows exactly what each seat saw, with its `k_*` tables | S |
| Per-viewer log | A seat view builds its log from that seat's event tables (80–82); omniscient view uses the narrative. The mock-up shows the narrative in every view, which leaks hidden information in a seat view | M |

### 2.3 Data: live games (Play)

| Feature | Notes | Size |
|---|---|---|
| Engine session | Start `robomage_nn --machine` (or `robomage` for non-`nn` opponents) with the human's seat on the RMSTATE stream: read frames from stdout, write one integer per decision (`docs/state_format.md` §9). Write `--decision-log` so every game can be opened in Replay | M |
| Opponents | `nn:<model>` / `nn:remote:<socket>` in process; `explore`, `random`; the scripted agent through the teacher channel at `--teacher-mix 1` (`--teacher-b "python3 ../python/teacher.py --tier hard"`), which already lets the teacher play a seat | S |
| F2 / F6 and stops | Client-side auto-pass. Recognise "pass priority" by the menu row's category, not its text. F2 stops when the stack gains an object or the turn ends; F6 takes row 0 at every decision until the turn ends; per-step stops pass priority automatically in unstopped steps when the stack is empty | M |
| Concede | `-2` / `-3` with a confirmation | S |
| Hidden information | Play renders only the human seat's frames, never omniscient ones | S |
| Model's read while playing | Featurize the human seat's frame in the workbench (`featurize_seat`) and evaluate it through `rmserve` with the torch-free remote evaluator (`src/actor/remote_evaluator`, `src/inference/protocol`); asynchronous, with a "thinking" indicator; can be turned off for an unaided game | M |

### 2.4 Card images and printings

| Feature | Notes | Size |
|---|---|---|
| Image cache | `tools/scryfall/fetch_images.py` (stdlib, like `tools/forge_fetch`): for every vocab card and token, list paper printings (`unique=prints`, oldest first), choose the oldest non-promo printing, download `normal` images into `~/.cache/robomage/scryfall/`, rate-limited, with a manifest. The C++ app reads only the cache, so it never touches the network | M |
| Printing overrides | `~/.config/robomage/printings.json`: card name → Scryfall set + collector number; the Settings tab lists all printings (the mock-up offers eight); a missing image is fetched on demand by the script | S |
| Special faces | Double-faced cards (combined scripts, `CARDS` face rows): show the face up; flip on hover. Face-down objects (morph, manifest, cloak): the card back, with the identity shown only when the viewer knows it. Tokens: Scryfall token printings, with a text-rendered fallback | M |
| Offline fallback | When an image is missing, render a text card from `CARDS` (name, cost, type line, oracle text, P/T) | S |
| Textures | stb_image decode, GPU textures with an LRU cache; images streamed in off the UI thread | S |

### 2.5 The Table

| Feature | Notes | Size |
|---|---|---|
| Layout from frames | Battlefield rows split into lands and other permanents; attachments (Auras, Equipment) drawn overlapping their host; identical untapped lands stacked; tokens; phased-out permanents dimmed (the frame flags them); emblems and player effects in a strip by the player | M |
| Object state | Tapped, counters by kind, effective P/T (post-layer, from `object`), damage marked, summoning sickness, loyalty, designations (monarch, day/night, speed …; `designations.def`), energy and poison on the player | M |
| Combat | Attackers set forward; arrows from blockers to attackers; damage assignment order | M |
| Targets | Hovering a stack object highlights its targets (`stack_target` rows, not text matching as in the mock-up); optional arrows | S |
| Card-to-choice linking | Use each menu row's `source` / `target` handles (`menu` table) to link rows and objects, replacing the mock-up's name matching | S |
| Auto-fit | Size cards to fill the panel, as mocked | S |

### 2.6 Other tools in the mock-up

| Tool | Real implementation | Size |
|---|---|---|
| Choices | The `menu` rows with descriptions; Play: choose; Replay: ► on the chosen row; π bars from the evaluations | S |
| Decision Inspector | From the frame (`header`, `pending`, `pending_cost`) and the episode (μ, `eval_*`, versions, teacher); one column per evaluated policy | S |
| State Tree | Generic over the SCHEMA: every table of the current frame as a tree node, rows as children, every column with its semantic tag; table-specific renderers for players, zones and objects; hover ↔ Table highlighting through handles | M |
| Value Graph | ImPlot: V per seat and per policy, the result, life strips, turn bands, cursor and scrubber; series chosen from the loaded policies | S |
| Swings | Computed from the loaded evaluations; seat disagreement needs both seats' evaluations (re-evaluate the other seat's frames, §2.7) | S |
| Model's Read | As mocked, from the current evaluations | S |
| Players | From `player` rows and zone counts | S |
| Zone windows | Graveyard and exile in order, with linked-exile sources; the library only where the viewer knows cards (`k_*`) | S |

### 2.7 Evaluation

| Feature | Notes | Size |
|---|---|---|
| Recorded evaluations | Read `eval_value`, `eval_log_pi`, `eval_policy`, `eval_version` from episodes | S |
| Re-evaluation | Evaluate any decision's seat frames (either seat) with any checkpoint through `rmserve` (`--policy ID=MODEL …`), batched over the whole game; cache results per (game, policy, seat) on disk | M |
| Model manager | List exports from a run (`runs/<run>/exports`, `ckpt`), start or attach to an `rmserve`, check compatibility (`check_compatible`, the model manifest) | M |

## 3. Additional analysis tools that fit the framework

Ordered roughly by value against effort. Each is one dockable tool.

| Tool | What it shows | Built on | Size |
|---|---|---|---|
| **Knowledge** | What the viewing seat knows or has deduced: opponent's hand candidates, known library positions, cards seen; the CLI's `know` panel as a tool | `k_*` tables (`docs/knowledge_store.md`) | S |
| **Policy Compare** | π and V of several checkpoints at the current decision and across the game: per-decision KL divergence, argmax agreement, where a newer checkpoint changed its mind | §2.7 re-evaluation | M |
| **Teacher Disagreement** | Where the scripted teacher's label differs from the model's choice, with both π; filterable list that jumps into Replay | episode `teacher` labels; or attach `teacher.py` to a replay | S |
| **Branch Explorer** | At a decision, take a different menu row and play the game out K times with chosen agents; per-row win rate next to π (a Monte Carlo Q). Uses the decision log's prefix plus the alternative reply as `--actions`, then in-process agents finish the game | engine subprocess, `--actions`, agents | M |
| **Influence Overlay** | Per decision, replace one card's input with the unknown-card vector (or remove its token) and re-evaluate: tint each card on the Table by its ΔV / Δπ(chosen) | featurizer + rmserve; the method of `learn/ablate.py` | M |
| **Blunder Finder** | Query decisions across many games: e.g. π(chosen) < 0.2 and |ΔV| > 0.3, or a row the teacher picks with π < 0.1; results open in Replay | episodes + evaluations | M |
| **Episode Browser** | Table of a run's episodes (decks, seats, result, length, policy versions, actor), filters and summary statistics; double-click opens Replay | `runs/<run>/episodes`, results files | M |
| **Bench Matrix** | Win-rate heatmap per deck pair from `rmtrain bench` / ladder, with confidence intervals; a cell lists its games | `runs/<run>/bench`, `ladder` | S |
| **Calibration** | V against the final result across games: reliability diagram, by turn, by deck, by seat | episodes + evaluations | S |
| **Training Curves** | Loss, entropy, value error, throughput over steps, with checkpoints marked | `metrics.jsonl`, `evals.jsonl` | S |
| **Confidence Timeline** | H(π), max π and the μ / π gap as extra Value Graph series; picks out decisions where the model was unsure | evaluations | S |
| **Card Statistics** | Per card across games: drawn, cast when castable, average ΔV after casting, win rate when drawn or in the opening hand | episodes + frames | M |
| **Mulligan & Sideboard** | Keep rates by hand, results after a mulligan; sideboard swaps (including `sb_explore` proposals) and later game results | episodes (bo3) | M |
| **Raw Frame** | The current frame's RMSTATE tables with SCHEMA column names and tags (the "hex editor" of the workbench), for engine and format debugging | decoder | S |
| **Tokens** | The featurizer's tokens and relations for the decision: types, categorical and numeric columns, the menu's `act_tok`, relation graph | `featurize_seat` | M |
| **Bookmarks & Notes** | Mark decisions with notes, stored beside the game; list, jump, export to Markdown | sidecar file | S |
| **Game Compare** | Two games side by side, e.g. one seed played by two checkpoints: synchronised by turn, two Tables and two value curves | two loaded games | M |
| **Belief Spread** (research) | V re-evaluated over hidden worlds consistent with the seat's knowledge, showing how much V depends on what the opponent holds | knowledge solver worlds; needs frames built for sampled worlds (state format register 22, deferred) | L |

## 4. Screens and views still to mock up

### 4.1 Starting and loading

1. **Start page**: recent games and runs, New Game, Open Replay, Open Run, inference-server status.
2. **New Game dialog**: your deck and seat, opponent (model / checkpoint, scripted tier, explore), best-of-one or best-of-three, seed, auto-mana, Model's Read on or off, knowledge level.
3. **Open Run / Episode Browser**: the run's episodes as a filterable table (§3).
4. **Model manager**: exports and checkpoints, rmserve start/attach, compatibility check.

### 4.2 Decisions in Play (direct manipulation on the Table)

5. **Mulligan**: the opening hand large, Keep / Mulligan, and for London mulligan the bottoming choice.
6. **Targeting**: legal targets highlighted on the Table and players, click to choose, arrows from the spell.
7. **Declare attackers / blockers**: click creatures to attack or block, arrows to their opponents, confirm (`-1`).
8. **Library search, dig, scry, surveil, reveal**: a card-picker window with the shown cards, top/bottom placement, fail-to-find.
9. **Modal and numeric choices**: choose modes (one or more, `mode:done`), choose X, divide damage among targets, order triggers.
10. **Name a card**: the limited candidate list (or every vocab card under `--closed-decklist`), with search.
11. **Mana payment**: under `--no-auto-mana`, tap lands and sources on the Table to pay, with the remaining cost (`pending_cost`).
12. **Optional costs and alternative costs**: e.g. Force of Will's pitch, kicker, convoke and crew pickers.
13. **Discard / sacrifice / return pickers** for mandatory choices.

### 4.3 Between and after games

14. **Sideboarding** (best-of-three): main deck and sideboard lists with counts, swap in/out, the model's suggested plan, Done.
15. **Game over / match result**: winner, score, the game's value curve and big swings, buttons to open the game in Replay or play again.
16. **Concede confirmation**: game (`-2`) or match (`-3`).

### 4.4 Analysis tools (from §3)

17. Knowledge.
18. Policy Compare.
19. Branch Explorer (with its rollout configuration and progress).
20. Influence Overlay (on the Table, with its legend and settings).
21. Blunder Finder (query builder and results).
22. Bench Matrix, Calibration, Training Curves, Card Statistics, Mulligan & Sideboard.
23. Raw Frame and Tokens.
24. Bookmarks & Notes.
25. Game Compare.

### 4.5 Settings and system

26. **Settings**, the remaining tabs: Engine (binary, flags), Models (rmserve socket, policies), Keys, Stops (defaults per step), Appearance (font size, UI scale).
27. **Layout manager**: save, rename and delete named layouts per mode.
28. **Error dialog**: engine or inference-server failure with the stderr tail.
29. **Card detail** (optional): the magnifier extended with oracle text, rulings links and the card's parse report, if a text view beside the image is wanted.

## 5. Suggested order

1. Shell, theme, docking and presets (§2.1).
2. Replay of a recorded `.rmlog` with all three viewers: Table, Players, Choices, Inspector, State Tree, Log, Value Graph from recorded evaluations (§2.2, §2.5, §2.6). Mock up §4.3's game-over screen first, since Replay needs it.
3. Card images and printings (§2.4).
4. Play against a model with F2 / F6 and stops, through the menu list first (§2.3); then the direct-manipulation decisions of §4.2, one category at a time.
5. Re-evaluation and the model manager (§2.7), then Policy Compare, Knowledge and Teacher Disagreement.
6. The run tools: Episode Browser, Bench Matrix, Calibration, Training Curves.
7. Branch Explorer, Influence Overlay, Blunder Finder, then the rest of §3.
