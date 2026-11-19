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
its own dockspace and layout (one `imgui.ini` under `~/.config/robomage/workbench/`); the tools as
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
in `settings.json` for the stages that use them; the error dialog (`App::report_error`). At start, when no
flag opens a game, replay or run or names a mode, preset, layout or tool, the start page (§4.1, item 1)
takes the work area in place of the mode's dockspace (Settings' `start_page`, "Show this page when nothing
is open"; `--start-page` forces it). Checked by
`rmwork --selftest` (the gate's opt-in `workbench` tier); `rmwork --screenshot` renders a PNG. Waiting
for later stages by design: the data links followed in §2.2.

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
decision count must be the log's, except for a run that stopped without a result (its `--max-decisions`
cap, a recorded `QUIT`): the engine wrote the frames of the decision it stopped at, which Replay shows as
the last decision, unanswered, with the result "no result: the run stopped at decision N". Best-of-three: the toolbar's game list holds the games and their
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
turn, step and deciding seat; next swing (enabled with §2.6's Swings). Episodes: File → Open Trajectory
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

**Status: done** (`tools/rmwork/`: `play_session`, `autopass`, `model_read`, `new_game_dialog`,
`play_dialogs`; the live match in `match`). Game → New Game vs Model (Ctrl+N, the button on Play's empty
Table, `rmwork --open-new-game`) chooses your deck (the engine's top-level, `league/` and `heldout/`
decks) and seat, the opponent (a model export: Settings → Models' policies and every run's
`exports/*.pt2`, `*.ts.pt`; an inference server's socket; explore; random; the scripted agent) and its
deck, best of one or three, the seed (with Random), auto-mana and Model's Read, checked as they are made
(an `nn:` opponent needs `robomage_nn`) and kept in `settings.json` (`new_game`). A game runs `<engine>
--machine --game-seed N --deck-a … --deck-b … [--bo3] [--no-auto-mana] --decision-log <games>/<time>-<seed>.rmlog
--agent-<opponent's seat> <spec>` in the engine's working directory, the games directory
`$XDG_DATA_HOME/robomage/workbench/games` (default `~/.local/share/…`): the human's seat has no agent, so
it reads the RMSTATE stream; the opponent is `nn:<export>` or `nn:remote:<socket>` (in the `robomage_nn`
beside Settings → Engine's binary, its absence named), `explore`, `random`, or `first` taught by `python3
python/teacher.py --tier hard --seed N` at `--teacher-mix 1` (the scripted agent playing the seat through
the teacher channel); every other opponent runs the plain `robomage`. The engine's stdin and stdout are
pipes (`posix_spawn`, its own process group, killed whole with what it started), its stderr
`<stem>.stderr`; a thread reads the RMSTATE records and the UI thread appends them to the live match (one
seat's frames, kept in `<stem>.seat-a.rmstate` / `.seat-b.rmstate` as its dump), which Play's tools read
through the session: the Table, Players and zone windows, Choices, Inspector, State Tree, the Game Log
(the seat's own log, with the row chosen after each decision), Value Graph, Swings and Model's Read, the
same views as Replay's, at the newest decision; earlier decisions are browsable read-only (the Game Log,
the Value Graph; the toolbar's Latest goes back). Only the human seat's frames reach them: at the match's
end the engine writes every seat's terminal frame to the stream, the other seat's with its once-per-match
frames, and the session drops a record of another viewer unread, by its header. Choosing: a click on a
Choices row, or ↑ / ↓ and Enter, sends its index (an attacker or blocker declaration's confirm is the last
row, the `-1` of `docs/state_format.md` §9); at a targeting or payment decision, and at a priority decision
offering mana abilities, a click on the Table does too (§4.2, items 6 and 11), as do a click in the card picker and on
the opening hand (items 5 and 8), through the same reply; nothing is sent while the engine runs ("Opponent is
thinking…" in the status bar). F2, F6 and the stops (`autopass`: a pure function of the decision's facts
from the seat's frame: the `PASS_PRIORITY` row at a decision with nothing pending, the game, turn and
step, the stack's handles): F2 passes until a later turn, a decision that is not a plain priority pass or a
new object on the stack; F6 takes row 0 until a later turn; with neither, a plain priority pass with an
empty stack in a step without a stop (Settings → Stops, for the turn's player; a click on a step of the
Table's step bar toggles the same stop, §2.5) is answered. Game →
Concede Game / Match asks first and sends `-2` / `-3` (while the opponent thinks, at your next decision).
The match's end opens the result screen (the winner, the bo3 score and each game's winner, your values'
curve when Model's Read ran; Open in Replay through Replay's loader, New Game, Close); File → Save
Decision Log (Ctrl+S, the file picker's save mode) copies the game's log (in Replay, the loaded one's).
Model's Read, asked for per game: on its own thread the engine's `--cards-frame` file and the
oracle-text table load the featurizer's inputs, the torch-free remote evaluator (`src/actor/
remote_evaluator`, `src/inference/protocol`, linked into `rmwork`) connects to Settings → Models' socket
for the policies listed there with numeric ids (else policy 0), and each of your decisions is featurized
with `featurize_seat` (without recent-event history) and evaluated; the values and π join the decisions
as episode steps, so Choices' π, the Model's Read, the Value Graph and the status bar's V show them. No
server is "not connected" (a note, and the game goes on); a refused handshake or a lost server opens the
error dialog. The engine's exit mid-game, a stream that breaks, a record that does not decode and a failed
write open the error dialog with the command and the stderr tail and end the session; quitting kills and
reaps the engine. Screenshot flags: `--new-game SPEC` (`opp=…,deck=…,opp-deck=…,seat=…,format=…,seed=…,
auto-mana=…,read=…`), `--play-decisions N` and `--play-until stack|combat|sideboard|target|mana|end` (a seeded
random row at each of your decisions until then), `--game-over`; for a sculpted game `--play-setup FLAGS` (the
engine's test-setup flags, `;`-separated, as `PlayConfig::setup`, which the decision log's header records) and
`--play-rows R,R,…` (your first decisions' rows). Checked by `rmwork --selftest-data` (`selftest_play`:
the engine's arguments per opponent and the binary chosen; a bo1 against explore, Model's Read not
connected, every kept record the seat's and the other seat's only after its last decision, the stops'
answers, the result screen, its log in Replay with the seat's decisions, replies and frames (event tables
aside), Save Decision Log; F2, F6 and the stops on that game's frames and on made-up facts, and given live;
a bo3 with sideboarding as seat B; the concession dialog, `-2` then a queued `-3`; the scripted opponent;
a killed engine, a broken stream and a record that is no frame) and `rmwork --selftest` (the `new_game`
settings). Model's Read was checked by hand against `rmserve` serving `runs/league26-24h/exports/final.ts.pt`
(not in the gate: it needs a server and a model). Sideboarding (§4.3, item 14; `sideboard_plan`,
`sideboard_view`): the Sideboard tool, a tab beside the Table in the Play, Replay and Table-only presets
(a layout saved before it existed gets it in the Table's tab group), brought to the front when the human's
sideboarding phase begins (Game → Sideboard… too) and the Table given back at the next game. The engine's
phase is one swap at a time, IN first: at a balanced deck "Done sideboarding" (`SIDEBOARD_DONE`, row 0) and
a `SIDEBOARD_IN` row per sideboard card; an IN makes `match.sb_drift` 1 (the card in `pending.source_card`)
and the next menu a `SIDEBOARD_OUT` row per deck card, Done withheld; the OUT completes the swap
(`sb_swaps` + 1); a card brought in is never offered out again in the phase, nor a cut card back in; at 15
swaps the phase ends with no Done; a one-row menu is the engine's own. Every STATE frame carries the
viewer's 75 (`own_deck`) and the DECKLISTS frame both registered lists and deck names. The tool shows one
seat's 75 as card images (the viewer's own; the omniscient viewer's the deciding seat's; during the
opponent's phase a seat's view says so and shows its own): the deck grouped by type and the sideboard, a
card once with its count, IN / OUT on the phase's locked cards, the header's coming game, the opponent and
its deck, the games so far, play or draw, and the deck, sideboard and swap counts. A docked tool rather than
a modal, as the mock-up has it: Choices, the Game Log and Model's Read stay beside it, and in Replay the
same tool follows the cursor. At the human's decision, clicks plan (a sideboard card in, a deck card out, a
planned card clicked again taken back; Undo, Reset) with the result's counts in the header; Confirm, once
balanced, sends it: at each of the human's decisions of the phase the reply is found in the live menu by
category and source card toward the planned deck (`next_sideboard_step`), each step checked in the frame
that follows, Done last; the next game's frame must show the planned deck. A menu without the row the plan
needs, a reply the session refuses or a step the next frame does not show stops the sending with the reason
in the tool. Plans: the league guide for the matchup, play or draw as `match.plays_first` says
(`python/sideboard_guides.json` from the repository: a `league/<stem>` deck against another; what the other
of play and draw changes; a held-out pair noted; no row, a non-league deck or a missing file said), each
line ticked when the plan meets it, Use This Plan (the moves from the current deck to the guide's target);
the model's π over this decision's rows when Model's Read evaluated it, listed and under each card (the
model decides one swap at a time, so the tool says π is for the decision in front of you; off, not
connected or failed said, never blocking); the phase's swaps so far, in Replay the phase in all (the deck
at its first decision against the viewer's frame after it), and the deck against its registered list.
Screenshot flags `--play-until sideboard`, `--sideboard-plan guide|+CARD;-CARD…`, `--tool sideboard`.
Checked by `rmwork --selftest-data` (`selftest_sideboard`: the plan's rules and the steps on made-up
decisions, a menu without the planned row refused; the guides file against its JSON; a live bo3 as seat A
sideboarded through the tool: the guide's plan for burn vs bug and Use This Plan, an unreachable target
stopped before anything is sent, a planned swap sent as IN, OUT and Done with game 2's frame showing it;
the match in Replay in seat A's view and seat B's). Waiting for later stages: the rest of §4.2's
direct-manipulation decisions (targeting, mana payment, combat, the card picker and the opening hand are answered
on the Table; every other decision goes through the Choices list). Beside each game's decision log Play
writes `<stem>.game.json` (the human's seat, the opponent's kind and model, the decks, the format, the
seed), which the start page reads; a game that ends without its result (stopped by File → Close, a new
game or quitting, or its engine gone) gets the `QUIT` line appended to its log, as a human seat's quit
writes it, so Replay opens it to the decision it stopped at.

| Feature | Notes | Size |
|---|---|---|
| Engine session | Start `robomage_nn --machine` (or `robomage` for non-`nn` opponents) with the human's seat on the RMSTATE stream: read frames from stdout, write one integer per decision (`docs/state_format.md` §9). Write `--decision-log` so every game can be opened in Replay | M |
| Opponents | `nn:<model>` / `nn:remote:<socket>` in process; `explore`, `random`; the scripted agent through the teacher channel at `--teacher-mix 1` (`--teacher-b "python3 ../python/teacher.py --tier hard"`), which already lets the teacher play a seat | S |
| F2 / F6 and stops | Client-side auto-pass. Recognise "pass priority" by the menu row's category, not its text. F2 stops when the stack gains an object or the turn ends; F6 takes row 0 at every decision until the turn ends; per-step stops pass priority automatically in unstopped steps when the stack is empty | M |
| Concede | `-2` / `-3` with a confirmation | S |
| Hidden information | Play renders only the human seat's frames, never omniscient ones | S |
| Model's read while playing | Featurize the human seat's frame in the workbench (`featurize_seat`) and evaluate it through `rmserve` with the torch-free remote evaluator (`src/actor/remote_evaluator`, `src/inference/protocol`); asynchronous, with a "thinking" indicator; can be turned off for an unaided game | M |

### 2.4 Card images and printings

**Status: done** (`tools/scryfall/fetch_images.py`; `tools/rmwork/`: `card_images`, `image_textures`,
`image_renderer`, `printings_tab`). The fetch tool (stdlib, its README) lists, for every vocab index and
every token script the decks make (`bin/resources/tokenscripts/`), the paper printings Scryfall has
(`/cards/named` then `oracleid:… game:paper`, `unique=prints`; a token by `t:token !"<name>"` with its
script's colours, types, P/T and keywords, relaxed until one matches), chooses the default (the oldest
by date, set and collector number that is paper, not digital, not a promo, not oversized, not from a
memorabilia / funny / minigame / alchemy / vanguard / planechase / archenemy / treasure-chest set, with
an image; else the oldest paper one with an image), downloads its `normal` images (each face of a
double-faced card) into `~/.cache/robomage/scryfall/images/<set>/<number>[-<face>].jpg` and writes
`manifest.json` (per card its key, vocab index, names, every paper printing with its faces' URLs and
files, the default; the cards Scryfall lacks); API requests 500 ms apart (the search and named-card
endpoints allow two a second), images 100 ms, retries honouring `Retry-After`, resumable, `--cards`,
`--printing NAME=SET:NUMBER`, `--dry-run`, `--selftest`. Run over the vocab it found every card and token (441 of 441 vocab
cards, 29 of 29 token scripts, 468 images, 56 MB, about 15 minutes). The workbench reads only the
cache: `CardImages` (the manifest, the overrides in `~/.config/robomage/printings.json`, which image a
face shows: the manifest card of its card id (a token's in the token band, an unregistered token by
name), the override's printing when the manifest lists it, else the default; of a double-faced
printing the face of the same name, else the back for a back face) and `ImageTextures` (a worker
thread decodes with stb_image and halves to the level a card is drawn at, 1/1 to 1/8; at most twelve a
frame are handed to ImGui 1.92 as user textures; at most 400 textures and 256 MB, the least recently
drawn evicted past either, never one drawn this frame or the last; another level of the file stands in
while one decodes; nothing blocks a frame). `ImageCardRenderer` draws the image with rounded corners,
turned a quarter when tapped and faded when phased out, and falls back to the text renderer for a
back (a face-down card keeps the drawn back with what the viewer knows of it; hidden cards are never
named), a card or token without a cached image, and an image still decoding. The Table (its cards and
the stack's thumbnails), the magnifier and the zone windows draw with it; a double-faced card shows the
face it is on, and the magnifier shows both faces side by side (one image when a card is printed whole
on one face, a split card). Edit → Settings → Card Printings: Card faces images / text (default images;
no manifest draws text), the cache directory and its status (vocab cards and tokens with images, what
Scryfall lacks, the command that fetches the rest, Reload), and the printings: a card list searched by
name, the chosen card's paper printings oldest first (thumbnail when cached, set name and code, number,
year, set type, default / promo / not downloaded); a row overrides the default (the default's row
resets), "Default (oldest)" and Reset All; a shown printing not downloaded shows the
`fetch_images.py --printing` command and a Fetch button that runs it (`/usr/bin/env python3`, its own
network access) and reloads the cache when it exits. Screenshot flags `--card-faces images|text` and
`--printings-card NAME`. Checked by `rmwork --selftest` (a synthetic cache of tiny PNGs: the manifest
and its refusals, the overrides file, the printing and face each card shows, the status, the LRU and
levels, the decode thread's hand-over, a file that does not decode, eviction, the renderer's image and
fallbacks), `rmwork --selftest-data` (the Table over sampled corpus decisions in every view with images
for half the cards: each card an image or text as the cache says, every image decoding) and
`fetch_images.py --selftest` (the choice rule on canned Scryfall JSON), all in the `workbench` tier. Not
done: "flip on hover" on the Table (the magnifier shows both faces instead); token images are Scryfall's
oldest matching token, which may differ in art from the set the card came from.

| Feature | Notes | Size |
|---|---|---|
| Image cache | `tools/scryfall/fetch_images.py` (stdlib, like `tools/forge_fetch`): for every vocab card and token, list paper printings (`unique=prints`, oldest first), choose the oldest non-promo printing, download `normal` images into `~/.cache/robomage/scryfall/`, rate-limited, with a manifest. The C++ app reads only the cache, so it never touches the network | M |
| Printing overrides | `~/.config/robomage/printings.json`: card name → Scryfall set + collector number; the Settings tab lists all printings (the mock-up offers eight); a missing image is fetched on demand by the script | S |
| Special faces | Double-faced cards (combined scripts, `CARDS` face rows): show the face up; flip on hover. Face-down objects (morph, manifest, cloak): the card back, with the identity shown only when the viewer knows it. Tokens: Scryfall token printings, with a text-rendered fallback | M |
| Offline fallback | When an image is missing, render a text card from `CARDS` (name, cost, type line, oracle text, P/T) | S |
| Textures | stb_image decode, GPU textures with an LRU cache; images streamed in off the UI thread | S |

### 2.5 The Table

**Status: done** (`tools/rmwork/`: `table_model`, `table_layout`, `table_view`, `card_book`,
`card_renderer`, `table_backdrop`, `links`). The Table shows Replay's current decision as the viewer saw
it, built only from that viewer's STATE frame and its own CARDS frame (the omniscient viewer sits as
seat A): the opponent's lands at the far edge and other permanents toward the middle, the middle strip
(turn, active player, step, deciding seat, day or night, the twelve-step bar, narrowed to the strip when that
is narrower; in Play a stop dot under each step, filled where Settings → Stops stops for the side whose turn
it is, and a click on a step (its label or dot) toggles that stop, saved with the settings like Settings →
Stops' own, a tooltip saying which; Replay's bar has no dots and takes no clicks;
the stack, top first, only when it is not empty: each object's card thumbnail, controller chip, kind and
type line, X, modes and targets), the viewer's other permanents, lands and hand on a shaded band, the
player chips (name, deck, life, hand and library counts, poison, energy, other counters, monarch, the
city's blessing, speed) with their player-effect and emblem chips beside them; the opponent's hand is its
chip's count. A land is an object whose post-layer types hold Land and not Creature; Auras and
Equipment peek above their host (an attachment's attachments join the root host's pile); rows are in
entry (timestamp) order and centred. The fit is the mock-up's: one card width for the four rows and the
hand (1.15×) so they fill the height and the widest row fits the width, every frame, so it follows a
resize; a row keeps an upright card's height when empty. A wide board (`table_layout`, ImGui-free):
every permanent is shown on its own while that fit gives cards at least 69 px wide (`LEGIBLE_CARD_W`:
the text renderer prints the type line from 6.5 px text at 0.095 of a card's width, so below 69 px a
card shows its name alone and an image's text is unreadable). Below it, while a row's width (not the
panel's height, nor the hand) is what holds the cards down, the widest row gets room: its largest group
of interchangeable permanents is stacked into one pile with an offset and a `×n` count (the model's
groups: the same card and face, or the same token, and controller; tapped or not alike; the same P/T,
damage, loyalty, counters, designations, colours, subtypes and summoning sickness; nothing attached and
not attached; identified, face up, not phased out, attacking or blocking: lands, tokens, creatures and
other permanents alike), then its next largest, and once it has none left it wraps onto another line
(up to three, its piles in order, split evenly) when that widens its cards without narrowing another
row's; it stops once the cards are 69 px wide, and what is left shrinks. A stack's front card stands for
it: hovering it says the count, it glows when a menu row acts with any of its cards and is outlined when
any of them is linked, and the magnifier on any of its cards adds "One of n identical". Cards are drawn by a `CardRenderer`: the card's image (§2.4), else
the text renderer, which prints the CARDS face in a rounded card in its frame colour (one colour, two
split, gold, colourless grey, a land tan with a land glyph and pips for its basic land types; a token
silver-bordered and "Token" in its type line): the name (wrapped on a small card), the cost as pips, the
type line once legible, and the oracle text from 120 px. Face-down objects are card backs (the identity
printed on the back only where the viewer's frame gives it); hidden cards are never named. Over the
cards: tapped cards turned a quarter, the post-layer P/T (green above the printed, red below, blue
mixed), marked damage, loyalty, counters by kind, designations (monstrous, exerted, prepared, a chosen
name, type, colour …), summoning sickness, phased-out permanents faded. Attackers stand forward of their
row; an arrow runs from each blocker to the attacker it blocks. Linking: `LinkState` holds the hover and
selection every tool shares (objects by handle, players, menu rows by their `source` / `target`
handles); the Table reports the card, stack object or player chip under the mouse and outlines whatever
it links to, hovering a stack object lights itself and its targets (`stack_target`; both ends highlighted, no
line drawn between them), a click selects;
when the viewer is the deciding seat the cards some menu row acts with glow. The magnifier: Q held over a
card (or a stack object), or a right-click, shows it large beside it with its state listed under it
(release Q; a right-click one closes on leaving the card, a click or Esc); a double-faced card (its
`CardBook` has both faces) shows both side by side, the face up first, outlined and captioned "Face up",
the other captioned "Back face" / "Front face", on the Table, the stack and the zone windows; a card the
viewer cannot identify stays a back. It is placed to the card's right, else its left, else centred, and
shrunk to stay on screen. The background is Settings →
Table's colour, one of the five textures generated as GPU tiles on first use, or an image (stb_image,
drawn covering the panel; one that cannot be read shows the wood and says so in the status bar). The
twelve-step bar's "Opening hand" phase now counts a bottom-of-library pick only while a player owes
London-mulligan cards (a scry-like effect's bottom pick was taken for the opening hands). `--table-hover-stack N`,
`--table-hover-step N` (Play) and `--table-magnify NAME` force a hover, a step's tooltip and the magnifier
for screenshots. Checked by `rmwork --selftest` (the layout and fit of synthetic boards: identical lands and
tokens stacked one pile a row on a narrow panel and not on a wide one, different creatures wrapped onto two
lines in order, tokens stacked before a row wraps, nothing wrapped on a panel too short for it; the step
bar's cells; a step's stop toggled and kept in the settings file; the magnifier's two faces side by side on
screen and shrunk on a small one; the links, the tiles, the new colour tokens' contrast) and `rmwork
--selftest-data` (every decision of five corpus logs and the played bo3 in every view against its frame at
four panel sizes, a narrow one among them: a stack's members interchangeable and identified, stacks and
wrapped rows only where the full-detail cards would be narrower than 69 px, a row wrapped only with no group
left to stack; drawn headless, and the hover and magnifier interactions, a click on Replay's step bar
toggling nothing; in Play, a click on a step's dot toggling its stop, settings.json keeping it, a second
click turning it back; `docs/ci.md`). Decisions on the Table (§4.2, items 6 and 11; `table_decision`): at a targeting decision (every row
`SELECT_TARGET` or `DIVIDE_DAMAGE`) or a payment one (a `PAYING_COSTS` row: `--no-auto-mana`'s mana payment,
a Phyrexian pip's life or mana) the objects and players a row names glow and the rest of the battlefield and
hand is dimmed, the spell or ability chosen or paid for is lit (its stack object, else a "Source" tag on its
permanent), targets chosen earlier in the same targeting are outlined in the accent colour with their number
(read from the earlier decisions: the frame has no target of a spell still being cast), a point of divided
damage given so far is counted on its target, and a prompt line at the foot of the middle strip says whose
decision it is ("YOUR DECISION" in Play, "A DECIDES" in Replay), what is decided (the CLI's pending-decision
line), the cost still to pay as pips with the life, energy and mana paid after it, the targets chosen so far,
and the rows naming nothing (Done selecting targets, Cancel casting, Pay 2 life (instead of {B}), Pay {B}) as
buttons on its right (on lines of their own when they would leave the prompt less than half the line). In Play
at the human's pending decision a click on a lit card, stack object or player chip answers with its row, and an
object several rows name (a dual land's two colours) opens a popup of them; at a priority decision offering
mana abilities a click on a land or source taps it (its `MANA_*` rows only); hovering a pick says what a click
does and shades its Choices row. In Replay, an earlier decision or while the opponent thinks the highlights show
and a click only selects. The player chips (and the Players tool) show floating mana as `pool {U}{R}`. Combat on
the Table (§4.2, item 7; `table_decision`'s combat kinds, `table_combat`): every attacker is labelled with what it
attacks ("→ Player B", "→ Kaito") during the combat decisions and whenever it attacks a planeswalker; at declare
attackers the creatures that can attack glow, at declare blockers those that can block, the attackers kept lit.
The engine declares a creature as soon as its row is answered and has no row taking it back, so in Play the human
plans the declaration and nothing is sent until the confirm button: a click plans an attacker against the defending
player (it stands forward, outlined in the accent colour); with a planeswalker to attack it stays armed, the
planeswalkers glowing and the player's chip outlined, and a click on a planeswalker (or the chip) changes what it
attacks; a click on a planned attacker takes it back. A click on a creature that can block arms it (a line follows
the mouse) and a click on an attacker, or a drag from the blocker onto it, plans the block, an accent arrow blocker →
attacker beside the declared blocks' red ones; a click on a planned blocker takes it back. Attackers a blocker cannot
block by the frame's evasion (flying without flying or reach, can't be blocked) are not lit and refuse the click,
saying why. The prompt line lists the declaration so far, and its buttons are "Attack with N" / "No attack" ("Block
with N" / "No blocks"), "Attack with all" and "Clear"; the confirm sends the plan one reply per decision (each
creature's row, its target's or attacker's row, the confirm row last, the engine taking a menu left with one row
itself), and stops where the engine's rows lack what was planned (a blocker that cannot block that attacker), the
rest of the plan kept, for the human to choose there. A creature declared in Choices stays declared. At
ATTACK_TARGET and BLOCK_TARGET (after a row answered in Choices, or a stopped plan) the player chip, the planeswalkers
or the attackers are picks, the attacker under the mouse joined to the blocker by an arrow. Damage assignment order:
the current rules have none (CR 510.1c: a creature blocked by two or more creatures divides its combat damage among
them as its controller chooses), and neither has the engine: when an attacker cannot kill all its blockers it asks
`ASSIGN_DAMAGE`, lethal damage to one blocker at a time, then Done. The Table lights the blockers that can still be
given lethal damage with "lethal N" (the row's), counts what each has been assigned ("N assigned", the `combat`
table's `assigned`), tags the attacker "Assigning", says what is left ("2 of 2 left to assign"), and a click gives a
blocker its lethal damage; Done is a button. For screenshots `--play-until attack|block|damage` stops at those
decisions and `--table-plan SPEC` plans a declaration. The opening hand and the card picker (§4.2, items 5 and 8;
`mulligan_plan`, `card_picker`, `card_picker_view`) are described there. Waiting for later stages: the rest of direct
manipulation in Play (§4.2), the Choices tool's side of the linking (§2.6).

| Feature | Notes | Size |
|---|---|---|
| Layout from frames | Battlefield rows split into lands and other permanents; attachments (Auras, Equipment) drawn overlapping their host; identical permanents stacked, and rows wrapped, only when the board is too wide to show each legibly; tokens; phased-out permanents dimmed (the frame flags them); emblems and player effects in a strip by the player | M |
| Object state | Tapped, counters by kind, effective P/T (post-layer, from `object`), damage marked, summoning sickness, loyalty, designations (monarch, day/night, speed …; `designations.def`), energy and poison on the player | M |
| Combat | Attackers set forward; arrows from blockers to attackers; damage assignment among blockers (no damage assignment order in the current rules, CR 510.1c) | M |
| Targets | Hovering a stack object highlights it and its targets (`stack_target` rows, not text matching as in the mock-up); no arrows: both ends are highlighted | S |
| Card-to-choice linking | Use each menu row's `source` / `target` handles (`menu` table) to link rows and objects, replacing the mock-up's name matching | S |
| Auto-fit | Size cards to fill the panel, as mocked | S |

### 2.6 Other tools in the mock-up

**Status: done** (`tools/rmwork/`: `replay_tools` and the views `players_view`, `choices_view`,
`inspector_view`, `state_tree_view`, `value_views`; their data `tool_data`, `decision_frame`,
`zone_model`, `state_tree`, `evaluations`; `magnifier`, `tool_widgets`). Every tool reads the cursor's
decision in the current viewer's STATE frame and that viewer's CARDS frame only. What a viewer sees of a
decision: the omniscient viewer and the deciding seat its menu and the decider's recorded evaluations (μ,
values, π, teacher); the other seat only whose decision it is. Of the evaluations over the match a seat
view sees its own seat's (series, swings, reads), the omniscient view both. Players: a box per player,
the viewer's opponent first (seat chip, the agent the episodes name, deck, life large, "deciding",
poison, energy, counters, designations, effects and emblems, the zone icons with counts; hovering a box
links the player). Zone windows (every zone icon opens one; floating, several at once, following the
cursor and viewer; Esc closes a right-click magnifier, else the focused window): graveyard and exile
newest first with an exiled card's linked source, a seat's library only its knowledge's exact positions
and the cards its frame shows (the omniscient viewer's in full), the opponent's hand only the cards
revealed or known (`k_known_hand`) and the rest as a counted card back; the cards drawn by the
`CardRenderer`, hover links, the magnifier (shared with the Table, `magnifier`). Choices: the menu rows,
► on the log's reply, π bars of the shown policy (the Inspector's choice, else the first) with π and μ,
↑ / ↓ moving a highlight, a click selecting the row and Enter / a double-click submitting it
(`take_submitted`, for §2.3), hovering a row lighting its source and target on the Table, the State Tree
and the zone windows, a row shaded when what it acts with is hovered there. Decision Inspector: an ImHex
Name / Value grid (decision, game, turn and active player, step, deciding seat and agent, the pending
query, the pending cost's remaining pips, life, energy and discards, stack depth, menu size; from the
episode V per policy on the deciding seat's or Player A's scale (a radio), H(π), log π(chosen), μ's
support, policy version, teacher label against the chosen row), the shown policy's radio, and the menu
with π per policy as bars, μ, ► chosen. State Tree: generic over the SCHEMA (`state_tree`): "game",
"player A/B" (life, counters, the zones with their objects, the player row), "stack", then "frame tables"
with every table and row, every column by its schema name, storage and semantic tag, its value by the
tag (enums and masks by their string tables, cards by name, a handle named only where the viewer's
object table identifies it); Col swatches in the card's colours (lands tan, colourless grey); nodes
expanded lazily and kept expanded across decisions by their keys; hover linked both ways with the Table.
Value Graph (ImPlot): V per seat and policy on Player A's scale, turn bands, game boundaries,
sideboarding shaded, each game's z with a line from the last value, a life strip, the cursor, click / drag
to scrub, a hover tooltip, a wrapping legend of toggles; without evaluations the life strip and a note.
Swings: the largest |ΔV| between a seat's consecutive decisions of a game and to its result, with the turn
and the viewer's own log lines between; in Omni the seats' disagreement (adjacent A and B decisions:
A's V against −V of B); a click goes to the decision; the toolbar's next swing walks the same rows.
Model's Read: the deciding seat's (a seat view: its own) V, the shown policy's preferred row with its π,
the model, a sparkline of the seat's values so far. The status bar's V is the deciding seat's when the
viewer sees it. The links (`LinkState`) take the menu and stack targets from the tools' data every frame,
so Choices, the Table, the State Tree and the zone windows link the same handles whichever are open; a
hovered stack object lights itself and its targets (cards, player chips and boxes), with no line drawn
(the Table's target arrows are gone; the combat lines stay). Screenshot and check flags: `--tool NAME`,
`--choices-hover N`, `--tree-expand`, `--open-zone SEAT:ZONE`, `--zone-magnify NAME`. Checked by `rmwork
--selftest` (the evaluations' series, swings, disagreements, turn bands and text on a synthetic match)
and `rmwork --selftest-data` (every decision of the corpus logs and the played bo3 in every view: the
Players' counts, the zone windows' cards exactly the frame's and its knowledge's, Choices the menu or
nothing in a non-deciding seat's view, the chosen row the reply, the Inspector's fields, the State Tree's
tables, rows and handle names, the links between Choices, the Table and the tree; a sample drawn with
every tool open, a Choices hover lighting the Table, zone windows opened, magnified in and closed by Esc;
the bo3 match with evaluations added to its episodes: the Value Graph's series, Swings and disagreements
recomputed, next swing, the Inspector, Choices' π, the Model's Read, the status bar). Not done:
re-evaluation (§2.7), so a second policy's series and the seats' disagreement come only from episodes that
recorded them (a self-play match evaluates both seats).

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
| Recorded evaluations | Read `eval_value`, `eval_log_pi`, `eval_policy`, `eval_version` from episodes. **Done** with §2.6 (`evaluations`: per seat and policy, shown by the Value Graph, Swings, the Inspector, Choices and the Model's Read) | S |
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

1. ~~**Start page**~~: built (`start_data`, `start_page`; mock-up `start.html#start`). A page over the whole
   work area in place of the mode's dockspace rather than a docked tool: with nothing open no tool has
   anything to show, and neither mode's layout (tabs, splits, the saved layouts) is touched while it is
   up, since the dockspace is simply not drawn, as for the other mode. It shows at start (setting
   `start_page`, its checkbox on the page; `--start-page`), from File → Start Page (Alt+Home) and after
   File → Close (Ctrl+W: the replay, or a Play game that is over; a running game is conceded first);
   opening a replay or run log, starting a game, the Start tab's ✕, the toolbar's mode switch and View's
   tools and layouts give the dockspace back. Left: New Game, Open Replay, Open Trajectory Episode, Open
   Run (Ctrl+Shift+O) with their keys, the keys to know, the setting. Middle (double-click opens): Play's
   recent games (the games directory's newest 40: when, your seat and deck, the opponent's agent from the
   game's sidecar and deck, format, result, decisions; "playing", "no result (stopped)", a malformed log
   named), the recent replays (settings' recent decision logs outside the games directory: decks, result,
   decisions, modified) and the recent runs (the recent run directories, then the runs directory's by
   last update: phase and running, updated, hours, games from `status.json`); each opens in Replay or the
   run browser. Results are never replayed for the page: a Play game's from its stderr's `GAME_RESULT` /
   `MATCH_RESULT` lines, else the replay cache's narrative of the log under the engine replays run with,
   else "…" until it is opened; the scan runs on the page's thread when shown and every 5 s while it is.
   Right: the engine (binary, built, working directory and `resources/`, the replays' plain engine or the
   `robomage_nn` beside it, older than the engine said, `make actor`), card images (cards and tokens
   cached, the fetch command), the inference server (socket configured and present, policies listed,
   `rmserve` beside the engine, Model's Read's last state, a refused or lost connection in red), files
   (the games directory, the replay cache's size), each linking its Settings tab. The model's
   compatibility check stays with the Model manager (item 4). Checked by `rmwork --selftest`
   (`selftest_start`: the result lines, the sidecar, a synthetic games directory with finished, stopped,
   malformed, running and unread games and more than the page lists, recent replays with and without a
   cached result, runs with and without `status.json`, the status blocks with and without engine,
   `robomage_nn`, working directory, images and socket; the App: the page in place of the dockspace, ✕,
   the key, Close with the setting on and off, a Settings link) and `--selftest-data` (Play's bo1 and bo3
   listed with their seats, opponents and results, the bo1 opened in Replay from the page, Close bringing
   it back; the game whose engine was killed listed as stopped and opened in Replay).
2. **New Game dialog**: your deck and seat, opponent (model / checkpoint, scripted tier, explore), best-of-one or best-of-three, seed, auto-mana, Model's Read on or off, knowledge level.
3. **Open Run / Episode Browser**: the run's episodes as a filterable table (§3).
4. **Model manager**: exports and checkpoints, rmserve start/attach, compatibility check.

### 4.2 Decisions in Play (direct manipulation on the Table)

5. ~~**Mulligan**~~: built (mock-ups `decisions.html#mulligan`, `#mulligan-bottom`; `table_decision`'s MULLIGAN and
   BOTTOM, `mulligan_plan`). At the opening hands the hand is drawn large on the Table (1.9 battlefield card widths,
   `PREGAME_HAND_SCALE`, in place of 1.15) and never dimmed; the prompt line says what is decided from the frame's
   `player.mulligans` / `to_bottom` ("Opening hand: keep these 7, or mulligan (draw 7, then put 1 on the bottom)",
   "Mulligan 2: keep these 7 and put 2 on the bottom, or mulligan to 4"; play or draw and the opponent's mulligans
   after it, dimmed) with Keep and Mulligan as buttons (rows 0 and 1). The London bottom picks (CR 103.5; one
   BOTTOM_DECK_CARD decision per card, no pending flow, a row per hand card, no row taking a card back) are planned
   on the hand rather than sent at each click (the mock-up's open question 1): a click on a lit hand card plans it
   ("Bottom 1", "Bottom 2" in the accent colour), a click on a planned card takes it back, a click past the cards owed
   is refused with a status message, Undo and Clear on the prompt line; once the plan holds every card owed, "Put N on
   the bottom" sends it one reply per decision (`next_bottom_row`: the row naming the first planned card still in the
   hand). The prompt says "Keeping 5: put 2 cards on the bottom of your library · planned: …" ("(1 of 2 done)" once
   one is). The opponent's mulligans are on the prompt line, not in a window (open question 2). Replay: the same
   highlights, the card chosen outlined in the accent colour with "Chosen" (as every pick the Table shows; a chosen
   button is highlighted), no plan. `--play-until mulligan|bottom` stop there and `--table-plan "NAME,NAME"` plans the
   bottom picks for screenshots. Checked by `rmwork --selftest-data` (`selftest_decisions`).
6. ~~**Targeting**~~: built (§2.5's decisions on the Table; mock-ups `decisions.html#targeting`,
   `#targeting-second`). Legal targets lit on the Table and the player chips, everything else dimmed; the spell
   (or the ability's source) lit, linked to its targets by highlighting both, no arrow (user, 2026-11-18: arrows
   only for blocks); a click on a target answers, a popup when rows share an object; the targets chosen so far
   numbered in the accent colour; Done selecting targets (and the other rows naming nothing) a button on the
   prompt line. Answered on the Table: `SELECT_TARGET` (a spell's, an ability's, a trigger's as it is put on the
   stack) and `DIVIDE_DAMAGE` (one more point to a target, the points so far counted on it); `ATTACK_TARGET` and
   `BLOCK_TARGET` belong to combat (item 7). A target the Table does not show (a card in a graveyard) is chosen
   in Choices. Checked by `rmwork --selftest-data` (`selftest_decisions`).
7. ~~**Declare attackers / blockers**~~: built (§2.5's combat on the Table; mock-ups `decisions.html#combat`,
   `#combat-attackers`, `#combat-blockers`, `#combat-damage`). The engine declares a creature as its row is
   answered and cannot take it back, so the Table plans the declaration and sends it on confirm: a click plans or
   takes back an attacker (with a planeswalker to attack, a second click on it or on the player's chip says what it
   attacks), a blocker then an attacker (or a drag) plans a block, drawn as an arrow blocker → attacker (user,
   2026-11-18: arrows for blocks only); "Attack with N" / "Block with N", "Attack with all" and "Clear" on the prompt
   line; the plan sent one reply per decision, stopping where the engine refuses a planned block. Combat damage
   assignment among blockers (`ASSIGN_DAMAGE`): the blockers lit with their lethal damage, the damage assigned so
   far on each, a click gives one its lethal damage, Done a button. Checked by `rmwork --selftest-data`
   (`selftest_decisions`).
8. ~~**Library search, dig, scry, surveil, reveal**~~: built (mock-ups `decisions.html#card-picker`, `-ponder`,
   `-brainstorm`, `-surveil`; `card_picker`, `card_picker_view`). A card-picker decision is one whose every row is
   SEARCH_LIBRARY (a row per legal find, the card its `source`; "Fail to find" names none), TOP_LIBRARY (Ponder's,
   Doomsday's ordering and scry's keep with an `ordinal`, the depth the pick fills; Doomsday's five picks
   "Thassa's Oracle (library)" / "Daze (graveyard)" and Brainstorm's put-back from the hand with none, each on top of
   the last), BOTTOM_DECK_CARD at a resolution (scry's other pile), DIG_CHOICE ("Take nothing (rest go to the bottom
   of library)" names none), CHOOSE_CARD (surveil's "Put X into graveyard", a tutor's, Delve's, Mox Diamond's
   discard), EXILE_FROM_YARD (Escape) or SYLVAN_CHOICE; interchangeable library and graveyard cards are one row
   (`interchangeable_cards`), hand cards never. The picker is a window inside the Table rather than a docked tool
   (open question 1): the stack object resolving, the prompt line and the board stay where they are and nothing has to
   be brought to the front and given back. It sits over the opponent's half (below the opponent's chip, down to the
   middle strip) as wide as its cards need, centred; when its cards need more room it takes the Table's width down to
   the prompt line (over the strip's step bar and stack; the title names what resolves), and a whole-zone pick too large
   for that (Doomsday with a 45-card library) takes the Table down to its foot, the prompt then at the picker's own
   foot; its title bar's button collapses it. The cards are the rows' (only what the viewer's frame shows: its menu,
   its objects, `zone_model`), grouped by zone (the viewer's library top first with each card's place, then its
   graveyard, exile, the other seat's zones), identical cards one entry with ×n; a search shows the rest of the library
   its reveal window shows, dimmed (open question 3: both, an "Unpickable" toggle hides them). A filter box (name or
   type line, case-insensitive) and a Cards / List switch above them, the list a line per entry with its count, type
   line and place. A click on a card answers its row; a card with several (scry's Top / Bottom, surveil's Top /
   Graveyard) has a button per row under it; the rows naming no card (Fail to find, Take nothing) are buttons under
   the cards. Ponder and Doomsday are answered a click per pick, as the engine asks (open question 2: the pile shows
   where each goes before it is clicked; the engine has no row taking a pick back). Beside the cards the pile being
   built, top first, as the library will be: the picks this resolution made so far (the replies of the deciding seat's
   decisions just before for the same pending source, read from their frames, since the frame no longer lists a card
   put back), each in its final place (an ordinal pick at its depth, "Top", "2nd", "3rd", the deepest first for
   Ponder; an on-top pick above the earlier ones, so Doomsday's first ends fifth), the place the next pick fills
   marked, then what went to the bottom or the graveyard; the pass restarts where the menus change kind (Doomsday's
   picks, then its ordering of the five). A decision whose cards are all on the Table (Brainstorm's put-back) is picked
   on the hand, the picker showing the pile alone at the right. Replay (and an earlier decision of Play): read-only, the
   card chosen marked and scrolled into view; a seat's view of the other seat's pick has none. Screenshot flags
   `--play-until pick`, `--picker-filter TEXT`, `--picker-list`. Not in the frame (the engine was not changed): how many
   picks a resolution will ask (Doomsday's five, Brainstorm's two), so the pile shows the picks so far and the next,
   not empty places for the rest; and the card a pick put back (the pile is read from the earlier decisions' replies).
   Checked by `rmwork --selftest-data` (`selftest_decisions`: a fetchland's search, Ponder, Preordain's scry,
   Doomsday's five picks with the filter, Replay).
9. **Modal and numeric choices**: choose modes (one or more, `mode:done`), choose X, divide damage among targets, order triggers.
10. **Name a card**: the limited candidate list (or every vocab card under `--closed-decklist`), with search.
11. ~~**Mana payment**~~: built (§2.5's decisions on the Table; mock-ups `decisions.html#mana-payment`,
    `#mana-payment-phyrexian`). Under `--no-auto-mana` (New Game's auto-mana, `auto-mana=off`): the remaining
    cost from `pending_cost` / `pending_cost_pip` as pips on the prompt line (life, energy and the mana paid
    after them), the sources that can pay lit, a click taps one (its row; a source with several, a dual land or a
    Mox, opens the popup of its colours), Cancel casting a button; a Phyrexian pip's "Pay 2 life (instead of
    {B})" and "Pay {B}" buttons beside it; at a priority decision a click on a land taps it for mana, the floating
    pool shown on the player chip and the Players tool. Checked by `rmwork --selftest-data`
    (`selftest_decisions`).
12. **Optional costs and alternative costs**: e.g. Force of Will's pitch, kicker, convoke and crew pickers.
13. **Discard / sacrifice / return pickers** for mandatory choices.

### 4.3 Between and after games

14. ~~**Sideboarding**~~ (best-of-three): built (§2.3, the Sideboard tool): main deck and sideboard as cards with counts, a plan of swaps sent one decision at a time, the league guide's and the model's plans, Done.
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
29. ~~**Card detail**~~: dropped (user, 2026-11-18). The magnifier is the enlargement: the card's image (its printing's set symbol and year are printed on it), and both faces side by side for a double-faced card (§2.5).

## 5. Suggested order

1. Shell, theme, docking and presets (§2.1).
2. Replay of a recorded `.rmlog` with all three viewers: Table, Players, Choices, Inspector, State Tree, Log, Value Graph from recorded evaluations (§2.2, §2.5, §2.6). Mock up §4.3's game-over screen first, since Replay needs it.
3. Card images and printings (§2.4).
4. Play against a model with F2 / F6 and stops, through the menu list first (§2.3); then the direct-manipulation decisions of §4.2, one category at a time.
5. Re-evaluation and the model manager (§2.7), then Policy Compare, Knowledge and Teacher Disagreement.
6. The run tools: Episode Browser, Bench Matrix, Calibration, Training Curves.
7. Branch Explorer, Influence Overlay, Blunder Finder, then the rest of §3.
