# Continuous integration & the test standard

The standard for changes to this repo: **`make check` must pass.** CI runs exactly it; run it
locally before pushing.

```bash
make check    # `make` (debug build), then bin/debug/robomage_check (every tier)
```

It needs the Forge card scripts provisioned first (the SessionStart hook does this in cloud
sessions only; on a fresh local clone run it yourself; stdlib only):

```bash
python3 tools/forge_fetch/provision_decks.py
```

It fetches every card in `src/card_vocab.h`, every card of the top-level, `league/`, `heldout/` and
`candidates/` decks, and the tokens they create, at the Forge commit pinned in
`tools/forge_fetch/FORGE_PIN`.

## The gate: `robomage_check`

`tools/robomage_check/` builds `bin/<config>/robomage_check` (part of `make`). It runs the engine
as subprocesses in parallel, with `bin/` as their working directory, and links no engine
translation unit (only the std-only decoder, the agents' frame reader and `src/featurize/`). It is
built `-fno-exceptions`; a scenario regex is compiled in a forked child first, so a malformed one
is reported, not an abort.

Every requested tier runs even if an earlier one fails. The run ends with each tier's PASS/FAIL, a
wall-time table, every warning and error, and a `Reproduce:` line (tiers, seed, mode, matchups).
Exit 1 on any **error** (warnings alone pass); exit 2 on bad usage, an unknown tier, or a missing
engine binary or card scripts.

```
robomage_check [--tier a,b,...] [-j N] [--seed N] [--binary PATH] [--out-dir DIR]
               [--fuzz-games N] [--manual-mana-games N] [--smoke-games N] [--matchups PRESET]
               [--mode MODE] [--scenario GLOB ...] [--show output|narrative|dump] [--record]
```

| Flag | Meaning |
|---|---|
| `--tier a,b` | run these tiers, in the order listed (default: all, in the table's order) |
| `-j N` / `--jobs N` | parallel engine processes (default: the cores) |
| `--seed N` | base seed of the `fuzz` and `smoke` games (default 1) |
| `--binary PATH` | the engine (default `robomage` beside the gate, so `bin/release/robomage_check` tests the release build); `rmknow`, `rmfeat` and `robomage_nn` are taken from the same directory |
| `--out-dir DIR` | transcripts, engine stderr and draw logs (default `ci_out/`) |
| `--fuzz-games N` / `--smoke-games N` | games per matchup (default 8 / 2) |
| `--manual-mana-games N` | `fuzz` tier: `--no-auto-mana` games per ring matchup (default 2; 0 skips them) |
| `--matchups PRESET` | over the playable decks (`league/` then `heldout/`): `mirrors`, `ring` (each deck vs the next), `mirrors+ring`, `all` (every ordered pair); or `"a:b,c:d"` (deck specs relative to `decks/`; `a:*` is `a` against every playable deck); default `fuzz` `mirrors`, `smoke` `mirrors+ring` |
| `--mode MODE` | the fuzz agent (bo1 and `--no-auto-mana` games): `explore` (default) or `explore:patient` |
| `--scenario GLOB` | `scenarios` tier: only the scenarios whose name matches (repeatable) |
| `--show output\|narrative\|dump` | `scenarios` tier: print each run's engine output, the narrative lines the assertions see, or its knowledge lines |
| `--record` | `replay` / `cli` / `vocab` tier: rewrite the corpus baselines / CLI goldens / parse-report baseline instead of checking them |

### The tiers

Times are wall clock for the debug build on 4 cores (`make check` total about 6.3 minutes).

| Tier | Checks | Time |
|---|---|---|
| `vocab` | `--check-decks`: every top-level, `league/` and `heldout/` deck card loads and is in the vocab; the WordPiece tokenizer against `tests/embed/tokens_golden.jsonl` (skipped with a notice without the pinned model's `vocab.txt` in `.cache/embed/`); the layout registry `src/featurize/registry.json` canonical, compiled in, needing no `make registry`, and append-only against HEAD unless `layout_version` was raised (`docs/featurizer.md` §6.1); the card digest `src/featurize/card_digest.json` current (§6.2); `--parse-report @vocab` against `tests/parse_report/vocab.json` (a new or vanished finding is an error; `--record` rewrites it); `tools/onboard/onboard.py check` over `decks/candidates/` (a tool failure is an error, a stale status file a warning, and a vocab card from id 367 up that no scenario names in a zone key or `deck_a/b` deck an error) | ~1 s |
| `concede` | the `-2` / `-3` sentinels (CR 104.3a) over RMSTATE sessions the gate answers itself: bo1, bo3 into sideboarding and game 2, between games, at parked queries; same seed and replies give the same frames; results, narrative and `--replay` of every decision log | ~2 s |
| `scenarios` | every `tests/scenarios/*.json` on the engine (presets as flags, `--play`, `--narrative --trace-decisions`), every assertion key of `tools/robomage_check/scenarios.h` | ~4 s |
| `cli` | each `tests/cli/<name>.json` (an engine command line and stdin lines): exit 0, empty stderr, stdout (the `Decision log:` path reduced to `<name>.rmlog`) byte-identical to `<name>.out` (a drift's output goes to `ci_out/cli/<name>.out`); `--replay` of the session's log clean | <1 s |
| `replay` | each `tests/corpus/<name>.rmlog` replayed, stdout byte-identical to `<name>.replay` (below) | ~17 s |
| `state` | the corpus replayed under `--selfcheck-knowledge` (same output, clean summaries); explore fuzz games with self-checks and dumps; every scenario at `--selfcheck-state-sample 1`; `--rmstate-read` on the dumps; decoder refusals of mutated frames; stream sessions against their dumps; name-a-card menus with open and closed decklists; the `CARDS` frame's ability parameters; include isolation of `src/knowledge/` and `src/featurize/` | ~2.2 min |
| `featurize` | the card tables and FEATURE_SPEC from `--cards-frame`, exported, read back and equal to `rmfeat`'s; the corpus and explore games' dumps under `--selfcheck-featurize`; every decision featurized against the spec (shapes, indices), deterministic and row-order invariant; the actor entry point refusing the omniscient frame; learner targets; `rmfeat --featurize` / `--targets` equal to in-process; the recent-event history option (`docs/featurizer.md` §3.7); the registry's append-only property (a card and a keyword appended, an enum reordered: same layout and tensors; capacities refused); `python/pycheck.py --part featurize` and `--part open_vocab` (the fixture sources' decisions hashing to `tests/open_vocab/inputs.json`; drop an intentionally moved one with `learn/open_vocab_acceptance.py inputs --drop-moved`); Attribution's perturbations (`docs/workbench_plan.md` §2.8) through `rmfeat --perturb --check` on the first explore game's dump, every third frame of each seat: the empty change byte-identical, every unit's change (tokens, zone groups, cards, slots, relation types, each type's first token's cells) within the spec's shapes and indices, removed tokens' relations gone and the rest renumbered in order, every unit naming a row, handle and player of the seat's own frame; then `--slot-stats` of the dump and the same checks under the `mean` and `resample` baselines on every eleventh frame | ~1.5 min |
| `traj` | the trajectory writer (`docs/trajectory_format.md`) under `ci_out/traj/`: in-process runs (bo1 with the omniscient part, bo3, concessions, a capped seat-A-only run), each `--replay` writing the same episodes; footers, replies and tensors against the decision log and dump; μ a distribution with mass at the reply; `python/rmtraj.py` vs the C++ reader; an `rm_driver.py --rmlog-dir` match re-featurized; the teacher channel (`python/teacher.py`) at mix 1 (scripted:hard's decisions, its stream the `--machine` stream byte for byte), mix 0 (decision log unchanged) and mix 0.5 (the coin, the μ mixture, the replay); broken teachers fatal | ~27 s |
| `actorloop` | the continuous actor (`docs/actor.md`) under `ci_out/actorloop/`: status and results files against the episodes' footers; a two-entry schedule's matches each replayed alone to the same episodes, a rerun with the same seed and id identical; the per-game cap; a taught entry's labels a fresh `teacher.py`'s on each replay; a schedule rewritten mid-run, then SIGTERM (exit 0, `QUIT` in the log, no partial episode); `sb_explore` (μ = `q` from replayed frames, swap counts covering 0–10, card weights moving frequencies); malformed schedules refused | ~10 s |
| `actor` | the policy projection's unit checks always; the rest opt-in (below): the reference policy built (`tools/nn/make_reference_policy.py`, under `ci_out/actor/`, rebuilt only when the spec or script changes); `learn/open_vocab_acceptance.py gate` (skipped without the original checkpoints under `runs/`); `nn:` vs explore bo3 matches (TorchScript and `.pt2` argmax identical, a `temp=0.02:thresh=0.03:disc=16` match's μ checked, a `--replay` re-featurized exactly, the three exports agreeing); `nn:remote:` through `rmserve` and `tools/nn/serve_check.py` (same games, evaluations within 1e-5, handshake refusals, a killed server failing the actor); hot reload under `--actor`; a continuous `--actor` smoke run; `sb_explore` with `nn:` self-play; the recent-event history case (`--feature-history 32`) | ~3 min (+~50 s when the reference policy is rebuilt) |
| `workbench` | opt-in (below): `rmwork --selftest`, headless, output in `ci_out/workbench/selftest.txt`: every preset layout in Play and Replay (the preset's tools open and no other, each docked into the mode's dockspace in its tab group, in order, the first tab selected, at the preset's split ratios), Reset Layout, each mode keeping its own layout across switches and an ini round trip, a closed tool reopening where it was, the key bindings (in their mode only; Esc closing the dialogs), every preset reading back from its dockspace as itself, named layouts (settings.json round trip, rebuilt exactly with their tab order and shown tabs, the live dockspace saved and restored, overwrite asked, rename and delete, malformed ones refused, Save Layout As and Manage Layouts), Settings → Keys rebinding (a captured chord, a conflicting one refused, Esc cancelling, resets), the later stages' settings (defaults from the repository, path resolution, round trip, refusals) and the status bar's engine and socket health, every embedded font family × weight drawn at the size asked for, the bars and default presets at 11 and 20 px, every colour theme in both modes (its ImGui and ImPlot colours, text contrast ≥ 4.5 and secondary text, series and icons ≥ 3 on their surfaces, the series in a plot ΔE ≥ 25, text on the Table's seat and counter chips ≥ 4.5), the Table's layout and fit on synthetic boards (every object placed once, attachments, attackers forward, inside the panel at four sizes; identical lands and tokens stacked only on a narrow panel, different creatures wrapped onto two lines, tokens stacked before a row wraps), the step bar's cells, a step's stop toggled and round-tripped, a priority decision's costs read from script costs and oracle text (a fetchland's, a token's sacrifice, loyalty, cycling, Uro's escape, a flashback with and without mana, Daze's and Force of Will's alternative costs, suspend) and their labels, the magnifier's two faces on screen, its hover and selection links and its background tiles, `settings.json` round-tripped in a temporary `$XDG_CONFIG_HOME` (the font and theme fallbacks too), the start page (the result lines and the game sidecar; a synthetic games directory with finished, stopped, malformed, running and unread games and more than the page lists; recent replays with and without a cached result; runs with and without `status.json`; the status blocks with and without engine, `robomage_nn`, working directory, card images and socket; the page in place of the dockspace, its ✕, File → Start Page's key, File → Close with its setting on and off, a status block's Settings link); then `rmwork --selftest-data --engine <engine>` (skipped with a message without the engine), output in `ci_out/workbench/selftest_data.txt`: Replay's data on real replays in a temporary `$XDG_CACHE_HOME` — corpus logs (a bo1 game and a `--no-auto-mana` one) and a bo3 match the engine plays for it (explore vs random, `--trajectory-out`): the decision count is the log's, every decision's A, B and omniscient frames decode with their viewer's schema and agree on its key, the menu only in the deciding seat's and the omniscient frame, the bo3 games and sideboarding phases and result, the narrative cut at every decision, each seat's log exactly its own frames' events (the opponent's opening draws unnamed), the cache (a hit, an engine of another identity replaying again, a truncated dump reaching the error dialog; the match index kept beside the cached dump: read by a second open, equal to the one built from the dump, a corrupt one and one of an older version rebuilt; Play's live match with none), episodes joined (a `--replay --trajectory-out` episode opened alone finds its log; the bo3 match's join by name, the random seat's μ), a run directory's listing, navigation and viewer switching updating the status bar, a malformed log, a diverging replay, a missing engine and a cancelled load, the file picker's filters; the Table (on those matches and three more corpus logs with blocks, attachments and targeted stack objects): every decision's model in every view against its frame (every battlefield object placed exactly once, attachments in their host's pile; tapped, damage, counters, loyalty, P/T carried; the stack top first with its targets resolved; nothing named that the frame hides; only the bottom seat's hand) and its layout inside the panel at four sizes, a narrow one among them (a stack's members interchangeable and identified, its count its size, stacks and wrapped rows only where full-detail cards would be narrower than 69 px, a row wrapped only with no group left to stack), every decision drawn in every view, hovering a stack object lighting its target, hovering a playable card lighting its menu row, Q and right-click magnifying, a click on Replay's step bar toggling no stop; the other tools (Players, zone windows, Choices, Decision Inspector, State Tree; every decision in every view): the players' counts the frame's, each zone window exactly the frame's cards and (in a seat view) its knowledge's (`k_position`, `k_known_hand`), names only for cards the frame gives, Choices the menu rows (none, and no evaluations, in a non-deciding seat's view) with ► on the log's reply, the Inspector's fields the frame's and the episode's, the State Tree holding every table and row of the frame and naming a handle only where the viewer's object table identifies it, a menu row's links to its objects the same handles in Choices, the Table and the tree, a Table object or menu row linked making the tree's row of its object the scroll target with the nodes above it expanded; a sample drawn with every tool open, a hovered Choices row lighting its source on the Table, a card hovered on the Table scrolling the tree (scrolled to its end) to its row once and not again while the hover stays, zone windows opened, magnified in and closed by Esc; then the bo3 match's episodes rewritten with made-up evaluations (two policies, the second on seat A only): the Value Graph's series the episodes' values on Player A's scale (a seat view only its own seat's), Swings and the seats' disagreement recomputed, next swing walking them, the Inspector's values on both scales and H(π), Choices' π (the shown policy's, else the first), the Model's Read, the status bar's V (none in the other seat's view); then, opt-in as above (`actor_setup` builds the reference policy `ci_out/actor/ref.ts.pt` when missing or stale), `rmwork --selftest-evals <ref.ts.pt> --engine <engine>`, output in `ci_out/workbench/selftest_evals.txt`: the `robomage_nn` beside the engine plays a bo3 with the reference policy's `nn:` agent on both seats (`--trajectory-out`, on the CPU), Replay opens it with every decision joined to its episode step, and the same checks run on its real evaluations (one policy, the agents the episodes name), plus every decision evaluated with π a distribution over its menu and each seat's Value Graph series a point per decision of the seat; then Play (in a temporary `$XDG_DATA_HOME`): the engine's arguments per opponent kind and the binary chosen (`robomage_nn` only for an `nn:` opponent), the New Game dialog's decks and checks, games played to their end through the session by a seeded random row (a bo1 as seat A against explore with Model's Read asked for and no server, a bo3 as seat B against random with sideboarding, the scripted agent through the teacher channel when `python3` runs); the Sideboard tool: the plan's rules (locks, balance, the 15-swap cap, undo) and its reply steps on made-up decisions (a menu without the planned row refused), the guides file against its JSON, a bo3 as seat A against explore sideboarded through the tool (the guide's plan for burn vs bug and Use This Plan, an unreachable target stopped before anything is sent, a planned swap sent as IN, OUT and Done and game 2's frame showing the planned deck), that match in Replay (the phase's swaps in seat A's view; seat B's own 75 and a note in seat B's): every record kept the human seat's (the other seat's match-end records dropped unread, only after its last decision), the stops answering exactly the plain priority passes with an empty stack in unstopped steps, the result screen, the decision log in Replay with the seat's decisions, replies and frames (event tables aside), Save Decision Log; F2, F6 and the stops on the recorded frames, on made-up facts and given live; a click on a step of Play's step bar toggling its stop, `settings.json` keeping it, a second click turning it back; decisions answered on the Table on two sculpted games as seat A against explore (the engine's `--hand-a`, `--battlefield-a` … flags, seat B holding only Islands): Lightning Bolt's target by a click on a creature, Prismari Charm's first target by a click on the opponent's chip and its second offered with the first marked chosen and answered by the prompt line's Done button, and under `--no-auto-mana` the cost to pay as pips, a click on a dual land opening the popup of its two rows and the popup's item answering, Dismember's Phyrexian pips answered by the prompt's life and mana buttons, lands clicked to pay, an untapped land clicked at a priority decision tapping it for mana and the pool on the player, each reply the row naming what was clicked and every pick an object the human's frame shows on the Table or a player; that targeting game in Replay (the highlights in seat A's view, none in seat B's, a click selecting and answering nothing); a sculpted combat as seat A against the `first` agent (Goblin Guide, Monastery Swiftspear and Grizzly Bears against Grizzly Bears, Baleful Strix and Kaito, Bane of Nightmares): attackers planned on the Table (one armed and aimed at the planeswalker by a second click, one at the player by the opponent's chip, one planned and taken back; the planned ones standing forward), nothing sent while planning, Attack with 2 sending the plan one reply per decision and the next frame's combat table holding exactly those attackers and targets; the double-blocked Goblin Guide's damage assignment (both blockers lit with their lethal damage, Done a button) answered by a click on the strix, which dies with the bears unharmed; the opponent's attack blocked by a plan (a blocker armed by a click and planned onto an attacker, taken back, then dragged onto the other) sent with Block with 1, the combat table holding the block; that game in Replay (the declaration lit, a click selecting and planning nothing); the card picker on sculpted games: a fetchland's search (the two finds lit, the rest of the library dimmed and hidden by the Unpickable toggle, Fail to find under the cards, a click on a find answering with its row), Ponder's two picks (the pile showing each in its final place, the next frame knowing the library's top three in the order picked), Preordain's scry (each card's Top and Bottom buttons, the Bolt sent to the bottom and listed there, Daze kept on top and drawn), Doomsday's five picks from the library and the graveyard (identical cards one entry with their count, the filter typed and set, the pile at the last pick the next frame's library top five), that game in Replay (the picker read-only with the chosen card marked, a click answering nothing, none in seat B's view); the opening hand (Keep and Mulligan on the prompt line, the hand drawn large, two mulligans and Keep, the two bottom picks planned by clicks: one taken back, a third refused, Undo, then sent with Put 2 on the bottom, one reply per decision, the cards gone from the hand) and its bottom pick in Replay (the chosen card marked, no plan buttons, a click planning nothing); modes, numbers, names, costs and pickers on sculpted games (each reply the row naming what was clicked, a later frame showing its effect): Kozilek's Command's X by the prompt line's stepper (the pips the value leaves) and two modes clicked in the choice panel (the second marking the first chosen with its target; Player B's two Spawn tokens once it resolves), Collective Brutality's Done by the prompt line's button (the bears die), Fury's evoke exile clicked on the hand, its triggers ordered in the panel and a point of divided damage clicked (the next decision counting it), Cabal Therapy's name searched and clicked in the panel (Player B discards it) and its flashback's sacrifice clicked, Daze's Island returned by a click and its unless cost declined by a button (the Bolt countered), Force of Will's blue card exiled by a click (only the blue cards lit), Murktide Regent's delve count by the stepper and a graveyard card exiled in the picker, a cleanup discard of two cards planned on the hand and sent, Thoughtseize's pick from Player B's revealed hand in the picker (its lands dimmed), Watery Grave's 2 life accepted by a button (life 18, untapped) and the legend rule's Thalia kept by a click, and the Kozilek game in Replay (the stepper at the X chosen, the panel read-only with the mode chosen marked, nothing in seat B's view); priority decisions on sculpted games: a Mountain played and Lightning Bolt cast by clicks on the hand (the playable cards lit, the prompt saying what may be done, Pass on the prompt line, the Bolt's cost {R}), its target clicked, Pass by the prompt line's button with the Bolt on the stack and the Space key at the next priority decision, Daze's two ways in a popup with their costs and the alternative chosen there, Polluted Delta's ability by a click ({T}, pay 1 life, sacrifice it; its search once it resolves), Wasteland's mana and ability in one popup under `--no-auto-mana` (the ability chosen, its target clicked), Cabal Therapy's flashback from the graveyard by the prompt line's button, and the first game in Replay (the Bolt's cast and a pass marked chosen, a click answering nothing, nothing in seat B's view); the concession dialog, `-2` then a queued `-3`; a killed engine, a broken stream and a record that is no frame reaching the error dialog; the start page listing the bo1 and bo3 games with their seats, opponents, decision counts and results (the bo3's the live match's), the bo1 opened in Replay from it and File → Close bringing it back, the game whose engine was killed listed as stopped and opened in Replay; `--selftest` adds the evaluations' units on a synthetic match and the `new_game` settings; Simulate (`docs/workbench_plan.md` §2.9) in all three: `--selftest` its units (the `simulate` settings' round trip and refusals, the configuration from the choices, the engine's arguments per seat (swapped, scripted on one or both seats with one `--teacher-mix`, a model's file and options, Play's setup flags after the opponent's agent), matches judged from canned output, the tally's Wilson interval, `simulation.json` read back and a mid-batch one opening cancelled, the past simulations newest first) and the App (the dialog from its key, the start page and File; a stand-in engine's match failing into the error dialog; the Simulations window, a failed row, New…, Open Folder…); `--selftest-data` explore vs random opened in Replay when it ends, the scripted agent against explore and against itself, three alternating matches on two jobs in the window (the swapped match's decks, `simulation.json` read back, a row opened in Replay), every finished match's result the engine's `--replay`'s, and a stand-in engine failing one seed and sleeping on another (the failure that match's alone, a cancel mid-batch keeping the finished logs and removing the partial one); `--selftest-evals` the reference policy against itself and against the scripted agent in `robomage_nn`, each result the plain engine's replay's; Attribution (`docs/workbench_plan.md` §2.8) in all three: `--selftest` its metrics on known evaluations, its cache entries (round trip, damaged and truncated ones refused, the paths, a live game's directory, pruning) and Shapley on an additive game; `--selftest-data` with the stand-in RMIP server on the corpus bo1 log (no server: nothing attributed; a decision of each seat in the omniscient view: the unchanged evaluation the re-evaluation's, every unit's evaluation that of the decision's tensors changed by it, the second pass; the decision visited again from the cache with a version check only; a token expanded into its cells and the Tokens tool's heat on them; the Table layer's tints on the cards and chips with values and none with the layer off, the omniscient view's mapped handles; a hovered row linking its object; the State Tree's impact sorted; a seat view at the other seat's decision attributing nothing and sending nothing; Shapley efficient; a cancel leaving a whole entry or none) and a live Play game explained at the human's decision, its entry found again when the game opens in Replay; `--selftest-evals` decisions of both seats through rmserve with the reference policy (the unchanged evaluation the episode's and the re-evaluation's within 1e-5, Shapley efficient for V, log π and the margin) and occlusion's cost (`ATTRIBUTION-COST` lines); Policy Compare (`docs/workbench_plan.md` §2.10) in all three: `--selftest` its metrics on made-up distributions against hand-computed values and a synthetic match's comparison, agreement (overall, per seat, per turn), ranked list and candidates per viewer; `--selftest-data` with the stand-in server (its policies' π differing) on the corpus bo1 log: the tool with no policy and with one, three re-evaluated policies' KL, TV and argmax at every decision against values computed from the merged records, the agreement counts, the default reference, the ranked list sorted and filtered to argmax differences, a row's jump and a decision panel row's link, every view (a seat view only its seat's decisions), the catalog popup listing a made-up runs directory and two of its policies brought in (served under the next ids, the older first, the replay re-evaluated with them alone, the newest the reference), a bring-in refused in Play; `--selftest-evals` the reference policy served twice through rmserve: the recorded and both re-evaluated policies agreeing at every decision (KL below 1e-6); card images on a synthetic cache of tiny PNGs in a temporary directory in both (`--selftest`: the manifest and its refusals, the printing overrides' file, the printing and face each card shows (an override, both faces of a double-faced card, a token by index and by name, an image not cached, a back), the cache's status, the LRU and its levels, the decode thread handing textures over, a file that does not decode, eviction past the budget, the renderer drawing the image or falling back to text; `--selftest-data`: every card the Table draws on sampled decisions in every view resolving to a cached image or to the text renderer as the cache says, each image decoding); then `python3 tools/scryfall/fetch_images.py --selftest` (the printing rule on canned Scryfall JSON, a token's printing from the token set of a card that makes it first and the token makers read from card scripts in a temporary directory, no network; skipped without `python3`), output in `ci_out/workbench/fetch_images.txt` | ~50 s |
| `knowledge` | `rmknow --selftest`, `--exactness 40`; explore games under `--selfcheck-knowledge`; stream sessions against their dumps (two under `--closed-decklist`); `rmknow --isolation` and `--truth` on every dump | ~60 s |
| `fuzz` | C++ explore vs explore in process over the matchups: `--fuzz-games` bo1 games each (matchup `k` game `i`: seed `seed + 1000k + i`); one bo3 `explore:patient` match per ring matchup (`seed + 500000 + k`); `--manual-mana-games` bo1 games per ring matchup under `--no-auto-mana` (`seed + 700000 + 1000k + i`; with an explicit `--matchups` list the ring is that list); the fixture decks (below). Transcripts `ci_out/fuzz_<a>__<b>.txt`, `fuzz_bo3_…`, `fuzz_manual_…`, `fuzz_mech_…` | ~50 s |
| `bench` | per-decision wall clock of a fixed workload (explore vs explore, bo1, three league pairings × three games, one engine at a time, process start left out) by stage: engine alone (`--replay`), + frames (knowledge `off`), + knowledge sidecar (`expert`), + featurizer; the stages must play the same games. With the `actor` opt-in, `nn:` in process and `nn:remote:` through `rmserve` (compute vs round trip). An optimised build (`bin/release/`, `bin/pgo/`) whose actor loop is below 1,500 decisions/s is a warning, never an error | ~1.7 min, ~3.5 min with evaluation (debug) |
| `smoke` | scripted hard vs hard through `python/rm_driver.py` (transcripts `ci_out/smoke_<a>__<b>.txt`) and `python/pycheck.py`: the Python decoder against the engine's `--state-dump json`, decoder refusals, the self-lethal guard, the driver's seeds and imports (stdlib only), the sideboard guides against `docs/sideboard_guides.md` and the decks, `scripted:hard+sb` bo3 matches at the guides' targets, `python/rmtb.py` on C++-written RMTB files. Skipped with a notice without `python3` | ~35 s |

**The `actor` tier is opt-in.** It needs `bin/<config>/robomage_nn` and `bin/<config>/rmserve`
beside the engine (`make actor LIBTORCH_DIR=…`) and `$ROBOMAGE_TORCH_PYTHON` naming a Python with
torch, onnx, onnxscript and onnxruntime; without either it prints `skipped: …` (not a warning) and
passes, so plain `make check` needs no torch. The `bench` tier's evaluation stages have the same
opt-in. The `workbench` tier is opt-in the same way: it runs `rmwork --selftest` and
`rmwork --selftest-data` when `bin/<config>/rmwork` exists (`make workbench`, which needs GLFW) and
otherwise prints `skipped: …` and passes; its `rmwork --selftest-evals` (real evaluations) also needs
the `actor` tier's opt-in (`robomage_nn` and `$ROBOMAGE_TORCH_PYTHON`), else it prints why it was skipped. The learner's checks (`learn/check_rmtrain.py`, its `--smoke` runs) are not part of the gate
and are run by hand (`docs/rmtrain.md` §9).

```bash
ROBOMAGE_TORCH_PYTHON=/path/to/venv/bin/python bin/debug/robomage_check --tier actor,bench
bin/debug/robomage_check --tier scenarios --scenario 'cr702_21a*' --show narrative
bin/debug/robomage_check --tier replay                 # one tier
bin/debug/robomage_check --tier cli --record           # rewrite the CLI goldens after an intended change
bin/debug/robomage_check --tier fuzz --matchups all --mode explore:patient --fuzz-games 20 --seed 7
bin/release/robomage_check --tier bench                # throughput of the release build
```

### Fixture decks (the `fuzz` tier)

A fuzz game passing proves nothing about a path it never reached, so a primitive gets a fixture: a
dense deck that reaches it in most games, and a list of what must have been reached.

- `bin/resources/decks/tests/mech_<primitive>.dk`: vocab cards only. It is outside `league/` and
  `heldout/`, so no matchup preset, training run or onboarding step plays it.
- `mech_<primitive>.expect` beside it: one coverage key per line, `<kind>:<name>` (`cost:SACRIFICE`,
  `cast:flashback`; the kinds and names are `docs/game_running.md`'s `--coverage` table), `#` starting
  a comment.

Every `fuzz` run plays each fixture against itself, `--fuzz-games` bo1 `<mode>` games and at least 16,
under `--coverage` (fixture `f` game `i`: seed `seed + 800000 + 1000 slot(f) + i`, the slot a hash of
the fixture's name), and sums the counts into `ci_out/coverage/mech_<primitive>.txt` in
`--coverage`'s format, so a new `.expect` is written by reading that file. An **error** names each
`.expect` key with no hits; so does a deck `--check-decks` refuses, a missing or empty `.expect`, an
`.expect` with no deck, and a game that wrote no counts. Crashes, error lines and draws report as
for any fuzz game. Only the mirror's counts count.

An explicit `--matchups` list plays only the fixtures it names: `--matchups tests/mech_<primitive>`
for one fixture, `--matchups 'tests/mech_<primitive>:*'` for it plus the deck against every
playable deck in every leg.

```bash
bin/debug/robomage_check --tier fuzz --matchups tests/mech_cost_paths --fuzz-games 100
```

### Throughput reference (bench)

The `bench` tier on the 7950X box, idle, one process at a time: wall clock per game (9 bo1 games,
process start left out; per decision in brackets).

| Stage | release | `make pgo` |
|---|---|---|
| engine alone (`--replay`) | 14.5 ms (138 us) | 11.7 ms (112 us) |
| + frames, knowledge off | +19.0 ms (+182 us) | +16.1 ms (+155 us) |
| + knowledge sidecar (expert) | +13.8 ms (+132 us) | +12.3 ms (+118 us) |
| + featurizer | +2.7 ms (+26 us) | +2.2 ms (+21 us) |
| the actor loop | 49.9 ms, 2,094 decisions/s | 42.4 ms, 2,463 decisions/s |

As one `--actor` process (40 bo3 matches, explore vs explore, five league pairings, knowledge
expert, `--featurize-seats`): 2,116 decisions/s release, 2,355 with `make pgo`.

### Error and warning classification

`tools/robomage_check/scan.h`:

- **Error** (fails the gate): an engine crash (nonzero exit / EOF mid-game), fewer games completed
  than requested, a failed check, any output line starting `ERROR:` / `FATAL:` or holding a Python
  traceback, `Segmentation fault`, a failed assertion, `Aborted` or `core dumped`, or a
  `SELFCHECK-*:` summary without `errors=0`.
- **Warning** (flagged for review, passes): a game with no winner — a draw (CR 104.4a) or a
  **stall** — or an engine `WARNING:` line. A drawn game's transcript is saved as
  `ci_out/<tier>_<a>__<b>_g<i>_draw_<stamp>.txt`.

`WARNING: Unrecognized ability param` lines stay in the transcripts but are not surfaced (they fire
in bulk for cosmetic / AI-hint params). Params the parser drops without warning are `ignored_keys`
in `src/parse.cpp`, and those of `param_form_checked`, which `--parse-report` reports outside their
safe forms.

## The replay corpus (`tests/corpus/`)

Each corpus game is two files:

- `<name>.rmlog` — its decision log (RMLOG v3), written while `python/record_corpus.py` plays the
  scripted agent (hard, both seats) through `python/rm_driver.py`: one bo1 game per pairing at a
  frozen seed — the curated `delver` / `doomsday` / `mav` decks in every seating (9 games) and the
  `lg_<a>__<b>` league pairings and mirrors (45 games) — plus two explore vs explore games under
  `--no-auto-mana` (`manual_doomsday_vs_delver`, `manual_mav_mirror`), which cover interactive
  mana payment;
- `<name>.replay` — its baseline: the stdout of `robomage --replay <name>.rmlog --machine
  --narrative --trace-decisions`, the banner's log path reduced to the file name and the trace's
  board lines left out (`tools/robomage_check/corpus.h`).

Checking needs no Python; re-recording needs the agent. The corpus is identical from debug and
release builds and across platforms: all gameplay randomness goes through `src/stable_rng.h`. Never
call `std::shuffle`, `std::uniform_int_distribution` or `rand()` for a gameplay decision: their
output differs between libstdc++ and libc++.

## Workflows

- **`.github/workflows/ci.yml`** — push / PR to `main`, and `workflow_dispatch`: `make check` with
  the fixed seed, so a red run is the diff's fault, not RNG. `ci_out/` is uploaded as
  `ci-transcripts-<run_id>` even on success.
- **`.github/workflows/nightly-fuzz.yml`** — daily 08:00 UTC, plus manual (`games`, `seed`
  inputs). Two matrix jobs (`explore`, `explore:patient`) run the `fuzz` tier with `--matchups all`,
  40 games per matchup and per manual-mana ring matchup by default, with a seed rotating by run
  number unless given. The job summary prints the local reproduce command; artifact
  `nightly-fuzz-explore[-patient]-<run>`.

Both use `.github/actions/setup-robomage`: Python 3.12, card-script provisioning, ccache and the
debug build. The card-script cache is keyed on the `FORGE_PIN` hash and saved right after
provisioning; the fetch tool is add-only, so a stale cache within a pin re-fetches only the delta.

## Reproducing a CI failure locally

1. **PR gate:** run `make check` (same command, same seed). The failing tier names its transcript or
   engine stderr under `ci_out/` (a replay drift writes the actual output to
   `ci_out/replay/<name>.replay` and names the first differing line); the `ci-transcripts-*`
   artifact has CI's copies. Re-run with the run's last line, its `Reproduce:` command.
2. **One scenario:** `bin/debug/robomage_check --tier scenarios --scenario <name> --show output`
   prints the engine's whole output; the engine command is the scenario's presets as flags
   (`scenarios.h`).
3. **One fuzz or corpus game by hand:** a fuzz game is `cd bin && ./debug/robomage --agent-a explore
   --agent-b explore --game-seed <seed> --deck-a league/<a> --deck-b league/<b> --narrative
   --trace-decisions` (bo3 leg: `explore:patient` and `--bo3`; manual-mana leg: `--no-auto-mana`;
   the seed is in the tier's report line); a corpus game is `cd bin && ./debug/robomage --replay
   ../tests/corpus/<name>.rmlog --machine --narrative --trace-decisions`.
4. **Nightly fuzz:** run the `robomage_check --tier fuzz …` line from the job summary; the
   `nightly-fuzz-*` artifact has the transcripts.

**Engine self-checks on any game.** `--selfcheck-state` (`docs/state_format.md` §10),
`--selfcheck-featurize` and `--selfcheck-knowledge` (`docs/knowledge_store.md` §9.5) change no other
output; failures are `ERROR:` lines on stderr, and each prints a `SELFCHECK-…: … errors=N` summary
at match end. `--selfcheck-state-sample N` (default 8) sets how often the sampled permutation and
snapshot checks run. `--rmstate-dump FILE` (`%p` = the process id) writes every viewer's frames;
`robomage --rmstate-read FILE` re-encodes them byte for byte, and `bin/debug/rmknow --isolation
--viewer A|B --in FILE` recomputes each seat frame's `k_*` tables and compares them with the
engine's. The driver passes engine flags through `--engine-flags "…"` or `$ROBOMAGE_ENGINE_FLAGS`.

## Intentionally changing behavior

The `replay` tier pins current behavior, so an **intended** change re-records the corpus in the
same commit:

```bash
make regen    # provision the card set, `make`, then python3 python/record_corpus.py
```

`record_corpus.py` replays the corpus games with the fresh debug binary, rewriting each
`<name>.rmlog`, then runs `robomage_check --tier replay --record` for the baselines. Review and
commit the changed files with the reason in the commit message. A scenario whose assertion an
intended change breaks is edited by hand.

`make refetch` deletes the local card/token scripts, re-provisions them at the pinned Forge commit
and runs `make check` **without** re-recording, to detect script drift from the committed corpus;
if the drift is intended, follow with `make regen`.
