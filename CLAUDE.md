# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code Style

- Don't put new functions in main.cpp
- Don't edit my comments for spelling or punctuation. Only change them if something substantive changed.
- Avoid inline logic for anything that will be repeated; write new functions that are reusable
- Declare local functions as private in the class, if the header contains a single class/struct, if header does not contain a class, write them as static functions in global namespace C-style.
- Iterate through mEntities when possible (working within a system class), rather than iterating through all entities
- Try to consolidate iterations through entities within a function, rather than iterating through many times
- To find battlefield permanents, use the shared accessors in `src/game_queries.h` — the
  `is_battlefield_permanent(entity, ctrl)` predicate as a loop guard / single-entity check,
  or `battlefield_permanents(mEntities, ctrl)` for the whole list — instead of open-coding
  the `Permanent` + `Zone` + `BATTLEFIELD` (+ controller) scan inline. These bake in the rule
  that a phased-out permanent is treated as not on the battlefield (CR 702.26e), so do **not**
  add your own `is_phased_out` check at the call site. The only code that reads
  `Permanent::is_phased_out` directly is the phasing subsystem itself (the untap-step
  phase-in/skip in `classes/game.cpp`) and the phase-out setter in `effects/effect_phases.cpp`.
  Add similar shared accessors (next to these) when a new entity-scan pattern starts repeating.
- Static (local) functions should be forward declared at top of source file for clarity
- C++17 with exceptions disabled (`-fno-exceptions`)
- Two interactive front ends, both Python, both sitting on the shared driver in
  `train/game_driver.py`: the Textual TUI (`train/tui_game.py`) and the PySide6 GUI
  (`train/gui_game.py`, with its analysis window in `train/gui_analysis.py`). There is NO C++
  front end (the old raylib GUI was removed); the engine is always a `--machine` subprocess.
- Uses clang-format configuration in `.clang-format`
- DO NOT MODIFY CARD SCRIPTS
- When given a long list of tasks or bugs to fix, do them one at a time (unless it is sensible to batch some)
  and use subagents for each one. Instruct the subagents to not spawn additional agents. 

## Project Overview

Robomage is a C++ implementation of a Magic: The Gathering game engine using an Entity Component System (ECS) architecture.

**Scope: two-player games only.** The engine simulates exactly one 1-v-1 matchup (Player A vs Player B). Multiplayer rules (CR 800+) — more than two players, ranges of influence, the monarch passing among 3+ players, "each opponent" over multiple opponents, leaving-the-game cleanup for a third seat, etc. — are out of scope. Implement card effects against the two-player case; do not add multiplayer-only machinery.

Every game decision logged as an integer. Games can be replayed deterministically when provided with the correct seed.

The python side of the project enables machine learning of the game and analysis.

## Rules Reference (authoritative)

The official **MTG Comprehensive Rules** are checked in at
[`docs/mtg_comprehensive_rules.txt`](docs/mtg_comprehensive_rules.txt), with a navigation guide
in [`docs/mtg_comprehensive_rules.md`](docs/mtg_comprehensive_rules.md).

**Whenever you implement or test a game mechanic, consult these rules** — they are the ground
truth for how a mechanic *should* behave (timing, priority, state-based actions, combat, keyword
definitions, the layer system), independent of the engine's current implementation. Don't read
the 9k-line file end to end: look up the relevant numbered rule with `grep` (e.g.
`grep -nE "^509\." docs/mtg_comprehensive_rules.txt` for combat, rule 702 for keyword abilities,
704 for state-based actions, 613 for layers). See the navigation guide for lookup recipes. Keep
this version current if a newer Comprehensive Rules release supersedes it.

## Build Commands

- **Build the project**: `make clean && make`
- **Clean build artifacts**: `make clean`
- **Build for release**: `make BUILD=RELEASE`

**The engine build is always headless** — the front ends (Textual TUI, PySide6 GUI) are pure
Python and drive `bin/robomage` over machine-mode stdio, so there is no front-end build option.
The deprecated C++ raylib GUI was removed entirely; there is no `GUI=TRUE` build option.
(`make HEADLESS=TRUE` is still accepted as a redundant no-op for backward compatibility.
PySide6 is an optional Python extra: `pip install -r train/requirements-gui.txt`.)

The compiled binary is output to `bin/<config>/robomage` — `bin/debug/robomage` from a plain
`make`, `bin/release/robomage` from `make BUILD=RELEASE` (separate object/output trees per
config, so switching `BUILD` never requires `make clean`).

**Which binary the Python tooling runs is a separate, independent choice from which one you
just built** — every `train/` entry point resolves its engine binary through `ROBOMAGE_BUILD`
(`train/cli_spec.py`), defaulting to a build tier per tool:

- **Debug by default** (`bin/debug/robomage`) — the test harness, `ci_check`/`make check`,
  `observe`, `baseline`, and anything else driven by `common_args()`'s plain default: these are
  correctness/verification tools, so the extra assertions (`-D_GLIBCXX_ASSERTIONS`, no `-O2`)
  are worth the slowdown.
- **Release by default** (`bin/release/robomage`) — the GUI (`gui.sh` / `play.py --gui`), the
  standalone TUI analysis browser (`tui_analysis.py` / `analysis.py`'s `interactive`/`search`/
  `browse` subcommands), and every PPO/AZ training driver (`train.py train/league/exploiter/
  sweep/fixed-model/alternate/az-selfplay/az-train/az-eval/az/az-league`). These push a lot of
  decisions through the engine where speed matters more than assertions, so a plain `make`
  alone isn't enough for them — you also need `make BUILD=RELEASE` before running one, or they
  fail to find `bin/release/robomage`.

**Override**: set the `ROBOMAGE_BUILD` environment variable to force every tool (both tiers
above) onto a specific config — e.g. `ROBOMAGE_BUILD=debug train/.venv/bin/python train/train.py
league` runs training against the debug binary instead (useful to reproduce an assertion
failure the debug build catches but release doesn't). Every subcommand also accepts an explicit
`--binary <path>` flag as a per-invocation escape hatch, independent of `ROBOMAGE_BUILD`.

**Codegen is part of the build.** The default `make` target runs `pygen` before compiling,
which regenerates ALL auto-generated files **unconditionally on every build** (they
write-if-changed — see `train/gen_util.py` — so an unchanged output keeps its mtime and forces
no recompile). Two of them derive purely from tracked sources and are **committed**:
`train/_enums.py` (from the C++ headers) and `src/gen/archetypes_gen.h` (from
`decks/archetypes.json`). The other three derive from **card-script CONTENT**
(`ManaCost`/keywords/types) — and card scripts are gitignored and fetched from Forge, so their
content is not expressible as a Make prerequisite. To keep a machine's stale local scripts from
committing stale matrices, those three are **UNTRACKED** and regenerated locally every build:
`train/card_costs.py`, `train/card_props.py`, and the C++ mirror header
`src/gen/card_costs_gen.h`. So **adding/removing a vocab entry and running `make` keeps the
cast-cost/property matrices in sync automatically**; there is nothing to commit for them. The
`pygen` CI tier only guards the two tracked outputs (the untracked three cannot go stale — they
are rebuilt from the pinned, provisioned scripts on every build and in CI).

**Run build/test commands plainly so they don't trigger a permission prompt.** A single
command, or a single pipeline whose programs are all allowlisted (`make`, the `train/...`
python entry points, `grep`/`head`/`tail`/`echo`), auto-approves. Avoid the shell plumbing
that forces a prompt: chaining statements with `;`/`&&`, and `${PIPESTATUS[0]}` / `$(...)`
expansions. Prefer bare `make` and `... | grep -i error` over
`make 2>&1 | grep ... ; echo "EXIT:${PIPESTATUS[0]}"`; read the printed output for status
instead of `$PIPESTATUS`.

## Long-running commands (Claude)

When Claude runs a long command (build, training, gate, fuzz, bench) it must run it **in the
background** and make the live output inspectable by the user:

- Always state the task's output-file path in chat when launching it, so the user can
  `tail -f` it while it runs. (No symlinks or other indirection — just the path.)
- Foreground runs stream nothing until they exit (for Claude AND the user), so anything
  expected to take more than ~30 s should be backgrounded rather than run with a long timeout.

## Testing guidelines
-**The standard gate is `make check`** (builds + runs `train/ci_check.py`: codegen-sync,
 vocab coverage, byte-identical replay corpus, deterministic league smoke, short fuzz). It
 must pass before pushing, and CI runs exactly it. See [`docs/ci.md`](docs/ci.md) for the
 tiers, how to reproduce a CI failure, and how to intentionally re-record the replay corpus /
 regenerate the codegen. `train/fuzz_campaign.py` remains the manual exploratory fuzz tool.
-Don't use sed or cat - if possible don't pipe a bunch of commands together in a way that will require asking my permission, run build and test tasks as simply as possible
-Non fatal errors are not acceptable
-Draws are not acceptable
-Do not attempt to test cards that are not already in `src/card_vocab.h`. Cards absent from the card vocab are considered unimplemented.
-Do not use commas when using the test harness. The harness splits `--play`, `--hand-a/-b`,
 `--library-a/-b`, and `--battlefield-a/-b` on commas, so a comma inside a card name (e.g.
 "Thalia, Guardian of Thraben", "Tamiyo, Inquisitive Student") breaks the parse. **The fix is
 simply to omit the comma from the card name**: pass `Ajani Nacatl Pariah`, `Thalia Guardian
 of Thraben` — the engine resolves card names via `name_to_uid`, which strips punctuation, so
 the comma-free form loads the exact same card (verified: a `--battlefield-a "Ajani Nacatl
 Pariah"` preset yields the real Ajani, Nacatl Pariah). This works in every comma-split list
 flag (`--hand`, `--library`, `--battlefield/graveyard/exile/sideboard` presets). In `--play`
 specs, use a comma-free unique substring (`cast:thalia`, `Tamiyo@own`); a stacked `temp/`
 deck file (one card per line, commas allowed there) also works.
-train.py observe is helpful for checking new builds (it replaced the old diag/watch
 commands — one command observes any {scripted|model} vs {scripted|model} matchup).
 Use `--games N` for a multi-game regression pass (per-game results + W/L/D summary),
 `--verbose` for the full per-decision transcript (board state + action menu + narrative),
 and supply `--deck`/`--opponent` to test cards/decks relevant to recently implemented features.
 **observe defaults to bo3 matches**; pass `--bo1` for single games (`--bo3` is a redundant no-op).
-**Running games programmatically: `runner.run_match`.** For scripting games from Python
 (agents, decks, bo1/bo3, seed, output mode as parameters), use
 `runner.run_match(agent_a, agent_b, deck_a=…, deck_b=…, games=, bo3=True, seed=1,
 transcript="compact|verbose|narrative|quiet", out=…)` — agent specs are scripted tiers
 ("scripted"/"hard", "easy", "random", "explore"), "human", "auto", "play:<specs>",
 "actions:<ints>", the generalist model ("gen", "az:gen", …), or an explicit checkpoint
 path. All game-running
 tools (harness, observe, baseline, play.py, fuzz, ci_check, analysis, bench) sit on the
 same `runner.drive_game` loop — do not hand-roll new decision loops; see
 [`docs/game_running.md`](docs/game_running.md).
-**`observe` requires torch.** `train/train.py` imports `stable_baselines3`/`sb3-contrib`
 (hence torch) at module load, so `observe` is unavailable wherever torch isn't installed —
 notably headless cloud containers, whose ephemeral filesystem would re-pay torch's ~0.5–1 GB
 install every session. For a torch-free **regression** that exercises the *same* C++ engine and
 the *same* rule-based `scripted_action`, run scripted full games through the test harness across
 a few seeds instead: `train/.venv/bin/python train/test_harness.py --deck-a <a> --deck-b <b>
 --scripted --seed N` (vary N). `observe`'s unique extra coverage is model-driven play (a trained
 checkpoint) and the gym env/extractor observation pipeline — relevant to RL eval, not to
 verifying a card's rules behavior. Prefer `observe` when torch is present; otherwise fall back to
 harness scripted games.

### Test harness for card behavior verification

`train/test_harness.py` runs the engine with `--machine --narrative`, with game narrative visible alongside decoded binary state.

**Shuffling:** by default the harness shuffles each library with the seeded RNG (deterministic per `--seed`). Pass `--no-shuffle` when feeding a **stacked deck** so deck-file order = draw order (first 7 cards become the starting hand). `--no-shuffle` is implied automatically when `--hand-a`/`--hand-b` are used (they build a stacked temp deck). So inline-hand and scenario examples below remain deck-ordered without needing the flag; a plain `--deck-a X --deck-b Y` run shuffles unless you add `--no-shuffle`.

**Engine flags used by the harness:**
- `--no-shuffle` — skip initial library shuffle; cards are drawn in deck file order (opt-in; see Shuffling above)
- `--narrative` — enable game_log output in machine mode (perfect information)

**Quick start — specify hands inline:**
```bash
train/.venv/bin/python train/test_harness.py \
  --hand-a "Mountain,Lightning Bolt" \
  --library-a "Island,Island,Mountain,Mountain,Mountain,Mountain,Mountain,Mountain" \
  --hand-b "Forest,Grizzly Bears" \
  --library-b "Forest,Forest,Forest,Forest,Forest,Forest,Forest,Forest" \
  --scripted --max-decisions 30
```

**Pre-set battlefield** — start with permanents already in play (no summoning sickness):
```bash
train/.venv/bin/python train/test_harness.py \
  --hand-a "Lightning Bolt" \
  --library-a "Mountain,Mountain,Mountain,Mountain,Mountain,Mountain,Mountain,Mountain" \
  --hand-b "Giant Growth" \
  --library-b "Forest,Forest,Forest,Forest,Forest,Forest,Forest,Forest" \
  --battlefield-a "Mountain,Mountain" \
  --battlefield-b "Grizzly Bears,Forest" \
  --scripted --max-decisions 30
```

**Using pre-made deck files** (for precise library sizes without auto-padding):
Write `.dk` files to `bin/resources/decks/temp/` with `1 CardName` per line (hand cards first), then pass the deck name relative to `decks/`:
```bash
train/.venv/bin/python train/test_harness.py \
  --deck-a temp/my_test_a --deck-b temp/my_test_b --scripted
```

**Play modes:**
- `--play "cast:Lightning Bolt,target:Grizzly Bears@opp,pass"` — **semantic action specs**
  (the preferred way to script a precise line of play, especially for Claude). Each spec is
  resolved against *this* decision's live menu by intent (verb + card name + controller), so
  it is robust to index reordering and authorable up front. See "Scripting with `--play`"
  below. An unmatched or ambiguous spec **fails loudly with the legal menu** instead of
  silently playing the wrong action.
- `--scripted` — rule-based agent (from env.py) plays both sides automatically
- `--actions "9,0,7,0,8"` — pre-scripted *positional* action index sequence (fragile; prefer
  `--play`). Indices shift if the menu reorders. Still useful as a canonical replay form —
  a `--play` run prints the `resolved --actions:` integer list it played.
- `--interactive` — prompt a human at the terminal for each decision. **Claude cannot use
  this mode** — there is no TTY for Claude to type into, so the `input()` prompt just hits
  EOF and spins. When *you* (Claude) are driving the harness, never pass `--interactive`;
  use `--play` (or precompute `--actions`) instead.
- Default (no flag) — auto-passes every decision

**Scripting with `--play` (the method for automated/Claude-driven testing):**
A `--play` value is a comma-separated list of semantic specs, one consumed per decision (the
list is global across both seats, like `--actions`). Grammar lives in `train/action_spec.py`;
the common verbs:

| Spec | Meaning |
|---|---|
| `A:<spec>` / `B:<spec>` | **seat key** — pin a spec to player A or B (see below) |
| `pass` | pass priority / take the first choice |
| `keep` / `mulligan` | the opening mulligan decision |
| `cast:<card>` | cast a spell from hand |
| `play:<card>` / `land:<card>` | play a land |
| `activate:<card>[@own/opp]` | activate a permanent's ability |
| `target:<card>[@own/opp]` or `target:<text>` | choose a target (card, or a player/modal by description text) |
| `attack:<card>` / `attack:done` | declare one attacker / confirm attackers |
| `block:<card>` / `block:done` | declare one blocker / confirm blockers |
| `mana:<w/u/b/r/g/c>` / `tap:<land>` | tap for mana |
| `search:<card>` / `search:fail` | library search / fail to find |
| `top:<card>` · `bottom:<card>` · `dig:<card>` | top-of-library / bottoming / dig pick |
| `desc:<text>` | match any action whose description contains `<text>` |
| `#<n>` or a bare integer | literal index escape hatch |

**Seat keys (`A:` / `B:`) — the easy way to drive both seats.** Because `--play` drives both
players from one global list, sequencing the priority hand-offs between them by hand is the
painful part. Prefix a spec with `A:` or `B:` (case-insensitive) to pin it to a seat. Before
applying the next spec the harness checks who currently holds priority: **if the next spec is
keyed to the seat that does _not_ have priority, the priority holder auto-passes** (the spec is
left unconsumed) until the keyed seat is on the clock. So you write each player's intended line
in order and never insert the intervening `pass`es yourself. A keyed spec also **auto-passes the
keyed seat forward through its _own_ priority windows until the action is actually legal** — e.g.
`A:attack:Voice of Victory` written while A still holds priority in its main phase keeps passing
(spec unconsumed) until A reaches the declare-attackers step where `attack:` is offered, so you
don't hand-count the `pass`es from main to combat either. (This forward-advance fires only on a
genuinely not-yet-legal action and only while a `pass` is available; an ambiguous or misspelled
spec, or one that reaches a mandatory choice — declare attackers/blockers, a target prompt — where
it still doesn't match, fails loudly with the legal menu rather than passing the game away.) The
seat key chooses *which decision* a spec applies to; it's independent of the `@own`/`@opp` target
suffix (which is the target's controller relative to the acting seat). Seat keys are optional — an
unkeyed spec applies to whoever has priority (the original behaviour) and is **not** auto-advanced
(it must match the current menu or fail loudly), so existing scripts are unchanged, and
keyed/unkeyed specs may be mixed. (Seat keys are meant for the dual-seat `--play` case; under
`observe --play-a/--play-b` each list already drives a single seat, so just leave specs unkeyed
or key them to that seat.)

Notes:
- Card names match case- and apostrophe-insensitively; an exact name wins, else a unique
  substring (`cast:bolt` → Lightning Bolt). Identical duplicate choices (e.g. four "Play
  Mountain" from a hand of duplicates) collapse to the first automatically; genuinely distinct
  matches are reported ambiguous — pin them with `@own`/`@opp`, `desc:`, or `#<n>`.
- The opening decisions are mulligans — start a sculpted-hand line with `keep,keep,…` (one
  `keep` per seat, since the harness drives both seats from the one list); with seat keys that's
  `A:keep,B:keep,…`.
- After the specs run out the game auto-advances (action `0`) to its end or `--max-decisions`.
- Read the transcript: every decision prints the numbered **Available actions** menu next to
  the decoded board state, so when a spec fails you can see exactly what was legal and fix it.
  Keep `--seed` constant (default `1`) so the prefix replays identically while you extend the line.

Example (Bolt kills a bear that is already in play), unkeyed:
```bash
train/.venv/bin/python train/test_harness.py \
  --hand-a "Mountain,Lightning Bolt" \
  --library-a "Island,Island,Mountain,Mountain,Mountain,Mountain,Mountain,Mountain" \
  --battlefield-b "Grizzly Bears" \
  --play "keep,keep,play:Mountain,pass,cast:Lightning Bolt,target:Grizzly Bears@opp"
```

Same line with seat keys (the harness fills in B's passes and A's post-cast passes for you):
```bash
train/.venv/bin/python train/test_harness.py \
  --hand-a "Lightning Bolt" --battlefield-a "Mountain" \
  --battlefield-b "Grizzly Bears" \
  --play "A:keep,B:keep,A:cast:Lightning Bolt,A:target:Grizzly Bears@opp,B:pass"
```

`train.py observe` takes per-seat `--play-a` / `--play-b` to drive one side by specs while the
other stays scripted/model (e.g. `observe --play-a "keep,play:Island,pass" --player-b scripted`).

For fully reproducible scenarios, capture the spec list (plus hands/libraries) in a JSON
scenario file via the `"play"` field (alongside or instead of `"actions"`).

**JSON scenario files:**
```bash
train/.venv/bin/python train/test_harness.py --scenario scenario.json
```
```json
{
  "name": "bolt_kills_bear",
  "hand_a": ["Mountain", "Lightning Bolt"],
  "library_a": ["Island", "Island", "Mountain", "Mountain", "Mountain", "Mountain", "Mountain", "Mountain"],
  "hand_b": ["Forest", "Grizzly Bears"],
  "library_b": ["Forest", "Forest", "Forest", "Forest", "Forest", "Forest", "Forest", "Forest"],
  "battlefield_a": [],
  "battlefield_b": ["Grizzly Bears"],
  "actions": [9, 0, 7, 0, 8],
  "seed": 1
}
```

**Output format** — at each decision point the harness prints:
- Narrative lines from the engine (casts, damage, zone changes, combat)
- Decoded game state (life, mana, hand contents, battlefield permanents with P/T/status, stack, graveyards)
- Available actions with human-readable descriptions (e.g. "Cast Lightning Bolt", "Target Grizzly Bears (opp)")

**Notes:**
- `--hand-a`/`--hand-b` auto-pads the library to a minimum 15-card deck. For precise small libraries (e.g. testing Thassa's Oracle with near-empty deck), write deck files manually to `decks/temp/` and use `--deck-a`/`--deck-b` instead.
- Card names in deck files omit apostrophes: `Thassas Oracle`, `Lions Eye Diamond`
- The scripted agent defaults to the **hard** (heuristic) tier everywhere a bare
  `scripted` spec is used — the test harness `--scripted`, the league/self-play scripted
  anchor, the mixed opponent pools, and observe/baseline/analysis. Hard adds a smart
  mulligan, combat simulation (profitable attacks/blocks), and evaluation-based targeting
  on top of the greedy baseline (casts spells when affordable, plays lands, attacks with
  all creatures); deck-specific combo lines (Doomsday) are preserved. Request the old
  greedy behaviour explicitly with `scripted:easy` / `scripted:greedy` (or `scripted:random`)
- Temp deck files in `decks/temp/` are cleaned up automatically when using `--hand-a`/`--hand-b`; manually created files in `decks/temp/` are not

## Architecture

### ECS Pattern

The codebase follows an Entity Component System architecture based on [Austin Morlan's ECS tutorial](https://austinmorlan.com/posts/entity_component_system/):

- **Entities** (`src/ecs/entity.h`): Simple uint32_t IDs, capped at `MAX_ENTITIES` (see `src/ecs/entity.h`)
- **Components** (`src/components/`): Pure data structs attached to entities
- **Systems** (`src/systems/`): Logic that operates on entities with specific component signatures
- **Coordinator** (`src/ecs/coordinator.h`): Central manager accessed via `global_coordinator` singleton

### Component Types

- **CardData**: Base card information (name, types, mana cost, oracle text, power/toughness, ability templates)
- **Zone**: Location tracking (library, hand, battlefield, stack, graveyard, exile, sideboard) with ownership and distance_from_top
- **Permanent**: Added when a card enters the battlefield; holds controller, tapped state, summoning sickness, and activated ability list
- **Ability**: Triggered, activated, or spell abilities with source/target/amount; also used as standalone stack entities for activated abilities
- **Creature**: Power/toughness, attacking/blocking state (added alongside Permanent for creatures)
- **Damage**: Damage counter tracking for creatures
- **Spell**: Marks a card entity that is currently on the stack as a spell
- **Effect**: Continuous effects (framework present, not yet applied)
- **Token**: Token permanent data (name, type)
- **Player**: Life total, mana pool, lands played this turn

### System Types

- **Orderer** (`src/systems/orderer.h`): Zone operations — card movement, drawing, shuffling, stack ordering
- **StateManager** (`src/systems/state_manager.h`): State-based effects (lethal damage, player death), permanent component lifecycle, `determine_legal_actions`
- **StackManager** (`src/systems/stack_manager.h`): Stack resolution — spells resolve to battlefield or graveyard; standalone ability entities resolve via `Ability::resolve()` then are destroyed

### Game Flow

The `Game` struct (`src/classes/game.h`) tracks:
- Current turn/step (UNTAP, UPKEEP, DRAW, FIRST_MAIN, BEGIN_COMBAT, DECLARE_ATTACKERS, DECLARE_BLOCKERS, FIRST_STRIKE_DAMAGE, COMBAT_DAMAGE, END_OF_COMBAT, SECOND_MAIN, END_STEP, CLEANUP)
- Active player and turn order
- Timestamp for ordering simultaneous events
- RNG seed and generator for reproducibility
- Delayed triggers (fire on specific future game events)
- Action history ring buffer (last 128 actions, used in ML observation)

Game loop in `src/main.cpp`:
1. State-based effects check (lethal damage, player death, permanent lifecycle, mandatory choices)
2. Mandatory choices (declare attackers, declare blockers, cleanup discard) handled via `proc_mandatory_choice`
3. Priority check and step advancement — if both players pass, advance step or resolve top of stack
4. Determine and display legal actions
5. Read player input via `InputLogger` (CLI, replay, or machine mode)
6. Execute chosen action via `process_action`

### Ability System

Ability categories resolved by `Ability::resolve()` in `src/components/ability.cpp`:
- `"AddMana"` — mana ability; handled at activation, never goes on the stack
- `"ChangeZone"` — zone search (e.g. fetch lands); prompts player to search, then moves card
- `"DealDamage"` — deals `amount` damage to `target` (player or creature)
- `"Destroy"` — moves `target` from battlefield to graveyard (checks target still on battlefield)
- `"Draw"` — draw cards
- `"Mill"` — put cards from library to graveyard
- `"Pump"` — add +X/+Y to a permanent's power/toughness
- `"Counter"` / `"PutCounter"` — put a counter on target permanent
- `"Token"` — create token permanents
- `"Attach"` — attach equipment or aura to target
- `"Untap"` — untap a permanent
- `"Phases"` — phase out target permanent
- `"Dig"` — look at top N cards, choose one matching filter; rest go to bottom
- `"Surveil"` / `"RearrangeTopOfLibrary"` / `"PeekAndReveal"` — look at and arrange top of library
- `"SylvanLibrary"` — draw 2, then choose: pay 4 life each or put on top
- `"DelayedTrigger"` — register ability to fire on a future game event
- `"ExaltedBonus"` / `"ProwessBonus"` — grant combat bonuses based on keyword count

Activated abilities with `valid_tgts != "N_A"` have their target selected before costs are paid and before the ability entity is pushed onto the stack. Target legality is re-verified at resolution.

**Name-a-card candidate set (deviation from CR 201.4).** "Name a card" effects (Cabal
Therapy's `SP$ NameCard`, Disruptor Flute's ETB, Petrified Hamlet's "name a land") do **not**
offer every card in existence as CR 201.4 allows. The shared builder
`build_name_card_choices()` (`src/name_card_choices.{h,cpp}`) returns a deliberately LIMITED,
context-driven candidate set derived from the match and the card — the distinct vocab cards
present in the relevant deck(s), filtered by the card's `ValidCards$` type. The `NameCardScope`
argument selects whose deck supplies the candidates: `CHOOSER_ONLY` (one player's whole deck,
used by Cabal Therapy / Disruptor Flute) or `BOTH_PLAYERS` (lands owned by **either** player A or
B, de-duped by name). Land-naming cards (Petrified Hamlet / Alpine Moon-style "name a land")
use `BOTH_PLAYERS`, so a land that exists only in the opponent's deck is still nameable.

### Card Parser (`src/parse.cpp`)

Parses `.txt` card scripts from `bin/resources/cardsfolder/`. Key script fields:

**Top-level card fields:**

| Field | Notes |
|---|---|
| `Name` | Card name |
| `ManaCost` | Mana cost string; `X` flag detected automatically |
| `Types` | Space-separated type line |
| `Oracle` | Oracle text; `\n` expanded to newlines |
| `PT` | Power/toughness as `P/T` |
| `A` | Activated or spell ability line (`AB$` / `SP$`) |
| `T` | Triggered ability line |
| `S` | Static ability or alternate cost line |
| `K` | Keyword list (Delve, Prowess, Equip, etc.) |
| `R` | Replacement effect line |
| `SVar` | Named variable substitution used in ability params |

**Ability parameter fields (within A/T lines):**

| Field | Notes |
|---|---|
| `AB$ <category>` | Activated ability; `Mana` normalized to `AddMana` |
| `SP$ <category>` | Spell ability |
| `Cost$` | Activation cost: `T` = tap, `PayLife<N>`, `Sac<qty/spec>`, `Return<qty/type>` |
| `Produced$` | Mana color for `AddMana`: `W/U/B/R/G/C`, `Any`, or `Combo W U G` |
| `ValidTgts$` | Target spec: `Any`, `Player`, `Creature`, `Land`, `Land.nonBasic`, combinations |
| `NumDmg$` | Damage amount for `DealDamage` |
| `ChangeType$` | Comma-separated subtypes to search (for `ChangeZone`) |
| `Origin$` | Source zone: `Library`, `Hand`, `Graveyard`, `Exile`, `Stack` |
| `Destination$` | Destination zone: `Battlefield`, `Library`, `Hand`, `Graveyard`, `Exile` |
| `Amount$` / `NumCards$` | Numeric amount or SVar reference |
| `TokenScript$` | Token name to create (for `Token` abilities) |
| `CounterType$` | Counter type string |
| `CounterNum$` | Counter count |
| `DigNum$` | Number of cards to look at (for `Dig`) |
| `Mandatory$` | `True` if the ability is mandatory |
| `SubAbility$` | SVar reference for chained sub-ability |
| `ActivationLimit$` | Max activations per turn |

Basic lands (Mountain, Forest, etc.) get their mana ability injected by `StateManager::apply_land_abilities` based on land subtypes, not from the script.

### Card Loading System

Cards are loaded on-demand from `bin/resources/cardsfolder/`:
- Card database (`src/card_db.h`): Maps card names to entity IDs, loads scripts on first access
- Parser (`src/parse.h` / `src/parse.cpp`): Parses card scripts into ECS entities with components
- Name to UID conversion: lowercase, spaces to underscores, other characters removed

**Double-faced cards are stored under ONE combined `<front>_<back>.txt` script — never
fetch/author a front-name-only duplicate.** Forge stores a DFC/MDFC/transform/Room card
under a single combined filename (e.g. `tamiyo_inquisitive_student_tamiyo_seasoned_scholar.txt`),
and the engine mirrors that: `load_card` (`src/card_db.cpp`) resolves the exact `<uid>.txt`
first and only falls back to scanning for a combined `<uid>_*.txt` when no exact file exists.
So writing a front-name `<uid>.txt` alongside an existing combined script adds the card to the
engine **twice** — the front-name file shadows the (correct) combined one. When provisioning
scripts, use the fetch tool below, which is DFC-aware; do not hand-create a `<front>.txt` for a
card whose combined script already exists.

**Fetching missing scripts — `tools/forge_fetch/fetch_script.py`** pulls card/token scripts from
the Card-Forge/forge repo (add-only; never overwrites without `--force`). Fetches are **pinned**
to the Forge commit in `tools/forge_fetch/FORGE_PIN`, so every clone fetches byte-identical
scripts to what the engine was tested against — an upstream master rewrite (this broke
`subability_roundtrip` once, when Forge rewrote Cloak and Dagger, Entwined) can only land
through a deliberate bump of that file; its header holds the bump recipe (diff the vocab
cards' scripts old-ref vs new-ref, adapt the engine, re-run `make check`). Pass `--ref master`
to preview live Forge without unpinning. It is the single
correct way to provision a missing script:
- Pass card names: `train/.venv/bin/python tools/forge_fetch/fetch_script.py "Brainstorm" "Tamiyo, Inquisitive Student"`.
  It treats a card as already-present if EITHER `<uid>.txt` OR a verified combined `<uid>_*.txt`
  exists, and on a front-name miss it discovers the combined DFC filename Forge serves (via the
  GitHub contents API) and fetches THAT under the combined name — so DFCs are never duplicated.
- Token scripts: add `--token` and pass the script stem (e.g. `--token b_0_0_orc_army`).
- Known gap: accented names aren't transliterated for the **filename/uid** — `name_to_uid`
  mirrors the C++ engine, which drops non-ASCII bytes rather than mapping them to a base letter
  (an accented "o" becomes nothing, not `o`), so it won't match Forge's transliterated filename;
  fetch/name such a card by its ASCII stem by hand. (The ML **vocab** match is separately
  accent-insensitive — `card_name_to_index` folds via `ascii_fold_card_name`, transliterating
  e.g. ó→o — so an accented `CardData::name` still resolves to its ASCII vocab entry.)
- The SessionStart hook (`.claude/hooks/session-start.sh`) calls this tool for the top-level,
  `meta/`, and `league/` decks so a fresh clone gets their card **and** token scripts
  automatically. The token pass scans each fetched card script (resolving combined DFC
  filenames) for `TokenScript$` stems and for the keyword/effect-synthesized tokens the engine
  hardcodes — Amass → `b_0_0_<subtype>_army`, Investigate → `c_a_clue_draw` (Clue), Mobilize →
  `r_1_1_warrior` (Warrior) — and fetches those too. After adding a card that creates a NEW
  engine-synthesized token kind (one with no `TokenScript$` field), add its stem to the hook's
  `keyword_tokens` map.

### Adding a New Card

When implementing a new card, **both** of the following steps are required:

1. Add the card to `src/card_vocab.h` — append a `{"Card Name", N}` entry where N is the next available index. `N_CARD_TYPES` in `src/machine_io.h` must be >= (highest index + 1).
2. Regenerate `train/card_costs.py` AND `train/card_props.py` — the cast-cost feature matrix
   used by the RL environment/extractor, and the frozen printed-property block of the
   network's card representation. A normal `make` does both for you (the `pygen` step
   regenerates ALL codegen on every build); run them by hand only to regenerate without a
   full build:
   ```
   train/.venv/bin/python train/gen_card_costs.py
   train/.venv/bin/python train/gen_card_props.py
   ```
   Both files (and the C++ mirror header `src/gen/card_costs_gen.h`) are **untracked** —
   generated from the fetched card scripts on every build — so there is **nothing to commit**
   for them; just make sure the card's script is provisioned so the regeneration is correct.
   (A new card only ADDS a property row — the column layout is a fixed constant in
   `gen_card_props.py`, so vocab growth never changes the network shape or invalidates
   checkpoints.)

**Parse script tags as intended — do not retag them.** When a card needs a mechanic the
engine lacks, implement the mechanic so the parser honors the script's actual tags
(`SP$`/`AB$`/`DB$` category, `Origin$`/`Destination$`/`ChangeType$`/`DefinedPlayer$`, etc.).
Do NOT rewrite one category into another or force a different Origin/Destination to shortcut
a single card's behavior — a retag that happens to satisfy one card silently corrupts every
other card that shares the tag. Add a real, general handler keyed on the tag's intended
meaning.

Follow-up: in limited cases it is acceptable to *ignore* an irrelevant tag when the card's
full functionality can already be inferred from the other tags (e.g. a cosmetic
`StackDescription$`/`TgtPrompt$`, or a `ChangeNum$` count-SVar when the effect already moves
all matching cards). Ignoring a tag is fine; repurposing a tag to mean something else is not.

### Deck Format

Deck files (`.dk`) in `bin/resources/decks/`:
```
<quantity> <card name>
...
SIDEBOARD:
<quantity> <card name>
```

## Key Globals

- `global_coordinator`: The ECS coordinator singleton
- `cur_game`: Current game state
- `RESOURCE_DIR`: Path to resources directory (set at runtime via `getcwd`)
- `card_db`: Card name to entity ID mapping

## Reinforcement Learning

The `train/` directory contains a Python gymnasium wrapper and PPO training script.

Python venv: `train/.venv/` — activate with `source train/.venv/bin/activate` or invoke directly via `train/.venv/bin/python`.

Dependencies: `gymnasium`, `stable-baselines3`, `sb3-contrib` (for `MaskablePPO` with action masking).

### Checkpoint naming: the one generalist (`gen`)

There is **one generalist model** that pilots *any* deck against *any* opponent —
not per-deck, not matchup-specific. It is stored under the fixed stem `gen`:
`checkpoints/gen__final.zip` (the current generalist) plus periodic
`gen__v{steps}.zip` snapshots (note the **double** underscore). The model
filename encodes **no deck** — so the deck a model pilots (and the opponent's
deck) must always travel as a **separate explicit parameter**. The self-play and
league pools, the `random-model` opponent-pool token, `play`, and `analysis` all
resolve the model as `gen` (`gen__final.zip`, else the newest `gen__v*.zip`) and
take the deck(s) separately. A bare *deck* stem (`delver`) is **no longer**
accepted as a checkpoint/model spec — the only model specs are `gen`, an explicit
`.zip`/`.pt` path, or the `az:gen`/`azraw:gen`/`mcts:gen` search wrappers. `gen`
is a reserved stem: a roster/league deck may not be named `gen`.

There is likewise **one generalist AZ net**, `checkpoints/az/gen__azfinal.pt`
(the gate-promoted incumbent) plus `gen__azv{steps}.pt` candidate snapshots.
It is always warm-started from the PPO `gen` checkpoint — AZ has no
from-scratch path — so build `gen__final.zip` (via `league`) before running
`az`/`az-train`.

Every training session **continues that one generalist**: each training
subcommand auto-resumes the existing `gen__final.zip` (or newest snapshot) and
accumulates the session's steps onto it, regardless of which deck/opponent this
session trains on. Pass `--fresh` to start the generalist over from scratch, or
`--load <path>` to resume a specific checkpoint. Because the filename carries no
deck, the deck must always be given explicitly (e.g. `baseline --deck`,
analysis's `--deck-a`/`--deck-b`).

### Training commands (run from repo root)

`train.py` uses subcommands (`train -h` for any subcommand's options). The
`train` subcommand is assumed when none is given, so one-liner training still
works without typing `train`.

**`league` is the primary way to train the PPO generalist** — it rotates the
roster and the PFSP snapshot pool for you. The single-matchup `train`/`sweep`
subcommands below are for scripting one deck-vs-deck session or inspecting a
checkpoint, not the main training loop.

```bash
# Training (the 'train' subcommand is implied when omitted)
train/.venv/bin/python train/train.py --deck delver --opponent mav                 # continue the generalist on delver vs mav
train/.venv/bin/python train/train.py --deck delver --opponent burn                # same gen__final.zip, now also on delver vs burn
train/.venv/bin/python train/train.py --deck delver --opponent mav --fresh         # start gen__final.zip from scratch
train/.venv/bin/python train/train.py train --deck delver --opponent mav --load checkpoints/gen__v500000.zip  # resume a specific snapshot
train/.venv/bin/python train/train.py --self-play --deck delver --opponent mav     # self-play vs the generalist's frozen snapshots

# Evaluation / inspection
# baseline: model vs scripted HARD, mirror decks. --deck is REQUIRED (the model
# encodes no deck); seats alternate per game, --seed reproducible.
train/.venv/bin/python train/train.py baseline gen --deck delver                      # win rate vs scripted:hard piloting delver
train/.venv/bin/python train/train.py baseline --all --games 100                      # round-robin the generalist (gen__final.zip): every roster deck vs every roster deck (scripted:hard), per-matchup + per-deck win rates; appends checkpoints/baseline_report.log (--log to override)
train/.venv/bin/python train/train.py baseline az:gen --all --games 50                # same NxN round-robin for ANY model spec — az:gen / azraw:gen / mcts:gen / an explicit .zip/.pt (e.g. an exp_* exploiter) — so the AZ net and exploiters get the same per-matchup report as the PPO gen
# observe: one command for any {scripted|model} vs {scripted|model} matchup
# (replaced the old diag/watch commands). --games N for a multi-game pass + summary,
# --verbose for the full per-decision transcript, --seed for reproducibility.
# observe runs bo3 MATCHES BY DEFAULT; --bo1 for single games.
train/.venv/bin/python train/train.py observe --player-a gen --player-b scripted --deck delver --opponent mav  # watch one match (per-side controller + deck)
train/.venv/bin/python train/train.py observe --deck delver --opponent mav                          # scripted vs scripted, one bo3 match (compact)
train/.venv/bin/python train/train.py observe --deck delver --opponent mav --games 10               # verify env: 10 matches + W/L/D summary
train/.venv/bin/python train/train.py observe --deck delver --opponent mav --verbose                # full transcript (state + action menu + narrative)
train/.venv/bin/python train/train.py observe --deck delver --opponent mav --games 10 --bo1         # 10 single (bo1) games

# Bulk training (was --train-all / --train-deck)
train/.venv/bin/python train/train.py sweep                                           # all deck×deck matchups
train/.venv/bin/python train/train.py sweep --deck delver                             # matchups featuring delver

# PFSP league (rotating learner over a shared snapshot pool)
train/.venv/bin/python train/train.py league                                          # train the decks/league/ roster
train/.venv/bin/python train/train.py league --resume                                 # resume an interrupted league run

# Archetype exploiters (AlphaStar-style main exploiters feeding the league pool)
train/.venv/bin/python train/train.py exploiter --archetype burn                      # pilot the burn decks vs the FROZEN gen; saves exp_burn__*.zip
train/.venv/bin/python train/train.py exploiter --archetype burn --resume             # resume that archetype's run

# AlphaZero (MCTS-trained net, warm-started from the PPO gen checkpoint — run league first)
train/.venv/bin/python train/train.py az --deck delver                                # one cycle: self-play -> train -> gate
train/.venv/bin/python train/train.py az-league                                       # rotate AZ cycles across decks/league/
train/.venv/bin/python train/train.py az-league --matrix --expert-decks league/wubg_doomsday  # whole-roster focus matrix + scripted:hard expert (BC) shards for the combo deck each slot
train/.venv/bin/python train/train.py az-league --exhaustive --rotations 2 --gate-every 0 --window 0 --workers 28  # EXACT matchup matrix each slot (one bo3 match vs scripted:hard per ordered deck pair + one self-play match per unordered pair = 155 on the 10-deck roster; with the C++ actor built the WHOLE matrix runs on it — vs-scripted cells ship the scripted seat's decisions to train/scripted_oracle.py — else pure Python), auto 2x shard window (0 = twice this slot's new shards, so the previous pass stays in-window), ungated (--gate-every 0: no gates; final candidate promoted to gen__azfinal unconditionally at completion)
train/.venv/bin/python train/train.py az-league --exhaustive-selfplay --exhaustive-repeats 2 --gate-every 0 --window 0 --workers 28  # same EXACT matrix narrowed to the PURE SELF-PLAY cells only (one bo3 match per UNORDERED deck pair, mirrors included = 55 on the 10-deck roster; NO vs-scripted cells, so the whole slot runs on the C++ actor with no Python fallback), --exhaustive-repeats N plays every cell N times per slot (2 = every self-play matchup twice). Both flags apply to `train.py az` too, and are persisted in the az-league resume sidecar. Plan file: curricula/az_selfplay_matrix.plan.json

# Curriculum: a multi-phase plan (league -> exploiter -> league -> az-league -> baseline) in one file
train/.venv/bin/python train/train.py curriculum --plan q3 --dry-run                  # print each phase's composed command
train/.venv/bin/python train/train.py curriculum --plan q3                            # run it (each phase is a subprocess)
train/.venv/bin/python train/train.py curriculum --plan q3 --status                   # per-phase state from the progress file
train/.venv/bin/python train/train.py curriculum --plan q3 --resume                   # continue after an interruption
```

**Curricula.** A curriculum is an ordered list of phases whose `kind` is a `train.py`
subcommand (`league`, `exploiter`, `az`, `az-league`, `baseline`), written as
`train/checkpoints/curricula/<name>.plan.json` (`"version": 1`) and executed by
`train.py curriculum` — one SUBPROCESS per phase, so PPO and AZ phases stay memory-isolated
and each phase reuses its own `--resume` machinery. A phase carries a few convenience fields
(`decks`, `steps`, `archetype`, `rotations`, `games`, `model`, `deck`) plus an `overrides`
object whose keys are that subcommand's **`cli_spec` arg dests** — the spec IS the schema, so
an unknown kind/override or a wrong value type fails loudly at load instead of at phase
launch. Progress lives in `<name>.progress.json` next to the plan; `--resume` skips completed
phases and relaunches the one that was in flight (with `--resume` of its own for the drivers
that support it), refusing to run if an already-executed phase was edited while allowing free
edits to the phases still ahead. The TUI's `curriculum` entry opens a plan builder screen
(phase list + per-phase form generated from that kind's `cli_spec` Sub) with Save/Load/Run/
Resume/Status. See `train/curriculum.py`.

**League resume.** A `league` run persists its driver progress (roster, total budget,
global steps done, current rotation, and every hyperparameter) to
`train/checkpoints/_league_progress.json`, rewritten every time a snapshot checkpoint is
saved and at each rotation boundary. If the session is interrupted, restart it with just
`train.py league --resume` (in the TUI, tick the league form's **--resume** checkbox) — the
sidecar restores the full run configuration and the loop continues from where it left off,
re-entering a partially-trained rotation for only its remainder. All other
flags are ignored when `--resume` is set; the one generalist's weights resume from
`gen__final.zip` / newest `gen__v*.zip` as usual.

**League opponent sampling (`LeaguePool`, `train/opponents.py`).** Each episode the pool
makes two coupled choices — which opponent *deck* and which *controller* pilots it — from
three branches: a small **latest-self** slot (`--self-play-frac`, default **0.2** — the
anti-collapse "keep up with your own frontier" mirror), a **scripted anchor** floor
(`--scripted-anchor-frac`, default 0.1, also the cold-start fallback), and, for the
majority of episodes, a **PFSP-weighted historical snapshot** where each snapshot's weight
is `(1 - learner_winrate_vs_it)^p` (`--pfsp-p`, default 2). That weighting is what
concentrates training on the learner's **worst** matchups — a 0%-win opponent gets weight
~1.0, a ~50% one ~0.25 — so keeping the fixed mirror slot **small** (not the old 0.8) hands
the bulk of games to the hard matchups rather than drowning them in ~50% mirror games. It is
self-tuning: as win-rates shift, PFSP reallocates automatically. To bias even harder toward
losing matchups, lower `--self-play-frac` (e.g. 0.1) or raise `--pfsp-p`.

**Active-pool composition (bounded but fair).** Pool entries are `(opponent_deck,
gen_snapshot)` pairs — the one generalist piloting each roster deck. The pool is capped to
`max(1, floor(opponent_ckpt_ratio * n_envs))` unique checkpoints (sharded across env
processes) to bound memory, but it is **not** a raw recency slice — that starved decks early
in the roster order. Instead it keeps, per deck, a **guaranteed** anchor (`gen__final` piloting
that deck, or the newest `gen` snapshot if no `__final` yet) that is never evicted, plus
**discretionary** `gen__v*` intermediates filled newest-first round-robin across decks. So every
roster deck is always represented as an opponent — even a perennial-loser deck stays in the pool
via the generalist's `__final` piloting it.

**Archetype exploiters (`train.py exploiter --archetype <arch>`).** A dedicated run whose
learner pilots ONE archetype's decks (every deck tagged with that archetype in
`bin/resources/decks/archetypes.json`) against a **frozen** opponent pool pinned to the
current `gen__final.zip` piloting the whole league roster — no snapshot rotation, no
latest-self mirror, so the target never moves. It saves under its own checkpoint stem
`exp_<archetype>` (`exp_burn__v{steps}.zip` / `exp_burn__final.zip`) and **never writes a
`gen` file**; `exp_*`, like `gen`, is a reserved stem no deck may be named. Weights
warm-start from `gen` (fresh step counter) unless `--fresh`; progress goes to
`checkpoints/_exploiter_<archetype>_progress.json` so `exploiter --archetype <arch> --resume`
continues an interrupted run. Later `league` runs then pick the exploiters up
automatically: each archetype's newest exploiter is a never-evicted pool entry piloting that
archetype's roster decks, older `exp_*__v*` compete for the discretionary budget, and
`--exploiter-floor` (default 0.1) reserves a minimum share of episodes for them on top of
their normal PFSP weight — so the generalist gets inoculated against burn/combo/control
styles without having to discover them itself.

**Snapshot promotion gate (`--promote-margin`, default 0.05).** The periodic
`gen__v{steps}.zip` snapshots are only saved when the learner's *recent-window* win-rate
clears `0.5 + margin` (first snapshot per deck exempt; `0` disables the gate; a negative
margin gates below 50%). This keeps the pool from filling with near-duplicate weak
intermediates. It gates **only** `gen__v*` snapshots — `gen__final` is always saved
unconditionally at rotation end (and always kept in the pool per the composition rule above),
so raising the margin never removes a deck from the field; it only thins the version history.

### Best-of-three mode

`--bo3` flag (C++ and Python) runs a best-of-three match in a single process:
- Player A goes first in game 1; loser goes first in subsequent games
- Each player keeps their own deck and seat across all games of the match
- Between games both players can sideboard (swap cards between main deck and sideboard)
- Match ends when either player reaches 2 wins

**C++ output protocol (machine mode):**
- `GAME_RESULT: N Player A wins` / `GAME_RESULT: N Player B wins` — after each game
- `MATCH_RESULT: Player A wins X-Y` — terminal signal for the match

**Reward structure (from Player A perspective) — PER GAME:**
- Individual game win/loss: **+1.0 / -1.0** (the primary signal; in bo3 it lands at every
  `GAME_RESULT`, so an episode's return is the discounted sum of the match's game results)
- Match win/loss: **0.0 / 0.0** — the separate match-terminal reward is retired. `MATCH_RESULT`
  still ends the episode; the constants (`MATCH_WIN_REWARD`/`MATCH_LOSS_REWARD`) and their
  plumbing stay in `train/env.py` so a match bonus can be dialed back in from one place.
- Why per-game: it is exactly the AlphaZero outcome target (`az_selfplay` prices every sample by
  the winner of the game it was played in), so a PPO checkpoint warm-starting an AZ net
  (`az_net.from_ppo`) hands over a critic already calibrated in AZ's units. (AZ additionally
  records an **n-step TD target** `td_q` per sample and trains on
  `(1 - q_mix) * z + q_mix * td_q` — knobs `--td-n` / `--q-mix`; see the "n-step TD value
  targets" section of [`docs/alphazero_status.md`](docs/alphazero_status.md). `z` is unchanged
  and stays the anchor.)
- Shaping is budgeted **per game** against that ±1.0 (`SHAPING_EPISODE_CAP` in `train/env.py`);
  bo1 and bo3 games are shaped identically (no bo3 /3 division).

**Bo3-relevant state-vector fields** (exact indices/normalizers live in the `src/machine_io.h`
layout block — don't hardcode them here):
- **Match context** — `game_number`, `self/opp_match_wins`, `is_sideboard_phase` (all 0.0 in single-game mode).
- **Library & post-board context** — `self/opp_library_ct`, `is_post_board` (1.0 in game 2+ of a bo3).
- **Known top-5 library cards** for the viewer — set as cards are placed on top (Ponder/Brainstorm/Rearrange/Sylvan), cleared to unknown on shuffle, and slid up when a tracked card is drawn.
- **Opponent revealed-cards multi-hot** — binary "has the opponent-of-viewer ever revealed card X this match", set when an opponent card enters a public zone (battlefield/stack/graveyard/exile) or is revealed by a tutor. It's the engine's deterministic "belief state" (a feedforward policy can't remember reveals across `reset()`), accumulated across a bo3's games and persisted over the per-game ECS reset. Tracked in `src/classes/match_state.{h,cpp}`.

### Machine mode protocol

`--machine` flag makes the game communicate over stdio for RL training:
- Game emits a `BQUERY` line on stdout at each decision point, followed by a binary payload
- Driver writes a single integer back on stdin
- All non-BQUERY stdout lines are game narrative and can be ignored

**BQUERY format:**
```
BQUERY: <N> <STATE_SIZE> <MAX_ACTIONS>\n
[float32 × STATE_SIZE  — state vector]
[int32   × MAX_ACTIONS — action categories (padded)]
[float32 × MAX_ACTIONS — action card IDs (padded)]
[float32 × MAX_ACTIONS — action controller_is_self flags (padded)]
[float32 × MAX_ACTIONS — action card_is_public flags (padded)]
[int32   × MAX_ACTIONS — action zone_ref (ActionRefZone; padded)]
[int32   × MAX_ACTIONS — action slot_ref (entity-reference slot, -1 = none; padded)]
[int32   × MAX_ACTIONS — action option_ordinal (mode/X/color/ability index, -1 = n/a; padded)]
```
- `N` = number of legal choices; the trailing `STATE_SIZE`/`MAX_ACTIONS` are a runtime
  layout handshake (the Python driver asserts them against its own imported constants so a
  C++ layout change without regenerated Python constants fails loudly instead of misframing
  the payload)
- State vector: `STATE_SIZE` floats, serialized from the **priority player's** ("self") perspective
- Action categories: `ActionCategory` enum integers
- Card IDs: `card_vocab_index / N_CARD_TYPES`, or `-1.0 / N_CARD_TYPES` as null sentinel
- Controller flags: `1.0` = self-controlled, `0.0` = opponent, null sentinel for non-entity actions
- Public flags: `1.0` if the choice's card identity is public knowledge to all players (a revealed tutor, e.g. Personal Tutor), else `0.0`. Lets observers show the card name for an otherwise-private choice (search / top-of-library). Consumed as a side-channel (`env._action_public`); **not** part of the gym observation vector, so `OBS_SIZE` and trained checkpoints are unaffected.

**Source of truth (do not duplicate the values here — they drift):**
- **State vector layout, `STATE_SIZE`, `N_CARD_TYPES`, per-slot field order** — the commented
  layout block and constants in `src/machine_io.h`. Card identity is a single normalized id
  float per slot (`norm_card_id`), *not* a one-hot.
- **Every block WIDTH** (player block, step one-hot, match/library context, pending decision,
  global extras, history entry, decklist slot, the stack sub-fields) — the "State-vector block
  widths" block in `src/machine_io.h`, mirrored into `train/_enums.py` by `gen_enums.py`.
  Never re-spell a width as a literal in `env.py`/`extractor.py`/`obs_builder.cpp`; import it.
- **Every block's absolute OFFSET** — `machine_io.h`'s `OFFSET_CHAIN`, pinned by its own
  `== STATE_SIZE` static_assert. `env.py` derives the same chain from the mirrored widths and
  `ci_check.py`'s `actorobs` tier proves the two equal block-by-block; `extractor.py` asserts
  itself against `env.py` at import. A block-by-block check, not a total-only one: a
  compensating change (a float moved between adjacent blocks) keeps the total and would
  otherwise misalign every field in between silently.
- **Per-action metadata block positions in the obs** — `env.py`'s `ACT_CATS_START` /
  `ACT_IDS_START` / `ACT_CTRL_START` / `ACT_ZONE_START` / `ACT_REFS_START` / `ACT_ORDS_START`.
  Slice those, never a hand-counted `STATE_SIZE + k * MAX_ACTIONS`.
- **`ActionCategory` values and meanings** — the enum in `src/classes/action.h` (mirrored to
  Python by codegen in `train/_enums.py`; `ACTION_CATEGORY_MAX` is generated from it).
- **`OBS_SIZE` and the observation composition** (`STATE_SIZE + N_ACTION_OBS_BLOCKS*MAX_ACTIONS`
  metadata + hand/battlefield cost features + the matchup tail) — `train/env.py`.
  `N_ACTION_OBS_BLOCKS` (`src/machine_io.h`) is the ONE source of truth for how many
  per-action arrays are folded into the obs; `pub` is emitted in the BQUERY but stays a
  side-channel and is not counted.

**Confirm slot convention:** mandatory attacker/blocker queries end with a confirm action. The Python env remaps `action = num_choices - 1` to `-1` before sending to the game.

**Concession sentinels (CR 104.3a):** besides an action index, the engine accepts
`-2` (`CONCEDE_GAME`) and `-3` (`CONCEDE_MATCH`) wherever it reads a decision — machine
mode, `--replay`, the in-process actor provider, and the CLI prompt. The seat the pending
decision belongs to loses the game (`-3` also ends the MATCH immediately, with
`MATCH_RESULT` naming the opponent whatever the score). The loss runs through the ordinary
`Game::player_loses` path, so `GAME_RESULT`, loser-on-the-play, sideboarding and the RL
reward are unchanged; the sentinel is written to the RMLOG decision log, so a concession
replays. Between games (the sideboard phase) there is no live game: `-2` is ignored safely,
`-3` still ends the match. Python constants live in `train/env.py` (`CONCEDE_GAME` /
`CONCEDE_MATCH`, stepped through `RoboMageEnv.step` without the confirm-slot remap);
`GameDriver.concede(match=…)` injects one for the human seat. Regression:
`train/test_concede.py` (`make check` tier `concede`).

**Two perspective flags** worth knowing without opening the source: in the state vector, one flag marks whether the priority player is the active player (perspective-relative) and another marks whether "self" is Player A (absolute); AND-ing their agreement recovers `active_is_a`. See `src/machine_io.h` for their exact indices.

### Interactive front ends: TUI, GUI, and the analysis window

`./tui.sh` (`train/tui.py`) is the overall Textual control panel — deck management, training,
league runs, observing games, and launching interactive play — and is a separate tool from the
two *game board* front ends described below. `./gui.sh` is a thin wrapper that launches
`train/gui_main.py` with no arguments (the app shell on its welcome pane; sessions start
via File ▸ New Session — no dialog is auto-opened).

Both game boards share one front-end-agnostic loop — `train/game_driver.py` (`GameDriver` on a
worker thread reporting `StateUpdate`s to a sink; `build_session` assembles env + opponent
controller). The engine is always a `--machine` subprocess; the opponent controller is any
`opponents.make_controller` spec (scripted tiers, `gen`, `az:`/`azraw:`/`mcts:` wrappers).

- **TUI board**: `train/play.py --human-deck X --model-deck Y` (default), or via `./tui.sh`.
- **GUI board** (PySide6): `play.py ... --gui`, or `python train/gui_game.py` / `./gui.sh` with
  no args for the app shell's welcome pane — File ▸ New Session opens the play/analysis dialogs
  (deck/opponent/seat/format pickers + search and analysis settings, persisted to
  `~/.robomage/gui_launcher.json`). Falls back to the TUI if PySide6 is missing.
- **Standalone analysis browser** (`train/tui_analysis.py`, Textual, not on `game_driver.py`):
  loads a checkpoint, simulates N games against a chosen opponent, and lets you page through
  each game's board states, seek via a clickable V(s) histogram, run any `analysis.py` REPL
  view, and branch `whatif` counterfactuals. Launch via `./tui.sh`'s `analysis-tui → browse`
  menu entry, or directly: `train/tui_analysis.py <model.zip|deck> --opponent scripted
  [--deck-b mav] [--n-games 20]`.
- **Headless smokes** (no display; used for sanity checks): `QT_QPA_PLATFORM=offscreen
  ROBOMAGE_GUI_SMOKE=N` auto-plays N decisions and exits 0; add `ROBOMAGE_ANALYSIS_SMOKE=1`
  to force the analysis window on (uniform evaluator, torch-free) and fail unless a live
  analysis run delivered stats.

**The analysis window** (`train/gui_analysis.py`, GUI only; enable via the launcher checkbox or
`play.py --gui --analysis`; F9 toggles, F5 analyzes, F6 reviews the opponent's last decision,
Shift+F5 stops): live MCTS evaluation of
the current decision, chess-engine style. It runs `mcts.IncrementalSearch` — a chunked,
cancellable, resumable search bit-identical to `run_search` for the same world seeds, which
holds its root snapshot open so `pv()`/`walk()` can browse afterwards — on a **detached
analysis engine** (`SearchRoboMageEnv.spawn_detached_mirror`): a caller-owned engine copy that
is *never* registered in the live env's mirror pool and is kept in lockstep lazily by replaying
the primary's action-history delta (so the live game never blocks, and the analysis engine may
lag and only ever replays forward — a request for an EARLIER decision, i.e. the F6 review below,
is served by respawning the engine at that history prefix). Displays: per-action table (prior / visits / visit% / Q as
win%), a branch value chart (bold mean per top branch + thin per-world lines for the
determinization spread, always in the human's perspective), and a PV scrubber that walks a
branch's principal variation on the analysis engine and renders each hypothetical future board
(`MiniBoard`, reusing `CardWidget`). Evaluator is selectable (default `az:gen` — AZ checkpoint
else PPO warm-start via `opponents.load_az_evaluator`; also `mcts:gen`, `uniform`).
Constraints: since the snapshot-safe conversion (branch `snapshot_safe`) every decision kind is
loop-safe and a legal search root — the greyed-out state remains only for the documented
residual `safe=0` prompts (see docs/alphazero_status.md's safe-window section); opponent-decision analysis
sits behind an explicit reveal toggle (it exposes hidden information), where a search opponent's
own per-decision `SearchResult` is surfaced for free via the `SearchController.on_result` hook
and other opponents are analyzed retrospectively on the analysis engine. **Opponent's last
decision** (the F6 button, reveal toggle required) reviews a move already made: the analysis
engine rewinds to just before the opponent's most recent real decision, searches it from their
perspective, and marks the action they actually played `▶` (read from the live env's action
history); the live game keeps running and the engine catches up by forward replay at the next
analysis. The Qt-free session
core lives in `train/analysis_session.py`; regression tests in `train/test_analysis_session.py`
run as the **opt-in** `ci_check.py` tier `analysis` (not part of default `make check`):
`train/.venv/bin/python train/ci_check.py --tier analysis`.

**Shard recording of GUI play** (`train/shard_record.py`; the play launcher's "Record shards"
checkbox — search opponents only — or `play.py --gui --record-shards`): a play session is
recorded into **trainer-schema shards** (`az_selfplay.SHARD_KEYS`, one dir per session under
`train/az_data/recorded/rec_*`; `ROBOMAGE_RECORD_DIR` overrides the base). A search opponent's
searched decisions land with their full visit posterior / root value / explored flag (the
`SearchController.on_result` tap, CHAINED with the analysis window's sink); every other
>1-choice decision — the human's included — lands as a one-hot behavior row (`q=NaN`, the
`generate_expert` precedent) via the driver's `step_observer` hook, so both seats are
browsable. One shard file per match, atomically rewritten at each game boundary and on demand,
so the dir is always valid for every existing shard consumer (`az_train.load_window`,
`az_inspect --shards`, `tui_az_inspect --shards`, `tui_analysis --shards`, the GUI browser's
shard mode) — rows of the game still in progress carry `z=0` until it finishes. Each shard
also gets a same-stem **`.rmplay` replay sidecar** (engine seed + the match's full action
log + per-row replay positions, `gui_session_io.save_replay` format), which
`shard_replay.load_replay_sidecars` attaches to the browsable records — so a recording is
**exactly replayable**: both analysis browsers gained a `search` menu entry (**F6**,
`browse_session.run_replay_search` / `EngineCore.search_step`) that replays the recording to
the selected decision on a throwaway `SearchRoboMageEnv` (recorded seed + action prefix,
obs verified against the recorded state) and runs a determinized MCTS there from the mover's
perspective — the live analysis window's F6 review, offline. Works without a live env/model
(shard-browse and traces-only modes included); training-pool shards and pre-sidecar
recordings self-report as non-replayable. **View ▸
Analyze Recording… (F10)** flushes mid-game and opens the recording in a shard-mode
`BrowserPane` in a SECOND window (the live game keeps running); viewpoint seat defaults to the
opponent so you page the model's decisions, and the analysis dialog's seat toggle shows your
own. The browser (all modes) also has **net-probe menu entries** (`train/shard_probes.py`,
Qt-free glue over `az_inspect`'s probes with a lazily loaded AZ net, PPO warm-start fallback):
per-decision recorded-π-vs-net, obs-block permutation importance, card-identity swap and scalar
sweeps at the selected step, plus pooled KL(search‖net) and value calibration against realized
z. Recorder regression: `train/test_shard_record.py`, wired into default `make check` as the
`shardrec` tier; the opt-in `gui` tier adds a record-smoke leg.

### Key files

- `train/env.py` — `RoboMageEnv` gymnasium wrapper; `ModelVsScriptedEnv` scripted-opponent wrapper; `SelfPlayEnv` self-play wrapper. Lazily re-exports `scripted_action` for back-compat callers; the real rule-based agent logic lives in `train/scripted_agent.py`.
- `train/scripted_agent.py` — the rule-based `ScriptedAgent`/`scripted_action` implementation (smart mulligan, combat simulation, evaluation-based targeting, deck-specific combo lines); imported by `opponents.py`, `train.py`, `bench_engine.py`, `analysis.py`, and re-exported from `env.py`
- `train/runner.py` — THE game-running module: `drive_game` (the single decision loop, with per-decision hooks), `run_games` (env-per-game orchestration + transcripts), `run_match` (spec-based front door for scripting: agents/decks/bo3/seed/output as parameters). See `docs/game_running.md`.
- `train/opponents.py` — the `Controller` agent abstraction and `make_controller` spec grammar (scripted tiers, model checkpoints via the shared `resolve_checkpoint`, `play:`/`actions:` scripts, `human`, `auto`), plus the training opponent pools
- `train/extractor.py` — `CardGameExtractor` per-entity feature extractor for the policy network
- `train/train.py` — `MaskablePPO` training, baseline evaluation, observe mode, self-play
- `train/curriculum.py` — multi-phase training plans behind `train.py curriculum`: the plan
  schema (validated against `cli_spec`, which IS the schema), the composed per-phase argv, the
  progress sidecar + resume/hash rules, and the subprocess runner. Stdlib-only, so `tui.py`'s
  plan-builder screen imports it without torch. Regression: `train/test_curriculum.py`
  (`make check` tier `curriculum`)
- `train/progress_io.py` — the single crash-safe (write-temp + `os.replace`) JSON progress
  sidecar reader/writer shared by the league, exploiter, az-league, and curriculum drivers
- `train/analysis.py` — model-analysis tool: loads a checkpoint, simulates games for a matchup, and inspects play (card importance, SHAP, value swings, regret, entropy, calibration, an interactive REPL). Charts save to PNG under `train/analysis_out/` (headless-safe; `--show` for a GUI window) with terminal sparkline/bar fallbacks. The model is the one generalist (`gen`, an explicit `.zip`/`.pt` path, or `az:gen`/`azraw:gen`) and encodes **no deck**, so both the model's deck and the opponent's deck must be given explicitly with `--deck-a`/`--deck-b` for any model seat (a scripted opponent defaults to a mirror of `--deck-a`). (The older offline `.rmrec` recording subsystem and `train.py --record` were removed.)
- `train/viz.py` — headless-friendly chart helpers for analysis.py (Agg-by-default matplotlib save-or-show, plus terminal sparklines and diverging bars)
- `train/play.py` — interactive human-vs-model play (text mode, `--tui`, `--gui`, `--analysis`)
- `train/game_driver.py` — front-end-agnostic play loop + `build_session`; `StateUpdate` carries
  an obs COPY plus `search_safe`/`history_len` for the analysis window
- `train/tui_game.py` / `train/gui_game.py` — the Textual and PySide6 boards over that driver
- `train/tui.py` — the overall Textual control panel behind `./tui.sh` (deck management,
  training, league runs, observing games, launching play — composes `train.py`/`analysis.py`/
  `play.py` as subprocesses, distinct from the `game_driver.py` boards). `ArgFormMixin` builds
  every input form from `cli_spec`; the `curriculum` entry pushes `CurriculumScreen`, the
  multi-phase plan builder (phase list + per-phase form + Save/Load/Run/Resume/Status)
- `train/gui_analysis.py` — the analysis window (worker thread, live MCTS table, branch chart,
  PV scrubber + MiniBoard)
- `train/shard_record.py` — `ShardRecorder`: records GUI play into trainer-schema shards
  (searched rows via `SearchController.on_result`, one-hot rows via `GameDriver.step_observer`,
  per-game mover-perspective z backfill through `az_selfplay._backfill_and_pack`, atomic
  per-match rewrites safe to read mid-game). Regression `train/test_shard_record.py`
  (`make check` tier `shardrec`)
- `train/shard_probes.py` — Qt-free glue running `az_inspect`'s net probes over browsed
  records (snapshot on the UI thread, stack+torch on the worker); `PROBE_MENU` is appended to
  the GUI browser's analyses sidebar
- `train/tui_analysis.py` — standalone Textual analysis browser (game list, board-state pager,
  clickable V(s) histogram, every `analysis.py` REPL view, `whatif` branching); behind
  `./tui.sh`'s `analysis-tui → browse` menu entry
- `train/analysis_session.py` — Qt-free analysis core: `AnalysisSession` (detached engine,
  delta-replay lockstep, chunked analyze/pv/walk), `AnalysisConfig`, `load_analysis_evaluator`
- `train/search_env.py` — `SearchRoboMageEnv` (snapshot protocol client, mirror pool,
  `spawn_detached_mirror`)
- `train/mcts.py` — determinized PUCT search: `run_search`/`run_search_parallel` (now also
  reporting per-action `q`, `w_sum`, per-world `world_values`) and `IncrementalSearch`.
  **Duplicate-edge merging is ON by default** (`merge_dupes=True`): interchangeable duplicate
  menu actions (same-owner hand picks, library search picks, graveyard/exile picks — see
  `decode.menu_merge_reps`/`MENU_MERGE_WHITELIST`) collapse to one search edge, with the C++
  twin in `src/actor/menu_merge.h`; the two predicates MUST change in lockstep (the actor
  parity gate asserts bit-identical visits). See the "Duplicate-edge merging" section of
  `docs/alphazero_status.md`
- `train/test_analysis_session.py` — analysis-core regression (opt-in ci tier `analysis`)
- `train/az_inspect.py` — **static** inspection of an AZ checkpoint, never from a played game.
  Most views need only the WEIGHTS: card-embedding neighbours / label purity / clusters / PCA
  (the embedding's rows line up with `src/card_vocab.h`), the per-matchup value-head column
  map, checkpoint diffs, and **`exposure`** — which embedding rows and critic columns actually
  received gradient, read exactly from a checkpoint pair (an untrained row is bit-identical,
  since `az_train` exempts these tables from weight decay). The rest additionally read the
  recorded self-play shards (`az_data/gen/*.npz`): per-card occurrence counts, per-bucket
  calibration against realized outcomes, KL(search‖net) by action category, and the
  per-decision probes (block permutation importance, card-identity swap, scalar sweeps).
  Each view is a data function plus a `render_*` returning display lines, so the CLI and the
  TUI cannot diverge
- `train/tui_az_inspect.py` — Textual front end over those views (Embedding / Critic panes,
  clickable embedding drill-down); `./tui.sh`'s `az-inspect → inspect` menu entry. **Opens
  weights-only** (~1s, no shard pool needed) and lists only the views it can compute;
  `--with-shards` loads recorded self-play and adds the Probes pane
- `train/test_az_inspect.py` — inspector regression against a fresh net + synthetic shards
  (opt-in ci tier `azinspect`; needs torch, no engine binary)
- `train/gen_card_costs.py` — regenerates `train/card_costs.py` from `src/card_vocab.h`
- `train/gen_card_props.py` — regenerates `train/card_props.py`, the FROZEN printed-property
  block (96 fixed columns: pips/CMC/X, colors, types, land subtypes, P/T, printed keywords,
  ability-category heads, tribal subtypes) of the network's card representation. The
  extractor consumes `[card_emb | card_props]` per card id — the trainable 32-dim identity
  embedding carries only the behavioral residual the frozen printed facts can't express.
  DFC-face aware (a back-face vocab entry parses its own face; Delver does not get Flying)
  and token-band aware (token rows parse `bin/resources/tokenscripts/`)
- `train/test_harness.py` — LLM test harness for card behavior verification (see Testing guidelines)
- `train/fuzz_campaign.py` — batch fuzzing driver for the league fuzz campaigns: runs N scripted
  games for ONE matchup (both seats driven by the coverage `explore` fuzzer, independent per-seat
  novelty state) with per-game seed increments, and dumps the full verbose transcript to a file for
  bug/rules-deviation review. One invocation per (matchup, mode); the transcript file is then
  reviewed. `--mode explore` (default, broad coverage) or `--mode explore:patient` (big-mana profile)
  are the two campaign modes. Example: `train/.venv/bin/python train/fuzz_campaign.py --deck-a
  league/ur_delver --deck-b league/gw_maverick --mode explore --games 100 --seed 1 --out out.txt`
  (decks resolve relative to `bin/resources/decks/`; W/L/D summary to stdout, any draw is a finding).
- `train/action_spec.py` — shared semantic-action resolver: turns a `--play` spec string (`cast:Lightning Bolt`, `target:X@opp`, `pass`, …) into the matching legal action index against the current decision's decoded menu. Used by `PlayController` (test harness `--play`, `observe --play-a/--play-b`) and by `HumanController` (play.py text mode / `run_match(..., "human")`) for typed semantic input.
- `train/card_costs.py` — auto-generated cast-cost and ability-cost matrices (do not edit manually)
- `train/card_props.py` — auto-generated frozen card-property matrix (do not edit manually)
- `train/test_obs_invariants.py` — standalone regression script asserting per-decision structural
  invariants on the RAW machine-mode observation state vector across a few seeded scripted games
  (delver vs maverick, exile-heavy bw_dnt vs delver, and a Yorion companion probe). Checks: every
  card-id-family float (perm card/chosen-name/returnable ids, stack + target ids, GY/exile/hand/
  known-top/known-opp-hand slots, pending-decision, history) decodes to `-1` or `[0, N_CARD_TYPES)`;
  every `norm_ref` float round-trips into `[-1, 107]`; GY/exile blocks are recency-packed with no
  holes; a returnable-exile id implies that card is in an exile block; `chosen_name` only on
  non-empty slots; the step one-hot sums to 1 and the mandatory-choice one-hot to ≤1; player
  life/hand/library counts are finite and non-negative; a declared companion is revealed to the
  opponent for the whole game proper; the mana-development block is finite/non-negative with each
  per-color potential ≤ `potential_total`, `potential_total` ≥ the floating pool it includes,
  `land_drops_remaining` inside the normalizer's range, `lands_in_hand` ≤ hand size, and
  `lands_in_play` equal to the lands counted over that side's permanent slots (`is_land` &
  `!is_phased_out`) in the SAME observation; and each **log-scaled vital** equal to
  `log1p(max(v, 0)) / log1p(normalizer)` for the life/library count recovered from that same
  observation's LINEAR float (masked to all-zeros in the sideboard phase, asserted there
  instead — both branches counted so neither passes vacuously). Every offset/constant is imported from `env`/`_enums` (zero
  magic numbers, so it is layout-change-proof). **Wired into `ci_check.py` as the `obsinv` tier, so
  `make check` runs it**; also runnable standalone: `train/.venv/bin/python train/test_obs_invariants.py`
- `src/machine_io.h` — state vector layout documentation and constants
- `src/input_logger.cpp` — machine mode BQUERY emission, replay, and CLI input handling
- `src/card_vocab.h` — card name → vocab index mapping for one-hot encoding
