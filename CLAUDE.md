# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code Style

- Don't put new functions in main.cpp
- Don't edit my comments for spelling or punctuation. Only change them if something substantive changed.
- Do not include in comments explanations of how things were previously prior to a change. Document only how code currently behaves.
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
[`docs/mtg_comprehensive_rules.txt`](docs/mtg_comprehensive_rules.txt), with a navigation guide in
[`docs/mtg_comprehensive_rules.md`](docs/mtg_comprehensive_rules.md).

**Whenever you implement or test a mechanic, consult these rules** — they are ground truth for how
it *should* behave, independent of the engine's current state. Don't read the 9k-line file end to
end: `grep` the numbered rule (e.g. `grep -nE "^509\." docs/mtg_comprehensive_rules.txt` for
combat; 702 keywords, 704 state-based actions, 613 layers). See the navigation guide for recipes.

## Build Commands

- **Build the project**: `make && make actor`
- **Clean build artifacts**: `make clean`
- **Build for release**: `make BUILD=RELEASE && make actor BUILD=RELEASE`

The compiled binary is output to `bin/<config>/robomage` — `bin/debug/robomage` from a plain
`make`, `bin/release/robomage` from `make BUILD=RELEASE` (separate object/output trees per
config, so switching `BUILD` never requires `make clean`).

**Which binary the Python tooling runs is a separate, independent choice from which one you
just built** — every `train/` entry point resolves its engine binary through `ROBOMAGE_BUILD`
(`train/cli_spec.py`), defaulting to a build tier per tool:

- **Debug by default** (`bin/debug/robomage`) — the test harness, `ci_check`/`make check`,
  `observe`, `baseline`: correctness tools, so the extra assertions (`-D_GLIBCXX_ASSERTIONS`, no
  `-O2`) are worth the slowdown.
- **Release by default** (`bin/release/robomage`) — the GUI, the standalone TUI analysis browser,
  and every PPO/AZ training driver.

**Override**: `ROBOMAGE_BUILD=debug|release` forces every tool onto one config (e.g. to reproduce
a debug-only assertion failure); each subcommand also takes `--binary <path>` as a per-invocation
escape hatch.

**Codegen is part of the build.** The default `make` target runs `pygen` first, regenerating
ALL auto-generated files unconditionally every build (write-if-changed via `train/gen_util.py`,
so unchanged outputs force no recompile). Two derive from tracked sources and are **committed**:
`train/_enums.py` (from C++ headers) and `src/gen/archetypes_gen.h` (from `decks/archetypes.json`).
The other three derive from card-script CONTENT (ManaCost/keywords/types) — and scripts are
gitignored/Forge-fetched, not expressible as Make prerequisites — so they are **UNTRACKED** and
rebuilt locally every build: `train/card_costs.py`, `train/card_props.py`, and the C++ mirror
`src/gen/card_costs_gen.h`. Adding/removing a vocab entry and running `make` keeps them in sync
with nothing to commit. The `pygen` CI tier only guards the two committed outputs.

## Long-running commands (Claude)

When Claude runs a long command (build, training, gate, fuzz, bench) it must run it **in the
background** and make the live output inspectable by the user:

- Always state the task's output-file path in chat when launching it, so the user can
  `tail -f` it while it runs. (No symlinks or other indirection — just the path.)

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
 `--library-a/-b`, `--battlefield/graveyard/exile/sideboard-a/-b` on commas, so a comma inside a
 card name breaks the parse. **Fix: omit the comma** — `name_to_uid` strips punctuation, so
 `Thalia Guardian of Thraben` loads the same card. In `--play` specs use a comma-free unique
 substring (`cast:thalia`); a stacked `temp/` deck file (one card per line) also works.
-train.py observe is helpful for checking new builds (it replaced the old diag/watch
 commands — one command observes any {scripted|model} vs {scripted|model} matchup).
 Use `--games N` for a multi-game regression pass (per-game results + W/L/D summary),
 `--verbose` for the full per-decision transcript (board state + action menu + narrative),
 and supply `--deck`/`--opponent` to test cards/decks relevant to recently implemented features.
 **observe defaults to bo3 matches**; pass `--bo1` for single games (`--bo3` is a redundant no-op).
-**Running games programmatically: `runner.run_match`.** `runner.run_match(agent_a, agent_b,
 deck_a=…, deck_b=…, games=, bo3=True, seed=1, transcript="compact|verbose|narrative|quiet",
 out=…)` — agent specs are scripted tiers ("scripted"/"hard", "easy", "random", "explore"),
 "human", "auto", "play:<specs>", "actions:<ints>", the generalist model ("gen", "az:gen", …),
 or a checkpoint path. All game-running tools (harness, observe, baseline, play.py, fuzz,
 ci_check, analysis, bench) sit on the same `runner.drive_game` loop — do not hand-roll new
 decision loops; see [`docs/game_running.md`](docs/game_running.md).

### Test harness for card behavior verification

`train/test_harness.py` runs the engine with `--machine --narrative`, with game narrative visible alongside decoded binary state.

**Shuffling:** by default each library is shuffled with the seeded RNG (deterministic per `--seed`). Pass `--no-shuffle` for a **stacked deck** so deck-file order = draw order (first 7 = starting hand). `--no-shuffle` is implied by `--hand-a`/`--hand-b` (they build a stacked temp deck), so inline-hand and scenario examples stay deck-ordered; a plain `--deck-a X --deck-b Y` run shuffles unless you add `--no-shuffle`.

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
- `--interactive` — prompt a human per decision. **Claude cannot use this mode** (no TTY; the
  `input()` prompt hits EOF and spins) — use `--play` or `--actions` instead.
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

**Seat keys (`A:` / `B:`) — the easy way to drive both seats.** Prefix a spec with `A:` or `B:`
(case-insensitive) to pin it to a seat, so you don't hand-sequence the priority hand-offs. Before
each spec the harness checks who holds priority: **if the next spec is keyed to the seat lacking
priority, the priority holder auto-passes** (spec unconsumed) until the keyed seat is on the clock.
A keyed spec also **auto-passes the keyed seat forward through its own priority windows until the
action is legal** — e.g. `A:attack:Voice of Victory` written in A's main phase keeps passing until
declare-attackers. (Forward-advance fires only on a genuinely-not-yet-legal action while a `pass`
is available; an ambiguous/misspelled spec, or one hitting a mandatory choice it doesn't match,
fails loudly with the legal menu.) The seat key is independent of the `@own`/`@opp` target suffix.
Unkeyed specs apply to whoever has priority and are **not** auto-advanced (must match or fail),
so existing scripts are unchanged and keyed/unkeyed may be mixed. (Under `observe --play-a/-b`
each list already drives one seat, so leave specs unkeyed there.)

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
- A bare `scripted` spec defaults to the **hard** (heuristic) tier everywhere (harness,
  league/self-play anchor, opponent pools, observe/baseline/analysis): smart mulligan, combat
  simulation, evaluation-based targeting, deck-specific combo lines (Doomsday), on top of the
  greedy baseline. Request the old greedy behaviour with `scripted:easy`/`scripted:greedy` (or
  `scripted:random`).
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

**Name-a-card candidate set (deviation from CR 201.4).** "Name a card" effects (Cabal Therapy,
Disruptor Flute, Petrified Hamlet) do **not** offer every card. `build_name_card_choices()`
(`src/name_card_choices.{h,cpp}`) returns a LIMITED set — the distinct vocab cards in the
relevant deck(s), filtered by `ValidCards$`. `NameCardScope` selects the source: `CHOOSER_ONLY`
(one player's whole deck) or `BOTH_PLAYERS` (lands owned by either player, de-duped), so a
land-naming card can still name a land only in the opponent's deck.

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

**Double-faced cards live under ONE combined `<front>_<back>.txt` script — never author a
front-name-only duplicate.** Forge uses a single combined filename (e.g.
`tamiyo_inquisitive_student_tamiyo_seasoned_scholar.txt`); `load_card` (`src/card_db.cpp`)
resolves exact `<uid>.txt` first, else scans for a combined `<uid>_*.txt`. A front-name
`<uid>.txt` alongside a combined script adds the card **twice** (shadowing the combined one).
Use the DFC-aware fetch tool below; don't hand-create a `<front>.txt`.

**Fetching missing scripts — `tools/forge_fetch/fetch_script.py`** pulls card/token scripts from
Card-Forge/forge (add-only; no overwrite without `--force`), **pinned** to the commit in
`tools/forge_fetch/FORGE_PIN` so every clone fetches byte-identical scripts. Bumping the pin can
break tests (Forge once rewrote Cloak and Dagger); the FORGE_PIN header holds the bump recipe.
`--ref master` previews live Forge without unpinning. It is the single correct way to provision
a script:
- Pass card names: `fetch_script.py "Brainstorm" "Tamiyo, Inquisitive Student"`. Treats a card as
  present if `<uid>.txt` OR a combined `<uid>_*.txt` exists; on a front-name miss it discovers and
  fetches the combined DFC filename (GitHub contents API), so DFCs are never duplicated.
- Token scripts: `--token` plus the stem (e.g. `--token b_0_0_orc_army`).
- Known gap: accented names aren't transliterated for the **filename/uid** (`name_to_uid` drops
  non-ASCII bytes like the C++ engine) — fetch such a card by its ASCII stem by hand. (The ML
  **vocab** match IS accent-insensitive via `ascii_fold_card_name`, so `CardData::name` still
  resolves to its vocab entry.)
- The SessionStart hook (`.claude/hooks/session-start.sh`) auto-fetches card + token scripts for
  the top-level, `meta/`, and `league/` decks. Its token pass scans for `TokenScript$` stems and
  the engine's synthesized tokens (Amass → `b_0_0_<subtype>_army`, Investigate → `c_a_clue_draw`,
  Mobilize → `r_1_1_warrior`). After adding a card that makes a NEW synthesized token kind, add
  its stem to the hook's `keyword_tokens` map.

### Adding a New Card

When implementing a new card, **both** of the following steps are required:

1. Add the card to `src/card_vocab.h` — append a `{"Card Name", N}` entry where N is the next available index. `N_CARD_TYPES` in `src/machine_io.h` must be >= (highest index + 1).
2. Regenerate `train/card_costs.py` and `train/card_props.py` (the cast-cost matrix and frozen
   printed-property block). A normal `make` does both via `pygen`; regenerate without a build via
   `train/.venv/bin/python train/gen_card_costs.py` and `.../gen_card_props.py`. Both (and the
   C++ mirror `src/gen/card_costs_gen.h`) are **untracked** — rebuilt from scripts every build,
   nothing to commit; just provision the card's script first. A new card only ADDS a property
   row (fixed column layout), so vocab growth never changes the network shape or invalidates
   checkpoints.

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

**C++ output protocol (machine mode):**
- `GAME_RESULT: N Player A wins` / `GAME_RESULT: N Player B wins` — after each game
- `MATCH_RESULT: Player A wins X-Y` — terminal signal for the match

**Reward structure (from Player A perspective) — PER GAME:**
- Individual game win/loss: **+1.0 / -1.0** (the primary signal; in bo3 it lands at every
  `GAME_RESULT`, so an episode's return is the discounted sum of the match's game results)
- Match win/loss: **0.0 / 0.0** — the match-terminal reward is retired; `MATCH_RESULT` still ends
  the episode, and `MATCH_WIN_REWARD`/`MATCH_LOSS_REWARD` stay in `train/env.py` to dial a match
  bonus back in from one place.
- Why per-game: it is exactly the AlphaZero outcome target, so a PPO checkpoint warm-starting an
  AZ net (`az_net.from_ppo`) hands over a critic calibrated in AZ's units. AZ additionally records
  an **n-step TD target** `td_q` and trains on `(1 - q_mix) * z + q_mix * td_q` (knobs `--td-n` /
  `--q-mix`; `z` stays the anchor — see [`docs/alphazero_status.md`](docs/alphazero_status.md)).
- Shaping is budgeted **per game** against ±1.0 (`SHAPING_EPISODE_CAP` in `train/env.py`); bo1 and
  bo3 shaped identically.

**Bo3-relevant state-vector fields** (exact indices/normalizers live in the `src/machine_io.h`
layout block — don't hardcode them here):
- **Match context** — `game_number`, `self/opp_match_wins`, `is_sideboard_phase` (all 0.0 in single-game mode).
- **Library & post-board context** — `self/opp_library_ct`, `is_post_board` (1.0 in game 2+ of a bo3).
- **Known top-5 library cards** — set as cards are placed on top (Ponder/Brainstorm/etc.), cleared on shuffle, slid up when a tracked card is drawn.
- **Opponent revealed-cards multi-hot** — "has the opponent ever revealed card X this match", set when an opponent card enters a public zone or is revealed by a tutor. The engine's deterministic belief state (a feedforward policy can't remember reveals across `reset()`), persisted over the per-game ECS reset. Tracked in `src/classes/match_state.{h,cpp}`.

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
- **State vector layout, `STATE_SIZE`, `N_CARD_TYPES`, per-slot field order** — the layout block
  and constants in `src/machine_io.h`. Card identity is one normalized id float (`norm_card_id`),
  *not* a one-hot.
- **Every block WIDTH** — the "State-vector block widths" block in `src/machine_io.h`, mirrored to
  `train/_enums.py` by `gen_enums.py`. Never re-spell a width as a literal in
  `env.py`/`extractor.py`/`obs_builder.cpp`; import it.
- **Every block's absolute OFFSET** — `machine_io.h`'s `OFFSET_CHAIN` (static_assert `== STATE_SIZE`).
  `env.py` derives the same chain; `ci_check.py`'s `actorobs` tier proves them equal **block-by-block**
  (a total-only check would miss a float moved between adjacent blocks); `extractor.py` asserts vs
  `env.py` at import.
- **Per-action metadata positions in the obs** — `env.py`'s `ACT_CATS_START`/`ACT_IDS_START`/
  `ACT_CTRL_START`/`ACT_ZONE_START`/`ACT_REFS_START`/`ACT_ORDS_START`; slice those, never a
  hand-counted `STATE_SIZE + k * MAX_ACTIONS`.
- **`ActionCategory` values** — the enum in `src/classes/action.h` (mirrored to `train/_enums.py`;
  `ACTION_CATEGORY_MAX` generated from it).
- **`OBS_SIZE` and obs composition** (`STATE_SIZE + N_ACTION_OBS_BLOCKS*MAX_ACTIONS` + cost
  features + matchup tail) — `train/env.py`. `N_ACTION_OBS_BLOCKS` (`src/machine_io.h`) is the sole
  source for how many per-action arrays fold in; `pub` is a side-channel, not counted.

**Confirm slot convention:** mandatory attacker/blocker queries end with a confirm action. The Python env remaps `action = num_choices - 1` to `-1` before sending to the game.

**Concession sentinels (CR 104.3a):** the engine accepts `-2` (`CONCEDE_GAME`) and `-3`
(`CONCEDE_MATCH`) wherever it reads a decision (machine mode, `--replay`, actor provider, CLI).
The pending-decision seat loses the game (`-3` also ends the match immediately). Loss runs the
ordinary `Game::player_loses` path (so `GAME_RESULT`, loser-on-the-play, sideboarding, RL reward
are unchanged) and is written to the RMLOG so it replays. In the sideboard phase `-2` is ignored,
`-3` still ends the match. Python constants in `train/env.py`; `GameDriver.concede(match=…)`
injects one for the human seat. Regression `train/test_concede.py` (`make check` tier `concede`).

**Two perspective flags** worth knowing without opening the source: in the state vector, one flag marks whether the priority player is the active player (perspective-relative) and another marks whether "self" is Player A (absolute); AND-ing their agreement recovers `active_is_a`. See `src/machine_io.h` for their exact indices.

### Interactive front ends: TUI, GUI, and the analysis window

`./tui.sh` (`train/tui.py`) is the overall Textual control panel (deck management, training,
league runs, observing, launching play) — separate from the two *game board* front ends below.
`./gui.sh` launches `train/gui_main.py` on its welcome pane (sessions start via File ▸ New Session).

Both game boards share one front-end-agnostic loop — `train/game_driver.py` (`GameDriver` on a
worker thread reporting `StateUpdate`s to a sink; `build_session` assembles env + opponent
controller). The engine is always a `--machine` subprocess; the opponent is any
`opponents.make_controller` spec (scripted tiers, `gen`, `az:`/`azraw:`/`mcts:` wrappers).

- **TUI board**: `train/play.py --human-deck X --model-deck Y` (default), or via `./tui.sh`.
- **GUI board** (PySide6): `play.py ... --gui`, or `python train/gui_game.py` / `./gui.sh` with
  no args for the app shell's welcome pane — File ▸ New Session opens the play/analysis dialogs
  (deck/opponent/seat/format pickers + search and analysis settings, persisted to
  `~/.robomage/gui_launcher.json`). Falls back to the TUI if PySide6 is missing.
- **Standalone analysis browser** (`train/tui_analysis.py`, Textual, not on `game_driver.py`):
  simulates N games vs an opponent and lets you page board states, seek via a clickable V(s)
  histogram, run any `analysis.py` REPL view, and branch `whatif` counterfactuals. Launch via
  `./tui.sh`'s `analysis-tui → browse`, or `train/tui_analysis.py <model.zip|deck> --opponent
  scripted [--deck-b mav] [--n-games 20]`.
- **Headless smokes**: `QT_QPA_PLATFORM=offscreen ROBOMAGE_GUI_SMOKE=N` auto-plays N decisions and
  exits 0; add `ROBOMAGE_ANALYSIS_SMOKE=1` to force the analysis window on and fail unless it
  delivered stats.

**The analysis window** (`train/gui_analysis.py`, GUI only; enable via launcher checkbox or
`play.py --gui --analysis`; F9 toggle, F5 analyze, F6 review opponent's last decision, Shift+F5
stop): live chess-engine-style MCTS on the current decision. Runs `mcts.IncrementalSearch`
(chunked, cancellable, bit-identical to `run_search` for the same world seeds; holds its root
snapshot open for `pv()`/`walk()`) on a **detached analysis engine**
(`SearchRoboMageEnv.spawn_detached_mirror`) kept in lockstep by forward-replaying the primary's
action-history delta, so the live game never blocks. Displays a per-action table (prior / visits
/ visit% / Q as win%), a per-world branch value chart (human's perspective), and a PV scrubber
rendering hypothetical future boards (`MiniBoard`). Evaluator selectable (default `az:gen` — AZ
else PPO warm-start via `opponents.load_az_evaluator`; also `mcts:gen`, `uniform`). Every
decision kind is now a legal search root (residual `safe=0` prompts stay greyed — see
docs/alphazero_status.md). Opponent-decision analysis sits behind a reveal toggle (exposes hidden
info): a search opponent's own `SearchResult` comes via `SearchController.on_result`; F6 rewinds
the engine to just before the opponent's last real decision, searches it from their perspective,
and marks the played action `▶`. Qt-free core in `train/analysis_session.py`; regression
`train/test_analysis_session.py` is the **opt-in** `ci_check.py --tier analysis` (not in default
`make check`).

**Shard recording of GUI play** (`train/shard_record.py`; launcher's "Record shards" checkbox —
search opponents only — or `play.py --gui --record-shards`): records a session into
**trainer-schema shards** (`az_selfplay.SHARD_KEYS`, one dir per session under
`train/az_data/recorded/rec_*`; `ROBOMAGE_RECORD_DIR` overrides). A search opponent's decisions
land with full visit posterior / root value / explored flag (via `SearchController.on_result`,
chained with the analysis window's sink); every other >1-choice decision (the human's included)
lands as a one-hot behavior row (`q=NaN`) via the driver's `step_observer`. One file per match,
atomically rewritten at each game boundary, so every shard consumer (`az_train.load_window`,
`az_inspect`/`tui_az_inspect`/`tui_analysis --shards`, the GUI browser) always sees a valid dir
(in-progress rows carry `z=0`). Each shard gets a same-stem **`.rmplay` replay sidecar** (seed +
full action log + per-row positions) attached by `shard_replay.load_replay_sidecars`, making a
recording **exactly replayable**: both browsers' `search` entry (**F6**,
`browse_session.run_replay_search`) replays to the selected decision on a throwaway
`SearchRoboMageEnv` and runs determinized MCTS from the mover's perspective — the F6 review,
offline. Works without a live env/model; pre-sidecar and training-pool shards self-report as
non-replayable. **View ▸ Analyze Recording… (F10)** opens the recording in a shard-mode
`BrowserPane` in a second window (live game keeps running), viewpoint defaulting to the opponent.
The browser also has **net-probe entries** (`train/shard_probes.py`, Qt-free glue over
`az_inspect`'s probes): recorded-π-vs-net, block permutation importance, card-swap/scalar sweeps,
pooled KL(search‖net), value calibration. Regression `train/test_shard_record.py` = default
`make check` tier `shardrec`; opt-in `gui` tier adds a record-smoke leg.

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
- `train/analysis.py` — model-analysis tool: loads a checkpoint, simulates a matchup, inspects play (card importance, SHAP, value swings, regret, entropy, calibration, a REPL). Charts save to PNG under `train/analysis_out/` (headless-safe; `--show` for a window) with terminal fallbacks. The model (`gen`, a `.zip`/`.pt` path, or `az:gen`/`azraw:gen`) encodes **no deck**, so `--deck-a`/`--deck-b` are required for any model seat (scripted opponent mirrors `--deck-a`).
- `train/viz.py` — headless-friendly chart helpers for analysis.py (Agg-by-default matplotlib save-or-show, plus terminal sparklines and diverging bars)
- `train/play.py` — interactive human-vs-model play (text mode, `--tui`, `--gui`, `--analysis`)
- `train/game_driver.py` — front-end-agnostic play loop + `build_session`; `StateUpdate` carries
  an obs COPY plus `search_safe`/`history_len` for the analysis window
- `train/tui_game.py` / `train/gui_game.py` — the Textual and PySide6 boards over that driver
- `train/tui.py` — the Textual control panel behind `./tui.sh` (deck management, training, league,
  observe, launching play — composes `train.py`/`analysis.py`/`play.py` as subprocesses).
  `ArgFormMixin` builds every form from `cli_spec`; the `curriculum` entry opens the multi-phase
  plan builder.
- `train/gui_analysis.py` — the analysis window (worker thread, live MCTS table, branch chart,
  PV scrubber + MiniBoard)
- `train/shard_record.py` — `ShardRecorder`: records GUI play into trainer-schema shards (searched
  rows via `SearchController.on_result`, one-hot rows via `GameDriver.step_observer`, per-game z
  backfill, atomic per-match rewrites safe to read mid-game). Regression `train/test_shard_record.py`
  (`make check` tier `shardrec`).
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
- `train/mcts.py` — determinized PUCT search: `run_search`/`run_search_parallel` (report
  per-action `q`, `w_sum`, per-world `world_values`) and `IncrementalSearch`. **Duplicate-edge
  merging ON by default** (`merge_dupes=True`): interchangeable menu actions (same-owner hand /
  library-search / GY-exile picks, see `decode.menu_merge_reps`) collapse to one edge, mirrored by
  the C++ twin `src/actor/menu_merge.h` — the two predicates MUST change in lockstep (parity gate
  asserts bit-identical visits). See docs/alphazero_status.md.
- `train/test_analysis_session.py` — analysis-core regression (opt-in ci tier `analysis`)
- `train/az_inspect.py` — **static** inspection of an AZ checkpoint (never a played game).
  Weights-only views: card-embedding neighbours / purity / clusters / PCA (rows align with
  `src/card_vocab.h`), per-matchup value-head column map, checkpoint diffs, and **`exposure`**
  (which embedding rows / critic columns received gradient). Shard-backed views (`az_data/gen/*.npz`):
  per-card counts, calibration vs realized outcomes, KL(search‖net) by category, per-decision
  probes (permutation importance, card-swap, scalar sweeps). Each view = data fn + `render_*`, so
  CLI and TUI can't diverge.
- `train/tui_az_inspect.py` — Textual front end over those views (Embedding / Critic panes,
  clickable embedding drill-down); `./tui.sh`'s `az-inspect → inspect` menu entry. **Opens
  weights-only** (~1s, no shard pool needed) and lists only the views it can compute;
  `--with-shards` loads recorded self-play and adds the Probes pane
- `train/test_az_inspect.py` — inspector regression against a fresh net + synthetic shards
  (opt-in ci tier `azinspect`; needs torch, no engine binary)
- `train/gen_card_costs.py` — regenerates `train/card_costs.py` from `src/card_vocab.h`
- `train/gen_card_props.py` — regenerates `train/card_props.py`, the FROZEN printed-property block
  (96 fixed columns: pips/CMC/X, colors, types, land subtypes, P/T, keywords, ability-category
  heads, tribal subtypes). The extractor consumes `[card_emb | card_props]` per card id — the
  trainable 32-dim embedding carries only the behavioral residual. DFC-face aware and token-band
  aware (token rows parse `bin/resources/tokenscripts/`).
- `train/test_harness.py` — LLM test harness for card behavior verification (see Testing guidelines)
- `train/fuzz_campaign.py` — batch fuzz driver: runs N scripted games for ONE matchup (both seats
  driven by the `explore` fuzzer), dumping the verbose transcript to a file for bug review. Modes
  `--mode explore` (default) / `explore:patient` (big-mana). Example: `fuzz_campaign.py --deck-a
  league/ur_delver --deck-b league/gw_maverick --mode explore --games 100 --seed 1 --out out.txt`
  (decks relative to `bin/resources/decks/`; W/L/D to stdout, any draw is a finding).
- `train/action_spec.py` — shared semantic-action resolver: turns a `--play` spec string (`cast:Lightning Bolt`, `target:X@opp`, `pass`, …) into the matching legal action index against the current decision's decoded menu. Used by `PlayController` (test harness `--play`, `observe --play-a/--play-b`) and by `HumanController` (play.py text mode / `run_match(..., "human")`) for typed semantic input.
- `train/card_costs.py` — auto-generated cast-cost and ability-cost matrices (do not edit manually)
- `train/card_props.py` — auto-generated frozen card-property matrix (do not edit manually)
- `train/test_obs_invariants.py` — asserts per-decision structural invariants on the RAW
  machine-mode state vector across seeded scripted games (delver/maverick, bw_dnt/delver, a Yorion
  probe): card-id floats decode to `-1` or `[0, N_CARD_TYPES)`; `norm_ref` floats round-trip into
  `[-1, 107]`; GY/exile blocks recency-packed with no holes; one-hots sum correctly; the
  mana-development block consistent (per-color ≤ `potential_total`, `lands_in_play` = counted land
  slots, etc.); each log-scaled vital = `log1p(v)/log1p(normalizer)` vs the same obs's linear
  float. Every offset imported from `env`/`_enums` (layout-change-proof). Default `make check` tier
  `obsinv`; standalone `train/.venv/bin/python train/test_obs_invariants.py`.
- `src/machine_io.h` — state vector layout documentation and constants
- `src/input_logger.cpp` — machine mode BQUERY emission, replay, and CLI input handling
- `src/card_vocab.h` — card name → vocab index mapping for one-hot encoding
