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
- GUI is written in C99 with raylib
- Uses clang-format configuration in `.clang-format`
- DO NOT MODIFY CARD SCRIPTS

## Project Overview

Robomage is a C++ implementation of a Magic: The Gathering game engine using an Entity Component System (ECS) architecture.

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
- **Disable GUI**: `make HEADLESS=TRUE`

The compiled binary is output to `bin/robomage`.

**Codegen is part of the build.** The default `make` target runs `pygen` before compiling,
which regenerates the auto-generated Python files (`train/_enums.py`, `train/card_costs.py`)
whenever their declared C++ inputs change. `train/card_costs.py` is rebuilt from
`gen_card_costs.py` keyed on `src/card_vocab.h` and `src/machine_io.h`, so **adding/removing a
vocab entry and running `make` keeps the cast-cost matrix in sync automatically** — the manual
`gen_card_costs.py` invocation in "Adding a New Card" is only needed to regenerate without a
full build. (The generator reads each vocab card's `ManaCost` from its script; a card in the
vocab is assumed to have its script present, so the scripts themselves are not Make
prerequisites.)

**Run build/test commands plainly so they don't trigger a permission prompt.** A single
command, or a single pipeline whose programs are all allowlisted (`make`, the `train/...`
python entry points, `grep`/`head`/`tail`/`echo`), auto-approves. Avoid the shell plumbing
that forces a prompt: chaining statements with `;`/`&&`, and `${PIPESTATUS[0]}` / `$(...)`
expansions. Prefer bare `make` and `... | grep -i error` over
`make 2>&1 | grep ... ; echo "EXIT:${PIPESTATUS[0]}"`; read the printed output for status
instead of `$PIPESTATUS`.

## Testing guidelines
-Don't use sed or cat - if possible don't pipe a bunch of commands together in a way that will require asking my permission, run build and test tasks as simply as possible
-Non fatal errors are not acceptable
-Draws are not acceptable
-Do not attempt to test cards that are not already in `src/card_vocab.h`. Cards absent from the card vocab are considered unimplemented.
-Do not use commas when using the test harness. The harness splits `--play`, `--hand-a/-b`,
 `--library-a/-b`, and `--battlefield-a/-b` on commas, so a comma inside a card name (e.g.
 "Thalia, Guardian of Thraben", "Tamiyo, Inquisitive Student") breaks the parse. Refer to such
 cards by a comma-free unique substring (e.g. `cast:thalia`, `Tamiyo@own`) or use a stacked
 `temp/` deck file (one card per line, no commas) instead of inline comma-separated lists.
-train.py observe is helpful for checking new builds (it replaced the old diag/watch
 commands — one command observes any {scripted|model} vs {scripted|model} matchup).
 Use `--games N` for a multi-game regression pass (per-game results + W/L/D summary),
 `--verbose` for the full per-decision transcript (board state + action menu + narrative),
 and supply `--deck`/`--opponent` to test cards/decks relevant to recently implemented features.

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
- The scripted agent always keeps (no mulligan), casts spells when affordable, plays lands, and attacks with all creatures
- Temp deck files in `decks/temp/` are cleaned up automatically when using `--hand-a`/`--hand-b`; manually created files in `decks/temp/` are not

## Architecture

### ECS Pattern

The codebase follows an Entity Component System architecture based on [Austin Morlan's ECS tutorial](https://austinmorlan.com/posts/entity_component_system/):

- **Entities** (`src/ecs/entity.h`): Simple uint32_t IDs, maximum 5000 entities
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

### Adding a New Card

When implementing a new card, **both** of the following steps are required:

1. Add the card to `src/card_vocab.h` — append a `{"Card Name", N}` entry where N is the next available index. `N_CARD_TYPES` in `src/machine_io.h` must be >= (highest index + 1).
2. Regenerate `train/card_costs.py` — the cast-cost feature matrix used by the RL environment
   and extractor. A normal `make` does this for you (the `pygen` step regenerates it because
   `src/card_vocab.h` changed); run it by hand only to regenerate without a full build:
   ```
   train/.venv/bin/python train/gen_card_costs.py
   ```
   Either way, commit the regenerated `train/card_costs.py` alongside the vocab change.

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

### Training commands (run from repo root)

`train.py` uses subcommands (`train -h` for any subcommand's options). The
`train` subcommand is assumed when none is given, so one-liner training still
works without typing `train`.

```bash
# Training (the 'train' subcommand is implied when omitted)
train/.venv/bin/python train/train.py --deck delver --opponent mav                 # train from scratch
train/.venv/bin/python train/train.py train --opponent mav --load checkpoints/robomage_final.zip  # resume
train/.venv/bin/python train/train.py --self-play --deck delver --opponent mav     # self-play training

# Evaluation / inspection
train/.venv/bin/python train/train.py baseline checkpoints/robomage_final.zip         # win rate vs scripted
# observe: one command for any {scripted|model} vs {scripted|model} matchup
# (replaces the old diag/watch/observe). --games N for a multi-game pass + summary,
# --verbose for the full per-decision transcript, --seed for reproducibility, --bo3 for matches.
train/.venv/bin/python train/train.py observe --player-a checkpoints/robomage_final.zip --player-b scripted --deck delver --opponent mav  # watch one game (per-side controller + deck)
train/.venv/bin/python train/train.py observe --deck delver --opponent mav                          # scripted vs scripted, one game (compact)
train/.venv/bin/python train/train.py observe --deck delver --opponent mav --games 10               # verify env: 10 games + W/L/D summary
train/.venv/bin/python train/train.py observe --deck delver --opponent mav --verbose                # full transcript (state + action menu + narrative)
train/.venv/bin/python train/train.py observe --deck delver --opponent mav --games 10 --bo3         # verify bo3 env (10 matches)

# Bulk training (was --train-all / --train-deck)
train/.venv/bin/python train/train.py sweep                                           # all deck×deck matchups
train/.venv/bin/python train/train.py sweep --deck delver                             # matchups featuring delver
```

### Best-of-three mode

`--bo3` flag (C++ and Python) runs a best-of-three match in a single process:
- Player A goes first in game 1; loser goes first in subsequent games
- Each player keeps their own deck and seat across all games of the match
- Between games both players can sideboard (swap cards between main deck and sideboard)
- Match ends when either player reaches 2 wins

**C++ output protocol (machine mode):**
- `GAME_RESULT: N Player A wins` / `GAME_RESULT: N Player B wins` — after each game
- `MATCH_RESULT: Player A wins X-Y` — terminal signal for the match

**Reward structure (from Player A perspective):**
- Individual game win/loss: +0.3 / -0.3 (intermediate)
- Match win/loss: +1.0 / -1.0 (terminal)

**State vector match context** (indices 33018-33021, all 0.0 in single-game mode):
- `game_number / 3.0`, `self_match_wins / 2.0`, `opp_match_wins / 2.0`, `is_sideboard_phase`

**State vector library & post-board context** (indices 33022-33024):
- `self_library_ct / 60.0`, `opp_library_ct / 60.0`, `is_post_board` (1.0 if game 2+ of bo3)

**Current turn** (index 33025): `turn / 50.0`

**Known top-of-library** (indices 33026-33665): 5 × 128 card one-hots for the
top 5 cards of the viewer's library. All-zeros slot = unknown. Populated as
cards are placed on top (Ponder/Brainstorm/Rearrange/Sylvan) and cleared to
unknown whenever the library is shuffled. Also slides up when a card is drawn
from within the tracked top-5 window.

**Opponent revealed-cards multi-hot** (indices 33666-33793): 128-float binary
multi-hot of every card the opponent-of-viewer has revealed so far this match.
Bit `i` = 1 once the opponent has shown card vocab index `i`. Set when an
opponent card enters a public zone (battlefield/stack/graveyard/exile) or is
revealed by a tutor; accumulated across the games of a bo3 and persisting over
the per-game ECS reset (the engine's deterministic "belief state", since a
feedforward policy cannot remember reveals across `reset()`). Empty (all zeros)
at game-1 turn-1. Tracked in `src/classes/match_state.{h,cpp}`.

### Machine mode protocol

`--machine` flag makes the game communicate over stdio for RL training:
- Game emits a `BQUERY` line on stdout at each decision point, followed by a binary payload
- Driver writes a single integer back on stdin
- All non-BQUERY stdout lines are game narrative and can be ignored

**BQUERY format:**
```
BQUERY: <N>\n
[float32 × STATE_SIZE  — state vector]
[int32   × MAX_ACTIONS — action categories (padded)]
[float32 × MAX_ACTIONS — action card IDs (padded)]
[float32 × MAX_ACTIONS — action controller_is_self flags (padded)]
[float32 × MAX_ACTIONS — action card_is_public flags (padded)]
```
- `N` = number of legal choices
- State vector: 33794 floats (see `src/machine_io.h` for layout)
- Action categories: ActionCategory enum integers (0–44)
- Card IDs: `card_vocab_index / N_CARD_TYPES`, or `-1.0 / N_CARD_TYPES` (-0.0078125) as null sentinel
- Controller flags: `1.0` = self-controlled, `0.0` = opponent, null sentinel for non-entity actions
- Public flags: `1.0` if the choice's card identity is public knowledge to all players (a revealed tutor, e.g. Personal Tutor), else `0.0`. Lets observers show the card name for an otherwise-private choice (search / top-of-library). Consumed as a side-channel (`env._action_public`); **not** part of the gym observation vector yet, so `OBS_SIZE` and trained checkpoints are unaffected.

**ActionCategory values** (emitted per legal action):

| Value | Name | Meaning |
|---|---|---|
| 0 | PASS_PRIORITY | Pass priority |
| 2 | SELECT_ATTACKER | Choose a creature to attack with |
| 3 | CONFIRM_ATTACKERS | Confirm attacker declaration (sent as -1) |
| 4 | SELECT_BLOCKER | Choose a creature to block with |
| 5 | CONFIRM_BLOCKERS | Confirm blocker declaration (sent as -1) |
| 6 | ACTIVATE_ABILITY | Activate a non-mana ability |
| 7 | CAST_SPELL | Cast a spell from hand |
| 8 | SELECT_TARGET | Choose a target for a spell/ability |
| 9 | PLAY_LAND | Play a land from hand |
| 10 | OTHER_CHOICE | Generic/unclassified choice — fallback default for any decision not given a specific category below |
| 11 | MULLIGAN | Keep (0) or mulligan (1) |
| 12 | BOTTOM_DECK_CARD | Choose card to put on library bottom (post-mulligan) |
| 13–18 | MANA_W/U/B/R/G/C | Tap a land for the corresponding color |
| 19 | SEARCH_LIBRARY | Choose card from library search (0 = fail to find) |
| 20 | TOP_LIBRARY | Choose card to put on top of library |
| 21 | SHUFFLE | Choose whether to shuffle |
| 22 | PAYING_COSTS | Pay an optional cost |
| 23 | DIG_CHOICE | Choose creature/land from top N cards (e.g. Once Upon a Time) |
| 24 | SIDEBOARD_IN | Choose a card from sideboard to add to main deck (bo3) |
| 25 | SIDEBOARD_OUT | Choose a card from main deck to move to sideboard (bo3) |
| 26 | SIDEBOARD_DONE | Finish sideboarding (bo3) |
| 27 | SACRIFICE_PERMANENT | Choose a permanent to sacrifice (cost or effect) |
| 28 | RETURN_PERMANENT | Choose a permanent to return to its owner's hand |
| 29 | CHOOSE_X | Choose the value of X for an X cost |
| 30 | DISCARD | Choose a card to discard (cost, effect, or cleanup) |
| 31 | CHOOSE_MODE | Choose a modal/charm mode |
| 32 | CHOOSE_MANA_COLOR | Choose the color of a flexible mana producer (e.g. LED) |
| 33 | PAY_UNLESS | Pay-or-decline of a "counter unless pay" cost (Mana Leak/Daze) |
| 34 | NAME_CARD | Name a card |
| 35 | CHOOSE_TYPE | Choose a creature type |
| 36 | KEEP_LEGEND | Legend rule: choose which duplicate to keep |
| 37 | ORDER_TRIGGERS | Order simultaneous triggers onto the stack |
| 38 | CHOOSE_REPLACEMENT | Choose which replacement effect / dredge-or-draw to apply |
| 39 | ATTACK_TARGET | Choose what a creature attacks (player or planeswalker) |
| 40 | BLOCK_TARGET | Choose which attacker a blocker blocks |
| 41 | OPTIONAL_YESNO | Optional yes/no confirmation |
| 42 | PLAY_FREE | Play a card for free (e.g. cast from exile) |
| 43 | SYLVAN_CHOICE | Sylvan Library card pick / pay-4-life-or-top choice |
| 44 | CHOOSE_CARD | Choose a card from a (non-library) zone for a zone-change effect |
| 45 | ASSIGN_DAMAGE | Attacker assigns lethal combat damage to a chosen blocker (T3.10; only when 2+ blockers and power ≤ total lethal) |

**Confirm slot convention:** mandatory attacker/blocker queries end with a confirm action. The Python env remaps `action = num_choices - 1` to `-1` before sending to the game.

### Observation space

Total: **34392 floats** (OBS_SIZE in `train/env.py`)

| Range | Size | Content |
|---|---|---|
| `[0:33794]` | 33794 | State vector (see `src/machine_io.h`) |
| `[33794:33858]` | 64 | Action categories, padded to MAX_ACTIONS (64), normalised by ACTION_CATEGORY_MAX (45) |
| `[33858:33922]` | 64 | Action card IDs, padded to MAX_ACTIONS |
| `[33922:33986]` | 64 | Action controller_is_self flags, padded to MAX_ACTIONS |
| `[33986:34056]` | 70 | Hand cast costs (10 slots × 7 cost features) |
| `[34056:34392]` | 336 | Battlefield ability costs (48 slots × 7 cost features) |

State vector layout is documented in `src/machine_io.h`. Key indices: `obs[31]` = priority player is the active player (perspective-relative), `obs[32]` = priority player is Player A (absolute). To get `active_is_a`: `(obs[31] > 0.5) == (obs[32] > 0.5)`.

### Key files

- `train/env.py` — `RoboMageEnv` gymnasium wrapper; `ModelVsScriptedEnv` scripted-opponent wrapper; `SelfPlayEnv` self-play wrapper; `scripted_action` rule-based agent
- `train/extractor.py` — `CardGameExtractor` per-entity feature extractor for the policy network
- `train/train.py` — `MaskablePPO` training, baseline evaluation, observe mode, self-play
- `train/analysis.py` — post-game analysis tool (win rates, action frequencies, SHAP, replay from `.rmrec` recordings)
- `train/play.py` — interactive human-vs-model play
- `train/gen_card_costs.py` — regenerates `train/card_costs.py` from `src/card_vocab.h`
- `train/test_harness.py` — LLM test harness for card behavior verification (see Testing guidelines)
- `train/action_spec.py` — shared semantic-action resolver: turns a `--play` spec string (`cast:Lightning Bolt`, `target:X@opp`, `pass`, …) into the matching legal action index against the current decision's decoded menu. Used by `PlayController` (test harness `--play`, `observe --play-a/--play-b`) and, later, by `play.py` for typed human input.
- `train/card_costs.py` — auto-generated cast-cost and ability-cost matrices (do not edit manually)
- `src/machine_io.h` — state vector layout documentation and constants
- `src/input_logger.cpp` — machine mode BQUERY emission, replay, and CLI input handling
- `src/card_vocab.h` — card name → vocab index mapping for one-hot encoding
