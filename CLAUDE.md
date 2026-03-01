# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

- **Build the project**: `make`
- **Clean build artifacts**: `make clean`
- **Build for release**: `make BUILD=RELEASE`
- **Enable GUI build**: `make GUI=TRUE`

The compiled binary is output to `bin/robomage`.


## Code Style

- Don't put any functions in main.cpp besides main!
- Avoid inline logic for anything that will be repeated; write new functions
- Declare local functions as private in the class, if the header contains a single class/struct
- Iterate through mEntities when possible (working with a system class), rather than iterating through all entities
- Try to consolidate iterations through entities within a function, rather than iterating through many times
- Static (local) functions should be forward declared at top of source file for clarity
- C++17 with exceptions disabled (`-fno-exceptions`)
- Uses clang-format configuration in `.clang-format`
- Platform-specific: Supports Linux, macOS (Darwin), and Windows
- On macOS: uses clang/clang++, on Linux: uses gcc/g++

## Project Overview

Robomage is a C++ implementation of a Magic: The Gathering game engine using an Entity Component System (ECS) architecture. The project aims to simulate MTG game rules including priority, the stack, state-based effects, and turn structure.

The goal is to have every game decision logged as an integer, so that games can be replayed deterministically when provided with the correct seed.

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
- **Player**: Life total, mana pool, lands played this turn

### System Types

- **Orderer** (`src/systems/orderer.h`): Zone operations — card movement, drawing, shuffling, stack ordering
- **StateManager** (`src/systems/state_manager.h`): State-based effects (lethal damage, player death), permanent component lifecycle, `determine_legal_actions`
- **StackManager** (`src/systems/stack_manager.h`): Stack resolution — spells resolve to battlefield or graveyard; standalone ability entities resolve via `Ability::resolve()` then are destroyed

### Game Flow

The `Game` struct (`src/classes/game.h`) tracks:
- Current turn/step (UNTAP, UPKEEP, DRAW, FIRST_MAIN, BEGIN_COMBAT, DECLARE_ATTACKERS, DECLARE_BLOCKERS, COMBAT_DAMAGE, END_COMBAT, SECOND_MAIN, END, CLEANUP)
- Active player and turn order
- Timestamp for ordering simultaneous events
- RNG seed and generator for reproducibility

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

Activated abilities with `valid_tgts != "N_A"` have their target selected before costs are paid and before the ability entity is pushed onto the stack. Target legality is re-verified at resolution.

### Card Parser (`src/parse.cpp`)

Parses `.txt` card scripts from `bin/resources/cardsfolder/`. Key script fields:

| Field | Notes |
|---|---|
| `AB$ <category>` | Activated ability; category `Mana` is normalized to `AddMana` |
| `SP$ <category>` | Spell ability |
| `Cost$ T` | Tap cost; `Sac<1/CARDNAME>` = sacrifice self; `PayLife<N>` = pay N life |
| `Produced$ C/R/G/W/U/B` | Mana color for `AddMana` abilities (sets `color` and `amount=1`) |
| `ValidTgts$ <spec>` | Target spec: `Any`, `Player`, `Creature`, `Land`, `Land.nonBasic`, combinations |
| `NumDmg$ N` | Damage amount for `DealDamage` abilities |
| `ChangeType$ <types>` | Comma-separated subtypes to search (for `ChangeZone`) |
| `Origin$ <zone>` | Source zone for zone-change search |
| `Destination$ <zone>` | Destination zone for zone-change |

Basic lands (Mountain, Forest, etc.) get their mana ability injected by `StateManager::apply_land_abilities` based on land subtypes, not from the script.

### Card Loading System

Cards are loaded on-demand from `bin/resources/cardsfolder/`:
- Card database (`src/card_db.h`): Maps card names to entity IDs, loads scripts on first access
- Parser (`src/parse.h` / `src/parse.cpp`): Parses card scripts into ECS entities with components
- Name to UID conversion: lowercase, spaces to underscores, other characters removed

### Adding a New Card

When implementing a new card, **both** of the following steps are required:

1. Add the card to `src/card_vocab.h` — append a `{"Card Name", N}` entry where N is the next available index. `N_CARD_TYPES` in `src/machine_io.h` must be >= (highest index + 1).
2. Regenerate `train/card_costs.py` by running from the repo root:
   ```
   train/.venv/bin/python train/gen_card_costs.py
   ```
   This writes the cast-cost feature matrix used by the RL environment and extractor.

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

```bash
train/.venv/bin/python train/train.py                                          # train from scratch
train/.venv/bin/python train/train.py --load checkpoints/robomage_final.zip   # resume
train/.venv/bin/python train/train.py --baseline checkpoints/robomage_final.zip  # win rate vs random
train/.venv/bin/python train/train.py --observe checkpoints/robomage_final.zip   # watch one game
```

### Machine mode protocol

`--machine` flag makes the game communicate over stdio for RL training:
- Game emits a `QUERY` line on stdout at each decision point
- Driver writes a single integer back on stdin
- All non-QUERY stdout lines are game narrative and can be ignored

**QUERY line format:**
```
QUERY: <N> <f0>...<f1152> <cat0>...<catN-1> <id0>...<idN-1>
```
- `N` = number of legal choices
- `f0..f1152` = 1153-float state vector (see `src/machine_io.h` for layout)
- `cat0..catN-1` = ActionCategory integer per choice (0–19, see `ActionCategory` enum in `src/classes/action.h`)
- `id0..idN-1` = card vocab index float per choice: `card_vocab_index / N_CARD_TYPES` for card entities, `-0.03125` as null sentinel for players/confirm slots/non-card actions

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
| 10 | OTHER_CHOICE | Generic choice (e.g. attack target, block assignment) |
| 11 | MULLIGAN | Keep (0) or mulligan (1) |
| 12 | BOTTOM_DECK_CARD | Choose card to put on library bottom |
| 13–18 | MANA_W/U/B/R/G/C | Tap a land for the corresponding color |
| 19 | SEARCH_LIBRARY | Choose card from library search (0 = fail to find) |

**Confirm slot convention:** mandatory attacker/blocker queries end with a confirm action. The Python env remaps `action = num_choices - 1` to `-1` before sending to the game.

### Observation space

Total: **1427 floats**

| Range | Size | Content |
|---|---|---|
| `[0:1153]` | 1153 | State vector (see `src/machine_io.h`) |
| `[1153:1185]` | 32 | Action categories, padded to MAX_ACTIONS, normalised by 21 |
| `[1185:1217]` | 32 | Action card IDs, padded to MAX_ACTIONS |
| `[1217:1287]` | 70 | Hand cast costs (10 slots × 7 cost features) |
| `[1287:1427]` | 140 | Battlefield ability costs (20 slots × 7 cost features) |

State vector layout is documented in `src/machine_io.h`. Key indices: `obs[30]` = active player is A, `obs[31]` = priority player is A.

### Key files

- `train/env.py` — `RoboMageEnv` gymnasium wrapper; `ModelVsScriptedEnv` self-play wrapper; `scripted_action` rule-based agent
- `train/extractor.py` — `CardGameExtractor` per-entity feature extractor for the policy network
- `train/train.py` — `MaskablePPO` self-play training, baseline evaluation, observe mode
- `train/gen_card_costs.py` — regenerates `train/card_costs.py` from `src/card_vocab.h`
- `train/card_costs.py` — auto-generated cast-cost and ability-cost matrices (do not edit manually)
- `src/machine_io.h` — state vector layout documentation and constants
- `src/input_logger.cpp` — machine mode QUERY emission, replay, and CLI input handling
- `src/card_vocab.h` — card name → vocab index mapping for one-hot encoding
