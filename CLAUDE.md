# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

- **Build the project**: `make program`
- **Clean build artifacts**: `make clean`
- **Build for release**: `BUILD=RELEASE make program`
- **Enable GUI build**: `GUI=TRUE make program`

The compiled binary is output to `bin/robomage`.

## Project Overview

Robomage is a C++ implementation of a Magic: The Gathering game engine using an Entity Component System (ECS) architecture. The project aims to simulate MTG game rules including priority, the stack, state-based effects, and turn structure.

## Architecture

### ECS Pattern

The codebase follows an Entity Component System architecture based on [Austin Morlan's ECS tutorial](https://austinmorlan.com/posts/entity_component_system/):

- **Entities** (`src/ecs/entity.h`): Simple uint32_t IDs, maximum 5000 entities
- **Components** (`src/components/`): Pure data structs attached to entities
- **Systems** (`src/systems/`): Logic that operates on entities with specific component signatures
- **Coordinator** (`src/ecs/coordinator.h`): Central manager accessed via `global_coordinator` singleton

### Component Types

- **CardData**: Base card information (name, types, mana cost, oracle text, power/toughness)
- **Zone**: Location tracking (library, hand, battlefield, stack, graveyard, exile, sideboard) with ownership and distance_from_top
- **Ability**: Triggered, activated, or spell abilities with source/target/amount
- **Effect**: Continuous effects that cause state-based actions
- **Damage**: Damage tracking components
- **Player**: Player-specific data

### System Types

- **Orderer** (`src/systems/orderer.h`): Manages zone operations, card movement, drawing, shuffling, and stack ordering
- **StateManager** (`src/systems/state_manager.h`): Handles state-based effects (checking for lethal damage, empty libraries, etc.)
- **StackManager**: Manages the stack resolution (declared in main.cpp)

### Game Flow

The `Game` struct (`src/classes/game.h`) tracks:
- Current turn/step (UNTAP, UPKEEP, DRAW, FIRST_MAIN, BEGIN_COMBAT, etc.)
- Active player and turn order
- Timestamp for ordering simultaneous events
- RNG seed and generator for reproducibility

Game loop in `src/main.cpp`:
1. State-based effects check
2. Priority checks and step advancement
3. Display stack and legal actions
4. Process user input (not yet implemented)
5. Resolve abilities/spells

### Card Loading System

Cards are loaded on-demand from `bin/resources/cardsfolder/`:
- Card database (`card_db.h`): Maps card names to entity IDs
- Parser (`parse.h`): Parses card script files into entities with components
- Name to UID conversion handles card name normalization

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
- `RESOURCE_DIR`: Path to resources directory (set at runtime)
- `card_db`: Card name to entity ID mapping

## Code Style

- C++17 with exceptions disabled (`-fno-exceptions`)
- Uses clang-format configuration in `.clang-format`
- Platform-specific: Supports Linux, macOS (Darwin), and Windows
- On macOS: uses clang/clang++, on Linux: uses gcc/g++