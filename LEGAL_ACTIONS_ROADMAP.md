# Legal Actions Implementation Roadmap

This document outlines the steps required to fully implement the `StateManager::determine_legal_actions()` function in robomage.

## Overview

The `determine_legal_actions()` function must analyze the complete game state and return all legal actions available to the player with priority. This requires checking:
- Game phase/step restrictions
- Mana availability
- Timing rules
- Card-specific restrictions
- Priority state

## Current State

**Implemented:**
- Basic structure: `LegalAction` struct and `ActionType` enum in `src/classes/action.h`
- Placeholder function that only returns "Pass priority"
- Integration with `print_legal_actions()` in debug.cpp

**Not Implemented:**
- Actual legal action determination logic

## Implementation Phases

### Phase 1: Foundation - Game State Queries

Before determining legal actions, we need helper functions to query game state.

**Required additions to `Game` struct (src/classes/game.h):**
- `bool has_played_land_this_turn` - Track if active player has played a land
- `Entity get_priority_player_entity()` - Return entity ID of player with priority

**Required additions to `StateManager`:**
- `Zone::Ownership get_priority_owner(const Game& game)` - Helper to get priority player
- `bool is_main_phase(const Game& game)` - Check if in FIRST_MAIN or SECOND_MAIN
- `bool is_stack_empty()` - Check if stack has no objects

**Required additions to `Orderer`:**
- `std::vector<Entity> get_hand(Zone::Ownership owner)` - Already exists, verify signature
- `std::vector<Entity> get_battlefield(Zone::Ownership owner)` - Get permanents controlled by player

**Estimated complexity:** Medium
**Dependencies:** None

---

### Phase 2: Mana System

Determine available mana for casting spells and activating abilities.

**Required additions to `Player` component:**
- Already has `std::multiset<Colors> mana` for mana pool
- Add `std::vector<Entity> mana_sources` - Track which lands/mana abilities can be activated

**Required additions to components:**
- Extend `CardData` or create `ManaCost` component to represent mana costs
- May need to parse mana cost strings (e.g., "2UU", "XGG", "{2/U}")

**Required logic in `StateManager`:**
- `bool can_pay_mana_cost(Zone::Ownership player, const ManaCost& cost)` - Check if player can pay
- `std::vector<Colors> get_available_mana(Zone::Ownership player)` - Get mana from untapped sources
- Handle generic mana, colored mana, hybrid mana, Phyrexian mana, snow mana, etc.

**Estimated complexity:** High (mana system is complex)
**Dependencies:** Phase 1

---

### Phase 3: Cast Spell Actions

Determine which cards in hand can be legally cast.

**Required logic:**
- Iterate through priority player's hand
- For each card with `Ability::SPELL`:
  - **Timing check:**
    - Instants: Can cast any time with priority
    - Sorceries: Can cast only during main phase, when stack is empty, on player's turn
    - Creature/Artifact/Enchantment spells: Same as sorceries
    - Lands: Not castable (handled as special actions)
  - **Cost check:** Can player pay mana cost? (Phase 2)
  - **Legal targets:** Does spell require targets? Are legal targets available?
  - **Additional restrictions:** Color restrictions, can't be countered, etc.

**Required additions to components:**
- Need to distinguish card types (instant, sorcery, creature, etc.) - may already exist in `CardData`
- Need to identify targeting requirements

**Implementation approach:**
```cpp
// Pseudocode
for (Entity card : get_hand(priority_owner)) {
    auto& data = GetComponent<CardData>(card);

    // Check if it's a spell
    if (!is_spell(data)) continue;

    // Check timing
    if (is_sorcery_speed(data) && !can_cast_sorcery(game)) continue;

    // Check mana
    if (!can_pay_mana_cost(priority_owner, data.mana_cost)) continue;

    // Check for legal targets (if required)
    if (requires_target(data) && !has_legal_targets(data)) continue;

    // Add to legal actions
    actions.push_back(LegalAction(CAST_SPELL, card, "Cast " + data.name));
}
```

**Estimated complexity:** High
**Dependencies:** Phase 1, Phase 2

---

### Phase 4: Activated Abilities

Determine which activated abilities can be activated.

**Required logic:**
- Iterate through all permanents controlled by priority player
- Check each permanent for activated abilities
- For each activated ability:
  - **Timing restrictions:** Most can be activated any time, but some have restrictions (e.g., "Activate only as a sorcery")
  - **Cost check:** Tap symbols, mana costs, sacrifice costs, etc.
  - **Usage limits:** "Activate only once per turn", etc.
  - **Legal targets:** If ability requires targets

**Required additions to components:**
- `Ability` component needs richer representation:
  - Activation cost (mana, tap, sacrifice, etc.)
  - Timing restrictions
  - Usage counter (for "once per turn" abilities)
- May need a `Permanent` component to track tap state

**Required additions to `StateManager`:**
- `std::vector<Entity> get_abilities_on_permanent(Entity permanent)` - Get all abilities
- `bool can_activate_ability(Entity ability, const Game& game)` - Check all restrictions

**Implementation approach:**
```cpp
// Pseudocode
for (Entity permanent : get_battlefield(priority_owner)) {
    for (Entity ability : get_abilities_on_permanent(permanent)) {
        auto& ability_data = GetComponent<Ability>(ability);

        if (ability_data.ability_type != ACTIVATED) continue;

        // Check timing restrictions
        if (has_timing_restriction(ability) && !meets_timing(game)) continue;

        // Check if ability can be paid for
        if (!can_pay_activation_cost(priority_owner, ability)) continue;

        // Check usage limits
        if (ability_used_this_turn(ability)) continue;

        // Add to legal actions
        actions.push_back(LegalAction(ACTIVATE_ABILITY, permanent,
                         "Activate " + describe_ability(ability)));
    }
}
```

**Estimated complexity:** Very High
**Dependencies:** Phase 1, Phase 2

---

### Phase 5: Special Actions

Handle special game actions like playing lands.

**Play Land:**
- Check if it's a main phase
- Check if priority player's turn
- Check if stack is empty
- Check if player has already played a land this turn
- Check if player has additional land plays (e.g., from Exploration)

**Other special actions:**
- Turn face-up morph creatures (if implemented)
- Suspend cards (if implemented)
- Foretell (if implemented)

**Required additions to `Game`:**
- `bool has_played_land_this_turn` (mentioned in Phase 1)
- `uint8_t lands_played_this_turn` - Track count
- `uint8_t land_plays_available` - Usually 1, but can be modified by effects

**Implementation approach:**
```cpp
// Pseudocode for play land
if (is_main_phase(game) &&
    game.player_a_turn == game.player_a_has_priority &&
    is_stack_empty() &&
    game.lands_played_this_turn < game.land_plays_available) {

    for (Entity card : get_hand(priority_owner)) {
        auto& data = GetComponent<CardData>(card);
        if (is_land(data)) {
            actions.push_back(LegalAction(SPECIAL_ACTION, card,
                             "Play " + data.name));
        }
    }
}
```

**Estimated complexity:** Medium
**Dependencies:** Phase 1

---

### Phase 6: Split Second & Replacement Effects

Handle edge cases that prevent normal action-taking.

**Split Second:**
- When a spell with split second is on the stack, only special actions and mana abilities are legal

**Can't Cast Spells:**
- Effects like "Players can't cast spells" (e.g., Abeyance)

**Required additions:**
- Flag in `Game` or check stack for split second spells
- System to track continuous effects that restrict actions

**Estimated complexity:** Medium
**Dependencies:** Phase 3, Phase 4

---

### Phase 7: Targeting System

Full implementation of targeting rules.

This is a prerequisite for Phases 3 and 4 but complex enough to be its own phase.

**Required:**
- Determine legal targets for abilities/spells
- Check targeting restrictions (color, type, controller, etc.)
- Handle "up to N targets"
- Handle modal spells ("Choose one —")
- Handle targeting protection (hexproof, shroud, protection from X)

**Required additions:**
- Target specification in abilities
- Target validation logic
- Protection checking

**Estimated complexity:** Very High
**Dependencies:** Phase 1

---

## Integration Points

Once `determine_legal_actions()` returns a complete list, the next steps are:

1. **User Input System** - Allow player to select an action by number
2. **Action Execution** - Implement `execute_action(LegalAction action)` to perform the chosen action
3. **AI Integration** - Use legal actions list for AI decision-making

## Testing Strategy

Each phase should be tested incrementally:

1. **Unit tests** - Test individual helper functions
2. **Integration tests** - Create specific game states and verify correct actions are returned
3. **Edge cases** - Test with split second, cost increases/reductions, restrictions, etc.

## Estimated Total Complexity

**Overall difficulty:** Very High

The legal action determination system touches nearly every aspect of the MTG rules engine:
- Comprehensive rules knowledge required
- Complex interactions between components
- Performance considerations (checking thousands of entities per frame)

**Suggested order of implementation:**
1. Phase 1 (Foundation)
2. Phase 5 (Play land) - Simplest complete feature
3. Phase 2 (Mana system)
4. Phase 7 (Targeting) - Needed for spells
5. Phase 3 (Cast spell)
6. Phase 4 (Activated abilities) - Most complex
7. Phase 6 (Edge cases)

## Notes

- This system will need constant refinement as more MTG mechanics are added
- Consider performance: with 5000 max entities, checking all possibilities could be expensive
- May want to cache legal actions until game state changes
- The action system doesn't handle declare attackers/blockers - those are separate UI flows prompted at the appropriate step
