#include "state_manager.h"

#include "../components/effect.h"
#include "../components/zone.h"
#include "../components/carddata.h"
#include "../ecs/coordinator.h"
#include "../error.h"
#include "../classes/game.h"

void StateManager::init() {
    Signature signature;
    signature.set(global_coordinator.GetComponentType<Zone>());
    signature.set(global_coordinator.GetComponentType<Effect>());
    global_coordinator.SetSystemSignature<StateManager>(signature);
}

//layers / timestamps would be implemented here; for now order is arbitrary
void StateManager::state_based_effects(Game& game) {
    // Check for mandatory choices based on game step
    // These must be resolved before priority-based actions can occur

    // Declare attackers step - active player must declare attackers
    if (game.cur_step == DECLARE_ATTACKERS && !game.attackers_declared) {
        game.pending_choice = DECLARE_ATTACKERS_CHOICE;
        return;
    }

    // Declare blockers step - defending player must declare blockers
    if (game.cur_step == DECLARE_BLOCKERS && !game.blockers_declared) {
        game.pending_choice = DECLARE_BLOCKERS_CHOICE;
        return;
    }

    // Cleanup step - check if active player exceeds maximum hand size
    if (game.cur_step == CLEANUP) {
        // TODO: Check actual hand size vs maximum hand size
        // For now, assume no discard needed
        // If hand_size > max_hand_size:
        //     game.pending_choice = CLEANUP_DISCARD;
        //     return;
    }

    // TODO: Check for legend rule violations
    // If multiple legendary permanents with same name exist:
    //     game.pending_choice = CHOOSE_ENTITY;
    //     return;

    // No mandatory choice pending
    game.pending_choice = NONE;

    // Process continuous effects
    for (auto &&entity : mEntities) {
        // if we are dealing with an effect
        if (global_coordinator.entity_has_component<Effect>(entity)) {
            // TODO: Apply continuous effects based on layers/timestamps
        }
    }
}

// Placeholder implementation - returns minimal legal actions
// TODO: Implement full legal action determination (see LEGAL_ACTIONS_ROADMAP.md)
std::vector<LegalAction> StateManager::determine_legal_actions(const Game& game) {
    std::vector<LegalAction> actions;

    // Pass priority is always legal
    actions.push_back(LegalAction(PASS_PRIORITY, "Pass priority"));

    // TODO: Check for castable spells in hand
    // - Verify timing restrictions (main phase, stack empty for sorceries)
    // - Check mana availability
    // - Check if player has priority

    // TODO: Check for activatable abilities
    // - Check activation costs (mana, tap, sacrifice, etc.)
    // - Verify timing restrictions
    // - Check if abilities are on battlefield/in appropriate zone

    // TODO: Check for special actions
    // - Play land (main phase, player's turn, haven't played land this turn, stack empty)
    // - Turn face-up morph creatures

    return actions;
}
