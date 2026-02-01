#include "cli.h"

#include <cstdio>
#include "classes/action.h"

// Helper function to get integer input
static int get_int_input() {
    int choice = -1;
    if (scanf("%d", &choice) != 1) {
        // Clear input buffer on error
        int c;
        while ((c = getchar()) != '\n' && c != EOF);
        return -1;
    }
    // Clear the newline left by scanf
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
    return choice;
}

// Get input for mandatory choices (declare attackers/blockers, cleanup discard, etc.)
void get_mandatory_choice_input(Game& cur_game) {
    int choice;

    switch (cur_game.pending_choice) {
        case DECLARE_ATTACKERS_CHOICE:
            // TODO: List all creatures that can attack with numbers
            // Format: "0: Declare no attackers"
            //         "1: Creature Name (ID: X)"
            //         "2: Creature Name (ID: Y)"
            // For multiple attackers, could prompt repeatedly or use multi-select
            printf("Enter choice: ");
            choice = get_int_input();
            if (choice == 0) {
                printf("No attackers declared.\n");
            } else if (choice > 0) {
                printf("TODO: Declare entity %d as attacker\n", choice);
                // TODO: Mark specified creature(s) as attacking
            }
            cur_game.attackers_declared = true;
            cur_game.pending_choice = NONE;
            break;

        case DECLARE_BLOCKERS_CHOICE:
            // TODO: List all valid blocking assignments with numbers
            // Format: "0: Declare no blockers"
            //         "1: Blocker Name blocks Attacker Name"
            //         "2: Blocker Name blocks Attacker Name"
            printf("Enter choice: ");
            choice = get_int_input();
            if (choice == 0) {
                printf("No blockers declared.\n");
            } else if (choice > 0) {
                printf("TODO: Execute blocking assignment %d\n", choice);
                // TODO: Set up specified blocking assignment
            }
            cur_game.blockers_declared = true;
            cur_game.pending_choice = NONE;
            break;

        case CLEANUP_DISCARD:
            // TODO: List cards in hand with numbers
            // Format: "Enter card number to discard (repeat until hand size is valid)"
            //         "1: Card Name"
            //         "2: Card Name"
            printf("Enter card to discard: ");
            choice = get_int_input();
            if (choice > 0) {
                printf("TODO: Discard card %d\n", choice);
                // TODO: Move specified card to graveyard
                // TODO: Check if more discards needed, if so don't clear pending_choice
            }
            cur_game.pending_choice = NONE;
            break;

        case CHOOSE_ENTITY:
            // TODO: List entities involved in the choice
            // Format: "Choose which permanent to keep (legend rule):"
            //         "1: Legendary Creature (ID: X)"
            //         "2: Legendary Creature (ID: Y)"
            printf("Enter choice: ");
            choice = get_int_input();
            if (choice > 0) {
                printf("TODO: Keep entity %d, sacrifice others\n", choice);
                // TODO: Execute the choice (keep one, sacrifice others for legend rule)
            }
            cur_game.pending_choice = NONE;
            break;

        case NONE:
            // Should not reach here
            break;
    }
}

// Get input for priority-based actions (cast spell, activate ability, pass, etc.)
void get_user_action_input(Game& cur_game, std::shared_ptr<StateManager> state_manager) {
    auto legal_actions = state_manager->determine_legal_actions(cur_game);

    printf("\nEnter action number: ");
    int choice = get_int_input();

    // Validate choice
    if (choice < 0 || choice >= static_cast<int>(legal_actions.size())) {
        printf("Invalid choice. Passing priority.\n");
        cur_game.pass_priority();
        return;
    }

    // Execute the chosen action
    const LegalAction& action = legal_actions[static_cast<size_t>(choice)];

    switch (action.type) {
        case PASS_PRIORITY:
            cur_game.pass_priority();
            break;

        case CAST_SPELL:
            printf("TODO: Cast spell from entity %u\n", action.source_entity);
            // TODO: Implement spell casting
            // - Move card to stack
            // - Pay costs (may need additional prompts for mana choices)
            // - Choose targets (may need additional prompts)
            // - Create ability on stack
            cur_game.take_action();
            break;

        case ACTIVATE_ABILITY:
            printf("TODO: Activate ability from entity %u\n", action.source_entity);
            // TODO: Implement ability activation
            // - Pay activation costs (may need additional prompts)
            // - Choose targets (may need additional prompts)
            // - Put ability on stack
            cur_game.take_action();
            break;

        case SPECIAL_ACTION:
            printf("TODO: Execute special action (entity %u)\n", action.source_entity);
            // TODO: Implement special actions
            // - Play land: Move land from hand to battlefield
            // - Morph: Turn face-up morph creature, pay costs
            cur_game.take_action();
            break;
    }
}
