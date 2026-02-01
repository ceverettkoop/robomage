#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include "card_db.h"
#include "classes/deck.h"
#include "classes/game.h"
#include "cli.h"
#include "components/ability.h"
#include "components/carddata.h"
#include "components/damage.h"
#include "components/effect.h"
#include "components/player.h"
#include "components/zone.h"
#include "debug.h"
#include "ecs/coordinator.h"
#include "systems/orderer.h"
#include "systems/stack_manager.h"
#include "systems/state_manager.h"

#ifndef VERSION_NUMBER
#define VERSION_NUMBER "0.001"
#endif

/*
extern "C" {
#include "gui.h"
}
*/

std::string RESOURCE_DIR;
Coordinator global_coordinator = Coordinator();
Deck DEFAULT_DECK_ONE;
Deck DEFAULT_DECK_TWO;
Game cur_game;

int main(int argc, char const *argv[]) {
    int choice;
    char buf[FILENAME_MAX];
    RESOURCE_DIR = getcwd(buf, FILENAME_MAX);
    RESOURCE_DIR += "/resources";

    DEFAULT_DECK_ONE = Deck(RESOURCE_DIR + "/decks/deck_one.dk");
    DEFAULT_DECK_TWO = Deck(RESOURCE_DIR + "/decks/deck_two.dk");

    printf("robomage %s\n", VERSION_NUMBER);
    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    std::srand(seed);

    global_coordinator.Init();
    global_coordinator.RegisterComponent<Ability>();
    global_coordinator.RegisterComponent<CardData>();
    global_coordinator.RegisterComponent<Damage>();
    global_coordinator.RegisterComponent<Player>();
    global_coordinator.RegisterComponent<Zone>();
    global_coordinator.RegisterComponent<Effect>();

    auto orderer = global_coordinator.RegisterSystem<Orderer>();
    auto state_manager = global_coordinator.RegisterSystem<StateManager>();
    auto stack_manager = global_coordinator.RegisterSystem<StackManager>();
    Orderer::init();  // system signature is set here
    StateManager::init();
    StackManager::init();

    // one time setup for this game
    Game cur_game(seed);
    cur_game.generate_players(DEFAULT_DECK_ONE, DEFAULT_DECK_TWO);
    orderer->generate_libraries(DEFAULT_DECK_ONE, DEFAULT_DECK_TWO);

    // TODO MULLIGANS, COMPANION ETC
    orderer->draw_hands();
    print_hand(orderer, Zone::PLAYER_A);
    print_hand(orderer, Zone::PLAYER_B);

    // PLAYER A IS ALWAYS ON THE PLAY IN THIS WORLD

    // game loop
    while (cur_game.ended != true) {
        print_step(cur_game);
        state_manager->state_based_effects(cur_game);
        // mandatory choices
        // e.g. declare target, declare attackers or declare blockers - discard at cleanup - legend rule; choice at
        // resolution; declare target
        if (cur_game.is_mandatory_choice_pending()) {
            // describe which player has to make a choice and how to input it
            choice = -1;
            while (choice == -1) {
                print_mandatory_choice_description(cur_game);
                int choice = get_int_input();
                if(choice == -1) printf("Invalid input\n");
            }
            proc_mandatory_choice(choice);
            continue;
        }
        // will return true if priority has been passed and there is nothing on stack
        if (cur_game.advance_step(stack_manager)) {
            continue;
        }
        // prompt for action
        print_stack(orderer);
        print_legal_actions(cur_game, state_manager);
        choice = get_int_input();
        proc_user_action(choice);
    }

    // repeat
    return 0;
}
