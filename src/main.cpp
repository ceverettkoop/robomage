#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include "action_processor.h"
#include "card_db.h"
#include "classes/action.h"
#include "classes/deck.h"
#include "classes/game.h"
#include "components/ability.h"
#include "components/carddata.h"
#include "components/creature.h"
#include "components/damage.h"
#include "components/effect.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/spell.h"
#include "components/zone.h"
#include "debug.h"
#include "ecs/coordinator.h"
#include "ecs/events.h"
#include "input_logger.h"
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

// London Mulligan: both players decide to keep or mulligan, then bottom-deck.
// Each decision is a serializable integer logged via InputLogger.
static void do_london_mulligan(std::shared_ptr<Orderer> orderer) {
    int mulligans_a = 0;
    int mulligans_b = 0;

    // Phase 1: Player A decides
    {
        cur_game.player_a_has_priority = true;
        bool keeping = false;
        while (!keeping) {
            print_hand(orderer, Zone::PLAYER_A);
            printf("Player A: 0=Keep, 1=Mulligan (taken %d)\n", mulligans_a);
            std::vector<ActionCategory> cats = {ActionCategory::MULLIGAN, ActionCategory::MULLIGAN};
            int choice = InputLogger::instance().get_logged_input(0, cats);
            if (choice == 0) {
                keeping = true;
            } else {
                mulligans_a++;
                auto hand = orderer->get_hand(Zone::PLAYER_A);
                for (auto card : hand) {
                    orderer->add_to_zone(false, card, Zone::LIBRARY);
                }
                orderer->shuffle_library(Zone::PLAYER_A);
                orderer->draw(Zone::PLAYER_A, 7);
            }
        }
    }

    // Phase 2: Player B decides
    {
        cur_game.player_a_has_priority = false;
        bool keeping = false;
        while (!keeping) {
            print_hand(orderer, Zone::PLAYER_B);
            printf("Player B: 0=Keep, 1=Mulligan (taken %d)\n", mulligans_b);
            std::vector<ActionCategory> cats = {ActionCategory::MULLIGAN, ActionCategory::MULLIGAN};
            int choice = InputLogger::instance().get_logged_input(0, cats);
            if (choice == 0) {
                keeping = true;
            } else {
                mulligans_b++;
                auto hand = orderer->get_hand(Zone::PLAYER_B);
                for (auto card : hand) {
                    orderer->add_to_zone(false, card, Zone::LIBRARY);
                }
                orderer->shuffle_library(Zone::PLAYER_B);
                orderer->draw(Zone::PLAYER_B, 7);
            }
        }
    }

    // Phase 3: Player A bottom-decks mulligans_a cards, one at a time
    cur_game.player_a_has_priority = true;
    for (int i = 0; i < mulligans_a; i++) {
        auto hand = orderer->get_hand(Zone::PLAYER_A);
        if (hand.empty()) break;
        printf("Player A: Choose card to put on library bottom (%d remaining):\n", mulligans_a - i);
        for (size_t j = 0; j < hand.size(); j++) {
            auto& cd = global_coordinator.GetComponent<CardData>(hand[j]);
            printf("  %zu: %s\n", j, cd.name.c_str());
        }
        std::vector<ActionCategory> cats(hand.size(), ActionCategory::BOTTOM_DECK_CARD);
        int choice = InputLogger::instance().get_logged_input(0, cats, hand);
        if (choice >= 0 && choice < static_cast<int>(hand.size())) {
            orderer->add_to_zone(true, hand[static_cast<size_t>(choice)], Zone::LIBRARY);
        }
    }

    // Phase 4: Player B bottom-decks mulligans_b cards, one at a time
    cur_game.player_a_has_priority = false;
    for (int i = 0; i < mulligans_b; i++) {
        auto hand = orderer->get_hand(Zone::PLAYER_B);
        if (hand.empty()) break;
        printf("Player B: Choose card to put on library bottom (%d remaining):\n", mulligans_b - i);
        for (size_t j = 0; j < hand.size(); j++) {
            auto& cd = global_coordinator.GetComponent<CardData>(hand[j]);
            printf("  %zu: %s\n", j, cd.name.c_str());
        }
        std::vector<ActionCategory> cats(hand.size(), ActionCategory::BOTTOM_DECK_CARD);
        int choice = InputLogger::instance().get_logged_input(0, cats, hand);
        if (choice >= 0 && choice < static_cast<int>(hand.size())) {
            orderer->add_to_zone(true, hand[static_cast<size_t>(choice)], Zone::LIBRARY);
        }
    }
}

int main(int argc, char const *argv[]) {
    int choice;
    char buf[FILENAME_MAX];
    RESOURCE_DIR = getcwd(buf, FILENAME_MAX);
    RESOURCE_DIR += "/resources";

    // Parse command-line arguments
    std::string replay_file_path;
    bool replay_mode = false;
    bool machine_mode = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--replay" && i + 1 < argc) {
            replay_mode = true;
            replay_file_path = argv[i + 1];
            i++;
        } else if (std::string(argv[i]) == "--machine") {
            machine_mode = true;
        } else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            printf("robomage %s\n", VERSION_NUMBER);
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --replay <logfile>  Replay a previously logged game\n");
            printf("  --machine           Machine mode: emit QUERY lines for AI input\n");
            printf("  --help, -h          Show this help message\n");
            return 0;
        }
    }

    // Use minimal test deck for both players
    DEFAULT_DECK_ONE = Deck(RESOURCE_DIR + "/decks/test_minimal.dk");
    DEFAULT_DECK_TWO = Deck(RESOURCE_DIR + "/decks/test_minimal.dk");

    printf("robomage %s\n", VERSION_NUMBER);

    // Setup seed and input logging
    unsigned int seed;
    if (replay_mode) {
        InputLogger::instance().init_replay(replay_file_path);
        seed = InputLogger::instance().get_replay_seed();
    } else {
        seed = static_cast<unsigned int>(time(nullptr));
        // Create logs directory
        std::string mkdir_cmd = "mkdir -p " + RESOURCE_DIR + "/logs";
        int result = system(mkdir_cmd.c_str());
        (void)result;  // Ignore return value
        if (machine_mode) {
            InputLogger::instance().init_machine(seed, RESOURCE_DIR);
        } else {
            InputLogger::instance().init_logging(seed, RESOURCE_DIR);
        }
    }
    std::srand(seed);
    printf("Using seed: %u\n", seed);

    global_coordinator.Init();
    global_coordinator.RegisterComponent<Ability>();
    global_coordinator.RegisterComponent<CardData>();
    global_coordinator.RegisterComponent<Creature>();
    global_coordinator.RegisterComponent<Damage>();
    global_coordinator.RegisterComponent<Permanent>();
    global_coordinator.RegisterComponent<Player>();
    global_coordinator.RegisterComponent<Spell>();
    global_coordinator.RegisterComponent<Zone>();
    global_coordinator.RegisterComponent<Effect>();

    auto orderer = global_coordinator.RegisterSystem<Orderer>();
    auto state_manager = global_coordinator.RegisterSystem<StateManager>();
    auto stack_manager = global_coordinator.RegisterSystem<StackManager>();
    Orderer::init();  // system signature is set here
    StateManager::init();
    StackManager::init();

    // Register event listeners
    global_coordinator.AddEventListener(Events::CREATURE_DIED, [](Event& event) {
        Entity dead = event.GetParam<Entity>(Params::ENTITY);
        if (global_coordinator.entity_has_component<CardData>(dead)) {
            auto& cd = global_coordinator.GetComponent<CardData>(dead);
            printf("[EVENT] CREATURE_DIED: %s\n", cd.name.c_str());
        }
    });

    // one time setup for this game
    cur_game = Game(seed);
    cur_game.generate_players(DEFAULT_DECK_ONE, DEFAULT_DECK_TWO);
    orderer->generate_libraries(DEFAULT_DECK_ONE, DEFAULT_DECK_TWO);

    // London Mulligan (companion not yet implemented)
    orderer->draw_hands();
    do_london_mulligan(orderer);
    cur_game.player_a_has_priority = true;  // restore for game start

    // PLAYER A IS ALWAYS ON THE PLAY IN THIS WORLD
    // game loop
    while (cur_game.ended != true) {
        state_manager->state_based_effects(cur_game, orderer);
        // mandatory choices
        // e.g. declare target, declare attackers or declare blockers - discard at cleanup - legend rule; choice at
        // resolution; declare target
        if (cur_game.is_mandatory_choice_pending()) {
            proc_mandatory_choice(cur_game, orderer);
            continue;
        }
        //move to next step if nothing else can occur or if both players have passed priority
        //in those cases advance_step will return true
        if (cur_game.advance_step(stack_manager, orderer)) {
            continue;
        }

        //active player can do something, list all options
        //if only option is pass priority we just do it
        auto legal_actions = state_manager->determine_legal_actions(cur_game, orderer, stack_manager);
        if (legal_actions.size() == 1) {
            cur_game.pass_priority();
            continue;
        }
        print_step(cur_game);
        // prompt for action
        print_stack(orderer);
        print_battlefield(orderer);
        print_mana_pools();
        print_hand(orderer, cur_game.player_a_has_priority ? Zone::PLAYER_A : Zone::PLAYER_B);

        print_legal_actions(cur_game, legal_actions);
        std::vector<ActionCategory> action_cats;
        std::vector<Entity> action_ents;
        action_cats.reserve(legal_actions.size());
        action_ents.reserve(legal_actions.size());
        for (auto& la : legal_actions) {
            action_cats.push_back(la.category);
            action_ents.push_back(la.source_entity);
        }
        choice = InputLogger::instance().get_logged_input(cur_game.turn, action_cats, action_ents);

        if (choice >= 0 && choice < static_cast<int>(legal_actions.size())) {
            process_action(legal_actions[static_cast<size_t>(choice)], cur_game, orderer);
        } else {
            printf("Invalid action\n");
        }
    }

    // repeat
    return 0;
}
