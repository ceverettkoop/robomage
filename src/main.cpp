#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include "action_processor.h"
#include "card_db.h"
#include "classes/deck.h"
#include "classes/game.h"
#include "cli_output.h"
#include "components/ability.h"
#include "components/carddata.h"
#include "components/color_identity.h"
#include "components/creature.h"
#include "components/damage.h"
#include "components/effect.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/spell.h"
#include "components/zone.h"
#include "ecs/coordinator.h"
#include "ecs/events.h"
#include "input_logger.h"
#include "machine_io.h"
#include "systems/orderer.h"
#include "systems/stack_manager.h"
#include "systems/state_manager.h"

#ifndef VERSION_NUMBER
#define VERSION_NUMBER "0.001"
#endif

#ifdef GUI
extern "C" {
#include "gui.h"
#include "pthread.h"
}
#endif

std::string RESOURCE_DIR;
Coordinator global_coordinator = Coordinator();
Deck DEFAULT_DECK_ONE;
Deck DEFAULT_DECK_TWO;
Game cur_game;
bool gui_mode = false;
bool has_human_player = false;
bool human_player_is_a = false;
extern volatile bool gui_killed;
pthread_t game_loop_thread;

GameState gs;
const GameState *gs_ptr = &gs;
std::string replay_file_path;
bool replay_mode = false;
bool machine_mode = false;


//runs in thread seperate from gui
static void *game_loop(void *args) {

    // Use minimal test deck for both players
    DEFAULT_DECK_ONE = Deck(RESOURCE_DIR + "/decks/test_minimal.dk");
    DEFAULT_DECK_TWO = Deck(RESOURCE_DIR + "/decks/test_minimal.dk");

    cli_print_version(VERSION_NUMBER);

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
    cli_print_seed(seed);

    global_coordinator.Init();
    global_coordinator.RegisterComponent<Ability>();
    global_coordinator.RegisterComponent<CardData>();
    global_coordinator.RegisterComponent<ColorIdentity>();
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

    // TODO deal with this; move it out of main; move printf to cli_output; implement some cards besides DRC that care
    // about this this has to be moved out of main and into its own unit; and the printf call needs to be handled
    // elsewhere
    global_coordinator.AddEventListener(Events::CREATURE_DIED, [](Event &event) {
        Entity dead = event.GetParam<Entity>(Params::ENTITY);
        if (global_coordinator.entity_has_component<CardData>(dead)) {
            auto &cd = global_coordinator.GetComponent<CardData>(dead);
            game_log("[EVENT] CREATURE_DIED: %s\n", cd.name.c_str());
        }
    });

    // one time setup for this game
    cur_game = Game(seed);
    cur_game.generate_players(DEFAULT_DECK_ONE, DEFAULT_DECK_TWO);
    orderer->generate_libraries(DEFAULT_DECK_ONE, DEFAULT_DECK_TWO);

    // draw 7 and run mulligan
    orderer->draw_hands();
    orderer->do_london_mulligan();
    cur_game.player_a_has_priority = true;  // restore for game start

    // PLAYER A IS ALWAYS ON THE PLAY IN THIS WORLD
    // game loop
    size_t prev_turn = (size_t)-1;
    while (cur_game.ended != true) {
        if (gui_killed) return NULL;
        // if new turn provide update
        if (!InputLogger::instance().is_machine_mode() && cur_game.turn != prev_turn) {
            cli_print_turn_header(cur_game.turn, cur_game.player_a_turn);
            prev_turn = cur_game.turn;
        }
        state_manager->state_based_effects(cur_game, orderer);
        // mandatory choices
        // e.g. declare target, declare attackers or declare blockers - discard at cleanup - legend rule; choice at
        // resolution; declare target
        if (cur_game.is_mandatory_choice_pending()) {
            proc_mandatory_choice(cur_game, orderer);
            continue;
        }
        // move to next step if nothing else can occur or if both players have passed priority
        // in those cases advance_step will return true
        if (cur_game.advance_step(stack_manager, orderer)) {
            continue;
        } else {
            // check state_based_effects again if something changed bc of resolution
            state_manager->state_based_effects(cur_game, orderer);
        }

        // active player can do something, list all options
        // if only option is pass priority we just do it
        auto legal_actions = state_manager->determine_legal_actions(cur_game, orderer, stack_manager);
        if (legal_actions.size() == 1) {
            cur_game.pass_priority();
            continue;
        }

        populate_gamestate(&gs);
        print_game_state(&gs);
        int choice = InputLogger::instance().get_input(legal_actions);
        process_action(legal_actions[static_cast<size_t>(choice)], cur_game, orderer);
    }
    return NULL;
}

int main(int argc, char const *argv[]) {
    char buf[FILENAME_MAX];
    RESOURCE_DIR = getcwd(buf, FILENAME_MAX);
    RESOURCE_DIR += "/resources";

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--replay" && i + 1 < argc) {
            replay_mode = true;
            replay_file_path = argv[i + 1];
            i++;
        } else if (std::string(argv[i]) == "--machine") {
            machine_mode = true;
        } else if (std::string(argv[i]) == "--gui") {
            gui_mode = true;
        } else if (std::string(argv[i]) == "--player" && i + 1 < argc) {
            has_human_player = true;
            std::string p = argv[i + 1];
            human_player_is_a = (p == "A" || p == "a");
            i++;
        } else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            cli_print_help(argv[0], VERSION_NUMBER);
            return 0;
        }
    }

    if (pthread_create(&game_loop_thread, NULL, game_loop, NULL) != 0) {
        perror("pthread_create");
        exit(1);
    }

    if (gui_mode) {
#ifdef GUI
        gui_set_resource_dir(RESOURCE_DIR.c_str());
        init_gui();
#else
        fatal_error("NOTE TO USE GUI; MUST BE COMPILED WITH FLAG GUI==TRUE, RUN AGAIN WITHOUT --gui FLAG OR RECOMPILE");
#endif
    }

    pthread_join(game_loop_thread, NULL);
    return 0;
}
