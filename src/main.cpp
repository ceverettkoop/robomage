#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include "action_processor.h"
#include "card_db.h"
#include "classes/deck.h"
#include "classes/game.h"
#include "cli_output.h"
#include "error.h"
#include "components/ability.h"
#include "components/carddata.h"
#include "components/color_identity.h"
#include "components/creature.h"
#include "components/damage.h"
#include "components/effect.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/spell.h"
#include "components/token.h"
#include "components/zone.h"
#include "ecs/coordinator.h"
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
std::string deck_a_name = "delver";
std::string deck_b_name = "delver";
bool seed_override = false;
unsigned int seed_value = 0;
bool no_shuffle = false;
bool narrative_mode = false;
bool bo3_mode = false;
std::vector<std::string> battlefield_a_cards;
std::vector<std::string> battlefield_b_cards;

// match state (accessible for state serialization)
int match_game_number = -1;  // -1 = single game, 0-2 = bo3 game index
int match_wins_a = 0;
int match_wins_b = 0;
bool sideboard_phase = false;

struct EcsSystems {
    std::shared_ptr<Orderer> orderer;
    std::shared_ptr<StateManager> state_manager;
    std::shared_ptr<StackManager> stack_manager;
};

static EcsSystems init_ecs();
static int play_single_game(EcsSystems &sys, const Deck &deck_a, const Deck &deck_b,
                            bool player_a_goes_first, unsigned int seed);
static void run_sideboard_phase(Deck &deck, Zone::Ownership player);
static std::vector<std::string> split_card_list(const std::string &csv);

static EcsSystems init_ecs() {
    card_db.clear();
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
    global_coordinator.RegisterComponent<Token>();

    auto orderer = global_coordinator.RegisterSystem<Orderer>();
    auto state_manager = global_coordinator.RegisterSystem<StateManager>();
    auto stack_manager = global_coordinator.RegisterSystem<StackManager>();
    Orderer::init();
    StateManager::init();
    StackManager::init();

    return {orderer, state_manager, stack_manager};
}

// Returns winner: Zone::PLAYER_A (1) or Zone::PLAYER_B (2)
static int play_single_game(EcsSystems &sys, const Deck &deck_a, const Deck &deck_b,
                            bool player_a_goes_first, unsigned int seed) {
    cur_game = Game(seed);
    cur_game.generate_players(deck_a, deck_b);
    sys.orderer->generate_libraries(deck_a, deck_b);

    sys.orderer->draw_hands();
    sys.orderer->do_london_mulligan();

    // place pre-set battlefield permanents
    std::vector<Entity> preplaced;
    if (!battlefield_a_cards.empty()) {
        auto placed = sys.orderer->place_on_battlefield(battlefield_a_cards, Zone::PLAYER_A);
        preplaced.insert(preplaced.end(), placed.begin(), placed.end());
    }
    if (!battlefield_b_cards.empty()) {
        auto placed = sys.orderer->place_on_battlefield(battlefield_b_cards, Zone::PLAYER_B);
        preplaced.insert(preplaced.end(), placed.begin(), placed.end());
    }
    // run SBE once to attach Permanent/Creature components, then clear summoning sickness
    if (!preplaced.empty()) {
        sys.state_manager->state_based_effects(cur_game, sys.orderer);
        for (auto e : preplaced) {
            if (global_coordinator.entity_has_component<Permanent>(e)) {
                global_coordinator.GetComponent<Permanent>(e).has_summoning_sickness = false;
            }
        }
    }

    cur_game.player_a_turn = player_a_goes_first;
    cur_game.player_a_has_priority = player_a_goes_first;

    size_t prev_turn = (size_t)-1;
    while (cur_game.ended != true) {
        if (gui_killed) return 0;
        if ((!InputLogger::instance().is_machine_mode() || narrative_mode) && cur_game.turn != prev_turn) {
            cli_print_turn_header(cur_game.turn, cur_game.player_a_turn);
            prev_turn = cur_game.turn;
        }
        Zone::Ownership viewer = (has_human_player)
            ? (human_player_is_a ? Zone::PLAYER_A : Zone::PLAYER_B)
            : Zone::UNKNOWN;

        sys.state_manager->process_turn_based_actions(cur_game, sys.orderer);
        if (cur_game.is_mandatory_choice_pending()) {
            populate_gamestate(&gs, viewer);
            proc_mandatory_choice(cur_game, sys.orderer);
            continue;
        }
        sys.state_manager->state_based_effects(cur_game, sys.orderer);
        if (cur_game.ended) break;
        if (cur_game.advance_step(sys.stack_manager, sys.orderer)) {
            continue;
        } else {
            sys.state_manager->state_based_effects(cur_game, sys.orderer);
            if (cur_game.ended) break;
        }

        auto legal_actions = sys.state_manager->determine_legal_actions(cur_game, sys.orderer, sys.stack_manager);
        if (legal_actions.size() == 1) {
            cur_game.pass_priority();
            if (!machine_mode) populate_gamestate(&gs, viewer);
            continue;
        }

        populate_gamestate(&gs, viewer);
        print_game_state(&gs);
        int choice = InputLogger::instance().get_input(legal_actions);
        process_action(legal_actions[static_cast<size_t>(choice)], cur_game, sys.orderer);
    }
    return cur_game.winner;
}

// present sideboard choices to a player via the standard query mechanism
static void run_sideboard_phase(Deck &deck, Zone::Ownership player) {
    sideboard_phase = true;
    const char *player_name = (player == Zone::PLAYER_A) ? "Player A" : "Player B";

    while (true) {
        if (gui_killed) break;

        // build action list: index 0 = done, 1..N = sideboard cards to bring in
        std::vector<LegalAction> actions;
        actions.emplace_back(ActionType::SPECIAL_ACTION, "Done sideboarding");
        actions.back().category = ActionCategory::SIDEBOARD_DONE;

        for (size_t i = 0; i < deck.sideboard.size(); i++) {
            std::string desc = "Sideboard in: " + std::to_string(deck.sideboard[i].first) + "x " + deck.sideboard[i].second;
            actions.emplace_back(ActionType::SPECIAL_ACTION, desc);
            actions.back().category = ActionCategory::SIDEBOARD_IN;
            // load the card so we can get its vocab index for ML
            Entity card_eid = load_card(deck.sideboard[i].second);
            actions.back().source_entity = card_eid;
        }

        if (actions.size() <= 1) break;  // no sideboard cards available

        game_log("\n%s sideboarding (%zu cards in sideboard):\n", player_name, deck.sideboard.size());

        populate_gamestate(&gs, player);
        print_game_state(&gs);
        int choice = InputLogger::instance().get_input(actions);

        if (choice == 0) break;  // done

        // player chose to bring in a sideboard card
        size_t sb_idx = static_cast<size_t>(choice - 1);
        std::string card_in = deck.sideboard[sb_idx].second;

        // now ask which main deck card to swap out
        std::vector<LegalAction> out_actions;
        for (size_t i = 0; i < deck.main_deck.size(); i++) {
            std::string desc = "Sideboard out: " + std::to_string(deck.main_deck[i].first) + "x " + deck.main_deck[i].second;
            out_actions.emplace_back(ActionType::SPECIAL_ACTION, desc);
            out_actions.back().category = ActionCategory::SIDEBOARD_OUT;
            Entity card_eid = load_card(deck.main_deck[i].second);
            out_actions.back().source_entity = card_eid;
        }

        game_log("Choose card to remove from main deck (replacing with %s):\n", card_in.c_str());
        populate_gamestate(&gs, player);
        int out_choice = InputLogger::instance().get_input(out_actions);

        size_t md_idx = static_cast<size_t>(out_choice);
        std::string card_out = deck.main_deck[md_idx].second;

        // perform the swap (one copy at a time)
        if (deck.sideboard[sb_idx].first > 1) {
            deck.sideboard[sb_idx].first--;
        } else {
            deck.sideboard.erase(deck.sideboard.begin() + static_cast<long>(sb_idx));
        }
        // add the removed main deck card to sideboard
        bool found_in_sb = false;
        for (auto &entry : deck.sideboard) {
            if (entry.second == card_out) {
                entry.first++;
                found_in_sb = true;
                break;
            }
        }
        if (!found_in_sb) {
            deck.sideboard.push_back({1, card_out});
        }
        // add the sideboard card to main deck
        bool found_in_md = false;
        for (auto &entry : deck.main_deck) {
            if (entry.second == card_in) {
                entry.first++;
                found_in_md = true;
                break;
            }
        }
        if (!found_in_md) {
            deck.main_deck.push_back({1, card_in});
        }
        // remove one copy from main deck
        if (deck.main_deck[md_idx].first > 1) {
            deck.main_deck[md_idx].first--;
        } else {
            deck.main_deck.erase(deck.main_deck.begin() + static_cast<long>(md_idx));
        }

        game_log("Swapped out %s for %s\n", card_out.c_str(), card_in.c_str());
    }

    sideboard_phase = false;
}

static std::vector<std::string> split_card_list(const std::string &csv) {
    std::vector<std::string> result;
    size_t start = 0;
    while (start < csv.size()) {
        size_t end = csv.find(',', start);
        if (end == std::string::npos) end = csv.size();
        // trim whitespace
        size_t s = start;
        while (s < end && csv[s] == ' ') s++;
        size_t e = end;
        while (e > s && csv[e - 1] == ' ') e--;
        if (e > s) result.push_back(csv.substr(s, e - s));
        start = end + 1;
    }
    return result;
}

//runs in thread seperate from gui
static void *game_loop(void *args) {

    DEFAULT_DECK_ONE = Deck(RESOURCE_DIR + "/decks/" + deck_a_name + ".dk");
    DEFAULT_DECK_TWO = Deck(RESOURCE_DIR + "/decks/" + deck_b_name + ".dk");

    cli_print_version(VERSION_NUMBER);

    // Setup seed and input logging
    unsigned int seed;
    if (replay_mode) {
        InputLogger::instance().init_replay(replay_file_path);
        seed = InputLogger::instance().get_replay_seed();
    } else {
        seed = seed_override ? seed_value : static_cast<unsigned int>(time(nullptr));
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

    if (bo3_mode) {
        Deck deck_a = DEFAULT_DECK_ONE;
        Deck deck_b = DEFAULT_DECK_TWO;
        bool a_goes_first = true;
        match_wins_a = 0;
        match_wins_b = 0;

        for (int game_num = 0; game_num < 3; game_num++) {
            match_game_number = game_num;
            game_log("\n===== MATCH GAME %d of 3 =====\n", game_num + 1);

            auto sys = init_ecs();
            int winner = play_single_game(sys, deck_a, deck_b, a_goes_first, seed + static_cast<unsigned int>(game_num));

            if (winner == Zone::PLAYER_A) {
                match_wins_a++;
                printf("GAME_RESULT: %d Player A wins\n", game_num + 1);
            } else {
                match_wins_b++;
                printf("GAME_RESULT: %d Player B wins\n", game_num + 1);
            }

            if (match_wins_a == 2) {
                printf("MATCH_RESULT: Player A wins %d-%d\n", match_wins_a, match_wins_b);
                break;
            }
            if (match_wins_b == 2) {
                printf("MATCH_RESULT: Player B wins %d-%d\n", match_wins_a, match_wins_b);
                break;
            }

            // loser goes first next game
            a_goes_first = (winner != Zone::PLAYER_A);

            // swap decks between players
            std::swap(deck_a, deck_b);

            // sideboarding phase - ECS from the just-ended game is still valid
            // (player entities exist for populate_gamestate, card_db works for load_card)
            run_sideboard_phase(deck_a, Zone::PLAYER_A);
            run_sideboard_phase(deck_b, Zone::PLAYER_B);
        }
        match_game_number = -1;
    } else {
        auto sys = init_ecs();
        play_single_game(sys, DEFAULT_DECK_ONE, DEFAULT_DECK_TWO, true, seed);
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
        } else if (std::string(argv[i]) == "--deck" && i + 1 < argc) {
            deck_a_name = argv[i + 1];
            deck_b_name = argv[i + 1];
            i++;
        } else if (std::string(argv[i]) == "--deck-a" && i + 1 < argc) {
            deck_a_name = argv[i + 1];
            i++;
        } else if (std::string(argv[i]) == "--deck-b" && i + 1 < argc) {
            deck_b_name = argv[i + 1];
            i++;
        } else if (std::string(argv[i]) == "--seed" && i + 1 < argc) {
            seed_override = true;
            seed_value = static_cast<unsigned int>(std::stoul(argv[i + 1]));
            i++;
        } else if (std::string(argv[i]) == "--no-shuffle") {
            no_shuffle = true;
        } else if (std::string(argv[i]) == "--battlefield-a" && i + 1 < argc) {
            battlefield_a_cards = split_card_list(argv[i + 1]);
            i++;
        } else if (std::string(argv[i]) == "--battlefield-b" && i + 1 < argc) {
            battlefield_b_cards = split_card_list(argv[i + 1]);
            i++;
        } else if (std::string(argv[i]) == "--narrative") {
            narrative_mode = true;
        } else if (std::string(argv[i]) == "--bo3") {
            bo3_mode = true;
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
