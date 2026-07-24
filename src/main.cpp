#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <unordered_set>

#include "classes/deck.h"
#include "classes/game.h"
#include "classes/match_state.h"
#include "cli_output.h"
#include "error.h"
#include "components/zone.h"
#include "game_driver.h"
#include "input_logger.h"
#include "search_server.h"

#ifndef VERSION_NUMBER
#define VERSION_NUMBER "0.2"
#endif

static std::vector<std::string> split_card_list(const std::string &csv);

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

static void game_loop() {

    cli_print_version(VERSION_NUMBER);

    // Setup decks, seed and input logging
    unsigned int seed;
    if (replay_mode) {
        // An RMLOG v2 log is self-contained: the seed, the state-affecting launch
        // flags, and both decklists come from its header. Embedded values override
        // anything given on the command line (--deck-a/--deck-b are ignored).
        InputLogger::instance().init_replay(replay_file_path);
        seed = InputLogger::instance().get_replay_seed();
        const DecisionLogHeader &hdr = InputLogger::instance().get_replay_header();
        DEFAULT_DECK_ONE = InputLogger::instance().get_replay_deck_a();
        DEFAULT_DECK_TWO = InputLogger::instance().get_replay_deck_b();
        deck_a_name = hdr.deck_a_name;
        deck_b_name = hdr.deck_b_name;
        for (const std::string &tok : InputLogger::instance().get_replay_flags()) {
            std::string key = tok;
            std::string value;
            size_t eq = tok.find('=');
            if (eq != std::string::npos) {
                key = tok.substr(0, eq);
                value = tok.substr(eq + 1);
            }
            if (key == "no-shuffle") {
                no_shuffle = true;
            } else if (key == "machine") {
                // decision-schedule marker; consumed by InputLogger::init_replay
            } else if (key == "bo3") {
                bo3_mode = true;
            } else if (key == "battlefield-a") {
                battlefield_a_cards = split_card_list(value);
            } else if (key == "battlefield-b") {
                battlefield_b_cards = split_card_list(value);
            } else if (key == "graveyard-a") {
                graveyard_a_cards = split_card_list(value);
            } else if (key == "graveyard-b") {
                graveyard_b_cards = split_card_list(value);
            } else if (key == "exile-a") {
                exile_a_cards = split_card_list(value);
            } else if (key == "exile-b") {
                exile_b_cards = split_card_list(value);
            } else if (key == "sideboard-a") {
                sideboard_a_cards = split_card_list(value);
            } else if (key == "sideboard-b") {
                sideboard_b_cards = split_card_list(value);
            } else if (key == "life-a") {
                life_a_override = std::stoi(value);
            } else if (key == "life-b") {
                life_b_override = std::stoi(value);
            } else {
                fatal_error("Replay log FLAGS: unknown flag '" + tok + "'");
            }
        }
    } else {
        DEFAULT_DECK_ONE = Deck(RESOURCE_DIR + "/decks/" + deck_a_name + ".dk");
        DEFAULT_DECK_TWO = Deck(RESOURCE_DIR + "/decks/" + deck_b_name + ".dk");
        seed = seed_override ? seed_value : static_cast<unsigned int>(time(nullptr));
        // Machine mode only logs decisions when --log-decisions is given; only
        // create the logs directory when a log will actually be written.
        if (!machine_mode || log_decisions_flag) {
            std::string mkdir_cmd = "mkdir -p " + RESOURCE_DIR + "/logs";
            int result = system(mkdir_cmd.c_str());
            (void)result;  // Ignore return value
        }
        // v2 log header: record everything that shapes the initial game state so
        // the log replays without re-supplying decks or flags.
        DecisionLogHeader hdr;
        if (no_shuffle) hdr.add_flag("no-shuffle");
        if (bo3_mode) hdr.add_flag("bo3");
        // Machine mode's decision schedule differs from the interactive one; mark it so
        // --replay knows to follow the machine schedule (InputLogger::is_machine_schedule).
        if (machine_mode) hdr.add_flag("machine");
        hdr.add_flag_list("battlefield-a", battlefield_a_cards);
        hdr.add_flag_list("battlefield-b", battlefield_b_cards);
        hdr.add_flag_list("graveyard-a", graveyard_a_cards);
        hdr.add_flag_list("graveyard-b", graveyard_b_cards);
        hdr.add_flag_list("exile-a", exile_a_cards);
        hdr.add_flag_list("exile-b", exile_b_cards);
        hdr.add_flag_list("sideboard-a", sideboard_a_cards);
        hdr.add_flag_list("sideboard-b", sideboard_b_cards);
        if (life_a_override >= 0) hdr.add_flag("life-a", std::to_string(life_a_override));
        if (life_b_override >= 0) hdr.add_flag("life-b", std::to_string(life_b_override));
        hdr.deck_a_name = deck_a_name;
        hdr.deck_a_text = DEFAULT_DECK_ONE.raw_text;
        hdr.deck_b_name = deck_b_name;
        hdr.deck_b_text = DEFAULT_DECK_TWO.raw_text;
        if (machine_mode) {
            InputLogger::instance().init_machine(seed, RESOURCE_DIR, log_decisions_flag, hdr);
        } else {
            InputLogger::instance().init_logging(seed, RESOURCE_DIR, hdr);
        }
    }
    std::srand(seed);
    cli_print_seed(seed);

    if (bo3_mode) {
        // The full bo3 match sequencing lives in game_driver's play_bo3_match so
        // main.cpp and bin/az_actor share ONE implementation (game boundaries,
        // loser-on-the-play, revealed accumulator, sideboarding between games).
        play_bo3_match(DEFAULT_DECK_ONE, DEFAULT_DECK_TWO, seed);
    } else {
        match_reset_revealed();  // accumulator works in single-game mode too
        auto sys = init_ecs();
        play_single_game(sys, DEFAULT_DECK_ONE, DEFAULT_DECK_TWO, true, seed);
    }
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
        } else if (std::string(argv[i]) == "--search-server") {
            // MCTS search protocol (SNAPSHOT/RESTORE/DETERMINIZE/RELEASE); a
            // machine-mode extension, so it implies --machine.
            search_server_mode = true;
            machine_mode = true;
        } else if (std::string(argv[i]) == "--player" && i + 1 < argc) {
            has_human_player = true;
            std::string p = argv[i + 1];
            human_player_is_a = (p == "A" || p == "a");
            i++;
        } else if (std::string(argv[i]) == "--log-viewer" && i + 1 < argc) {
            // Narrative-only viewer: redact game_log_private to one seat's view
            // without setting has_human_player (input stays on the machine path).
            log_viewer_set = true;
            std::string p = argv[i + 1];
            log_viewer_owner = (p == "A" || p == "a") ? Zone::PLAYER_A : Zone::PLAYER_B;
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
        } else if (std::string(argv[i]) == "--graveyard-a" && i + 1 < argc) {
            graveyard_a_cards = split_card_list(argv[i + 1]);
            i++;
        } else if (std::string(argv[i]) == "--graveyard-b" && i + 1 < argc) {
            graveyard_b_cards = split_card_list(argv[i + 1]);
            i++;
        } else if (std::string(argv[i]) == "--exile-a" && i + 1 < argc) {
            exile_a_cards = split_card_list(argv[i + 1]);
            i++;
        } else if (std::string(argv[i]) == "--exile-b" && i + 1 < argc) {
            exile_b_cards = split_card_list(argv[i + 1]);
            i++;
        } else if (std::string(argv[i]) == "--sideboard-a" && i + 1 < argc) {
            sideboard_a_cards = split_card_list(argv[i + 1]);
            i++;
        } else if (std::string(argv[i]) == "--sideboard-b" && i + 1 < argc) {
            sideboard_b_cards = split_card_list(argv[i + 1]);
            i++;
        } else if (std::string(argv[i]) == "--life-a" && i + 1 < argc) {
            life_a_override = std::stoi(argv[i + 1]);
            i++;
        } else if (std::string(argv[i]) == "--life-b" && i + 1 < argc) {
            life_b_override = std::stoi(argv[i + 1]);
            i++;
        } else if (std::string(argv[i]) == "--narrative") {
            narrative_mode = true;
        } else if (std::string(argv[i]) == "--log-decisions") {
            log_decisions_flag = true;
        } else if (std::string(argv[i]) == "--broadcast-steps") {
            // Passive BSTATE frames at forced auto-pass windows (machine mode
            // observers only; no stdin response expected, nothing recorded).
            broadcast_steps_mode = true;
        } else if (std::string(argv[i]) == "--bo3") {
            bo3_mode = true;
        } else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            cli_print_help(argv[0], VERSION_NUMBER);
            return 0;
        }
    }

    // Search commands would interleave into (or replay out of sync with) the
    // deterministic decision record, and the cooperative unwind cannot route
    // through a human prompt.
    if (search_server_mode && (replay_mode || log_decisions_flag || has_human_player)) {
        fatal_error("--search-server is incompatible with --replay, --log-decisions, and --player");
    }

    game_loop();
    return 0;
}
