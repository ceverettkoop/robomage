#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include "classes/deck.h"
#include "classes/game.h"
#include "classes/match_state.h"
#include "cli_args.h"
#include "cli_output.h"
#include "error.h"
#include "components/zone.h"
#include "game_driver.h"
#include "input_logger.h"
#include "search_server.h"

#ifndef VERSION_NUMBER
#define VERSION_NUMBER "0.2"
#endif

static void game_loop() {

    cli_print_version(VERSION_NUMBER);

    // Setup decks, seed and input logging
    unsigned int seed;
    if (replay_mode) {
        // An RMLOG v2 log is self-contained: the seed, the state-affecting launch
        // flags, and both decklists come from its header. parse_command_line leaves
        // every setup option at its default in replay mode, so the header alone
        // determines the setup.
        InputLogger::instance().init_replay(replay_file_path);
        seed = InputLogger::instance().get_replay_seed();
        const DecisionLogHeader &hdr = InputLogger::instance().get_replay_header();
        DEFAULT_DECK_ONE = InputLogger::instance().get_replay_deck_a();
        DEFAULT_DECK_TWO = InputLogger::instance().get_replay_deck_b();
        deck_a_name = hdr.deck_a_name;
        deck_b_name = hdr.deck_b_name;
        apply_replay_header_flags(InputLogger::instance().get_replay_flags());
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
        record_header_flags(hdr);
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

    if (parse_command_line(argc, argv) == CliParseResult::SHOW_HELP) {
        cli_print_help(argv[0], VERSION_NUMBER);
        return 0;
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
