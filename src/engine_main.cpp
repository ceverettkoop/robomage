#include "engine_main.h"

#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include "agents/driver.h"
#include "classes/deck.h"
#include "classes/game.h"
#include "cli_args.h"
#include "cli_output.h"
#include "components/zone.h"
#include "deck_check.h"
#include "error.h"
#include "game_driver.h"
#include "input_logger.h"
#include "state_format/cards_file.h"
#include "state_format/recode.h"
#include "state_format/stream.h"
#include "text_table.h"

static void game_loop();

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
        DEFAULT_DECK_ONE = load_launch_deck(Zone::PLAYER_A);
        DEFAULT_DECK_TWO = load_launch_deck(Zone::PLAYER_B);
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
        if (!decision_log_path.empty()) InputLogger::instance().set_log_path(decision_log_path);
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
    agents_install(seed);

    if (bo3_mode) {
        // The full bo3 match sequencing lives in game_driver's play_bo3_match so
        // every in-process driver shares ONE implementation (game boundaries,
        // loser-on-the-play, sideboarding between games).
        play_bo3_match(DEFAULT_DECK_ONE, DEFAULT_DECK_TWO, seed);
    } else {
        auto sys = init_ecs();
        print_game_result(1, play_single_game(sys, DEFAULT_DECK_ONE, DEFAULT_DECK_TWO, true, seed));
    }
}

int engine_main(int argc, char const *argv[]) {
    char buf[FILENAME_MAX];
    RESOURCE_DIR = getcwd(buf, FILENAME_MAX);
    RESOURCE_DIR += "/resources";

    if (parse_command_line(argc, argv) == CliParseResult::SHOW_HELP) {
        cli_print_help(argv[0], VERSION_NUMBER);
        return 0;
    }

    // --check-decks: the decks' cards against the vocabulary; no game is played.
    if (check_decks_flag) return check_decks(check_decks_dirs);

    // --text-table-check / --text-table-inputs: the oracle-text table; no game is played.
    if (text_table_requested()) return run_text_table();

    // --cards-frame: the whole-vocab CARDS frame for the card tables; no game is played.
    if (!cards_frame_path.empty()) return write_cards_frame_file(cards_frame_path);

    // --rmstate-read: an offline stream check; no game is played.
    if (!rmstate_read_path.empty()) return rmstate_read_stream(rmstate_read_path, state_dump_json);
    rmstate_outputs_init();

    game_loop();
    return 0;
}
