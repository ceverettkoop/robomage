#include "engine_main.h"

#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include "agents/actor.h"
#include "agents/driver.h"
#include "agents/teacher.h"
#include "agents/trajectory.h"
#include "classes/deck.h"
#include "classes/game.h"
#include "cli_args.h"
#include "cli_output.h"
#include "components/zone.h"
#include "deck_check.h"
#include "error.h"
#include "game_driver.h"
#include "input_logger.h"
#include "parse_report.h"
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
        // An RMLOG v3 log is self-contained: the seed, the state-affecting launch
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
        const DecisionLogHeader hdr = match_log_header(DEFAULT_DECK_ONE, DEFAULT_DECK_TWO);
        trajectory_configure_decision_log(seed);
        if (!decision_log_path.empty()) InputLogger::instance().set_log_path(decision_log_path);
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
        const int winner = play_single_game(sys, DEFAULT_DECK_ONE, DEFAULT_DECK_TWO, true, seed);
        report_game_end(1, winner);
    }
    teacher_finish();
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

    // --parse-report: what the parser does not model in cards' scripts; no game is played.
    if (parse_report_requested()) return run_parse_report();

    // --text-table-check / --text-table-inputs: the oracle-text table; no game is played.
    if (text_table_requested()) return run_text_table();

    // --cards-frame: the whole-vocab CARDS frame for the card tables; no game is played.
    if (!cards_frame_path.empty()) return write_cards_frame_file(cards_frame_path);

    // --rmstate-read: an offline stream check; no game is played.
    if (!rmstate_read_path.empty()) return rmstate_read_stream(rmstate_read_path, state_dump_json);

    // --actor: match after match from a schedule, every seat decided in process (agents/actor.h).
    if (actor_mode) return actor_main();
    rmstate_outputs_init();

    game_loop();
    return 0;
}
