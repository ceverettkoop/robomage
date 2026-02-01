#include "input_logger.h"

#include <cstdio>
#include <iostream>

#include "cli.h"
#include "error.h"

extern std::string RESOURCE_DIR;

InputLogger& InputLogger::instance() {
    static InputLogger logger;
    return logger;
}

void InputLogger::init_logging(unsigned int seed, const std::string& resource_dir) {
    log_path = resource_dir + "/logs/game_" + std::to_string(seed) + ".log";
    log_file.open(log_path);
    if (!log_file.is_open()) {
        non_fatal_error("Failed to open log file: " + log_path);
        return;
    }
    // First line is seed
    log_file << seed << std::endl;
    log_file.flush();
    replay_mode = false;
    printf("Logging inputs to: %s\n", log_path.c_str());
}

void InputLogger::init_replay(const std::string& replay_path) {
    replay_file.open(replay_path);
    if (!replay_file.is_open()) {
        fatal_error("Failed to open replay file: " + replay_path);
    }
    // Read seed from first line
    if (!(replay_file >> replay_seed)) {
        fatal_error("Failed to read seed from replay file");
    }
    replay_mode = true;
    printf("REPLAY MODE: Using seed %u from %s\n", replay_seed, replay_path.c_str());
}

bool InputLogger::is_replay_mode() const { return replay_mode; }

unsigned int InputLogger::get_replay_seed() const { return replay_seed; }

int InputLogger::get_logged_input() {
    if (replay_mode) {
        int choice;
        if (!(replay_file >> choice)) {
            fatal_error("Replay file ended unexpectedly");
        }
        printf("(REPLAY) Input: %d\n", choice);
        return choice;
    } else {
        int choice = get_int_input();
        if (choice != -1 && log_file.is_open()) {
            log_file << choice << std::endl;
            log_file.flush();  // Ensure immediate write for crash safety
        }
        return choice;
    }
}
