#include "input_logger.h"

#include <cstdio>
#include <iostream>

#include "card_vocab.h"
#include "cli.h"
#include "components/carddata.h"
#include "ecs/coordinator.h"
#include "error.h"
#include "classes/game.h"
#include "machine_io.h"

extern std::string RESOURCE_DIR;
extern Coordinator global_coordinator;

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

void InputLogger::init_machine(unsigned int seed, const std::string& resource_dir) {
    log_path = resource_dir + "/logs/game_" + std::to_string(seed) + ".log";
    log_file.open(log_path);
    if (!log_file.is_open()) {
        non_fatal_error("Failed to open log file: " + log_path);
    } else {
        log_file << seed << std::endl;
        log_file.flush();
    }
    machine_mode = true;
    replay_mode = false;
}

bool InputLogger::is_replay_mode() const { return replay_mode; }
bool InputLogger::is_machine_mode() const { return machine_mode; }

unsigned int InputLogger::get_replay_seed() const { return replay_seed; }

int InputLogger::get_logged_input(size_t cur_turn,
                                   const std::vector<ActionCategory>& action_categories,
                                   const std::vector<Entity>& entities) {
    int num_choices = static_cast<int>(action_categories.size());

    if (replay_mode) {
        int choice;
        if (!(replay_file >> choice)) {
            fatal_error("Replay file ended unexpectedly");
        }
        printf("(REPLAY) Input: %d\n", choice);
        return choice;
    }

    if (machine_mode) {
        // Emit QUERY line:
        //   "QUERY: <num_choices> <f0>...<f1152> <cat0>...<catN-1> <id0>...<idN-1>"
        // State vector is followed by one ActionCategory int per legal action,
        // then one normalized card vocab index float per action slot.
        std::vector<float> state = serialize_state();
        printf("QUERY: %d", num_choices);
        for (float f : state) printf(" %.4f", f);
        for (ActionCategory cat : action_categories) printf(" %d", static_cast<int>(cat));
        for (int i = 0; i < num_choices; i++) {
            int idx = -1;
            auto ui = static_cast<size_t>(i);
            if (ui < entities.size() && entities[ui] != 0 &&
                global_coordinator.entity_has_component<CardData>(entities[ui])) {
                idx = card_name_to_index(
                    global_coordinator.GetComponent<CardData>(entities[ui]).name);
            }
            printf(" %.4f", idx / static_cast<float>(N_CARD_TYPES));
        }
        printf("\n");
        fflush(stdout);

        // Read response integer from stdin
        int choice = -1;
        if (scanf("%d", &choice) != 1) choice = -1;
        // consume rest of line
        int c;
        while ((c = getchar()) != '\n' && c != EOF);

        if (log_file.is_open()) { log_file << choice << std::endl; log_file.flush(); }
        return choice;
    }

    // Auto-pass mode: return 0 and log it until we reach the target turn,
    // but halt at Declare Attackers so the player can act.
    if (auto_pass_until_turn >= 0) {
        if (cur_game.cur_step == DECLARE_ATTACKERS) {
            auto_pass_until_turn = -1;
        } else if ((int)cur_turn < auto_pass_until_turn) {
            if (log_file.is_open()) { log_file << 0 << std::endl; log_file.flush(); }
            return 0;
        }
        auto_pass_until_turn = -1;
    }

    int choice = get_int_input();
    if (choice == PASS_TURN_CMD) {
        printf("Auto-passing turn.\n");
        auto_pass_until_turn = (int)cur_turn + 1;
        if (log_file.is_open()) { log_file << 0 << std::endl; log_file.flush(); }
        return 0;
    }
    if (choice != -1 && log_file.is_open()) {
        log_file << choice << std::endl;
        log_file.flush();
    }
    return choice;
}
