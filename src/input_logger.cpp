#include "input_logger.h"
#include <pthread.h>

#include <cstdio>
#include <iostream>

#include "card_vocab.h"
#include "classes/game.h"
#include "components/carddata.h"
#include "components/permanent.h"
#include "components/zone.h"
#include "debug.h"
#include "ecs/coordinator.h"
#include "error.h"
#include "machine_io.h"

extern std::string RESOURCE_DIR;
extern Coordinator global_coordinator;
extern Game cur_game;
extern bool gui_mode;
extern bool gui_input_requested;
extern bool gui_input_sent;
extern int gui_cmd;
extern pthread_mutex_t input_mutex;
extern bool gui_killed;

// helper migrated from cli.cpp
#define PASS_TURN_CMD (-2)

static int get_int_input() {
    if (gui_mode) {
        gui_input_requested = true;
        while (!gui_input_sent) {
            if(gui_killed){
                printf("User exited GUI, quitting\n");
                exit(0);
            }
        }
        pthread_mutex_lock(&input_mutex);
        gui_input_sent = false;
        pthread_mutex_unlock(&input_mutex);
        gui_input_requested = false;
        return gui_cmd;
    } else {
        int c;
        while ((c = getchar()) == ' ' || c == '\t');
        if (c == EOF || c == '\n') return -1;
        if (c == 'q' || c == 'Q') {
            printf("Quitting.\n");
            exit(0);
        }
        if (c == 'z' || c == 'Z') {
            while ((c = getchar()) != '\n' && c != EOF);
            return PASS_TURN_CMD;
        }
        ungetc(c, stdin);
        int choice = -1;
        if (scanf("%d", &choice) != 1) {
            while ((c = getchar()) != '\n' && c != EOF);
            return -1;
        }
        while ((c = getchar()) != '\n' && c != EOF);
        return choice;
    }
}

InputLogger &InputLogger::instance() {
    static InputLogger logger;
    return logger;
}

void InputLogger::init_logging(unsigned int seed, const std::string &resource_dir) {
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

void InputLogger::init_replay(const std::string &replay_path) {
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

void InputLogger::init_machine(unsigned int seed, const std::string &resource_dir) {
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

bool InputLogger::is_replay_mode() const {
    return replay_mode;
}
bool InputLogger::is_machine_mode() const {
    return machine_mode;
}

unsigned int InputLogger::get_replay_seed() const {
    return replay_seed;
}

int InputLogger::get_logged_input(size_t cur_turn, const std::vector<LegalAction>& actions) {
    if (replay_mode) {
        int choice;
        if (!(replay_file >> choice)) {
            fatal_error("Replay file ended unexpectedly");
        }
        Zone::Ownership priority = cur_game.player_a_has_priority ? Zone::PLAYER_A : Zone::PLAYER_B;
        printf("(REPLAY) [T%zu | %s | %s] Input: %d\n", cur_game.turn, step_to_string(cur_game.cur_step),
            player_name(priority).c_str(), choice);
        return choice;
    }

    if (machine_mode) {
        GameState gs;
        Query q;
        populate_gamestate(&gs);
        populate_query(&q, actions);

        auto state_vec = serialize_state(&gs);
        printf("QUERY: %d", q.num_choices);
        for (float f : state_vec) printf(" %.4f", f);
        for (int i = 0; i < q.num_choices; i++) printf(" %d", q.choices[i].category);
        const float id_null = -1.0f / static_cast<float>(N_CARD_TYPES);
        for (int i = 0; i < q.num_choices; i++) {
            float id_f = q.choices[i].card_vocab_idx >= 0
                ? static_cast<float>(q.choices[i].card_vocab_idx) / static_cast<float>(N_CARD_TYPES)
                : id_null;
            printf(" %.4f", id_f);
        }
        const float ctrl_null = -1.0f / static_cast<float>(N_CARD_TYPES);
        for (int i = 0; i < q.num_choices; i++) {
            float ctrl = (q.choices[i].zone_ref != REF_NONE)
                ? (q.choices[i].controller_is_self ? 1.0f : 0.0f)
                : ctrl_null;
            printf(" %.4f", ctrl);
        }
        printf("\n");
        fflush(stdout);

        int choice = -1;
        if (scanf("%d", &choice) != 1) choice = -1;
        int c;
        while ((c = getchar()) != '\n' && c != EOF);

        if (log_file.is_open()) {
            log_file << choice << std::endl;
            log_file.flush();
        }
        return choice;
    }

    // Auto-pass mode: return 0 until we reach the target turn
    if (auto_pass_until_turn >= 0) {
        if (cur_game.cur_step == DECLARE_ATTACKERS) {
            auto_pass_until_turn = -1;
        } else if ((int)cur_turn < auto_pass_until_turn) {
            if (log_file.is_open()) {
                log_file << 0 << std::endl;
                log_file.flush();
            }
            return 0;
        }
        auto_pass_until_turn = -1;
    }

    // Typical CLI input
    int choice = get_int_input();
    if (choice == PASS_TURN_CMD) {
        printf("Auto-passing turn.\n");
        auto_pass_until_turn = (int)cur_turn + 1;
        if (log_file.is_open()) {
            log_file << 0 << std::endl;
            log_file.flush();
        }
        return 0;
    }
    if (choice != -1 && log_file.is_open()) {
        log_file << choice << std::endl;
        log_file.flush();
    }
    return choice;
}

int InputLogger::get_logged_input(
    size_t cur_turn, const std::vector<ActionCategory>& action_categories, const std::vector<Entity>& entities) {
    std::vector<LegalAction> actions;
    actions.reserve(action_categories.size());
    for (size_t i = 0; i < action_categories.size(); i++) {
        Entity e = (i < entities.size()) ? entities[i] : static_cast<Entity>(0);
        LegalAction la(PASS_PRIORITY, e, "");
        la.category = action_categories[i];
        actions.push_back(la);
    }
    return get_logged_input(cur_turn, actions);
}
