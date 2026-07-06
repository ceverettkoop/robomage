#include "input_logger.h"

#include <cstdio>
#include <iostream>

#include "card_vocab.h"
#include "classes/game.h"
#include "cli_output.h"
#include "components/carddata.h"
#include "components/ability.h"
#include "components/zone.h"
#include "ecs/coordinator.h"
#include "error.h"
#include "machine_io.h"

extern std::string RESOURCE_DIR;
extern Coordinator global_coordinator;
extern Game cur_game;

static int get_int_input();
static void record_chosen_action(const std::vector<LegalAction> &actions, int choice);
static std::vector<std::string> parse_flag_tokens(const std::string &flags_line);
static std::string read_header_field(std::ifstream &file, const std::string &key);
static std::string read_embedded_deck(std::ifstream &file, const std::string &deck_key);

static int get_int_input() {
    int c;
    while ((c = getchar()) == ' ' || c == '\t');
    if (c == EOF || c == '\n') return -1;
    if (c == 'q' || c == 'Q') {
        game_log("Quitting.\n");
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

void DecisionLogHeader::add_flag(const std::string &token) {
    if (!flags.empty()) flags += " ";
    flags += token;
}

void DecisionLogHeader::add_flag(const std::string &key, const std::string &value) {
    add_flag(key + "=" + value);
}

void DecisionLogHeader::add_flag_list(const std::string &key, const std::vector<std::string> &values) {
    if (values.empty()) return;
    std::string joined;
    for (const std::string &v : values) {
        if (!joined.empty()) joined += ",";
        joined += v;
    }
    add_flag(key, joined);
}

// Split an RMLOG v2 FLAGS line back into flag tokens. Tokens are space-separated,
// but a key=value token's value may itself contain spaces (card names in the
// preset-zone lists, e.g. battlefield-b=Grizzly Bears,Mountain). A fragment
// without '=' that is not a known bare flag is therefore a continuation of the
// previous token, and is glued back on with the space it was split at.
static std::vector<std::string> parse_flag_tokens(const std::string &flags_line) {
    std::vector<std::string> tokens;
    if (flags_line.empty() || flags_line == "-") return tokens;
    size_t start = 0;
    while (start <= flags_line.size()) {
        size_t end = flags_line.find(' ', start);
        if (end == std::string::npos) end = flags_line.size();
        std::string piece = flags_line.substr(start, end - start);
        if (!piece.empty()) {
            bool bare_flag = (piece == "no-shuffle" || piece == "bo3" || piece == "machine");
            bool has_key = piece.find('=') != std::string::npos;
            if (tokens.empty() || bare_flag || has_key) {
                tokens.push_back(piece);
            } else {
                tokens.back() += " " + piece;
            }
        }
        start = end + 1;
    }
    return tokens;
}

// Read one header line and strip a required "KEY " prefix (or the bare keyword
// itself), fatal-erroring with context when the log doesn't match the v2 layout.
static std::string read_header_field(std::ifstream &file, const std::string &key) {
    std::string line;
    if (!std::getline(file, line)) {
        fatal_error("Replay log ended early: missing " + key + " header line");
    }
    if (line == key) return "";
    if (line.rfind(key + " ", 0) != 0) {
        fatal_error("Replay log header: expected '" + key + " ...', got '" + line + "'");
    }
    return line.substr(key.size() + 1);
}

// Accumulate verbatim decklist lines up to the END_DECK sentinel.
static std::string read_embedded_deck(std::ifstream &file, const std::string &deck_key) {
    std::string text;
    std::string line;
    while (true) {
        if (!std::getline(file, line)) {
            fatal_error("Replay log ended early: no END_DECK after " + deck_key);
        }
        if (line == "END_DECK") return text;
        text += line + "\n";
    }
}

InputLogger &InputLogger::instance() {
    static InputLogger logger;
    return logger;
}

void InputLogger::write_header(unsigned int seed, const DecisionLogHeader &header) {
    log_file << "RMLOG v2\n";
    log_file << "SEED " << seed << "\n";
    log_file << "FLAGS " << (header.flags.empty() ? "-" : header.flags) << "\n";
    log_file << "DECK_A " << header.deck_a_name << "\n";
    log_file << header.deck_a_text;
    if (header.deck_a_text.empty() || header.deck_a_text.back() != '\n') log_file << "\n";
    log_file << "END_DECK\n";
    log_file << "DECK_B " << header.deck_b_name << "\n";
    log_file << header.deck_b_text;
    if (header.deck_b_text.empty() || header.deck_b_text.back() != '\n') log_file << "\n";
    log_file << "END_DECK\n";
    log_file << "DECISIONS" << std::endl;
    log_file.flush();
}

void InputLogger::init_logging(unsigned int seed, const std::string &resource_dir, const DecisionLogHeader &header) {
    log_path = resource_dir + "/logs/game_" + std::to_string(seed) + ".log";
    log_file.open(log_path);
    if (!log_file.is_open()) {
        non_fatal_error("Failed to open log file: " + log_path);
        return;
    }
    write_header(seed, header);
    replay_mode = false;
    game_log("Logging inputs to: %s\n", log_path.c_str());
}

void InputLogger::init_replay(const std::string &replay_path) {
    replay_file.open(replay_path);
    if (!replay_file.is_open()) {
        fatal_error("Failed to open replay file: " + replay_path);
    }
    std::string line;
    if (!std::getline(replay_file, line)) {
        fatal_error("Replay file is empty: " + replay_path);
    }
    if (line != "RMLOG v2") {
        if (!line.empty() && line.find_first_not_of("0123456789") == std::string::npos) {
            fatal_error("Replay log predates the RMLOG v2 format (first line is a bare seed). "
                        "Old logs record only seed + choices and cannot be replayed self-contained; "
                        "re-record the game with the current engine.");
        }
        fatal_error("Unrecognized replay log format: expected 'RMLOG v2', got '" + line + "'");
    }
    std::string seed_str = read_header_field(replay_file, "SEED");
    if (seed_str.empty() || seed_str.find_first_not_of("0123456789") != std::string::npos) {
        fatal_error("Replay log header: SEED value is not a number: '" + seed_str + "'");
    }
    replay_seed = static_cast<unsigned int>(std::stoul(seed_str));
    replay_header.flags = read_header_field(replay_file, "FLAGS");
    replay_flags = parse_flag_tokens(replay_header.flags);
    replay_header.deck_a_name = read_header_field(replay_file, "DECK_A");
    replay_header.deck_a_text = read_embedded_deck(replay_file, "DECK_A");
    replay_header.deck_b_name = read_header_field(replay_file, "DECK_B");
    replay_header.deck_b_text = read_embedded_deck(replay_file, "DECK_B");
    read_header_field(replay_file, "DECISIONS");
    replay_deck_a = Deck::from_string(replay_header.deck_a_text);
    replay_deck_b = Deck::from_string(replay_header.deck_b_text);
    // A machine-mode log's decision schedule differs from the interactive one (auto-paid
    // mana, hidden mana-source menu entries, ...); replaying it must follow the machine
    // schedule or the logged indices land on the wrong menus.
    for (const std::string &tok : replay_flags) {
        if (tok == "machine") replay_machine_schedule = true;
    }
    replay_mode = true;
    game_log("REPLAY MODE: Using seed %u from %s\n", replay_seed, replay_path.c_str());
    game_log("REPLAY MODE: Using embedded decks '%s' / '%s' (command-line deck args ignored)\n",
             replay_header.deck_a_name.c_str(), replay_header.deck_b_name.c_str());
}

void InputLogger::init_machine(unsigned int seed, const std::string &resource_dir, bool enable_log,
                               const DecisionLogHeader &header) {
    machine_mode = true;
    replay_mode = false;
    // Machine mode (RL training / test harness) writes no decision log unless
    // --log-decisions is given — training runs millions of episodes.
    if (!enable_log) return;
    log_path = resource_dir + "/logs/game_" + std::to_string(seed) + ".log";
    log_file.open(log_path);
    if (!log_file.is_open()) {
        non_fatal_error("Failed to open log file: " + log_path);
        return;
    }
    write_header(seed, header);
    game_log("Logging inputs to: %s\n", log_path.c_str());
}

bool InputLogger::is_replay_mode() const {
    return replay_mode;
}
bool InputLogger::is_machine_mode() const {
    return machine_mode;
}
bool InputLogger::is_machine_schedule() const {
    return machine_mode || replay_machine_schedule;
}

unsigned int InputLogger::get_replay_seed() const {
    return replay_seed;
}

const DecisionLogHeader &InputLogger::get_replay_header() const {
    return replay_header;
}

const Deck &InputLogger::get_replay_deck_a() const {
    return replay_deck_a;
}

const Deck &InputLogger::get_replay_deck_b() const {
    return replay_deck_b;
}

const std::vector<std::string> &InputLogger::get_replay_flags() const {
    return replay_flags;
}

static void record_chosen_action(const std::vector<LegalAction> &actions, int choice) {
    if (choice < 0 || choice >= static_cast<int>(actions.size())) return;
    const LegalAction &la = actions[static_cast<size_t>(choice)];
    int cat = static_cast<int>(la.category);
    // Same vocab-index chain populate_query emits for this action, so the logged id
    // matches the queried id (previously this omitted the Permanent/token and
    // ability-source-permanent cases, drifting from the BQUERY payload).
    int vocab = action_card_vocab_idx(la);
    cur_game.record_action(cat, vocab, cur_game.player_a_has_priority);
}

void InputLogger::commit_choice(const std::vector<LegalAction> &actions, int choice) {
    if (log_file.is_open()) {
        log_file << choice << std::endl;
        log_file.flush();
    }
    record_chosen_action(actions, choice);
}

int InputLogger::get_input(const std::vector<LegalAction> &actions) {
    extern bool has_human_player;
    extern bool human_player_is_a;
    bool human_has_priority = has_human_player && (human_player_is_a == cur_game.player_a_has_priority);

    if (replay_mode) {
        int choice;
        if (!(replay_file >> choice)) {
            fatal_error("Replay file ended unexpectedly");
        }
        replay_decision_no++;
        // Remap -1 (old confirm sentinel) to last slot for old replay file compatibility
        if (choice == -1 && !actions.empty()) {
            choice = static_cast<int>(actions.size()) - 1;
        }
        // Divergence guard: a logged index outside the live menu means the replayed game
        // has drifted from the recorded one — fail here with context instead of letting
        // the caller index out of bounds.
        if (choice < 0 || choice >= static_cast<int>(actions.size())) {
            fatal_error("replay diverged at decision " + std::to_string(replay_decision_no) +
                        ": logged index " + std::to_string(choice) + " but menu has " +
                        std::to_string(actions.size()) + " actions");
        }
        Zone::Ownership priority = cur_game.player_a_has_priority ? Zone::PLAYER_A : Zone::PLAYER_B;
        game_log("(REPLAY) [T%zu | %s | %s] Input: %d\n", cur_game.turn, step_to_string(cur_game.cur_step),
            player_name(priority).c_str(), choice);
        record_chosen_action(actions, choice);
        return choice;
    }

    if (machine_mode && !human_has_priority) {
        extern bool sideboard_phase;
        extern Zone::Ownership sideboard_phase_player;
        GameState gs;
        Query q;
        // During sideboarding, priority/active-player flags don't reflect who is
        // sideboarding. Use the sideboard player as the viewer so obs[32]
        // (self_is_player_a) correctly identifies the sideboarding side.
        Zone::Ownership viewer = sideboard_phase ? sideboard_phase_player : Zone::UNKNOWN;
        populate_gamestate(&gs, viewer);
        populate_query(&q, actions);
        cli_emit_machine_query(&q, &gs);

        int choice = -1;
        if (scanf("%d", &choice) != 1) choice = -1;
        int c;
        while ((c = getchar()) != '\n' && c != EOF);

        // Remap -1 (confirm sentinel from Python env) to last slot
        if (choice == -1 && !actions.empty()) {
            choice = static_cast<int>(actions.size()) - 1;
        }
        commit_choice(actions, choice);
        return choice;
    }

    // Auto-pass mode: return 0 until we reach the target turn
    if (auto_pass_until_turn >= 0) {
        if (cur_game.cur_step == DECLARE_ATTACKERS) {
            auto_pass_until_turn = -1;
        } else if ((int)cur_game.turn < auto_pass_until_turn) {
            commit_choice(actions, 0);
            return 0;
        }
        auto_pass_until_turn = -1;
    }

    // CLI input loop: print query, validate, log
    Zone::Ownership viewer = (has_human_player)
        ? (human_player_is_a ? Zone::PLAYER_A : Zone::PLAYER_B)
        : Zone::UNKNOWN;
    GameState gs;
    Query q;
    while (true) {
        populate_gamestate(&gs, viewer);
        populate_query(&q, actions);
        print_query(&q, cur_game.player_a_has_priority);
        int choice = get_int_input();
        // handling flagged choices
        switch (choice) {
            case PASS_TURN_CMD:
                game_log("Auto-passing turn.\n");
                auto_pass_until_turn = (int)cur_game.turn + 1;
                commit_choice(actions, 0);
                return 0;
                break;
            case FLAG_QUIT:
                exit(0);
                break;
            default:
                break;
        }
        // other input checked against choices
        if (choice < 0 || choice >= static_cast<int>(actions.size())) {
            cli_print_invalid_action();
            continue;
        }
        commit_choice(actions, choice);
        return choice;
    }
}

bool request_optional_yesno(Zone::Ownership chooser, const std::string& prompt) {
    std::vector<LegalAction> yn;
    LegalAction decline(PASS_PRIORITY, std::string("Decline: ") + prompt);
    decline.category = ActionCategory::OPTIONAL_YESNO;
    yn.push_back(decline);
    LegalAction accept(PASS_PRIORITY, std::string("Accept: ") + prompt);
    accept.category = ActionCategory::OPTIONAL_YESNO;
    yn.push_back(accept);
    bool prev_priority = cur_game.player_a_has_priority;
    cur_game.player_a_has_priority = (chooser == Zone::PLAYER_A);
    int choice = InputLogger::instance().get_input(yn);
    cur_game.player_a_has_priority = prev_priority;
    return choice == 1;
}
