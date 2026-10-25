#include "input_logger.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "agents/actor.h"
#include "agents/driver.h"
#include "agents/trajectory.h"
#include "card_vocab.h"
#include "classes/game.h"
#include "choice_labels.h"
#include "cli_args.h"
#include "cli_output.h"
#include "components/carddata.h"
#include "components/ability.h"
#include "components/zone.h"
#include "ecs/coordinator.h"
#include "error.h"
#include "game_driver.h"
#include "queries/players.h"
#include "queries/zones.h"
#include "state_format/dump.h"
#include "state_format/selfcheck.h"
#include "state_format/stream.h"
#include "state_format/handles.h"

extern std::string RESOURCE_DIR;
extern Coordinator global_coordinator;
extern Game cur_game;

static Zone::Ownership deciding_player();
static int apply_concede_input(int choice);
static void check_machine_choice(const std::vector<LegalAction> &actions, int choice);
static bool interchangeable_choices(const LegalAction &a, const LegalAction &b);
static std::vector<size_t> distinct_choice_indices(const std::vector<LegalAction> &actions);
static std::vector<std::string> parse_flag_tokens(const std::string &flags_line);
static std::string read_header_field(std::ifstream &file, const std::string &key);
static std::string read_embedded_deck(std::ifstream &file, const std::string &deck_key);
static bool parse_logged_choice(const std::string &token, int *choice);

// The seat a decision belongs to — i.e. who a concession read at it is
// attributed to. Normally the priority holder (every mid-resolution prompt
// seats priority at its chooser, e.g. request_optional_yesno below); between
// games the priority flags are stale, so the sideboarding player owns it.
static Zone::Ownership deciding_player() {
    extern bool sideboard_phase;
    extern Zone::Ownership sideboard_phase_player;
    if (sideboard_phase && sideboard_phase_player != Zone::UNKNOWN) return sideboard_phase_player;
    return priority_seat();
}

// Apply a CONCEDE_GAME / CONCEDE_MATCH sentinel read at the current decision
// (CR 104.3a) and return the index the caller should act on. That is choice 0,
// the cooperative unwind's first-choice hand-back: the game is already decided,
// so the auto-choices below simply carry control back to the main loop, which
// discards the conceded game's suspended state and exits (see game_driver.h).
static int apply_concede_input(int choice) {
    concede_current_game(deciding_player(), choice == CONCEDE_MATCH);
    return concede_unwind_choice();
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

// Split an RMLOG FLAGS line back into flag tokens. Tokens are space-separated,
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
            bool bare_flag = is_bare_header_flag(piece);
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

void InputLogger::set_input_provider(InputProvider provider, bool covers_a, bool covers_b, bool real_decisions) {
    input_provider = std::move(provider);
    provider_covers[0] = covers_a;
    provider_covers[1] = covers_b;
    provider_real_decisions = real_decisions;
}

void InputLogger::clear_input_provider() {
    input_provider = nullptr;
    provider_covers[0] = provider_covers[1] = false;
}

void InputLogger::write_header(unsigned int seed, const DecisionLogHeader &header) {
    log_file << "RMLOG v3\n";
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

void InputLogger::set_log_path(const std::string &path) {
    named_log_path = path;
}

bool InputLogger::open_log(unsigned int seed, const std::string &resource_dir, const DecisionLogHeader &header) {
    log_path = named_log_path.empty() ? resource_dir + "/logs/game_" + std::to_string(seed) + ".log" : named_log_path;
    log_file.open(log_path);
    if (!log_file.is_open()) {
        non_fatal_error("Failed to open log file: " + log_path);
        return false;
    }
    write_header(seed, header);
    return true;
}

void InputLogger::init_logging(unsigned int seed, const std::string &resource_dir, const DecisionLogHeader &header) {
    if (!open_log(seed, resource_dir, header)) return;
    replay_mode = false;
    if (named_log_path.empty()) game_log("Logging inputs to: %s\n", log_path.c_str());
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
    if (line != "RMLOG v3") {
        if (!line.empty() && line.find_first_not_of("0123456789") == std::string::npos) {
            fatal_error("Replay log predates the RMLOG v2 format (first line is a bare seed). "
                        "Old logs record only seed + choices and cannot be replayed self-contained; "
                        "re-record the game with the current engine.");
        }
        if (line == "RMLOG v2")
            fatal_error("Replay log " + replay_path +
                        " is RMLOG v2: it records the one-row menus the engine now takes without asking, so it "
                        "cannot be replayed; re-record the game with the current engine.");
        fatal_error("Unrecognized replay log format: expected 'RMLOG v3', got '" + line + "'");
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
    replay_mode = true;
    game_log("REPLAY MODE: Using seed %u from %s\n", replay_seed, replay_path.c_str());
    game_log("REPLAY MODE: Using embedded decks '%s' / '%s' (command-line deck args ignored)\n",
             replay_header.deck_a_name.c_str(), replay_header.deck_b_name.c_str());
}

void InputLogger::init_machine(unsigned int seed, const std::string &resource_dir, bool enable_log,
                               const DecisionLogHeader &header) {
    machine_mode = true;
    replay_mode = false;
    // Machine mode (agent games, the gate) writes no decision log unless
    // --log-decisions (or --decision-log) is given — agent runs play many games.
    if (!enable_log) return;
    if (!open_log(seed, resource_dir, header)) return;
    if (named_log_path.empty()) game_log("Logging inputs to: %s\n", log_path.c_str());
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

const std::string &InputLogger::log_file_name() const {
    static const std::string none;
    return log_file.is_open() ? log_path : none;
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

// Machine-mode divergence guard: an index outside the live menu means the driver's
// view of this decision has drifted from the engine's (e.g. a search tree replaying
// a stale menu after a world-consistency break). Fail here with full context instead
// of letting the caller index legal_actions out of bounds (which reads garbage and
// segfaults copying it — the exact failure this replaced).
static void check_machine_choice(const std::vector<LegalAction> &actions, int choice) {
    if (choice >= 0 && choice < static_cast<int>(actions.size())) return;
    std::string menu;
    for (size_t i = 0; i < actions.size(); i++)
        menu += "\n  [" + std::to_string(i) + "] " + actions[i].description;
    fatal_error("machine input out of range: got index " + std::to_string(choice) +
                " but menu has " + std::to_string(actions.size()) + " actions (turn=" +
                std::to_string(cur_game.turn_state.turn) + " step=" +
                std::string(step_to_string(cur_game.turn_state.step)) + ")" + menu);
}

// A decision line of the log: a whole-token integer (a menu index or a sentinel).
static bool parse_logged_choice(const std::string &token, int *choice) {
    char *end = nullptr;
    const long v = std::strtol(token.c_str(), &end, 10);
    if (end == token.c_str() || *end != '\0') return false;
    *choice = static_cast<int>(v);
    return true;
}

void InputLogger::write_quit_marker() {
    if (!log_file.is_open()) return;
    log_file << QUIT_MARKER << std::endl;
    log_file.flush();
}

void InputLogger::end_run_by_quit() {
    trajectory_end_run(featurize::EndReason::QUIT);
    write_quit_marker();
    std::fflush(stdout);
    std::exit(0);
}

void InputLogger::end_run_by_signal() {
    write_quit_marker();
    actor_stopped_at_decision();
    std::fflush(stdout);
    std::exit(0);
}

bool InputLogger::begin_match_log(unsigned int seed, const std::string &path, const DecisionLogHeader &header) {
    end_match_log();
    machine_mode = true;
    replay_mode = false;
    decisions_read = 0;
    game_decisions_read = 0;
    last_reply = 0;
    named_log_path = path;
    return open_log(seed, RESOURCE_DIR, header);
}

void InputLogger::end_match_log() {
    if (log_file.is_open()) log_file.close();
    log_file.clear();
}

void InputLogger::begin_game() {
    game_decisions_read = 0;
}

void InputLogger::commit_choice(const std::vector<LegalAction> &actions, int choice) {
    if (log_file.is_open()) {
        log_file << choice << std::endl;
        log_file.flush();
    }
}

int InputLogger::accept_machine_choice(const std::vector<LegalAction> &actions, int choice) {
    if (choice == CONCEDE_GAME || choice == CONCEDE_MATCH) {
        commit_choice(actions, choice);
        last_reply = choice;
        return apply_concede_input(choice);
    }
    if (choice == -1 && !actions.empty()) {
        choice = static_cast<int>(actions.size()) - 1;
    }
    check_machine_choice(actions, choice);
    commit_choice(actions, choice);
    last_reply = choice;
    return choice;
}

// Two menu entries that make the same choice: the same kind of action on interchangeable objects
// (two copies of a card in a graveyard or library — Surgical Extraction targeting one of several
// copies, a search finding one) with the same label and cast / activation variant.
static bool interchangeable_choices(const LegalAction &a, const LegalAction &b) {
    return a.type == b.type && a.category == b.category && a.description == b.description &&
           a.option_ordinal == b.option_ordinal && a.target_entity == b.target_entity &&
           a.use_alt_cost == b.use_alt_cost && a.use_flashback == b.use_flashback &&
           a.use_offspring == b.use_offspring && a.use_escape == b.use_escape && a.impulse_cast == b.impulse_cast &&
           a.without_mana_cost == b.without_mana_cost && a.play_back_face == b.play_back_face &&
           a.cast_back_face == b.cast_back_face && a.companion_to_hand == b.companion_to_hand &&
           a.suspend_action == b.suspend_action && a.card_is_public == b.card_is_public &&
           a.source_entity != b.source_entity && interchangeable_cards(a.source_entity, b.source_entity);
}

// Indices of the menu entries kept when every entry interchangeable with an earlier one is left
// out, in menu order.
static std::vector<size_t> distinct_choice_indices(const std::vector<LegalAction> &actions) {
    std::vector<size_t> kept;
    kept.reserve(actions.size());
    for (size_t i = 0; i < actions.size(); i++) {
        bool duplicate = false;
        for (size_t k : kept)
            if (interchangeable_choices(actions[k], actions[i])) {
                duplicate = true;
                break;
            }
        if (!duplicate) kept.push_back(i);
    }
    return kept;
}

int InputLogger::get_input(const std::vector<LegalAction> &actions) {
    std::vector<size_t> kept = distinct_choice_indices(actions);
    if (kept.size() == actions.size()) return choose(actions);
    std::vector<LegalAction> menu;
    menu.reserve(kept.size());
    for (size_t i : kept) menu.push_back(actions[i]);
    int choice = choose(menu);
    return (choice >= 0 && choice < static_cast<int>(kept.size()))
               ? static_cast<int>(kept[static_cast<size_t>(choice)])
               : choice;
}

int InputLogger::choose(const std::vector<LegalAction> &actions) {
    // Cooperative unwind after a concession (CR 104.3a): the game is already
    // decided, so hand back the first choice at every remaining decision --
    // emitting no query, consuming no stdin, committing nothing -- until control
    // returns to the main loop, which exits on the ended game. Nothing these
    // auto-choices touch can change the outcome (state-based effects early-return
    // once the game has ended). Checked before every input path so a replay
    // consumes exactly the decisions the recorded run committed.
    if (concede_unwind_pending()) return concede_unwind_choice();

    // A menu with one row (after the interchangeable-choice collapse) is taken here and never asked,
    // for every consumer: no frame, no decision number, no decision-log line (its events reach the
    // seat's next frame), and the decision caps do not count it.
    if (actions.size() == 1) {
        agents_auto_decision(actions[0]);
        return 0;
    }

    // --max-game-decisions: a live game past the cap stops before this decision is asked
    // (cap_current_game), and the unwind answers it.
    if (max_game_decisions >= 0 && !sideboard_phase && !cur_game.ended &&
        game_decisions_read >= static_cast<size_t>(max_game_decisions)) {
        cap_current_game(game_decisions_read);
        return concede_unwind_choice();
    }

    // --trace-handles: every viewer's handle table as of this decision (state_format/handles.h).
    if (trace_handles) trace_handle_tables();

    const Zone::Ownership seat = rmstate_deciding_seat();

    // --rmstate-dump: every viewer's state-format frames at this decision (state_format/dump.h),
    // after the bookkeeping every RMSTATE writer shares (state_format/stream.h), then the
    // --selfcheck-state checks of this decision's frames (state_format/selfcheck.h; not for the
    // decisions of a provider that also answers simulated ones, a search agent's).
    rmstate_begin_decision();
    rmstate_dump_decision(actions);
    if (!provider_decides(seat) || provider_real_decisions) selfcheck_state_decision(actions);

    // --max-decisions: the run stops at the first decision past the cap, before anyone decides it.
    if (max_decisions >= 0 && decisions_read >= static_cast<size_t>(max_decisions))
        end_run_at_decision_cap(decisions_read);
    // An actor asked to stop (agents/actor.h): the run ends before this decision is decided.
    if (actor_stop_requested()) end_run_by_signal();
    ++decisions_read;
    if (!sideboard_phase) ++game_decisions_read;

    // The in-process agents' frames for this decision, its --trace-decisions block and the
    // trajectory writer's record (agents/driver.h).
    agents_begin_decision(actions, decisions_read);
    const int choice = read_choice(actions, seat);
    agents_end_decision(choice, last_reply);
    return choice;
}

bool InputLogger::provider_decides(Zone::Ownership seat) const {
    if (!input_provider) return false;
    return (seat == Zone::PLAYER_A && provider_covers[0]) || (seat == Zone::PLAYER_B && provider_covers[1]);
}

int InputLogger::read_choice(const std::vector<LegalAction> &actions, Zone::Ownership seat) {
    if (replay_mode) {
        std::string token;
        if (!(replay_file >> token)) fatal_error("Replay file ended unexpectedly");
        // The recorded session quit at this decision: the replay ends here, with no result.
        if (token == QUIT_MARKER) end_run_at_recorded_quit(replay_decision_no + 1);
        int choice;
        if (!parse_logged_choice(token, &choice))
            fatal_error("Replay file: '" + token + "' where decision " + std::to_string(replay_decision_no + 1) +
                        " was expected");
        replay_decision_no++;
        // A logged concession replays as a concession, from the same seat.
        if (choice == CONCEDE_GAME || choice == CONCEDE_MATCH) {
            game_log("(REPLAY) [T%zu | %s] Input: %d (concede)\n", cur_game.turn_state.turn,
                     step_to_string(cur_game.turn_state.step), choice);
            last_reply = choice;
            return apply_concede_input(choice);
        }
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
        Zone::Ownership priority = priority_seat();
        game_log("(REPLAY) [T%zu | %s | %s] Input: %d\n", cur_game.turn_state.turn,
            step_to_string(cur_game.turn_state.step), player_name(priority).c_str(), choice);
        last_reply = choice;
        return choice;
    }

    // In-process decision hook (an agent or a human seat of the CLI): the choice comes straight
    // from the provider, no frame is written and the engine reads no stdin. The choice goes
    // through accept_machine_choice (concession sentinels, -1 → last-slot remap, commit_choice), as
    // the stream's does, so the decision log stays complete. The provider can consult
    // snapshot_loop_safe() for restore-safety without any extra plumbing.
    if (provider_decides(seat)) return accept_machine_choice(actions, input_provider(seat, actions));
    // Every other seat decides over the RMSTATE stream: its frame, answered by one integer line
    // (state_format/stream.h).
    if (!machine_mode || !rmstate_stream_enabled())
        fatal_error("a decision with no input path (no agent, no human seat, no stream)");
    return accept_machine_choice(actions, rmstate_stream_exchange(actions));
}

bool request_optional_yesno(Zone::Ownership chooser, const std::string& prompt) {
    std::vector<LegalAction> yn = optional_yesno_menu(prompt);
    bool prev_priority = cur_game.priority.player_a_has_priority;
    cur_game.priority.player_a_has_priority = (chooser == Zone::PLAYER_A);
    int choice = InputLogger::instance().get_input(yn);
    cur_game.priority.player_a_has_priority = prev_priority;
    return choice == 1;
}
