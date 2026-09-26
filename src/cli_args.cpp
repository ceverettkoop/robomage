#include "cli_args.h"

#include <cstring>

#include "cli_output.h"
#include "components/zone.h"
#include "error.h"
#include "game_driver.h"
#include "search_server.h"

// One command-line option. The option table below is the single definition of which flags
// exist, which take a value, which shape the game's setup, and which the log header records.
struct CliOption {
    const char *name;   // spelling on the command line, e.g. "--deck-a"
    const char *alias;  // alternate spelling, or nullptr
    bool takes_value;
    // Shapes the game's initial state. Under --replay these come only from the log header, so a
    // command-line occurrence is ignored.
    bool setup;
    // Stores the option in its global (value is nullptr for a flag that takes none); nullptr for
    // --help, which the parser handles itself.
    void (*apply)(const char *value);
    // Writes the option's RMLOG v2 FLAGS token under `key` (the name without its leading "--")
    // when the option is in effect; nullptr for options the FLAGS line does not carry.
    void (*record)(DecisionLogHeader &header, const std::string &key);
};

// An option found on the command line, with its value (nullptr for a flag that takes none).
struct ParsedOption {
    const CliOption *option;
    const char *value;
};

static std::vector<std::string> split_card_list(const std::string &csv);
static bool is_seat_a(const char *value);
static const CliOption *find_option(const std::string &name);
static const CliOption *find_header_option(const std::string &key);
static std::string header_key(const CliOption &option);
static void apply_parsed(const std::vector<ParsedOption> &parsed, bool setup);
static void warn_ignored_in_replay(const std::vector<ParsedOption> &parsed);

// Options carried in the FLAGS line are listed first, in the order they are written there.
static const CliOption OPTIONS[] = {
    {"--no-shuffle", nullptr, false, true, [](const char *) { no_shuffle = true; },
        [](DecisionLogHeader &h, const std::string &k) {
            if (no_shuffle) h.add_flag(k);
        }},
    {"--bo3", nullptr, false, true, [](const char *) { bo3_mode = true; },
        [](DecisionLogHeader &h, const std::string &k) {
            if (bo3_mode) h.add_flag(k);
        }},
    // Machine mode's decision schedule differs from the interactive one; the "machine" marker
    // tells --replay to follow the machine schedule (InputLogger::is_machine_schedule). In replay
    // mode the flag itself only affects output.
    {"--machine", nullptr, false, false, [](const char *) { machine_mode = true; },
        [](DecisionLogHeader &h, const std::string &k) {
            if (machine_mode) h.add_flag(k);
        }},
    {"--battlefield-a", nullptr, true, true, [](const char *v) { battlefield_a_cards = split_card_list(v); },
        [](DecisionLogHeader &h, const std::string &k) { h.add_flag_list(k, battlefield_a_cards); }},
    {"--battlefield-b", nullptr, true, true, [](const char *v) { battlefield_b_cards = split_card_list(v); },
        [](DecisionLogHeader &h, const std::string &k) { h.add_flag_list(k, battlefield_b_cards); }},
    {"--graveyard-a", nullptr, true, true, [](const char *v) { graveyard_a_cards = split_card_list(v); },
        [](DecisionLogHeader &h, const std::string &k) { h.add_flag_list(k, graveyard_a_cards); }},
    {"--graveyard-b", nullptr, true, true, [](const char *v) { graveyard_b_cards = split_card_list(v); },
        [](DecisionLogHeader &h, const std::string &k) { h.add_flag_list(k, graveyard_b_cards); }},
    {"--exile-a", nullptr, true, true, [](const char *v) { exile_a_cards = split_card_list(v); },
        [](DecisionLogHeader &h, const std::string &k) { h.add_flag_list(k, exile_a_cards); }},
    {"--exile-b", nullptr, true, true, [](const char *v) { exile_b_cards = split_card_list(v); },
        [](DecisionLogHeader &h, const std::string &k) { h.add_flag_list(k, exile_b_cards); }},
    {"--sideboard-a", nullptr, true, true, [](const char *v) { sideboard_a_cards = split_card_list(v); },
        [](DecisionLogHeader &h, const std::string &k) { h.add_flag_list(k, sideboard_a_cards); }},
    {"--sideboard-b", nullptr, true, true, [](const char *v) { sideboard_b_cards = split_card_list(v); },
        [](DecisionLogHeader &h, const std::string &k) { h.add_flag_list(k, sideboard_b_cards); }},
    {"--life-a", nullptr, true, true, [](const char *v) { life_a_override = std::stoi(v); },
        [](DecisionLogHeader &h, const std::string &k) {
            if (life_a_override >= 0) h.add_flag(k, std::to_string(life_a_override));
        }},
    {"--life-b", nullptr, true, true, [](const char *v) { life_b_override = std::stoi(v); },
        [](DecisionLogHeader &h, const std::string &k) {
            if (life_b_override >= 0) h.add_flag(k, std::to_string(life_b_override));
        }},

    // Setup options recorded in their own header fields (DECK_A / DECK_B / SEED)
    {"--deck", nullptr, true, true,
        [](const char *v) {
            deck_a_name = v;
            deck_b_name = v;
        },
        nullptr},
    {"--deck-a", nullptr, true, true, [](const char *v) { deck_a_name = v; }, nullptr},
    {"--deck-b", nullptr, true, true, [](const char *v) { deck_b_name = v; }, nullptr},
    {"--seed", nullptr, true, true,
        [](const char *v) {
            seed_override = true;
            seed_value = static_cast<unsigned int>(std::stoul(v));
        },
        nullptr},

    // Input / output options; honored in replay mode
    {"--replay", nullptr, true, false,
        [](const char *v) {
            replay_mode = true;
            replay_file_path = v;
        },
        nullptr},
    // MCTS search protocol (SNAPSHOT/RESTORE/DETERMINIZE/RELEASE); a machine-mode extension, so it
    // implies --machine.
    {"--search-server", nullptr, false, false,
        [](const char *) {
            search_server_mode = true;
            machine_mode = true;
        },
        nullptr},
    {"--player", nullptr, true, false,
        [](const char *v) {
            has_human_player = true;
            human_player_is_a = is_seat_a(v);
        },
        nullptr},
    // Narrative-only viewer: redact game_log_private to one seat's view without setting
    // has_human_player (input stays on the machine path).
    {"--log-viewer", nullptr, true, false,
        [](const char *v) {
            log_viewer_set = true;
            log_viewer_owner = is_seat_a(v) ? Zone::PLAYER_A : Zone::PLAYER_B;
        },
        nullptr},
    {"--narrative", nullptr, false, false, [](const char *) { narrative_mode = true; }, nullptr},
    {"--log-decisions", nullptr, false, false, [](const char *) { log_decisions_flag = true; }, nullptr},
    // Passive BSTATE frames at forced auto-pass windows (machine mode observers only; no stdin
    // response expected, nothing recorded).
    {"--broadcast-steps", nullptr, false, false, [](const char *) { broadcast_steps_mode = true; }, nullptr},
    {"--help", "-h", false, false, nullptr, nullptr},
};

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

// Seat argument of --player / --log-viewer: "A" or "a" is Player A, anything else Player B.
static bool is_seat_a(const char *value) {
    return std::strcmp(value, "A") == 0 || std::strcmp(value, "a") == 0;
}

static const CliOption *find_option(const std::string &name) {
    for (const CliOption &option : OPTIONS) {
        if (name == option.name || (option.alias && name == option.alias)) return &option;
    }
    return nullptr;
}

// The option whose FLAGS token is keyed `key`, or nullptr.
static const CliOption *find_header_option(const std::string &key) {
    for (const CliOption &option : OPTIONS) {
        if (option.record && key == header_key(option)) return &option;
    }
    return nullptr;
}

static std::string header_key(const CliOption &option) {
    return std::string(option.name + 2);
}

// Applies, in command-line order, every parsed option whose `setup` matches.
static void apply_parsed(const std::vector<ParsedOption> &parsed, bool setup) {
    for (const ParsedOption &p : parsed) {
        if (p.option->setup == setup) p.option->apply(p.value);
    }
}

static void warn_ignored_in_replay(const std::vector<ParsedOption> &parsed) {
    std::string ignored;
    for (const ParsedOption &p : parsed) {
        if (!p.option->setup) continue;
        if (!ignored.empty()) ignored += ", ";
        ignored += p.option->name;
        if (p.value) ignored += std::string(" ") + p.value;
    }
    if (ignored.empty()) return;
    cli_warning("ignored in replay mode (the log header determines the game setup): " + ignored);
}

CliParseResult parse_command_line(int argc, char const *argv[]) {
    std::vector<ParsedOption> parsed;
    bool show_help = false;
    for (int i = 1; i < argc; i++) {
        const CliOption *option = find_option(argv[i]);
        if (!option) {
            cli_warning("unrecognized option '" + std::string(argv[i]) + "' (ignored); see --help");
            continue;
        }
        if (!option->apply) {
            show_help = true;
            continue;
        }
        const char *value = nullptr;
        if (option->takes_value) {
            if (i + 1 >= argc) {
                cli_warning("option '" + std::string(argv[i]) + "' requires a value (ignored); see --help");
                continue;
            }
            value = argv[++i];
        }
        parsed.push_back({option, value});
    }
    if (show_help) return CliParseResult::SHOW_HELP;

    // Input/output options first: --replay decides whether the setup options apply.
    apply_parsed(parsed, false);
    if (replay_mode) {
        warn_ignored_in_replay(parsed);
    } else {
        apply_parsed(parsed, true);
    }
    return CliParseResult::RUN;
}

void record_header_flags(DecisionLogHeader &header) {
    for (const CliOption &option : OPTIONS) {
        if (option.record) option.record(header, header_key(option));
    }
}

bool is_bare_header_flag(const std::string &token) {
    const CliOption *option = find_header_option(token);
    return option && !option->takes_value;
}

void apply_replay_header_flags(const std::vector<std::string> &tokens) {
    for (const std::string &tok : tokens) {
        std::string key = tok;
        std::string value;
        size_t eq = tok.find('=');
        if (eq != std::string::npos) {
            key = tok.substr(0, eq);
            value = tok.substr(eq + 1);
        }
        const CliOption *option = find_header_option(key);
        if (!option) fatal_error("Replay log FLAGS: unknown flag '" + tok + "'");
        // The "machine" decision-schedule marker is consumed by InputLogger::init_replay.
        if (!option->setup) continue;
        option->apply(option->takes_value ? value.c_str() : nullptr);
    }
}
