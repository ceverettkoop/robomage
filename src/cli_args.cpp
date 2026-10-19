#include "cli_args.h"

#include <cstdlib>
#include <cstring>

#include "agents/game_seed.h"
#include "agents/trajectory.h"
#include "cli_output.h"
#include "components/zone.h"
#include "error.h"
#include "game_driver.h"
#include "state_format/cards_file.h"
#include "state_format/selfcheck_featurize.h"
#include "text_table.h"

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

// A removed option: giving it is fatal, with a hint at what replaces it.
struct RemovedOption {
        const char *name;
        const char *hint;
};

static std::vector<std::string> split_card_list(const std::string &csv);
static const RemovedOption *find_removed_option(const std::string &name);
static bool is_seat_a(const char *value);
static int knowledge_level_value(const char *value);
static long non_negative_value(const char *flag, const char *value);
static const CliOption *find_option(const std::string &name);
static const CliOption *find_header_option(const std::string &key);
static std::string header_key(const CliOption &option);
static void apply_parsed(const std::vector<ParsedOption> &parsed, bool setup);
static void warn_ignored_in_replay(const std::vector<ParsedOption> &parsed);
static void resolve_human_seats();

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
    // The "machine" marker records that the game was played over the machine protocol. The
    // decision schedule is the same for every consumer, so in replay mode the marker changes
    // nothing and the flag itself only affects output.
    {"--machine", nullptr, false, false, [](const char *) { machine_mode = true; },
        [](DecisionLogHeader &h, const std::string &k) {
            if (machine_mode) h.add_flag(k);
        }},
    // Turns auto-mana (game_driver.h) off for the whole match; a FLAGS line without it replays with
    // auto-mana on.
    {"--no-auto-mana", nullptr, false, true, [](const char *) { auto_mana = false; },
        [](DecisionLogHeader &h, const std::string &k) {
            if (!auto_mana) h.add_flag(k);
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
    {"--offer-cancel", nullptr, false, true, [](const char *) { offer_cast_cancel = true; },
        [](DecisionLogHeader &h, const std::string &k) {
            if (offer_cast_cancel) h.add_flag(k);
        }},
    // Each seat's state-format frames carry only its own registered decklist (decision 11).
    {"--closed-decklist", nullptr, false, true, [](const char *) { closed_decklist = true; },
        [](DecisionLogHeader &h, const std::string &k) {
            if (closed_decklist) h.add_flag(k);
        }},
    // Stop the run at the first decision past N (end_run_at_decision_cap), so a capped log replays
    // to the same point.
    {"--max-decisions", nullptr, true, true,
        [](const char *v) { max_decisions = non_negative_value("--max-decisions", v); },
        [](DecisionLogHeader &h, const std::string &k) {
            if (max_decisions >= 0) h.add_flag(k, std::to_string(max_decisions));
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
    // The game the Python drivers call seed N: engine seed engine_seed_for_game(N), the agents'
    // generator seeded with N (agents/game_seed.h).
    {"--game-seed", nullptr, true, true,
        [](const char *v) {
            game_seed_given = true;
            game_seed_value = static_cast<uint64_t>(non_negative_value("--game-seed", v));
            seed_override = true;
            seed_value = agents::engine_seed_for_game(game_seed_value);
        },
        nullptr},
    // A seat's stacked deck: these cards first (the opening hand), then its --library-a/-b
    // (load_launch_deck). Implies --no-shuffle for both seats.
    {"--hand-a", nullptr, true, true,
        [](const char *v) {
            hand_a_cards = split_card_list(v);
            no_shuffle = true;
        },
        nullptr},
    {"--hand-b", nullptr, true, true,
        [](const char *v) {
            hand_b_cards = split_card_list(v);
            no_shuffle = true;
        },
        nullptr},
    {"--library-a", nullptr, true, true, [](const char *v) { library_a_cards = split_card_list(v); }, nullptr},
    {"--library-b", nullptr, true, true, [](const char *v) { library_b_cards = split_card_list(v); }, nullptr},

    // Input / output options; honored in replay mode
    {"--replay", nullptr, true, false,
        [](const char *v) {
            replay_mode = true;
            replay_file_path = v;
        },
        nullptr},
    // A human seat (resolve_human_seats).
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
    // In-process agents (agents/driver.h); each implies --machine.
    {"--agent-a", nullptr, true, false,
        [](const char *v) {
            agent_a_spec = v;
            machine_mode = true;
        },
        nullptr},
    {"--agent-b", nullptr, true, false,
        [](const char *v) {
            agent_b_spec = v;
            machine_mode = true;
        },
        nullptr},
    {"--play", nullptr, true, false,
        [](const char *v) {
            play_script_given = true;
            play_script_specs = v;
            machine_mode = true;
        },
        nullptr},
    {"--actions", nullptr, true, false,
        [](const char *v) {
            actions_script_given = true;
            actions_script = v;
            machine_mode = true;
        },
        nullptr},
    // The board and menu of every decision on the game log (agents/trace.h).
    {"--trace-decisions", nullptr, false, false, [](const char *) { trace_decisions = true; }, nullptr},
    // Featurize every decision an in-process agent seat makes and drop the tensors (the bench tier's
    // featurizer stage; agents/driver.h).
    {"--featurize-seats", nullptr, false, false, [](const char *) { featurize_seats = true; }, nullptr},
    // Write every game's trajectory episode (agents/trajectory.h): the directory, the recorded
    // seats and the learner-side omniscient featurization.
    {"--trajectory-out", nullptr, true, false, [](const char *v) { trajectory_out_dir = v; }, nullptr},
    {"--trajectory-seats", nullptr, true, false, [](const char *v) { trajectory_seats_option = v; }, nullptr},
    {"--trajectory-omniscient", nullptr, false, false, [](const char *) { trajectory_omniscient = true; }, nullptr},
    // Check the decks' cards against the card vocabulary and their scripts instead of playing
    // (deck_check.h).
    {"--check-decks", nullptr, false, false, [](const char *) { check_decks_flag = true; }, nullptr},
    {"--check-decks-dir", nullptr, true, false, [](const char *v) { check_decks_dirs.push_back(v); }, nullptr},
    // Check the oracle-text table against the vocab and the model pin, or write the inputs of its
    // stale rows, instead of playing (text_table.h).
    {"--text-table-check", nullptr, true, false, [](const char *v) { text_table_check_path = v; }, nullptr},
    {"--text-table-inputs", nullptr, true, false, [](const char *v) { text_table_inputs_path = v; }, nullptr},
    {"--text-table", nullptr, true, false, [](const char *v) { text_table_path = v; }, nullptr},
    {"--model-pin", nullptr, true, false, [](const char *v) { model_pin_path = v; }, nullptr},
    {"--narrative", nullptr, false, false, [](const char *) { narrative_mode = true; }, nullptr},
    {"--log-decisions", nullptr, false, false, [](const char *) { log_decisions_flag = true; }, nullptr},
    {"--decision-log", nullptr, true, false,
        [](const char *v) {
            decision_log_path = v;
            log_decisions_flag = true;
        },
        nullptr},
    // Narrate reveal windows opening and closing (VISIBILITY: lines in the game log).
    {"--trace-visibility", nullptr, false, false, [](const char *) { trace_visibility = true; }, nullptr},
    // Print the per-viewer handle tables at each decision and narrate handle issue/link/retire
    // (HANDLES: / HANDLE: lines in the game log).
    {"--trace-handles", nullptr, false, false, [](const char *) { trace_handles = true; }, nullptr},
    // Narrate each observation event as it is queued, per viewer (EVENT: lines in the game log).
    {"--trace-events", nullptr, false, false, [](const char *) { trace_events = true; }, nullptr},
    // Write every viewer's state-format frames (RMSTATE records) to a file at each decision.
    {"--rmstate-dump", nullptr, true, false, [](const char *v) { rmstate_dump_path = v; }, nullptr},
    // The machine protocol's stdout format: rmstate, the only one (state_format/stream.h), so the
    // same as --machine.
    {"--machine-format", nullptr, true, false,
        [](const char *v) {
            if (std::strcmp(v, "bquery") == 0)
                fatal_error("--machine-format bquery was removed: the machine protocol is RMSTATE (--machine)");
            if (std::strcmp(v, "rmstate") != 0)
                fatal_error("--machine-format: expected rmstate, got '" + std::string(v) + "'");
            machine_mode = true;
        },
        nullptr},
    // Every CARDS frame carries each card's and token's raw script text (its `script` column).
    {"--rmstate-card-scripts", nullptr, false, false, [](const char *) { rmstate_card_scripts = true; }, nullptr},
    // Render the RMSTATE frames --rmstate-dump writes (or --rmstate-read reads) as JSON lines.
    {"--state-dump", nullptr, true, false,
        [](const char *v) {
            if (std::strcmp(v, "json") != 0)
                fatal_error("--state-dump: the only format is json, got '" + std::string(v) + "'");
            state_dump_json = true;
        },
        nullptr},
    // Write the whole-vocab CARDS frame (with the SCHEMA and METAGAME frames) instead of playing
    // (state_format/cards_file.h).
    {"--cards-frame", nullptr, true, false, [](const char *v) { cards_frame_path = v; }, nullptr},
    // Decode, re-encode and compare an RMSTATE stream instead of playing (state_format/recode.h).
    {"--rmstate-read", nullptr, true, false, [](const char *v) { rmstate_read_path = v; }, nullptr},
    // Each seat's knowledge sidecar level in its RMSTATE frames (state_format/knowledge_feed.h).
    // Output only, like the RMSTATE writers: not recorded in the log header, honored in replay.
    {"--knowledge-a", nullptr, true, false, [](const char *v) { knowledge_level_a = knowledge_level_value(v); },
        nullptr},
    {"--knowledge-b", nullptr, true, false, [](const char *v) { knowledge_level_b = knowledge_level_value(v); },
        nullptr},
    // Run the state format's self-checks at every decision (state_format/selfcheck.h).
    {"--selfcheck-state", nullptr, false, false, [](const char *) { selfcheck_state = true; }, nullptr},
    // Check the knowledge sidecar against the truth at every decision (state_format/selfcheck_knowledge.h);
    // implies --selfcheck-state, whose knowledge instances it checks.
    {"--selfcheck-knowledge", nullptr, false, false,
        [](const char *) {
            selfcheck_knowledge = true;
            selfcheck_state = true;
        },
        nullptr},
    // Check the featurizer under the state self-check (state_format/selfcheck_featurize.h); implies
    // --selfcheck-state, whose scramble and permutation frames it featurizes.
    {"--selfcheck-featurize", nullptr, false, false,
        [](const char *) {
            selfcheck_featurize = true;
            selfcheck_state = true;
        },
        nullptr},
    // Start the sampled self-checks (permutation leak, snapshot) at every Nth decision.
    {"--selfcheck-state-sample", nullptr, true, false,
        [](const char *v) {
            char *end = nullptr;
            const long n = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || n < 1)
                fatal_error("--selfcheck-state-sample: expected a positive integer, got '" + std::string(v) + "'");
            selfcheck_state_sample = static_cast<int>(n);
        },
        nullptr},
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

// The value of --knowledge-a / --knowledge-b: off (0), expert (1) or expert+belief (2); anything
// else is fatal.
static int knowledge_level_value(const char *value) {
    if (std::strcmp(value, "off") == 0) return 0;
    if (std::strcmp(value, "expert") == 0) return 1;
    if (std::strcmp(value, "expert+belief") == 0) return 2;
    fatal_error("--knowledge-a/--knowledge-b: expected off, expert or expert+belief, got '" + std::string(value) + "'");
    return 0;
}

// A non-negative integer option value; anything else is fatal.
static long non_negative_value(const char *flag, const char *value) {
    char *end = nullptr;
    const long long n = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0' || n < 0)
        fatal_error(std::string(flag) + ": expected a non-negative integer, got '" + value + "'");
    return static_cast<long>(n);
}

static const RemovedOption REMOVED_OPTIONS[] = {
    {"--search-server", "the MCTS search protocol is gone; snapshots stay in-process (snapshot.h)"},
    {"--broadcast-steps", "BSTATE frames went with BQUERY; the RMSTATE frames are the machine protocol"},
};

static const RemovedOption *find_removed_option(const std::string &name) {
    for (const RemovedOption &option : REMOVED_OPTIONS) {
        if (name == option.name) return &option;
    }
    return nullptr;
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
        if (const RemovedOption *removed = find_removed_option(argv[i]))
            fatal_error(std::string(removed->name) + " was removed: " + removed->hint);
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
    resolve_human_seats();
    return CliParseResult::RUN;
}

// The human seats (the C++ CLI, cli/human.h). Outside --replay, a seat with no in-process agent
// is a human seat when it is the --player seat, or when --machine is not given (every agent and
// script implies it): a plain run is hotseat, `--player A --agent-b explore` a human against an
// agent, and `--player A` with --play the human taking over once the script runs out. A human seat
// decides on its frames like the agents, so it turns machine mode on (no narrative unless
// --narrative), and the decision log, which a human game always writes. It reads stdin, so no
// other seat may read the RMSTATE stream (a seat with neither an agent nor a script would).
static void resolve_human_seats() {
    if (replay_mode) return;
    const bool scripted = play_script_given || actions_script_given;
    const std::string *specs[2] = {&agent_a_spec, &agent_b_spec};
    const char *flags[2] = {"--agent-a", "--agent-b"};
    bool any = false;
    for (int s = 0; s < 2; ++s) {
        const bool player_seat = has_human_player && human_player_is_a == (s == 0);
        if (player_seat && !specs[s]->empty())
            fatal_error(
                std::string("--player ") + (s == 0 ? "A" : "B") + " is a human seat; it cannot also have " + flags[s]);
        human_seat[s] = specs[s]->empty() && (player_seat || !machine_mode);
        any = any || human_seat[s];
    }
    if (!any) return;
    for (int s = 0; s < 2; ++s)
        if (!human_seat[s] && specs[s]->empty() && !scripted)
            fatal_error(std::string("Player ") + (s == 0 ? "A" : "B") +
                        " has no in-process agent, so it would read the RMSTATE stream on stdin, which the human "
                        "seat reads; give it one (" +
                        flags[s] + " explore, ...) or drop --machine for hotseat");
    machine_mode = true;
    log_decisions_flag = true;
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
        // The "machine" marker is not a setup option (see --machine above).
        if (!option->setup) continue;
        option->apply(option->takes_value ? value.c_str() : nullptr);
    }
}
