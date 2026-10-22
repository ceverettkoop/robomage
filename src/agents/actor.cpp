#include "actor.h"

#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <set>
#include <sstream>
#include <vector>

#include "../cli_args.h"
#include "../cli_output.h"
#include "../error.h"
#include "../featurize/json.h"
#include "../file_util.h"
#include "../game_driver.h"
#include "../input_logger.h"
#include "actor_schedule.h"
#include "driver.h"
#include "game_seed.h"
#include "teacher.h"
#include "trajectory.h"

bool actor_mode = false;
std::string actor_schedule_path;
std::string actor_out_dir;
int64_t actor_id = 0;
uint64_t actor_run_seed = 0;
int64_t actor_start = 0;
int64_t actor_matches = -1;

namespace {

using Clock = std::chrono::steady_clock;

// The status file is rewritten at least this often while matches are played (and after each one).
const double STATUS_INTERVAL_S = 2.0;
// The status file's recent decision rate is measured over windows at least this long.
const double RATE_WINDOW_S = 10.0;
// The per-game decision cap when neither the command line nor the schedule sets one.
const int64_t DEFAULT_MAX_GAME_DECISIONS = 5000;

// What identifies the schedule file's contents: its stat fields.
struct FileStamp {
        bool ok = false;
        int64_t mtime_ns = 0, size = 0;
        uint64_t ino = 0, dev = 0;
        bool operator==(const FileStamp &o) const {
            return ok == o.ok && mtime_ns == o.mtime_ns && size == o.size && ino == o.ino && dev == o.dev;
        }
};

// One seat's sideboarding phase before the game: its coin, its moves in order.
struct SideboardTally {
        bool explored = false;
        int swaps = 0;  // completed swaps: the cuts
        std::vector<std::string> in, out;
};

// The game being played (its episode: the sideboarding before it included), for its results line.
struct GameTally {
        int game_number = 1;  // 1-based
        bool a_first = true;
        int wins_a = 0, wins_b = 0;
        size_t decisions = 0;
        size_t seat_decisions[2] = {0, 0};
        std::set<uint64_t> versions[2];
        bool conceded = false;
        SideboardTally sb[2];
        Clock::time_point started = Clock::now();
};

// The match being played: its index, seeds, entry and plan, and what each seat got.
struct MatchInfo {
        int64_t index = 0;
        uint64_t game_seed = 0;
        unsigned int engine_seed = 0;
        agents::ScheduleEntry entry;
        agents::MatchPlan plan;
        std::string stem, rmlog, trajectory_dir;
        std::string deck[2], agent[2], teacher[2];
        size_t decisions = 0;
};

struct ActorRun {
        std::string status_path, results_path;
        FILE *results = nullptr;
        agents::ScheduleEntry base;  // the command line's values under every entry
        std::string base_teacher_timeout;
        agents::Schedule schedule;
        FileStamp schedule_stamp;
        double schedule_loaded_at = 0;
        std::string schedule_error;
        size_t schedule_loads = 0;
        Clock::time_point started = Clock::now();
        double started_unix = 0;
        int64_t next_match = 0;
        uint64_t matches = 0, games = 0, decisions = 0;
        uint64_t caps = 0, draws = 0, concessions = 0, errors = 0;
        std::set<uint64_t> versions;
        uint64_t last_version = 0;
        bool in_match = false;
        MatchInfo match;
        GameTally game;
        Clock::time_point last_status = Clock::now();
        // The recent rate: decisions over the last window of at least RATE_WINDOW_S (-1 before the
        // first window closes: the overall rate is reported).
        Clock::time_point rate_since = Clock::now();
        uint64_t rate_decisions = 0;
        double recent_rate = -1;
        std::string state = "starting";
        std::string last_error, exit_reason;
};

}  // namespace

static ActorRun *actor = nullptr;
static volatile sig_atomic_t stop_signal = 0;
static volatile sig_atomic_t reload_signal = 0;

static void on_stop_signal(int sig);
static void on_reload_signal(int sig);
static void install_signals();
static void on_error(const std::string &message, bool fatal);
static void refuse_options();
static agents::ScheduleEntry command_line_entry();
static FileStamp file_stamp(const std::string &path);
static bool load_schedule(std::string *err);
static void reload_schedule_if_changed();
static std::string actor_file(const char *suffix);
static std::string purpose_rmlog_dir(const MatchInfo &m);
static std::string purpose_trajectory_dir(const MatchInfo &m);
static std::string seat_letters(const MatchInfo &m);
static void plan_match_globals(int64_t index);
static void run_match();
static void game_started(int game_num, bool a_goes_first);
static void game_ended(int game_num, int winner);
static void write_results_line(int winner, const char *end_reason);
static void note_version(uint64_t v);
static double unix_now();
static double seconds_since(Clock::time_point t);
static void write_status();
static void maybe_write_status();
static void add_str(std::string &o, const char *key, const std::string &value);
static void add_int(std::string &o, const char *key, int64_t value);
static void add_num(std::string &o, const char *key, double value, int decimals);
static void add_raw(std::string &o, const char *key, const std::string &raw);
static std::string versions_json(const std::set<uint64_t> &versions);
static std::string names_json(const std::vector<std::string> &names);
static std::string sideboard_json(const GameTally &g);
static const char *knowledge_name(int level);

uint64_t actor_game_seed(uint64_t run_seed, uint64_t actor, uint64_t match) {
    uint64_t h = agents::splitmix64(run_seed);
    h = agents::splitmix64(h ^ actor);
    h = agents::splitmix64(h ^ match);
    return h >> 2;
}

static void on_stop_signal(int sig) {
    stop_signal = sig;
}

static void on_reload_signal(int) {
    reload_signal = 1;
}

// SIGTERM / SIGINT: stop at the next decision (a second one kills: the handler resets); SIGHUP:
// re-read the schedule before the next match. Interrupted system calls restart.
static void install_signals() {
    struct sigaction stop = {};
    stop.sa_handler = on_stop_signal;
    sigemptyset(&stop.sa_mask);
    stop.sa_flags = SA_RESTART | SA_RESETHAND;
    sigaction(SIGTERM, &stop, nullptr);
    sigaction(SIGINT, &stop, nullptr);
    struct sigaction hup = {};
    hup.sa_handler = on_reload_signal;
    sigemptyset(&hup.sa_mask);
    hup.sa_flags = SA_RESTART;
    sigaction(SIGHUP, &hup, nullptr);
}

bool actor_stop_requested() {
    return actor && stop_signal != 0;
}

void actor_stopped_at_decision() {
    if (!actor) return;
    actor->state = "stopped";
    actor->exit_reason = "signal " + std::to_string(static_cast<int>(stop_signal)) + " in match " +
                         std::to_string(actor->match.index) +
                         " (its unfinished game has no episode; its decision log ends with QUIT)";
    write_status();
}

// The error hook (error.h): a fatal error is the run's end, recorded with its reason.
static void on_error(const std::string &message, bool fatal) {
    if (!actor) return;
    actor->last_error = message;
    if (!fatal) {
        ++actor->errors;
        return;
    }
    actor->state = "failed";
    actor->exit_reason = message;
    write_status();
}

// The options that fix one game, or name files the schedule names, are refused under --actor.
static void refuse_options() {
    struct Refused {
            bool given;
            const char *what;
    };
    const Refused refused[] = {{replay_mode, "--replay"}, {seed_override, "--seed / --game-seed"},
        {play_script_given || actions_script_given, "--play / --actions"}, {has_human_player, "--player"},
        {!decision_log_path.empty() || log_decisions_flag, "--decision-log / --log-decisions"},
        {max_decisions >= 0, "--max-decisions (use --max-game-decisions)"},
        {!trajectory_out_dir.empty(), "--trajectory-out (the schedule's outputs name the directories)"},
        {!rmstate_dump_path.empty(), "--rmstate-dump"}, {selfcheck_state, "--selfcheck-*"},
        {!battlefield_a_cards.empty() || !battlefield_b_cards.empty() || !graveyard_a_cards.empty() ||
                !graveyard_b_cards.empty() || !exile_a_cards.empty() || !exile_b_cards.empty() ||
                !sideboard_a_cards.empty() || !sideboard_b_cards.empty(),
            "the zone presets"},
        {life_a_override >= 0 || life_b_override >= 0, "--life-a / --life-b"},
        {!hand_a_cards.empty() || !hand_b_cards.empty() || !library_a_cards.empty() || !library_b_cards.empty(),
            "--hand-a/-b / --library-a/-b"}};
    for (const Refused &r : refused)
        if (r.given) fatal_error(std::string("--actor: ") + r.what + " cannot be combined with --actor");
    if (actor_schedule_path.empty()) fatal_error("--actor needs --actor-schedule FILE");
    if (actor_out_dir.empty()) fatal_error("--actor needs --actor-out DIR");
}

// The command line's values, under every entry.
static agents::ScheduleEntry command_line_entry() {
    agents::ScheduleEntry e;
    e.deck[0] = deck_a_name;
    e.deck[1] = deck_b_name;
    e.bo3 = true;
    e.agent[0] = agent_a_spec;
    e.agent[1] = agent_b_spec;
    e.teacher[0] = teacher_command[0];
    e.teacher[1] = teacher_command[1];
    if (!teacher_mix_option.empty()) {
        char *end = nullptr;
        e.teacher_mix = std::strtod(teacher_mix_option.c_str(), &end);
        if (end == teacher_mix_option.c_str() || *end != '\0' || !(e.teacher_mix >= 0 && e.teacher_mix <= 1))
            fatal_error("--teacher-mix: expected a number in [0, 1], got '" + teacher_mix_option + "'");
    }
    e.knowledge[0] = knowledge_level_a;
    e.knowledge[1] = knowledge_level_b;
    e.max_game_decisions = max_game_decisions >= 0 ? max_game_decisions : DEFAULT_MAX_GAME_DECISIONS;
    e.trajectory_seats = trajectory_seats_option;
    e.trajectory_omniscient = trajectory_omniscient;
    return e;
}

static FileStamp file_stamp(const std::string &path) {
    FileStamp f;
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return f;
    f.ok = true;
    f.mtime_ns = static_cast<int64_t>(st.st_mtim.tv_sec) * 1000000000 + static_cast<int64_t>(st.st_mtim.tv_nsec);
    f.size = static_cast<int64_t>(st.st_size);
    f.ino = static_cast<uint64_t>(st.st_ino);
    f.dev = static_cast<uint64_t>(st.st_dev);
    return f;
}

// Reads and parses the schedule file; on success it replaces the schedule in use.
static bool load_schedule(std::string *err) {
    const FileStamp stamp = file_stamp(actor_schedule_path);
    std::ifstream in(actor_schedule_path, std::ios::binary);
    if (!in) {
        *err = actor_schedule_path + ": cannot be read";
        return false;
    }
    std::stringstream text;
    text << in.rdbuf();
    agents::Schedule s;
    if (!agents::parse_schedule(text.str(), actor->base, RESOURCE_DIR + "/decks", &s, err)) {
        *err = actor_schedule_path + ": " + *err;
        return false;
    }
    actor->schedule = std::move(s);
    actor->schedule_stamp = stamp;
    actor->schedule_loaded_at = unix_now();
    actor->schedule_error.clear();
    ++actor->schedule_loads;
    return true;
}

// Between matches: a SIGHUP, or a schedule file that changed, re-reads it. A schedule that does not
// parse is not used (the last good one stays), and its error is in the status file.
static void reload_schedule_if_changed() {
    const bool asked = reload_signal != 0;
    reload_signal = 0;
    const FileStamp now = file_stamp(actor_schedule_path);
    if (!asked && now == actor->schedule_stamp) return;
    std::string err;
    if (load_schedule(&err)) return;
    actor->schedule_stamp = now;
    actor->schedule_error = err;
    actor->last_error = err;
    cli_error("--actor-schedule: " + err + " (the last good schedule stays in use)");
}

// <actor out>/a<k><suffix>.
static std::string actor_file(const char *suffix) {
    return actor_out_dir + "/a" + std::to_string(actor_id) + suffix;
}

// The match's decision log directory: the entry's, the schedule's for its purpose, else
// <actor out>/<purpose>/rmlog.
static std::string purpose_rmlog_dir(const MatchInfo &m) {
    if (m.entry.outputs.rmlog_set) return m.entry.outputs.rmlog_dir;
    auto it = actor->schedule.outputs.find(m.entry.purpose);
    if (it != actor->schedule.outputs.end() && it->second.rmlog_set) return it->second.rmlog_dir;
    return actor_out_dir + "/" + m.entry.purpose + "/rmlog";
}

// The match's episode directory ("" none): the entry's, the schedule's for its purpose, else
// <actor out>/train/traj for train and none for eval.
static std::string purpose_trajectory_dir(const MatchInfo &m) {
    if (m.entry.outputs.trajectory_set) return m.entry.outputs.trajectory_dir;
    auto it = actor->schedule.outputs.find(m.entry.purpose);
    if (it != actor->schedule.outputs.end() && it->second.trajectory_set) return it->second.trajectory_dir;
    return m.entry.purpose == "train" ? actor_out_dir + "/train/traj" : "";
}

// The recorded seats: those whose side's letter is in the entry's trajectory_seats.
static std::string seat_letters(const MatchInfo &m) {
    std::string out;
    for (int s = 0; s < 2; ++s)
        if (m.entry.trajectory_seats.find(m.plan.side_of_seat[s] == 0 ? 'A' : 'B') != std::string::npos)
            out += s == 0 ? "A" : "B";
    return out;
}

// Match `index`'s seeds and plan, put into the launch-option globals the match is played with.
static void plan_match_globals(int64_t index) {
    MatchInfo &m = actor->match;
    m = MatchInfo();
    m.index = index;
    m.game_seed = actor_game_seed(actor_run_seed, static_cast<uint64_t>(actor_id), static_cast<uint64_t>(index));
    m.engine_seed = agents::engine_seed_for_game(m.game_seed);
    m.plan = agents::plan_match(actor->schedule, m.game_seed);
    m.entry = actor->schedule.entries[m.plan.entry];
    m.stem = "a" + std::to_string(actor_id) + "-m" + std::to_string(index);
    for (int s = 0; s < 2; ++s) {
        m.deck[s] = m.entry.deck[m.plan.deck_of_seat[s]];
        m.agent[s] = m.entry.agent[m.plan.side_of_seat[s]];
        m.teacher[s] = m.entry.teacher[m.plan.side_of_seat[s]];
    }
    deck_a_name = m.deck[0];
    deck_b_name = m.deck[1];
    bo3_mode = m.entry.bo3;
    agent_a_spec = m.agent[0];
    agent_b_spec = m.agent[1];
    teacher_command[0] = m.teacher[0];
    teacher_command[1] = m.teacher[1];
    const bool taught = !m.teacher[0].empty() || !m.teacher[1].empty();
    char mix[32];
    std::snprintf(mix, sizeof mix, "%.17g", m.entry.teacher_mix);
    teacher_mix_option = taught && m.entry.teacher_mix > 0 ? mix : "";
    teacher_timeout_option = taught ? actor->base_teacher_timeout : "";
    knowledge_level_a = m.entry.knowledge[m.plan.side_of_seat[0]];
    knowledge_level_b = m.entry.knowledge[m.plan.side_of_seat[1]];
    max_game_decisions = m.entry.max_game_decisions;
    m.trajectory_dir = purpose_trajectory_dir(m);
    trajectory_out_dir = m.trajectory_dir;
    const std::string seats = seat_letters(m);
    const bool recording = !m.trajectory_dir.empty() && !seats.empty();
    if (!recording) trajectory_out_dir.clear();
    trajectory_seats_option = recording ? seats : "AB";
    trajectory_omniscient = recording && m.entry.trajectory_omniscient;
    game_seed_given = true;
    game_seed_value = m.game_seed;
    seed_override = true;
    seed_value = m.engine_seed;
    const std::string rmlog_dir = purpose_rmlog_dir(m);
    if (!make_dirs(rmlog_dir)) fatal_error("--actor: cannot create " + rmlog_dir);
    m.rmlog = rmlog_dir + "/" + m.stem + ".rmlog";
    decision_log_path = m.rmlog;
    agents::SbExplore explore[2];
    for (int s = 0; s < 2; ++s) explore[s] = agents::sb_explore_for_deck(m.entry.sb_explore, m.deck[s]);
    agents_set_sb_explore(explore, m.entry.sb_explore.every_agent);
}

// One match from its planned globals: its decision log, the agents, the games.
static void run_match() {
    MatchInfo &m = actor->match;
    DEFAULT_DECK_ONE = load_launch_deck(Zone::PLAYER_A);
    DEFAULT_DECK_TWO = load_launch_deck(Zone::PLAYER_B);
    if (!InputLogger::instance().begin_match_log(
            m.engine_seed, m.rmlog, match_log_header(DEFAULT_DECK_ONE, DEFAULT_DECK_TWO)))
        fatal_error("--actor: cannot write the decision log " + m.rmlog);
    std::srand(m.engine_seed);
    cli_print_seed(m.engine_seed);
    agents_install(m.engine_seed);
    std::fflush(stdout);
    std::printf("ACTOR-MATCH: %lld entry=%s purpose=%s format=%s game_seed=%llu agents=%s|%s decks=%s|%s\n",
        static_cast<long long>(m.index), m.entry.id.c_str(), m.entry.purpose.c_str(), bo3_mode ? "bo3" : "bo1",
        static_cast<unsigned long long>(m.game_seed), m.agent[0].c_str(), m.agent[1].c_str(), m.deck[0].c_str(),
        m.deck[1].c_str());
    std::fflush(stdout);
    actor->game = GameTally();
    actor->in_match = true;
    if (bo3_mode) {
        play_bo3_match(DEFAULT_DECK_ONE, DEFAULT_DECK_TWO, m.engine_seed, game_started, game_ended);
    } else {
        reset_match_state();
        game_started(0, true);
        auto sys = init_ecs();
        const int winner = play_single_game(sys, DEFAULT_DECK_ONE, DEFAULT_DECK_TWO, true, m.engine_seed);
        report_game_end(1, winner);
        game_ended(0, winner);
    }
    actor->in_match = false;
    InputLogger::instance().end_match_log();
    ++actor->matches;
}

// A game of the match starts (play_bo3_match's before_game; game_num 0-based).
static void game_started(int game_num, bool a_goes_first) {
    GameTally &g = actor->game;
    g.game_number = game_num + 1;
    g.a_first = a_goes_first;
    g.wins_a = match_wins_a;
    g.wins_b = match_wins_b;
}

// A game of the match ended (play_bo3_match's after_game, after its GAME_RESULT line and episode):
// its results line; the next game's tally starts (its sideboarding decisions count toward it).
static void game_ended(int, int winner) {
    const char *reason = game_capped()                                          ? "cap"
                         : actor->game.conceded                                 ? "concession"
                         : winner == Zone::PLAYER_A || winner == Zone::PLAYER_B ? "win"
                                                                                : "draw";
    write_results_line(winner, reason);
    ++actor->games;
    if (game_capped())
        ++actor->caps;
    else if (actor->game.conceded)
        ++actor->concessions;
    else if (winner != Zone::PLAYER_A && winner != Zone::PLAYER_B)
        ++actor->draws;
    actor->game = GameTally();
    maybe_write_status();
}

void actor_note_decision(int seat, int reply, const agents::Behaviour *behaviour) {
    if (!actor || !actor->in_match) return;
    GameTally &g = actor->game;
    ++g.decisions;
    ++actor->match.decisions;
    ++actor->decisions;
    if (seat == 0 || seat == 1) ++g.seat_decisions[seat];
    if ((reply == CONCEDE_GAME && !sideboard_phase) || reply == CONCEDE_MATCH) g.conceded = true;
    if (behaviour && !behaviour->mu.empty() && (seat == 0 || seat == 1)) {
        g.versions[seat].insert(behaviour->policy_version);
        note_version(behaviour->policy_version);
    }
    maybe_write_status();
}

void actor_note_sideboard_explore(int seat, bool explored) {
    if (!actor || !actor->in_match || (seat != 0 && seat != 1)) return;
    actor->game.sb[seat].explored = explored;
}

void actor_note_sideboard_move(int seat, bool in, const std::string &card) {
    if (!actor || !actor->in_match || (seat != 0 && seat != 1)) return;
    SideboardTally &t = actor->game.sb[seat];
    (in ? t.in : t.out).push_back(card);
    if (!in) ++t.swaps;
}

static void note_version(uint64_t v) {
    actor->versions.insert(v);
    actor->last_version = v;
}

// One JSON line per finished game in the results file (docs/actor.md §4).
static void write_results_line(int winner, const char *end_reason) {
    const MatchInfo &m = actor->match;
    const GameTally &g = actor->game;
    const int result_a = std::string(end_reason) == "cap" ? 0
                         : winner == Zone::PLAYER_A       ? 1
                         : winner == Zone::PLAYER_B       ? -1
                                                          : 0;
    std::string o = "{";
    add_int(o, "actor", actor_id);
    add_int(o, "match", m.index);
    add_int(o, "game", g.game_number);
    add_str(o, "entry", m.entry.id);
    add_str(o, "purpose", m.entry.purpose);
    add_str(o, "schedule", actor->schedule.version);
    add_str(o, "format", m.entry.bo3 ? "bo3" : "bo1");
    add_raw(o, "swapped", m.plan.swapped ? "true" : "false");
    add_str(o, "side_a_seat", m.plan.side_of_seat[0] == 0 ? "A" : "B");
    add_str(o, "deck_a", m.deck[0]);
    add_str(o, "deck_b", m.deck[1]);
    add_str(o, "agent_a", m.agent[0]);
    add_str(o, "agent_b", m.agent[1]);
    add_str(o, "teacher_a", m.teacher[0]);
    add_str(o, "teacher_b", m.teacher[1]);
    add_num(o, "teacher_mix", m.entry.teacher_mix, 6);
    add_str(o, "knowledge_a", knowledge_name(knowledge_level_a));
    add_str(o, "knowledge_b", knowledge_name(knowledge_level_b));
    add_int(o, "game_seed", static_cast<int64_t>(m.game_seed));
    add_int(o, "engine_seed", m.engine_seed);
    add_str(o, "first", g.a_first ? "A" : "B");
    add_int(o, "wins_a", g.wins_a);
    add_int(o, "wins_b", g.wins_b);
    add_int(o, "result_a", result_a);
    add_int(o, "result_b", -result_a);
    add_str(o, "end_reason", end_reason);
    add_int(o, "decisions", static_cast<int64_t>(g.decisions));
    add_int(o, "decisions_a", static_cast<int64_t>(g.seat_decisions[0]));
    add_int(o, "decisions_b", static_cast<int64_t>(g.seat_decisions[1]));
    add_raw(o, "policy_versions_a", versions_json(g.versions[0]));
    add_raw(o, "policy_versions_b", versions_json(g.versions[1]));
    if (g.game_number >= 2) add_raw(o, "sb", sideboard_json(g));
    add_str(o, "rmlog", m.rmlog);
    if (trajectory_out_dir.empty())
        add_raw(o, "episode", "null");
    else
        add_str(o, "episode", trajectory_out_dir + "/" + m.stem + ".g" + std::to_string(g.game_number) + ".rmtraj");
    add_num(o, "seconds", seconds_since(g.started), 3);
    o += "}\n";
    if (std::fwrite(o.data(), 1, o.size(), actor->results) != o.size() || std::fflush(actor->results) != 0)
        fatal_error("--actor: cannot write " + actor->results_path);
}

static void add_str(std::string &o, const char *key, const std::string &value) {
    add_raw(o, key, featurize::json_quote(value));
}

static void add_int(std::string &o, const char *key, int64_t value) {
    add_raw(o, key, std::to_string(value));
}

// A number with `decimals` digits after the point.
static void add_num(std::string &o, const char *key, double value, int decimals) {
    char buf[48];
    std::snprintf(buf, sizeof buf, "%.*f", decimals, value);
    add_raw(o, key, buf);
}

static void add_raw(std::string &o, const char *key, const std::string &raw) {
    if (o.size() > 1) o += ",";
    o += featurize::json_quote(key) + ":" + raw;
}

// A --knowledge-a/-b level's name.
static const char *knowledge_name(int level) {
    return level == 0 ? "off" : level == 2 ? "expert+belief" : "expert";
}

static std::string versions_json(const std::set<uint64_t> &versions) {
    std::string out = "[";
    for (const uint64_t v : versions) out += (out.size() > 1 ? "," : "") + std::to_string(v);
    return out + "]";
}

static std::string names_json(const std::vector<std::string> &names) {
    std::string out = "[";
    for (const std::string &n : names) out += (out.size() > 1 ? "," : "") + featurize::json_quote(n);
    return out + "]";
}

// The sideboarding phase before the game, per seat: {"A": {...}, "B": {...}}.
static std::string sideboard_json(const GameTally &g) {
    std::string o = "{";
    for (int s = 0; s < 2; ++s) {
        const SideboardTally &t = g.sb[s];
        std::string seat = "{";
        add_raw(seat, "explored", t.explored ? "true" : "false");
        add_int(seat, "swaps", t.swaps);
        add_raw(seat, "in", names_json(t.in));
        add_raw(seat, "out", names_json(t.out));
        add_raw(o, s == 0 ? "A" : "B", seat + "}");
    }
    return o + "}";
}

static double unix_now() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

static double seconds_since(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

static void maybe_write_status() {
    if (seconds_since(actor->last_status) >= STATUS_INTERVAL_S) write_status();
}

// The status file (docs/actor.md §3), written whole.
static void write_status() {
    const double window = seconds_since(actor->rate_since);
    if (window >= RATE_WINDOW_S) {
        actor->recent_rate = static_cast<double>(actor->decisions - actor->rate_decisions) / window;
        actor->rate_since = Clock::now();
        actor->rate_decisions = actor->decisions;
    }
    actor->last_status = Clock::now();
    const double uptime = seconds_since(actor->started);
    const double overall_rate = uptime > 0 ? static_cast<double>(actor->decisions) / uptime : 0;
    std::string o = "{";
    add_int(o, "actor", actor_id);
    add_int(o, "pid", static_cast<int64_t>(getpid()));
    add_int(o, "run_seed", static_cast<int64_t>(actor_run_seed));
    add_str(o, "state", actor->state);
    add_num(o, "started_at", actor->started_unix, 3);
    add_num(o, "updated_at", unix_now(), 3);
    add_num(o, "uptime_s", uptime, 3);
    add_int(o, "first_match", actor_start);
    add_int(o, "next_match", actor->next_match);
    add_int(o, "matches", static_cast<int64_t>(actor->matches));
    add_int(o, "games", static_cast<int64_t>(actor->games));
    add_int(o, "decisions", static_cast<int64_t>(actor->decisions));
    add_num(o, "decisions_per_s", overall_rate, 1);
    add_num(o, "recent_decisions_per_s", actor->recent_rate < 0 ? overall_rate : actor->recent_rate, 1);
    add_num(o, "matches_per_s", uptime > 0 ? static_cast<double>(actor->matches) / uptime : 0, 4);
    add_int(o, "caps", static_cast<int64_t>(actor->caps));
    add_int(o, "draws", static_cast<int64_t>(actor->draws));
    add_int(o, "concessions", static_cast<int64_t>(actor->concessions));
    add_int(o, "errors", static_cast<int64_t>(actor->errors));
    if (actor->in_match) {
        const MatchInfo &m = actor->match;
        std::string c = "{";
        add_int(c, "match", m.index);
        add_str(c, "entry", m.entry.id);
        add_str(c, "purpose", m.entry.purpose);
        add_int(c, "game_seed", static_cast<int64_t>(m.game_seed));
        add_int(c, "game", actor->game.game_number);
        add_int(c, "decisions", static_cast<int64_t>(m.decisions));
        add_str(c, "rmlog", m.rmlog);
        add_raw(o, "current", c + "}");
    } else {
        add_raw(o, "current", "null");
    }
    std::string sched = "{";
    add_str(sched, "path", actor_schedule_path);
    add_str(sched, "version", actor->schedule.version);
    add_num(sched, "loaded_at", actor->schedule_loaded_at, 3);
    add_int(sched, "loads", static_cast<int64_t>(actor->schedule_loads));
    add_int(sched, "entries", static_cast<int64_t>(actor->schedule.entries.size()));
    add_str(sched, "error", actor->schedule_error);
    add_raw(o, "schedule", sched + "}");
    std::string pv = "{";
    add_int(pv, "count", static_cast<int64_t>(actor->versions.size()));
    if (!actor->versions.empty()) {
        add_int(pv, "min", static_cast<int64_t>(*actor->versions.begin()));
        add_int(pv, "max", static_cast<int64_t>(*actor->versions.rbegin()));
        add_int(pv, "last", static_cast<int64_t>(actor->last_version));
    }
    add_raw(o, "policy_versions", pv + "}");
    add_str(o, "last_error", actor->last_error);
    add_str(o, "exit_reason", actor->exit_reason);
    o += "}\n";
    std::string err;
    if (!write_file_atomic(actor->status_path, o, &err)) cli_error("--actor: the status file: " + err);
}

int actor_main() {
    refuse_options();
    actor = new ActorRun();
    actor->started_unix = unix_now();
    actor->base = command_line_entry();
    actor->base_teacher_timeout = teacher_timeout_option;
    actor->next_match = actor_start;
    if (!make_dirs(actor_out_dir)) fatal_error("--actor-out: cannot create " + actor_out_dir);
    actor->status_path = actor_file(".status.json");
    actor->results_path = actor_file(".results.jsonl");
    actor->results = std::fopen(actor->results_path.c_str(), "ab");
    if (!actor->results) fatal_error("--actor: cannot open " + actor->results_path);
    set_error_hook(on_error);
    install_signals();
    std::string err;
    if (!load_schedule(&err)) fatal_error("--actor-schedule: " + err);
    actor->state = "running";
    write_status();
    for (int64_t i = actor_start; actor_matches < 0 || i < actor_start + actor_matches; ++i) {
        if (stop_signal) break;
        reload_schedule_if_changed();
        plan_match_globals(i);
        run_match();
        actor->next_match = i + 1;
        write_status();
    }
    teacher_finish();
    actor->state = stop_signal ? "stopped" : "done";
    if (stop_signal)
        actor->exit_reason = "signal " + std::to_string(static_cast<int>(stop_signal)) + " between matches";
    write_status();
    std::fclose(actor->results);
    actor->results = nullptr;
    return 0;
}
