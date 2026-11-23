#include "simulation.h"

#include <dirent.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <thread>

#include "../../src/featurize/json.h"
#include "files.h"
#include "model_zip.h"
#include "paths.h"
#include "rmlog.h"

namespace {

const char *const SUMMARY_FILE = "simulation.json";
const char *const STATE_KEYS[] = {"waiting", "running", "finished", "failed", "cancelled"};
// The 95% normal quantile of the Wilson interval.
const double Z95 = 1.959964;

}  // namespace

static std::string base_name(const std::string &path);
static std::string file_safe(const std::string &text);
static std::string relative_to(const std::string &path, const std::string &dir);
static std::string in_dir(const std::string &dir, const std::string &name);
static std::string match_stem(int index);
static std::string first_fatal_line(const std::string &text);
static std::string tail_of(const std::string &text, size_t max_bytes);
static bool find_state(const std::string &key, rmwork::SimState *out);
static std::string entrant_json(const rmwork::SimEntrant &e);
static bool read_entrant(const featurize::Json &v, rmwork::SimEntrant *out, std::string *err);
static std::string match_json(const rmwork::SimMatch &m, const std::string &dir);
static bool read_match(const featurize::Json &v, const std::string &dir, rmwork::SimMatch *out, std::string *err);
static std::string clock_text(double seconds);

static std::string base_name(const std::string &path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

// Letters, digits, '-' and '_' kept; anything else as '_'.
static std::string file_safe(const std::string &text) {
    std::string out;
    for (char c : text) out += std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' ? c : '_';
    return out;
}

// `path` relative to `dir` when under it, else as it is.
static std::string relative_to(const std::string &path, const std::string &dir) {
    if (!dir.empty() && path.size() > dir.size() + 1 && path.compare(0, dir.size(), dir) == 0 && path[dir.size()] == '/')
        return path.substr(dir.size() + 1);
    return path;
}

// `name` under `dir` unless it is absolute or empty.
static std::string in_dir(const std::string &dir, const std::string &name) {
    if (name.empty() || name[0] == '/') return name;
    return dir + "/" + name;
}

// "m001" for match 0.
static std::string match_stem(int index) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "m%03d", index + 1);
    return buf;
}

// The first line of `text` that holds "FATAL", "" none.
static std::string first_fatal_line(const std::string &text) {
    const size_t at = text.find("FATAL");
    if (at == std::string::npos) return "";
    const size_t begin = text.rfind('\n', at);
    const size_t from = begin == std::string::npos ? 0 : begin + 1;
    const size_t end = text.find('\n', at);
    return text.substr(from, end == std::string::npos ? std::string::npos : end - from);
}

// The last `max_bytes` of `text`, from a line's start.
static std::string tail_of(const std::string &text, size_t max_bytes) {
    if (text.size() <= max_bytes) return text;
    const size_t from = text.find('\n', text.size() - max_bytes);
    return text.substr(from == std::string::npos ? text.size() - max_bytes : from + 1);
}

static bool find_state(const std::string &key, rmwork::SimState *out) {
    for (int i = 0; i < 5; ++i)
        if (key == STATE_KEYS[i]) {
            *out = static_cast<rmwork::SimState>(i);
            return true;
        }
    return false;
}

static std::string entrant_json(const rmwork::SimEntrant &e) {
    using featurize::json_quote;
    return "{\"agent\": " + json_quote(rmwork::opponent_key(e.agent.kind)) + ", \"model\": " + json_quote(e.agent.model) +
           ", \"options\": " + json_quote(e.agent.options) + ", \"deck\": " + json_quote(e.deck) +
           ", \"label\": " + json_quote(rmwork::sim_entrant_text(e)) + "}";
}

static bool read_entrant(const featurize::Json &v, rmwork::SimEntrant *out, std::string *err) {
    const featurize::Json *agent = v.get("agent"), *model = v.get("model"), *options = v.get("options"),
                          *deck = v.get("deck");
    const int kind = agent && agent->is_string() ? rmwork::find_opponent(agent->as_string()) : -1;
    if (!v.is_object() || kind < 0 || !deck || !deck->is_string()) {
        *err = "an entrant without a known agent or a deck";
        return false;
    }
    out->agent.kind = static_cast<rmwork::Opponent>(kind);
    out->agent.model = model && model->is_string() ? model->as_string() : "";
    out->agent.options = options && options->is_string() ? options->as_string() : "";
    out->deck = deck->as_string();
    return true;
}

static std::string match_json(const rmwork::SimMatch &m, const std::string &dir) {
    using featurize::json_quote;
    std::string games;
    for (size_t g = 0; g < m.result.games.size(); ++g) games += (g ? ", " : "") + std::to_string(m.result.games[g]);
    char seconds[32];
    std::snprintf(seconds, sizeof seconds, "%.3f", m.seconds);
    return "{\"match\": " + std::to_string(m.index) + ", \"seed\": " + std::to_string(m.seed) +
           ", \"seat_a\": " + std::to_string(m.entrant(0)) + ", \"state\": " + json_quote(rmwork::sim_state_key(m.state)) +
           ", \"result\": " + json_quote(rmwork::sim_result_text(m)) + ", \"winner_seat\": " +
           std::to_string(m.winner_seat) + ", \"winner\": " + std::to_string(m.winner()) + ", \"games\": [" + games +
           "], \"match_score\": [" + std::to_string(m.result.wins[0]) + ", " + std::to_string(m.result.wins[1]) +
           "], \"decisions\": " + std::to_string(m.decisions) + ", \"seconds\": " + seconds +
           ", \"log\": " + json_quote(relative_to(m.log, dir)) +
           ", \"stderr\": " + json_quote(relative_to(m.stderr_path, dir)) + ", \"error\": " + json_quote(m.error) + "}";
}

static bool read_match(const featurize::Json &v, const std::string &dir, rmwork::SimMatch *out, std::string *err) {
    const featurize::Json *index = v.get("match"), *seed = v.get("seed"), *seat_a = v.get("seat_a"),
                          *state = v.get("state");
    if (!v.is_object() || !index || !index->is_integer() || !seed || !seed->is_integer() || !seat_a ||
        !seat_a->is_integer() || !state || !state->is_string() || !find_state(state->as_string(), &out->state)) {
        *err = "a match without its index, seed, seats or state";
        return false;
    }
    out->index = static_cast<int>(index->as_int());
    out->seed = static_cast<uint64_t>(seed->as_int());
    out->swapped = seat_a->as_int() == 1;
    if (const featurize::Json *w = v.get("winner_seat"))
        if (w->is_integer()) out->winner_seat = static_cast<int>(w->as_int());
    if (const featurize::Json *g = v.get("games"))
        if (g->is_array())
            for (const featurize::Json &x : g->items())
                if (x.is_integer()) out->result.games.push_back(static_cast<int>(x.as_int()));
    if (const featurize::Json *s = v.get("match_score"))
        if (s->is_array() && s->items().size() == 2) {
            out->result.wins[0] = static_cast<int>(s->items()[0].as_int());
            out->result.wins[1] = static_cast<int>(s->items()[1].as_int());
        }
    if (const featurize::Json *d = v.get("decisions"))
        if (d->is_integer()) out->decisions = static_cast<size_t>(d->as_int());
    if (const featurize::Json *s = v.get("seconds"))
        if (s->is_number()) out->seconds = s->as_number();
    if (const featurize::Json *l = v.get("log"))
        if (l->is_string()) out->log = in_dir(dir, l->as_string());
    if (const featurize::Json *e = v.get("stderr"))
        if (e->is_string()) out->stderr_path = in_dir(dir, e->as_string());
    if (const featurize::Json *e = v.get("error"))
        if (e->is_string()) out->error = e->as_string();
    return true;
}

// "0:42", "1:02:03".
static std::string clock_text(double seconds) {
    const long s = static_cast<long>(seconds);
    char buf[32];
    if (s >= 3600)
        std::snprintf(buf, sizeof buf, "%ld:%02ld:%02ld", s / 3600, s / 60 % 60, s % 60);
    else
        std::snprintf(buf, sizeof buf, "%ld:%02ld", s / 60, s % 60);
    return buf;
}

namespace rmwork {

std::string nn_options(bool argmax, double temperature) {
    if (argmax) return ":argmax";
    if (std::fabs(temperature - 1.0) < 1e-9) return "";
    char buf[48];
    std::snprintf(buf, sizeof buf, ":temp=%g", temperature);
    return buf;
}

bool sim_config_from(const SimulateSettings &s, SimConfig *out, std::string *why) {
    *out = SimConfig();
    for (size_t i = 0; i < 2; ++i) {
        const SimulateSeatSettings &seat = s.seats[i];
        const int kind = find_opponent(seat.agent);
        if (kind < 0) {
            *why = "no agent kind \"" + seat.agent + "\"";
            return false;
        }
        SimEntrant &e = out->entrants[i];
        e.agent.kind = static_cast<Opponent>(kind);
        e.agent.model = e.agent.kind == Opponent::MODEL    ? resolve_path(seat.model, "")
                        : e.agent.kind == Opponent::SERVER ? resolve_path(seat.server, "")
                                                           : "";
        if (opponent_needs_nn(e.agent.kind)) e.agent.options = nn_options(seat.argmax, seat.temperature);
        e.deck = seat.deck;
    }
    out->bo3 = s.bo3;
    out->auto_mana = s.auto_mana;
    out->seed = s.seed;
    out->matches = std::max(1, s.matches);
    out->alternate = s.alternate;
    out->jobs = s.jobs > 0 ? s.jobs : default_simulation_jobs();
    return true;
}

int default_simulation_jobs() {
    const unsigned hw = std::thread::hardware_concurrency();
    return std::max(1, static_cast<int>(hw / 4));
}

std::string sim_agent_text(const SeatAgent &a) {
    GameSidecar side;
    side.opponent = opponent_key(a.kind);
    side.model = a.model;
    std::string out = sidecar_opponent_text(side);
    if (!a.options.empty()) {
        std::string opts = a.options;
        std::replace(opts.begin(), opts.end(), ':', ' ');
        out += opts;
    }
    return out;
}

std::string sim_entrant_text(const SimEntrant &e) {
    return sim_agent_text(e.agent) + " \xC2\xB7 " + e.deck;
}

const char *sim_state_key(SimState s) {
    return STATE_KEYS[static_cast<int>(s)];
}

SimTally tally_simulation(const std::vector<SimMatch> &matches) {
    SimTally t;
    for (const SimMatch &m : matches) {
        switch (m.state) {
            case SimState::WAITING:
                ++t.waiting;
                continue;
            case SimState::RUNNING:
                ++t.running;
                continue;
            case SimState::FAILED:
                ++t.failed;
                continue;
            case SimState::CANCELLED:
                ++t.cancelled;
                continue;
            case SimState::FINISHED:
                ++t.finished;
                break;
        }
        const int w = m.winner();
        if (w < 0) {
            ++t.draws;
        } else {
            ++t.wins[w];
            ++t.seat_wins[m.winner_seat];
        }
        for (int g : m.result.games) {
            if (g < 0)
                ++t.game_draws;
            else
                ++t.game_wins[m.entrant(g)];
        }
    }
    if (t.finished > 0) {
        const double n = t.finished, p = (t.wins[0] + 0.5 * t.draws) / n;
        const double z2 = Z95 * Z95, centre = (p + z2 / (2 * n)) / (1 + z2 / n);
        const double half = Z95 * std::sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n);
        t.share = p;
        t.low = std::max(0.0, centre - half);
        t.high = std::min(1.0, centre + half);
    }
    return t;
}

std::string sim_result_text(const SimMatch &m) {
    switch (m.state) {
        case SimState::WAITING:
        case SimState::RUNNING:
        case SimState::FAILED:
        case SimState::CANCELLED:
            return sim_state_key(m.state);
        case SimState::FINISHED:
            break;
    }
    if (m.winner_seat < 0) return "draw";
    std::string out = m.winner_seat == 0 ? "A wins" : "B wins";
    if (m.result.match_line) out += " " + std::to_string(m.result.wins[0]) + "-" + std::to_string(m.result.wins[1]);
    return out;
}

void judge_match(SimMatch *m, bool exited_ok, const std::string &status, const std::string &stderr_text, bool bo3) {
    m->result = parse_result_lines(stderr_text);
    m->winner_seat = -1;
    m->error.clear();
    const std::string fatal = first_fatal_line(stderr_text);
    if (!fatal.empty()) {
        m->error = fatal;
    } else if (!exited_ok) {
        m->error = "the engine stopped (" + status + ")";
    } else if (!m->result.known(bo3)) {
        m->error = bo3 ? "no MATCH_RESULT line" : "no GAME_RESULT line";
    }
    if (!m->error.empty()) {
        m->state = SimState::FAILED;
        return;
    }
    m->state = SimState::FINISHED;
    m->winner_seat = bo3 ? m->result.match_winner : m->result.games.front();
}

std::vector<std::string> simulation_arguments(const SimConfig &c, const SimMatch &m, const std::string &log,
    const std::array<std::string, 2> &models, const std::string &repo, const std::string &python) {
    SeatAgent seats[2];
    std::string decks[2];
    for (int s = 0; s < 2; ++s) {
        const int e = m.entrant(s);
        seats[s] = c.entrants[static_cast<size_t>(e)].agent;
        if (seats[s].kind == Opponent::MODEL) seats[s].model = models[static_cast<size_t>(e)];
        decks[s] = c.entrants[static_cast<size_t>(e)].deck;
    }
    std::vector<std::string> a = {"--game-seed", std::to_string(m.seed), "--deck-a", decks[0], "--deck-b", decks[1]};
    if (c.bo3) a.push_back("--bo3");
    if (!c.auto_mana) a.push_back("--no-auto-mana");
    a.insert(a.end(), {"--decision-log", log});
    const std::vector<std::string> agents = agent_arguments(&seats[0], &seats[1], m.seed, repo, python);
    a.insert(a.end(), agents.begin(), agents.end());
    return a;
}

std::string simulation_description(const SimConfig &c) {
    std::string out;
    for (size_t i = 0; i < 2; ++i) {
        const SimEntrant &e = c.entrants[i];
        out += std::string(i ? "-vs-" : "") + opponent_key(e.agent.kind) + "-" + file_safe(base_name(e.deck));
    }
    return out;
}

std::string simulation_json(const SimConfig &c, const std::vector<SimMatch> &matches, const std::string &engine,
    const std::string &state, const std::string &dir) {
    using featurize::json_quote;
    const SimTally t = tally_simulation(matches);
    std::string out = "{\n  \"version\": 1,\n  \"state\": " + json_quote(state) + ",\n  \"engine\": " +
                      json_quote(engine) + ",\n  \"format\": " + json_quote(c.bo3 ? "bo3" : "bo1") +
                      ",\n  \"auto_mana\": " + (c.auto_mana ? "true" : "false") + ",\n  \"seed\": " +
                      std::to_string(c.seed) + ",\n  \"matches\": " + std::to_string(c.matches) +
                      ",\n  \"alternate\": " + (c.alternate ? "true" : "false") + ",\n  \"jobs\": " +
                      std::to_string(c.jobs) + ",\n  \"entrants\": [" + entrant_json(c.entrants[0]) + ", " +
                      entrant_json(c.entrants[1]) + "],\n";
    out += "  \"tally\": {\"finished\": " + std::to_string(t.finished) + ", \"failed\": " + std::to_string(t.failed) +
           ", \"cancelled\": " + std::to_string(t.cancelled) + ", \"wins\": [" + std::to_string(t.wins[0]) + ", " +
           std::to_string(t.wins[1]) + "], \"draws\": " + std::to_string(t.draws) + ", \"game_wins\": [" +
           std::to_string(t.game_wins[0]) + ", " + std::to_string(t.game_wins[1]) + "]},\n  \"results\": [";
    for (size_t i = 0; i < matches.size(); ++i) out += (i ? ",\n    " : "\n    ") + match_json(matches[i], dir);
    out += matches.empty() ? "]\n}\n" : "\n  ]\n}\n";
    return out;
}

bool parse_simulation(const std::string &text, const std::string &dir, SimConfig *c, std::vector<SimMatch> *matches,
    std::string *engine, std::string *state, std::string *err) {
    featurize::Json doc;
    if (!featurize::Json::parse(text, &doc, err)) return false;
    const featurize::Json *entrants = doc.get("entrants"), *results = doc.get("results");
    if (!doc.is_object() || !entrants || !entrants->is_array() || entrants->items().size() != 2 || !results ||
        !results->is_array()) {
        *err = "not a simulation summary (no two entrants, no results)";
        return false;
    }
    *c = SimConfig();
    for (size_t i = 0; i < 2; ++i)
        if (!read_entrant(entrants->items()[i], &c->entrants[i], err)) return false;
    const featurize::Json *v = nullptr;
    if ((v = doc.get("format")) && v->is_string()) c->bo3 = v->as_string() == "bo3";
    if ((v = doc.get("auto_mana")) && v->is_bool()) c->auto_mana = v->as_bool();
    if ((v = doc.get("seed")) && v->is_integer()) c->seed = static_cast<uint64_t>(v->as_int());
    if ((v = doc.get("matches")) && v->is_integer()) c->matches = static_cast<int>(v->as_int());
    if ((v = doc.get("alternate")) && v->is_bool()) c->alternate = v->as_bool();
    if ((v = doc.get("jobs")) && v->is_integer()) c->jobs = static_cast<int>(v->as_int());
    *engine = (v = doc.get("engine")) && v->is_string() ? v->as_string() : "";
    *state = (v = doc.get("state")) && v->is_string() ? v->as_string() : "";
    matches->clear();
    for (const featurize::Json &r : results->items()) {
        SimMatch m;
        if (!read_match(r, dir, &m, err)) return false;
        matches->push_back(std::move(m));
    }
    return true;
}

std::vector<SimulationEntry> list_simulations(const std::string &root, size_t max) {
    std::vector<std::string> names;
    if (DIR *d = opendir(root.c_str())) {
        while (const dirent *e = readdir(d)) {
            const std::string n = e->d_name;
            if (!n.empty() && n[0] != '.' && file_exists(root + "/" + n + "/" + SUMMARY_FILE)) names.push_back(n);
        }
        closedir(d);
    }
    // The names start with their timestamp: newest first.
    std::sort(names.begin(), names.end(), [](const std::string &a, const std::string &b) { return a > b; });
    if (names.size() > max) names.resize(max);
    std::vector<SimulationEntry> out;
    for (const std::string &n : names) {
        SimulationEntry e;
        e.dir = root + "/" + n;
        std::string text, err, engine, state;
        SimConfig c;
        std::vector<SimMatch> matches;
        if (!read_file(e.dir + "/" + SUMMARY_FILE, &text) ||
            !parse_simulation(text, e.dir, &c, &matches, &engine, &state, &err)) {
            e.label = n + " (" + (err.empty() ? "unreadable" : err) + ")";
            out.push_back(std::move(e));
            continue;
        }
        e.matches = static_cast<int>(matches.size());
        for (const SimMatch &m : matches) e.finished += m.state == SimState::FINISHED ? 1 : 0;
        const std::string when = n.size() >= 13 ? n.substr(4, 2) + "-" + n.substr(6, 2) + " " + n.substr(9, 2) + ":" +
                                                      n.substr(11, 2)
                                                : n;
        e.label = when + " \xC2\xB7 " + sim_entrant_text(c.entrants[0]) + " vs " + sim_entrant_text(c.entrants[1]) +
                  " \xC2\xB7 " + std::to_string(e.matches) + (c.bo3 ? " bo3" : " bo1");
        out.push_back(std::move(e));
    }
    return out;
}

Simulation::~Simulation() {
    if (running()) cancel();
}

bool Simulation::start(const SimConfig &c, const SimPaths &p, std::string *err) {
    if (running()) cancel();
    const SimEntrant &a = c.entrants[0], &b = c.entrants[1];
    const Opponent nn = opponent_needs_nn(a.agent.kind) ? a.agent.kind : b.agent.kind;
    std::string why;
    const std::string engine = absolute_path(play_engine_binary(absolute_path(p.engine), nn, &why));
    if (engine.empty()) {
        *err = why;
        return false;
    }
    std::array<std::string, 2> models;
    for (size_t i = 0; i < 2; ++i) {
        if (c.entrants[i].agent.kind != Opponent::MODEL) continue;
        models[i] = model_file(c.entrants[i].agent.model, &why);
        if (models[i].empty()) {
            *err = "the model of entrant " + std::to_string(i + 1) + " could not be read: " + why;
            return false;
        }
    }
    if (!make_dirs(p.root)) {
        *err = "cannot create the simulations directory " + p.root;
        return false;
    }
    const std::string base = p.root + "/" + file_timestamp() + "-" + simulation_description(c);
    std::string dir = base;
    for (int i = 2; path_kind(dir) != PathKind::MISSING; ++i) dir = base + "-" + std::to_string(i);
    if (!make_dirs(dir)) {
        *err = "cannot create " + dir;
        return false;
    }
    config_ = c;
    config_.matches = std::max(1, c.matches);
    config_.jobs = std::max(1, c.jobs);
    paths_ = p;
    dir_ = dir;
    engine_ = engine;
    models_ = models;
    matches_.clear();
    children_.clear();
    commands_.clear();
    started_.clear();
    for (int i = 0; i < config_.matches; ++i) {
        SimMatch m;
        m.index = i;
        m.seed = config_.seed + static_cast<uint64_t>(i);
        m.swapped = config_.alternate && i % 2 == 1;
        m.log = dir_ + "/" + match_stem(i) + ".rmlog";
        m.stderr_path = dir_ + "/" + match_stem(i) + ".stderr";
        matches_.push_back(std::move(m));
        children_.emplace_back();
        commands_.emplace_back();
        started_.emplace_back();
    }
    next_ = 0;
    finished_event_ = false;
    elapsed_ = 0;
    began_ = std::chrono::steady_clock::now();
    state_ = State::RUNNING;
    write_summary();
    poll();
    return true;
}

void Simulation::poll() {
    if (state_ != State::RUNNING) return;
    bool changed = false;
    int running = 0;
    for (size_t i = 0; i < matches_.size(); ++i) {
        if (matches_[i].state != SimState::RUNNING) continue;
        if (children_[i]->poll()) {
            ++running;
            continue;
        }
        reap(i);
        changed = true;
    }
    while (running < config_.jobs && next_ < matches_.size()) {
        launch(next_++);
        changed = true;
        if (matches_[next_ - 1].state == SimState::RUNNING) ++running;
    }
    if (running == 0 && next_ >= matches_.size()) {
        state_ = State::DONE;
        elapsed_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - began_).count();
        finished_event_ = true;
        changed = true;
    }
    if (changed) write_summary();
}

void Simulation::launch(size_t i) {
    SimMatch &m = matches_[i];
    commands_[i] = {engine_};
    const std::vector<std::string> args =
        simulation_arguments(config_, m, m.log, models_, paths_.repo, paths_.python);
    commands_[i].insert(commands_[i].end(), args.begin(), args.end());
    children_[i].reset(new ChildProcess());
    std::string err;
    started_[i] = std::chrono::steady_clock::now();
    // With no seat on a stream the engine writes its result lines to stdout: both go to the one file.
    if (!children_[i]->start(commands_[i], paths_.workdir, m.stderr_path, m.stderr_path, &err, true)) {
        m.state = SimState::FAILED;
        m.error = err;
        children_[i].reset();
        return;
    }
    m.state = SimState::RUNNING;
}

// A match whose engine exited: judged from its exit and stderr; its log's decisions counted.
void Simulation::reap(size_t i) {
    SimMatch &m = matches_[i];
    ChildProcess &child = *children_[i];
    // What the engine started (a teacher) goes with it.
    child.kill_and_reap();
    m.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_[i]).count();
    std::string text;
    read_file(m.stderr_path, &text);
    judge_match(&m, child.ok(), child.status_text(), text, config_.bo3);
    std::string bytes, err;
    RmlogInfo info;
    if (read_file(m.log, &bytes) && parse_rmlog(bytes, &info, &err)) m.decisions = info.replies.size();
    children_[i].reset();
}

void Simulation::cancel() {
    if (state_ != State::RUNNING) return;
    for (size_t i = 0; i < matches_.size(); ++i) {
        SimMatch &m = matches_[i];
        if (m.state == SimState::RUNNING) {
            children_[i]->kill_and_reap();
            children_[i].reset();
            m.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_[i]).count();
            std::remove(m.log.c_str());
            m.log.clear();
            m.state = SimState::CANCELLED;
        } else if (m.state == SimState::WAITING) {
            m.log.clear();
            m.stderr_path.clear();
            m.state = SimState::CANCELLED;
        }
    }
    next_ = matches_.size();
    elapsed_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - began_).count();
    state_ = State::CANCELLED;
    write_summary();
}

bool Simulation::open(const std::string &dir, std::string *err) {
    const std::string d = absolute_path(dir);
    std::string text, engine, state;
    if (!read_file(d + "/" + SUMMARY_FILE, &text)) {
        *err = "no " + std::string(SUMMARY_FILE) + " in " + d;
        return false;
    }
    SimConfig c;
    std::vector<SimMatch> matches;
    if (!parse_simulation(text, d, &c, &matches, &engine, &state, err)) {
        *err = d + "/" + SUMMARY_FILE + ": " + *err;
        return false;
    }
    if (running()) cancel();
    bool unfinished = false;
    for (SimMatch &m : matches)
        if (m.state == SimState::WAITING || m.state == SimState::RUNNING) {
            m.state = SimState::CANCELLED;
            unfinished = true;
        }
    config_ = c;
    matches_ = std::move(matches);
    dir_ = d;
    engine_ = engine;
    children_.clear();
    children_.resize(matches_.size());
    commands_.assign(matches_.size(), {});
    started_.assign(matches_.size(), std::chrono::steady_clock::time_point());
    next_ = matches_.size();
    elapsed_ = 0;
    for (const SimMatch &m : matches_) elapsed_ = std::max(elapsed_, m.seconds);
    finished_event_ = false;
    state_ = unfinished || state == "cancelled" ? State::CANCELLED : State::DONE;
    return true;
}

void Simulation::close() {
    if (running()) cancel();
    state_ = State::NONE;
    matches_.clear();
    children_.clear();
    commands_.clear();
    started_.clear();
    dir_.clear();
}

bool Simulation::take_finished() {
    const bool f = finished_event_;
    finished_event_ = false;
    return f;
}

double Simulation::fraction() const {
    if (matches_.empty()) return 0;
    size_t ended = 0;
    for (const SimMatch &m : matches_) ended += m.state == SimState::WAITING || m.state == SimState::RUNNING ? 0 : 1;
    return static_cast<double>(ended) / static_cast<double>(matches_.size());
}

std::string Simulation::progress_text() const {
    const SimTally t = tally_simulation(matches_);
    const int ended = t.finished + t.failed + t.cancelled;
    std::string out = std::to_string(ended) + " / " + std::to_string(matches_.size()) +
                      (matches_.size() == 1 ? " match" : " matches");
    if (t.running) out += " \xC2\xB7 " + std::to_string(t.running) + " running";
    if (t.failed) out += " \xC2\xB7 " + std::to_string(t.failed) + " failed";
    if (t.cancelled) out += " \xC2\xB7 " + std::to_string(t.cancelled) + " cancelled";
    const double s = running() ? std::chrono::duration<double>(std::chrono::steady_clock::now() - began_).count()
                               : elapsed_;
    if (s > 0) out += " \xC2\xB7 " + clock_text(s);
    return out;
}

std::string Simulation::stderr_tail(size_t i, size_t max_bytes) const {
    std::string text;
    if (i >= matches_.size() || matches_[i].stderr_path.empty() || !read_file(matches_[i].stderr_path, &text))
        return "";
    return tail_of(text, max_bytes);
}

const char *Simulation::state_key() const {
    switch (state_) {
        case State::RUNNING:
            return "running";
        case State::CANCELLED:
            return "cancelled";
        case State::DONE:
        case State::NONE:
            break;
    }
    return "done";
}

void Simulation::write_summary() {
    std::string err;
    write_file_atomic(summary_path(), simulation_json(config_, matches_, engine_, state_key(), dir_), &err);
}

}  // namespace rmwork
