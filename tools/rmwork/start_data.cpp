#include "start_data.h"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <set>

#include "../../src/featurize/json.h"
#include "file_picker.h"
#include "files.h"
#include "paths.h"
#include "replay_cache.h"
#include "rmlog.h"

namespace {

// How much of a Play game's stderr and of a cached narrative is read for the result lines: the
// MATCH_RESULT and the last GAME_RESULT are at the end.
const size_t STDERR_TAIL = 256 * 1024;
const size_t NARRATIVE_TAIL = 16 * 1024;
// A decision log larger than this is not read for the page (Play's are a few KB).
const uint64_t LOG_MAX_BYTES = 8u << 20;
const char *const ELLIPSIS = "\xE2\x80\xA6";

}  // namespace

static bool ends_with(const std::string &s, const std::string &suffix);
static std::string base_name(const std::string &path);
static std::string strip_suffix(const std::string &s, const std::string &suffix);
static std::vector<std::string> list_names(const std::string &dir);
static bool stat_path(const std::string &path, struct stat *st);
static bool read_tail(const std::string &path, size_t max_bytes, std::string *out);
static int64_t stem_time(const std::string &stem);
static int parse_player(const std::string &word);
static bool under_dir(const std::string &path, const std::string &dir);
static std::string file_time(const std::string &path, int64_t *mtime);
static void cache_size(const std::string &root, uint64_t *bytes, size_t *entries);

static bool ends_with(const std::string &s, const std::string &suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::string base_name(const std::string &path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

static std::string strip_suffix(const std::string &s, const std::string &suffix) {
    return ends_with(s, suffix) ? s.substr(0, s.size() - suffix.size()) : s;
}

// The directory's entry names (not . and ..); empty when it cannot be read.
static std::vector<std::string> list_names(const std::string &dir) {
    std::vector<std::string> out;
    DIR *d = opendir(dir.c_str());
    if (!d) return out;
    while (const dirent *e = readdir(d)) {
        const std::string name = e->d_name;
        if (name != "." && name != "..") out.push_back(name);
    }
    closedir(d);
    return out;
}

static bool stat_path(const std::string &path, struct stat *st) {
    return stat(path.c_str(), st) == 0;
}

// The last `max_bytes` of the file (the whole of a smaller one).
static bool read_tail(const std::string &path, size_t max_bytes, std::string *out) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    out->clear();
    if (std::fseek(f, 0, SEEK_END) == 0) {
        const long size = std::ftell(f);
        const long from = size > static_cast<long>(max_bytes) ? size - static_cast<long>(max_bytes) : 0;
        std::fseek(f, from, SEEK_SET);
    }
    char buf[8192];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out->append(buf, n);
    std::fclose(f);
    return true;
}

// Play's stem "20261116-191502-3" as local time, 0 when it is not one.
static int64_t stem_time(const std::string &stem) {
    std::tm tm{};
    int y, mo, d, h, mi, s;
    char tail;
    if (std::sscanf(stem.c_str(), "%4d%2d%2d-%2d%2d%2d%c", &y, &mo, &d, &h, &mi, &s, &tail) != 7 || tail != '-')
        return 0;
    tm.tm_year = y - 1900;
    tm.tm_mon = mo - 1;
    tm.tm_mday = d;
    tm.tm_hour = h;
    tm.tm_min = mi;
    tm.tm_sec = s;
    tm.tm_isdst = -1;
    const std::time_t t = std::mktime(&tm);
    return t < 0 ? 0 : static_cast<int64_t>(t);
}

// "A" -> 0, "B" -> 1, else -1.
static int parse_player(const std::string &word) {
    return word == "A" ? 0 : word == "B" ? 1 : -1;
}

static bool under_dir(const std::string &path, const std::string &dir) {
    return !dir.empty() && path.size() > dir.size() && path.compare(0, dir.size(), dir) == 0 && path[dir.size()] == '/';
}

// The file's modification time, "11-17 15:31" ("" when it is missing).
static std::string file_time(const std::string &path, int64_t *mtime) {
    struct stat st;
    if (!stat_path(path, &st)) return "";
    *mtime = static_cast<int64_t>(st.st_mtime);
    return rmwork::short_time(*mtime);
}

// The replay cache's complete entries and their files' total size.
static void cache_size(const std::string &root, uint64_t *bytes, size_t *entries) {
    for (const std::string &key : list_names(root)) {
        const std::string dir = root + "/" + key;
        struct stat st;
        if (!stat_path(dir, &st) || !S_ISDIR(st.st_mode)) continue;
        if (key.find(".tmp-") == std::string::npos) ++*entries;
        for (const std::string &f : list_names(dir))
            if (stat_path(dir + "/" + f, &st) && S_ISREG(st.st_mode)) *bytes += static_cast<uint64_t>(st.st_size);
    }
}

namespace rmwork {

LogResult parse_result_lines(const std::string &text) {
    LogResult r;
    size_t start = 0;
    while (start < text.size()) {
        size_t nl = text.find('\n', start);
        if (nl == std::string::npos) nl = text.size();
        const std::string line = text.substr(start, nl - start);
        start = nl + 1;
        char who[8] = "", what[16] = "";
        int n = 0, a = 0, b = 0;
        if (std::sscanf(line.c_str(), "GAME_RESULT: %d Player %7s %15s", &n, who, what) == 3 &&
            std::string(what) == "wins") {
            r.games.push_back(parse_player(who));
        } else if (std::sscanf(line.c_str(), "GAME_RESULT: %d %15s", &n, what) == 2 && std::string(what) == "draw") {
            r.games.push_back(-1);
        } else if (std::sscanf(line.c_str(), "MATCH_RESULT: Player %7s wins %d-%d", who, &a, &b) == 3) {
            r.match_line = true;
            r.match_winner = parse_player(who);
            r.wins[0] = a;
            r.wins[1] = b;
        }
    }
    return r;
}

std::string short_time(int64_t seconds) {
    const std::time_t t = static_cast<std::time_t>(seconds);
    std::tm tm;
    localtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, "%m-%d %H:%M", &tm);
    return buf;
}

std::string game_result_text(const RecentGame &g, int *tone) {
    *tone = 0;
    switch (g.state) {
        case GameState::MALFORMED:
            *tone = -1;
            return "not a decision log";
        case GameState::PLAYING:
            *tone = 2;
            return "playing";
        case GameState::STOPPED:
            *tone = 2;
            return "no result (stopped)";
        case GameState::UNKNOWN:
            return ELLIPSIS;
        case GameState::FINISHED:
            break;
    }
    const LogResult &r = g.result;
    int winner = -1;
    std::string score;
    if (r.match_line) {
        winner = r.match_winner;
        if (g.bo3) {
            const int me = g.seat == 1 ? 1 : 0;
            score = " " + std::to_string(r.wins[me]) + "-" + std::to_string(r.wins[1 - me]);
        }
    } else if (!r.games.empty()) {
        winner = r.games.back();
    }
    if (winner < 0) return "draw";
    if (g.seat < 0) return std::string(winner == 0 ? "A" : "B") + " wins" + score;
    *tone = winner == g.seat ? 1 : -1;
    return (winner == g.seat ? "won" : "lost") + score;
}

const char *game_format_text(const RecentGame &g) {
    return g.bo3 ? "bo3" : "bo1";
}

std::string game_own_deck(const RecentGame &g) {
    return g.seat == 1 ? g.deck_b : g.deck_a;
}

std::string game_opponent_deck(const RecentGame &g) {
    return g.seat == 1 ? g.deck_a : g.deck_b;
}

std::string sidecar_path(const std::string &log) {
    return strip_suffix(log, ".rmlog") + ".game.json";
}

std::string sidecar_json(const GameSidecar &s) {
    using featurize::json_quote;
    return std::string("{\"seat\": ") + json_quote(s.seat == 1 ? "b" : "a") +
           ", \"opponent\": " + json_quote(s.opponent) + ", \"model\": " + json_quote(s.model) +
           ", \"deck\": " + json_quote(s.deck) + ", \"opponent_deck\": " + json_quote(s.opponent_deck) +
           ", \"format\": " + json_quote(s.bo3 ? "bo3" : "bo1") + ", \"seed\": " + std::to_string(s.seed) + "}\n";
}

bool parse_sidecar(const std::string &text, GameSidecar *out) {
    featurize::Json doc;
    std::string err;
    if (!featurize::Json::parse(text, &doc, &err) || !doc.is_object()) return false;
    const featurize::Json *seat = doc.get("seat"), *opp = doc.get("opponent");
    if (!seat || !seat->is_string() || !opp || !opp->is_string()) return false;
    *out = GameSidecar();
    out->seat = seat->as_string() == "b" ? 1 : 0;
    out->opponent = opp->as_string();
    if (const featurize::Json *v = doc.get("model"))
        if (v->is_string()) out->model = v->as_string();
    if (const featurize::Json *v = doc.get("deck"))
        if (v->is_string()) out->deck = v->as_string();
    if (const featurize::Json *v = doc.get("opponent_deck"))
        if (v->is_string()) out->opponent_deck = v->as_string();
    if (const featurize::Json *v = doc.get("format")) out->bo3 = v->is_string() && v->as_string() == "bo3";
    if (const featurize::Json *v = doc.get("seed"))
        if (v->is_number() && v->as_number() >= 0) out->seed = static_cast<uint64_t>(v->as_number());
    return true;
}

std::string sidecar_opponent_text(const GameSidecar &s) {
    if (s.opponent == "model") return "nn: " + base_name(s.model);
    if (s.opponent == "server") return "nn:remote: " + base_name(s.model);
    if (s.opponent == "scripted") return "scripted (hard)";
    return s.opponent;
}

RecentGame read_recent_game(const std::string &log, bool play, const StartScanRequest &r) {
    RecentGame g;
    g.path = log;
    const std::string stem = strip_suffix(log, ".rmlog");
    struct stat st;
    const bool exists = stat_path(log, &st);
    g.time = stem_time(base_name(stem));
    if (g.time == 0 && exists) g.time = static_cast<int64_t>(st.st_mtime);
    g.when = g.time ? short_time(g.time) : ELLIPSIS;
    std::string bytes, err;
    RmlogInfo info;
    if (!exists || static_cast<uint64_t>(st.st_size) > LOG_MAX_BYTES || !read_file(log, &bytes) ||
        !parse_rmlog(bytes, &info, &err)) {
        g.state = GameState::MALFORMED;
        g.error = !exists ? "the file is gone" : err.empty() ? "cannot read it" : err;
        return g;
    }
    g.deck_a = info.deck_a;
    g.deck_b = info.deck_b;
    g.bo3 = info.has_flag("bo3");
    g.decisions = info.replies.size();
    std::string text;
    if (play) {
        GameSidecar side;
        if (read_file(sidecar_path(log), &text) && parse_sidecar(text, &side)) {
            g.seat = side.seat;
            g.opponent = sidecar_opponent_text(side);
            g.game_seed = side.seed;
        } else if (file_exists(stem + ".seat-a.rmstate")) {
            g.seat = 0;
        } else if (file_exists(stem + ".seat-b.rmstate")) {
            g.seat = 1;
        }
        if (log == r.live_log) {
            g.state = GameState::PLAYING;
            return g;
        }
        if (read_tail(stem + ".stderr", STDERR_TAIL, &text)) {
            g.result = parse_result_lines(text);
            g.state = g.result.known(g.bo3) ? GameState::FINISHED : GameState::STOPPED;
            if (g.state == GameState::FINISHED) return g;
        }
    }
    // The replay cache's narrative of this log, under the engine replays run with.
    const EngineIdentity engine = engine_identity(r.replay_engine);
    if (!engine.found) return g;
    const CacheEntry e = cache_entry(r.cache_root, replay_cache_key(bytes, engine));
    if (!cache_entry_complete(e) || !read_tail(e.narrative, NARRATIVE_TAIL, &text)) return g;
    const LogResult cached = parse_result_lines(text);
    if (!cached.known(g.bo3)) return g;
    g.result = cached;
    g.cached = true;
    g.state = GameState::FINISHED;
    return g;
}

bool is_run_directory(const std::string &dir) {
    return path_kind(dir + "/status.json") == PathKind::FILE || path_kind(dir + "/rmlog") == PathKind::DIRECTORY;
}

RecentRun read_recent_run(const std::string &dir, bool recent) {
    RecentRun run;
    run.path = dir;
    run.name = base_name(dir);
    run.recent = recent;
    struct stat st;
    if (stat_path(dir, &st)) run.updated = static_cast<double>(st.st_mtime);
    std::string text;
    if (!read_file(dir + "/status.json", &text)) return run;
    featurize::Json doc;
    if (!featurize::Json::parse(text, &doc, &run.error) || !doc.is_object()) {
        if (run.error.empty()) run.error = "not a JSON object";
        return run;
    }
    run.status = true;
    if (const featurize::Json *v = doc.get("phase"))
        if (v->is_string()) run.phase = v->as_string();
    if (const featurize::Json *v = doc.get("running")) run.running = v->truthy();
    if (const featurize::Json *v = doc.get("updated_at"))
        if (v->is_number()) run.updated = v->as_number();
    if (const featurize::Json *v = doc.get("uptime_s"))
        if (v->is_number()) run.hours = v->as_number() / 3600.0;
    if (const featurize::Json *totals = doc.get("totals"))
        if (const featurize::Json *v = totals->get("games"))
            if (v->is_number()) run.games = v->as_int();
    return run;
}

StartData scan_start(const StartScanRequest &r) {
    StartData d;
    // Play's games: the newest by name (their names start with the time), then read.
    d.games_dir_exists = path_kind(r.games_dir) == PathKind::DIRECTORY;
    std::vector<std::string> logs;
    for (const std::string &name : list_names(r.games_dir))
        if (ends_with(name, ".rmlog")) logs.push_back(name);
    d.games_total = logs.size();
    std::sort(logs.begin(), logs.end(), [](const std::string &a, const std::string &b) { return natural_less(b, a); });
    if (logs.size() > r.max_games) logs.resize(r.max_games);
    for (const std::string &name : logs) d.games.push_back(read_recent_game(r.games_dir + "/" + name, true, r));
    std::stable_sort(
        d.games.begin(), d.games.end(), [](const RecentGame &a, const RecentGame &b) { return a.time > b.time; });
    // Recent decision logs elsewhere, and recent directories that hold runs.
    std::set<std::string> seen;
    for (const std::string &path : r.recent) {
        const PathKind kind = path_kind(path);
        if (kind == PathKind::FILE && ends_with(path, ".rmlog") && !under_dir(path, r.games_dir) &&
            d.replays.size() < r.max_replays) {
            d.replays.push_back(read_recent_game(path, false, r));
        } else if (kind == PathKind::DIRECTORY && d.runs.size() < r.max_runs && seen.insert(path).second) {
            d.runs.push_back(read_recent_run(path, true));
        }
    }
    std::vector<RecentRun> found;
    for (const std::string &name : list_names(r.run_dir)) {
        const std::string dir = r.run_dir + "/" + name;
        if (seen.count(dir) || path_kind(dir) != PathKind::DIRECTORY || !is_run_directory(dir)) continue;
        found.push_back(read_recent_run(dir, false));
    }
    std::sort(found.begin(), found.end(), [](const RecentRun &a, const RecentRun &b) {
        return a.updated != b.updated ? a.updated > b.updated : a.name < b.name;
    });
    for (RecentRun &run : found) {
        if (d.runs.size() >= r.max_runs) break;
        d.runs.push_back(std::move(run));
    }
    cache_size(r.cache_root, &d.cache_bytes, &d.cache_entries);
    return d;
}

StatusBlock engine_status(const std::string &binary, const std::string &workdir) {
    StatusBlock b;
    b.title = "Engine";
    int64_t engine_time = 0, nn_time = 0, plain_time = 0;
    const bool found = is_executable(binary);
    b.health = found ? Health::GOOD : Health::BAD;
    b.lines.push_back(
        {"binary", display_path(binary) + (found ? "" : " (missing)"), found ? Health::NONE : Health::BAD});
    if (found) b.lines.push_back({"built", file_time(binary, &engine_time), Health::NONE});
    const PathKind wd = path_kind(workdir);
    const bool resources = path_kind(workdir + "/resources") == PathKind::DIRECTORY;
    b.lines.push_back({"working dir",
        display_path(workdir) + (wd != PathKind::DIRECTORY ? " (missing)"
                                    : resources            ? ""
                                                           : " (no resources/ in it)"),
        wd == PathKind::DIRECTORY && resources ? Health::NONE : Health::BAD});
    if (wd != PathKind::DIRECTORY || !resources) b.health = Health::BAD;
    const std::string dir = parent_dir(binary);
    const bool is_nn = base_name(binary) == "robomage_nn";
    const std::string plain = dir + "/robomage", nn = dir + "/robomage_nn";
    if (is_nn && is_executable(plain)) {
        b.lines.push_back(
            {"replays with", display_path(plain) + " \xC2\xB7 built " + file_time(plain, &plain_time), Health::NONE});
        nn_time = engine_time;
        engine_time = plain_time;
    } else if (!is_nn && is_executable(nn)) {
        b.lines.push_back({"robomage_nn", "built " + file_time(nn, &nn_time) + " (nn: opponents)", Health::NONE});
    } else if (!is_nn) {
        b.lines.push_back({"robomage_nn", "none beside it: an nn: opponent needs it", Health::WARN});
        b.fix = "make actor";
    }
    if (nn_time && engine_time && nn_time + 3600 < engine_time) {
        const long days = static_cast<long>((engine_time - nn_time) / 86400);
        b.lines.push_back({"",
            "robomage_nn is " + (days >= 1 ? std::to_string(days) + " days" : std::string("hours")) +
                " older than robomage: an nn: opponent plays on that build",
            Health::WARN});
        b.fix = "make actor";
        if (b.health == Health::GOOD) b.health = Health::WARN;
    }
    return b;
}

StatusBlock image_status(const std::string &cache_dir, const ImageCacheStatus &s) {
    StatusBlock b;
    b.title = "Card images";
    b.lines.push_back({"cache", display_path(cache_dir), Health::NONE});
    if (!s.manifest) {
        b.health = Health::WARN;
        b.lines.push_back(
            {"cards", s.error.empty() ? "no images: cards are drawn as text" : "the manifest does not read: " + s.error,
                Health::WARN});
        b.fix = "python3 tools/scryfall/fetch_images.py";
        return b;
    }
    const int missing = s.vocab - s.with_images;
    b.health = missing == 0 ? Health::GOOD : Health::WARN;
    b.lines.push_back({"cards",
        std::to_string(s.with_images) + " of " + std::to_string(s.vocab) + " cached" +
            (missing ? " \xC2\xB7 " + std::to_string(missing) + " drawn as text" : ""),
        missing ? Health::WARN : Health::NONE});
    b.lines.push_back(
        {"tokens", std::to_string(s.token_images) + " of " + std::to_string(s.tokens) + " cached", Health::NONE});
    if (missing) b.fix = "python3 tools/scryfall/fetch_images.py";
    return b;
}

StatusBlock inference_status(const std::string &socket, size_t policies, const std::string &engine,
    const std::string &read_state, bool read_failed) {
    StatusBlock b;
    b.title = "Inference server";
    const bool present = !socket.empty() && path_kind(socket) == PathKind::SOCKET;
    b.health = read_failed ? Health::BAD : present ? Health::GOOD : Health::NONE;
    b.lines.push_back({"socket", socket.empty() ? "none configured" : display_path(socket), Health::NONE});
    b.lines.push_back({"state",
        socket.empty() ? "Model's Read and nn:remote: need one"
        : present      ? "socket present"
                       : "no server at the socket",
        present ? Health::GOOD : Health::NONE});
    b.lines.push_back(
        {"policies", policies ? std::to_string(policies) + " listed" : "none listed (policy 0 asked)", Health::NONE});
    if (!read_state.empty()) b.lines.push_back({"last read", read_state, read_failed ? Health::BAD : Health::NONE});
    const std::string rmserve = parent_dir(engine) + "/rmserve";
    int64_t t = 0;
    if (is_executable(rmserve)) {
        b.lines.push_back(
            {"rmserve", display_path(rmserve) + " \xC2\xB7 built " + file_time(rmserve, &t), Health::NONE});
    } else {
        b.lines.push_back({"rmserve", "not built", Health::NONE});
        b.fix = "make actor";
    }
    return b;
}

StatusBlock files_status(const StartData &d, const std::string &games_dir, const std::string &cache_root) {
    StatusBlock b;
    b.title = "Files";
    b.health = Health::NONE;
    b.lines.push_back({"games",
        display_path(games_dir) + " \xC2\xB7 " +
            (d.games_dir_exists ? std::to_string(d.games_total) + " game" + (d.games_total == 1 ? "" : "s")
                                : std::string("not made yet")),
        Health::NONE});
    b.lines.push_back({"replay cache",
        display_path(cache_root) + " \xC2\xB7 " + std::to_string(d.cache_entries) + " replays, " +
            byte_size_text(d.cache_bytes) + " of " + byte_size_text(CACHE_MAX_BYTES),
        Health::NONE});
    return b;
}

std::string display_path(const std::string &path) {
    const std::string repo = repo_root();
    if (under_dir(path, repo)) return path.substr(repo.size() + 1);
    const char *home = std::getenv("HOME");
    if (home && *home && under_dir(path, home)) return "~" + path.substr(std::string(home).size());
    return path;
}

std::string byte_size_text(uint64_t bytes) {
    char buf[32];
    if (bytes >= (1ull << 30)) {
        std::snprintf(buf, sizeof buf, "%.1f GB", static_cast<double>(bytes) / (1ull << 30));
    } else if (bytes >= (1ull << 20)) {
        std::snprintf(buf, sizeof buf, "%.0f MB", static_cast<double>(bytes) / (1ull << 20));
    } else {
        std::snprintf(buf, sizeof buf, "%.0f KB", static_cast<double>(bytes) / 1024.0);
    }
    return buf;
}

}  // namespace rmwork
