#include "selftest_start.h"

#include <sys/stat.h>
#include <sys/time.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "files.h"
#include "imgui_internal.h"
#include "paths.h"
#include "replay_cache.h"

namespace {

using rmwork::CheckFn;
using Frames = std::function<void(int)>;

const double LOAD_TIMEOUT_S = 300;
const char *const START_WINDOW = "##rmwork.start";

}  // namespace

static std::string rmlog_text(const std::string &flags, const std::vector<int> &replies, const std::string &last);
static bool write(const std::string &path, const std::string &text);
static bool make_executable(const std::string &path);
static void set_mtime(const std::string &path, int64_t seconds);
static const rmwork::RecentGame *find_game(const std::vector<rmwork::RecentGame> &v, const std::string &path);
static const rmwork::RecentRun *find_run(const std::vector<rmwork::RecentRun> &v, const std::string &path);
static std::string result_of(const rmwork::RecentGame &g);
static bool window_shown(const char *name);
static bool run_until(const Frames &frames, const std::function<bool()> &done, double timeout_s);
static void check_result_lines(const CheckFn &check);
static void check_sidecar(const CheckFn &check);
static void check_scan(const CheckFn &check, const std::string &dir);
static void check_status(const CheckFn &check, const std::string &dir);
static void press_chord(rmwork::App &app, ImGuiKeyChord chord);
static void frames_of(rmwork::App &app, int n);

// A decision log's text: the header, two one-card decklists, `replies` and the last line.
static std::string rmlog_text(const std::string &flags, const std::vector<int> &replies, const std::string &last) {
    std::string t = "RMLOG v3\nSEED 7\nFLAGS " + flags + "\nDECK_A league/burn\n60 Mountain\nEND_DECK\n" +
                    "DECK_B league/bug\n60 Forest\nEND_DECK\nDECISIONS\n";
    for (int r : replies) t += std::to_string(r) + "\n";
    return t + last;
}

static bool write(const std::string &path, const std::string &text) {
    std::string err;
    return rmwork::make_dirs(rmwork::parent_dir(path)) && rmwork::write_file_atomic(path, text, &err);
}

static bool make_executable(const std::string &path) {
    return write(path, "#!/bin/sh\nexit 0\n") && chmod(path.c_str(), 0755) == 0;
}

static void set_mtime(const std::string &path, int64_t seconds) {
    timeval tv[2];
    tv[0].tv_sec = tv[1].tv_sec = static_cast<time_t>(seconds);
    tv[0].tv_usec = tv[1].tv_usec = 0;
    utimes(path.c_str(), tv);
}

static const rmwork::RecentGame *find_game(const std::vector<rmwork::RecentGame> &v, const std::string &path) {
    for (const rmwork::RecentGame &g : v)
        if (g.path == path) return &g;
    return nullptr;
}

static const rmwork::RecentRun *find_run(const std::vector<rmwork::RecentRun> &v, const std::string &path) {
    for (const rmwork::RecentRun &r : v)
        if (r.path == path) return &r;
    return nullptr;
}

static std::string result_of(const rmwork::RecentGame &g) {
    int tone = 0;
    return rmwork::game_result_text(g, &tone);
}

// The window was submitted in the last frame.
static bool window_shown(const char *name) {
    const ImGuiWindow *w = ImGui::FindWindowByName(name);
    return w && w->Active;
}

static bool run_until(const Frames &frames, const std::function<bool()> &done, double timeout_s) {
    const auto start = std::chrono::steady_clock::now();
    while (!done()) {
        frames(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > timeout_s) return false;
    }
    frames(2);
    return true;
}

static void frames_of(rmwork::App &app, int n) {
    for (int i = 0; i < n; ++i) {
        app.begin_frame();
        ImGui::NewFrame();
        app.frame();
        ImGui::Render();
    }
}

static void press_chord(rmwork::App &app, ImGuiKeyChord chord) {
    ImGuiIO &io = ImGui::GetIO();
    const ImGuiKeyChord mods = chord & ImGuiMod_Mask_;
    const ImGuiKey key = static_cast<ImGuiKey>(chord & ~ImGuiMod_Mask_);
    const ImGuiKey mod_keys[] = {ImGuiMod_Ctrl, ImGuiMod_Shift, ImGuiMod_Alt};
    for (ImGuiKey m : mod_keys)
        if (mods & m) io.AddKeyEvent(m, true);
    io.AddKeyEvent(key, true);
    frames_of(app, 1);
    io.AddKeyEvent(key, false);
    for (ImGuiKey m : mod_keys)
        if (mods & m) io.AddKeyEvent(m, false);
    frames_of(app, 1);
}

static void check_result_lines(const CheckFn &check) {
    using namespace rmwork;
    LogResult r =
        parse_result_lines("robomage 0.3\n\nPlayer B has -1 life - Player A wins!\nGAME_RESULT: 1 Player A wins\n");
    check(r.games.size() == 1 && r.games[0] == 0 && !r.match_line && r.known(false) && !r.known(true),
        "Start: a best of one's GAME_RESULT line");
    r = parse_result_lines(
        "GAME_RESULT: 1 Player B wins\nGAME_RESULT: 2 draw\nGAME_RESULT: 3 Player A wins\n"
        "GAME_RESULT: 4 Player B wins\nMATCH_RESULT: Player B wins 1-2\n");
    check(r.games == std::vector<int>({1, -1, 0, 1}) && r.match_line && r.match_winner == 1 && r.wins[0] == 1 &&
              r.wins[1] == 2 && r.known(true),
        "Start: a best of three's result lines");
    r = parse_result_lines("GAME_RESULT: x\nMATCH_RESULT: nobody\nPlayer A wins\n");
    check(r.games.empty() && !r.match_line, "Start: lines that are no result lines read as results");
}

static void check_sidecar(const CheckFn &check) {
    using namespace rmwork;
    GameSidecar s;
    s.seat = 1;
    s.opponent = "model";
    s.model = "/runs/x/exports/final.ts.pt";
    s.deck = "league/bug";
    s.opponent_deck = "league/burn";
    s.bo3 = true;
    s.seed = 42;
    GameSidecar back;
    check(parse_sidecar(sidecar_json(s), &back) && back.seat == 1 && back.opponent == "model" &&
              back.model == s.model && back.deck == s.deck && back.opponent_deck == s.opponent_deck && back.bo3 &&
              back.seed == 42,
        "Start: the game sidecar does not round-trip");
    check(sidecar_opponent_text(back) == "nn: final.ts.pt",
        "Start: a model opponent's name: " + sidecar_opponent_text(back));
    back.opponent = "scripted";
    check(sidecar_opponent_text(back) == "scripted (hard)", "Start: the scripted opponent's name");
    check(!parse_sidecar("{\"seat\": 1}", &back) && !parse_sidecar("[]", &back), "Start: a bad sidecar is read");
    check(sidecar_path("/g/20261101-100000-1.rmlog") == "/g/20261101-100000-1.game.json", "Start: the sidecar's path");
}

// scan_start over synthetic files under `dir`.
static void check_scan(const CheckFn &check, const std::string &dir) {
    using namespace rmwork;
    const std::string games = dir + "/games";
    const auto game = [&games](const std::string &stem) { return games + "/" + stem + ".rmlog"; };
    bool ok = true;
    // A finished best of one (seat A, explore), a best of three won as seat B against a model.
    ok &= write(game("20261101-100000-1"), rmlog_text("machine", {0, 1, 2}, ""));
    ok &= write(games + "/20261101-100000-1.stderr", "robomage 0.3\n\nGAME_RESULT: 1 Player A wins\n");
    GameSidecar side;
    side.opponent = "explore";
    ok &= write(sidecar_path(game("20261101-100000-1")), sidecar_json(side));
    ok &= write(game("20261102-100000-2"), rmlog_text("machine bo3", {0, 0, 1, 0}, ""));
    ok &= write(games + "/20261102-100000-2.stderr",
        "GAME_RESULT: 1 Player A wins\nGAME_RESULT: 2 Player B wins\nGAME_RESULT: 3 Player B wins\n"
        "MATCH_RESULT: Player B wins 1-2\n");
    side.seat = 1;
    side.opponent = "model";
    side.model = "/x/exports/final.ts.pt";
    side.bo3 = true;
    side.seed = 2;
    ok &= write(sidecar_path(game("20261102-100000-2")), sidecar_json(side));
    // Stopped (no sidecar: the seat from the stream's name), malformed, the running game, nothing to read.
    ok &= write(game("20261103-100000-3"), rmlog_text("machine", {0, 3}, "QUIT\n"));
    ok &= write(games + "/20261103-100000-3.stderr", "robomage 0.3\n");
    ok &= write(games + "/20261103-100000-3.seat-b.rmstate", "");
    ok &= write(game("20261104-100000-4"), "not a decision log\n");
    ok &= write(game("20261105-100000-5"), rmlog_text("machine", {0}, ""));
    ok &= write(game("20261106-100000-6"), rmlog_text("machine", {1, 1}, ""));
    ok &= write(games + "/notes.txt", "not listed\n");
    // Recent replays: one the cache has replayed (its narrative's result), one it has not.
    const std::string engine = dir + "/bin/robomage";
    ok &= make_executable(engine);
    const std::string logs = dir + "/logs", cache = dir + "/cache";
    const std::string cached_log = logs + "/a0-m1.rmlog", plain_log = logs + "/a0-m2.rmlog";
    const std::string cached_text = rmlog_text("machine bo3", {0, 1}, "");
    ok &= write(cached_log, cached_text) && write(plain_log, rmlog_text("machine", {2}, ""));
    const CacheEntry e = cache_entry(cache, replay_cache_key(cached_text, engine_identity(engine)));
    ok &= write(e.dump, "frames") &&
          write(e.narrative, "...\nGAME_RESULT: 2 Player A wins\nMATCH_RESULT: Player A wins 2-0\n");
    std::string err;
    ok &= write_cache_entry(e, cached_log, engine_identity(engine), 2, &err);
    // Runs: a recent one elsewhere, the runs directory's (status.json, rmlog/ only, a malformed status,
    // a directory that is no run).
    const std::string runs = dir + "/runs", elsewhere = dir + "/elsewhere/r0";
    ok &= make_dirs(elsewhere + "/rmlog");
    ok &= write(runs + "/r1/status.json",
        "{\"phase\": \"B3\", \"running\": true, \"updated_at\": 1800000000, \"uptime_s\": 7200, "
        "\"totals\": {\"games\": 1234}}");
    ok &= make_dirs(runs + "/r2/rmlog");
    set_mtime(runs + "/r2", 1700000000);
    ok &= write(runs + "/r4/status.json", "{broken");
    ok &= make_dirs(runs + "/r3");
    check(ok, "Start: cannot write the synthetic files under " + dir);

    StartScanRequest r;
    r.games_dir = games;
    r.recent = {
        cached_log, game("20261101-100000-1"), dir + "/missing.rmlog", logs + "/x.rmtraj", plain_log, elsewhere};
    r.run_dir = runs;
    r.replay_engine = engine;
    r.cache_root = cache;
    r.live_log = game("20261105-100000-5");
    StartData d = scan_start(r);
    check(d.games_dir_exists && d.games_total == 6 && d.games.size() == 6,
        "Start: the games directory's logs: " + std::to_string(d.games.size()) + " of " +
            std::to_string(d.games_total));
    check(d.games.size() == 6 && d.games.front().path == game("20261106-100000-6") &&
              d.games.back().path == game("20261101-100000-1"),
        "Start: the games are not newest first");
    const RecentGame *g = find_game(d.games, game("20261101-100000-1"));
    check(g && g->state == GameState::FINISHED && g->seat == 0 && g->opponent == "explore" && g->decisions == 3 &&
              !g->bo3 && result_of(*g) == "won" && g->deck_a == "league/burn" && g->when == "11-01 10:00",
        "Start: the finished best of one: " + (g ? result_of(*g) + " " + g->when : std::string("missing")));
    g = find_game(d.games, game("20261102-100000-2"));
    check(g && g->state == GameState::FINISHED && g->seat == 1 && g->opponent == "nn: final.ts.pt" && g->bo3 &&
              g->game_seed == 2 && result_of(*g) == "won 2-1" && game_own_deck(*g) == "league/bug" &&
              game_opponent_deck(*g) == "league/burn",
        "Start: the best of three won as B: " + (g ? result_of(*g) : std::string("missing")));
    g = find_game(d.games, game("20261103-100000-3"));
    check(g && g->state == GameState::STOPPED && g->seat == 1 && g->opponent.empty() && g->decisions == 2 &&
              result_of(*g) == "no result (stopped)",
        "Start: the stopped game: " + (g ? result_of(*g) : std::string("missing")));
    g = find_game(d.games, game("20261104-100000-4"));
    check(g && g->state == GameState::MALFORMED && !g->error.empty() && result_of(*g) == "not a decision log",
        "Start: the malformed log");
    g = find_game(d.games, game("20261105-100000-5"));
    check(g && g->state == GameState::PLAYING && result_of(*g) == "playing", "Start: the running game");
    g = find_game(d.games, game("20261106-100000-6"));
    check(g && g->state == GameState::UNKNOWN && result_of(*g) == "\xE2\x80\xA6" && g->seat < 0,
        "Start: a game with nothing to read");
    check(d.replays.size() == 2 && d.replays[0].path == cached_log && d.replays[1].path == plain_log,
        "Start: the recent replays are not the two logs outside the games directory, latest first");
    if (d.replays.size() == 2) {
        check(d.replays[0].state == GameState::FINISHED && d.replays[0].cached && d.replays[0].seat < 0 &&
                  result_of(d.replays[0]) == "A wins 2-0",
            "Start: the cached replay's result: " + result_of(d.replays[0]));
        check(d.replays[1].state == GameState::UNKNOWN && result_of(d.replays[1]) == "\xE2\x80\xA6",
            "Start: a replay not cached has a result");
    }
    check(d.runs.size() == 4 && d.runs[0].path == elsewhere && d.runs[0].recent,
        "Start: the runs are not the recent one, then the runs directory's three");
    const RecentRun *run = find_run(d.runs, runs + "/r1");
    check(run && run->status && run->phase == "B3" && run->running && run->games == 1234 && run->hours == 2.0 &&
              run->updated == 1800000000.0 && !run->recent,
        "Start: the run with a status.json");
    run = find_run(d.runs, runs + "/r2");
    check(run && !run->status && run->games < 0 && run->updated == 1700000000.0, "Start: the run without status.json");
    run = find_run(d.runs, runs + "/r4");
    check(run && !run->status && !run->error.empty(), "Start: the run with a malformed status.json");
    check(!find_run(d.runs, runs + "/r3"), "Start: a directory that is no run is listed");
    check(d.runs.size() == 4 && d.runs[1].path == runs + "/r1", "Start: the runs directory's are not newest first");
    check(d.cache_entries == 1 && d.cache_bytes > 0, "Start: the replay cache's size");
    r.max_games = 2;
    r.max_runs = 2;
    d = scan_start(r);
    check(
        d.games.size() == 2 && d.games_total == 6 && d.games[0].path == game("20261106-100000-6") && d.runs.size() == 2,
        "Start: more games or runs than the page lists");
    r.games_dir = dir + "/no-games";
    r.run_dir = dir + "/no-runs";
    r.recent.clear();
    d = scan_start(r);
    check(!d.games_dir_exists && d.games.empty() && d.replays.empty() && d.runs.empty(),
        "Start: missing directories list something");
}

// The status blocks with and without what they look for.
static void check_status(const CheckFn &check, const std::string &dir) {
    using namespace rmwork;
    StatusBlock b = engine_status(dir + "/none/robomage", dir + "/none");
    check(b.health == Health::BAD && b.lines.size() >= 2 && b.lines[0].value.find("(missing)") != std::string::npos,
        "Start: a missing engine is not bad");
    const std::string bin = dir + "/sbin", work = dir + "/swork";
    bool ok = make_executable(bin + "/robomage") && make_dirs(work + "/resources");
    check(ok, "Start: cannot make the status files");
    b = engine_status(bin + "/robomage", work);
    check(b.health == Health::GOOD && b.fix == "make actor", "Start: an engine without robomage_nn");
    ok = make_executable(bin + "/robomage_nn");
    set_mtime(bin + "/robomage", 1800000000);
    set_mtime(bin + "/robomage_nn", 1800000000 - 2 * 86400 - 60);
    b = engine_status(bin + "/robomage", work);
    bool older = false;
    for (const StatusLine &l : b.lines) older = older || l.value.find("2 days older") != std::string::npos;
    check(ok && b.health == Health::WARN && older, "Start: an older robomage_nn is not a warning");
    b = engine_status(bin + "/robomage_nn", work);
    bool replays = false;
    for (const StatusLine &l : b.lines) replays = replays || l.label == "replays with";
    check(b.health == Health::WARN && replays, "Start: robomage_nn configured: the replays' engine and the warning");
    set_mtime(bin + "/robomage_nn", 1800000000);
    b = engine_status(bin + "/robomage", dir + "/sbin");
    check(b.health == Health::BAD, "Start: a working directory without resources/ is not bad");
    ImageCacheStatus s;
    b = image_status(dir + "/images", s);
    check(b.health == Health::WARN && !b.fix.empty(), "Start: no image cache is not a warning with its fix");
    s.manifest = true;
    s.vocab = 10;
    s.with_images = 10;
    s.tokens = 3;
    s.token_images = 3;
    b = image_status(dir + "/images", s);
    check(b.health == Health::GOOD && b.fix.empty() && b.lines.size() == 3 && b.lines[1].value == "10 of 10 cached",
        "Start: a full image cache");
    s.with_images = 7;
    b = image_status(dir + "/images", s);
    check(b.health == Health::WARN && b.lines[1].value.find("3 drawn as text") != std::string::npos,
        "Start: a partial image cache");
    b = inference_status("", 0, bin + "/robomage", "", false);
    check(b.health == Health::NONE && b.lines[0].value == "none configured", "Start: no socket");
    b = inference_status(dir + "/no.sock", 2, bin + "/robomage", "Model's Read: failed: refused", true);
    check(b.health == Health::BAD && b.lines[1].value == "no server at the socket" && b.lines[2].value == "2 listed",
        "Start: a refused Model's Read is not bad");
    StartData d;
    d.games_dir_exists = true;
    d.games_total = 3;
    d.cache_entries = 2;
    d.cache_bytes = 3u << 20;
    b = files_status(d, dir + "/g", dir + "/c");
    check(b.lines.size() == 2 && b.lines[0].value.find("3 games") != std::string::npos &&
              b.lines[1].value.find("2 replays, 3 MB of 4.0 GB") != std::string::npos,
        "Start: the files block: " + (b.lines.size() == 2 ? b.lines[1].value : std::string()));
}

namespace rmwork {

void check_start_units(const CheckFn &check, const std::string &tmp) {
    const std::string dir = tmp + "/start_units";
    check_result_lines(check);
    check_sidecar(check);
    check_scan(check, dir);
    check_status(check, dir);
    remove_tree(dir);
}

void check_start_app(const CheckFn &check, const ConfigPaths &paths, const std::string &tmp) {
    const std::string dir = tmp + "/start_app";
    const char *old_data = std::getenv("XDG_DATA_HOME");
    const std::string saved = old_data ? old_data : "";
    setenv("XDG_DATA_HOME", (dir + "/data").c_str(), 1);
    const std::string games = default_games_dir();
    const std::string log = games + "/20261101-100000-1.rmlog";
    check(write(log, rmlog_text("machine", {0, 1}, "")) &&
              write(games + "/20261101-100000-1.stderr", "GAME_RESULT: 1 Player B wins\n") &&
              write(games + "/20261101-100000-1.seat-a.rmstate", ""),
        "Start: cannot write the App's games");
    Settings s = default_settings();
    s.models.run_dir = dir + "/runs";
    s.engine.binary = dir + "/no-engine";
    {
        App app(s, paths, false);
        frames_of(app, 2);
        check(!app.start_page_shown() && window_shown(dockspace_host_name(app.mode())) && !window_shown(START_WINDOW),
            "Start: an App starts with the start page");
        app.show_start_page(true);
        const Frames frames = [&app](int n) { frames_of(app, n); };
        check(run_until(frames, [&app] { return app.start_page().fresh(); }, 30), "Start: the scan does not finish");
        check(app.start_page_shown() && window_shown(START_WINDOW) && !window_shown(dockspace_host_name(app.mode())),
            "Start: the page is not shown in place of the dockspace");
        check(!app.working(), "Start: the App is still working once the page's scan is done");
        const StartData &d = app.start_page().data();
        check(d.games.size() == 1 && d.games[0].path == log && d.games[0].state == GameState::FINISHED &&
                  d.games[0].seat == 0 && result_of(d.games[0]) == "lost",
            "Start: the App's page does not list its games directory's game");
        bool engine_bad = false;
        for (const StartStatus &st : app.start_status())
            if (st.block.title == "Engine") engine_bad = st.block.health == Health::BAD && st.settings_tab >= 0;
        check(engine_bad, "Start: a missing engine is not bad on the page, with its Settings tab");
        StartRequest hide;
        hide.kind = StartRequest::Kind::HIDE;
        app.run_start_request(hide);
        frames_of(app, 2);
        check(!app.start_page_shown() && window_shown(dockspace_host_name(app.mode())) && !window_shown(START_WINDOW),
            "Start: its \xE2\x9C\x95 does not give the dockspace back");
        // Shown again, it lists what changed since: a game written while it was hidden.
        check(write(games + "/20261102-100000-2.rmlog", rmlog_text("machine", {1}, "")),
            "Start: cannot write a second game");
        press_chord(app, app.settings().keys[action_index(Action::START_PAGE)]);
        check(app.start_page_shown() && window_shown(START_WINDOW), "Start: the Start Page key does not show it");
        check(run_until(
                  frames, [&app] { return app.start_page().fresh(); }, 30) &&
                  app.start_page().data().games.size() == 2,
            "Start: the page shown again lists what it listed before");
        app.set_mode(Mode::PLAY);
        frames_of(app, 2);
        check(app.start_page_shown(), "Start: setting the mode (as opening a run does) hides the page");
        app.request_preset(default_preset(Mode::PLAY));
        frames_of(app, 3);
        check(!app.start_page_shown() && window_shown(dockspace_host_name(Mode::PLAY)),
            "Start: a layout chosen does not show the dockspace");
        check(app.close_session() && app.start_page_shown(),
            "Start: File \xE2\x86\x92 Close does not bring the page back");
        app.run_start_request(hide);
        StartRequest settings;
        settings.kind = StartRequest::Kind::SETTINGS;
        settings.settings_tab = SettingsDialog::ENGINE;
        app.run_start_request(settings);
        frames_of(app, 2);
        check(app.settings_open(), "Start: a status block's Settings link does not open Settings");
    }
    s.start_page = false;
    {
        App app(s, paths, false);
        frames_of(app, 2);
        check(app.close_session() && !app.start_page_shown(), "Start: Close shows the page with its setting off");
    }
    Settings back;
    std::string err;
    check(settings_from_json(settings_to_json(s), &back, &err) && !back.start_page &&
              settings_from_json("{}", &back, &err) && back.start_page &&
              !settings_from_json("{\"start_page\": 1}", &back, &err),
        "Start: the start_page setting's file round trip: " + err);
    if (old_data) {
        setenv("XDG_DATA_HOME", saved.c_str(), 1);
    } else {
        unsetenv("XDG_DATA_HOME");
    }
    remove_tree(dir);
}

void check_start_games(const CheckFn &check, App &app, const std::function<void(int)> &frames) {
    const Match *live = app.play().match();
    const std::string bo3_log = app.play().log_path();
    app.show_start_page(true);
    check(run_until(frames, [&app] { return app.start_page().fresh(); }, 30), "Start page: the scan does not finish");
    const StartData &d = app.start_page().data();
    const RecentGame *bo1 = nullptr, *bo3 = find_game(d.games, bo3_log);
    for (const RecentGame &g : d.games)
        if (!g.bo3 && g.opponent == "explore" && g.game_seed == 3 && g.seat == 0) bo1 = &g;
    check(bo1 && bo1->state == GameState::FINISHED && bo1->deck_a == "league/burn" && bo1->deck_b == "league/bug" &&
              (result_of(*bo1) == "won" || result_of(*bo1) == "lost" || result_of(*bo1) == "draw"),
        "Start page: Play's bo1 game is not listed with its result");
    if (bo3 && live) {
        RmlogInfo info;
        std::string text, err;
        const bool read = read_file(bo3_log, &text) && parse_rmlog(text, &info, &err);
        const int me = 1;
        const std::string want = std::string(live->result.winner == me ? "won " : "lost ") +
                                 std::to_string(live->result.wins[me]) + "-" +
                                 std::to_string(live->result.wins[1 - me]);
        check(bo3->state == GameState::FINISHED && bo3->bo3 && bo3->seat == 1 && bo3->opponent == "random" && read &&
                  bo3->decisions == info.replies.size() && result_of(*bo3) == want,
            "Start page: Play's bo3 game: " + result_of(*bo3) + ", the match " + want);
    } else {
        check(false, "Start page: Play's bo3 game is not listed");
    }
    if (!bo1) return;
    const std::string path = bo1->path;
    StartRequest open;
    open.kind = StartRequest::Kind::OPEN_LOG;
    open.path = path;
    app.run_start_request(open);
    check(!app.start_page_shown(), "Start page: opening a recent game leaves the page shown");
    run_until(frames, [&app] { return !app.loading(); }, LOAD_TIMEOUT_S);
    check(app.mode() == Mode::REPLAY && app.replay().loaded() && app.replay().match()->log_path == path,
        "Start page: the recent game did not open in Replay: " + app.error_message());
    check(app.close_session() && app.start_page_shown() && !app.replay().loaded(),
        "Start page: File \xE2\x86\x92 Close does not bring the page back with nothing loaded");
    frames(2);
}

void check_start_stopped(const CheckFn &check, App &app, const std::function<void(int)> &frames) {
    app.show_start_page(true);
    check(run_until(frames, [&app] { return app.start_page().fresh(); }, 30), "Start page: the scan does not finish");
    const StartData &d = app.start_page().data();
    const RecentGame *stopped = nullptr;
    for (const RecentGame &g : d.games)
        if (g.state == GameState::STOPPED && g.game_seed == 11) stopped = &g;
    check(stopped && result_of(*stopped) == "no result (stopped)",
        "Start page: the game whose engine was killed is not listed as stopped");
    if (!stopped) return;
    const std::string path = stopped->path;
    check(app.open_replay(path), "Start page: the stopped game does not start opening");
    run_until(frames, [&app] { return !app.loading(); }, LOAD_TIMEOUT_S);
    check(app.replay().loaded() && app.replay().match()->log_path == path && app.replay().match()->stopped,
        "Start page: the stopped game does not open in Replay: " + app.error_title() + ": " + app.error_message());
    app.close_session();
    app.show_start_page(false);
    frames(2);
}

}  // namespace rmwork
