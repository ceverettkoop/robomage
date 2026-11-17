#include "selftest_data.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "../../src/agents/decision_view.h"
#include "../../src/cli/board.h"
#include "../../src/featurize/rmtb.h"
#include "app.h"
#include "episodes.h"
#include "file_picker.h"
#include "files.h"
#include "paths.h"
#include "replay_cache.h"
#include "rmlog.h"
#include "run_browser.h"
#include "selftest_table.h"
#include "selftest_tools.h"
#include "subprocess.h"
#include "theme.h"

namespace {

struct Tally {
        int checks = 0;
        int failures = 0;
};

// The test's files: the temporary root and the engine.
struct Env {
        std::string tmp;
        std::string engine;
        std::string workdir;
        std::string corpus;
};

const double LOAD_TIMEOUT_S = 600;
const char *const BO1_LOG = "delver_mirror.rmlog";
const char *const MANUAL_LOG = "manual_doomsday_vs_delver.rmlog";
// Corpus logs with blocks, attachments and stacks of targeted objects, for the Table's checks.
const char *const TABLE_LOGS[] = {
    "lg_death_and_taxes__wrb_energy.rmlog", "lg_reanimator__reanimator.rmlog", "lg_reanimator__wrb_energy.rmlog"};

}  // namespace

static void check(Tally *t, bool ok, const std::string &what);
static void begin_headless();
static void run_frames(rmwork::App &app, int n);
static void press(rmwork::App &app, ImGuiKeyChord chord);
static bool wait_load(rmwork::App &app);
static bool run_engine(const Env &env, const std::vector<std::string> &args, const std::string &log, std::string *err);
static std::vector<std::string> files_with(const std::string &dir, const char *suffix);
static bool copy_executable(const std::string &from, const std::string &to);
static size_t cache_entries(const std::string &root);
static bool open_and_wait(Tally *t, rmwork::App &app, const std::string &log, const std::string &what);
static void check_frames(Tally *t, const rmwork::Match &m, const std::string &at);
static void check_seat_logs(Tally *t, const rmwork::Match &m, const std::string &at);
static void check_narrative(Tally *t, const rmwork::Match &m, const std::string &at);
static void check_match(Tally *t, rmwork::App &app, const std::string &log, const std::string &at,
    rmwork::TableCoverage *cov, rmwork::ToolsCoverage *tools);
static void check_table(Tally *t, rmwork::App &app, const std::string &at, rmwork::TableCoverage *cov);
static void check_tools(Tally *t, rmwork::App &app, const std::string &at, rmwork::ToolsCoverage *cov);
static void check_tools_coverage(Tally *t, const rmwork::ToolsCoverage &cov);
static std::string make_eval_match(Tally *t, const Env &env, const std::string &bo3_log);
static bool add_evaluations(const std::string &from, const std::string &to, std::string *err);
static void check_table_coverage(Tally *t, const rmwork::TableCoverage &cov);
static void check_navigation(Tally *t, rmwork::App &app);
static void check_cache(Tally *t, rmwork::App &app, const Env &env);
static void check_corpus_episode(Tally *t, rmwork::App &app, const Env &env);
static std::string make_bo3(Tally *t, const Env &env);
static void check_bo3(
    Tally *t, rmwork::App &app, const std::string &log, rmwork::TableCoverage *cov, rmwork::ToolsCoverage *tools);
static void check_errors(Tally *t, rmwork::App &app, const Env &env);
static void check_run_directory(Tally *t, const Env &env, const std::string &bo3_log);
static void check_picker(Tally *t, rmwork::App &app, const Env &env, const std::string &bo3_log);

static void check(Tally *t, bool ok, const std::string &what) {
    ++t->checks;
    if (ok) return;
    ++t->failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

static void begin_headless() {
    rmwork::create_ui_context();
    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize = ImVec2(1600, 1000);
    io.DeltaTime = 1.0f / 60.0f;
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
}

static void run_frames(rmwork::App &app, int n) {
    for (int i = 0; i < n; ++i) {
        app.begin_frame();
        ImGui::NewFrame();
        app.frame();
        ImGui::Render();
    }
}

// Presses and releases `chord` (no modifiers), a frame each.
static void press(rmwork::App &app, ImGuiKeyChord chord) {
    ImGuiIO &io = ImGui::GetIO();
    const ImGuiKey key = static_cast<ImGuiKey>(chord & ~ImGuiMod_Mask_);
    io.AddKeyEvent(key, true);
    run_frames(app, 1);
    io.AddKeyEvent(key, false);
    run_frames(app, 1);
}

// Frames until the loader has handed its outcome over; false past the timeout.
static bool wait_load(rmwork::App &app) {
    const auto start = std::chrono::steady_clock::now();
    while (app.loading()) {
        run_frames(app, 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (s > LOAD_TIMEOUT_S) return false;
    }
    run_frames(app, 2);
    return true;
}

// The engine with `args` in its working directory, its output into `log`.
static bool run_engine(const Env &env, const std::vector<std::string> &args, const std::string &log, std::string *err) {
    std::vector<std::string> argv = {env.engine};
    argv.insert(argv.end(), args.begin(), args.end());
    rmwork::ChildProcess child;
    if (!child.start(argv, env.workdir, log, log + ".err", err)) return false;
    while (child.poll()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (!child.ok()) *err = child.status_text() + " (see " + log + ".err)";
    return child.ok();
}

static std::vector<std::string> files_with(const std::string &dir, const char *suffix) {
    std::vector<std::string> out;
    DIR *d = opendir(dir.c_str());
    if (!d) return out;
    const std::string sfx = suffix;
    while (const dirent *e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > sfx.size() && n.compare(n.size() - sfx.size(), sfx.size(), sfx) == 0)
            out.push_back(dir + "/" + n);
    }
    closedir(d);
    std::sort(out.begin(), out.end(), rmwork::natural_less);
    return out;
}

// A copy of the engine (another path and modification time: another identity).
static bool copy_executable(const std::string &from, const std::string &to) {
    std::string bytes, err;
    if (!rmwork::read_file(from, &bytes) || !rmwork::write_file_atomic(to, bytes, &err)) return false;
    return chmod(to.c_str(), 0755) == 0;
}

// Complete cache entries under `root`.
static size_t cache_entries(const std::string &root) {
    size_t n = 0;
    DIR *d = opendir(root.c_str());
    if (!d) return 0;
    while (const dirent *e = readdir(d)) {
        const std::string name = e->d_name;
        if (name == "." || name == "..") continue;
        if (rmwork::cache_entry_complete(rmwork::cache_entry(root, name))) ++n;
    }
    closedir(d);
    return n;
}

static bool open_and_wait(Tally *t, rmwork::App &app, const std::string &log, const std::string &what) {
    check(t, app.open_replay(log), what + ": open_replay refused");
    const bool done = wait_load(app);
    check(t, done, what + ": the load did not finish");
    return done;
}

// Every decision's three frames decode with their viewer's schema and name the decision's key; the
// menu is in the deciding seat's frame and the omniscient one only; the cache returns its frame.
static void check_frames(Tally *t, const rmwork::Match &m, const std::string &at) {
    using namespace rmwork;
    bool all_refs = true, decoded = true, keys = true, viewers = true, menus = true;
    std::string first_problem;
    for (size_t i = 0; i <= m.decisions.size(); ++i) {
        const bool terminal = i == m.decisions.size();
        for (int v = 0; v < VIEW_COUNT; ++v) {
            const FrameRef &ref =
                terminal ? m.terminal[static_cast<size_t>(v)] : m.decisions[i].frames[static_cast<size_t>(v)];
            if (!ref.valid()) {
                all_refs = false;
                continue;
            }
            std::string err;
            const auto f = m.frame(i, v, &err);
            if (!f) {
                if (decoded) first_problem = "decision " + std::to_string(i + 1) + ": " + err;
                decoded = false;
                continue;
            }
            if (f->header.viewer != v) viewers = false;
            const agents::Table header(*f, m.stream(v).schema(), "header");
            const bool term = header.rows() && header.u(0, header.col("terminal")) != 0;
            if (term != terminal) keys = false;
            if (terminal) continue;
            const DecisionInfo &d = m.decisions[i];
            if (!header.rows() || static_cast<int>(header.u(0, header.col("game"))) != d.game ||
                (header.u(0, header.col("sideboarding")) != 0) != d.sideboarding ||
                static_cast<int64_t>(header.u(0, header.col("decision"))) != d.decision)
                keys = false;
            const size_t menu = agents::Table(*f, m.stream(v).schema(), "menu").rows();
            if ((v == VIEW_OMNI || v == d.decider) != (menu > 0)) menus = false;
            if (v == VIEW_OMNI && static_cast<int>(menu) != d.menu_rows) menus = false;
        }
    }
    check(t, all_refs, at + ": a decision lacks a viewer's STATE frame");
    check(t, decoded, at + ": a frame does not decode with its viewer's schema: " + first_problem);
    check(t, viewers, at + ": a frame's header names another viewer");
    check(t, keys, at + ": a frame's header is not its decision's key");
    check(t, menus, at + ": the menu is not exactly in the deciding seat's and the omniscient frame");
    std::string err;
    const auto a = m.frame(0, VIEW_A, &err), b = m.frame(0, VIEW_A, &err);
    check(t, a && a == b, at + ": the frame cache decodes a frame it holds again");
    check(t, !m.frame(m.decisions.size() + 1, VIEW_A, &err), at + ": a frame past the terminal one is read");
}

// Each seat's log is its own frames' events, in order, each at the frame's decision (the terminal
// frame's at the last), turn headers between; the opponent's opening draws carry no card names.
static void check_seat_logs(Tally *t, const rmwork::Match &m, const std::string &at) {
    using namespace rmwork;
    for (int v = 0; v < 2; ++v) {
        const std::string who = at + ": " + view_name(v) + "'s log";
        std::vector<std::pair<int, std::string>> expected;
        for (size_t i = 0; i <= m.decisions.size(); ++i) {
            std::string err;
            const auto f = m.frame(i, v, &err);
            if (!f) continue;
            const agents::DecisionView dv(m.stream(v), *f, v);
            const int at_decision = static_cast<int>(i == m.decisions.size() ? i - 1 : i);
            for (const std::string &line : cli::event_lines(dv)) {
                int turn = 0;
                std::string text;
                expected.emplace_back(at_decision, split_turn_prefix(line, &turn, &text) ? text : line);
            }
        }
        std::vector<std::pair<int, std::string>> got;
        bool ordered = true;
        int last = 0;
        for (const LogLine &l : m.logs[static_cast<size_t>(v)]) {
            if (l.decision < last) ordered = false;
            last = l.decision;
            const bool turn_header = l.kind == LogLine::Kind::HEADER && l.text.compare(0, 5, "Turn ") == 0 &&
                                     l.text.find_first_not_of("0123456789", 5) == std::string::npos;
            if (!turn_header) got.emplace_back(l.decision, l.text);
        }
        check(t, !got.empty(), who + " is empty");
        check(t, got == expected,
            who + " is not exactly its frames' events (" + std::to_string(got.size()) + " lines, its frames give " +
                std::to_string(expected.size()) + ")");
        check(t, ordered, who + ": lines out of decision order");
        const std::string opp = v == 0 ? "Player B draws" : "Player A draws";
        bool hidden = true, any = false;
        for (const LogLine &l : m.logs[static_cast<size_t>(v)]) {
            if (l.text.compare(0, opp.size(), opp) != 0 || l.text.find("(opening hand)") == std::string::npos) continue;
            any = true;
            if (l.text.find("(opening hand):") != std::string::npos) hidden = false;
        }
        check(t, any, who + " has no line for the opponent's opening draws");
        check(t, hidden, who + " names the opponent's opening hand");
        for (const LogLine &l : m.logs[static_cast<size_t>(v)])
            if (l.kind == LogLine::Kind::CHOICE || l.text.find("(REPLAY)") != std::string::npos) {
                check(t, false, who + " holds a narrative line: " + l.text);
                break;
            }
    }
}

// The narrative's decision markers match the decisions, one chosen-row line each, in order.
static void check_narrative(Tally *t, const rmwork::Match &m, const std::string &at) {
    using namespace rmwork;
    check(t, m.narrative_mapped, at + ": the narrative's decision markers do not match the decisions");
    size_t choices = 0;
    bool in_order = true;
    int last = 0;
    for (const LogLine &l : m.logs[VIEW_OMNI]) {
        if (l.decision < last) in_order = false;
        last = l.decision;
        if (l.kind != LogLine::Kind::CHOICE) continue;
        if (l.decision != static_cast<int>(choices)) in_order = false;
        ++choices;
    }
    check(t, choices == m.decisions.size(),
        at + ": the narrative has " + std::to_string(choices) + " chosen-row lines for " +
            std::to_string(m.decisions.size()) + " decisions");
    check(t, in_order, at + ": the narrative's lines are out of decision order");
}

// The loaded match against its log: the decision count, the frames, the logs, the result, the Table
// and the other tools.
static void check_match(Tally *t, rmwork::App &app, const std::string &log, const std::string &at,
    rmwork::TableCoverage *cov, rmwork::ToolsCoverage *tools) {
    using namespace rmwork;
    check(t, app.replay().loaded(), at + ": no match loaded (" + app.error_title() + ": " + app.error_message() + ")");
    if (!app.replay().loaded()) return;
    const Match &m = *app.replay().match();
    std::string bytes, err;
    RmlogInfo info;
    check(t, read_file(log, &bytes) && parse_rmlog(bytes, &info, &err), at + ": the log does not parse: " + err);
    check(t, m.decisions.size() == info.replies.size() && !m.decisions.empty(),
        at + ": " + std::to_string(m.decisions.size()) + " decisions, the log holds " +
            std::to_string(info.replies.size()));
    bool deciders = true, replies = true;
    for (size_t i = 0; i < m.decisions.size() && i < info.replies.size(); ++i) {
        if (m.decisions[i].decider != 0 && m.decisions[i].decider != 1) deciders = false;
        if (m.decisions[i].reply != info.replies[i]) replies = false;
    }
    check(t, deciders, at + ": a decision has no deciding seat");
    check(t, replies, at + ": a decision's reply is not the log's");
    check(t, m.result.known, at + ": no match result");
    check_frames(t, m, at);
    check_seat_logs(t, m, at);
    check_narrative(t, m, at);
    check_table(t, app, at, cov);
    check_tools(t, app, at, tools);
}

// The Table of the loaded match: its model and layout of every frame, then every decision drawn
// in every view and the hover and magnifier interactions.
static void check_table(Tally *t, rmwork::App &app, const std::string &at, rmwork::TableCoverage *cov) {
    using namespace rmwork;
    const CheckFn fn = [t](bool ok, const std::string &what) { check(t, ok, "Table: " + what); };
    check_table_frames(fn, app.table(), *app.replay().match(), at, cov);
    check_table_drawing(fn, app, [&app](int n) { run_frames(app, n); }, at, cov);
    app.replay().go_to(0);
    app.replay().set_view(VIEW_A);
}

// The tools beside the Table on the loaded match: every frame's models, then a sample drawn.
static void check_tools(Tally *t, rmwork::App &app, const std::string &at, rmwork::ToolsCoverage *cov) {
    using namespace rmwork;
    const CheckFn fn = [t](bool ok, const std::string &what) { check(t, ok, "Tools: " + what); };
    check_tools_frames(fn, app, at, cov);
    check_tools_drawing(fn, app, [&app](int n) { run_frames(app, n); }, at, cov);
}

// The matches met each thing the tools show.
static void check_tools_coverage(Tally *t, const rmwork::ToolsCoverage &cov) {
    check(t, cov.frames > 0 && cov.drawn > 0, "Tools: no frame checked or drawn");
    check(t, cov.menus > 0 && cov.withheld > 0, "Tools: no menu seen, or none withheld from a seat");
    check(t, cov.zone_cards > 0, "Tools: no zone window card met");
    check(t, cov.known_hand > 0, "Tools: no card of the opponent's hand known by knowledge alone");
    check(t, cov.known_library > 0, "Tools: no library card known by position alone");
    check(t, cov.pending > 0 && cov.pending_costs > 0, "Tools: no pending query or cost met");
    check(t, cov.trees > 0 && cov.tree_handles > 0, "Tools: no State Tree walked");
    check(t, cov.links > 0, "Tools: no menu row linked");
    check(t, cov.episode_decisions > 0, "Tools: no decision checked against its evaluations");
    std::printf(
        "Tools: %d frames (%d menus, %d withheld), %d drawn; %d zone cards (%d known hand, %d known library, %d linked "
        "exile); %d pending, %d costs; %d trees, %d handles; %d links; %d evaluated decisions\n",
        cov.frames, cov.menus, cov.withheld, cov.drawn, cov.zone_cards, cov.known_hand, cov.known_library,
        cov.linked_exile, cov.pending, cov.pending_costs, cov.trees, cov.tree_handles, cov.links,
        cov.episode_decisions);
}

// The logs' frames met each thing the Table draws.
static void check_table_coverage(Tally *t, const rmwork::TableCoverage &cov) {
    check(t, cov.frames > 0, "Table: no frame checked");
    check(t, cov.land_stacks > 0, "Table: no stack of identical lands met");
    check(t, cov.attachments > 0, "Table: no attachment met");
    check(t, cov.stack_targets > 0, "Table: no stack target met");
    check(t, cov.blocks > 0, "Table: no block met");
    check(t, cov.counters > 0, "Table: no counter met");
    check(t, cov.tapped > 0, "Table: no tapped permanent met");
    check(t, cov.hover_checks > 0, "Table: no stack object's targets hovered");
    check(t, cov.magnifier_checks > 0, "Table: no card magnified");
    std::printf(
        "Table: %d frames; %d land stacks, %d attachments, %d stack targets, %d blocks, %d counters, %d hidden; "
        "%d hovers, %d magnifiers\n",
        cov.frames, cov.land_stacks, cov.attachments, cov.stack_targets, cov.blocks, cov.counters, cov.hidden,
        cov.hover_checks, cov.magnifier_checks);
}

// The keys, autoplay and the viewer toggle move the cursor and update the status bar's fields.
static void check_navigation(Tally *t, rmwork::App &app) {
    using namespace rmwork;
    ReplaySession &s = app.replay();
    const KeyBindings keys = default_bindings();
    const size_t n = s.count();
    const auto fields_match = [&](size_t i) {
        const DecisionInfo &d = s.match()->decisions[i];
        const std::string decider = d.decider == 0 ? "Player A decides" : "Player B decides";
        return s.decision_text() == "Decision " + std::to_string(i + 1) + " / " + std::to_string(n) &&
               s.turn_text().find("Game " + std::to_string(d.game + 1)) == 0 &&
               s.turn_text().find(d.phase) != std::string::npos &&
               (d.sideboarding || s.turn_text().find("Turn " + std::to_string(d.turn + 1)) != std::string::npos) &&
               s.decider_text() == decider;
    };
    press(app, keys[action_index(Action::REPLAY_LAST)]);
    check(t, s.cursor() == n - 1 && fields_match(n - 1), "End does not go to the last decision");
    press(app, keys[action_index(Action::REPLAY_PREV)]);
    check(t, s.cursor() == n - 2 && fields_match(n - 2), "← does not go back a decision");
    press(app, keys[action_index(Action::REPLAY_FIRST)]);
    check(t, s.cursor() == 0 && fields_match(0), "Home does not go to the first decision");
    press(app, keys[action_index(Action::REPLAY_NEXT)]);
    check(t, s.cursor() == 1 && fields_match(1), "→ does not go to the next decision");
    for (size_t i = 0; i < n; i += std::max<size_t>(1, n / 17)) {
        s.go_to(i);
        run_frames(app, 1);
        check(t, fields_match(i), "the status bar's fields at decision " + std::to_string(i + 1));
    }
    s.go_to(3);
    const double now = ImGui::GetTime();
    s.set_playing(true, now);
    s.tick(now + s.interval * 1.01);
    check(t, s.cursor() == 4 && s.playing(), "autoplay does not step");
    press(app, keys[action_index(Action::REPLAY_NEXT)]);
    check(t, !s.playing(), "a key does not stop autoplay");
    const char *names[] = {"Replay · view seat A", "Replay · view seat B", "Replay · view omniscient"};
    for (int v = 0; v < VIEW_COUNT; ++v) {
        const unsigned g = s.generation();
        s.set_view(v);
        run_frames(app, 1);
        check(t, app.status_mode_text() == names[v], std::string("the status bar does not show ") + names[v]);
        check(t, &GameLogView::lines(*s.match(), s.view()) == &s.match()->logs[static_cast<size_t>(v)],
            std::string("the Game Log does not read ") + view_name(v) + "'s log");
        check(t, v == VIEW_A || s.generation() != g, "switching the viewer does not move the Game Log");
    }
    s.set_view(VIEW_A);
    check(t, GameLogView::caption(*s.match(), VIEW_A).find("Narrative") == std::string::npos,
        "a seat view's Game Log is captioned as the narrative");
}

// A second open of a log replays nothing; an engine of another identity replays again; a cached
// dump that no longer decodes reaches the error dialog.
static void check_cache(Tally *t, rmwork::App &app, const Env &env) {
    using namespace rmwork;
    const std::string log = env.corpus + "/" + BO1_LOG;
    const std::string root = replay_cache_root();
    const size_t before = cache_entries(root);
    if (!open_and_wait(t, app, log, "cache hit")) return;
    check(t, app.replay().loaded() && app.replay().match()->from_cache, "a second open of a log replays it again");
    check(t, cache_entries(root) == before, "a cache hit adds an entry");
    const std::string copy = env.tmp + "/engine_copy/robomage";
    make_dirs(parent_dir(copy));
    check(t, copy_executable(env.engine, copy), "cannot copy the engine for the identity check");
    check(t,
        engine_identity(copy).found &&
            replay_cache_key("x", engine_identity(copy)) != replay_cache_key("x", engine_identity(env.engine)),
        "a copied engine has the same identity");
    app.set_engine_override(copy);
    if (open_and_wait(t, app, log, "engine identity")) {
        check(t, app.replay().loaded() && !app.replay().match()->from_cache,
            "an engine of another identity reuses the cached replay");
        check(t, cache_entries(root) == before + 1, "an engine of another identity adds no cache entry");
        // Truncate this entry's dump: its next open is a decode error.
        const std::string dump = app.replay().match()->dump_path;
        std::string bytes;
        read_file(dump, &bytes);
        app.replay().clear();
        std::string err;
        write_file_atomic(dump, bytes.substr(0, bytes.size() / 2), &err);
        open_and_wait(t, app, log, "a truncated dump");
        check(t, !app.replay().loaded() && app.error_open() && app.error_title() == "Replay not decoded",
            "a cached dump that does not decode does not reach the error dialog (" + app.error_title() + ")");
        press(app, ImGuiKey_Escape);
        remove_tree(parent_dir(dump));
    }
    app.set_engine_override(env.engine);
    remove_tree(parent_dir(copy));
}

// An episode of a corpus log (--replay --trajectory-out), opened on its own: its log is found and
// every decision joins with the episode's seat and reply.
static void check_corpus_episode(Tally *t, rmwork::App &app, const Env &env) {
    using namespace rmwork;
    const std::string log = env.corpus + "/" + BO1_LOG;
    const std::string dir = env.tmp + "/corpus_episodes";
    std::string err;
    check(t, run_engine(env, {"--replay", log, "--trajectory-out", dir}, env.tmp + "/corpus_episodes.out", &err),
        "the engine does not write the corpus log's episode: " + err);
    const std::vector<std::string> eps = files_with(dir, ".rmtraj");
    check(t, eps.size() == 1, "the corpus log's replay wrote " + std::to_string(eps.size()) + " episodes");
    if (eps.size() != 1) return;
    check(t, log_for_episode(eps[0], env.workdir, &err) == log, "the episode's log is not found: " + err);
    if (!open_and_wait(t, app, log, "corpus episode")) return;
    app.replay().go_to(7);
    check(t, app.open_episode(eps[0]), "open_episode refused");
    check(t, wait_load(app), "the episode's load did not finish");
    check(t, !app.error_open(), "opening the episode reports: " + app.error_title() + ": " + app.error_message());
    if (!app.replay().loaded()) return;
    const Match &m = *app.replay().match();
    check(t, app.replay().cursor() == 7, "opening an episode of the loaded log moves the cursor");
    size_t joined = 0;
    bool same = true;
    for (const DecisionInfo &d : m.decisions) {
        if (d.episode_step < 0) continue;
        ++joined;
        const EpisodeStepInfo &e = m.episode_steps[static_cast<size_t>(d.episode_step)];
        if (e.seat != d.decider || e.chosen != d.reply) same = false;
    }
    check(t, joined == m.decisions.size(),
        "the episode joins " + std::to_string(joined) + " of " + std::to_string(m.decisions.size()) + " decisions");
    check(t, same, "a joined decision's seat or reply differs from the episode's");
    check(t, m.episode_files.size() == 1 && m.episode_files[0] == eps[0], "the match does not list the episode");
    check(t, !app.settings().recent.empty() && app.settings().recent.front() == eps[0],
        "the opened episode is not the first recent file");
}

// A bo3 match of explore (A) and random (B) with its episodes beside its log; the log's path.
static std::string make_bo3(Tally *t, const Env &env) {
    const std::string dir = env.tmp + "/bo3";
    std::string err;
    const bool ok = run_engine(env,
        {"--agent-a", "explore", "--agent-b", "random", "--deck-a", "league/burn", "--deck-b", "league/ur_tempo",
            "--bo3", "--game-seed", "3", "--trajectory-out", dir},
        env.tmp + "/bo3.out", &err);
    check(t, ok, "the engine does not play the bo3 match: " + err);
    const std::vector<std::string> logs = files_with(dir, ".rmlog");
    check(t, logs.size() == 1, "the bo3 match wrote " + std::to_string(logs.size()) + " decision logs");
    return logs.size() == 1 ? logs[0] : "";
}

// The games and sideboarding phases, the result, the game list, the episodes joined by name.
static void check_bo3(
    Tally *t, rmwork::App &app, const std::string &log, rmwork::TableCoverage *cov, rmwork::ToolsCoverage *tools) {
    using namespace rmwork;
    if (!open_and_wait(t, app, log, "bo3")) return;
    check_match(t, app, log, "bo3", cov, tools);
    if (!app.replay().loaded()) return;
    const Match &m = *app.replay().match();
    check(t, m.bo3(), "bo3: the log is not a bo3 match");
    size_t games = 0, sideboards = 0;
    bool order = true;
    for (size_t g = 0; g < m.games.size(); ++g) {
        const GameSpan &s = m.games[g];
        if (s.sideboarding) {
            ++sideboards;
            if (g + 1 >= m.games.size() || m.games[g + 1].sideboarding || m.games[g + 1].game != s.game ||
                s.label != "Sideboarding " + std::to_string(s.game + 1))
                order = false;
        } else {
            if (s.game != static_cast<int>(games) || s.label != "Game " + std::to_string(s.game + 1)) order = false;
            ++games;
        }
        if (g > 0 && s.first != m.games[g - 1].first + m.games[g - 1].count) order = false;
    }
    check(t, games >= 2 && games <= 3, "bo3: " + std::to_string(games) + " games");
    check(t, sideboards == games - 1,
        "bo3: " + std::to_string(sideboards) + " sideboarding phases for " + std::to_string(games) + " games");
    check(t, order, "bo3: the games and sideboarding phases are not in match order");
    check(t, m.result.game_winners.size() == games,
        "bo3: game results for " + std::to_string(m.result.game_winners.size()) + " games");
    int wins[2] = {0, 0};
    for (int w : m.result.game_winners)
        if (w >= 0) ++wins[w];
    check(t,
        m.result.wins[0] == wins[0] && m.result.wins[1] == wins[1] && m.result.winner >= 0 &&
            m.result.wins[static_cast<size_t>(m.result.winner)] == 2,
        "bo3: the match result disagrees with the games' (" + m.result.text + ")");
    for (const GameSpan &s : m.games) {
        app.replay().go_to(s.first);
        run_frames(app, 1);
        check(t, app.replay().game_label() == s.label,
            "bo3: going to " + s.label + " shows " + app.replay().game_label());
    }
    app.open_game_combo();
    run_frames(app, 2);
    check(t, ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId), "bo3: the game list does not open");
    press(app, ImGuiKey_Escape);
    const size_t files = files_with(parent_dir(log), ".rmtraj").size();
    check(t, m.episode_files.size() == files && files == games,
        "bo3: " + std::to_string(m.episode_files.size()) + " of the " + std::to_string(files) + " episodes joined");
    size_t joined = 0, mu_b = 0, b = 0;
    bool same = true;
    for (const DecisionInfo &d : m.decisions) {
        if (d.decider == 1) ++b;
        if (d.episode_step < 0) continue;
        ++joined;
        const EpisodeStepInfo &e = m.episode_steps[static_cast<size_t>(d.episode_step)];
        if (e.seat != d.decider || e.chosen != d.reply) same = false;
        if (e.seat == 1 && e.has_mu && static_cast<int>(e.mu.size()) == d.menu_rows) ++mu_b;
    }
    check(t, joined == m.decisions.size(),
        "bo3: the episodes join " + std::to_string(joined) + " of " + std::to_string(m.decisions.size()) +
            " decisions");
    check(t, same, "bo3: a joined decision's seat or reply differs from the episode's");
    check(t, mu_b == b && b > 0,
        "bo3: the random seat's μ is on " + std::to_string(mu_b) + " of its " + std::to_string(b) + " decisions");
}

// The episode at `from` with evaluations added, written to `to`: per decision policy 0 (version 100)
// and, on seat A's decisions only, policy 1 (version 50), each a value and a log π over the menu
// rows, made up from the decision's index.
static bool add_evaluations(const std::string &from, const std::string &to, std::string *err) {
    featurize::RmtbFile f;
    if (!featurize::rmtb_read_file(from, &f, err)) return false;
    std::vector<int64_t> seat, game, offsets;
    const featurize::RmtbArray *s = f.find("seat"), *g = f.find("game"), *o = f.find("menu_offsets");
    if (!s || !g || !o || !featurize::rmtb_values(*s, &seat) || !featurize::rmtb_values(*g, &game) ||
        !featurize::rmtb_values(*o, &offsets)) {
        *err = "the episode lacks its seat, game or menu_offsets arrays";
        return false;
    }
    const size_t d = seat.size(), a = static_cast<size_t>(offsets.back());
    std::vector<int64_t> ids(d * 2), versions(d * 2);
    std::vector<float> values(d * 2), log_pi(a * 2);
    for (size_t i = 0; i < d; ++i) {
        const double x = static_cast<double>(i);
        ids[i * 2] = 0;
        versions[i * 2] = 100;
        values[i * 2] = static_cast<float>(0.9 * std::sin(0.37 * x + 1.7 * static_cast<double>(game[i])));
        ids[i * 2 + 1] = seat[i] == 0 ? 1 : -1;
        versions[i * 2 + 1] = seat[i] == 0 ? 50 : 0;
        values[i * 2 + 1] = seat[i] == 0 ? static_cast<float>(0.6 * std::cos(0.23 * x)) : 0.0f;
        for (int p = 0; p < 2; ++p) {
            double norm = 0;
            const size_t b = static_cast<size_t>(offsets[i]), e = static_cast<size_t>(offsets[i + 1]);
            for (size_t k = b; k < e; ++k)
                norm += std::exp(p == 0 ? std::sin(1.3 * static_cast<double>(k) + x)
                                        : std::cos(0.7 * static_cast<double>(k) + x / 2));
            for (size_t k = b; k < e; ++k) {
                const double z = p == 0 ? std::sin(1.3 * static_cast<double>(k) + x)
                                        : std::cos(0.7 * static_cast<double>(k) + x / 2);
                log_pi[k * 2 + static_cast<size_t>(p)] = static_cast<float>(z - std::log(norm));
            }
        }
    }
    f.arrays.erase(std::remove_if(f.arrays.begin(), f.arrays.end(),
                       [](const featurize::RmtbArray &x) { return x.name.compare(0, 5, "eval_") == 0; }),
        f.arrays.end());
    f.arrays.push_back(featurize::rmtb_ints("eval_policy", featurize::Dtype::I64, {d, 2}, ids));
    f.arrays.push_back(featurize::rmtb_ints("eval_version", featurize::Dtype::I64, {d, 2}, versions));
    f.arrays.push_back(featurize::rmtb_floats("eval_value", featurize::Dtype::F32, {d, 2}, values));
    f.arrays.push_back(featurize::rmtb_floats("eval_log_pi", featurize::Dtype::F32, {a, 2}, log_pi));
    return featurize::rmtb_write_file(to, f, err);
}

// The bo3 match's log and episodes, with evaluations added, in a directory of their own; the log's path.
static std::string make_eval_match(Tally *t, const Env &env, const std::string &bo3_log) {
    const std::string dir = env.tmp + "/bo3_eval";
    rmwork::make_dirs(dir);
    std::string bytes, err;
    const std::string log = dir + "/" + bo3_log.substr(bo3_log.rfind('/') + 1);
    check(t, rmwork::read_file(bo3_log, &bytes) && rmwork::write_file_atomic(log, bytes, &err),
        "evaluations: cannot copy the bo3 log: " + err);
    const std::vector<std::string> eps = files_with(rmwork::parent_dir(bo3_log), ".rmtraj");
    bool ok = !eps.empty();
    for (const std::string &e : eps) ok = ok && add_evaluations(e, dir + "/" + e.substr(e.rfind('/') + 1), &err);
    check(t, ok, "evaluations: cannot write the episodes with evaluations: " + err);
    return ok ? log : "";
}

// A malformed log, a diverging replay, a missing engine and a cancelled load.
static void check_errors(Tally *t, rmwork::App &app, const Env &env) {
    using namespace rmwork;
    std::string err;
    const std::string bad = env.tmp + "/malformed.rmlog";
    write_file_atomic(bad, "this is not a decision log\n", &err);
    open_and_wait(t, app, bad, "malformed");
    check(t, app.error_open() && app.error_title() == "Not a decision log",
        "a malformed log does not reach the error dialog (" + app.error_title() + ")");
    press(app, ImGuiKey_Escape);
    std::string bytes;
    read_file(env.corpus + "/" + BO1_LOG, &bytes);
    RmlogInfo info;
    parse_rmlog(bytes, &info, &err);
    // The 20th reply made one no menu has: the replay diverges and the engine exits 1.
    size_t at = bytes.find("DECISIONS\n") + 10;
    for (int i = 0; i < 20; ++i) at = bytes.find('\n', at) + 1;
    const std::string diverging = bytes.substr(0, at) + "77\n" + bytes.substr(bytes.find('\n', at) + 1);
    const std::string div = env.tmp + "/diverging.rmlog";
    write_file_atomic(div, diverging, &err);
    const std::string root = replay_cache_root();
    const size_t entries = cache_entries(root);
    open_and_wait(t, app, div, "diverging");
    check(t,
        app.error_open() && app.error_title() == "Engine exited" &&
            app.error_message().find("after decision 20 of") != std::string::npos,
        "a diverging replay does not reach the error dialog (" + app.error_title() + ": " + app.error_message() + ")");
    check(t, cache_entries(root) == entries, "a failed replay is cached");
    std::vector<std::string> tmp_dirs;
    if (DIR *d = opendir(root.c_str())) {
        while (const dirent *e = readdir(d))
            if (std::string(e->d_name).find(".tmp-") != std::string::npos) tmp_dirs.push_back(e->d_name);
        closedir(d);
    }
    check(t, tmp_dirs.empty(), "a failed replay leaves a temporary cache entry");
    press(app, ImGuiKey_Escape);
    app.set_engine_override(env.tmp + "/no/such/robomage");
    open_and_wait(t, app, env.corpus + "/" + BO1_LOG, "missing engine");
    check(t, app.error_open() && app.error_title() == "Engine not found",
        "a missing engine does not reach the error dialog (" + app.error_title() + ")");
    press(app, ImGuiKey_Escape);
    app.set_engine_override(env.engine);
    // A log not yet cached, cancelled as it starts.
    const std::string fresh = env.tmp + "/fresh.rmlog";
    write_file_atomic(fresh, bytes + "\n", &err);
    check(t, app.open_replay(fresh), "open_replay refused");
    check(t, !app.open_replay(fresh), "a second open is accepted while one runs");
    app.cancel_load();
    check(t, wait_load(app), "a cancelled load does not finish");
    check(t, !app.error_open() && app.status_message() == "Opening cancelled",
        "a cancelled load reports an error or no cancel (" + app.status_message() + ")");
    check(t, cache_entries(root) == entries, "a cancelled replay is cached");
}

// A run directory: its logs by purpose with the games their episodes cover; the log ↔ episode
// correspondence by name in both directions.
static void check_run_directory(Tally *t, const Env &env, const std::string &bo3_log) {
    using namespace rmwork;
    const std::string run = env.tmp + "/run";
    const std::string logs = run + "/rmlog/selfplay", eps = run + "/episodes/selfplay";
    make_dirs(logs);
    make_dirs(eps);
    std::string bytes, err;
    read_file(bo3_log, &bytes);
    const std::string stem = "a0-m7";
    write_file_atomic(logs + "/" + stem + ".rmlog", bytes, &err);
    write_file_atomic(logs + "/a0-m10.rmlog", bytes, &err);
    write_file_atomic(logs + "/a0-m2.rmlog", bytes, &err);
    const std::vector<std::string> src = files_with(parent_dir(bo3_log), ".rmtraj");
    for (size_t g = 0; g < src.size(); ++g) {
        featurize::RmtbFile f;
        featurize::rmtb_read_file(src[g], &f, &err);
        for (featurize::RmtbAttr &a : f.attrs)
            if (a.key == "rmlog") a.s = "/no/such/dir/" + stem + ".rmlog";
        featurize::rmtb_write_file(eps + "/" + stem + ".g" + std::to_string(g + 1) + ".rmtraj", f, &err);
    }
    const std::vector<RunPurpose> ps = scan_run_directory(run);
    check(t, ps.size() == 1 && ps[0].name == "selfplay" && ps[0].logs.size() == 3,
        "run directory: the selfplay logs are not listed");
    if (ps.size() == 1 && ps[0].logs.size() == 3) {
        check(t, ps[0].logs[0].name == "a0-m2" && ps[0].logs[1].name == "a0-m7" && ps[0].logs[2].name == "a0-m10",
            "run directory: the logs are not in natural order");
        check(t, ps[0].logs[1].games.size() == src.size() && ps[0].logs[0].games.empty() && ps[0].with_episodes == 1,
            "run directory: the episodes are not matched to their log");
    }
    check(t, episode_files_for_log(logs + "/" + stem + ".rmlog").size() == src.size(),
        "run directory: the log's episodes are not found by name");
    check(t, log_for_episode(eps + "/" + stem + ".g1.rmtraj", env.workdir, &err) == logs + "/" + stem + ".rmlog",
        "run directory: an episode's log is not found by name: " + err);
    check(t, scan_run_directory(env.tmp + "/no_such_run").empty(), "an empty run directory lists logs");
}

// The picker lists a directory's logs (or episodes) after its subdirectories, in natural order.
static void check_picker(Tally *t, rmwork::App &app, const Env &env, const std::string &bo3_log) {
    using namespace rmwork;
    check(t, natural_less("a0-m2", "a0-m10") && !natural_less("a0-m10", "a0-m2") && natural_less("g1", "g01x"),
        "natural order");
    app.open_file_picker(App::PICK_REPLAY);
    run_frames(app, 2);
    FilePicker &p = app.file_picker();
    check(t, p.is_open() && p.filter() == FileFilter::DECISION_LOGS, "the replay picker does not open on logs");
    check(t, p.navigate(env.corpus), "the picker cannot list the corpus");
    size_t logs = 0;
    bool only_logs = true;
    for (const PickerEntry &e : p.entries()) {
        if (e.dir) continue;
        if (e.name.size() > 6 && e.name.compare(e.name.size() - 6, 6, ".rmlog") == 0) {
            ++logs;
        } else {
            only_logs = false;
        }
    }
    check(t, only_logs && logs == files_with(env.corpus, ".rmlog").size(),
        "the picker's log filter shows " + std::to_string(logs) + " logs");
    p.navigate(parent_dir(bo3_log));
    p.set_filter(FileFilter::EPISODES);
    check(t, p.entries().size() == files_with(parent_dir(bo3_log), ".rmtraj").size(),
        "the picker's episode filter does not show the episodes alone");
    p.set_filter(FileFilter::DIRECTORIES);
    check(t, p.entries().empty(), "the directory filter shows files");
    check(t, !p.navigate(env.tmp + "/no/such/dir") && p.directory() == parent_dir(bo3_log),
        "the picker leaves its directory for one it cannot read");
    press(app, ImGuiKey_Escape);
    check(t, !p.is_open(), "Esc does not close the picker");
    check(t, !app.settings().recent.empty() && app.settings().recent.size() <= RECENT_MAX,
        "the opened logs are not remembered as recent files");
}

namespace rmwork {

int run_selftest_data(const std::string &engine) {
    Tally t;
    Env env;
    env.engine = engine.empty() ? parent_dir(executable_path()) + "/robomage" : engine;
    if (!is_executable(env.engine)) {
        std::fprintf(stderr, "rmwork: no engine at %s (build it with make, or pass --engine)\n", env.engine.c_str());
        return 1;
    }
    char buf[PATH_MAX];
    if (realpath(env.engine.c_str(), buf)) env.engine = buf;
    env.workdir = default_engine_workdir();
    env.corpus = repo_root() + "/tests/corpus";
    env.tmp = make_temp_dir("rmwork_selftest_data_");
    if (env.tmp.empty()) {
        std::fprintf(stderr, "rmwork: cannot make a temporary directory\n");
        return 1;
    }
    setenv("XDG_CONFIG_HOME", (env.tmp + "/config").c_str(), 1);
    setenv("XDG_CACHE_HOME", (env.tmp + "/cache").c_str(), 1);
    begin_headless();
    {
        App app(default_settings(), config_paths(""), false);
        app.set_engine_override(env.engine);
        run_frames(app, 2);
        const std::string bo1 = env.corpus + "/" + BO1_LOG;
        TableCoverage cov;
        ToolsCoverage tools;
        if (open_and_wait(&t, app, bo1, "bo1")) {
            check_match(&t, app, bo1, "bo1", &cov, &tools);
            check(&t, app.replay().loaded() && !app.replay().match()->from_cache, "bo1: the first open is cached");
            if (app.replay().loaded()) {
                const Match &m = *app.replay().match();
                check(&t, m.games.size() == 1 && m.games[0].label == "Game 1" && !m.bo3(), "bo1: not one game");
                check_navigation(&t, app);
            }
        }
        const std::string manual = env.corpus + "/" + MANUAL_LOG;
        if (file_exists(manual) && open_and_wait(&t, app, manual, "no-auto-mana"))
            check_match(&t, app, manual, "no-auto-mana", &cov, &tools);
        for (const char *name : TABLE_LOGS) {
            const std::string log = env.corpus + "/" + name;
            if (open_and_wait(&t, app, log, name)) check_match(&t, app, log, name, &cov, &tools);
        }
        check_cache(&t, app, env);
        check_corpus_episode(&t, app, env);
        const std::string bo3 = make_bo3(&t, env);
        if (!bo3.empty()) {
            check_bo3(&t, app, bo3, &cov, &tools);
            const std::string eval = make_eval_match(&t, env, bo3);
            if (!eval.empty() && open_and_wait(&t, app, eval, "evaluations")) {
                const CheckFn fn = [&t](bool ok, const std::string &what) { check(&t, ok, "Tools: " + what); };
                check_tools_values(fn, app, [&app](int n) { run_frames(app, n); }, "evaluations", &tools);
            }
            check_run_directory(&t, env, bo3);
            check_picker(&t, app, env, bo3);
        }
        check_errors(&t, app, env);
        check_table_coverage(&t, cov);
        check_tools_coverage(&t, tools);
    }
    destroy_ui_context();
    remove_tree(env.tmp);
    std::printf("RMWORK-SELFTEST-DATA: %d checks, %d failures\n", t.checks, t.failures);
    return t.failures == 0 ? 0 : 1;
}

}  // namespace rmwork
