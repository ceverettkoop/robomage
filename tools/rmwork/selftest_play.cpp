#include "selftest_play.h"

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "../../src/agents/decision_view.h"
#include "../../src/state_format/decode.h"
#include "autopass.h"
#include "files.h"
#include "new_game_dialog.h"
#include "rmlog.h"
#include "selftest_sideboard.h"
#include "selftest_start.h"

namespace {

using rmwork::CheckFn;
using Frames = std::function<void(int)>;

const double GAME_TIMEOUT_S = 300;

// What the passing checks read of each of a recorded game's decisions.
struct Recorded {
        std::vector<rmwork::PassFacts> facts;
        std::vector<std::vector<std::string>> categories;
        std::vector<bool> pending;  // the frame has a pending decision or cost
};

}  // namespace

static bool has_pair(const std::vector<std::string> &v, const std::string &a, const std::string &b);
static void press_key(const Frames &frames, ImGuiKey key);
static bool run_until(const Frames &frames, const std::function<bool()> &done, double timeout_s);
static bool make_executable(const std::string &path, const std::string &text);
static rmwork::NewGameSettings game(
    const std::string &opp, const std::string &deck, const std::string &opp_deck, int seat, bool bo3, uint64_t seed);
static bool play_game(const CheckFn &check, rmwork::App &app, const Frames &frames, const rmwork::NewGameSettings &g,
    const std::string &at);
static Recorded record(const rmwork::Match &m);
static bool same_except_events(const rmstate::DecodedFrame &a, const rmstate::DecodedFrame &b, std::string *why);
static void check_arguments(const CheckFn &check, const rmwork::PlayTestEnv &env);
static void check_dialog(const CheckFn &check, rmwork::App &app, const Frames &frames, const rmwork::PlayTestEnv &env);
static void check_stream_file(const CheckFn &check, const rmwork::PlaySession &s, const std::string &at);
static void check_stops_answered(const CheckFn &check, const rmwork::PlaySession &s, const std::string &at);
static void check_replay(const CheckFn &check, rmwork::App &app, const Frames &frames, const std::string &at);
static void check_passing_recorded(const CheckFn &check, const Recorded &r);
static void check_passing_units(const CheckFn &check);
static void check_bo1(const CheckFn &check, rmwork::App &app, const Frames &frames, const rmwork::PlayTestEnv &env);
static void check_bo3(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_live_passing(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_step_bar(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_concede(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_scripted(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_failures(
    const CheckFn &check, rmwork::App &app, const Frames &frames, const rmwork::PlayTestEnv &env);

// `v` holds `a` immediately followed by `b`.
static bool has_pair(const std::vector<std::string> &v, const std::string &a, const std::string &b) {
    for (size_t i = 0; i + 1 < v.size(); ++i)
        if (v[i] == a && v[i + 1] == b) return true;
    return false;
}

// Presses and releases `key`, a frame each.
static void press_key(const Frames &frames, ImGuiKey key) {
    ImGui::GetIO().AddKeyEvent(key, true);
    frames(1);
    ImGui::GetIO().AddKeyEvent(key, false);
    frames(1);
}

// Frames until `done`; false past the timeout.
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

static bool make_executable(const std::string &path, const std::string &text) {
    std::string err;
    return rmwork::write_file_atomic(path, text, &err) && chmod(path.c_str(), 0755) == 0;
}

static rmwork::NewGameSettings game(
    const std::string &opp, const std::string &deck, const std::string &opp_deck, int seat, bool bo3, uint64_t seed) {
    rmwork::NewGameSettings g;
    g.opponent = opp;
    g.deck = deck;
    g.opponent_deck = opp_deck;
    g.seat = seat;
    g.bo3 = bo3;
    g.seed = seed;
    return g;
}

// Starts `g` and answers every decision with auto_play's rule until the match ends; true when it
// finished.
static bool play_game(const CheckFn &check, rmwork::App &app, const Frames &frames, const rmwork::NewGameSettings &g,
    const std::string &at) {
    const bool started = app.start_game(g);
    check(started, at + ": the game does not start");
    if (!started) return false;
    check(app.mode() == rmwork::Mode::PLAY, at + ": starting a game does not switch to Play");
    app.auto_play(-1, "end");
    const bool ended = run_until(frames, [&app] { return !app.play().running(); }, GAME_TIMEOUT_S);
    check(ended, at + ": the game did not end within " + std::to_string(static_cast<int>(GAME_TIMEOUT_S)) + " s");
    const bool finished = app.play().state() == rmwork::PlaySession::State::FINISHED;
    check(finished, at + ": the session did not finish (" + app.error_title() + ": " + app.error_message() + ")");
    const rmwork::Match *m = app.play().match();
    std::printf("rmwork: %s: %zu decisions of yours (%zu answered by the passing), %zu records, %zu withheld; %s\n",
        at.c_str(), m ? m->decisions.size() : 0, app.play().auto_answered().size(), app.play().records(),
        app.play().withheld(), m ? m->result.text.c_str() : "no match");
    return finished;
}

// Each decision's passing facts, menu categories and whether a decision or cost is pending.
static Recorded record(const rmwork::Match &m) {
    Recorded r;
    for (size_t i = 0; i < m.decisions.size(); ++i) {
        std::string err;
        const int seat = m.decisions[i].decider;
        const std::shared_ptr<const rmstate::DecodedFrame> f = m.frame(i, seat, &err);
        if (!f) continue;
        const agents::DecisionView v(m.stream(seat), *f, seat);
        r.facts.push_back(rmwork::read_pass_facts(v));
        std::vector<std::string> cats;
        for (size_t k = 0; k < v.n(); ++k) cats.push_back(v.category(k));
        r.categories.push_back(cats);
        r.pending.push_back(v.table("pending").rows() > 0 || v.table("pending_cost").rows() > 0);
    }
    return r;
}

// Every table of `a` but the event tables equals `b`'s (text by its string, other cells by value).
static bool same_except_events(const rmstate::DecodedFrame &a, const rmstate::DecodedFrame &b, std::string *why) {
    const auto keep = [](const rmstate::DecodedTable &t) { return t.def && t.def->name.compare(0, 5, "event") != 0; };
    std::vector<const rmstate::DecodedTable *> ta, tb;
    for (const rmstate::DecodedTable &t : a.tables)
        if (keep(t)) ta.push_back(&t);
    for (const rmstate::DecodedTable &t : b.tables)
        if (keep(t)) tb.push_back(&t);
    if (ta.size() != tb.size()) {
        *why = "the frames hold " + std::to_string(ta.size()) + " and " + std::to_string(tb.size()) + " tables";
        return false;
    }
    for (size_t i = 0; i < ta.size(); ++i) {
        const rmstate::DecodedTable &x = *ta[i], &y = *tb[i];
        if (x.id != y.id || x.rows.size() != y.rows.size()) {
            *why = "table " + x.def->name + " differs in its rows";
            return false;
        }
        for (size_t r = 0; r < x.rows.size(); ++r)
            for (size_t c = 0; c < x.rows[r].size() && c < x.def->columns.size(); ++c) {
                const bool text = x.def->columns[c].storage == rmstate::Storage::TEXT;
                const rmstate::Value &u = x.rows[r][c], &v = y.rows[r][c];
                if (text ? u.text != v.text : u.bits != v.bits) {
                    *why = "table " + x.def->name + " row " + std::to_string(r) + " column " + x.def->columns[c].name;
                    return false;
                }
            }
    }
    return true;
}

// The engine's arguments per opponent and the engine binary chosen for one.
static void check_arguments(const CheckFn &check, const rmwork::PlayTestEnv &env) {
    using namespace rmwork;
    PlayConfig c;
    c.deck = "league/burn";
    c.opponent_deck = "league/bug";
    c.seed = 7;
    std::vector<std::string> a = play_arguments(c, "/g/x.rmlog", "/repo", "python3");
    check(has_pair(a, "--agent-b", "explore") && has_pair(a, "--deck-a", "league/burn") &&
              has_pair(a, "--deck-b", "league/bug") && has_pair(a, "--game-seed", "7") &&
              has_pair(a, "--decision-log", "/g/x.rmlog") && a[0] == "--machine" &&
              std::find(a.begin(), a.end(), "--bo3") == a.end(),
        "Play: the engine's arguments against explore");
    c.seat = 1;
    c.bo3 = true;
    c.auto_mana = false;
    c.opponent = Opponent::RANDOM;
    a = play_arguments(c, "/g/x.rmlog", "/repo", "python3");
    check(has_pair(a, "--agent-a", "random") && has_pair(a, "--deck-a", "league/bug") &&
              has_pair(a, "--deck-b", "league/burn") && std::find(a.begin(), a.end(), "--bo3") != a.end() &&
              std::find(a.begin(), a.end(), "--no-auto-mana") != a.end(),
        "Play: the engine's arguments as seat B, bo3, --no-auto-mana");
    c.opponent = Opponent::SCRIPTED;
    a = play_arguments(c, "/g/x.rmlog", "/r p", "python3");
    check(has_pair(a, "--agent-a", "first") && has_pair(a, "--teacher-mix", "1") &&
              has_pair(a, "--teacher-a", "'python3' '/r p/python/teacher.py' --tier hard --seed 7"),
        "Play: the scripted opponent's teacher channel");
    c.opponent = Opponent::MODEL;
    c.model = "/m/final.pt2";
    check(has_pair(play_arguments(c, "/g", "/r", "python3"), "--agent-a", "nn:/m/final.pt2"),
        "Play: the model opponent's agent");
    c.opponent = Opponent::SERVER;
    c.model = "/s.sock";
    check(has_pair(play_arguments(c, "/g", "/r", "python3"), "--agent-a", "nn:remote:/s.sock"),
        "Play: the server opponent's agent");
    // The engine chosen: robomage_nn only for nn: opponents.
    const std::string with = env.tmp + "/withnn", without = env.tmp + "/nnless";
    make_dirs(with);
    make_dirs(without);
    const bool made = make_executable(with + "/robomage", "") && make_executable(with + "/robomage_nn", "") &&
                      make_executable(without + "/robomage", "");
    check(made, "Play: cannot make the engine stand-ins");
    std::string why;
    check(play_engine_binary(with + "/robomage_nn", Opponent::EXPLORE, &why) == with + "/robomage" &&
              play_engine_binary(with + "/robomage", Opponent::MODEL, &why) == with + "/robomage_nn" &&
              play_engine_binary(with + "/robomage_nn", Opponent::SERVER, &why) == with + "/robomage_nn",
        "Play: robomage for explore, robomage_nn for an nn: opponent");
    why.clear();
    check(play_engine_binary(without + "/robomage", Opponent::MODEL, &why).empty() &&
              why.find("robomage_nn") != std::string::npos && why.find("make actor") != std::string::npos,
        "Play: an nn: opponent without robomage_nn is not reported: " + why);
    NewGameSettings g;
    check(NewGameDialog::check(g, env.engine).empty(), "New Game: the default choices do not start");
    g.opponent = "model";
    g.model = with + "/robomage";
    check(NewGameDialog::check(g, without + "/robomage").find("robomage_nn") != std::string::npos,
        "New Game: a model opponent without robomage_nn is accepted");
    g.model.clear();
    check(!NewGameDialog::check(g, with + "/robomage").empty(), "New Game: a model opponent without a model starts");
    g.opponent = "server";
    g.server = env.tmp + "/no.sock";
    check(!NewGameDialog::check(g, with + "/robomage").empty(), "New Game: a server opponent without a server starts");
}

// The dialog's decks and its opening and closing.
static void check_dialog(const CheckFn &check, rmwork::App &app, const Frames &frames, const rmwork::PlayTestEnv &env) {
    using namespace rmwork;
    const std::vector<std::string> decks = NewGameDialog::list_decks(env.workdir);
    const auto has = [&decks](const char *d) { return std::find(decks.begin(), decks.end(), d) != decks.end(); };
    check(has("delver") && has("league/burn") && has("heldout/storm") && !has("candidates/storm"),
        "New Game: the decks are not the top-level, league/ and heldout/ ones");
    app.open_new_game();
    frames(2);
    check(app.new_game_dialog().is_open() && app.new_game_dialog().problem().empty(),
        "New Game: the dialog does not open ready to start: " + app.new_game_dialog().problem());
    press_key(frames, ImGuiKey_Escape);
    check(!app.new_game_dialog().is_open(), "New Game: Esc does not close the dialog");
}

// Every record kept in the stream file is the human seat's, and the other seat's records came
// after the human's last decision.
static void check_stream_file(const CheckFn &check, const rmwork::PlaySession &s, const std::string &at) {
    FILE *f = std::fopen(s.stream_path().c_str(), "rb");
    check(f != nullptr, at + ": no stream file " + s.stream_path());
    if (!f) return;
    rmstate::RecordReader reader(f);
    std::string payload;
    rmstate::DecodeResult res;
    size_t n = 0, others = 0;
    while (reader.next(&payload, &res)) {
        ++n;
        if (payload.size() < 8 || static_cast<uint8_t>(payload[7]) != s.config().seat) ++others;
    }
    std::fclose(f);
    check(res.ok() && others == 0 && n + s.withheld() == s.records(),
        at + ": the stream file holds " + std::to_string(others) + " records of another viewer, " + std::to_string(n) +
            " kept of " + std::to_string(s.records()) + " read");
    check(s.withheld() == 0 || s.first_withheld_record() > s.last_decision_record(),
        at + ": another viewer's record came before the human's last decision");
}

// The stops answered exactly the plain priority passes with an empty stack in steps without a
// stop (no F2 or F6 in the game), each with its pass row.
static void check_stops_answered(const CheckFn &check, const rmwork::PlaySession &s, const std::string &at) {
    const rmwork::Match *m = s.match();
    if (!m) return;
    const Recorded r = record(*m);
    const rmwork::StopSettings stops = rmwork::default_stops();
    size_t passed = 0, wrong = 0;
    for (size_t i = 0; i < r.facts.size(); ++i) {
        const rmwork::PassFacts &f = r.facts[i];
        const bool stop = f.step >= 0 && (f.own_turn ? stops.own : stops.opponent)[static_cast<size_t>(f.step)];
        const bool should = f.pass_row >= 0 && f.stack.empty() && f.step >= 0 && !stop;
        const bool did = std::find(s.auto_answered().begin(), s.auto_answered().end(), i) != s.auto_answered().end();
        if (should != did || (did && m->decisions[i].reply != f.pass_row)) ++wrong;
        passed += did ? 1 : 0;
    }
    check(wrong == 0, at + ": " + std::to_string(wrong) + " decisions the stops answered wrongly");
    check(passed > 0, at + ": the stops answered no decision");
}

// The decision log in Replay: the log's count, the human's decisions in order with the same keys,
// replies and frames (event tables aside).
static void check_replay(const CheckFn &check, rmwork::App &app, const Frames &frames, const std::string &at) {
    using namespace rmwork;
    const Match *live = app.play().match();
    const std::string log = app.play().log_path();
    const int seat = app.play().config().seat;
    check(app.open_replay(log), at + ": the decision log does not open in Replay");
    run_until(frames, [&app] { return !app.loading(); }, GAME_TIMEOUT_S);
    check(app.mode() == Mode::REPLAY && app.replay().loaded(),
        at + ": the decision log did not load: " + app.error_message());
    if (!app.replay().loaded() || !live) return;
    const Match &rm = *app.replay().match();
    check(rm.decisions.size() == rm.log.replies.size(), at + ": Replay's decision count is not the log's");
    std::vector<size_t> mine;
    for (size_t i = 0; i < rm.decisions.size(); ++i)
        if (rm.decisions[i].decider == seat) mine.push_back(i);
    check(mine.size() == live->decisions.size(), at + ": Replay has " + std::to_string(mine.size()) +
                                                     " decisions of the human's seat, the game " +
                                                     std::to_string(live->decisions.size()));
    size_t bad = 0;
    std::string first;
    for (size_t k = 0; k < std::min(mine.size(), live->decisions.size()); ++k) {
        const DecisionInfo &a = live->decisions[k], &b = rm.decisions[mine[k]];
        std::string why;
        if (a.game != b.game || a.sideboarding != b.sideboarding || a.decision != b.decision) why = "another key";
        if (why.empty() && a.reply != b.reply)
            why = "reply " + std::to_string(a.reply) + " vs " + std::to_string(b.reply);
        std::string err;
        const std::shared_ptr<const rmstate::DecodedFrame> fa = live->frame(k, seat, &err);
        const std::shared_ptr<const rmstate::DecodedFrame> fb = rm.frame(mine[k], seat, &err);
        if (why.empty() && (!fa || !fb)) why = "a frame does not decode: " + err;
        if (why.empty()) same_except_events(*fa, *fb, &why);
        if (!why.empty()) {
            if (!bad++) first = "decision " + std::to_string(k + 1) + ": " + why;
        }
    }
    check(bad == 0, at + ": " + std::to_string(bad) + " of the human's decisions differ in Replay; " + first);
}

// F2, F6 and the stops over the recorded facts.
static void check_passing_recorded(const CheckFn &check, const Recorded &r) {
    using namespace rmwork;
    StopSettings all, none;
    all.own.fill(true);
    all.opponent.fill(true);
    size_t f2 = 0, f6 = 0, wrong = 0, priority = 0;
    for (size_t i = 0; i < r.facts.size(); ++i) {
        const PassFacts &f = r.facts[i];
        // The pass row is the PASS_PRIORITY row of a frame with nothing pending.
        int expect = -1;
        for (size_t k = 0; k < r.categories[i].size() && expect < 0 && !r.pending[i]; ++k)
            if (r.categories[i][k] == "PASS_PRIORITY") expect = static_cast<int>(k);
        if (f.pass_row != expect) ++wrong;
        priority += f.pass_row >= 0 ? 1 : 0;
        AutoPass idle;
        if (auto_reply(&idle, f, all).row != -1) ++wrong;
        const AutoReply loose = auto_reply(&idle, f, none);
        if (loose.row != ((f.pass_row >= 0 && f.stack.empty() && f.step >= 0) ? f.pass_row : -1)) ++wrong;
        // F2 from here: passes while the turn lasts, at plain priority decisions, with no new object
        // on the stack; F6 takes row 0 for the turn.
        AutoPass ap, ap6;
        start_auto_pass(&ap, PassMode::END_OF_TURN, f);
        start_auto_pass(&ap6, PassMode::SKIP_TURN, f);
        std::vector<uint64_t> seen = f.stack;
        bool f2_on = true, f6_on = true;
        for (size_t j = i; j < r.facts.size() && (f2_on || f6_on); ++j) {
            const PassFacts &g = r.facts[j];
            const bool same_turn = g.game == f.game && g.sideboarding == f.sideboarding && g.turn == f.turn;
            if (f2_on) {
                bool gained = false;
                for (uint64_t h : g.stack) gained = gained || std::find(seen.begin(), seen.end(), h) == seen.end();
                const AutoReply a = auto_reply(&ap, g, all);
                const bool passes = same_turn && g.pass_row >= 0 && !gained;
                if (passes ? a.row != g.pass_row : (a.row != -1 || a.ended.empty())) ++wrong;
                if (!passes) {
                    f2_on = false;
                    ++f2;
                }
                seen = g.stack;
            }
            if (f6_on) {
                const AutoReply a = auto_reply(&ap6, g, all);
                if (same_turn ? a.row != 0 : (a.row != -1 || a.ended != "the turn ended")) ++wrong;
                if (!same_turn) {
                    f6_on = false;
                    ++f6;
                }
            }
        }
    }
    check(wrong == 0, "Passing: " + std::to_string(wrong) + " answers differ from F2 / F6 / the stops' rules on " +
                          std::to_string(r.facts.size()) + " recorded decisions");
    check(priority > 0 && f2 > 0 && f6 > 0, "Passing: the recorded game has no priority pass, or F2 / F6 never stop");
}

// The rules on made-up facts: a new object on the stack stops F2, an old one does not; a pending
// decision is not a priority pass; the stops by the turn's player.
static void check_passing_units(const CheckFn &check) {
    using namespace rmwork;
    PassFacts f;
    f.rows = 3;
    f.pass_row = 1;
    f.turn = 4;
    f.own_turn = true;
    f.step = 3;
    f.stack = {11};
    AutoPass ap;
    StopSettings all;
    all.own.fill(true);
    all.opponent.fill(true);
    start_auto_pass(&ap, PassMode::END_OF_TURN, f);
    check(auto_reply(&ap, f, all).row == 1, "F2: an object on the stack when F2 was given stops it");
    PassFacts g = f;
    g.stack = {11, 12};
    AutoReply r = auto_reply(&ap, g, all);
    check(r.row == -1 && r.ended == "something went on the stack" && ap.mode == PassMode::NONE,
        "F2: a new object on the stack does not stop it");
    start_auto_pass(&ap, PassMode::END_OF_TURN, g);
    g.stack = {11};
    check(auto_reply(&ap, g, all).row == 1, "F2: an object leaving the stack stops it");
    g.pass_row = -1;
    r = auto_reply(&ap, g, all);
    check(r.row == -1 && r.ended == "this decision is not a priority pass", "F2: a decision that is no pass");
    StopSettings stops = default_stops();
    PassFacts e;
    e.pass_row = 0;
    e.step = 9;  // main 2: no stop by default
    e.own_turn = false;
    AutoPass idle;
    check(auto_reply(&idle, e, stops).row == 0 && auto_reply(&idle, e, stops).by_stop,
        "Stops: an empty stack in a step without a stop is not passed");
    e.step = 3;
    check(auto_reply(&idle, e, stops).row == -1, "Stops: a step with a stop is passed");
    stops.opponent[3] = false;
    check(auto_reply(&idle, e, stops).row == 0, "Stops: the opponent's turn reads the own turn's stops");
    e.stack = {5};
    check(auto_reply(&idle, e, stops).row == -1, "Stops: a decision with an object on the stack is passed");
}

// A bo1 game against explore with Model's Read asked for and no server; the log in Replay; the
// passing on its frames; Save Decision Log.
static void check_bo1(const CheckFn &check, rmwork::App &app, const Frames &frames, const rmwork::PlayTestEnv &env) {
    using namespace rmwork;
    NewGameSettings g = game("explore", "league/burn", "league/bug", 0, false, 3);
    g.model_read = true;
    if (!play_game(check, app, frames, g, "Play bo1")) return;
    PlaySession &s = app.play();
    const Match *m = s.match();
    check(m && m->finished && m->result.known && !m->decisions.empty(), "Play bo1: no result or no decision");
    if (!m) return;
    check(s.model_read().state() == ModelReader::State::NOT_CONNECTED,
        "Play bo1: Model's Read without a server is not 'not connected': " + s.model_read().status_text());
    bool answered = true;
    for (const DecisionInfo &d : m->decisions)
        answered = answered && d.reply != REPLY_PENDING && d.decider == 0 && d.menu_rows > 0;
    check(answered, "Play bo1: a decision is not the human's, or not answered");
    check(app.game_over().is_open() &&
              app.game_over().summary().title == (m->result.winner == 0 ? "You win" : "You lose"),
        "Play bo1: the result screen does not open with the result");
    press_key(frames, ImGuiKey_Escape);
    check(!app.game_over().is_open(), "Play bo1: Esc does not close the result screen");
    check_stream_file(check, s, "Play bo1");
    check(s.withheld() > 0, "Play bo1: the stream carried no other viewer's terminal records (the engine changed?)");
    check_stops_answered(check, s, "Play bo1");
    std::string err;
    const std::string saved = env.tmp + "/saved.rmlog";
    std::string a, b;
    check(app.save_log_to(saved, &err) && read_file(saved, &a) && read_file(s.log_path(), &b) && a == b && !a.empty(),
        "Save Decision Log: the copy is not the log: " + err);
    RmlogInfo info;
    check(parse_rmlog(b, &info, &err) && !info.replies.empty(), "Play bo1: the decision log does not parse: " + err);
    const Recorded r = record(*m);
    check_passing_recorded(check, r);
    check_replay(check, app, frames, "Play bo1");
}

// A bo3 match as seat B against random, with its sideboarding.
static void check_bo3(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    if (!play_game(check, app, frames, game("random", "league/ur_tempo", "league/burn", 1, true, 5), "Play bo3"))
        return;
    const Match *m = app.play().match();
    size_t sideboarding = 0;
    for (const DecisionInfo &d : m->decisions) sideboarding += d.sideboarding ? 1 : 0;
    check(
        m->bo3() && m->result.known && m->result.wins[0] + m->result.wins[1] >= 2 && m->result.game_winners.size() >= 2,
        "Play bo3: the match result is not a best of three's: " + m->result.text);
    check(sideboarding > 0, "Play bo3: the human had no sideboarding decision");
    check(app.game_over().is_open() && !app.game_over().summary().score.empty(),
        "Play bo3: the result screen has no score");
    press_key(frames, ImGuiKey_Escape);
    check_stream_file(check, app.play(), "Play bo3");
}

// F2 and F6 given in a live game: F2 at a plain priority decision, F6 at the next decision after
// that turn; each answers the decisions of its turn as its rule says and leaves the first one it
// stops at to the human, with a note.
static void check_live_passing(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    PlaySession &s = app.play();
    check(app.start_game(game("explore", "league/uw_control", "league/bug", 0, false, 13)),
        "Live passing: the game does not start");
    uint64_t rng = 99;
    int given = 0;  // 0 none yet, 1 F2 given, 2 F6 given
    size_t f2_at = 0, f6_at = 0, f2_stop = 0, f6_stop = 0;
    const auto facts_at = [&s](size_t i) {
        std::string err;
        const std::shared_ptr<const rmstate::DecodedFrame> f = s.match()->frame(i, s.config().seat, &err);
        return f ? read_pass_facts(agents::DecisionView(s.match()->stream(s.config().seat), *f, s.config().seat))
                 : PassFacts();
    };
    while (
        given < 3 && run_until(frames, [&s] { return s.deciding() || !s.running(); }, GAME_TIMEOUT_S) && s.running()) {
        const size_t i = static_cast<size_t>(s.pending_index());
        const PassFacts f = facts_at(i);
        if (given == 0 && f.pass_row >= 0) {
            f2_at = i;
            app.play_pass(PassMode::END_OF_TURN);
            given = 1;
            continue;
        }
        if (given == 1 && s.pass_mode() == PassMode::NONE && i > f2_at) {
            f2_stop = i;
            f6_at = i;
            app.play_pass(PassMode::SKIP_TURN);
            given = 2;
            continue;
        }
        if (given == 2 && s.pass_mode() == PassMode::NONE && i > f6_at) {
            f6_stop = i;
            given = 3;
            break;
        }
        rng = rng * 6364136223846793005ull + 1442695040888963407ull;
        app.play_reply(static_cast<int>((rng >> 33) % static_cast<uint64_t>(std::max(1, f.rows))));
    }
    check(given == 3, "Live passing: the game ended before F2 and F6 were both given and stopped");
    if (given == 3) {
        const PassFacts a = facts_at(f2_at), stop = facts_at(f2_stop);
        size_t wrong = 0;
        for (size_t i = f2_at; i < f2_stop; ++i) {
            const PassFacts f = facts_at(i);
            const bool same_turn = f.game == a.game && f.turn == a.turn;
            if (!same_turn || s.match()->decisions[i].reply != f.pass_row) ++wrong;
        }
        bool gained = false;
        const PassFacts prev = facts_at(f2_stop - 1);
        for (uint64_t h : stop.stack)
            gained = gained || std::find(prev.stack.begin(), prev.stack.end(), h) == prev.stack.end();
        check(wrong == 0 && (stop.turn != a.turn || stop.game != a.game || stop.pass_row < 0 || gained),
            "Live passing: F2 answered " + std::to_string(f2_stop - f2_at) + " decisions, " + std::to_string(wrong) +
                " wrongly, and stopped at a decision it should have passed");
        const PassFacts b = facts_at(f6_at);
        wrong = 0;
        for (size_t i = f6_at; i < f6_stop; ++i) {
            const PassFacts f = facts_at(i);
            if (f.turn != b.turn || f.game != b.game || s.match()->decisions[i].reply != 0) ++wrong;
        }
        const PassFacts end = facts_at(f6_stop);
        check(wrong == 0 && (end.turn != b.turn || end.game != b.game),
            "Live passing: F6 answered " + std::to_string(f6_stop - f6_at) + " decisions, " + std::to_string(wrong) +
                " wrongly, or stopped inside its turn");
        std::printf("rmwork: Live passing: F2 answered %zu decisions, F6 %zu\n", f2_stop - f2_at, f6_stop - f6_at);
    }
    s.stop();
}

// Play's step bar on a live game: at the first of the human's decisions in a turn's step, clicks on
// a step toggle its stop and settings.json keeps it (selftest_table.h).
static void check_step_bar(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    PlaySession &s = app.play();
    check(app.start_game(game("explore", "league/burn", "league/bug", 0, false, 21)),
        "Step bar: the game does not start");
    uint64_t rng = 7;
    bool reached = false;
    while (run_until(frames, [&s] { return s.deciding() || !s.running(); }, GAME_TIMEOUT_S) && s.running()) {
        frames(2);
        if (app.table().model() && !app.table().layout().step_cells.empty()) {
            reached = true;
            break;
        }
        std::string err;
        const auto f = s.match()->frame(static_cast<size_t>(s.pending_index()), s.config().seat, &err);
        const int rows =
            f ? read_pass_facts(agents::DecisionView(s.match()->stream(s.config().seat), *f, s.config().seat)).rows : 1;
        rng = rng * 6364136223846793005ull + 1442695040888963407ull;
        app.play_reply(static_cast<int>((rng >> 33) % static_cast<uint64_t>(std::max(1, rows))));
    }
    check(reached, "Step bar: the game ended before a decision in a turn's step");
    if (reached) check_step_bar_clicks(check, app, frames, config_paths("").settings);
    s.stop();
}

// The concession dialog; a game conceded at a decision, the match conceded while the opponent
// thinks (queued).
static void check_concede(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    PlaySession &s = app.play();
    check(app.start_game(game("explore", "league/burn", "league/bug", 0, true, 9)), "Concede: the game does not start");
    run_until(frames, [&s] { return s.deciding() || !s.running(); }, GAME_TIMEOUT_S);
    app.open_concede(true);
    frames(2);
    check(app.concede_dialog().is_open(), "Concede: Game → Concede Match does not ask");
    press_key(frames, ImGuiKey_Escape);
    check(!app.concede_dialog().is_open() && s.running(), "Concede: Esc does not cancel");
    s.concede(false);
    check(!s.deciding() && !s.concede_queued(), "Concede: conceding at a decision does not answer it");
    s.concede(true);
    check(s.concede_queued(), "Concede: conceding while the opponent thinks is not queued");
    run_until(frames, [&s] { return !s.running(); }, GAME_TIMEOUT_S);
    const Match *m = s.match();
    check(s.state() == PlaySession::State::FINISHED && m && m->result.known && m->result.winner == 1,
        "Concede: the match did not end with the opponent winning");
    std::string text, err;
    RmlogInfo info;
    const bool parsed = read_file(s.log_path(), &text) && parse_rmlog(text, &info, &err);
    check(parsed && std::count(info.replies.begin(), info.replies.end(), -2) == 1 && !info.replies.empty() &&
              info.replies.back() == -3,
        "Concede: the decision log does not hold -2 and then -3 last");
    press_key(frames, ImGuiKey_Escape);
}

// The scripted agent through the teacher channel, when python3 runs.
static void check_scripted(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    if (std::system("python3 -c 'import sys' >/dev/null 2>&1") != 0) {
        std::printf("rmwork: python3 does not run: the scripted opponent is not checked\n");
        return;
    }
    if (!play_game(check, app, frames, game("scripted", "league/bug", "league/burn", 1, false, 4), "Play scripted"))
        return;
    const std::string tail = app.play().stderr_tail();
    check(tail.find("FATAL") == std::string::npos, "Play scripted: the engine reports " + tail);
    press_key(frames, ImGuiKey_Escape);
}

// The engine killed mid-game, a stream that does not decode, a record that is not a frame.
static void check_failures(
    const CheckFn &check, rmwork::App &app, const Frames &frames, const rmwork::PlayTestEnv &env) {
    using namespace rmwork;
    PlaySession &s = app.play();
    check(app.start_game(game("explore", "league/burn", "league/bug", 0, false, 11)),
        "Failures: the game does not start");
    run_until(frames, [&s] { return s.deciding() || !s.running(); }, GAME_TIMEOUT_S);
    s.kill_engine();
    run_until(frames, [&s] { return !s.running(); }, 30);
    check(s.state() == PlaySession::State::FAILED && app.error_open() &&
              app.error_title() == "The engine exited mid-game",
        "Failures: a killed engine does not reach the error dialog: " + app.error_title());
    press_key(frames, ImGuiKey_Escape);
    const struct {
            const char *name, *script, *title;
    } fakes[] = {{"garbage_engine", "#!/bin/sh\nprintf 'garbage\\n'\nsleep 30\n", "The engine's stream broke"},
        {"short_engine", "#!/bin/sh\nprintf 'RMSTATE 5\\nhello'\nsleep 30\n",
            "A frame from the engine could not be read"}};
    for (const auto &fk : fakes) {
        const std::string path = env.tmp + "/" + fk.name;
        check(make_executable(path, fk.script), std::string("Failures: cannot write ") + path);
        app.set_engine_override(path);
        check(app.start_game(game("explore", "league/burn", "league/bug", 0, false, 1)),
            std::string("Failures: ") + fk.name + " does not start");
        run_until(frames, [&s] { return !s.running(); }, 30);
        check(s.state() == PlaySession::State::FAILED && app.error_open() && app.error_title() == fk.title,
            std::string("Failures: ") + fk.name + " gives '" + app.error_title() + "'");
        press_key(frames, ImGuiKey_Escape);
    }
    app.set_engine_override(env.engine);
}

namespace rmwork {

void check_play(const CheckFn &check, App &app, const std::function<void(int)> &frames, const PlayTestEnv &env) {
    const auto start = std::chrono::steady_clock::now();
    app.set_engine_override(env.engine);
    check_arguments(check, env);
    check_passing_units(check);
    check_dialog(check, app, frames, env);
    check_bo1(check, app, frames, env);
    check_bo3(check, app, frames);
    check_start_games(check, app, frames);
    check_sideboard(check, app, frames, env);
    check_live_passing(check, app, frames);
    check_step_bar(check, app, frames);
    check_concede(check, app, frames);
    check_scripted(check, app, frames);
    check_failures(check, app, frames, env);
    check_start_stopped(check, app, frames);
    app.play().stop();
    app.set_mode(Mode::REPLAY);
    frames(2);
    std::printf("rmwork: Play's checks took %.1f s\n",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}

}  // namespace rmwork
