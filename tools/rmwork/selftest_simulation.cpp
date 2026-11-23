#include "selftest_simulation.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "files.h"
#include "paths.h"
#include "rmlog.h"
#include "subprocess.h"

namespace {

using Frames = std::function<void(int)>;
using rmwork::CheckFn;

// How long a simulation, a replay or a load may take.
const double TIMEOUT_S = 600;

}  // namespace

static void frames_of(rmwork::App &app, int n);
static void press_chord(rmwork::App &app, ImGuiKeyChord chord);
static bool make_executable(const std::string &path, const std::string &text);
static bool write(const std::string &path, const std::string &text);
static bool has_pair(const std::vector<std::string> &a, const std::string &key, const std::string &value);
static size_t count_of(const std::vector<std::string> &a, const std::string &item);
static bool wait_until(const Frames &frames, const std::function<bool()> &done, double seconds);
static bool replay_result(const rmwork::SimulationTestEnv &env, const std::string &log, const std::string &out,
    rmwork::LogResult *result, std::string *err);
static void check_replays(
    const CheckFn &check, const rmwork::Simulation &sim, const rmwork::SimulationTestEnv &env, const std::string &at);
static bool python_runs();
static bool base_name_is(const std::string &path, const std::string &name);
static rmwork::SimulateSettings seats(const std::string &a, const std::string &deck_a, const std::string &b,
    const std::string &deck_b, int matches, uint64_t seed);
static bool run_simulation(const CheckFn &check, rmwork::App &app, const Frames &frames,
    const rmwork::SimulateSettings &g, const std::string &at);
static void check_settings_units(const CheckFn &check);
static void check_argument_units(const CheckFn &check);
static void check_judge_units(const CheckFn &check);
static void check_tally_units(const CheckFn &check);
static void check_summary_units(const CheckFn &check, const std::string &tmp);
static void check_single(const CheckFn &check, rmwork::App &app, const Frames &frames,
    const rmwork::SimulationTestEnv &env);
static void check_batch(const CheckFn &check, rmwork::App &app, const Frames &frames,
    const rmwork::SimulationTestEnv &env);
static void check_failure_and_cancel(const CheckFn &check, const rmwork::SimulationTestEnv &env);

static void frames_of(rmwork::App &app, int n) {
    for (int i = 0; i < n; ++i) {
        app.begin_frame();
        ImGui::NewFrame();
        app.frame();
        ImGui::Render();
    }
}

// Presses and releases `chord` with its modifiers, a frame each.
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

static bool make_executable(const std::string &path, const std::string &text) {
    return write(path, text) && chmod(path.c_str(), 0755) == 0;
}

static bool write(const std::string &path, const std::string &text) {
    std::string err;
    return rmwork::make_dirs(rmwork::parent_dir(path)) && rmwork::write_file_atomic(path, text, &err);
}

static bool has_pair(const std::vector<std::string> &a, const std::string &key, const std::string &value) {
    for (size_t i = 0; i + 1 < a.size(); ++i)
        if (a[i] == key && a[i + 1] == value) return true;
    return false;
}

static size_t count_of(const std::vector<std::string> &a, const std::string &item) {
    return static_cast<size_t>(std::count(a.begin(), a.end(), item));
}

// Frames until `done` holds; false past `seconds`.
static bool wait_until(const Frames &frames, const std::function<bool()> &done, double seconds) {
    const auto start = std::chrono::steady_clock::now();
    while (!done()) {
        frames(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > seconds) return false;
    }
    return true;
}

// The result lines the engine's --replay of `log` writes (its output into `out`); false with `err`.
static bool replay_result(const rmwork::SimulationTestEnv &env, const std::string &log, const std::string &out,
    rmwork::LogResult *result, std::string *err) {
    rmwork::ChildProcess child;
    if (!child.start({env.engine, "--replay", log}, env.workdir, out, out, err)) return false;
    const auto start = std::chrono::steady_clock::now();
    while (child.poll()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > TIMEOUT_S) {
            child.kill_and_reap();
            *err = "the replay did not end";
            return false;
        }
    }
    std::string text;
    rmwork::read_file(out, &text);
    *result = rmwork::parse_result_lines(text);
    if (!child.ok()) *err = "--replay " + child.status_text() + ": " + text.substr(text.size() > 400 ? text.size() - 400 : 0);
    return child.ok();
}

// Every finished match's result lines are what the engine's --replay of its log reports.
static void check_replays(
    const CheckFn &check, const rmwork::Simulation &sim, const rmwork::SimulationTestEnv &env, const std::string &at) {
    using namespace rmwork;
    for (const SimMatch &m : sim.matches()) {
        if (m.state != SimState::FINISHED) continue;
        const std::string what = at + " match " + std::to_string(m.index + 1);
        LogResult r;
        std::string err;
        const bool ok = replay_result(env, m.log, env.tmp + "/replay.out", &r, &err);
        check(ok, what + ": its log does not replay: " + err);
        if (!ok) continue;
        check(r.games == m.result.games && r.match_line == m.result.match_line &&
                  r.match_winner == m.result.match_winner && r.wins[0] == m.result.wins[0] &&
                  r.wins[1] == m.result.wins[1],
            what + ": the replay's result is not the simulation's (" + sim_result_text(m) + ")");
        std::string bytes;
        RmlogInfo info;
        check(read_file(m.log, &bytes) && parse_rmlog(bytes, &info, &err) && info.replies.size() == m.decisions &&
                  m.decisions > 0,
            what + ": the decisions counted are not the log's");
    }
}

static bool python_runs() {
    return std::system("python3 -c 'import sys' >/dev/null 2>&1") == 0;
}

static bool base_name_is(const std::string &path, const std::string &name) {
    const size_t slash = path.rfind('/');
    return (slash == std::string::npos ? path : path.substr(slash + 1)) == name;
}

static rmwork::SimulateSettings seats(const std::string &a, const std::string &deck_a, const std::string &b,
    const std::string &deck_b, int matches, uint64_t seed) {
    rmwork::SimulateSettings g = rmwork::default_simulate_settings();
    g.seats[0].agent = a;
    g.seats[0].deck = deck_a;
    g.seats[1].agent = b;
    g.seats[1].deck = deck_b;
    g.matches = matches;
    g.seed = seed;
    g.alternate = false;
    g.jobs = 1;
    return g;
}

// Starts `g` through the App and waits for its end (and a single match's load in Replay).
static bool run_simulation(const CheckFn &check, rmwork::App &app, const Frames &frames,
    const rmwork::SimulateSettings &g, const std::string &at) {
    const bool started = app.start_simulation(g);
    check(started, at + ": the simulation does not start: " + app.error_message());
    if (!started) return false;
    const bool done =
        wait_until(frames, [&app] { return !app.simulation().running() && !app.loading(); }, TIMEOUT_S);
    check(done, at + ": the simulation does not end");
    frames(2);
    return done;
}

static void check_settings_units(const CheckFn &check) {
    using namespace rmwork;
    const Settings def = default_settings();
    check(def.simulate.seats[0].agent == "explore" && def.simulate.seats[1].agent == "random" &&
              def.simulate.seats[1].deck == "league/bug" && def.simulate.jobs == 0 && def.simulate.alternate,
        "Simulate: the default choices are not explore vs random, alternating, default jobs");
    Settings s = def;
    s.simulate.seats[0] = {"model", "runs/x/exports/m9.ts.pt", "/s.sock", "heldout/storm", true, 0.5};
    s.simulate.seats[1].agent = "scripted";
    s.simulate.seats[1].deck = "delver";
    s.simulate.seats[1].temperature = 2.25;
    s.simulate.bo3 = true;
    s.simulate.seed = 77;
    s.simulate.auto_mana = false;
    s.simulate.matches = 25;
    s.simulate.alternate = false;
    s.simulate.jobs = 3;
    Settings back;
    std::string err;
    check(settings_from_json(settings_to_json(s), &back, &err) && settings_to_json(back) == settings_to_json(s),
        "Simulate: the settings do not round-trip: " + err);
    const SimulateSettings &b = back.simulate;
    check(b.seats[0].agent == "model" && b.seats[0].model == "runs/x/exports/m9.ts.pt" && b.seats[0].argmax &&
              std::fabs(b.seats[0].temperature - 0.5) < 1e-9 && b.seats[0].deck == "heldout/storm" &&
              b.seats[1].agent == "scripted" && std::fabs(b.seats[1].temperature - 2.25) < 1e-9 && b.bo3 &&
              b.seed == 77 && !b.auto_mana && b.matches == 25 && !b.alternate && b.jobs == 3,
        "Simulate: the read-back choices differ");
    const char *refused[] = {"{\"simulate\": 3}", "{\"simulate\": {\"seats\": []}}",
        "{\"simulate\": {\"format\": \"bo5\"}}", "{\"simulate\": {\"matches\": 0}}", "{\"simulate\": {\"jobs\": -1}}",
        "{\"simulate\": {\"seed\": -2}}", "{\"simulate\": {\"seats\": [{\"temperature\": 0}, {}]}}",
        "{\"simulate\": {\"seats\": [{\"argmax\": \"yes\"}, {}]}}"};
    for (const char *text : refused)
        check(!settings_from_json(text, &back, &err), std::string("Simulate: accepted ") + text);
    check(settings_from_json("{\"simulate\": {\"matches\": 4}}", &back, &err) && back.simulate.matches == 4 &&
              back.simulate.seats[1].agent == "random",
        "Simulate: a partial \"simulate\" does not keep the other defaults: " + err);

    SimulateSettings g = default_simulate_settings();
    g.seats[0].agent = "model";
    g.seats[0].model = "runs/a/m.ts.pt";
    g.seats[0].argmax = true;
    g.seats[1].agent = "server";
    g.seats[1].server = "~/s.sock";
    g.seats[1].temperature = 0.5;
    g.jobs = 0;
    SimConfig c;
    std::string why;
    check(sim_config_from(g, &c, &why) && c.entrants[0].agent.model == repo_root() + "/runs/a/m.ts.pt" &&
              c.entrants[0].agent.options == ":argmax" &&
              c.entrants[1].agent.model == std::string(std::getenv("HOME")) + "/s.sock" &&
              c.entrants[1].agent.options == ":temp=0.5" && c.jobs == default_simulation_jobs() && c.jobs >= 1,
        "Simulate: the configuration of the choices: models resolved, the nn: options, the default jobs");
    g.seats[0].agent = "explore";
    check(sim_config_from(g, &c, &why) && c.entrants[0].agent.options.empty() && c.entrants[0].agent.model.empty(),
        "Simulate: an explore seat takes a model's options");
    g.seats[1].agent = "nobody";
    check(!sim_config_from(g, &c, &why) && why.find("nobody") != std::string::npos,
        "Simulate: an unknown agent kind is accepted");
    check(nn_options(false, 1.0).empty() && nn_options(true, 0.3) == ":argmax" && nn_options(false, 2) == ":temp=2",
        "Simulate: the nn: options of argmax and a temperature");
}

static void check_argument_units(const CheckFn &check) {
    using namespace rmwork;
    SimConfig c;
    c.entrants[0].agent.kind = Opponent::SCRIPTED;
    c.entrants[0].deck = "league/burn";
    c.entrants[1].agent.kind = Opponent::EXPLORE;
    c.entrants[1].deck = "league/bug";
    SimMatch m;
    m.index = 1;
    m.seed = 8;
    m.swapped = true;
    std::vector<std::string> a = simulation_arguments(c, m, "/l.rmlog", {"", ""}, "/r", "python3");
    check(has_pair(a, "--deck-a", "league/bug") && has_pair(a, "--deck-b", "league/burn") &&
              has_pair(a, "--agent-a", "explore") && has_pair(a, "--agent-b", "first") &&
              has_pair(a, "--teacher-b", "'python3' '/r/python/teacher.py' --tier hard --seed 8") &&
              has_pair(a, "--teacher-mix", "1") && has_pair(a, "--game-seed", "8") &&
              has_pair(a, "--decision-log", "/l.rmlog") && count_of(a, "--machine") == 0 &&
              count_of(a, "--teacher-a") == 0 && count_of(a, "--bo3") == 0,
        "Simulate: a swapped match's arguments (seats, decks, the scripted seat's teacher)");
    c.entrants[1].agent.kind = Opponent::SCRIPTED;
    c.bo3 = true;
    c.auto_mana = false;
    a = simulation_arguments(c, m, "/l.rmlog", {"", ""}, "/r", "python3");
    check(count_of(a, "--teacher-a") == 1 && count_of(a, "--teacher-b") == 1 && count_of(a, "--teacher-mix") == 1 &&
              has_pair(a, "--agent-a", "first") && has_pair(a, "--agent-b", "first") && count_of(a, "--bo3") == 1 &&
              count_of(a, "--no-auto-mana") == 1,
        "Simulate: both seats scripted: two teachers, one --teacher-mix");
    c.entrants[0].agent = {Opponent::MODEL, "/models/m.zip!exports/m.ts.pt", ":argmax"};
    c.entrants[1].agent = {Opponent::SERVER, "/s.sock", ":temp=0.5"};
    m.swapped = false;
    a = simulation_arguments(c, m, "/l.rmlog", {"/cache/m.ts.pt", ""}, "/r", "python3");
    check(has_pair(a, "--agent-a", "nn:/cache/m.ts.pt:argmax") && has_pair(a, "--agent-b", "nn:remote:/s.sock:temp=0.5") &&
              count_of(a, "--teacher-mix") == 0,
        "Simulate: a model's file (a zipped one extracted) and the nn: options");
    // Play's arguments go through the same builder; a sculpted game's own --agent-b comes after it.
    PlayConfig pc;
    pc.opponent = Opponent::EXPLORE;
    pc.setup = {"--agent-b", "first"};
    const std::vector<std::string> play = play_arguments(pc, "/g/x.rmlog", "/r", "python3");
    const auto last = std::find(play.rbegin(), play.rend(), "--agent-b");
    check(count_of(play, "--agent-b") == 2 && last != play.rend() && last != play.rbegin() && *(last - 1) == "first",
        "Simulate: Play's setup flags do not override the opponent's agent");
    check(simulation_description(c) == "model-burn-vs-server-bug",
        "Simulate: the directory's description is " + simulation_description(c));
    SimEntrant e;
    e.agent = {Opponent::MODEL, "/runs/latest/policy.ts.pt", ":argmax"};
    e.deck = "league/burn";
    check(sim_entrant_text(e) == "nn: policy.ts.pt argmax \xC2\xB7 league/burn",
        "Simulate: an entrant's text is " + sim_entrant_text(e));
}

static void check_judge_units(const CheckFn &check) {
    using namespace rmwork;
    SimMatch m;
    m.swapped = true;
    judge_match(&m, true, "exit 0", "robomage 0.3\nPlayer A has 0 life\nGAME_RESULT: 1 Player B wins\n", false);
    check(m.state == SimState::FINISHED && m.winner_seat == 1 && m.winner() == 0 && m.games() == 1 &&
              sim_result_text(m) == "B wins",
        "Simulate: a bo1 win, swapped, judged");
    judge_match(&m, true, "exit 0", "GAME_RESULT: 1 draw\n", false);
    check(m.state == SimState::FINISHED && m.winner_seat == -1 && m.winner() == -1 && sim_result_text(m) == "draw",
        "Simulate: a drawn game judged");
    m.swapped = false;
    judge_match(&m, true, "exit 0",
        "GAME_RESULT: 1 Player A wins\nGAME_RESULT: 2 Player B wins\nGAME_RESULT: 3 Player A wins\n"
        "MATCH_RESULT: Player A wins 2-1\n",
        true);
    check(m.state == SimState::FINISHED && m.winner_seat == 0 && m.games() == 3 && sim_result_text(m) == "A wins 2-1",
        "Simulate: a bo3 match judged: " + sim_result_text(m));
    judge_match(&m, false, "exit 1", "robomage 0.3\nFATAL: --deck-a: no deck x\n", false);
    check(m.state == SimState::FAILED && m.error == "FATAL: --deck-a: no deck x" && sim_result_text(m) == "failed",
        "Simulate: a FATAL judged a failure: " + m.error);
    judge_match(&m, false, "killed by signal 9", "GAME_RESULT: 1 Player A wins\n", false);
    check(m.state == SimState::FAILED && m.error.find("signal 9") != std::string::npos,
        "Simulate: a killed engine judged a failure");
    judge_match(&m, true, "exit 0", "robomage 0.3\n", false);
    check(m.state == SimState::FAILED && m.error == "no GAME_RESULT line", "Simulate: no result judged a failure");
    judge_match(&m, true, "exit 0", "GAME_RESULT: 1 Player A wins\n", true);
    check(m.state == SimState::FAILED && m.error == "no MATCH_RESULT line",
        "Simulate: a bo3 match without its MATCH_RESULT judged finished");
}

static void check_tally_units(const CheckFn &check) {
    using namespace rmwork;
    std::vector<SimMatch> ms;
    for (int i = 0; i < 10; ++i) {
        SimMatch m;
        m.index = i;
        m.swapped = i % 2 == 1;
        m.state = SimState::FINISHED;
        // Entrant 0 wins matches 0-5, entrant 1 the rest.
        const int winner = i < 6 ? 0 : 1;
        m.winner_seat = m.swapped ? 1 - winner : winner;
        m.result.games = {m.winner_seat};
        ms.push_back(m);
    }
    SimMatch failed, cancelled, waiting;
    failed.state = SimState::FAILED;
    cancelled.state = SimState::CANCELLED;
    waiting.state = SimState::WAITING;
    ms.push_back(failed);
    ms.push_back(cancelled);
    ms.push_back(waiting);
    const SimTally t = tally_simulation(ms);
    check(t.finished == 10 && t.failed == 1 && t.cancelled == 1 && t.waiting == 1 && t.wins[0] == 6 &&
              t.wins[1] == 4 && t.game_wins[0] == 6 && t.game_wins[1] == 4 && t.seat_wins[0] + t.seat_wins[1] == 10,
        "Simulate: the tally's counts");
    check(std::fabs(t.share - 0.6) < 1e-9 && std::fabs(t.low - 0.31267) < 1e-3 && std::fabs(t.high - 0.83182) < 1e-3,
        "Simulate: the Wilson interval of 6 of 10 is not 31.3%-83.2%");
    ms[0].winner_seat = -1;
    ms[0].result.games = {-1};
    const SimTally d = tally_simulation(ms);
    check(d.draws == 1 && d.game_draws == 1 && std::fabs(d.share - 0.55) < 1e-9, "Simulate: a draw counts half");
    check(tally_simulation({}).share < 0, "Simulate: an empty tally has a share");
}

static void check_summary_units(const CheckFn &check, const std::string &tmp) {
    using namespace rmwork;
    const std::string dir = tmp + "/sim_units/20261102-100000-explore-burn-vs-random-bug";
    SimConfig c;
    c.entrants[0].deck = "league/burn";
    c.entrants[1].agent.kind = Opponent::RANDOM;
    c.entrants[1].deck = "league/bug";
    c.matches = 3;
    c.alternate = true;
    c.seed = 40;
    c.jobs = 2;
    std::vector<SimMatch> ms(3);
    for (int i = 0; i < 3; ++i) {
        ms[i].index = i;
        ms[i].seed = 40 + static_cast<uint64_t>(i);
        ms[i].swapped = i == 1;
        ms[i].log = dir + "/m00" + std::to_string(i + 1) + ".rmlog";
        ms[i].stderr_path = dir + "/m00" + std::to_string(i + 1) + ".stderr";
    }
    judge_match(&ms[0], true, "exit 0", "GAME_RESULT: 1 Player B wins\n", false);
    ms[0].decisions = 91;
    ms[0].seconds = 1.5;
    judge_match(&ms[1], false, "exit 2", "FATAL: x\n", false);
    ms[2].state = SimState::RUNNING;
    const std::string json = simulation_json(c, ms, "/e/robomage", "running", dir);
    check(json.find("\"log\": \"m001.rmlog\"") != std::string::npos, "Simulate: the summary's paths are not relative");
    SimConfig rc;
    std::vector<SimMatch> rm;
    std::string engine, state, err;
    check(parse_simulation(json, dir, &rc, &rm, &engine, &state, &err) && engine == "/e/robomage" &&
              state == "running" && rm.size() == 3 && rc.seed == 40 && rc.alternate && rc.jobs == 2 &&
              rc.entrants[1].agent.kind == Opponent::RANDOM && rc.entrants[1].deck == "league/bug" &&
              rm[0].log == ms[0].log && rm[0].state == SimState::FINISHED && rm[0].winner_seat == 1 &&
              rm[0].decisions == 91 && rm[0].result.games == ms[0].result.games && rm[1].swapped &&
              rm[1].state == SimState::FAILED && rm[1].error == "FATAL: x" && rm[2].state == SimState::RUNNING,
        "Simulate: simulation.json does not read back as written: " + err);
    check(!parse_simulation("{}", dir, &rc, &rm, &engine, &state, &err) &&
              !parse_simulation("[", dir, &rc, &rm, &engine, &state, &err),
        "Simulate: a malformed summary reads");
    check(write(dir + "/simulation.json", json) &&
              write(tmp + "/sim_units/20261101-100000-explore-burn-vs-random-bug/simulation.json", json) &&
              make_dirs(tmp + "/sim_units/20261103-100000-not-a-simulation"),
        "Simulate: cannot write the past simulations");
    Simulation sim;
    check(sim.open(dir, &err) && sim.state() == Simulation::State::CANCELLED &&
              sim.matches()[2].state == SimState::CANCELLED && sim.matches()[0].state == SimState::FINISHED,
        "Simulate: a summary written mid-batch does not open as cancelled: " + err);
    check(!sim.open(tmp + "/sim_units/20261103-100000-not-a-simulation", &err) && sim.dir() == dir,
        "Simulate: a directory without simulation.json opens");
    const std::vector<SimulationEntry> past = list_simulations(tmp + "/sim_units", 10);
    check(past.size() == 2 && past[0].dir == dir && past[0].matches == 3 && past[0].finished == 1 &&
              past[0].label.find("explore \xC2\xB7 league/burn vs random \xC2\xB7 league/bug") != std::string::npos &&
              list_simulations(tmp + "/sim_units", 1).size() == 1,
        "Simulate: the past simulations are not listed newest first");
}

static void check_single(const CheckFn &check, rmwork::App &app, const Frames &frames,
    const rmwork::SimulationTestEnv &env) {
    using namespace rmwork;
    if (run_simulation(check, app, frames, seats("explore", "league/burn", "random", "league/bug", 1, 5), "Simulate single")) {
        const Simulation &sim = app.simulation();
        check(sim.matches().size() == 1 && sim.matches()[0].state == SimState::FINISHED,
            "Simulate single: the match did not finish: " + (sim.matches().empty() ? "" : sim.matches()[0].error));
        check(app.mode() == Mode::REPLAY && app.replay().loaded() && !sim.matches().empty() &&
                  app.replay().match()->log_path == sim.matches()[0].log &&
                  app.replay().match()->decisions.size() == sim.matches()[0].decisions,
            "Simulate single: the match is not open in Replay when it ends");
        check(!app.simulation_view().is_open(), "Simulate single: one match opened the Simulations window");
        check(base_name_is(sim.engine(), "robomage"), "Simulate single: explore vs random runs " + sim.engine());
        check_replays(check, sim, env, "Simulate single");
    }
    if (!python_runs()) {
        std::printf("rmwork: python3 does not run: the scripted agent's simulation is not checked\n");
        return;
    }
    if (run_simulation(check, app, frames, seats("scripted", "league/burn", "explore", "league/bug", 1, 6),
            "Simulate scripted")) {
        const Simulation &sim = app.simulation();
        check(!sim.matches().empty() && sim.matches()[0].state == SimState::FINISHED,
            "Simulate scripted: the match did not finish: " + (sim.matches().empty() ? "" : sim.matches()[0].error));
        check_replays(check, sim, env, "Simulate scripted");
    }
    if (run_simulation(check, app, frames, seats("scripted", "league/burn", "scripted", "league/bug", 1, 7),
            "Simulate scripted pair")) {
        const Simulation &sim = app.simulation();
        check(!sim.matches().empty() && sim.matches()[0].state == SimState::FINISHED,
            "Simulate scripted pair: both seats scripted did not finish: " +
                (sim.matches().empty() ? "" : sim.matches()[0].error));
        check_replays(check, sim, env, "Simulate scripted pair");
    }
}

static void check_batch(const CheckFn &check, rmwork::App &app, const Frames &frames,
    const rmwork::SimulationTestEnv &env) {
    using namespace rmwork;
    SimulateSettings g = seats("explore", "league/burn", "random", "league/bug", 3, 21);
    g.alternate = true;
    g.jobs = 2;
    if (!run_simulation(check, app, frames, g, "Simulate batch")) return;
    const Simulation &sim = app.simulation();
    check(app.simulation_view().is_open(), "Simulate batch: the Simulations window did not open");
    check(sim.state() == Simulation::State::DONE && sim.config().jobs == 2 && sim.matches().size() == 3,
        "Simulate batch: not three matches done");
    bool all = true;
    for (const SimMatch &m : sim.matches()) all = all && m.state == SimState::FINISHED;
    check(all, "Simulate batch: a match did not finish");
    if (sim.matches().size() != 3) return;
    const SimMatch &swapped = sim.matches()[1];
    std::string bytes, err;
    RmlogInfo info;
    check(swapped.swapped && !sim.matches()[0].swapped && swapped.seed == 22 && read_file(swapped.log, &bytes) &&
              parse_rmlog(bytes, &info, &err) && info.deck_a.rfind("league/bug", 0) == 0 &&
              info.deck_b.rfind("league/burn", 0) == 0,
        "Simulate batch: the second match does not swap the entrants' seats");
    check_replays(check, sim, env, "Simulate batch");
    Simulation back;
    check(back.open(sim.dir(), &err), "Simulate batch: simulation.json does not open: " + err);
    bool same = back.matches().size() == sim.matches().size() && back.state() == Simulation::State::DONE;
    for (size_t i = 0; same && i < back.matches().size(); ++i) {
        const SimMatch &a = sim.matches()[i], &b = back.matches()[i];
        same = a.state == b.state && a.seed == b.seed && a.swapped == b.swapped && a.winner_seat == b.winner_seat &&
               a.result.games == b.result.games && a.decisions == b.decisions && a.log == b.log;
    }
    check(same, "Simulate batch: simulation.json does not read back as the batch ran");
    const std::vector<std::string> lines = SimulationView::summary_lines(sim);
    const SimTally t = tally_simulation(sim.matches());
    check(lines.size() == 4 && t.wins[0] + t.wins[1] + t.draws == 3 && lines[3].find("95% interval") != std::string::npos,
        "Simulate batch: the summary has no interval for alternating seats");
    // A row opened in Replay.
    SimulationRequest r;
    r.kind = SimulationRequest::Kind::OPEN_LOG;
    r.path = sim.matches()[2].log;
    r.match = 2;
    app.run_simulation_request(r);
    check(wait_until(frames, [&app] { return !app.loading(); }, TIMEOUT_S), "Simulate batch: the row's load does not end");
    frames(2);
    check(app.replay().loaded() && app.replay().match()->log_path == sim.matches()[2].log &&
              app.replay().match()->decisions.size() == sim.matches()[2].decisions,
        "Simulate batch: a row does not open its match in Replay");
    check(app.simulation_view().is_open(), "Simulate batch: opening a row closed the Simulations window");
    std::printf("rmwork: Simulate batch: %s; %s\n", sim.progress_text().c_str(), lines.back().c_str());
}

// A stand-in engine: the real one, but seed 302 fails with a FATAL line and seed 304 sleeps.
static void check_failure_and_cancel(const CheckFn &check, const rmwork::SimulationTestEnv &env) {
    using namespace rmwork;
    const std::string fake = env.tmp + "/sim_fake/robomage";
    check(make_executable(fake,
              "#!/bin/sh\ncase \" $* \" in\n  *\" --game-seed 302 \"*) echo \"FATAL: stand-in failure\" >&2; exit 3;;\n"
              "  *\" --game-seed 304 \"*) exec sleep 60;;\nesac\nexec '" +
                  env.engine + "' \"$@\"\n"),
        "Simulate: cannot write the stand-in engine");
    SimConfig c;
    std::string why;
    check(sim_config_from(seats("explore", "league/burn", "random", "league/bug", 5, 301), &c, &why),
        "Simulate: the stand-in's configuration: " + why);
    SimPaths p;
    p.engine = fake;
    p.workdir = env.workdir;
    p.root = env.tmp + "/sim_fake/simulations";
    p.repo = repo_root();
    Simulation sim;
    std::string err;
    check(sim.start(c, p, &err), "Simulate: the stand-in batch does not start: " + err);
    if (!sim.running()) return;
    const auto start = std::chrono::steady_clock::now();
    while (sim.matches()[3].state != SimState::RUNNING && sim.running()) {
        sim.poll();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > TIMEOUT_S) break;
    }
    const std::vector<SimMatch> &ms = sim.matches();
    check(ms[3].state == SimState::RUNNING && ms[4].state == SimState::WAITING,
        "Simulate: the stand-in's sleeping match is not running with the next waiting");
    check(ms[0].state == SimState::FINISHED && ms[2].state == SimState::FINISHED,
        "Simulate: a failed match stopped the others");
    check(ms[1].state == SimState::FAILED && ms[1].error == "FATAL: stand-in failure",
        "Simulate: the failed match's error is " + ms[1].error);
    check(sim.stderr_tail(1).find("FATAL: stand-in failure") != std::string::npos,
        "Simulate: the failed match's output does not hold its FATAL");
    const std::string sleeping_log = ms[3].log, waiting_log = ms[4].log;
    const auto cancel_at = std::chrono::steady_clock::now();
    sim.cancel();
    const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - cancel_at).count();
    check(took < 5, "Simulate: cancelling took " + std::to_string(took) + " s");
    check(sim.state() == Simulation::State::CANCELLED && ms[3].state == SimState::CANCELLED &&
              ms[4].state == SimState::CANCELLED && ms[3].log.empty() && !file_exists(sleeping_log) &&
              !file_exists(waiting_log),
        "Simulate: the cancelled matches are not cancelled without logs");
    check(file_exists(ms[0].log) && file_exists(ms[2].log), "Simulate: a cancel removed a finished match's log");
    check(!sim.take_finished(), "Simulate: a cancel is taken as the batch's end");
    Simulation back;
    check(back.open(sim.dir(), &err) && back.state() == Simulation::State::CANCELLED,
        "Simulate: the cancelled batch's summary does not open as cancelled: " + err);
    const SimTally t = tally_simulation(back.matches());
    check(t.finished == 2 && t.failed == 1 && t.cancelled == 2 && t.running == 0 && t.waiting == 0,
        "Simulate: the cancelled batch's summary does not count 2 finished, 1 failed, 2 cancelled");
    check_replays(check, back, env, "Simulate stand-in");
    std::printf("rmwork: Simulate stand-in: %s, cancelled in %.2f s\n", back.progress_text().c_str(), took);
}

namespace rmwork {

void check_simulation_units(const CheckFn &check, const std::string &tmp) {
    check_settings_units(check);
    check_argument_units(check);
    check_judge_units(check);
    check_tally_units(check);
    check_summary_units(check, tmp);
}

void check_simulation_app(const CheckFn &check, const ConfigPaths &paths, const std::string &tmp) {
    const std::string dir = tmp + "/sim_app";
    const char *old_data = std::getenv("XDG_DATA_HOME");
    const std::string saved = old_data ? old_data : "";
    setenv("XDG_DATA_HOME", (dir + "/data").c_str(), 1);
    const std::string engine = dir + "/bin/robomage";
    check(make_executable(engine, "#!/bin/sh\necho robomage 0.3\nexit 0\n"), "Simulate app: cannot write the engine");
    Settings s = default_settings();
    s.engine.binary = engine;
    s.engine.workdir = dir;
    {
        App app(s, paths, false);
        frames_of(app, 2);
        app.open_simulate();
        frames_of(app, 3);
        check(app.simulate_dialog().is_open() && app.simulate_dialog().problem().empty(),
            "Simulate app: the dialog does not open ready to run: " + app.simulate_dialog().problem());
        press_chord(app, ImGuiKey_Escape);
        check(!app.simulate_dialog().is_open(), "Simulate app: Esc does not close the dialog");
        press_chord(app, default_bindings()[action_index(Action::SIMULATE)]);
        frames_of(app, 1);
        check(app.simulate_dialog().is_open(), "Simulate app: the key does not open the dialog");
        press_chord(app, ImGuiKey_Escape);
        app.run_start_request({StartRequest::Kind::SIMULATE, "", 0});
        frames_of(app, 2);
        check(app.simulate_dialog().is_open(), "Simulate app: the start page's action does not open the dialog");
        press_chord(app, ImGuiKey_Escape);
        SimulateSettings g = default_simulate_settings();
        g.seats[0].agent = "model";
        g.seats[0].model = dir + "/none.ts.pt";
        check(SimulateDialog::check(g, engine).rfind("Seat A: ", 0) == 0,
            "Simulate app: a model seat without its model runs: " + SimulateDialog::check(g, engine));
        g.seats[0].model = engine;
        check(SimulateDialog::check(g, engine).find("robomage_nn") != std::string::npos,
            "Simulate app: a model seat without robomage_nn runs");
        g = default_simulate_settings();
        g.matches = 1;
        g.seed = 99;
        // The stand-in writes no result: the one match fails, and the error dialog says so.
        check(app.start_simulation(g) && app.settings().simulate.seed == 99,
            "Simulate app: the choices are not kept when a simulation starts");
        const Frames frames = [&app](int n) { frames_of(app, n); };
        check(wait_until(frames, [&app] { return !app.simulation().running(); }, 60),
            "Simulate app: the stand-in's simulation does not end");
        frames_of(app, 3);
        check(app.error_open() && app.error_title() == "Match 1 failed" &&
                  app.error_message().find("no GAME_RESULT line") != std::string::npos,
            "Simulate app: a single match's failure is not in the error dialog: " + app.error_title());
        press_chord(app, ImGuiKey_Escape);
        // The window over the past simulations.
        const std::string past = app.simulation().dir();
        app.simulation().close();
        check(app.open_simulation(past) && app.simulation_view().is_open() && app.simulation().loaded() &&
                  app.simulation().matches().size() == 1,
            "Simulate app: a past simulation does not open in the window");
        frames_of(app, 3);
        check(!app.simulation_view().past().empty() && app.simulation_view().past()[0].dir == past,
            "Simulate app: the window does not list the past simulation");
        const std::vector<std::string> lines = SimulationView::summary_lines(app.simulation());
        check(lines.size() == 3 && lines[2].find("1 failed") != std::string::npos,
            "Simulate app: the summary does not count the failure");
        app.run_simulation_request({SimulationRequest::Kind::FAILURE, "", 0});
        frames_of(app, 2);
        check(app.error_open() && app.error_title() == "Match 1 failed", "Simulate app: a failed row shows no error");
        press_chord(app, ImGuiKey_Escape);
        app.run_simulation_request({SimulationRequest::Kind::NEW, "", 0});
        frames_of(app, 2);
        check(app.simulate_dialog().is_open(), "Simulate app: the window's New does not open the dialog");
        press_chord(app, ImGuiKey_Escape);
        app.run_simulation_request({SimulationRequest::Kind::PICK, "", 0});
        frames_of(app, 2);
        check(app.file_picker().is_open() && app.file_picker().purpose() == App::PICK_SIMULATION,
            "Simulate app: Open Folder does not open the file picker");
        press_chord(app, ImGuiKey_Escape);
    }
    if (old_data)
        setenv("XDG_DATA_HOME", saved.c_str(), 1);
    else
        unsetenv("XDG_DATA_HOME");
}

void check_simulation_data(
    const CheckFn &check, App &app, const std::function<void(int)> &frames, const SimulationTestEnv &env) {
    check_single(check, app, frames, env);
    check_batch(check, app, frames, env);
    check_failure_and_cancel(check, env);
    app.simulation_view().close();
}

void check_simulation_evals(const CheckFn &check, App &app, const std::function<void(int)> &frames,
    const SimulationTestEnv &env, const std::string &model) {
    SimulateSettings g = seats("model", "league/burn", "model", "league/ur_tempo", 1, 9);
    g.seats[0].model = model;
    g.seats[1].model = model;
    g.seats[1].argmax = true;
    if (run_simulation(check, app, frames, g, "Simulate nn")) {
        const Simulation &sim = app.simulation();
        check(!sim.matches().empty() && sim.matches()[0].state == SimState::FINISHED,
            "Simulate nn: the model's match did not finish: " + (sim.matches().empty() ? "" : sim.matches()[0].error));
        check(base_name_is(sim.engine(), "robomage_nn"), "Simulate nn: the models play in " + sim.engine());
        check(app.replay().loaded() && !sim.matches().empty() && app.replay().match()->log_path == sim.matches()[0].log,
            "Simulate nn: the match is not open in Replay");
        check_replays(check, sim, env, "Simulate nn");
    }
    if (!python_runs()) return;
    g = seats("model", "league/burn", "scripted", "league/bug", 2, 30);
    g.seats[0].model = model;
    g.alternate = true;
    g.jobs = 2;
    if (run_simulation(check, app, frames, g, "Simulate nn vs scripted")) {
        const Simulation &sim = app.simulation();
        bool all = sim.matches().size() == 2;
        for (const SimMatch &m : sim.matches()) all = all && m.state == SimState::FINISHED;
        check(all, "Simulate nn vs scripted: a match did not finish");
        check_replays(check, sim, env, "Simulate nn vs scripted");
        std::printf("rmwork: Simulate nn vs scripted: %s\n", SimulationView::summary_lines(sim).back().c_str());
    }
    app.simulation_view().close();
}

}  // namespace rmwork
