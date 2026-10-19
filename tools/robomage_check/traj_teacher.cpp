#include <cmath>
#include <cstdio>
#include <functional>

#include "../../src/agents/py_random.h"
#include "frames.h"
#include "gate.h"
#include "proc.h"
#include "scan.h"
#include "traj.h"
#include "util.h"

// ── The traj tier's teacher channel checks (--teacher-a/-b, agents/teacher.h; plan T4) ─────────
//
// python/teacher.py (the scripted agent, hard tier) labels seats of in-process runs; each part
// writes under <out>/traj/teacher/<part>/ and reports its own failures.

namespace {

const size_t MAX_FAILS = 20;
const char *const DECK_A = "league/bug";
const char *const DECK_B = "league/burn";

using Fails = std::vector<std::string>;

}  // namespace

static std::string shell_quote(const std::string &s);
static std::string teacher_command(const rmcheck::Options &opt, long seed, const std::string &record);
static rmcheck::ProcResult run_engine(const rmcheck::Options &opt, const std::vector<std::string> &args);
static bool run_ok(const std::string &label, const rmcheck::ProcResult &r, const std::string &text, Fails *fails);
static bool episodes(
    const std::string &label, const std::string &dir, std::vector<rmcheck::EpisodeFile> *out, Fails *fails);
static std::vector<const featurize::EpisodeStep *> seat_steps(const std::vector<rmcheck::EpisodeFile> &eps, int seat);
static std::string the_log(const std::string &dir);
static void check_offline_labels(const rmcheck::Options &opt, const std::string &label, long seed,
    const std::vector<std::string> &records, const std::vector<const featurize::EpisodeStep *> &steps, Fails *fails);
static void mix1_part(const rmcheck::Options &opt, const std::string &dir, Fails *fails, std::string *note);
static void mix0_part(const rmcheck::Options &opt, const std::string &dir, Fails *fails, std::string *note);
static void mix_half_part(const rmcheck::Options &opt, const std::string &dir, Fails *fails, std::string *note);
static void failure_part(const rmcheck::Options &opt, const std::string &dir, Fails *fails, std::string *note);

static std::string shell_quote(const std::string &s) {
    std::string out = "'";
    for (char c : s) out += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return out + "'";
}

// python/teacher.py at the hard tier, seeded with `seed`, recording its stream into `record`.
static std::string teacher_command(const rmcheck::Options &opt, long seed, const std::string &record) {
    return shell_quote(opt.python) + " " + shell_quote(rmcheck::path_join(opt.repo, "python/teacher.py")) + " --seed " +
           std::to_string(seed) + " --record " + shell_quote(record);
}

// The engine with `args`, from bin/, stderr merged into the output.
static rmcheck::ProcResult run_engine(const rmcheck::Options &opt, const std::vector<std::string> &args) {
    rmcheck::ProcSpec spec;
    spec.argv = {opt.binary};
    spec.argv.insert(spec.argv.end(), args.begin(), args.end());
    spec.cwd = opt.bin_dir;
    spec.timeout_s = 900;
    spec.merge_stderr = true;
    return rmcheck::run_process(spec);
}

// Whether a run exited with status 0 and printed no error line (`text` its output); a failure if not.
static bool run_ok(const std::string &label, const rmcheck::ProcResult &r, const std::string &text, Fails *fails) {
    std::vector<rmcheck::ScanLine> errors;
    rmcheck::scan_transcript(text, &errors, nullptr);
    if (r.ok() && errors.empty()) return true;
    fails->push_back(label + ": " + r.status_text() + (errors.empty() ? "" : ": " + errors[0].line));
    return false;
}

static bool episodes(
    const std::string &label, const std::string &dir, std::vector<rmcheck::EpisodeFile> *out, Fails *fails) {
    std::string err;
    if (rmcheck::read_episodes(dir, out, &err) && !out->empty()) return true;
    fails->push_back(label + ": no episodes in " + dir + " " + err);
    return false;
}

// The steps of `seat` over the episodes, in order.
static std::vector<const featurize::EpisodeStep *> seat_steps(const std::vector<rmcheck::EpisodeFile> &eps, int seat) {
    std::vector<const featurize::EpisodeStep *> out;
    for (const rmcheck::EpisodeFile &e : eps)
        for (const featurize::EpisodeStep &s : e.ep.steps)
            if (s.seat == seat) out.push_back(&s);
    return out;
}

// The one decision log a --trajectory-out directory holds ("" when not exactly one).
static std::string the_log(const std::string &dir) {
    const std::vector<std::string> logs = rmcheck::list_dir(dir, ".rmlog");
    return logs.size() == 1 ? rmcheck::path_join(dir, logs[0]) : "";
}

// python/teacher.py run again on the streams teachers recorded, one after the other on its stdin
// (successive matches of one stream): its replies are the steps' labels.
static void check_offline_labels(const rmcheck::Options &opt, const std::string &label, long seed,
    const std::vector<std::string> &records, const std::vector<const featurize::EpisodeStep *> &steps, Fails *fails) {
    rmcheck::ProcSpec spec;
    spec.argv = {opt.python, rmcheck::path_join(opt.repo, "python/teacher.py"), "--seed", std::to_string(seed)};
    for (const std::string &record : records) {
        std::string data;
        if (!rmcheck::read_file(record, &data)) {
            fails->push_back(label + ": no recorded stream " + record);
            return;
        }
        spec.stdin_data += data;
    }
    spec.timeout_s = 900;
    const rmcheck::ProcResult r = rmcheck::run_process(spec);
    if (!r.ok()) {
        fails->push_back(label + ": teacher.py on its recorded stream " + r.status_text() + ": " + r.err);
        return;
    }
    const std::vector<std::string> lines = rmcheck::split_lines(r.out);
    if (lines.size() != steps.size()) {
        fails->push_back(label + ": teacher.py answers its recorded stream " + std::to_string(lines.size()) +
                         " times, the seat has " + std::to_string(steps.size()) + " steps");
        return;
    }
    for (size_t k = 0; k < steps.size(); ++k)
        if (std::strtoll(lines[k].c_str(), nullptr, 10) != steps[k]->teacher) {
            fails->push_back(label + ": step " + std::to_string(k) + "'s label " + std::to_string(steps[k]->teacher) +
                             ", teacher.py on the recorded stream says " + lines[k]);
            return;
        }
}

// --teacher-mix 1 on seat A (explore deciding, first on B), bo3: the decisions are scripted:hard's
// through rm_driver.py with the same seed, the teacher's recorded stream is byte for byte the
// --machine stream of the same game (its seat A records, terminal frames included), every A step's
// label its reply with μ one-hot there, B's steps unlabelled.
static void mix1_part(const rmcheck::Options &opt, const std::string &dir, Fails *fails, std::string *note) {
    const long seed = opt.seed + 500;
    const std::string s = std::to_string(seed);
    const std::string record = rmcheck::path_join(dir, "teacher_a.rms");
    rmcheck::ProcSpec drive;
    drive.argv = {opt.python, rmcheck::path_join(opt.repo, "python/rm_driver.py"), "--player-a", "scripted",
        "--player-b", "first", "--deck-a", DECK_A, "--deck-b", DECK_B, "--format", "bo3", "--games", "1", "--seed", s,
        "--quiet", "--rmlog-dir", rmcheck::path_join(dir, "driver"), "--binary", opt.binary};
    drive.timeout_s = 900;
    drive.merge_stderr = true;
    const rmcheck::ProcResult d = rmcheck::run_process(drive);
    const rmcheck::ProcResult t =
        run_engine(opt, {"--agent-a", "explore", "--agent-b", "first", "--teacher-a",
                            teacher_command(opt, seed, record), "--teacher-mix", "1", "--deck-a", DECK_A, "--deck-b",
                            DECK_B, "--bo3", "--game-seed", s, "--trajectory-out", rmcheck::path_join(dir, "traj")});
    if (!run_ok("mix1: rm_driver.py", d, d.out, fails) || !run_ok("mix1: the taught run", t, t.out, fails)) return;
    std::vector<rmcheck::EpisodeFile> eps;
    if (!episodes("mix1", rmcheck::path_join(dir, "traj"), &eps, fails)) return;
    const std::vector<int64_t> driven = rmcheck::logged_replies(the_log(rmcheck::path_join(dir, "driver")));
    const std::vector<int64_t> taught = rmcheck::logged_replies(the_log(rmcheck::path_join(dir, "traj")));
    if (driven.empty() || driven != taught)
        fails->push_back("mix1: the taught run's decisions (" + std::to_string(taught.size()) +
                         ") are not scripted:hard's through rm_driver.py (" + std::to_string(driven.size()) + ")");
    const std::vector<const featurize::EpisodeStep *> a = seat_steps(eps, 0), b = seat_steps(eps, 1);
    size_t sideboarding = 0;
    for (size_t k = 0; k < a.size() && fails->size() < MAX_FAILS; ++k) {
        const featurize::EpisodeStep &st = *a[k];
        sideboarding += st.sideboarding ? 1 : 0;
        const size_t rows = st.tensors.a();
        bool onehot = st.has_mu && st.mu.size() == rows && st.chosen >= 0 && st.chosen < static_cast<int64_t>(rows);
        for (size_t r = 0; onehot && r < rows; ++r)
            onehot = st.mu[r] == (static_cast<int64_t>(r) == st.chosen ? 1.0f : 0.0f);
        if (st.teacher != st.chosen || !onehot)
            fails->push_back("mix1: A step " + std::to_string(k) + ": label " + std::to_string(st.teacher) +
                             ", reply " + std::to_string(st.chosen) + (onehot ? "" : ", μ not one-hot at the reply"));
    }
    for (const featurize::EpisodeStep *st : b)
        if (st->teacher != -1 || st->has_mu) {
            fails->push_back("mix1: an untaught B step with a label or μ");
            break;
        }
    if (eps.size() < 2 || sideboarding == 0)
        fails->push_back("mix1: " + std::to_string(eps.size()) + " games, " + std::to_string(sideboarding) +
                         " A sideboarding steps (the bo3 match should sideboard)");
    // The --machine stream of the same game, seat A answered with the taught run's replies.
    size_t next = 0;
    rmcheck::ProcSpec stream;
    stream.argv = {opt.binary, "--machine", "--agent-b", "first", "--deck-a", DECK_A, "--deck-b", DECK_B, "--bo3",
        "--game-seed", s, "--decision-log", rmcheck::path_join(dir, "stream.rmlog")};
    stream.cwd = opt.bin_dir;
    stream.timeout_s = 900;
    const rmcheck::StreamRun run = rmcheck::run_stream(stream, [&](const rmcheck::StreamDecision &sd) {
        if (sd.viewer != rmcheck::VIEWER_A || next >= a.size()) return -2;
        return static_cast<int>(a[next++]->chosen);
    });
    if (!run.error.empty() || !run_ok("mix1: the --machine run", run.proc, run.proc.err, fails)) {
        if (!run.error.empty()) fails->push_back("mix1: the --machine run: " + run.error);
        return;
    }
    if (rmcheck::logged_replies(rmcheck::path_join(dir, "stream.rmlog")) != taught ||
        rmcheck::game_results(run.proc.err) != rmcheck::game_results(t.out))
        fails->push_back("mix1: the --machine run with the same replies played another game");
    std::vector<std::string> seen, sent;
    for (const std::string &p : run.payloads)
        if (rmcheck::frame_viewer(p) == rmcheck::VIEWER_A) sent.push_back(p);
    std::string err;
    if (!rmcheck::read_records(record, &seen, &err) || seen != sent)
        fails->push_back("mix1: the teacher's stream (" + std::to_string(seen.size()) +
                         " records) is not seat A's --machine stream (" + std::to_string(sent.size()) + ") " + err);
    *note = "mix1: " + std::to_string(taught.size()) + " decisions as rm_driver.py's scripted:hard, " +
            std::to_string(sent.size()) + " records as the --machine stream's";
}

// --teacher-mix 0 (both seats taught; explore against random, bo3): the decision log is byte for
// byte the one without teachers, every step's reply and μ the same, every step labelled, and the
// labels those teacher.py gives on the stream each teacher recorded, and on both streams read as two
// successive matches of one stream.
static void mix0_part(const rmcheck::Options &opt, const std::string &dir, Fails *fails, std::string *note) {
    const long seed = opt.seed + 501;
    const std::string s = std::to_string(seed);
    const std::vector<std::string> base = {"--agent-a", "explore", "--agent-b", "random", "--deck-a", DECK_A,
        "--deck-b", DECK_B, "--bo3", "--game-seed", s};
    std::vector<std::string> plain_args = base, taught_args = base;
    plain_args.insert(plain_args.end(), {"--trajectory-out", rmcheck::path_join(dir, "plain")});
    const std::string rec_a = rmcheck::path_join(dir, "teacher_a.rms"),
                      rec_b = rmcheck::path_join(dir, "teacher_b.rms");
    taught_args.insert(taught_args.end(),
        {"--teacher-a", teacher_command(opt, seed, rec_a), "--teacher-b", teacher_command(opt, seed, rec_b),
            "--teacher-mix", "0", "--trajectory-out", rmcheck::path_join(dir, "taught")});
    const rmcheck::ProcResult p = run_engine(opt, plain_args);
    const rmcheck::ProcResult t = run_engine(opt, taught_args);
    if (!run_ok("mix0: the run without teachers", p, p.out, fails) || !run_ok("mix0: the taught run", t, t.out, fails))
        return;
    std::string plain_log, taught_log;
    if (!rmcheck::read_file(the_log(rmcheck::path_join(dir, "plain")), &plain_log) ||
        !rmcheck::read_file(the_log(rmcheck::path_join(dir, "taught")), &taught_log) || plain_log != taught_log)
        fails->push_back("mix0: the decision log differs with teachers");
    std::vector<rmcheck::EpisodeFile> pe, te;
    if (!episodes("mix0 plain", rmcheck::path_join(dir, "plain"), &pe, fails) ||
        !episodes("mix0 taught", rmcheck::path_join(dir, "taught"), &te, fails))
        return;
    size_t steps = 0;
    std::vector<const featurize::EpisodeStep *> both;
    for (int seat = 0; seat < 2; ++seat) {
        const std::vector<const featurize::EpisodeStep *> a = seat_steps(pe, seat), b = seat_steps(te, seat);
        if (a.size() != b.size()) {
            fails->push_back("mix0: seat " + std::to_string(seat) + " steps differ in number");
            continue;
        }
        for (size_t k = 0; k < a.size(); ++k)
            if (a[k]->chosen != b[k]->chosen || a[k]->has_mu != b[k]->has_mu || a[k]->mu != b[k]->mu ||
                a[k]->teacher != -1 || b[k]->teacher < 0 || b[k]->teacher >= static_cast<int64_t>(b[k]->tensors.a())) {
                fails->push_back("mix0: seat " + std::to_string(seat) + " step " + std::to_string(k) +
                                 ": the reply or μ differ with teachers, or a label is missing");
                break;
            }
        steps += b.size();
        both.insert(both.end(), b.begin(), b.end());
        check_offline_labels(
            opt, std::string("mix0: seat ") + (seat ? "B" : "A"), seed, {seat ? rec_b : rec_a}, b, fails);
    }
    check_offline_labels(opt, "mix0: both streams as two matches of one stream", seed, {rec_a, rec_b}, both, fails);
    char buf[160];
    std::snprintf(buf, sizeof buf,
        "mix0: %zu decisions labelled, play unchanged (%.1f s with two teachers, %.1f s without)", steps, t.seconds,
        p.seconds);
    *note = buf;
}

// --teacher-mix 0.5 on seat A (the random agent, uniform μ; explore on B), bo3: the reply is the
// label exactly when the coin (Python's random.Random(seed + 2**32)) falls under 0.5, μ the mixture
// 0.5·onehot(label) + 0.5·uniform, the labels teacher.py's on the recorded stream, and a --replay of
// the decision log (no teacher) writes the same episodes and results.
static void mix_half_part(const rmcheck::Options &opt, const std::string &dir, Fails *fails, std::string *note) {
    const long seed = opt.seed + 502;
    const std::string s = std::to_string(seed);
    const std::string record = rmcheck::path_join(dir, "teacher_a.rms");
    const rmcheck::ProcResult t =
        run_engine(opt, {"--agent-a", "random", "--agent-b", "explore", "--teacher-a",
                            teacher_command(opt, seed, record), "--teacher-mix", "0.5", "--deck-a", DECK_A, "--deck-b",
                            DECK_B, "--bo3", "--game-seed", s, "--trajectory-out", rmcheck::path_join(dir, "live")});
    if (!run_ok("mix0.5: the taught run", t, t.out, fails)) return;
    const std::string log = the_log(rmcheck::path_join(dir, "live"));
    const rmcheck::ProcResult r =
        run_engine(opt, {"--replay", log, "--trajectory-out", rmcheck::path_join(dir, "replay")});
    if (!run_ok("mix0.5: the replay", r, r.out, fails)) return;
    std::vector<rmcheck::EpisodeFile> live, replayed;
    if (!episodes("mix0.5 live", rmcheck::path_join(dir, "live"), &live, fails) ||
        !episodes("mix0.5 replay", rmcheck::path_join(dir, "replay"), &replayed, fails))
        return;
    for (const std::string &f : rmcheck::compare_replayed(live, replayed)) fails->push_back("mix0.5: " + f);
    if (rmcheck::game_results(t.out) != rmcheck::game_results(r.out) || rmcheck::game_results(t.out).empty())
        fails->push_back("mix0.5: the replay's results differ");
    agents::PyRandom coin(static_cast<uint64_t>(seed) + (uint64_t{1} << 32));
    const std::vector<const featurize::EpisodeStep *> a = seat_steps(live, 0);
    size_t played = 0, overruled = 0;
    for (size_t k = 0; k < a.size() && fails->size() < MAX_FAILS; ++k) {
        const featurize::EpisodeStep &st = *a[k];
        const size_t rows = st.tensors.a();
        const bool teacher_plays = coin.random() < 0.5;
        if (st.teacher < 0 || st.teacher >= static_cast<int64_t>(rows) || (teacher_plays && st.chosen != st.teacher)) {
            fails->push_back("mix0.5: A step " + std::to_string(k) + ": label " + std::to_string(st.teacher) +
                             ", reply " + std::to_string(st.chosen) + ", coin " +
                             (teacher_plays ? "teacher" : "agent"));
            continue;
        }
        played += teacher_plays ? 1 : 0;
        overruled += !teacher_plays && st.chosen != st.teacher ? 1 : 0;
        bool mixture = st.has_mu && st.mu.size() == rows;
        for (size_t i = 0; mixture && i < rows; ++i) {
            const double want =
                0.5 * (static_cast<int64_t>(i) == st.teacher ? 1.0 : 0.0) + 0.5 / static_cast<double>(rows);
            mixture = std::fabs(st.mu[i] - want) < 1e-6;
        }
        if (!mixture) fails->push_back("mix0.5: A step " + std::to_string(k) + ": μ is not the documented mixture");
    }
    if (played == 0 || overruled == 0)
        fails->push_back("mix0.5: the teacher played " + std::to_string(played) + " and the agent overruled it " +
                         std::to_string(overruled) + " times");
    check_offline_labels(opt, "mix0.5", seed, {record}, a, fails);
    *note = "mix0.5: " + std::to_string(a.size()) + " A steps (" + std::to_string(played) +
            " labels played by the coin), replayed identically";
}

// A teacher that exits at once, replies garbage, hangs past --teacher-timeout or exits with status 3
// at the end, and --teacher-mix on a seat the stream decides: each a fatal error naming the seat.
static void failure_part(const rmcheck::Options &opt, const std::string &dir, Fails *fails, std::string *note) {
    struct Case {
            const char *name;
            std::vector<std::string> args;
            const char *expect;
    };
    const std::string s = std::to_string(opt.seed + 503);
    const std::string scripted = teacher_command(opt, opt.seed + 503, rmcheck::path_join(dir, "exit3.rms"));
    const std::vector<std::string> game = {"--deck-a", DECK_A, "--deck-b", DECK_B, "--game-seed", s};
    std::vector<Case> cases = {
        {"exits", {"--agent-a", "explore", "--agent-b", "first", "--teacher-a", "true"}, "FATAL: teacher A (`true`)"},
        {"garbage", {"--agent-a", "explore", "--agent-b", "first", "--teacher-a", "echo garbage; exec cat >/dev/null"},
            "reply 'garbage' at decision 1"},
        {"hangs",
            {"--agent-a", "explore", "--agent-b", "first", "--teacher-a", "exec sleep 60", "--teacher-timeout", "1"},
            "FATAL: teacher A (`exec sleep 60`): no progress replying to decision 1 within 1 s"},
        {"exit_status", {"--agent-a", "explore", "--agent-b", "explore", "--teacher-b", scripted + "; exit 3"},
            "exited with status 3 at the end of the run"},
        {"mix_stream_seat", {"--agent-a", "explore", "--teacher-b", "exec cat >/dev/null", "--teacher-mix", "0.5"},
            "FATAL: --teacher-mix: seat B is taught but not decided by an in-process agent"},
    };
    for (Case &c : cases) {
        c.args.insert(c.args.end(), game.begin(), game.end());
        const rmcheck::ProcResult r = run_engine(opt, c.args);
        if (r.exit_code != 1 || r.timed_out || !rmcheck::contains(r.out, c.expect))
            fails->push_back(std::string("failures: ") + c.name + ": " + r.status_text() + ", expected '" + c.expect +
                             "' (" + rmcheck::trim(r.out.substr(r.out.size() > 300 ? r.out.size() - 300 : 0)) + ")");
    }
    *note = "failures: " + std::to_string(cases.size()) + " broken teachers end the run naming their seat";
}

namespace rmcheck {

void traj_teacher_part(const Options &opt, Report &rep) {
    if (opt.python.empty()) {
        rep.info("traj: the teacher channel skipped (no python3 on PATH)");
        return;
    }
    const double t0 = now_seconds();
    using Part = void (*)(const Options &, const std::string &, Fails *, std::string *);
    const std::vector<std::pair<const char *, Part>> parts = {
        {"mix1", mix1_part}, {"mix0", mix0_part}, {"mix_half", mix_half_part}, {"failures", failure_part}};
    std::vector<Fails> fails(parts.size());
    std::vector<std::string> notes(parts.size());
    parallel_for(parts.size(), opt.jobs, [&](size_t i) {
        const std::string dir = path_join(path_join(opt.out("traj"), "teacher"), parts[i].first);
        remove_tree(dir);
        make_dirs(dir);
        parts[i].second(opt, dir, &fails[i], &notes[i]);
    });
    for (size_t i = 0; i < parts.size(); ++i) {
        for (size_t k = 0; k < fails[i].size() && k < MAX_FAILS; ++k) rep.error("traj", "teacher: " + fails[i][k]);
        if (fails[i].empty()) rep.info("traj: teacher: " + notes[i]);
    }
    rep.info("traj: the teacher channel checked in " + fmt1(now_seconds() - t0) + "s");
}

}  // namespace rmcheck
