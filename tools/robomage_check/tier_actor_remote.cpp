#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>

#include "../../src/featurize/rmtb.h"
#include "../../src/inference/protocol.h"
#include "actor.h"
#include "proc.h"
#include "scan.h"
#include "traj.h"
#include "util.h"

namespace {

// Remote and in-process evaluations agree within this, absolute plus relative.
const double EVAL_TOLERANCE = 1e-5;
// The kill check's server stops answering after this many requests.
const long KILL_AFTER = 40;
// A server signalled to stop is killed (and reported as timed out) past this.
const double STOP_TIMEOUT_S = 30;
// The version rmserve gives policy 1 (the .pt2 export).
const char *const POLICY1_VERSION = "7";

// One bo1 game of the remote checks, nn: argmax on both seats.
struct RemoteGame {
        std::string deck_a, deck_b;
        long seed = 0;
};

// The largest differences between two evals= files' evaluations of one policy.
struct EvalDiff {
        size_t decisions = 0;
        double value = 0, logits = 0, log_pi = 0;
        std::vector<int64_t> versions;  // the compared policy's versions seen
        std::string problem;            // why they cannot be compared, or the first difference past the tolerance
};

}  // namespace

static rmcheck::ProcSpec game_spec(const rmcheck::Options &opt, const rmcheck::ActorSetup &a, const RemoteGame &g,
    const std::string &spec_a, const std::string &spec_b, const std::string &rmlog);
static std::string game_problem(const rmcheck::ProcResult &r, const std::string &transcript);
static std::string decisions_of(const std::string &rmlog);
static std::string fmt_g(double v);
static std::string stat_field(const std::string &text, const char *line_prefix, const char *key);
static bool read_evals(const std::string &path, size_t *policies, std::vector<float> *value, std::vector<float> *logits,
    std::vector<float> *log_pi, std::vector<int64_t> *version, std::vector<int64_t> *offsets, std::string *err);
static EvalDiff compare_evals(const std::string &want, const std::string &got, size_t column);
static void note_diff(double want, double got, double *worst, const std::string &what, std::string *problem);
static std::string hello_refusal(const std::string &socket, const inference::Hello &h);
static void check_handshake(
    const rmcheck::Options &opt, rmcheck::Report &rep, const rmcheck::ActorSetup &a, const std::string &socket);
static void check_kill(const rmcheck::Options &opt, rmcheck::Report &rep, const rmcheck::ActorSetup &a,
    const std::string &rmserve, const std::string &dir, const RemoteGame &g);
static bool python_game(const rmcheck::Options &opt, rmcheck::Report &rep, const rmcheck::ActorSetup &a,
    const std::string &dir, const RemoteGame &g);
static void check_remote_trajectories(
    rmcheck::Report &rep, const std::string &dir, const std::string &evals_a, const std::string &evals_b);

// Remote game 0's trajectory episodes (--trajectory-out) against its seats' evals= files: seat A
// (policies=0,1) records both policies' values and log π, seat B its one, each step's μ and reply.
static void check_remote_trajectories(
    rmcheck::Report &rep, const std::string &dir, const std::string &evals_a, const std::string &evals_b) {
    std::vector<rmcheck::EpisodeFile> eps;
    std::string err;
    if (!rmcheck::read_episodes(dir, &eps, &err) || eps.empty()) {
        rep.error("actor", "remote game 0's trajectory episodes: " + (err.empty() ? "none in " + dir : err));
        return;
    }
    size_t a = 0, b = 0;
    std::vector<std::string> fails = rmcheck::check_against_evals(eps, 0, evals_a, &a);
    const std::vector<std::string> fb = rmcheck::check_against_evals(eps, 1, evals_b, &b);
    fails.insert(fails.end(), fb.begin(), fb.end());
    for (const std::string &f : fails) rep.error("actor", "remote trajectories: " + f);
    if (fails.empty())
        rep.info("remote trajectories: " + std::to_string(a + b) +
                 " steps carry the evals= files' mu, replies and every requested policy's log pi and value (seat A "
                 "policies 0,1)");
}

static rmcheck::ProcSpec game_spec(const rmcheck::Options &opt, const rmcheck::ActorSetup &a, const RemoteGame &g,
    const std::string &spec_a, const std::string &spec_b, const std::string &rmlog) {
    rmcheck::ProcSpec spec;
    spec.argv = {a.nn_binary, "--agent-a", spec_a, "--agent-b", spec_b, "--deck-a", g.deck_a, "--deck-b", g.deck_b,
        "--game-seed", std::to_string(g.seed), "--narrative", "--decision-log", rmlog};
    spec.cwd = opt.bin_dir;
    spec.timeout_s = 1800;
    return spec;
}

// "" when the game ran to a result with no error or warning line, else what went wrong (the
// transcript is saved to `transcript`).
static std::string game_problem(const rmcheck::ProcResult &r, const std::string &transcript) {
    rmcheck::write_file(transcript, r.out + "\n=== engine stderr ===\n" + r.err);
    if (!r.ok()) return "robomage_nn " + r.status_text() + " (see " + transcript + ")";
    bool result = false;
    for (const std::string &line : rmcheck::split_lines(r.out)) {
        if (!rmcheck::starts_with(line, "GAME_RESULT: ")) continue;
        if (rmcheck::ends_with(line, " draw")) return "a drawn game (see " + transcript + ")";
        result = true;
    }
    if (!result) return "no game result (see " + transcript + ")";
    std::vector<rmcheck::ScanLine> errs, warns;
    rmcheck::scan_transcript(r.out + r.err, &errs, &warns);
    if (!errs.empty()) return rmcheck::quoted(errs[0].line) + " (see " + transcript + ")";
    if (!warns.empty()) return rmcheck::quoted(warns[0].line) + " (see " + transcript + ")";
    return "";
}

// A decision log's decisions (the lines after DECISIONS), "" when it cannot be read.
static std::string decisions_of(const std::string &rmlog) {
    std::string text;
    if (!rmcheck::read_file(rmlog, &text)) return "";
    const size_t at = text.find("DECISIONS\n");
    return at == std::string::npos ? "" : text.substr(at + 10);
}

// Three significant digits.
static std::string fmt_g(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.3g", v);
    return buf;
}

// The value of ` key=` on the first line of `text` starting with `line_prefix`, "" when absent.
static std::string stat_field(const std::string &text, const char *line_prefix, const char *key) {
    const std::string k = std::string(" ") + key + "=";
    for (const std::string &line : rmcheck::split_lines(text)) {
        if (!rmcheck::starts_with(line, line_prefix)) continue;
        const size_t at = line.find(k);
        if (at == std::string::npos) return "";
        const size_t end = line.find(' ', at + k.size());
        return line.substr(at + k.size(), end == std::string::npos ? std::string::npos : end - at - k.size());
    }
    return "";
}

static bool read_evals(const std::string &path, size_t *policies, std::vector<float> *value, std::vector<float> *logits,
    std::vector<float> *log_pi, std::vector<int64_t> *version, std::vector<int64_t> *offsets, std::string *err) {
    featurize::RmtbFile f;
    if (!featurize::rmtb_read_file(path, &f, err)) return false;
    const featurize::RmtbArray *pol = f.find("policy"), *v = f.find("value"), *l = f.find("logits"),
                               *lp = f.find("log_pi"), *ver = f.find("version"), *off = f.find("menu_offsets");
    if (f.kind != "evaluations" || !pol || !v || !l || !lp || !ver || !off || !featurize::rmtb_values(*v, value) ||
        !featurize::rmtb_values(*l, logits) || !featurize::rmtb_values(*lp, log_pi) ||
        !featurize::rmtb_values(*ver, version) || !featurize::rmtb_values(*off, offsets) || offsets->empty()) {
        *err = path + " is not an evaluations file";
        return false;
    }
    *policies = static_cast<size_t>(pol->count());
    return *policies > 0;
}

// Records |want - got| into `worst`; past the tolerance, the first such difference into `problem`.
static void note_diff(double want, double got, double *worst, const std::string &what, std::string *problem) {
    const double d = std::fabs(want - got);
    *worst = std::max(*worst, d);
    if (d > EVAL_TOLERANCE * (1.0 + std::fabs(want)) && problem->empty())
        *problem = what + ": " + std::to_string(want) + " vs " + std::to_string(got);
}

// The in-process file `want` (one policy) against policy `column` of `got`, decision by decision.
static EvalDiff compare_evals(const std::string &want, const std::string &got, size_t column) {
    EvalDiff d;
    size_t pw = 0, pg = 0;
    std::vector<float> wv, wl, wlp, gv, gl, glp;
    std::vector<int64_t> wver, woff, gver, goff;
    std::string err;
    if (!read_evals(want, &pw, &wv, &wl, &wlp, &wver, &woff, &err) ||
        !read_evals(got, &pg, &gv, &gl, &glp, &gver, &goff, &err)) {
        d.problem = err;
        return d;
    }
    if (woff != goff || column >= pg) {
        d.problem = "the two files hold other decisions or menus (" + want + ", " + got + ")";
        return d;
    }
    d.decisions = woff.size() - 1;
    for (size_t i = 0; i < d.decisions; ++i) {
        const std::string at = "decision " + std::to_string(i);
        note_diff(wv[i * pw], gv[i * pg + column], &d.value, at + " value", &d.problem);
        if (std::find(d.versions.begin(), d.versions.end(), gver[i * pg + column]) == d.versions.end())
            d.versions.push_back(gver[i * pg + column]);
        for (int64_t r = woff[i]; r < woff[i + 1]; ++r) {
            const size_t row = static_cast<size_t>(r);
            note_diff(wl[row * pw], gl[row * pg + column], &d.logits, at + " logit", &d.problem);
            note_diff(wlp[row * pw], glp[row * pg + column], &d.log_pi, at + " log pi", &d.problem);
        }
    }
    return d;
}

// The ERROR code the server answers `h` with ("" when it answers otherwise).
static std::string hello_refusal(const std::string &socket, const inference::Hello &h) {
    std::string err, code, message;
    const int fd = inference::connect_unix(socket, &err);
    if (fd < 0) return "";
    inference::Message m;
    if (inference::send_message(fd, inference::MSG_HELLO, inference::hello_json(h), &err) &&
        inference::recv_message(fd, 30, &m, &err) == inference::RecvStatus::OK && m.type == inference::MSG_ERROR)
        inference::parse_error(m.body, &code, &message);
    close(fd);
    return code;
}

// The server refuses a handshake of another spec hash, protocol or an unserved policy (raw HELLOs),
// and an actor asking for an unserved policy fails naming the endpoint.
static void check_handshake(
    const rmcheck::Options &opt, rmcheck::Report &rep, const rmcheck::ActorSetup &a, const std::string &socket) {
    inference::Hello base;
    base.client = "robomage_check";
    base.spec_hash = "0123456789abcdef";
    base.vocab_hash = "0123456789abcdef";
    base.text_model = "none";
    base.shm = "/rmip-none";
    base.request_bytes = base.response_bytes = 8;
    base.policies = {0};
    inference::Hello protocol = base, policy = base;
    protocol.protocol = 2;
    policy.policies = {9};
    const struct {
            const inference::Hello *hello;
            const char *what, *code;
    } cases[] = {{&base, "a spec hash mismatch", "spec_hash"}, {&protocol, "protocol 2", "protocol"},
        {&policy, "policy 9", "policy"}};
    for (const auto &c : cases) {
        const std::string code = hello_refusal(socket, *c.hello);
        if (code != c.code)
            rep.error("actor", std::string("rmserve: a HELLO with ") + c.what + " was not refused with " + c.code +
                                   " (got " + rmcheck::quoted(code) + ")");
    }
    rmcheck::ProcSpec spec;
    spec.argv = {
        a.nn_binary, "--agent-a", "nn:remote:" + socket + ":policies=9", "--agent-b", "explore", "--game-seed", "1"};
    spec.cwd = opt.bin_dir;
    spec.merge_stderr = true;
    const rmcheck::ProcResult r = rmcheck::run_process(spec);
    if (r.ok() || !rmcheck::contains(r.out, "policy 9 is not served") || !rmcheck::contains(r.out, socket.c_str()))
        rep.error("actor", "nn:remote:...:policies=9: expected a fatal refusal naming the endpoint, got " +
                               r.status_text() + ": " + rmcheck::quoted(rmcheck::trim(r.out).substr(0, 300)));
    rep.info(
        "handshake: rmserve refuses a spec hash mismatch, protocol 2 and an unserved policy (ERROR spec_hash, "
        "protocol, policy); nn:remote: asking for one exits naming the endpoint");
}

// An rmserve that stops answering after KILL_AFTER requests is killed: the actor (both seats
// nn:remote:) exits nonzero naming the endpoint, its decision log holding exactly the answered
// decisions.
static void check_kill(const rmcheck::Options &opt, rmcheck::Report &rep, const rmcheck::ActorSetup &a,
    const std::string &rmserve, const std::string &dir, const RemoteGame &g) {
    const std::string sock = rmcheck::path_join(dir, "kill.sock"), rmlog = rmcheck::path_join(dir, "kill.rmlog");
    rmcheck::ServerRun server;
    std::string err;
    if (!rmcheck::start_server(opt,
            {rmserve, "--socket", sock, "--policy", "0=" + a.ts(), "--deadline-us", "0", "--pause-after",
                std::to_string(KILL_AFTER)},
            sock, &server, &err)) {
        rep.error("actor", "kill: " + err);
        return;
    }
    const std::string spec = "nn:remote:" + sock + ":argmax:timeout=300";
    rmcheck::Session actor;
    if (!actor.start(game_spec(opt, a, g, spec, spec, rmlog), &err)) {
        rep.error("actor", "kill: robomage_nn: " + err);
        return;
    }
    const bool paused = server.session.wait_stderr("paused after");
    const rmcheck::ProcResult s = rmcheck::stop_server(&server, SIGKILL);
    const double t0 = rmcheck::now_seconds();
    const rmcheck::ProcResult r = actor.finish();
    const double waited = rmcheck::now_seconds() - t0;
    const std::string transcript = rmcheck::path_join(dir, "kill.txt");
    rmcheck::write_file(transcript, r.out + "\n=== engine stderr ===\n" + r.err + "\n=== rmserve ===\n" + s.err);
    const size_t decisions = rmcheck::split_lines(decisions_of(rmlog)).size();
    std::string fatal;
    for (const std::string &line : rmcheck::split_lines(r.err))
        if (rmcheck::starts_with(line, "FATAL: ")) fatal = line;
    if (!paused || r.ok() || r.signal || fatal.empty() || !rmcheck::contains(fatal, sock.c_str()) ||
        decisions != static_cast<size_t>(KILL_AFTER)) {
        rep.error("actor", "kill: after rmserve was killed past " + std::to_string(KILL_AFTER) +
                               " requests the actor " + r.status_text() + " with " + std::to_string(decisions) +
                               " logged decisions and " + rmcheck::quoted(fatal) + " (see " + transcript + ")");
        return;
    }
    rep.info("kill: rmserve killed after " + std::to_string(KILL_AFTER) + " answers: the actor " + r.status_text() +
             " " + rmcheck::fmt1(waited) + " s later, " + std::to_string(decisions) + " decisions logged, " +
             rmcheck::quoted(fatal));
}

// serve_check.py (a Python server) plays game `g` (written to python.rmlog, python_A.rmtb).
static bool python_game(const rmcheck::Options &opt, rmcheck::Report &rep, const rmcheck::ActorSetup &a,
    const std::string &dir, const RemoteGame &g) {
    const std::string sock = rmcheck::path_join(dir, "python.sock");
    rmcheck::ServerRun server;
    std::string err;
    const std::string script = rmcheck::path_join(opt.repo, "tools/nn/serve_check.py");
    if (!rmcheck::start_server(
            opt, {a.python, script, "--socket", sock, "--policy", "0=" + a.ts()}, sock, &server, &err)) {
        rep.error("actor", "python server: " + err);
        return false;
    }
    const std::string base = "nn:remote:" + sock + ":argmax:timeout=300:evals=";
    const rmcheck::ProcResult r =
        rmcheck::run_process(game_spec(opt, a, g, base + rmcheck::path_join(dir, "python_A.rmtb"),
            base + rmcheck::path_join(dir, "python_B.rmtb"), rmcheck::path_join(dir, "python.rmlog")));
    const rmcheck::ProcResult s = rmcheck::stop_server(&server, SIGTERM);
    rmcheck::write_file(rmcheck::path_join(dir, "python_server.txt"), s.err);
    const std::string problem = game_problem(r, rmcheck::path_join(dir, "python.txt"));
    if (!problem.empty()) rep.error("actor", "python server game: " + problem);
    if (!s.ok()) rep.error("actor", "serve_check.py did not stop cleanly on SIGTERM: " + s.status_text());
    return problem.empty();
}

namespace rmcheck {

// Starts the server and waits for its "listening on" line.
bool start_server(const Options &opt, const std::vector<std::string> &argv, const std::string &socket, ServerRun *s,
    std::string *err) {
    ProcSpec spec;
    spec.argv = argv;
    spec.cwd = opt.bin_dir;
    spec.timeout_s = 1800;
    std::remove(socket.c_str());
    s->socket = socket;
    if (!s->session.start(spec, err)) return false;
    if (s->session.wait_stderr("listening on")) return true;
    *err = basename_of(argv[0]) + " did not start listening: " + quoted(trim(s->session.stderr_text()).substr(0, 300));
    return false;
}

// Signals the server and collects its output.
ProcResult stop_server(ServerRun *s, int sig) {
    s->session.send_signal(sig);
    s->session.set_timeout(STOP_TIMEOUT_S);
    return s->session.finish();
}

// The remote checks of the actor tier (docs/ci.md): in-process argmax games, the same games
// through one rmserve by concurrent actors (same decisions, evaluations within 1e-5, batches
// across clients), the handshake refusals, the kill check and the Python server's game.
void actor_remote_checks(const Options &opt, Report &rep, const ActorSetup &a) {
    const std::string rmserve = path_join(dirname_of(a.nn_binary), "rmserve");
    if (access(rmserve.c_str(), X_OK) != 0) {
        rep.error("actor", "no " + rmserve + " beside robomage_nn (`make actor` builds both)");
        return;
    }
    const std::string dir = path_join(a.dir, "remote");
    remove_tree(dir);
    make_dirs(dir);
    const std::vector<RemoteGame> games = {{"league/bug", "league/burn", opt.seed + 11},
        {"league/ur_delver", "league/uw_control", opt.seed + 12}, {"league/reanimator", "league/bw_dnt", opt.seed + 13},
        {"league/gw_maverick", "league/wrb_energy", opt.seed + 14}};
    const size_t n = games.size();
    const std::string sock = path_join(dir, "rmserve.sock");
    rmcheck::ServerRun server;
    std::string err;
    if (!rmcheck::start_server(opt,
            {rmserve, "--socket", sock, "--policy", "0=" + a.ts(), "--policy", "1=" + a.pt2(), "--version",
                std::string("1=") + POLICY1_VERSION, "--max-batch", "64", "--deadline-us", "3000"},
            sock, &server, &err)) {
        rep.error("actor", "rmserve: " + err);
        return;
    }
    const auto file = [&dir](const std::string &kind, size_t g, const char *ext) {
        return path_join(dir, kind + "_" + std::to_string(g) + ext);
    };
    std::vector<ProcResult> local(n), remote(n);
    bool python_ok = false;
    parallel_for(n + 3, opt.jobs, [&](size_t i) {
        if (i < n) {
            const std::string spec = "nn:" + a.ts() + ":argmax:evals=";
            local[i] = run_process(game_spec(opt, a, games[i], spec + file("local", i, "_A.rmtb"),
                spec + file("local", i, "_B.rmtb"), file("local", i, ".rmlog")));
        } else if (i == n) {
            check_kill(opt, rep, a, rmserve, dir, games[1]);
        } else if (i == n + 1) {
            check_handshake(opt, rep, a, sock);
        } else {
            python_ok = python_game(opt, rep, a, dir, games[0]);
        }
    });
    const double t0 = now_seconds();
    const std::string traj_dir = path_join(dir, "remote_0_traj");
    remove_tree(traj_dir);
    parallel_for(n, static_cast<int>(n), [&](size_t i) {
        const std::string base = "nn:remote:" + sock + ":argmax:stats:timeout=300";
        ProcSpec spec = game_spec(opt, a, games[i], base + ":policies=0,1:evals=" + file("remote", i, "_A.rmtb"),
            base + ":evals=" + file("remote", i, "_B.rmtb"), file("remote", i, ".rmlog"));
        if (i == 0) spec.argv.insert(spec.argv.end(), {"--trajectory-out", traj_dir});
        remote[i] = run_process(spec);
    });
    const double wall = now_seconds() - t0;
    const ProcResult s = rmcheck::stop_server(&server, SIGTERM);
    write_file(path_join(dir, "rmserve.txt"), s.err);
    if (!s.ok()) rep.error("actor", "rmserve did not stop cleanly on SIGTERM: " + s.status_text());

    size_t decisions = 0, compared = 0;
    double value = 0, logits = 0, log_pi = 0, pt2 = 0;
    bool same = true;
    for (size_t g = 0; g < n; ++g) {
        const std::string lp = game_problem(local[g], file("local", g, ".txt"));
        const std::string rp = game_problem(remote[g], file("remote", g, ".txt"));
        if (!lp.empty()) rep.error("actor", "in-process game " + std::to_string(g) + ": " + lp);
        if (!rp.empty()) rep.error("actor", "remote game " + std::to_string(g) + ": " + rp);
        if (!lp.empty() || !rp.empty()) {
            same = false;
            continue;
        }
        const std::string want = decisions_of(file("local", g, ".rmlog"));
        if (want.empty() || want != decisions_of(file("remote", g, ".rmlog"))) {
            rep.error("actor", "remote game " + std::to_string(g) + " is not the in-process game (" +
                                   file("local", g, ".rmlog") + " vs " + file("remote", g, ".rmlog") + ")");
            same = false;
            continue;
        }
        decisions += split_lines(want).size();
        for (const char *seat : {"_A.rmtb", "_B.rmtb"}) {
            const EvalDiff d = compare_evals(file("local", g, seat), file("remote", g, seat), 0);
            compared += d.decisions;
            value = std::max(value, d.value);
            logits = std::max(logits, d.logits);
            log_pi = std::max(log_pi, d.log_pi);
            if (!d.problem.empty() || d.versions != std::vector<int64_t>{0})
                rep.error("actor", "remote game " + std::to_string(g) + " " + seat + ": policy 0's evaluations " +
                                       (d.problem.empty() ? "carry another version" : d.problem));
        }
        const EvalDiff d1 = compare_evals(file("local", g, "_A.rmtb"), file("remote", g, "_A.rmtb"), 1);
        pt2 = std::max(pt2, std::max(d1.logits, d1.value));
        if (!d1.problem.empty() || d1.versions != std::vector<int64_t>{std::atoll(POLICY1_VERSION)})
            rep.error("actor",
                "remote game " + std::to_string(g) + ": policy 1 (.pt2) " +
                    (d1.problem.empty() ? "does not carry version " + std::string(POLICY1_VERSION) : d1.problem));
    }
    if (same)
        rep.info("remote: " + std::to_string(n) + " argmax bo1 games through rmserve equal the in-process ones (" +
                 std::to_string(decisions) + " decisions); over " + std::to_string(compared) +
                 " evaluations the largest differences: logits " + fmt_g(logits) + ", log pi " + fmt_g(log_pi) +
                 ", value " + fmt_g(value) + "; policy 1 (.pt2, version " + POLICY1_VERSION +
                 ") vs in-process TorchScript " + fmt_g(pt2));
    check_remote_trajectories(rep, traj_dir, file("remote", 0, "_A.rmtb"), file("remote", 0, "_B.rmtb"));
    const std::string batches = stat_field(s.err, "RMSERVE-STATS:", "batches"),
                      largest = stat_field(s.err, "RMSERVE-STATS:", "max_batch");
    if (std::atol(largest.c_str()) < 2)
        rep.error("actor", "rmserve never batched across clients (largest batch " + quoted(largest) + ", see " +
                               path_join(dir, "rmserve.txt") + ")");
    else
        rep.info("batching: " + std::to_string(n) + " concurrent actors (" + std::to_string(2 * n) +
                 " clients) against one rmserve: " + std::to_string(decisions) + " decisions in " + fmt1(wall) +
                 " s (" + fmt1(static_cast<double>(decisions) / wall) + " decisions/s), " + batches +
                 " batches, mean " + stat_field(s.err, "RMSERVE-STATS:", "mean_batch") + ", largest " + largest +
                 ", server compute " + stat_field(s.err, "RMSERVE-STATS:", "compute_s") + " s");
    if (!python_ok || !local[0].ok()) return;
    const std::string want = decisions_of(file("local", 0, ".rmlog"));
    const EvalDiff d = compare_evals(file("local", 0, "_A.rmtb"), path_join(dir, "python_A.rmtb"), 0);
    if (want.empty() || want != decisions_of(path_join(dir, "python.rmlog")) || !d.problem.empty())
        rep.error("actor", "the Python server's game is not the in-process game 0 " +
                               (d.problem.empty() ? "(" + path_join(dir, "python.rmlog") + ")" : d.problem));
    else
        rep.info("python server: tools/nn/serve_check.py plays in-process game 0 decision for decision (" +
                 std::to_string(split_lines(want).size()) + " decisions; seat A's logits within " + fmt_g(d.logits) +
                 ", values within " + fmt_g(d.value) + ")");
}

}  // namespace rmcheck
