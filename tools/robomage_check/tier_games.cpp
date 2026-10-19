#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "actor.h"
#include "embed.h"
#include "gate.h"
#include "proc.h"
#include "scan.h"
#include "util.h"

// One live game (bo1) or match (bo3) of a fuzz / smoke matchup.
struct GameJob {
        std::string deck_a, deck_b;
        long seed = 0;
        bool bo3 = false;
        bool no_auto_mana = false;  // --no-auto-mana: mana paid source by source
        std::string mode;
        rmcheck::ProcResult result;
        std::string transcript;  // what the per-matchup file gets for this game
        int a_wins = 0, b_wins = 0, draws = 0;
        bool finished = false;  // a result for every game, no crash
};

// Runs of each bench game and stage; each figure is the fastest (process timings on a shared machine
// vary by 10-20% from run to run).
const int BENCH_REPS = 2;
// The training plan's target for the actor loop (engine, knowledge, featurizer; T7), in decisions/s
// per core: an optimised build (bin/release/, bin/pgo/) below it gets a warning, never an error.
const double BENCH_TARGET = 1500;

// One stage of the bench tier: its workload's totals.
struct BenchStage {
        std::string label;
        std::vector<std::string> flags;
        size_t decisions = 0;
        double wall = 0;
        double start = 0;  // the same processes' start alone (--max-decisions 0)
        size_t processes = 0;
        size_t featurized = 0;  // --featurize-seats: decisions featurized and the time it took
        double featurize_s = 0;
        // Microseconds of wall clock per decision, process start included.
        double per_decision() const { return decisions ? wall * 1e6 / static_cast<double>(decisions) : 0; }
        // The same with the process start left out.
        double net_per_decision() const {
            return decisions ? (wall - start) * 1e6 / static_cast<double>(decisions) : 0;
        }
        // Milliseconds of one process start.
        double start_ms() const { return processes ? start * 1e3 / static_cast<double>(processes) : 0; }
};

// The nn: agents' NN-STATS lines of a stage, summed.
struct NnTotals {
        size_t decisions = 0;
        double featurize_s = 0, evaluate_s = 0, server_s = 0;
        std::vector<std::string> logs;  // the stage's decision logs, in game order
};

static void count_results(GameJob *job);
static std::string stamp();
static void report_matchups(const rmcheck::Options &opt, rmcheck::Report &rep, const char *tier, const char *leg,
    const std::vector<std::pair<std::string, std::string>> &pairs, std::vector<GameJob> &jobs, int per_matchup);
static std::string job_tags(const GameJob &job);
static std::vector<std::string> engine_game_args(const GameJob &job);
static void report_pycheck(const rmcheck::Options &opt, rmcheck::Report &rep, const rmcheck::ProcResult &r);
static size_t bench_decisions(const std::string &rmlog);
static bool bench_run(const rmcheck::Options &opt, rmcheck::Report &rep, const std::vector<std::string> &argv,
    const std::string &rmlog, BenchStage *stage);
static double bench_start(const rmcheck::Options &opt, std::vector<std::string> argv);
static std::string bench_pad(const std::string &label);
static bool bench_nn_stage(const rmcheck::Options &opt, rmcheck::Report &rep, const rmcheck::ActorSetup &a,
    const std::vector<std::pair<std::string, std::string>> &pairs, const std::string &dir, const std::string &nn,
    const std::string &tag, BenchStage *stage, NnTotals *totals);
static void bench_evaluation(const rmcheck::Options &opt, rmcheck::Report &rep,
    const std::vector<std::pair<std::string, std::string>> &pairs, const std::string &dir);

// The games' winners from the engine's GAME_RESULT / MATCH_RESULT lines. A bo1 game is drawn
// when its GAME_RESULT says so, and a game with no result (a stall) counts as a draw too; a bo3
// match counts its drawn games.
static void count_results(GameJob *job) {
    int results = 0;
    bool match_result = false;
    for (const std::string &line : rmcheck::split_lines(job->result.out)) {
        if (rmcheck::starts_with(line, "GAME_RESULT: ")) {
            ++results;
            if (rmcheck::contains(line, "Player A wins"))
                job->a_wins += job->bo3 ? 0 : 1;
            else if (rmcheck::contains(line, "Player B wins"))
                job->b_wins += job->bo3 ? 0 : 1;
            else
                ++job->draws;
        } else if (rmcheck::starts_with(line, "MATCH_RESULT: ")) {
            match_result = true;
            if (rmcheck::contains(line, "Player A wins"))
                ++job->a_wins;
            else if (rmcheck::contains(line, "Player B wins"))
                ++job->b_wins;
        }
    }
    job->finished = job->result.ok() && results > 0 && (!job->bo3 || match_result);
    if (job->result.ok() && !job->finished) ++job->draws;  // a stall: no result
}

static std::string stamp() {
    char buf[32];
    const std::time_t t = std::time(nullptr);
    std::strftime(buf, sizeof buf, "%Y%m%d_%H%M%S", std::localtime(&t));
    return buf;
}

static std::vector<std::string> engine_game_args(const GameJob &job) {
    std::vector<std::string> a = {"--agent-a", job.mode, "--agent-b", job.mode, "--game-seed", std::to_string(job.seed),
        "--deck-a", job.deck_a, "--deck-b", job.deck_b, "--narrative", "--trace-decisions"};
    if (job.bo3) a.push_back("--bo3");
    if (job.no_auto_mana) a.push_back("--no-auto-mana");
    return a;
}

// A game's agent and format as its report lines show them ("explore:patient bo3").
static std::string job_tags(const GameJob &job) {
    return job.mode + (job.bo3 ? " bo3" : "") + (job.no_auto_mana ? " no-auto-mana" : "");
}

// Writes each matchup's transcript (<out>/<leg>_<a>__<b>.txt) and reports, under the tier, its
// crashes, missing results, error and warning lines and draws (a drawn game's transcript also saved
// as <leg>_<a>__<b>_g<i>_draw_<stamp>.txt). jobs[k * per_matchup + i] is game i of matchup k.
static void report_matchups(const rmcheck::Options &opt, rmcheck::Report &rep, const char *tier, const char *leg,
    const std::vector<std::pair<std::string, std::string>> &pairs, std::vector<GameJob> &jobs, int per_matchup) {
    const std::string out_dir = opt.out("");
    for (size_t k = 0; k < pairs.size(); ++k) {
        const std::string stem =
            std::string(leg) + "_" + rmcheck::deck_stem(pairs[k].first) + "__" + rmcheck::deck_stem(pairs[k].second);
        const std::string path = rmcheck::path_join(out_dir, stem + ".txt");
        std::string file;
        int w = 0, l = 0, d = 0;
        const GameJob *first = per_matchup ? &jobs[k * static_cast<size_t>(per_matchup)] : nullptr;
        const std::string label = rmcheck::deck_stem(pairs[k].first) + " vs " + rmcheck::deck_stem(pairs[k].second) +
                                  " [" + (first ? job_tags(*first) : opt.mode) + "] seed " +
                                  std::to_string(first ? first->seed : 0);
        size_t crashes = 0;
        for (int i = 0; i < per_matchup; ++i) {
            GameJob &job = jobs[k * static_cast<size_t>(per_matchup) + static_cast<size_t>(i)];
            file += "--- Game " + std::to_string(i + 1) + "/" + std::to_string(per_matchup) + " [" + job_tags(job) +
                    "] seed " + std::to_string(job.seed) + " ---\n" + job.transcript + "\n";
            if (!job.result.ok()) {
                ++crashes;
                file += "=== game " + std::to_string(i + 1) + "/" + std::to_string(per_matchup) + ": ENGINE CRASH — " +
                        job.result.status_text() + " ===\n";
                rep.error(tier, label + ": game " + std::to_string(i + 1) + " (seed " + std::to_string(job.seed) +
                                    ") engine crashed — " + job.result.status_text() + " (see " + path + ")");
                continue;
            }
            w += job.a_wins;
            l += job.b_wins;
            d += job.draws;
            if (job.draws) {
                const std::string log =
                    rmcheck::path_join(out_dir, stem + "_g" + std::to_string(i + 1) + "_draw_" + stamp() + ".txt");
                rmcheck::write_file(log, job.transcript);
                file += "=== " +
                        std::string(job.finished ? "drawn game(s), both players lost at once (CR 104.4a)"
                                                 : "no result (stall)") +
                        " — review; full log saved to " + log + " ===\n";
            }
        }
        file += "\n" + std::to_string(w) + "W / " + std::to_string(l) + "L / " + std::to_string(d) + "D over " +
                std::to_string(per_matchup) + " games\n";
        rmcheck::write_file(path, file);
        if (crashes) continue;
        if (d > 0) rep.warn(tier, label + ": " + std::to_string(d) + " draw(s) — review " + path);
        std::vector<rmcheck::ScanLine> errs, warns;
        rmcheck::scan_transcript(file, &errs, &warns);
        if (!errs.empty())
            rep.error(tier, label + ": " + std::to_string(errs.size()) + " error line(s) in transcript, first at " +
                                path + ":" + std::to_string(errs[0].lineno) + ": " + rmcheck::quoted(errs[0].line));
        if (!warns.empty())
            rep.warn(tier, label + ": " + std::to_string(warns.size()) + " engine WARNING line(s) — review " + path +
                               " (e.g. " + path + ":" + std::to_string(warns[0].lineno) + ": " +
                               rmcheck::quoted(warns[0].line) + ")");
        rep.info(label + ": " + std::to_string(w) + "W / " + std::to_string(l) + "L / " + std::to_string(d) + "D -> " +
                 path);
    }
}

// The kept Python's own checks (python/pycheck.py: its decoder against the engine's, the
// decoders' refusals, the driver's seeds and imports), run beside the smoke games.
static void report_pycheck(const rmcheck::Options &opt, rmcheck::Report &rep, const rmcheck::ProcResult &r) {
    const std::string path = rmcheck::path_join(opt.out(""), "smoke_pycheck.txt");
    rmcheck::write_file(path, r.out);
    size_t fails = 0;
    for (const std::string &line : rmcheck::split_lines(r.out)) {
        const std::string t = rmcheck::trim(line);
        if (rmcheck::starts_with(t, "FAIL: ")) {
            ++fails;
            rep.error("smoke", "pycheck: " + t.substr(6));
        } else if (rmcheck::starts_with(t, "OK   ")) {
            rep.info("pycheck: " + t.substr(5));
        }
    }
    if (!r.ok() && fails == 0)
        rep.error("smoke", "pycheck: python/pycheck.py failed (" + r.status_text() + ", see " + path + ")");
    std::vector<rmcheck::ScanLine> errs, warns;
    rmcheck::scan_transcript(r.out, &errs, &warns);
    if (!errs.empty())
        rep.error("smoke", "pycheck: " + std::to_string(errs.size()) + " error line(s), first at " + path + ":" +
                               std::to_string(errs[0].lineno) + ": " + rmcheck::quoted(errs[0].line));
    if (!warns.empty())
        rep.warn("smoke", "pycheck: " + std::to_string(warns.size()) + " engine WARNING line(s) — review " + path);
}

// A decision log's decisions (its lines after DECISIONS).
static size_t bench_decisions(const std::string &rmlog) {
    std::string text;
    size_t n = 0;
    rmcheck::read_file(rmlog, &text);
    const size_t at = text.find("DECISIONS\n");
    if (at != std::string::npos)
        for (const std::string &line : rmcheck::split_lines(text.substr(at + 10)))
            n += rmcheck::trim(line).empty() ? 0 : 1;
    return n;
}

// Runs one bench game into `stage` (its decisions are the decision log's), BENCH_REPS times, its
// wall clock and its start each the fastest of the runs; false after an error.
static bool bench_run(const rmcheck::Options &opt, rmcheck::Report &rep, const std::vector<std::string> &argv,
    const std::string &rmlog, BenchStage *stage) {
    rmcheck::ProcSpec spec;
    spec.argv = argv;
    spec.cwd = opt.bin_dir;
    spec.timeout_s = 1800;
    rmcheck::ProcResult r;
    double wall = 0, start = 0;
    for (int i = 0; i < BENCH_REPS; ++i) {
        r = rmcheck::run_process(spec);
        if (!r.ok()) {
            rep.error("bench", stage->label + ": " + rmcheck::join(argv, " ") + ": " + r.status_text());
            return false;
        }
        const double s = bench_start(opt, argv);
        wall = i ? std::min(wall, r.seconds) : r.seconds;
        start = i ? std::min(start, s) : s;
    }
    stage->decisions += bench_decisions(rmlog);
    stage->wall += wall;
    stage->start += start;
    ++stage->processes;
    for (const std::string &line : rmcheck::split_lines(r.out + "\n" + r.err)) {
        size_t n = 0;
        double sec = 0;
        if (std::sscanf(line.c_str(), "FEATURIZE-SEATS: %zu decisions in %lf s", &n, &sec) == 2) {
            stage->featurized += n;
            stage->featurize_s += sec;
        }
    }
    return true;
}

// The wall clock of `argv` stopped before its first decision: the process start, the match setup
// and any input built before play. A live run is the same command under --max-decisions 0, its
// decision log written to <log>.start; a replay (whose cap the log sets) replays the <log>.start of
// the live run that wrote its log.
static double bench_start(const rmcheck::Options &opt, std::vector<std::string> argv) {
    bool replay = false;
    for (size_t i = 0; i + 1 < argv.size(); ++i)
        if (argv[i] == "--decision-log" || argv[i] == "--replay") {
            replay = replay || argv[i] == "--replay";
            argv[i + 1] += ".start";
        }
    if (!replay) argv.insert(argv.end(), {"--max-decisions", "0"});
    rmcheck::ProcSpec spec;
    spec.argv = argv;
    spec.cwd = opt.bin_dir;
    return rmcheck::run_process(spec).seconds;
}

// A stage label padded for the bench table.
static std::string bench_pad(const std::string &label) {
    return label + std::string(label.size() < 36 ? 36 - label.size() : 1, ' ');
}

// One nn: stage of the bench: the agent spec `nn` on both seats of the workload's pairings and
// seeds through the actor binary, the decision logs <dir>/<tag><k>_<s>.rmlog; false after an error.
static bool bench_nn_stage(const rmcheck::Options &opt, rmcheck::Report &rep, const rmcheck::ActorSetup &a,
    const std::vector<std::pair<std::string, std::string>> &pairs, const std::string &dir, const std::string &nn,
    const std::string &tag, BenchStage *stage, NnTotals *totals) {
    for (size_t k = 0; k < pairs.size(); ++k)
        for (long s = 1; s <= 3; ++s) {
            const std::string log =
                rmcheck::path_join(dir, tag + std::to_string(k) + "_" + std::to_string(s) + ".rmlog");
            rmcheck::ProcSpec spec;
            spec.argv = {a.nn_binary, "--agent-a", nn, "--agent-b", nn, "--game-seed", std::to_string(s), "--deck-a",
                pairs[k].first, "--deck-b", pairs[k].second, "--decision-log", log};
            spec.cwd = opt.bin_dir;
            spec.timeout_s = 1800;
            const rmcheck::ProcResult r = rmcheck::run_process(spec);
            if (!r.ok()) {
                rep.error("bench", "evaluation: " + rmcheck::join(spec.argv, " ") + ": " + r.status_text());
                return false;
            }
            stage->decisions += bench_decisions(log);
            stage->wall += r.seconds;
            stage->start += bench_start(opt, spec.argv);
            ++stage->processes;
            totals->logs.push_back(log);
            for (const std::string &line : rmcheck::split_lines(r.out + "\n" + r.err)) {
                char seat = 0;
                size_t n = 0;
                double fs = 0, es = 0;
                if (std::sscanf(line.c_str(), "NN-STATS: seat %c decisions=%zu featurize_s=%lf evaluate_s=%lf", &seat,
                        &n, &fs, &es) != 4)
                    continue;
                totals->decisions += n;
                totals->featurize_s += fs;
                totals->evaluate_s += es;
                const size_t at = line.find(" server_s=");
                if (at != std::string::npos) totals->server_s += std::atof(line.c_str() + at + 10);
            }
        }
    return true;
}

// The evaluation stages: nn: (the reference policy, TorchScript, sampled) on both seats of the
// workload's pairings and seeds through the actor binary, in process, then through rmserve (one
// actor at a time, no batching deadline: the round trip's own cost); skipped without them
// (actor.h).
static void bench_evaluation(const rmcheck::Options &opt, rmcheck::Report &rep,
    const std::vector<std::pair<std::string, std::string>> &pairs, const std::string &dir) {
    rmcheck::ActorSetup a;
    std::string skip;
    if (!rmcheck::actor_setup(opt, rep, "bench", &a, &skip)) {
        if (!skip.empty()) rep.info("  " + bench_pad("evaluation") + "skipped: " + skip);
        return;
    }
    BenchStage stage;
    stage.label = "evaluation (robomage_nn)";
    NnTotals local;
    if (!bench_nn_stage(opt, rep, a, pairs, dir, "nn:" + a.ts() + ":stats", "nn", &stage, &local)) return;
    const double per = static_cast<double>(local.decisions ? local.decisions : 1);
    rep.info("  " + bench_pad(stage.label) + rmcheck::fmt1(stage.net_per_decision()) + " us, start " +
             rmcheck::fmt1(stage.start_ms()) + " ms (nn: vs nn:, " + std::to_string(stage.decisions) +
             " decisions of other games; in-process featurize " + rmcheck::fmt1(local.featurize_s * 1e6 / per) +
             " us + evaluate " + rmcheck::fmt1(local.evaluate_s * 1e6 / per) + " us per decision, " + a.ts() + ")");
    const std::string rmserve = rmcheck::path_join(rmcheck::dirname_of(a.nn_binary), "rmserve");
    const std::string sock = rmcheck::path_join(dir, "rmserve.sock");
    rmcheck::ServerRun server;
    if (!rmcheck::start_server(
            opt, {rmserve, "--socket", sock, "--policy", "0=" + a.ts(), "--deadline-us", "0"}, sock, &server, &skip)) {
        rep.error("bench", "evaluation: rmserve: " + skip);
        return;
    }
    BenchStage remote;
    remote.label = "evaluation remote (rmserve)";
    NnTotals totals;
    const bool ran = bench_nn_stage(
        opt, rep, a, pairs, dir, "nn:remote:" + sock + ":stats:timeout=300", "nn_remote", &remote, &totals);
    rmcheck::stop_server(&server, SIGTERM);
    if (!ran) return;
    bool same = local.logs.size() == totals.logs.size();
    for (size_t i = 0; same && i < local.logs.size(); ++i) {
        std::string x, y;
        same = rmcheck::read_file(local.logs[i], &x) && rmcheck::read_file(totals.logs[i], &y) &&
               x.substr(x.find("DECISIONS")) == y.substr(y.find("DECISIONS"));
    }
    const double rper = static_cast<double>(totals.decisions ? totals.decisions : 1);
    rep.info("  " + bench_pad(remote.label) + rmcheck::fmt1(remote.net_per_decision()) + " us, start " +
             rmcheck::fmt1(remote.start_ms()) + " ms (" + (same ? "the same games" : "other games") +
             "; per decision the client's evaluate " + rmcheck::fmt1(totals.evaluate_s * 1e6 / rper) +
             " us, of it the server's compute " + rmcheck::fmt1(totals.server_s * 1e6 / rper) + " us, round trip " +
             rmcheck::fmt1((totals.evaluate_s - totals.server_s) * 1e6 / rper) + " us)");
}

namespace rmcheck {

// The fuzz tier: C++ explore vs explore in-process (--agent-a/-b <mode>, the Python explore tier's
// games for the same seeds), bo1 over the matchups (default the league mirrors, --fuzz-games per
// matchup; matchup k's game i has game seed seed + 1000k + i), then one bo3 explore:patient match
// per ring matchup (seed + 500000 + k), then --manual-mana-games bo1 <mode> games per ring matchup
// under --no-auto-mana (ring matchup k's game i: seed + 700000 + 1000k + i).
void tier_fuzz(const Options &opt, Report &rep) {
    const std::vector<std::pair<std::string, std::string>> pairs =
        matchups(opt, opt.matchups.empty() ? "mirrors" : opt.matchups);
    const std::vector<std::pair<std::string, std::string>> ring = matchups(opt, "ring");
    std::vector<GameJob> bo1, bo3, manual;
    for (size_t k = 0; k < pairs.size(); ++k)
        for (int i = 0; i < opt.fuzz_games; ++i) {
            GameJob j;
            j.deck_a = pairs[k].first;
            j.deck_b = pairs[k].second;
            j.seed = opt.seed + 1000 * static_cast<long>(k) + i;
            j.mode = opt.mode;
            bo1.push_back(j);
        }
    for (size_t k = 0; k < ring.size(); ++k) {
        GameJob j;
        j.deck_a = ring[k].first;
        j.deck_b = ring[k].second;
        j.seed = opt.seed + 500000 + static_cast<long>(k);
        j.mode = "explore:patient";
        j.bo3 = true;
        bo3.push_back(j);
    }
    for (size_t k = 0; k < ring.size(); ++k)
        for (int i = 0; i < opt.manual_mana_games; ++i) {
            GameJob j;
            j.deck_a = ring[k].first;
            j.deck_b = ring[k].second;
            j.seed = opt.seed + 700000 + 1000 * static_cast<long>(k) + i;
            j.mode = opt.mode;
            j.no_auto_mana = true;
            manual.push_back(j);
        }
    std::vector<GameJob *> all;
    for (GameJob &j : bo1) all.push_back(&j);
    for (GameJob &j : bo3) all.push_back(&j);
    for (GameJob &j : manual) all.push_back(&j);
    parallel_for(all.size(), opt.jobs, [&](size_t i) {
        GameJob &job = *all[i];
        ProcSpec spec;
        spec.argv = {opt.binary};
        for (const std::string &a : engine_game_args(job)) spec.argv.push_back(a);
        spec.cwd = opt.bin_dir;
        spec.timeout_s = 1800;
        job.result = run_process(spec);
        job.transcript =
            job.result.out + (trim(job.result.err).empty() ? "" : "\n=== engine stderr ===\n" + job.result.err);
        count_results(&job);
    });
    report_matchups(opt, rep, "fuzz", "fuzz", pairs, bo1, opt.fuzz_games);
    report_matchups(opt, rep, "fuzz", "fuzz_bo3", ring, bo3, 1);
    if (opt.manual_mana_games > 0)
        report_matchups(opt, rep, "fuzz", "fuzz_manual", ring, manual, opt.manual_mana_games);
    size_t decisions = 0;
    double cpu = 0;
    for (const GameJob *j : all) {
        for (const std::string &line : split_lines(j->result.out)) decisions += starts_with(line, "| >> ") ? 1 : 0;
        cpu += j->result.seconds;
    }
    rep.info("fuzz: " + std::to_string(all.size()) + " games/matches, " + std::to_string(decisions) + " decisions, " +
             fmt1(cpu) + " engine-seconds");
}

// The bench tier (docs/engine_fork_plan.md §5d, throughput accounting): a fixed workload — explore
// vs explore, bo1, three league pairings × three games (game seeds 1-3), one engine at a time (so
// the figures do not depend on -j) — per stage, each stage the previous one plus a part of the
// actor loop, so the differences per decision are the parts' costs: the engine alone (--replay of
// the games' decision logs), + the frames encoded and decoded for the in-process agents with
// knowledge off, + the knowledge sidecar (expert, the default), + the featurizer
// (--featurize-seats). Each game runs BENCH_REPS times per stage and its fastest run counts. Each
// figure is wall clock per decision with the process start (the same command stopped by
// --max-decisions 0: start, match setup, inputs built before play) left out and reported beside
// it; the stages must play the same games (the same decision logs). The headline decisions/s is
// the knowledge stage's, process start included. Then, when the actor binary and the reference
// policy are available (actor.h), the evaluation stage: the same pairings and seeds with nn: on
// both seats (different games, one run each), wall clock per decision (start left out) and the
// agents' own featurize and evaluate time per decision (NN-STATS). The decision count is each
// game's decision log's length.
void tier_bench(const Options &opt, Report &rep) {
    const std::vector<std::pair<std::string, std::string>> pairs = {{"league/ur_delver", "league/gw_maverick"},
        {"league/bug", "league/ur_delver"}, {"league/reanimator", "league/burn"}};
    const std::string dir = make_temp_dir(opt.out("bench"), "run_");
    std::vector<BenchStage> stages(4);
    stages[0].label = "engine alone (--replay)";
    stages[1].label = "+ frames, knowledge off";
    stages[1].flags = {"--knowledge-a", "off", "--knowledge-b", "off"};
    stages[2].label = "+ knowledge sidecar (expert)";
    stages[3].label = "+ featurizer (--featurize-seats)";
    stages[3].flags = {"--featurize-seats"};
    size_t games = 0;
    for (size_t k = 0; k < pairs.size(); ++k)
        for (long s = 1; s <= 3; ++s) {
            const std::string tag = std::to_string(k) + "_" + std::to_string(s);
            // The live stages first (the knowledge stage's log is the one the replay stage replays).
            for (size_t st = 1; st < stages.size(); ++st) {
                const std::string log = path_join(dir, "g" + tag + "_" + std::to_string(st) + ".rmlog");
                std::vector<std::string> args = {opt.binary, "--agent-a", "explore", "--agent-b", "explore",
                    "--game-seed", std::to_string(s), "--deck-a", pairs[k].first, "--deck-b", pairs[k].second,
                    "--decision-log", log};
                args.insert(args.end(), stages[st].flags.begin(), stages[st].flags.end());
                if (!bench_run(opt, rep, args, log, &stages[st])) return;
            }
            const std::string log = path_join(dir, "g" + tag + "_2.rmlog");
            if (!bench_run(opt, rep, {opt.binary, "--replay", log}, log, &stages[0])) return;
            for (size_t st = 1; st < stages.size(); ++st)
                if (bench_decisions(path_join(dir, "g" + tag + "_" + std::to_string(st) + ".rmlog")) !=
                    bench_decisions(log))
                    rep.error("bench", "stage '" + stages[st].label +
                                           "' played another game than the knowledge stage (" + pairs[k].first +
                                           " vs " + pairs[k].second + " seed " + std::to_string(s) + ")");
            ++games;
        }
    const double base = stages[2].per_decision();
    rep.info("bench: " + std::to_string(games) + " explore games, " + std::to_string(stages[2].decisions) +
             " decisions in " + fmt1(stages[2].wall) +
             " s: " + std::to_string(static_cast<long>(base > 0 ? 1e6 / base : 0)) + " decisions/s, " +
             fmt1(static_cast<double>(games) / (stages[2].wall > 0 ? stages[2].wall : 1)) + " games/s (" + opt.binary +
             ")");
    rep.info(
        "bench: wall clock per decision, process start (each process again with --max-decisions 0) left out; "
        "the stages play the same games:");
    for (size_t st = 0; st < 3; ++st) {
        const double us = stages[st].net_per_decision();
        const double delta = st ? us - stages[st - 1].net_per_decision() : us;
        rep.info("  " + bench_pad(stages[st].label) + fmt1(us) + " us (" + (st && delta >= 0 ? "+" : "") + fmt1(delta) +
                 " us), start " + fmt1(stages[st].start_ms()) + " ms");
    }
    // The featurizer's cost is its in-process time: building its inputs at start parses every vocab
    // card, which the other stages parse during play (the first CARDS frames), so the stage's own
    // wall clock without its start understates it.
    const double feat =
        stages[3].featurize_s * 1e6 / static_cast<double>(stages[3].decisions ? stages[3].decisions : 1);
    const double loop_us = stages[2].net_per_decision() + feat;
    rep.info("  " + bench_pad(stages[3].label) + fmt1(loop_us) + " us (+" + fmt1(feat) + " us in process over " +
             std::to_string(stages[3].featurized) + " featurized decisions), start " + fmt1(stages[3].start_ms()) +
             " ms (its inputs built); the stage's wall clock " + fmt1(stages[3].net_per_decision()) + " us");
    const double loop_rate = loop_us > 0 ? 1e6 / loop_us : 0;
    const bool optimised =
        opt.binary.find("/release/") != std::string::npos || opt.binary.find("/pgo/") != std::string::npos;
    const double game_ms =
        games ? loop_us * static_cast<double>(stages[2].decisions) / static_cast<double>(games) / 1e3 : 0;
    rep.info("bench: the actor loop (engine, knowledge, featurizer) " + std::to_string(static_cast<long>(loop_rate)) +
             " decisions/s, " + fmt1(game_ms) + " ms a game; target " +
             std::to_string(static_cast<long>(BENCH_TARGET)) + " decisions/s (an optimised build)");
    if (optimised && loop_rate < BENCH_TARGET)
        rep.warn("bench", "the actor loop runs " + std::to_string(static_cast<long>(loop_rate)) +
                              " decisions/s, below the target " + std::to_string(static_cast<long>(BENCH_TARGET)) +
                              " (a loaded machine, or a regression: docs/ci.md bench)");
    bench_evaluation(opt, rep, pairs, dir);
    remove_tree(dir);
}

// The smoke tier: scripted hard vs hard (the Python agent: python/rm_driver.py, one process per
// matchup) over the league mirrors and the ring (--smoke-games per matchup, game i of matchup k
// with seed seed + 1000k + i), and beside them the kept Python's own checks (python/pycheck.py; its rmtb
// part reads the RMTB samples written here, embed.h).
// Skipped with a notice when no Python interpreter is found.
void tier_smoke(const Options &opt, Report &rep) {
    if (opt.python.empty()) {
        rep.info("smoke: skipped (no python3 interpreter on PATH)");
        return;
    }
    const std::vector<std::pair<std::string, std::string>> pairs =
        matchups(opt, opt.matchups.empty() ? "mirrors+ring" : opt.matchups);
    const std::string out_dir = opt.out("");
    std::vector<ProcResult> results(pairs.size());
    std::vector<std::string> work(pairs.size());
    const std::string rmtb_samples = write_rmtb_samples(opt, rep);
    ProcResult pycheck;
    // Job 0 is the Python checks (the longest job), job k + 1 matchup k.
    parallel_for(pairs.size() + 1, opt.jobs, [&](size_t job) {
        ProcSpec spec;
        if (job == 0) {
            spec.argv = {opt.python, opt.repo + "/python/pycheck.py", "--binary", opt.binary, "--out-dir", out_dir};
            if (!rmtb_samples.empty()) spec.argv.insert(spec.argv.end(), {"--rmtb-samples", rmtb_samples});
            spec.cwd = opt.repo;
            spec.timeout_s = 3600;
            spec.merge_stderr = true;
            pycheck = run_process(spec);
            return;
        }
        const size_t k = job - 1;
        work[k] = make_temp_dir(out_dir, ".smoke_");
        spec.argv = {opt.python, opt.repo + "/python/rm_driver.py", "--player-a", "scripted", "--player-b", "scripted",
            "--deck-a", pairs[k].first, "--deck-b", pairs[k].second, "--format", "bo1", "--games",
            std::to_string(opt.smoke_games), "--seed", std::to_string(opt.seed + 1000 * static_cast<long>(k)),
            "--verbose", "--binary", opt.binary};
        spec.cwd = work[k];
        spec.timeout_s = 3600;
        spec.merge_stderr = true;
        results[k] = run_process(spec);
    });
    for (size_t k = 0; k < pairs.size(); ++k) {
        const std::string stem = "smoke_" + deck_stem(pairs[k].first) + "__" + deck_stem(pairs[k].second);
        const std::string path = path_join(out_dir, stem + ".txt");
        const ProcResult &r = results[k];
        write_file(path, r.out);
        const std::string label = deck_stem(pairs[k].first) + " vs " + deck_stem(pairs[k].second) +
                                  " [scripted] seed " + std::to_string(opt.seed + 1000 * static_cast<long>(k));
        for (const std::string &draw : list_dir(work[k], ".txt")) {
            std::string text;
            read_file(path_join(work[k], draw), &text);
            write_file(path_join(out_dir, stem + "_" + draw), text);
        }
        remove_tree(work[k]);
        if (!r.ok()) {
            rep.error("smoke", label + ": the driver failed (" + r.status_text() + ", see " + path + ")");
            continue;
        }
        // The driver's results: a W/L/D line for several games, a "=== <winner> wins ===" or
        // "=== DRAW ===" line for one; a draw (or a drawn game in a match) also names its saved log.
        std::string summary;
        int draws = 0, results = 0;
        for (const std::string &line : split_lines(r.out)) {
            const std::string t = trim(line);
            if (contains(t, "full log saved to")) ++draws;
            if (opt.smoke_games > 1 && contains(t, "W / ") && contains(t, " over ")) {
                summary = t;
                long w = 0, l = 0, d = 0;
                if (std::sscanf(t.c_str(), "%ldW / %ldL / %ldD", &w, &l, &d) == 3)
                    results = static_cast<int>(w + l + d);
            } else if (opt.smoke_games == 1 && starts_with(t, "=== ") &&
                       (contains(t, " wins ===") || t == "=== DRAW ===")) {
                summary = t;
                results = 1;
            }
        }
        if (results < opt.smoke_games)
            rep.error("smoke", label + ": only " + std::to_string(results) + "/" + std::to_string(opt.smoke_games) +
                                   " games completed (see " + path + ")");
        if (draws > 0) rep.warn("smoke", label + ": " + std::to_string(draws) + " draw(s) — review " + path);
        std::vector<ScanLine> errs, warns;
        scan_transcript(r.out, &errs, &warns);
        if (!errs.empty())
            rep.error("smoke", label + ": " + std::to_string(errs.size()) + " error line(s) in transcript, first at " +
                                   path + ":" + std::to_string(errs[0].lineno) + ": " + quoted(errs[0].line));
        if (!warns.empty())
            rep.warn("smoke", label + ": " + std::to_string(warns.size()) + " engine WARNING line(s) — review " + path);
        rep.info(label + ": " + summary + " -> " + path);
    }
    report_pycheck(opt, rep, pycheck);
}

}  // namespace rmcheck
