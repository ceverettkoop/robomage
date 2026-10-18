#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "gate.h"
#include "proc.h"
#include "scan.h"
#include "util.h"

// One live game (bo1) or match (bo3) of a fuzz / smoke matchup.
struct GameJob {
        std::string deck_a, deck_b;
        long seed = 0;
        bool bo3 = false;
        std::string mode;
        rmcheck::ProcResult result;
        std::string transcript;  // what the per-matchup file gets for this game
        int a_wins = 0, b_wins = 0, draws = 0;
        bool finished = false;  // a result for every game, no crash
};

static void count_results(GameJob *job);
static std::string stamp();
static void report_matchups(const rmcheck::Options &opt, rmcheck::Report &rep, const char *tier,
    const std::vector<std::pair<std::string, std::string>> &pairs, std::vector<GameJob> &jobs, int per_matchup);
static std::vector<std::string> engine_game_args(const GameJob &job);
static void report_pycheck(const rmcheck::Options &opt, rmcheck::Report &rep, const rmcheck::ProcResult &r);

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
    return a;
}

// Writes each matchup's transcript (<out>/<tier>_<a>__<b>.txt) and reports its crashes, missing
// results, error and warning lines and draws (a drawn game's transcript also saved as
// <tier>_<a>__<b>_g<i>_draw_<stamp>.txt). jobs[k * per_matchup + i] is game i of matchup k.
static void report_matchups(const rmcheck::Options &opt, rmcheck::Report &rep, const char *tier,
    const std::vector<std::pair<std::string, std::string>> &pairs, std::vector<GameJob> &jobs, int per_matchup) {
    const std::string out_dir = opt.out("");
    for (size_t k = 0; k < pairs.size(); ++k) {
        const std::string stem =
            std::string(tier) + "_" + rmcheck::deck_stem(pairs[k].first) + "__" + rmcheck::deck_stem(pairs[k].second);
        const std::string path = rmcheck::path_join(out_dir, stem + ".txt");
        std::string file;
        int w = 0, l = 0, d = 0;
        const GameJob *first = per_matchup ? &jobs[k * static_cast<size_t>(per_matchup)] : nullptr;
        const std::string label = rmcheck::deck_stem(pairs[k].first) + " vs " + rmcheck::deck_stem(pairs[k].second) +
                                  " [" + (first ? first->mode : opt.mode) + (first && first->bo3 ? " bo3" : "") +
                                  "] seed " + std::to_string(first ? first->seed : 0);
        size_t crashes = 0;
        for (int i = 0; i < per_matchup; ++i) {
            GameJob &job = jobs[k * static_cast<size_t>(per_matchup) + static_cast<size_t>(i)];
            file += "--- Game " + std::to_string(i + 1) + "/" + std::to_string(per_matchup) + " [" + job.mode +
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

// The kept Python's own checks (train/pycheck.py: its decoder against the engine's, the
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
        rep.error("smoke", "pycheck: train/pycheck.py failed (" + r.status_text() + ", see " + path + ")");
    std::vector<rmcheck::ScanLine> errs, warns;
    rmcheck::scan_transcript(r.out, &errs, &warns);
    if (!errs.empty())
        rep.error("smoke", "pycheck: " + std::to_string(errs.size()) + " error line(s), first at " + path + ":" +
                               std::to_string(errs[0].lineno) + ": " + rmcheck::quoted(errs[0].line));
    if (!warns.empty())
        rep.warn("smoke", "pycheck: " + std::to_string(warns.size()) + " engine WARNING line(s) — review " + path);
}

namespace rmcheck {

// The fuzz tier: C++ explore vs explore in-process (--agent-a/-b <mode>, the Python explore tier's
// games for the same seeds), bo1 over the matchups (default the league mirrors, --fuzz-games per
// matchup; matchup k's game i has game seed seed + 1000k + i), then one bo3 explore:patient match
// per ring matchup (seed + 500000 + k).
void tier_fuzz(const Options &opt, Report &rep) {
    const std::vector<std::pair<std::string, std::string>> pairs =
        matchups(opt, opt.matchups.empty() ? "mirrors" : opt.matchups);
    const std::vector<std::pair<std::string, std::string>> ring = matchups(opt, "ring");
    std::vector<GameJob> bo1, bo3;
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
    std::vector<GameJob *> all;
    for (GameJob &j : bo1) all.push_back(&j);
    for (GameJob &j : bo3) all.push_back(&j);
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
    report_matchups(opt, rep, "fuzz", pairs, bo1, opt.fuzz_games);
    report_matchups(opt, rep, "fuzz_bo3", ring, bo3, 1);
    size_t decisions = 0;
    double cpu = 0;
    for (const GameJob *j : all) {
        for (const std::string &line : split_lines(j->result.out)) decisions += starts_with(line, "| >> ") ? 1 : 0;
        cpu += j->result.seconds;
    }
    rep.info("fuzz: " + std::to_string(all.size()) + " games/matches, " + std::to_string(decisions) + " decisions, " +
             fmt1(cpu) + " engine-seconds");
}

// The bench tier: decisions per second of a fixed workload — explore vs explore, bo1, three
// league pairings × three games (game seeds 1-3), one engine at a time (so the figure does not
// depend on -j). The decision count is each game's decision log's length.
void tier_bench(const Options &opt, Report &rep) {
    const std::vector<std::pair<std::string, std::string>> pairs = {{"league/ur_delver", "league/gw_maverick"},
        {"league/bug", "league/ur_delver"}, {"league/reanimator", "league/burn"}};
    const std::string dir = make_temp_dir(opt.out("bench"), "run_");
    size_t decisions = 0, games = 0;
    double wall = 0;
    for (const auto &p : pairs)
        for (long s = 1; s <= 3; ++s) {
            const std::string log = path_join(dir, "g.rmlog");
            ProcSpec spec;
            spec.argv = {opt.binary, "--agent-a", "explore", "--agent-b", "explore", "--game-seed", std::to_string(s),
                "--deck-a", p.first, "--deck-b", p.second, "--decision-log", log};
            spec.cwd = opt.bin_dir;
            const ProcResult r = run_process(spec);
            if (!r.ok()) {
                rep.error("bench", p.first + " vs " + p.second + " seed " + std::to_string(s) + ": " + r.status_text());
                continue;
            }
            std::string text;
            read_file(log, &text);
            const size_t at = text.find("DECISIONS\n");
            if (at != std::string::npos)
                for (const std::string &line : split_lines(text.substr(at + 10)))
                    decisions += trim(line).empty() ? 0 : 1;
            wall += r.seconds;
            ++games;
        }
    remove_tree(dir);
    rep.info("bench: " + std::to_string(games) + " explore games, " + std::to_string(decisions) + " decisions in " +
             fmt1(wall) +
             " s: " + std::to_string(static_cast<long>(static_cast<double>(decisions) / (wall > 0 ? wall : 1))) +
             " decisions/s, " + fmt1(static_cast<double>(games) / (wall > 0 ? wall : 1)) + " games/s (" + opt.binary +
             ")");
}

// The smoke tier: scripted hard vs hard (the Python agent: train/rm_driver.py, one process per
// matchup) over the league mirrors and the ring (--smoke-games per matchup, game i of matchup k
// with seed seed + 1000k + i), and beside them the kept Python's own checks (train/pycheck.py).
// Skipped with a notice when no Python interpreter is found.
void tier_smoke(const Options &opt, Report &rep) {
    if (opt.python.empty()) {
        rep.info("smoke: skipped (no Python interpreter: train/.venv/bin/python or python3)");
        return;
    }
    const std::vector<std::pair<std::string, std::string>> pairs =
        matchups(opt, opt.matchups.empty() ? "mirrors+ring" : opt.matchups);
    const std::string out_dir = opt.out("");
    std::vector<ProcResult> results(pairs.size());
    std::vector<std::string> work(pairs.size());
    ProcResult pycheck;
    // Job 0 is the Python checks (the longest job), job k + 1 matchup k.
    parallel_for(pairs.size() + 1, opt.jobs, [&](size_t job) {
        ProcSpec spec;
        if (job == 0) {
            spec.argv = {opt.python, opt.repo + "/train/pycheck.py", "--binary", opt.binary, "--out-dir", out_dir};
            spec.cwd = opt.repo;
            spec.timeout_s = 3600;
            spec.merge_stderr = true;
            pycheck = run_process(spec);
            return;
        }
        const size_t k = job - 1;
        work[k] = make_temp_dir(out_dir, ".smoke_");
        spec.argv = {opt.python, opt.repo + "/train/rm_driver.py", "--player-a", "scripted", "--player-b", "scripted",
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
