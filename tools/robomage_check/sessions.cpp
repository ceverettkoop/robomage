#include "sessions.h"

#include "util.h"

static const double RMKNOW_TIMEOUT_S = 600;

static std::string summary_line(const std::string &text, const char *prefix);
static std::vector<std::string> error_lines(const std::string &text, size_t max);

static std::string summary_line(const std::string &text, const char *prefix) {
    std::string last;
    for (const std::string &l : rmcheck::split_lines(text))
        if (rmcheck::starts_with(l, prefix)) last = l;
    return last;
}

// The lines of `text` holding "ERROR:", at most `max`.
static std::vector<std::string> error_lines(const std::string &text, size_t max) {
    std::vector<std::string> out;
    for (const std::string &l : rmcheck::split_lines(text))
        if (rmcheck::contains(l, "ERROR:") && out.size() < max) out.push_back(l);
    return out;
}

namespace rmcheck {

std::string fuzz_label(const FuzzGame &g) {
    bool sample = false;
    for (const std::string &f : g.flags) sample = sample || f == "--selfcheck-state-sample";
    return "fuzz " + deck_stem(g.deck_a) + " vs " + deck_stem(g.deck_b) + (g.bo3 ? " bo3" : " bo1") + " seed " +
           std::to_string(g.seed) + (sample ? " sample=1" : "");
}

FuzzOutcome play_fuzz_game(const Options &opt, const FuzzGame &g, const std::string &dir) {
    FuzzOutcome o;
    o.label = fuzz_label(g);
    o.err_path = path_join(dir, "fuzz_" + std::to_string(g.seed) + ".err");
    ProcSpec spec;
    spec.argv = {opt.binary, "--agent-a", "explore", "--agent-b", "explore", "--game-seed", std::to_string(g.seed),
        "--deck-a", g.deck_a, "--deck-b", g.deck_b, "--max-decisions", "2500"};
    if (g.bo3) spec.argv.push_back("--bo3");
    spec.argv.insert(spec.argv.end(), g.flags.begin(), g.flags.end());
    if (g.dump) {
        o.dump = path_join(dir, "fuzz_" + std::to_string(g.seed) + ".rms");
        spec.argv.push_back("--rmstate-dump");
        spec.argv.push_back(o.dump);
    }
    spec.cwd = opt.bin_dir;
    spec.timeout_s = 3600;
    o.result = run_process(spec);
    write_file(o.err_path, o.result.err);
    o.finished = o.result.ok() && !contains(o.result.out, "MAX_DECISIONS:");
    return o;
}

std::string stream_label(const char *what, const StreamGame &g) {
    bool closed = false;
    for (const std::string &f : g.flags) closed = closed || f == "--closed-decklist";
    return std::string(what) + " " + deck_stem(g.deck_a) + " vs " + deck_stem(g.deck_b) + (g.bo3 ? " bo3" : " bo1") +
           " seed " + std::to_string(g.seed) + (closed ? " closed" : "");
}

StreamRun play_stream_session(const Options &opt, const StreamGame &g, const std::string &dump, uint64_t cap) {
    ProcSpec spec;
    spec.argv = {opt.binary, "--machine-format", "rmstate", "--seed", std::to_string(g.seed), "--deck-a", g.deck_a,
        "--deck-b", g.deck_b};
    if (g.bo3) spec.argv.push_back("--bo3");
    if (!dump.empty()) {
        spec.argv.push_back("--rmstate-dump");
        spec.argv.push_back(dump);
    }
    spec.argv.insert(spec.argv.end(), g.flags.begin(), g.flags.end());
    spec.cwd = opt.bin_dir;
    spec.timeout_s = 1800;
    StreamRun run = run_stream(spec, random_replies(static_cast<uint64_t>(g.seed), cap));
    if (run.error.empty() && !run.proc.ok()) run.error = "engine " + run.proc.status_text();
    if (run.error.empty() && run.terminals != 2)
        run.error = std::to_string(run.terminals) + " terminal frames, expected 2";
    return run;
}

std::vector<std::string> rmknow_isolation(
    const Options &opt, const std::string &path, const char *level, std::string *summary) {
    std::vector<std::string> fails;
    std::vector<std::string> sums;
    for (const char *seat : {"A", "B"}) {
        ProcSpec spec;
        spec.argv = {opt.rmknow, "--viewer", seat, "--level", level, "--isolation", "--in", path};
        spec.timeout_s = RMKNOW_TIMEOUT_S;
        const ProcResult r = run_process(spec);
        const std::string sum = summary_line(r.err, "RMKNOW:");
        if (!r.ok() || sum.empty() || contains(sum, "isolation_frames=0 "))
            fails.push_back(basename_of(path) + " viewer " + seat + ": rmknow --isolation " + r.status_text() + ": " +
                            join(error_lines(r.err, 3), " | ") + " " + sum);
        sums.push_back(sum);
    }
    if (summary) *summary = join(sums, "; ");
    return fails;
}

std::vector<std::string> rmknow_truth(
    const Options &opt, const std::string &path, const char *seat, const char *level, std::string *summary) {
    ProcSpec spec;
    spec.argv = {opt.rmknow, "--viewer", seat, "--level", level, "--truth", "--quiet", "--in", path};
    spec.timeout_s = RMKNOW_TIMEOUT_S;
    const ProcResult r = run_process(spec);
    const std::string sum = summary_line(r.err, "RMKNOW:");
    if (summary) *summary = sum;
    if (!r.ok() || sum.empty() || contains(sum, " truth_frames=0 "))
        return {"truth " + basename_of(path) + " viewer " + seat + ": rmknow --truth " + r.status_text() + ": " +
                join(error_lines(r.err, 3), " | ") + " " + sum};
    return {};
}

std::vector<std::string> recode_dump(const Options &opt, const std::string &path, std::string *summary) {
    std::vector<std::string> fails;
    ProcSpec spec;
    spec.argv = {opt.binary, "--rmstate-read", path, "--state-dump", "json"};
    spec.cwd = opt.bin_dir;
    spec.timeout_s = RMKNOW_TIMEOUT_S;
    const ProcResult r = run_process(spec);
    const std::string sum = summary_line(r.err, "RMSTATE_READ:");
    if (summary) *summary = sum;
    if (!r.ok() || contains(r.err, "ERROR:"))
        fails.push_back(basename_of(path) + ": --rmstate-read " + r.status_text() + ": " +
                        join(error_lines(r.err, 3), " | ") + " " + sum);
    std::string live;
    if (read_file(path + ".jsonl", &live) && live != r.out)
        fails.push_back(basename_of(path) + ": the live --state-dump json file differs from --rmstate-read's JSON");
    return fails;
}

}  // namespace rmcheck
