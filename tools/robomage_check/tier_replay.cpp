#include "corpus.h"
#include "gate.h"
#include "proc.h"
#include "scan.h"
#include "util.h"

namespace rmcheck {

// The replay tier (corpus.h): every corpus decision log replayed, its output against its baseline;
// with --record the baselines are rewritten instead.
void tier_replay(const Options &opt, Report &rep) {
    const std::vector<std::string> names = corpus_names(opt);
    if (names.empty()) {
        rep.error("replay", "no decision log (*.rmlog) in " + opt.corpus_dir() +
                                " (record them: train/regression/replay_diff.py record)");
        return;
    }
    for (const std::string &f : list_dir(opt.corpus_dir(), ".replay")) {
        const std::string stem = f.substr(0, f.size() - 7);
        if (!file_exists(corpus_log(opt, stem)))
            rep.error("replay", "baseline " + f + " has no decision log " + stem + ".rmlog");
    }
    const std::string out_dir = opt.out("replay");
    std::vector<std::string> results(names.size());
    parallel_for(names.size(), opt.jobs, [&](size_t i) {
        const std::string &name = names[i];
        const ProcResult r = replay_corpus_game(opt, name, {});
        std::vector<ScanLine> errors;
        scan_transcript(r.out + r.err, &errors, nullptr);
        if (!r.ok() || !errors.empty()) {
            write_file(path_join(out_dir, name + ".out"), r.out + r.err);
            rep.error("replay", name + ": the replay failed (" + r.status_text() + ")" +
                                    (errors.empty() ? "" : ": " + errors[0].line) + " (see " +
                                    path_join(out_dir, name + ".out") + ")");
            return;
        }
        const std::string text = baseline_text(r.out, corpus_log(opt, name), name);
        const std::string path = corpus_baseline(opt, name);
        if (opt.record) {
            std::string old;
            const bool had = read_file(path, &old);
            if (!write_file(path, text)) {
                rep.error("replay", name + ": cannot write " + path);
                return;
            }
            results[i] = std::string(had && old == text ? "unchanged " : "recorded  ") + name + " (" +
                         std::to_string(text.size()) + " bytes)";
            return;
        }
        std::string expected;
        if (!read_file(path, &expected)) {
            rep.error("replay", name + ": no baseline " + path + " (record it: robomage_check --tier replay --record)");
            return;
        }
        const std::string diff = first_difference(expected, text);
        if (!diff.empty()) {
            const std::string actual = path_join(out_dir, name + ".replay");
            write_file(actual, text);
            rep.error("replay", "DRIFT " + name + ": " + diff + " (actual output: " + actual +
                                    "; re-record an intended change with `make regen`)");
            return;
        }
        results[i] = "OK   " + name;
    });
    size_t ok = 0;
    for (const std::string &r : results) {
        if (r.empty()) continue;
        ++ok;
        if (opt.record) rep.info(r);
    }
    rep.info(std::string(opt.record ? "recorded " : "replay: ") + std::to_string(ok) + "/" +
             std::to_string(names.size()) + (opt.record ? " baselines" : " corpus games byte-identical"));
}

}  // namespace rmcheck
