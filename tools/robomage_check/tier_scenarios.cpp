#include "embed.h"
#include "gate.h"
#include "parse_report_check.h"
#include "proc.h"
#include "registry_check.h"
#include "scenarios.h"
#include "util.h"

namespace rmcheck {

// The vocab tier: the engine's --check-decks over the playable decks (the top-level, league/ and
// heldout/ decks), the text generator's tokenizer against its golden ids (embed.h), the layout
// registry (registry_check.h), the parse report over the vocab and the candidates' onboarding
// status (parse_report_check.h).
void tier_vocab(const Options &opt, Report &rep) {
    ProcSpec spec;
    spec.argv = {opt.binary, "--check-decks"};
    spec.cwd = opt.bin_dir;
    spec.timeout_s = 300;
    const ProcResult r = run_process(spec);
    for (const std::string &line : split_lines(r.out)) rep.info(line);
    if (!r.ok()) rep.error("vocab", "robomage --check-decks failed (" + r.status_text() + "):\n" + r.out + r.err);
    embed_tokenizer_check(opt, rep);
    registry_check_part(opt, rep);
    parse_report_part(opt, rep);
    onboard_check_part(opt, rep);
}

// The scenarios tier: every rules-regression scenario (scenarios.h), in parallel.
void tier_scenarios(const Options &opt, Report &rep) {
    const std::vector<std::string> paths = scenario_paths(opt);
    if (paths.empty()) {
        rep.error("scenarios", "no scenario matched");
        return;
    }
    prevalidate_scenarios(paths);
    std::vector<ScenarioRun> runs(paths.size());
    parallel_for(paths.size(), opt.jobs, [&](size_t i) { runs[i] = run_scenario(opt, paths[i], {}); });
    size_t failed = 0;
    for (const ScenarioRun &run : runs) {
        if (opt.show == "output") {
            std::printf("##### %s\n%s\n", run.name.c_str(), run.output.c_str());
        } else if (opt.show == "narrative") {
            std::vector<std::string> narrative, actions;
            split_engine_output(run.output, &narrative, &actions);
            std::printf("##### %s\n%s\n", run.name.c_str(), join(narrative, "\n").c_str());
        } else if (opt.show == "dump") {
            std::printf("##### %s\n", run.name.c_str());
            for (const auto &f : run.frames)
                for (const std::string &l : f.second) std::printf("%s\n", l.c_str());
        }
        if (run.fails.empty()) continue;
        ++failed;
        std::string msg = run.name + ": " + run.fails[0];
        for (size_t k = 1; k < run.fails.size(); ++k) msg += "\n           " + run.fails[k];
        if (!run.reproduce.empty()) msg += "\n           reproduce: " + run.reproduce;
        rep.error("scenarios", msg);
    }
    rep.info("scenarios: " + std::to_string(runs.size() - failed) + "/" + std::to_string(runs.size()) + " passed");
}

}  // namespace rmcheck
