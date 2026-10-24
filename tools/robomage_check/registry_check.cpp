#include "registry_check.h"

#include "proc.h"
#include "scan.h"
#include "util.h"

namespace rmcheck {

void registry_check_part(const Options &opt, Report &rep) {
    const std::string dir = opt.out("vocab");
    make_dirs(dir);
    const std::string cards = path_join(dir, "cards.rmst"), base = path_join(dir, "registry_head.json");
    const std::string registry = path_join(opt.repo, "src/featurize/registry.json");
    ProcSpec spec;
    spec.argv = {opt.binary, "--cards-frame", cards};
    spec.cwd = opt.bin_dir;
    const ProcResult frame = run_process(spec);
    if (!frame.ok()) {
        rep.error("vocab", "robomage --cards-frame " + frame.status_text() + ": " + frame.err);
        return;
    }
    const std::string git = find_program("git");
    bool have_base = false;
    if (!git.empty()) {
        spec.argv = {git, "-C", opt.repo, "show", "HEAD:src/featurize/registry.json"};
        spec.cwd = "";
        const ProcResult head = run_process(spec);
        have_base = head.ok() && !head.out.empty() && write_file(base, head.out);
    }
    if (!have_base) rep.info("     registry: no committed registry to compare with (git show HEAD failed)");
    spec.argv = {opt.rmfeat, "--registry-check", registry, "--cards", cards, "--text-table",
        path_join(opt.bin_dir, "resources/embeddings/oracle_text.rmtb")};
    if (have_base) {
        spec.argv.push_back("--base");
        spec.argv.push_back(base);
    }
    const ProcResult r = run_process(spec);
    for (const std::string &line : split_lines(r.out))
        if (starts_with(line, "REGISTRY: ")) rep.info("     " + line);
    if (!r.ok()) {
        std::string why;
        for (const std::string &line : split_lines(r.out + r.err))
            if (starts_with(line, "REGISTRY-FAIL: ") || starts_with(line, "rmfeat: ")) why += "\n           " + line;
        rep.error("vocab", "rmfeat --registry-check " + r.status_text() + why);
    } else {
        rep.info("OK   registry: src/featurize/registry.json current, compiled in" +
                 std::string(have_base ? " and no entry moved since HEAD" : ""));
    }
    digest_check_part(opt, rep, cards);
    remove_tree(cards);
}

void digest_check_part(const Options &opt, Report &rep, const std::string &cards) {
    ProcSpec spec;
    spec.argv = {opt.rmfeat, "--digest-check", path_join(opt.repo, "src/featurize/card_digest.json"), "--cards", cards,
        "--text-table", path_join(opt.bin_dir, "resources/embeddings/oracle_text.rmtb")};
    const ProcResult r = run_process(spec);
    if (r.ok()) {
        for (const std::string &line : split_lines(r.out))
            if (starts_with(line, "DIGEST: ")) rep.info("OK   card digest: " + line.substr(8));
        return;
    }
    std::string why;
    for (const std::string &line : split_lines(r.out + r.err))
        if (starts_with(line, "DIGEST-FAIL: ") || starts_with(line, "rmfeat: ")) why += "\n           " + line;
    rep.error("vocab",
        "rmfeat --digest-check (a known card's table rows changed: the drift report, "
        "docs/featurizer.md §6.2) " +
            r.status_text() + why);
}

}  // namespace rmcheck
