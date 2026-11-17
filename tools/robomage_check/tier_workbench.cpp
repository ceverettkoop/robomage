#include <unistd.h>

#include <string>
#include <vector>

#include "gate.h"
#include "proc.h"
#include "util.h"

// ── The workbench tier: rmwork --selftest and --selftest-data (opt-in) ─────────────────────────
//
// The workbench (bin/<config>/rmwork, `make workbench`) is not part of the default build: without
// it beside the engine the tier prints that it was skipped and passes. With it, `rmwork --selftest`
// (headless: no display or GPU) must exit 0, and so must `rmwork --selftest-data --engine <engine>`
// (Replay's data on corpus logs the engine replays; skipped with a message when the engine is
// absent); their output is kept in <out>/workbench/selftest.txt and selftest_data.txt. Then
// `tools/scryfall/fetch_images.py --selftest` (the card images' printing rule on canned Scryfall
// JSON, no network) must exit 0 (skipped without python3), its output in fetch_images.txt.

static std::string summary_line(const std::string &out, const char *prefix);
static void run_selftest(const rmcheck::Options &opt, rmcheck::Report &rep, const std::vector<std::string> &argv,
    const char *prefix, const char *file);

// The self-test's "<prefix> …" summary line, or "".
static std::string summary_line(const std::string &out, const char *prefix) {
    for (const std::string &line : rmcheck::split_lines(out))
        if (rmcheck::starts_with(line, prefix)) return line;
    return "";
}

// Runs one of rmwork's self-tests; an error unless it exits 0 with its summary line.
static void run_selftest(const rmcheck::Options &opt, rmcheck::Report &rep, const std::vector<std::string> &argv,
    const char *prefix, const char *file) {
    const std::string log = rmcheck::path_join(opt.out("workbench"), file);
    rmcheck::ProcSpec spec;
    spec.argv = argv;
    spec.timeout_s = 900;
    spec.merge_stderr = true;
    const rmcheck::ProcResult r = rmcheck::run_process(spec);
    rmcheck::write_file(log, r.out + r.err);
    const std::string summary = summary_line(r.out, prefix);
    if (!r.ok() || summary.empty()) {
        rep.error("workbench", argv[0] + " " + argv[1] + " " + r.status_text() +
                                   (summary.empty() ? "" : ": " + summary) + " (see " + log + ")");
        return;
    }
    rep.info("workbench: " + summary);
}

namespace rmcheck {

void tier_workbench(const Options &opt, Report &rep) {
    const std::string bin = path_join(dirname_of(opt.binary), "rmwork");
    if (access(bin.c_str(), X_OK) != 0) {
        rep.info("workbench: skipped: no " + bin + " (`make workbench` builds it)");
        return;
    }
    run_selftest(opt, rep, {bin, "--selftest"}, "RMWORK-SELFTEST:", "selftest.txt");
    if (access(opt.binary.c_str(), X_OK) != 0) {
        rep.info("workbench: --selftest-data skipped: no engine " + opt.binary);
        return;
    }
    run_selftest(
        opt, rep, {bin, "--selftest-data", "--engine", opt.binary}, "RMWORK-SELFTEST-DATA:", "selftest_data.txt");
    if (opt.python.empty()) {
        rep.info("workbench: fetch_images.py --selftest skipped (no python3 on PATH)");
        return;
    }
    run_selftest(opt, rep, {opt.python, path_join(opt.repo, "tools/scryfall/fetch_images.py"), "--selftest"},
        "FETCH-IMAGES-SELFTEST:", "fetch_images.txt");
}

}  // namespace rmcheck
