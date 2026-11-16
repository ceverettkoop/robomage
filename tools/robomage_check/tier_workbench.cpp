#include <unistd.h>

#include <string>
#include <vector>

#include "gate.h"
#include "proc.h"
#include "util.h"

// ── The workbench tier: rmwork --selftest (opt-in) ──────────────────────────────────────────────
//
// The workbench (bin/<config>/rmwork, `make workbench`) is not part of the default build: without
// it beside the engine the tier prints that it was skipped and passes. With it, `rmwork --selftest`
// (headless: no display or GPU) must exit 0; its output is kept in <out>/workbench/selftest.txt.

static std::string summary_line(const std::string &out);

// The self-test's "RMWORK-SELFTEST: …" line, or "".
static std::string summary_line(const std::string &out) {
    for (const std::string &line : rmcheck::split_lines(out))
        if (rmcheck::starts_with(line, "RMWORK-SELFTEST:")) return line;
    return "";
}

namespace rmcheck {

void tier_workbench(const Options &opt, Report &rep) {
    const std::string bin = path_join(dirname_of(opt.binary), "rmwork");
    if (access(bin.c_str(), X_OK) != 0) {
        rep.info("workbench: skipped: no " + bin + " (`make workbench` builds it)");
        return;
    }
    const std::string log = path_join(opt.out("workbench"), "selftest.txt");
    ProcSpec spec;
    spec.argv = {bin, "--selftest"};
    spec.timeout_s = 300;
    spec.merge_stderr = true;
    const ProcResult r = run_process(spec);
    write_file(log, r.out + r.err);
    const std::string summary = summary_line(r.out);
    if (!r.ok() || summary.empty()) {
        rep.error("workbench",
            "rmwork --selftest " + r.status_text() + (summary.empty() ? "" : ": " + summary) + " (see " + log + ")");
        return;
    }
    rep.info("workbench: " + summary);
}

}  // namespace rmcheck
