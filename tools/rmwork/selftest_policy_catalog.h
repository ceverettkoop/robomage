#ifndef RMWORK_SELFTEST_POLICY_CATALOG_H
#define RMWORK_SELFTEST_POLICY_CATALOG_H

#include <string>

#include "selftest_table.h"

// ── The policy catalog's checks: --selftest (a synthetic runs directory) ────────────────────────

namespace rmwork {

// Writes the synthetic runs directory below at `runs` (alpha, beta, latest/); false when a file cannot
// be written.
bool write_policy_fixture(const std::string &runs);

// A synthetic runs directory under `tmp`: two runs with fixed points, final, periodic checkpoints, the
// seed, a live export, the old layouts' copies, evaluation exports and temporary files, metrics.jsonl
// dates that disagree with the files' modification times (a torn last line, noise lines), a policy no
// metrics line names, and latest/. Checked: latest pinned first and never a run, the order by
// training date, the hidden rows, what is never listed, the formats merged into one row, the export
// played, the learner steps, the modification-time fallback, a metrics file read again after it grew,
// and New Game's and the Add row's lists over it (latest first and the default model, a chosen model
// kept, no hidden rows).
void check_policy_catalog_units(const CheckFn &check, const std::string &tmp);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_POLICY_CATALOG_H */
