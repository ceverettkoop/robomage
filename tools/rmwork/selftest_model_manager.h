#ifndef RMWORK_SELFTEST_MODEL_MANAGER_H
#define RMWORK_SELFTEST_MODEL_MANAGER_H

#include <string>

#include "settings.h"
#include "selftest_table.h"

// ── The Models tool's checks: --selftest (the policy catalog's synthetic runs directory) ─────────

namespace rmwork {

// Without ImGui: the view's order (date descending by default, latest first under every column and
// direction), the filter, the run, Show all, the runs listed; the compatibility verdicts on synthetic
// manifest pairs (a tables manifest.json and latest.json's object: compatible, an export refused
// with its retable hint where a checkpoint passes, latest.json without a vocab, without a manifest),
// and the checker over the fixture with a given engine manifest (no engine run).
void check_model_manager_units(const CheckFn &check, const std::string &tmp);

// An App (headless, within a UI context) with the tool open over the fixture: it draws and lists the
// rows newest first, latest first; the checker's verdicts reach the rows; the filter, the run and
// Show all narrow or widen it; a run written while the tool is closed is listed when it opens again;
// Serve adds a policy once under the next free id and turns the server on; Play against opens New
// Game with that model as the opponent's; Start, Stop and Settings → Models.
void check_model_manager_app(const CheckFn &check, const ConfigPaths &paths, const std::string &tmp);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_MODEL_MANAGER_H */
