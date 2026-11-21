#ifndef RMWORK_SELFTEST_MODEL_ZIP_H
#define RMWORK_SELFTEST_MODEL_ZIP_H

#include <string>

#include "selftest_table.h"

// ── Zipped models' checks: --selftest (synthetic archives) ───────────────────────────────────────

namespace rmwork {

// A synthetic archive under `tmp` (a stored and a deflated export, a README, a directory entry):
// its entries and exports, the labels, references to a member and to the bare archive, extraction
// into the cache ($XDG_CACHE_HOME pointed into `tmp` for the call) and its reuse, and the refusals: a
// missing member, a bare archive of two exports, a damaged CRC-32, a member path leaving its
// directory, a file that is not a zip, an encrypted member.
void check_model_zip_units(const CheckFn &check, const std::string &tmp);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_MODEL_ZIP_H */
