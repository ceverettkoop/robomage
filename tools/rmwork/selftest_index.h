#ifndef RMWORK_SELFTEST_INDEX_H
#define RMWORK_SELFTEST_INDEX_H

#include <functional>
#include <string>

#include "app.h"
#include "selftest_table.h"

// ── rmwork --selftest-data's checks of the cached match index (match_index.h) ────────────────────
//
// On decision log `log`, already opened once (its replay and index cached): a second open reads the
// index; the index read back equals the one index_match builds from the dump (every decision's
// fields, the games, the three logs, the terminal frames, the result, each decision by its key, the
// viewers' streams); a corrupt index and one of an older version are rebuilt (that open indexes the
// dump, and writes an index the next open reads).

namespace rmwork {

void check_match_index(const CheckFn &check, App &app, const std::string &log, const std::function<bool()> &open);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_INDEX_H */
