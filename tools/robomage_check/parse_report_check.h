#ifndef ROBOMAGE_CHECK_PARSE_REPORT_CHECK_H
#define ROBOMAGE_CHECK_PARSE_REPORT_CHECK_H

#include "gate.h"

// ── robomage_check: the parse report over the vocab, and the candidates' onboarding status ─────
//
// Part of the vocab tier. parse_report_part: robomage --parse-report @vocab, every vocab card's
// findings (kind:name, src/parse_report.h) against the committed baseline
// tests/parse_report/vocab.json — the vocab cards' known parser gaps. A finding the baseline lacks
// (a parser change that drops something a known card says, or an evaluator form the report's
// recognizer misses) and a baseline finding gone (a gap fixed: the baseline is out of date) are
// errors; --record rewrites the baseline. A vocab card with no script is an error.
// onboard_check_part: python3 tools/onboard/onboard.py check (stages 2, 4 and 5 of every candidate
// in decks/candidates/, nothing written, no network): a tool failure is an error, a committed
// status file whose result the current engine and vocab no longer give is a warning (rerun
// onboard.py all). Skipped without a Python.

namespace rmcheck {

void parse_report_part(const Options &opt, Report &rep);
void onboard_check_part(const Options &opt, Report &rep);

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_PARSE_REPORT_CHECK_H */
