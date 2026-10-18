#ifndef ROBOMAGE_CHECK_SCAN_H
#define ROBOMAGE_CHECK_SCAN_H

#include <string>
#include <vector>

// ── robomage_check: what the gate looks for in engine output ──────────────────────────────────
//
// The transcript scan (train/ci_check.py's scan_transcript): an ERROR is a line starting "ERROR:"
// or "FATAL:", or holding "Traceback (most recent call last)", "Segmentation fault",
// "Assertion … failed", the word "Aborted" or "core dumped"; a WARNING is a line starting
// "WARNING:" other than the routine "Unrecognized ability param" parse notice.
//
// The self-check summaries (train/test_selfcheck_state.py's scan_engine_log): every
// "SELFCHECK-STATE: decisions=… errors=N" / "SELFCHECK-KNOWLEDGE: …" line must say errors=0.

namespace rmcheck {

struct ScanLine {
        size_t lineno;  // 1-based
        std::string line;
};

bool is_error_line(const std::string &line);
bool is_warning_line(const std::string &line);
// The error and warning lines of `text`.
void scan_transcript(const std::string &text, std::vector<ScanLine> *errors, std::vector<ScanLine> *warnings);

// Failure messages for one engine's stderr `text` (named `where` in them, labelled `label`): every
// error line (scan_transcript), every self-check summary without errors=0, and, when `summaries`
// is not negative, a count of summary lines of each of `kinds` ("STATE", "KNOWLEDGE") other than
// `summaries`. At most 20 messages.
std::vector<std::string> selfcheck_failures(const std::string &text, const std::string &where, const std::string &label,
    int summaries, const std::vector<std::string> &kinds);
// The last "SELFCHECK-KNOWLEDGE:" line of `text` (trimmed), or a note that there is none.
std::string knowledge_summary(const std::string &text);

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_SCAN_H */
