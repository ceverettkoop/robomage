#ifndef ROBOMAGE_CHECK_SCAN_H
#define ROBOMAGE_CHECK_SCAN_H

#include <map>
#include <regex>
#include <set>
#include <string>
#include <utility>
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
//
// The shadow classifier (train/test_knowledge.py's classify): the signatures of
// tests/knowledge_divergences.json, tried in order.

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

// The divergence signatures of knowledge_divergences.json.
class ShadowSignatures {
    public:
        // Loads and validates `path` (each signature's scenario must exist in `scenario_dir`);
        // false with `err` naming the malformed entry. Its patterns are compiled in forked children
        // first (util.h's first_bad_regex): call it while no other thread runs.
        bool load(const std::string &path, const std::string &scenario_dir, std::string *err);
        // The name of the first signature that classifies the SHADOW line `line`, or "".
        std::string classify(const std::string &line) const;
        struct Signature {
                std::string name;
                std::set<std::string> kinds;
                std::string direction;
                std::string owner;  // "" = any
                std::regex regex;
                std::string scenario;
        };
        const std::vector<Signature> &signatures() const { return sigs_; }
        // Failure messages for one log `text` (named `where`): every SHADOW line classified (at most
        // 10 shown), and with `summaries` not negative exactly that many SHADOW-KNOWLEDGE lines.
        // `counts` (optional) gets the lines per signature name.
        std::vector<std::string> log_failures(const std::string &text, const std::string &where,
            const std::string &label, int summaries, std::map<std::string, int> *counts) const;

    private:
        std::vector<Signature> sigs_;
};

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_SCAN_H */
