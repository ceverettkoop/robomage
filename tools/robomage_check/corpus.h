#ifndef ROBOMAGE_CHECK_CORPUS_H
#define ROBOMAGE_CHECK_CORPUS_H

#include <string>
#include <vector>

#include "gate.h"
#include "proc.h"

// ── robomage_check: the replay corpus (train/regression/corpus/) ──────────────────────────────
//
// Each corpus game is a decision log, <name>.rmlog (RMLOG v2: the seed, the setup flags, both
// decklists and every decision), recorded from the scripted agent's games by
// train/regression/replay_diff.py record, and a baseline, <name>.replay: the engine's
//
//   robomage --replay <name>.rmlog --machine --narrative --trace-decisions
//
// stdout (--machine: the narrative without the interactive CLI's board printouts) made independent
// of where the corpus lives and of the trace's board rendering: the replay banner's log path
// becomes the log's file name, and the trace keeps its decision header, menu and choice lines (the
// board lines between "| decision" and "| menu:" are left out). The replay tier compares that
// text with the baseline byte for byte; --record rewrites the baselines.

namespace rmcheck {

// The corpus games: the stems of <corpus>/*.rmlog, sorted.
std::vector<std::string> corpus_names(const Options &opt);
std::string corpus_log(const Options &opt, const std::string &name);
std::string corpus_baseline(const Options &opt, const std::string &name);
// Replays one corpus log with the baseline's flags plus `extra`.
ProcResult replay_corpus_game(const Options &opt, const std::string &name, const std::vector<std::string> &extra);
// The baseline text of a replay's stdout.
std::string baseline_text(const std::string &out, const std::string &log_path, const std::string &name);
// The first line (1-based) where `a` and `b` differ, with both lines; "" when equal.
std::string first_difference(const std::string &expected, const std::string &actual);

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_CORPUS_H */
