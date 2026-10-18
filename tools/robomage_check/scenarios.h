#ifndef ROBOMAGE_CHECK_SCENARIOS_H
#define ROBOMAGE_CHECK_SCENARIOS_H

#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "gate.h"
#include "json.h"

// ── robomage_check: the rules-regression scenarios (train/regression/scenarios/*.json) ─────────
//
// train/test_scenarios.py's runner on the engine alone: each scenario plays as one engine run with
// its presets as engine flags and its `play` script on the engine's --play resolver
// (train/compare_cpp_agents.py's engine_scenario_cmd: --game-seed <seed, default 1> --narrative
// --trace-decisions --max-decisions <default 60>, --bo3 for format bo3, --hand-*/--library-* or
// --deck-* (default delver), the zone presets comma-joined with apostrophes dropped, --life-*,
// --play / --actions, else --agent-a first --agent-b first; --offer-cancel and engine_flags as
// given; --rmstate-dump for the expect_dump* keys). The assertions are the Python runner's, with
// the same failure messages: over the narrative (every non-blank line that is not a
// --trace-decisions line, from the first turn banner on), the offered actions (the trace's menu
// rows), the GAME_RESULT, the knowledge lines (knowledge_lines.h) and the variants' expect_same
// lines. A scenario also fails on an error line (ERROR: / FATAL: — a failed --play spec is
// "FATAL: --play: …" — a crash, "SCRIPT INCOMPLETE") or a nonzero engine exit.
//
// The regexes are std::regex ECMAScript, searched in each line as Python's re.search does: the
// scenario files use no Python-only syntax (lookaheads and non-capturing groups are ECMAScript).

namespace rmcheck {

struct Scenario {
        std::string name, path;
        Json doc;
};

// The scenario files of `opt` (its scenario_patterns as name globs, else all), sorted.
std::vector<std::string> scenario_paths(const Options &opt);
// Compiles every regex of the scenario files `paths` in forked children (util.h's first_bad_regex),
// so that load_scenario reports a malformed one instead of the process aborting on it. Call it
// before running those scenarios, while no other thread runs.
void prevalidate_scenarios(const std::vector<std::string> &paths);
// Reads and validates one scenario file (test_scenarios.py's load_scenario); false with `err`.
bool load_scenario(const std::string &path, Scenario *out, std::string *err);
// Whether the scenario has an expect_dump* key.
bool wants_dump(const Json &scn);

struct ScenarioRun {
        std::string name;
        std::vector<std::string> fails;                                 // empty = pass
        std::string output;                                             // the engine's stdout + stderr
        std::vector<std::pair<char, std::vector<std::string>>> frames;  // knowledge lines, when dumped
        std::string reproduce;                                          // the engine command line
};

// Plays one scenario (and its variants) with `extra_flags` added to its engine flags.
ScenarioRun run_scenario(const Options &opt, const std::string &path, const std::vector<std::string> &extra_flags);

// The narrative lines (from the first turn banner) and offered-action descriptions of one run.
void split_engine_output(
    const std::string &text, std::vector<std::string> *narrative, std::vector<std::string> *actions);

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_SCENARIOS_H */
