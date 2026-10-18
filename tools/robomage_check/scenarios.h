#ifndef ROBOMAGE_CHECK_SCENARIOS_H
#define ROBOMAGE_CHECK_SCENARIOS_H

#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "gate.h"
#include "json.h"

// ── robomage_check: the rules-regression scenarios (tests/scenarios/*.json) ───────────────────
//
// Each scenario plays as one engine run with its presets as engine flags and its `play` script on
// the engine's --play resolver (--game-seed <seed, default 1> --narrative --trace-decisions
// --max-decisions <default 60>, --bo3 for format bo3, --hand-*/--library-* or --deck-* (default
// delver), the zone presets comma-joined with apostrophes dropped, --life-*, --play / --actions,
// else --agent-a first --agent-b first; --offer-cancel and engine_flags as given; --rmstate-dump
// for the expect_dump* keys). The assertions are over the narrative (every non-blank line that is
// not a --trace-decisions line, from the first turn banner on, so the pregame's opening draws are
// left out), the offered actions (the trace's menu rows), the GAME_RESULT, the knowledge lines
// (knowledge_lines.h) and the variants' expect_same lines. A scenario also fails on an error line
// (ERROR: / FATAL: — a failed --play spec is "FATAL: --play: …" — a crash, "SCRIPT INCOMPLETE")
// or a nonzero engine exit.
//
// Scenario keys beyond the presets (name, hand_a/b, library_a/b, battlefield_a/b, graveyard_a/b,
// exile_a/b, sideboard_a/b, life_a/b, play or actions, seed, max_decisions):
//
//   why            one line: the CR rule and the commit that fixed the behavior (required)
//   expect         regexes that must each match some narrative line
//   expect_not     regexes that must match no narrative line
//   expect_order   regexes that must match narrative lines in this order
//   expect_count   {regex: n}: each regex must match exactly n narrative lines
//   expect_menu    regexes that must each match some offered action (any decision)
//   expect_menu_not  regexes that must match no offered action
//   expect_result  "A" | "B" | "draw": the GAME_RESULT the game must end with
//   expect_unique  regexes with capture groups: over all matching narrative lines, the matched
//                  groups (joined) must be distinct (e.g. no handle issued twice)
//   variants       a list of {preset key: value} overrides: the scenario is also run once per
//                  variant (e.g. the same position built from a different deck order); a variant
//                  may carry its own expect_same, which replaces the scenario's for that variant
//   expect_same    regexes: the narrative lines matching any of them must be identical, in
//                  order, between the scenario's run and each variant's run (needs variants)
//   format         "bo1" (default) or "bo3": a bo3 scenario plays on through sideboarding and the
//                  later games once its script runs out (presets apply to game 1)
//   player_a / player_b  accepted, not read: once its script runs out a seat takes action 0
//   deck_a / deck_b      a deck for a seat with no hand preset
//   offer_cancel   true: run with --offer-cancel
//   engine_flags   extra engine flags (a list, e.g. ["--trace-visibility"])
//   expect_dump    regexes that must each match some knowledge line, any frame
//   expect_dump_not  regexes that must match no knowledge line
//   expect_dump_final  regexes that must each match a knowledge line of a seat's last frame
//   expect_dump_final_not  regexes that must match no knowledge line of either seat's last frame
//
// A seat's last frame is its last STATE frame in the dump (the decision at max_decisions, or the
// terminal frame of a finished game). Variants do not check the expect_dump* keys. Keep
// max_decisions just past the asserted event.
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
// Reads and validates one scenario file; false with `err`.
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
