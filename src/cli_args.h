#ifndef CLI_ARGS_H
#define CLI_ARGS_H

#include <string>
#include <vector>

#include "input_logger.h"

enum class CliParseResult {
    RUN,
    SHOW_HELP
};

// Parses argv into the launch-option globals (game_driver.h), driven by one option table that
// defines which flags exist and which take a value. An unrecognized argument, or a value-taking
// flag that ends argv with no value, is skipped with a warning on stderr (stdout carries the
// machine protocol). Under --replay the setup options (decks, seed, --bo3, --no-shuffle, zone
// presets, starting life) are not applied, so they keep their defaults until the log header
// supplies them; any given on the command line are reported on stderr as ignored.
// Returns SHOW_HELP when --help / -h is present (no option is applied then).
CliParseResult parse_command_line(int argc, char const *argv[]);

// Appends the RMLOG v2 FLAGS tokens for every setup option in effect (plus the "machine"
// decision-schedule marker) to `header`. Every setup option other than the decks and the seed,
// which have their own header fields, is written here.
void record_header_flags(DecisionLogHeader &header);

// True when `token` is the FLAGS key of an option that takes no value (e.g. "no-shuffle"), i.e. a
// complete token on its own rather than the continuation of a key=value token split at a space.
bool is_bare_header_flag(const std::string &token);

// Applies a replay log's FLAGS tokens to the launch-option globals. A token missing from the
// FLAGS line leaves that option at its default. Fatal on a token no option records.
void apply_replay_header_flags(const std::vector<std::string> &tokens);

#endif /* CLI_ARGS_H */
