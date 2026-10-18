#ifndef ROBOMAGE_CHECK_KNOWLEDGE_LINES_H
#define ROBOMAGE_CHECK_KNOWLEDGE_LINES_H

#include <string>
#include <utility>
#include <vector>

// ── robomage_check: the seats' knowledge tables as text lines (the expect_dump* assertions) ────
//
// train/test_scenarios.py's knowledge_frames, read from the binary --rmstate-dump instead of its
// JSON rendering (the same values: state_format/json.h's rules). Every seat STATE frame's k_*
// tables, one fact per line, prefixed by the seat and the frame:
//
//   <A|B> d=<decision> g<game> <t<turn>|sb> <fact>
//
// with the facts meta, seen, pool, position, known_hand, list, archetype and stack_target as that
// module's docstring lists them.

namespace rmcheck {

// [(seat 'A' / 'B', the frame's lines)] for every seat STATE frame of `payloads` (a dump's
// records), in order; false with `err` when a frame does not decode.
bool knowledge_frames(const std::vector<std::string> &payloads,
    std::vector<std::pair<char, std::vector<std::string>>> *frames, std::string *err);

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_KNOWLEDGE_LINES_H */
