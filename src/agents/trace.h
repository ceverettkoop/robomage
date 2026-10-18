#ifndef AGENTS_TRACE_H
#define AGENTS_TRACE_H

#include <cstddef>
#include <string>
#include <vector>

#include "decision_view.h"

// ── --trace-decisions: the board and menu at every decision, from the deciding seat's frame ───
//
// Rendered from the decoded STATE frame the deciding seat receives, by the CLI's renderer
// (cli/board.h: so it shows what that seat may see, nothing more, as the human seat sees it).
// Every line starts with "| ", which no narrative line does, so a reader of the game log can tell
// the trace from the narrative:
//
//   | decision <n>: Player <X> (game <g>, turn <t>, <STEP>, Player <Y> active)   [or: sideboarding]
//   | <board>                     cli::board_lines, then cli::pending_lines
//   | menu:
//   |   <i>: <description>        one line per menu row
//
// then, once the decision is made:
//
//   | >> Player <X>: <i> (<description>)     or     | >> Player <X>: -2 (concede game) / -3 (concede match)
//
// <n> counts the run's decisions from 1; <g> and <t> are the frame's 0-based game and turn. The
// header, menu and choice lines keep this exact shape (the gate reads them: a menu row is "|   "
// then the index and ": "); the board's lines never take the shape of a menu row.

namespace agents {

std::vector<std::string> trace_decision_lines(const DecisionView &v, size_t number);
std::string trace_choice_line(const DecisionView &v, int choice);

}  // namespace agents

#endif /* AGENTS_TRACE_H */
