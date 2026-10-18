#ifndef AGENTS_TRACE_H
#define AGENTS_TRACE_H

#include <cstddef>
#include <string>
#include <vector>

#include "decision_view.h"

// ── --trace-decisions: the board and menu at every decision, from the deciding seat's frame ───
//
// Rendered from the decoded STATE frame the deciding seat receives (so it shows what that seat
// may see, nothing more). Every line starts with "| ", which no narrative line does, so a reader
// of the game log can tell the trace from the narrative:
//
//   | decision <n>: Player <X> (game <g>, turn <t>, <STEP>, Player <Y> active)   [or: sideboarding]
//   | Player <X>: life <l>, hand <h>, library <k>, graveyard <g>, exile <e>[, pool <W..C>]
//   | Player <X> battlefield: <name>[ <p>/<t>][ (tapped, sick, attacking, ..., <counter> x<n>)], ...
//   | stack: <name> (Player <X>), ...                       (top first)
//   | Player <X> hand: <name>, ...                           (the deciding seat's own hand)
//   | Player <X> graveyard: <name>, ...
//   | menu:
//   |   <i>: <description>                                   (one line per menu row)
//
// then, once the decision is made:
//
//   | >> Player <X>: <i> (<description>)     or     | >> Player <X>: -2 (concede game) / -3 (concede match)
//
// <n> counts the run's decisions from 1.

namespace agents {

std::vector<std::string> trace_decision_lines(const DecisionView &v, size_t number);
std::string trace_choice_line(const DecisionView &v, int choice);

}  // namespace agents

#endif /* AGENTS_TRACE_H */
