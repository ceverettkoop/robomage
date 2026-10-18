#ifndef AGENTS_DRIVER_H
#define AGENTS_DRIVER_H

#include <vector>

#include "../classes/action.h"

// ── The in-process agents in the engine: options, the decision hook, --trace-decisions ────────
//
// The engine side of src/agents/ (the agents themselves read only frames, agents/agent.h).
//
// Options (cli_args.cpp; the globals live in game_driver.cpp):
//   --agent-a <spec> / --agent-b <spec>  that seat's decisions go to an in-process agent
//   --play <specs>     one global --play script (agents/play_spec.h) for both seats; once it
//                      runs out each seat's --agent-* decides (first when none is given)
//   --actions <ints>   one global list of replies for both seats, then the seats' agents
//   --trace-decisions  the board and menu of every decision (agents/trace.h) on the game log
// Any of the first three implies --machine (the machine decision schedule); a seat with no
// agent and no script keeps its own input path (the RMSTATE stream on stdin/stdout, or the
// terminal for a --player seat). They cannot be combined with --replay or an agent on a
// --player seat.
//
// At the end of the run (process exit, the --max-decisions stop included) a --play script prints
// "resolved --actions: <i,i,...>" (every reply it gave) and, when some of its specs were never
// applied, "SCRIPT INCOMPLETE: <n> --play spec(s) never applied: <specs>".

// Validates the options and installs the agents and the decision hook, before the match: the
// shared generator is seeded with the game seed (--game-seed) when given, else `engine_seed`.
// Fatal on an unknown spec or a conflicting option.
void agents_install(unsigned int engine_seed);

// Called as each decision is read (InputLogger::choose), after the frame writers and before the
// input is read: for a seat an agent decides, or under --trace-decisions, builds and decodes the
// deciding seat's frames and prints the trace block. Returns whether the decision is traced.
bool agents_begin_decision(const std::vector<LegalAction> &actions, size_t number);
// After a traced decision: the choice line.
void agents_trace_choice(int choice);

#endif /* AGENTS_DRIVER_H */
