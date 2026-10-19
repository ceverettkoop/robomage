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
//   --featurize-seats  every decision of a seat an in-process agent decides is also featurized
//                      (featurize_seat on the seat's decoded frame, the inputs built in process:
//                      state_format/cards_file.h) and the tensors dropped; at exit
//                      "FEATURIZE-SEATS: <n> decisions in <s> s". Play is unchanged: the bench
//                      tier's featurizer stage
// The trajectory writer (--trajectory-out, agents/trajectory.h) and the teacher channel
// (--teacher-a/-b, agents/teacher.h) are installed here too: the seats they record or teach are
// served by the in-process feed whatever decides them.
// Any of the first three implies --machine; a seat with no agent and no script keeps its own
// input path: the RMSTATE stream on stdin/stdout, or, for a human seat (cli_args.cpp: the
// --player seat, or every seat of a run without --machine), the CLI (cli/human.h), installed
// here as that seat's agent (it takes over a --play / --actions script's seat once the script runs
// out). They cannot be combined with --replay or an agent on a --player seat.
//
// At the end of the run (process exit, the --max-decisions stop included) a --play script prints
// "resolved --actions: <i,i,...>" (every reply it gave) and, when some of its specs were never
// applied, "SCRIPT INCOMPLETE: <n> --play spec(s) never applied: <specs>".

// Validates the options and installs the agents, the decision hook and the trajectory writer,
// before the match: the shared generator is seeded with the game seed (--game-seed) when given,
// else `engine_seed`. Fatal on an unknown spec or a conflicting option. An actor (agents/actor.h)
// calls it before each of its matches, with that match's options: the generator is reseeded, each
// seat gets its agent (an agent of an earlier match with the same spec is reused, after its
// begin_match), each seat's decoded stream starts empty, the in-process feed and the knowledge
// sidecar start the match afresh (every seat gets its once-per-match frames again), and the
// trajectory writer and the teacher channel are installed again.
void agents_install(unsigned int engine_seed);

// Called as each decision is read (InputLogger::choose), after the frame writers and before the
// input is read: for a seat an agent decides, a seat the trajectory writer records
// (agents/trajectory.h), a taught seat (agents/teacher.h: its frames are sent to its teacher) or
// under --trace-decisions, builds and decodes the deciding seat's frames and prints the trace
// block; then hands the decision to the trajectory writer.
void agents_begin_decision(const std::vector<LegalAction> &actions, size_t number);
// A one-row menu the engine took without asking (InputLogger::choose): under --trace-decisions a
// "| auto: Player <X>: <description>" line on the game log; a --play script whose next spec names
// that row consumes the spec (PlayScript::auto_taken).
void agents_auto_decision(const LegalAction &taken);
// After the decision: `choice` the action index taken (the choice line of a traced decision),
// `reply` what the decision log holds for it (a menu row or a concession sentinel), for the
// trajectory writer's record (with the teacher's label, read here when a taught seat's reply was
// not needed earlier).
void agents_end_decision(int choice, int reply);

#endif /* AGENTS_DRIVER_H */
