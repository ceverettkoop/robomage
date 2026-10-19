#ifndef ENGINE_MAIN_H
#define ENGINE_MAIN_H

// ── The engine's process entry: the flags, the no-game modes, then the match ──────────────────
//
// main() of bin/<config>/robomage (src/main.cpp) is this alone. A binary that adds to the engine
// links every engine object but main.o and calls it from its own main() after its additions: the
// actor's bin/<config>/robomage_nn (src/actor/robomage_nn.cpp) registers the nn: agent kind
// (agents/agent.h, register_agent_kind) first.

// Parses the command line and runs what it asks for; the process exit status.
int engine_main(int argc, char const *argv[]);

#endif /* ENGINE_MAIN_H */
