// bin/<config>/robomage_nn: the engine with the nn: agent (plan step 5d, src/actor/nn_agent.h).
//
// The engine's own entry (engine_main.h) after the actor's agent kind is registered: every engine
// flag, mode and output is the plain binary's, and --agent-a / --agent-b accept nn:<model>[:…].

#include "../engine_main.h"
#include "nn_agent.h"

int main(int argc, char const *argv[]) {
    actor::register_nn_agent();
    return engine_main(argc, argv);
}
