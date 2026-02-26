#ifndef ACTION_PROCESSOR_H
#define ACTION_PROCESSOR_H

#include <memory>

#include "classes/action.h"
#include "classes/game.h"

// Forward declarations
class Orderer;

// Process a legal action selected by the user
void process_action(const LegalAction& action, Game& game, std::shared_ptr<Orderer> orderer);

// Handle the current mandatory choice (declare attackers, blockers, etc.)
void proc_mandatory_choice(Game& game, std::shared_ptr<Orderer> orderer);

#endif
