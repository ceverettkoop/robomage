#ifndef ACTION_PROCESSOR_H
#define ACTION_PROCESSOR_H

#include <memory>
#include <vector>

#include "classes/action.h"
#include "classes/game.h"
#include "components/ability.h"
#include "ecs/entity.h"

// Forward declarations
class Orderer;

// Process a legal action selected by the user
void process_action(const LegalAction& action, Game& game, std::shared_ptr<Orderer> orderer);

// Handle the current mandatory choice (declare attackers, blockers, etc.)
void proc_mandatory_choice(Game& game, std::shared_ptr<Orderer> orderer);

// Returns true if the ability has no targeting requirement or at least one legal target exists.
bool has_legal_targets(const Ability& ability, std::shared_ptr<Orderer> orderer);

// Prompts the active player to choose a target and sets ability.target.
// Caller must ensure has_legal_targets() is true before calling.
void select_target(Ability& ability, std::shared_ptr<Orderer> orderer);

#endif
