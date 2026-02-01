#ifndef CLI_H
#define CLI_H

#include "classes/game.h"
#include "systems/state_manager.h"
#include <memory>

void get_mandatory_choice_input(Game& cur_game);
void get_user_action_input(Game& cur_game, std::shared_ptr<StateManager> state_manager);

#endif /* CLI_H */
