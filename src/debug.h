#ifndef DEBUG_H
#define DEBUG_H

#include <vector>
#include "components/zone.h"
#include "ecs/entity.h"
#include "systems/orderer.h"
#include "systems/state_manager.h"
#include "classes/game.h"

void print_library(std::shared_ptr<Orderer> orderer, Zone::Ownership owner);
void print_hand(std::shared_ptr<Orderer> orderer, Zone::Ownership owner);
void print_stack(std::shared_ptr<Orderer> orderer);
void print_step(const Game& cur_game);
void print_legal_actions(const Game& cur_game, std::vector<LegalAction> legal_actions);
void print_mandatory_choice_description(const Game& cur_game);
void print_battlefield(std::shared_ptr<Orderer> orderer);
void print_mana_pools();
std::string player_name(Zone::Ownership owner);
void print_draw(Zone::Ownership player, const std::vector<Entity>& cards);


#endif /* DEBUG_H */
