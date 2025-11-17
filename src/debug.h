#ifndef DEBUG_H
#define DEBUG_H

#include "components/zone.h"
#include "systems/orderer.h"
#include "classes/game.h"

void print_library(std::shared_ptr<Orderer> orderer, Zone::Ownership owner);
void print_hand(std::shared_ptr<Orderer> orderer, Zone::Ownership owner);
void print_stack(std::shared_ptr<Orderer> orderer);
void print_step(const Game& cur_game);
std::string player_name(Zone::Ownership owner);


#endif /* DEBUG_H */
