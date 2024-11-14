#ifndef GAMEOBJECTDB_H
#define GAMEOBJECTDB_H

#include "entity.h"
#include <vector>

struct GameObjects{
    std::vector<EntityID> otp_library;
    std::vector<EntityID> otd_library;
    std::vector<EntityID> otp_hand;
    std::vector<EntityID> otd_hand;
    std::vector<EntityID> battlefield;
    std::vector<EntityID> stack;
    std::vector<EntityID> otp_graveyard;
    std::vector<EntityID> otd_graveyard;
    std::vector<EntityID> exile;
    std::vector<EntityID> otp_sideboard;
    std::vector<EntityID> otd_sideboard;
};

#endif /* GAMEOBJECTDB_H */
