#ifndef STATE_MANAGER_H
#define STATE_MANAGER_H

#include "../ecs/system.h"
#include "../ecs/entity.h"
#include "../components/zone.h"
#include <vector>
#include <memory>

struct Deck;

class StateManager : public System {

public:
    static void init();
    void state_based_effects();

};

#endif /* STATE_MANAGER_H */
