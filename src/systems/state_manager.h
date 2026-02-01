#ifndef STATE_MANAGER_H
#define STATE_MANAGER_H

#include "../ecs/system.h"
#include "../ecs/entity.h"
#include "../components/zone.h"
#include "../classes/action.h"
#include <vector>
#include <memory>

struct Deck;
struct Game;

class StateManager : public System {

public:
    static void init();
    void state_based_effects(Game& game);
    std::vector<LegalAction> determine_legal_actions(const Game& game);

};

#endif /* STATE_MANAGER_H */
