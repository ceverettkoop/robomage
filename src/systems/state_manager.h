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

class Orderer;
class StackManager;

class StateManager : public System {

public:
    static void init();
    void state_based_effects(Game& game, std::shared_ptr<Orderer> orderer);
    std::vector<LegalAction> determine_legal_actions(const Game& game, std::shared_ptr<Orderer> orderer,
                                                      std::shared_ptr<StackManager> stack_manager);

};

#endif /* STATE_MANAGER_H */
