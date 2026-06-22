#include "effects.h"

#include "../classes/game.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;

namespace effects {

bool draw(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // "Target player draws" (e.g. Deep Analysis) draws for the chosen target
    // player; otherwise the effect's controller (source owner) draws.
    Zone::Ownership owner;
    if (ab.target != 0 && global_coordinator.entity_has_component<Player>(ab.target))
        owner = (ab.target == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
    else
        owner = global_coordinator.GetComponent<Zone>(ab.source).owner;
    orderer->draw(owner, ab.amount);
    return true;
}

}  // namespace effects
