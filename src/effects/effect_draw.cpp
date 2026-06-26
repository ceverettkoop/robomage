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
    // A Draw with no NumCards$ draws a single card (Forge default), e.g. Kozilek's
    // Command's "then draws a card" rider (DB$ Draw | Defined$ ParentTarget).
    size_t count = ab.amount > 0 ? ab.amount : 1;
    orderer->draw(owner, count);
    return true;
}

}  // namespace effects
