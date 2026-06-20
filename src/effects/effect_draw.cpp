#include "effects.h"

#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;

namespace effects {

bool draw(Ability &ab, std::shared_ptr<Orderer> orderer) {
    Zone::Ownership owner = global_coordinator.GetComponent<Zone>(ab.source).owner;
    orderer->draw(owner, ab.amount);
    return true;
}

}  // namespace effects
