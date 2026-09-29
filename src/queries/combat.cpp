#include "combat.h"

#include "../classes/game.h"
#include "damage.h"

// See declaration in combat.h.
bool is_unblocked_attacker(Entity e) {
    if (!cur_game.combat.blockers_declared || cur_game.turn_state.step < DECLARE_BLOCKERS ||
        cur_game.turn_state.step > END_OF_COMBAT)
        return false;
    return is_attacking_creature(e) && !global_coordinator.GetComponent<Creature>(e).is_blocked;
}

std::vector<Entity> blockers_of(Entity attacker, const std::set<Entity> &entities) {
    std::vector<Entity> out;
    for (auto b : entities)
        if (is_blocking_creature(b) &&
            global_coordinator.GetComponent<Creature>(b).blocking_target.get() == attacker)
            out.push_back(b);
    return out;
}

uint32_t lethal_needed_for_blocker(Entity attacker, Entity blocker) {
    const Creature &acr = global_coordinator.GetComponent<Creature>(attacker);
    const Creature &bcr = global_coordinator.GetComponent<Creature>(blocker);
    if (creature_has_keyword(acr, "Deathtouch")) return bcr.toughness > 0 ? 1u : 0u;
    uint32_t marked = marked_damage_on(blocker);
    return (bcr.toughness > marked) ? bcr.toughness - marked : 0u;
}
