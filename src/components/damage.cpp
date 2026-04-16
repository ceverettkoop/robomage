#include "damage.h"
#include "creature.h"
#include "../ecs/coordinator.h"

static bool source_has_keyword(Entity source, const char *kw) {
    Coordinator &coord = Coordinator::global();
    if (!coord.entity_has_component<Creature>(source)) return false;
    auto &cr = coord.GetComponent<Creature>(source);
    for (const auto &k : cr.keywords)
        if (k == kw) return true;
    return false;
}

//returns "was damage dealt this way"
bool deal_damage(Entity source, Entity target, size_t amount) {

    Coordinator& coordinator = Coordinator::global();

    //does nothing if target cannot be damaged and/or no longer is valid
    if(!coordinator.entity_has_component<Damage>(target)){
        return false;
    }

    if (coordinator.entity_has_component<Creature>(target)) {
        if (has_protection_from(coordinator.GetComponent<Creature>(target), source))
            return false;
    }

    auto &dmg = coordinator.GetComponent<Damage>(target);
    dmg.damage_counters += amount;
    // Deathtouch flag propagates per-creature until cleanup clears damage.
    if (amount > 0 && source_has_keyword(source, "Deathtouch"))
        dmg.has_deathtouch_damage = true;
    return true;
}