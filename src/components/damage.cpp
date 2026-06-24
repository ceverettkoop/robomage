#include "damage.h"
#include "creature.h"
#include "permanent.h"
#include "player.h"
#include "zone.h"
#include "../cli_output.h"
#include "../ecs/coordinator.h"
#include "../mana_system.h"

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

bool deal_damage_to_player(Entity source, Entity player_entity, size_t amount) {
    Coordinator &coord = Coordinator::global();
    if (!coord.entity_has_component<Player>(player_entity)) return false;

    auto &player = coord.GetComponent<Player>(player_entity);
    player.life_total -= static_cast<int32_t>(amount);

    // Lifelink: the source's controller gains life equal to the damage dealt.
    if (amount > 0 && source_has_keyword(source, "Lifelink") &&
        coord.entity_has_component<Permanent>(source)) {
        Zone::Ownership ctrl = coord.GetComponent<Permanent>(source).controller;
        Entity ctrl_entity = get_player_entity(ctrl);
        if (coord.entity_has_component<Player>(ctrl_entity)) {
            coord.GetComponent<Player>(ctrl_entity).life_total += static_cast<int32_t>(amount);
            game_log("%s gains %zu life (lifelink)\n", player_name(ctrl).c_str(), amount);
        }
    }
    return true;
}