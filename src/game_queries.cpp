#include "game_queries.h"

#include "classes/game.h"

extern Coordinator global_coordinator;

// The effective_* accessors implement CR 608.2h: use the object's current information while it
// is in the zone it is expected to be in (the battlefield, for a permanent's continuous-effect-
// and counter-modified characteristics), otherwise its last-known information. They live here
// rather than inline in game_queries.h so the header does not need to depend on game.h (and the
// cur_game.last_known_info store).

// Look up a leaving-the-battlefield snapshot, if one was captured for `e`.
static const LastKnownInfo *lki_for(Entity e) {
    auto it = cur_game.last_known_info.find(e);
    return it == cur_game.last_known_info.end() ? nullptr : &it->second;
}

int effective_power(Entity e) {
    if (is_battlefield_permanent(e) && global_coordinator.entity_has_component<Creature>(e))
        return static_cast<int>(global_coordinator.GetComponent<Creature>(e).power);
    if (const LastKnownInfo *lki = lki_for(e)) return lki->power;
    if (global_coordinator.entity_has_component<CardData>(e))
        return static_cast<int>(global_coordinator.GetComponent<CardData>(e).power);
    return 0;
}

int effective_toughness(Entity e) {
    if (is_battlefield_permanent(e) && global_coordinator.entity_has_component<Creature>(e))
        return static_cast<int>(global_coordinator.GetComponent<Creature>(e).toughness);
    if (const LastKnownInfo *lki = lki_for(e)) return lki->toughness;
    if (global_coordinator.entity_has_component<CardData>(e))
        return static_cast<int>(global_coordinator.GetComponent<CardData>(e).toughness);
    return 0;
}

std::set<Colors> effective_colors(Entity e) {
    // On the battlefield, effective color is the layer-5 (613.1e) result. No current-vocab card
    // changes color, so this reads the printed colors today; this is the single seam a future
    // layer-5 color-changing effect plugs into without touching any consumer.
    if (is_battlefield_permanent(e) && global_coordinator.entity_has_component<CardData>(e))
        return card_colors(global_coordinator.GetComponent<CardData>(e));
    if (const LastKnownInfo *lki = lki_for(e)) return lki->colors;
    if (global_coordinator.entity_has_component<CardData>(e))
        return card_colors(global_coordinator.GetComponent<CardData>(e));
    return {};
}
