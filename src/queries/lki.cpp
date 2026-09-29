#include "lki.h"

#include "../classes/game.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"

// Look up a leaving-the-battlefield snapshot, if one was captured for `e` (declared in
// lki.h).
const LastKnownInfo *lki_for(Entity e) {
    const LastKnownInfo *lki = departed_lki_for(e);
    return (lki && !lki->superseded) ? lki : nullptr;
}

const LastKnownInfo *departed_lki_for(Entity e) {
    auto it = cur_game.last_known_info.find(e);
    if (it == cur_game.last_known_info.end()) return nullptr;
    // Captured for an earlier holder of the id (the id was destroyed and issued again).
    if (it->second.issue != global_coordinator.GetIssueCount(e)) return nullptr;
    return &it->second;
}

void supersede_last_known_info(Entity e) {
    auto it = cur_game.last_known_info.find(e);
    if (it != cur_game.last_known_info.end()) it->second.superseded = true;
}

void supersede_departed_cards() {
    for (auto &kv : cur_game.last_known_info)
        if (!kv.second.is_token) kv.second.superseded = true;
}

uint64_t stamp_object_gen(Entity e) {
    if (e == 0 || !global_coordinator.entity_has_component<Zone>(e)) return 0;
    auto &z = global_coordinator.GetComponent<Zone>(e);
    if (z.obj_gen == 0) z.obj_gen = cur_game.identity.next_obj_gen++;
    return z.obj_gen;
}

std::string last_known_name(Entity e) {
    const LastKnownInfo *lki = lki_for(e);
    if (!lki || lki->name.empty()) return "";
    return lki->is_token ? lki->name + " token" : lki->name;
}
