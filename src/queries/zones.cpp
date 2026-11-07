#include "zones.h"

#include <string>

#include "../classes/game.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../error.h"
#include "../object_ref.h"
#include "../systems/rules_modifying.h"
#include "characteristics.h"
#include "permissions.h"
#include "players.h"
#include "types.h"
#include "visibility.h"

// The set zoned_entities() returns: a system's mEntities, owned by the ECS. A snapshot restore
// assigns into that same set, so the binding holds until the next init_ecs rebinds it.
static const EntitySet *g_zoned_entities = nullptr;

const EntitySet &zoned_entities() {
    if (!g_zoned_entities) fatal_error("zoned_entities: read before init_ecs bound the set");
    return *g_zoned_entities;
}

void bind_zoned_entities(const EntitySet &entities) {
    g_zoned_entities = &entities;
}

std::vector<Entity> zone_objects(const EntitySet &entities, Zone::ZoneValue zone, Zone::Ownership owner) {
    std::vector<Entity> out;
    for (Entity e : entities) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        const Zone &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location == zone && (owner == Zone::UNKNOWN || z.owner == owner)) out.push_back(e);
    }
    return out;
}

Entity library_top(Zone::Ownership owner) {
    for (Entity e : zoned_entities()) {
        const Zone &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location == Zone::LIBRARY && z.owner == owner && z.distance_from_top == 0) return e;
    }
    return 0;
}

int graveyard_card_types(Zone::Ownership owner, const EntitySet &entities, Entity except) {
    std::set<std::string> type_names;
    for (auto entity : entities) {
        if (entity == except) continue;
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(entity);
        if (z.location != Zone::GRAVEYARD || z.owner != owner) continue;
        if (!global_coordinator.entity_has_component<CardData>(entity)) continue;
        for (auto &t : effective_types(entity))
            if (t.kind == TYPE) type_names.insert(t.name);
    }
    return static_cast<int>(type_names.size());
}

int graveyard_card_count(Zone::Ownership owner, const EntitySet &entities, Entity except) {
    int n = 0;
    for (auto entity : entities) {
        if (entity == except) continue;
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(entity);
        if (z.location != Zone::GRAVEYARD || z.owner != owner) continue;
        if (!global_coordinator.entity_has_component<CardData>(entity)) continue;
        n++;
    }
    return n;
}

Entity exiled_with_card(Entity source) {
    if (source == 0 || !global_coordinator.entity_has_component<Permanent>(source)) return 0;
    const auto &ew = global_coordinator.GetComponent<Permanent>(source).exiled_with;
    for (auto it = ew.rbegin(); it != ew.rend(); ++it)
        if (Entity card = it->get()) return card;
    return 0;
}

Entity returnable_exiled_card(Entity host) {
    if (host == 0 || !global_coordinator.entity_has_component<Permanent>(host)) return 0;
    const auto &exiled = global_coordinator.GetComponent<Permanent>(host).exiled_with;
    // Most-recent-first: cards are push_back'd as they are exiled, so the last entry is the
    // most recently exiled card. Return the first one that still has a live return path.
    for (auto it = exiled.rbegin(); it != exiled.rend(); ++it) {
        // The card must still be the object that was exiled, sitting in the exile zone: one that
        // left exile, or a ceased token whose id was reissued, is no longer "exiled with" it.
        Entity card = it->get();
        if (card == 0) continue;
        if (global_coordinator.GetComponent<Zone>(card).location != Zone::EXILE) continue;
        for (const auto &dt : cur_game.delayed_triggers) {
            const Ability &fa = dt.ability;
            // A return path is a ChangeZone that would pull the card OUT of exile.
            if (fa.def->kind != EffectKind::ChangeZone) continue;
            if (fa.def->origin != Zone::EXILE || fa.def->destination == Zone::EXILE) continue;
            // Shape A tags the card in the fire ability's restore_remembered_exiled_with (and
            // watches the host); shape B tags it in the trigger's remembered_objects (and mirrors
            // it into restore_remembered_exiled_with). Either reference means this card returns.
            if (refs_contain(fa.restore_remembered_exiled_with, card) ||
                refs_contain(dt.remembered_objects, card))
                return card;
        }
    }
    return 0;
}

bool interchangeable_cards(Entity a, Entity b) {
    if (a == b) return true;
    for (Entity e : {a, b})
        if (!global_coordinator.entity_has_component<Zone>(e) ||
            !global_coordinator.entity_has_component<CardData>(e))
            return false;
    const Zone &za = global_coordinator.GetComponent<Zone>(a);
    const Zone &zb = global_coordinator.GetComponent<Zone>(b);
    if (za.location != zb.location || za.owner != zb.owner ||
        global_coordinator.GetComponent<CardData>(a).name !=
            global_coordinator.GetComponent<CardData>(b).name)
        return false;
    if (za.location == Zone::GRAVEYARD) {
        for (Zone::Ownership player : {Zone::PLAYER_A, Zone::PLAYER_B}) {
            PlayPermissions pa = play_permissions(a, player);
            PlayPermissions pb = play_permissions(b, player);
            if (pa.sources != pb.sources || pa.expires_this_turn != pb.expires_this_turn)
                return false;
        }
        return true;
    }
    if (za.location == Zone::LIBRARY) {
        for (Zone::Ownership seat : {Zone::PLAYER_A, Zone::PLAYER_B})
            if (identity_visible(a, seat) != identity_visible(b, seat)) return false;
        return true;
    }
    return false;
}
