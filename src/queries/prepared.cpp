#include "prepared.h"

#include <vector>

#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/prepared_copy.h"
#include "../components/token.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../systems/orderer.h"
#include "battlefield.h"

static const CardData *prepare_spell_of(Entity permanent);
static void make_prepared_copy(Entity permanent, Permanent &perm, const std::shared_ptr<Orderer> &orderer);
static bool copy_goes(Entity copy, Entity permanent);

// CR 722.2b: a card's second set of characteristics, or the one a token copy of such a card has.
static const CardData *prepare_spell_of(Entity permanent) {
    if (global_coordinator.entity_has_component<CardData>(permanent)) {
        const CardData &cd = global_coordinator.GetComponent<CardData>(permanent);
        return cd.is_prepare ? cd.backside.get() : nullptr;
    }
    if (global_coordinator.entity_has_component<Token>(permanent))
        return global_coordinator.GetComponent<Token>(permanent).prepare_spell.get();
    return nullptr;
}

static void make_prepared_copy(Entity permanent, Permanent &perm, const std::shared_ptr<Orderer> &orderer) {
    const CardData *spell = prepare_spell_of(permanent);
    if (spell == nullptr) return;
    const Entity copy = global_coordinator.CreateEntity();
    global_coordinator.AddComponent(copy, CardData(*spell));
    global_coordinator.AddComponent(copy, PreparedCopy{ObjectRef::of(permanent)});
    orderer->place_created_in_exile(copy, perm.controller);
    perm.prepared_copy = ObjectRef::of(copy);
    game_log("%s is prepared: a copy of %s is in exile.\n", perm.name.c_str(), spell->name.c_str());
}

static bool copy_goes(Entity copy, Entity permanent) {
    const Zone::ZoneValue zone = global_coordinator.GetComponent<Zone>(copy).location;
    if (zone == Zone::STACK) return false;  // being cast: nothing here decides it until it is cast
    return zone != Zone::EXILE || !is_battlefield_permanent(permanent) ||
           !global_coordinator.GetComponent<Permanent>(permanent).prepared;
}

void sync_prepared_copies(const EntitySet &entities, const std::shared_ptr<Orderer> &orderer) {
    std::vector<Entity> gone;
    std::vector<Entity> needing;
    for (Entity e : entities) {
        if (global_coordinator.entity_has_component<PreparedCopy>(e) &&
            copy_goes(e, global_coordinator.GetComponent<PreparedCopy>(e).permanent.get()))
            gone.push_back(e);
        if (!is_battlefield_permanent(e)) continue;
        const Permanent &perm = global_coordinator.GetComponent<Permanent>(e);
        if (perm.prepared && perm.prepared_copy.empty()) needing.push_back(e);
    }
    for (Entity copy : gone) {
        const Entity permanent = global_coordinator.GetComponent<PreparedCopy>(copy).permanent.get();
        // Still prepared, only phased out: it makes a copy again as it phases in (CR 722.3c).
        if (permanent != 0 && global_coordinator.entity_has_component<Permanent>(permanent) &&
            global_coordinator.GetComponent<Zone>(copy).location == Zone::EXILE)
            global_coordinator.GetComponent<Permanent>(permanent).prepared_copy = ObjectRef{};
        game_log("%s ceases to exist.\n", global_coordinator.GetComponent<CardData>(copy).name.c_str());
        orderer->remove_created_object(copy);
    }
    for (Entity e : needing) make_prepared_copy(e, global_coordinator.GetComponent<Permanent>(e), orderer);
}

void prepared_copy_cast(Entity card) {
    if (!global_coordinator.entity_has_component<PreparedCopy>(card)) return;
    const Entity permanent = global_coordinator.GetComponent<PreparedCopy>(card).permanent.get();
    if (permanent != 0 && global_coordinator.entity_has_component<Permanent>(permanent)) {
        Permanent &perm = global_coordinator.GetComponent<Permanent>(permanent);
        perm.prepared = false;
        perm.prepared_copy = ObjectRef{};
        game_log("%s is no longer prepared.\n", perm.name.c_str());
    }
    global_coordinator.RemoveComponent<PreparedCopy>(card);
}
