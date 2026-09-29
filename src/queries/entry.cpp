#include "entry.h"

#include "../components/carddata.h"
#include "../components/entry_info.h"
#include "../ecs/coordinator.h"
#include "characteristics.h"

const CardData &entering_face(Entity e, const CardData &cd) {
    const EntryInfo *entry = find_entry_info(e);
    if (cd.backside && entry && entry->enters_transformed) return *cd.backside;
    return active_face(e, cd);
}

EntryInfo &entry_info(Entity e) {
    if (!global_coordinator.entity_has_component<EntryInfo>(e))
        global_coordinator.AddComponent(e, EntryInfo{});
    return global_coordinator.GetComponent<EntryInfo>(e);
}

EntryInfo *find_entry_info(Entity e) {
    if (!global_coordinator.entity_has_component<EntryInfo>(e)) return nullptr;
    return &global_coordinator.GetComponent<EntryInfo>(e);
}

void drop_entry_info(Entity e) {
    if (global_coordinator.entity_has_component<EntryInfo>(e))
        global_coordinator.RemoveComponent<EntryInfo>(e);
}

void drop_entry_info_if_consumed(Entity e) {
    const EntryInfo *entry = find_entry_info(e);
    if (entry && entry->consumed()) global_coordinator.RemoveComponent<EntryInfo>(e);
}
