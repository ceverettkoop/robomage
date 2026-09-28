#include "object_ref.h"

#include "classes/game.h"
#include "components/zone.h"
#include "ecs/coordinator.h"
#include "game_driver.h"
#include "game_queries.h"

ObjectRef ObjectRef::of(Entity e) {
    ObjectRef r;
    r.e = e;
    r.gen = stamp_object_gen(e);
    return r;
}

Entity ObjectRef::get() const {
    if (e == 0) return 0;
    const bool has_zone = global_coordinator.entity_has_component<Zone>(e);
    if (gen == 0) return has_zone ? 0 : e;
    if (!has_zone) return 0;
    const uint64_t cur = global_coordinator.GetComponent<Zone>(e).obj_gen;
    if (cur == gen) return e;
    if (cur_game.follow_window_open) {
        auto it = cur_game.follow_window_origins.find(e);
        if (it != cur_game.follow_window_origins.end() && gen >= it->second && gen < cur) return e;
    }
    return 0;
}

std::vector<ObjectRef> refs_of(const std::vector<Entity> &entities) {
    std::vector<ObjectRef> out;
    out.reserve(entities.size());
    for (Entity e : entities) out.push_back(ObjectRef::of(e));
    return out;
}

std::vector<Entity> live_entities(const std::vector<ObjectRef> &refs) {
    std::vector<Entity> out;
    out.reserve(refs.size());
    for (const ObjectRef &r : refs)
        if (Entity e = r.get()) out.push_back(e);
    return out;
}

std::vector<Entity> lki_entities(const std::vector<ObjectRef> &refs) {
    std::vector<Entity> out;
    out.reserve(refs.size());
    for (const ObjectRef &r : refs) out.push_back(r.lki_entity());
    return out;
}

std::vector<ObjectRef> restamp_live(const std::vector<ObjectRef> &refs) {
    return refs_of(live_entities(refs));
}

bool refs_contain(const std::vector<ObjectRef> &refs, Entity e) {
    if (e == 0) return false;
    for (const ObjectRef &r : refs)
        if (r.e == e && r.get() == e) return true;
    return false;
}

void erase_refs_to(std::vector<ObjectRef> &refs, Entity e) {
    std::vector<ObjectRef> kept;
    kept.reserve(refs.size());
    for (const ObjectRef &r : refs)
        if (r.e != e) kept.push_back(r);
    refs.swap(kept);
}

size_t ObjectSet::count(Entity e) const {
    auto it = members.find(e);
    return it != members.end() && it->second.get() == e ? 1 : 0;
}

std::vector<Entity> ObjectSet::live() const {
    std::vector<Entity> out;
    for (const auto &kv : members)
        if (kv.second.get() == kv.first) out.push_back(kv.first);
    return out;
}

void open_follow_window() {
    cur_game.follow_window_open = true;
    cur_game.follow_window_origins.clear();
}

void close_follow_window() {
    cur_game.follow_window_open = false;
    cur_game.follow_window_origins.clear();
}

bool follow_window_open() { return cur_game.follow_window_open; }

void note_object_moved(Entity e, uint64_t old_gen) {
    if (!cur_game.follow_window_open || old_gen == 0) return;
    cur_game.follow_window_origins.emplace(e, old_gen);
}
