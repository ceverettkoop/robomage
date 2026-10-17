#include "handles.h"

#include <algorithm>
#include <string>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../ecs/coordinator.h"
#include "../game_driver.h"
#include "../queries/characteristics.h"
#include "../queries/visibility.h"
#include "../queries/zones.h"

using Table = Game::HandleTable;
using Entry = Game::HandleTable::Entry;

// The viewers in the order every per-viewer loop (and the trace) visits them.
static const Zone::Ownership VIEWERS[] = {Zone::PLAYER_A, Zone::PLAYER_B, Zone::UNKNOWN};

static Zone::Ownership viewer_key(Zone::Ownership viewer);
static Table &table_of(Zone::Ownership viewer);
static bool entry_current(const Entry &en, Entity e);
static bool entry_live(const Entry &en, Entity e, Zone::Ownership viewer);
static uint32_t issue(Zone::Ownership viewer, Entity e, bool narrate);
static void retire(Zone::Ownership viewer, Entity e, const char *reason);
static void retire_in_handle_order(Zone::Ownership viewer, std::vector<Entity> objects, const char *reason);
static void sweep(Zone::Ownership viewer);
static std::string describe(Entity e, Zone::Ownership viewer);
static const char *viewer_word(Zone::Ownership viewer);

// Seats are their own viewers; every other value is the omniscient one.
static Zone::Ownership viewer_key(Zone::Ownership viewer) {
    return viewer == Zone::PLAYER_A || viewer == Zone::PLAYER_B ? viewer : Zone::UNKNOWN;
}

static Table &table_of(Zone::Ownership viewer) {
    return cur_game.handle_tables[HandleMove::slot(viewer)];
}

// Whether `en` still names `e` as the object it was issued for: the same holder of the id and the
// exact identity stamp (not ObjectRef::get, whose CR 400.7j follow window would carry the handle
// over to the object a move made).
static bool entry_current(const Entry &en, Entity e) {
    if (en.obj.lki_entity() != e || !global_coordinator.entity_has_component<Zone>(e)) return false;
    return global_coordinator.GetComponent<Zone>(e).obj_gen == en.obj.gen;
}

static bool entry_live(const Entry &en, Entity e, Zone::Ownership viewer) {
    return entry_current(en, e) && individually_identifiable(e, viewer);
}

// A fresh handle for `e` (which holds no live one); the issue is narrated unless the caller
// narrates it itself.
static uint32_t issue(Zone::Ownership viewer, Entity e, bool narrate) {
    Table &t = table_of(viewer);
    Entry en;
    en.obj = ObjectRef::of(e);
    en.handle = t.next++;
    t.live[e] = en;
    if (narrate && trace_handles)
        game_log("HANDLE: issue %s h%u %s\n", viewer_word(viewer), en.handle, describe(e, Zone::UNKNOWN).c_str());
    return en.handle;
}

// Drops `viewer`'s handle for `e`, if they hold one; it is never issued again.
static void retire(Zone::Ownership viewer, Entity e, const char *reason) {
    Table &t = table_of(viewer);
    auto it = t.live.find(e);
    if (it == t.live.end()) return;
    if (trace_handles) {
        const std::string what = entry_current(it->second, e) ? describe(e, Zone::UNKNOWN) + " " : std::string();
        game_log("HANDLE: retire %s h%u %s(%s)\n", viewer_word(viewer), it->second.handle, what.c_str(), reason);
    }
    t.live.erase(it);
}

// Retires `viewer`'s handles for `objects` in ascending handle order, so the order the objects
// arrive in (entity ids, window order) never shows.
static void retire_in_handle_order(Zone::Ownership viewer, std::vector<Entity> objects, const char *reason) {
    const Table &t = table_of(viewer);
    std::sort(
        objects.begin(), objects.end(), [&t](Entity a, Entity b) { return t.live.at(a).handle < t.live.at(b).handle; });
    for (Entity e : objects) retire(viewer, e, reason);
}

// Retires every handle of `viewer` whose object is gone or that the viewer can no longer identify.
static void sweep(Zone::Ownership viewer) {
    std::vector<Entity> gone;
    std::vector<Entity> hidden;
    for (const auto &kv : table_of(viewer).live) {
        if (!entry_current(kv.second, kv.first))
            gone.push_back(kv.first);
        else if (!individually_identifiable(kv.first, viewer))
            hidden.push_back(kv.first);
    }
    retire_in_handle_order(viewer, gone, "gone");
    retire_in_handle_order(viewer, hidden, "unidentifiable");
}

// `e` as `viewer` sees it, for the trace: zone, owner, library position if any, and its name, or
// "face-down" when its identity is hidden from the viewer.
static std::string describe(Entity e, Zone::Ownership viewer) {
    if (!global_coordinator.entity_has_component<Zone>(e)) return "(gone)";
    const Zone &z = global_coordinator.GetComponent<Zone>(e);
    std::string out = std::string(trace_zone_word(z.location)) + ":" + trace_seat_word(z.owner);
    if (z.location == Zone::LIBRARY) out += "@" + std::to_string(z.distance_from_top);
    out += " ";
    out += identity_visible(e, viewer) ? entity_name(e) : std::string("face-down");
    if (z.is_face_down && viewer_key(viewer) == Zone::UNKNOWN) out += " (face down)";
    return out;
}

static const char *viewer_word(Zone::Ownership viewer) {
    const Zone::Ownership v = viewer_key(viewer);
    return v == Zone::PLAYER_A ? "A" : v == Zone::PLAYER_B ? "B" : "omni";
}

uint32_t handle_of(Zone::Ownership viewer, Entity e) {
    Table &t = table_of(viewer);
    auto it = t.live.find(e);
    if (it != t.live.end()) {
        if (entry_live(it->second, e, viewer)) return it->second.handle;
        retire(viewer, e, entry_current(it->second, e) ? "unidentifiable" : "gone");
    }
    if (!individually_identifiable(e, viewer)) return 0;
    return issue(viewer, e, true);
}

uint32_t handle_peek(Zone::Ownership viewer, Entity e) {
    const Table &t = table_of(viewer);
    auto it = t.live.find(e);
    return it != t.live.end() && entry_live(it->second, e, viewer) ? it->second.handle : 0;
}

void issue_handles_canonical(Zone::Ownership viewer) {
    sweep(viewer);
    const Table &t = table_of(viewer);
    std::vector<Entity> fresh;
    for (Entity e : zoned_entities())
        if (t.live.find(e) == t.live.end() && individually_identifiable(e, viewer)) fresh.push_back(e);
    for (Entity e : canonical_object_order(fresh, viewer)) issue(viewer, e, true);
}

std::vector<HandleRow> handle_table(Zone::Ownership viewer) {
    std::vector<HandleRow> rows;
    for (const auto &kv : table_of(viewer).live) {
        if (!entry_live(kv.second, kv.first, viewer)) continue;
        HandleRow r;
        r.handle = kv.second.handle;
        r.entity = kv.first;
        rows.push_back(r);
    }
    std::sort(rows.begin(), rows.end(), [](const HandleRow &a, const HandleRow &b) { return a.handle < b.handle; });
    return rows;
}

HandleMove handles_begin_move(Entity e) {
    HandleMove move;
    for (Zone::Ownership viewer : VIEWERS) {
        const Table &t = table_of(viewer);
        auto it = t.live.find(e);
        if (it == t.live.end()) continue;
        if (entry_live(it->second, e, viewer)) move.before_handle[HandleMove::slot(viewer)] = it->second.handle;
        retire(viewer, e, "zone change");
    }
    return move;
}

void handles_end_move(HandleMove &move, Entity e) {
    for (Zone::Ownership viewer : VIEWERS) {
        const uint32_t before = move.before(viewer);
        if (before == 0 || !individually_identifiable(e, viewer)) continue;
        uint32_t after = handle_peek(viewer, e);
        if (after == 0) {
            retire(viewer, e, "gone");
            after = issue(viewer, e, false);
        }
        move.after_handle[HandleMove::slot(viewer)] = after;
        if (trace_handles)
            game_log("HANDLE: link %s h%u -> h%u %s\n", viewer_word(viewer), before, after,
                describe(e, Zone::UNKNOWN).c_str());
    }
}

void retire_object_handles(Entity e) {
    for (Zone::Ownership viewer : VIEWERS) retire(viewer, e, "ceased to exist");
}

void retire_unidentifiable_handles(const std::vector<Entity> &cards) {
    for (Zone::Ownership viewer : VIEWERS) {
        const Table &t = table_of(viewer);
        std::vector<Entity> lost;
        for (Entity card : cards) {
            auto it = t.live.find(card);
            if (it != t.live.end() && !entry_live(it->second, card, viewer) &&
                std::find(lost.begin(), lost.end(), card) == lost.end())
                lost.push_back(card);
        }
        retire_in_handle_order(viewer, lost, "window closed");
    }
}

void trace_handle_tables() {
    for (Zone::Ownership viewer : VIEWERS) {
        issue_handles_canonical(viewer);
        std::string line = std::string("HANDLES: ") + viewer_word(viewer);
        for (const HandleRow &r : handle_table(viewer))
            line += " [h" + std::to_string(r.handle) + " " + describe(r.entity, viewer) + "]";
        game_log("%s\n", line.c_str());
    }
}
