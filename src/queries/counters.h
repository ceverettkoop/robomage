#ifndef QUERIES_COUNTERS_H
#define QUERIES_COUNTERS_H

#include <string>
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"

// ── Counters (122.1) ────────────────────────────────────────────────────────
// Every counter kind (+1/+1, -1/-1, loyalty, keyword) lives in Permanent::counters
// keyed by type; these helpers are the single add/remove/query path (T2.4).

// True if a counter type names a keyword ability, so a counter of that type grants the
// keyword to its permanent (CR 122.1d: keyword counters). The names match the keyword
// strings the engine checks via creature_has_keyword (e.g. "Flying", "Trample"). Single
// source so the keyword-counter rebuild and any future "remove keyword counter" share it.
bool is_keyword_counter_type(const std::string &type);

// Number of counters of `type` on `e` (0 if none, or `e` is not a permanent).
inline int get_counters(Entity e, const std::string &type) {
    if (!global_coordinator.entity_has_component<Permanent>(e)) return 0;
    auto &perm = global_coordinator.GetComponent<Permanent>(e);
    auto it = perm.counters.find(type);
    return it == perm.counters.end() ? 0 : it->second;
}

// Recompute a creature's cached +1/+1 − -1/-1 P/T contribution from its counters
// (layer 7c, 613.4c) and refresh effective P/T. No-op if `e` is not a creature.
void refresh_counter_pt(Entity e);

// Add `delta` counters of `type` to `e` (delta may be negative). An entry reaching
// exactly 0 is erased so the map only holds live counters. +1/+1 and -1/-1 changes
// resync the creature's P/T. Returns the new total; no-op (returns current) if not a permanent.
int add_counters(Entity e, const std::string &type, int delta);

// Counters on a card in exile: its suspend time counters plus a void counter (Dauthi
// Voidwalker). An exiled card is not a permanent, so these are on its Zone (Zone::counters)
// rather than Permanent::counters. Defined in counters.cpp.
int exiled_card_counters(Entity card);

// Number of `type` counters on an object (CR 122.1): a battlefield permanent's
// Permanent::counters, or an exiled card's VOID / TIME counters as exiled_card_counters tracks
// them. 0 for any other object. Defined in counters.cpp.
int object_counters(Entity e, const std::string &type);

#endif /* QUERIES_COUNTERS_H */
