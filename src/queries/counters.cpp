#include "counters.h"

#include <algorithm>
#include <set>

#include "../components/creature.h"
#include "../components/zone.h"
#include "battlefield.h"

bool is_keyword_counter_type(const std::string &type) {
    static const std::set<std::string> kKeywordCounters = {
        "Flying", "First Strike", "Double Strike", "Deathtouch", "Haste", "Hexproof",
        "Indestructible", "Lifelink", "Menace", "Reach", "Trample", "Vigilance",
        "Shadow", "Skulk", "Fear", "Intimidate", "Horsemanship", "Infect", "Wither",
        "Toxic", "Defender", "Flash", "Persist", "Undying", "Decayed"};
    return kKeywordCounters.count(type) != 0;
}

void refresh_counter_pt(Entity e) {
    if (!global_coordinator.entity_has_component<Creature>(e)) return;
    auto &cr = global_coordinator.GetComponent<Creature>(e);
    cr.counter_pt_bonus = get_counters(e, "P1P1") - get_counters(e, "M1M1");
    recompute_pt(cr);
}

int add_counters(Entity e, const std::string &type, int delta) {
    if (delta == 0 || !global_coordinator.entity_has_component<Permanent>(e))
        return get_counters(e, type);
    auto &perm = global_coordinator.GetComponent<Permanent>(e);
    int total = perm.counters[type] + delta;
    if (total == 0) perm.counters.erase(type);
    else perm.counters[type] = total;
    if (type == "P1P1" || type == "M1M1") refresh_counter_pt(e);
    return total;
}

int exiled_card_counters(Entity card) {
    int n = 0;
    for (const auto &c : global_coordinator.GetComponent<Zone>(card).counters)
        if (c.second > 0) n += c.second;
    return n;
}

int object_counters(Entity e, const std::string &type) {
    if (is_battlefield_permanent(e)) return get_counters(e, type);
    if (!global_coordinator.entity_has_component<Zone>(e) ||
        global_coordinator.GetComponent<Zone>(e).location != Zone::EXILE)
        return 0;
    const CounterMap &counters = global_coordinator.GetComponent<Zone>(e).counters;
    auto it = counters.find(type);
    return it == counters.end() ? 0 : std::max(it->second, 0);
}
