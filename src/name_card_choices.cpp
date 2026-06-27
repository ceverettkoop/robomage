#include "name_card_choices.h"

#include <algorithm>
#include <string>

#include "card_vocab.h"
#include "components/carddata.h"
#include "components/zone.h"
#include "ecs/coordinator.h"

extern Coordinator global_coordinator;

// True if the card has the Land card type.
static bool card_is_land(const CardData &cd) {
    for (auto &t : cd.types)
        if (t.name == "Land") return true;
    return false;
}

std::vector<LegalAction> build_name_card_choices(const std::set<Entity> &entities,
                                                 Zone::Ownership owner, bool exclude_lands,
                                                 std::vector<std::string> &out_names,
                                                 bool only_lands) {
    // Distinct owner-owned vocab card names, each with a representative entity (so the
    // per-action card id encodes the candidate) and a copy count for ordering.
    std::vector<std::string> names;
    std::vector<Entity> reps;
    std::vector<int> copies;
    for (auto e : entities) {
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        if (global_coordinator.GetComponent<Zone>(e).owner != owner) continue;
        auto &cd = global_coordinator.GetComponent<CardData>(e);
        if (card_name_to_index(cd.name) < 0) continue;  // restrict to vocab cards
        if (only_lands && !card_is_land(cd)) continue;
        else if (exclude_lands && card_is_land(cd)) continue;
        bool found = false;
        for (size_t i = 0; i < names.size(); i++)
            if (names[i] == cd.name) { copies[i]++; found = true; break; }
        if (!found) { names.push_back(cd.name); reps.push_back(e); copies.push_back(1); }
    }

    // Order by copies desc, then name asc (deterministic for replay).
    std::vector<size_t> order(names.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (copies[a] != copies[b]) return copies[a] > copies[b];
        return names[a] < names[b];
    });
    if (order.size() > static_cast<size_t>(MAX_ACTIONS)) order.resize(MAX_ACTIONS);

    std::vector<LegalAction> name_choices;
    out_names.clear();
    for (size_t idx : order) {
        LegalAction la(PASS_PRIORITY, reps[idx], "Name card: " + names[idx]);
        la.category = ActionCategory::NAME_CARD;
        la.card_is_public = true;
        name_choices.push_back(la);
        out_names.push_back(names[idx]);
    }
    return name_choices;
}
