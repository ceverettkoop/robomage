#include "name_card_choices.h"

#include <algorithm>
#include <string>

#include "card_vocab.h"
#include "components/carddata.h"
#include "components/zone.h"
#include "ecs/coordinator.h"
#include "game_queries.h"

extern Coordinator global_coordinator;

std::vector<LegalAction> build_name_card_choices(const std::set<Entity> &entities,
                                                 Zone::Ownership owner,
                                                 const std::string &valid_filter,
                                                 std::vector<std::string> &out_names,
                                                 NameCardScope scope) {
    // Distinct vocab card names in scope, each with a representative entity (so the
    // per-action card id encodes the candidate) and a copy count for ordering.
    // CHOOSER_ONLY keeps only cards owned by `owner`; BOTH_PLAYERS accepts cards owned
    // by either player (A or B), de-duped by name across both decks.
    std::vector<std::string> names;
    std::vector<Entity> reps;
    std::vector<int> copies;
    for (auto e : entities) {
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        Zone::Ownership card_owner = global_coordinator.GetComponent<Zone>(e).owner;
        if (scope == NameCardScope::CHOOSER_ONLY) {
            if (card_owner != owner) continue;
        } else {  // BOTH_PLAYERS
            if (card_owner != Zone::PLAYER_A && card_owner != Zone::PLAYER_B) continue;
        }
        auto &cd = global_coordinator.GetComponent<CardData>(e);
        if (card_name_to_index(cd.name) < 0) continue;  // restrict to vocab cards
        // Apply the ability's ValidCards$ filter against the candidate's printed
        // characteristics via the unified matcher, so every filter token (Card.nonLand,
        // Land, …) works uniformly. Empty filter = no restriction.
        if (!valid_filter.empty() && !card_matches_any(e, valid_filter)) continue;
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
