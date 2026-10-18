#include "name_card_choices.h"

#include <algorithm>
#include <string>

#include "card_db.h"
#include "card_vocab.h"
#include "components/carddata.h"
#include "components/zone.h"
#include "ecs/coordinator.h"
#include "game_driver.h"
#include "queries/filters.h"

extern Coordinator global_coordinator;

static std::vector<LegalAction> vocab_name_card_choices(
    const std::string &valid_filter, std::vector<std::string> &out_names);

std::vector<LegalAction> build_name_card_choices(const std::set<Entity> &entities,
                                                 Zone::Ownership owner,
                                                 const std::string &valid_filter,
                                                 std::vector<std::string> &out_names,
                                                 NameCardScope scope) {
    // A closed decklist hides the opponent's list, so the candidates cannot come from the decks
    // (docs/state_format.md register 11a).
    if (closed_decklist) return vocab_name_card_choices(valid_filter, out_names);
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

// The --closed-decklist candidate set: every vocab card (by its front face; a back face's own
// vocab entry is not a separate card) whose printed characteristics pass `valid_filter`, in name
// order, uncapped. The cards are read from the process-wide parse cache (card_data_by_name), so
// cards no deck holds create no entity and touch no game state; no object stands for a
// candidate, so each choice carries its card as `named_card`.
static std::vector<LegalAction> vocab_name_card_choices(
    const std::string &valid_filter, std::vector<std::string> &out_names) {
    std::vector<std::pair<std::string, int>> cards;
    for (int i = 0; i < CARD_VOCAB_SIZE; i++) {
        const CardData *cd = card_data_by_name(card_vocab_entries[i].name);
        if (cd == nullptr || cd->name != card_vocab_entries[i].name) continue;
        if (!valid_filter.empty() && !card_matches_any(*cd, valid_filter)) continue;
        cards.push_back({cd->name, card_vocab_entries[i].index});
    }
    std::sort(cards.begin(), cards.end());
    std::vector<LegalAction> name_choices;
    out_names.clear();
    for (const auto &card : cards) {
        LegalAction la(PASS_PRIORITY, 0, "Name card: " + card.first);
        la.category = ActionCategory::NAME_CARD;
        la.card_is_public = true;
        la.named_card = card.second;
        name_choices.push_back(la);
        out_names.push_back(card.first);
    }
    return name_choices;
}
