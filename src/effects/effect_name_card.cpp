#include "effects.h"

#include <algorithm>
#include <string>
#include <vector>

#include "../card_vocab.h"
#include "../classes/action.h"
#include "../classes/game.h"
#include "../classes/gamestate.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Forward declarations (see definitions below).
static bool card_is_land(const CardData &cd);

// SP$/DB$ NameCard (Cabal Therapy): the ability's controller chooses a card name
// (CR 201.4). The chosen name is recorded in cur_game.named_card so a chained
// Card.NamedCard sub-ability — here a RevealDiscardAll discard that makes a player discard
// every copy of the named card — can reference it.
//
// A player may name any card. We restrict the offered candidate set to the distinct
// nonland vocab cards owned by the spell's target player (the player who will discard),
// which is the only sensible, decidable candidate set for the engine and mirrors the
// Disruptor Flute ETB name-a-card decision (state_manager_statics.cpp). The ValidCards$
// Card.nonLand filter on the NameCard ability is honored by excluding lands.
bool name_card(Ability &ab, std::shared_ptr<Orderer> orderer) {
    (void)orderer;

    // The card name is being chosen for the benefit of the chained discard, which targets a
    // player. Offer names from that player's owned cards; fall back to the controller's
    // opponent if no target player is set.
    Zone::Ownership name_owner;
    if (ab.target != 0 && global_coordinator.entity_has_component<Player>(ab.target)) {
        name_owner = (ab.target == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
    } else {
        name_owner = (ab.controller == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;
    }

    // Distinct owned nonland vocab card names, each with a representative entity (so the
    // per-action card id encodes the candidate) and a copy count for deterministic ordering.
    std::vector<std::string> names;
    std::vector<Entity> reps;
    std::vector<int> copies;
    for (auto e : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        if (global_coordinator.GetComponent<Zone>(e).owner != name_owner) continue;
        auto &cd = global_coordinator.GetComponent<CardData>(e);
        if (card_name_to_index(cd.name) < 0) continue;  // restrict to vocab cards
        if (card_is_land(cd)) continue;                 // ValidCards$ Card.nonLand
        bool found = false;
        for (size_t i = 0; i < names.size(); i++)
            if (names[i] == cd.name) { copies[i]++; found = true; break; }
        if (!found) { names.push_back(cd.name); reps.push_back(e); copies.push_back(1); }
    }

    if (names.empty()) {
        // No nameable card; the chained Card.NamedCard discard will find no match.
        cur_game.named_card = "";
        game_log("%s names no card (no eligible card to name).\n",
                 player_name(ab.controller).c_str());
        return true;
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
    for (size_t idx : order) {
        LegalAction la(PASS_PRIORITY, reps[idx], "Name card: " + names[idx]);
        la.category = ActionCategory::NAME_CARD;
        la.card_is_public = true;
        name_choices.push_back(la);
    }

    bool prev_priority = cur_game.player_a_has_priority;
    cur_game.player_a_has_priority = (ab.controller == Zone::PLAYER_A);
    game_log("%s names a card:\n", player_name(ab.controller).c_str());
    int choice = InputLogger::instance().get_input(name_choices);
    cur_game.player_a_has_priority = prev_priority;

    cur_game.named_card = names[order[static_cast<size_t>(choice)]];
    game_log("%s names card: %s\n", player_name(ab.controller).c_str(), cur_game.named_card.c_str());
    return true;
}

// True if the card has the Land card type.
static bool card_is_land(const CardData &cd) {
    for (auto &t : cd.types)
        if (t.name == "Land") return true;
    return false;
}

}  // namespace effects
