#include "effects.h"

#include <string>
#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// DB$ DigUntil (Amped Raptor): exile cards from the top of the controller's library one at a
// time until one matches Valid$ (here Card.nonLand). The cards passed over go to
// RevealedDestination$; the matching card goes to FoundDestination$ (both Exile for Amped
// Raptor — an impulse-style exile). RememberFound$ True records the matching card in
// cur_game.remembered_entities so a chained DB$ Play can cast it. If the library empties first,
// stop gracefully (nothing is remembered). General over the destinations and the filter.
bool dig_until(Ability &ab, std::shared_ptr<Orderer> orderer) {
    Zone::Ownership owner = ab.controller;
    Zone::ZoneValue found_dest = static_cast<Zone::ZoneValue>(ab.dig_until_found_dest);
    Zone::ZoneValue revealed_dest = static_cast<Zone::ZoneValue>(ab.dig_until_revealed_dest);

    // RememberFound$ True replaces any previously remembered cards with the found one, so a
    // downstream Defined$ Remembered (DB$ Play) reads exactly this card. Clear up front so a
    // failed dig (library empties) leaves nothing remembered.
    if (ab.dig_until_remember_found) cur_game.remembered_entities.clear();

    // Walk the library from the top, one card at a time (the library shrinks as we exile,
    // so re-read the current top each step rather than snapshotting).
    while (true) {
        std::vector<Entity> top = orderer->get_library_top(owner, 1);
        if (top.empty()) {
            game_log("%s's library is empty; the dig stops.\n", player_name(owner).c_str());
            break;
        }
        Entity card = top[0];
        bool matches = ab.change_valid.empty() ||
                       card_matches_filter(card, ab.change_valid);
        const std::string nm = global_coordinator.entity_has_component<CardData>(card)
            ? global_coordinator.GetComponent<CardData>(card).name : "a card";
        if (matches) {
            orderer->add_to_zone(false, card, found_dest);
            game_log("%s exiles %s.\n", player_name(owner).c_str(), nm.c_str());
            if (ab.dig_until_remember_found)
                cur_game.remembered_entities.push_back(card);
            break;  // the until-condition is satisfied — stop
        }
        // A non-matching card passed over: send it to the revealed destination.
        orderer->add_to_zone(false, card, revealed_dest);
        game_log("%s exiles %s.\n", player_name(owner).c_str(), nm.c_str());
    }
    return true;
}

bool parse_dig_until(Ability &ab, const std::string &key, const std::string &value) {
    auto zone_from = [](const std::string &v) -> int {
        if (v == "Library") return Zone::LIBRARY;
        if (v == "Hand") return Zone::HAND;
        if (v == "Graveyard") return Zone::GRAVEYARD;
        if (v == "Exile") return Zone::EXILE;
        if (v == "Battlefield") return Zone::BATTLEFIELD;
        return Zone::HAND;
    };
    // Valid$ — the until-filter. Scoped to DigUntil so it doesn't shadow any other effect's
    // Valid$; stored in the shared change_valid filter slot the matcher already reads.
    if (key == "Valid" && ab.category == "DigUntil") {
        ab.change_valid = value;
        return true;
    } else if (key == "FoundDestination") {
        ab.dig_until_found_dest = zone_from(value);
        return true;
    } else if (key == "RevealedDestination") {
        ab.dig_until_revealed_dest = zone_from(value);
        return true;
    } else if (key == "RememberFound") {
        ab.dig_until_remember_found = (value == "True");
        return true;
    }
    // Valid$ is shared with the Dig/ChangeZone grammar via change_valid; handled below.
    return false;
}

}  // namespace effects
