#include "effects.h"

#include <cstddef>
#include <string>
#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../stable_rng.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// DB$ DigUntil: reveal cards from the top of the controller's library one at a time until one
// matches Valid$ (CR 701.16 reveal + 401 library ordering). The matching card goes to
// FoundDestination$; the cards passed over go to RevealedDestination$.
//   - Amped Raptor: found -> Exile, revealed -> Exile (impulse), RememberFound$ True records the
//     matching card in cur_game.remembered_entities so a chained DB$ Play can cast it.
//   - Raph & Mikey, Troublemakers: found -> Battlefield entering Tapped$ and Attacking$ (CR 508.4:
//     put onto the battlefield attacking, not declared — no new "attacks" triggers), revealed ->
//     the bottom of the library (RevealedLibraryPosition$ -1) in a random order (RevealRandomOrder$).
// If the library empties before a match, stop gracefully (nothing is remembered/found). General
// over the destinations, the filter, and the entering flags.
HandlerResult dig_until(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    Zone::Ownership owner = ab.controller;
    Zone::ZoneValue found_dest = static_cast<Zone::ZoneValue>(ab.dig_until_found_dest);
    Zone::ZoneValue revealed_dest = static_cast<Zone::ZoneValue>(ab.dig_until_revealed_dest);

    // RememberFound$ True replaces any previously remembered cards with the found one, so a
    // downstream Defined$ Remembered (DB$ Play) reads exactly this card. Clear up front so a
    // failed dig (library empties) leaves nothing remembered.
    if (ab.dig_until_remember_found) cur_game.remembered_entities.clear();

    // Reveal from the top (a stable top-first snapshot — nothing leaves the library until the
    // reveal has resolved) until a card matches. `revealed` holds the non-matching cards passed
    // over, `found` the matching card (0 if the library ran out first).
    std::vector<Entity> lib = orderer->get_library_top(owner, static_cast<size_t>(-1));
    Entity found = 0;
    std::vector<Entity> revealed;
    for (Entity card : lib) {
        const std::string nm = global_coordinator.entity_has_component<CardData>(card)
            ? global_coordinator.GetComponent<CardData>(card).name : "a card";
        game_log("%s reveals %s.\n", player_name(owner).c_str(), nm.c_str());
        bool matches = ab.change_valid.empty() || card_matches_filter(card, ab.change_valid);
        if (matches) { found = card; break; }
        revealed.push_back(card);
    }
    if (found == 0)
        game_log("%s's library is empty; the dig finds no match.\n", player_name(owner).c_str());

    // Place the passed-over cards first, then the found card (this preserves the exile ordering of
    // the one-at-a-time impulse dig for Amped Raptor). RevealRandomOrder$ shuffles them with the
    // seeded RNG (deterministic, platform-stable — see stable_rng.h); a Library destination honors
    // RevealedLibraryPosition$ (-1 / unset = bottom, 0 = top).
    if (!revealed.empty()) {
        if (ab.rest_random_order) stable_shuffle(revealed, cur_game.gen);
        bool on_bottom = (revealed_dest == Zone::LIBRARY && ab.dig_library_position != 0);
        for (Entity card : revealed)
            orderer->add_to_zone(on_bottom, card, revealed_dest);
    }

    if (found != 0) {
        const std::string nm = global_coordinator.entity_has_component<CardData>(found)
            ? global_coordinator.GetComponent<CardData>(found).name : "a card";
        if (found_dest == Zone::BATTLEFIELD) {
            // Tapped$ / Attacking$ (Raph & Mikey): the found creature enters tapped and attacking
            // the same defender the source is attacking (CR 508.4). Reuse the ninjutsu one-shots —
            // apply_permanent_components consumes them once the card's Creature component exists.
            if (ab.enters_tapped) cur_game.pending_enters_tapped.insert(found);
            if (ab.dig_until_attacking) {
                Entity attack_target = 0;
                if (global_coordinator.entity_has_component<Creature>(ab.source))
                    attack_target = global_coordinator.GetComponent<Creature>(ab.source).attack_target;
                if (attack_target != 0) cur_game.pending_enters_attacking[found] = attack_target;
            }
            orderer->add_to_zone(false, found, Zone::BATTLEFIELD);
            // The card enters under the digging player's control (CR 608.2 — it comes from their
            // own library).
            if (global_coordinator.GetComponent<Zone>(found).location == Zone::BATTLEFIELD)
                global_coordinator.GetComponent<Zone>(found).controller = owner;
            game_log("%s puts %s onto the battlefield tapped and attacking.\n",
                     player_name(owner).c_str(), nm.c_str());
        } else {
            orderer->add_to_zone(false, found, found_dest);
            game_log("%s exiles %s.\n", player_name(owner).c_str(), nm.c_str());
        }
        if (ab.dig_until_remember_found) cur_game.remembered_entities.push_back(found);
    }
    return HandlerResult::DONE_RUN_SUBS;
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
    } else if (key == "Attacking" && ab.category == "DigUntil") {
        ab.dig_until_attacking = (value == "True");
        return true;
    } else if (key == "RevealedLibraryPosition") {
        // Where the revealed (passed-over) cards go in RevealedDestination$ Library: -1 = bottom
        // (the default), 0 = top. Stored in the shared dig_library_position slot the mover reads.
        ab.dig_library_position = std::stoi(value);
        return true;
    } else if (key == "RevealRandomOrder") {
        ab.rest_random_order = (value == "True");
        return true;
    }
    // Valid$ is shared with the Dig/ChangeZone grammar via change_valid; handled below.
    return false;
}

}  // namespace effects
