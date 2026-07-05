#include "effects.h"

#include <random>
#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../stable_rng.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

bool change_zone_all(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // Same-name move-all (e.g. Extirpate's ExileYard): ChangeType$ Remembered.sameName.
    if (ab.change_type.find("sameName") != std::string::npos)
        return change_zone_same_name(ab, orderer, /*force_all=*/true);

    // A targeted ability resolved with no target chosen (e.g. Endurance's "up to
    // one target player", TargetMin$ 0) affects no one — do nothing rather than
    // falling back to the controller's own zones. Untargeted ChangeZoneAll
    // (valid_tgts "N_A", e.g. Doomsday) still operates on the controller below.
    if (ab.valid_tgts != "N_A" && ab.target == 0 && ab.targets.empty())
        return true;

    Zone::Ownership owner = ab.controller;
    // If this targets a player (e.g. Endurance: "target player puts the cards
    // from their graveyard on the bottom of their library"), operate on the
    // targeted player's zones rather than the controller's.
    if (ab.target != 0 && global_coordinator.entity_has_component<Player>(ab.target)) {
        owner = (ab.target == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
    }
    // Defined$ TriggeredCardOwner (Emrakul's death trigger: "its owner shuffles their graveyard
    // into their library"): operate on the OWNER of the card that triggered this ability (the
    // source), not the last controller. Read the owner off the source's Zone (it is in the
    // graveyard by now). CR 608.2g/400.3: ownership is fixed regardless of who controlled it.
    if (ab.defined == "TriggeredCardOwner" && ab.source != 0 &&
        global_coordinator.entity_has_component<Zone>(ab.source)) {
        Zone::Ownership src_owner = global_coordinator.GetComponent<Zone>(ab.source).owner;
        if (src_owner != Zone::UNKNOWN) owner = src_owner;
    }

    // Determine which zones to search
    std::vector<Zone::ZoneValue> search_zones;
    if (ab.origins.size() > 1) {
        search_zones = ab.origins;
    } else {
        search_zones.push_back(ab.origin);
    }

    // Collect all cards from the specified zones
    std::vector<Entity> zone_contents;
    for (auto zone : search_zones) {
        if (zone == Zone::LIBRARY) {
            auto lib = orderer->get_library_contents(owner);
            zone_contents.insert(zone_contents.end(), lib.begin(), lib.end());
        } else if (zone == Zone::HAND) {
            auto hand = orderer->get_hand(owner);
            zone_contents.insert(zone_contents.end(), hand.begin(), hand.end());
        } else if (zone == Zone::GRAVEYARD) {
            for (auto e : orderer->mEntities) {
                if (!global_coordinator.entity_has_component<Zone>(e)) continue;
                auto &z = global_coordinator.GetComponent<Zone>(e);
                if (z.location == Zone::GRAVEYARD && z.owner == owner) zone_contents.push_back(e);
            }
        }
    }

    // Filter by change_type (supports the "not remembered" restriction used by
    // Doomsday — written ChangeType$ Card.!IsRemembered in Forge syntax, also
    // accepted as IsNotRemembered). Without this, the "exile the rest" step would
    // exile the cards just placed on top, emptying the library.
    bool filter_not_remembered = (ab.change_type.find("!IsRemembered") != std::string::npos
                                  || ab.change_type.find("IsNotRemembered") != std::string::npos);
    std::vector<Entity> to_move;
    for (auto entity : zone_contents) {
        if (filter_not_remembered) {
            bool is_remembered = false;
            for (auto re : cur_game.remembered_entities) {
                if (re == entity) { is_remembered = true; break; }
            }
            if (is_remembered) continue;
        }
        to_move.push_back(entity);
    }

    // RandomOrder$ — randomize the moved cards with the seeded RNG (deterministic
    // per game seed, platform-stable — see stable_rng.h). Used for "in a random
    // order" library placement (Endurance).
    if (ab.rest_random_order) {
        stable_shuffle(to_move, cur_game.gen);
    }

    // Library destinations honor LibraryPosition$ (-1 / unset = bottom).
    bool on_bottom = (ab.destination == Zone::LIBRARY && ab.dig_library_position != 0);
    const char *dest_str = ab.destination == Zone::EXILE       ? "exile"
                           : ab.destination == Zone::GRAVEYARD ? "graveyard"
                           : ab.destination == Zone::HAND      ? "hand"
                           : ab.destination == Zone::BATTLEFIELD ? "the battlefield"
                           : ab.destination == Zone::LIBRARY   ? (on_bottom ? "the bottom of their library"
                                                                            : "the top of their library")
                                                               : "zone";

    size_t moved = 0;
    for (auto entity : to_move) {
        if (global_coordinator.entity_has_component<CardData>(entity)) {
            auto &cd = global_coordinator.GetComponent<CardData>(entity);
            game_log("%s moves %s to %s\n", player_name(owner).c_str(), cd.name.c_str(), dest_str);
        }
        orderer->add_to_zone(on_bottom, entity, ab.destination);
        // RememberChanged$ True: stash every moved card in the remembered set, mirroring the
        // single-target ChangeZone path (effect_change_zone.cpp). A later SVar can then count
        // these cards (Canoptek Scarab Swarm: X = Remembered$Valid Land,Artifact, "for each
        // artifact or land card exiled this way"); cleared by the paired DBCleanup ClearRemembered$.
        if (ab.remember_changed) cur_game.remembered_entities.push_back(entity);
        moved++;
    }
    game_log("%s moves %zu card(s) to %s\n", player_name(owner).c_str(), moved, dest_str);

    // Shuffle$ True (Emrakul's death trigger: "shuffle their graveyard into their library"): after
    // moving the cards into the library, shuffle it. shuffle_library also clears the known-top-of-
    // library tracking for that player (CR 701.20).
    if (ab.shuffle_after && ab.destination == Zone::LIBRARY) {
        orderer->shuffle_library(owner);
        game_log("%s shuffles their library.\n", player_name(owner).c_str());
    }
    return true;
}

}  // namespace effects
