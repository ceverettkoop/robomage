#include "effects.h"

#include <string>
#include <vector>

#include "../classes/game.h"
#include "../classes/match_state.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

static bool search_reveals_card(const Ability &ab);

// A library search reveals the chosen card when it must satisfy a restriction
// more specific than "any card" — the searcher proves the card qualifies (e.g.
// Personal Tutor: "search for a sorcery card, reveal it"). An unrestricted
// "Card" search (Vampiric/Demonic Tutor, Doomsday) reveals nothing. Searches of
// other zones are public already, so this only matters for the library.
static bool search_reveals_card(const Ability &ab) {
    bool from_library = (ab.origin == Zone::LIBRARY);
    for (auto z : ab.origins) {
        if (z == Zone::LIBRARY) from_library = true;
    }
    bool specific_type = !ab.change_type.empty() && ab.change_type != "Card";
    return from_library && specific_type;
}

bool change_zone(Ability &ab, std::shared_ptr<Orderer> orderer) {
    Zone::Ownership owner = global_coordinator.GetComponent<Zone>(ab.source).owner;

    const char *dest_str = ab.destination == Zone::BATTLEFIELD ? "the battlefield"
                           : ab.destination == Zone::LIBRARY   ? "top of library"
                           : ab.destination == Zone::GRAVEYARD ? "graveyard"
                           : ab.destination == Zone::HAND      ? "hand"
                                                               : "exile";

    // Targeted ChangeZone (e.g. Swords to Plowshares, Faerie Macabre, Life from the
    // Loam): move target(s) directly. A targeted ability with no chosen targets (e.g.
    // "up to three" with zero picked) does nothing rather than searching.
    if (ab.valid_tgts != "N_A") {
        std::vector<Entity> to_move;
        if (!ab.targets.empty()) {
            to_move = ab.targets;
        } else if (ab.target != 0) {
            to_move.push_back(ab.target);
        }
        for (auto tgt : to_move) {
            if (!global_coordinator.entity_has_component<Zone>(tgt)) continue;
            std::string tname = global_coordinator.entity_has_component<CardData>(tgt)
                                    ? global_coordinator.GetComponent<CardData>(tgt).name
                                    : (global_coordinator.entity_has_component<Permanent>(tgt)
                                              ? global_coordinator.GetComponent<Permanent>(tgt).name
                                              : "<unknown>");
            orderer->add_to_zone(false, tgt, ab.destination);
            if (ab.destination == Zone::EXILE && ab.source != 0 &&
                global_coordinator.entity_has_component<Permanent>(ab.source)) {
                global_coordinator.GetComponent<Permanent>(ab.source).exiled_with.push_back(tgt);
            }
            game_log("%s is moved to %s\n", tname.c_str(), dest_str);
        }
        return true;
    }

    // Defined$ Self — move the source card directly (e.g. Talon Gates putting itself onto battlefield from hand)
    if (ab.defined_self && ab.source != 0) {
        std::string sname = global_coordinator.entity_has_component<CardData>(ab.source)
                                ? global_coordinator.GetComponent<CardData>(ab.source).name
                                : "<unknown>";
        orderer->add_to_zone(false, ab.source, ab.destination);
        if (ab.destination == Zone::BATTLEFIELD) {
            auto &src_zone = global_coordinator.GetComponent<Zone>(ab.source);
            src_zone.controller = owner;
        }
        game_log("%s is moved to %s\n", sname.c_str(), dest_str);
        return true;
    }

    // Search-based ChangeZone (e.g. fetch lands, Green Sun's Zenith)
    size_t num_to_move = (ab.amount > 0) ? ab.amount : 1;
    bool multi_zone = ab.origins.size() > 1;
    bool reveal = search_reveals_card(ab);

    for (size_t i = 0; i < num_to_move; i++) {
        Entity chosen = 0;
        if (multi_zone) {
            chosen = search_multi_zone(orderer, owner, ab.origins, ab.change_type, ab.mandatory, ab.destination,
                reveal);
        } else {
            chosen = search_zone(orderer, owner, ab.origin, ab.change_type, ab.mandatory, ab.destination,
                reveal);
        }

        // after we have chosen but before we place it where it goes, if we messed with library shuffle it
        if (ab.origin == Zone::LIBRARY) {
            orderer->shuffle_library(owner);
            game_log("%s shuffles their library\n", player_name(owner).c_str());
        }

        if (chosen != 0) {
            auto &chosen_cd = global_coordinator.GetComponent<CardData>(chosen);
            auto &chosen_zone = global_coordinator.GetComponent<Zone>(chosen);
            orderer->add_to_zone(false, chosen, ab.destination);
            if (ab.destination == Zone::BATTLEFIELD) {
                chosen_zone.controller = owner;
                if (ab.enters_tapped) cur_game.pending_enters_tapped.insert(chosen);
            }
            if (ab.remember_changed) {
                cur_game.remembered_entities.push_back(chosen);
            }
            bool dest_public =
                (ab.destination == Zone::BATTLEFIELD || ab.destination == Zone::GRAVEYARD || ab.destination == Zone::EXILE);
            if (dest_public) {
                game_log("%s puts %s to %s\n", player_name(owner).c_str(), chosen_cd.name.c_str(), dest_str);
            } else if (reveal) {
                // Hidden destination, but the card was revealed — it's public knowledge.
                // The orderer reveal hook only fires for public zones, so mark it here.
                mark_card_revealed(chosen, owner);
                game_log("%s reveals %s and puts it to %s\n", player_name(owner).c_str(), chosen_cd.name.c_str(),
                    dest_str);
            } else {
                game_log_private(
                    owner, "%s puts %s to %s\n", player_name(owner).c_str(), chosen_cd.name.c_str(), dest_str);
                game_log("%s puts a card to %s\n", player_name(owner).c_str(), dest_str);
            }
        } else {
            game_log("%s fails to find\n", player_name(owner).c_str());
            break;
        }
    }
    return true;
}

// Owns the zone-movement param keys shared by the ChangeZone family (ChangeZone
// and ChangeZoneAll both consume origin/destination/etc. at resolve).
bool parse_change_zone(Ability &ab, const std::string &key, const std::string &value) {
    if (key == "ChangeType") {
        ab.change_type = value;
        return true;
    } else if (key == "RememberChanged") {
        ab.remember_changed = (value == "True");
        return true;
    } else if (key == "Origin") {
        // Handle comma-separated origins (e.g. "Graveyard,Library")
        auto parse_zone = [](const std::string &s) -> Zone::ZoneValue {
            if (s == "Library")        return Zone::LIBRARY;
            if (s == "Hand")           return Zone::HAND;
            if (s == "Graveyard")      return Zone::GRAVEYARD;
            if (s == "Exile")          return Zone::EXILE;
            if (s == "Stack")          return Zone::STACK;
            return Zone::LIBRARY;
        };
        ab.origins.clear();
        size_t zp = 0;
        while (true) {
            size_t comma = value.find(',', zp);
            if (comma == std::string::npos) {
                ab.origins.push_back(parse_zone(value.substr(zp)));
                break;
            }
            ab.origins.push_back(parse_zone(value.substr(zp, comma - zp)));
            zp = comma + 1;
        }
        ab.origin = ab.origins[0];  // backward compat
        return true;
    } else if (key == "Destination") {
        if (value == "Battlefield")    ab.destination = Zone::BATTLEFIELD;
        else if (value == "Library")   ab.destination = Zone::LIBRARY;
        else if (value == "Hand")      ab.destination = Zone::HAND;
        else if (value == "Graveyard") ab.destination = Zone::GRAVEYARD;
        else if (value == "Exile")     ab.destination = Zone::EXILE;
        return true;
    } else if (key == "MayShuffle") {
        ab.may_shuffle = (value == "True");
        return true;
    } else if (key == "Tapped") {
        ab.enters_tapped = (value == "True");
        return true;
    }
    return false;
}

}  // namespace effects
