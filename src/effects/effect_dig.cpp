#include "effects.h"

#include <algorithm>
#include <cctype>
#include <random>
#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

bool dig(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // Look at top N cards, player picks one matching filter, rest go to bottom
    Zone::Ownership dig_owner = ab.controller;
    // Resolve dynamic dig count (e.g. Count$Devotion.Blue)
    size_t effective_dig_num = ab.dig_num;
    if (!ab.dig_num_expr.empty()) {
        effective_dig_num = evaluate_dynamic_amount(ab.dig_num_expr, dig_owner, orderer, 0);
    }
    std::vector<Entity> lib = orderer->get_library_contents(dig_owner);
    std::sort(lib.begin(), lib.end(), [](Entity a, Entity b) {
        return global_coordinator.GetComponent<Zone>(a).distance_from_top <
               global_coordinator.GetComponent<Zone>(b).distance_from_top;
    });
    if (lib.size() > effective_dig_num) lib.resize(effective_dig_num);

    // Parse change_valid filters (comma-separated "Card.Creature,Card.Land" etc.)
    std::vector<std::string> filters;
    if (!ab.change_valid.empty()) {
        size_t fp = 0;
        while (true) {
            size_t comma = ab.change_valid.find(',', fp);
            if (comma == std::string::npos) {
                filters.push_back(ab.change_valid.substr(fp));
                break;
            }
            filters.push_back(ab.change_valid.substr(fp, comma - fp));
            fp = comma + 1;
        }
    }

    // Filter matching cards
    std::vector<Entity> matching;
    for (auto e : lib) {
        if (filters.empty()) {
            matching.push_back(e);
            continue;
        }
        bool card_matches = false;
        auto &cd = global_coordinator.GetComponent<CardData>(e);
        for (auto &f : filters) {
            std::string type_name;
            size_t dot = f.find('.');
            if (dot != std::string::npos)
                type_name = f.substr(dot + 1);
            else
                type_name = f;
            for (auto &t : cd.types) {
                if (t.name == type_name) {
                    card_matches = true;
                    break;
                }
            }
            if (card_matches) break;
        }
        if (card_matches) matching.push_back(e);
    }

    game_log("%s looks at the top %zu card(s) of their library.\n", player_name(dig_owner).c_str(), lib.size());

    // Present choices
    std::vector<LegalAction> dig_actions;
    if (ab.optional_choice) {
        LegalAction la(PASS_PRIORITY, "Take nothing");
        la.category = ActionCategory::DIG_CHOICE;
        dig_actions.push_back(la);
    }
    for (auto e : matching) {
        auto &cd = global_coordinator.GetComponent<CardData>(e);
        LegalAction la(PASS_PRIORITY, e, cd.name);
        la.category = ActionCategory::DIG_CHOICE;
        dig_actions.push_back(la);
    }
    // If no matching and not optional, fall through (all go to bottom)
    Entity chosen = 0;
    if (!dig_actions.empty()) {
        int choice = InputLogger::instance().get_input(dig_actions);
        chosen = dig_actions[static_cast<size_t>(choice)].source_entity;
    }

    if (chosen != 0) {
        // Determine destination: default is HAND, but DestinationZone$ can override
        Zone::ZoneValue chosen_dest = Zone::HAND;
        bool on_bottom = false;
        if (ab.dig_destination >= 0) {
            chosen_dest = static_cast<Zone::ZoneValue>(ab.dig_destination);
            // LibraryPosition$ 0 = top of library
            on_bottom = (ab.dig_library_position != 0);
        }
        orderer->add_to_zone(on_bottom, chosen, chosen_dest);
        auto &cd = global_coordinator.GetComponent<CardData>(chosen);
        if (chosen_dest == Zone::LIBRARY) {
            game_log_private(dig_owner, "%s puts %s on top of their library.\n", player_name(dig_owner).c_str(),
                cd.name.c_str());
        } else {
            game_log_private(dig_owner, "%s puts %s into hand.\n", player_name(dig_owner).c_str(), cd.name.c_str());
        }
    }

    // Remaining cards go to bottom of library
    std::vector<Entity> remaining;
    for (auto e : lib) {
        if (e != chosen) remaining.push_back(e);
    }
    if (ab.rest_random_order) {
        // Shuffle remaining with game RNG
        for (size_t i = remaining.size(); i > 1; --i) {
            std::uniform_int_distribution<size_t> dist(0, i - 1);
            size_t j = dist(cur_game.gen);
            std::swap(remaining[i - 1], remaining[j]);
        }
    }
    for (auto e : remaining) {
        orderer->add_to_zone(true, e, Zone::LIBRARY);
    }
    game_log(
        "%s puts %zu card(s) on the bottom of their library.\n", player_name(dig_owner).c_str(), remaining.size());
    return true;
}

bool parse_dig(Ability &ab, const std::string &key, const std::string &value) {
    if (key == "DigNum") {
        // Value may be a literal int or an SVar reference (e.g. "X")
        if (!value.empty() && (std::isdigit(value[0]) || value[0] == '-')) {
            ab.dig_num = static_cast<size_t>(std::stoi(value));
        } else {
            ab.dig_num = 0;
            ab.dig_num_expr = value;
        }
        return true;
    } else if (key == "DestinationZone") {
        if (value == "Library") ab.dig_destination = Zone::LIBRARY;
        else if (value == "Hand") ab.dig_destination = Zone::HAND;
        else if (value == "Graveyard") ab.dig_destination = Zone::GRAVEYARD;
        return true;
    } else if (key == "LibraryPosition") {
        ab.dig_library_position = std::stoi(value);
        return true;
    } else if (key == "ChangeValid") {
        ab.change_valid = value;
        return true;
    } else if (key == "RestRandomOrder" || key == "RandomOrder") {
        ab.rest_random_order = (value == "True");
        return true;
    }
    return false;
}

}  // namespace effects
