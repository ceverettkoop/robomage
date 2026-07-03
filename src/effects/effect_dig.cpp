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
#include "../game_queries.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../svar_eval.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

bool dig(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // Look at top N cards, player picks one matching filter, rest go to bottom.
    // When the ability targets a player (Fateseal, e.g. Jace +2), the dug library is
    // the TARGET player's, not the controller's.
    Zone::Ownership dig_owner = ab.controller;
    if (ab.target != 0 && global_coordinator.entity_has_component<Player>(ab.target))
        dig_owner = (ab.target == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
    // Resolve dynamic dig count (e.g. Count$Devotion.Blue)
    size_t effective_dig_num = ab.dig_num;
    if (!ab.dig_num_expr.empty()) {
        effective_dig_num = evaluate_dynamic_amount(ab.dig_num_expr, dig_owner, orderer, 0);
    }
    std::vector<Entity> lib = orderer->get_library_top(dig_owner, effective_dig_num);

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

    // A filter like "Creature.cmcLEX" or "Card.cmcLEX" carries a mana-value bound X, resolved
    // from dynamic_amount_expr (e.g. Birthing Ritual: 1 + sacrificed creature's mana value).
    bool has_cmc_le = !ab.change_valid.empty() && ab.change_valid.find("cmcLE") != std::string::npos;
    int cmc_threshold = 0;
    if (has_cmc_le && !ab.dynamic_amount_expr.empty())
        cmc_threshold = static_cast<int>(evaluate_dynamic_amount(ab.dynamic_amount_expr, dig_owner, orderer, ab.target));

    // Filter matching cards. A filter is dot-separated: "Card" is a type wildcard, a "cmc.."
    // token is a mana-value bound (handled above), and any other token is the required type
    // (so both "Card.Creature" and "Creature.cmcLEX" name the Creature type).
    std::vector<Entity> matching;
    for (auto e : lib) {
        auto &cd = global_coordinator.GetComponent<CardData>(e);
        bool card_matches = filters.empty();
        for (auto &f : filters) {
            std::string want_type;
            size_t start = 0;
            while (start <= f.size()) {
                size_t dot = f.find('.', start);
                std::string part = f.substr(start, dot == std::string::npos ? std::string::npos : dot - start);
                if (!part.empty() && part != "Card" && part.rfind("cmc", 0) != 0)
                    want_type = part;
                if (dot == std::string::npos) break;
                start = dot + 1;
            }
            bool type_ok = want_type.empty();
            if (!type_ok)
                for (auto &t : cd.types)
                    if (t.name == want_type) { type_ok = true; break; }
            if (type_ok) { card_matches = true; break; }
        }
        if (!card_matches) continue;
        // Apply the mana-value bound if present (cmc <= threshold).
        if (has_cmc_le && card_mana_value(cd) > cmc_threshold) continue;
        matching.push_back(e);
    }

    game_log("%s looks at the top %zu card(s) of their library.\n", player_name(dig_owner).c_str(), lib.size());

    // How many of the looked-at cards may be taken (default 1). A conditional
    // ChangeNum$ (Flow State) raises this to its true-value when the summed
    // graveyard counts satisfy the compare; a plain numeric ChangeNum$ uses amount.
    // ChangeNum$ Any (Fateseal) means the player may take any number (0..pool) of the
    // looked-at cards; treat it as optional with a take limit of the whole pool.
    bool any_count = ab.change_num_any;
    bool optional = ab.optional_choice || any_count;
    size_t take_count = 1;
    if (ab.change_num >= 0) {
        // Explicit ChangeNum$ N (incl. 0 = "look but take nothing", Birthing Ritual DBDigBis).
        take_count = static_cast<size_t>(ab.change_num);
    } else if (ab.cond_amount_active) {
        int sum = 0;
        for (auto &expr : ab.cond_amount_exprs)
            sum += static_cast<int>(evaluate_dynamic_amount(expr, dig_owner, orderer, 0));
        take_count = compare_svar(sum, ab.cond_amount_compare) ? ab.cond_amount_if_true : ab.amount;
    } else if (any_count) {
        take_count = matching.size();
    } else if (ab.amount > 0) {
        take_count = ab.amount;
    }

    // Present choices, one card at a time until take_count are taken (or the player
    // declines / the pool runs dry).
    std::vector<Entity> chosen_cards;
    std::vector<Entity> pool = matching;
    for (size_t pick = 0; pick < take_count; ++pick) {
        std::vector<LegalAction> dig_actions;
        if (optional) {
            LegalAction la(PASS_PRIORITY, "Take nothing");
            la.category = ActionCategory::DIG_CHOICE;
            dig_actions.push_back(la);
        }
        for (auto e : pool) {
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            LegalAction la(PASS_PRIORITY, e, cd.name);
            la.category = ActionCategory::DIG_CHOICE;
            dig_actions.push_back(la);
        }
        // If no matching and not optional, fall through (all go to bottom)
        if (dig_actions.empty()) break;
        int choice = InputLogger::instance().get_input(dig_actions);
        Entity sel = dig_actions[static_cast<size_t>(choice)].source_entity;
        if (sel == 0) break;  // chose "Take nothing"
        chosen_cards.push_back(sel);
        pool.erase(std::remove(pool.begin(), pool.end(), sel), pool.end());
    }

    // Determine destination: default is HAND, but DestinationZone$ can override
    Zone::ZoneValue chosen_dest = Zone::HAND;
    bool on_bottom = false;
    if (ab.dig_destination >= 0) {
        chosen_dest = static_cast<Zone::ZoneValue>(ab.dig_destination);
        // LibraryPosition$ 0 = top of library
        on_bottom = (ab.dig_library_position != 0);
    }
    for (Entity chosen : chosen_cards) {
        orderer->add_to_zone(on_bottom, chosen, chosen_dest);
        auto &cd = global_coordinator.GetComponent<CardData>(chosen);
        if (chosen_dest == Zone::LIBRARY) {
            game_log_private(dig_owner, "%s puts %s on the %s of their library.\n", player_name(dig_owner).c_str(),
                cd.name.c_str(), on_bottom ? "bottom" : "top");
            game_log_redacted(dig_owner, "%s puts a card on the %s of their library.\n",
                player_name(dig_owner).c_str(), on_bottom ? "bottom" : "top");
        } else if (chosen_dest == Zone::BATTLEFIELD) {
            // Public information once it hits the battlefield.
            game_log("%s puts %s onto the battlefield.\n", player_name(dig_owner).c_str(), cd.name.c_str());
        } else {
            game_log_private(dig_owner, "%s puts %s into hand.\n", player_name(dig_owner).c_str(), cd.name.c_str());
            game_log_redacted(dig_owner, "%s puts a card into hand.\n", player_name(dig_owner).c_str());
        }
    }

    // Remaining cards go to bottom of library
    std::vector<Entity> remaining;
    for (auto e : lib) {
        if (std::find(chosen_cards.begin(), chosen_cards.end(), e) == chosen_cards.end()) remaining.push_back(e);
    }
    if (ab.rest_random_order) {
        // Shuffle remaining with game RNG
        for (size_t i = remaining.size(); i > 1; --i) {
            std::uniform_int_distribution<size_t> dist(0, i - 1);
            size_t j = dist(cur_game.gen);
            std::swap(remaining[i - 1], remaining[j]);
        }
    }
    // Unchosen cards normally go to the bottom; LibraryPosition2$ 0 (Fateseal) keeps
    // them on top instead (i.e. you may bottom the looked-at card, else it stays put).
    bool rest_on_bottom = (ab.dig_rest_library_position != 0);
    for (auto e : remaining) {
        orderer->add_to_zone(rest_on_bottom, e, Zone::LIBRARY);
    }
    game_log("%s puts %zu card(s) on the %s of their library.\n", player_name(dig_owner).c_str(),
             remaining.size(), rest_on_bottom ? "bottom" : "top");
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
        else if (value == "Battlefield") ab.dig_destination = Zone::BATTLEFIELD;
        return true;
    } else if (key == "LibraryPosition") {
        ab.dig_library_position = std::stoi(value);
        return true;
    } else if (key == "LibraryPosition2") {
        // Where the looked-at-but-unchosen cards go: 0 = top, otherwise bottom.
        ab.dig_rest_library_position = std::stoi(value);
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
