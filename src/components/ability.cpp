#include "ability.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../components/carddata.h"
#include "../debug.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../systems/orderer.h"
#include "damage.h"
#include "player.h"
#include "spell.h"

extern Coordinator global_coordinator;
extern Game cur_game;

bool Ability::identical_activated_ability(const Ability &other) {
    if (other.category != this->category) return false;
    if (other.valid_tgts != this->valid_tgts) return false;
    if (other.amount != this->amount) return false;
    if (other.tap_cost != this->tap_cost) return false;
    if (other.activation_mana_cost != this->activation_mana_cost) return false;
    if (other.sac_self != this->sac_self) return false;
    if (other.change_type != this->change_type) return false;
    if (other.origin != this->origin) return false;
    if (other.destination != this->destination) return false;
    return true;
};

// Searches a zone for cards whose types match any entry in the comma-separated
// change_type string. Presents all matches plus a "fail to find" option (index 0).
// Returns the chosen Entity, or 0 for fail to find.
// 0 is a valid entity but will always be player a  so is never correct
Entity search_zone(
    std::shared_ptr<Orderer> orderer, Zone::Ownership owner, Zone::ZoneValue zone, const std::string &change_type,
    bool mandatory, Zone::ZoneValue destination) {
    //  comma-separated subtypes
    std::vector<std::string> subtypes;
    size_t p = 0;
    while (true) {
        size_t comma = change_type.find(',', p);
        if (comma == std::string::npos) {
            subtypes.push_back(change_type.substr(p));
            break;
        }
        subtypes.push_back(change_type.substr(p, comma - p));
        p = comma + 1;
    }

    // Collect zone contents
    std::vector<Entity> zone_contents;
    if (zone == Zone::LIBRARY) {
        zone_contents = orderer->get_library_contents(owner);
    } else if (zone == Zone::HAND) {
        zone_contents = orderer->get_hand(owner);
    }
    // TODO OTHER ZONES

    // Filter to matching cards; empty change_type means all cards match
    std::vector<Entity> choices;
    if (change_type.empty()) {
        choices = zone_contents;
    } else {
        for (auto entity : zone_contents) {
            auto &cd = global_coordinator.GetComponent<CardData>(entity);
            bool matches = false;
            for (auto &t : cd.types) {
                for (auto &st : subtypes) {
                    if (t.name == st) {
                        matches = true;
                        break;
                    }
                }
                if (matches) break;
            }
            if (matches) choices.push_back(entity);
        }
    }

    const char *zone_name = (zone == Zone::LIBRARY)     ? "library"
                            : (zone == Zone::HAND)      ? "hand"
                            : (zone == Zone::GRAVEYARD) ? "graveyard"
                            : (zone == Zone::EXILE)     ? "exile"
                                                        : "zone";
    // Determine category: library searches going to top of library use TOP_LIBRARY,
    // other library searches use SEARCH_LIBRARY, hand picks use OTHER_CHOICE
    ActionCategory cat = (zone == Zone::LIBRARY && destination == Zone::LIBRARY) ? ActionCategory::TOP_LIBRARY
                       : (zone == Zone::LIBRARY)                                 ? ActionCategory::SEARCH_LIBRARY
                                                                                 : ActionCategory::OTHER_CHOICE;

    // Fail-to-find is shown when: not mandatory, OR zone is empty (nothing else to choose)
    bool show_fail_to_find = !mandatory || choices.empty();

    if (mandatory && choices.empty()) {
        // Nothing left to move; return immediately without prompting
        return 0;
    }

    if (zone == Zone::LIBRARY) {
        printf("Searching %s's %s:\n", player_name(owner).c_str(), zone_name);
    } else {
        printf("%s chooses a card from %s %s:\n", player_name(owner).c_str(), player_name(owner).c_str(), zone_name);
    }

    std::vector<ActionCategory> cats;
    std::vector<Entity> search_entities;

    size_t display_index = 0;
    if (show_fail_to_find) {
        printf("  %zu: Fail to find\n", display_index);
        cats.push_back(cat);
        search_entities.push_back(Entity(0));
        display_index++;
    }
    for (size_t i = 0; i < choices.size(); i++) {
        auto &cd = global_coordinator.GetComponent<CardData>(choices[i]);
        printf("  %zu: %s\n", display_index + i, cd.name.c_str());
        cats.push_back(cat);
        search_entities.push_back(choices[i]);
    }

    int choice = InputLogger::instance().get_logged_input(cur_game.turn, cats, search_entities);
    // Map choice back: if fail-to-find is shown, index 0 = fail-to-find, 1..N = choices
    // If fail-to-find suppressed, index 0..N-1 = choices directly
    if (show_fail_to_find) {
        if (choice >= 1 && choice <= static_cast<int>(choices.size()))
            return choices[static_cast<size_t>(choice - 1)];
        return 0;
    } else {
        if (choice >= 0 && choice < static_cast<int>(choices.size()))
            return choices[static_cast<size_t>(choice)];
        return 0;
    }
}

 
void Ability::resolve_change_zone(std::shared_ptr<Orderer> orderer) {
    Zone::Ownership owner = global_coordinator.GetComponent<Zone>(source).owner;
    size_t num_to_move = (amount > 0) ? amount : 1;

    for (size_t i = 0; i < num_to_move; i++) {
        Entity chosen = search_zone(orderer, owner, origin, change_type, mandatory, destination);
        if (chosen != 0) {
            auto &chosen_cd = global_coordinator.GetComponent<CardData>(chosen);
            auto &chosen_zone = global_coordinator.GetComponent<Zone>(chosen);
            orderer->add_to_zone(false, chosen, destination);
            if (destination == Zone::BATTLEFIELD) {
                chosen_zone.controller = owner;
            }
            printf("%s puts %s to %s\n", player_name(owner).c_str(), chosen_cd.name.c_str(),
                   destination == Zone::BATTLEFIELD ? "the battlefield" :
                   destination == Zone::LIBRARY     ? "top of library" :
                   destination == Zone::GRAVEYARD   ? "graveyard"      :
                   destination == Zone::HAND        ? "hand"           : "exile");
        } else {
            printf("%s fails to find\n", player_name(owner).c_str());
            break;
        }
    }

    if (origin == Zone::LIBRARY) {
        orderer->shuffle_library(owner);
        printf("%s shuffles their library\n", player_name(owner).c_str());
    }
}

void Ability::resolve_rearrange_top_of_library(std::shared_ptr<Orderer> orderer) {
    Zone::Ownership owner = global_coordinator.GetComponent<Zone>(source).owner;

    std::vector<Entity> lib = orderer->get_library_contents(owner);
    // Sort by distance_from_top ascending so lib[0] is the actual top card
    std::sort(lib.begin(), lib.end(), [](Entity a, Entity b) {
        return global_coordinator.GetComponent<Zone>(a).distance_from_top
             < global_coordinator.GetComponent<Zone>(b).distance_from_top;
    });
    //looking at top n only
    if (lib.size() > amount) lib.resize(amount);
    size_t actual = lib.size();
    std::vector<Entity> remaining = lib;
    
    printf("%s looks at the top %zu card(s) of their library.\n", player_name(owner).c_str(), actual);

    std::vector<Entity> chosen_order;
    // Player picks N-1 cards; the last is automatic
    for (size_t pick = 0; pick + 1 < actual; pick++) {
        printf("Choose which card goes on top next (pick %zu of %zu):\n", pick + 1, actual);
        std::vector<ActionCategory> cats;
        std::vector<Entity> pick_entities;
        for (size_t i = 0; i < remaining.size(); i++) {
            auto &cd = global_coordinator.GetComponent<CardData>(remaining[i]);
            printf("  %zu: %s\n", i, cd.name.c_str());
            cats.push_back(ActionCategory::TOP_LIBRARY);
            pick_entities.push_back(remaining[i]);
        }
        int choice = InputLogger::instance().get_logged_input(cur_game.turn, cats, pick_entities);
        if (choice < 0 || choice >= static_cast<int>(remaining.size())) choice = 0;
        chosen_order.push_back(remaining[static_cast<size_t>(choice)]);
        remaining.erase(remaining.begin() + choice);
    }
    // Last card is forced
    if (!remaining.empty()) {
        chosen_order.push_back(remaining[0]);
    }

    // Put cards back: chosen_order[0] should end up on top, so place in reverse
    for (auto it = chosen_order.rbegin(); it != chosen_order.rend(); ++it) {
        orderer->add_to_zone(false, *it, Zone::LIBRARY);
    }

    if (may_shuffle) {
        printf("You may shuffle your library. 0: Don't shuffle  1: Shuffle\n");
        std::vector<ActionCategory> shuffle_cats = {ActionCategory::SHUFFLE, ActionCategory::SHUFFLE};
        std::vector<Entity> shuffle_entities = {Entity(0), Entity(0)};
        int shuffle_choice = InputLogger::instance().get_logged_input(cur_game.turn, shuffle_cats, shuffle_entities);
        if (shuffle_choice == 1) {
            orderer->shuffle_library(owner);
            printf("%s shuffles their library.\n", player_name(owner).c_str());
        }
    }
}

void Ability::resolve(std::shared_ptr<Orderer> orderer) {
    printf("Resolving ability (category: %s, amount: %zu)\n", category.c_str(), amount);

    if (category == "Draw") {
        Zone::Ownership owner = global_coordinator.GetComponent<Zone>(source).owner;
        orderer->draw(owner, amount);
    } else if (category == "ChangeZone") {
        resolve_change_zone(orderer);
    } else if (category == "RearrangeTopOfLibrary") {
        resolve_rearrange_top_of_library(orderer);
    } else if (category == "DealDamage") {
        if (global_coordinator.entity_has_component<Player>(target)) {
            auto &player = global_coordinator.GetComponent<Player>(target);
            player.life_total -= static_cast<int32_t>(amount);
            printf("Dealt %zu damage to player (now at %d life)\n", amount, player.life_total);
        } else {
            if (deal_damage(source, target, amount)) {
                printf("Dealt %zu damage to creature\n", amount);
            } else {
                printf("Invalid target\n");
            }
        }
    } else if (category == "Destroy") {
        resolve_destroy(orderer);
    } else if (category == "Counter") {
        if (global_coordinator.entity_has_component<Zone>(target)) {
            auto& tz = global_coordinator.GetComponent<Zone>(target);
            if (tz.location == Zone::STACK) {
                std::string name = global_coordinator.entity_has_component<CardData>(target)
                    ? global_coordinator.GetComponent<CardData>(target).name : "<unknown>";
                if (global_coordinator.entity_has_component<Ability>(target))
                    global_coordinator.RemoveComponent<Ability>(target);
                if (global_coordinator.entity_has_component<Spell>(target))
                    global_coordinator.RemoveComponent<Spell>(target);
                orderer->add_to_zone(false, target, Zone::GRAVEYARD);
                printf("%s is countered\n", name.c_str());
            } else {
                printf("Counter: target is no longer on the stack\n");
            }
        } else {
            printf("Counter: target is no longer on the stack\n");
        }
    }

    //if there are subabilities, resolve them in sequence
    for (auto sub_ab : this->subabilities) {
        sub_ab.source = this->source;
        sub_ab.resolve(orderer);
    }
}

void Ability::resolve_destroy(std::shared_ptr<Orderer> orderer) {
    if (!global_coordinator.entity_has_component<Zone>(target)) {
        printf("Destroy: target is no longer in play\n");
        return;
    }
    auto &tz = global_coordinator.GetComponent<Zone>(target);
    if (tz.location != Zone::BATTLEFIELD) {
        printf("Destroy: target is no longer on the battlefield\n");
        return;
    }
    //TODO OTHER REASONS TARGET IS NOW ILLEGAL
    std::string name = global_coordinator.entity_has_component<CardData>(target)
        ? global_coordinator.GetComponent<CardData>(target).name
        : "<unknown>";
    orderer->add_to_zone(false, target, Zone::GRAVEYARD);
    printf("%s is destroyed\n", name.c_str());
}
