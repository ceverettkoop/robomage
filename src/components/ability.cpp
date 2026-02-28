#include "ability.h"

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
    std::shared_ptr<Orderer> orderer, Zone::Ownership owner, Zone::ZoneValue zone, const std::string &change_type) {
    // Parse comma-separated subtypes
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

    // Filter to matching cards
    std::vector<Entity> choices;
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

    const char *zone_name = (zone == Zone::LIBRARY)     ? "library"
                            : (zone == Zone::HAND)      ? "hand"
                            : (zone == Zone::GRAVEYARD) ? "graveyard"
                            : (zone == Zone::EXILE)     ? "exile"
                                                        : "zone";
    printf("Searching %s's %s:\n", player_name(owner).c_str(), zone_name);
    printf("  0: Fail to find\n");
    for (size_t i = 0; i < choices.size(); i++) {
        auto &cd = global_coordinator.GetComponent<CardData>(choices[i]);
        printf("  %zu: %s\n", i + 1, cd.name.c_str());
    }

    std::vector<ActionCategory> cats(choices.size() + 1, ActionCategory::SEARCH_LIBRARY);
    std::vector<Entity> search_entities;
    search_entities.push_back(Entity(0));  // index 0 = fail-to-find (null sentinel)
    for (auto e : choices) search_entities.push_back(e);
    int choice = InputLogger::instance().get_logged_input(cur_game.turn, cats, search_entities);
    if (choice >= 1 && choice <= static_cast<int>(choices.size())) {
        return choices[static_cast<size_t>(choice - 1)];
    }
    return 0;
}

void Ability::resolve_change_zone(std::shared_ptr<Orderer> orderer) {
    Zone::Ownership owner = global_coordinator.GetComponent<Zone>(source).owner;

    Entity chosen = search_zone(orderer, owner, origin, change_type);
    if (chosen != 0) {
        auto &chosen_cd = global_coordinator.GetComponent<CardData>(chosen);
        auto &chosen_zone = global_coordinator.GetComponent<Zone>(chosen);
        orderer->add_to_zone(false, chosen, destination);
        if (destination == Zone::BATTLEFIELD) {
            chosen_zone.controller = owner;
        }
        printf("%s searches and puts %s onto the battlefield\n", player_name(owner).c_str(), chosen_cd.name.c_str());
    } else {
        printf("%s fails to find\n", player_name(owner).c_str());
    }

    if (origin == Zone::LIBRARY) {
        orderer->shuffle_library(owner);
        printf("%s shuffles their library\n", player_name(owner).c_str());
    }
}

void Ability::resolve(std::shared_ptr<Orderer> orderer) {
    printf("Resolving ability (category: %s, amount: %zu)\n", category.c_str(), amount);

    if (category == "ChangeZone") {
        resolve_change_zone(orderer);
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
