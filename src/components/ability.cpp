#include "ability.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../mana_system.h"
#include "../systems/orderer.h"
#include "../error.h"
#include "creature.h"
#include "damage.h"
#include "permanent.h"
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
        game_log("Searching %s's %s:\n", player_name(owner).c_str(), zone_name);
    } else {
        game_log("%s chooses a card from %s %s:\n", player_name(owner).c_str(), player_name(owner).c_str(), zone_name);
    }

    std::vector<LegalAction> search_actions;
    if (show_fail_to_find) {
        LegalAction ftf(PASS_PRIORITY, Entity(0), std::string("Fail to find"));
        ftf.category = cat;
        search_actions.push_back(ftf);
    }
    for (auto entity : choices) {
        auto &cd = global_coordinator.GetComponent<CardData>(entity);
        LegalAction la(PASS_PRIORITY, entity, cd.name);
        la.category = cat;
        search_actions.push_back(la);
    }

    int choice = InputLogger::instance().get_input(search_actions);
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
            game_log("%s puts %s to %s\n", player_name(owner).c_str(), chosen_cd.name.c_str(),
                   destination == Zone::BATTLEFIELD ? "the battlefield" :
                   destination == Zone::LIBRARY     ? "top of library" :
                   destination == Zone::GRAVEYARD   ? "graveyard"      :
                   destination == Zone::HAND        ? "hand"           : "exile");
        } else {
            game_log("%s fails to find\n", player_name(owner).c_str());
            break;
        }
    }

    if (origin == Zone::LIBRARY) {
        orderer->shuffle_library(owner);
        game_log("%s shuffles their library\n", player_name(owner).c_str());
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

    game_log("%s looks at the top %zu card(s) of their library.\n", player_name(owner).c_str(), actual);

    std::vector<Entity> chosen_order;
    // Player picks N-1 cards; the last is automatic
    for (size_t pick = 0; pick + 1 < actual; pick++) {
        game_log("Choose which card goes on top next (pick %zu of %zu):\n", pick + 1, actual);
        std::vector<LegalAction> pick_actions;
        for (auto card : remaining) {
            auto &cd = global_coordinator.GetComponent<CardData>(card);
            LegalAction la(PASS_PRIORITY, card, cd.name);
            la.category = ActionCategory::TOP_LIBRARY;
            pick_actions.push_back(la);
        }
        int choice = InputLogger::instance().get_input(pick_actions);
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
        std::vector<LegalAction> shuffle_actions = {
            LegalAction(PASS_PRIORITY, std::string("Don't shuffle")),
            LegalAction(PASS_PRIORITY, std::string("Shuffle")),
        };
        shuffle_actions[0].category = ActionCategory::SHUFFLE;
        shuffle_actions[1].category = ActionCategory::SHUFFLE;
        int shuffle_choice = InputLogger::instance().get_input(shuffle_actions);
        if (shuffle_choice == 1) {
            orderer->shuffle_library(owner);
            game_log("%s shuffles their library.\n", player_name(owner).c_str());
        }
    }
}

// Returns true if the spell should be countered (controller declined or couldn't pay).
static bool run_unless_loop(size_t cost, Zone::Ownership controller,
                            std::shared_ptr<Orderer> orderer) {
    std::multiset<Colors> cond_cost;
    for (size_t i = 0; i < cost; i++) cond_cost.insert(GENERIC);

    while (true) {
        std::vector<LegalAction> unless_actions;

        for (auto e : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
            auto& perm = global_coordinator.GetComponent<Permanent>(e);
            if (perm.controller != controller || perm.is_tapped) continue;
            for (auto& ab : perm.abilities) {
                if (ab.category != "AddMana") continue;
                auto& cd = global_coordinator.GetComponent<CardData>(e);
                ActionCategory mcat = ActionCategory::MANA_C;
                switch (ab.color) {
                    case WHITE: mcat = ActionCategory::MANA_W; break;
                    case BLUE:  mcat = ActionCategory::MANA_U; break;
                    case BLACK: mcat = ActionCategory::MANA_B; break;
                    case RED:   mcat = ActionCategory::MANA_R; break;
                    case GREEN: mcat = ActionCategory::MANA_G; break;
                    default: break;
                }
                LegalAction la(PASS_PRIORITY, e, std::string("Tap ") + cd.name + " for {" + mana_symbol(ab.color) + "}");
                la.category = mcat;
                unless_actions.push_back(la);
                break;
            }
        }

        bool can_pay = can_afford(controller, cond_cost);
        size_t pay_idx = unless_actions.size();
        if (can_pay) {
            LegalAction pay(PASS_PRIORITY, std::string("Pay {") + std::to_string(cost) + "} (spell is not countered)");
            pay.category = ActionCategory::OTHER_CHOICE;
            unless_actions.push_back(pay);
        }
        size_t decline_idx = unless_actions.size();
        {
            LegalAction decline(PASS_PRIORITY, std::string("Don't pay (spell is countered)"));
            decline.category = ActionCategory::OTHER_CHOICE;
            unless_actions.push_back(decline);
        }

        int choice = InputLogger::instance().get_input(unless_actions);

        if (choice == static_cast<int>(decline_idx)) return true;

        if (can_pay && choice == static_cast<int>(pay_idx)) {
            spend_mana(controller, cond_cost);
            game_log("%s pays {%zu} — spell is not countered\n",
                   player_name(controller).c_str(), cost);
            return false;
        }

        if (choice >= 0 && choice < static_cast<int>(pay_idx)) {
            Entity land = unless_actions[static_cast<size_t>(choice)].source_entity;
            auto& perm = global_coordinator.GetComponent<Permanent>(land);
            auto& cd   = global_coordinator.GetComponent<CardData>(land);
            for (auto& ab : perm.abilities) {
                if (ab.category != "AddMana") continue;
                perm.is_tapped = true;
                add_mana(controller, ab.color, ab.amount);
                game_log("%s tapped %s for {%s}\n", player_name(controller).c_str(),
                       cd.name.c_str(), mana_symbol(ab.color).c_str());
                break;
            }
        }
    }
}

void Ability::fizzle(std::shared_ptr<Orderer> orderer){
    //stack manager present behavior moves everything to graveyard or destroys it
    //so for now this is a stub
    game_log("%s fizzles\n", this->category.c_str());
    return;

}

//TODO fix
//redundant with call in action processor and not generalizable!
bool Ability::is_target_valid() const {
    const std::string &vt = valid_tgts;

    if (target_type == "Spell") {
        return global_coordinator.entity_has_component<Zone>(target)
            && global_coordinator.GetComponent<Zone>(target).location == Zone::STACK
            && global_coordinator.entity_has_component<Spell>(target);
    }

    bool any          = (vt == "Any");
    bool inc_players   = any || vt.find("Player")   != std::string::npos;
    bool inc_creatures = any || vt.find("Creature") != std::string::npos;
    bool inc_lands     = any || vt.find("Land")     != std::string::npos;
    bool nonbasic_only = vt.find("nonBasic")        != std::string::npos;

    if (inc_players && global_coordinator.entity_has_component<Player>(target)) return true;

    if (!global_coordinator.entity_has_component<Zone>(target)) return false;
    auto &tz = global_coordinator.GetComponent<Zone>(target);
    if (tz.location != Zone::BATTLEFIELD) return false;
    if (!global_coordinator.entity_has_component<Permanent>(target)) return false;

    if (inc_creatures && global_coordinator.entity_has_component<Creature>(target)) return true;

    if (inc_lands && global_coordinator.entity_has_component<CardData>(target)) {
        auto &tcd = global_coordinator.GetComponent<CardData>(target);
        bool is_land = false, is_basic = false;
        for (auto &t : tcd.types) {
            if (t.kind == TYPE && t.name == "Land") is_land = true;
            if (t.kind == SUPERTYPE && t.name == "Basic") is_basic = true;
        }
        if (is_land && (!nonbasic_only || !is_basic)) return true;
    }

    return false;
}

void Ability::resolve(std::shared_ptr<Orderer> orderer) {
    //check if target is still valid!!!
    if(valid_tgts != "N_A"){
        if(!is_target_valid()){
            fizzle(orderer);
            return; //subabilities do not fire; TODO revisit this in light of cards e.g. k-command
        }
    }
    game_log("Resolving ability (category: %s, amount: %zu)\n", category.c_str(), amount);

    if (category == "GainLife") {
        Zone::Ownership controller;
        if (global_coordinator.entity_has_component<Permanent>(source)) {
            controller = global_coordinator.GetComponent<Permanent>(source).controller;
        } else {
            controller = global_coordinator.GetComponent<Zone>(source).owner;
        }
        Entity ctrl_entity = get_player_entity(controller);
        auto &player = global_coordinator.GetComponent<Player>(ctrl_entity);
        player.life_total += static_cast<int32_t>(amount);
        game_log("%s gains %zu life (now at %d)\n", player_name(controller).c_str(), amount, player.life_total);
    } else if (category == "Draw") {
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
            game_log("Dealt %zu damage to player (now at %d life)\n", amount, player.life_total);
        } else {
            if (deal_damage(source, target, amount)) {
                game_log("Dealt %zu damage to creature\n", amount);
            } else {
                non_fatal_error("Damage should have fizzled prior to this");
            }
        }
    } else if (category == "Destroy") {
        resolve_destroy(orderer);
    } else if (category == "Counter") {
        if (global_coordinator.entity_has_component<Zone>(target)) {
            auto& tz = global_coordinator.GetComponent<Zone>(target);
            if (tz.location == Zone::STACK) {
                Zone::Ownership target_controller = global_coordinator.entity_has_component<Spell>(target)
                    ? global_coordinator.GetComponent<Spell>(target).caster
                    : tz.owner;

                bool do_counter = true;
                if (unless_generic_cost > 0) {
                    std::string tname = global_coordinator.entity_has_component<CardData>(target)
                        ? global_coordinator.GetComponent<CardData>(target).name : "<unknown>";
                    game_log("%s's controller may pay {%zu} to save it:\n",
                           tname.c_str(), unless_generic_cost);
                    do_counter = run_unless_loop(unless_generic_cost, target_controller, orderer);
                }

                if (do_counter) {
                    std::string name = global_coordinator.entity_has_component<CardData>(target)
                        ? global_coordinator.GetComponent<CardData>(target).name : "<unknown>";
                    if (global_coordinator.entity_has_component<Ability>(target))
                        global_coordinator.RemoveComponent<Ability>(target);
                    if (global_coordinator.entity_has_component<Spell>(target))
                        global_coordinator.RemoveComponent<Spell>(target);
                    orderer->add_to_zone(false, target, Zone::GRAVEYARD);
                    game_log("%s is countered\n", name.c_str());
                }
            } else {
                non_fatal_error("Counter should have fizzled prior to this");
            }
        } else {
            non_fatal_error("Counter should have fizzled prior to this");
        }
    } else if (category == "Surveil") {
        resolve_surveil(orderer);
        //DONT SKIP SUBABILITIES

    //THIS BLOCK IS ALL SPECIFIC TO DELVER
    //TODO MAKE THIS GENERALIZABLE AND MOVE TO ITS OWN FUNCTION
    } else if (category == "PeekAndReveal") {
        auto& src_perm = global_coordinator.GetComponent<Permanent>(source);
        Zone::Ownership controller = src_perm.controller;

        // Find top card of controller's library
        Entity top_card = 0;
        for (auto e : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            auto& z = global_coordinator.GetComponent<Zone>(e);
            if (z.location == Zone::LIBRARY && z.owner == controller && z.distance_from_top == 0) {
                top_card = e;
                break;
            }
        }

        if (top_card == 0) {
            game_log("Library is empty — nothing to peek.\n");
            return;
        }
        auto& top_cd = global_coordinator.GetComponent<CardData>(top_card);
        game_log("Top card of library: %s\n", top_cd.name.c_str());
        std::vector<LegalAction> reveal_actions = {
            LegalAction(PASS_PRIORITY, std::string("Don't reveal")),
            LegalAction(PASS_PRIORITY, std::string("Reveal")),
        };
        int reveal_choice = InputLogger::instance().get_input(reveal_actions);

        if (reveal_choice == 1) {
            game_log("Revealed: %s\n", top_cd.name.c_str());
            bool is_instant_or_sorcery = false;
            for (auto& t : top_cd.types) {
                if (t.kind == TYPE && (t.name == "Instant" || t.name == "Sorcery")) {
                    is_instant_or_sorcery = true;
                    break;
                }
            }
            if (is_instant_or_sorcery && global_coordinator.entity_has_component<CardData>(source)) {
                auto& src_cd = global_coordinator.GetComponent<CardData>(source);
                if (src_cd.backside && !src_perm.transformed) {
                    src_perm.transformed = true;
                    if (global_coordinator.entity_has_component<Creature>(source))
                        global_coordinator.RemoveComponent<Creature>(source);
                    if (global_coordinator.entity_has_component<Damage>(source))
                        global_coordinator.RemoveComponent<Damage>(source);
                    Creature back_creature;
                    back_creature.power      = src_cd.backside->power;
                    back_creature.toughness  = src_cd.backside->toughness;
                    back_creature.keywords   = src_cd.backside->keywords;
                    global_coordinator.AddComponent(source, back_creature);
                    Damage dmg;
                    dmg.damage_counters = 0;
                    global_coordinator.AddComponent(source, dmg);
                    game_log("%s transforms into %s!\n",
                           src_cd.name.c_str(), src_cd.backside->name.c_str());
                }
            }
        }
        return;  // transform logic handled inline; skip subabilities loop
    }

    //if there are subabilities, resolve them in sequence
    for (auto sub_ab : this->subabilities) {
        sub_ab.source = this->source;
        sub_ab.resolve(orderer);
    }
}

void Ability::resolve_surveil(std::shared_ptr<Orderer> orderer) {
    Zone::Ownership controller;
    if (global_coordinator.entity_has_component<Permanent>(source)) {
        controller = global_coordinator.GetComponent<Permanent>(source).controller;
    } else {
        controller = global_coordinator.GetComponent<Zone>(source).owner;
    }

    for (size_t i = 0; i < amount; i++) {
        std::vector<Entity> lib = orderer->get_library_contents(controller);
        if (lib.empty()) {
            game_log("%s's library is empty — nothing to surveil.\n", player_name(controller).c_str());
            break;
        }

        // Find the top card (minimum distance_from_top)
        Entity top_card = lib[0];
        size_t min_dist = global_coordinator.GetComponent<Zone>(lib[0]).distance_from_top;
        for (auto e : lib) {
            size_t d = global_coordinator.GetComponent<Zone>(e).distance_from_top;
            if (d < min_dist) { min_dist = d; top_card = e; }
        }

        auto &top_cd = global_coordinator.GetComponent<CardData>(top_card);
        game_log("Top card of %s's library: %s\n", player_name(controller).c_str(), top_cd.name.c_str());
        std::vector<LegalAction> surveil_actions = {
            LegalAction(PASS_PRIORITY, std::string("Keep on top")),
            LegalAction(PASS_PRIORITY, std::string("Put in graveyard")),
        };
        int choice = InputLogger::instance().get_input(surveil_actions);

        if (choice == 1) {
            orderer->add_to_zone(false, top_card, Zone::GRAVEYARD);
            game_log("%s puts %s into the graveyard.\n", player_name(controller).c_str(), top_cd.name.c_str());
        }
    }
}

void Ability::resolve_destroy(std::shared_ptr<Orderer> orderer) {
    if (!global_coordinator.entity_has_component<Zone>(target)) {
        game_log("Destroy: target is no longer in play\n");
        return;
    }
    auto &tz = global_coordinator.GetComponent<Zone>(target);
    if (tz.location != Zone::BATTLEFIELD) {
        game_log("Destroy: target is no longer on the battlefield\n");
        return;
    }
    //TODO OTHER REASONS TARGET IS NOW ILLEGAL
    std::string name = global_coordinator.entity_has_component<CardData>(target)
        ? global_coordinator.GetComponent<CardData>(target).name
        : "<unknown>";
    orderer->add_to_zone(false, target, Zone::GRAVEYARD);
    game_log("%s is destroyed\n", name.c_str());
}
