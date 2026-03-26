#include "ability.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/token.h"
#include "../components/types.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../game_queries.h"
#include "../input_logger.h"
#include "../mana_system.h"
#include "../systems/orderer.h"
#include "../error.h"
#include "../parse.h"
#include "creature.h"
#include "damage.h"
#include "permanent.h"
#include "player.h"
#include "spell.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static size_t evaluate_dynamic_amount(const std::string &expr, Zone::Ownership ctrl,
                                      std::shared_ptr<Orderer> orderer, Entity target);

//edge case of two identical abilities being applied from two sources not handled
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

// Checks if a card entity matches a single filter spec.
// Supports: plain type name ("Forest"), dot-qualified color ("Creature.Green"),
// and +cmcLEX (CMC <= cur_game.x_paid).
static bool matches_filter_spec(Entity entity, const std::string &spec) {
    auto &cd = global_coordinator.GetComponent<CardData>(entity);

    // Split on '+' for additional constraints (e.g. "Creature.Green+cmcLEX")
    std::string type_part = spec;
    bool has_cmc_le_x = false;
    size_t plus_pos = spec.find('+');
    if (plus_pos != std::string::npos) {
        type_part = spec.substr(0, plus_pos);
        std::string constraint = spec.substr(plus_pos + 1);
        if (constraint == "cmcLEX") has_cmc_le_x = true;
    }

    // Split type_part on '.' for color qualifier (e.g. "Creature.Green")
    std::string type_name = type_part;
    std::string color_qualifier;
    size_t dot_pos = type_part.find('.');
    if (dot_pos != std::string::npos) {
        type_name = type_part.substr(0, dot_pos);
        color_qualifier = type_part.substr(dot_pos + 1);
    }

    // Check type match
    bool type_matches = false;
    for (auto &t : cd.types) {
        if (t.name == type_name) { type_matches = true; break; }
    }
    if (!type_matches) return false;

    // Check color qualifier
    if (!color_qualifier.empty()) {
        Colors required_color = NO_COLOR;
        if      (color_qualifier == "Green") required_color = GREEN;
        else if (color_qualifier == "White") required_color = WHITE;
        else if (color_qualifier == "Blue")  required_color = BLUE;
        else if (color_qualifier == "Black") required_color = BLACK;
        else if (color_qualifier == "Red")   required_color = RED;

        // Check explicit_colors first, then mana cost colors
        bool has_color = false;
        if (!cd.explicit_colors.empty()) {
            has_color = cd.explicit_colors.count(required_color) > 0;
        } else {
            has_color = cd.mana_cost.count(required_color) > 0;
        }
        if (!has_color) return false;
    }

    // Check CMC <= X constraint
    if (has_cmc_le_x) {
        size_t cmc = cd.mana_cost.size();
        if (cmc > cur_game.x_paid) return false;
    }

    return true;
}

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
        // Check if any filter spec uses extended syntax (dot/plus qualifiers)
        bool has_extended = false;
        for (auto &st : subtypes) {
            if (st.find('.') != std::string::npos || st.find('+') != std::string::npos) {
                has_extended = true;
                break;
            }
        }

        for (auto entity : zone_contents) {
            bool matches = false;
            if (has_extended) {
                for (auto &st : subtypes) {
                    if (matches_filter_spec(entity, st)) { matches = true; break; }
                }
            } else {
                auto &cd = global_coordinator.GetComponent<CardData>(entity);
                for (auto &t : cd.types) {
                    for (auto &st : subtypes) {
                        if (t.name == st) { matches = true; break; }
                    }
                    if (matches) break;
                }
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

    const char* dest_str = destination == Zone::BATTLEFIELD ? "the battlefield" :
                           destination == Zone::LIBRARY     ? "top of library" :
                           destination == Zone::GRAVEYARD   ? "graveyard"      :
                           destination == Zone::HAND        ? "hand"           : "exile";

    // Targeted ChangeZone (e.g. Swords to Plowshares): move the target directly
    if (valid_tgts != "N_A" && target != 0) {
        if (!global_coordinator.entity_has_component<Zone>(target)) return;
        std::string tname = global_coordinator.entity_has_component<CardData>(target)
            ? global_coordinator.GetComponent<CardData>(target).name
            : (global_coordinator.entity_has_component<Permanent>(target)
                ? global_coordinator.GetComponent<Permanent>(target).name : "<unknown>");
        orderer->add_to_zone(false, target, destination);
        // Track exiled_with on the source permanent (for Keen-Eyed Curator)
        if (destination == Zone::EXILE && source != 0 &&
            global_coordinator.entity_has_component<Permanent>(source)) {
            global_coordinator.GetComponent<Permanent>(source).exiled_with.push_back(target);
        }
        game_log("%s is moved to %s\n", tname.c_str(), dest_str);
        return;
    }

    // Defined$ Self — move the source card directly (e.g. Talon Gates putting itself onto battlefield from hand)
    if (defined_self && source != 0) {
        std::string sname = global_coordinator.entity_has_component<CardData>(source)
            ? global_coordinator.GetComponent<CardData>(source).name : "<unknown>";
        orderer->add_to_zone(false, source, destination);
        if (destination == Zone::BATTLEFIELD) {
            auto &src_zone = global_coordinator.GetComponent<Zone>(source);
            src_zone.controller = owner;
        }
        game_log("%s is moved to %s\n", sname.c_str(), dest_str);
        return;
    }

    // Search-based ChangeZone (e.g. fetch lands, Green Sun's Zenith)
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
            bool dest_public = (destination == Zone::BATTLEFIELD ||
                                destination == Zone::GRAVEYARD   ||
                                destination == Zone::EXILE);
            if (dest_public) {
                game_log("%s puts %s to %s\n", player_name(owner).c_str(), chosen_cd.name.c_str(), dest_str);
            } else {
                game_log_private(owner, "%s puts %s to %s\n", player_name(owner).c_str(), chosen_cd.name.c_str(), dest_str);
                game_log("%s puts a card to %s\n", player_name(owner).c_str(), dest_str);
            }
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

    size_t num_cards = amount;
    if (!dynamic_amount_expr.empty())
        num_cards = evaluate_dynamic_amount(dynamic_amount_expr, owner, orderer, target);

    std::vector<Entity> lib = orderer->get_library_contents(owner);
    // Sort by distance_from_top ascending so lib[0] is the actual top card
    std::sort(lib.begin(), lib.end(), [](Entity a, Entity b) {
        return global_coordinator.GetComponent<Zone>(a).distance_from_top
             < global_coordinator.GetComponent<Zone>(b).distance_from_top;
    });
    //looking at top n only
    if (lib.size() > num_cards) lib.resize(num_cards);
    size_t actual = lib.size();
    std::vector<Entity> remaining = lib;

    game_log("%s looks at the top %zu card(s) of their library.\n", player_name(owner).c_str(), actual);

    std::vector<Entity> chosen_order;
    // Player picks N-1 cards; the last is automatic
    for (size_t pick = 0; pick + 1 < actual; pick++) {
        game_log("Choose which card goes %zu from top:\n", actual - pick);
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
    for (auto it : chosen_order) {
        orderer->add_to_zone(false, it, Zone::LIBRARY);
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
                            std::shared_ptr<Orderer> orderer, Entity paid_for) {
    std::multiset<Colors> cond_cost;
    for (size_t i = 0; i < cost; i++) cond_cost.insert(GENERIC);

    // the target's controller decides whether to pay, not the Daze caster
    bool prev_priority = cur_game.player_a_has_priority;
    cur_game.player_a_has_priority = (controller == Zone::PLAYER_A);

    while (true) {
        std::vector<LegalAction> unless_actions = collect_mana_legal_actions(controller, orderer);

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

        if (choice == static_cast<int>(decline_idx)) {
            cur_game.player_a_has_priority = prev_priority;
            return true;
        }

        if (can_pay && choice == static_cast<int>(pay_idx)) {
            spend_mana(controller, cond_cost, paid_for);
            game_log("%s pays {%zu} — spell is not countered\n",
                   player_name(controller).c_str(), cost);
            cur_game.player_a_has_priority = prev_priority;
            return false;
        }

        if (choice >= 0 && choice < static_cast<int>(pay_idx)) {
            auto &chosen = unless_actions[static_cast<size_t>(choice)];
            Entity land = chosen.source_entity;
            auto& perm = global_coordinator.GetComponent<Permanent>(land);
            perm.is_tapped = true;
            add_mana(controller, chosen.ability.color, chosen.ability.amount);
            game_log("%s tapped %s for {%s}\n", player_name(controller).c_str(),
                   perm.name.c_str(), mana_symbol(chosen.ability.color).c_str());
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
    // Optional targeting: no target chosen is valid
    if (target == 0 && target_min == 0) return true;

    const std::string &vt = valid_tgts;

    if (target_type == "Spell") {
        return global_coordinator.entity_has_component<Zone>(target)
            && global_coordinator.GetComponent<Zone>(target).location == Zone::STACK
            && global_coordinator.entity_has_component<Spell>(target);
    }

    bool any           = (vt == "Any");
    bool opp_only      = (vt == "Opponent");
    bool inc_players   = any || opp_only || vt.find("Player") != std::string::npos;
    bool inc_creatures = any || vt.find("Creature") != std::string::npos;
    bool inc_lands     = vt.find("Land")     != std::string::npos;
    bool nonbasic_only = vt.find("nonBasic")        != std::string::npos;
    bool legendary_only = vt.find("Legendary")      != std::string::npos;

    if (inc_players && global_coordinator.entity_has_component<Player>(target)) return true;

    if (!global_coordinator.entity_has_component<Zone>(target)) return false;
    auto &tz = global_coordinator.GetComponent<Zone>(target);
    if (tz.location != Zone::BATTLEFIELD) return false;
    if (!global_coordinator.entity_has_component<Permanent>(target)) return false;

    if (inc_creatures && global_coordinator.entity_has_component<Creature>(target)) {
        if (legendary_only) {
            auto &cperm = global_coordinator.GetComponent<Permanent>(target);
            bool is_legendary = false;
            for (auto &t : cperm.types)
                if (t.kind == SUPERTYPE && t.name == "Legendary") { is_legendary = true; break; }
            if (!is_legendary) return false;
        }
        if (has_protection_from(global_coordinator.GetComponent<Creature>(target), source))
            return false;
        return true;
    }

    if (inc_lands) {
        auto &tperm = global_coordinator.GetComponent<Permanent>(target);
        bool is_land = false, is_basic = false;
        for (auto &t : tperm.types) {
            if (t.kind == TYPE && t.name == "Land") is_land = true;
            if (t.kind == SUPERTYPE && t.name == "Basic") is_basic = true;
        }
        if (is_land && (!nonbasic_only || !is_basic)) return true;
    }

    return false;
}

// Evaluates a condition SVar expression against cur_game state.
static int evaluate_condition_svar(const std::string &expr, Entity src) {
    if (expr == "Count$ResolvedThisTurn") {
        auto it = cur_game.ability_resolution_counts.find(src);
        return (it != cur_game.ability_resolution_counts.end()) ? it->second : 0;
    }
    return 0;
}

// Returns true if val passes the compare spec (e.g. "EQ2", "NE2", "GE1", "LE3").
static bool compare_svar(int val, const std::string &spec) {
    if (spec.size() < 3) return true;  // unknown spec: always pass
    std::string op = spec.substr(0, 2);
    int rhs = std::stoi(spec.substr(2));
    if (op == "EQ") return val == rhs;
    if (op == "NE") return val != rhs;
    if (op == "GE") return val >= rhs;
    if (op == "LE") return val <= rhs;
    if (op == "GT") return val >  rhs;
    if (op == "LT") return val <  rhs;
    return true;
}

// Evaluates a dynamic_amount_expr at runtime for the given controller.
// Supports: Count$InYourLibrary, Count$YourLifeTotal, Count$YourLifeTotal/HalfUp,
//           Count$Valid Creature.YouCtrl, Targeted$CardPower.
static size_t evaluate_dynamic_amount(const std::string &expr, Zone::Ownership ctrl,
                                      std::shared_ptr<Orderer> orderer, Entity target) {
    if (expr.find("Count$InYourLibrary") != std::string::npos) {
        return orderer->get_library_contents(ctrl).size();
    }
    if (expr.find("Count$YourLifeTotal") != std::string::npos) {
        Entity ctrl_entity = (ctrl == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
        auto &player = global_coordinator.GetComponent<Player>(ctrl_entity);
        int life = player.life_total;
        if (life < 0) life = 0;
        if (expr.find("/HalfUp") != std::string::npos) {
            return static_cast<size_t>((life + 1) / 2);
        }
        return static_cast<size_t>(life);
    }
    if (expr.find("Count$Valid Creature.YouCtrl") != std::string::npos) {
        size_t count = 0;
        for (auto e : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Creature>(e)) continue;
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location != Zone::BATTLEFIELD) continue;
            if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
            if (global_coordinator.GetComponent<Permanent>(e).controller != ctrl) continue;
            count++;
        }
        return count;
    }
    if (expr.find("Targeted$CardPower") != std::string::npos) {
        if (global_coordinator.entity_has_component<CardData>(target))
            return static_cast<size_t>(global_coordinator.GetComponent<CardData>(target).power);
        if (global_coordinator.entity_has_component<Creature>(target))
            return static_cast<size_t>(global_coordinator.GetComponent<Creature>(target).power);
    }
    return 0;
}

void Ability::resolve(std::shared_ptr<Orderer> orderer) {
    // Pre-resolve target validity check — skipped for categories that select their own target internally
    if (valid_tgts != "N_A" && category != "Pump") {
        if (!is_target_valid()) {
            fizzle(orderer);
            return; //subabilities do not fire; TODO revisit this in light of cards e.g. k-command
        }
    }
    game_log("Resolving ability (category: %s, amount: %zu)\n", category.c_str(), amount);

    // Conditional execution: if condition fails, skip this ability's body but still chain subabilities
    bool condition_passed = true;
    if (!condition_check_svar.empty()) {
        int val = evaluate_condition_svar(condition_check_svar, source);
        condition_passed = compare_svar(val, condition_svar_compare);
    }
    if (!condition_passed) {
        for (auto sub_ab : this->subabilities) {
            sub_ab.source = this->source;
            sub_ab.target = this->target;
            sub_ab.controller = this->controller;
            sub_ab.resolve(orderer);
        }
        return;
    }

    if (category == "GainLife") {
        Zone::Ownership gain_controller;
        if (defined_targeted_controller && global_coordinator.entity_has_component<Zone>(target)) {
            // Swords to Plowshares: gain life goes to the exiled creature's controller
            gain_controller = global_coordinator.GetComponent<Zone>(target).controller;
            if (gain_controller == Zone::UNKNOWN && global_coordinator.entity_has_component<Permanent>(target))
                gain_controller = global_coordinator.GetComponent<Permanent>(target).controller;
        } else if (global_coordinator.entity_has_component<Permanent>(source)) {
            gain_controller = global_coordinator.GetComponent<Permanent>(source).controller;
        } else {
            gain_controller = global_coordinator.GetComponent<Zone>(source).owner;
        }
        // Evaluate dynamic amount if set (e.g. "Targeted$CardPower")
        size_t gain_amount = amount;
        if (!dynamic_amount_expr.empty() && dynamic_amount_expr.find("Targeted$CardPower") != std::string::npos) {
            if (global_coordinator.entity_has_component<CardData>(target)) {
                gain_amount = static_cast<size_t>(global_coordinator.GetComponent<CardData>(target).power);
            } else if (global_coordinator.entity_has_component<Creature>(target)) {
                gain_amount = static_cast<size_t>(global_coordinator.GetComponent<Creature>(target).power);
            }
        }
        Entity ctrl_entity = get_player_entity(gain_controller);
        auto &player = global_coordinator.GetComponent<Player>(ctrl_entity);
        player.life_total += static_cast<int32_t>(gain_amount);
        game_log("%s gains %zu life (now at %d)\n", player_name(gain_controller).c_str(), gain_amount, player.life_total);
    } else if (category == "LoseLife") {
        Zone::Ownership lose_controller = controller;
        size_t lose_amount = amount;
        if (!dynamic_amount_expr.empty())
            lose_amount = evaluate_dynamic_amount(dynamic_amount_expr, lose_controller, orderer, target);
        Entity ctrl_entity = (lose_controller == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
        auto &player = global_coordinator.GetComponent<Player>(ctrl_entity);
        player.life_total -= static_cast<int32_t>(lose_amount);
        game_log("%s loses %zu life (now at %d)\n", player_name(lose_controller).c_str(), lose_amount, player.life_total);
    } else if (category == "Discard") {
        // RevealYouChoose: target player reveals hand, caster picks a card matching filter
        Zone::Ownership tgt_owner = Zone::PLAYER_A;
        if (global_coordinator.entity_has_component<Player>(target)) {
            tgt_owner = (target == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
        }
        std::vector<Entity> hand = orderer->get_hand(tgt_owner);
        game_log("%s reveals their hand:\n", player_name(tgt_owner).c_str());
        for (auto e : hand) {
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            game_log("  %s\n", cd.name.c_str());
        }

        // Filter by DiscardValid$
        // Format: "Card.nonLand", "Card.nonCreature+nonLand"
        std::vector<Entity> valid;
        for (auto e : hand) {
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            bool passes = true;
            if (!discard_valid.empty()) {
                // Parse the filter: strip "Card." prefix, split on '+' for constraints
                std::string filter = discard_valid;
                if (filter.rfind("Card.", 0) == 0) filter = filter.substr(5);
                // Split on '+'
                size_t fp = 0;
                while (fp < filter.size()) {
                    size_t plus = filter.find('+', fp);
                    if (plus == std::string::npos) plus = filter.size();
                    std::string constraint = filter.substr(fp, plus - fp);
                    // "nonLand" means card must NOT be a Land
                    if (constraint.rfind("non", 0) == 0) {
                        std::string excluded_type = constraint.substr(3);
                        for (auto &t : cd.types) {
                            if (t.name == excluded_type) { passes = false; break; }
                        }
                    }
                    if (!passes) break;
                    fp = plus + 1;
                }
            }
            if (passes) valid.push_back(e);
        }

        if (valid.empty()) {
            game_log("No valid cards to discard.\n");
        } else {
            // Caster chooses which card to discard
            bool prev_priority = cur_game.player_a_has_priority;
            cur_game.player_a_has_priority = (controller == Zone::PLAYER_A);
            game_log("%s chooses a card to discard:\n", player_name(controller).c_str());
            std::vector<LegalAction> discard_actions;
            for (auto e : valid) {
                auto &cd = global_coordinator.GetComponent<CardData>(e);
                LegalAction la(PASS_PRIORITY, e, cd.name);
                la.category = ActionCategory::OTHER_CHOICE;
                discard_actions.push_back(la);
            }
            int choice = InputLogger::instance().get_input(discard_actions);
            Entity chosen = valid[static_cast<size_t>(choice)];
            auto &cd = global_coordinator.GetComponent<CardData>(chosen);
            game_log("%s discards %s\n", player_name(tgt_owner).c_str(), cd.name.c_str());
            orderer->add_to_zone(false, chosen, Zone::GRAVEYARD);
            cur_game.player_a_has_priority = prev_priority;
        }
    } else if (category == "Draw") {
        Zone::Ownership owner = global_coordinator.GetComponent<Zone>(source).owner;
        orderer->draw(owner, amount);
    } else if (category == "ChangeZone") {
        resolve_change_zone(orderer);
    } else if (category == "RearrangeTopOfLibrary") {
        resolve_rearrange_top_of_library(orderer);
    } else if (category == "DealDamage") {
        // Delirium-conditional damage (Unholy Heat)
        size_t dmg = amount;
        if (amount_is_delirium_scale) {
            Zone::Ownership caster = global_coordinator.entity_has_component<Permanent>(source)
                ? global_coordinator.GetComponent<Permanent>(source).controller
                : global_coordinator.GetComponent<Zone>(source).owner;
            if (check_delirium(caster, orderer->mEntities)) dmg = amount_delirium;
        }
        if (global_coordinator.entity_has_component<Player>(target)) {
            auto &player = global_coordinator.GetComponent<Player>(target);
            player.life_total -= static_cast<int32_t>(dmg);
            game_log("Dealt %zu damage to player (now at %d life)\n", dmg, player.life_total);
        } else {
            if (deal_damage(source, target, dmg)) {
                game_log("Dealt %zu damage to creature\n", dmg);
            } else {
                #ifndef NDEBUG
                fprintf(stderr,"SOURCE:");
                dump_entity(source);
                fprintf(stderr,"TARGET:");
                dump_entity(target);
                #endif
                non_fatal_error("Damage should have fizzled prior to this");
            }
        }
    } else if (category == "PutCounter") {
        resolve_put_counter();
    } else if (category == "ProwessBonus") {
        if (global_coordinator.entity_has_component<Creature>(source)) {
            auto &cr = global_coordinator.GetComponent<Creature>(source);
            cr.prowess_bonus += static_cast<int>(amount);
            cr.power     += static_cast<uint32_t>(amount);
            cr.toughness += static_cast<uint32_t>(amount);
            game_log("Prowess: creature gets +%zu/+%zu until end of turn.\n", amount, amount);
        }
    } else if (category == "ExaltedBonus") {
        if (target != 0 && global_coordinator.entity_has_component<Creature>(target)) {
            auto &cr = global_coordinator.GetComponent<Creature>(target);
            cr.prowess_bonus += static_cast<int>(amount);
            cr.power     += static_cast<uint32_t>(amount);
            cr.toughness += static_cast<uint32_t>(amount);
            std::string tgt_name = global_coordinator.entity_has_component<CardData>(target)
                ? global_coordinator.GetComponent<CardData>(target).name
                : (global_coordinator.entity_has_component<Permanent>(target)
                    ? global_coordinator.GetComponent<Permanent>(target).name : "creature");
            std::string src_name = global_coordinator.entity_has_component<CardData>(source)
                ? global_coordinator.GetComponent<CardData>(source).name
                : (global_coordinator.entity_has_component<Permanent>(source)
                    ? global_coordinator.GetComponent<Permanent>(source).name : "permanent");
            game_log("Exalted (%s): %s gets +%zu/+%zu until end of turn.\n",
                     src_name.c_str(), tgt_name.c_str(), amount, amount);
        }
    } else if (category == "Token") {
        resolve_token(orderer);
    } else if (category == "Attach") {
        // Equip the source equipment to the remembered entity
        Entity equip_entity = source;
        Entity target_creature = defined_remembered ? cur_game.remembered_entity : target;

        if (optional) {
            // Ask the controller whether to attach
            game_log("Attach equipment to token? (0=No 1=Yes)\n");
            std::vector<LegalAction> attach_actions = {
                LegalAction(PASS_PRIORITY, std::string("No")),
                LegalAction(PASS_PRIORITY, std::string("Yes")),
            };
            attach_actions[0].category = ActionCategory::OTHER_CHOICE;
            attach_actions[1].category = ActionCategory::OTHER_CHOICE;
            int choice = InputLogger::instance().get_input(attach_actions);
            if (choice == 0) goto attach_done;
        }
        if (target_creature != 0 &&
            global_coordinator.entity_has_component<Permanent>(equip_entity) &&
            global_coordinator.entity_has_component<Permanent>(target_creature)) {
            auto &eq_perm = global_coordinator.GetComponent<Permanent>(equip_entity);
            // Detach from previous creature
            if (eq_perm.equipped_to != 0 &&
                global_coordinator.entity_has_component<Permanent>(eq_perm.equipped_to)) {
                global_coordinator.GetComponent<Permanent>(eq_perm.equipped_to).equipped_by = 0;
            }
            eq_perm.equipped_to = target_creature;
            global_coordinator.GetComponent<Permanent>(target_creature).equipped_by = equip_entity;
            game_log("Equipment attached.\n");
        }
        attach_done:;
    } else if (category == "Mill") {
        // Move top N cards from target player's library to graveyard
        Zone::Ownership mill_owner = controller;
        size_t mill_count = (amount > 0) ? amount : 1;
        std::vector<Entity> lib = orderer->get_library_contents(mill_owner);
        std::sort(lib.begin(), lib.end(), [](Entity a, Entity b) {
            return global_coordinator.GetComponent<Zone>(a).distance_from_top
                 < global_coordinator.GetComponent<Zone>(b).distance_from_top;
        });
        for (size_t i = 0; i < mill_count && i < lib.size(); i++) {
            std::string cname = global_coordinator.entity_has_component<CardData>(lib[i])
                ? global_coordinator.GetComponent<CardData>(lib[i]).name : "card";
            orderer->add_to_zone(false, lib[i], Zone::GRAVEYARD);
            game_log("%s mills %s.\n", player_name(mill_owner).c_str(), cname.c_str());
        }
    } else if (category == "Pump") {
        // Present target selection, then chain subabilities with that target
        Zone::Ownership ctrl = controller;
        std::vector<Entity> pump_targets;
        for (Entity e = 0; e < MAX_ENTITIES; ++e) {
            if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
            if (!global_coordinator.entity_has_component<Creature>(e)) continue;
            auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location != Zone::BATTLEFIELD) continue;
            auto &p = global_coordinator.GetComponent<Permanent>(e);
            if (valid_tgts.find("YouCtrl") != std::string::npos && p.controller != ctrl) continue;
            pump_targets.push_back(e);
        }
        if (pump_targets.empty()) {
            game_log("Pump: no valid targets.\n");
            // still chain subabilities with no target
        } else {
            game_log("Choose a creature for Pump:\n");
            std::vector<LegalAction> tgt_actions;
            for (auto te : pump_targets) {
                std::string ename = global_coordinator.GetComponent<Permanent>(te).name;
                auto &tcr = global_coordinator.GetComponent<Creature>(te);
                LegalAction la(PASS_PRIORITY, te, ename + " [" + std::to_string(tcr.power) + "/" + std::to_string(tcr.toughness) + "]");
                la.category = ActionCategory::SELECT_TARGET;
                tgt_actions.push_back(la);
            }
            int choice = InputLogger::instance().get_input(tgt_actions);
            if (choice >= 0 && choice < static_cast<int>(pump_targets.size()))
                this->target = pump_targets[static_cast<size_t>(choice)];
        }
    } else if (category == "MultiplyCounter") {
        // Double all P1P1 counters on target creature
        Entity tgt = (target != 0) ? target : source;
        if (global_coordinator.entity_has_component<Creature>(tgt)) {
            auto &cr = global_coordinator.GetComponent<Creature>(tgt);
            if (cr.plus_one_counters > 0) {
                cr.plus_one_counters *= 2;
                cr.power     += static_cast<uint32_t>(cr.plus_one_counters / 2);
                cr.toughness += static_cast<uint32_t>(cr.plus_one_counters / 2);
                game_log("MultiplyCounter: doubled +1/+1 counters on creature (now %u/%u).\n", cr.power, cr.toughness);
            }
        }
    } else if (category == "Cleanup") {
        if (clear_remembered) cur_game.remembered_entity = 0;
    } else if (category == "DelayedTrigger") {
        resolve_delayed_trigger();
    } else if (category == "Untap") {
        if (global_coordinator.entity_has_component<Permanent>(target)) {
            auto &tperm = global_coordinator.GetComponent<Permanent>(target);
            tperm.is_tapped = false;
            std::string tname = tperm.name;
            game_log("%s untaps\n", tname.c_str());
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
                    do_counter = run_unless_loop(unless_generic_cost, target_controller, orderer, target);
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

    //THIS BLOCK IS ALL SPECIFIC TO DELVER (and Mishra's Bauble peek variant)
    //TODO MAKE THIS GENERALIZABLE AND MOVE TO ITS OWN FUNCTION
    } else if (category == "PeekAndReveal") {
        if (is_peek_no_reveal) {
            // Mishra's Bauble: look at target player's top card privately, no reveal choice
            Zone::Ownership peek_owner = global_coordinator.entity_has_component<Player>(target)
                ? (target == cur_game.player_a_entity ? Zone::PLAYER_A : Zone::PLAYER_B)
                : controller;
            Entity top_card = 0;
            for (auto e : orderer->mEntities) {
                if (!global_coordinator.entity_has_component<Zone>(e)) continue;
                auto& z = global_coordinator.GetComponent<Zone>(e);
                if (z.location == Zone::LIBRARY && z.owner == peek_owner && z.distance_from_top == 0) {
                    top_card = e;
                    break;
                }
            }
            if (top_card == 0) {
                game_log("%s's library is empty — nothing to peek.\n", player_name(peek_owner).c_str());
            } else if (global_coordinator.entity_has_component<CardData>(top_card)) {
                auto& top_cd = global_coordinator.GetComponent<CardData>(top_card);
                game_log_private(controller, "%s looks at top of %s's library: %s\n",
                    player_name(controller).c_str(), player_name(peek_owner).c_str(), top_cd.name.c_str());
            }
            // fall through to subabilities (DelayedTrigger sub-ability fires next upkeep)
        } else {
            // Delver of Secrets: peek own library top, optionally reveal
            if (!global_coordinator.entity_has_component<Permanent>(source)) {
                fizzle(orderer);
                return;
            }
            auto& src_perm = global_coordinator.GetComponent<Permanent>(source);
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
            game_log_private(controller, "Top card of library: %s\n", top_cd.name.c_str());
            std::vector<LegalAction> reveal_actions = {
                LegalAction(PASS_PRIORITY, top_card, std::string("Don't reveal")),
                LegalAction(PASS_PRIORITY, top_card, std::string("Reveal")),
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
                               src_perm.name.c_str(), src_cd.backside->name.c_str());
                    }
                }
            }
            return;  // transform logic handled inline; skip subabilities loop
        }
    } else if (category == "Phases") {
        // Phase out target permanent
        if (target != 0 && global_coordinator.entity_has_component<Permanent>(target)) {
            auto &tgt_perm = global_coordinator.GetComponent<Permanent>(target);
            tgt_perm.is_phased_out = true;
            game_log("%s phases out\n", tgt_perm.name.c_str());
        }
    } else if (category == "Dig") {
        // Look at top N cards, player picks one matching filter, rest go to bottom
        Zone::Ownership dig_owner = controller;
        std::vector<Entity> lib = orderer->get_library_contents(dig_owner);
        std::sort(lib.begin(), lib.end(), [](Entity a, Entity b) {
            return global_coordinator.GetComponent<Zone>(a).distance_from_top
                 < global_coordinator.GetComponent<Zone>(b).distance_from_top;
        });
        if (lib.size() > dig_num) lib.resize(dig_num);

        // Parse change_valid filters (comma-separated "Card.Creature,Card.Land" etc.)
        std::vector<std::string> filters;
        if (!change_valid.empty()) {
            size_t fp = 0;
            while (true) {
                size_t comma = change_valid.find(',', fp);
                if (comma == std::string::npos) {
                    filters.push_back(change_valid.substr(fp));
                    break;
                }
                filters.push_back(change_valid.substr(fp, comma - fp));
                fp = comma + 1;
            }
        }

        // Filter matching cards
        std::vector<Entity> matching;
        for (auto e : lib) {
            if (filters.empty()) { matching.push_back(e); continue; }
            bool card_matches = false;
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            for (auto &f : filters) {
                // "Card.Creature" → check for Creature type
                // "Card.Land" → check for Land type
                std::string type_name;
                size_t dot = f.find('.');
                if (dot != std::string::npos) type_name = f.substr(dot + 1);
                else type_name = f;
                for (auto &t : cd.types) {
                    if (t.name == type_name) { card_matches = true; break; }
                }
                if (card_matches) break;
            }
            if (card_matches) matching.push_back(e);
        }

        game_log("%s looks at the top %zu card(s) of their library.\n",
                 player_name(dig_owner).c_str(), lib.size());

        // Present choices
        std::vector<LegalAction> dig_actions;
        if (optional_choice) {
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
            orderer->add_to_zone(false, chosen, Zone::HAND);
            auto &cd = global_coordinator.GetComponent<CardData>(chosen);
            game_log_private(dig_owner, "%s puts %s into hand.\n",
                     player_name(dig_owner).c_str(), cd.name.c_str());
        }

        // Remaining cards go to bottom of library
        std::vector<Entity> remaining;
        for (auto e : lib) {
            if (e != chosen) remaining.push_back(e);
        }
        if (rest_random_order) {
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
        game_log("%s puts %zu card(s) on the bottom of their library.\n",
                 player_name(dig_owner).c_str(), remaining.size());
    } else if (category == "SylvanLibrary") {
        // Draw 2, then for each card drawn this turn still in hand, choose: pay 4 life or put on top
        Zone::Ownership ctrl = controller;
        Entity ctrl_entity = (ctrl == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
        auto &pl = global_coordinator.GetComponent<Player>(ctrl_entity);

        // Draw 2 cards
        orderer->draw(ctrl, 2);
        game_log("%s draws 2 cards (Sylvan Library)\n", player_name(ctrl).c_str());

        // Get cards drawn this turn that are still in hand
        std::vector<Entity> drawn_in_hand;
        for (auto e : pl.cards_drawn_this_turn) {
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location == Zone::HAND && z.owner == ctrl) {
                drawn_in_hand.push_back(e);
            }
        }

        // Player chooses 2 cards (or fewer if not enough in hand)
        size_t to_choose = std::min(drawn_in_hand.size(), static_cast<size_t>(2));
        std::vector<Entity> chosen_cards;
        for (size_t pick = 0; pick < to_choose; pick++) {
            game_log("Choose a card drawn this turn (%zu remaining):\n", to_choose - pick);
            std::vector<LegalAction> choose_actions;
            for (auto e : drawn_in_hand) {
                // Skip already chosen
                bool already = false;
                for (auto c : chosen_cards) { if (c == e) { already = true; break; } }
                if (already) continue;
                auto &cd = global_coordinator.GetComponent<CardData>(e);
                LegalAction la(PASS_PRIORITY, e, cd.name);
                la.category = ActionCategory::OTHER_CHOICE;
                choose_actions.push_back(la);
            }
            if (choose_actions.empty()) break;
            int choice = InputLogger::instance().get_input(choose_actions);
            chosen_cards.push_back(choose_actions[static_cast<size_t>(choice)].source_entity);
        }

        // For each chosen card: pay 4 life or put on top of library
        for (auto card : chosen_cards) {
            auto &cd = global_coordinator.GetComponent<CardData>(card);
            game_log("For %s: pay 4 life or put on top of library?\n", cd.name.c_str());
            std::vector<LegalAction> pay_actions = {
                LegalAction(PASS_PRIORITY, std::string("Pay 4 life")),
                LegalAction(PASS_PRIORITY, std::string("Put on top of library")),
            };
            pay_actions[0].category = ActionCategory::OTHER_CHOICE;
            pay_actions[1].category = ActionCategory::OTHER_CHOICE;
            int choice = InputLogger::instance().get_input(pay_actions);
            if (choice == 0) {
                pl.life_total -= 4;
                game_log("%s pays 4 life (now at %d)\n", player_name(ctrl).c_str(), pl.life_total);
            } else {
                orderer->add_to_zone(false, card, Zone::LIBRARY);
                game_log("%s puts %s on top of library\n", player_name(ctrl).c_str(), cd.name.c_str());
            }
        }
    }

    //if there are subabilities, resolve them in sequence
    for (auto sub_ab : this->subabilities) {
        sub_ab.source = this->source;
        sub_ab.target = this->target;  // propagate target so GainLife etc. can reference it
        sub_ab.controller = this->controller;
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
        game_log_private(controller, "Top card of %s's library: %s\n", player_name(controller).c_str(), top_cd.name.c_str());
        std::vector<LegalAction> surveil_actions = {
            LegalAction(PASS_PRIORITY, top_card, std::string("Keep on top")),
            LegalAction(PASS_PRIORITY, top_card, std::string("Put in graveyard")),
        };
        int choice = InputLogger::instance().get_input(surveil_actions);

        if (choice == 1) {
            orderer->add_to_zone(false, top_card, Zone::GRAVEYARD);
            game_log("%s puts %s into the graveyard.\n", player_name(controller).c_str(), top_cd.name.c_str());
        }
    }
}

void Ability::resolve_put_counter() {
    // Use target if set (e.g. from a Pump parent), otherwise put counters on source
    Entity counter_tgt = (target != 0 && global_coordinator.entity_has_component<Creature>(target))
                         ? target : source;
    if (!global_coordinator.entity_has_component<Creature>(counter_tgt)) return;
    auto &cr = global_coordinator.GetComponent<Creature>(counter_tgt);
    if (counter_type == "P1P1") {
        int n = counter_count;
        if (counter_count_from_delve) {
            n = static_cast<int>(cur_game.delve_exiled.size());
            cur_game.delve_exiled.clear();
        }
        if (n <= 0) return;
        cr.plus_one_counters += n;
        cr.power     += static_cast<uint32_t>(n);
        cr.toughness += static_cast<uint32_t>(n);
        game_log("Put %d +1/+1 counter(s) on creature (now %u/%u).\n", n, cr.power, cr.toughness);
    }
}

// Parses a token script string of the form "<color>_<power>_<toughness>_<name>[_<kw1>[_<kw2>...]]"
// e.g. "w_1_1_monk_prowess"
void Ability::resolve_token(std::shared_ptr<Orderer> orderer) {
    Token tok = parse_token_script(token_script);
    if (tok.name.empty()) {
        game_log("resolve_token: failed to parse token script '%s'\n", token_script.c_str());
        return;
    }

    Zone::Ownership ctrl = global_coordinator.entity_has_component<Permanent>(source)
        ? global_coordinator.GetComponent<Permanent>(source).controller
        : global_coordinator.GetComponent<Zone>(source).owner;

    Entity tok_entity = global_coordinator.CreateEntity();
    global_coordinator.AddComponent(tok_entity, Zone(Zone::HAND, ctrl, ctrl));
    global_coordinator.AddComponent(tok_entity, tok);
    orderer->add_to_zone(false, tok_entity, Zone::BATTLEFIELD);

    // Add Permanent + Creature + Damage immediately so subabilities (e.g. Attach) can see them
    // before the next apply_permanent_components pass.
    Permanent perm;
    perm.name = tok.name;
    perm.types = tok.types;
    perm.is_token = true;
    perm.controller = ctrl;
    perm.has_summoning_sickness = true;
    perm.is_tapped = false;
    perm.timestamp_entered_battlefield = cur_game.timestamp++;
    global_coordinator.AddComponent(tok_entity, perm);

    Creature creature;
    creature.power = tok.power;
    creature.toughness = tok.toughness;
    creature.keywords = tok.keywords;
    global_coordinator.AddComponent(tok_entity, creature);

    Damage damage;
    damage.damage_counters = 0;
    global_coordinator.AddComponent(tok_entity, damage);

    cur_game.remembered_entity = tok_entity;
    game_log("Token created: %u/%u %s\n", tok.power, tok.toughness, tok.name.c_str());
}

void Ability::resolve_delayed_trigger() {
    Zone::Ownership owner = global_coordinator.entity_has_component<Permanent>(source)
        ? global_coordinator.GetComponent<Permanent>(source).controller
        : global_coordinator.GetComponent<Zone>(source).owner;
    Entity owner_entity = get_player_entity(owner);

    Ability draw_ab;
    draw_ab.ability_type = Ability::TRIGGERED;
    draw_ab.category     = "Draw";
    draw_ab.amount       = 1;
    draw_ab.source       = source;

    DelayedTrigger dt;
    dt.ability       = draw_ab;
    dt.fire_on       = Events::UPKEEP_BEGAN;
    dt.owner_entity  = owner_entity;
    dt.fire_on_turn  = delayed_trigger_next_turn ? cur_game.turn + 1 : cur_game.turn;
    cur_game.delayed_triggers.push_back(dt);
    game_log("Delayed trigger registered: draw 1 at next upkeep.\n");
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
    std::string name = global_coordinator.entity_has_component<Permanent>(target)
        ? global_coordinator.GetComponent<Permanent>(target).name
        : "<unknown>";
    orderer->add_to_zone(false, target, Zone::GRAVEYARD);
    game_log("%s is destroyed\n", name.c_str());
}
