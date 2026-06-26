#include "ability.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/color_identity.h"
#include "../components/creature.h"
#include "../components/token.h"
#include "../components/types.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../error.h"
#include "../action_processor.h"
#include "../game_queries.h"
#include "../input_logger.h"
#include "../mana_system.h"
#include "../parse.h"
#include "../svar_eval.h"
#include "../effects/effects.h"
#include "../systems/orderer.h"
#include "creature.h"
#include "damage.h"
#include "permanent.h"
#include "player.h"
#include "spell.h"

extern Coordinator global_coordinator;
extern Game cur_game;

// evaluate_dynamic_amount, search_multi_zone, and run_unless_loop are declared in
// effects.h (de-static'd so effect handlers in src/effects/ can share them);
// their definitions remain below.

// edge case of two identical abilities being applied from two sources not handled
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
    if (other.color != this->color) return false;
    if (other.mana_choices != this->mana_choices) return false;
    if (other.restrict_to_chosen_type_creature != this->restrict_to_chosen_type_creature) return false;
    if (other.adds_no_counter != this->adds_no_counter) return false;
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
        if (t.name == type_name) {
            type_matches = true;
            break;
        }
    }
    if (!type_matches) return false;

    // Check color qualifier
    if (!color_qualifier.empty()) {
        // IsRemembered: entity must be in cur_game.remembered_entities
        if (color_qualifier == "IsRemembered") {
            bool found = false;
            for (auto re : cur_game.remembered_entities) {
                if (re == entity) { found = true; break; }
            }
            if (!found) return false;
        } else if (color_qualifier == "Basic" || color_qualifier == "nonBasic") {
            // Basic/nonBasic supertype qualifier (e.g. "Land.Basic" for fetch ramp).
            bool is_basic = has_basic_supertype(cd.types);
            if (color_qualifier == "Basic" && !is_basic) return false;
            if (color_qualifier == "nonBasic" && is_basic) return false;
        } else {
            Colors required_color = NO_COLOR;
            if (color_qualifier == "Green")
                required_color = GREEN;
            else if (color_qualifier == "White")
                required_color = WHITE;
            else if (color_qualifier == "Blue")
                required_color = BLUE;
            else if (color_qualifier == "Black")
                required_color = BLACK;
            else if (color_qualifier == "Red")
                required_color = RED;

            // Check explicit_colors first, then mana cost colors
            bool has_color = false;
            if (!cd.explicit_colors.empty()) {
                has_color = cd.explicit_colors.count(required_color) > 0;
            } else {
                has_color = cd.mana_cost.count(required_color) > 0;
            }
            if (!has_color) return false;
        }
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
Entity search_zone(std::shared_ptr<Orderer> orderer, Zone::Ownership owner, Zone::ZoneValue zone,
    const std::string &change_type, bool mandatory, Zone::ZoneValue destination, bool reveal) {
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
    } else if (zone == Zone::GRAVEYARD) {
        for (auto e : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location == Zone::GRAVEYARD && z.owner == owner) zone_contents.push_back(e);
        }
    }

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
                    if (matches_filter_spec(entity, st)) {
                        matches = true;
                        break;
                    }
                }
            } else {
                auto &cd = global_coordinator.GetComponent<CardData>(entity);
                for (auto &t : cd.types) {
                    for (auto &st : subtypes) {
                        if (t.name == st) {
                            matches = true;
                            break;
                        }
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
    // other library searches use SEARCH_LIBRARY, non-library zone picks use CHOOSE_CARD
    ActionCategory cat = (destination == Zone::LIBRARY && (zone == Zone::LIBRARY || zone == Zone::HAND))
                             ? ActionCategory::TOP_LIBRARY
                         : (zone == Zone::LIBRARY) ? ActionCategory::SEARCH_LIBRARY
                                                   : ActionCategory::CHOOSE_CARD;

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
        la.card_is_public = reveal;
        search_actions.push_back(la);
    }

    int choice = InputLogger::instance().get_input(search_actions);
    // Map choice back: if fail-to-find is shown, index 0 = fail-to-find, 1..N = choices
    // If fail-to-find suppressed, index 0..N-1 = choices directly
    if (show_fail_to_find) {
        if (choice >= 1 && choice <= static_cast<int>(choices.size())) return choices[static_cast<size_t>(choice - 1)];
        return 0;
    } else {
        if (choice >= 0 && choice < static_cast<int>(choices.size())) return choices[static_cast<size_t>(choice)];
        return 0;
    }
}

// Searches multiple zones combined for cards matching change_type.
// Used by Doomsday (Origin$ Graveyard,Library).
Entity search_multi_zone(std::shared_ptr<Orderer> orderer, Zone::Ownership owner,
    const std::vector<Zone::ZoneValue> &zones, const std::string &change_type, bool mandatory,
    Zone::ZoneValue destination, bool reveal) {
    // Collect contents from all zones
    std::vector<Entity> zone_contents;
    for (auto zone : zones) {
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

    // Exclude already-remembered entities (e.g. Doomsday picking 5 cards one at a time)
    if (!cur_game.remembered_entities.empty()) {
        std::vector<Entity> filtered;
        for (auto e : zone_contents) {
            bool already = false;
            for (auto re : cur_game.remembered_entities) {
                if (re == e) {
                    already = true;
                    break;
                }
            }
            if (!already) filtered.push_back(e);
        }
        zone_contents = filtered;
    }

    // Filter by change_type — "Card" matches everything
    std::vector<Entity> choices;
    if (change_type.empty() || change_type == "Card") {
        choices = zone_contents;
    } else {
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
                    if (matches_filter_spec(entity, st)) {
                        matches = true;
                        break;
                    }
                }
            } else {
                auto &cd = global_coordinator.GetComponent<CardData>(entity);
                for (auto &t : cd.types) {
                    for (auto &st : subtypes) {
                        if (t.name == st) {
                            matches = true;
                            break;
                        }
                    }
                    if (matches) break;
                }
            }
            if (matches) choices.push_back(entity);
        }
    }

    bool show_fail_to_find = !mandatory || choices.empty();
    if (mandatory && choices.empty()) return 0;

    game_log("Searching %s's library and graveyard:\n", player_name(owner).c_str());

    ActionCategory cat = (destination == Zone::LIBRARY) ? ActionCategory::TOP_LIBRARY : ActionCategory::SEARCH_LIBRARY;

    std::vector<LegalAction> search_actions;
    if (show_fail_to_find) {
        LegalAction ftf(PASS_PRIORITY, Entity(0), std::string("Fail to find"));
        ftf.category = cat;
        search_actions.push_back(ftf);
    }
    for (auto entity : choices) {
        auto &cd = global_coordinator.GetComponent<CardData>(entity);
        auto &z = global_coordinator.GetComponent<Zone>(entity);
        const char *zone_label = (z.location == Zone::GRAVEYARD) ? " (graveyard)" : " (library)";
        LegalAction la(PASS_PRIORITY, entity, cd.name + zone_label);
        la.category = cat;
        la.card_is_public = reveal;
        search_actions.push_back(la);
    }

    int choice = InputLogger::instance().get_input(search_actions);
    if (show_fail_to_find) {
        if (choice >= 1 && choice <= static_cast<int>(choices.size())) return choices[static_cast<size_t>(choice - 1)];
        return 0;
    } else {
        if (choice >= 0 && choice < static_cast<int>(choices.size())) return choices[static_cast<size_t>(choice)];
        return 0;
    }
}




// Returns true if the spell should be countered (controller declined or couldn't pay).
bool run_unless_loop(
    size_t cost, Zone::Ownership controller, std::shared_ptr<Orderer> orderer, Entity paid_for) {
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
            pay.category = ActionCategory::PAY_UNLESS;
            unless_actions.push_back(pay);
        }
        size_t decline_idx = unless_actions.size();
        {
            LegalAction decline(PASS_PRIORITY, std::string("Don't pay (spell is countered)"));
            decline.category = ActionCategory::PAY_UNLESS;
            unless_actions.push_back(decline);
        }

        int choice = InputLogger::instance().get_input(unless_actions);

        if (choice == static_cast<int>(decline_idx)) {
            cur_game.player_a_has_priority = prev_priority;
            return true;
        }

        if (can_pay && choice == static_cast<int>(pay_idx)) {
            spend_mana(controller, cond_cost, paid_for);
            game_log("%s pays {%zu} — spell is not countered\n", player_name(controller).c_str(), cost);
            cur_game.player_a_has_priority = prev_priority;
            return false;
        }

        if (choice >= 0 && choice < static_cast<int>(pay_idx)) {
            auto &chosen = unless_actions[static_cast<size_t>(choice)];
            Entity land = chosen.source_entity;
            auto &perm = global_coordinator.GetComponent<Permanent>(land);
            perm.is_tapped = true;
            add_mana(controller, chosen.ability.color, chosen.ability.amount);
            game_log("%s tapped %s for {%s}\n", player_name(controller).c_str(), perm.name.c_str(),
                mana_symbol(chosen.ability.color).c_str());
        }
    }
}

void Ability::fizzle(std::shared_ptr<Orderer> orderer) {
    // stack manager present behavior moves everything to graveyard or destroys it
    // so for now this is a stub
    game_log("%s fizzles\n", this->category.c_str());
    return;
}

// True if the card counts as the given color (explicit Colors: override first,
// else the colors present in its mana cost).
static bool card_is_color(const CardData &cd, Colors c) {
    if (!cd.explicit_colors.empty()) return cd.explicit_colors.count(c) > 0;
    return cd.mana_cost.count(c) > 0;
}

// Enforces "non<Color>" target restrictions (e.g. ValidTgts$ Creature.nonBlack on
// Snuff Out). Returns false when the candidate is one of the excluded colors.
static bool passes_noncolor_restriction(const std::string &vt, const CardData &cd) {
    static const struct { const char *tok; Colors col; } table[] = {
        {"nonWhite", WHITE}, {"nonBlue", BLUE}, {"nonBlack", BLACK}, {"nonRed", RED}, {"nonGreen", GREEN}};
    for (auto &e : table)
        if (vt.find(e.tok) != std::string::npos && card_is_color(cd, e.col)) return false;
    return true;
}

// Single source of truth for target legality (see header). build_valid_targets
// enumerates candidates and filters them through this; is_target_valid re-runs the
// chosen target(s) through it at resolution. Keeping both on one predicate is what
// prevents the enumeration and re-verification rules from drifting apart.
bool Ability::is_legal_target(Entity cand, Zone::Ownership caster) const {
    if (cand == 0) return false;

    // NOTE: Pyroblast/Hydroblast (ConditionPresent$ <type>.<Color>) intentionally do
    // NOT restrict target legality by color — they may target any spell/permanent and
    // their counter/destroy effect is conditional on the target's color (enforced in
    // effects::counter / effects::destroy via target_color_condition_met). So no color
    // filter is applied here.

    const std::string &vt = valid_tgts;

    // Spell on the stack (counterspells etc.)
    if (target_type == "Spell") {
        if (!global_coordinator.entity_has_component<Zone>(cand)) return false;
        if (global_coordinator.GetComponent<Zone>(cand).location != Zone::STACK) return false;
        if (!global_coordinator.entity_has_component<Spell>(cand)) return false;
        bool non_creature_only = vt.find("nonCreature") != std::string::npos;
        bool instant_sorcery_only =
            (vt.find("Instant") != std::string::npos || vt.find("Sorcery") != std::string::npos) &&
            vt.find("Creature") == std::string::npos;
        if ((non_creature_only || instant_sorcery_only) &&
            global_coordinator.entity_has_component<CardData>(cand)) {
            auto &cd = global_coordinator.GetComponent<CardData>(cand);
            bool is_creature = false, is_instant = false, is_sorcery = false;
            for (auto &t : cd.types) {
                if (t.name == "Creature") is_creature = true;
                if (t.name == "Instant") is_instant = true;
                if (t.name == "Sorcery") is_sorcery = true;
            }
            if (non_creature_only && is_creature) return false;
            if (instant_sorcery_only && !is_instant && !is_sorcery) return false;
        }
        return true;
    }

    // Stifle: standalone activated/triggered ability on the stack
    if (target_type.find("Activated") != std::string::npos ||
        target_type.find("Triggered") != std::string::npos) {
        if (!global_coordinator.entity_has_component<Zone>(cand)) return false;
        if (global_coordinator.GetComponent<Zone>(cand).location != Zone::STACK) return false;
        if (global_coordinator.entity_has_component<Spell>(cand)) return false;  // spells aren't abilities
        if (!global_coordinator.entity_has_component<Ability>(cand)) return false;
        auto &ab = global_coordinator.GetComponent<Ability>(cand);
        bool want_activated = target_type.find("Activated") != std::string::npos;
        bool want_triggered = target_type.find("Triggered") != std::string::npos;
        if (want_activated && ab.ability_type == Ability::ACTIVATED) return true;
        if (want_triggered && ab.ability_type == Ability::TRIGGERED) return true;
        return false;
    }

    // Card in a non-battlefield zone (e.g. Faerie Macabre targeting graveyard cards)
    if (vt == "Card" && category == "ChangeZone" && origin == Zone::GRAVEYARD) {
        return global_coordinator.entity_has_component<Zone>(cand) &&
               global_coordinator.GetComponent<Zone>(cand).location == Zone::GRAVEYARD;
    }

    bool any = (vt == "Any");
    bool opp_only = (vt == "Opponent");
    bool inc_players = any || opp_only || vt.find("Player") != std::string::npos;
    bool inc_creatures = any || vt.find("Creature") != std::string::npos;
    bool inc_lands = vt.find("Land") != std::string::npos;
    bool nonbasic_only = vt.find("nonBasic") != std::string::npos;
    bool legendary_only = vt.find("Legendary") != std::string::npos;
    // "Any" is "any target" (creature/player/planeswalker), NOT non-creature
    // artifacts/enchantments — those require the type named explicitly.
    bool inc_artifacts = vt.find("Artifact") != std::string::npos;
    bool inc_enchantments = vt.find("Enchantment") != std::string::npos;
    bool inc_permanents = vt.find("Permanent") != std::string::npos;
    // "Any" includes planeswalkers (a damage spell like Lightning Bolt can hit a walker, 306.7).
    bool inc_planeswalkers = any || vt.find("Planeswalker") != std::string::npos;
    int cmc_le = -1;
    {
        size_t cmc_pos = vt.find("cmcLE");
        if (cmc_pos != std::string::npos) cmc_le = std::stoi(vt.substr(cmc_pos + 5));
    }

    // Card in a graveyard targeted by a ChangeZone with a type filter (e.g. Life from
    // the Loam: ValidTgts$ Land.YouCtrl, Origin$ Graveyard). Filter by zone, owner
    // (YouCtrl/OppCtrl), and card type.
    if (target_in_graveyard ||
        (category == "ChangeZone" && origin == Zone::GRAVEYARD && destination != Zone::BATTLEFIELD)) {
        if (!global_coordinator.entity_has_component<Zone>(cand)) return false;
        auto &cz = global_coordinator.GetComponent<Zone>(cand);
        if (cz.location != Zone::GRAVEYARD) return false;
        bool you_ctrl = vt.find("YouCtrl") != std::string::npos;
        bool opp_ctrl = vt.find("OppCtrl") != std::string::npos;
        if (you_ctrl && cz.owner != caster) return false;
        if (opp_ctrl && cz.owner == caster) return false;
        if (!global_coordinator.entity_has_component<CardData>(cand)) return false;
        auto &cd = global_coordinator.GetComponent<CardData>(cand);
        bool type_ok = !(inc_creatures || inc_lands || inc_artifacts || inc_enchantments);
        for (auto &t : cd.types) {
            if (t.kind != TYPE) continue;
            if (inc_creatures    && t.name == "Creature")    type_ok = true;
            if (inc_lands        && t.name == "Land")        type_ok = true;
            if (inc_artifacts    && t.name == "Artifact")    type_ok = true;
            if (inc_enchantments && t.name == "Enchantment") type_ok = true;
        }
        if (nonbasic_only && has_basic_supertype(cd.types)) return false;
        return type_ok;
    }

    // Player target
    if (global_coordinator.entity_has_component<Player>(cand)) {
        if (!inc_players) return false;
        if (opp_only) {
            Zone::Ownership opp = (caster == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;
            return cand == get_player_entity(opp);
        }
        return true;
    }

    // Battlefield permanent target (phased-out permanents can't be targeted, 702.26e)
    if (!is_battlefield_permanent(cand)) return false;
    auto &tperm = global_coordinator.GetComponent<Permanent>(cand);

    if (cmc_le >= 0 && global_coordinator.entity_has_component<CardData>(cand)) {
        int cmc = static_cast<int>(global_coordinator.GetComponent<CardData>(cand).mana_cost.size());
        if (cmc > cmc_le) return false;
    }

    if (inc_creatures && global_coordinator.entity_has_component<Creature>(cand)) {
        if (legendary_only) {
            bool is_legendary = false;
            for (auto &t : tperm.types)
                if (t.kind == SUPERTYPE && t.name == "Legendary") { is_legendary = true; break; }
            if (!is_legendary) return false;
        }
        if (global_coordinator.entity_has_component<CardData>(cand) &&
            !passes_noncolor_restriction(vt, global_coordinator.GetComponent<CardData>(cand)))
            return false;
        if (has_protection_from(global_coordinator.GetComponent<Creature>(cand), source)) return false;
        return true;
    }
    if (inc_lands) {
        bool is_land = false;
        for (auto &t : tperm.types)
            if (t.kind == TYPE && t.name == "Land") { is_land = true; break; }
        if (is_land && (!nonbasic_only || !has_basic_supertype(tperm.types))) return true;
    }
    if (inc_planeswalkers) {
        for (auto &t : tperm.types)
            if (t.kind == TYPE && t.name == "Planeswalker") return true;
    }
    if (inc_permanents) return true;
    if (inc_artifacts || inc_enchantments) {
        for (auto &t : tperm.types) {
            if (t.kind == TYPE && ((inc_artifacts && t.name == "Artifact") ||
                                   (inc_enchantments && t.name == "Enchantment")))
                return true;
        }
    }

    return false;
}

bool Ability::is_target_valid() const {
    // Optional targeting: no target chosen is valid
    if (target == 0 && targets.empty() && target_min == 0) return true;

    // Multi-target abilities (target_max > 1) populate `targets`; verify every one.
    if (!targets.empty()) {
        for (Entity t : targets)
            if (!is_legal_target(t, controller)) return false;
        return true;
    }

    return is_legal_target(target, controller);
}

// Evaluates a condition SVar expression against cur_game state.
static int evaluate_condition_svar(const std::string &expr, Entity src, Zone::Ownership ctrl = Zone::PLAYER_A,
    std::shared_ptr<Orderer> orderer = nullptr) {
    if (expr == "Count$ResolvedThisTurn") {
        auto it = cur_game.ability_resolution_counts.find(src);
        return (it != cur_game.ability_resolution_counts.end()) ? it->second : 0;
    }
    // Delegate to evaluate_dynamic_amount for Count$ expressions
    if (orderer && expr.find("Count$") != std::string::npos) {
        return static_cast<int>(evaluate_dynamic_amount(expr, ctrl, orderer, 0));
    }
    return 0;
}

// Returns true if val passes the compare spec (e.g. "EQ2", "NE2", "GE1", "LE3").
// When svar_rhs is non-empty, it is evaluated as the RHS instead of parsing an int
// from spec. Unlike svar_eval::compare_svar this is permissive — a missing/unknown
// operator passes — so the operator application (but not the defaulting) is shared
// via apply_svar_op.
static bool compare_svar(int val, const std::string &spec, const std::string &svar_rhs = "", Entity src = 0,
    Zone::Ownership ctrl = Zone::PLAYER_A, std::shared_ptr<Orderer> orderer = nullptr) {
    if (spec.size() < 2) return true;
    std::string op = spec.substr(0, 2);
    int rhs;
    if (!svar_rhs.empty() && orderer) {
        rhs = evaluate_condition_svar(svar_rhs, src, ctrl, orderer);
    } else {
        if (spec.size() < 3) return true;
        rhs = std::stoi(spec.substr(2));
    }
    if (op == "EQ" || op == "NE" || op == "GE" ||
        op == "LE" || op == "GT" || op == "LT") return apply_svar_op(val, op, rhs);
    return true;  // unknown operator passes (permissive)
}

// Evaluates a dynamic_amount_expr at runtime for the given controller.
// Supports: Count$InYourLibrary, Count$YourLifeTotal, Count$YourLifeTotal/HalfUp,
//           Count$Valid Creature.YouCtrl, Targeted$CardPower.
size_t evaluate_dynamic_amount(
    const std::string &expr, Zone::Ownership ctrl, std::shared_ptr<Orderer> orderer, Entity target) {
    if (expr.find("Count$Devotion.") != std::string::npos) {
        // Count mana symbols of a given color in mana costs of permanents you control
        Colors devotion_color = NO_COLOR;
        if (expr.find("Devotion.Blue") != std::string::npos)
            devotion_color = BLUE;
        else if (expr.find("Devotion.Black") != std::string::npos)
            devotion_color = BLACK;
        else if (expr.find("Devotion.Red") != std::string::npos)
            devotion_color = RED;
        else if (expr.find("Devotion.Green") != std::string::npos)
            devotion_color = GREEN;
        else if (expr.find("Devotion.White") != std::string::npos)
            devotion_color = WHITE;
        size_t count = 0;
        for (auto e : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location != Zone::BATTLEFIELD) continue;
            auto &perm = global_coordinator.GetComponent<Permanent>(e);
            if (perm.controller != ctrl) continue;
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            count += cd.mana_cost.count(devotion_color);
        }
        return count;
    }
    if (expr.find("Count$InYourLibrary") != std::string::npos ||
        expr.find("Count$ValidLibrary Card.YouOwn") != std::string::npos) {
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
    // Count$Revolt.high.low — returns high if revolt active for controller, low otherwise
    if (expr.find("Count$Revolt.") != std::string::npos) {
        size_t dot1 = expr.find("Revolt.") + 7;
        size_t dot2 = expr.find('.', dot1);
        int high_val = std::stoi(expr.substr(dot1, dot2 - dot1));
        int low_val = std::stoi(expr.substr(dot2 + 1));
        bool revolt = (ctrl == Zone::PLAYER_A) ? cur_game.revolt_player_a : cur_game.revolt_player_b;
        return static_cast<size_t>(revolt ? high_val : low_val);
    }
    if (expr.find("Targeted$CardPower") != std::string::npos) {
        if (global_coordinator.entity_has_component<CardData>(target))
            return static_cast<size_t>(global_coordinator.GetComponent<CardData>(target).power);
        if (global_coordinator.entity_has_component<Creature>(target))
            return static_cast<size_t>(global_coordinator.GetComponent<Creature>(target).power);
    }
    // Remembered$CardManaCost[/Plus.N] — mana value of the first remembered card (Birthing
    // Ritual: X = 1 plus the sacrificed creature's mana value).
    if (expr.find("Remembered$CardManaCost") != std::string::npos) {
        int base = 0;
        if (!cur_game.remembered_entities.empty()) {
            Entity r = cur_game.remembered_entities[0];
            if (global_coordinator.entity_has_component<CardData>(r))
                base = static_cast<int>(global_coordinator.GetComponent<CardData>(r).mana_cost.size());
        }
        size_t plus = expr.find("/Plus.");
        if (plus != std::string::npos) base += std::stoi(expr.substr(plus + 6));
        return static_cast<size_t>(base < 0 ? 0 : base);
    }
    // Fall back to the shared static-ability SVar evaluator for graveyard-count
    // expressions (Count$TypeInYourYard / Count$ValidGraveyard / CardTypes). It
    // returns 0 for anything it doesn't recognise, so this preserves the prior
    // default while making one set of Count$ handlers serve both paths.
    int sa_val = evaluate_sa_svar(expr, ctrl);
    return sa_val > 0 ? static_cast<size_t>(sa_val) : 0;
}

void Ability::resolve(std::shared_ptr<Orderer> orderer) {
    // 603.4 intervening-if: re-check the trigger's "if" condition on resolution. If it is no
    // longer true the ability is removed from the stack and does nothing — not even its
    // subabilities fire (unlike a ConditionCheckSVar gate).
    if (intervening_if && !evaluate_present_condition(*this, controller, orderer)) {
        game_log("Triggered ability's intervening-if condition is no longer true; it does nothing.\n");
        return;
    }
    // Pre-resolve target validity check — skipped for categories that select their own target internally
    if (valid_tgts != "N_A" && category != "Pump") {
        if (!is_target_valid()) {
            fizzle(orderer);
            return;  // subabilities do not fire; TODO revisit this in light of cards e.g. k-command
        }
    }
    // RememberTargets/RememberObjects: stash the target(s) so chained
    // ChangeType$ Remembered.sameName subabilities can match by name (Surgical Extraction).
    if (remember_targeted) {
        cur_game.remembered_entities.clear();
        if (!targets.empty())
            for (auto t : targets) cur_game.remembered_entities.push_back(t);
        else if (target != 0)
            cur_game.remembered_entities.push_back(target);
    }
    game_log("Resolving ability (category: %s, amount: %zu)\n", category.c_str(), amount);

    // Conditional execution: if condition fails, skip this ability's body but still chain subabilities
    bool condition_passed = true;
    if (!condition_check_svar.empty()) {
        int val = evaluate_condition_svar(condition_check_svar, source, controller, orderer);
        condition_passed =
            compare_svar(val, condition_svar_compare, condition_compare_svar_expr, source, controller, orderer);
    }
    // ConditionDefined$ Remembered gate (Birthing Ritual): the dig only happens if a creature
    // was sacrificed (remembered count satisfies condition_present/condition_compare). Like the
    // SVar gate, failure skips this body but still chains subabilities.
    if (condition_passed && condition_on_remembered)
        condition_passed = evaluate_present_condition(*this, controller, orderer);
    if (!condition_passed) {
        for (auto sub_ab : this->subabilities) {
            sub_ab.source = this->source;
            sub_ab.target = this->target;
            sub_ab.controller = this->controller;
            sub_ab.resolve(orderer);
        }
        return;
    }

    // Table-driven dispatch: every effect category resolves through its handler
    // in src/effects/. handler_for() returns nullptr only for categories with no
    // resolve-time handler (e.g. "Equip", handled at activation) — those simply
    // chain subabilities, matching the legacy chain's fall-through behavior.
    effects::EffectHandler handler = effects::handler_for(effect_kind_from_string(category));
    bool run_subs = handler ? handler(*this, orderer) : true;

    // Chain subabilities unless the handler opted out (Charm/WinsGame and the
    // non-peek PeekAndReveal path return false to handle their own resolution).
    if (run_subs) {
        for (auto sub_ab : this->subabilities) {
            sub_ab.source = this->source;
            sub_ab.target = this->target;  // propagate target so GainLife etc. can reference it
            sub_ab.controller = this->controller;
            sub_ab.resolve(orderer);
        }
    }
}







