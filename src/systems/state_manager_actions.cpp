#include "state_manager.h"
#include "state_manager_internal.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "../action_processor.h"
#include "../card_vocab.h"
#include "../classes/game.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/color_identity.h"
#include "../components/creature.h"
#include "../components/static_ability.h"
#include "../components/damage.h"
#include "../components/effect.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/token.h"
#include "../components/types.h"
#include "../type_constants.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../cli_output.h"
#include "../game_queries.h"
#include "../input_logger.h"
#include "../mana_system.h"
#include "../svar_eval.h"
#include "../systems/stack_manager.h"
#include "orderer.h"

static bool check_condition_present(const Ability &ab, Zone::Ownership caster, std::shared_ptr<Orderer> orderer);
static bool can_afford_alt(const AltCost& alt_cost, Zone::Ownership priority_player,
                           Entity card_entity, std::shared_ptr<Orderer> orderer) {
    if (!alt_cost.has_alt_cost) return false;

    // Check SVar condition (e.g. Once Upon a Time: free only if first spell this game)
    if (!alt_cost.condition_svar.empty()) {
        int svar_value = 0;
        if (alt_cost.condition_svar.find("Count$YouCastThisGame") != std::string::npos) {
            Entity pp_entity = get_player_entity(priority_player);
            svar_value = static_cast<int>(global_coordinator.GetComponent<Player>(pp_entity).spells_cast_this_game);
        }
        if (!compare_svar(svar_value, alt_cost.condition_compare)) return false;
    }

    // IsPresent$ <type>[.YouCtrl] — the alt cost is only available while the
    // caster controls a matching permanent (e.g. Snuff Out: control a Swamp).
    if (!alt_cost.condition_is_present.empty()) {
        std::string filter = alt_cost.condition_is_present;
        size_t dot = filter.find('.');
        std::string type_name = (dot == std::string::npos) ? filter : filter.substr(0, dot);
        bool found = false;
        for (auto e : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            if (global_coordinator.GetComponent<Zone>(e).location != Zone::BATTLEFIELD) continue;
            auto &perm = global_coordinator.GetComponent<Permanent>(e);
            if (perm.controller != priority_player) continue;
            for (auto &t : perm.types) {
                if (t.name == type_name) { found = true; break; }
            }
            if (found) break;
        }
        if (!found) return false;
    }

    // Free alt cost: no further affordability checks needed
    if (alt_cost.is_free) return true;

    if (alt_cost.return_to_hand_count > 0) {
        int matching = 0;
        const std::string& sub = alt_cost.return_to_hand_type;
        for (auto e : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
            auto& perm = global_coordinator.GetComponent<Permanent>(e);
            if (perm.controller != priority_player) continue;
            for (auto& t : perm.types) {
                if (t.kind == SUBTYPE && t.name == sub) { matching++; break; }
            }
        }
        return matching >= alt_cost.return_to_hand_count;
    }

    if (alt_cost.life_cost > 0) {
        Entity pp_entity = get_player_entity(priority_player);
        if (global_coordinator.GetComponent<Player>(pp_entity).life_total < alt_cost.life_cost)
            return false;
    }

    // Condition: not your turn (Force of Negation, Force of Vigor)
    if (alt_cost.condition_not_your_turn) {
        bool is_my_turn = (priority_player == Zone::PLAYER_A) ? cur_game.player_a_turn : !cur_game.player_a_turn;
        if (is_my_turn) return false;
    }

    if (alt_cost.exile_from_hand_count > 0) {
        Colors required_color = alt_cost.exile_from_hand_color;
        bool has_match = false;
        for (auto e : orderer->get_hand(priority_player)) {
            if (e == card_entity) continue;
            if (required_color != NO_COLOR && global_coordinator.entity_has_component<ColorIdentity>(e) &&
                global_coordinator.GetComponent<ColorIdentity>(e).colors.count(required_color)) {
                has_match = true; break;
            }
        }
        if (!has_match) return false;
    }

    // Mana portion of the alt cost (e.g. Evoke:R)
    if (!alt_cost.mana_cost.empty()) {
        if (!can_pay_mana(priority_player, alt_cost.mana_cost, card_entity, orderer)) return false;
    }

    return true;
}

// Check ConditionPresent$ / ConditionCompare$ castability condition.
// Counts battlefield permanents matching the filter and compares against the threshold.
// Filter format: "Type.YouCtrl" or "Type.OppCtrl" (e.g. "Land.YouCtrl").
static bool check_condition_present(const Ability &ab, Zone::Ownership caster, std::shared_ptr<Orderer> orderer) {
    if (ab.condition_present.empty()) return true;

    // Parse filter: "Land.YouCtrl" → type_filter="Land", controller check
    std::string filter = ab.condition_present;
    std::string type_filter;
    bool you_ctrl = false;
    bool opp_ctrl = false;
    size_t dot = filter.find('.');
    if (dot != std::string::npos) {
        type_filter = filter.substr(0, dot);
        std::string qualifier = filter.substr(dot + 1);
        if (qualifier == "YouCtrl") you_ctrl = true;
        else if (qualifier == "OppCtrl") opp_ctrl = true;
    } else {
        type_filter = filter;
    }

    Zone::Ownership required_ctrl = you_ctrl ? caster :
        opp_ctrl ? (caster == Zone::PLAYER_A ? Zone::PLAYER_B : Zone::PLAYER_A) :
        Zone::UNKNOWN;

    size_t count = 0;
    for (auto e : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location != Zone::BATTLEFIELD) continue;
        if (required_ctrl != Zone::UNKNOWN) {
            auto &perm = global_coordinator.GetComponent<Permanent>(e);
            if (perm.controller != required_ctrl) continue;
        }
        if (!type_filter.empty() && global_coordinator.entity_has_component<CardData>(e)) {
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            bool match = false;
            for (const auto &t : cd.types) {
                if (t.name == type_filter) { match = true; break; }
            }
            if (!match) continue;
        }
        count++;
    }

    return compare_svar(static_cast<int>(count), ab.condition_compare);
}


std::vector<LegalAction> StateManager::determine_legal_actions(
    const Game &game, std::shared_ptr<Orderer> orderer, std::shared_ptr<StackManager> stack_manager) {
    std::vector<LegalAction> actions;          // return value
    std::vector<LegalAction> pending_actions;  // non mana-ability actions possible if costs could be paid; used to
                                               // check what mana abilities can be rationally activated

    // Determine whose turn/priority it is
    Zone::Ownership priority_player = game.player_a_has_priority ? Zone::PLAYER_A : Zone::PLAYER_B;
    Entity priority_player_entity = get_player_entity(priority_player);

    // PASS PRIORITY
    LegalAction la(PASS_PRIORITY, "Pass priority");
    la.category = ActionCategory::PASS_PRIORITY;
    actions.push_back(la);

    // LAND FROM HAND — requires empty stack (sorcery-speed)
    if ((game.cur_step == FIRST_MAIN || game.cur_step == SECOND_MAIN) &&
        game.player_a_turn == game.player_a_has_priority && stack_manager->is_empty() &&
        global_coordinator.entity_has_component<Player>(priority_player_entity)) {
        auto &player = global_coordinator.GetComponent<Player>(priority_player_entity);

        // Compute effective land play limit (base 1 + AdjustLandPlays statics)
        int land_play_limit = 1;
        bool may_play_from_graveyard = false;
        for (const auto &as : g_active_statics) {
            if (as.controller != priority_player) continue;
            if (as.sa->adjust_land_plays > 0) land_play_limit += as.sa->adjust_land_plays;
            if (as.sa->may_play_from_graveyard) may_play_from_graveyard = true;
        }

        if (player.lands_played_this_turn < land_play_limit) {
            // Check hand for lands
            auto hand = orderer->get_hand(priority_player);
            for (auto card_entity : hand) {
                auto &card_data = global_coordinator.GetComponent<CardData>(card_entity);
                if (is_land_card(card_data)) {
                    std::string desc = "Play " + card_data.name;
                    LegalAction la(SPECIAL_ACTION, card_entity, desc);
                    la.category = ActionCategory::PLAY_LAND;
                    actions.push_back(la);
                }
            }
            // Check graveyard for lands if MayPlay from graveyard is active
            if (may_play_from_graveyard) {
                Entity max_e = global_coordinator.GetMaxIssuedEntity();
                for (Entity gy_e = 0; gy_e < max_e; ++gy_e) {
                    if (!global_coordinator.entity_has_component<Zone>(gy_e)) continue;
                    auto &gz = global_coordinator.GetComponent<Zone>(gy_e);
                    if (gz.location != Zone::GRAVEYARD || gz.owner != priority_player) continue;
                    if (!global_coordinator.entity_has_component<CardData>(gy_e)) continue;
                    auto &gcd = global_coordinator.GetComponent<CardData>(gy_e);
                    if (is_land_card(gcd)) {
                        std::string desc = "Play " + gcd.name + " (from graveyard)";
                        LegalAction la(SPECIAL_ACTION, gy_e, desc);
                        la.category = ActionCategory::PLAY_LAND;
                        actions.push_back(la);
                    }
                }
            }
        }
    }
    // checking for spells to cast from hand
    // TODO spells cast from elsewhere
    bool stack_empty = stack_manager->is_empty();
    auto hand = orderer->get_hand(priority_player);
    for (auto card_entity : hand) {
        auto &card_data = global_coordinator.GetComponent<CardData>(card_entity);
        bool is_instant = false;
        bool is_land = false;
        for (auto &type : card_data.types) {
            if (type.kind == TYPE) {
                if (type.name == "Instant") {
                    is_instant = true;
                } else if (type.name == "Land") {
                    is_land = true;  // can't cast land
                    break;
                }
            }
        }
        if (is_land) continue;
        // Flash keyword grants instant-speed casting
        if (!is_instant) {
            for (const auto &kw : card_data.keywords) {
                if (kw == "Flash") { is_instant = true; break; }
            }
        }
        // Timing restrictions
        bool can_cast_now = false;
        if (is_instant) {
            can_cast_now = true;  // cast anytime you have priority... TODO handle edge cases
        } else {
            // Sorcery speed: main phase, your turn, stack empty
            can_cast_now = (game.cur_step == FIRST_MAIN || game.cur_step == SECOND_MAIN) &&
                           (game.player_a_turn == game.player_a_has_priority) && stack_empty;
        }
        // Check that at least one legal target exists for any targeting requirement
        // and that any ConditionPresent$ castability condition is met
        bool tgt_ok = true;
        bool condition_ok = true;
        for (const auto &ab : card_data.abilities) {
            if (ab.ability_type != Ability::SPELL) continue;
            tgt_ok = has_legal_targets(ab, orderer);
            // Target-conditional abilities (ConditionDefined$ Targeted, e.g. Fatal Push)
            // may target anything legal; the condition is checked on the target at
            // resolution, so it must not gate cast-time legality.
            if (!ab.condition_present.empty() && !ab.condition_on_target)
                condition_ok = check_condition_present(ab, priority_player, orderer);
            break;
        }
        // Machine mode only: action-masking optimization — don't offer a conditional-destroy
        // spell to the RL agent when no target on the board would currently pass the
        // condition (e.g. Fatal Push: only show if a creature with mana value <= the current
        // revolt-aware threshold exists). This is a masking heuristic, NOT a rules gate —
        // the spell can still legally target any creature in CLI/interactive play.
        if (InputLogger::instance().is_machine_mode() && tgt_ok && condition_ok) {
            for (const auto &ab : card_data.abilities) {
                if (ab.ability_type != Ability::SPELL) continue;
                if (ab.condition_present.find("cmcLEX") != std::string::npos &&
                    !ab.dynamic_amount_expr.empty()) {
                    // Evaluate Revolt threshold inline
                    int threshold = 2;
                    if (ab.dynamic_amount_expr.find("Count$Revolt.") != std::string::npos) {
                        size_t dot1 = ab.dynamic_amount_expr.find("Revolt.") + 7;
                        size_t dot2 = ab.dynamic_amount_expr.find('.', dot1);
                        int high_val = std::stoi(ab.dynamic_amount_expr.substr(dot1, dot2 - dot1));
                        int low_val = std::stoi(ab.dynamic_amount_expr.substr(dot2 + 1));
                        bool revolt = (priority_player == Zone::PLAYER_A)
                            ? cur_game.revolt_player_a : cur_game.revolt_player_b;
                        threshold = revolt ? high_val : low_val;
                    }
                    bool any_valid = false;
                    for (auto ce : mEntities) {
                        if (!global_coordinator.entity_has_component<Creature>(ce)) continue;
                        if (!global_coordinator.entity_has_component<Zone>(ce)) continue;
                        auto &cz = global_coordinator.GetComponent<Zone>(ce);
                        if (cz.location != Zone::BATTLEFIELD) continue;
                        if (!global_coordinator.entity_has_component<CardData>(ce)) continue;
                        int cmc = static_cast<int>(global_coordinator.GetComponent<CardData>(ce).mana_cost.size());
                        if (cmc <= threshold) { any_valid = true; break; }
                    }
                    if (!any_valid) tgt_ok = false;
                }
                break;
            }
        }

        auto pf_it = cur_game.payment_fail_counts.find(card_entity);
        bool payment_blocked = pf_it != cur_game.payment_fail_counts.end() && pf_it->second >= 2;
        if (can_cast_now && tgt_ok && condition_ok && !payment_blocked) {
            std::string desc = "Cast " + card_data.name;
            LegalAction la(CAST_SPELL, card_entity, desc);
            la.category = ActionCategory::CAST_SPELL;

            // Check CantBeCast statics from cached active_statics
            bool card_is_creature = is_creature_card(card_data);
            bool cast_blocked = false;
            for (const auto &as : g_active_statics) {
                if (as.sa->category != "CantBeCast") continue;
                // Skip if the spell doesn't match the filter (creatures are unaffected by nonCreature restriction)
                if (as.sa->cant_cast_filter.find("nonCreature") != std::string::npos && card_is_creature)
                    continue;
                Entity pp_entity = get_player_entity(priority_player);
                auto &pp = global_coordinator.GetComponent<Player>(pp_entity);
                if (as.sa->cant_cast_limit_per_turn > 0 &&
                    static_cast<int>(pp.noncreature_spells_cast_this_turn) >= as.sa->cant_cast_limit_per_turn) {
                    cast_blocked = true;
                }
            }
            if (cast_blocked) continue;

            ManaValue effective_cost = effective_base_cost(card_data);

            // X-cost spells: base cost (without X) is enough to be castable;
            // X value is chosen at cast time in action_processor
            bool can_regular = can_pay_mana(priority_player, effective_cost, card_entity,
                                            orderer, card_data.has_delve);

            bool can_alt = can_afford_alt(card_data.alt_cost, priority_player, card_entity, orderer);

            if (can_regular) actions.push_back(la);
            if (can_alt) {
                LegalAction alt_la = la;
                alt_la.use_alt_cost = true;
                alt_la.description = "Cast " + card_data.name + " (alternate cost)";
                actions.push_back(alt_la);
            }
            if (!can_regular && !can_alt) pending_actions.push_back(la);
        }
    }
    // checking graveyard for flashback spells
    for (auto gy_entity : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Zone>(gy_entity)) continue;
        auto &gz = global_coordinator.GetComponent<Zone>(gy_entity);
        if (gz.location != Zone::GRAVEYARD || gz.owner != priority_player) continue;
        if (!global_coordinator.entity_has_component<CardData>(gy_entity)) continue;
        auto &gcd = global_coordinator.GetComponent<CardData>(gy_entity);
        if (!gcd.has_flashback) continue;

        bool is_instant = false;
        for (auto &type : gcd.types) {
            if (type.kind == TYPE && type.name == "Instant") { is_instant = true; break; }
        }
        bool can_cast_now = false;
        if (is_instant) {
            can_cast_now = true;
        } else {
            can_cast_now = (game.cur_step == FIRST_MAIN || game.cur_step == SECOND_MAIN) &&
                           (game.player_a_turn == game.player_a_has_priority) && stack_empty;
        }
        if (!can_cast_now) continue;

        bool tgt_ok = true;
        for (const auto &ab : gcd.abilities) {
            if (ab.ability_type != Ability::SPELL) continue;
            tgt_ok = has_legal_targets(ab, orderer);
            break;
        }
        if (!tgt_ok) continue;

        // Check affordability: flashback mana cost + life cost
        bool can_afford_fb = can_pay_mana(priority_player, gcd.flashback_mana_cost, gy_entity, orderer);
        if (can_afford_fb && gcd.flashback_alt_cost.life_cost > 0) {
            Entity pp_entity = get_player_entity(priority_player);
            if (global_coordinator.GetComponent<Player>(pp_entity).life_total < gcd.flashback_alt_cost.life_cost)
                can_afford_fb = false;
        }
        if (!can_afford_fb) continue;

        LegalAction fb_la(CAST_SPELL, gy_entity, "Cast " + gcd.name + " (flashback)");
        fb_la.category = ActionCategory::CAST_SPELL;
        fb_la.use_flashback = true;
        actions.push_back(fb_la);
    }
    // checking permanents for activated abilities
    // mana abilities parsed last, after pending_actions complete
    // Simple tap-only mana sources collected via shared function
    std::vector<LegalAction> legal_mana_abilities = collect_mana_legal_actions(priority_player, orderer);
    for (auto entity : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location != Zone::BATTLEFIELD) continue;
        auto &permanent = global_coordinator.GetComponent<Permanent>(entity);
        if (permanent.controller != priority_player) continue;
        if (permanent.is_phased_out) continue;

        // Check if any CantBeActivated static suppresses this permanent's abilities.
        // (Mana abilities are collected separately above, so they remain usable — this
        // matches Disruptor Flute's ValidSA$ Activated.!ManaAbility.)
        bool cant_activate = false;
        for (const auto &as : g_active_statics) {
            if (as.sa->category != "CantBeActivated") continue;
            if (as.sa->match_named_card) {
                // NamedCard (Disruptor Flute): suppress sources whose name matches the chosen name
                if (!global_coordinator.entity_has_component<Permanent>(as.entity)) continue;
                auto &src = global_coordinator.GetComponent<Permanent>(as.entity);
                if (!src.chosen_name.empty() && src.chosen_name == permanent.name) cant_activate = true;
            } else if (as.sa->cant_activate_card_filter == "Artifact") {
                for (auto &t : permanent.types)
                    if (t.kind == TYPE && t.name == "Artifact") { cant_activate = true; break; }
            }
            if (cant_activate) break;
        }
        if (cant_activate) continue;

        // EQUIP: equipment's equip ability is sorcery-speed (main phase, your turn, empty stack).
        // The Equip keyword is parsed into is_equipment/equip_cost but produces no stored Ability,
        // so synthesise the action here when there is a creature to equip and the cost is payable.
        if (global_coordinator.entity_has_component<CardData>(entity)) {
            auto &cd = global_coordinator.GetComponent<CardData>(entity);
            bool main_phase = (game.cur_step == FIRST_MAIN || game.cur_step == SECOND_MAIN) &&
                              (game.player_a_turn == game.player_a_has_priority) &&
                              stack_manager->is_empty();
            if (cd.is_equipment && main_phase) {
                bool has_creature = false;
                for (auto e2 : orderer->mEntities) {
                    if (!global_coordinator.entity_has_component<Permanent>(e2)) continue;
                    if (!global_coordinator.entity_has_component<Creature>(e2)) continue;
                    if (global_coordinator.GetComponent<Zone>(e2).location != Zone::BATTLEFIELD) continue;
                    if (global_coordinator.GetComponent<Permanent>(e2).controller != priority_player) continue;
                    has_creature = true;
                    break;
                }
                if (has_creature && can_pay_mana(priority_player, cd.equip_cost, entity, orderer)) {
                    Ability equip_ab;
                    equip_ab.ability_type = Ability::ACTIVATED;
                    equip_ab.category = "Equip";
                    equip_ab.source = entity;
                    equip_ab.activation_mana_cost = cd.equip_cost;
                    std::string desc = "Equip " + entity_name(entity);
                    LegalAction equip_la(ACTIVATE_ABILITY, entity, equip_ab, desc);
                    equip_la.category = ActionCategory::ACTIVATE_ABILITY;
                    actions.push_back(equip_la);
                }
            }
        }

        for (auto ab : permanent.abilities) {
            if (ab.ability_type != Ability::ACTIVATED) continue;
            if (ab.activation_zone == Zone::HAND) continue;  // hand-only ability, not usable from battlefield
            // todo handle this elswewhere, tapping check
            if (ab.tap_cost && permanent.is_tapped) continue;
            if (ab.tap_cost && permanent.has_summoning_sickness &&
                global_coordinator.entity_has_component<Creature>(entity)) {
                auto &cr = global_coordinator.GetComponent<Creature>(entity);
                bool has_haste = false;
                for (const auto &kw : cr.keywords) {
                    if (kw == "Haste") { has_haste = true; break; }
                }
                if (!has_haste) continue;
            }
            // Activation limit check
            if (ab.activation_limit > 0 && ab.activations_this_turn >= ab.activation_limit) continue;
            // sac_cost_spec: require controller has a permanent matching type
            if (!ab.sac_cost_spec.empty() &&
                controlled_permanents_matching(priority_player, ab.sac_cost_spec, orderer->mEntities).empty())
                continue;
            // Return cost: require controller has a land of given subtype
            if (!ab.return_cost_type.empty() &&
                controlled_permanents_matching(priority_player, ab.return_cost_type, orderer->mEntities).empty())
                continue;
            if (ab.category == "AddMana" && !ab.instant_speed) {
                // Normal mana abilities collected via collect_mana_legal_actions above
                // InstantSpeed$ abilities (e.g. LED) are not mana abilities and go on the stack
                continue;
            } else {
                // Non-mana activated ability (e.g. ChangeZone for fetch lands, Destroy for Wasteland)
                if (!ab.activation_mana_cost.empty() && !can_pay_mana(priority_player, ab.activation_mana_cost, ab.source, orderer)) continue;
                if (ab.valid_tgts != "N_A" && !has_legal_targets(ab, orderer)) continue;
                { auto it = cur_game.payment_fail_counts.find(ab.source);
                  if (it != cur_game.payment_fail_counts.end() && it->second >= 2) continue; }
                std::string src_name = entity_name(ab.source);
                std::string desc = "Activate " + src_name + " (" + ab.category + ")";
                LegalAction non_mana_la(ACTIVATE_ABILITY, ab.source, ab, desc);
                non_mana_la.category = ActionCategory::ACTIVATE_ABILITY;
                actions.push_back(non_mana_la);
            }
        }
    }
    // Check hand for cards with ActivationZone$ Hand abilities (e.g. Talon Gates of Madara)
    for (auto card_entity : hand) {
        auto &card_data = global_coordinator.GetComponent<CardData>(card_entity);
        for (const auto &ab : card_data.abilities) {
            if (ab.ability_type != Ability::ACTIVATED) continue;
            if (ab.activation_zone != Zone::HAND) continue;
            // Check mana affordability
            if (!ab.activation_mana_cost.empty() && !can_pay_mana(priority_player, ab.activation_mana_cost, card_entity, orderer)) continue;
            // Check target legality
            if (ab.valid_tgts != "N_A" && ab.target_min > 0 && !has_legal_targets(ab, orderer)) continue;
            // sac_cost_spec: require controller has a permanent matching type
            if (!ab.sac_cost_spec.empty() &&
                controlled_permanents_matching(priority_player, ab.sac_cost_spec, orderer->mEntities).empty())
                continue;
            { auto it = cur_game.payment_fail_counts.find(card_entity);
              if (it != cur_game.payment_fail_counts.end() && it->second >= 2) continue; }
            std::string desc = "Activate " + card_data.name + " from hand (" + ab.category + ")";
            LegalAction la(ACTIVATE_ABILITY, card_entity, ab, desc);
            la.category = ActionCategory::ACTIVATE_ABILITY;
            actions.push_back(la);
        }
    }

    // not filtering mana abilities based on if they contribute to a spell- will revisit this if it makes ML harder
    /*
    for (auto &ma : useful_mana_abilities(legal_mana_abilities, pending_actions)) {
        actions.push_back(ma);
    }
    */
    if (!InputLogger::instance().is_machine_mode()) {
        for (auto &ma : legal_mana_abilities) {
            actions.push_back(ma);
        }
    }
    return actions;
}