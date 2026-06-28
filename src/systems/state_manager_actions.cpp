#include "state_manager.h"
#include "state_manager_internal.h"
#include "rules_modifying.h"

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

static bool count_intervening_condition(const std::string &expr, Zone::Ownership caster, int &out);

static bool can_afford_alt(const AltCost& alt_cost, Zone::Ownership priority_player,
                           Entity card_entity, std::shared_ptr<Orderer> orderer) {
    if (!alt_cost.has_alt_cost) return false;

    // Check SVar condition (e.g. Once Upon a Time: free only if first spell this game)
    if (!alt_cost.condition_svar.empty()) {
        const std::string &cond = alt_cost.condition_svar;
        // Mindbreak Trap (Trap alt cost, CR 702.59): "If an opponent cast three or more
        // spells this turn, you may pay {0}…". Scripted as
        // PlayerCountOpponents$Condition<OP><N> SpellsCastThisTurn — the alt cost is
        // enabled when at least one opponent's per-turn spell count satisfies the
        // condition. condition_compare is unset; the comparison op/threshold are embedded
        // in the SVar's "Condition<OP><N>" token.
        if (cond.find("PlayerCountOpponents$") != std::string::npos &&
            cond.find("SpellsCastThisTurn") != std::string::npos) {
            std::string compare;  // e.g. "GE3"
            size_t cpos = cond.find("Condition");
            if (cpos != std::string::npos) compare = cond.substr(cpos + 9);  // strip "Condition"
            // Truncate at the first space (the count metric follows the condition token).
            size_t sp = compare.find(' ');
            if (sp != std::string::npos) compare = compare.substr(0, sp);
            Zone::Ownership opp = (priority_player == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;
            int opp_spells =
                static_cast<int>(global_coordinator.GetComponent<Player>(get_player_entity(opp)).spells_cast_this_turn);
            if (!compare_svar(opp_spells, compare)) return false;
        } else {
            int svar_value = 0;
            if (cond.find("Count$YouCastThisGame") != std::string::npos) {
                Entity pp_entity = get_player_entity(priority_player);
                svar_value = static_cast<int>(global_coordinator.GetComponent<Player>(pp_entity).spells_cast_this_game);
            }
            if (!compare_svar(svar_value, alt_cost.condition_compare)) return false;
        }
    }

    // IsPresent$ <type>[.YouCtrl] — the alt cost is only available while the
    // caster controls a matching permanent (e.g. Snuff Out: control a Swamp).
    if (!alt_cost.condition_is_present.empty()) {
        std::string filter = alt_cost.condition_is_present;
        size_t dot = filter.find('.');
        std::string type_name = (dot == std::string::npos) ? filter : filter.substr(0, dot);
        bool found = false;
        for (auto e : orderer->mEntities) {
            if (!is_battlefield_permanent(e, priority_player)) continue;
            auto &perm = global_coordinator.GetComponent<Permanent>(e);
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
            if (!is_battlefield_permanent(e, priority_player)) continue;
            auto& perm = global_coordinator.GetComponent<Permanent>(e);
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

// Check ConditionPresent$ / ConditionCompare$ condition (rule-603.4 intervening-if, spell
// castability, and ConditionDefined$ Remembered subability gates all share this).
// Counts battlefield permanents matching the filter (or remembered cards when
// condition_on_remembered) and compares against the threshold (default ">= 1").
// Filter format: "Type.YouCtrl" or "Type.OppCtrl" (e.g. "Land.YouCtrl"); "Card" matches any.
// Evaluate a Count$<...> intervening-if expression (CR 603.4 dynamic condition) to an integer.
// Returns true and sets `out` when the token is recognized; returns false for an unrecognized
// Count$ token so the caller fails loudly instead of mis-reading the raw string as a board
// filter. This is the single place the intervening-if Count$ tokens are listed — a new
// "count X this turn / this game" condition is added here, once, rather than as another literal
// branch in evaluate_present_condition. Each token mirrors a count the engine tracks on Player.
static bool count_intervening_condition(const std::string &expr, Zone::Ownership caster, int &out) {
    Entity pe = get_player_entity(caster);
    const Player *pl = global_coordinator.entity_has_component<Player>(pe)
                           ? &global_coordinator.GetComponent<Player>(pe)
                           : nullptr;
    // Ocelot Pride: "if you gained life this turn".
    if (expr.find("LifeYouGainedThisTurn") != std::string::npos) {
        out = pl ? pl->life_gained_this_turn : 0;
        return true;
    }
    // Arclight Phoenix: "if you've cast three or more instant and sorcery spells this turn"
    // (the engine tracks the combined instant+sorcery count on the player).
    if (expr.find("ThisTurnCast") != std::string::npos &&
        (expr.find("Instant") != std::string::npos || expr.find("Sorcery") != std::string::npos)) {
        out = pl ? static_cast<int>(pl->instant_sorcery_spells_cast_this_turn) : 0;
        return true;
    }
    return false;
}

bool evaluate_present_condition(const Ability &ab, Zone::Ownership caster, std::shared_ptr<Orderer> orderer) {
    if (ab.condition_present.empty()) return true;
    // Empty compare means the bare "if you control a <thing>" form → at least one.
    std::string compare = ab.condition_compare.empty() ? "GE1" : ab.condition_compare;

    // ConditionDefined$ Remembered: count remembered cards, not battlefield permanents.
    if (ab.condition_on_remembered) {
        // ConditionPresent$ Card.ExiledWithSource (Skyclave Apparition's TrigToken): only the
        // remembered cards that are STILL exiled (currently in the exile zone) count. CR 707/the
        // card's reminder text: when Skyclave leaves, the token is made only if the exiled card
        // is still exiled — if it has already returned to another zone, no token (and a card that
        // can't be found / is gone yields none either).
        if (ab.condition_present == "Card.ExiledWithSource") {
            size_t still_exiled = 0;
            for (auto e : cur_game.remembered_entities) {
                if (global_coordinator.entity_has_component<Zone>(e) &&
                    global_coordinator.GetComponent<Zone>(e).location == Zone::EXILE)
                    still_exiled++;
            }
            return compare_svar(static_cast<int>(still_exiled), compare);
        }
        // A specific filter (other than the bare "Card") counts only the remembered cards that
        // match it as battlefield permanents. Phelia's "If it entered under your control, put a
        // +1/+1 counter on it" is ConditionDefined$ Imprinted | ConditionPresent$
        // Card.YouCtrl+ThisTurnEntered — count the returned card iff it is now a permanent the
        // ability's controller controls that entered this turn (CR 122 / the card text). Phelia's
        // counter gate checks the card the delayed trigger just put back onto the battlefield.
        if (!ab.condition_present.empty() && ab.condition_present != "Card") {
            MatchCtx ctx;
            ctx.controller = caster;
            ctx.source = ab.source;
            size_t matching = 0;
            for (auto e : cur_game.remembered_entities) {
                if (permanent_matches_filter(e, ab.condition_present, ctx)) {
                    matching++;
                    continue;
                }
                // A card returned to the battlefield earlier in THIS resolution (Phelia's
                // exile-and-return) has its Zone set to BATTLEFIELD with its controller assigned,
                // but its Permanent component is created by the deferred SBA pass — so it is not
                // yet a battlefield permanent for permanent_matches_filter. Fall back to Zone data:
                // it entered this turn by definition (it just returned), so the YouCtrl/OppCtrl
                // clause is checked against Zone.controller.
                if (!global_coordinator.entity_has_component<Zone>(e)) continue;
                const auto &z = global_coordinator.GetComponent<Zone>(e);
                if (z.location != Zone::BATTLEFIELD) {
                    // A remembered card in a non-battlefield zone (a revealed top-of-library
                    // card, or a remembered card in hand/graveyard/exile) is matched by its
                    // PRINTED characteristics — permanent_matches_filter is battlefield-only, so
                    // fall back to card_matches_filter for the type/color portion of the filter
                    // (a YouCtrl/OppCtrl qualifier is a no-op off the battlefield there).
                    if (!card_matches_filter(e, ab.condition_present, ctx)) continue;
                    // A controller qualifier on a card that LEFT the battlefield (Boomerang
                    // Basics: "If you controlled that permanent" against the bounced permanent) is
                    // resolved from last-known information — the controller it had as it left
                    // (CR 608.2g) — captured in cur_game.last_known_info when it left play.
                    bool youctrl = ab.condition_present.find("YouCtrl") != std::string::npos;
                    bool oppctrl = ab.condition_present.find("OppCtrl") != std::string::npos;
                    if (youctrl || oppctrl) {
                        auto lit = cur_game.last_known_info.find(e);
                        Zone::Ownership lc = lit != cur_game.last_known_info.end()
                                                 ? lit->second.controller : Zone::UNKNOWN;
                        bool ctrl_ok = youctrl ? (lc == caster)
                                               : (lc != caster && lc != Zone::UNKNOWN);
                        if (!ctrl_ok) continue;
                    }
                    matching++;
                    continue;
                }
                if (global_coordinator.entity_has_component<Permanent>(e)) continue;  // handled above
                bool youctrl = ab.condition_present.find("YouCtrl") != std::string::npos;
                bool oppctrl = ab.condition_present.find("OppCtrl") != std::string::npos;
                bool ctrl_ok = youctrl ? (z.controller == caster)
                             : oppctrl ? (z.controller != caster)
                                       : true;
                if (ctrl_ok) matching++;
            }
            return compare_svar(static_cast<int>(matching), compare);
        }
        size_t count = cur_game.remembered_entities.size();
        return compare_svar(static_cast<int>(count), compare);
    }

    // ConditionPresent$ Card.wasCastFromYourHandByYou (Amped Raptor): the ability's source —
    // the card that triggered it — must have entered the battlefield as a spell its controller
    // cast from their own hand. Read the persisted flag off its Permanent (set when the
    // permanent was created from a hand cast). A non-hand entry leaves the flag false.
    if (ab.condition_present == "Card.wasCastFromYourHandByYou") {
        return global_coordinator.entity_has_component<Permanent>(ab.source) &&
               global_coordinator.GetComponent<Permanent>(ab.source).cast_from_hand_by_controller;
    }

    // IsPresent$ Card.Self: the source must itself be on the battlefield (Kappa Cannoneer's
    // "Whenever another artifact you control enters" only functions while Kappa is in play).
    if (ab.condition_present == "Card.Self") {
        return is_battlefield_permanent(ab.source);
    }

    // IsPresent$ Card.Self+counters_GE<N>_<TYPE>: the source must be on the battlefield AND
    // carry at least N counters of the given type (Moonshadow: "while this creature has a
    // -1/-1 counter on it" → Card.Self+counters_GE1_M1M1). CR 122.1/603.4 — the counter
    // count is re-checked when the trigger would go on the stack and again on resolution, so
    // once the last -1/-1 counter is gone the trigger stops firing.
    if (ab.condition_present.rfind("Card.Self+counters_GE", 0) == 0) {
        if (!is_battlefield_permanent(ab.source)) return false;
        std::string rest = ab.condition_present.substr(std::string("Card.Self+counters_GE").size());
        size_t us = rest.find('_');
        int need = 1;
        std::string ctype = "M1M1";
        if (us != std::string::npos) {
            need = std::stoi(rest.substr(0, us));
            ctype = rest.substr(us + 1);
        } else if (!rest.empty()) {
            need = std::stoi(rest);
        }
        return get_counters(ab.source, ctype) >= need;
    }

    // Count$<...> dynamic intervening-if (Ocelot Pride's life gained this turn, Arclight
    // Phoenix's instant/sorcery spells cast this turn, ...). These are counts over game history
    // / player state, NOT board presence, so route every Count$ condition through the shared
    // count helper and compare. A Count$ token the helper does not recognize fails loudly here
    // rather than falling through to the permanent-presence scan below — where the raw
    // "Count$..." string would be read as a permanent type name, match nothing, and yield a
    // confident-but-wrong count (silently suppressing or firing the trigger).
    if (ab.condition_present.rfind("Count$", 0) == 0) {
        int value = 0;
        if (!count_intervening_condition(ab.condition_present, caster, value)) {
            game_log("WARNING: unrecognized Count$ intervening-if condition '%s' — treated as unmet.\n",
                     ab.condition_present.c_str());
            return false;
        }
        return compare_svar(value, compare);
    }

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
    if (type_filter == "Card") type_filter.clear();  // "Card" = any permanent

    Zone::Ownership required_ctrl = you_ctrl ? caster :
        opp_ctrl ? (caster == Zone::PLAYER_A ? Zone::PLAYER_B : Zone::PLAYER_A) :
        Zone::UNKNOWN;

    size_t count = 0;
    for (auto e : orderer->mEntities) {
        if (!is_battlefield_permanent(e, required_ctrl)) continue;
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

    return compare_svar(static_cast<int>(count), compare);
}


std::vector<LegalAction> StateManager::determine_legal_actions(
    const Game &game, std::shared_ptr<Orderer> orderer, std::shared_ptr<StackManager> stack_manager) {
    std::vector<LegalAction> actions;          // return value

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
        int land_play_limit = 1 + rules_mod::land_play_bonus(priority_player);
        bool may_play_from_graveyard = rules_mod::may_play_lands_from_graveyard(priority_player);

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
                condition_ok = evaluate_present_condition(ab, priority_player, orderer);
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
            if (rules_mod::cast_prohibited(priority_player, card_data)) continue;

            ManaValue effective_cost = effective_base_cost(card_data, priority_player);

            // X-cost spells: base cost (without X) is enough to be castable;
            // X value is chosen at cast time in action_processor
            bool can_regular = can_pay_mana(priority_player, effective_cost, card_entity,
                                            orderer, card_data.has_delve, card_data.has_improvise);

            // Additional Sacrifice-a-<type> cost on the spell itself (Natural Order:
            // "As an additional cost to cast this spell, sacrifice a green creature").
            // The spell can't be cast unless a matching permanent is available to
            // sacrifice (CR 601.2f). General: any spell whose SPELL ability Cost$ carries
            // a Sac<...> token, matched (incl. color qualifier) like an activation cost.
            std::string spell_sac_spec = spell_additional_sac_spec(card_data);
            if (!spell_sac_spec.empty() &&
                controlled_permanents_matching(priority_player, spell_sac_spec,
                                               orderer->mEntities, card_entity).empty())
                can_regular = false;

            bool can_alt = can_afford_alt(card_data.alt_cost, priority_player, card_entity, orderer);

            if (can_regular) actions.push_back(la);
            if (can_alt) {
                LegalAction alt_la = la;
                alt_la.use_alt_cost = true;
                alt_la.description = "Cast " + card_data.name + " (alternate cost)";
                actions.push_back(alt_la);
            }
            // Offspring (CR 702.171): optional ADDITIONAL cost. Offer a separate cast option
            // that must pay the base cost plus the offspring cost together.
            if (card_data.has_offspring) {
                ManaValue offspring_total = effective_cost;
                for (Colors c : card_data.offspring_cost) offspring_total.insert(c);
                if (can_pay_mana(priority_player, offspring_total, card_entity, orderer,
                                 card_data.has_delve, card_data.has_improvise)) {
                    LegalAction off_la = la;
                    off_la.use_offspring = true;
                    off_la.description = "Cast " + card_data.name + " (offspring)";
                    actions.push_back(off_la);
                }
            }
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

        // Flashback sacrifice cost (Cabal Therapy: Flashback—Sacrifice a creature): can't be
        // cast unless a matching permanent is available to sacrifice (CR 601.2f / 601.3a).
        if (!gcd.flashback_alt_cost.sac_cost_spec.empty() &&
            controlled_permanents_matching(priority_player, gcd.flashback_alt_cost.sac_cost_spec,
                                           orderer->mEntities, gy_entity).empty())
            continue;

        // A graveyard-cast static (Grafdigger's Cage: Origin$ Graveyard) prohibits flashback.
        if (rules_mod::cast_prohibited(priority_player, gcd, Zone::GRAVEYARD)) continue;

        LegalAction fb_la(CAST_SPELL, gy_entity, "Cast " + gcd.name + " (flashback)");
        fb_la.category = ActionCategory::CAST_SPELL;
        fb_la.use_flashback = true;
        actions.push_back(fb_la);
    }
    // ESCAPE (CR 702.139): a card in its owner's graveyard may be cast from there for its
    // escape cost (mana + additional cost). Same sorcery-speed timing flashback uses unless the
    // card is an instant. Nethergoyf's additional cost is ExileFromGrave with a "≥N card types
    // among the chosen cards" constraint, so the cast is only legal when OTHER cards in the
    // caster's graveyard collectively cover those N types (otherwise the cost is unpayable).
    for (auto gy_entity : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Zone>(gy_entity)) continue;
        auto &gz = global_coordinator.GetComponent<Zone>(gy_entity);
        if (gz.location != Zone::GRAVEYARD || gz.owner != priority_player) continue;
        if (!global_coordinator.entity_has_component<CardData>(gy_entity)) continue;
        auto &gcd = global_coordinator.GetComponent<CardData>(gy_entity);
        if (!gcd.has_escape) continue;

        bool esc_is_instant = card_has_type(gcd, "Instant");
        bool can_cast_now = esc_is_instant ||
                            ((game.cur_step == FIRST_MAIN || game.cur_step == SECOND_MAIN) &&
                             (game.player_a_turn == game.player_a_has_priority) && stack_empty);
        if (!can_cast_now) continue;

        // Spell-target legality (Nethergoyf has none, but keep general for future escape cards).
        bool tgt_ok = true;
        for (const auto &ab : gcd.abilities) {
            if (ab.ability_type != Ability::SPELL) continue;
            tgt_ok = has_legal_targets(ab, orderer);
            break;
        }
        if (!tgt_ok) continue;

        if (!can_pay_mana(priority_player, gcd.escape_mana_cost, gy_entity, orderer)) continue;

        // ExileFromGrave group-type constraint: enough OTHER graveyard cards must be available
        // to collectively reach the required number of distinct card types (CR 601.2f).
        if (gcd.escape_alt_cost.exile_grave_min_types > 0 &&
            graveyard_card_types(priority_player, orderer->mEntities, gy_entity) <
                gcd.escape_alt_cost.exile_grave_min_types)
            continue;

        if (rules_mod::cast_prohibited(priority_player, gcd, Zone::GRAVEYARD)) continue;

        LegalAction esc_la(CAST_SPELL, gy_entity, "Cast " + gcd.name + " (escape)");
        esc_la.category = ActionCategory::CAST_SPELL;
        esc_la.use_escape = true;
        actions.push_back(esc_la);
    }
    // CAST-FROM-GRAVEYARD PERMISSIONS (Emry's AB$ Effect): a card the priority player has
    // been granted permission to cast this turn (CR 601.3e). It is cast from the graveyard
    // for its normal cost, at the timing its type allows. Only the granting player may
    // cast it, so filter the permission set by graveyard owner.
    for (auto gy_entity : cur_game.may_cast_this_turn) {
        if (!global_coordinator.entity_has_component<Zone>(gy_entity)) continue;
        auto &gz = global_coordinator.GetComponent<Zone>(gy_entity);
        if (gz.location != Zone::GRAVEYARD || gz.owner != priority_player) continue;
        if (!global_coordinator.entity_has_component<CardData>(gy_entity)) continue;
        auto &gcd = global_coordinator.GetComponent<CardData>(gy_entity);
        if (is_land_card(gcd)) continue;  // lands aren't cast (601.1)

        // Flash / instant cards may be cast anytime; everything else is sorcery-speed.
        bool can_cast_at_instant_speed = card_has_type(gcd, "Instant");
        for (const auto &kw : gcd.keywords)
            if (kw == "Flash") { can_cast_at_instant_speed = true; break; }
        bool can_cast_now = can_cast_at_instant_speed ||
            ((game.cur_step == FIRST_MAIN || game.cur_step == SECOND_MAIN) &&
             (game.player_a_turn == game.player_a_has_priority) && stack_empty);
        if (!can_cast_now) continue;

        // Any targeting requirement must have at least one legal target.
        bool tgt_ok = true;
        for (const auto &ab : gcd.abilities) {
            if (ab.ability_type != Ability::SPELL) continue;
            tgt_ok = has_legal_targets(ab, orderer);
            break;
        }
        if (!tgt_ok) continue;

        if (rules_mod::cast_prohibited(priority_player, gcd, Zone::GRAVEYARD))
            continue;

        ManaValue gy_cost = effective_base_cost(gcd, priority_player);
        if (!can_pay_mana(priority_player, gy_cost, gy_entity, orderer, gcd.has_delve, gcd.has_improvise))
            continue;

        LegalAction gy_la(CAST_SPELL, gy_entity, "Cast " + gcd.name + " (from graveyard)");
        gy_la.category = ActionCategory::CAST_SPELL;
        actions.push_back(gy_la);
    }
    // IMPULSE-CAST PERMISSIONS (Amped Raptor's DB$ Play): a card exiled this turn that its
    // controller may cast, paying an alternative RESOURCE cost (energy or life equal to its
    // mana value) instead of its mana cost (CR 707 / 118.9). Cast from EXILE at the timing its
    // type allows; only the granted player may cast it, and only if they can pay the resource.
    for (const auto &[ex_entity, perm_grant] : cur_game.impulse_cast_permission) {
        if (perm_grant.caster != priority_player) continue;
        if (!global_coordinator.entity_has_component<Zone>(ex_entity)) continue;
        auto &ez = global_coordinator.GetComponent<Zone>(ex_entity);
        if (ez.location != Zone::EXILE) continue;  // must still be in exile
        if (!global_coordinator.entity_has_component<CardData>(ex_entity)) continue;
        auto &ecd = global_coordinator.GetComponent<CardData>(ex_entity);
        if (is_land_card(ecd)) continue;  // lands aren't cast (601.1)

        // Timing: instants / Flash cards anytime; everything else sorcery-speed.
        bool can_cast_at_instant_speed = card_has_type(ecd, "Instant");
        for (const auto &kw : ecd.keywords)
            if (kw == "Flash") { can_cast_at_instant_speed = true; break; }
        bool can_cast_now = can_cast_at_instant_speed ||
            ((game.cur_step == FIRST_MAIN || game.cur_step == SECOND_MAIN) &&
             (game.player_a_turn == game.player_a_has_priority) && stack_empty);
        if (!can_cast_now) continue;

        // Affordability of the alternative resource cost.
        Entity pe = get_player_entity(priority_player);
        if (!global_coordinator.entity_has_component<Player>(pe)) continue;
        auto &ppl = global_coordinator.GetComponent<Player>(pe);
        if (perm_grant.resource == Game::ImpulseCastPermission::ENERGY) {
            if (player_energy(ppl) < perm_grant.amount) continue;
        } else {  // LIFE — must be able to pay without the cost itself being lethal is not a
                  // legality bar in MTG, but a player won't be forced; require enough life so
                  // the optional cast is sensibly offered.
            if (ppl.life_total < perm_grant.amount) continue;
        }

        // Any targeting requirement must have at least one legal target.
        bool tgt_ok = true;
        for (const auto &ab : ecd.abilities) {
            if (ab.ability_type != Ability::SPELL) continue;
            tgt_ok = has_legal_targets(ab, orderer);
            break;
        }
        if (!tgt_ok) continue;

        if (rules_mod::cast_prohibited(priority_player, ecd, Zone::EXILE))
            continue;

        LegalAction imp_la(CAST_SPELL, ex_entity, "Cast " + ecd.name + " (impulse, alt cost)");
        imp_la.category = ActionCategory::CAST_SPELL;
        imp_la.impulse_cast = true;
        actions.push_back(imp_la);
    }
    // checking permanents for activated abilities
    // mana abilities parsed last
    // Simple tap-only mana sources collected via shared function
    std::vector<LegalAction> legal_mana_abilities =
        collect_mana_legal_actions(priority_player, orderer, 0, /*at_priority=*/true);
    for (auto entity : orderer->mEntities) {
        if (!is_battlefield_permanent(entity, priority_player)) continue;
        auto &permanent = global_coordinator.GetComponent<Permanent>(entity);

        // Sorcery-speed window: controller's main phase with an empty stack. Gates both the
        // Equip ability and planeswalker loyalty abilities (606.3), so it is computed once.
        bool sorcery_speed = (game.cur_step == FIRST_MAIN || game.cur_step == SECOND_MAIN) &&
                             (game.player_a_turn == game.player_a_has_priority) &&
                             stack_manager->is_empty();

        // Check if any CantBeActivated static suppresses this permanent's abilities.
        // (Mana abilities are collected separately above, so they remain usable — this
        // matches Disruptor Flute's ValidSA$ Activated.!ManaAbility.)
        if (rules_mod::activation_prohibited(entity)) continue;

        // EQUIP: equipment's equip ability is sorcery-speed (main phase, your turn, empty stack).
        // The Equip keyword is parsed into is_equipment/equip_cost but produces no stored Ability,
        // so synthesise the action here when there is a creature to equip and the cost is payable.
        if (global_coordinator.entity_has_component<CardData>(entity)) {
            auto &cd = global_coordinator.GetComponent<CardData>(entity);
            if (cd.is_equipment && sorcery_speed) {
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
            // Loyalty abilities (606.3): sorcery-speed only, once per turn per permanent across
            // all its loyalty abilities, and a minus ability needs enough loyalty (606.6; equality
            // is legal — may go to exactly 0 and die to the SBA).
            if (ab.is_loyalty_ability) {
                if (!sorcery_speed) continue;
                if (permanent.loyalty_ability_activated_this_turn) continue;
                if (ab.loyalty_cost < 0 && get_counters(entity, "LOYALTY") < -ab.loyalty_cost) continue;
            }
            // SorcerySpeed$ True (Ba Sing Se's earthbend): activatable only any time its
            // controller could cast a sorcery (CR 605.x) — main phase, their turn, empty stack.
            if (ab.sorcery_speed_only && !sorcery_speed) continue;
            // Activation$ gate (CR 602.5): "activate only if <condition>" (e.g. Metalcraft) —
            // illegal unless the controller meets the named condition. (Mana abilities take the
            // same gate in collect_available_mana_sources; this covers non-mana gated activations.)
            if (!activation_condition_met(ab, priority_player, orderer->mEntities, entity)) continue;
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
            // sac_cost_spec: require controller has a permanent matching type (honouring a
            // .Other self-exclusion against the ability's source — "another creature").
            if (!ab.sac_cost_spec.empty() &&
                controlled_permanents_matching(priority_player, ab.sac_cost_spec, orderer->mEntities, ab.source).empty())
                continue;
            // Return cost: require controller has a land of given subtype
            if (!ab.return_cost_type.empty() &&
                controlled_permanents_matching(priority_player, ab.return_cost_type, orderer->mEntities).empty())
                continue;
            if (ability_is_mana(ab)) {
                // All mana abilities — including InstantSpeed$ ones (e.g. LED) and AB$
                // ManaReflected (Mox Amber) — are collected via collect_mana_legal_actions above
                // and resolve off-stack. None go on the stack.
                continue;
            } else {
                // Non-mana activated ability (e.g. ChangeZone for fetch lands, Destroy for Wasteland).
                // Gate on the post-ReduceCost$ cost so legality matches what payment will charge.
                ManaValue ab_cost = effective_activation_mana_cost(ab, priority_player, orderer);
                if (!ab_cost.empty() && !can_pay_mana(priority_player, ab_cost, ab.source, orderer)) continue;
                // PayEnergy<N> additional cost (CR 122.1c): you can't pay {E} you don't have.
                if (ab.energy_cost > 0 &&
                    player_energy(global_coordinator.GetComponent<Player>(get_player_entity(priority_player))) < ab.energy_cost)
                    continue;
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
            // Check mana affordability against the post-ReduceCost$ cost (Eiganjo's Channel is
            // cheaper per legendary creature you control), so legality matches payment.
            ManaValue from_hand_cost = effective_activation_mana_cost(ab, priority_player, orderer);
            if (!from_hand_cost.empty() && !can_pay_mana(priority_player, from_hand_cost, card_entity, orderer)) continue;
            // PayEnergy<N> additional cost (CR 122.1c): you can't pay {E} you don't have.
            if (ab.energy_cost > 0 &&
                player_energy(global_coordinator.GetComponent<Player>(get_player_entity(priority_player))) < ab.energy_cost)
                continue;
            // Check target legality
            if (ab.valid_tgts != "N_A" && ab.target_min > 0 && !has_legal_targets(ab, orderer)) continue;
            // sac_cost_spec: require controller has a permanent matching type (honouring a
            // .Other self-exclusion against the activating card).
            if (!ab.sac_cost_spec.empty() &&
                controlled_permanents_matching(priority_player, ab.sac_cost_spec, orderer->mEntities, card_entity).empty())
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
    bool machine = InputLogger::instance().is_machine_mode();
    for (auto &ma : legal_mana_abilities) {
        // In machine mode, normal mana sources stay hidden and are auto-paid during cost
        // payment. Instant-speed sources (e.g. LED) can only be activated at priority to
        // float mana, so the ML agent must be offered those explicitly.
        if (machine && !ma.ability.instant_speed) continue;
        actions.push_back(ma);
    }
    return actions;
}