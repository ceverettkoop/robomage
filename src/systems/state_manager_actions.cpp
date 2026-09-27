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
static bool present_condition_raw(const Ability &ab, Zone::Ownership caster, std::shared_ptr<Orderer> orderer);
static bool aura_enchant_target_available(const CardData &card_data, Zone::Ownership caster,
                                          std::shared_ptr<Orderer> orderer);
static void offer_modal_back_face_casts(std::vector<LegalAction> &actions,
                                        Zone::Ownership priority_player,
                                        std::shared_ptr<Orderer> orderer, bool sorcery_window);
static std::string loyalty_cost_label(const Ability &ab);
static std::vector<Entity> stack_removal_targets(std::shared_ptr<Orderer> orderer);
static bool sorcery_timing_ok(const Game &game, Zone::Ownership seat, bool stack_empty);
static bool spell_timing_ok(const CardData &face, Zone::Ownership caster, bool sorcery_window);

// The sorcery-timing window for `seat` (CR 307.1): its own turn, a main phase, and an empty
// stack. Casting a sorcery, playing a land (CR 305.2), activating Equip (CR 702.6a), a loyalty
// ability (CR 606.3) or any "activate only as a sorcery" ability, and the companion special
// action (CR 702.139a) all wait for it.
static bool sorcery_timing_ok(const Game &game, Zone::Ownership seat, bool stack_empty) {
    return stack_empty && (game.cur_step == FIRST_MAIN || game.cur_step == SECOND_MAIN) &&
           game.player_a_turn == (seat == Zone::PLAYER_A);
}

// May `caster` cast a spell with `face`'s characteristics now, as far as timing goes (CR 601.3,
// 307.1)? An instant, a Flash card, or a spell a cast-with-flash permission covers (Teferi, Time
// Raveler's +1, CR 702.8) may be cast any time its caster holds priority; any other spell only in
// the caster's sorcery window. An opponent's sorcery-speed lock (Teferi's static "each opponent can
// cast spells only any time they could cast a sorcery") overrides both, so under it every spell —
// even an instant, or one the caster's own permission grants flash — waits for the sorcery window.
// Every cast-offering path shares this one rule: hand, split / modal DFC back face, flashback,
// escape, the graveyard and exile cast permissions, and the suspend special action.
static bool spell_timing_ok(const CardData &face, Zone::Ownership caster, bool sorcery_window) {
    if (sorcery_window) return true;
    if (rules_mod::opponent_sorcery_speed_locked(caster)) return false;
    return card_has_type(face, "Instant") || card_has_keyword(face, "Flash") ||
           rules_mod::cast_with_flash_active(caster, face);
}

// An Aura (CR 303.4 / 601.2c) targets the object it will enchant as it is cast, so EVERY
// cast-offering path — hand, modal back face, flashback, escape, a cast-from-graveyard
// permission (Emry), an impulse/free cast from exile (Amped Raptor, suspend) — may only offer
// it while a legal object matching its Enchant restriction exists. Builds the same transient
// targeting ability the cast flow's AURA_TARGET step uses, so the offer gate and the cast-time
// target menu can never disagree. The enchant filter is controller-relative (Sheltered by
// Ghosts: Enchant Creature.YouCtrl), so the transient ability MUST carry the actual caster as
// controller — has_legal_targets evaluates YouCtrl/OppCtrl from ability_perspective_player,
// which defaults to PLAYER_A when left unset. Returns true for non-Auras.
// (This gate was originally inlined only in the hand-cast loop; an Amped Raptor impulse grant
// exiling Animate Dead off an empty-graveyard board then offered the cast ungated and crashed
// AURA_TARGET's empty target menu — CR 601.2c requires the gate on every offering path.)
static bool aura_enchant_target_available(const CardData &card_data, Zone::Ownership caster,
                                          std::shared_ptr<Orderer> orderer) {
    if (card_data.enchant_filter.empty()) return true;  // not an Aura
    return has_legal_targets(enchant_target_ability(0, card_data, caster), orderer);
}

// Chosen targets of every stack object that would destroy or exile a battlefield
// permanent (category "Destroy", or "ChangeZone" battlefield->exile). Player-entity
// targets may be included; callers only membership-test against permanent entities.
static std::vector<Entity> stack_removal_targets(std::shared_ptr<Orderer> orderer) {
    std::vector<Entity> tgts;
    for (Entity e : orderer->get_stack()) {
        if (!global_coordinator.entity_has_component<Ability>(e)) continue;
        auto &ab = global_coordinator.GetComponent<Ability>(e);
        bool exiles_permanent =
            ab.category == "ChangeZone" && ab.destination == Zone::EXILE &&
            (ab.origin == Zone::BATTLEFIELD ||
             std::find(ab.origins.begin(), ab.origins.end(), Zone::BATTLEFIELD) != ab.origins.end());
        if (ab.category != "Destroy" && !exiles_permanent) continue;
        if (ab.target != 0) tgts.push_back(ab.target);
        tgts.insert(tgts.end(), ab.targets.begin(), ab.targets.end());
    }
    return tgts;
}

// A planeswalker loyalty ability's cost as an MTG-notation suffix (" [+1]",
// " [0]", " [-3]", " [-X]"); empty for a non-loyalty ability. Shown in the
// action menu so the player sees each ability's loyalty cost, not just its
// effect category.
static std::string loyalty_cost_label(const Ability &ab) {
    if (!ab.is_loyalty_ability) return "";
    if (ab.loyalty_cost_is_x) return ab.loyalty_cost < 0 ? " [-X]" : " [+X]";
    if (ab.loyalty_cost == 0) return " [0]";
    int magnitude = ab.loyalty_cost < 0 ? -ab.loyalty_cost : ab.loyalty_cost;
    std::string sign = ab.loyalty_cost < 0 ? "-" : "+";
    return " [" + sign + std::to_string(magnitude) + "]";
}

static bool can_afford_alt(const CardData& card_data, const AltCost& alt_cost,
                           Zone::Ownership priority_player,
                           Entity card_entity, std::shared_ptr<Orderer> orderer) {
    if (!alt_cost.has_alt_cost) return false;

    // Miracle (CR 702.94) is never a normal priority-menu cast: it is castable ONLY through the
    // dedicated immediate "cast it for its miracle cost / do not cast" decision presented right when
    // its linked triggered ability resolves (proc_mandatory_choice's miracle-cast branch, gated on
    // its own affordability check). So the alternate-cost cast option is never offered here.
    if (alt_cost.is_miracle) return false;

    // Spectacle (CR 702.107a): the spectacle cost may be paid only if an opponent of the
    // caster lost life this turn. Two-player game — the sole opponent is the other seat.
    if (alt_cost.is_spectacle) {
        Zone::Ownership opp = opponent_of(priority_player);
        if (global_coordinator.GetComponent<Player>(get_player_entity(opp)).life_lost_this_turn <= 0)
            return false;
    }

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
            Zone::Ownership opp = opponent_of(priority_player);
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

    // Mana portion of the alt cost (e.g. Evoke:R) with the active SetCost floor
    // (Trinisphere) folded in — CR 601.2f applies the floor AFTER the alternative cost is
    // substituted, so even a Cost$ 0 / pitch cast must be able to pay up to the floor.
    ManaValue alt_mana = floored_alt_mana_cost(card_data, alt_cost.mana_cost, priority_player);

    // Free alt cost: castable iff any floor imposed on it is payable
    if (alt_cost.is_free)
        return alt_mana.empty() || can_pay_mana(priority_player, alt_mana, card_entity, orderer);

    if (alt_cost.return_to_hand_count > 0) {
        size_t matching = controlled_permanents_matching(priority_player, alt_cost.return_to_hand_type,
                                                         orderer->mEntities).size();
        // Fall through (don't return true here): a pitch-style cost (Daze) still has to
        // cover the floored mana portion checked below when a SetCost floor is active.
        if (matching < static_cast<size_t>(alt_cost.return_to_hand_count)) return false;
    }

    if (alt_cost.life_cost > 0) {
        Entity pp_entity = get_player_entity(priority_player);
        if (!can_pay_life(global_coordinator.GetComponent<Player>(pp_entity), alt_cost.life_cost))
            return false;
    }

    // Sac<N/Type> alt cost (CR 118.9, Fireblast): the caster must control at least N permanents
    // matching the filter to sacrifice. (Fall through — a floored mana portion is still checked below.)
    if (alt_cost.sac_cost_count > 0) {
        if (static_cast<int>(controlled_permanents_matching(priority_player, alt_cost.sac_cost_spec,
                                                            orderer->mEntities, card_entity).size()) <
            alt_cost.sac_cost_count)
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
            if (required_color != NO_COLOR && effective_colors(e).count(required_color)) {
                has_match = true; break;
            }
        }
        if (!has_match) return false;
    }

    // Floored mana portion of the alt cost (computed above)
    if (!alt_mana.empty()) {
        if (!can_pay_mana(priority_player, alt_mana, card_entity, orderer)) return false;
    }

    return true;
}

// Check ConditionPresent$ / ConditionCompare$ condition (rule-603.4 intervening-if and the
// CR 608.2c resolution-time "if" gate, including ConditionDefined$ Remembered, share this).
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

static bool present_condition_raw(const Ability &ab, Zone::Ownership caster, std::shared_ptr<Orderer> orderer) {
    if (ab.condition_present.empty()) return true;
    // Empty compare means the bare "if you control a <thing>" form → at least one.
    std::string compare = ab.condition_compare.empty() ? "GE1" : ab.condition_compare;

    // ConditionDefined$ ExiledWith (The Creation of Avacyn II & III): the condition is a property
    // check on the card the source Saga exiled face down — is it a Creature card? Match the card's
    // PRINTED characteristics against condition_present (card_matches_filter is battlefield-agnostic;
    // the exiled card sits in exile). Absent card ⇒ 0 matches (condition unmet).
    if (ab.condition_on_exiled_with) {
        Entity ew = exiled_with_card(ab.source);
        int matches = 0;
        if (ew != 0 && global_coordinator.entity_has_component<CardData>(ew)) {
            MatchCtx ctx;
            ctx.controller = caster;
            ctx.source = ab.source;
            if (card_matches_filter(ew, ab.condition_present, ctx)) matches = 1;
        }
        return compare_svar(matches, compare);
    }

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
                        const LastKnownInfo *lki = lki_for(e);
                        Zone::Ownership lc = lki ? lki->controller : Zone::UNKNOWN;
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
    // permanent was created from a hand cast). A non-hand entry leaves the flag false. If the
    // source has already left the battlefield before this trigger resolves (e.g. Amped Raptor
    // killed in response to its own ETB trigger), the Permanent is gone; fall back to the
    // last-known-information snapshot captured as it left play (CR 603.10 / 608.2h) so the
    // exile-cast clause is not silently lost.
    if (ab.condition_present == "Card.wasCastFromYourHandByYou") {
        if (global_coordinator.entity_has_component<Permanent>(ab.source))
            return global_coordinator.GetComponent<Permanent>(ab.source).cast_from_hand_by_controller;
        const LastKnownInfo *lki = departed_lki_for(ab.source);
        return lki && lki->cast_from_hand_by_controller;
    }

    // IsPresent$ Card.Self: the source must itself be on the battlefield (Kappa Cannoneer's
    // "Whenever another artifact you control enters" only functions while Kappa is in play).
    if (ab.condition_present == "Card.Self") {
        return is_battlefield_permanent(ab.source);
    }

    // IsPresent$ Card.StrictlySelf (Animate Dead's ETB: "When CARDNAME enters, if it's on the
    // battlefield, ..."): strictly this object itself must be on the battlefield when the
    // trigger would go on the stack AND when it resolves (CR 603.4 — either check failing means
    // the trigger does nothing). Unlike Card.Self this is NOT Forge's "another permanent
    // enters" idiom, so the source's own entry satisfies it (no trigger_self_excluded in
    // parse.cpp). Must not fall through to the generic presence scan below, which would read
    // the type token "Card" as any-permanent and pass vacuously whenever anything is in play.
    if (ab.condition_present == "Card.StrictlySelf") {
        return is_battlefield_permanent(ab.source);
    }

    // Card.Self+escaped: the source permanent entered because its spell was cast from the
    // graveyard for its Escape cost (CR 702.139). Read the persisted flag off its Permanent.
    // Uro's TrigSac uses ConditionNotPresent$ Card.Self+escaped ("sacrifice it unless it
    // escaped"), so condition_negate inverts this in the wrapper below. General — any escape
    // card with an "if it escaped" clause reuses it.
    if (ab.condition_present == "Card.Self+escaped") {
        return global_coordinator.entity_has_component<Permanent>(ab.source) &&
               global_coordinator.GetComponent<Permanent>(ab.source).cast_with_escape;
    }

    // IsPresent$ Card.Self+counters_<OP><N>_<TYPE>: the source must be on the battlefield AND its
    // count of counters of the given type satisfy the comparison <OP><N> (Forge spells the op as
    // GE/LE/EQ/NE/GT/LT). Moonshadow: "while this creature has a -1/-1 counter on it" →
    // counters_GE1_M1M1; Dark Depths: "when this has no ice counters on it" → counters_EQ0_ICE.
    // CR 122.1/603.4/603.8 — the counter count is re-checked whenever the condition is evaluated
    // (trigger placement, resolution, and each state-based check for a Mode$ Always state trigger).
    if (ab.condition_present.rfind("Card.Self+counters_", 0) == 0) {
        if (!is_battlefield_permanent(ab.source)) return false;
        std::string rest = ab.condition_present.substr(std::string("Card.Self+counters_").size());
        // rest is "<OP><N>_<TYPE>", e.g. "EQ0_ICE" / "GE1_M1M1".
        std::string op = rest.substr(0, 2);          // two-letter comparator
        std::string after = rest.substr(2);          // "<N>_<TYPE>"
        size_t us = after.find('_');
        std::string num = (us != std::string::npos) ? after.substr(0, us) : after;
        std::string ctype = (us != std::string::npos) ? after.substr(us + 1) : "M1M1";
        return compare_svar(get_counters(ab.source, ctype), op + num);
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

    // A board-presence condition (Birthing Ritual's Creature.YouCtrl, Edge of Autumn's
    // Land.YouCtrl, Permanent.Red+YouCtrl+Other): count the battlefield permanents matching the
    // whole filter by their current characteristics (a Clue token is not a creature; an animated
    // manland is), relative to this ability's controller and source.
    int count = count_battlefield_matching(ab.condition_present, caster, ab.source);
    return compare_svar(count, compare);
}

// Public entry point: evaluate the present condition, applying ConditionNotPresent$ negation.
// An empty condition_present is "no condition" → always satisfied (the negate flag is never set
// in that case, since ConditionNotPresent always carries a filter). Used by the intervening-if
// trigger checks (CR 603.4) and the resolution-time condition gate (CR 608.2c) alike.
bool evaluate_present_condition(const Ability &ab, Zone::Ownership caster, std::shared_ptr<Orderer> orderer) {
    bool raw = present_condition_raw(ab, caster, orderer);
    return ab.condition_negate ? !raw : raw;
}


// Modal DFC whose BACK face is a NONLAND spell (Tergrid, God of Fright // Tergrid's Lantern):
// offer a CAST_SPELL that casts the BACK face — paying the back's mana cost and using the back
// face's characteristics/abilities (CR 712.8). The front face is offered as a normal cast by the
// main hand loop, and the LAND-back case is offered there as a PLAY_LAND; only the nonland-back
// CAST is added here. Kept in its own hand pass so a prohibition/`continue` on the front face
// doesn't suppress the back-face option (the two faces are cast independently).
static void offer_modal_back_face_casts(std::vector<LegalAction> &actions,
                                        Zone::Ownership priority_player,
                                        std::shared_ptr<Orderer> orderer, bool sorcery_window) {
    auto hand = orderer->get_hand(priority_player);
    for (auto card_entity : hand) {
        auto &front = global_coordinator.GetComponent<CardData>(card_entity);
        // Offer the back half for both modal DFCs (nonland back) and split cards (CR 709): both
        // store the second face in `backside` and cast it via the shared cast_back_face path.
        if ((!front.is_modal_dfc && !front.is_split) || !front.backside) continue;
        const CardData &back = *front.backside;
        if (is_land_card(back)) continue;  // land back is a PLAY_LAND, handled in the main loop

        // Timing is the back face's own (Gone is an instant), under the shared rule.
        if (!spell_timing_ok(back, priority_player, sorcery_window)) continue;

        // Spell-target legality (mirrors the front-face checks).
        bool tgt_ok = true;
        for (const auto &ab : back.abilities) {
            if (ab.ability_type != Ability::SPELL) continue;
            tgt_ok = has_legal_targets(cast_gate_probe(ab, card_entity, priority_player), orderer);
            break;
        }
        if (!tgt_ok) continue;
        // Aura enchant-target gate (CR 303.4 / 601.2c) — see aura_enchant_target_available.
        if (!aura_enchant_target_available(back, priority_player, orderer)) continue;

        if (rules_mod::cast_prohibited(priority_player, back)) continue;

        auto pf_it = cur_game.payment_fail_counts.find(card_entity);
        if (pf_it != cur_game.payment_fail_counts.end() && pf_it->second >= 2) continue;

        ManaValue cost = effective_base_cost(back, priority_player);
        if (!can_pay_mana(priority_player, cost, card_entity, orderer,
                          back.has_delve, back.has_improvise))
            continue;

        LegalAction la(CAST_SPELL, card_entity, "Cast " + back.name);
        la.category = ActionCategory::CAST_SPELL;
        la.cast_back_face = true;
        la.option_ordinal = 3;  // cast variant: 3 = modal-DFC back face
        actions.push_back(la);
    }
}

std::vector<LegalAction> StateManager::determine_legal_actions(
    const Game &game, std::shared_ptr<Orderer> orderer, std::shared_ptr<StackManager> stack_manager) {
    std::vector<LegalAction> actions;          // return value

    // Determine whose turn/priority it is
    Zone::Ownership priority_player = priority_seat();
    Entity priority_player_entity = get_player_entity(priority_player);
    bool stack_empty = stack_manager->is_empty();
    bool sorcery_window = sorcery_timing_ok(game, priority_player, stack_empty);

    // Graveyard / exile cards the priority player has some play route for (the shared
    // card_play_permission predicate, which ignores timing and cost), in ascending entity
    // order, with their route bits. Every graveyard / exile play loop below walks this list
    // and applies its own timing, cost and target gates.
    std::vector<std::pair<Entity, unsigned>> zone_play_routes;
    for (auto e : orderer->mEntities) {
        unsigned routes = card_play_permission(e, priority_player).sources;
        if (routes) zone_play_routes.emplace_back(e, routes);
    }

    // PASS PRIORITY
    LegalAction la(PASS_PRIORITY, "Pass priority");
    la.category = ActionCategory::PASS_PRIORITY;
    actions.push_back(la);

    // LAND FROM HAND — requires empty stack (sorcery-speed)
    if (sorcery_window && global_coordinator.entity_has_component<Player>(priority_player_entity)) {
        // Effective land play allowance (base 1 + AdjustLandPlays statics) minus the
        // lands already played, through the shared rules_mod expression the ML
        // observation's mana-development block reports.
        if (rules_mod::land_drops_remaining(priority_player) > 0) {
            // Check hand for lands
            auto hand = orderer->get_hand(priority_player);
            for (auto card_entity : hand) {
                auto &card_data = global_coordinator.GetComponent<CardData>(card_entity);
                // A modal DFC whose BACK face is a land (Witch Enchanter // Witch-Blessed
                // Meadow): playing the back face is a land play, subject to the same one-
                // land-per-turn drop (CR 712.x / 305.2). The front face is still offered as a
                // cast in the spell loop below. (A modal DFC whose back is a nonland spell
                // would instead be a cast — keyed on the back face's card type.)
                if (!card_playable_as_land(card_data)) continue;
                bool back_face = !is_land_card(card_data);
                std::string desc =
                    "Play " + (back_face ? card_data.backside->name : card_data.name);
                LegalAction la(SPECIAL_ACTION, card_entity, desc);
                la.category = ActionCategory::PLAY_LAND;
                la.play_back_face = back_face;
                actions.push_back(la);
            }
            // Lands in the graveyard while a play-lands-from-graveyard static applies
            // (the GRAVEYARD_LAND route).
            for (const auto &[gy_e, routes] : zone_play_routes) {
                if (!(routes & CardPlayPermission::GRAVEYARD_LAND)) continue;
                auto &gcd = global_coordinator.GetComponent<CardData>(gy_e);
                std::string desc = "Play " + gcd.name + " (from graveyard)";
                LegalAction la(SPECIAL_ACTION, gy_e, desc);
                la.category = ActionCategory::PLAY_LAND;
                actions.push_back(la);
            }
        }
    }
    // COMPANION (CR 702.139) — at sorcery speed, the player may pay {3} ONCE per game to put their
    // chosen companion from the sideboard ("outside the game") into their hand. The chosen companion
    // and the deckbuilding-restriction gate were resolved at game start (setup_companions); here we
    // only offer the special action while the companion is still in the sideboard, it hasn't been
    // used yet, and {3} is affordable. Mirrors the play-land special action's timing gate.
    if (sorcery_window && global_coordinator.entity_has_component<Player>(priority_player_entity)) {
        auto &player = global_coordinator.GetComponent<Player>(priority_player_entity);
        Entity comp = player.chosen_companion;
        if (comp != 0 && !player.companion_brought_to_hand &&
            global_coordinator.entity_has_component<Zone>(comp) &&
            global_coordinator.GetComponent<Zone>(comp).location == Zone::SIDEBOARD &&
            global_coordinator.entity_has_component<CardData>(comp)) {
            ManaValue three = {GENERIC, GENERIC, GENERIC};
            if (can_pay_mana(priority_player, three, comp, orderer)) {
                auto &cd = global_coordinator.GetComponent<CardData>(comp);
                LegalAction la(SPECIAL_ACTION, comp,
                               "Companion: pay {3}, put " + cd.name + " into your hand");
                la.category = ActionCategory::COMPANION;
                la.companion_to_hand = true;
                la.card_is_public = true;
                actions.push_back(la);
            }
        }
    }

    // checking for spells to cast from hand
    // TODO spells cast from elsewhere
    auto hand = orderer->get_hand(priority_player);
    for (auto card_entity : hand) {
        auto &card_data = global_coordinator.GetComponent<CardData>(card_entity);
        if (card_has_type(card_data, "Land")) continue;  // can't cast land
        bool can_cast_now = spell_timing_ok(card_data, priority_player, sorcery_window);
        // Check that at least one legal target exists for any targeting requirement. A
        // ConditionPresent$ "if ..." clause is checked only at resolution (CR 608.2c), so it never
        // gates the cast.
        bool tgt_ok = true;
        for (const auto &ab : card_data.abilities) {
            if (ab.ability_type != Ability::SPELL) continue;
            // Mode-aware target legality (CR 601.2c): for a Gift spell the required target type
            // switches on the gift promise (Into the Flood Maw: a creature without the gift, a
            // nonland permanent with it), so the spell is castable iff a legal target exists for
            // at least one reachable mode. Reduces to has_legal_targets for ordinary spells.
            // Probe with the real cast source/controller (card_entity) so source-dependent target
            // restrictions — protection from this spell's color, OppCtrl — match select_target and
            // a protected-only target (Emrakul vs white, Scryb Ranger vs blue) is not offered.
            Ability probe = cast_gate_probe(ab, card_entity, priority_player);
            tgt_ok = spell_has_castable_targets(probe, orderer, priority_player, card_data.has_gift);
            break;
        }
        // Aura enchant-target gate (CR 303.4 / 601.2c) — see aura_enchant_target_available.
        if (tgt_ok)
            tgt_ok = aura_enchant_target_available(card_data, priority_player, orderer);
        // Machine mode only: action-masking optimization — don't offer a conditional-destroy
        // spell to the RL agent when no target on the board would currently pass the
        // condition (e.g. Fatal Push: only show if a creature with mana value <= the current
        // revolt-aware threshold exists). This is a masking heuristic, NOT a rules gate —
        // the spell can still legally target any creature in CLI/interactive play.
        if (InputLogger::instance().is_machine_schedule() && tgt_ok) {
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
                        if (!is_battlefield_permanent(ce)) continue;
                        if (!global_coordinator.entity_has_component<Creature>(ce)) continue;
                        // A token creature carries no CardData; a non-copy token has no mana
                        // cost and therefore mana value 0 (CR 111.7), which is always <= the
                        // threshold, so it is a valid conditional-destroy target. Mirror
                        // effect_destroy.cpp, which likewise treats a CardData-less target as
                        // mana value 0 and destroys it — without this, boards whose only small
                        // creatures are tokens (e.g. Monk/Orc Army) hid the legal Fatal Push.
                        int cmc = global_coordinator.entity_has_component<CardData>(ce)
                                      ? card_mana_value(global_coordinator.GetComponent<CardData>(ce))
                                      : 0;
                        if (cmc <= threshold) { any_valid = true; break; }
                    }
                    if (!any_valid) tgt_ok = false;
                }
                break;
            }
        }

        auto pf_it = cur_game.payment_fail_counts.find(card_entity);
        bool payment_blocked = pf_it != cur_game.payment_fail_counts.end() && pf_it->second >= 2;
        if (can_cast_now && tgt_ok && !payment_blocked) {
            std::string desc = "Cast " + card_data.name;
            LegalAction la(CAST_SPELL, card_entity, desc);
            la.category = ActionCategory::CAST_SPELL;
            la.option_ordinal = 0;  // cast variant: 0 = normal

            // Check CantBeCast statics from cached active_statics
            if (rules_mod::cast_prohibited(priority_player, card_data)) continue;

            ManaValue effective_cost = effective_base_cost(card_data, priority_player);

            // X-cost spells: base cost (without X) is enough to be castable;
            // X value is chosen at cast time in action_processor. Hybrid pips ({W/U}, {2/W})
            // are folded in via resolve_hybrid_cost (castable iff SOME hybrid assignment is
            // payable); with no hybrids this is exactly can_pay_mana.
            bool can_regular = resolve_hybrid_cost(priority_player, effective_cost,
                                                   card_data.hybrid_mana, card_entity, orderer,
                                                   card_data.has_delve, card_data.has_improvise);

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

            bool can_alt = can_afford_alt(card_data, card_data.alt_cost, priority_player, card_entity, orderer);

            if (can_regular) actions.push_back(la);
            if (can_alt) {
                LegalAction alt_la = la;
                alt_la.use_alt_cost = true;
                alt_la.option_ordinal = 1;  // cast variant: 1 = alternate/impending cost
                alt_la.description = "Cast " + card_data.name +
                    (card_data.alt_cost.is_impending ? " (impending)"
                     : card_data.alt_cost.is_spectacle ? " (spectacle)"
                     : card_data.alt_cost.is_miracle   ? " (miracle)"
                     : card_data.alt_cost.is_warp      ? " (warp)"
                                                        : " (alternate cost)");
                actions.push_back(alt_la);
            }
            // Offspring (CR 702.171): optional ADDITIONAL cost. Offer a separate cast option
            // that must pay the base cost plus the offspring cost together.
            if (card_data.has_offspring) {
                ManaValue offspring_total = effective_cost;
                for (Colors c : card_data.offspring_cost) offspring_total.insert(c);
                if (resolve_hybrid_cost(priority_player, offspring_total, card_data.hybrid_mana,
                                        card_entity, orderer, card_data.has_delve,
                                        card_data.has_improvise)) {
                    LegalAction off_la = la;
                    off_la.use_offspring = true;
                    off_la.option_ordinal = 2;  // cast variant: 2 = offspring
                    off_la.description = "Cast " + card_data.name + " (offspring)";
                    actions.push_back(off_la);
                }
            }
        }
        // SUSPEND (CR 702.62a, first ability): from the hand, at the timing you could begin to cast
        // the card (`can_cast_now` — sorcery speed for a sorcery), its owner may instead pay the
        // suspend cost and exile it with N time counters. This is a special action (doesn't use the
        // stack). Its targets are chosen only later, when the last counter is removed and it is cast
        // for free, so no legal target is required now (702.62 casts it then, not here); the card
        // just must not be under a cast prohibition (702.62c) and the suspend mana cost must be
        // affordable. General over any Suspend card. Offered independently of the normal-cast block.
        if (card_data.has_suspend && can_cast_now &&
            !rules_mod::cast_prohibited(priority_player, card_data) &&
            can_pay_mana(priority_player, card_data.suspend_cost, card_entity, orderer)) {
            LegalAction sus_la(SPECIAL_ACTION, card_entity, "Suspend " + card_data.name);
            sus_la.category = ActionCategory::CAST_SPELL;  // cast-adjacent "play this card" action
            sus_la.suspend_action = true;
            actions.push_back(sus_la);
        }
    }
    // Modal DFC nonland back faces: offer the BACK face as a CAST_SPELL (the front face's normal
    // cast and the land-back PLAY_LAND were handled in the hand loop above).
    offer_modal_back_face_casts(actions, priority_player, orderer, sorcery_window);
    // checking graveyard for flashback spells (the FLASHBACK route)
    for (const auto &[gy_entity, routes] : zone_play_routes) {
        if (!(routes & CardPlayPermission::FLASHBACK)) continue;
        auto &gcd = global_coordinator.GetComponent<CardData>(gy_entity);

        if (!spell_timing_ok(gcd, priority_player, sorcery_window)) continue;

        bool tgt_ok = true;
        for (const auto &ab : gcd.abilities) {
            if (ab.ability_type != Ability::SPELL) continue;
            tgt_ok = has_legal_targets(cast_gate_probe(ab, gy_entity, priority_player), orderer);
            break;
        }
        if (!tgt_ok) continue;
        // Aura enchant-target gate (CR 303.4 / 601.2c) — see aura_enchant_target_available.
        if (!aura_enchant_target_available(gcd, priority_player, orderer)) continue;

        // Check affordability: flashback mana cost (floored — flashback is an alternative
        // cost, CR 702.34a, so an active SetCost floor applies to it too) + life cost
        bool can_afford_fb = can_pay_mana(
            priority_player, floored_alt_mana_cost(gcd, gcd.flashback_mana_cost, priority_player), gy_entity, orderer);
        if (can_afford_fb && gcd.flashback_alt_cost.life_cost > 0) {
            Entity pp_entity = get_player_entity(priority_player);
            if (!can_pay_life(global_coordinator.GetComponent<Player>(pp_entity),
                              gcd.flashback_alt_cost.life_cost))
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
        fb_la.option_ordinal = 4;  // cast variant: 4 = flashback
        actions.push_back(fb_la);
    }
    // ESCAPE (CR 702.139): a card in its owner's graveyard may be cast from there for its
    // escape cost (mana + additional cost). Same sorcery-speed timing flashback uses unless the
    // card is an instant. Nethergoyf's additional cost is ExileFromGrave with a "≥N card types
    // among the chosen cards" constraint, so the cast is only legal when OTHER cards in the
    // caster's graveyard collectively cover those N types (otherwise the cost is unpayable).
    // The permission itself is the ESCAPE route.
    for (const auto &[gy_entity, routes] : zone_play_routes) {
        if (!(routes & CardPlayPermission::ESCAPE)) continue;
        auto &gcd = global_coordinator.GetComponent<CardData>(gy_entity);

        if (!spell_timing_ok(gcd, priority_player, sorcery_window)) continue;

        // Spell-target legality (Nethergoyf has none, but keep general for future escape cards).
        bool tgt_ok = true;
        for (const auto &ab : gcd.abilities) {
            if (ab.ability_type != Ability::SPELL) continue;
            tgt_ok = has_legal_targets(cast_gate_probe(ab, gy_entity, priority_player), orderer);
            break;
        }
        if (!tgt_ok) continue;
        // Aura enchant-target gate (CR 303.4 / 601.2c) — see aura_enchant_target_available.
        if (!aura_enchant_target_available(gcd, priority_player, orderer)) continue;

        // An escape life cost must be payable (CR 119.4).
        if (!can_pay_life(global_coordinator.GetComponent<Player>(get_player_entity(priority_player)),
                          gcd.escape_alt_cost.life_cost))
            continue;

        // Escape is an alternative cost (CR 702.139a): fold in any active SetCost floor.
        if (!can_pay_mana(priority_player, floored_alt_mana_cost(gcd, gcd.escape_mana_cost, priority_player),
                          gy_entity, orderer))
            continue;

        // ExileFromGrave group-type constraint: enough OTHER graveyard cards must be available
        // to collectively reach the required number of distinct card types (CR 601.2f).
        if (gcd.escape_alt_cost.exile_grave_min_types > 0 &&
            graveyard_card_types(priority_player, orderer->mEntities, gy_entity) <
                gcd.escape_alt_cost.exile_grave_min_types)
            continue;

        // ExileFromGrave literal-count constraint (Uro: exile FIVE other cards): enough OTHER
        // graveyard cards must exist to pay the cost (CR 601.2f).
        if (gcd.escape_alt_cost.exile_grave_count > 0 &&
            graveyard_card_count(priority_player, orderer->mEntities, gy_entity) <
                gcd.escape_alt_cost.exile_grave_count)
            continue;

        if (rules_mod::cast_prohibited(priority_player, gcd, Zone::GRAVEYARD)) continue;

        LegalAction esc_la(CAST_SPELL, gy_entity, "Cast " + gcd.name + " (escape)");
        esc_la.category = ActionCategory::CAST_SPELL;
        esc_la.use_escape = true;
        esc_la.option_ordinal = 5;  // cast variant: 5 = escape
        actions.push_back(esc_la);
    }
    // CAST-FROM-GRAVEYARD PERMISSIONS (Emry's AB$ Effect): a card the priority player has
    // been granted permission to cast this turn (CR 601.3e). It is cast from the graveyard
    // for its normal cost, at the timing its type allows. The GRAVEYARD_CAST route covers
    // only the graveyard owner and never a land (601.1).
    for (const auto &[gy_entity, routes] : zone_play_routes) {
        if (!(routes & CardPlayPermission::GRAVEYARD_CAST)) continue;
        auto &gcd = global_coordinator.GetComponent<CardData>(gy_entity);

        if (!spell_timing_ok(gcd, priority_player, sorcery_window)) continue;

        // Any targeting requirement must have at least one legal target.
        bool tgt_ok = true;
        for (const auto &ab : gcd.abilities) {
            if (ab.ability_type != Ability::SPELL) continue;
            tgt_ok = has_legal_targets(cast_gate_probe(ab, gy_entity, priority_player), orderer);
            break;
        }
        if (!tgt_ok) continue;
        // Aura enchant-target gate (CR 303.4 / 601.2c) — see aura_enchant_target_available.
        if (!aura_enchant_target_available(gcd, priority_player, orderer)) continue;

        if (rules_mod::cast_prohibited(priority_player, gcd, Zone::GRAVEYARD))
            continue;

        ManaValue gy_cost = effective_base_cost(gcd, priority_player);
        if (!can_pay_mana(priority_player, gy_cost, gy_entity, orderer, gcd.has_delve, gcd.has_improvise))
            continue;

        LegalAction gy_la(CAST_SPELL, gy_entity, "Cast " + gcd.name + " (from graveyard)");
        gy_la.category = ActionCategory::CAST_SPELL;
        gy_la.option_ordinal = 6;  // cast variant: 6 = cast-from-graveyard permission (Emry)
        actions.push_back(gy_la);
    }
    // IMPULSE-CAST PERMISSIONS (Amped Raptor's DB$ Play): a card exiled this turn that its
    // controller may cast, paying an alternative RESOURCE cost (energy or life equal to its
    // mana value) instead of its mana cost (CR 707 / 118.9). Cast from EXILE at the timing its
    // type allows; only the granted player may cast it, and only if they can pay the resource.
    // The EXILE_GRANT route covers a card still in exile whose grant names this player (and
    // a land only under a NORMAL grant that allows lands).
    for (const auto &[ex_entity, routes] : zone_play_routes) {
        if (!(routes & CardPlayPermission::EXILE_GRANT)) continue;
        const auto &perm_grant = cur_game.impulse_cast_permission.at(ex_entity);
        auto &ecd = global_coordinator.GetComponent<CardData>(ex_entity);

        // A LAND among the exiled cards: only a NORMAL "play" permission (Light Up the Stage's
        // "you may PLAY those cards") may play it — a land play, sorcery-timing, own main phase,
        // empty stack, and a land drop remaining (CR 305.2 / 601.3e). A free/energy/life "cast"
        // grant (Ugin -11 / Amped Raptor) can't play a land (601.1), and card_play_permission
        // reports no route for a land under one.
        if (is_land_card(ecd)) {
            if (!sorcery_window) continue;
            // Same shared land-drop expression the hand loop above uses.
            if (rules_mod::land_drops_remaining(priority_player) <= 0) continue;
            LegalAction land_la(SPECIAL_ACTION, ex_entity, "Play " + ecd.name + " (from exile)");
            land_la.category = ActionCategory::PLAY_LAND;
            actions.push_back(land_la);
            continue;
        }

        bool can_cast_now = spell_timing_ok(ecd, priority_player, sorcery_window);
        // A suspend free cast (CR 702.62a) is made as an effect of resolving the last-time-counter
        // triggered ability during the caster's own upkeep, so it ignores the card's normal
        // sorcery/instant timing — offer it at any priority window this caster holds (until the
        // permission lapses at cleanup, i.e. "if you don't, it remains exiled").
        if (perm_grant.from_suspend) can_cast_now = true;
        if (!can_cast_now) continue;

        // Affordability of the alternative resource cost.
        Entity pe = get_player_entity(priority_player);
        if (!global_coordinator.entity_has_component<Player>(pe)) continue;
        auto &ppl = global_coordinator.GetComponent<Player>(pe);
        bool is_normal_play = (perm_grant.resource == Game::ImpulseCastPermission::NORMAL);
        if (perm_grant.resource == Game::ImpulseCastPermission::FREE) {
            // No cost to pay (Ugin -11 grant) — always affordable.
        } else if (is_normal_play) {
            // Play a nonland card for its NORMAL mana cost (Light Up the Stage): affordable iff
            // the full (cost-increase-adjusted, hybrid-resolved) base cost can be paid.
            ManaValue base = effective_base_cost(ecd, priority_player);
            if (!resolve_hybrid_cost(priority_player, base, ecd.hybrid_mana, ex_entity, orderer,
                                     ecd.has_delve, ecd.has_improvise))
                continue;
        } else if (perm_grant.resource == Game::ImpulseCastPermission::ENERGY) {
            if (player_energy(ppl) < perm_grant.amount) continue;
        } else {  // LIFE — must be able to pay without the cost itself being lethal is not a
                  // legality bar in MTG, but a player won't be forced; require enough life so
                  // the optional cast is sensibly offered.
            if (!can_pay_life(ppl, perm_grant.amount)) continue;
        }

        // Cost-increase / SetCost-floor statics apply to alternative costs too (CR 118.9d /
        // 601.2f): an impulse/free cast substitutes a {0} mana cost, but an active Trinisphere
        // floor pads that up to its minimum ({3}) and Thalia adds its surcharge — payable ON TOP
        // of the energy/life resource cost. Require the floored mana; empty (no floor/increase)
        // means no extra mana and this gate is a no-op. NORMAL plays already pay the full base
        // cost above, so this alt-cost floor doesn't apply to them.
        if (!is_normal_play) {
            ManaValue floor_mana = floored_alt_mana_cost(ecd, ManaValue{}, priority_player);
            if (!floor_mana.empty() && !can_pay_mana(priority_player, floor_mana, ex_entity, orderer))
                continue;
        }

        // Any targeting requirement must have at least one legal target.
        bool tgt_ok = true;
        for (const auto &ab : ecd.abilities) {
            if (ab.ability_type != Ability::SPELL) continue;
            tgt_ok = has_legal_targets(cast_gate_probe(ab, ex_entity, priority_player), orderer);
            break;
        }
        if (!tgt_ok) continue;
        // Aura enchant-target gate (CR 303.4 / 601.2c) — see aura_enchant_target_available.
        // The concrete crash this fixes: Animate Dead reanimates the opponent's Amped Raptor
        // (emptying the graveyard), the Raptor's impulse exiles a SECOND Animate Dead and
        // grants this energy-cast permission — which must not be offered while no creature
        // card is in any graveyard.
        if (!aura_enchant_target_available(ecd, priority_player, orderer)) continue;

        if (rules_mod::cast_prohibited(priority_player, ecd, Zone::EXILE))
            continue;

        const char *imp_suffix = (perm_grant.resource == Game::ImpulseCastPermission::FREE)
                                     ? " (from exile, no cost)"
                                 : is_normal_play ? " (from exile)"
                                                  : " (impulse, alt cost)";
        LegalAction imp_la(CAST_SPELL, ex_entity, "Cast " + ecd.name + imp_suffix);
        imp_la.category = ActionCategory::CAST_SPELL;
        imp_la.impulse_cast = true;
        imp_la.option_ordinal = 7;  // cast variant: 7 = impulse/free cast from exile
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

        // Check if any CantBeActivated static suppresses this permanent's abilities.
        // (Mana abilities are collected separately above, so they remain usable — this
        // matches Disruptor Flute's ValidSA$ Activated.!ManaAbility.)
        if (rules_mod::activation_prohibited(entity)) continue;

        // EQUIP: equipment's equip ability is sorcery-speed (main phase, your turn, empty stack).
        // The Equip keyword is parsed into is_equipment/equip_cost but produces no stored Ability,
        // so synthesise the action here when there is a creature to equip and the cost is payable.
        if (global_coordinator.entity_has_component<CardData>(entity)) {
            auto &cd = global_coordinator.GetComponent<CardData>(entity);
            if (cd.is_equipment && sorcery_window) {
                bool has_creature = !equip_candidates(entity, priority_player, orderer->mEntities).empty();
                if (has_creature && can_pay_mana(priority_player, cd.equip_cost, entity, orderer)) {
                    Ability equip_ab;
                    equip_ab.ability_type = Ability::ACTIVATED;
                    equip_ab.category = "Equip";
                    equip_ab.source = entity;
                    equip_ab.activation_mana_cost = cd.equip_cost;
                    std::string desc = (cd.is_reconfigure ? "Reconfigure " : "Equip ") + entity_name(entity);
                    LegalAction equip_la(ACTIVATE_ABILITY, entity, equip_ab, desc);
                    equip_la.category = ActionCategory::ACTIVATE_ABILITY;
                    // Synthesised activation (no stored Ability): fixed ordinal above
                    // any plausible ability-list index so it can't collide with the
                    // per-ability ordinals below (normalizer OPTION_ORDINAL_MAX = 63).
                    equip_la.option_ordinal = 32;
                    actions.push_back(equip_la);
                }
                // Reconfigure (CR 702.151): while attached, pay the cost to unattach. Sorcery-speed,
                // same cost as the attach. The unattach makes the permanent a creature again.
                if (cd.is_reconfigure && permanent.equipped_to != 0 &&
                    can_pay_mana(priority_player, cd.equip_cost, entity, orderer)) {
                    Ability unattach_ab;
                    unattach_ab.ability_type = Ability::ACTIVATED;
                    unattach_ab.category = "Unattach";
                    unattach_ab.source = entity;
                    unattach_ab.activation_mana_cost = cd.equip_cost;
                    std::string desc = "Unattach " + entity_name(entity);
                    LegalAction unattach_la(ACTIVATE_ABILITY, entity, unattach_ab, desc);
                    unattach_la.category = ActionCategory::ACTIVATE_ABILITY;
                    unattach_la.option_ordinal = 33;  // synthesised: see equip above
                    actions.push_back(unattach_la);
                }
            }
        }

        // ability_index: the ability's stable position in this permanent's ability
        // list, emitted as the action's option_ordinal so the ML observation can
        // tell same-permanent activations apart (e.g. a planeswalker's loyalty
        // abilities, which are otherwise feature-identical). Counted over the FULL
        // list (including skipped/mana abilities) so an ability keeps one ordinal
        // regardless of which of its siblings happen to be legal right now.
        int ability_index = -1;
        for (const auto &ab : permanent.abilities) {
            ++ability_index;
            if (ab.ability_type != Ability::ACTIVATED) continue;
            if (ab.activation_zone == Zone::HAND) continue;  // hand-only ability, not usable from battlefield
            // Loyalty abilities (606.3): sorcery-speed only, once per turn per permanent across
            // all its loyalty abilities, and a minus ability needs enough loyalty (606.6; equality
            // is legal — may go to exactly 0 and die to the SBA).
            if (ab.is_loyalty_ability) {
                if (!sorcery_window) continue;
                if (permanent.loyalty_ability_activated_this_turn) continue;
                // A fixed minus cost needs enough loyalty (606.6). An X minus cost (Chandra,
                // Flamecaller's [-X]) is legal at any loyalty — X is chosen 0..current loyalty.
                if (!ab.loyalty_cost_is_x && ab.loyalty_cost < 0 &&
                    get_counters(entity, "LOYALTY") < -ab.loyalty_cost) continue;
            }
            // SorcerySpeed$ True (Ba Sing Se's earthbend): activatable only any time its
            // controller could cast a sorcery (CR 605.x) — main phase, their turn, empty stack.
            if (ab.sorcery_speed_only && !sorcery_window) continue;
            // Activation$ gate (CR 602.5): "activate only if <condition>" (e.g. Metalcraft) —
            // illegal unless the controller meets the named condition. (Mana abilities take the
            // same gate in collect_available_mana_sources; this covers non-mana gated activations.)
            if (!activation_condition_met(ab, priority_player, orderer->mEntities, entity)) continue;
            // todo handle this elswewhere, tapping check
            if (ab.tap_cost && permanent.is_tapped) continue;
            if (ab.tap_cost && is_summoning_sick(entity)) continue;
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
                // A {T} in the ability's own cost spends the source's tap, so its mana ability is
                // NOT also available to pay with — exclude it, or a Blast Zone whose only other
                // land is an Ancient Tomb reads as able to pay {3} off 2 mana plus its own {C}.
                ManaValue ab_cost = effective_activation_mana_cost(ab, priority_player, orderer);
                if (!ab_cost.empty() &&
                    !can_pay_mana(priority_player, ab_cost, ab.source, orderer,
                                  /*has_delve=*/false, /*has_improvise=*/false,
                                  /*exclude_entity=*/ab.tap_cost ? entity : 0))
                    continue;
                // PayEnergy<N> additional cost (CR 122.1c): you can't pay {E} you don't have.
                if (ab.energy_cost > 0 &&
                    player_energy(global_coordinator.GetComponent<Player>(get_player_entity(priority_player))) < ab.energy_cost)
                    continue;
                // PayLife<N> additional cost (CR 119.4): you can't pay life you don't have. A
                // fetch land (Pay 1 life) at 1 life is still legal (you pay down to 0, then die);
                // only an ability costing MORE life than you have is filtered out here.
                if (!can_pay_life(global_coordinator.GetComponent<Player>(get_player_entity(priority_player)),
                                  ab.life_cost))
                    continue;
                if (ab.valid_tgts != "N_A" && !has_legal_targets(ab, orderer)) continue;
                { auto it = cur_game.payment_fail_counts.find(ab.source);
                  if (it != cur_game.payment_fail_counts.end() && it->second >= 2) continue; }
                std::string src_name = entity_name(ab.source);
                std::string desc = "Activate " + src_name + loyalty_cost_label(ab)
                                   + " (" + ab.category + ")";
                LegalAction non_mana_la(ACTIVATE_ABILITY, ab.source, ab, desc);
                non_mana_la.category = ActionCategory::ACTIVATE_ABILITY;
                non_mana_la.option_ordinal = ability_index;
                actions.push_back(non_mana_la);
            }
        }
    }
    // Check hand for cards with ActivationZone$ Hand abilities (e.g. Talon Gates of Madara)
    for (auto card_entity : hand) {
        auto &card_data = global_coordinator.GetComponent<CardData>(card_entity);
        int hand_ability_index = -1;  // ordinal: see the battlefield loop above
        for (const auto &ab : card_data.abilities) {
            ++hand_ability_index;
            if (ab.ability_type != Ability::ACTIVATED) continue;
            if (ab.activation_zone != Zone::HAND) continue;
            // Ninjutsu (CR 702.49e): activatable only during the declare-blockers step, after
            // blockers are declared, while the activator controls an unblocked attacker.
            if (ab.is_ninjutsu) {
                if (game.cur_step != DECLARE_BLOCKERS) continue;
                if (unblocked_attackers(orderer->mEntities, priority_player).empty()) continue;
            }
            // Check mana affordability against the post-ReduceCost$ cost (Eiganjo's Channel is
            // cheaper per legendary creature you control), so legality matches payment.
            ManaValue from_hand_cost = effective_activation_mana_cost(ab, priority_player, orderer);
            if (!from_hand_cost.empty() && !can_pay_mana(priority_player, from_hand_cost, card_entity, orderer)) continue;
            // PayEnergy<N> additional cost (CR 122.1c): you can't pay {E} you don't have.
            if (ab.energy_cost > 0 &&
                player_energy(global_coordinator.GetComponent<Player>(get_player_entity(priority_player))) < ab.energy_cost)
                continue;
            // PayLife<N> additional cost (CR 119.4): you can't pay life you don't have.
            if (!can_pay_life(global_coordinator.GetComponent<Player>(get_player_entity(priority_player)),
                              ab.life_cost))
                continue;
            // Check target legality. The bare CardData ability carries no source/controller, and
            // ability_perspective_player would fall back to the default-initialized controller
            // (player A) — evaluating .OppCtrl from the wrong seat when B activates (Boseiju's
            // Channel was offered targeting B's own nonbasic land). Stamp the real activator via
            // cast_gate_probe so the existence check matches what target selection will offer.
            if (ab.valid_tgts != "N_A" && ab.target_min > 0 &&
                !has_legal_targets(cast_gate_probe(ab, card_entity, priority_player), orderer)) continue;
            // sac_cost_spec: require controller has a permanent matching type (honouring a
            // .Other self-exclusion against the activating card).
            if (!ab.sac_cost_spec.empty() &&
                controlled_permanents_matching(priority_player, ab.sac_cost_spec, orderer->mEntities, card_entity).empty())
                continue;
            { auto it = cur_game.payment_fail_counts.find(card_entity);
              if (it != cur_game.payment_fail_counts.end() && it->second >= 2) continue; }
            std::string desc = ab.is_ninjutsu
                ? ("Ninjutsu " + card_data.name)
                : ("Activate " + card_data.name + " from hand (" + ab.category + ")");
            LegalAction la(ACTIVATE_ABILITY, card_entity, ab, desc);
            la.category = ActionCategory::ACTIVATE_ABILITY;
            la.option_ordinal = hand_ability_index;
            actions.push_back(la);
        }
    }

    // Check the graveyard for cards with ActivationZone$ Graveyard abilities (Unearth, CR 702.84).
    // Such abilities are activated from the graveyard at sorcery speed (controller's main phase,
    // empty stack, holding priority) and return the card to the battlefield.
    {
        for (auto card_entity : orderer->get_graveyard(priority_player)) {
            if (!global_coordinator.entity_has_component<CardData>(card_entity)) continue;
            auto &card_data = global_coordinator.GetComponent<CardData>(card_entity);
            int gy_ability_index = -1;  // ordinal: see the battlefield loop above
            for (const auto &ab : card_data.abilities) {
                ++gy_ability_index;
                if (ab.ability_type != Ability::ACTIVATED) continue;
                if (ab.activation_zone != Zone::GRAVEYARD) continue;
                if (ab.sorcery_speed_only && !sorcery_window) continue;
                ManaValue gy_cost = effective_activation_mana_cost(ab, priority_player, orderer);
                if (!gy_cost.empty() && !can_pay_mana(priority_player, gy_cost, card_entity, orderer)) continue;
                // Target-existence gate (CR 601.2c), stamped like the hand loop above — today's
                // graveyard activations (Unearth) don't target, but a targeted one must not be
                // offered with zero legal targets.
                if (ab.valid_tgts != "N_A" && ab.target_min > 0 &&
                    !has_legal_targets(cast_gate_probe(ab, card_entity, priority_player), orderer)) continue;
                { auto it = cur_game.payment_fail_counts.find(card_entity);
                  if (it != cur_game.payment_fail_counts.end() && it->second >= 2) continue; }
                std::string desc = "Unearth " + card_data.name;
                LegalAction la(ACTIVATE_ABILITY, card_entity, ab, desc);
                la.category = ActionCategory::ACTIVATE_ABILITY;
                la.option_ordinal = gy_ability_index;
                actions.push_back(la);
            }
        }
    }

    // CLI mode lists every collected mana ability so a human can float mana manually,
    // whether or not it contributes to a castable spell.
    bool machine = InputLogger::instance().is_machine_schedule();
    std::vector<Entity> removal_tgts;
    if (machine && !legal_mana_abilities.empty()) removal_tgts = stack_removal_targets(orderer);
    for (auto &ma : legal_mana_abilities) {
        // In machine mode, normal mana sources stay hidden and are auto-paid during cost
        // payment, with two exceptions offered at priority so the agent can float mana:
        // instant-speed sources (e.g. LED, only activatable here), and any source that is
        // a chosen target of a Destroy/exile effect on the stack (float in response to
        // removal, e.g. Wasteland, before the source leaves the battlefield).
        if (machine && !ma.ability.instant_speed &&
            std::find(removal_tgts.begin(), removal_tgts.end(), ma.source_entity) ==
                removal_tgts.end())
            continue;
        actions.push_back(ma);
    }
    return actions;
}