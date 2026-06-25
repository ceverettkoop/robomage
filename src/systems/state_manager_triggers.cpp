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

namespace {
// A triggered ability that has fired and is waiting to be put on the stack. We collect every
// trigger from the current batch of events first, then place them all in APNAP order (603.3b);
// the old code pushed each trigger the instant it was found (raw entity-ID order), which is not
// APNAP and gave no player a chance to order their own simultaneous triggers.
struct PendingTrigger {
    Ability ab;                 // fully prepared (source / controller / event-derived fields set)
    Zone::Ownership controller; // whose trigger this is (drives APNAP partitioning)
    Entity source = 0;          // source permanent (for logging)
    std::string label;          // choice label when its controller orders simultaneous triggers
    std::string log_line;       // narrative line emitted when it is placed on the stack
    bool needs_target = false;  // select a target at placement time if it still has legal targets
};
}  // namespace

// A short, distinct label so a player ordering two triggers from the same source can tell them
// apart (e.g. Endurance's evoke-sacrifice trigger vs. its enters-the-battlefield trigger).
static std::string trigger_label(const std::string &name, const Ability &ab);

// 603.3b: place all simultaneously-triggered abilities on the stack in APNAP order — the active
// player's triggers first (so they resolve last), then the non-active player's. Each player
// orders their own group via a mandatory choice when more than one trigger is theirs.
static void place_triggers_apnap(Game &game, std::shared_ptr<Orderer> orderer,
                                 std::vector<PendingTrigger> &pending);

// Drains all buffered events since the last call and puts any triggered abilities
// from battlefield permanents whose trigger condition matches onto the stack.
void StateManager::check_triggered_abilities(Game &game, std::shared_ptr<Orderer> orderer) {
    auto events = global_coordinator.drain_pending_events();

    // Every ability that triggers off this batch of events is collected here first, then placed
    // on the stack together in APNAP order (603.3b) — nothing is pushed mid-scan.
    std::vector<PendingTrigger> pending;

    // Fire any delayed triggers that match current events
    {
        std::vector<size_t> to_remove;
        for (size_t i = 0; i < game.delayed_triggers.size(); i++) {
            auto &dt = game.delayed_triggers[i];
            bool matched = false;
            for (const auto &ev : events) {
                if (ev.GetType() != dt.fire_on) continue;
                if (dt.fire_on == Events::UPKEEP_BEGAN && game.turn < dt.fire_on_turn) continue;
                // Owner check: only fire on the correct player's upkeep
                if (ev.HasParam(Params::PLAYER) &&
                    ev.GetParam<Entity>(Params::PLAYER) != dt.owner_entity) continue;
                matched = true;
                break;
            }
            if (matched) {
                // Determine controller from owner_entity
                Zone::Ownership ctrl = (dt.owner_entity == game.player_a_entity)
                                       ? Zone::PLAYER_A : Zone::PLAYER_B;
                Ability trigger_ab = dt.ability;
                trigger_ab.controller = ctrl;
                PendingTrigger pt;
                pt.ab = trigger_ab;
                pt.controller = ctrl;
                pt.source = trigger_ab.source;
                pt.label = "Delayed trigger";
                pt.log_line = "Delayed trigger fires.";
                pt.needs_target = (trigger_ab.valid_tgts != "N_A" && trigger_ab.target == 0);
                pending.push_back(pt);
                to_remove.push_back(i);
            }
        }
        for (auto it = to_remove.rbegin(); it != to_remove.rend(); ++it)
            game.delayed_triggers.erase(game.delayed_triggers.begin() + static_cast<ptrdiff_t>(*it));
    }

    if (!events.empty()) {
    for (auto entity : mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location != Zone::BATTLEFIELD) continue;

        auto &perm = global_coordinator.GetComponent<Permanent>(entity);
        if (perm.is_phased_out) continue;

        // Gather triggered abilities from all sources:
        // CardData/Token for innate abilities, Permanent for keyword-granted abilities
        std::vector<const std::vector<Ability>*> ab_sources;
        if (global_coordinator.entity_has_component<CardData>(entity))
            ab_sources.push_back(&global_coordinator.GetComponent<CardData>(entity).abilities);
        if (global_coordinator.entity_has_component<Token>(entity))
            ab_sources.push_back(&global_coordinator.GetComponent<Token>(entity).abilities);
        ab_sources.push_back(&perm.abilities);
        if (ab_sources.empty()) continue;

        const std::string ent_name = entity_name(entity);

        for (const auto &ev : events) {
            for (const auto *src : ab_sources) {
            for (const auto &ab : *src) {
                if (ab.ability_type != Ability::TRIGGERED) continue;
                if (ab.trigger_on == 0 || ab.trigger_on != ev.GetType()) continue;
                // "another" check: skip if the event entity is the triggering permanent itself
                if (ab.trigger_self_excluded && ev.HasParam(Params::ENTITY) &&
                    ev.GetParam<Entity>(Params::ENTITY) == entity) continue;
                // Card.Self: only fire when the event entity is the triggering permanent itself
                if (ab.trigger_only_self && ev.HasParam(Params::ENTITY) &&
                    ev.GetParam<Entity>(Params::ENTITY) != entity) continue;
                // Evoke self-sacrifice only fires when this permanent was cast via evoke
                if (ab.is_evoke_sacrifice && !perm.evoked) continue;
                // Don't fire front-face triggers on a transformed permanent
                if (perm.transformed) continue;
                // ValidPlayer$ You: only fire when the event's player matches the permanent's controller
                if (ab.trigger_valid_player_is_controller && ev.HasParam(Params::PLAYER)) {
                    Entity event_player = ev.GetParam<Entity>(Params::PLAYER);
                    Entity ctrl_entity = get_player_entity(perm.controller);
                    if (event_player != ctrl_entity) continue;
                }
                // DisableTriggers check (Doorkeeper Thrull): suppress ETB triggers caused by matching card types
                if (ev.GetType() == Events::CARD_CHANGED_ZONE &&
                    ev.GetParam<Zone::ZoneValue>(Params::DESTINATION) == Zone::BATTLEFIELD) {
                    bool suppressed = false;
                    Entity entering = ev.HasParam(Params::ENTITY) ? ev.GetParam<Entity>(Params::ENTITY) : 0;
                    for (const auto &as : g_active_statics) {
                        if (as.sa->category != "DisableTriggers") continue;
                        if (entering != 0 && global_coordinator.entity_has_component<CardData>(entering)) {
                            auto &ecd = global_coordinator.GetComponent<CardData>(entering);
                            for (auto &t : ecd.types) {
                                if (as.sa->disable_triggers_cause.find(t.name) != std::string::npos) {
                                    suppressed = true; break;
                                }
                            }
                        }
                        if (suppressed) break;
                    }
                    if (suppressed) continue;
                }

                // CARD_CHANGED_ZONE filters: origin, destination, card type
                if (ev.GetType() == Events::CARD_CHANGED_ZONE) {
                    Zone::ZoneValue ev_origin = ev.GetParam<Zone::ZoneValue>(Params::ORIGIN);
                    Zone::ZoneValue ev_dest   = ev.GetParam<Zone::ZoneValue>(Params::DESTINATION);
                    if (ab.trigger_zone_origin >= 0 &&
                        ev_origin != static_cast<Zone::ZoneValue>(ab.trigger_zone_origin)) continue;
                    if (ab.trigger_zone_destination >= 0 &&
                        ev_dest != static_cast<Zone::ZoneValue>(ab.trigger_zone_destination)) continue;
                    // ValidCard$ Creature filter
                    if (ab.trigger_valid_card_is_creature && ev.HasParam(Params::ENTITY)) {
                        Entity ev_card = ev.GetParam<Entity>(Params::ENTITY);
                        bool is_creature = global_coordinator.entity_has_component<Token>(ev_card);
                        if (!is_creature && global_coordinator.entity_has_component<CardData>(ev_card))
                            is_creature = is_creature_card(global_coordinator.GetComponent<CardData>(ev_card));
                        if (!is_creature) continue;
                    }
                    // ValidCard$ Instant/Sorcery filter (Murktide Regent)
                    if (ab.trigger_valid_card_is_instant_or_sorcery && ev.HasParam(Params::ENTITY)) {
                        Entity ev_card = ev.GetParam<Entity>(Params::ENTITY);
                        if (!global_coordinator.entity_has_component<CardData>(ev_card)) continue;
                        bool ok = false;
                        for (auto &t : global_coordinator.GetComponent<CardData>(ev_card).types)
                            if (t.kind == TYPE && (t.name == "Instant" || t.name == "Sorcery")) { ok = true; break; }
                        if (!ok) continue;
                    }
                    // ValidCard$ Land.* filter (landfall)
                    if (ab.trigger_valid_card_is_land && ev.HasParam(Params::ENTITY)) {
                        Entity ev_card = ev.GetParam<Entity>(Params::ENTITY);
                        bool is_land = global_coordinator.entity_has_component<CardData>(ev_card) &&
                                       is_land_card(global_coordinator.GetComponent<CardData>(ev_card));
                        if (!is_land) continue;
                    }
                }
                // Drawn trigger filters (Orcish Bowmasters): PLAYER_DREW_CARD
                if (ev.GetType() == Events::PLAYER_DREW_CARD) {
                    // ValidCard$ Card.OppOwn — the drawn card must be owned by an
                    // opponent of the source's controller (drawer != controller).
                    if (ab.trigger_valid_card_opp_own && ev.HasParam(Params::PLAYER)) {
                        Entity drawer = ev.GetParam<Entity>(Params::PLAYER);
                        Entity ctrl_entity = get_player_entity(perm.controller);
                        if (drawer == ctrl_entity) continue;
                    }
                    // FirstCardInDrawStep$ False — ignore the first card drawn in the
                    // drawer's draw step (the turn-based draw).
                    if (ab.trigger_exclude_first_draw_step && ev.HasParam(Params::FIRST_IN_STEP) &&
                        ev.GetParam<int>(Params::FIRST_IN_STEP) == 1)
                        continue;
                }

                // Spell count filter (Cori-Steel Cutter)
                if (ab.trigger_spell_count_eq > 0 && ev.HasParam(Params::PLAYER)) {
                    Entity ev_player = ev.GetParam<Entity>(Params::PLAYER);
                    if (!global_coordinator.entity_has_component<Player>(ev_player)) continue;
                    auto &pl = global_coordinator.GetComponent<Player>(ev_player);
                    if (pl.spells_cast_this_turn != ab.trigger_spell_count_eq) continue;
                }

                // Prepare the triggered ability and queue it; APNAP placement (and any target
                // selection) happens after the full scan, in place_triggers_apnap().
                Ability trigger_ab = ab;
                trigger_ab.source = entity;
                trigger_ab.controller = perm.controller;
                // For exalted, target the sole attacker from the event
                if (trigger_ab.category == "ExaltedBonus" && ev.HasParam(Params::ENTITY))
                    trigger_ab.target = ev.GetParam<Entity>(Params::ENTITY);
                // For combat damage triggers, capture the damage amount
                if (ev.GetType() == Events::COMBAT_DAMAGE_TO_PLAYER && ev.HasParam(Params::AMOUNT))
                    trigger_ab.trigger_damage_amount = ev.GetParam<uint32_t>(Params::AMOUNT);

                // 603.4 intervening-if: a trigger whose "if" condition is false right now does
                // not go on the stack at all (it is re-checked again on resolution).
                if (trigger_ab.intervening_if &&
                    !evaluate_present_condition(trigger_ab, perm.controller, orderer))
                    continue;

                PendingTrigger pt;
                pt.ab = trigger_ab;
                pt.controller = perm.controller;
                pt.source = entity;
                pt.label = trigger_label(ent_name, trigger_ab);
                pt.log_line = ent_name + " triggered";
                // Triggered abilities that require a target (e.g. Talon Gates of Madara's
                // "up to one target creature phases out") choose their target as the ability
                // goes on the stack, by the controller, in APNAP placement order.
                pt.needs_target = (trigger_ab.valid_tgts != "N_A" && trigger_ab.target == 0);
                pending.push_back(pt);
            }
            }
        }
    }
    }

    place_triggers_apnap(game, orderer, pending);
}

static std::string trigger_label(const std::string &name, const Ability &ab) {
    if (ab.is_evoke_sacrifice) return name + " (evoke: sacrifice)";
    std::string s = name + " (" + ab.category;
    if (ab.valid_tgts != "N_A" && !ab.valid_tgts.empty()) s += ", targeted";
    s += ")";
    return s;
}

static void place_triggers_apnap(Game &game, std::shared_ptr<Orderer> orderer,
                                 std::vector<PendingTrigger> &pending) {
    if (pending.empty()) return;

    Zone::Ownership active = game.player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B;
    Zone::Ownership apnap[2] = {active,
                                active == Zone::PLAYER_A ? Zone::PLAYER_B : Zone::PLAYER_A};

    for (Zone::Ownership owner : apnap) {
        // This owner's pending triggers, in the order they were collected.
        std::vector<size_t> group;
        for (size_t i = 0; i < pending.size(); i++)
            if (pending[i].controller == owner) group.push_back(i);

        // 603.3b: the owner puts their simultaneously-triggered abilities on the stack in an
        // order of their choosing. We place one at a time; the first chosen ends up on the
        // bottom of the stack (resolves last). A single trigger needs no choice.
        while (!group.empty()) {
            size_t pick = 0;  // position within `group`
            if (group.size() > 1) {
                std::vector<LegalAction> choices;
                for (size_t gi : group) {
                    LegalAction la(PASS_PRIORITY, pending[gi].source, pending[gi].label);
                    la.category = ActionCategory::OTHER_CHOICE;
                    choices.push_back(la);
                }
                game_log("%s orders %zu simultaneous triggers (pick which goes on the stack next).\n",
                         player_name(owner).c_str(), group.size());
                pick = static_cast<size_t>(InputLogger::instance().get_input(choices));
            }
            PendingTrigger &pt = pending[group[pick]];
            if (pt.needs_target && has_legal_targets(pt.ab, orderer))
                select_target(pt.ab, orderer, pt.controller);
            orderer->push_ability_onto_stack(pt.ab, pt.controller);
            game_log("%s\n", pt.log_line.c_str());
            group.erase(group.begin() + static_cast<ptrdiff_t>(pick));
        }
    }
}
