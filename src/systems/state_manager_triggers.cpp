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
#include "../components/spell.h"
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

// Storm count (CR 702.40a): the number of OTHER spells cast before the storm spell this turn,
// counting spells cast by EITHER player. The per-player spells_cast_this_turn counters already
// include the storm spell itself (the cast is recorded before SPELL_CAST fires), so subtract one.
static size_t storm_count_this_turn(const Game &game) {
    size_t total = 0;
    for (Entity pe : {game.player_a_entity, game.player_b_entity})
        if (global_coordinator.entity_has_component<Player>(pe))
            total += global_coordinator.GetComponent<Player>(pe).spells_cast_this_turn;
    return total > 0 ? total - 1 : 0;
}

// 603.3b: place all simultaneously-triggered abilities on the stack in APNAP order — the active
// player's triggers first (so they resolve last), then the non-active player's. Each player
// orders their own group via a mandatory choice when more than one trigger is theirs.
static void place_triggers_apnap(Game &game, std::shared_ptr<Orderer> orderer,
                                 std::vector<PendingTrigger> &pending);

// Bind the triggering player (the event's PLAYER, e.g. the caster of the triggering spell)
// onto any ability in the tree that uses Defined$ TriggeredActivator (CR 603.x). The
// LoseLife/etc. effect lives in a DB$ subability under Execute$, so recurse into
// subabilities/charm_choices. Only abilities flagged defined_triggered_activator are touched.
static void bind_triggered_activator(Ability &ab, Entity activator_entity) {
    Zone::Ownership activator = (activator_entity == get_player_entity(Zone::PLAYER_A)) ? Zone::PLAYER_A
                              : (activator_entity == get_player_entity(Zone::PLAYER_B)) ? Zone::PLAYER_B
                                                                                        : Zone::UNKNOWN;
    if (ab.defined_triggered_activator) ab.triggered_activator = activator;
    for (auto &sub : ab.subabilities) bind_triggered_activator(sub, activator_entity);
    for (auto &c : ab.charm_choices) bind_triggered_activator(c, activator_entity);
}

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
                // Entity-watched "when THIS permanent leaves the battlefield" delayed trigger
                // (CR 603.6e, the earthbend return-tapped trigger): match only the watched
                // entity changing zone from the battlefield. Skip the phase/owner gating below,
                // which is for the phase-based delayed triggers.
                if (dt.fire_on_leave_battlefield) {
                    if (ev.GetType() != Events::CARD_CHANGED_ZONE) continue;
                    if (!ev.HasParam(Params::ENTITY)) continue;
                    if (ev.GetParam<Entity>(Params::ENTITY) != dt.watch_entity) continue;
                    if (ev.GetParam<Zone::ZoneValue>(Params::ORIGIN) != Zone::BATTLEFIELD) continue;
                    matched = true;
                    break;
                }
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

    // Floating triggered abilities (CR 603.7e): transient until-end-of-turn triggers created by a
    // DB$ Effect | Triggers$ SVar (Forth Eorlingas!). They have no source permanent, so they are
    // scanned here against the drained events rather than by the battlefield loop. Currently the
    // one supported floating trigger is "Mode$ DamageAll | ValidSource$ Creature.YouCtrl |
    // ValidTarget$ Player | CombatDamage$ True" — fires once per combat-damage batch when one or
    // more creatures the trigger's controller controls deal combat damage to one or more players.
    for (const auto &ft : game.floating_triggers) {
        // Mode$ Attacks hosted on a command-zone Effect (Tamiyo, Seasoned Scholar's +2:
        // "whenever a creature an opponent controls attacks you or a planeswalker you control, it
        // gets -1/-0"). Fires once per matching attacker (CREATURE_ATTACKED), binding the attacker
        // as the effect's (Pump) target. The Attacked$ You,Planeswalker.YouCtrl clause is satisfied
        // whenever an opponent's creature attacks, in the two-player engine (the only defender it
        // can have is this effect's controller or their planeswalker).
        if (ft.trigger_on == Events::CREATURE_ATTACKED) {
            for (const auto &ev : events) {
                if (ev.GetType() != Events::CREATURE_ATTACKED) continue;
                if (!ev.HasParam(Params::ENTITY)) continue;
                Entity attacker = ev.GetParam<Entity>(Params::ENTITY);
                // ValidCard$ Creature.OppCtrl — the attacker is controlled by an opponent of the
                // effect's controller.
                if (ft.trigger_attacker_opp_ctrl &&
                    source_controller(attacker) == ft.controller) continue;
                Ability trigger_ab = ft;
                trigger_ab.controller = ft.controller;
                // Defined$ TriggeredAttacker(LKICopy) — bind the attacker as the pump's target.
                if (trigger_ab.defined_triggered_attacker_lki) trigger_ab.target = attacker;
                PendingTrigger pt;
                pt.ab = trigger_ab;
                pt.controller = ft.controller;
                pt.source = 0;
                pt.label = "Floating trigger (" + trigger_ab.category + ")";
                pt.log_line = "A floating triggered ability triggers.";
                pt.needs_target = false;
                pending.push_back(pt);
            }
            continue;
        }
        if (ft.trigger_on != Events::COMBAT_DAMAGE_TO_PLAYER) continue;
        bool matched = false;
        for (const auto &ev : events) {
            if (ev.GetType() != Events::COMBAT_DAMAGE_TO_PLAYER) continue;
            if (ft.trigger_damage_source_youctrl) {
                if (!ev.HasParam(Params::ENTITY)) continue;
                Entity dmg_src = ev.GetParam<Entity>(Params::ENTITY);
                if (source_controller(dmg_src) != ft.controller) continue;
            }
            matched = true;
            break;
        }
        if (!matched) continue;
        Ability trigger_ab = ft;
        trigger_ab.controller = ft.controller;
        PendingTrigger pt;
        pt.ab = trigger_ab;
        pt.controller = ft.controller;
        pt.source = 0;
        pt.label = "Floating trigger (" + trigger_ab.category + ")";
        pt.log_line = "A floating triggered ability triggers.";
        pt.needs_target = false;
        pending.push_back(pt);
    }

    // The monarch's inherent sourceless triggered abilities (CR 725.2): "At the beginning of the
    // monarch's end step, that player draws a card" and "Whenever a creature deals combat damage to
    // the monarch, its controller becomes the monarch." These have no source object and are not
    // card abilities, so they are produced here directly from the drained events against
    // game.monarch_entity. General over any monarch card (Forth Eorlingas! and future ones).
    if (game.monarch_entity != MAX_ENTITIES) {
        Zone::Ownership monarch_ctrl = (game.monarch_entity == game.player_a_entity)
                                       ? Zone::PLAYER_A : Zone::PLAYER_B;
        for (const auto &ev : events) {
            // End-step draw: the monarch draws an extra card at the beginning of their end step.
            if (ev.GetType() == Events::END_STEP_BEGAN && ev.HasParam(Params::PLAYER) &&
                ev.GetParam<Entity>(Params::PLAYER) == game.monarch_entity) {
                Ability draw_ab;
                draw_ab.ability_type = Ability::TRIGGERED;
                draw_ab.category = "Draw";
                draw_ab.amount = 1;
                draw_ab.controller = monarch_ctrl;
                draw_ab.target = game.monarch_entity;  // effect_draw reads target player when set
                PendingTrigger pt;
                pt.ab = draw_ab;
                pt.controller = monarch_ctrl;
                pt.source = 0;
                pt.label = "Monarch (end-step draw)";
                pt.log_line = "The monarch draws a card at the beginning of their end step.";
                pt.needs_target = false;
                pending.push_back(pt);
            }
            // Steal: a creature dealing combat damage to the monarch makes its controller the
            // monarch. The damaged player is the event's PLAYER; the new monarch is the controller
            // of the damaging creature (the event's ENTITY).
            if (ev.GetType() == Events::COMBAT_DAMAGE_TO_PLAYER && ev.HasParam(Params::PLAYER) &&
                ev.GetParam<Entity>(Params::PLAYER) == game.monarch_entity &&
                ev.HasParam(Params::ENTITY)) {
                Zone::Ownership attacker_ctrl = source_controller(ev.GetParam<Entity>(Params::ENTITY));
                if (attacker_ctrl == Zone::UNKNOWN) continue;
                Entity new_monarch = get_player_entity(attacker_ctrl);
                if (new_monarch == game.monarch_entity) continue;  // already the monarch
                Ability steal_ab;
                steal_ab.ability_type = Ability::TRIGGERED;
                steal_ab.category = "BecomeMonarch";
                steal_ab.controller = attacker_ctrl;
                PendingTrigger pt;
                pt.ab = steal_ab;
                pt.controller = attacker_ctrl;
                pt.source = 0;
                pt.label = "Monarch (steal on combat damage)";
                pt.log_line = "A creature dealt combat damage to the monarch.";
                pt.needs_target = false;
                pending.push_back(pt);
            }
        }
    }

    // Saga chapter abilities (CR 714.3): a SAGA_CHAPTER event is emitted by the lore-counter
    // machinery (saga.cpp) when a Saga's lore counters reach chapter N. The chapter ability lives on
    // the Saga's CardData::saga_chapters (parsed from K:Chapter), not as a T: trigger, so it is
    // produced here directly and queued for APNAP placement / target selection like any other
    // triggered ability. The chapter ability is marked is_saga_chapter so the 714.4 sacrifice gate
    // (Permanent::saga_chapters_in_flight) is released when it leaves the stack.
    for (const auto &ev : events) {
        if (ev.GetType() != Events::SAGA_CHAPTER) continue;
        if (!ev.HasParam(Params::ENTITY) || !ev.HasParam(Params::AMOUNT)) continue;
        Entity saga = ev.GetParam<Entity>(Params::ENTITY);
        int chapter = static_cast<int>(ev.GetParam<uint32_t>(Params::AMOUNT));
        if (!global_coordinator.entity_has_component<CardData>(saga)) continue;
        if (!global_coordinator.entity_has_component<Permanent>(saga)) continue;
        const auto &cd = global_coordinator.GetComponent<CardData>(saga);
        if (chapter < 1 || chapter > static_cast<int>(cd.saga_chapters.size())) continue;
        Zone::Ownership ctrl = global_coordinator.GetComponent<Permanent>(saga).controller;

        Ability trigger_ab = cd.saga_chapters[static_cast<size_t>(chapter - 1)];
        trigger_ab.ability_type = Ability::TRIGGERED;
        trigger_ab.source = saga;
        trigger_ab.controller = ctrl;
        trigger_ab.is_saga_chapter = true;

        PendingTrigger pt;
        pt.ab = trigger_ab;
        pt.controller = ctrl;
        pt.source = saga;
        pt.label = entity_name(saga) + " (chapter " + std::to_string(chapter) + ")";
        pt.log_line = entity_name(saga) + " chapter " + std::to_string(chapter) + " triggers.";
        pt.needs_target = (trigger_ab.valid_tgts != "N_A" && trigger_ab.target == 0);
        pending.push_back(pt);
    }

    if (!events.empty()) {
    for (auto entity : mEntities) {
        if (!is_battlefield_permanent(entity)) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(entity);

        // Gather triggered abilities from all sources:
        // CardData/Token for innate abilities, Permanent for keyword-granted abilities.
        // A transformed DFC functions with its ACTIVE (back) face's abilities (CR 712.4): its
        // triggered abilities are the back face's, and the front face's are suppressed. So pull the
        // innate abilities from whichever face is up — this is the single place front/back trigger
        // selection happens (the per-ability "if (perm.transformed) continue" front-face skip is
        // therefore unnecessary, and would wrongly suppress the back face's own triggers).
        std::vector<const std::vector<Ability>*> ab_sources;
        if (global_coordinator.entity_has_component<CardData>(entity)) {
            const CardData &cd = global_coordinator.GetComponent<CardData>(entity);
            if (perm.transformed && cd.backside)
                ab_sources.push_back(&cd.backside->abilities);
            else
                ab_sources.push_back(&cd.abilities);
        }
        if (global_coordinator.entity_has_component<Token>(entity))
            ab_sources.push_back(&global_coordinator.GetComponent<Token>(entity).abilities);
        ab_sources.push_back(&perm.abilities);
        if (ab_sources.empty()) continue;

        const std::string ent_name = entity_name(entity);

        for (const auto &ev : events) {
            for (const auto *src : ab_sources) {
            for (const auto &ab : *src) {
                if (ab.ability_type != Ability::TRIGGERED) continue;
                // Static$ True mana-additional triggers (Badgermole Cub's TapsForMana) never go
                // on the stack — they resolve immediately inside the mana system (CR 605.1a).
                if (ab.trigger_taps_for_mana_static) continue;
                // A graveyard-functioning trigger (TriggerZones$ Graveyard, e.g. Arclight
                // Phoenix's begin-combat return) functions ONLY while its source is in the
                // graveyard — TriggerZones overrides the default battlefield functioning
                // (CR 113.6). It is handled by the dedicated graveyard scan below; firing it
                // from the battlefield would place a spurious (no-op) trigger on the stack.
                if (ab.trigger_from_graveyard) continue;
                if (ab.trigger_on == 0 || ab.trigger_on != ev.GetType()) continue;
                // "another" check: skip if the event entity is the triggering permanent itself
                if (ab.trigger_self_excluded && ev.HasParam(Params::ENTITY) &&
                    ev.GetParam<Entity>(Params::ENTITY) == entity) continue;
                // Card.Self: only fire when the event entity is the triggering permanent itself.
                // BECAME_TARGET is exempt: there ENTITY is the targeting object (not the source),
                // and ValidTarget$ Card.Self is matched against the TARGET param in the dedicated
                // BECAME_TARGET block below instead.
                if (ab.trigger_only_self && ev.GetType() != Events::BECAME_TARGET &&
                    ev.HasParam(Params::ENTITY) &&
                    ev.GetParam<Entity>(Params::ENTITY) != entity) continue;
                // "If you cast it" (ValidCard$ Card.wasCastByYou): a self ETB trigger that only
                // fires when its source permanent entered the battlefield by being cast (CR
                // 614.12; The One Ring's protection grant). perm is the source (trigger_only_self).
                if (ab.trigger_requires_entered_by_cast && !perm.entered_by_cast) continue;
                // Evoke self-sacrifice only fires when this permanent was cast via evoke
                if (ab.is_evoke_sacrifice && !perm.evoked) continue;
                // Offspring token copy only fires when this permanent was cast with offspring
                if (ab.is_offspring_token && !perm.entered_with_offspring) continue;
                // (Front/back face selection is done once when ab_sources is built above, so a
                // transformed permanent already only sees its active face's triggers here.)
                // ValidPlayer$ You: only fire when the event's player matches the permanent's
                // controller. BECAME_TARGET is exempt, like trigger_only_self above: there the
                // PLAYER param is the targeting spell's controller (the OPPONENT, typically), not
                // a "you" reference, so gating on it would wrongly suppress a becomes-target
                // trigger authored with ValidPlayer$ You. Such a trigger's own ValidSource$/
                // ValidTarget$ clauses constrain it in the dedicated BECAME_TARGET block below.
                if (ab.trigger_valid_player_is_controller && ev.GetType() != Events::BECAME_TARGET &&
                    ev.HasParam(Params::PLAYER)) {
                    Entity event_player = ev.GetParam<Entity>(Params::PLAYER);
                    Entity ctrl_entity = get_player_entity(perm.controller);
                    if (event_player != ctrl_entity) continue;
                }
                // DisableTriggers check (Doorkeeper Thrull): suppress ETB triggers caused by matching card types
                if (ev.GetType() == Events::CARD_CHANGED_ZONE &&
                    ev.GetParam<Zone::ZoneValue>(Params::DESTINATION) == Zone::BATTLEFIELD) {
                    Entity entering = ev.HasParam(Params::ENTITY) ? ev.GetParam<Entity>(Params::ENTITY) : 0;
                    if (rules_mod::etb_triggers_suppressed(entering)) continue;
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
                    // ValidCard$ Card.nonCreature filter on a counted SpellCast (The Fantasticar):
                    // the triggering spell must NOT be a creature.
                    if (ab.trigger_valid_card_non_creature && ev.HasParam(Params::ENTITY)) {
                        Entity ev_card = ev.GetParam<Entity>(Params::ENTITY);
                        bool is_creature = global_coordinator.entity_has_component<Token>(ev_card);
                        if (!is_creature && global_coordinator.entity_has_component<CardData>(ev_card))
                            is_creature = is_creature_card(global_coordinator.GetComponent<CardData>(ev_card));
                        if (is_creature) continue;
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
                    // ValidCard$ Artifact.* filter (Kappa Cannoneer: another artifact entering).
                    // Token artifacts have their types on the Token component, not CardData.
                    if (ab.trigger_valid_card_is_artifact && ev.HasParam(Params::ENTITY)) {
                        Entity ev_card = ev.GetParam<Entity>(Params::ENTITY);
                        bool is_art = false;
                        if (global_coordinator.entity_has_component<CardData>(ev_card))
                            is_art = card_has_type(global_coordinator.GetComponent<CardData>(ev_card),
                                                   "Artifact");
                        if (!is_art && global_coordinator.entity_has_component<Token>(ev_card)) {
                            for (auto &t : global_coordinator.GetComponent<Token>(ev_card).types)
                                if (t.name == "Artifact") { is_art = true; break; }
                        }
                        if (!is_art) continue;
                    }
                    // ValidCard$ ...+!token — only real cards match, not tokens (CR 110.1).
                    // Token permanents carry a Token component (their identity lives there
                    // rather than on CardData). Moonshadow's "permanent CARDS put into your
                    // graveyard" must ignore a dying token you own.
                    if (ab.trigger_valid_card_non_token && ev.HasParam(Params::ENTITY)) {
                        Entity ev_card = ev.GetParam<Entity>(Params::ENTITY);
                        if (global_coordinator.entity_has_component<Token>(ev_card)) continue;
                    }
                    // ValidCard$ Permanent — only permanent card types match (CR 110.4a);
                    // an instant/sorcery moving to the graveyard must not trigger (Moonshadow:
                    // "permanent cards put into your graveyard").
                    if (ab.trigger_valid_card_is_permanent && ev.HasParam(Params::ENTITY)) {
                        Entity ev_card = ev.GetParam<Entity>(Params::ENTITY);
                        if (!global_coordinator.entity_has_component<CardData>(ev_card)) continue;
                        if (!is_permanent_card(global_coordinator.GetComponent<CardData>(ev_card)))
                            continue;
                    }
                    // ValidCard(s)$ <Subtype> filter (Ajani: a Cat changing zone). Checked
                    // against the changing card's CardData or Token subtypes.
                    if (!ab.trigger_valid_card_subtype.empty() && ev.HasParam(Params::ENTITY)) {
                        Entity ev_card = ev.GetParam<Entity>(Params::ENTITY);
                        bool has_sub = false;
                        if (global_coordinator.entity_has_component<CardData>(ev_card)) {
                            for (auto &t : global_coordinator.GetComponent<CardData>(ev_card).types)
                                if (t.name == ab.trigger_valid_card_subtype) { has_sub = true; break; }
                        }
                        if (!has_sub && global_coordinator.entity_has_component<Token>(ev_card)) {
                            for (auto &t : global_coordinator.GetComponent<Token>(ev_card).types)
                                if (t.name == ab.trigger_valid_card_subtype) { has_sub = true; break; }
                        }
                        // 603.10 look-back: a token that died has already ceased to exist, so
                        // fall back to its last-known type names captured at battlefield-leave.
                        if (!has_sub) {
                            auto it = game.lk_battlefield_types.find(ev_card);
                            if (it != game.lk_battlefield_types.end())
                                for (auto &n : it->second)
                                    if (n == ab.trigger_valid_card_subtype) { has_sub = true; break; }
                        }
                        if (!has_sub) continue;
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
                    // Number$ N (Tamiyo, Inquisitive Student: "your THIRD card in a turn") — fire
                    // only when this draw is the drawer's Nth this turn. AMOUNT is the per-draw
                    // running ordinal stamped on the event (1-based), so exactly the Nth draw fires.
                    if (ab.trigger_draw_number_eq > 0) {
                        if (!ev.HasParam(Params::AMOUNT) ||
                            ev.GetParam<uint32_t>(Params::AMOUNT) != ab.trigger_draw_number_eq)
                            continue;
                    }
                }

                // Colorless filter (Glaring Fleshraker): the cast spell (SPELL_CAST) or the
                // entering card (CARD_CHANGED_ZONE) must be colorless (CR 105.2c). The card is
                // carried as Params::ENTITY on both event types.
                if (ab.trigger_valid_card_colorless && ev.HasParam(Params::ENTITY)) {
                    Entity ev_card = ev.GetParam<Entity>(Params::ENTITY);
                    if (!is_colorless_entity(ev_card)) continue;
                }

                // Spell count filter (Cori-Steel Cutter)
                if (ab.trigger_spell_count_eq > 0 && ev.HasParam(Params::PLAYER)) {
                    Entity ev_player = ev.GetParam<Entity>(Params::PLAYER);
                    if (!global_coordinator.entity_has_component<Player>(ev_player)) continue;
                    auto &pl = global_coordinator.GetComponent<Player>(ev_player);
                    // The Fantasticar counts only noncreature spells; Cori-Steel Cutter counts all.
                    size_t cast_count = ab.trigger_spell_count_noncreature
                                            ? pl.noncreature_spells_cast_this_turn
                                            : pl.spells_cast_this_turn;
                    if (cast_count != ab.trigger_spell_count_eq) continue;
                }

                // Dynamic mana-value filter on the cast spell (Chalice of the Void:
                // ValidCard$ Card.cmcEQY, Y = Count$CardCounters.CHARGE). Compare the cast
                // spell's mana value to the count resolved against this source permanent.
                if (!ab.trigger_cmc_expr.empty() && ev.GetType() == Events::SPELL_CAST) {
                    if (!ev.HasParam(Params::ENTITY)) continue;
                    Entity spell_e = ev.GetParam<Entity>(Params::ENTITY);
                    if (!global_coordinator.entity_has_component<CardData>(spell_e)) continue;
                    int spell_mv = static_cast<int>(
                        global_coordinator.GetComponent<CardData>(spell_e).mana_cost.size());
                    int bound = evaluate_sa_svar(ab.trigger_cmc_expr, perm.controller, entity);
                    const std::string &op = ab.trigger_cmc_op;
                    bool ok = (op == "EQ") ? (spell_mv == bound)
                            : (op == "LE") ? (spell_mv <= bound)
                            : (op == "GE") ? (spell_mv >= bound)
                            : (op == "LT") ? (spell_mv <  bound)
                            : (op == "GT") ? (spell_mv >  bound)
                            : (op == "NE") ? (spell_mv != bound)
                            : (spell_mv == bound);
                    if (!ok) continue;
                }

                // BECAME_TARGET filters (Reality Smasher): the trigger fires only for the
                // permanent that became a target (TARGET == this source, i.e. ValidTarget$
                // Card.Self, already enforced by trigger_only_self against ENTITY below is NOT
                // applicable here because ENTITY is the targeting object, not the targeted
                // permanent — so the self check is done explicitly against TARGET) and only when
                // the targeting object is a spell controlled by an opponent (ValidSource$
                // Spell.OppCtrl).
                if (ev.GetType() == Events::BECAME_TARGET) {
                    // ValidTarget$ Card.Self: the permanent that became a target must be this one.
                    Entity targeted = ev.HasParam(Params::TARGET) ? ev.GetParam<Entity>(Params::TARGET) : 0;
                    if (ab.trigger_only_self && targeted != entity) continue;
                    Entity targeting = ev.HasParam(Params::ENTITY) ? ev.GetParam<Entity>(Params::ENTITY) : 0;
                    // ValidSource$ Spell — the targeting object must be a spell on the stack.
                    if (ab.trigger_source_must_be_spell &&
                        !global_coordinator.entity_has_component<Spell>(targeting)) continue;
                    // ValidSource$ ...OppCtrl — controlled by an opponent of this source's controller.
                    if (ab.trigger_source_opp_ctrl && ev.HasParam(Params::PLAYER)) {
                        Entity src_player = ev.GetParam<Entity>(Params::PLAYER);
                        if (src_player == get_player_entity(perm.controller)) continue;
                    }
                }

                // Prepare the triggered ability and queue it; APNAP placement (and any target
                // selection) happens after the full scan, in place_triggers_apnap().
                Ability trigger_ab = ab;
                trigger_ab.source = entity;
                trigger_ab.controller = perm.controller;
                // Defined$ TriggeredSourceSA — the Counter effect acts on the spell that targeted
                // this permanent. Bind it as the ability's target from the event's ENTITY (the
                // targeting object). UnlessPayer$ TriggeredSourceSAController binds the payer of the
                // unless-cost to that spell's controller (the opponent), captured from PLAYER.
                if (ev.GetType() == Events::BECAME_TARGET) {
                    if (trigger_ab.defined_triggered_source_sa && ev.HasParam(Params::ENTITY))
                        trigger_ab.target = ev.GetParam<Entity>(Params::ENTITY);
                    if (trigger_ab.unless_payer_is_triggered_source_sa_ctrl && ev.HasParam(Params::PLAYER)) {
                        Entity src_player = ev.GetParam<Entity>(Params::PLAYER);
                        trigger_ab.unless_payer = (src_player == get_player_entity(Zone::PLAYER_A))
                                                  ? Zone::PLAYER_A : Zone::PLAYER_B;
                    }
                }
                // Defined$ TriggeredSpellAbility — the effect (Counter) acts on the spell that
                // fired this trigger. Capture it from the event as the ability's target.
                if (trigger_ab.defined_triggered_spell && ev.HasParam(Params::ENTITY))
                    trigger_ab.target = ev.GetParam<Entity>(Params::ENTITY);
                // Defined$ TriggeredActivator — bind the player who caused the trigger (the
                // event's PLAYER, e.g. the caster of the noncreature spell) onto this ability
                // and its subabilities, so the effect (LoseLife etc.) resolves against them.
                if (ev.HasParam(Params::PLAYER))
                    bind_triggered_activator(trigger_ab, ev.GetParam<Entity>(Params::PLAYER));
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

    // Self zone-change triggers that fire as the source itself moves (CR 603.6b / 603.10): a
    // triggered ability that watches its own source change zones (Flagstones of Trokair: "When
    // CARDNAME is put into a graveyard from the battlefield, ..."; Emrakul, the Aeons Torn: "When
    // CARDNAME is put into a graveyard from ANYWHERE, ...") cannot be caught by the battlefield
    // scan above — the moved card is no longer a battlefield permanent (and for a non-battlefield
    // origin, e.g. a discard from hand or a mill from library, it never was). The game instead
    // looks back at the moment just before the event (603.10) to decide whether the ability
    // triggered. We do that here: for each CARD_CHANGED_ZONE event, re-scan the changing card's
    // own CardData abilities for a Card.Self trigger whose origin/destination filter matches, and
    // queue it. The per-ability origin filter (below) keeps an Origin$ Battlefield "dies" trigger
    // from firing on a hand/library departure, while an Origin$ Any trigger (filter unset) fires
    // from any zone. CardData (and the card's persisted last controller / owner in its Zone)
    // survives the move, so the source is fully recoverable.
    for (const auto &ev : events) {
        if (ev.GetType() != Events::CARD_CHANGED_ZONE) continue;
        if (!ev.HasParam(Params::ENTITY)) continue;
        Zone::ZoneValue ev_origin = ev.GetParam<Zone::ZoneValue>(Params::ORIGIN);
        Zone::ZoneValue ev_dest = ev.GetParam<Zone::ZoneValue>(Params::DESTINATION);
        // Enters-the-battlefield self-triggers (destination battlefield) are handled by the
        // battlefield scan above (the source IS a battlefield permanent once it arrives), so
        // exclude them here to avoid double-firing. This block covers self moves to any OTHER
        // zone (graveyard, exile, hand, library) from any origin.
        if (ev_dest == Zone::BATTLEFIELD) continue;
        Entity entity = ev.GetParam<Entity>(Params::ENTITY);
        if (!global_coordinator.entity_has_component<CardData>(entity)) continue;
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        // The source's controller as it left the battlefield persists on its Zone component
        // (only overwritten when a card enters the battlefield), so it still names the last
        // controller; fall back to the owner if it was never set.
        auto &z = global_coordinator.GetComponent<Zone>(entity);
        Zone::Ownership ctrl = (z.controller != Zone::UNKNOWN) ? z.controller : z.owner;
        const std::string ent_name = entity_name(entity);
        for (const auto &ab : global_coordinator.GetComponent<CardData>(entity).abilities) {
            if (ab.ability_type != Ability::TRIGGERED) continue;
            if (ab.trigger_on != Events::CARD_CHANGED_ZONE) continue;
            if (!ab.trigger_only_self) continue;  // Card.Self — only the source's own move
            if (ab.trigger_zone_origin >= 0 &&
                ev_origin != static_cast<Zone::ZoneValue>(ab.trigger_zone_origin)) continue;
            if (ab.trigger_zone_destination >= 0 &&
                ev_dest != static_cast<Zone::ZoneValue>(ab.trigger_zone_destination)) continue;

            Ability trigger_ab = ab;
            trigger_ab.source = entity;
            trigger_ab.controller = ctrl;

            // A leaves-the-battlefield ability that operates on the cards the source had exiled
            // (Skyclave Apparition's TrigToken sizes/owns the Illusion by the exiled card, and
            // gates on it still being exiled): the source has already left the battlefield, so its
            // exiled_with list lives in the last-known-info snapshot captured at departure. Carry
            // it onto the trigger so resolve() can restore the remembered set (CR 608.2h).
            {
                auto lki_it = game.last_known_info.find(entity);
                if (lki_it != game.last_known_info.end() && !lki_it->second.exiled_with.empty())
                    trigger_ab.restore_remembered_exiled_with = lki_it->second.exiled_with;
            }

            PendingTrigger pt;
            pt.ab = trigger_ab;
            pt.controller = ctrl;
            pt.source = entity;
            pt.label = trigger_label(ent_name, trigger_ab);
            pt.log_line = ent_name + " triggered";
            pt.needs_target = (trigger_ab.valid_tgts != "N_A" && trigger_ab.target == 0);
            pending.push_back(pt);
        }
    }

    // Self-cast triggered abilities (CR 702.33e/f, Wastescape Battlemage): "When you cast this
    // spell, [if it was kicked with its [N] kicker,] ..." fires on SPELL_CAST of the spell
    // ITSELF, while it sits on the stack — not from the battlefield — so the battlefield scan
    // above never sees it. The source is the cast spell entity (the event's ENTITY); read its
    // own CardData abilities for a trigger_only_self SPELL_CAST trigger and, when the trigger
    // names a kicker (trigger_kicked_index > 0), gate it on that kicker having been paid
    // (Spell::kicked). General over any self-cast / kicked-N SpellCast trigger.
    for (const auto &ev : events) {
        if (ev.GetType() != Events::SPELL_CAST) continue;
        if (!ev.HasParam(Params::ENTITY)) continue;
        Entity spell_e = ev.GetParam<Entity>(Params::ENTITY);
        if (!global_coordinator.entity_has_component<CardData>(spell_e)) continue;
        if (!global_coordinator.entity_has_component<Spell>(spell_e)) continue;
        const auto &spell = global_coordinator.GetComponent<Spell>(spell_e);
        Zone::Ownership ctrl = spell.caster;
        const std::string ent_name = entity_name(spell_e);
        for (const auto &ab : global_coordinator.GetComponent<CardData>(spell_e).abilities) {
            if (ab.ability_type != Ability::TRIGGERED) continue;
            if (ab.trigger_on != Events::SPELL_CAST) continue;
            if (!ab.trigger_only_self) continue;  // ValidCard$ Card.Self — only the cast spell itself
            // Linked kicker condition (CR 702.33f): fires only if the named kicker was paid.
            if (ab.trigger_kicked_index > 0) {
                size_t idx = static_cast<size_t>(ab.trigger_kicked_index - 1);
                if (idx >= spell.kicked.size() || !spell.kicked[idx]) continue;
            }

            Ability trigger_ab = ab;
            trigger_ab.source = spell_e;
            trigger_ab.controller = ctrl;

            // Storm (CR 702.40a): lock in the copy count = number of OTHER spells cast before
            // this one this turn, by EITHER player. The per-player cast counters were already
            // bumped for this storm spell before SPELL_CAST fired (action_processor records the
            // cast first), so the both-player total minus this spell is the storm count. Snapshot
            // it now, at trigger-fire time — spells cast in RESPONSE to the storm trigger come
            // after this spell and must not inflate the count.
            if (trigger_ab.category == "Storm")
                trigger_ab.amount = storm_count_this_turn(game);

            PendingTrigger pt;
            pt.ab = trigger_ab;
            pt.controller = ctrl;
            pt.source = spell_e;
            pt.label = trigger_label(ent_name, trigger_ab);
            pt.log_line = ent_name + " triggered";
            pt.needs_target = (trigger_ab.valid_tgts != "N_A" && trigger_ab.target == 0);
            pending.push_back(pt);
        }
    }

    // Graveyard-functioning triggered abilities (CR 113.6 / 603.6): a card whose triggered
    // ability has TriggerZones$ Graveyard (e.g. Arclight Phoenix's "at the beginning of
    // combat on your turn, ... return ~ from your graveyard to the battlefield") functions
    // while in its owner's graveyard, so it is never scanned by the battlefield loop above.
    // Scan the graveyards for these abilities. Only phase-type triggers (PLAYER param) are
    // supported here, which covers the current card; ValidPlayer$ You and the 603.4
    // intervening-if are honoured exactly as on the battlefield.
    for (auto entity : mEntities) {
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(entity);
        if (z.location != Zone::GRAVEYARD) continue;
        if (!global_coordinator.entity_has_component<CardData>(entity)) continue;
        Zone::Ownership owner = z.owner;
        const std::string ent_name = entity_name(entity);
        for (const auto &ev : events) {
            for (const auto &ab : global_coordinator.GetComponent<CardData>(entity).abilities) {
                if (ab.ability_type != Ability::TRIGGERED) continue;
                if (!ab.trigger_from_graveyard) continue;
                if (ab.trigger_on == 0 || ab.trigger_on != ev.GetType()) continue;
                // ValidPlayer$ You: the source's owner must be the event's player (the
                // active player whose combat / phase began).
                if (ab.trigger_valid_player_is_controller && ev.HasParam(Params::PLAYER)) {
                    if (ev.GetParam<Entity>(Params::PLAYER) != get_player_entity(owner)) continue;
                }

                Ability trigger_ab = ab;
                trigger_ab.source = entity;
                trigger_ab.controller = owner;

                // 603.4 intervening-if: a trigger whose "if" condition is false right now
                // does not go on the stack at all (re-checked again on resolution).
                if (trigger_ab.intervening_if &&
                    !evaluate_present_condition(trigger_ab, owner, orderer))
                    continue;

                PendingTrigger pt;
                pt.ab = trigger_ab;
                pt.controller = owner;
                pt.source = entity;
                pt.label = trigger_label(ent_name, trigger_ab);
                pt.log_line = ent_name + " triggered";
                pt.needs_target = (trigger_ab.valid_tgts != "N_A" && trigger_ab.target == 0);
                pending.push_back(pt);
            }
        }
    }
    }

    place_triggers_apnap(game, orderer, pending);

    // Last-known type snapshots are only valid for this batch of leave-the-battlefield
    // events; clear them so a later, unrelated trigger can't match a stale entity id.
    game.lk_battlefield_types.clear();
}

static std::string trigger_label(const std::string &name, const Ability &ab) {
    if (ab.is_evoke_sacrifice) return name + " (evoke: sacrifice)";
    if (ab.is_offspring_token) return name + " (offspring: token copy)";
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
                    la.category = ActionCategory::ORDER_TRIGGERS;
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
