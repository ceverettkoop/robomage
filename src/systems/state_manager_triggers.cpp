#include "../coverage.h"
#include "state_manager.h"
#include "state_manager_internal.h"
#include "rules_modifying.h"

#include <algorithm>
#include <cstddef>
#include <set>
#include <string>
#include <vector>

#include "../action_processor.h"
#include "../announce.h"
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
#include "../components/spell.h"
#include "../components/token.h"
#include "../components/types.h"
#include "../type_constants.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../cli_output.h"
#include "../error.h"
#include "../game_driver.h"
#include "../queries/activation.h"
#include "../queries/battlefield.h"
#include "../queries/characteristics.h"
#include "../queries/counters.h"
#include "../queries/filters.h"
#include "../queries/lki.h"
#include "../queries/players.h"
#include "../queries/types.h"
#include "../queries/zones.h"
#include "../input_logger.h"
#include "../mana_system.h"
#include "../svar_eval.h"
#include "../svar_eval.h"
#include "../systems/stack_manager.h"
#include "orderer.h"
#include "../resolution.h"
#include "../targeting.h"
#include "../trigger_modes.h"

// A triggered ability that has fired and is waiting to be put on the stack. We collect every
// trigger from the current batch of events first, then place them all in APNAP order (603.3b);
// the old code pushed each trigger the instant it was found (raw entity-ID order), which is not
// APNAP and gave no player a chance to order their own simultaneous triggers. The persisted
// struct lives in resolution_frame.h (Game::pending.trigger_placement holds the flattened queue while
// a placement decision is parked); this alias keeps the collection code unchanged.
using PendingTrigger = PendingTriggerRT;

namespace {
// TargetAsker for trigger-placement target selection (CR 603.3d): arms the
// pick as a loop-top pending decision with the family tag TRIGGER_PLACE and
// the trigger's source as the pending-decision context — the same
// PendingDecisionScope(ability.source) the blocking select_target held.
// Priority is already seated at the trigger's controller (the placement loop
// seats it per group), so arm/consume leave the seat unchanged.
class TriggerPlaceTargetAsker final : public TargetAsker {
    public:
        int ask(const std::vector<LegalAction> &menu, Entity decision_source) override {
            PendingQuery &pq = cur_game.pending.query;
            if (pq.active) {
                // Consume the latched answer for THIS ask. The re-entered
                // machine must have rebuilt the identical menu (pure builds,
                // nothing runs between suspend and resume).
                if (!pq.answered || pq.tag != PendingQuery::TRIGGER_PLACE)
                    fatal_error("trigger target resume without a latched TRIGGER_PLACE answer");
                if (pq.menu.size() != menu.size())
                    fatal_error("trigger target menu size changed between arm and resume (" +
                                std::to_string(pq.menu.size()) + " vs " +
                                std::to_string(menu.size()) + ")");
                int answer = pq.answer;
                cur_game.priority.player_a_has_priority = pq.prev_priority;
                pq = PendingQuery{};
                return answer;
            }
            // Suspend: park the query for the main loop to emit.
            pq = PendingQuery{};
            pq.tag = PendingQuery::TRIGGER_PLACE;
            pq.active = true;
            pq.menu = menu;
            pq.chooser_is_a = cur_game.priority.player_a_has_priority;
            pq.decision_source = decision_source;
            pq.prev_priority = cur_game.priority.player_a_has_priority;
            return -1;
        }
        bool resuming() const override {
            return cur_game.pending.query.active && cur_game.pending.query.answered;
        }
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

// Announce the modes and targets of the trigger at the queue front as it is put on the stack
// (CR 603.3c/d, run_announce) — asking through a parked TRIGGER_PLACE query, or inline when the
// placement can't suspend. SUSPENDED means a pick parked; re-enter with the same tp.
static AnnounceStatus announce_trigger(PendingTrigger &pt, TriggerPlacementRT &tp,
                                       bool suspendable, std::shared_ptr<Orderer> orderer);

// Where a trigger scan finds the abilities it reads (CR 113.6).
enum class TriggerScan : uint8_t {
    BATTLEFIELD,    // a permanent on the battlefield
    DEPARTED,       // a source that left with the objects it watches: the look-back (CR 603.10a)
    GRAVEYARD,      // a card in its owner's graveyard, for TriggerZones$ Graveyard
    EFFECT_OBJECT,  // an effect object, which is in no zone and has no source object (CR 603.7e)
};

// What holds the abilities a trigger scan reads. `perm` is the source's Permanent while it is on
// the battlefield; `object` is set for an effect object, and `entity` is then 0.
struct TriggerHost {
    TriggerScan scan = TriggerScan::BATTLEFIELD;
    Entity entity = 0;
    Zone::Ownership controller = Zone::UNKNOWN;
    const Permanent *perm = nullptr;
    std::string name;
    const EffectObject *object = nullptr;
};

// Queue every ability of `host` in `ab_defs` that one of `events` triggers (CR 603.2). An effect
// object's `ab_defs` are its triggers' definitions, in order.
static void match_event_triggers(const TriggerHost &host, const std::vector<const AbilityDef *> &ab_defs,
                                 const std::vector<Event> &events, std::shared_ptr<Orderer> orderer,
                                 std::vector<PendingTrigger> &pending);

// `trigger_ab` has triggered: a Static$ True ability resolves now, off the stack (it never goes on
// it), and any other joins `pending` for APNAP placement.
static void fire_trigger(Ability &trigger_ab, Zone::Ownership controller, Entity source, const std::string &label,
                         const std::string &log_line, std::shared_ptr<Orderer> orderer,
                         std::vector<PendingTrigger> &pending);

// What the object's floating triggers are shown as, having no source to name them by.
static std::string floating_trigger_label(const Ability &ab);

// Leaves-the-battlefield abilities of the cards that left the battlefield in this batch of events
// (CR 603.10a): each looks back to just before the event, so a watcher that left along with the
// objects it watches (Super Shredder's "another permanent leaves the battlefield" when both die
// in one state-based-action check) still triggers, with its last-known abilities and controller.
// Tokens keep no abilities to look back at once off the battlefield.
static bool is_self_etb_event(const Event &ev, Entity entity);
static ObjectRef moved_object(const Event &ev);
static void match_departed_watcher_triggers(const std::vector<Event> &events,
                                            std::shared_ptr<Orderer> orderer,
                                            std::vector<PendingTrigger> &pending);

// Bind the triggering player (the event's PLAYER, e.g. the caster of the triggering spell)
// onto any ability in the tree that uses Defined$ TriggeredActivator (CR 603.x). The
// LoseLife/etc. effect lives in a DB$ subability under Execute$, so recurse into
// subabilities/charm_choices. Only abilities naming Defined$ TriggeredActivator are touched.
// The object a CARD_CHANGED_ZONE event moved, as it was before the move.
static ObjectRef moved_object(const Event &ev) {
    return ObjectRef::of_object(
        ev.GetParam<Entity>(Params::ENTITY),
        ev.HasParam(Params::OBJECT_GEN) ? ev.GetParam<uint64_t>(Params::OBJECT_GEN) : 0);
}

// Is `ev` the object `entity` itself entering the battlefield (an ETB event for its own
// enters-the-battlefield triggered abilities)?
static bool is_self_etb_event(const Event &ev, Entity entity) {
    return ev.GetType() == Events::CARD_CHANGED_ZONE && ev.HasParam(Params::ENTITY) &&
           ev.GetParam<Entity>(Params::ENTITY) == entity &&
           ev.GetParam<Zone::ZoneValue>(Params::DESTINATION) == Zone::BATTLEFIELD;
}

static void bind_triggered_activator(Ability &ab, Entity activator_entity) {
    Zone::Ownership activator = seat_of_player(activator_entity);
    if (ab.def->defined.is(ObjectSpec::TRIGGERED_ACTIVATOR)) ab.triggered_activator = activator;
    for (auto &sub : ab.subabilities) bind_triggered_activator(sub, activator_entity);
    for (auto &c : ab.charm_choices) bind_triggered_activator(c, activator_entity);
}

// Bind the triggering event's player (Defined$ TriggeredPlayer, e.g. the active player whose
// upkeep began — Roiling Vortex) onto any ability in the tree that reads it. Sibling of
// bind_triggered_activator; the DealDamage/etc. effect lives in a DB$ subability under Execute$,
// so recurse into subabilities/charm_choices. Only abilities naming Defined$ TriggeredPlayer
// are touched.
static void bind_triggered_player(Ability &ab, Entity player_entity) {
    Zone::Ownership who = seat_of_player(player_entity);
    if (ab.def->defined.is(ObjectSpec::TRIGGERED_PLAYER)) ab.triggered_player = who;
    for (auto &sub : ab.subabilities) bind_triggered_player(sub, player_entity);
    for (auto &c : ab.charm_choices) bind_triggered_player(c, player_entity);
}

// Drains all buffered events since the last call and records every ability they trigger (CR
// 603.2) in Game::waiting_triggers, to be put on the stack by place_waiting_triggers.
void StateManager::collect_triggered_abilities(Game &game, std::shared_ptr<Orderer> orderer) {
    // A suspended trigger placement owns the trigger flow: the main loop's
    // TRIGGER_PLACE dispatch (resume_trigger_placement) drives it to
    // completion, and this scan must not drain events or collect a second
    // batch underneath it. (Belt-and-braces — the main loop never reaches a
    // state_based_effects call while a pending query is parked.)
    if (game.pending.trigger_placement.active) return;
    auto events = global_coordinator.drain_pending_events();

    // Every ability that triggers off this batch of events is collected here, then recorded as
    // waiting to be put on the stack — nothing is pushed mid-scan.
    std::vector<PendingTrigger> pending;

    // Fire any delayed triggers that match current events
    {
        std::vector<size_t> to_remove;
        for (size_t i = 0; i < game.delayed_triggers.size(); i++) {
            auto &dt = game.delayed_triggers[i];
            bool matched = false;
            bool expired = false;
            for (const auto &ev : events) {
                if (ev.GetType() != dt.fire_on) continue;
                // Entity-watched "when THIS permanent leaves the battlefield" delayed trigger
                // (CR 603.6e, the earthbend return-tapped trigger): match only the watched
                // entity changing zone from the battlefield. Skip the phase/owner gating below,
                // which is for the phase-based delayed triggers.
                if (dt.fire_on_leave_battlefield) {
                    if (ev.GetType() != Events::CARD_CHANGED_ZONE) continue;
                    if (!ev.HasParam(Params::ENTITY)) continue;
                    if (moved_object(ev) != dt.watched) continue;
                    if (ev.GetParam<Zone::ZoneValue>(Params::ORIGIN) != Zone::BATTLEFIELD) continue;
                    // Destination filter (e.g. earthbend's "when it dies or is exiled"): when the
                    // trigger names specific destination zones, a move to any other zone (bounce
                    // to hand, shuffle into library) does not fire it. Either way the watched
                    // object has left the battlefield — a later re-entry is a NEW object
                    // (CR 400.7), so a non-matching departure EXPIRES the trigger unfired rather
                    // than leaving it armed against the reused entity id.
                    if (!dt.fire_dest_zones.empty()) {
                        Zone::ZoneValue dest = ev.GetParam<Zone::ZoneValue>(Params::DESTINATION);
                        bool dest_ok = false;
                        for (Zone::ZoneValue z : dt.fire_dest_zones)
                            if (z == dest) { dest_ok = true; break; }
                        if (!dest_ok) { expired = true; break; }
                    }
                    matched = true;
                    break;
                }
                if (dt.fire_on == Events::UPKEEP_BEGAN && game.turn_state.turn < dt.fire_on_turn) continue;
                // Phase-player restriction: only an explicit ValidPlayer$ restriction pins the
                // trigger to one player's phase. Unrestricted (restrict_player == 0), "the next
                // turn's upkeep" / "the next end step" fires at the next occurrence of the phase
                // regardless of whose turn it is (CR 603.7 — Mishra's Bauble draws at the
                // opponent's upkeep when sacrificed on your own turn).
                if (dt.restrict_player != 0 && ev.HasParam(Params::PLAYER) &&
                    ev.GetParam<Entity>(Params::PLAYER) != dt.restrict_player) continue;
                matched = true;
                break;
            }
            if (matched) {
                // Determine controller from owner_entity
                Zone::Ownership ctrl = seat_of_player(dt.owner_entity);
                // The copy carries dt.ability.delayed_link (seq, creator, subjects) onto the
                // stack object, so the observation's delayed-trigger block keeps following
                // this trigger until its stack object leaves the stack.
                Ability trigger_ab = dt.ability;
                trigger_ab.controller = ctrl;
                // Defined$ TriggeredCardController (Searing Blood): bind the fire ability's player
                // to the last-known controller of the object whose departure fired this trigger
                // (CR 608.2g — it is in the graveyard by now). Shares triggered_player storage.
                if (trigger_ab.def->defined.is(ObjectSpec::TRIGGERED_CARD_CONTROLLER) && dt.fire_on_leave_battlefield)
                    trigger_ab.triggered_player = last_known_controller(dt.watched.lki_entity());
                // CR 400.7e: an ability that triggers on the watched object leaving the
                // battlefield can find the new object it became in a public zone (earthbend's
                // "return it to the battlefield").
                const Entity departed = dt.watched.lki_entity();
                if (dt.fire_on_leave_battlefield && trigger_ab.source == dt.watched &&
                    global_coordinator.entity_has_component<Zone>(departed) &&
                    is_public_zone(global_coordinator.GetComponent<Zone>(departed).location))
                    trigger_ab.source = ObjectRef::of(departed);
                PendingTrigger pt;
                pt.ab = trigger_ab;
                pt.controller = ctrl;
                pt.source = trigger_ab.source.lki_entity();
                pt.label = "Delayed trigger";
                pt.log_line = "Delayed trigger fires.";
                pending.push_back(pt);
                to_remove.push_back(i);
            } else if (expired) {
                to_remove.push_back(i);
            }
        }
        for (auto it = to_remove.rbegin(); it != to_remove.rend(); ++it)
            game.delayed_triggers.erase(game.delayed_triggers.begin() + static_cast<ptrdiff_t>(*it));
    }

    // Floating triggered abilities (CR 603.7e): those an effect object holds, created by a DB$ Effect
    // | Triggers$ SVar (Forth Eorlingas!, Tamiyo, Seasoned Scholar's +2) or on an emblem. They have no
    // source permanent, so they are scanned here against the drained events rather than by the
    // battlefield loop.
    for (const EffectObject &object : game.resolved_effects.effect_objects) {
        std::vector<const AbilityDef *> defs;
        for (const Ability &ft : object.triggers) defs.push_back(ft.def);
        TriggerHost host;
        host.scan = TriggerScan::EFFECT_OBJECT;
        host.controller = object.controller;
        host.object = &object;
        match_event_triggers(host, defs, events, orderer, pending);
    }

    // The monarch's inherent sourceless triggered abilities (CR 725.2): "At the beginning of the
    // monarch's end step, that player draws a card" and "Whenever a creature deals combat damage to
    // the monarch, its controller becomes the monarch." These have no source object and are not
    // card abilities, so they are produced here directly from the drained events against
    // game.monarch_entity. General over any monarch card (Forth Eorlingas! and future ones).
    if (game.monarch_entity != MAX_ENTITIES) {
        Zone::Ownership monarch_ctrl = seat_of_player(game.monarch_entity);
        for (const auto &ev : events) {
            // End-step draw: the monarch draws an extra card at the beginning of their end step.
            if (ev.GetType() == Events::END_STEP_BEGAN && ev.HasParam(Params::PLAYER) &&
                ev.GetParam<Entity>(Params::PLAYER) == game.monarch_entity) {
                Ability draw_ab(draw_one_trigger_def());
                draw_ab.controller = monarch_ctrl;
                draw_ab.target = ObjectRef::of(game.monarch_entity);  // effect_draw reads target player when set
                PendingTrigger pt;
                pt.ab = draw_ab;
                pt.controller = monarch_ctrl;
                pt.source = 0;
                pt.label = "Monarch (end-step draw)";
                pt.log_line = "The monarch draws a card at the beginning of their end step.";
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
                Ability steal_ab(triggered_effect_def("BecomeMonarch"));
                steal_ab.controller = attacker_ctrl;
                PendingTrigger pt;
                pt.ab = steal_ab;
                pt.controller = attacker_ctrl;
                pt.source = 0;
                pt.label = "Monarch (steal on combat damage)";
                pt.log_line = "A creature dealt combat damage to the monarch.";
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
        // Layer-6 ability removal: a Saga whose abilities are removed (turned into a Mountain
        // by Magus of the Moon, CR 305.7) has no chapter abilities — the trigger never happens.
        // The lore-counter machinery is gated too (saga.cpp), so this only catches an event
        // emitted in the same window the removal landed; release the 714.4 in-flight gate the
        // emitter took since no chapter ability will go on (and thus leave) the stack.
        if (global_coordinator.GetComponent<Permanent>(saga).abilities_removed) {
            auto &saga_perm = global_coordinator.GetComponent<Permanent>(saga);
            if (saga_perm.saga_chapters_in_flight > 0) saga_perm.saga_chapters_in_flight--;
            continue;
        }
        Zone::Ownership ctrl = global_coordinator.GetComponent<Permanent>(saga).controller;

        Ability trigger_ab(cd.saga_chapters[static_cast<size_t>(chapter - 1)]);
        trigger_ab.source = ObjectRef::of(saga);
        trigger_ab.controller = ctrl;

        PendingTrigger pt;
        pt.ab = trigger_ab;
        pt.controller = ctrl;
        pt.source = saga;
        pt.label = entity_name(saga) + " (chapter " + std::to_string(chapter) + ")";
        pt.log_line = entity_name(saga) + " chapter " + std::to_string(chapter) + " triggers.";
        pending.push_back(pt);
    }

    // Impending (CR 702.175e): "At the beginning of your end step, remove a time counter from
    // [each impending permanent you control]." This is a genuine triggered ability — one per
    // impending permanent — so it is produced here from the END_STEP_BEGAN event and placed on
    // the stack (players get priority, opponents can respond), NOT applied as an immediate
    // step side effect. When the last time counter is removed on resolution the permanent
    // becomes a creature on the next state-based pass (the impending suppression in
    // apply_permanent_components is gated on TIME > 0). General over any Impending card.
    for (const auto &ev : events) {
        if (ev.GetType() != Events::END_STEP_BEGAN || !ev.HasParam(Params::PLAYER)) continue;
        Entity end_step_player = ev.GetParam<Entity>(Params::PLAYER);
        Zone::Ownership ctrl = (end_step_player == game.player_a_entity) ? Zone::PLAYER_A
                             : (end_step_player == game.player_b_entity) ? Zone::PLAYER_B
                                                                         : Zone::UNKNOWN;
        if (ctrl == Zone::UNKNOWN) continue;
        for (auto entity : mEntities) {
            if (!is_battlefield_permanent(entity, ctrl)) continue;
            auto &perm = global_coordinator.GetComponent<Permanent>(entity);
            // Only impending permanents that still have an impending time counter to shed.
            if (!perm.entered_via_impending) continue;
            if (get_counters(entity, "TIME") <= 0) continue;

            // Reuses effect_remove_counter (Defined$ Self).
            Ability shed_ab(keyed_ability_def("impending_shed", [] {
                AbilityDef d;
                d.ability_type = AbilityDef::TRIGGERED;
                d.category = "RemoveCounter";
                d.params = CounterParams{"TIME", 1};
                return d;
            }));
            shed_ab.source = ObjectRef::of(entity);             // remove_counter falls back to source (target == 0)
            shed_ab.controller = ctrl;

            PendingTrigger pt;
            pt.ab = shed_ab;
            pt.controller = ctrl;
            pt.source = entity;
            pt.label = perm.name + " (impending: remove a time counter)";
            pt.log_line = perm.name + " triggers: remove a time counter (impending).";
            pending.push_back(pt);
        }
    }

    // Suspend (CR 702.62a, second ability): "At the beginning of your upkeep, if this card is
    // suspended, remove a time counter from it." This is a genuine triggered ability — one per
    // suspended card its owner controls — produced from the UPKEEP_BEGAN event and placed on the
    // stack (players get priority, opponents can respond), NOT applied as an immediate step side
    // effect. The removal is handled by the SuspendTick effect on resolution, and removing the last
    // counter triggers the free cast (SuspendCast); see effect_suspend_tick. A suspended card is one in the
    // exile zone with a positive time-counter count (702.62b), kept on its Zone (an exiled card
    // is not a permanent). General over any Suspend card.
    for (const auto &ev : events) {
        if (ev.GetType() != Events::UPKEEP_BEGAN || !ev.HasParam(Params::PLAYER)) continue;
        Entity upkeep_player = ev.GetParam<Entity>(Params::PLAYER);
        Zone::Ownership ctrl = (upkeep_player == game.player_a_entity) ? Zone::PLAYER_A
                             : (upkeep_player == game.player_b_entity) ? Zone::PLAYER_B
                                                                       : Zone::UNKNOWN;
        if (ctrl == Zone::UNKNOWN) continue;
        for (Entity card : mEntities) {
            auto &cz = global_coordinator.GetComponent<Zone>(card);
            if (cz.location != Zone::EXILE || cz.owner != ctrl) continue;  // still suspended, this player's
            auto time = cz.counters.find("TIME");
            if (time == cz.counters.end() || time->second <= 0) continue;
            std::string cname = global_coordinator.entity_has_component<CardData>(card)
                                    ? global_coordinator.GetComponent<CardData>(card).name : "card";

            Ability tick_ab(triggered_effect_def("SuspendTick"));
            tick_ab.source = ObjectRef::of(card);  // the exiled suspended card
            tick_ab.controller = ctrl;

            PendingTrigger pt;
            pt.ab = tick_ab;
            pt.controller = ctrl;
            pt.source = card;
            pt.label = cname + " (suspend: remove a time counter)";
            pt.log_line = cname + " triggers: remove a time counter (suspend).";
            pending.push_back(pt);
        }
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
        // Layer-6 ability removal (CR 613.1f / 305.7): a permanent whose abilities are removed
        // (Humility "lose all abilities", or a land turned into a basic type by Magus of the
        // Moon) has NO innate triggered abilities either — skip its CardData/Token sources.
        // perm.abilities stays in the list: the layer pass already filtered it down to what
        // survives the removal (an ability granted by the remover itself, or the regenerated
        // subtype-derived mana ability), so anything still there is legitimately functioning.
        std::vector<const AbilityDef *> ab_defs = object_abilities(entity);
        if (ab_defs.empty()) continue;

        TriggerHost host;
        host.entity = entity;
        host.controller = perm.controller;
        host.perm = &perm;
        host.name = entity_name(entity);
        match_event_triggers(host, ab_defs, events, orderer, pending);
    }
    match_departed_watcher_triggers(events, orderer, pending);

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
    const TriggerMode *mode = trigger_mode_on(Events::CARD_CHANGED_ZONE);
    for (const auto &ev : events) {
        if (ev.GetType() != Events::CARD_CHANGED_ZONE) continue;
        if (!ev.HasParam(Params::ENTITY)) continue;
        Zone::ZoneValue ev_origin = ev.GetParam<Zone::ZoneValue>(Params::ORIGIN);
        Zone::ZoneValue ev_dest = ev.GetParam<Zone::ZoneValue>(Params::DESTINATION);
        Entity entity = ev.GetParam<Entity>(Params::ENTITY);
        // Enters-the-battlefield self-triggers (destination battlefield) are handled by the
        // battlefield scan above (the source IS a battlefield permanent once it arrives) — but
        // only while it is STILL there at collection time. A permanent that entered and then
        // left again before triggers were collected (legend-rule keep-the-other-copy, CR 704.5j;
        // a 0-toughness SBA death) still DID enter, so its ETB trigger fired (CR 603.3a) and is
        // collected here from the event with last-known information (603.10d). Skip only the
        // still-on-battlefield case to avoid double-firing.
        bool etb_lookback = false;
        if (ev_dest == Zone::BATTLEFIELD) {
            if (is_battlefield_permanent(entity)) continue;
            etb_lookback = true;
        }
        if (!global_coordinator.entity_has_component<CardData>(entity)) continue;
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        // The source's controller as it left the battlefield persists on its Zone component
        // (only overwritten when a card enters the battlefield), so it still names the last
        // controller; fall back to the owner if it was never set.
        auto &z = global_coordinator.GetComponent<Zone>(entity);
        Zone::Ownership ctrl = (z.controller != Zone::UNKNOWN) ? z.controller : z.owner;
        // The how-it-entered gates ("if you cast it", evoke, offspring) and the active DFC face
        // lived on the stripped Permanent; the look-back reads them from the LKI snapshot taken
        // as the permanent left the battlefield.
        // A battlefield-departure event (or an ETB look-back) refers to the object that left
        // play, so it reads that departed object's snapshot even if the card moved again within
        // the same resolution (a flicker). A move between other zones is a new object's own
        // event (CR 400.7) and sees only a current snapshot.
        const LastKnownInfo *lki = (ev_origin == Zone::BATTLEFIELD || etb_lookback)
                                       ? departed_lki_for(entity)
                                       : lki_for(entity);
        const std::string ent_name =
            (lki && lki->copied_card) ? lki->copied_card->name : entity_name(entity);
        // A transformed DFC functions with its active (back) face's abilities (CR 712.4). A
        // permanent that left play as an in-place copy looks back at the copy's
        // characteristics (CR 603.10), not the printed card it reverted to (CR 400.7).
        const CardData &cd = (lki && lki->copied_card)
                                 ? *lki->copied_card
                                 : global_coordinator.GetComponent<CardData>(entity);
        // Layer-6 ability removal (CR 613.1f / 305.7) via the LKI look-back (CR 603.10): a
        // permanent whose abilities were removed as it left play (Humility "lose all abilities")
        // had NO triggered abilities to fire on leaving — the look-back uses its last-known
        // abilities, which were none. Suppress its own leaves/dies self-trigger entirely,
        // regardless of active DFC face, mirroring the on-battlefield scan's abilities_removed
        // gate above. (The subtype-derived mana ability / remover-granted abilities that survive
        // removal live on perm.abilities, not on the printed CardData scanned here.)
        if (lki && lki->abilities_removed) continue;
        const std::vector<const AbilityDef *> &self_abs =
            (lki && lki->transformed && cd.backside) ? cd.backside->abilities : cd.abilities;
        for (const AbilityDef *abp : self_abs) {
            const AbilityDef &ab = *abp;
            if (ab.ability_type != AbilityDef::TRIGGERED) continue;
            if (ab.trigger_on != Events::CARD_CHANGED_ZONE) continue;
            if (!ab.trigger_only_self) continue;  // Card.Self — only the source's own move
            // Its row: the zones, the ValidCard$ and DisableTriggers (Doorkeeper Thrull), as in the
            // battlefield scan.
            if (!mode->match({ab, entity, ctrl, ev})) continue;
            // ETB gates mirrored from the battlefield scan, read from the LKI snapshot:
            // "if you cast it" (The One Ring), evoke self-sacrifice, offspring token copy.
            if (ab.trigger_requires_entered_by_cast && !(lki && lki->entered_by_cast)) continue;
            if (ab.is_evoke_sacrifice && !(lki && lki->evoked)) continue;
            if (ab.is_offspring_token && !(lki && lki->entered_with_offspring)) continue;
            // ValidCard$ ...+untapped: the look-back path only runs once the permanent has
            // already left the battlefield again, and its tapped-as-it-entered state is not
            // captured in LKI — the live Permanent check the battlefield scan uses is gone.
            // Be conservative and don't fire the untapped-gated trigger here.
            if (ab.trigger_valid_card_untapped) continue;

            Ability trigger_ab(&ab);
            trigger_ab.source = ObjectRef::of(entity);
            trigger_ab.controller = ctrl;
            if (mode->capture) mode->capture(trigger_ab, ev);

            // A leaves-the-battlefield ability that operates on the cards the source had exiled
            // (Skyclave Apparition's TrigToken sizes/owns the Illusion by the exiled card, and
            // gates on it still being exiled): the source has already left the battlefield, so its
            // exiled_with list lives in the last-known-info snapshot captured at departure. Carry
            // it onto the trigger so resolve() can restore the remembered set (CR 608.2h).
            if (lki && !lki->exiled_with.empty())
                trigger_ab.restore_remembered_exiled_with = lki->exiled_with;

            if (etb_lookback) {
                // Parity with the battlefield ETB scan: bind Defined$ TriggeredActivator from the
                // event's player, and apply the 603.4 intervening-if (evaluated now, as the
                // trigger would go on the stack).
                if (ev.HasParam(Params::PLAYER)) {
                    bind_triggered_activator(trigger_ab, ev.GetParam<Entity>(Params::PLAYER));
                    bind_triggered_player(trigger_ab, ev.GetParam<Entity>(Params::PLAYER));
                }
                if (trigger_ab.def->intervening_if &&
                    !evaluate_present_condition(trigger_ab, ctrl, orderer))
                    continue;
            }

            fire_trigger(trigger_ab, ctrl, entity, trigger_label(ent_name, trigger_ab), ent_name + " triggered", orderer,
                         pending);
        }
    }

    // Self-cast triggered abilities (CR 702.33e/f, Wastescape Battlemage): "When you cast this
    // spell, [if it was kicked with its [N] kicker,] ..." fires on SPELL_CAST of the spell
    // ITSELF, while it sits on the stack — not from the battlefield — so the battlefield scan
    // above never sees it. The source is the cast spell entity (the event's ENTITY); read its
    // own CardData abilities for a trigger_only_self SPELL_CAST trigger, matched by its row.
    // General over any self-cast / kicked-N SpellCast trigger.
    for (const auto &ev : events) {
        if (ev.GetType() != Events::SPELL_CAST) continue;
        if (!ev.HasParam(Params::ENTITY)) continue;
        Entity spell_e = ev.GetParam<Entity>(Params::ENTITY);
        if (!global_coordinator.entity_has_component<CardData>(spell_e)) continue;
        if (!global_coordinator.entity_has_component<Spell>(spell_e)) continue;
        Zone::Ownership ctrl = global_coordinator.GetComponent<Spell>(spell_e).caster;
        const std::string ent_name = entity_name(spell_e);
        const TriggerMode *mode = trigger_mode_on(Events::SPELL_CAST);
        for (const auto &ab : global_coordinator.GetComponent<CardData>(spell_e).abilities) {
            if (ab->ability_type != AbilityDef::TRIGGERED) continue;
            if (ab->trigger_on != Events::SPELL_CAST) continue;
            if (!ab->trigger_only_self) continue;  // ValidCard$ Card.Self — only the cast spell itself
            if (mode->match && !mode->match({*ab, spell_e, ctrl, ev})) continue;

            Ability trigger_ab(ab);
            trigger_ab.source = ObjectRef::of(spell_e);
            trigger_ab.controller = ctrl;
            if (mode->capture) mode->capture(trigger_ab, ev);

            // Storm (CR 702.40a): lock in the copy count = number of OTHER spells cast before
            // this one this turn, by EITHER player. The per-player cast counters were already
            // bumped for this storm spell before SPELL_CAST fired (action_processor records the
            // cast first), so the both-player total minus this spell is the storm count. Snapshot
            // it now, at trigger-fire time — spells cast in RESPONSE to the storm trigger come
            // after this spell and must not inflate the count.
            if (trigger_ab.def->kind == EffectKind::Storm) {
                const int storm_count = static_cast<int>(storm_count_this_turn(game));
                trigger_ab.def = derived_ability_def(trigger_ab.def, "storm_count", storm_count,
                                                     [storm_count](AbilityDef &d) {
                                                         d.amount = static_cast<size_t>(storm_count);
                                                     });
            }

            fire_trigger(trigger_ab, ctrl, spell_e, trigger_label(ent_name, trigger_ab), ent_name + " triggered", orderer,
                         pending);
        }
    }

    // Graveyard-functioning triggered abilities (CR 113.6 / 603.6): a card whose triggered
    // ability has TriggerZones$ Graveyard (e.g. Arclight Phoenix's "at the beginning of
    // combat on your turn, ... return ~ from your graveyard to the battlefield") functions
    // while in its owner's graveyard, so it is never scanned by the battlefield loop above.
    for (auto entity : mEntities) {
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(entity);
        if (z.location != Zone::GRAVEYARD) continue;
        if (!global_coordinator.entity_has_component<CardData>(entity)) continue;
        TriggerHost host;
        host.scan = TriggerScan::GRAVEYARD;
        host.entity = entity;
        host.controller = z.owner;
        host.name = entity_name(entity);
        match_event_triggers(host, global_coordinator.GetComponent<CardData>(entity).abilities, events, orderer, pending);
    }
    }

    // ── State-triggered abilities (Mode$ Always, CR 603.8) ──────────────────────────────────
    // A state trigger fires off a game STATE (its IsPresent$ intervening-if), not a drained
    // event, so it is scanned every time this function runs — even when the event batch is empty
    // (outside the `if (!events.empty())` guard above). For each battlefield permanent's state
    // trigger: evaluate the condition against live state; if it is true and the trigger is not
    // already armed, queue it and arm it; if it is false, disarm it so it can fire again the next
    // time the condition becomes true (603.8: "won't trigger again until it has become false and
    // then true again"). The per-permanent latch (Permanent::state_triggers_armed) prevents the
    // trigger from re-queuing on every SBA pass while the condition stays true (e.g. Dark Depths
    // sits on the battlefield with 0 ice counters until its sacrifice trigger resolves).
    for (auto entity : mEntities) {
        if (!is_battlefield_permanent(entity)) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(entity);
        const std::string ent_name = entity_name(entity);
        for (const AbilityDef *abp : object_abilities(entity)) {
            {
                const AbilityDef &ab = *abp;
                if (ab.ability_type != AbilityDef::TRIGGERED) continue;
                if (!ab.trigger_state_condition) continue;

                Ability trigger_ab(&ab);
                trigger_ab.source = ObjectRef::of(entity);
                trigger_ab.controller = perm.controller;

                // Stable per-trigger signature for the latch (category + the state condition),
                // so two distinct state triggers on one permanent latch independently.
                const std::string sig = ab.category + "|" + ab.condition_present.text();
                bool cond = evaluate_present_condition(trigger_ab, perm.controller, orderer);
                bool armed = perm.state_triggers_armed.count(sig) != 0;
                if (!cond) {
                    if (armed) perm.state_triggers_armed.erase(sig);  // re-arm (603.8)
                    continue;
                }
                if (armed) continue;  // already fired while the condition has stayed true
                perm.state_triggers_armed.insert(sig);

                PendingTrigger pt;
                pt.ab = trigger_ab;
                pt.controller = perm.controller;
                pt.source = entity;
                pt.label = trigger_label(ent_name, trigger_ab);
                pt.log_line = ent_name + " triggered";
                pending.push_back(pt);
            }
        }
    }

    // ResolvedLimit$ (CR 603.4 intervening-if): an ability that has already resolved its limit's
    // number of times this turn doesn't trigger.
    for (PendingTrigger &pt : pending)
        if (!trigger_resolved_limit_reached(pt.ab)) game.waiting_triggers.push_back(std::move(pt));
}

void StateManager::place_waiting_triggers(Game &game, std::shared_ptr<Orderer> orderer) {
    if (game.pending.trigger_placement.active) return;
    std::vector<PendingTrigger> pending;
    for (PendingTrigger &pt : game.waiting_triggers) {
        // A trigger queued outside the event scan (Ward, a reflexive trigger) is labeled here
        // for its controller's ordering choice.
        if (pt.label.empty()) pt.label = trigger_label(entity_name(pt.source), pt.ab);
        pending.push_back(std::move(pt));
    }
    game.waiting_triggers.clear();
    place_triggers_apnap(game, orderer, pending);
}

static void match_departed_watcher_triggers(const std::vector<Event> &events,
                                            std::shared_ptr<Orderer> orderer,
                                            std::vector<PendingTrigger> &pending) {
    std::set<Entity> seen;
    for (const auto &ev : events) {
        if (ev.GetType() != Events::CARD_CHANGED_ZONE || !ev.HasParam(Params::ENTITY)) continue;
        if (ev.GetParam<Zone::ZoneValue>(Params::ORIGIN) != Zone::BATTLEFIELD) continue;
        Entity watcher = ev.GetParam<Entity>(Params::ENTITY);
        if (!seen.insert(watcher).second) continue;
        // Back on the battlefield already: the battlefield scan reads it as the object it is now.
        if (is_battlefield_permanent(watcher)) continue;
        if (!global_coordinator.entity_has_component<CardData>(watcher)) continue;
        const LastKnownInfo *lki = departed_lki_for(watcher);
        if (!lki || lki->abilities_removed) continue;
        const CardData &cd = lki->copied_card ? *lki->copied_card
                                              : global_coordinator.GetComponent<CardData>(watcher);
        const std::vector<const AbilityDef *> &ab_defs =
            (lki->transformed && cd.backside) ? cd.backside->abilities : cd.abilities;
        TriggerHost host;
        host.scan = TriggerScan::DEPARTED;
        host.entity = watcher;
        host.controller = last_known_controller(watcher);
        host.name = lki->name.empty() ? entity_name(watcher) : lki->name;
        match_event_triggers(host, ab_defs, events, orderer, pending);
    }
}

static void match_event_triggers(const TriggerHost &host, const std::vector<const AbilityDef *> &ab_defs,
                                 const std::vector<Event> &events, std::shared_ptr<Orderer> orderer,
                                 std::vector<PendingTrigger> &pending) {
    const Entity entity = host.entity;
    const Zone::Ownership controller = host.controller;
    const Permanent *perm = host.perm;
    const bool departed = host.scan == TriggerScan::DEPARTED;
    // Mode$ ChangesZoneAll batch triggers (CR 603.2c) fire once for a whole group of
    // simultaneous zone changes, not once per matching card. Track which of this permanent's
    // batch triggers have already queued this scan so later matching events in the same batch
    // are skipped. Keyed by the ability template address (&ab), which is stable across the
    // event loop (it lives in CardData/Token/perm.abilities).
    std::set<const AbilityDef *> batch_zone_all_fired;

    for (const auto &ev : events) {
        // The row reading this event. Hand-built triggers have no Mode$, so it is found by event.
        const TriggerMode *mode = trigger_mode_on(ev.GetType());
        const bool names_target = mode && mode->names_target;
        for (size_t ab_index = 0; ab_index < ab_defs.size(); ab_index++) {
            const AbilityDef &ab = *ab_defs[ab_index];
            if (ab.ability_type != AbilityDef::TRIGGERED) continue;
            // Static$ True mana-additional triggers (Badgermole Cub's TapsForMana) never go
            // on the stack — they resolve immediately inside the mana system (CR 605.1a).
            if (ab.trigger_taps_for_mana_static) continue;
            // A graveyard-functioning trigger (TriggerZones$ Graveyard, e.g. Arclight
            // Phoenix's begin-combat return) functions ONLY while its source is in the
            // graveyard — TriggerZones overrides the default battlefield functioning
            // (CR 113.6) — and that scan reads no other.
            if (ab.trigger_from_graveyard != (host.scan == TriggerScan::GRAVEYARD)) continue;
            // Match the primary trigger event OR any additional event in trigger_on_extra (a
            // phase trigger listing several phases, e.g. Carpet of Flowers' Phase$ Main1,Main2).
            {
                bool fires = (ab.trigger_on != 0) &&
                             (ab.trigger_on == ev.GetType() ||
                              std::find(ab.trigger_on_extra.begin(), ab.trigger_on_extra.end(),
                                        ev.GetType()) != ab.trigger_on_extra.end());
                if (!fires) continue;
            }
            // A source that left the battlefield along with the moving objects triggers only
            // through the leaves-the-battlefield look-back (CR 603.10a): its Origin$ Battlefield
            // zone-change abilities, on a move off the battlefield. Its own Card.Self departure is
            // the self zone-change look-back's (collect_triggered_abilities).
            if (departed && (ab.trigger_on != Events::CARD_CHANGED_ZONE || ab.trigger_only_self ||
                             ab.trigger_zone_origin != static_cast<int>(Zone::BATTLEFIELD) ||
                             ev.GetType() != Events::CARD_CHANGED_ZONE ||
                             ev.GetParam<Zone::ZoneValue>(Params::ORIGIN) != Zone::BATTLEFIELD))
                continue;
            // "another" check: skip if the event entity is the triggering permanent itself
            if (ab.trigger_self_excluded && ev.HasParam(Params::ENTITY) &&
                ev.GetParam<Entity>(Params::ENTITY) == entity) continue;
            // Card.Self: only fire when the event entity is the triggering permanent itself.
            // BECAME_TARGET is exempt: there ENTITY is the targeting object (not the source),
            // and ValidTarget$ Card.Self is matched against the TARGET param by its row instead.
            if (ab.trigger_only_self && !names_target &&
                ev.HasParam(Params::ENTITY) &&
                ev.GetParam<Entity>(Params::ENTITY) != entity) continue;
            // "If you cast it" (ValidCard$ Card.wasCastByYou): a self ETB trigger that only
            // fires when its source permanent entered the battlefield by being cast (CR
            // 614.12; The One Ring's protection grant). perm is the source (trigger_only_self).
            if (ab.trigger_requires_entered_by_cast && !(perm && perm->entered_by_cast)) continue;
            // Evoke self-sacrifice only fires when this permanent was cast via evoke
            if (ab.is_evoke_sacrifice && !(perm && perm->evoked)) continue;
            // Offspring token copy only fires when this permanent was cast with offspring
            if (ab.is_offspring_token && !(perm && perm->entered_with_offspring)) continue;
            // (Front/back face selection is done once when ab_sources is built above, so a
            // transformed permanent already only sees its active face's triggers here.)
            // ValidPlayer$ You: only fire when the event's player matches the permanent's
            // controller. BECAME_TARGET is exempt, like trigger_only_self above: there the
            // PLAYER param is the targeting spell's controller (the OPPONENT, typically), not
            // a "you" reference, so gating on it would wrongly suppress a becomes-target
            // trigger authored with ValidPlayer$ You. Such a trigger's own ValidSource$/
            // ValidTarget$ clauses constrain it in its row.
            if (ab.trigger_valid_player_is_controller && !names_target &&
                ev.HasParam(Params::PLAYER)) {
                Entity event_player = ev.GetParam<Entity>(Params::PLAYER);
                Entity ctrl_entity = get_player_entity(controller);
                if (event_player != ctrl_entity) continue;
            }
            // ValidActivatingPlayer$ Opponent (Lavinia, Azorius Renegade): only fire when the
            // acting player is an OPPONENT of this source's controller (the event's PLAYER is
            // the caster on SPELL_CAST). In a two-player game "opponent" = not the controller.
            if (ab.trigger_valid_player_is_opponent && !names_target &&
                ev.HasParam(Params::PLAYER)) {
                Entity event_player = ev.GetParam<Entity>(Params::PLAYER);
                Entity ctrl_entity = get_player_entity(controller);
                if (event_player == ctrl_entity) continue;
            }
            // The mode's own filters on this event: the trigger_modes.h row's matcher.
            if (mode && mode->match && !mode->match({ab, entity, controller, ev})) continue;

            // Prepare the triggered ability and queue it; APNAP placement (and any target
            // selection) happens after the full scan, in place_triggers_apnap().
            Ability trigger_ab = host.object ? host.object->triggers[ab_index] : Ability(&ab);
            if (!host.object) trigger_ab.source = ObjectRef::of(entity);
            trigger_ab.controller = controller;
            // CR 107.3m: an enters-the-battlefield trigger of a permanent uses the X its spell was
            // cast with.
            if (perm && is_self_etb_event(ev, entity)) trigger_ab.x_paid = perm->entered_x;
            if (mode && mode->capture) mode->capture(trigger_ab, ev);
            // Defined$ TriggeredSpellAbility — the effect (Counter) acts on the spell that
            // fired this trigger. Capture it from the event as the ability's target.
            if (trigger_ab.def->defined.is(ObjectSpec::TRIGGERED_SPELL) && ev.HasParam(Params::ENTITY))
                trigger_ab.target = ObjectRef::of(ev.GetParam<Entity>(Params::ENTITY));
            // Defined$ TriggeredActivator — bind the player who caused the trigger (the
            // event's PLAYER, e.g. the caster of the noncreature spell) onto this ability
            // and its subabilities, so the effect (LoseLife etc.) resolves against them.
            if (ev.HasParam(Params::PLAYER)) {
                bind_triggered_activator(trigger_ab, ev.GetParam<Entity>(Params::PLAYER));
                // Defined$ TriggeredPlayer — bind the player whose event fired (e.g. the active
                // player whose upkeep began, Roiling Vortex's "each player's upkeep").
                bind_triggered_player(trigger_ab, ev.GetParam<Entity>(Params::PLAYER));
            }
            // For exalted, target the sole attacker from the event
            if (trigger_ab.def->kind == EffectKind::ExaltedBonus && ev.HasParam(Params::ENTITY))
                trigger_ab.target = ObjectRef::of(ev.GetParam<Entity>(Params::ENTITY));

            // 603.4 intervening-if: a trigger whose "if" condition is false right now does
            // not go on the stack at all (it is re-checked again on resolution).
            if (trigger_ab.def->intervening_if &&
                !evaluate_present_condition(trigger_ab, controller, orderer))
                continue;
            // Per-permanent stored-SVar gate (a CheckSVar$ on a Number$ scratch-int latch): the
            // trigger does not go on the stack unless the source's latched scratch int satisfies
            // the comparison.
            if (!stored_svar_gate_passes(entity, trigger_ab.def->stored_svar_gate_name,
                                         trigger_ab.def->stored_svar_gate_compare))
                continue;

            // Mode$ ChangesZoneAll batch trigger (CR 603.2c): dedupe to a single firing for
            // the whole simultaneous group. The first matching event queues it; further
            // matching events for this same ability template are skipped.
            if (ab.trigger_batch_zone_all && !batch_zone_all_fired.insert(&ab).second)
                continue;

            // Triggered abilities that require a target (e.g. Talon Gates of Madara's
            // "up to one target creature phases out") choose their target as the ability
            // goes on the stack, by the controller, in APNAP placement order.
            if (host.object)
                fire_trigger(trigger_ab, controller, 0, floating_trigger_label(trigger_ab),
                             "A floating triggered ability triggers.", orderer, pending);
            else
                fire_trigger(trigger_ab, controller, entity, trigger_label(host.name, trigger_ab),
                             host.name + " triggered", orderer, pending);
        }
    }
}

static void fire_trigger(Ability &trigger_ab, Zone::Ownership controller, Entity source, const std::string &label,
                         const std::string &log_line, std::shared_ptr<Orderer> orderer,
                         std::vector<PendingTrigger> &pending) {
    if (trigger_ab.def->trigger_static_offstack) {
        coverage_trigger(trigger_ab);
        resolve_ability(trigger_ab, orderer);
        return;
    }
    PendingTrigger pt;
    pt.ab = trigger_ab;
    pt.controller = controller;
    pt.source = source;
    pt.label = label;
    pt.log_line = log_line;
    pending.push_back(pt);
}

static std::string floating_trigger_label(const Ability &ab) {
    return "Floating trigger (" + ab.def->category + ")";
}

static std::string trigger_label(const std::string &name, const Ability &ab) {
    if (ab.def->is_evoke_sacrifice) return name + " (evoke: sacrifice)";
    if (ab.def->is_offspring_token) return name + " (offspring: token copy)";
    std::string s = name + " (" + ab.def->category;
    if (ab.def->valid_tgts != "N_A" && !ab.def->valid_tgts.empty()) s += ", targeted";
    s += ")";
    return s;
}

static void place_triggers_apnap(Game &game, std::shared_ptr<Orderer> orderer,
                                 std::vector<PendingTrigger> &pending) {
    if (pending.empty()) return;
    if (game.pending.trigger_placement.active)
        fatal_error("place_triggers_apnap re-entered with a placement already in flight");

    Zone::Ownership active = active_seat();
    Zone::Ownership apnap[2] = {active,
                                opponent_of(active)};

    // Flatten into the persisted APNAP queue — the active player's group first
    // (so their triggers end up on the bottom and resolve last), collection
    // order within each group — and drive it. When a placement decision is
    // needed mid-queue, resume_trigger_placement parks it as a loop-top
    // pending decision and returns; the main loop dispatches the answer back
    // into it until the queue empties.
    TriggerPlacementRT &tp = game.pending.trigger_placement;
    tp.active = true;
    tp.saved_priority = cur_game.priority.player_a_has_priority;
    tp.announce = AnnounceRT{};
    tp.queue.clear();
    tp.placed.clear();
    for (Zone::Ownership owner : apnap)
        for (const auto &pt : pending)
            if (pt.controller == owner) tp.queue.push_back(pt);
    resume_trigger_placement(game, orderer);
}

static AnnounceStatus announce_trigger(PendingTrigger &pt, TriggerPlacementRT &tp,
                                       bool suspendable, std::shared_ptr<Orderer> orderer) {
    if (!suspendable) {
        BlockingTargetAsker asker;
        return run_announce(pt.ab, tp.announce, asker, orderer, pt.controller, true);
    }
    TriggerPlaceTargetAsker asker;
    return run_announce(pt.ab, tp.announce, asker, orderer, pt.controller, true);
}

void resume_trigger_placement(Game &game, std::shared_ptr<Orderer> orderer) {
    TriggerPlacementRT &tp = game.pending.trigger_placement;
    if (!tp.active)
        fatal_error("resume_trigger_placement without an active trigger placement");
    // Pre-game placements (test-harness preplaced permanents) have no loop top
    // to park a query at — they block inline exactly as before.
    bool suspendable = in_main_loop();

    while (!tp.queue.empty()) {
        Zone::Ownership owner = tp.queue.front().controller;
        // Placement-time decisions (the 603.3b ordering pick and 603.3d target selection)
        // belong to the TRIGGER'S controller, who need not be the player holding priority
        // (e.g. the opponent's draw fired our Orcish Bowmasters). The deciding seat
        // follows cur_game.priority.player_a_has_priority, so seat the queries on the owner;
        // tp.saved_priority is restored when the whole placement completes.
        cur_game.priority.player_a_has_priority = (owner == Zone::PLAYER_A);

        // The current group: the leading run of queue entries sharing the front's controller.
        size_t group_size = 1;
        while (group_size < tp.queue.size() && tp.queue[group_size].controller == owner)
            group_size++;

        // 603.3b: the owner puts their simultaneously-triggered abilities on the stack in an
        // order of their choosing. We place one at a time; the first chosen ends up on the
        // bottom of the stack (resolves last). A single trigger needs no choice.
        size_t pick = 0;  // position within the group
        if (!tp.announce.active && group_size > 1) {
            PendingQuery &pq = game.pending.query;
            if (suspendable && pq.active) {
                // Consume the latched ordering answer (the queue is untouched between
                // arm and resume, so the group it indexes is identical).
                if (!pq.answered || pq.tag != PendingQuery::TRIGGER_PLACE)
                    fatal_error("trigger ordering resume without a latched TRIGGER_PLACE answer");
                if (pq.menu.size() != group_size)
                    fatal_error("trigger ordering menu size changed between arm and resume (" +
                                std::to_string(pq.menu.size()) + " vs " +
                                std::to_string(group_size) + ")");
                pick = static_cast<size_t>(pq.answer);
                pq = PendingQuery{};
            } else {
                std::vector<LegalAction> choices;
                for (size_t gi = 0; gi < group_size; gi++) {
                    LegalAction la(PASS_PRIORITY, tp.queue[gi].source, tp.queue[gi].label);
                    la.category = ActionCategory::ORDER_TRIGGERS;
                    // Menu position disambiguates two triggers that share one source
                    // permanent (same-entity collision).
                    la.option_ordinal = static_cast<int>(choices.size());
                    choices.push_back(la);
                }
                game_log("%s orders %zu simultaneous triggers (pick which goes on the stack next).\n",
                         player_name(owner).c_str(), group_size);
                // The group's leading trigger source is the pending-decision source
                // (each menu entry still carries its own trigger's source). A floating
                // trigger has no source object; the card whose effect created it stands in.
                const PendingTriggerRT &lead = tp.queue.front();
                Entity order_source = lead.source != 0 ? lead.source : lead.ab.floating_creator;
                if (!suspendable) {
                    PendingDecisionScope pending(order_source);
                    pick = static_cast<size_t>(InputLogger::instance().get_input(choices));
                } else {
                    // Park the ordering pick for the main loop.
                    pq = PendingQuery{};
                    pq.tag = PendingQuery::TRIGGER_PLACE;
                    pq.menu = std::move(choices);
                    pq.chooser_is_a = (owner == Zone::PLAYER_A);
                    pq.decision_source = order_source;
                    pq.prev_priority = cur_game.priority.player_a_has_priority;
                    pq.answered = false;
                    pq.answer = -1;
                    pq.active = true;
                    return;
                }
            }
        }
        // Apply the ordering pick: the chosen entry moves to the queue front.
        if (pick > 0)
            std::rotate(tp.queue.begin(), tp.queue.begin() + static_cast<ptrdiff_t>(pick),
                        tp.queue.begin() + static_cast<ptrdiff_t>(pick) + 1);

        PendingTrigger &pt = tp.queue.front();
        const AnnounceStatus announced = announce_trigger(pt, tp, suspendable, orderer);
        if (announced == AnnounceStatus::SUSPENDED) return;
        if (announced == AnnounceStatus::REMOVED) {
            // CR 603.3c/d: if no mode can be chosen, or no legal choice can be made for a
            // required target, as the triggered ability would go on the stack, the ability is
            // simply removed — never placed mode- or target-less (it would fizzle confusingly,
            // or worse, resolve against target 0). Optional targeting (target_min 0) always
            // passes.
            const bool no_mode = is_modal(pt.ab) && pt.ab.charm_chosen.empty();
            game_log("%s's trigger is removed - no legal %s\n", entity_name(pt.source).c_str(),
                     no_mode ? "modes (603.3c)" : "targets (603.3d)");
            tp.queue.erase(tp.queue.begin());
            continue;
        }
        // A triggered ability whose X nothing defined (CR 107.3m/n) has X = 0 (CR 107.3i: the
        // source's own X is 0 off the stack), so it never reads a stale X from another spell.
        if (pt.ab.x_paid < 0) pt.ab.x_paid = 0;
        tp.placed.push_back(orderer->push_ability_onto_stack(pt.ab, pt.controller));
        coverage_trigger(pt.ab);
        game_log("%s\n", pt.log_line.c_str());
        tp.queue.erase(tp.queue.begin());
    }

    // The placed abilities' targets became their targets as each went on the stack (CR 603.3d);
    // the Ward and becomes-target abilities that triggered meanwhile are put on the stack after
    // this placement, above every ability it placed (CR 603.3b).
    for (Entity placed : tp.placed)
        fire_targeting_hooks(placed, global_coordinator.GetComponent<Ability>(placed).controller);

    // Placement complete: restore the pre-placement priority seat.
    cur_game.priority.player_a_has_priority = tp.saved_priority;
    tp = TriggerPlacementRT{};
}
