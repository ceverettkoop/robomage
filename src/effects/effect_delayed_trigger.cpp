#include "effects.h"

#include <cstdint>
#include <string>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../game_queries.h"
#include "../mana_system.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

static uint32_t phase_string_to_event(const std::string &phase) {
    if (phase == "Upkeep") return Events::UPKEEP_BEGAN;
    if (phase == "Draw") return Events::DRAW_STEP_BEGAN;
    // Forge writes the end step as either "EndStep" or "End of Turn" (the parser keeps the raw
    // Phase$ value). Both map to the beginning of the end step (CR 513).
    if (phase == "EndStep" || phase == "End of Turn") return Events::END_STEP_BEGAN;
    return Events::UPKEEP_BEGAN;  // default
}

HandlerResult delayed_trigger(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    Zone::Ownership owner = source_controller(ab.source);
    Entity owner_entity = get_player_entity(owner);

    const DelayedTriggerParams *dp = std::get_if<DelayedTriggerParams>(&ab.params);
    bool has_execute = dp && !dp->execute_svar.empty();
    std::string phase = dp ? dp->phase : std::string();
    bool next_turn = dp && dp->next_turn;

    // Build the ability to fire from the Execute$ sub-ability (marked from_delayed_execute at
    // parse). Any trailing SubAbility$ on the DB$ DelayedTrigger (e.g. a DBCleanup that clears
    // remembered objects) chains after the Execute ability, so it runs once the fire ability has
    // resolved. Falls back to Draw 1 if no Execute was parsed.
    Ability fire_ab;
    bool found_execute = false;
    if (has_execute) {
        for (size_t i = 0; i < ab.subabilities.size(); i++) {
            if (!ab.subabilities[i].from_delayed_execute) continue;
            fire_ab = ab.subabilities[i];
            for (size_t j = 0; j < ab.subabilities.size(); j++)
                if (j != i) fire_ab.subabilities.push_back(ab.subabilities[j]);
            found_execute = true;
            break;
        }
    }
    if (!found_execute) {
        fire_ab.ability_type = Ability::TRIGGERED;
        fire_ab.category = "Draw";
        fire_ab.amount = 1;
    }
    fire_ab.source = ab.source;

    // Mode$ ChangesZone (Searing Blood): a "when THAT object changes zone, do Y" delayed trigger
    // (CR 603.7b). The watched object is the parent spell's target — RememberObjects$ Targeted
    // put it in cur_game.remembered_entities before this handler ran. Register a
    // fire_on_leave_battlefield watch (reusing the earthbend infrastructure) filtered to the
    // Destination$ zone(s) so only the matching move fires it; ThisTurn$ bounds it to this turn.
    if (dp && dp->mode_changes_zone) {
        if (cur_game.remembered_entities.empty()) return HandlerResult::DONE_NO_SUBS;
        Entity watched = cur_game.remembered_entities[0];
        DelayedTrigger dt;
        dt.ability = fire_ab;
        dt.fire_on = Events::CARD_CHANGED_ZONE;
        dt.owner_entity = owner_entity;
        dt.fire_on_turn = cur_game.turn;
        dt.watch_entity = watched;
        dt.fire_on_leave_battlefield = true;
        // Origin$ Battlefield / Destination$ Graveyard parsed onto this ability by parse_change_zone.
        // The origin is implicit (leave-battlefield watch); the destination becomes the zone filter
        // so a bounce/exile of the watched object does not fire a "when it dies" trigger.
        if (ab.destination == Zone::GRAVEYARD || ab.destination == Zone::EXILE ||
            ab.destination == Zone::HAND || ab.destination == Zone::LIBRARY)
            dt.fire_dest_zones = {ab.destination};
        dt.expires_end_of_turn = dp->this_turn;
        cur_game.delayed_triggers.push_back(dt);
        game_log("Delayed trigger registered: %s when watched permanent leaves the battlefield.\n",
                 fire_ab.category.c_str());
        return HandlerResult::DONE_NO_SUBS;
    }

    uint32_t event_id = phase.empty() ? Events::UPKEEP_BEGAN : phase_string_to_event(phase);

    DelayedTrigger dt;
    // RememberObjects$ RememberedLKI: capture the objects the preceding RememberChanged$
    // ChangeZone just moved (CR 603.7a). They are restored into cur_game.remembered_entities
    // when the trigger fires so the Execute ability's Defined$ DelayTriggerRememberedLKI acts on
    // exactly those objects (Flickerwisp / Phelia return the card they exiled).
    if (dp && dp->remember_objects_lki) {
        // Nothing remembered means nothing to act on (CR 603.6-style): an "up to one
        // target" exile with no target chosen (or the target gone by resolution) moved
        // nothing, so do NOT register the delayed return at all. Registering it anyway
        // made the return fire on whatever the global remembered set held at end of turn
        // (a phantom "<unknown>" entering the battlefield, falsely firing ETB watchers).
        if (cur_game.remembered_entities.empty()) return HandlerResult::DONE_NO_SUBS;
        dt.remembered_objects = cur_game.remembered_entities;
        fire_ab.restore_remembered_exiled_with = cur_game.remembered_entities;
    }
    dt.ability = fire_ab;
    dt.fire_on = event_id;
    dt.owner_entity = owner_entity;
    // ValidPlayer$ pins the firing phase to one player's turn: "You" = the ability's controller,
    // "Opponent" = the other player. "Player" (any player) or an absent ValidPlayer$ leaves the
    // trigger unrestricted — it fires at the NEXT occurrence of the phase whoever is active
    // (Mishra's Bauble's "the next turn's upkeep", Flickerwisp's "the next end step").
    if (dp) {
        if (dp->valid_player == "You")
            dt.restrict_player = owner_entity;
        else if (dp->valid_player == "Opponent")
            dt.restrict_player =
                get_player_entity(owner == Zone::PLAYER_A ? Zone::PLAYER_B : Zone::PLAYER_A);
    }
    dt.fire_on_turn = next_turn ? cur_game.turn + 1 : cur_game.turn;
    cur_game.delayed_triggers.push_back(dt);
    game_log("Delayed trigger registered: %s at next %s.\n", fire_ab.category.c_str(),
        phase.empty() ? "upkeep" : phase.c_str());
    // Return false so resolve() does NOT chain this DB$ DelayedTrigger's subabilities inline:
    // the Execute$ ability (and any trailing cleanup) is deferred onto the delayed trigger to
    // run when it fires later, not now (CR 603.7a). Chaining them here would resolve the return
    // immediately instead of at the scheduled phase.
    return HandlerResult::DONE_NO_SUBS;
}

bool parse_delayed_trigger(Ability &ab, const std::string &key, const std::string &value) {
    if (ab.category != "DelayedTrigger") return false;
    // Mode$ on a DB$ DelayedTrigger: Phase (the default) fires at a future step; ChangesZone
    // (Searing Blood) instead watches a specific object leaving one zone for another. Consume
    // Mode either way so it isn't flagged as an unrecognized param.
    if (key == "Mode") {
        if (value == "ChangesZone")
            effect_params<DelayedTriggerParams>(ab).mode_changes_zone = true;
        return true;
    }
    if (key == "NextTurn") { effect_params<DelayedTriggerParams>(ab).next_turn = (value == "True"); return true; }
    if (key == "Phase")    { effect_params<DelayedTriggerParams>(ab).phase = value; return true; }
    if (key == "Execute")  { effect_params<DelayedTriggerParams>(ab).execute_svar = value; return true; }
    if (key == "ValidPlayer") { effect_params<DelayedTriggerParams>(ab).valid_player = value; return true; }
    // ThisTurn$ True bounds a ChangesZone watch to its registration turn (CR 603.7b).
    if (key == "ThisTurn") { effect_params<DelayedTriggerParams>(ab).this_turn = (value == "True"); return true; }
    // ValidCard$ Card.IsTriggerRemembered restates that the watched card is the remembered
    // (Targeted) object; the watch entity already comes from RememberObjects$ Targeted, so the
    // filter is informational here. Consume it so it isn't flagged unrecognized.
    if (key == "ValidCard") return true;
    return false;
}

}  // namespace effects
