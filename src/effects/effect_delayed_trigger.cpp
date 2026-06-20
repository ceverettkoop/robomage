#include "effects.h"

#include <cstdint>
#include <string>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../mana_system.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

static uint32_t phase_string_to_event(const std::string &phase) {
    if (phase == "Upkeep") return Events::UPKEEP_BEGAN;
    if (phase == "Draw") return Events::DRAW_STEP_BEGAN;
    if (phase == "EndStep") return Events::END_STEP_BEGAN;
    return Events::UPKEEP_BEGAN;  // default
}

bool delayed_trigger(Ability &ab, std::shared_ptr<Orderer> orderer) {
    (void)orderer;
    Zone::Ownership owner = global_coordinator.entity_has_component<Permanent>(ab.source)
                                ? global_coordinator.GetComponent<Permanent>(ab.source).controller
                                : global_coordinator.GetComponent<Zone>(ab.source).owner;
    Entity owner_entity = get_player_entity(owner);

    const DelayedTriggerParams *dp = std::get_if<DelayedTriggerParams>(&ab.params);
    bool has_execute = dp && !dp->execute_svar.empty();
    std::string phase = dp ? dp->phase : std::string();
    bool next_turn = dp && dp->next_turn;

    // Build the ability to fire: use Execute$ sub-ability if parsed, else fall back to Draw 1
    Ability fire_ab;
    if (!ab.subabilities.empty() && has_execute) {
        fire_ab = ab.subabilities.back();
        fire_ab.source = ab.source;
    } else {
        fire_ab.ability_type = Ability::TRIGGERED;
        fire_ab.category = "Draw";
        fire_ab.amount = 1;
        fire_ab.source = ab.source;
    }

    uint32_t event_id = phase.empty() ? Events::UPKEEP_BEGAN : phase_string_to_event(phase);

    DelayedTrigger dt;
    dt.ability = fire_ab;
    dt.fire_on = event_id;
    dt.owner_entity = owner_entity;
    dt.fire_on_turn = next_turn ? cur_game.turn + 1 : cur_game.turn;
    cur_game.delayed_triggers.push_back(dt);
    game_log("Delayed trigger registered: %s at next %s.\n", fire_ab.category.c_str(),
        phase.empty() ? "upkeep" : phase.c_str());
    return true;
}

bool parse_delayed_trigger(Ability &ab, const std::string &key, const std::string &value) {
    if (key == "NextTurn") { effect_params<DelayedTriggerParams>(ab).next_turn = (value == "True"); return true; }
    if (key == "Phase")    { effect_params<DelayedTriggerParams>(ab).phase = value; return true; }
    if (key == "Execute")  { effect_params<DelayedTriggerParams>(ab).execute_svar = value; return true; }
    if (key == "ValidPlayer") { effect_params<DelayedTriggerParams>(ab).valid_player = value; return true; }
    return false;
}

}  // namespace effects
