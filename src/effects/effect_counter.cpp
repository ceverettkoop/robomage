#include "effects.h"

#include <string>

#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/color_identity.h"
#include "../components/spell.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../error.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;

namespace effects {

bool target_color_condition_met(const Ability &ab, Entity target) {
    if (ab.condition_present.empty()) return true;
    const std::string &c = ab.condition_present;
    Colors required = NO_COLOR;
    if (c.find(".Red") != std::string::npos) required = RED;
    else if (c.find(".Blue") != std::string::npos) required = BLUE;
    else if (c.find(".Green") != std::string::npos) required = GREEN;
    else if (c.find(".White") != std::string::npos) required = WHITE;
    else if (c.find(".Black") != std::string::npos) required = BLACK;
    if (required == NO_COLOR) return true;  // non-color condition (e.g. cmcLEX) — not handled here
    if (!global_coordinator.entity_has_component<ColorIdentity>(target)) return false;
    return global_coordinator.GetComponent<ColorIdentity>(target).colors.count(required) > 0;
}

bool counter(Ability &ab, std::shared_ptr<Orderer> orderer) {
    if (global_coordinator.entity_has_component<Zone>(ab.target)) {
        auto &tz = global_coordinator.GetComponent<Zone>(ab.target);
        if (tz.location == Zone::STACK) {
            Zone::Ownership target_controller = global_coordinator.entity_has_component<Spell>(ab.target)
                                                    ? global_coordinator.GetComponent<Spell>(ab.target).caster
                                                    : tz.owner;

            bool do_counter = true;
            // Pyroblast/Hydroblast: only counter if the target is the required color.
            // The spell still resolves (and is put to the graveyard) doing nothing otherwise.
            if (!target_color_condition_met(ab, ab.target)) {
                std::string tname = global_coordinator.entity_has_component<CardData>(ab.target)
                                        ? global_coordinator.GetComponent<CardData>(ab.target).name
                                        : "<unknown>";
                game_log("%s is not the required color — not countered\n", tname.c_str());
                do_counter = false;
            }
            if (do_counter && ab.unless_generic_cost > 0) {
                std::string tname = global_coordinator.entity_has_component<CardData>(ab.target)
                                        ? global_coordinator.GetComponent<CardData>(ab.target).name
                                        : "<unknown>";
                game_log("%s's controller may pay {%zu} to save it:\n", tname.c_str(), ab.unless_generic_cost);
                do_counter = run_unless_loop(ab.unless_generic_cost, target_controller, orderer, ab.target);
            }

            // Can't be countered check (Cavern of Souls)
            if (do_counter && global_coordinator.entity_has_component<Spell>(ab.target) &&
                global_coordinator.GetComponent<Spell>(ab.target).cant_be_countered) {
                std::string name = global_coordinator.entity_has_component<CardData>(ab.target)
                                       ? global_coordinator.GetComponent<CardData>(ab.target).name
                                       : "<unknown>";
                game_log("%s can't be countered\n", name.c_str());
                do_counter = false;
            }

            if (do_counter) {
                std::string name = global_coordinator.entity_has_component<CardData>(ab.target)
                                       ? global_coordinator.GetComponent<CardData>(ab.target).name
                                       : "<unknown>";
                bool is_standalone_ability = !global_coordinator.entity_has_component<Spell>(ab.target) &&
                                            global_coordinator.entity_has_component<Ability>(ab.target);
                if (global_coordinator.entity_has_component<Ability>(ab.target))
                    global_coordinator.RemoveComponent<Ability>(ab.target);
                if (global_coordinator.entity_has_component<Spell>(ab.target))
                    global_coordinator.RemoveComponent<Spell>(ab.target);
                if (is_standalone_ability) {
                    // Standalone ability entities (activated/triggered) have no card to send
                    // to a zone — remove from stack and destroy (rule 701.5a)
                    orderer->add_to_zone(false, ab.target, Zone::EXILE);
                    global_coordinator.DestroyEntity(ab.target);
                } else {
                    Zone::ZoneValue counter_dest = (ab.destination == Zone::EXILE) ? Zone::EXILE : Zone::GRAVEYARD;
                    orderer->add_to_zone(false, ab.target, counter_dest);
                }
                game_log("%s is countered\n", name.c_str());
            }
        } else {
            non_fatal_error("Counter should have fizzled prior to this");
        }
    } else {
        non_fatal_error("Counter should have fizzled prior to this");
    }
    return true;
}

}  // namespace effects
