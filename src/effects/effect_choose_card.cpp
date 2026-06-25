#include "effects.h"

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/spell.h"
#include "../components/types.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"
#include "../input_logger.h"
#include "../action_processor.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

bool choose_card(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // Dauthi Voidwalker: choose an exiled card owned by opponent with a void counter,
    // then play it without paying its mana cost.
    Zone::Ownership ctrl = ab.controller;
    Zone::Ownership opp = (ctrl == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;

    std::vector<Entity> choices;
    for (Entity e = 0; e < global_coordinator.GetMaxIssuedEntity(); ++e) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location != Zone::EXILE) continue;
        if (z.owner != opp) continue;
        if (cur_game.void_countered.find(e) == cur_game.void_countered.end()) continue;
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        choices.push_back(e);
    }

    if (!choices.empty()) {
        std::vector<LegalAction> pick_actions;
        for (auto e : choices) {
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            LegalAction la(PASS_PRIORITY, e, "Play " + cd.name + " (exiled, free)");
            la.category = ActionCategory::PLAY_FREE;
            pick_actions.push_back(la);
        }

        game_log("Choose an exiled card with a void counter:\n");
        int choice = InputLogger::instance().get_input(pick_actions);
        Entity chosen = choices[static_cast<size_t>(choice)];
        auto &cd = global_coordinator.GetComponent<CardData>(chosen);
        cur_game.void_countered.erase(chosen);

        // Determine if permanent or instant/sorcery
        bool is_permanent = false;
        for (auto &t : cd.types) {
            if (t.kind == TYPE && (t.name == "Creature" || t.name == "Artifact" ||
                t.name == "Enchantment" || t.name == "Planeswalker" || t.name == "Land")) {
                is_permanent = true;
                break;
            }
        }

        if (is_permanent) {
            orderer->add_to_zone(false, chosen, Zone::BATTLEFIELD);
            auto &cz = global_coordinator.GetComponent<Zone>(chosen);
            cz.controller = ctrl;
            game_log("%s plays %s from exile (Dauthi Voidwalker).\n",
                     player_name(ctrl).c_str(), cd.name.c_str());
        } else {
            // Instant/Sorcery: put on stack as a spell, let normal resolution handle it
            Spell sp;
            sp.caster = ctrl;
            global_coordinator.AddComponent(chosen, sp);
            for (auto &spell_ab : cd.abilities) {
                if (spell_ab.ability_type != Ability::SPELL) continue;
                Ability cast_ab = spell_ab;
                cast_ab.source = chosen;
                cast_ab.controller = ctrl;
                // Select a target now (as casting normally would) so the spell
                // doesn't fizzle for want of a target on resolution.
                if (cast_ab.valid_tgts != "N_A" && has_legal_targets(cast_ab, orderer)) {
                    select_target(cast_ab, orderer, ctrl);
                }
                global_coordinator.AddComponent(chosen, cast_ab);
                break;
            }
            orderer->add_to_zone(false, chosen, Zone::STACK);
            game_log("%s casts %s from exile (Dauthi Voidwalker).\n",
                     player_name(ctrl).c_str(), cd.name.c_str());
        }
    } else {
        game_log("No exiled cards with void counters to choose.\n");
    }
    return true;
}

}  // namespace effects
