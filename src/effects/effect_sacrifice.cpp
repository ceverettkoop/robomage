#include "effects.h"

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// DB$ Sacrifice (an EFFECT, distinct from a Sac activation cost): the ability's controller
// sacrifices a permanent they control matching SacValid$ (Birthing Ritual: a creature).
// Optional$ True lets them decline. RememberSacrificed$ True records the sacrificed entity in
// cur_game.remembered_entities so a later sub-ability can read its mana value / identity (and a
// ConditionDefined$ Remembered gate can tell whether anything was sacrificed).
bool sacrifice(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // Type filter from SacValid$ (strip any ".YouCtrl"-style qualifier — you only ever
    // sacrifice your own permanents). "Card"/"Permanent" mean any permanent you control.
    std::string type_filter = ab.sac_valid;
    size_t dot = type_filter.find('.');
    if (dot != std::string::npos) type_filter = type_filter.substr(0, dot);
    if (type_filter == "Card" || type_filter == "Permanent") type_filter.clear();

    // RememberSacrificed resets the remembered set to exactly what is sacrificed here, so a
    // downstream ConditionDefined$ Remembered gate sees 0 (declined) or 1 (sacrificed).
    if (ab.remember_sacrificed) cur_game.remembered_entities.clear();

    std::vector<Entity> candidates;
    for (auto e : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        if (global_coordinator.GetComponent<Zone>(e).location != Zone::BATTLEFIELD) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(e);
        if (perm.controller != ab.controller) continue;
        if (!type_filter.empty()) {
            bool match = false;
            for (auto &t : perm.types) if (t.name == type_filter) { match = true; break; }
            if (!match) continue;
        }
        candidates.push_back(e);
    }

    if (candidates.empty()) return true;  // nothing to sacrifice; chain subabilities

    // Choose which to sacrifice (optional → may decline).
    std::vector<LegalAction> choices;
    if (ab.optional_choice) {
        LegalAction la(PASS_PRIORITY, std::string("Don't sacrifice"));
        la.category = ActionCategory::OTHER_CHOICE;
        choices.push_back(la);
    }
    for (auto e : candidates) {
        const std::string &nm = global_coordinator.GetComponent<Permanent>(e).name;
        LegalAction la(PASS_PRIORITY, e, "Sacrifice " + nm);
        la.category = ActionCategory::OTHER_CHOICE;
        choices.push_back(la);
    }
    int choice = InputLogger::instance().get_input(choices);
    Entity chosen = choices[static_cast<size_t>(choice)].source_entity;
    if (chosen == 0) {
        game_log("%s declines to sacrifice.\n", player_name(ab.controller).c_str());
        return true;
    }

    std::string name = global_coordinator.GetComponent<Permanent>(chosen).name;
    orderer->add_to_zone(false, chosen, Zone::GRAVEYARD);
    game_log("%s sacrifices %s.\n", player_name(ab.controller).c_str(), name.c_str());
    if (ab.remember_sacrificed) cur_game.remembered_entities.push_back(chosen);
    return true;
}

}  // namespace effects
