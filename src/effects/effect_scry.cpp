#include "effects.h"

#include <algorithm>
#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Scry N (CR 701.18): the chosen player looks at the top N cards of their library, then
// may put any number of them on the bottom of their library and the rest back on top in
// any order. Modeled as a per-card top-or-bottom choice from the top down; cards left on
// top keep their relative order (the optional reorder-among-kept is omitted as a
// simplification). The player is ValidTgts$ Player (ab.target); absent a target the
// source's controller scries. After scrying, any SubAbility$ chains with the same target
// (Kozilek's Command: "scries X, then draws a card" — DBDraw with Defined$ ParentTarget).
bool scry(Ability &ab, std::shared_ptr<Orderer> orderer) {
    Zone::Ownership owner;
    if (ab.target != 0 && global_coordinator.entity_has_component<Player>(ab.target))
        owner = (ab.target == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
    else if (global_coordinator.entity_has_component<Permanent>(ab.source))
        owner = global_coordinator.GetComponent<Permanent>(ab.source).controller;
    else
        owner = global_coordinator.GetComponent<Zone>(ab.source).owner;

    size_t num = ab.amount;
    if (!ab.dynamic_amount_expr.empty())
        num = evaluate_dynamic_amount(ab.dynamic_amount_expr, owner, orderer, ab.target);
    if (num == 0) return true;

    std::vector<Entity> lib = orderer->get_library_contents(owner);
    // Sort so lib[0] is the actual top card.
    std::sort(lib.begin(), lib.end(), [](Entity a, Entity b) {
        return global_coordinator.GetComponent<Zone>(a).distance_from_top <
               global_coordinator.GetComponent<Zone>(b).distance_from_top;
    });
    if (lib.size() > num) lib.resize(num);
    if (lib.empty()) {
        game_log("%s's library is empty — nothing to scry.\n", player_name(owner).c_str());
        return true;
    }

    game_log("%s scries %zu.\n", player_name(owner).c_str(), lib.size());
    // Decide top-to-bottom per card. Bottomed cards move to the library bottom; cards left
    // on top stay in place (their distance_from_top compacts as bottomed cards leave).
    for (Entity card : lib) {
        auto &cd = global_coordinator.GetComponent<CardData>(card);
        game_log_private(owner, "Scry: top card is %s\n", cd.name.c_str());
        std::vector<LegalAction> scry_actions = {
            LegalAction(PASS_PRIORITY, card, std::string("Keep on top")),
            LegalAction(PASS_PRIORITY, card, std::string("Put on bottom")),
        };
        scry_actions[0].category = ActionCategory::TOP_LIBRARY;
        scry_actions[1].category = ActionCategory::TOP_LIBRARY;
        int choice = InputLogger::instance().get_input(scry_actions);
        if (choice == 1) {
            orderer->add_to_zone(true, card, Zone::LIBRARY);
            game_log("%s puts a card on the bottom of their library.\n", player_name(owner).c_str());
        }
    }
    return true;
}

}  // namespace effects
