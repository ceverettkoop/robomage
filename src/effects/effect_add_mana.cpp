#include "effects.h"

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../input_logger.h"
#include "../mana_system.h"

extern Game cur_game;

namespace effects {

bool add_mana(Ability &ab, std::shared_ptr<Orderer> orderer) {
    (void)orderer;
    // Non-mana-ability that adds mana on resolution (Dark Ritual, Lion's Eye Diamond)
    Zone::Ownership mana_controller = ab.controller;
    size_t mana_amount = (ab.amount > 0) ? ab.amount : 1;
    Colors mana_color = ab.color;
    if (!ab.mana_choices.empty()) {
        // Prompt player to choose a color (e.g. LED: "Any" → 3 mana of one chosen color)
        bool prev_priority = cur_game.player_a_has_priority;
        cur_game.player_a_has_priority = (mana_controller == Zone::PLAYER_A);
        std::vector<LegalAction> color_actions;
        for (auto c : ab.mana_choices) {
            std::string desc = "Add " + std::to_string(mana_amount) + "{" + mana_symbol(c) + "}";
            LegalAction la(PASS_PRIORITY, std::string(desc));
            la.category = ActionCategory::OTHER_CHOICE;
            color_actions.push_back(la);
        }
        int choice = InputLogger::instance().get_input(color_actions);
        mana_color = ab.mana_choices[static_cast<size_t>(choice)];
        cur_game.player_a_has_priority = prev_priority;
    }
    ::add_mana(mana_controller, mana_color, mana_amount);
    game_log("%s adds %zu{%s}\n", player_name(mana_controller).c_str(), mana_amount, mana_symbol(mana_color).c_str());
    return true;
}

}  // namespace effects
