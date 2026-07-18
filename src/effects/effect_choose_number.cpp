#include "effects.h"

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../input_logger.h"

extern Game cur_game;

namespace effects {

// DB$ ChooseNumber: the resolving controller chooses an integer in [0, Max]. Max comes from
// ab.dynamic_amount_expr — the runtime Count$ expression named by Max$ (Wrath of the Skies:
// Count$YourCountersEnergy = the controller's current energy, capping how much {E} they may
// choose to pay). The pick is stored in cur_game.chosen_number so a chained sub-ability can
// read it via Count$ChosenNumber (here the DestroyAll's mana-value bound Y and its
// PayEnergy<Y> unless-cost). General over any "choose a number up to N" effect.
HandlerResult choose_number(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    PendingDecisionScope pending_scope(ab.source);
    int max = 0;
    if (!ab.dynamic_amount_expr.empty())
        max = static_cast<int>(
            evaluate_dynamic_amount(ab.dynamic_amount_expr, ab.controller, orderer, ab.target));
    if (max < 0) max = 0;

    // Hand priority to the choosing player so the input is logged against the right seat,
    // mirroring the optional/unless resolution prompts (run_unless_loop).
    bool prev_priority = cur_game.player_a_has_priority;
    cur_game.player_a_has_priority = (ab.controller == Zone::PLAYER_A);

    std::vector<LegalAction> choices;
    for (int n = 0; n <= max; n++) {
        LegalAction la(PASS_PRIORITY, std::to_string(n));
        la.category = ActionCategory::CHOOSE_X;
        la.option_ordinal = n;  // the chosen number
        choices.push_back(la);
    }
    int choice = InputLogger::instance().get_input(choices);
    cur_game.player_a_has_priority = prev_priority;

    if (choice < 0) choice = 0;
    if (choice > max) choice = max;
    cur_game.chosen_number = choice;
    game_log("%s chooses %d.\n", player_name(ab.controller).c_str(), choice);
    return HandlerResult::DONE_RUN_SUBS;
}

// Max$ — the SVar (resolved at parse time to a runtime Count$ expression) bounding the choice;
// stored in dynamic_amount_expr and evaluated at resolution. ListTitle$ is cosmetic prompt prose.
bool parse_choose_number(Ability &ab, const std::string &key, const std::string &value) {
    if (key == "Max") { ab.dynamic_amount_expr = value; return true; }
    if (key == "ListTitle") return true;  // cosmetic prompt text
    return false;
}

}  // namespace effects
