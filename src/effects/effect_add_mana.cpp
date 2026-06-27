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
            la.category = ActionCategory::CHOOSE_MANA_COLOR;
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

bool parse_add_mana(Ability &ab, const std::string &key, const std::string &value) {
    // AB$ ManaReflected (Mox Amber): a mana ability that produces "one mana of any color among"
    // the permanents its Valid$ filter matches (CR 605). Valid$ holds the (controller-scoped)
    // permanent filter whose colors are reflected; ColorOrType$ Color / ReflectProperty$ Is are
    // the only supported mode (reflect the colors the matching permanents actually are), so they
    // are validated and consumed here rather than warned as unrecognized.
    if (ab.category == "ManaReflected") {
        if (key == "Valid") {
            ab.reflected_mana_filter = value;
            // ManaReflected adds exactly one mana of the chosen color ("Add one mana of any
            // color among …"); set the amount here so the off-stack mana add and its narrative
            // produce 1, not the default 0.
            if (ab.amount == 0) ab.amount = 1;
            return true;
        }
        if (key == "ColorOrType" || key == "ReflectProperty") {
            return true;  // Color / Is — the implemented mode; consumed, no extra state needed
        }
    }
    if (key != "Produced") return false;
    // A mana ability may specify how much mana it makes via Amount$ (e.g. Ancient Tomb:
    // Produced$ C | Amount$ 2). Amount$ is consumed separately and sets ab.amount, but the
    // two params can appear in either order on the line, so only fall back to the default of
    // 1 when no explicit amount has been parsed yet (ab.amount still at its 0 default).
    size_t default_amount = (ab.amount > 0) ? ab.amount : 1;
    if (value == "Any") {
        // Birds of Paradise: produce any color
        ab.mana_choices = {WHITE, BLUE, BLACK, RED, GREEN};
        ab.amount = default_amount;
    } else if (value.find("Combo") != std::string::npos) {
        // Noble Hierarch: "Combo W U G" — space-separated colors after "Combo"
        size_t combo_pos = value.find("Combo");
        size_t start = combo_pos + 5;  // skip "Combo"
        while (start < value.size() && value[start] == ' ') start++;
        for (size_t ci = start; ci <= value.size(); ci++) {
            if (ci == value.size() || value[ci] == ' ') {
                if (ci > start) {
                    char tok = value[start];
                    if      (tok == 'W') ab.mana_choices.push_back(WHITE);
                    else if (tok == 'U') ab.mana_choices.push_back(BLUE);
                    else if (tok == 'B') ab.mana_choices.push_back(BLACK);
                    else if (tok == 'R') ab.mana_choices.push_back(RED);
                    else if (tok == 'G') ab.mana_choices.push_back(GREEN);
                    else if (tok == 'C') ab.mana_choices.push_back(COLORLESS);
                }
                start = ci + 1;
            }
        }
        ab.amount = default_amount;
    } else {
        for (char c : value) {
            if      (c == 'W') { ab.color = WHITE;     break; }
            else if (c == 'U') { ab.color = BLUE;      break; }
            else if (c == 'B') { ab.color = BLACK;     break; }
            else if (c == 'R') { ab.color = RED;       break; }
            else if (c == 'G') { ab.color = GREEN;     break; }
            else if (c == 'C') { ab.color = COLORLESS; break; }
        }
        ab.amount = default_amount;
    }
    return true;
}

}  // namespace effects
