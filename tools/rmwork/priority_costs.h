#ifndef RMWORK_PRIORITY_COSTS_H
#define RMWORK_PRIORITY_COSTS_H

#include <string>
#include <vector>

#include "../../src/agents/decision_view.h"
#include "card_book.h"

// ── What each way of a priority decision costs ──────────────────────────────────────────────────
//
// A priority menu's rows (CAST_SPELL, PLAY_LAND, ACTIVATE_ABILITY, TURN_FACE_UP, COMPANION, MANA_*)
// name their card (`source`, `source_card`) and say how it is cast in their description ("Cast Daze
// (alternate cost)") and `variant` mask, but not what that way costs. The Table's popup and hover say
// it, read from the viewer's stream alone (ImGui-free):
//   - a cast for the card's mana cost (CARDS' `mana_cost`; a back face's or an Adventure's from the
//     back face), nothing for a cast without paying its mana cost, {3} cast face down (CR 702.37c) or
//     for a companion (CR 702.139a);
//   - a keyword's cost (flashback, escape, evoke, warp, impending, overload, bestow, suspend; turning a
//     permanent face up: morph, megamorph, disguise) from the oracle text's line of that keyword
//     ("Escape—{G}{G}{U}{U}, Exile five other cards from your graveyard.");
//   - an alternative cost (CR 118.9) from the oracle text's "You may … rather than pay this spell's
//     mana cost" ("return an Island you control to its owner's hand");
//   - an activated ability's cost from CARDS' `card_ability.cost` (the row's `ab_card` / `ab_index`), in
//     the script's cost syntax ("T PayLife<1> Sac<1/CARDNAME>": {T}, pay 1 life, sacrifice it).
// A land is played for nothing; a mana row's description says it all.

namespace rmwork {

struct WayCost {
        std::vector<std::string> pips;  // mana symbols as the cards print them: "1", "U", "W/U", "T" (tap)
        std::string text;               // the rest: "pay 1 life, sacrifice it"; "" for none
        bool empty() const { return pips.empty() && text.empty(); }
};

// What menu row `row` of the priority decision `v` costs (`book`: `v`'s stream's CARDS faces).
WayCost way_cost(const agents::DecisionView &v, const CardBook &book, size_t row);

// An activated ability's cost in the script's syntax ("T PayLife<1> Sac<1/CARDNAME>", "1 R",
// "SubCounter<3/LOYALTY>") as pips and words.
WayCost ability_cost(const std::string &script_cost);

// The cost of keyword `keyword` from a card's oracle text: its line's pips and words ("Flashback—
// Sacrifice a creature." gives "sacrifice a creature"); empty when no line starts with it.
WayCost keyword_cost(const std::string &oracle, const std::string &keyword);

// An alternative cost from a card's oracle text: what "You may … rather than pay this spell's mana cost"
// asks ("pay 1 life and exile a blue card from your hand"); "" when it has none.
std::string alternative_cost(const std::string &oracle);

// The cost in text: "{1}{U}", "{T}, pay 1 life, sacrifice it", "return an Island you control …".
std::string way_cost_text(const WayCost &c);

// A row's description with its cost: "Cast Lightning Bolt {R}", "Cast Daze (alternate cost) — return an
// Island you control to its owner's hand"; the description alone when the cost is empty.
std::string way_label(const std::string &description, const WayCost &c);

}  // namespace rmwork

#endif /* RMWORK_PRIORITY_COSTS_H */
