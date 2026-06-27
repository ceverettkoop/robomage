#include "effects.h"

#include <string>
#include <vector>

#include "../classes/colors.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// True if `e`'s card characteristics include color `c` (explicit Colors: override, else
// the colors in its mana cost). Mirrors the color test used for target legality.
static bool perm_is_color(Entity e, Colors c) {
    if (!global_coordinator.entity_has_component<CardData>(e)) return false;
    const auto &cd = global_coordinator.GetComponent<CardData>(e);
    if (!cd.explicit_colors.empty()) return cd.explicit_colors.count(c) > 0;
    return cd.mana_cost.count(c) > 0;
}

bool permanent_matches_cards_filter(Entity e, const std::string &spec,
                                    Zone::Ownership controller, Entity source) {
    if (spec.empty()) return false;
    if (!is_battlefield_permanent(e)) return false;
    auto &perm = global_coordinator.GetComponent<Permanent>(e);

    // Head token (type or subtype) before the first '.'/'+' separator.
    size_t firstsep = spec.find_first_of(".+");
    std::string head = spec.substr(0, firstsep);
    if (!(head.empty() || head == "Permanent" || head == "Card") &&
        !permanent_has_type(perm, head))
        return false;
    if (firstsep == std::string::npos) return true;

    // Qualifiers after the head, '+'-delimited (Forge writes the first with '.', the
    // rest with '+'; both were captured by find_first_of above).
    std::string rest = spec.substr(firstsep + 1);
    size_t p = 0;
    while (p <= rest.size()) {
        size_t plus = rest.find('+', p);
        if (plus == std::string::npos) plus = rest.size();
        std::string q = rest.substr(p, plus - p);
        p = plus + 1;
        if (q.empty()) continue;
        if (q == "YouCtrl") { if (perm.controller != controller) return false; }
        else if (q == "OppCtrl") { if (perm.controller == controller) return false; }
        else if (q == "Other") { if (e == source) return false; }
        else if (q == "nonLand") { if (permanent_has_type(perm, "Land")) return false; }
        else if (q == "nonToken" || q == "!token") { if (perm.is_token) return false; }
        else if (q == "token") { if (!perm.is_token) return false; }
        else if (q == "ThisTurnEntered") { if (perm.entered_on_turn != cur_game.turn) return false; }
        else if (q == "nonChosenCard") { if (cur_game.chosen_cards.count(e)) return false; }
        else if (q == "White") { if (!perm_is_color(e, WHITE)) return false; }
        else if (q == "Blue")  { if (!perm_is_color(e, BLUE))  return false; }
        else if (q == "Black") { if (!perm_is_color(e, BLACK)) return false; }
        else if (q == "Red")   { if (!perm_is_color(e, RED))   return false; }
        else if (q == "Green") { if (!perm_is_color(e, GREEN)) return false; }
        // Generic non<Type> negation (e.g. nonArtifact, nonCreature): exclude any
        // permanent that carries that type/subtype. The specific "nonLand"/"nonToken"
        // cases above keep their existing handling; this covers every other type name.
        else if (q.size() > 3 && q.compare(0, 3, "non") == 0) {
            if (permanent_has_type(perm, q.substr(3))) return false;
        }
        else return false;  // unknown qualifier: fail closed so a mass effect never over-selects
    }
    return true;
}

// PutCounterAll (Ajani's +2): put CounterNum counters of CounterType on every permanent
// matching the ValidCards$ filter (e.g. a +1/+1 counter on each Cat you control).
bool put_counter_all(Ability &ab, std::shared_ptr<Orderer> orderer) {
    const CounterParams *cp = std::get_if<CounterParams>(&ab.params);
    std::string ctype = (cp && !cp->type.empty()) ? cp->type : "P1P1";
    int n = cp ? cp->count : 1;
    if (n <= 0) return true;

    std::vector<Entity> targets;
    for (auto e : orderer->mEntities)
        if (permanent_matches_cards_filter(e, ab.valid_cards_filter, ab.controller, ab.source))
            targets.push_back(e);

    for (auto e : targets) {
        add_counters(e, ctype, n);
        game_log("Put %d %s counter(s) on %s.\n", n, ctype.c_str(),
                 global_coordinator.GetComponent<Permanent>(e).name.c_str());
    }
    return true;
}

}  // namespace effects
