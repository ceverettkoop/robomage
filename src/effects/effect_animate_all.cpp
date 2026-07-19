#include "effects.h"

#include <algorithm>
#include <string>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;

namespace effects {

// AB$ AnimateAll (Shadowspear: "{1}: Permanents your opponents control lose hexproof and
// indestructible until end of turn"). A mass, until-end-of-turn continuous effect applied to
// EVERY battlefield permanent matching ValidCards$ (here Permanent.OppCtrl — opponents'
// permanents). Two general directions:
//   * RemoveKeywords$ — record the keyword(s) in the permanent's removed_keywords_eot set, so the
//     effective-keyword accessors (permanent_has_keyword / is_indestructible) treat them as absent
//     for the rest of the turn (CR 613 layer 6 keyword removal). The loss has real consequences:
//     a hexproof creature becomes targetable by its controller's opponents again (CR 702.11b
//     enforced through is_legal_target), and an indestructible permanent can be destroyed or die
//     to lethal damage again (CR 702.12b enforced through the lethal-damage SBA / Destroy effects).
//   * AddKeyword$/Keywords$ — grant the keyword(s) until end of turn via the creature EOT bucket
//     (cr.eot_keywords), the same machinery PumpAll uses (the add direction is wired for
//     generality; Shadowspear exercises only the removal direction).
// The ValidCards$ filter is matched through permanent_matches_filter so YouCtrl/OppCtrl controller
// scoping and the full qualifier grammar come for free. Cleared at the cleanup step (514.2).
HandlerResult animate_all(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &fctx) {
    MatchCtx ctx{ab.controller, ab.source};
    int affected = 0;
    for (auto e : orderer->mEntities) {
        if (!is_battlefield_permanent(e)) continue;
        if (!permanent_matches_filter(e, ab.valid_cards_filter, ctx)) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(e);

        // Removal direction (CR 613 layer 6): suppress the named keyword(s) until end of turn.
        for (const auto &kw : ab.remove_keywords)
            perm.removed_keywords_eot.insert(kw);

        // Grant direction (until end of turn) — creatures only (keywords live on Creature).
        if (!ab.add_keywords.empty() && global_coordinator.entity_has_component<Creature>(e)) {
            auto &cr = global_coordinator.GetComponent<Creature>(e);
            for (const auto &kw : ab.add_keywords) {
                if (std::find(cr.eot_keywords.begin(), cr.eot_keywords.end(), kw) == cr.eot_keywords.end())
                    cr.eot_keywords.push_back(kw);
                if (std::find(cr.keywords.begin(), cr.keywords.end(), kw) == cr.keywords.end())
                    cr.keywords.push_back(kw);
            }
        }
        ++affected;
    }
    if (!ab.remove_keywords.empty()) {
        std::string list;
        for (size_t i = 0; i < ab.remove_keywords.size(); ++i)
            list += (i ? ", " : "") + ab.remove_keywords[i];
        game_log("%d permanent(s) lose %s until end of turn.\n", affected, list.c_str());
    }
    return HandlerResult::DONE_RUN_SUBS;
}

// Parse hook for AB$ AnimateAll. Claims RemoveKeywords$ (the removal list) and, when the ability
// is an AnimateAll, ValidCards$ (the affected-permanents filter) and AddKeyword$/Keywords$ (the
// grant list). Forge separates multiple keywords with " & " (e.g. "Hexproof & Indestructible").
static void split_keywords(const std::string &value, std::vector<std::string> &out) {
    size_t start = 0;
    while (start < value.size()) {
        size_t amp = value.find(" & ", start);
        std::string kw = (amp == std::string::npos) ? value.substr(start)
                                                     : value.substr(start, amp - start);
        size_t b = kw.find_first_not_of(" \t");
        size_t e = kw.find_last_not_of(" \t");
        if (b != std::string::npos) out.push_back(kw.substr(b, e - b + 1));
        if (amp == std::string::npos) break;
        start = amp + 3;
    }
}

bool parse_animate_all(Ability &ab, const std::string &key, const std::string &value) {
    if (ab.category != "AnimateAll") return false;
    if (key == "RemoveKeywords") { split_keywords(value, ab.remove_keywords); return true; }
    if (key == "AddKeyword" || key == "Keywords") { split_keywords(value, ab.add_keywords); return true; }
    if (key == "ValidCards") { ab.valid_cards_filter = value; return true; }
    return false;
}

}  // namespace effects
