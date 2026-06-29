#include "effects.h"

#include <vector>

#include "../cli_output.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;

namespace effects {

// DB$/AB$ UntapAll (Paradox Engine: "untap all nonland permanents you control"). The mass
// counterpart of single-target Untap (CR 701.21): untap every battlefield permanent matching
// ValidCards$ (e.g. "Permanent.YouCtrl+nonLand"). The filter is matched through the shared
// permanent_matches_filter so YouCtrl/OppCtrl controller scoping and the full qualifier grammar
// (nonLand, subtypes, colors, …) come for free — general over any "untap all <filter>" effect,
// not special-cased to Paradox Engine. ab.valid_cards_filter is populated by parse_destroy_all
// (the shared ValidCards$ hook).
bool untap_all(Ability &ab, std::shared_ptr<Orderer> orderer) {
    MatchCtx ctx{ab.controller, ab.source};
    for (auto e : orderer->mEntities) {
        if (!is_battlefield_permanent(e)) continue;
        if (!permanent_matches_filter(e, ab.valid_cards_filter, ctx)) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(e);
        if (!perm.is_tapped) continue;
        perm.is_tapped = false;
        game_log("%s untaps\n", perm.name.c_str());
    }
    return true;
}

}  // namespace effects
