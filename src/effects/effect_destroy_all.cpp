#include "effects.h"

#include <cstdio>
#include <string>
#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/types.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../svar_eval.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

HandlerResult destroy_all(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    // Destroy all permanents matching the filter (e.g. Meltdown: "Artifact.cmcLEX")
    const std::string &filter = ab.valid_cards_filter;

    const DestroyAllParams *dp = std::get_if<DestroyAllParams>(&ab.params);

    // UnlessCost$ PayEnergy<N> (Wrath of the Skies: with UnlessSwitched$ True). N
    // (Count$ChosenNumber) is the amount of energy chosen earlier this resolution.
    if (dp && !dp->energy_unless_expr.empty()) {
        int n = static_cast<int>(
            evaluate_dynamic_amount(dp->energy_unless_expr, ab.controller, orderer, ab.target));
        Entity ctrl_entity = get_player_entity(ab.controller);
        auto &pl = global_coordinator.GetComponent<Player>(ctrl_entity);
        if (dp->energy_unless_switched) {
            // Switched: the spell's controller pays N {E} as a single cost of resolution and the
            // destroy proceeds ONLY IF it is paid (CR 122.1c). Wrath of the Skies: pay the chosen
            // amount, then destroy each artifact/creature/enchantment with MV <= that amount.
            bool paid = pay_energy(pl, n);  // n <= 0 is a trivially-payable no-op (returns true)
            if (n > 0)
                game_log("%s pays %d energy.\n", player_name(ab.controller).c_str(), n);
            if (!paid) return HandlerResult::DONE_RUN_SUBS;  // couldn't pay -> nothing is destroyed
        } else {
            // Non-switched genuine "destroy each X unless its controller pays {E}": the payment is
            // per-permanent by EACH affected permanent's controller, and paying PREVENTS that
            // permanent's destruction — a shape this single-payment path can't model. No shipping
            // card uses it. Fail closed (warn once) rather than incorrectly charge the spell's
            // controller and destroy regardless; the destroy below then proceeds as the
            // no-one-paid default.
            static bool warned = false;
            if (!warned) {
                warned = true;
                printf("WARNING: non-switched energy UnlessCost in DestroyAll is unsupported "
                       "(per-permanent unless-payment not modeled); destroying without the "
                       "optional cost\n");
            }
        }
    }

    // The filter is matched against each permanent's live characteristics by the shared
    // matcher: types, nonLand, control qualifiers and mana value (a token's is 0, CR 111.4 /
    // 202.3). Legacy cmcLEX keys off the X paid at cast (Meltdown); a dynamic cmc<op><SVar>
    // bound (Wrath of the Skies' cmcLEY = energy paid, Blast Zone's cmcEQY = its charge
    // counters) is resolved from the parsed Count$ expression here and supplied as the bound.
    MatchCtx mctx;
    mctx.controller = ab.controller;
    mctx.source = ab.source;
    if (dp && !dp->cmc_expr.empty()) {
        mctx.cmc_bound = static_cast<int>(
            evaluate_dynamic_amount(dp->cmc_expr, ab.controller, orderer, ab.target, ab.source));
        mctx.cmc_op = dp->cmc_op.empty() ? "LE" : dp->cmc_op;
    }
    std::vector<Entity> to_destroy;
    for (auto e : orderer->mEntities)
        if (permanent_matches_filter(e, filter, mctx)) to_destroy.push_back(e);
    for (auto e : to_destroy) {
        std::string ename = entity_name(e);
        // CR 702.12b: an indestructible permanent can't be destroyed by a mass-destroy effect.
        if (is_indestructible(e)) {
            game_log("%s is indestructible — not destroyed\n", ename.c_str());
            continue;
        }
        orderer->add_to_zone(false, e, Zone::GRAVEYARD);
        game_log("%s is destroyed\n", ename.c_str());
    }
    return HandlerResult::DONE_RUN_SUBS;
}

bool parse_destroy_all(Ability &ab, const std::string &key, const std::string &value) {
    if (key != "ValidCards") return false;
    // Shared by DestroyAll / SacrificeAll / PutCounterAll — a plain member so it can
    // coexist with each effect's own params block.
    ab.valid_cards_filter = value;
    return true;
}

}  // namespace effects
