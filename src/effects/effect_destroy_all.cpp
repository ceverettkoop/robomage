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
    std::string filter = ab.valid_cards_filter;
    bool filter_artifact = filter.find("Artifact") != std::string::npos;
    bool filter_creature = filter.find("Creature") != std::string::npos;
    bool filter_enchantment = filter.find("Enchantment") != std::string::npos;
    // "nonLand" exclusion (Blast Zone: "each nonland permanent …"). Without honouring it, a
    // filter that names no positive card type — "Permanent.nonLand" — falls to the
    // match-anything default below and would wrongly destroy lands too.
    bool filter_nonland = filter.find("nonLand") != std::string::npos;

    const DestroyAllParams *dp = std::get_if<DestroyAllParams>(&ab.params);

    // UnlessCost$ PayEnergy<N> (Wrath of the Skies: with UnlessSwitched$ True). N
    // (Count$ChosenNumber) is the amount of energy chosen earlier this resolution.
    if (dp && !dp->energy_unless_expr.empty()) {
        int n = static_cast<int>(
            evaluate_dynamic_amount(dp->energy_unless_expr, ab.controller, orderer, ab.target));
        Entity ctrl_entity =
            (ab.controller == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
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

    // CMC filter bound. Legacy cmcLEX keys off the X paid at cast (Meltdown). A dynamic
    // cmcLE<SVar> bound (Wrath of the Skies' cmcLEY, Y = Count$ChosenNumber = energy paid) is
    // resolved from the parsed Count$ expression at resolution.
    // cmc_bound is the mana-value threshold; cmc_op is how a card's MV is compared to it. The
    // legacy cmcLEX / numeric path is "<=" (LE); a dynamic cmc<op><SVar> filter (Blast Zone's
    // cmcEQY = "MV equal to the charge-counter count") carries its own operator in dp->cmc_op.
    int cmc_bound = -1;
    std::string cmc_op = "LE";
    if (dp && !dp->cmc_expr.empty()) {
        cmc_bound = static_cast<int>(
            evaluate_dynamic_amount(dp->cmc_expr, ab.controller, orderer, ab.target, ab.source));
        if (!dp->cmc_op.empty()) cmc_op = dp->cmc_op;
    } else if (filter.find("cmcLEX") != std::string::npos) {
        cmc_bound = static_cast<int>(cur_game.x_paid);
    }
    std::vector<Entity> to_destroy;
    for (auto e : orderer->mEntities) {
        if (!is_battlefield_permanent(e)) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(e);
        // Type filter
        bool type_match = (!filter_artifact && !filter_creature && !filter_enchantment);
        for (auto &t : perm.types) {
            if (filter_artifact && t.kind == TYPE && t.name == "Artifact") type_match = true;
            if (filter_creature && t.kind == TYPE && t.name == "Creature") type_match = true;
            if (filter_enchantment && t.kind == TYPE && t.name == "Enchantment") type_match = true;
        }
        if (!type_match) continue;
        // nonLand exclusion (CR 110.4a): "each nonland permanent" never hits a land.
        if (filter_nonland) {
            bool is_land = false;
            for (auto &t : perm.types)
                if (t.kind == TYPE && t.name == "Land") { is_land = true; break; }
            if (is_land) continue;
        }
        // CMC filter, comparing the card's mana value to the bound with the parsed operator
        // (EQ for Blast Zone, LE for the legacy cmcLEX / Wrath-of-the-Skies path).
        if (cmc_bound >= 0 && global_coordinator.entity_has_component<CardData>(e)) {
            int cmc = card_mana_value(global_coordinator.GetComponent<CardData>(e));
            if (!apply_svar_op(cmc, cmc_op, cmc_bound)) continue;
        }
        to_destroy.push_back(e);
    }
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
