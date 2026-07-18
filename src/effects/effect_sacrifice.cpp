#include "effects.h"

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../input_logger.h"
#include "../mana_system.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// One player sacrifices a permanent they control matching SacValid$. `sacrificer` is BOTH the
// player making the choice and the controller of the permanent being sacrificed — this is what
// makes an edict an edict (CR 701.16 + 701.16a: to sacrifice a permanent, the player who
// controls it chooses which one). `optional` lets them decline (Optional$ True). Returns the
// chosen entity, or 0 if nothing was sacrificed (no legal permanent, or declined). If the pick
// suspended (parked for the loop top), sets `suspended` and returns 0 mutating nothing — the
// candidate scan and menu rebuild identically on resume.
static Entity sacrifice_one(const Ability &ab, Zone::Ownership sacrificer, bool optional,
                            std::shared_ptr<Orderer> orderer, FrameCtx &ctx, bool &suspended) {
    suspended = false;
    std::vector<Entity> candidates;
    for (auto e : orderer->mEntities) {
        if (!is_battlefield_permanent(e, sacrificer)) continue;
        // SacValid$ filter, evaluated against the sacrificer's permanents (YouCtrl/token/!token,
        // colors, nonLand, …). Filter controller is `sacrificer` so YouCtrl-style qualifiers and
        // negations resolve relative to the player who is sacrificing.
        if (!permanent_matches_filter(e, ab.sac_valid, MatchCtx{sacrificer, ab.source})) continue;
        candidates.push_back(e);
    }

    if (candidates.empty()) return 0;  // no matching permanent → sacrifice nothing

    std::vector<LegalAction> choices;
    if (optional) {
        LegalAction la(PASS_PRIORITY, std::string("Don't sacrifice"));
        la.category = ActionCategory::SACRIFICE_PERMANENT;
        choices.push_back(la);
    }
    for (auto e : candidates) {
        const std::string &nm = global_coordinator.GetComponent<Permanent>(e).name;
        LegalAction la(PASS_PRIORITY, e, "Sacrifice " + nm);
        la.category = ActionCategory::SACRIFICE_PERMANENT;
        choices.push_back(la);
    }
    // CR 701.16a: the SACRIFICER chooses which permanent to sacrifice, and they need not be
    // the player holding priority (Sheoldred's Edict: the caster holds priority while the
    // opponent picks). The ask seats the query on the sacrificer (persisting priority there
    // when it suspends) and restores afterwards, exactly as the old inline swap did.
    int choice = ctx.ask(choices, sacrificer, ab.source);
    if (choice < 0 && decision_suspended()) {
        suspended = true;
        return 0;
    }
    Entity chosen = choices[static_cast<size_t>(choice)].source_entity;
    if (chosen == 0) {
        game_log("%s declines to sacrifice.\n", player_name(sacrificer).c_str());
        return 0;
    }

    std::string name = global_coordinator.GetComponent<Permanent>(chosen).name;
    orderer->add_to_zone(false, chosen, Zone::GRAVEYARD);
    game_log("%s sacrifices %s.\n", player_name(sacrificer).c_str(), name.c_str());
    return chosen;
}

// DB$ Sacrifice (an EFFECT, distinct from a Sac activation cost). By default the ability's
// controller sacrifices a permanent they control matching SacValid$ (Birthing Ritual: a
// creature). With Defined$ Opponent (Sheoldred's Edict: "Each opponent sacrifices a …") it is
// an EDICT — each opponent of the controller chooses and sacrifices one of THEIR matching
// permanents (the choosing player controls the permanent; CR 701.16a). Optional$ True lets the
// sacrificer decline. RememberSacrificed$ True records the sacrificed entity in
// cur_game.remembered_entities so a later sub-ability can read its mana value / identity (and a
// ConditionDefined$ Remembered gate can tell whether anything was sacrificed).
// Self-sacrifice (CR 701.16): the source object's controller sacrifices it. Used for a bare
// DB$ Sacrifice (no SacValid$ / Defined$) — "sacrifice CARDNAME". No choice is involved. Returns
// the sacrificed entity, or 0 if the source was no longer on the battlefield.
static Entity sacrifice_self(const Ability &ab, std::shared_ptr<Orderer> orderer) {
    Entity src = ab.source;
    if (src == 0 || !is_battlefield_permanent(src)) return 0;
    std::string name = global_coordinator.GetComponent<Permanent>(src).name;
    orderer->add_to_zone(false, src, Zone::GRAVEYARD);
    game_log("%s sacrifices %s.\n", player_name(ab.controller).c_str(), name.c_str());
    return src;
}

HandlerResult sacrifice(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    // Non-repeatable preamble (the remembered reset would erase an already-
    // sacrificed entity, and the unless-energy loop would re-prompt) runs once;
    // the persisted rt skips it on resume.
    SacrificeRt local_rt;
    SacrificeRt &rt = ctx.can_suspend() ? ctx.rt<SacrificeRt>() : local_rt;
    if (!rt.pre_done) {
        // RememberSacrificed resets the remembered set to exactly what is sacrificed here, so a
        // downstream ConditionDefined$ Remembered gate sees 0 (declined) or 1 (sacrificed).
        if (ab.remember_sacrificed) cur_game.remembered_entities.clear();

        // UnlessCost$ PayEnergy<N> (Static Prison: "sacrifice CARDNAME unless you pay {E}"). The payer
        // (UnlessPayer$ You ⇒ the controller) may pay N energy to prevent the sacrifice; run_unless_loop
        // returns false when paid (don't sacrifice) and true when declined / unaffordable (sacrifice).
        // (Still a blocking prompt this batch — Shape B unless-loops convert with the pool batch.)
        if (ab.unless_cost_is_energy) {
            Zone::Ownership payer = (ab.unless_payer != Zone::UNKNOWN) ? ab.unless_payer : ab.controller;
            game_log("%s may pay %zu energy to avoid sacrificing:\n", player_name(payer).c_str(),
                     ab.unless_generic_cost);
            if (!run_unless_loop(ab.unless_generic_cost, payer, orderer, ab.source, UnlessPayKind::ENERGY))
                return HandlerResult::DONE_RUN_SUBS;  // paid — nothing is sacrificed
        }
        rt.pre_done = true;
    }

    // No SacValid$ filter and not an edict ⇒ self-sacrifice (sacrifice CARDNAME), CR 701.16.
    if (ab.sac_valid.empty() && !ab.defined_each_opponent) {
        Entity sacked = sacrifice_self(ab, orderer);
        if (ab.remember_sacrificed && sacked) cur_game.remembered_entities.push_back(sacked);
        return HandlerResult::DONE_RUN_SUBS;
    }

    // Defined$ Opponent — edict: each opponent sacrifices on their own. CR 109.5 / 102.1: in a
    // two-player game "each opponent" is the controller's single opponent. The edict is mandatory
    // for an opponent who controls a matching permanent (they must pick one), so it is never
    // optional regardless of Optional$.
    if (ab.defined_each_opponent) {
        Zone::Ownership opp = (ab.controller == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;
        // sac_count > 1: the sacrificer chooses and sacrifices that many permanents, one at a
        // time (Annihilator N, CR 702.85b). Each iteration re-gathers candidates, so the choices
        // shrink as permanents leave; sacrifice_one returns 0 when none remain, so a player who
        // controls fewer than sac_count permanents simply sacrifices all they have. The
        // iteration index persists in rt so a resume re-enters the suspended pick.
        for (; rt.iter < ab.sac_count; ++rt.iter) {
            bool suspended = false;
            Entity sacked = sacrifice_one(ab, opp, /*optional=*/false, orderer, ctx, suspended);
            if (suspended) return HandlerResult::SUSPENDED;
            if (ab.remember_sacrificed && sacked) cur_game.remembered_entities.push_back(sacked);
            if (sacked == 0) break;
        }
        return HandlerResult::DONE_RUN_SUBS;
    }

    for (; rt.iter < ab.sac_count; ++rt.iter) {
        bool suspended = false;
        Entity sacked = sacrifice_one(ab, ab.controller, ab.optional_choice, orderer, ctx, suspended);
        if (suspended) return HandlerResult::SUSPENDED;
        if (ab.remember_sacrificed && sacked) cur_game.remembered_entities.push_back(sacked);
        if (sacked == 0) break;
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
