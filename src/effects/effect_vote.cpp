#include "effects.h"

#include <algorithm>
#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"
#include "../game_queries.h"
#include "../input_logger.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// SP$/AB$ Vote (Council's Judgment) — will-of-the-council (CR 701.32).
//
// The printed effect: "Starting with you, each player votes for a nonland permanent you don't
// control. Exile each permanent with the most votes or tied for most votes." Full multiplayer
// voting (CR 701.32 / the 800-series) is out of scope: the engine is two-player only (see
// CLAUDE.md). Per the user-directed two-player reduction, the vote collapses to "the spell's
// controller CHOOSES one permanent matching VoteCard$, and it is exiled." (You vote first, and
// with only one opponent any tie is the single object you picked, so a one-pick choice is the
// correct two-player outcome.)
//
// This is a CHOICE, not a target (CR 115.10a/701.32): the permanent is selected on resolution,
// not at cast time, and an opponent's hexproof/shroud does NOT protect it (those are targeting
// restrictions only). We therefore enumerate candidates with permanent_matches_filter and never
// run is_legal_target on them.
//
// The chosen permanent is placed in cur_game.remembered_entities so the VoteSubAbility$ DBExile
// (DB$ ChangeZone | Defined$ Remembered | Origin$ Battlefield | Destination$ Exile), parsed into
// this ability's subabilities, exiles it via the standard change_zone resolution. Returning true
// chains that subability. If no permanent matches the filter (the opponent controls no nonland
// permanents), the spell still resolves and does nothing.
bool vote(Ability &ab, std::shared_ptr<Orderer> orderer) {
    PendingDecisionScope pending_scope(ab.source);
    // Translate Forge's "YouDontCtrl" (a permanent you don't control) into the evaluator's
    // OppCtrl, exactly as Ability::is_legal_target does, so "Permanent.nonLand+YouDontCtrl"
    // matches the opponent's nonland permanents. The controller is the "you" reference.
    std::string spec = ab.vote_card_filter.empty() ? std::string("Permanent.nonLand") : ab.vote_card_filter;
    for (size_t pos = spec.find("YouDontCtrl"); pos != std::string::npos;
         pos = spec.find("YouDontCtrl", pos))
        spec.replace(pos, std::string("YouDontCtrl").size(), "OppCtrl");

    MatchCtx ctx;
    ctx.controller = ab.controller;
    ctx.source = ab.source;

    std::vector<Entity> candidates;
    for (auto e : orderer->mEntities) {
        if (!is_battlefield_permanent(e)) continue;  // control restriction is in the filter
        if (permanent_matches_filter(e, spec, ctx)) candidates.push_back(e);
    }

    // Clear any remembered entities so only the voted-for permanent is fed to the
    // Defined$ Remembered DBExile sub-ability.
    cur_game.remembered_entities.clear();

    if (candidates.empty()) {
        game_log("Will of the Council: no eligible permanent to vote for; nothing is exiled.\n");
        return true;  // still resolves; chain subs (DBExile finds nothing remembered)
    }

    // Present the controller (the voter) the choice over the eligible permanents. This is a
    // CHOOSE_CARD-style decision (not a target), so it works even against hexproof/shroud.
    bool prev_priority = cur_game.player_a_has_priority;
    cur_game.player_a_has_priority = (ab.controller == Zone::PLAYER_A);

    std::vector<LegalAction> picks;
    for (auto e : candidates) {
        const std::string &nm = global_coordinator.GetComponent<Permanent>(e).name;
        LegalAction la(PASS_PRIORITY, e, "Vote to exile " + nm);
        la.category = ActionCategory::CHOOSE_CARD;
        la.card_is_public = true;
        picks.push_back(la);
    }

    game_log("%s votes for a permanent to exile:\n", player_name(ab.controller).c_str());
    int choice = InputLogger::instance().get_input(picks);
    cur_game.player_a_has_priority = prev_priority;

    Entity chosen = candidates[static_cast<size_t>(choice)];
    cur_game.remembered_entities.push_back(chosen);
    game_log("%s votes for %s.\n", player_name(ab.controller).c_str(),
             global_coordinator.GetComponent<Permanent>(chosen).name.c_str());
    return true;  // chain the VoteSubAbility$ DBExile (Defined$ Remembered → Exile)
}

}  // namespace effects
