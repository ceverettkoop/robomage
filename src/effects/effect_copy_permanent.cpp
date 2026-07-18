#include "effects.h"

#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../components/token.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Builds the Token characteristics for a copy of permanent `src`. A token source carries
// its own Token component (the cleanest copyable snapshot, 707.2); a nontoken permanent is
// reconstructed from its CardData + Creature so the copy gets the printed (not pumped /
// countered) P/T and its keywords/types.
static Token copyable_token_of(Entity src) {
    if (global_coordinator.entity_has_component<Token>(src))
        return global_coordinator.GetComponent<Token>(src);
    Token tok;
    if (global_coordinator.entity_has_component<Permanent>(src)) {
        auto &perm = global_coordinator.GetComponent<Permanent>(src);
        tok.name = perm.name;
        tok.types = perm.types;
    }
    if (global_coordinator.entity_has_component<CardData>(src)) {
        auto &cd = global_coordinator.GetComponent<CardData>(src);
        if (tok.name.empty()) tok.name = cd.name;
        tok.keywords = cd.keywords;
    }
    if (global_coordinator.entity_has_component<Creature>(src)) {
        auto &cr = global_coordinator.GetComponent<Creature>(src);
        tok.power = static_cast<uint32_t>(cr.base_power < 0 ? 0 : cr.base_power);
        tok.toughness = static_cast<uint32_t>(cr.base_toughness < 0 ? 0 : cr.base_toughness);
        if (tok.keywords.empty()) tok.keywords = cr.keywords;
    }
    return tok;
}

// CopyPermanent (Ocelot Pride): "for each token you control that entered this turn, create a
// token that's a copy of it." The set of permanents matching Defined$ Valid <filter> is
// snapshotted *before* any copy is made, so the freshly created copies — which also entered
// this turn — are not themselves re-copied. Each copy is a new token under the controller's
// control. CopyPermanent's filter is carried in valid_cards_filter (parsed from Defined$ Valid).
HandlerResult copy_permanent(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    Zone::Ownership ctrl = ab.controller;
    if (global_coordinator.entity_has_component<Permanent>(ab.source))
        ctrl = global_coordinator.GetComponent<Permanent>(ab.source).controller;

    // Offspring (CR 702.171c): "create a 1/1 token that's a copy of" the source creature.
    // Copy the source permanent itself, then override the copy's P/T to 1/1.
    if (ab.is_offspring_token) {
        Token tok = copyable_token_of(ab.source);
        if (tok.name.empty()) return HandlerResult::DONE_RUN_SUBS;
        tok.power = 1;
        tok.toughness = 1;
        Entity tok_entity = global_coordinator.CreateEntity();
        global_coordinator.AddComponent(tok_entity, Zone(Zone::HAND, ctrl, ctrl));
        global_coordinator.AddComponent(tok_entity, tok);
        orderer->add_to_zone(false, tok_entity, Zone::BATTLEFIELD);
        bootstrap_token_components(tok_entity, tok, ctrl, cur_game.timestamp);
        game_log("Offspring token copy created: %u/%u %s\n", tok.power, tok.toughness, tok.name.c_str());
        return HandlerResult::DONE_RUN_SUBS;
    }

    // Snapshot the matching permanents first (707.2 / "for each ... that entered this turn").
    std::vector<Entity> sources;
    for (auto e : orderer->mEntities)
        if (permanent_matches_filter(e, ab.valid_cards_filter, MatchCtx{ctrl, ab.source}))
            sources.push_back(e);

    for (auto src : sources) {
        Token tok = copyable_token_of(src);
        if (tok.name.empty()) continue;
        Entity tok_entity = global_coordinator.CreateEntity();
        global_coordinator.AddComponent(tok_entity, Zone(Zone::HAND, ctrl, ctrl));
        global_coordinator.AddComponent(tok_entity, tok);
        orderer->add_to_zone(false, tok_entity, Zone::BATTLEFIELD);
        bootstrap_token_components(tok_entity, tok, ctrl, cur_game.timestamp);
        game_log("Token copy created: %u/%u %s\n", tok.power, tok.toughness, tok.name.c_str());
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
