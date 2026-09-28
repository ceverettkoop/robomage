#include "effects.h"

#include <memory>
#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../components/token.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../queries/battlefield.h"
#include "../queries/characteristics.h"
#include "../queries/filters.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Builds the Token characteristics for a copy of permanent `src` from its copiable values
// (CR 707.2): a token source's own Token component; a nontoken permanent's printed card, on the
// face it shows (CR 712.8e) — name, types, colors, mana value, P/T, keywords, and its triggered,
// activated and static abilities and replacement effects. Counters, pumps and type- or ability-changing effects are not
// copiable, so the live Permanent / Creature state is not read.
static Token copyable_token_of(Entity src) {
    if (global_coordinator.entity_has_component<Token>(src))
        return global_coordinator.GetComponent<Token>(src);
    Token tok;
    if (!global_coordinator.entity_has_component<CardData>(src)) return tok;
    const CardData &card = global_coordinator.GetComponent<CardData>(src);
    const CardData &face = active_face(src, card);
    bool transformed = global_coordinator.entity_has_component<Permanent>(src) &&
                       global_coordinator.GetComponent<Permanent>(src).transformed;
    tok.name = face.name;
    tok.types = face.types;
    tok.explicit_colors = card_colors(face);
    tok.mana_value = card_mana_value(mana_value_face(card, transformed));
    tok.power = face.power;
    tok.toughness = face.toughness;
    tok.keywords = face.keywords;
    tok.abilities = face.abilities;
    tok.static_abilities = face.static_abilities;
    tok.replacement_effects = face.replacement_effects;
    return tok;
}

// CopyPermanent (Ocelot Pride): "for each token you control that entered this turn, create a
// token that's a copy of it." The set of permanents matching Defined$ Valid <filter> is
// snapshotted *before* any copy is made, so the freshly created copies — which also entered
// this turn — are not themselves re-copied. Each copy is a new token under the controller's
// control. CopyPermanent's filter is carried in valid_cards_filter (parsed from Defined$ Valid).
HandlerResult copy_permanent(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    Zone::Ownership ctrl = ab.controller;
    const Entity self = ab.source.get();
    if (self != 0 && global_coordinator.entity_has_component<Permanent>(self))
        ctrl = global_coordinator.GetComponent<Permanent>(self).controller;

    // Offspring (CR 702.175a): "create a token that's a copy of it, except it's 1/1."
    // Copy the source permanent itself, then override the copy's P/T to 1/1.
    if (ab.def.is_offspring_token) {
        Token tok = copyable_token_of(ab.source.lki_entity());
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
        if (permanent_matches_filter(e, ab.def.valid_cards_filter, MatchCtx{ctrl, ab.source.lki_entity()}))
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

// AB$ Clone (Thespian's Stage: "{2}, {T}: CARDNAME becomes a copy of target land, except it has
// this ability."). CR 707.2 / 707.9b in-place copy: the SOURCE permanent (Thespian's Stage) becomes
// a copy of the target land while staying the SAME object — same entity, same counters, tapped
// state, and attachments. Unlike copy_permanent (which spawns a token), this overwrites the source
// permanent's copiable characteristics (name, types, abilities, keywords, static abilities) with the
// target's, leaving non-copiable state untouched (707.2). Because a copied land is a fresh set of
// copiable values, any counters already on the source remain: a Stage cloning Dark Depths becomes a
// Dark Depths with ZERO ice counters (the ice counters come from Dark Depths' etbCounter, which
// applies only on ENTRY, never on becoming a copy), so its Mode$ Always state trigger sees no ice
// counters and fires. The card's printed CardData is stashed on Permanent::printed_card (first copy
// only) and restored by Orderer::add_to_zone when the permanent leaves the battlefield (CR 400.7).
HandlerResult clone(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    (void)ctx;
    Entity src = ab.source.get();  // the permanent that becomes a copy
    Entity tgt = ab.target.get();  // the land being copied
    if (!is_battlefield_permanent(src) || !global_coordinator.entity_has_component<CardData>(src)) {
        game_log("Clone: source is no longer on the battlefield\n");
        return HandlerResult::DONE_RUN_SUBS;
    }
    // CR 608.2b re-verify the target: a nontoken land still on the battlefield.
    if (tgt == 0 || !is_battlefield_permanent(tgt) ||
        !global_coordinator.entity_has_component<CardData>(tgt)) {
        game_log("Clone: target land is no longer valid\n");
        return HandlerResult::DONE_RUN_SUBS;
    }

    // GainThisAbility$ True: preserve the source's own Clone ability(ies) — captured BEFORE the
    // overwrite — so the copy keeps "…except it has this ability." and can re-clone later.
    std::vector<AbilityDef> retained;
    if (ab.def.gain_this_ability) {
        for (const auto &a : global_coordinator.GetComponent<CardData>(src).abilities)
            if (a.category == "Clone") retained.push_back(a);
    }

    // Build the copy from the target's copiable characteristics (a value copy of its CardData),
    // then re-add the retained Clone ability(ies).
    CardData new_cd = global_coordinator.GetComponent<CardData>(tgt);
    for (const auto &a : retained) new_cd.abilities.push_back(a);

    // Apply the copy in place: overwrite the source's CardData and reset its derived permanent
    // state (name/types and the ability/static lists) so the next state-based pass re-derives the
    // permanent's activated/mana/triggered abilities and static abilities from the NEW
    // characteristics (apply_permanent_components + apply_land_abilities). Counters, tapped state,
    // and attachments on the Permanent are deliberately NOT touched (707.2). The printed card is
    // stashed before the first overwrite only, so a re-copy keeps the original printed identity.
    auto &perm = global_coordinator.GetComponent<Permanent>(src);
    if (!perm.printed_card)
        perm.printed_card =
            std::make_shared<const CardData>(global_coordinator.GetComponent<CardData>(src));
    global_coordinator.GetComponent<CardData>(src) = new_cd;
    std::string old_name = perm.name;
    perm.name = new_cd.name;
    perm.types = new_cd.types;
    perm.abilities.clear();         // re-derived from the new CardData next SBA pass
    perm.static_abilities.clear();  // re-derived from the new CardData next SBA pass
    game_log("%s becomes a copy of %s.\n", old_name.c_str(), new_cd.name.c_str());
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
