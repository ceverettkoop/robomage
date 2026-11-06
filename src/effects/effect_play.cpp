#include "effects.h"

#include <cstdlib>
#include <string>
#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../queries/characteristics.h"
#include "../queries/filters.h"
#include "../queries/types.h"
#include "../queries/spells.h"
#include "../queries/zones.h"
#include "../resolution_memory.h"
#include "../action_processor.h"
#include "../resolution_frame.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static std::vector<Entity> play_candidates(const Ability &ab, const PlayParams &pp);
static ResolutionCastTerms play_terms(const Ability &ab, const PlayParams &pp, Entity card);
static int play_cost_amount(const PlayParams &pp, Entity card);

namespace effects {

// DB$ Play: "You may cast [a card]." The cast is part of this effect (CR 608.2g): it is offered
// while the ability resolves, ignoring the card's type-based timing, and made through the ordinary
// cast flow (cast_during_resolution), so the spell goes on the stack above the resolving ability and
// cast triggers, targets and cost floors behave as for any cast. If it isn't cast then, the card
// stays where it is; no permission outlives the resolution.
//
// The cards are Defined$ (Amped Raptor's Remembered: the nonland card DigUntil just exiled), or those
// of ValidZone$ that Valid$ matches (Bilbo's graveyard). PlayCost$ replaces the mana cost with a
// resource (CR 118.9): PayEnergy / PayLife of a literal amount or of the spell's mana value
// ("ConvertedManaCost"; X counts as 0 for a card in exile, CR 202.3e). ValidSA$: the spells it may
// be cast as, so a land (CR 601.1) gets no offer. Optional$ True: the controller may decline (every
// DB$ Play in the card pool is optional).
HandlerResult play(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    ResolutionCastRt local_rt;
    ResolutionCastRt &rt = ctx.can_suspend() ? ctx.rt<ResolutionCastRt>() : local_rt;
    const PlayParams &pp = std::get<PlayParams>(ab.def->params);
    // While the cast is in flight the card has left its zone, so the offer is built only before it.
    std::vector<ResolutionCastOffer> offers;
    std::string decline_label;
    if (rt.stage == ResolutionCastRt::OFFER) {
        if (pp.needs_bargain && !spell_bargained(ab.source.get())) return HandlerResult::DONE_RUN_SUBS;
        const std::vector<Entity> cards = play_candidates(ab, pp);
        for (Entity card : cards) {
            const ResolutionCastTerms terms = play_terms(ab, pp, card);
            const bool resource = terms.grant.resource == GrantResource::ENERGY || terms.grant.resource == GrantResource::LIFE;
            if (resource && !ctx.resuming()) {
                const char *res = (terms.grant.resource == GrantResource::LIFE) ? "life" : "energy";
                game_log("%s may cast %s by paying %d %s rather than its mana cost.\n",
                         player_name(ab.controller).c_str(), entity_name(card).c_str(), terms.grant.amount, res);
            }
            const std::vector<ResolutionCastOffer> rows = resolution_cast_offers(card, ab.controller, terms, orderer);
            offers.insert(offers.end(), rows.begin(), rows.end());
        }
        decline_label = cards.size() == 1 ? "Do not cast " + entity_name(cards.front()) : "Do not cast a spell";
    }
    const ResolutionCastStatus status = cast_during_resolution(offers, decline_label, ab.controller, rt, ctx, orderer);
    if (status == ResolutionCastStatus::SUSPENDED) return HandlerResult::SUSPENDED;
    if (status == ResolutionCastStatus::CAST) report_object(ab, EffectObjects::PLAYED, rt.card);
    return HandlerResult::DONE_RUN_SUBS;
}

bool parse_play(AbilityDef &ab, const std::string &key, const std::string &value) {
    if (ab.category != "Play") return false;
    PlayParams &pp = effect_params<PlayParams>(ab);
    if (key == "ValidSA") {
        pp.valid_sa = FilterSpec(value);
        return true;
    } else if (key == "WithoutManaCost" && value == "True") {
        pp.cost_resource = GrantResource::FREE;
        return true;
    } else if (key == "Condition" && value == "Bargain") {
        pp.needs_bargain = true;
        return true;
    } else if (key == "ValidZone" && value == "Graveyard") {
        pp.from_zone = true;
        pp.zone = Zone::GRAVEYARD;
        return true;
    } else if (key == "Valid") {
        pp.valid = FilterSpec(value);
        return true;
    } else if (key == "ReplaceGraveyard" && value == "Exile") {
        pp.replaces_graveyard = true;
        pp.graveyard_replaced_by = Zone::EXILE;
        return true;
    } else if (key == "ReplaceGraveyardValid") {
        pp.graveyard_replaced_valid = FilterSpec(value);
        return true;
    } else if (key == "PlayCost") {
        // PlayCost$ PayEnergy<amount> / PayLife<amount>. The amount inside the angle brackets is
        // either a literal int or "ConvertedManaCost" (the cast card's mana value). The resource
        // generalizes the alt-cost-cast over energy ({E}) and life.
        std::string amount_expr;
        size_t lt = value.find('<');
        size_t gt = value.find('>');
        if (lt != std::string::npos && gt != std::string::npos && gt > lt)
            amount_expr = value.substr(lt + 1, gt - lt - 1);
        if (value.rfind("PayEnergy", 0) == 0) {
            pp.cost_resource = GrantResource::ENERGY;
            pp.cost_expr = amount_expr;
        } else if (value.rfind("PayLife", 0) == 0) {
            pp.cost_resource = GrantResource::LIFE;
            pp.cost_expr = amount_expr;
        }
        return true;
    }
    return false;
}

}  // namespace effects

// The cards `ab` may cast.
static std::vector<Entity> play_candidates(const Ability &ab, const PlayParams &pp) {
    std::vector<Entity> cards;
    if (pp.from_zone) {
        MatchCtx match;
        match.controller = ab.controller;
        for (Entity e : zone_objects(zoned_entities(), pp.zone))
            if (global_coordinator.entity_has_component<CardData>(e) && card_matches_filter(e, pp.valid, match))
                cards.push_back(e);
        return cards;
    }
    Entity card = 0;
    if (ab.def->defined.is_remembered() && !cur_game.resolution.memory.remembered.empty())
        card = cur_game.resolution.memory.remembered[0].get();
    else
        card = ab.target.get();  // fallback: a directly-defined/targeted card
    if (card != 0 && global_coordinator.entity_has_component<CardData>(card) &&
        global_coordinator.entity_has_component<Zone>(card))
        cards.push_back(card);
    return cards;
}

// What `ab` asks for casting `card`.
static ResolutionCastTerms play_terms(const Ability &ab, const PlayParams &pp, Entity card) {
    ResolutionCastTerms terms;
    terms.grant.resource = pp.cost_resource;
    terms.amount_is_mana_value = pp.cost_expr == "ConvertedManaCost";
    terms.grant.amount = play_cost_amount(pp, card);
    terms.valid_sa = &pp.valid_sa;
    terms.cast_by = ab.def;
    return terms;
}

// The PlayCost$ amount for `card`: its mana value for "ConvertedManaCost" (of the face cast, once
// it is known), else the literal.
static int play_cost_amount(const PlayParams &pp, Entity card) {
    if (pp.cost_expr == "ConvertedManaCost") return card_mana_value(global_coordinator.GetComponent<CardData>(card));
    if (!pp.cost_expr.empty()) return std::atoi(pp.cost_expr.c_str());
    return 0;
}
