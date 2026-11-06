#include "effects.h"

#include <cstdlib>
#include <string>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../queries/characteristics.h"
#include "../queries/types.h"
#include "../action_processor.h"
#include "../resolution_frame.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static int play_cost_amount(const PlayParams &pp, Entity card);

namespace effects {

// DB$ Play (Amped Raptor): "You may cast that card by paying [an alternative cost] rather than
// paying its mana cost." The cast is part of this effect (CR 608.2g): it is offered while the
// ability resolves, ignoring the card's type-based timing, and made through the ordinary cast
// flow (cast_during_resolution), so the spell goes on the stack above the resolving ability and
// cast triggers, targets and cost floors behave as for any cast. If it isn't cast then, the card
// stays where it is; no permission outlives the resolution.
//
// The card is Defined$ Remembered — the nonland card DigUntil just exiled. The alternative cost
// replaces the mana cost with the PlayCost$ resource (CR 118.9): PayEnergy / PayLife of a literal
// amount or of the card's mana value ("ConvertedManaCost"; X counts as 0 for a card in exile,
// CR 202.3e). ValidSA$: the spells it may be cast as, so a land (CR 601.1) gets no offer.
// Optional$ True: the controller may decline (every DB$ Play in the card pool is
// optional).
HandlerResult play(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    ResolutionCastRt local_rt;
    ResolutionCastRt &rt = ctx.can_suspend() ? ctx.rt<ResolutionCastRt>() : local_rt;
    const PlayParams &pp = std::get<PlayParams>(ab.def->params);
    // Resolve the card to play (Defined$ Remembered). While the cast is in flight the card has
    // left exile, so the card is looked up only for the offer.
    Entity card = 0;
    if (ab.def->defined.is_remembered() && !cur_game.resolution.memory.remembered.empty())
        card = cur_game.resolution.memory.remembered[0].get();
    else
        card = ab.target.get();  // fallback: a directly-defined/targeted card
    if (card == 0 || !global_coordinator.entity_has_component<CardData>(card) ||
        !global_coordinator.entity_has_component<Zone>(card))
        return HandlerResult::DONE_RUN_SUBS;
    if (rt.stage == ResolutionCastRt::OFFER) {
        const CardData &cd = global_coordinator.GetComponent<CardData>(card);
        if (global_coordinator.GetComponent<Zone>(card).location != Zone::EXILE)
            return HandlerResult::DONE_RUN_SUBS;
        if (is_land_card(cd)) return HandlerResult::DONE_RUN_SUBS;
    }

    ResolutionCastTerms terms;
    terms.grant.resource = pp.cost_resource == PlayParams::LIFE ? GrantResource::LIFE : GrantResource::ENERGY;
    terms.amount_is_mana_value = pp.cost_expr == "ConvertedManaCost";
    terms.grant.amount = play_cost_amount(pp, card);
    terms.valid_sa = &pp.valid_sa;
    if (rt.stage == ResolutionCastRt::OFFER && !ctx.resuming()) {
        const char *res = (terms.grant.resource == GrantResource::LIFE) ? "life" : "energy";
        game_log("%s may cast %s by paying %d %s rather than its mana cost.\n",
                 player_name(ab.controller).c_str(), entity_name(card).c_str(), terms.grant.amount, res);
    }
    if (cast_during_resolution(card, ab.controller, terms, rt, ctx, orderer) == ResolutionCastStatus::SUSPENDED)
        return HandlerResult::SUSPENDED;
    return HandlerResult::DONE_RUN_SUBS;
}

bool parse_play(AbilityDef &ab, const std::string &key, const std::string &value) {
    if (ab.category != "Play") return false;
    PlayParams &pp = effect_params<PlayParams>(ab);
    if (key == "ValidSA") {
        pp.valid_sa = FilterSpec(value);
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
            pp.cost_resource = PlayParams::ENERGY;
            pp.cost_expr = amount_expr;
        } else if (value.rfind("PayLife", 0) == 0) {
            pp.cost_resource = PlayParams::LIFE;
            pp.cost_expr = amount_expr;
        }
        return true;
    }
    return false;
}

}  // namespace effects

// The PlayCost$ amount for `card`: its mana value for "ConvertedManaCost" (of the face cast, once
// it is known), else the literal.
static int play_cost_amount(const PlayParams &pp, Entity card) {
    if (pp.cost_expr == "ConvertedManaCost") return card_mana_value(global_coordinator.GetComponent<CardData>(card));
    if (!pp.cost_expr.empty()) return std::atoi(pp.cost_expr.c_str());
    return 0;
}
