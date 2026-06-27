#include "effects.h"

#include <cstdlib>
#include <string>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// DB$ Play (Amped Raptor): "You may cast that card by paying [an alternative cost] rather than
// paying its mana cost." (CR 707 impulse / 118.9 alternative cost.)
//
// The card to cast is Defined$ Remembered — the nonland card DigUntil just exiled. Rather than
// reentrantly cast a spell mid-resolution (the DB$ Play ability is itself resolving from the
// stack), this grants a one-shot permission (cur_game.impulse_cast_permission) to cast that
// exiled card this turn, with its mana cost replaced by the alternative RESOURCE cost. The
// normal casting pipeline (determine_legal_actions → CAST_SPELL) then offers it at the
// controller's next priority and funnels it onto the stack like any other cast, so targeting,
// triggers, and the stack all work unchanged.
//
// General over the resource (PlayCostResource) and the amount (play_cost_expr =
// "ConvertedManaCost" → the card's mana value, else a literal), so a future Bolas's Citadel
// ("play the top card of your library, paying life equal to its mana value") reuses this with
// play_cost_resource = LIFE. Optional$ True (optional_choice) lets the controller decline — the
// permission is offered, not forced, so declining simply means the card stays in exile.
//
// ValidSA$ Spell (play_valid_sa_spell): only a card castable as a nonland spell may be played
// this way; a land (or a card with no SPELL ability) gets no permission and stays exiled.
bool play(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // Resolve the card to play (Defined$ Remembered).
    Entity card = 0;
    if (ab.defined_remembered && !cur_game.remembered_entities.empty())
        card = cur_game.remembered_entities[0];
    else
        card = ab.target;  // fallback: a directly-defined/targeted card
    if (card == 0 || !global_coordinator.entity_has_component<CardData>(card)) return true;

    auto &cd = global_coordinator.GetComponent<CardData>(card);

    // ValidSA$ Spell: only a card that can be cast as a (nonland) spell qualifies. Every
    // nonland card is cast as a spell (CR 601.1) — instants/sorceries via their SP$ ability,
    // permanents (creatures, artifacts, ...) cast directly onto the stack — so the gate is
    // simply "not a land". Lands can't be cast (601.1), so a land remembered here does nothing.
    if (ab.play_valid_sa_spell && is_land_card(cd)) return true;

    // Resolve the alternative cost amount. "ConvertedManaCost" = the card's mana value (the
    // number of mana symbols in its printed cost); otherwise a literal. X spells count X as 0
    // for this mechanic (the card is exiled with no X chosen), so the printed cost's symbols
    // already give mana value with X = 0.
    int amount = 0;
    if (ab.play_cost_expr == "ConvertedManaCost")
        amount = static_cast<int>(cd.mana_cost.size());
    else if (!ab.play_cost_expr.empty())
        amount = std::atoi(ab.play_cost_expr.c_str());

    Game::ImpulseCastPermission perm;
    perm.resource = (ab.play_cost_resource == Ability::PLAY_COST_LIFE)
                        ? Game::ImpulseCastPermission::LIFE
                        : Game::ImpulseCastPermission::ENERGY;
    perm.amount = amount;
    perm.caster = ab.controller;
    cur_game.impulse_cast_permission[card] = perm;

    const char *res = (perm.resource == Game::ImpulseCastPermission::LIFE) ? "life" : "energy";
    game_log("%s may cast %s this turn by paying %d %s rather than its mana cost.\n",
             player_name(ab.controller).c_str(), cd.name.c_str(), amount, res);
    return true;
}

bool parse_play(Ability &ab, const std::string &key, const std::string &value) {
    if (key == "ValidSA") {
        // ValidSA$ Spell — only a castable nonland spell may be played this way.
        ab.play_valid_sa_spell = (value == "Spell");
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
            ab.play_cost_resource = Ability::PLAY_COST_ENERGY;
            ab.play_cost_expr = amount_expr;
        } else if (value.rfind("PayLife", 0) == 0) {
            ab.play_cost_resource = Ability::PLAY_COST_LIFE;
            ab.play_cost_expr = amount_expr;
        }
        return true;
    }
    return false;
}

}  // namespace effects
