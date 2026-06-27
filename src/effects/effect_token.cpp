#include "effects.h"

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/token.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../parse.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Parses a token script string of the form "<color>_<power>_<toughness>_<name>[_<kw1>...]"
// e.g. "w_1_1_monk_prowess"
bool token(Ability &ab, std::shared_ptr<Orderer> orderer) {
    const TokenParams *tp = std::get_if<TokenParams>(&ab.params);
    std::string script = tp ? tp->script : "";
    Token tok = parse_token_script(script);
    if (tok.name.empty()) {
        game_log("resolve_token: failed to parse token script '%s'\n", script.c_str());
        return true;
    }

    Zone::Ownership ctrl = source_controller(ab.source);
    // TokenOwner$ TargetedPlayer (Kozilek's Command): the targeted player creates and
    // controls the tokens, not the spell's controller.
    if (tp && tp->owner_is_target && ab.target != 0 &&
        global_coordinator.entity_has_component<Player>(ab.target))
        ctrl = (ab.target == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
    // TokenOwner$ RememberedOwner (Skyclave Apparition): the token is owned and controlled by
    // the OWNER of the first remembered card — the exiled permanent's owner — so if Skyclave
    // exiled your permanent, you get the Illusion when Skyclave dies (CR 707/the card text).
    if (tp && tp->owner_is_remembered && !cur_game.remembered_entities.empty()) {
        Entity r = cur_game.remembered_entities[0];
        if (global_coordinator.entity_has_component<Zone>(r))
            ctrl = global_coordinator.GetComponent<Zone>(r).owner;
    }

    // TokenPower$/TokenToughness$ from an SVar (Skyclave Apparition: X = Remembered$CardManaCost):
    // override the token script's printed P/T so the created token enters as an X/X. Evaluated
    // once here against the live remembered card; both default to the script's value when absent.
    if (tp && !tp->power_expr.empty())
        tok.power = static_cast<uint32_t>(evaluate_dynamic_amount(tp->power_expr, ctrl, orderer, ab.target));
    if (tp && !tp->toughness_expr.empty())
        tok.toughness = static_cast<uint32_t>(evaluate_dynamic_amount(tp->toughness_expr, ctrl, orderer, ab.target));

    // TokenAmount$ N (default 1): create N identical tokens. The count may be dynamic
    // (Count$xPaid → X). amount==0 with no dynamic expr means the single-token default.
    size_t count = ab.amount;
    if (!ab.dynamic_amount_expr.empty())
        count = evaluate_dynamic_amount(ab.dynamic_amount_expr, ctrl, orderer, ab.target);
    else if (count == 0)
        count = 1;

    cur_game.remembered_entities.clear();
    for (size_t n = 0; n < count; n++) {
        Entity tok_entity = global_coordinator.CreateEntity();
        global_coordinator.AddComponent(tok_entity, Zone(Zone::HAND, ctrl, ctrl));
        global_coordinator.AddComponent(tok_entity, tok);
        orderer->add_to_zone(false, tok_entity, Zone::BATTLEFIELD);

        // Add Permanent + Creature + Damage immediately so subabilities (e.g. Attach) can see them
        // before the next apply_permanent_components pass.
        bootstrap_token_components(tok_entity, tok, ctrl, cur_game.timestamp);
        cur_game.remembered_entities.push_back(tok_entity);
        game_log("Token created: %u/%u %s\n", tok.power, tok.toughness, tok.name.c_str());
    }
    return true;
}

bool parse_token(Ability &ab, const std::string &key, const std::string &value) {
    if (key == "TokenScript") {
        effect_params<TokenParams>(ab).script = value;
        return true;
    }
    if (key == "TokenOwner") {
        // TokenOwner$ TargetedPlayer routes the tokens to the spell's chosen target;
        // TokenOwner$ RememberedOwner routes them to the owner of the first remembered card
        // (Skyclave Apparition: the exiled card's owner gets the Illusion). Other values
        // (You / the default) leave the token under the ability's controller.
        if (value.find("Targeted") != std::string::npos)
            effect_params<TokenParams>(ab).owner_is_target = true;
        else if (value.find("Remembered") != std::string::npos)
            effect_params<TokenParams>(ab).owner_is_remembered = true;
        return true;
    }
    if (key == "TokenPower") {
        effect_params<TokenParams>(ab).power_expr = value;
        return true;
    }
    if (key == "TokenToughness") {
        effect_params<TokenParams>(ab).toughness_expr = value;
        return true;
    }
    return false;
}

}  // namespace effects
