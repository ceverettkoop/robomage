#include "effects.h"

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/permanent.h"
#include "../components/token.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../parse.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Parses a token script string of the form "<color>_<power>_<toughness>_<name>[_<kw1>...]"
// e.g. "w_1_1_monk_prowess"
bool token(Ability &ab, std::shared_ptr<Orderer> orderer) {
    Token tok = parse_token_script(ab.token_script);
    if (tok.name.empty()) {
        game_log("resolve_token: failed to parse token script '%s'\n", ab.token_script.c_str());
        return true;
    }

    Zone::Ownership ctrl = global_coordinator.entity_has_component<Permanent>(ab.source)
                               ? global_coordinator.GetComponent<Permanent>(ab.source).controller
                               : global_coordinator.GetComponent<Zone>(ab.source).owner;

    Entity tok_entity = global_coordinator.CreateEntity();
    global_coordinator.AddComponent(tok_entity, Zone(Zone::HAND, ctrl, ctrl));
    global_coordinator.AddComponent(tok_entity, tok);
    orderer->add_to_zone(false, tok_entity, Zone::BATTLEFIELD);

    // Add Permanent + Creature + Damage immediately so subabilities (e.g. Attach) can see them
    // before the next apply_permanent_components pass.
    bootstrap_token_components(tok_entity, tok, ctrl, cur_game.timestamp);

    cur_game.remembered_entities.clear();
    cur_game.remembered_entities.push_back(tok_entity);
    game_log("Token created: %u/%u %s\n", tok.power, tok.toughness, tok.name.c_str());
    return true;
}

bool parse_token(Ability &ab, const std::string &key, const std::string &value) {
    if (key != "TokenScript") return false;
    ab.token_script = value;
    return true;
}

}  // namespace effects
