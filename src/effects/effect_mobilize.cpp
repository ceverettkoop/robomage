#include "effects.h"

#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../components/token.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../game_queries.h"
#include "../mana_system.h"
#include "../parse.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Mobilize N (rule 702.176, Tarkir: Dragonstorm): "Whenever this creature attacks, create
// N tapped and attacking 1/1 red Warrior creature tokens. Sacrifice them at the beginning
// of the next end step." The tokens enter already attacking the same player/planeswalker the
// Mobilize creature is attacking (508.4a — they are put onto the battlefield attacking, so no
// new attack is declared and no further "attacks" triggers fire). At the next end step the
// controller sacrifices exactly the tokens this instance created, via a delayed trigger.
bool mobilize(Ability &ab, std::shared_ptr<Orderer> orderer) {
    int n = static_cast<int>(ab.amount);
    if (n <= 0) return true;

    Zone::Ownership ctrl = source_controller(ab.source);

    // The defender the Mobilize creature is attacking; the tokens attack the same target.
    Entity attack_target = 0;
    if (global_coordinator.entity_has_component<Creature>(ab.source))
        attack_target = global_coordinator.GetComponent<Creature>(ab.source).attack_target;

    std::vector<Entity> created;
    for (int i = 0; i < n; i++) {
        Token tok = parse_token_script("r_1_1_warrior");
        if (tok.name.empty()) {
            game_log("Mobilize: no token script for Warrior.\n");
            break;
        }
        Entity tok_entity = global_coordinator.CreateEntity();
        global_coordinator.AddComponent(tok_entity, Zone(Zone::HAND, ctrl, ctrl));
        global_coordinator.AddComponent(tok_entity, tok);
        orderer->add_to_zone(false, tok_entity, Zone::BATTLEFIELD);
        bootstrap_token_components(tok_entity, tok, ctrl, cur_game.timestamp);

        // Enters tapped and attacking (put onto the battlefield attacking — not declared as
        // an attacker, so no summoning-sickness / declare-attackers restrictions apply).
        auto &perm = global_coordinator.GetComponent<Permanent>(tok_entity);
        perm.is_tapped = true;
        auto &cr = global_coordinator.GetComponent<Creature>(tok_entity);
        cr.is_attacking = true;
        cr.attack_target = attack_target;
        created.push_back(tok_entity);
    }

    if (created.empty()) return true;
    game_log("Mobilize %d: %s creates %zu tapped and attacking 1/1 Warrior(s).\n", n,
             player_name(ctrl).c_str(), created.size());

    // Register the "sacrifice them at the beginning of the next end step" delayed trigger
    // (same turn — the attack happened during this turn's combat, so the next end step is
    // this turn's). The sacrifice ability carries exactly the tokens created here.
    Ability sac_ab;
    sac_ab.ability_type = Ability::TRIGGERED;
    sac_ab.category = "SacrificeTokens";
    sac_ab.source = ab.source;
    sac_ab.targets = created;

    DelayedTrigger dt;
    dt.ability = sac_ab;
    dt.fire_on = Events::END_STEP_BEGAN;
    dt.owner_entity = get_player_entity(ctrl);
    dt.fire_on_turn = cur_game.turn;
    cur_game.delayed_triggers.push_back(dt);
    return true;
}

// Delayed end-step sacrifice fired by Mobilize. Sacrifices each token in ab.targets that is
// still on the battlefield (some may already have died/left). Tokens cease to exist when they
// hit the graveyard, matching "sacrifice them."
bool sacrifice_tokens(Ability &ab, std::shared_ptr<Orderer> orderer) {
    for (Entity tok : ab.targets) {
        if (!global_coordinator.entity_has_component<Zone>(tok)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(tok);
        if (z.location != Zone::BATTLEFIELD) continue;
        std::string name = global_coordinator.entity_has_component<Permanent>(tok)
                               ? global_coordinator.GetComponent<Permanent>(tok).name
                               : "token";
        orderer->add_to_zone(false, tok, Zone::GRAVEYARD);
        game_log("Mobilize: %s is sacrificed.\n", name.c_str());
    }
    return true;
}

}  // namespace effects
