#include "token.h"
#include "permanent.h"
#include "creature.h"
#include "damage.h"
#include "../classes/game.h"
#include "../ecs/coordinator.h"

extern Game cur_game;

void bootstrap_token_components(Entity tok_entity, const Token &tok,
                                Zone::Ownership controller, size_t &timestamp) {
    if (!global_coordinator.entity_has_component<Permanent>(tok_entity)) {
        Permanent perm;
        perm.name = tok.name;
        perm.types = tok.types;
        perm.is_token = true;
        perm.controller = controller;
        perm.has_summoning_sickness = true;
        perm.is_tapped = false;
        perm.timestamp_entered_battlefield = timestamp++;
        perm.entered_on_turn = cur_game.turn;
        global_coordinator.AddComponent(tok_entity, perm);
    }
    if (!global_coordinator.entity_has_component<Creature>(tok_entity)) {
        Creature creature;
        creature.base_power = static_cast<int>(tok.power);
        creature.base_toughness = static_cast<int>(tok.toughness);
        creature.keywords = tok.keywords;
        recompute_pt(creature);
        global_coordinator.AddComponent(tok_entity, creature);

        Damage damage;
        damage.damage_counters = 0;
        global_coordinator.AddComponent(tok_entity, damage);
    }
}
