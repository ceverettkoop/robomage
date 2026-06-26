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
        // Carry the token's intrinsic activated abilities onto the permanent so they are
        // offered as legal actions (a card's activated abilities are read from
        // Permanent::abilities, not the source component). The Eldrazi Spawn token's
        // "Sacrifice this creature: Add {C}." mana ability reaches the player this way.
        // Triggered abilities stay on Token::abilities (the trigger scan reads them there).
        for (const auto &ab : tok.abilities) {
            if (ab.ability_type != Ability::ACTIVATED && ab.ability_type != Ability::SPELL)
                continue;
            Ability copy = ab;
            copy.source = tok_entity;
            perm.abilities.push_back(copy);
        }
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
