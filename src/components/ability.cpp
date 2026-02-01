#include "ability.h"

#include "../ecs/coordinator.h"
#include "damage.h"
#include "player.h"

#include <cstdio>

extern Coordinator global_coordinator;

void Ability::resolve() {
    printf("Resolving ability (category: %s, amount: %zu)\n", category.c_str(), amount);

    if (category == "DealDamage" && target != 0) {
        // Deal damage to target
        if (global_coordinator.entity_has_component<Player>(target)) {
            // Target is a player
            auto& player = global_coordinator.GetComponent<Player>(target);
            player.life_total -= static_cast<int32_t>(amount);
            printf("Dealt %zu damage to player (now at %d life)\n", amount, player.life_total);
        } else {
            // Target is a creature/permanent
            if (deal_damage(source, target, amount)) {
                printf("Dealt %zu damage to creature\n", amount);
            } else {
                printf("Failed to deal damage to target\n");
            }
        }
    }
}