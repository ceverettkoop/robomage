#include "ability.h"

#include "../ecs/coordinator.h"
#include "damage.h"
#include "player.h"

#include <cstdio>

extern Coordinator global_coordinator;

bool Ability::identical_activated_ability(const Ability &other){ 
    if(other.category != this->category) return false;
    if(other.valid_tgts != this->valid_tgts) return false;    
    if(other.amount != this->amount) return false;
    if(other.tap_cost != this->tap_cost) return false;
    if(other.activation_mana_cost != this->activation_mana_cost) return false;
    if(other.sac_self != this->sac_self) return false;
    if(other.change_type != this->change_type) return false;
    return true;
};

void Ability::resolve() {
    printf("Resolving ability (category: %s, amount: %zu)\n", category.c_str(), amount);

    if (category == "DealDamage") {
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
                printf("Invalid target\n");
            }
        }
    }
}