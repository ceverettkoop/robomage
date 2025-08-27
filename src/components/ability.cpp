#include "ability.h"
#include "damage.h"

void Ability::resolve() {
    if(category == "DealDamage"){
        deal_damage(source, target, amount);
    }
}