#include "ability.h"
#include "damage.h"

void Ability::resolve() {

    if(type == "DealDamage"){
        deal_damage(source, target, amount);
    }
}