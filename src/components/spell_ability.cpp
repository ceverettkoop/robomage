#include "spell_ability.h"
#include "damage.h"

void SpellAbility::resolve() {

    if(type == "DealDamage"){
        deal_damage(source, target, amount);
    }
}