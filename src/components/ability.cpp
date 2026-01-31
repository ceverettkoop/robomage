#include "ability.h"
#include "damage.h"
#include <cstdio>

void Ability::resolve() {
    // TODO: Implement ability resolution based on category and type
    printf("Resolving ability (category: %s, amount: %zu)\n", category.c_str(), amount);

    // Example: if category is "damage", deal damage to target
    if (category == "damage" && target != 0) {
        deal_damage(source, target, amount);
    }
}