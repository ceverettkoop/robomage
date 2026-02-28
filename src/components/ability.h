#ifndef ABILITY_H
#define ABILITY_H

#include "../classes/colors.h"
#include "../ecs/entity.h"
#include <string>

struct Ability{

    enum AbilityType{
        TRIGGERED,
        ACTIVATED,
        SPELL
    };

    AbilityType ability_type = SPELL;
    std::string category = "";
    // TODO: support multiple targets (e.g. "deal 1 damage to each of up to two targets")
    std::string valid_tgts = "N_A";  // Value of ValidTgts$ param; "N_A" if no targeting required
    Entity source = 0;
    Entity target = 0;
    // TODO: support multiple effects per ability (e.g. "deal 3 damage and gain 3 life")
    size_t amount = 0;
    Colors color = NO_COLOR; //for mana ability

    // Activated ability costs
    bool tap_cost = false;              // {T} is part of the activation cost
    ManaValue activation_mana_cost;     // Mana that must be paid to activate
    int life_cost = 0;                  // PayLife<N> — life paid at activation
    bool sac_self = false;              // Sac<1/CARDNAME> — sacrifice source permanent as cost
    std::string change_type = "";        // ChangeType$ — comma-separated land subtypes to search

    void resolve();
    bool identical_activated_ability(const Ability& other);
};

#endif /* ABILITY_H */
