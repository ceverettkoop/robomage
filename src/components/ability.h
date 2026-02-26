#ifndef ABILITY_H
#define ABILITY_H

#include "../ecs/entity.h"
#include <string>

struct Ability{

    enum AbilityType{
        TRIGGERED,
        ACTIVATED,
        SPELL
    };

    AbilityType ability_type = SPELL;
    std::string category;
    // TODO: support multiple targets (e.g. "deal 1 damage to each of up to two targets")
    std::string valid_tgts = "N_A";  // Value of ValidTgts$ param; "N_A" if no targeting required
    Entity source = 0;
    Entity target = 0;
    // TODO: support multiple effects per ability (e.g. "deal 3 damage and gain 3 life")
    size_t amount = 0;

    void resolve();

    // TODO order these by the order they appear on the card
    bool operator<(const Ability& other) const {
        if (ability_type != other.ability_type) return ability_type < other.ability_type;
        if (category != other.category) return category < other.category;
        if (valid_tgts != other.valid_tgts) return valid_tgts < other.valid_tgts;
        return amount < other.amount;
    }
};

#endif /* ABILITY_H */
