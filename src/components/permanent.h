#ifndef PERMANENT_H
#define PERMANENT_H

#include "../ecs/entity.h"
#include "zone.h"
#include "ability.h"
#include "static_ability.h"
#include "types.h"
#include <set>
#include <string>
#include <vector>

struct Permanent {
    std::string name;
    std::set<Type> types;
    bool is_token = false;
    bool is_tapped = false;
    bool has_summoning_sickness = true;
    std::vector<Ability> abilities;
    std::vector<StaticAbility> static_abilities;
    Zone::Ownership controller = Zone::UNKNOWN;
    size_t timestamp_entered_battlefield = 0;  // For ordering simultaneous ETBs
    bool transformed = false;  // true when DFC is showing its back face
    Entity equipped_to = 0;   // for equipment: which creature entity is equipped (0 = unattached)
    Entity equipped_by = 0;   // for creatures: which equipment is attached (0 = none)
    bool is_phased_out = false;
    int loyalty = 0;  // number of loyalty counters on a planeswalker (306.5c); current loyalty
    // 606.3: a loyalty ability can be activated only if no loyalty ability of THIS
    // permanent has been activated this turn. The gate is per-permanent (across all its
    // loyalty abilities), not per-ability, so it can't reuse Ability::activations_this_turn.
    bool loyalty_ability_activated_this_turn = false;
    bool evoked = false;  // entered via its evoke alternate cost — fires the evoke sacrifice ETB trigger
    std::string chosen_type = "";  // creature type chosen on ETB (Cavern of Souls)
    std::string chosen_name = "";  // card name chosen on ETB (Disruptor Flute) — keys Card.NamedCard statics
    std::vector<Entity> exiled_with;  // entities exiled by this permanent (for Keen-Eyed Curator)
};

#endif /* PERMANENT_H */
