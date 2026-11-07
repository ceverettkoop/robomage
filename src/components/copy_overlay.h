#ifndef COPY_OVERLAY_H
#define COPY_OVERLAY_H

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "../classes/colors.h"
#include "types.h"

// The exceptions a copy effect makes to the copying process (CR 707.9): what it sets or adds is
// part of the copy's copiable values, so a copy of the copy has them too (707.9a, b). One value
// for every copy effect, whether the copy is a new token or a permanent that becomes a copy where
// it is (queries/copies.h); a new exception is a field here and a line of apply_copy_overlay. Fixed
// by the effect, so it lives on the AbilityDef.
struct CopyOverlay {
    // "Except it's 1/1" (Offspring, CR 702.175a; 707.9b).
    bool sets_pt = false;
    uint32_t power = 0;
    uint32_t toughness = 0;
    // GainThisAbility$ True (Thespian's Stage: "except it has this ability", 707.9a): the copy keeps
    // the Clone abilities of the permanent that became it.
    bool gains_this_ability = false;
    // AddKeywords$ (Reflection of Kiki-Jiki: "except it has haste", 707.9a).
    std::vector<std::string> add_keywords;
    // AddTypes$ (Mockingbird: "except it's a Bird in addition to its other types", 707.9b).
    std::vector<Type> add_types;
    // SetCreatureTypes$ (Lazotep Quarry: "except it's a ... Zombie"): these replace its creature
    // types, and its other subtypes stay (205.1a).
    std::vector<Type> set_creature_types;
    // SetColor$ (Lazotep Quarry: "except it's ... black"): what it is instead of what it copied.
    std::set<Colors> set_colors;
};

#endif /* COPY_OVERLAY_H */
