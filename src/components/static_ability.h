#ifndef STATIC_ABILITY_H
#define STATIC_ABILITY_H
#include <cstdint>
#include <string>

struct StaticAbility {
    std::string category  = "";  // "Continuous", "MustAttack"
    std::string condition = "";  // "Delirium", etc.; empty = always active
    // Continuous mode fields:
    int add_power     = 0;
    int add_toughness = 0;
    std::string add_power_svar = "";      // e.g. "Count$TypeInYourYard.Land" — evaluated at SBE time
    std::string add_toughness_svar = "";
    int last_applied_power = 0;           // tracks dynamic delta last applied (reset when condition lost)
    int last_applied_toughness = 0;
    std::string add_keyword = "";
    std::string affected = "";        // "EquippedBy" = apply buff to equipped creature, not source
    // ETB counter fields (category = "EtbCounter"):
    std::string counter_type = "";    // "P1P1" for +1/+1 counters
    bool counter_count_from_delve = false;  // counter count = cur_game.delve_exiled.size()
    // RaiseCost fields (category = "RaiseCost"):
    int raise_cost = 0;               // generic mana added to cost of matching spells
    std::string raise_cost_filter = ""; // "nonCreature" = apply to non-creature spells
    // State tracking — lives in the Permanent's copy, not in CardData template:
    bool applied = false;             // true when the condition was met on the last SBE pass
    uint32_t last_applied_entity = 0; // for EquippedBy: entity that last received the buff
};

#endif /* STATIC_ABILITY_H */
