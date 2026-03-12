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
    std::string add_keyword = "";
    std::string affected = "";        // "EquippedBy" = apply buff to equipped creature, not source
    // ETB counter fields (category = "EtbCounter"):
    std::string counter_type = "";    // "P1P1" for +1/+1 counters
    bool counter_count_from_delve = false;  // counter count = cur_game.delve_exiled.size()
    // State tracking — lives in the Permanent's copy, not in CardData template:
    bool applied = false;             // true when the condition was met on the last SBE pass
    uint32_t last_applied_entity = 0; // for EquippedBy: entity that last received the buff
};

#endif /* STATIC_ABILITY_H */
