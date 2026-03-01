#ifndef STATIC_ABILITY_H
#define STATIC_ABILITY_H
#include <string>

struct StaticAbility {
    std::string category  = "";  // "Continuous", "MustAttack"
    std::string condition = "";  // "Delirium", etc.; empty = always active
    // Continuous mode fields:
    int add_power     = 0;
    int add_toughness = 0;
    std::string add_keyword = "";
    // State tracking — lives in the Permanent's copy, not in CardData template:
    bool applied = false;  // true when the condition was met on the last SBE pass
};

#endif /* STATIC_ABILITY_H */
