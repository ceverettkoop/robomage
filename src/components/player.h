#ifndef PLAYER_H
#define PLAYER_H

#include <cstdint>
#include <set>
#include <vector>
#include "../classes/colors.h"
#include "../ecs/entity.h"

struct Player {
    int32_t life_total = 20;
    int32_t life_gained_this_turn = 0;  // total life gained this turn (any source); reset each turn (Ocelot Pride end-step check)
    bool has_city_blessing = false;     // 702.131c: city's blessing designation, kept for the rest of the game once obtained
    uint8_t poison_counters = 0;
    std::multiset<Colors> mana;
    uint8_t lands_played_this_turn = 0;
    size_t spells_cast_this_turn = 0;
    size_t spells_cast_this_game = 0;
    size_t noncreature_spells_cast_this_turn = 0;
    size_t instant_sorcery_spells_cast_this_turn = 0;  // Count$ThisTurnCast_Instant.YouCtrl,Sorcery.YouCtrl (Arclight Phoenix)
    std::vector<Entity> cards_drawn_this_turn;
    size_t cards_drawn_this_draw_step = 0;  // reset each turn; used to detect the first draw of a draw step (Orcish Bowmasters)
    // creature subtypes in this player's deck: pair<list_index, all_subtypes_index>
    std::vector<std::pair<int, int>> creature_subtypes;
};

#endif /* PLAYER_H */
