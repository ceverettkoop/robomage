#ifndef PLAYER_H
#define PLAYER_H

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "../classes/colors.h"
#include "../ecs/entity.h"

struct Player {
    int32_t life_total = 20;
    int32_t life_gained_this_turn = 0;  // total life gained this turn (any source); reset each turn (Ocelot Pride end-step check)
    bool has_city_blessing = false;     // 702.131c: city's blessing designation, kept for the rest of the game once obtained
    // 122.1: every kind of counter a player can have, keyed by counter type. Poison lives
    // here under "POISON" (replacing the old dedicated poison_counters field); energy ({E})
    // lives under "ENERGY" (CR 122.1c). Mutate via the get/add helpers below so all counter
    // kinds share one store. An entry is absent (not 0) when the player has none of that type.
    std::map<std::string, int> counters;
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

    // Number of counters of `type` on this player (0 if none). Single read path for the
    // counter map so every counter kind (POISON, ENERGY, …) is queried the same way.
    int counter_count(const std::string &type) const {
        auto it = counters.find(type);
        return it == counters.end() ? 0 : it->second;
    }
    // Add `delta` counters of `type` to this player (delta may be negative). An entry
    // reaching exactly 0 is erased so the map only holds live counters. Returns the new total.
    int add_counters(const std::string &type, int delta) {
        if (delta == 0) return counter_count(type);
        int total = counters[type] + delta;
        if (total == 0) counters.erase(type);
        else counters[type] = total;
        return total;
    }
};

#endif /* PLAYER_H */
