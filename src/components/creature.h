#ifndef CREATURE_H
#define CREATURE_H

#include <cstdint>
#include <set>
#include <string>
#include <vector>
#include "../ecs/entity.h"
#include "../classes/colors.h"

struct Creature {
    // Effective (current) power/toughness. These are CACHED values derived from the
    // signed contributions below via recompute_pt(). Never mutate them directly — change
    // a contribution and call recompute_pt(). They stay uint32_t so the ~75 readers
    // (combat, damage, ML serialization) are unchanged.
    uint32_t power = 0;
    uint32_t toughness = 0;

    bool is_attacking = false;
    Entity attack_target = 0;  // Entity of player or planeswalker being attacked (0 = none)
    bool is_blocking = false;
    Entity blocking_target = 0;  // Entity of attacker being blocked (0 = none)
    bool is_blocked = false;     // attacker was blocked at declare-blockers; stays blocked even if all blockers leave (509.1h)
    std::vector<std::string> keywords;
    bool must_attack = false;        // set by MustAttack static ability; enforced in declare_attackers

    // --- P/T contributions (all signed; effective = sum, floored at 0) ---
    int base_power = 0;              // characteristic base P (printed, token, CDA-set, or Pump-modified)
    int base_toughness = 0;          // characteristic base T
    int plus_one_counters = 0;       // +1/+1 counters (symmetric; affects both P and T)
    int prowess_bonus = 0;           // temporary +N/+N from Prowess/Exalted; cleared at cleanup step
    int eot_power_bonus = 0;         // temporary +N/-N P from "until end of turn" pumps; cleared at cleanup step
    int eot_toughness_bonus = 0;     // temporary +N/-N T from "until end of turn" pumps; cleared at cleanup step
    int static_power_bonus = 0;      // P from continuous static abilities; recomputed each SBE pass
    int static_toughness_bonus = 0;  // T from continuous static abilities; recomputed each SBE pass
};

// Recompute the cached effective power/toughness from the signed contributions.
// Uses signed arithmetic and floors at 0 so a transient negative (e.g. -X/-X pump or
// a mid-recompute static revert) can never underflow the uint32_t fields.
void recompute_pt(Creature &cr);

std::set<Colors> get_protection_colors(const Creature &cr);
bool has_protection_from(const Creature &cr, Entity source);

#endif /* CREATURE_H */
