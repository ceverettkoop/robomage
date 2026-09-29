#ifndef QUERIES_COMBAT_H
#define QUERIES_COMBAT_H

#include <cstdint>
#include <set>
#include <vector>
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"
#include "battlefield.h"
#include "keywords.h"

// ── Combat (CR 506-511) ───────────────────────────────────────────────────────

// True if `e` is a live battlefield creature that is attacking / blocking. A creature removed
// from combat (CR 506.4) or phased out (CR 702.26b) is neither.
inline bool is_attacking_creature(Entity e) {
    return is_battlefield_permanent(e) && global_coordinator.entity_has_component<Creature>(e) &&
           global_coordinator.GetComponent<Creature>(e).is_attacking;
}
inline bool is_blocking_creature(Entity e) {
    return is_battlefield_permanent(e) && global_coordinator.entity_has_component<Creature>(e) &&
           global_coordinator.GetComponent<Creature>(e).is_blocking;
}

// True if `e` is an unblocked attacking creature (CR 509.1h): an attacking creature that no
// creature blocked, from the moment blockers are declared until it is removed from combat or
// combat ends. Before blockers are declared an attacker is neither blocked nor unblocked. A
// creature put onto the battlefield attacking after blockers are declared is unblocked. Defined
// in combat.cpp (needs cur_game).
bool is_unblocked_attacker(Entity e);

// The live battlefield creatures blocking `attacker` (CR 509.1g), in entity order. Shared by the
// combat-damage step and its damage-assignment prompt so both divide damage among the same set.
std::vector<Entity> blockers_of(Entity attacker, const std::set<Entity> &entities);

// CR 302.6: a creature can't attack, and its abilities with {T} in the cost can't be activated,
// unless it has been under its controller's control continuously since their most recent turn
// began — or it has haste (CR 702.10b). True when `e` is a creature under that restriction now.
// Single source for the attack-eligibility and tap-cost gates.
inline bool is_summoning_sick(Entity e) {
    if (!global_coordinator.entity_has_component<Creature>(e) ||
        !global_coordinator.entity_has_component<Permanent>(e))
        return false;
    return global_coordinator.GetComponent<Permanent>(e).has_summoning_sickness &&
           !permanent_has_keyword(e, "Haste");
}

// True if the creature deals damage during the first-strike combat damage step
// (it has First Strike or Double Strike). Single source for "does a first-strike
// damage step matter": the step-skip scan and the per-creature damage gate both
// consume this so they cannot drift on the keyword literals.
inline bool creature_deals_first_strike_damage(const Creature &cr) {
    return creature_has_keyword(cr, "First Strike") ||
           creature_has_keyword(cr, "Double Strike");
}

// Damage `attacker` must assign to `blocker` for that blocker to count as receiving
// lethal damage (used both for the auto-assign path and the "can it kill everything?"
// threshold). Deathtouch makes any nonzero amount lethal (702.2c); otherwise lethal is
// the blocker's remaining toughness after damage already marked on it (702.19b / T3.11).
uint32_t lethal_needed_for_blocker(Entity attacker, Entity blocker);

#endif /* QUERIES_COMBAT_H */
