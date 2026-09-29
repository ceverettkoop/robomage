#ifndef QUERIES_ACTIVATION_H
#define QUERIES_ACTIVATION_H

#include <set>
#include "../components/ability.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/entity.h"

// Activations counted against a permanent's once-per-turn gates: every ability's
// ActivationLimit$ counter (Ability::activations_this_turn, only advanced for a limited
// ability) summed, plus 1 if one of its loyalty abilities was activated (CR 606.3). Both
// reset for every battlefield permanent at each untap step (reset_permanent_activations_this_turn).
inline int permanent_activations_this_turn(const Permanent &perm) {
    int n = perm.loyalty_ability_activated_this_turn ? 1 : 0;
    for (const auto &ab : perm.abilities) n += ab.activations_this_turn;
    return n;
}

// Clear the once-per-turn activation gates counted by permanent_activations_this_turn. Called
// for every battlefield permanent, whichever player controls it, as each turn begins: "Activate
// only once each turn" (CR 602.5b) counts the opponent's turns too.
inline void reset_permanent_activations_this_turn(Permanent &perm) {
    for (auto &ab : perm.abilities) ab.activations_this_turn = 0;
    perm.loyalty_ability_activated_this_turn = false;
}

// Number of this turn's triggered-ability resolutions whose source is `source`
// (Game::ability_resolution_counts — the Count$ResolvedThisTurn value, Scythecat Cub).
// Defined in activation.cpp (needs cur_game).
int ability_resolutions_this_turn(Entity source);

// ── Activation conditions (CR 602.5 "activate only if …") ───────────────────
// Named gates that make an activated ability illegal to activate unless the named
// condition holds for its controller. Parsed from Activation$ <name>; evaluated at
// activation-legality time (mana-source enumeration, non-mana activated enumeration,
// and the action processor guard). Keep general: add a named condition + a case in
// activation_condition_met() rather than special-casing one card.

// Metalcraft (CR 702.46): the player controls three or more artifacts. The activating
// permanent (e.g. Mox Opal itself, an artifact) is counted. Reusable by any Metalcraft card.
bool controller_has_metalcraft(Zone::Ownership controller, const std::set<Entity> &entities);

// True if `ab`'s Activation$ gate (if any) is satisfied for `controller`. An ability with
// no activation_condition is always allowed (returns true). `source` is the gated ability's
// source permanent (needed by per-permanent gates like NotMonstrous; pass 0 if unknown).
// Unknown condition names fail closed (return false) so a misparsed gate never silently
// permits activation.
bool activation_condition_met(const Ability &ab, Zone::Ownership controller,
                              const std::set<Entity> &entities, Entity source = 0);

// The state `source` must be in for `controller` to activate its ability `ab` (CR 602.5): the
// Activation$ condition holds, a {T} in the cost finds an untapped permanent free of summoning
// sickness (CR 302.6), and the per-turn activation limit isn't reached. Shared by the mana-source
// enumeration and the non-mana activated-ability offers, whichever zone the source is in.
bool activation_source_ready(const Ability &ab, Entity source, Zone::Ownership controller,
                             const std::set<Entity> &entities);

#endif /* QUERIES_ACTIVATION_H */
