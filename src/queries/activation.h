#ifndef QUERIES_ACTIVATION_H
#define QUERIES_ACTIVATION_H

#include <set>
#include <string>

#include "../components/ability.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/entity.h"
#include "../ecs/entity_set.h"

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

// ResolvedLimit$ (CR 603.4): true when `ab` is a limited triggered ability that has already
// resolved its limit's number of times this turn for its source object (ab.source as it is now)
// and controller (Game::limited_resolution_counts). A trigger in that state doesn't trigger, and
// one already on the stack does nothing.
bool trigger_resolved_limit_reached(const Ability &ab);
// Records that `ab` resolves (CR 608.2: its optional "you may" accepted, its intervening-if and
// targets held, before its effects run; an ability that does nothing is not recorded): one
// resolution toward its ResolvedLimit$ (no-op for an unlimited ability, or once its source is no
// longer the object that triggered) and, for a triggered ability resolving from the stack
// (`from_stack`), one toward its source's Count$ResolvedThisTurn
// (Game::ability_resolution_counts).
void count_resolution(const Ability &ab, bool from_stack);

// How many times the ActivationLimit$ ability `def` of `source` triggered this turn, as the object
// `source` is now.
int triggers_this_turn(const AbilityDef *def, Entity source);
// Records that `def` of `source` triggered: one toward its ActivationLimit$.
void count_limited_trigger(const AbilityDef *def, Entity source);

// ── Activation rules (CR 602.5) ──────────────────────────────────────────────
// What restricts activating an ability is the list on its definition (ActivationRule), evaluated
// at activation-legality time (mana-source enumeration, non-mana activated enumeration, and the
// action processor guard). Keep general: add a row to the table of named conditions rather than
// special-casing one card.

// Metalcraft (CR 702.46): the player controls three or more artifacts. The activating
// permanent (e.g. Mox Opal itself, an artifact) is counted. Reusable by any Metalcraft card.
bool controller_has_metalcraft(Zone::Ownership controller, const EntitySet &entities);

// The row of a named condition (Activation$ <name>). An unknown name gets a row that never holds,
// so a misparsed gate never silently permits activation.
const ActivationCondition *activation_condition_named(const std::string &name);
// Is `name` a row of the table, not the row that never holds?
bool activation_condition_known(const std::string &name);

// Is `def` activated only as a sorcery (CR 602.5d)?
bool activates_as_sorcery(const AbilityDef &def);

// Do `ab`'s rules let `controller` activate it from `source` now (CR 602.5), and isn't the
// per-turn activation limit reached? `sorcery_window` is whether a sorcery could be cast: the
// mana-source enumeration leaves it set, no mana ability having a timing rule. Shared by that
// enumeration and the non-mana activated-ability offers, whichever zone the source is in; a {T}
// in the cost is source_can_tap's.
bool activation_source_ready(const Ability &ab, Entity source, Zone::Ownership controller, const EntitySet &entities,
    bool sorcery_window = true);

// Can a {T} in a cost be paid by `source`: an untapped permanent free of summoning sickness
// (CR 302.6)?
bool source_can_tap(Entity source);

#endif /* QUERIES_ACTIVATION_H */
