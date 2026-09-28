#ifndef SVAR_EVAL_H
#define SVAR_EVAL_H

#include <cstddef>
#include <string>

#include "ecs/entity.h"
#include "components/zone.h"

// SVar evaluation: the one evaluator for every Forge count expression a card script names
// (Count$..., Targeted$..., Remembered$..., PlayerCount...$..., integer literals) and the one
// comparator for its "GE4"-style comparisons, shared by effect amounts, target counts, resolution
// conditions, static abilities, alternative-cost conditions, replacement and trigger gates.

// Apply a two-letter comparison operator (EQ/NE/GE/LE/GT/LT) to lhs and rhs. Any other operator
// is a script the engine can't read: reported as an error, and the comparison fails.
bool apply_svar_op(int lhs, const std::string &op2, int rhs);

// Compare an integer against a Forge-style comparator string ("GE4", "LT2", ...): a two-letter
// operator followed by an integer. A malformed comparator is reported as an error and fails.
bool compare_svar(int value, const std::string &compare);

// Evaluate an SVar expression to an integer from `controller`'s view ("you"). `source` is the
// object the expression belongs to (Count$CardCounters, Count$ResolvedThisTurn, +Other, the
// cards exiled with it), `target` the ability's chosen target (Targeted$CardPower,
// TargetedPlayerCtrl). Trailing /Plus.N, /Times.N, /HalfUp and /LimitMax.N adjust the value. An
// expression the engine doesn't recognize is reported as an error and evaluates to 0.
int evaluate_svar(const std::string &expr, Zone::Ownership controller, Entity source = 0,
                  Entity target = 0);

// evaluate_svar as an amount (a number of cards, counters, damage, ...): never below 0.
size_t evaluate_amount(const std::string &expr, Zone::Ownership controller, Entity source = 0,
                       Entity target = 0);

// Per-permanent stored-SVar trigger gate (Carpet of Flowers' once-per-turn CheckSVar latch): read
// `source`'s Permanent::stored_svars[name] (absent reads as 0) and test it against `compare`
// ("EQ0", "GE2", ...). An empty name means "no gate" and passes. Used at both trigger placement
// (state_manager_triggers) and resolution (resolution.cpp).
bool stored_svar_gate_passes(Entity source, const std::string &name, const std::string &compare);

#endif /* SVAR_EVAL_H */
