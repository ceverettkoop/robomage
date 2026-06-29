#ifndef SVAR_EVAL_H
#define SVAR_EVAL_H

#include <string>

#include "ecs/entity.h"
#include "components/zone.h"

// SVar evaluation shared across the static-ability, alt-cost, and castability-
// condition code. Extracted from state_manager.cpp so the same comparison and
// graveyard-count logic can be reused instead of re-inlined per call site.

// Apply a two-letter comparison operator (EQ/NE/GE/LE/GT/LT) to lhs and rhs.
// Returns false for an unrecognised operator. Shared by every SVar comparator so
// the operator table lives in one place (state_manager statics + ability.cpp).
bool apply_svar_op(int lhs, const std::string &op2, int rhs);

// Compare an integer against a Forge-style comparator string ("GE4", "LT2", ...).
// Recognised prefixes: EQ, NE, GE, LE, GT, LT. Returns false for anything else.
bool compare_svar(int value, const std::string &compare);

// Evaluate a StaticAbility SVar expression (e.g. "Count$TypeInYourYard.Land",
// "Count$ValidGraveyard Land.YouOwn") to an integer, from `controller`'s view.
// `source` is the permanent the SVar belongs to; only expressions scoped to a
// specific source (e.g. "Count$ValidExile ... CardTypes", which counts card types
// among that permanent's exiled-with pile) need it — others ignore it.
int evaluate_sa_svar(const std::string &expr, Zone::Ownership controller, Entity source = 0);

// Per-permanent stored-SVar trigger gate (Carpet of Flowers' once-per-turn CheckSVar latch): read
// `source`'s Permanent::stored_svars[name] (absent reads as 0) and test it against `compare`
// ("EQ0", "GE2", ...). An empty name means "no gate" and passes; an empty/unrecognized comparator
// fails. Used at both trigger placement (state_manager_triggers) and resolution (ability.cpp).
bool stored_svar_gate_passes(Entity source, const std::string &name, const std::string &compare);

#endif /* SVAR_EVAL_H */
