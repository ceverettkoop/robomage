#ifndef SVAR_EVAL_H
#define SVAR_EVAL_H

#include <string>

#include "components/zone.h"

// SVar evaluation shared across the static-ability, alt-cost, and castability-
// condition code. Extracted from state_manager.cpp so the same comparison and
// graveyard-count logic can be reused instead of re-inlined per call site.

// Compare an integer against a Forge-style comparator string ("GE4", "LT2", ...).
// Recognised prefixes: EQ, NE, GE, LE, GT, LT. Returns false for anything else.
bool compare_svar(int value, const std::string &compare);

// Evaluate a StaticAbility SVar expression (e.g. "Count$TypeInYourYard.Land",
// "Count$ValidGraveyard Land.YouOwn") to an integer, from `controller`'s view.
int evaluate_sa_svar(const std::string &expr, Zone::Ownership controller);

#endif /* SVAR_EVAL_H */
