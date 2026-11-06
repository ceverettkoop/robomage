#ifndef SVAR_EVAL_H
#define SVAR_EVAL_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ecs/entity.h"
#include "components/zone.h"

struct ResolveCtx;

// SVar evaluation: the one evaluator for every Forge count expression a card script names
// (Count$..., Targeted$..., Remembered$..., PlayerCount...$..., integer literals) and the one
// comparator for its "GE4"-style comparisons, shared by effect amounts, target counts, resolution
// conditions, static abilities, alternative-cost conditions, replacement and trigger gates.
//
// An expression is compiled once, by text, through the tables of svar_eval.cpp: a base (a row of
// its own, or a set, a property and an aggregate: <Set>$<Property>) and its trailing steps (the
// suffix table). What --parse-report calls recognized is what these tables read, so a new
// expression, set, property or suffix is one row.

// A two-letter comparison operator, read once from a script (UNKNOWN: any other spelling).
enum class SvarOp : uint8_t { EQ, NE, GE, LE, GT, LT, UNKNOWN };
SvarOp svar_op_of(const std::string &op2);

// Apply a two-letter comparison operator (EQ/NE/GE/LE/GT/LT) to lhs and rhs. Any other operator
// is a script the engine can't read: reported as an error, and the comparison fails.
bool apply_svar_op(int lhs, const std::string &op2, int rhs);
bool apply_svar_op(int lhs, SvarOp op, int rhs);

// Compare an integer against a Forge-style comparator string ("GE4", "LT2", ...): a two-letter
// operator followed by an integer. A malformed comparator is reported as an error and fails.
bool compare_svar(int value, const std::string &compare);

// Evaluate an SVar expression to an integer from `controller`'s view ("you"). `source` is the
// object the expression belongs to (Count$CardCounters, Count$ResolvedThisTurn, +Other, the
// cards exiled with it), `target` the ability's chosen target (Targeted$CardPower,
// TargetedPlayerCtrl). Trailing /Plus.N, /Times.N, /HalfUp, /LimitMax.N and /DivideEvenlyDown.N adjust
// the value. An expression the engine doesn't recognize is reported as an error and evaluates to 0.
int evaluate_svar(const std::string &expr, Zone::Ownership controller, Entity source = 0,
                  Entity target = 0);

// The same, reading the expression from `ctx`: its controller's view, its source, and the chosen
// target of its ability instance (none for a static).
int evaluate_svar(const std::string &expr, const ResolveCtx &ctx);

// True when evaluate_svar has a reading for `expr` (it would not report it unrecognized), checked
// without a game (--parse-report, parse_report.h).
bool svar_expr_recognized(const std::string &expr);

// How a script spells each expression that names a filter over a set, up to the filter
// ("Count$Valid ", "Remembered$Valid "): what --parse-report checks against the filter matcher.
std::vector<std::string> svar_filter_spellings();

// evaluate_svar as an amount (a number of cards, counters, damage, ...): never below 0.
size_t evaluate_amount(const std::string &expr, Zone::Ownership controller, Entity source = 0,
                       Entity target = 0);
size_t evaluate_amount(const std::string &expr, const ResolveCtx &ctx);

// Per-permanent stored-SVar trigger gate (a trigger's CheckSVar$ on a Number$ scratch SVar): read
// `source`'s Permanent::stored_svars[name] (absent reads as 0) and test it against `compare`
// ("EQ0", "GE2", ...). An empty name means "no gate" and passes. Used at both trigger placement
// (state_manager_triggers) and resolution (resolution.cpp).
bool stored_svar_gate_passes(Entity source, const std::string &name, const std::string &compare);

#endif /* SVAR_EVAL_H */
