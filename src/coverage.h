#ifndef COVERAGE_H
#define COVERAGE_H

#include <string>

struct Ability;
enum class ActionCategory;
enum class EffectKind;

// ── --coverage FILE: which engine paths a run reached ────────────────────────────────────────
//
// Hit counts per key, written to FILE when the process exits (atomically, write_file_atomic) as
// `<kind> <name> <count>` lines sorted by kind and name, so a fixed seed gives the same file. The
// `fuzz` tier sums them over a fixture deck's games and fails on an `.expect` key with no hits
// (docs/ci.md). Nothing here reads the RNG or the game, or prints; without --coverage every call
// below returns at once.
//
// The counts live in this file and not in `Game` or the ECS on purpose: a snapshot restore (the
// --selfcheck-state snapshot check, a search agent) must not rewind them. They count what the
// process ran, so work a restore repeats is counted again.

// --coverage FILE: starts counting and arranges the write at exit (a run ended by fatal_error
// writes too; one killed by a signal does not).
void coverage_open(const std::string &path);

// effect <category>: an ability finished its one dispatch through effects::handler_for (an
// ability whose handler suspended counts when the re-entry finishes it).
void coverage_effect(EffectKind kind);
// trigger <mode>: a triggered ability went on the stack, or resolved at once if it uses no stack.
// The mode is its T: line's Mode$; a trigger with no such line is named by what made it: `Delayed`,
// its keyword or engine key, else `Reflexive`.
void coverage_trigger(const Ability &ab);
// replacement <kind>: a replacement effect was applied to a real event (a mana simulation's is not
// counted).
void coverage_replacement(const char *kind);
// category <ActionCategory>: a decision was asked and answered, by the category of the row taken.
// A one-row menu is never asked, and a concession takes no row, so neither counts.
void coverage_decision(ActionCategory category);
// row <name>: a filter or SVar table row was read that no other kind tells from its neighbours.
void coverage_row(const char *name);
// cost <part>: a cast or activation paid this part of its cost.
void coverage_cost(const char *part);
// cast <mode>: a spell became cast with this LegalAction flag set (one hit per flag), or a special
// action of that kind was taken.
void coverage_cast(const char *mode);

#endif /* COVERAGE_H */
