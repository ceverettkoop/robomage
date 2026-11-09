#ifndef RESOLUTION_MEMORY_H
#define RESOLUTION_MEMORY_H

#include <cstdint>
#include <string>

#include "components/memory_ops.h"
#include "ecs/entity.h"

struct Ability;
struct AbilityDef;

// The memory table's reader for param `key`: true when `key` is one of its rows, whatever the
// value; a value the row's form takes adds the row's op to `def`.
bool read_memory_param(AbilityDef &def, const std::string &key, const std::string &value);

// Adds the rows of `def`'s effect kind: what it writes with no param asking.
void bind_memory_ops(AbilityDef &def);

// An effect reporting to the step: `sources` happened, with no object yet (a REPLACE row then
// empties its slot even if none follows). Does nothing unless a row of `ab` reads them.
void report_effect(Ability &ab, uint32_t sources);
// An effect reporting object `e` from `sources`, stamped now. Does nothing unless a row of `ab`
// reads them.
void report_object(Ability &ab, uint32_t sources, Entity e);
// Reports `ab`'s live targets as EffectObjects::TARGETS.
void report_targets(Ability &ab);
// True if `ab` has reported `e`, as the object it is now, since the step last ran.
bool reported_object(const Ability &ab, Entity e);

// Drops `e` from the remembered set.
void forget_remembered(Entity e);

// The step: writes what `ab` has reported into the resolution's memory through its definition's
// ops, and drops the reports.
void apply_memory_ops(Ability &ab);

#endif /* RESOLUTION_MEMORY_H */
