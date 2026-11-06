#ifndef MEMORY_OPS_H
#define MEMORY_OPS_H

#include <cstdint>
#include <vector>

#include "../object_ref.h"

// What a resolution remembers of its effects (CR 608.2c: a later instruction may act on what an
// earlier one did) is data on the definition: the MemoryOps parsed from its params, rows of the
// table in resolution_memory.cpp. A handler only reports what it did; apply_memory_ops, one step
// of resolve_ability, writes the resolution's memory from those reports.

// What an effect can report: the objects it did something to, or that it did nothing to them. A
// bit each, so a row can read several and one report can feed several rows.
struct EffectObjects {
    enum : uint32_t {
        TARGETS = 1u << 0,  // the ability's targets, as it starts to resolve
    };
};

// The part of the resolution's memory a row writes (ResolutionMemory).
enum class MemorySlot { REMEMBERED };

enum class MemoryAction {
    APPEND,   // add the reported objects
    REPLACE,  // the reported objects become the slot's whole content, none at all included
};

// One write: the objects reported from `sources` go to `slot`. Rows of an ability that share a
// slot and an action merge into one op, so an object reported from both goes in once.
struct MemoryOp {
    MemorySlot slot = MemorySlot::REMEMBERED;
    MemoryAction action = MemoryAction::APPEND;
    uint32_t sources = 0;
};

// One object an effect reported, stamped as it was then.
struct ReportedObject {
    uint32_t sources = 0;
    ObjectRef ref;
};

// What a resolving effect has reported so far and apply_memory_ops has not yet written. It lives
// on the resolving Ability, so a suspended effect's reports persist in its FrameLevel.
struct EffectReport {
    uint32_t reported = 0;  // the sources reported, with or without objects
    std::vector<ReportedObject> objects;
};

#endif /* MEMORY_OPS_H */
