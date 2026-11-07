#ifndef TARGET_GATE_H
#define TARGET_GATE_H

#include <vector>

#include "components/ability.h"
#include "ecs/entity.h"
#include "resolution_frame.h"

// One pick of a target (run_target_select): the ability choosing, how many targets it holds for
// this instance of the word "target" (the first `picked` of ability.targets), and what the
// caller asks of a candidate, when it asks anything.
struct TargetPick {
    const Ability &ability;
    int picked;
    const TargetPickGate *gate;
};

// Narrow `pool`, the pick's legal candidates, through the constraints across a spell's or
// ability's targets: a row of target_gate.cpp's table each. A constraint on a group of targets is
// a row there, read at every pick of every stack object.
void apply_target_gate(std::vector<Entity> &pool, const TargetPick &pick);

#endif /* TARGET_GATE_H */
