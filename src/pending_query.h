#ifndef PENDING_QUERY_H
#define PENDING_QUERY_H

#include <cstdint>
#include <vector>

#include "classes/action.h"
#include "ecs/entity.h"

// A mid-flow decision parked for the main loop to emit (the suspension half of
// the snapshot-safe decision protocol). The suspending site stores the EXACT
// menu it would have prompted with — the loop top re-emits it verbatim rather
// than re-deriving it, so re-emission after a snapshot restore is trivially
// byte-identical (nothing mutates between suspend and resume). Lives on Game by
// value so a cur_game copy covers it for free. At most one is active game-wide.
struct PendingQuery {
    // Which flow parked the decision — the loop-top dispatcher switches on this
    // after the answer is read to hand control back to the right resume path.
    enum Tag {
        NONE,
        RESOLUTION,     // effect handler under StackManager::resolve_top
        SBE_LATCHED,    // SBE-pass prompt re-derived on resume (legend rule, ETB choices)
        TURN_DRAW,      // draw-step replacement (dredge) mini-frame
        CAST,           // PendingCast state machine (cast-time prompts)
        ACTIVATION,     // PendingActivation state machine
        ATTACK_TARGET,  // declare-attackers target sub-prompt
        BLOCK_TARGET,   // declare-blockers target sub-prompt
        DAMAGE_ASSIGN,  // combat damage assignment order
        TRIGGER_PLACE   // trigger ordering / trigger target selection
    };
    Tag tag = NONE;
    bool active = false;
    std::vector<LegalAction> menu;  // the suspended menu, re-emitted verbatim
    bool chooser_is_a = true;       // who answers (priority is persisted here on suspend)
    Entity decision_source = 0;     // pending-decision context for the observation
    uint64_t key = 0;               // site key for latched-answer re-convergence (SBE flavor)
    bool answered = false;
    int answer = -1;
};

#endif /* PENDING_QUERY_H */
