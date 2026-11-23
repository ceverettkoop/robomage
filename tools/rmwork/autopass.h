#ifndef RMWORK_AUTOPASS_H
#define RMWORK_AUTOPASS_H

#include <cstdint>
#include <string>
#include <vector>

#include "../../src/agents/decision_view.h"
#include "settings.h"

// ── Play's client-side passing: F2, F6 and the per-step stops ───────────────────────────────────
//
// The engine asks the human seat at every decision with more than one row; the workbench answers
// some of them itself. What it reads of a decision is read_pass_facts' PassFacts, from the human
// seat's frame only, and what it answers is auto_reply's, a pure function of those facts, the pass
// state and the stops, so both are checked on recorded frames without a window or an engine.
//
// A priority pass is the menu row whose action category is PASS_PRIORITY, at a plain priority
// decision (the frame has no pending decision and no pending cost); its text is never read.
//
//   - F2 (to the end of the turn): passes at each plain priority decision of the turn it was given
//     in; it stops (asks) at the first decision of a later turn, game or sideboarding phase, at a
//     decision that is not a plain priority decision, and when the stack holds an object it did not
//     hold at the previous decision F2 answered (or when F2 was given).
//   - F6 (skip the turn): takes row 0 at every decision of the turn it was given in; it stops at the
//     first decision of a later turn, game or sideboarding phase.
//   - Stops (Settings → Stops, per step for your turn and the opponent's): with neither active, a
//     plain priority decision with an empty stack in a step without a stop for the turn's player is
//     passed.
// When F2 or F6 stops at a new turn, the stops still apply to that decision.

namespace rmwork {

struct PassFacts {
        int rows = 0;       // the menu's rows
        int pass_row = -1;  // the priority pass row, -1 when this is not a plain priority decision
        int game = 0;
        bool sideboarding = false;
        uint64_t turn = 0;
        bool own_turn = false;        // the frame's seat is the active player
        int step = -1;                // the stop step (settings.h), -1 outside the turn's steps
        std::vector<uint64_t> stack;  // the stack's objects' handles
};

// The facts of the decision `v` (the deciding seat's frame).
PassFacts read_pass_facts(const agents::DecisionView &v);

enum class PassMode {
    NONE,
    END_OF_TURN,  // F2
    SKIP_TURN     // F6
};

struct AutoPass {
        PassMode mode = PassMode::NONE;
        int game = 0;
        bool sideboarding = false;
        uint64_t turn = 0;
        std::vector<uint64_t> stack;  // the stack at the decision it last answered (or was given at)
};

// Starts F2 or F6 in the turn of `at` (the latest decision's facts).
void start_auto_pass(AutoPass *ap, PassMode mode, const PassFacts &at);

struct AutoReply {
        int row = -1;          // the row to send; -1: the human decides
        bool by_stop = false;  // a priority pass in a step without a stop
        std::string ended;     // F2 or F6 ended at this decision: why ("the turn ended", …)
};

// What the workbench answers at decision `f`; ends F2 / F6 (ap->mode NONE) where they stop.
AutoReply auto_reply(AutoPass *ap, const PassFacts &f, const StopSettings &stops);

}  // namespace rmwork

#endif /* RMWORK_AUTOPASS_H */
