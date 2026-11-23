#ifndef RMWORK_INSPECTOR_VIEW_H
#define RMWORK_INSPECTOR_VIEW_H

#include <string>
#include <vector>

#include "links.h"
#include "tool_data.h"

// ── The Decision Inspector: the decision's facts, ImHex "Data Inspector" style ──────────────────
//
// A Name / Value grid: the decision (n of N), game, turn and active player, step, the deciding seat
// and its agent, the pending query (the `pending` row's tag and source) and the cost being paid
// (`pending_cost`'s remaining pips, life, energy, discards) when there is one, the stack's depth,
// the menu's size; from the decision's episode, when the viewer sees the decision: V per evaluated
// policy on the chosen scale (the deciding seat's own, or Player A's), H(π) and log π(chosen) of the
// shown policy, μ's support, the policy version and the teacher's label against the chosen row.
// Under it the value-scale radio, the shown policy's radio (with two or more), and the menu: per
// row π of every evaluated policy as a bar, μ, ► on the chosen row.

namespace rmwork {

struct InspectorField {
        std::string name, value;
        bool has_sign = false;  // the value is signed (coloured by its sign)
        double sign = 0;
};

// The grid's rows for the current decision.
std::vector<InspectorField> inspector_fields(const ToolData &data);

class InspectorView {
    public:
        void draw(ToolData &data, LinkState *links);
};

}  // namespace rmwork

#endif /* RMWORK_INSPECTOR_VIEW_H */
