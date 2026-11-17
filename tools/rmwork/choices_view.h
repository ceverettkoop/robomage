#ifndef RMWORK_CHOICES_VIEW_H
#define RMWORK_CHOICES_VIEW_H

#include <vector>

#include "links.h"
#include "tool_data.h"

// ── The Choices tool: the decision's menu rows ──────────────────────────────────────────────────
//
// The `menu` rows of the viewer's frame, each with its description; only in the deciding seat's and
// the omniscient frame (a non-deciding seat's view says whose decision it is and shows nothing of
// it). In Replay ► marks the row the decision log chose; under each row a bar of the shown policy's
// π (ToolData::shown_policy, the Inspector's choice), with π and μ (when the episode has it) as
// numbers (the first policy's on a decision the shown one did not evaluate). ↑ / ↓ move a highlight while the tool has
// focus (it starts on the chosen row); hovering a row links its source and target objects and players (the Table
// outlines them), and a row an object hovered elsewhere acts with or on is shaded. Enter or a double-click submits the
// highlighted row (Play's choosing; take_submitted), a click selects it.

namespace rmwork {

class ChoicesView {
    public:
        void draw(ToolData &data, LinkState *links, bool play);
        // The highlighted row (-1 none).
        int highlighted() const { return highlight_; }
        // Screenshots and checks: hover menu row `row` every frame (-1 off).
        void force_hover(int row) { force_hover_ = row; }
        // The row Enter or a double-click submitted, -1 none; taken once.
        int take_submitted();
        // The evaluation whose π the tool shows: the shown policy's, else the decision's first
        // (nullptr when the viewer sees none, or it is not over the menu's rows).
        static const featurize::PolicyEvalRecord *shown_record(const ToolData &data);
        // Its π per menu row (empty when none).
        static std::vector<double> shown_pi(const ToolData &data);

    private:
        int highlight_ = -1;
        unsigned generation_ = ~0u;
        int force_hover_ = -1;
        int submitted_ = -1;
        bool scroll_to_ = false;
};

}  // namespace rmwork

#endif /* RMWORK_CHOICES_VIEW_H */
