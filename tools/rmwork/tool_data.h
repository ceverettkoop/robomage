#ifndef RMWORK_TOOL_DATA_H
#define RMWORK_TOOL_DATA_H

#include <string>
#include <vector>

#include "decision_frame.h"
#include "evaluations.h"
#include "replay_session.h"
#include "table_view.h"

// ── What the tools beside the Table share: the decision's frame and models, the evaluations ─────
//
// refresh() reads the session's decision in its viewer's frame once per change of decision or
// viewer (the session's generation): the DecisionFrame and its FrameModels. The match's
// evaluations (evaluations.h: the episodes' and the re-evaluations merged with them, reevaluation.h)
// are read once per match, set of joined episodes and change of its evaluations. The Inspector's
// choices are kept here because other tools follow them: the policy whose π and values the tools
// show (-1: the first evaluated) and the value scale (the deciding seat's own, or Player A's).
//
// What a viewer may see of the evaluations: the omniscient viewer every seat's values; a seat only
// its own (its frames' evaluations), so a seat view's series, swings and reads are that seat's.

namespace rmwork {

class ToolData {
    public:
        // The session's current decision and viewer; false (with error()) when its frame cannot be
        // read or nothing is loaded.
        bool refresh(ReplaySession &s, TableView &table);
        bool ok() const { return ok_; }
        const std::string &error() const { return error_; }
        const DecisionFrame &frame() const { return frame_; }
        const FrameModels &models() const { return models_; }
        unsigned generation() const { return generation_; }
        // The match's evaluations (empty before a match).
        const MatchValues &values() const { return values_; }
        // The policy the tools show (the selected one when evaluated, else the first), -1 when none.
        int64_t shown_policy() const { return values_.resolve(policy); }
        // The seats whose evaluations viewer `view` sees: both for the omniscient viewer, else its own.
        static bool sees_seat(int view, int seat) { return view == VIEW_OMNI || view == seat; }
        // The largest swings the viewer sees (its seats, the shown policy), at most `n`.
        std::vector<Swing> swings(int view, size_t n) const;

        int64_t policy = -1;   // the Inspector's choice; -1 the first
        bool scale_a = false;  // values on Player A's scale (else the deciding seat's)

    private:
        bool ok_ = false;
        bool have_ = false;
        unsigned generation_ = 0;
        std::string error_;
        DecisionFrame frame_;
        FrameModels models_;
        const Match *values_match_ = nullptr;
        std::string values_dump_;
        size_t values_steps_ = 0;
        unsigned values_changed_ = 0;
        MatchValues values_;
};

}  // namespace rmwork

#endif /* RMWORK_TOOL_DATA_H */
