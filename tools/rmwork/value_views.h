#ifndef RMWORK_VALUE_VIEWS_H
#define RMWORK_VALUE_VIEWS_H

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "tool_data.h"

// ── The tools of the recorded evaluations: the Value Graph, Swings, the Model's Read ───────────
//
// Value Graph (ImPlot): over the match's decisions, on Player A's scale, V of each seat by each
// evaluated policy the viewer sees (ToolData::sees_seat; a B decision's value negated), turn bands,
// game boundaries, each game's result (z) and a life strip (both players' life, public at every
// decision); the current decision's cursor; a click or drag scrubs; hovering names the decision; a
// legend of toggles. Without evaluations: the life strip and a note.
//
// Swings: the viewer's seats' largest value changes (evaluations.h), with the turn and what
// happened between in the viewer's own log; in the omniscient view also the seats' disagreement.
// A click on a row goes to its decision.
//
// Model's Read: the deciding seat's (a seat view: the viewer's own) value at the current decision
// (else its last), the row the shown policy prefers when that seat decides, the model, a sparkline
// of the seat's values so far.

namespace rmwork {

// Inside an ImPlot plot whose x axis is the match's decisions: a click in the plot (`hovered`) starts a
// drag, and while the left button is held the session's cursor follows the decision under the mouse.
void plot_scrub(ReplaySession &s, bool hovered, bool *dragging);

// One line of the Value Graph's legend.
struct GraphSeries {
        int seat = 0;
        int64_t policy = 0;
        std::string label;  // "V (A) · p0 v9834", "−V of B seat · p0 v9834"
        ImU32 color = 0;
        int dash = 0;  // 0 solid, 1 dashed, 2 dotted
};

class ValueGraphView {
    public:
        void draw(ToolData &data, ReplaySession &s);
        // The series the viewer's legend lists, in order.
        static std::vector<GraphSeries> series_for(const ToolData &data, int view);
        // The points (decision, value on Player A's scale) a series draws.
        static std::vector<std::pair<size_t, double>> points(const ToolData &data, int seat, int64_t policy);
        bool shown(int seat, int64_t policy) const;
        void set_shown(int seat, int64_t policy, bool on) { hidden_[{seat, policy}] = !on; }
        bool life_shown() const { return life_; }

    private:
        void draw_legend(const std::vector<GraphSeries> &series, bool any);
        void draw_values(ToolData &data, ReplaySession &s, const std::vector<GraphSeries> &series);
        void draw_life(ReplaySession &s);
        void scrub(ReplaySession &s, bool hovered);

        std::map<std::pair<int, int64_t>, bool> hidden_;
        bool life_ = true;
        bool dragging_ = false;
};

class SwingsView {
    public:
        void draw(ToolData &data, ReplaySession &s);
        inline static constexpr size_t ROWS = 12;
};

class ModelReadView {
    public:
        void draw(ToolData &data, ReplaySession &s);
        // The seat the tool reads: the decider, or in a seat view the viewer.
        static int seat_of(const ToolData &data);
        // That seat's values (shown policy) up to the current decision, and the decision of the last.
        static std::vector<double> history(const ToolData &data, size_t *last_decision);
};

}  // namespace rmwork

#endif /* RMWORK_VALUE_VIEWS_H */
