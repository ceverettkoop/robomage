#include "tool_data.h"

namespace rmwork {

bool ToolData::refresh(ReplaySession &s, TableView &table) {
    if (!s.loaded()) {
        ok_ = have_ = false;
        error_ = "No game loaded";
        values_match_ = nullptr;
        values_ = MatchValues();
        frame_ = DecisionFrame();
        models_ = FrameModels();
        return false;
    }
    const Match &m = *s.match();
    if (values_match_ != &m || values_dump_ != m.dump_path || values_steps_ != m.episode_steps.size() ||
        values_changed_ != m.evaluations_changed) {
        values_match_ = &m;
        values_dump_ = m.dump_path;
        values_steps_ = m.episode_steps.size();
        values_changed_ = m.evaluations_changed;
        // A merged evaluation changes what the decision's tools show (the frame's step is the same).
        have_ = false;
        values_ = match_values(m);
    }
    if (have_ && s.generation() == generation_) return ok_;
    have_ = true;
    generation_ = s.generation();
    error_.clear();
    ok_ = load_decision_frame(m, s.cursor(), s.view(), table.book(m, s.view()), &frame_, &error_);
    models_ = ok_ ? build_frame_models(frame_) : FrameModels();
    return ok_;
}

std::vector<Swing> ToolData::swings(int view, size_t n) const {
    if (!values_match_ || !values_.any()) return {};
    return top_swings(*values_match_, values_, view == VIEW_OMNI ? -1 : view, shown_policy(), n);
}

}  // namespace rmwork
