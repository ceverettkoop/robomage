#ifndef RMWORK_POLICY_COMPARE_VIEW_H
#define RMWORK_POLICY_COMPARE_VIEW_H

#include <map>
#include <set>
#include <string>
#include <vector>

#include "layout.h"
#include "links.h"
#include "model_manager.h"
#include "policy_compare.h"
#include "reevaluation.h"
#include "settings.h"
#include "tool_data.h"

// ── The Policy Compare tool (docs/workbench_plan.md §2.10) ───────────────────────────────────────
//
// The header: a chip per policy the viewer can compare (policy_compare.h: a tick to compare it, a
// radio for the reference; every one ticked at first, the reference the last), "Add checkpoints…" and
// the metric (KL or TV). Then, top to bottom: the cursor's decision (per menu row each policy's π as a
// bar in its colour, its argmax ● and the chosen row ►; per policy V, KL and TV against the reference;
// hovering a row links it as Choices does); the plots over the match (KL or TV against the reference per
// policy, argmax differences marked, and V on Player A's scale; the cursor, a click or a drag scrubs); per
// policy the agreement (overall, per seat, mean KL and TV, a strip per turn); the ranked disagreements
// (a policy or all, "argmax differs only"), each row naming the reference's and the policy's preferred
// rows, a click going to its decision. The comparison is rebuilt when the match's evaluations, the
// viewer or the choices change.
//
// "Add checkpoints…" opens a popup of the policy catalog (policy_catalog.h: the rows that play an
// export, hidden ones left out) with each row's compatibility (model_manager.h's CompatChecker against
// the engine's tables; an incompatible row cannot be ticked) and a filter; "Serve and re-evaluate" asks
// the App to bring the ticked rows in (CompareRequest::BRING_IN, oldest first), which serves them and
// re-evaluates the match with them (App::compare_bring_in).

namespace rmwork {

struct CompareContext {
        Mode mode = Mode::REPLAY;
        ReplaySession *session = nullptr;
        ToolData *data = nullptr;
        LinkState *links = nullptr;
        const Settings *settings = nullptr;
        std::string engine;                // the engine binary whose tables the catalog rows are checked against
        std::vector<ServedPolicy> served;  // Settings → Models' served policies
        bool reevaluating = false;         // a re-evaluation runs
        std::string reeval_phase;          // its phase
        double reeval_fraction = -1;
        std::string note;  // what the App says of the last bring-in ("" none)
};

struct CompareRequest {
        enum class Kind {
            NONE,
            BRING_IN,  // serve `models` (oldest first) and re-evaluate the match with them
            SETTINGS   // Settings → Models
        };
        Kind kind = Kind::NONE;
        std::vector<std::string> models;
};

class PolicyCompareView {
    public:
        enum class State {
            NO_MATCH,     // nothing loaded
            NO_POLICIES,  // the viewer sees no evaluated policy
            ONE_POLICY,   // one: nothing to compare it with
            ONE_CHOSEN,   // fewer than two ticked
            READY
        };
        CompareRequest draw(const CompareContext &c);
        // Rebuilds the comparison for the session's match and viewer if they or the choices changed (draw
        // does it first; the checks call it alone).
        void update(const ReplaySession &s, const ToolData &data);
        State state() const { return state_; }
        const MatchComparison &comparison() const { return cmp_; }
        // The policies the viewer can compare (comparable_policies), as at the last update.
        const std::vector<int64_t> &candidates() const { return candidates_; }
        // The reference chosen (-1: the default, the last candidate).
        int64_t reference_choice() const { return ref_; }
        void set_reference(int64_t id) { ref_ = id; }
        // The reference by its policy_name ("p0", "re p1"), taken once a candidate has that name ("" none).
        void set_reference_name(const std::string &name) { ref_name_ = name; }
        bool included(int64_t id) const { return excluded_.count(id) == 0; }
        void set_included(int64_t id, bool on);
        CompareMetric metric() const { return metric_; }
        void set_metric(CompareMetric m) { metric_ = m; }
        bool argmax_only() const { return argmax_only_; }
        void set_argmax_only(bool on) { argmax_only_ = on; }
        // The policy the ranked list ranks (-1: every one but the reference).
        int64_t rank_policy() const { return rank_policy_; }
        void set_rank_policy(int64_t p) { rank_policy_ = p; }
        // The ranked list as at the last update.
        const std::vector<RankedDecision> &ranked() const { return ranked_; }
        // The cursor's decision as the comparison has it (nullptr when it has none: the viewer does not
        // see it, or the reference did not evaluate it), as at the last update.
        const DecisionCompare *current() const { return current_; }
        // Checks and screenshots: hover menu row `row` of the decision panel every frame (-1 off); at the
        // next draw, act as if ranked row `k` were clicked (-1 none); open the catalog popup at the next draw.
        void force_hover(int row) { force_hover_ = row; }
        void force_jump(int k) { force_jump_ = k; }
        void open_picker() { open_picker_ = true; }
        // Screenshots: the popup opened and kept open (opened again whenever something closes it).
        void force_picker(bool on) { force_picker_ = on; }
        bool picker_asked() const { return open_picker_ || (force_picker_ && !picker_shown_); }
        // The catalog's rows the popup lists (after it opened once), and whether row `i` is ticked.
        const std::vector<PolicyRow> &picker_rows() const { return rows_; }
        bool picker_open() const { return picker_shown_; }
        // The comparison as text (the command line's --compare-report): the policies, each one's agreement with
        // the reference and its mean KL and TV, then the `n` decisions the ranked list puts first with the rows
        // the reference and the policy prefer; updated for `s` first.
        std::string report(
            const ReplaySession &s, const ToolData &data, const std::vector<ServedPolicy> &served, size_t n);
        // The colour of the policy at position `k` of the compared list.
        static ImU32 policy_color(int k);

    private:
        void draw_header(const CompareContext &c, CompareRequest *req);
        void draw_decision(const CompareContext &c, float height);
        void draw_plots(const CompareContext &c, float height);
        void draw_agreement(const CompareContext &c);
        void draw_ranked(const CompareContext &c);
        void draw_picker(const CompareContext &c, CompareRequest *req);
        void scan_catalog(const CompareContext &c);
        // A table row's height; the agreement table's; the decision panel's with every menu row shown.
        static float table_row_height();
        float agreement_height() const;
        float decision_height() const;
        const std::vector<std::string> &menu_texts(const Match &m, size_t index, int view);
        std::string row_text(const Match &m, size_t index, int view, int row);
        std::string label_of(int64_t policy) const;
        ImU32 color_of(int64_t policy) const;
        std::string model_of(const std::vector<ServedPolicy> &served, int64_t policy) const;
        bool go_to(ReplaySession *s, size_t index);

        State state_ = State::NO_MATCH;
        std::vector<int64_t> candidates_;
        std::set<int64_t> excluded_;
        int64_t ref_ = -1;
        std::string ref_name_;
        CompareMetric metric_ = CompareMetric::TV;
        bool argmax_only_ = false;
        int64_t rank_policy_ = -1;
        MatchComparison cmp_;
        std::vector<RankedDecision> ranked_;
        const DecisionCompare *current_ = nullptr;
        std::map<int64_t, uint64_t> versions_;  // per policy its version (MatchValues::versions)
        std::string key_;                       // what the comparison was built from
        std::string rank_key_;
        std::map<size_t, std::vector<std::string>> texts_;  // per decision its menu rows' descriptions
        std::string texts_key_;
        int force_hover_ = -1;
        int force_jump_ = -1;
        bool dragging_ = false;
        // The catalog popup.
        bool open_picker_ = false;
        bool force_picker_ = false;
        bool picker_shown_ = false;
        std::vector<PolicyRow> rows_;
        std::set<std::string> ticked_;  // by the export each row plays
        char filter_[96] = {0};
        CompatChecker checker_;
        double requested_at_ = -1;
};

}  // namespace rmwork

#endif /* RMWORK_POLICY_COMPARE_VIEW_H */
