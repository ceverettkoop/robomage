#ifndef RMWORK_ATTRIBUTION_VIEW_H
#define RMWORK_ATTRIBUTION_VIEW_H

#include <set>
#include <string>
#include <vector>

#include "attribution.h"
#include "layout.h"
#include "links.h"
#include "reevaluation.h"
#include "tool_data.h"

// ── The Attribution tool (docs/workbench_plan.md §2.8) ──────────────────────────────────────────
//
// The header: the decision and the seat whose frame is attributed, the policy (a served one), the
// target (V; π: TV; log π(focus); focus vs rival), the baseline, the method (occlusion, Shapley), the
// level (state or tensors), the Table layer's toggle, Explain (and in Play Auto) and the progress with
// Cancel; the focus and rival rows. The body: the ranked list of the attribution's rows (attribution.h)
// with a signed bar of the target and the metrics as sortable columns (a click on a header sorts by it,
// by |value|); a row with children expands (asking for them when they were never asked). At the state
// level a row is labelled by what it is in the game ("Ragavan (B, battlefield)", "Player A · life"), at
// the tensors level by the spec ("object[12]", "num[17] object.power (signed_log1p)"). Hovering a row
// links its object or player (the Table, the State Tree, the zone windows and Choices follow); a click
// selects it, which scrolls the State Tree to its row.

namespace rmwork {

struct AttributionContext {
        Mode mode = Mode::REPLAY;
        Attribution *att = nullptr;
        ToolData *data = nullptr;
        LinkState *links = nullptr;
        std::vector<ServedPolicy> policies;  // Settings → Models' served policies
        bool server = false;                 // a server listens at the socket
        int seat = -1;                       // the seat whose frame the view attributes (-1 none)
        bool current = false;                // the attribution is of the cursor's decision and that seat
        bool waiting = false;                // Play: waiting for Model's Read before starting
        bool no_menu = false;                // the seat's frame has no menu here (another seat decides)
        const AttrTints *tints = nullptr;
        bool *layer = nullptr;       // the Table layer shown
        bool stats_missing = false;  // a mean / resample baseline without slot statistics
        std::string stats_note;      // what the slot statistics are doing
};

class AttributionView {
    public:
        void draw(const AttributionContext &c);
        // Explain pressed (taken once).
        bool take_explain();
        AttrTarget target() const { return target_; }
        void set_target(AttrTarget t) { target_ = t; }
        bool tensors_level() const { return tensors_; }
        void set_tensors_level(bool on) { tensors_ = on; }
        featurize::Baseline baseline() const { return baseline_; }
        void set_baseline(featurize::Baseline b) { baseline_ = b; }
        AttrMethod method() const { return method_; }
        void set_method(AttrMethod m) { method_ = m; }
        // The served policy chosen (-1: the shown policy's when served, else the first).
        int64_t policy() const { return policy_; }
        void set_policy(int64_t p) { policy_ = p; }
        // Play: attribute each of the human's decisions while the tool is open.
        bool auto_explain() const { return auto_; }
        void set_auto(bool on) { auto_ = on; }
        // The rows shown at the last draw (indices into the attribution's rows), in order.
        const std::vector<int> &shown() const { return shown_; }
        // The unit keys expanded.
        std::set<std::string> &open() { return open_; }
        // Screenshots and checks: hover the row of unit `key` every frame ("" off).
        void force_hover(const std::string &key) { force_hover_ = key; }
        // The label a row shows at the current level.
        std::string label(const Attribution &a, const AttrRow &r) const;

    private:
        void draw_header(const AttributionContext &c);
        void draw_rows(const AttributionContext &c);
        void draw_row(const AttributionContext &c, int index, double scale, int metric);
        std::vector<int> sorted(const Attribution &a, const std::vector<int> &rows, int metric) const;

        AttrTarget target_ = AttrTarget::V;
        bool tensors_ = false;
        featurize::Baseline baseline_ = featurize::Baseline::NONE;
        AttrMethod method_ = AttrMethod::OCCLUSION;
        int64_t policy_ = -1;
        bool auto_ = false;
        bool explain_ = false;
        int sort_metric_ = -1;  // a metric column (-1: the target's)
        std::set<std::string> open_;
        std::string expand_;  // the row whose children the last draw asked for
        std::string force_hover_;
        std::vector<int> shown_;
};

}  // namespace rmwork

#endif /* RMWORK_ATTRIBUTION_VIEW_H */
