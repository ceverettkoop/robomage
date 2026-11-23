#ifndef RMWORK_REEVALUATE_DIALOG_H
#define RMWORK_REEVALUATE_DIALOG_H

#include <array>
#include <string>
#include <vector>

#include "reevaluation.h"

// ── Analysis → Re-evaluate…: the served policies and seats to re-evaluate the open replay with ──
//
// A modal opened from the menu or its key: a tick per served policy (Settings → Models' numeric ids,
// each with its model; every one ticked when the dialog first opens, the choices kept for the
// session), the seats (both by default), whether to re-evaluate a replay when it opens, then
// Re-evaluate (Enter) or Cancel. With no served policy it says so and offers Settings → Models.

namespace rmwork {

class ReevaluateDialog {
    public:
        // Opens for `policies` (the served ones now) and the setting's state.
        void open(const std::vector<ServedPolicy> &policies, bool on_open);
        enum class Choice {
            NONE,
            REEVALUATE,
            SETTINGS
        };
        // The button pressed this frame; REEVALUATE with the request's policies and seats in `out` and
        // the setting in `*on_open`.
        Choice draw(ReevalRequest *out, bool *on_open);
        bool is_open() const { return open_ || want_open_; }
        // The policies ticked (by served id) and the seats, as the next Re-evaluate sends them.
        std::vector<ServedPolicy> chosen() const;
        const std::array<bool, 2> &seats() const { return seats_; }

    private:
        bool want_open_ = false;
        bool open_ = false;
        bool on_open_ = true;
        std::vector<ServedPolicy> policies_;
        std::vector<bool> ticked_;
        std::vector<uint32_t> unticked_;  // the served ids the user unticked this session
        std::array<bool, 2> seats_{{true, true}};
};

}  // namespace rmwork

#endif /* RMWORK_REEVALUATE_DIALOG_H */
