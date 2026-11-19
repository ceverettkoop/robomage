#ifndef RMWORK_SELFTEST_LIVE_H
#define RMWORK_SELFTEST_LIVE_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "../../src/agents/decision_view.h"
#include "../../src/state_format/decode.h"
#include "app.h"
#include "selftest_table.h"

// ── The self-tests' helpers for live Play games driven through the UI ───────────────────────────
//
// Shared by selftest_decisions.cpp and selftest_choices.cpp: a sculpted game as seat A against explore
// (PlayConfig::setup's engine flags), the human's pending decision and its frame, rows found by
// category and text, simulated clicks on the Table's cards, its prompt line's buttons and the card
// picker's items, each checked against the reply recorded, and the game's log opened in Replay.

namespace rmwork {
namespace live {

using Frames = std::function<void(int)>;
using RowTest = std::function<bool(const std::string &category, const std::string &text)>;

// How long a decision may take to arrive.
inline constexpr double DECISION_TIMEOUT_S = 120;
// Decisions drive_to answers with row 0 before giving up.
inline constexpr int DRIVE_LIMIT = 40;

// The human's pending decision: its frame and the view over it.
struct Pending {
        std::shared_ptr<const rmstate::DecodedFrame> frame;
        std::unique_ptr<agents::DecisionView> view;
        size_t index = 0;
        bool ok() const { return view != nullptr; }
};

// Frames until the human decides or the game is over; whether the human decides.
bool wait_decision(const Frames &frames, PlaySession &s);
// The human's pending decision (not ok() when none).
Pending pending(PlaySession &s);
// The first row `test` holds for, -1 none.
int find_row(const Pending &p, const RowTest &test);
// Answers the human's decisions with row 0 until one has a row `test` holds for; that row (the decision
// left pending), -1 when none came.
int drive_to(App &app, const Frames &frames, const RowTest &test);
// The one row naming object `handle` (or player `seat`) whose description contains `containing`; -1
// when none or several do.
int row_naming(const Pending &p, uint64_t handle, int seat, const std::string &containing);
// The first object on the Table named `name` (untapped only when `untapped`), 0 when none.
uint64_t object_named(const App &app, const std::string &name, bool untapped);
// The first battlefield object named `name` controlled by seat `controller`, 0 when none.
uint64_t object_of(const App &app, const std::string &name, int controller);
// The centre of `r` (the Table's panel coordinates) on screen.
ImVec2 screen_point(Mode m, const TRect &r);
// The rectangle object `handle`'s card is drawn in (its pile's front card).
bool card_rect(const App &app, uint64_t handle, TRect *out);
// A click at `p`; the reply then recorded at decision `index` (REPLY_PENDING when none was sent).
int click_reply(App &app, const Frames &frames, ImVec2 p, size_t index);
// Every pick of the Table's decision is an object the frame shows on the Table (the bottom seat's
// battlefield and hand, the stack) or a player.
void check_picks_shown(const CheckFn &check, const App &app, const std::string &at);
// Starts a sculpted game as seat A against explore (burn against bug, game seed `seed`) with the
// engine's setup flags `setup`; whether it started. Explore keeps its opening hand at seed 1 (at seed 5
// it mulligans, its hand then reshuffled).
bool start_sculpted(const CheckFn &check, App &app, bool auto_mana, const std::vector<std::string> &setup,
    const std::string &at, uint64_t seed = 5);
// A click on object `handle`'s card answers with row `want`.
bool click_object(
    const CheckFn &check, App &app, const Frames &frames, uint64_t handle, int want, const std::string &at);
// A click on the prompt line's button whose label contains `text` answers with its row.
bool click_button(const CheckFn &check, App &app, const Frames &frames, const std::string &text, const std::string &at);
// A click on the prompt line's button whose label contains `text`; false when there is none.
bool press_button(const CheckFn &check, App &app, const Frames &frames, const std::string &text, const std::string &at);
// Where object `handle`'s card is on screen (Play's Table).
bool card_point(const App &app, uint64_t handle, ImVec2 *out);
void click_card(App &app, const Frames &frames, uint64_t handle);
// Frames until the Table's plan is sent (or stopped), then until the human decides; whether it was.
bool wait_sent(const Frames &frames, App &app);
// The first of the human's decisions after `index` (the stops answer some unasked; their frames are
// recorded too) whose frame `test` holds for; not ok() when none.
Pending frame_after(const App &app, size_t index, const std::function<bool(const Pending &)> &test);
// The picker's card item named `name` (in zone `zone`, -1 any), as last drawn; nullptr when none.
const CardPickerView::Item *picker_card(App &app, const std::string &name, int zone);
// The picker's button labelled `label`: a placement button of the card named `entry`, or ("" entry) a
// button under the cards; nullptr when none.
const CardPickerView::Item *picker_button(App &app, const std::string &entry, const std::string &label);
// A click on the picker's item `it` answers with row `want`.
bool click_item(const CheckFn &check, App &app, const Frames &frames, const CardPickerView::Item *it, int want,
    const std::string &at);
// Card `card`'s name in the pending frame's stream, "?" for none.
std::string card_name(const Pending &p, int card);
// Stops Play's game and opens its decision log in Replay; whether it loaded.
bool open_log_in_replay(const CheckFn &check, App &app, const Frames &frames, const std::string &at);
// The first decision of seat `seat` whose frame `test` holds for, -1 none.
long first_decision(const Match &m, int seat, const std::function<bool(const agents::DecisionView &)> &test);

}  // namespace live
}  // namespace rmwork

#endif /* RMWORK_SELFTEST_LIVE_H */
