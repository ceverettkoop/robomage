#ifndef RMWORK_DECISION_FRAME_H
#define RMWORK_DECISION_FRAME_H

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "card_book.h"
#include "match.h"
#include "table_model.h"
#include "zone_model.h"

// ── One decision as one viewer saw it: the frame and what the tools read of it ──────────────────
//
// The tools beside the Table read the cursor's decision in the current viewer's STATE frame only
// (its stream's schema and CARDS frame), so a seat view never shows what the seat could not see.
// What the viewer sees of the decision itself: the omniscient viewer and the deciding seat see its
// menu (the frame carries it) and the decider's recorded evaluations (the episode's μ, values, π
// and teacher label, which come from the decider's own frame); the other seat sees only that the
// opponent is deciding.
//
// FrameModels: what the Players, Choices and zone windows show, built once per frame: the Table's
// model (players, menu links, the stack's targets), the zones (zone_model.h) and the menu rows.

namespace rmwork {

struct DecisionFrame {
        const Match *match = nullptr;
        size_t index = 0;  // the decision (the cursor)
        int view = VIEW_OMNI;
        const DecisionInfo *info = nullptr;
        std::shared_ptr<const rmstate::DecodedFrame> frame;
        const agents::SeatStream *stream = nullptr;
        const CardBook *book = nullptr;
        const EpisodeStepInfo *episode = nullptr;  // the decision's joined episode step, nullptr when none

        // The viewer is the omniscient one or the deciding seat.
        bool sees_decision() const { return view == VIEW_OMNI || (info && view == info->decider); }
        // The episode step, when the viewer sees the decision.
        const EpisodeStepInfo *visible_episode() const { return sees_decision() ? episode : nullptr; }
};

// The decision `index` of `m` in `view`'s frame, `book` that viewer's CardBook; false with `err`
// when the frame cannot be read.
bool load_decision_frame(
    const Match &m, size_t index, int view, const CardBook &book, DecisionFrame *out, std::string *err);

// One menu row as the viewer's frame gives it.
struct MenuRow {
        std::string text;      // the description (a seat frame's "(hidden card)" kept)
        std::string category;  // the action category's name ("CAST_SPELL")
        MenuLink link;
};

struct MenuModel {
        bool shown = false;  // the viewer sees the decision's menu
        int decider = -1;
        std::vector<MenuRow> rows;
        // The decision log's reply when it is a row of the menu and the viewer sees it, else -1.
        int chosen = -1;
};

MenuModel build_menu_model(const DecisionFrame &f);

struct FrameModels {
        TableModel table;
        ZoneModel zones;
        MenuModel menu;
};

FrameModels build_frame_models(const DecisionFrame &f);

// "Player A" / "Player B".
const char *seat_name(int seat);
// The agent of seat `seat` as the match's episodes name it, shortened for a label ("nn:current.ts.pt");
// "" when no episode names it.
std::string agent_label(const Match &m, int seat);

}  // namespace rmwork

#endif /* RMWORK_DECISION_FRAME_H */
