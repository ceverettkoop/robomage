#ifndef RMWORK_SIDEBOARD_VIEW_H
#define RMWORK_SIDEBOARD_VIEW_H

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "card_renderer.h"
#include "magnifier.h"
#include "play_session.h"
#include "sideboard_plan.h"
#include "tool_data.h"

// ── The Sideboard tool: a best of three's sideboarding as cards (plan §4.3, item 14) ────────────
//
// A docked tool, a tab beside the Table (Play opens and brings it to the front when the human's
// sideboarding phase begins and gives the Table back at the next game), reading the cursor's
// decision in the viewer's frame like the other tools. It shows one seat's 75 (the viewer's own;
// the omniscient viewer's the deciding seat's) as card images: the main deck grouped by type and the
// sideboard, each card once with its count, the phase's locks (a card brought in this phase is
// marked IN and is not cut again, a card cut is marked OUT and does not come back), the header's
// coming game, the opponent and its deck, the games so far and play or draw, and the deck, sideboard
// and swap counts. A seat view never shows the other seat's 75: during the opponent's sideboarding
// it says so and shows the viewer's own.
//
// Play, at the human's sideboarding decision: clicks build a plan (a sideboard card in, a deck card
// out; a click on a planned card takes it back; Undo, Reset) whose resulting counts the header shows;
// Confirm, once the plan is balanced, sends it: at each of the human's decisions of the phase the
// next reply is found in the live menu by category and source card (sideboard_plan.h's
// next_sideboard_step), checked against the frame that follows, and Done ends it; the next game's
// first frame must show the planned deck. A menu without the row the plan needs, a reply the session
// refuses, or a frame that does not show the step is shown and nothing more is sent.
//
// Plans: the league guide's plan for this matchup, play or draw (python/sideboard_guides.json, read
// from the repository), with Use This Plan; the model's π over this decision's rows when Model's
// Read evaluated it (the model decides one swap at a time, so π is for the decision in front of the
// human), also under each card; the swaps of the phase so far and, in Replay, what the phase swapped
// in all (the deck at the phase's first decision against the one the viewer's frame after the phase
// shows), and the deck against its registered list.

namespace rmwork {

// What the tool shows of the cursor's decision.
struct SideboardModel {
        bool ok = false;
        std::string error;
        int view = VIEW_OMNI;
        int seat = -1;              // the seat whose 75 is shown
        bool phase = false;         // the decision is of a sideboarding phase
        bool seat_decides = false;  // ... and the shown seat makes it
        std::string note;           // what the viewer cannot see ("Player B is sideboarding …")
        SideboardFacts facts;       // the shown seat's, in the viewer's frame
        SideboardState state;       // the phase's deck at its start and now
        bool have_end = false;      // the deck the phase left is known (the frame after it)
        DeckCounts end;
        Decklists lists;
        GuidePlan guide;
        std::vector<double> pi;  // per menu row, empty when not evaluated (or not visible)
        std::string pi_note;     // why there is no π
        std::string policy;      // the evaluation's policy label
        std::string opponent;    // the opponent's agent ("explore", "nn:final.ts.pt"), else its seat
        std::string results;     // "lost game 1"
        bool on_play = true;
        int deck_size = 60, side_size = 15;  // the registered list's
};

class SideboardView {
    public:
        // The model of the session's cursor decision (rebuilt when the decision or viewer changes).
        void update(ToolData &data, ReplaySession &s);
        // The tool; `choosing`: the human's pending decision is at the cursor (Play).
        void draw(ToolData &data, ReplaySession &s, bool choosing, ImGuiKeyChord magnify);
        // The renderer cards are drawn with (nullptr: the text renderer); it must outlive its use.
        void set_renderer(const CardRenderer *r) { renderer_ = r ? r : &text_renderer_; }
        // Play's session (nullptr in Replay): the opponent's name, Model's Read's state.
        void set_play(const PlaySession *p) { play_ = p; }

        // Play: sends the confirmed plan; once a frame after the session's poll.
        void step(PlaySession &p);

        // The actions the tool's clicks and buttons take (false with `why` when not allowed).
        bool click_side(int card, std::string *why);
        bool click_main(int card, std::string *why);
        bool undo() { return plan_.undo(); }
        void reset() { plan_.clear(); }
        bool use_guide(std::string *why);
        // Starts sending the plan (Play, at the human's sideboarding decision).
        bool confirm(std::string *why);
        // Starts sending toward main deck `target` as a confirmed plan would (the self-test's).
        bool send_target(const std::map<int, int> &target, std::string *why);
        void dismiss();
        // A plan spec, as clicks: "guide" (Use This Plan), or ";"-separated "+Card" (a sideboard card's
        // click) and "-Card" (a deck card's) items; card names as python/sideboard_guides.py's card_key
        // compares them. The screenshot flags'.
        bool apply_spec(const std::string &spec, std::string *why);
        // Whether the human's pending decision is at the cursor (draw sets it; update() outside a draw
        // reads it).
        void set_choosing(bool on) { choosing_ = on; }

        const SideboardModel &model() const { return model_; }
        const SideboardPlan &plan() const { return plan_; }
        bool sending() const { return send_.active; }
        int replies_sent() const { return send_.sent; }
        // The last send's failure ("" none) and its success message.
        const std::string &error() const { return error_; }
        const std::string &result() const { return result_; }
        // Whether the human may plan now (Play, at the shown seat's sideboarding decision, not sending).
        bool interactive() const { return choosing_ && model_.seat_decides && !send_.active; }
        const SideboardGuides &guides();

    private:
        struct Pile {
                int card = -1;
                bool main = true;       // in the deck (else the sideboard)
                int count = 0;          // copies once the plan is sent
                int delta = 0;          // planned copies into (+) or out of (-) this section
                bool lock_in = false;   // brought in this phase
                bool lock_out = false;  // cut this phase
                bool pending = false;   // the card brought in, waiting for its cut
                bool guide = false;     // the guide's plan moves it (not yet planned)
                double pi = -1;         // π of its menu row, -1 none
                bool top = false;       // the row π favours most
        };
        struct PileGroup {
                std::string label;
                std::vector<Pile> piles;
        };
        struct Sending {
                bool active = false;
                std::map<int, int> target;
                int seat = 0;
                int game = 0;
                long answered = -1;  // the decision answered last
                SideboardStep last;
                int last_card = -1;
                DeckCounts before;
                int sent = 0;
        };

        std::vector<PileGroup> build_groups(const DecisionFrame &f) const;
        float flow(const std::vector<PileGroup> &groups, float cw, float width, std::vector<ImVec2> *at,
            std::vector<ImVec2> *labels) const;
        void draw_header(const DecisionFrame &f);
        void draw_piles(
            const DecisionFrame &f, const std::vector<PileGroup> &groups, ImVec2 size, ImGuiKeyChord magnify);
        void draw_pile(const DecisionFrame &f, const Pile &p, ImVec2 a, ImVec2 b, bool hovered);
        void draw_plans(const DecisionFrame &f);
        void draw_footer();
        CardLook look_of(const DecisionFrame &f, int card) const;
        std::string name_of(const DecisionFrame &f, int card) const;
        std::string status_line(const DecisionFrame &f) const;
        void finish(const std::string &error, const std::string &result);
        bool start_sending(const std::map<int, int> &target, std::string *why);

        SideboardModel model_;
        SideboardPlan plan_;
        Sending send_;
        std::string error_, result_;
        std::string hint_;  // why the last click did nothing
        bool choosing_ = false;
        unsigned generation_ = ~0u;
        const Match *match_ = nullptr;
        size_t index_ = 0;
        const PlaySession *play_ = nullptr;
        SideboardGuides guides_;
        bool guides_read_ = false;
        std::string guides_error_;
        MagnifierState mag_;
        TextCardRenderer text_renderer_;
        const CardRenderer *renderer_ = &text_renderer_;
};

}  // namespace rmwork

#endif /* RMWORK_SIDEBOARD_VIEW_H */
