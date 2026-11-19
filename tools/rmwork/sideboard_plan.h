#ifndef RMWORK_SIDEBOARD_PLAN_H
#define RMWORK_SIDEBOARD_PLAN_H

#include <array>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../../src/agents/seat_frames.h"

// ── Sideboarding: a seat's 75, the plan the Sideboard tool builds, the league guides ─────────────
//
// What the engine asks (src/game_driver.cpp's run_sideboard_phase): between the games of a best of
// three each seat gets sideboarding decisions, one swap at a time and IN first. At a balanced deck
// the menu is "Done sideboarding" (SIDEBOARD_DONE, row 0) and a "Sideboard in: Nx Card" row
// (SIDEBOARD_IN) per sideboard card; the IN puts one copy in the main deck (the frame's match.sb_drift
// becomes 1, its pending.source_card that card) and the next menu is a "Sideboard out: Nx Card" row
// (SIDEBOARD_OUT) per main deck card, Done withheld; the OUT completes the swap (match.sb_swaps + 1).
// A card brought in this phase is never offered out again, nor a card cut offered back in (by name,
// for the phase). At 15 completed swaps the phase ends at the next balanced point with no Done
// decision; a menu of one row is taken by the engine without asking. Every STATE frame carries the
// viewer's current 75 (own_deck: a seat's frame its own seat's, the omniscient frame both) and the
// DECKLISTS frame the registered lists and deck names (both seats': open decklists).
//
// A plan is a multiset of cards to bring in and to cut, built from the phase's current deck; its
// result is the target main deck, and the replies are found one decision at a time from the live
// menu by category and source card (next_sideboard_step), so the pairing of ins with outs is the
// engine's order and a one-row decision the engine took itself is simply skipped.
//
// The guides: python/sideboard_guides.json (docs/sideboard_guides.md), per league deck and league
// opponent the cards in and out, and on the draw the deck's draw land out and the row's draw card
// in. A deck is a league deck when its registered name is "league/<stem>"; card names are compared
// as python/sideboard_guides.py's card_key does (accents folded, lower case, letters and digits:
// "Dead/Gone" is "Dead // Gone"), against the seat's own 75.

namespace rmwork {

// The engine's cap on a phase's completed swaps (src/game_driver.cpp, SIDEBOARD_SWAP_CAP).
inline constexpr int SIDEBOARD_SWAP_CAP = 15;

// One seat's 75: per card id its main deck and sideboard copies (no zero entries).
struct DeckCounts {
        std::map<int, int> main, side;
        int main_of(int card) const;
        int side_of(int card) const;
        int main_total() const;
        int side_total() const;
};

// A menu row as the plan reads it: its category's name and its source card (-1 none).
struct SideboardRow {
        std::string category;
        int card = -1;
};

// What one STATE frame shows of a seat's deck and its sideboarding.
struct SideboardFacts {
        bool sideboarding = false;  // header.sideboarding
        int game = 0;               // header.game (0-based): during sideboarding the coming game
        int plays_first = -1;       // match.plays_first: 0 A, 1 B (the coming game's while sideboarding)
        int swaps = 0;              // match.sb_swaps: the phase's completed swaps
        int drift = 0;              // match.sb_drift: 1 while a card brought in waits for its cut
        int pending_card = -1;      // that card (pending.source_card), -1 none
        bool has_deck = false;      // the frame has own_deck rows of the seat
        DeckCounts deck;            // the seat's current 75
        std::vector<SideboardRow> menu;
};

// The facts of frame `f` (of stream `s`) about seat `seat` (0 A, 1 B).
SideboardFacts read_sideboard_facts(const agents::SeatStream &s, const rmstate::DecodedFrame &f, int seat);

// The registered decklists of a stream's DECKLISTS frame, per seat.
struct Decklists {
        std::array<bool, 2> known{{false, false}};
        std::array<std::string, 2> name;  // "league/burn"
        std::array<DeckCounts, 2> lists;
};
Decklists read_decklists(const agents::SeatStream &s);

// The phase as the plan needs it: its deck at the phase's first decision and now.
struct SideboardState {
        DeckCounts start, now;
        int swaps = 0, drift = 0;
        int pending_card = -1;
        // Brought in this phase (more copies in the main deck than at its start): never cut again.
        bool in_locked(int card) const { return now.main_of(card) > start.main_of(card); }
        // Cut this phase: never brought back.
        bool out_locked(int card) const { return now.main_of(card) < start.main_of(card); }
};

class SideboardPlan {
    public:
        // A click on a sideboard card: cancels a planned cut of it, else plans one copy in; false
        // with `why` when neither is allowed.
        bool click_side(const SideboardState &s, int card, std::string *why);
        // A click on a main deck card: cancels a planned copy in, else plans one copy out.
        bool click_main(const SideboardState &s, int card, std::string *why);
        // Takes back the last click; false when there is none.
        bool undo();
        void clear();
        // The plan becomes the moves from the current deck to `target` (a main deck: card -> copies);
        // false with `why` (and the plan unchanged) when a move goes against the phase's locks, a
        // card has too few copies, or it takes more swaps than are left.
        bool set_target(const SideboardState &s, const std::map<int, int> &target, std::string *why);

        bool empty() const { return in_.empty() && out_.empty(); }
        int ins() const;
        int outs() const;
        int planned_in(int card) const;
        int planned_out(int card) const;
        const std::map<int, int> &in() const { return in_; }
        const std::map<int, int> &out() const { return out_; }
        // The 75 the plan leaves.
        DeckCounts result(const SideboardState &s) const;
        // The phase's swaps once the plan is sent (the outstanding one completed).
        int swaps_after(const SideboardState &s) const;
        // Every card in is matched by a cut (and the outstanding one): the deck ends at its size.
        bool balanced(const SideboardState &s) const;

    private:
        bool may_bring_in(const SideboardState &s, int card, std::string *why) const;
        bool may_cut(const SideboardState &s, int card, std::string *why) const;
        std::map<int, int> in_, out_;
        std::vector<std::pair<bool, int>> history_;  // (in, card) per click
};

// The reply toward the main deck `target` at a sideboarding decision whose frame shows `f`.
struct SideboardStep {
        int row = -1;       // the menu row to send, -1 none
        bool done = false;  // it is Done
        std::string what;   // "in: Smash to Smithereens", "out: Fireblast", "Done"
        std::string error;  // why no row can be sent (the menu does not offer what the plan needs)
};
// With a card waiting for its cut: the first SIDEBOARD_OUT row of a card the deck has more of than
// `target`; balanced and at the target: the Done row; balanced otherwise: the first SIDEBOARD_IN row
// of a card the deck has fewer of. `names` names cards in the error.
SideboardStep next_sideboard_step(
    const SideboardFacts &f, const std::map<int, int> &target, const std::function<std::string(int)> &names);
// The deck's main deck is `target` exactly.
bool main_matches(const DeckCounts &d, const std::map<int, int> &target);

// python/sideboard_guides.py's card_key.
std::string card_key(const std::string &name);

// The guides file.
struct GuideRow {
        std::vector<std::pair<std::string, int>> in, out;  // as the file lists them
        std::string draw;                                  // the draw card ("" none)
};
class SideboardGuides {
    public:
        // Reads `path`; false with `err` when it is missing or malformed.
        bool load(const std::string &path, std::string *err);
        bool loaded() const { return loaded_; }
        const std::string &path() const { return path_; }
        // The row of deck `own` against `opp` (stems: "burn"), nullptr when none.
        const GuideRow *row(const std::string &own, const std::string &opp) const;
        // The land `own` cuts on the draw ("" none).
        std::string draw_land(const std::string &own) const;
        bool holdout(const std::string &own, const std::string &opp) const;
        bool guided(const std::string &own) const { return draw_lands_.count(own) != 0; }

    private:
        bool loaded_ = false;
        std::string path_;
        std::map<std::string, std::string> draw_lands_;
        std::map<std::pair<std::string, std::string>, GuideRow> rows_;
        std::vector<std::pair<std::string, std::string>> holdout_;
};

// The league stem of a registered deck name ("league/burn" -> "burn"; "" for another deck).
std::string league_stem(const std::string &deck_name);

// The guide's plan for seat `seat` in the coming game.
struct GuidePlan {
        bool found = false;
        std::string why;       // no plan: why ("no guide row for …")
        std::string own, opp;  // the decks' stems
        bool on_play = true;
        bool holdout = false;  // a held-out matchup (the scripted tier keeps its deck there)
        // The lines for this game (the draw swap included on the draw): card name, copies.
        std::vector<std::pair<std::string, int>> in, out;
        std::string other;               // what the other of play / draw changes ("On the draw also: …")
        std::map<int, int> target;       // the main deck the guide wants, by card id
        std::map<std::string, int> ids;  // each line's card id (absent when not in the 75)
        std::string unresolved;          // a guide card the seat's 75 does not have
};
// `names` gives a card id's face names (front, back: CardNames).
GuidePlan guide_plan(
    const SideboardGuides &g, const Decklists &d, int seat, bool on_play, const agents::CardNames &names);

}  // namespace rmwork

#endif /* RMWORK_SIDEBOARD_PLAN_H */
