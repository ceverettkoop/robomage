#ifndef RMWORK_CARD_PICKER_H
#define RMWORK_CARD_PICKER_H

#include <cstdint>
#include <string>
#include <vector>

#include "../../src/agents/decision_view.h"
#include "card_book.h"
#include "match.h"
#include "zone_model.h"

// ── The card picker's model: a decision among cards the Table does not show ─────────────────────
//
// docs/workbench_plan.md §4.2, item 8. A card-picker decision is one whose every row is a library
// search (SEARCH_LIBRARY), a card put on top of a library (TOP_LIBRARY: Ponder's and Doomsday's
// ordering, Brainstorm's put-back, scry's and surveil's keep), on the bottom at a resolution
// (BOTTOM_DECK_CARD: scry's other pile), a dig's pick (DIG_CHOICE), a card chosen for a zone change
// (CHOOSE_CARD: surveil's graveyard, a tutor's pick, Delve), a graveyard card exiled for a cost
// (EXILE_FROM_YARD), Sylvan Library's (SYLVAN_CHOICE), a card discarded (DISCARD), a permanent sacrificed
// or returned (SACRIFICE_PERMANENT, RETURN_PERMANENT) or the legend kept (KEEP_LEGEND), or a row of
// those categories naming no card (Fail to find, Take nothing). A row names its card as its `source`.
// Rows naming a card the Table shows (the viewer's hand, the battlefield) are answered on the Table; the
// picker gathers the rest (Thoughtseize's pick from the opponent's revealed hand):
//   - grouped by zone (the viewer's library first, then its graveyard, exile, the other seat's zones,
//     anything else), each card as the viewer's frame shows it (zone_model.h: only what the viewer may
//     see), identical cards of one zone one entry with their count (the engine offers one row for
//     interchangeable cards, src/queries/zones.h), a library's cards in order from the top. The other
//     cards a reveal window shows in a library with a pick (a search's whole library) and the other cards
//     the frame shows of the other seat's hand (Thoughtseize's lands) are entries with no row, shown
//     dimmed;
//   - an entry's rows are its placements when it has several (scry: "Top", "Bottom"; surveil: "Top",
//     "Graveyard"), else one pick;
//   - the pile being built in its final order: the picks this resolution made so far (the replies of
//     the deciding seat's decisions just before, for the same pending source, read from their frames:
//     the frame no longer lists a card put back), as the library's top will be. A row's `ordinal` is
//     the depth it fills (Ponder's first pick 2, "3rd from the top"; scry's 0, then 1); a TOP_LIBRARY row
//     with none puts the card on top of the cards put there before (Doomsday's five picks, Brainstorm's
//     put-back), so the first ends deepest. A pass ends where the menu's kind changes (Doomsday's picks,
//     then its ordering of the five). Cards put on the bottom (scry) and into the graveyard (surveil) are
//     listed apart, and the other categories' picks as chosen so far (a dig choosing two, delve's exiles);
//     a discard, sacrifice, return or legend pick builds none.

namespace rmwork {

// Whether every row of `v`'s menu is of a card-picker category (above), at least one naming a card.
bool card_pick_menu(const agents::DecisionView &v);

struct CardPickEntry {
        int seat = 0;  // the zone's owner: 0 A, 1 B; -1 for a card in no zone the model lists
        ZoneKind zone = ZoneKind::LIBRARY;
        bool other_zone = false;        // in none of the four zones (the row's card alone is known)
        std::vector<uint64_t> handles;  // the copies, the one a row names first
        int card = -1;
        bool back = false;
        std::string name;
        std::string type_line;  // the face's, "" when the book has none
        uint8_t colors = 0;
        bool land = false, token = false;
        std::vector<std::string> subtypes;
        int pos = -1;                      // a library card's position from the top (the first copy's), else -1
        std::vector<int> rows;             // the menu rows naming a copy, in menu order; empty: shown only
        std::vector<std::string> actions;  // per row: its placement ("Top", "3rd from the top", "Bottom",
                                           // "Graveyard"), "" for a plain pick
        int count() const { return static_cast<int>(handles.size()); }
        bool pickable() const { return !rows.empty(); }
};

struct CardPickGroup {
        int seat = 0;
        ZoneKind zone = ZoneKind::LIBRARY;
        bool other_zone = false;
        std::string title;  // "Your library", "Player B's graveyard", "Other cards"
        int zone_size = 0;  // the player row's count of the zone (0 for other cards)
        std::vector<CardPickEntry> entries;
};

struct PileCard {
        int card = -1;
        bool back = false;
        std::string name;
        std::string from;  // the zone it was picked from ("library", "graveyard", "hand")
};

// One place of the library's top: "Top", "2nd", …, filled by a pick made, or the place the next pick fills.
struct PileSlot {
        std::string label;
        bool next = false;
        bool filled = false;
        PileCard card;
};

struct PickerPile {
        bool shown = false;
        bool ordered = false;          // the picks fill given depths (else each goes on top of the last)
        std::vector<PileSlot> top;     // top first
        std::vector<PileCard> bottom;  // put on the bottom, in the order put
        std::vector<PileCard> graveyard;
        std::vector<PileCard> chosen;  // the other categories' picks so far
};

struct CardPicker {
        bool active = false;      // a card-picker decision with something to show (cards off the Table, a pile)
        uint64_t source = 0;      // the pending source (the spell or ability resolving, or being cast)
        std::string source_name;  // "Doomsday", "" when none
        std::string title;        // "Doomsday · your library and graveyard · 18 cards"
        std::vector<CardPickGroup> groups;
        std::vector<int> buttons;  // the rows naming no card (Fail to find, Take nothing), in menu order
        std::vector<std::string> row_text;
        PickerPile pile;
        int chosen_row = -1;  // the row the decision chose (Replay, an earlier decision), -1 none
        // Whether any entry may be picked (a card off the Table some row names).
        bool off_table() const;
        size_t entry_count() const;
        // The entry some row `row` names, as (group, entry); false when none.
        bool entry_of_row(int row, size_t *group, size_t *entry) const;
};

// The picker of decision `index` of `m` in viewer `view`'s frame (`v`, read with the viewer's book and
// zone model): inactive unless card_pick_menu holds and the menu names a card off the Table or builds a
// pile. `chosen_row` is the reply made there (-1 none: the human's pending decision).
CardPicker read_card_picker(const Match &m, size_t index, int view, const agents::DecisionView &v, const CardBook &book,
    const ZoneModel &zones, int chosen_row);

// Whether `e` matches `filter` (a case-insensitive substring of its name or type line; "" matches all).
bool picker_entry_matches(const CardPickEntry &e, const std::string &filter);

// "Top", "2nd", "3rd", … for depth 0, 1, 2, ….
std::string depth_label(int depth);

}  // namespace rmwork

#endif /* RMWORK_CARD_PICKER_H */
