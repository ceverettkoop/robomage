#ifndef RMWORK_LINKS_H
#define RMWORK_LINKS_H

#include <cstdint>
#include <map>
#include <vector>

#include "table_model.h"

// ── The tools' shared hover and selection: objects, players, stack objects and menu rows ───────
//
// A tool reports what the mouse is over while it draws (hover_object, hover_player,
// hover_menu_row) and a click selects it (select_*). new_frame(), called once before the tools draw,
// makes the last frame's hover current and works out what it links to, so every tool reads the
// same links whatever order they draw in:
//   - an object (a permanent, a hand card, a stack object): itself; the menu rows acting with it or
//     on it (a row's source or target); a stack object's targets (objects and players);
//   - a player: the menu rows whose source or target is that player;
//   - a menu row: its source and target objects and players.
// With nothing hovered, the selection links the same way. Handles are the viewer's own, so a change
// of decision or viewer (set_context) clears the hover and the selection.

namespace rmwork {

struct LinkItem {
        enum class Kind {
            NONE,
            OBJECT,
            PLAYER,
            MENU_ROW
        } kind = Kind::NONE;
        uint64_t handle = 0;  // OBJECT
        int seat = -1;        // PLAYER: 0 A, 1 B
        int row = -1;         // MENU_ROW
        bool operator==(const LinkItem &o) const {
            return kind == o.kind && handle == o.handle && seat == o.seat && row == o.row;
        }
        bool operator!=(const LinkItem &o) const { return !(*this == o); }
};

class LinkState {
    public:
        // The decision and viewer the tools show; a change clears the hover and the selection.
        void set_context(unsigned generation);
        // The current decision's menu links and stack targets (the Table sets them as it draws;
        // they apply from the next new_frame).
        void set_menu(const std::vector<MenuLink> &menu) { menu_ = menu; }
        void set_stack_targets(const std::map<uint64_t, std::vector<StackTarget>> &t) { stack_targets_ = t; }

        void new_frame();
        void hover_object(uint64_t handle);
        void hover_player(int seat);
        void hover_menu_row(int row);
        void select(const LinkItem &item);
        void clear_selection() { selected_ = LinkItem(); }

        // What is hovered (last frame) and selected.
        const LinkItem &hovered() const { return hovered_; }
        const LinkItem &selected() const { return selected_; }
        // Whether the hovered item (else the selected one) links to it.
        bool object_linked(uint64_t handle) const;
        bool player_linked(int seat) const;
        bool row_linked(int row) const;
        // The menu rows acting with object `handle` as their source.
        bool is_source(uint64_t handle) const;

    private:
        void link(const LinkItem &item);

        unsigned generation_ = 0;
        bool have_generation_ = false;
        LinkItem hovered_, next_hover_, selected_;
        std::vector<MenuLink> menu_;
        std::map<uint64_t, std::vector<StackTarget>> stack_targets_;
        std::vector<uint64_t> objects_;
        std::vector<int> players_, rows_;
};

}  // namespace rmwork

#endif /* RMWORK_LINKS_H */
