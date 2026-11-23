#ifndef RMWORK_PLAYERS_VIEW_H
#define RMWORK_PLAYERS_VIEW_H

#include <array>
#include <string>
#include <vector>

#include "card_renderer.h"
#include "links.h"
#include "magnifier.h"
#include "tool_data.h"

// ── The Players tool and the zone windows ───────────────────────────────────────────────────────
//
// Players: a box per player, the viewer's opponent first (seat B for the omniscient viewer): the
// seat chip, the agent the episodes name (else the seat) and the deck, the life large, "deciding"
// on the deciding seat, poison, energy, other counters and designations, the player's effects and
// emblems, and the zone icons with their counts — hand, library, graveyard, exile — each opening a
// zone window. Hovering a box links the player (the Table's chip, menu rows naming the player).
//
// Zone windows: floating windows ("Player A · Graveyard"), several at once, each following the
// current decision and viewer: the zone as the viewer knows it (zone_model.h) drawn as cards by the
// card renderer, in order, with what is noted of each (an exiled card's source, a library card's
// position, a known card of the opponent's hand); the cards the viewer cannot name as one card back
// with their count. Hovering a card links it (the menu rows acting with it); the magnifier works as
// on the Table (Q held, or a right-click, which also focuses the window). Esc closes a right-click
// magnifier, else the focused window.
//
// Zone tools (Player A · Graveyard, …, layout.h): the same zone drawn the same way in a tool's window,
// docked wherever the layout puts it; each keeps its own magnifier.

namespace rmwork {

class PlayersView {
    public:
        void draw(ToolData &data, ReplaySession &s, LinkState *links);
        // The open zone windows, as floating windows (call after the tools).
        void draw_zone_windows(ToolData &data, LinkState *links, ImGuiKeyChord magnify);
        // A zone tool's body (Player A · Graveyard, …, layout.h): the zone as a zone window shows it, in
        // the tool's docked or floating window.
        void draw_zone_tool(ToolData &data, int seat, ZoneKind z, LinkState *links, ImGuiKeyChord magnify);
        void open_zone(int seat, ZoneKind z);
        // The renderer cards are drawn with (nullptr: the text renderer); it must outlive its use.
        void set_renderer(const CardRenderer *r) { renderer_ = r ? r : &text_renderer_; }
        void close_zone(int seat, ZoneKind z);
        bool zone_open(int seat, ZoneKind z) const;
        size_t open_zones() const { return windows_.size(); }
        // Screenshots: open the magnifier on the first card of an open zone window whose name holds
        // `name` ("" off).
        void force_magnify(const std::string &name) { force_mag_ = name; }
        // The card the last drawn zone windows (then zone tools) magnify (0 none).
        uint64_t magnified() const;
        // Where the zone window of (`seat`, `z`), else its zone tool, last drew its card `index` (screen
        // coordinates); false when neither drew it.
        bool card_rect(int seat, ZoneKind z, size_t index, ImVec2 *min, ImVec2 *max) const;
        // The window's title ("Player A · Graveyard").
        static std::string zone_window_title(int seat, ZoneKind z);

    private:
        // One zone's cards drawn somewhere: a zone window's or a zone tool's.
        struct ZonePane {
                int seat = 0;
                ZoneKind kind = ZoneKind::GRAVEYARD;
                MagnifierState mag;
                std::vector<std::pair<ImVec2, ImVec2>> rects;  // the cards last drawn
        };
        struct ZoneWindow {
                ZonePane pane;
                bool placed = false;
        };
        void draw_box(const ToolData &data, int seat, LinkState *links);
        bool draw_zone_window(
            ToolData &data, ZoneWindow &w, LinkState *links, ImGuiKeyChord magnify, size_t index, bool esc_closes);
        // The zone's caption, cards, magnifier and the magnify hint, in the current window.
        void draw_zone_pane(ToolData &data, ZonePane &p, LinkState *links, ImGuiKeyChord magnify);
        CardLook look_of(const ToolData &data, const ZoneCard &c) const;
        static std::string zone_caption(const ToolData &data, const ZoneList &z);
        static std::string card_label(const ZoneList &z, const ZoneCard &c);
        static std::vector<std::string> card_lines(const ZoneList &z, const ZoneCard &c);

        std::vector<ZoneWindow> windows_;
        std::array<ZonePane, 2 * ZONE_KINDS> tool_panes_;  // the zone tools', by seat then zone
        TextCardRenderer text_renderer_;
        const CardRenderer *renderer_ = &text_renderer_;
        std::string force_mag_;
};

}  // namespace rmwork

#endif /* RMWORK_PLAYERS_VIEW_H */
