#ifndef RMWORK_MAGNIFIER_H
#define RMWORK_MAGNIFIER_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "card_renderer.h"
#include "imgui.h"

// ── The magnifier: a card shown large beside where it lies, its state listed under it ───────────
//
// Shared by the Table and the zone windows. MagnifierState follows the keys and the mouse: holding
// the Magnify key (Q) over a card shows it until the key is released or the mouse leaves the card;
// a right-click on a card shows it until the mouse leaves that card, a left click or Esc.
// draw_magnified_card draws it on the foreground draw list.

namespace rmwork {

class MagnifierState {
    public:
        // `under` is the card under the mouse (0 none), `over` that the mouse is over the tool,
        // `key` the Magnify chord.
        void update(uint64_t under, bool over, ImGuiKeyChord key);
        // Shows `handle` as a right-click does.
        void open(uint64_t handle);
        void close();
        uint64_t handle() const { return handle_; }

    private:
        uint64_t handle_ = 0;
        bool by_click_ = false;
};

// Draws `look` large beside the screen rectangle (min, max) — to its right, else its left, else
// centred — with `lines` under it (the first in the text colour, the rest dimmed); `overlay` draws
// over the large card (its rectangle given) when set. A double-faced card's other face
// (`look.other`) is drawn beside it, as large.
void draw_magnified_card(const CardRenderer &renderer, const CardLook &look, const std::vector<std::string> &lines,
    ImVec2 min, ImVec2 max, const std::function<void(ImDrawList *, ImVec2, ImVec2)> &overlay = {});

}  // namespace rmwork

#endif /* RMWORK_MAGNIFIER_H */
