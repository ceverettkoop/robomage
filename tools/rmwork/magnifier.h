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

// Where the magnifier draws, in screen coordinates: the face up, a double-faced card's other face
// to its right (each with its caption above it), and the box of lines under them.
struct MagnifierRects {
        int faces = 1;
        ImVec2 face_min[2], face_max[2];
        ImVec2 caption_min[2];  // two faces: each one's caption line
        ImVec2 box_min, box_max;
};
// The magnifier's rectangles for the card at (min, max) on a screen of (vp_min, vp_max): the
// card large (up to 340 px wide, 42% of the screen's height), with `faces` (1 or 2) side by side
// 8 px apart and `caption_h` above them (two faces), and a box `box_h` tall under them across their
// width; to the card's right, else its left, else centred, and moved and shrunk to stay on the
// screen with 8 px to spare.
MagnifierRects magnifier_rects(
    ImVec2 min, ImVec2 max, ImVec2 vp_min, ImVec2 vp_max, int faces, float caption_h, float box_h);

// Draws `look` large beside the screen rectangle (min, max), as magnifier_rects places it, with
// `lines` under it (the first in the text colour, the rest dimmed); `overlay` draws over the large
// card (its rectangle given) when set. A double-faced card's other face (`look.other`, which the
// caller sets only for a card whose faces the viewer may see) is drawn to its right, as large; the
// face up is outlined and captioned "Face up".
void draw_magnified_card(const CardRenderer &renderer, const CardLook &look, const std::vector<std::string> &lines,
    ImVec2 min, ImVec2 max, const std::function<void(ImDrawList *, ImVec2, ImVec2)> &overlay = {});

}  // namespace rmwork

#endif /* RMWORK_MAGNIFIER_H */
