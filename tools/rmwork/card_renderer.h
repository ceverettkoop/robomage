#ifndef RMWORK_CARD_RENDERER_H
#define RMWORK_CARD_RENDERER_H

#include <cstdint>
#include <string>
#include <vector>

#include "card_book.h"
#include "imgui.h"

// ── Drawing one card: its face or its back, upright or tapped ───────────────────────────────────
//
// A CardRenderer draws what a card looks like and nothing of its state on the table (the Table
// draws P/T, counters, damage and the rest over it), so the text renderer here and the image
// renderer of the card-image stage (docs/workbench_plan.md §2.4) are interchangeable. A card is
// drawn into `slot`: upright it fills the slot (width × CARD_ASPECT tall); `rotated` (tapped) it
// lies on its side, turned a quarter clockwise, in a slot as wide as the card is tall.
//
// TextCardRenderer prints the face from CARDS: a black-bordered (a token: silver-bordered) rounded
// card in its frame colour (one colour; two colours split down the middle; three or more gold;
// colourless grey; a land tan with a land glyph and pips for the basic land types it has), the name
// (wrapped on a small card), the mana cost as pips, the type line and, once the card is large
// enough, the oracle text. The colours are card content, not theme. The back is a brown card
// with an oval; `note` (what the viewer knows of a face-down card) is printed on it when it fits.

namespace rmwork {

struct CardLook {
        const CardFace *face = nullptr;  // nullptr: the card's back
        uint8_t colors = 0;              // the frame's colours (WUBRG bits; a permanent's post-layer ones)
        bool land = false;
        bool token = false;
        uint8_t land_mana = 0;  // a land's basic land types, as WUBRG bits
        std::string note;       // printed on a back
};

class CardRenderer {
    public:
        virtual ~CardRenderer() = default;
        // Draws `look` into `slot` (screen coordinates), faded by `alpha`.
        virtual void draw(ImDrawList *dl, const CardLook &look, ImVec2 slot_min, ImVec2 slot_max, bool rotated,
            float alpha) const = 0;
};

class TextCardRenderer : public CardRenderer {
    public:
        void draw(ImDrawList *dl, const CardLook &look, ImVec2 slot_min, ImVec2 slot_max, bool rotated,
            float alpha) const override;
};

// A colour swatch of a card's frame colours over the rectangle: one colour, two split down the
// middle, gold for more, colourless grey, a land (with no colour) tan.
void draw_card_swatch(ImDrawList *dl, ImVec2 min, ImVec2 max, uint8_t colors, bool land);

// The WUBRG bits of the basic land types among `subtypes` (Plains W, Island U, Swamp B, Mountain R,
// Forest G).
uint8_t basic_land_mana(const std::vector<std::string> &subtypes);

// Writes `text` at `pos` with `font` at `size`, cut to `max_w` with an ellipsis; returns its width.
float draw_fitted_text(
    ImDrawList *dl, ImFont *font, float size, ImVec2 pos, ImU32 color, const std::string &text, float max_w);
// `text` wrapped to `max_w` in at most `max_lines` lines (the last cut with an ellipsis when more
// remains).
std::vector<std::string> wrap_text(ImFont *font, float size, const std::string &text, float max_w, int max_lines);

}  // namespace rmwork

#endif /* RMWORK_CARD_RENDERER_H */
