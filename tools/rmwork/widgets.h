#ifndef RMWORK_WIDGETS_H
#define RMWORK_WIDGETS_H

#include <string>

#include "imgui.h"

// ── Small widgets of the shell: drawn icons, icon buttons, segmented toggles, swatches ──────────

namespace rmwork {

struct Palette;

// The toolbar's icons, drawn with the draw list from the mock-up's SVG paths.
enum class Icon {
    LOGO,
    OPEN,
    SAVE,
    MODE_PLAY,
    MODE_REPLAY,
    FIRST,
    PREV,
    PLAY,
    PAUSE,
    NEXT,
    LAST,
    SWING
};

// The icon's natural size in pixels.
ImVec2 icon_size(Icon icon);
// Draws `icon` with its top-left corner at `pos`; `alpha` fades it (disabled).
void draw_icon(ImDrawList *dl, Icon icon, ImVec2 pos, float alpha = 1.0f);
// A toolbar button showing `icon`, 24 wide and the frame height tall (so it lines up with the
// toolbar's other items at any font size), with `tooltip`; true when clicked.
bool icon_button(const char *id, Icon icon, const char *tooltip, bool enabled = true);
// A vertical rule between toolbar groups.
void toolbar_separator();

// One segment of a segmented toggle.
struct Segment {
        const char *label;
        bool has_icon;
        Icon icon;
        const char *tooltip;
};

// A row of joined buttons, the `*selected` one pressed; true when the selection changed.
bool segmented(const char *id, const Segment *segments, int count, int *selected);

// A 56×36 swatch drawing the table texture `key` ("wood", …), outlined when `selected`; true when
// clicked.
bool texture_swatch(const char *id, const std::string &key, bool selected);
// The same for a flat colour.
bool color_swatch(const char *id, ImU32 color, bool selected);
// Draws the table texture `key` over the rectangle (a flat fallback for an unknown key).
void draw_table_texture(ImDrawList *dl, ImVec2 min, ImVec2 max, const std::string &key);

// A row of square swatches of `p` (background, panel, selected tab, accent, text and the series
// colours), each `height` square, framed, as an item on the current line.
void palette_strip(const Palette &p, float height);
float palette_strip_width(float height);

// "#rrggbb" ↔ colour (false for malformed text).
bool parse_hex_color(const std::string &text, ImU32 *out);
std::string format_hex_color(ImU32 color);

// A text block centred in the remaining content region, wrapped to at most `max_width`.
void centered_text_block(const char *title, const char *body, float max_width);

}  // namespace rmwork

#endif /* RMWORK_WIDGETS_H */
