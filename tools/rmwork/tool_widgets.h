#ifndef RMWORK_TOOL_WIDGETS_H
#define RMWORK_TOOL_WIDGETS_H

#include <cstdint>
#include <string>
#include <vector>

#include "imgui.h"
#include "zone_model.h"

// ── Small widgets of the tools beside the Table ──────────────────────────────────────────────────
//
// ImHex-style grids (a header row, alternating row shades, column borders), the seat chips, the
// zone icons (a fanned hand, a library's stack, a graveyard's tombstone, exile's swirl), the π bars,
// value text coloured by its sign, card-colour swatches and a sparkline. Every colour is a theme
// token, except a swatch's card colours and the library icon's card back, which are card content.

namespace rmwork {

// Begins an ImHex-style table of `columns` columns filling `size` (0: the remaining width, rows'
// height); `scroll` gives it its own vertical scrollbar (the caller freezes the header row).
bool begin_grid(const char *id, int columns, ImVec2 size = ImVec2(0, 0), bool scroll = false);
// The seat's letter on its colour, as an item on the current line.
void seat_chip(int seat);
float seat_chip_width();
// A zone icon (16 px at the 14 px font, scaled with it) with `count` after it, as a button, outlined
// when `active` (its zone window is open); true when clicked.
bool zone_button(const char *id, ZoneKind zone, int count, const std::string &tooltip, bool active);
void draw_zone_icon(ImDrawList *dl, ZoneKind zone, ImVec2 pos, float size);
// A bar `fraction` full (0-1), `width` wide, a third of the line tall, as an item.
void value_bar(float fraction, float width, ImU32 fill);
// A bar drawn at `pos`, `height` tall.
void draw_bar(ImDrawList *dl, ImVec2 pos, float width, float height, float fraction, ImU32 fill);
// `text` in the positive or negative colour by the sign of `v`.
void signed_value(double v, const std::string &text);
ImU32 signed_color(double v);
// Attribution's heat colour of `v` (`scale`: the largest |value|): signed_color's positive or negative
// colour, the accent for an unsigned target, its alpha growing with |v| / scale up to `max_alpha`.
ImU32 heat_color(double v, double scale, bool is_signed, int max_alpha = 200);
// A heat swatch as an item, `width` × the line height × 0.7 (an outline alone without a value).
void heat_swatch(bool has, double v, double scale, bool is_signed, float width);
// The heat legend at `pos`, `width` wide: the negative colour through clear to the positive (clear to
// the accent for an unsigned target), its ends labelled with the scale, `label` above it; its height.
float draw_heat_legend(ImDrawList *dl, ImVec2 pos, float width, double scale, bool is_signed, const std::string &label);
// A card-colour swatch as an item, `width` × the line height × 0.8.
void card_swatch(uint8_t colors, bool land, float width);
// The background of a row linked to what is hovered elsewhere.
ImU32 linked_row_color();
// A small line chart of `values` (−1 to +1) over `size` as an item: a zero line, the series in
// `color`, the last point marked.
void sparkline(const char *id, const std::vector<double> &values, ImVec2 size, ImU32 color);

}  // namespace rmwork

#endif /* RMWORK_TOOL_WIDGETS_H */
