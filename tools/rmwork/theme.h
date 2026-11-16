#ifndef RMWORK_THEME_H
#define RMWORK_THEME_H

#include <cstdint>
#include <string>

#include "imgui.h"

// ── The workbench's look: colour themes, the ImGui and ImPlot styles ────────────────────────────
//
// Every colour the workbench draws comes from the current theme's Palette (tok()): the mock-up's
// colour tokens, a full set per theme. apply_theme sets the palette and the ImGui and ImPlot colours
// from it (between frames); the metrics (padding, rounding) are the same for every theme. The table
// backgrounds a user picks in Settings → Table are content, not theme, and keep their colours.

namespace rmwork {

// An opaque (or `alpha`) colour from 0xRRGGBB.
constexpr ImU32 hex_color(uint32_t rgb, uint32_t alpha = 255) {
    return IM_COL32((rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF, alpha);
}

// One theme's colour tokens.
struct Palette {
        ImU32 BG;            // behind the dockspace (the desktop)
        ImU32 BG_TEXT;       // text drawn straight on BG
        ImU32 PANEL;         // tool windows
        ImU32 PANEL_2;       // alternate table rows
        ImU32 HEADER;        // table headers; a hovered icon button
        ImU32 TITLE;         // a floating window's title bar
        ImU32 TITLE_ACTIVE;  // a dialog's title bar
        ImU32 TITLE_TEXT;    // text on TITLE_ACTIVE
        ImU32 TAB;           // tabs
        ImU32 TAB_ACTIVE;    // the selected tab
        ImU32 TAB_DIM;       // the selected tab of an unfocused group
        ImU32 LINE;          // borders, rules, separators
        ImU32 TEXT;          // text
        ImU32 DIM;           // secondary text
        ImU32 MENU;          // the menu bar and toolbar
        ImU32 DROP;          // popups, menus, tooltips
        ImU32 STATUS;        // the status bar
        ImU32 BTN;           // buttons
        ImU32 BTN_HOVER;     // a hovered button or row
        ImU32 FIELD;         // input fields, combo boxes, sliders
        ImU32 ACCENT;        // check marks, slider grabs, focus, drop targets
        ImU32 SEL;           // selected rows, a selected segment, text selection
        ImU32 PLOT_BG;       // plot areas, previews, detail text boxes
        ImU32 GRID;          // plot grid lines
        ImU32 SCROLL_THUMB;  // scrollbar thumbs
        ImU32 S_A;           // the series V (A); seat A's icons
        ImU32 S_A6;          // the series V (A, re-evaluated)
        ImU32 S_B;           // the series −V of the B seat
        ImU32 LIFE_A;        // life of A
        ImU32 LIFE_B;        // life of B
        ImU32 BAD;           // errors, the swing icon
        ImU32 GOOD;          // the play icons
        ImU32 FOLDER;        // the open icon
        ImU32 LOGO_BG;       // the logo's square
        ImU32 LOGO_FG;       // the logo's letter
        // Derived from the tokens above by apply_theme's palette build:
        ImU32 FIELD_HOVER;          // a hovered field
        ImU32 FIELD_ACTIVE;         // an active field
        ImU32 BTN_ACTIVE;           // a pressed button
        ImU32 SCROLL_THUMB_HOVER;   // a hovered scrollbar thumb
        ImU32 SCROLL_THUMB_ACTIVE;  // a dragged scrollbar thumb
        ImU32 ACCENT_ACTIVE;        // a dragged slider grab
        ImU32 MODAL_DIM;            // over the workbench behind a dialog (translucent)
};

struct ThemeInfo {
        const char *id;    // settings key ("imhex-dark")
        const char *name;  // display name ("ImHex Dark")
        bool light;
};

inline constexpr int THEME_COUNT = 8;
inline constexpr const char *DEFAULT_THEME = "imhex-dark";

const ThemeInfo &theme_info(int index);
// The theme with settings key `id`, or -1.
int find_theme(const std::string &id);
// A theme's palette (whether or not it is the current one).
const Palette &theme_palette(int index);
// The current theme's palette.
const Palette &tok();
// The current theme's index.
int current_theme();
// Makes theme `id` (the default for an unknown one) current: its palette and the ImGui and ImPlot
// colours. Call between frames.
void apply_theme(const std::string &id);

// The WCAG 2 contrast ratio of two opaque colours (1 to 21).
float contrast_ratio(ImU32 a, ImU32 b);
// The CIE76 colour difference ΔE*ab of two opaque colours (about 2.3: just noticeable).
float color_difference(ImU32 a, ImU32 b);

// Creates the ImGui and ImPlot contexts with docking on, the workbench's styles, the default theme
// and the embedded fonts (fonts.h), the default font applied; no ini file (the caller sets
// io.IniFilename).
void create_ui_context();
void destroy_ui_context();

}  // namespace rmwork

#endif /* RMWORK_THEME_H */
