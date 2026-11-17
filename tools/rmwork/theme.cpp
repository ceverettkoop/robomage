#include "theme.h"

#include <algorithm>
#include <cmath>

#include "fonts.h"
#include "implot.h"

namespace {

struct ThemeEntry {
        rmwork::ThemeInfo info;
        rmwork::Palette (*build)();
};

// Every theme's palette (built once), the current theme and, per ImPlot context, each theme's
// colormap.
struct ThemeState {
        rmwork::Palette palettes[rmwork::THEME_COUNT];
        int current = 0;
        int colormaps[rmwork::THEME_COUNT] = {};
};

}  // namespace

static rmwork::Palette imhex_dark_palette();
static rmwork::Palette imhex_light_palette();
static rmwork::Palette imgui_dark_palette();
static rmwork::Palette imgui_classic_palette();
static rmwork::Palette solarized_dark_palette();
static rmwork::Palette solarized_light_palette();
static rmwork::Palette nord_palette();
static rmwork::Palette windows_95_palette();
static ThemeState &state();
static ImU32 mix(ImU32 a, ImU32 b, float t);
static void derive(rmwork::Palette *p, bool light);
static ImVec4 v4(ImU32 c);
static ImVec4 v4a(ImU32 c, float alpha);
static void apply_imgui_metrics();
static void apply_imgui_colors(const rmwork::Palette &t);
static void apply_plot_metrics();
static void apply_plot_colors(const rmwork::Palette &t, int colormap);
static void add_colormaps();
static float linear(float channel);
static void to_lab(ImU32 c, float lab[3]);

namespace {

const ThemeEntry THEMES[rmwork::THEME_COUNT] = {
    {{"imhex-dark", "ImHex Dark", false}, imhex_dark_palette},
    {{"imhex-light", "ImHex Light", true}, imhex_light_palette},
    {{"imgui-dark", "ImGui Dark", false}, imgui_dark_palette},
    {{"imgui-classic", "ImGui Classic", false}, imgui_classic_palette},
    {{"solarized-dark", "Solarized Dark", false}, solarized_dark_palette},
    {{"solarized-light", "Solarized Light", true}, solarized_light_palette},
    {{"nord", "Nord", false}, nord_palette},
    {{"windows-95", "Windows 95", true}, windows_95_palette},
};

}  // namespace

// The mock-up's tokens: ImHex's dark look.
static rmwork::Palette imhex_dark_palette() {
    using rmwork::hex_color;
    rmwork::Palette p{};
    p.BG = hex_color(0x121418);
    p.BG_TEXT = hex_color(0x8a929d);
    p.PANEL = hex_color(0x1d2025);
    p.PANEL_2 = hex_color(0x22262c);
    p.HEADER = hex_color(0x2a2f37);
    p.TITLE = hex_color(0x181b20);
    p.TITLE_ACTIVE = hex_color(0x2a5a8c);
    p.TITLE_TEXT = hex_color(0xdde2e8);
    p.TAB = hex_color(0x1d2836);
    p.TAB_ACTIVE = hex_color(0x2a5a8c);
    p.TAB_DIM = hex_color(0x24405f);
    p.LINE = hex_color(0x343a43);
    p.TEXT = hex_color(0xdde2e8);
    p.DIM = hex_color(0x8a929d);
    p.MENU = hex_color(0x17191d);
    p.DROP = hex_color(0x1f2227);
    p.STATUS = hex_color(0x101215);
    p.BTN = hex_color(0x28466a);
    p.BTN_HOVER = hex_color(0x31598a);
    p.FIELD = hex_color(0x2a2f37);
    p.ACCENT = hex_color(0x4a8fe0);
    p.SEL = hex_color(0x26507d);
    p.PLOT_BG = hex_color(0x16181c);
    p.GRID = hex_color(0x2c3138);
    p.SCROLL_THUMB = hex_color(0x454b54);
    p.S_A = hex_color(0x5b9be6);
    p.S_A6 = hex_color(0xd374d3);
    p.S_B = hex_color(0x43c2aa);
    p.LIFE_A = hex_color(0x5b9be6);
    p.LIFE_B = hex_color(0xe06c6c);
    p.BAD = hex_color(0xe05252);
    p.GOOD = hex_color(0x6ac46a);
    p.FOLDER = hex_color(0xe0b44a);
    p.LOGO_BG = hex_color(0x2a5a8c);
    p.LOGO_FG = hex_color(0xffffff);
    p.GLOW = hex_color(0xffd84a);
    p.LINK = hex_color(0x00e0ff);
    p.COUNTER = hex_color(0x7a3cc0);
    p.SEAT_A = hex_color(0x24456b);
    p.SEAT_B = hex_color(0x6b2b2b);
    return p;
}

// ImHex's light look: pale panels, blue selection.
static rmwork::Palette imhex_light_palette() {
    using rmwork::hex_color;
    rmwork::Palette p{};
    p.BG = hex_color(0xc9ced6);
    p.BG_TEXT = hex_color(0x2f3640);
    p.PANEL = hex_color(0xf6f7f9);
    p.PANEL_2 = hex_color(0xedf0f3);
    p.HEADER = hex_color(0xdde2e8);
    p.TITLE = hex_color(0xe2e6eb);
    p.TITLE_ACTIVE = hex_color(0x2f6fc0);
    p.TITLE_TEXT = hex_color(0xffffff);
    p.TAB = hex_color(0xdae1ea);
    p.TAB_ACTIVE = hex_color(0xb7d1f0);
    p.TAB_DIM = hex_color(0xcbd8e6);
    p.LINE = hex_color(0xc3c9d1);
    p.TEXT = hex_color(0x1b2027);
    p.DIM = hex_color(0x5c6570);
    p.MENU = hex_color(0xe8ebef);
    p.DROP = hex_color(0xfbfcfd);
    p.STATUS = hex_color(0xdfe3e8);
    p.BTN = hex_color(0xc9dcf2);
    p.BTN_HOVER = hex_color(0xaecbee);
    p.FIELD = hex_color(0xe4e8ed);
    p.ACCENT = hex_color(0x2a6cc2);
    p.SEL = hex_color(0xbcd5f2);
    p.PLOT_BG = hex_color(0xffffff);
    p.GRID = hex_color(0xe1e5ea);
    p.SCROLL_THUMB = hex_color(0xb3bac3);
    p.S_A = hex_color(0x1f66c4);
    p.S_A6 = hex_color(0xb02fb0);
    p.S_B = hex_color(0x0b8571);
    p.LIFE_A = hex_color(0x1f66c4);
    p.LIFE_B = hex_color(0xc62f2f);
    p.BAD = hex_color(0xc62f2f);
    p.GOOD = hex_color(0x2a8a2a);
    p.FOLDER = hex_color(0xb07d00);
    p.LOGO_BG = hex_color(0x2a6cc2);
    p.LOGO_FG = hex_color(0xffffff);
    p.GLOW = hex_color(0xf5b800);
    p.LINK = hex_color(0x0096d6);
    p.COUNTER = hex_color(0xdccbf5);
    p.SEAT_A = hex_color(0xbcd5f2);
    p.SEAT_B = hex_color(0xf5c9c9);
    return p;
}

// Dear ImGui's StyleColorsDark, its translucent colours flattened over the window background.
static rmwork::Palette imgui_dark_palette() {
    using rmwork::hex_color;
    rmwork::Palette p{};
    p.BG = hex_color(0x0a0a0a);
    p.BG_TEXT = hex_color(0x8c8c8c);
    p.PANEL = hex_color(0x0f0f0f);
    p.PANEL_2 = hex_color(0x181818);
    p.HEADER = hex_color(0x303033);
    p.TITLE = hex_color(0x0a0a0a);
    p.TITLE_ACTIVE = hex_color(0x294a7a);
    p.TITLE_TEXT = hex_color(0xffffff);
    p.TAB = hex_color(0x1b3150);
    p.TAB_ACTIVE = hex_color(0x2a5a97);
    p.TAB_DIM = hex_color(0x1d3655);
    p.LINE = hex_color(0x3a3a40);
    p.TEXT = hex_color(0xffffff);
    p.DIM = hex_color(0x808080);
    p.MENU = hex_color(0x242424);
    p.DROP = hex_color(0x141414);
    p.STATUS = hex_color(0x141414);
    p.BTN = hex_color(0x24456d);
    p.BTN_HOVER = hex_color(0x3270bd);
    p.FIELD = hex_color(0x1d2f49);
    p.ACCENT = hex_color(0x4296fa);
    p.SEL = hex_color(0x1d3a5f);
    p.PLOT_BG = hex_color(0x080808);
    p.GRID = hex_color(0x2e2e2e);
    p.SCROLL_THUMB = hex_color(0x4f4f4f);
    p.S_A = hex_color(0x5b8fd9);
    p.S_A6 = hex_color(0xdd8452);
    p.S_B = hex_color(0x55b46c);
    p.LIFE_A = hex_color(0x5b8fd9);
    p.LIFE_B = hex_color(0xd9575b);
    p.BAD = hex_color(0xd9575b);
    p.GOOD = hex_color(0x55b46c);
    p.FOLDER = hex_color(0xe6b54a);
    p.LOGO_BG = hex_color(0x294a7a);
    p.LOGO_FG = hex_color(0xffffff);
    p.GLOW = hex_color(0xffd84a);
    p.LINK = hex_color(0x00e0ff);
    p.COUNTER = hex_color(0x6b38b0);
    p.SEAT_A = hex_color(0x284a80);
    p.SEAT_B = hex_color(0x803030);
    return p;
}

// Dear ImGui's StyleColorsClassic, flattened the same way.
static rmwork::Palette imgui_classic_palette() {
    using rmwork::hex_color;
    rmwork::Palette p{};
    p.BG = hex_color(0x000000);
    p.BG_TEXT = hex_color(0x999999);
    p.PANEL = hex_color(0x0d0d0d);
    p.PANEL_2 = hex_color(0x1a1a24);
    p.HEADER = hex_color(0x34345a);
    p.TITLE = hex_color(0x262650);
    p.TITLE_ACTIVE = hex_color(0x474789);
    p.TITLE_TEXT = hex_color(0xe6e6e6);
    p.TAB = hex_color(0x34346c);
    p.TAB_ACTIVE = hex_color(0x5252a6);
    p.TAB_DIM = hex_color(0x4a4a90);
    p.LINE = hex_color(0x5a5a5a);
    p.TEXT = hex_color(0xe6e6e6);
    p.DIM = hex_color(0x999999);
    p.MENU = hex_color(0x3d3d58);
    p.DROP = hex_color(0x1c1c24);
    p.STATUS = hex_color(0x262633);
    p.BTN = hex_color(0x373f60);
    p.BTN_HOVER = hex_color(0x4b5a88);
    p.FIELD = hex_color(0x2a2a2a);
    p.ACCENT = hex_color(0x8080e6);
    p.SEL = hex_color(0x2e2e67);
    p.PLOT_BG = hex_color(0x000000);
    p.GRID = hex_color(0x333333);
    p.SCROLL_THUMB = hex_color(0x45456a);
    p.S_A = hex_color(0x8a8aff);
    p.S_A6 = hex_color(0xff8ae0);
    p.S_B = hex_color(0x66d9b3);
    p.LIFE_A = hex_color(0x8a8aff);
    p.LIFE_B = hex_color(0xff7070);
    p.BAD = hex_color(0xff6060);
    p.GOOD = hex_color(0x70e070);
    p.FOLDER = hex_color(0xe6c34d);
    p.LOGO_BG = hex_color(0x474789);
    p.LOGO_FG = hex_color(0xffffff);
    p.GLOW = hex_color(0xffd84a);
    p.LINK = hex_color(0x00e0ff);
    p.COUNTER = hex_color(0x6b38b0);
    p.SEAT_A = hex_color(0x3a3a8c);
    p.SEAT_B = hex_color(0x8c3a3a);
    return p;
}

// Ethan Schoonover's Solarized, dark (base03 panels).
static rmwork::Palette solarized_dark_palette() {
    using rmwork::hex_color;
    rmwork::Palette p{};
    p.BG = hex_color(0x001c24);
    p.BG_TEXT = hex_color(0x839496);
    p.PANEL = hex_color(0x002b36);
    p.PANEL_2 = hex_color(0x04313d);
    p.HEADER = hex_color(0x073642);
    p.TITLE = hex_color(0x00252f);
    p.TITLE_ACTIVE = hex_color(0x0b4f63);
    p.TITLE_TEXT = hex_color(0xeee8d5);
    p.TAB = hex_color(0x073642);
    p.TAB_ACTIVE = hex_color(0x0e5a70);
    p.TAB_DIM = hex_color(0x0a4757);
    p.LINE = hex_color(0x2a4a52);
    p.TEXT = hex_color(0xd5d8cf);
    p.DIM = hex_color(0x839496);
    p.MENU = hex_color(0x00232c);
    p.DROP = hex_color(0x073642);
    p.STATUS = hex_color(0x00202a);
    p.BTN = hex_color(0x0b4a5c);
    p.BTN_HOVER = hex_color(0x11607a);
    p.FIELD = hex_color(0x073642);
    p.ACCENT = hex_color(0x268bd2);
    p.SEL = hex_color(0x0e5068);
    p.PLOT_BG = hex_color(0x00252f);
    p.GRID = hex_color(0x0d3b47);
    p.SCROLL_THUMB = hex_color(0x2f525b);
    p.S_A = hex_color(0x3a9ae0);
    p.S_A6 = hex_color(0xd33682);
    p.S_B = hex_color(0x2aa198);
    p.LIFE_A = hex_color(0x3a9ae0);
    p.LIFE_B = hex_color(0xe0503c);
    p.BAD = hex_color(0xdc322f);
    p.GOOD = hex_color(0x859900);
    p.FOLDER = hex_color(0xb58900);
    p.LOGO_BG = hex_color(0x268bd2);
    p.LOGO_FG = hex_color(0xfdf6e3);
    p.GLOW = hex_color(0xe6b800);
    p.LINK = hex_color(0x35d0c5);
    p.COUNTER = hex_color(0x4b4f9a);
    p.SEAT_A = hex_color(0x1a4a6e);
    p.SEAT_B = hex_color(0x6e2424);
    return p;
}

// Solarized, light (base3 panels).
static rmwork::Palette solarized_light_palette() {
    using rmwork::hex_color;
    rmwork::Palette p{};
    p.BG = hex_color(0xe4ddc8);
    p.BG_TEXT = hex_color(0x4a5d64);
    p.PANEL = hex_color(0xfdf6e3);
    p.PANEL_2 = hex_color(0xf6efdc);
    p.HEADER = hex_color(0xeee8d5);
    p.TITLE = hex_color(0xeee8d5);
    p.TITLE_ACTIVE = hex_color(0x1a6aa6);
    p.TITLE_TEXT = hex_color(0xfdf6e3);
    p.TAB = hex_color(0xe9e2cd);
    p.TAB_ACTIVE = hex_color(0xc9e0ef);
    p.TAB_DIM = hex_color(0xdcdccf);
    p.LINE = hex_color(0xd6cfb8);
    p.TEXT = hex_color(0x3b4d54);
    p.DIM = hex_color(0x6c7f85);
    p.MENU = hex_color(0xeee8d5);
    p.DROP = hex_color(0xfdf6e3);
    p.STATUS = hex_color(0xeee8d5);
    p.BTN = hex_color(0xd9e6ea);
    p.BTN_HOVER = hex_color(0xc3dbe8);
    p.FIELD = hex_color(0xeee8d5);
    p.ACCENT = hex_color(0x1f77b8);
    p.SEL = hex_color(0xd3e3ec);
    p.PLOT_BG = hex_color(0xfffbee);
    p.GRID = hex_color(0xece5d2);
    p.SCROLL_THUMB = hex_color(0xc7c0aa);
    p.S_A = hex_color(0x1f77b8);
    p.S_A6 = hex_color(0xc2307a);
    p.S_B = hex_color(0x15867f);
    p.LIFE_A = hex_color(0x1f77b8);
    p.LIFE_B = hex_color(0xcb3a1b);
    p.BAD = hex_color(0xc62a28);
    p.GOOD = hex_color(0x5f7400);
    p.FOLDER = hex_color(0x9a7300);
    p.LOGO_BG = hex_color(0x268bd2);
    p.LOGO_FG = hex_color(0xfdf6e3);
    p.GLOW = hex_color(0xd9a400);
    p.LINK = hex_color(0x1a9e94);
    p.COUNTER = hex_color(0xd8d5f2);
    p.SEAT_A = hex_color(0xcfe0ee);
    p.SEAT_B = hex_color(0xf2d3cf);
    return p;
}

// Arctic Ice Studio's Nord: Polar Night panels, Frost and Aurora accents.
static rmwork::Palette nord_palette() {
    using rmwork::hex_color;
    rmwork::Palette p{};
    p.BG = hex_color(0x242933);
    p.BG_TEXT = hex_color(0x9aa5b8);
    p.PANEL = hex_color(0x2e3440);
    p.PANEL_2 = hex_color(0x333a47);
    p.HEADER = hex_color(0x3b4252);
    p.TITLE = hex_color(0x2a303b);
    p.TITLE_ACTIVE = hex_color(0x40577a);
    p.TITLE_TEXT = hex_color(0xeceff4);
    p.TAB = hex_color(0x353c4a);
    p.TAB_ACTIVE = hex_color(0x4a5f80);
    p.TAB_DIM = hex_color(0x3e4a5e);
    p.LINE = hex_color(0x434c5e);
    p.TEXT = hex_color(0xe5e9f0);
    p.DIM = hex_color(0x98a2b4);
    p.MENU = hex_color(0x2a2f3a);
    p.DROP = hex_color(0x353b48);
    p.STATUS = hex_color(0x272c36);
    p.BTN = hex_color(0x434c5e);
    p.BTN_HOVER = hex_color(0x4c5870);
    p.FIELD = hex_color(0x3b4252);
    p.ACCENT = hex_color(0x88c0d0);
    p.SEL = hex_color(0x4a5a73);
    p.PLOT_BG = hex_color(0x2a2f3a);
    p.GRID = hex_color(0x3b4252);
    p.SCROLL_THUMB = hex_color(0x4c566a);
    p.S_A = hex_color(0x81a1c1);
    p.S_A6 = hex_color(0xd08770);
    p.S_B = hex_color(0xa3be8c);
    p.LIFE_A = hex_color(0x81a1c1);
    p.LIFE_B = hex_color(0xbf616a);
    p.BAD = hex_color(0xd0717a);
    p.GOOD = hex_color(0xa3be8c);
    p.FOLDER = hex_color(0xebcb8b);
    p.LOGO_BG = hex_color(0x5e81ac);
    p.LOGO_FG = hex_color(0xeceff4);
    p.GLOW = hex_color(0xebcb8b);
    p.LINK = hex_color(0x88c0d0);
    p.COUNTER = hex_color(0x6a4f8a);
    p.SEAT_A = hex_color(0x3b5a80);
    p.SEAT_B = hex_color(0x7a3b44);
    return p;
}

// The first mock-up: grey faces, navy titles, white fields, the teal desktop.
static rmwork::Palette windows_95_palette() {
    using rmwork::hex_color;
    rmwork::Palette p{};
    p.BG = hex_color(0x008080);
    p.BG_TEXT = hex_color(0xffffff);
    p.PANEL = hex_color(0xc0c0c0);
    p.PANEL_2 = hex_color(0xb8b8b8);
    p.HEADER = hex_color(0xd4d0c8);
    p.TITLE = hex_color(0x9c9c9c);
    p.TITLE_ACTIVE = hex_color(0x000080);
    p.TITLE_TEXT = hex_color(0xffffff);
    p.TAB = hex_color(0xc0c0c0);
    p.TAB_ACTIVE = hex_color(0xe0e0e0);
    p.TAB_DIM = hex_color(0xb4b4b4);
    p.LINE = hex_color(0x808080);
    p.TEXT = hex_color(0x000000);
    p.DIM = hex_color(0x4a4a4a);
    p.MENU = hex_color(0xc0c0c0);
    p.DROP = hex_color(0xd4d0c8);
    p.STATUS = hex_color(0xc0c0c0);
    p.BTN = hex_color(0xd8d8d8);
    p.BTN_HOVER = hex_color(0xe8e8e8);
    p.FIELD = hex_color(0xffffff);
    p.ACCENT = hex_color(0x000080);
    p.SEL = hex_color(0xa6caf0);
    p.PLOT_BG = hex_color(0xffffff);
    p.GRID = hex_color(0xd8d8d8);
    p.SCROLL_THUMB = hex_color(0x9a9a9a);
    p.S_A = hex_color(0x0000c0);
    p.S_A6 = hex_color(0xb000b0);
    p.S_B = hex_color(0x006666);
    p.LIFE_A = hex_color(0x0000c0);
    p.LIFE_B = hex_color(0xc00000);
    p.BAD = hex_color(0xc00000);
    p.GOOD = hex_color(0x007800);
    p.FOLDER = hex_color(0x806000);
    p.LOGO_BG = hex_color(0x000080);
    p.LOGO_FG = hex_color(0xffffff);
    p.GLOW = hex_color(0xffff00);
    p.LINK = hex_color(0x00c0ff);
    p.COUNTER = hex_color(0xc8b4ea);
    p.SEAT_A = hex_color(0xa8c4ea);
    p.SEAT_B = hex_color(0xeab0b0);
    return p;
}

static ThemeState &state() {
    static ThemeState s = [] {
        ThemeState built;
        for (int i = 0; i < rmwork::THEME_COUNT; ++i) {
            built.palettes[i] = THEMES[i].build();
            derive(&built.palettes[i], THEMES[i].info.light);
        }
        return built;
    }();
    return s;
}

// `a` moved `t` of the way to `b`, channel by channel (opaque).
static ImU32 mix(ImU32 a, ImU32 b, float t) {
    const ImVec4 x = ImGui::ColorConvertU32ToFloat4(a), y = ImGui::ColorConvertU32ToFloat4(b);
    return ImGui::ColorConvertFloat4ToU32(
        ImVec4(x.x + (y.x - x.x) * t, x.y + (y.y - x.y) * t, x.z + (y.z - x.z) * t, 1.0f));
}

// The pressed and hovered shades, each a step toward the text colour (lighter on a dark theme,
// darker on a light one), and the dialogs' dimming.
static void derive(rmwork::Palette *p, bool light) {
    p->FIELD_HOVER = mix(p->FIELD, p->TEXT, 0.06f);
    p->FIELD_ACTIVE = mix(p->FIELD, p->TEXT, 0.10f);
    p->BTN_ACTIVE = mix(p->BTN_HOVER, p->TEXT, 0.10f);
    p->SCROLL_THUMB_HOVER = mix(p->SCROLL_THUMB, p->TEXT, 0.12f);
    p->SCROLL_THUMB_ACTIVE = mix(p->SCROLL_THUMB, p->TEXT, 0.25f);
    p->ACCENT_ACTIVE = mix(p->ACCENT, p->TEXT, 0.20f);
    p->MODAL_DIM = rmwork::hex_color(0x000000, light ? 64 : 90);
    p->TABLE_SHADE = (p->PANEL & ~IM_COL32_A_MASK) | (static_cast<ImU32>(224) << IM_COL32_A_SHIFT);
}

static ImVec4 v4(ImU32 c) {
    return ImGui::ColorConvertU32ToFloat4(c);
}

static ImVec4 v4a(ImU32 c, float alpha) {
    ImVec4 v = v4(c);
    v.w = alpha;
    return v;
}

// ImHex-like metrics for every theme: square panels and tabs, small frame rounding, compact
// padding.
static void apply_imgui_metrics() {
    ImGuiStyle &s = ImGui::GetStyle();
    s = ImGuiStyle();
    s.WindowPadding = ImVec2(6, 6);
    s.FramePadding = ImVec2(6, 3);
    s.CellPadding = ImVec2(6, 2);
    s.ItemSpacing = ImVec2(6, 4);
    s.ItemInnerSpacing = ImVec2(5, 4);
    s.IndentSpacing = 14;
    s.ScrollbarSize = 10;
    s.GrabMinSize = 8;
    s.WindowBorderSize = 1;
    s.ChildBorderSize = 1;
    s.PopupBorderSize = 1;
    s.FrameBorderSize = 0;
    s.TabBorderSize = 0;
    s.TabBarBorderSize = 2;
    s.TabBarOverlineSize = 0;
    s.WindowRounding = 0;
    s.ChildRounding = 0;
    s.FrameRounding = 2;
    s.PopupRounding = 0;
    s.ScrollbarRounding = 5;
    s.GrabRounding = 2;
    s.TabRounding = 0;
    s.WindowTitleAlign = ImVec2(0.0f, 0.5f);
    s.WindowMenuButtonPosition = ImGuiDir_None;
    s.DockingSeparatorSize = 4;
    s.SeparatorTextBorderSize = 1;
}

// Every ImGui colour from the palette.
static void apply_imgui_colors(const rmwork::Palette &t) {
    ImVec4 *c = ImGui::GetStyle().Colors;
    c[ImGuiCol_Text] = v4(t.TEXT);
    c[ImGuiCol_TextDisabled] = v4(t.DIM);
    c[ImGuiCol_WindowBg] = v4(t.PANEL);
    c[ImGuiCol_ChildBg] = v4a(t.PANEL, 0.0f);
    c[ImGuiCol_PopupBg] = v4(t.DROP);
    c[ImGuiCol_Border] = v4(t.LINE);
    c[ImGuiCol_BorderShadow] = v4a(t.BG, 0.0f);
    c[ImGuiCol_FrameBg] = v4(t.FIELD);
    c[ImGuiCol_FrameBgHovered] = v4(t.FIELD_HOVER);
    c[ImGuiCol_FrameBgActive] = v4(t.FIELD_ACTIVE);
    c[ImGuiCol_TitleBg] = v4(t.TITLE);
    c[ImGuiCol_TitleBgActive] = v4(t.TITLE);
    c[ImGuiCol_TitleBgCollapsed] = v4(t.TITLE);
    c[ImGuiCol_MenuBarBg] = v4(t.MENU);
    c[ImGuiCol_ScrollbarBg] = v4(t.PANEL);
    c[ImGuiCol_ScrollbarGrab] = v4(t.SCROLL_THUMB);
    c[ImGuiCol_ScrollbarGrabHovered] = v4(t.SCROLL_THUMB_HOVER);
    c[ImGuiCol_ScrollbarGrabActive] = v4(t.SCROLL_THUMB_ACTIVE);
    c[ImGuiCol_CheckMark] = v4(t.ACCENT);
    c[ImGuiCol_CheckboxSelectedBg] = v4(t.FIELD_ACTIVE);
    c[ImGuiCol_SliderGrab] = v4(t.ACCENT);
    c[ImGuiCol_SliderGrabActive] = v4(t.ACCENT_ACTIVE);
    c[ImGuiCol_Button] = v4(t.BTN);
    c[ImGuiCol_ButtonHovered] = v4(t.BTN_HOVER);
    c[ImGuiCol_ButtonActive] = v4(t.BTN_ACTIVE);
    c[ImGuiCol_Header] = v4(t.SEL);
    c[ImGuiCol_HeaderHovered] = v4(t.BTN_HOVER);
    c[ImGuiCol_HeaderActive] = v4(t.BTN_ACTIVE);
    c[ImGuiCol_Separator] = v4(t.LINE);
    c[ImGuiCol_SeparatorHovered] = v4(t.ACCENT);
    c[ImGuiCol_SeparatorActive] = v4(t.ACCENT);
    c[ImGuiCol_ResizeGrip] = v4a(t.ACCENT, 0.0f);
    c[ImGuiCol_ResizeGripHovered] = v4a(t.ACCENT, 0.6f);
    c[ImGuiCol_ResizeGripActive] = v4(t.ACCENT);
    c[ImGuiCol_InputTextCursor] = v4(t.TEXT);
    c[ImGuiCol_TabHovered] = v4(t.BTN_HOVER);
    c[ImGuiCol_Tab] = v4(t.TAB);
    c[ImGuiCol_TabSelected] = v4(t.TAB_ACTIVE);
    c[ImGuiCol_TabSelectedOverline] = v4(t.TAB_ACTIVE);
    c[ImGuiCol_TabDimmed] = v4(t.TAB);
    c[ImGuiCol_TabDimmedSelected] = v4(t.TAB_DIM);
    c[ImGuiCol_TabDimmedSelectedOverline] = v4(t.TAB_DIM);
    c[ImGuiCol_DockingPreview] = v4a(t.ACCENT, 0.45f);
    c[ImGuiCol_DockingEmptyBg] = v4(t.BG);
    c[ImGuiCol_PlotLines] = v4(t.S_A);
    c[ImGuiCol_PlotLinesHovered] = v4(t.ACCENT);
    c[ImGuiCol_PlotHistogram] = v4(t.ACCENT);
    c[ImGuiCol_PlotHistogramHovered] = v4(t.BTN_HOVER);
    c[ImGuiCol_TableHeaderBg] = v4(t.HEADER);
    c[ImGuiCol_TableBorderStrong] = v4(t.LINE);
    c[ImGuiCol_TableBorderLight] = v4(t.LINE);
    c[ImGuiCol_TableRowBg] = v4a(t.PANEL, 0.0f);
    c[ImGuiCol_TableRowBgAlt] = v4(t.PANEL_2);
    c[ImGuiCol_TextLink] = v4(t.ACCENT);
    c[ImGuiCol_TextSelectedBg] = v4a(t.SEL, 0.85f);
    c[ImGuiCol_TreeLines] = v4(t.LINE);
    c[ImGuiCol_DragDropTarget] = v4(t.ACCENT);
    c[ImGuiCol_DragDropTargetBg] = v4a(t.ACCENT, 0.0f);
    c[ImGuiCol_UnsavedMarker] = v4(t.TEXT);
    c[ImGuiCol_NavCursor] = v4(t.ACCENT);
    c[ImGuiCol_NavWindowingHighlight] = v4a(t.TEXT, 0.7f);
    c[ImGuiCol_NavWindowingDimBg] = v4(t.MODAL_DIM);
    c[ImGuiCol_ModalWindowDimBg] = v4(t.MODAL_DIM);
}

static void apply_plot_metrics() {
    ImPlotStyle &s = ImPlot::GetStyle();
    s.PlotPadding = ImVec2(6, 6);
    s.LabelPadding = ImVec2(4, 3);
    s.LineWeight = 1.6f;
    s.PlotBorderSize = 1;
    s.MajorTickLen = ImVec2(4, 4);
    s.MinorTickLen = ImVec2(2, 2);
    s.PlotMinSize = ImVec2(60, 30);
}

// ImPlot in the palette; the series colours are the theme's colormap.
static void apply_plot_colors(const rmwork::Palette &t, int colormap) {
    ImPlotStyle &s = ImPlot::GetStyle();
    ImVec4 *c = s.Colors;
    c[ImPlotCol_FrameBg] = v4(t.PLOT_BG);
    c[ImPlotCol_PlotBg] = v4(t.PLOT_BG);
    c[ImPlotCol_PlotBorder] = v4(t.LINE);
    c[ImPlotCol_LegendBg] = v4a(t.DROP, 0.9f);
    c[ImPlotCol_LegendBorder] = v4(t.LINE);
    c[ImPlotCol_LegendText] = v4(t.TEXT);
    c[ImPlotCol_TitleText] = v4(t.TEXT);
    c[ImPlotCol_InlayText] = v4(t.DIM);
    c[ImPlotCol_AxisText] = v4(t.DIM);
    c[ImPlotCol_AxisGrid] = v4(t.GRID);
    c[ImPlotCol_AxisTick] = v4(t.GRID);
    c[ImPlotCol_AxisBg] = v4a(t.PLOT_BG, 0.0f);
    c[ImPlotCol_AxisBgHovered] = v4a(t.BTN_HOVER, 0.5f);
    c[ImPlotCol_AxisBgActive] = v4a(t.BTN_HOVER, 0.7f);
    c[ImPlotCol_Selection] = v4(t.ACCENT);
    c[ImPlotCol_Crosshairs] = v4(t.TEXT);
    c[ImPlotCol_ErrorBar] = v4(t.TEXT);
    s.Colormap = colormap;
}

// One colormap per theme in the current ImPlot context: V (A), V (A, re-eval), −V of B, life A,
// life B.
static void add_colormaps() {
    ThemeState &st = state();
    for (int i = 0; i < rmwork::THEME_COUNT; ++i) {
        const rmwork::Palette &t = st.palettes[i];
        const ImVec4 series[] = {v4(t.S_A), v4(t.S_A6), v4(t.S_B), v4(t.LIFE_A), v4(t.LIFE_B)};
        const std::string name = std::string("Robomage ") + THEMES[i].info.name;
        st.colormaps[i] = ImPlot::AddColormap(name.c_str(), series, static_cast<int>(sizeof series / sizeof series[0]));
    }
}

// An sRGB channel (0–1) to linear light.
static float linear(float channel) {
    return channel <= 0.04045f ? channel / 12.92f : std::pow((channel + 0.055f) / 1.055f, 2.4f);
}

// CIE L*a*b* (D65) of an opaque colour.
static void to_lab(ImU32 c, float lab[3]) {
    const ImVec4 v = ImGui::ColorConvertU32ToFloat4(c);
    const float r = linear(v.x), g = linear(v.y), b = linear(v.z);
    const float xyz[3] = {(0.4124f * r + 0.3576f * g + 0.1805f * b) / 0.95047f, 0.2126f * r + 0.7152f * g + 0.0722f * b,
        (0.0193f * r + 0.1192f * g + 0.9505f * b) / 1.08883f};
    float f[3];
    for (int i = 0; i < 3; ++i) f[i] = xyz[i] > 0.008856f ? std::cbrt(xyz[i]) : 7.787f * xyz[i] + 16.0f / 116.0f;
    lab[0] = 116.0f * f[1] - 16.0f;
    lab[1] = 500.0f * (f[0] - f[1]);
    lab[2] = 200.0f * (f[1] - f[2]);
}

namespace rmwork {

const ThemeInfo &theme_info(int index) {
    return THEMES[index].info;
}

int find_theme(const std::string &id) {
    for (int i = 0; i < THEME_COUNT; ++i)
        if (id == THEMES[i].info.id) return i;
    return -1;
}

const Palette &theme_palette(int index) {
    return state().palettes[index];
}

const Palette &tok() {
    ThemeState &st = state();
    return st.palettes[st.current];
}

int current_theme() {
    return state().current;
}

void apply_theme(const std::string &id) {
    ThemeState &st = state();
    const int i = find_theme(id);
    st.current = i >= 0 ? i : find_theme(DEFAULT_THEME);
    apply_imgui_colors(st.palettes[st.current]);
    apply_plot_colors(st.palettes[st.current], st.colormaps[st.current]);
}

float contrast_ratio(ImU32 a, ImU32 b) {
    const auto luminance = [](ImU32 c) {
        const ImVec4 v = ImGui::ColorConvertU32ToFloat4(c);
        return 0.2126f * linear(v.x) + 0.7152f * linear(v.y) + 0.0722f * linear(v.z);
    };
    const float la = luminance(a), lb = luminance(b);
    return (std::max(la, lb) + 0.05f) / (std::min(la, lb) + 0.05f);
}

float color_difference(ImU32 a, ImU32 b) {
    float x[3], y[3];
    to_lab(a, x);
    to_lab(b, y);
    return std::sqrt((x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]) + (x[2] - y[2]) * (x[2] - y[2]));
}

void create_ui_context() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigWindowsMoveFromTitleBarOnly = true;
    io.IniFilename = nullptr;
    apply_imgui_metrics();
    apply_plot_metrics();
    add_colormaps();
    apply_theme(DEFAULT_THEME);
    load_ui_fonts();
    apply_font(FontChoice());
}

void destroy_ui_context() {
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
}

}  // namespace rmwork
