#include "selftest.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "app.h"
#include "files.h"
#include "imgui_internal.h"
#include "implot.h"
#include "theme.h"

namespace {

// The checks' tally; each failure is printed as it happens.
struct Tally {
        int checks = 0;
        int failures = 0;
};

const ImVec2 DISPLAY(1600, 1000);
// How far a group's node may sit from the preset's split of the dockspace (splitters, rounding).
const float RECT_TOLERANCE = 8.0f;

}  // namespace

static void check(Tally *t, bool ok, const std::string &what);
static void begin_headless();
static void run_frames(rmwork::App &app, int n);
static ImGuiWindow *tool_window(rmwork::Mode m, rmwork::Tool t);
static bool window_active(const ImGuiWindow *w);
static std::string tab_names(const ImGuiDockNode *node);
static std::string where(rmwork::Mode m, const rmwork::Preset &p);
static void check_layout(Tally *t, const rmwork::App &app, rmwork::Mode m, int preset);
static void check_other_mode_hidden(Tally *t, rmwork::Mode m);
static void check_presets(Tally *t, rmwork::App &app);
static void check_mode_switching(Tally *t, rmwork::App &app);
static void check_reopen(Tally *t, rmwork::App &app);
static void press(rmwork::App &app, ImGuiKeyChord chord);
static void check_keys(Tally *t, rmwork::App &app);
static void begin_inspected_frame(rmwork::App &app);
static void end_inspected_frame(rmwork::App &app);
static void check_font_frame(Tally *t, int family, rmwork::FontWeight w, int size);
static void check_fonts(Tally *t, rmwork::App &app);
static void check_bars(Tally *t, const std::string &at);
static void check_font_sizes(Tally *t, rmwork::App &app);
static rmwork::FontChoice font_from_json(const std::string &block, bool *ok);
static void check_themes(Tally *t, rmwork::App &app);
static void check_min_contrast(Tally *t, const std::string &at, const char *fg_name, ImU32 fg,
    const std::vector<std::pair<const char *, ImU32>> &backgrounds, float minimum);
static void check_theme_contrast(Tally *t);
static void check_theme_settings(Tally *t);
static void check_font_settings(Tally *t);
static void check_ini_round_trip(Tally *t, const rmwork::ConfigPaths &paths);
static void check_settings_file(Tally *t, const rmwork::ConfigPaths &paths, const std::string &tmp);
static void check_app_persists(Tally *t, const rmwork::ConfigPaths &paths);

static void check(Tally *t, bool ok, const std::string &what) {
    ++t->checks;
    if (ok) return;
    ++t->failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

// A fresh context sized like a window. It claims a renderer that takes texture requests, so the
// font atlas needs no building; nothing is drawn.
static void begin_headless() {
    rmwork::create_ui_context();
    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize = DISPLAY;
    io.DeltaTime = 1.0f / 60.0f;
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
}

static void run_frames(rmwork::App &app, int n) {
    for (int i = 0; i < n; ++i) {
        app.begin_frame();
        ImGui::NewFrame();
        app.frame();
        ImGui::Render();
    }
}

static ImGuiWindow *tool_window(rmwork::Mode m, rmwork::Tool t) {
    return ImGui::FindWindowByName(rmwork::tool_window_name(m, t).c_str());
}

// Submitted in the last frame.
static bool window_active(const ImGuiWindow *w) {
    return w && w->Active;
}

// The node's tabs in order (" Players Decision Inspector(closed)"), for a failure's message.
static std::string tab_names(const ImGuiDockNode *node) {
    std::string out;
    if (!node->TabBar) return out;
    for (const ImGuiTabItem &tab : node->TabBar->Tabs)
        if (tab.Window) out += std::string(" ") + tab.Window->Name + (window_active(tab.Window) ? "" : "(closed)");
    return out;
}

static std::string where(rmwork::Mode m, const rmwork::Preset &p) {
    return std::string(rmwork::mode_title(m)) + " / " + p.name + ": ";
}

// The last frame shows exactly preset `preset`'s tools in mode `m`, each group in one dock node of
// the mode's dockspace, in order, the first tab selected, at the preset's split ratios.
static void check_layout(Tally *t, const rmwork::App &app, rmwork::Mode m, int preset) {
    using namespace rmwork;
    const Preset &p = presets()[static_cast<size_t>(preset)];
    const std::string at = where(m, p);
    const ToolFlags expected = preset_tools(p);
    for (int i = 0; i < TOOL_COUNT; ++i) {
        const Tool tool = static_cast<Tool>(i);
        const std::string name = tool_info(tool).title;
        check(t, app.tool_open(m, tool) == expected[i], at + name + (expected[i] ? " is closed" : " is open"));
        check(t, window_active(tool_window(m, tool)) == expected[i],
            at + name + (expected[i] ? " is not drawn" : " is drawn though closed"));
    }
    const ImGuiDockNode *root = ImGui::DockBuilderGetNode(dockspace_id(m));
    check(t, root != nullptr, at + "no dockspace node");
    if (!root) return;
    const std::vector<std::vector<Tool>> groups = preset_groups(p);
    const std::vector<std::pair<ImVec2, ImVec2>> rects = preset_group_rects(p, root->Pos, root->Size);
    std::vector<const ImGuiDockNode *> nodes;
    for (size_t g = 0; g < groups.size(); ++g) {
        const ImGuiWindow *first = tool_window(m, groups[g].front());
        const ImGuiDockNode *node = first ? first->DockNode : nullptr;
        const std::string gname = at + "group of " + tool_info(groups[g].front()).title;
        check(t, node != nullptr, gname + " is not docked");
        if (!node) continue;
        check(t, ImGui::DockNodeGetRootNode(const_cast<ImGuiDockNode *>(node))->ID == root->ID,
            gname + " is docked outside the mode's dockspace");
        for (const ImGuiDockNode *other : nodes) check(t, other != node, gname + " shares a node with another group");
        nodes.push_back(node);
        std::vector<const ImGuiWindow *> want;
        for (Tool tool : groups[g]) {
            const ImGuiWindow *w = tool_window(m, tool);
            want.push_back(w);
            check(t, w && w->DockNode == node, at + tool_info(tool).title + " is not in its group's tab node");
        }
        std::vector<const ImGuiWindow *> tabs;
        if (node->TabBar)
            for (const ImGuiTabItem &tab : node->TabBar->Tabs)
                if (tab.Window && window_active(tab.Window)) tabs.push_back(tab.Window);
        check(t, tabs == want, gname + ": tabs not in the preset's order:" + tab_names(node));
        check(t, node->VisibleWindow == first, gname + ": the first tab is not the selected one");
        const ImVec2 lo = rects[g].first, hi = rects[g].second;
        const ImVec2 nlo = node->Pos, nhi = ImVec2(node->Pos.x + node->Size.x, node->Pos.y + node->Size.y);
        const bool near = std::fabs(nlo.x - lo.x) <= RECT_TOLERANCE && std::fabs(nlo.y - lo.y) <= RECT_TOLERANCE &&
                          std::fabs(nhi.x - hi.x) <= RECT_TOLERANCE && std::fabs(nhi.y - hi.y) <= RECT_TOLERANCE;
        char buf[200];
        std::snprintf(buf, sizeof buf, ": at (%.0f,%.0f)-(%.0f,%.0f), the preset puts it at (%.0f,%.0f)-(%.0f,%.0f)",
            nlo.x, nlo.y, nhi.x, nhi.y, lo.x, lo.y, hi.x, hi.y);
        check(t, near, gname + buf);
    }
    check_other_mode_hidden(t, m);
}

// No tool window of the mode other than `m` was drawn.
static void check_other_mode_hidden(Tally *t, rmwork::Mode m) {
    using namespace rmwork;
    const Mode other = m == Mode::PLAY ? Mode::REPLAY : Mode::PLAY;
    for (int i = 0; i < TOOL_COUNT; ++i)
        check(t, !window_active(tool_window(other, static_cast<Tool>(i))),
            std::string(mode_title(m)) + ": " + mode_title(other) + "'s " + tool_info(static_cast<Tool>(i)).title +
                " is drawn");
}

// Every preset in every mode, then Reset Layout.
static void check_presets(Tally *t, rmwork::App &app) {
    using namespace rmwork;
    for (int mi = 0; mi < MODE_COUNT; ++mi) {
        const Mode m = static_cast<Mode>(mi);
        app.set_mode(m);
        run_frames(app, 3);
        for (size_t p = 0; p < presets().size(); ++p) {
            app.request_preset(static_cast<int>(p));
            run_frames(app, 3);
            check_layout(t, app, m, static_cast<int>(p));
        }
        app.request_preset(default_preset(m));
        run_frames(app, 3);
        check_layout(t, app, m, default_preset(m));
    }
}

// Each mode keeps its own layout while the other is shown.
static void check_mode_switching(Tally *t, rmwork::App &app) {
    using namespace rmwork;
    const int analysis = find_preset("analysis"), table = find_preset("table");
    app.set_mode(Mode::REPLAY);
    app.request_preset(analysis);
    run_frames(app, 3);
    app.set_mode(Mode::PLAY);
    app.request_preset(table);
    run_frames(app, 3);
    check_layout(t, app, Mode::PLAY, table);
    app.set_mode(Mode::REPLAY);
    run_frames(app, 3);
    check_layout(t, app, Mode::REPLAY, analysis);
    app.set_mode(Mode::PLAY);
    run_frames(app, 3);
    check_layout(t, app, Mode::PLAY, table);
}

// A closed tool reopens in the node it left, even the only tool of its group; a tool outside the
// preset opens behind the first group.
static void check_reopen(Tally *t, rmwork::App &app) {
    using namespace rmwork;
    app.set_mode(Mode::REPLAY);
    app.request_preset(default_preset(Mode::REPLAY));
    run_frames(app, 3);
    for (Tool tool : {Tool::GRAPH, Tool::PLAYERS}) {
        const std::string name = tool_info(tool).title;
        const ImGuiWindow *w = tool_window(Mode::REPLAY, tool);
        const ImGuiID before = w && w->DockNode ? w->DockNode->ID : 0;
        app.set_tool_open(Mode::REPLAY, tool, false);
        run_frames(app, 3);
        check(t, !window_active(tool_window(Mode::REPLAY, tool)), "close: " + name + " is still drawn");
        app.set_tool_open(Mode::REPLAY, tool, true);
        run_frames(app, 3);
        w = tool_window(Mode::REPLAY, tool);
        check(t, window_active(w) && w->DockNode && w->DockNode->ID == before,
            "reopen: " + name + " is not back in the node it was closed from");
    }
    run_frames(app, 2);
    check_layout(t, app, Mode::REPLAY, default_preset(Mode::REPLAY));
    app.request_preset(find_preset("table"));
    run_frames(app, 3);
    app.set_tool_open(Mode::REPLAY, Tool::GRAPH, true);
    run_frames(app, 3);
    const ImGuiWindow *graph = tool_window(Mode::REPLAY, Tool::GRAPH);
    const ImGuiWindow *table = tool_window(Mode::REPLAY, Tool::TABLE);
    check(t, window_active(graph) && table && graph->DockNode && graph->DockNode == table->DockNode,
        "View: a tool outside the preset does not open behind its first group");
}

// Presses and releases `chord`, a frame each.
static void press(rmwork::App &app, ImGuiKeyChord chord) {
    ImGuiIO &io = ImGui::GetIO();
    const ImGuiKeyChord mods = chord & ImGuiMod_Mask_;
    const ImGuiKey key = static_cast<ImGuiKey>(chord & ~ImGuiMod_Mask_);
    if (mods & ImGuiMod_Ctrl) io.AddKeyEvent(ImGuiMod_Ctrl, true);
    io.AddKeyEvent(key, true);
    run_frames(app, 1);
    io.AddKeyEvent(key, false);
    if (mods & ImGuiMod_Ctrl) io.AddKeyEvent(ImGuiMod_Ctrl, false);
    run_frames(app, 1);
}

// The bindings act in their mode only and write a status message without a game; Ctrl+, opens
// Settings, Esc closes it and the error dialog.
static void check_keys(Tally *t, rmwork::App &app) {
    using namespace rmwork;
    const KeyBindings keys = default_bindings();
    app.set_mode(Mode::REPLAY);
    run_frames(app, 2);
    app.status("");
    press(app, keys[action_index(Action::REPLAY_NEXT)]);
    check(t, app.status_message() == "No game loaded: Next decision", "Replay: → does not report the missing game");
    app.status("");
    press(app, keys[action_index(Action::PLAY_TO_END_OF_TURN)]);
    check(t, app.status_message().empty(), "Replay: F2 acts outside Play");
    app.set_mode(Mode::PLAY);
    run_frames(app, 2);
    app.status("");
    press(app, keys[action_index(Action::PLAY_SKIP_TURN)]);
    check(t, app.status_message() == "No game loaded: Skip turn", "Play: F6 does not report the missing game");
    app.status("");
    press(app, keys[action_index(Action::REPLAY_FIRST)]);
    check(t, app.status_message().empty(), "Play: Home acts outside Replay");
    press(app, keys[action_index(Action::OPEN_REPLAY)]);
    check(t, app.status_message() == "not built yet: Open Replay", "Ctrl+O does not report that it is not built");
    press(app, keys[action_index(Action::OPEN_SETTINGS)]);
    check(t, app.settings_open(), "Ctrl+, does not open Settings");
    app.status("");
    press(app, keys[action_index(Action::PLAY_SKIP_TURN)]);
    check(t, app.status_message().empty(), "a key acts behind the Settings dialog");
    press(app, ImGuiKey_Escape);
    check(t, !app.settings_open(), "Esc does not close Settings");
    app.report_error("Title", "Message", "detail\n");
    run_frames(app, 2);
    check(t, app.error_open(), "report_error does not open the error dialog");
    press(app, ImGuiKey_Escape);
    check(t, !app.error_open(), "Esc does not close the error dialog");
}

// A frame of `app` left open after NewFrame, for checks that need the frame's font state.
static void begin_inspected_frame(rmwork::App &app) {
    app.begin_frame();
    ImGui::NewFrame();
}

static void end_inspected_frame(rmwork::App &app) {
    app.frame();
    ImGui::Render();
}

// Inside a frame: the current font is `family` × `w` at `size` (the size the setting asks for, not
// the size the font was added with), its own glyphs drawn from its own file and the UI's
// non-ASCII ones present (its own or JetBrains Mono's); monospace families draw "iiii" as wide as
// "MMMM".
static void check_font_frame(Tally *t, int family, rmwork::FontWeight w, int size) {
    using namespace rmwork;
    const std::string at = std::string("font ") + font_family(family).name + " " + font_weight_name(family, w) + " " +
                           std::to_string(size) + ": ";
    ImFont *font = ImGui::GetFont();
    check(t, font == ui_font(family, w), at + "is not the frame's font");
    check(t, ImGui::GetFontSize() == static_cast<float>(size),
        at + "drawn at " + std::to_string(static_cast<int>(ImGui::GetFontSize())) + " px");
    check(t, !font->Sources.empty() && std::string(font->Sources[0]->Name) == font_family(family).name,
        at + "the font's first source is not the family's file");
    const ImFont *other = ui_font(family, w == FontWeight::REGULAR ? FontWeight::HEAVY : FontWeight::REGULAR);
    check(t, other != font && other->Sources[0]->FontData != font->Sources[0]->FontData,
        at + "the two weights share a file");
    ImFontBaked *baked = ImGui::GetFontBaked();
    const ImFontGlyph *m = baked->FindGlyphNoFallback('M');
    check(t, m && m->SourceIdx == 0 && m->Visible && m->AdvanceX > 0, at + "'M' is not the family's own glyph");
    for (ImWchar c : {ImWchar(0x2192), ImWchar(0x2014), ImWchar(0x00B7), ImWchar(0x2026), ImWchar(0x2212)})
        check(t, baked->FindGlyphNoFallback(c) != nullptr, at + "no glyph for U+" + std::to_string(c));
    const bool mono = ImGui::CalcTextSize("iiii").x == ImGui::CalcTextSize("MMMM").x;
    check(t, mono == !font_family(family).proportional,
        at + (mono ? "draws as monospace though proportional" : "draws as proportional though monospace"));
}

// Every family × weight becomes the UI font at the next frame and draws one.
static void check_fonts(Tally *t, rmwork::App &app) {
    using namespace rmwork;
    for (int f = 0; f < FONT_FAMILY_COUNT; ++f) {
        for (FontWeight w : {FontWeight::REGULAR, FontWeight::HEAVY}) {
            FontChoice c;
            c.family = font_family(f).id;
            c.weight = w;
            c.size = 16;
            app.set_font(c);
            begin_inspected_frame(app);
            check_font_frame(t, f, w, c.size);
            end_inspected_frame(app);
            const ImDrawData *dd = ImGui::GetDrawData();
            check(
                t, dd && dd->TotalVtxCount > 0, std::string("font ") + font_family(f).id + ": the frame draws nothing");
        }
    }
    app.set_font(FontChoice());
    run_frames(app, 2);
}

// The bars of the last frame fit the font: the menu bar holds the logo, the toolbar its frame-high
// items with its padding, the status bar a line of text; tab strips are at least a line high.
static void check_bars(Tally *t, const std::string &at) {
    const ImGuiStyle &style = ImGui::GetStyle();
    const float line = ImGui::GetFontSize();
    const float frame = line + style.FramePadding.y * 2;
    const ImGuiWindow *menu = ImGui::FindWindowByName("##MainMenuBar");
    check(t, menu && menu->MenuBarHeight >= std::max(18.0f, line), at + "the menu bar is lower than its text or logo");
    const ImGuiWindow *toolbar = ImGui::FindWindowByName("##rmwork.toolbar");
    check(
        t, toolbar && std::fabs(toolbar->Size.y - (frame + 9)) < 0.5f, at + "the toolbar is not the frame height + 9");
    check(
        t, toolbar && toolbar->ContentSize.y <= toolbar->Size.y - 2 * 4 + 0.5f, at + "the toolbar's items overflow it");
    const ImGuiWindow *status = ImGui::FindWindowByName("##rmwork.status");
    check(t, status && std::fabs(status->Size.y - (line + 7)) < 0.5f, at + "the status bar is not a line + 7");
    const ImGuiDockNode *root = ImGui::DockBuilderGetNode(rmwork::dockspace_id(rmwork::Mode::REPLAY));
    const ImGuiWindow *table = tool_window(rmwork::Mode::REPLAY, rmwork::Tool::TABLE);
    check(t,
        root && table && table->DockNode && table->DockNode->TabBar &&
            table->DockNode->TabBar->BarRect.GetHeight() >= line,
        at + "the Table's tab strip is lower than a line");
}

// At the smallest and largest sizes the bars fit and each mode's default preset still lays out.
static void check_font_sizes(Tally *t, rmwork::App &app) {
    using namespace rmwork;
    app.set_mode(Mode::REPLAY);
    for (int size : {FONT_SIZE_MIN, FONT_SIZE_MAX}) {
        FontChoice c;
        c.size = size;
        app.set_font(c);
        for (int mi = 0; mi < MODE_COUNT; ++mi) {
            const Mode m = static_cast<Mode>(mi);
            app.set_mode(m);
            app.request_preset(default_preset(m));
            run_frames(app, 4);
            check_layout(t, app, m, default_preset(m));
        }
        app.set_mode(Mode::REPLAY);
        run_frames(app, 2);
        const std::string at = "size " + std::to_string(size) + ": ";
        check(t, ImGui::GetStyle().FontSizeBase == static_cast<float>(size), at + "not the frames' font size");
        check_bars(t, at);
    }
    app.set_font(FontChoice());
    run_frames(app, 2);
}

// The font a settings.json holding `block` as its "font" reads as; `*ok` false when refused.
static rmwork::FontChoice font_from_json(const std::string &block, bool *ok) {
    rmwork::Settings s;
    std::string err;
    *ok = rmwork::settings_from_json("{\"font\": " + block + "}", &s, &err);
    return s.font;
}

// The font block: each weight written as its family names it and read back; an unknown family or
// weight, or a size out of range, read as the default's; a block of the wrong kind refused.
static void check_font_settings(Tally *t) {
    using namespace rmwork;
    for (int f = 0; f < FONT_FAMILY_COUNT; ++f) {
        for (FontWeight w : {FontWeight::REGULAR, FontWeight::HEAVY}) {
            Settings s = default_settings();
            s.font.family = font_family(f).id;
            s.font.weight = w;
            s.font.size = FONT_SIZE_MIN + f;
            const std::string json = settings_to_json(s);
            const std::string key = "\"weight\": \"" + font_weight_key(f, w) + "\"";
            check(t, json.find(key) != std::string::npos, std::string("settings.json does not hold ") + key);
            Settings back;
            std::string err;
            check(t, settings_from_json(json, &back, &err) && back.font == s.font,
                std::string("the font ") + font_choice_label(s.font) + " does not read back");
        }
    }
    check(t,
        font_weight_key(find_font_family("cascadia-mono"), FontWeight::HEAVY) == "semibold" &&
            font_weight_key(find_font_family("hack"), FontWeight::HEAVY) == "bold",
        "the heavier weights are not named as their families name them");
    const FontChoice def;
    check(t, def.family == "jetbrains-mono" && def.weight == FontWeight::HEAVY && def.size == 14,
        "the default font is not JetBrains Mono Medium 14");
    const struct {
            const char *block;
            const char *family;
            FontWeight weight;
            int size;
            const char *what;
    } cases[] = {
        {"{\"family\": \"inter\", \"weight\": \"regular\", \"size\": 17}", "inter", FontWeight::REGULAR, 17,
            "a valid font"},
        {"{\"family\": \"comic-sans\", \"weight\": \"regular\", \"size\": 16}", "jetbrains-mono", FontWeight::REGULAR,
            16, "an unknown family"},
        {"{\"family\": \"inter\", \"weight\": \"bold\", \"size\": 16}", "inter", FontWeight::HEAVY, 16,
            "a weight the family lacks"},
        {"{\"family\": \"hack\", \"weight\": \"medium\"}", "hack", FontWeight::HEAVY, 14, "another family's weight"},
        {"{\"size\": 10}", "jetbrains-mono", FontWeight::HEAVY, 14, "a size below the range"},
        {"{\"size\": 21}", "jetbrains-mono", FontWeight::HEAVY, 14, "a size above the range"},
        {"{\"size\": 15.5}", "jetbrains-mono", FontWeight::HEAVY, 14, "a fractional size"},
        {"{\"size\": 4294967310}", "jetbrains-mono", FontWeight::HEAVY, 14, "a size past int"},
        {"{\"family\": \"hack\", \"weight\": \"bold\", \"size\": 20}", "hack", FontWeight::HEAVY, 20,
            "a family's own heavy name"},
    };
    for (const auto &c : cases) {
        bool ok = false;
        const FontChoice got = font_from_json(c.block, &ok);
        check(t, ok && got.family == c.family && got.weight == c.weight && got.size == c.size,
            std::string("font block with ") + c.what + " reads as " + font_choice_label(got));
    }
    for (const char *bad : {"\"big\"", "{\"size\": \"14\"}", "{\"family\": 3}", "{\"weight\": true}"}) {
        bool ok = true;
        font_from_json(bad, &ok);
        check(t, !ok, std::string("font block ") + bad + " is accepted");
    }
    FontChoice odd;
    odd.family = "papyrus";
    odd.size = 99;
    check(t, sanitize_font_choice(odd) == def, "sanitize_font_choice keeps an unknown family or size");
}

// Every theme becomes current at the next frame (the ImGui and ImPlot colours its palette's) and
// draws both modes' default presets.
static void check_themes(Tally *t, rmwork::App &app) {
    using namespace rmwork;
    for (int i = 0; i < THEME_COUNT; ++i) {
        const std::string at = std::string("theme ") + theme_info(i).name + ": ";
        app.set_theme(theme_info(i).id);
        for (int mi = 0; mi < MODE_COUNT; ++mi) {
            const Mode m = static_cast<Mode>(mi);
            app.set_mode(m);
            app.request_preset(default_preset(m));
            run_frames(app, 3);
            check_layout(t, app, m, default_preset(m));
            const ImDrawData *dd = ImGui::GetDrawData();
            check(t, dd && dd->TotalVtxCount > 0, at + mode_title(m) + " draws nothing");
        }
        const Palette &p = theme_palette(i);
        const ImVec4 *c = ImGui::GetStyle().Colors;
        check(t, current_theme() == i && &tok() == &p, at + "is not the current theme");
        check(t,
            ImGui::ColorConvertFloat4ToU32(c[ImGuiCol_WindowBg]) == p.PANEL &&
                ImGui::ColorConvertFloat4ToU32(c[ImGuiCol_Text]) == p.TEXT &&
                ImGui::ColorConvertFloat4ToU32(ImPlot::GetStyle().Colors[ImPlotCol_PlotBg]) == p.PLOT_BG,
            at + "the ImGui or ImPlot colours are not the palette's");
        check(t,
            ImGui::ColorConvertFloat4ToU32(ImPlot::GetColormapColor(0, ImPlot::GetStyle().Colormap)) == p.S_A &&
                ImGui::ColorConvertFloat4ToU32(ImPlot::GetColormapColor(4, ImPlot::GetStyle().Colormap)) == p.LIFE_B,
            at + "the plot colormap is not the theme's series");
    }
    app.set_theme("no-such-theme");
    run_frames(app, 1);
    check(t, theme_info(current_theme()).id == std::string(DEFAULT_THEME), "an unknown theme is not the default");
    app.set_theme(DEFAULT_THEME);
    app.set_mode(Mode::REPLAY);
    run_frames(app, 2);
}

// `fg` against each background reaches `minimum` (a WCAG contrast ratio).
static void check_min_contrast(Tally *t, const std::string &at, const char *fg_name, ImU32 fg,
    const std::vector<std::pair<const char *, ImU32>> &backgrounds, float minimum) {
    for (const auto &bg : backgrounds) {
        const float r = rmwork::contrast_ratio(fg, bg.second);
        char buf[160];
        std::snprintf(buf, sizeof buf, "%s on %s: contrast %.2f, below %.1f", fg_name, bg.first, r, minimum);
        check(t, r >= minimum, at + buf);
    }
}

// Per theme: text on every surface it is drawn on at least 4.5:1, secondary text 3:1, the series,
// icons and accent 3:1 on what they are drawn on, and the series sharing a plot apart (ΔE ≥ 25).
static void check_theme_contrast(Tally *t) {
    using namespace rmwork;
    for (int i = 0; i < THEME_COUNT; ++i) {
        const Palette &p = theme_palette(i);
        const std::string at = std::string("theme ") + theme_info(i).name + ": ";
        check_min_contrast(t, at, "text", p.TEXT,
            {{"panel", p.PANEL}, {"alternate row", p.PANEL_2}, {"menu bar", p.MENU}, {"status bar", p.STATUS},
                {"popup", p.DROP}, {"field", p.FIELD}, {"hovered field", p.FIELD_HOVER}, {"button", p.BTN},
                {"hovered button", p.BTN_HOVER}, {"tab", p.TAB}, {"selected tab", p.TAB_ACTIVE},
                {"unfocused selected tab", p.TAB_DIM}, {"table header", p.HEADER}, {"selection", p.SEL},
                {"title bar", p.TITLE}, {"plot", p.PLOT_BG}},
            4.5f);
        check_min_contrast(t, at, "dialog title", p.TITLE_TEXT, {{"dialog title bar", p.TITLE_ACTIVE}}, 4.5f);
        check_min_contrast(t, at, "desktop text", p.BG_TEXT, {{"desktop", p.BG}}, 4.5f);
        check_min_contrast(t, at, "secondary text", p.DIM,
            {{"panel", p.PANEL}, {"menu bar", p.MENU}, {"status bar", p.STATUS}, {"popup", p.DROP}, {"field", p.FIELD},
                {"plot", p.PLOT_BG}},
            3.0f);
        const std::vector<std::pair<const char *, ImU32>> series = {{"V (A)", p.S_A}, {"V (A, re-eval)", p.S_A6},
            {"-V of B", p.S_B}, {"life A", p.LIFE_A}, {"life B", p.LIFE_B}};
        for (const auto &s : series)
            check_min_contrast(t, at, s.first, s.second, {{"plot", p.PLOT_BG}, {"panel", p.PANEL}}, 3.0f);
        for (const auto &icon : std::vector<std::pair<const char *, ImU32>>{{"play icon", p.GOOD},
                 {"swing icon", p.BAD}, {"open icon", p.FOLDER}, {"save icon", p.S_A}, {"nav icon", p.TEXT}})
            check_min_contrast(t, at, icon.first, icon.second, {{"toolbar", p.MENU}}, 3.0f);
        check_min_contrast(t, at, "accent", p.ACCENT, {{"panel", p.PANEL}, {"field", p.FIELD}}, 3.0f);
        check_min_contrast(t, at, "logo letter", p.LOGO_FG, {{"logo", p.LOGO_BG}}, 3.0f);
        const std::vector<std::vector<std::pair<const char *, ImU32>>> plots = {
            {series[0], series[1], series[2]}, {series[3], series[4]}};
        for (const auto &plot : plots) {
            for (size_t a = 0; a < plot.size(); ++a) {
                for (size_t b = a + 1; b < plot.size(); ++b) {
                    const float d = color_difference(plot[a].second, plot[b].second);
                    char buf[160];
                    std::snprintf(
                        buf, sizeof buf, "%s and %s differ by dE %.1f, below 25", plot[a].first, plot[b].first, d);
                    check(t, d >= 25.0f, at + buf);
                }
            }
        }
    }
}

// The theme in settings.json: each one written and read back, an unknown one read as the default,
// one of the wrong kind refused.
static void check_theme_settings(Tally *t) {
    using namespace rmwork;
    for (int i = 0; i < THEME_COUNT; ++i) {
        Settings s = default_settings();
        s.theme = theme_info(i).id;
        const std::string json = settings_to_json(s);
        Settings back;
        std::string err;
        check(t,
            json.find(std::string("\"theme\": \"") + theme_info(i).id + "\"") != std::string::npos &&
                settings_from_json(json, &back, &err) && back.theme == s.theme,
            std::string("theme ") + theme_info(i).id + " does not round-trip");
    }
    Settings back;
    std::string err;
    check(t, settings_from_json("{\"theme\": \"vaporwave\"}", &back, &err) && back.theme == DEFAULT_THEME,
        "an unknown theme does not read as the default");
    check(t, !settings_from_json("{\"theme\": 3}", &back, &err), "a theme that is not a string is accepted");
    check(t, default_settings().theme == "imhex-dark", "the default theme is not ImHex Dark");
}

// The layout saved to an ini and loaded into a fresh context is the one shown (no default preset).
static void check_ini_round_trip(Tally *t, const rmwork::ConfigPaths &paths) {
    using namespace rmwork;
    Settings settings = default_settings();
    std::string ini;
    {
        App app(settings, paths, false);
        run_frames(app, 3);
        app.request_preset(find_preset("analysis"));
        run_frames(app, 3);
        app.set_mode(Mode::PLAY);
        app.request_preset(find_preset("replay"));
        run_frames(app, 3);
        settings = app.settings();
        ini = ImGui::SaveIniSettingsToMemory();
    }
    destroy_ui_context();
    begin_headless();
    ImGui::LoadIniSettingsFromMemory(ini.c_str(), ini.size());
    App app(settings, paths, false);
    run_frames(app, 3);
    check_layout(t, app, Mode::PLAY, find_preset("replay"));
    app.set_mode(Mode::REPLAY);
    run_frames(app, 3);
    check_layout(t, app, Mode::REPLAY, find_preset("analysis"));
}

// settings.json: written and read back equal, defaults for a missing file or key, malformed text
// refused, every default binding's name parsed back to it.
static void check_settings_file(Tally *t, const rmwork::ConfigPaths &paths, const std::string &tmp) {
    using namespace rmwork;
    check(t, paths.settings == tmp + "/robomage/settings.json", "settings path is not under $XDG_CONFIG_HOME");
    check(t, paths.imgui_ini == tmp + "/robomage/workbench/imgui.ini", "imgui.ini path is not under $XDG_CONFIG_HOME");
    std::string err;
    Settings s;
    check(t, load_settings(paths.settings + ".missing", &s, &err), "a missing settings file is an error");
    check(
        t, settings_to_json(s) == settings_to_json(default_settings()), "a missing settings file is not the defaults");
    s.mode = Mode::PLAY;
    s.tools[mode_index(Mode::PLAY)][tool_index(Tool::SWINGS)] = true;
    s.tools[mode_index(Mode::REPLAY)][tool_index(Tool::TABLE)] = false;
    s.table = {TableBackground::Kind::IMAGE, "/pictures/table \"oak\".png"};
    s.keys[action_index(Action::REPLAY_NEXT)] = ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_L;
    s.keys[action_index(Action::MAGNIFY)] = 0;
    check(t, save_settings(paths.settings, s, &err), "save_settings: " + err);
    Settings back;
    check(t, load_settings(paths.settings, &back, &err), "load_settings: " + err);
    check(t, settings_to_json(back) == settings_to_json(s), "settings.json does not read back as written");
    write_file_atomic(paths.settings, "{\"mode\": \"play\", \"from_the_future\": [1, 2]}", &err);
    check(t,
        load_settings(paths.settings, &back, &err) && back.mode == Mode::PLAY && back.keys == default_bindings() &&
            back.tools == default_settings().tools,
        "unknown keys are not ignored, or missing ones not defaulted");
    write_file_atomic(paths.settings, "{\"mode\": ", &err);
    err.clear();
    check(t, !load_settings(paths.settings, &back, &err) && !err.empty(), "malformed settings.json is accepted");
    check(t, settings_to_json(back) == settings_to_json(default_settings()),
        "malformed settings.json leaves non-defaults");
    write_file_atomic(paths.settings, "{\"keys\": {\"replay.next\": \"Hyper+Q\"}}", &err);
    check(t, !load_settings(paths.settings, &back, &err), "an unknown key name is accepted");
    for (int a = 0; a < ACTION_COUNT; ++a) {
        const ImGuiKeyChord c = default_bindings()[a];
        ImGuiKeyChord parsed = 0;
        check(t, parse_chord(chord_name(c), &parsed) && parsed == c,
            std::string("binding ") + action_info(static_cast<Action>(a)).key + " does not round-trip");
    }
    std::remove(paths.settings.c_str());
}

// An App that persists writes the settings file when a setting changes, and not before.
static void check_app_persists(Tally *t, const rmwork::ConfigPaths &paths) {
    using namespace rmwork;
    std::remove(paths.settings.c_str());
    App app(default_settings(), paths, true);
    run_frames(app, 3);
    check(t, !file_exists(paths.settings), "settings.json written though nothing changed");
    app.set_mode(Mode::PLAY);
    app.set_tool_open(Mode::PLAY, Tool::SWINGS, true);
    run_frames(app, 2);
    Settings back;
    std::string err;
    check(t,
        load_settings(paths.settings, &back, &err) && back.mode == Mode::PLAY &&
            back.tools[mode_index(Mode::PLAY)][tool_index(Tool::SWINGS)],
        "a changed setting is not written to settings.json");
    FontChoice font;
    font.family = "ibm-plex-sans";
    font.weight = FontWeight::REGULAR;
    font.size = 18;
    app.set_font(font);
    run_frames(app, 1);
    check(t, load_settings(paths.settings, &back, &err) && back.font == font,
        "a changed font is not written to settings.json");
    app.set_theme("solarized-light");
    run_frames(app, 1);
    check(t, load_settings(paths.settings, &back, &err) && back.theme == "solarized-light",
        "a changed theme is not written to settings.json");
    check(t,
        ImGui::GetStyle().FontSizeBase == 18.0f &&
            ImGui::GetIO().FontDefault == ui_font(find_font_family("ibm-plex-sans"), FontWeight::REGULAR),
        "a changed font is not applied at the next frame");
}

namespace rmwork {

int run_selftest() {
    Tally t;
    const std::string tmp = make_temp_dir("rmwork_selftest_");
    if (tmp.empty()) {
        std::fprintf(stderr, "rmwork: cannot make a temporary directory\n");
        return 1;
    }
    setenv("XDG_CONFIG_HOME", tmp.c_str(), 1);
    const ConfigPaths paths = config_paths("");

    begin_headless();
    {
        App app(default_settings(), paths, false);
        run_frames(app, 3);
        check_layout(&t, app, Mode::REPLAY, default_preset(Mode::REPLAY));
        app.set_mode(Mode::PLAY);
        run_frames(app, 3);
        check_layout(&t, app, Mode::PLAY, default_preset(Mode::PLAY));
        check_presets(&t, app);
        check_mode_switching(&t, app);
        check_reopen(&t, app);
        check_keys(&t, app);
        check_fonts(&t, app);
        check_font_sizes(&t, app);
        check_themes(&t, app);
        app.open_settings(SettingsDialog::APPEARANCE);
        run_frames(app, 3);
        check(&t, app.settings_open(), "Settings does not open at Appearance");
    }
    check_ini_round_trip(&t, paths);
    check_app_persists(&t, paths);
    destroy_ui_context();
    check_settings_file(&t, paths, tmp);
    check_font_settings(&t);
    check_theme_settings(&t);
    begin_headless();
    check_theme_contrast(&t);
    destroy_ui_context();

    remove_tree(tmp);
    std::printf("RMWORK-SELFTEST: %d checks, %d failures\n", t.checks, t.failures);
    return t.failures == 0 ? 0 : 1;
}

}  // namespace rmwork
