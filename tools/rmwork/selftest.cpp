#include "selftest.h"

#include <sys/socket.h>
#include <sys/un.h>

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
#include "paths.h"
#include "selftest_table.h"
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
static void check_layout(Tally *t, const rmwork::App &app, rmwork::Mode m, int preset);
static void check_layout_tree(
    Tally *t, const rmwork::App &app, rmwork::Mode m, const rmwork::LayoutNode &layout, const std::string &label);
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
static void check_preset_capture(Tally *t, rmwork::App &app);
static rmwork::LayoutNode custom_layout();
static void check_named_layouts(Tally *t, const rmwork::ConfigPaths &paths);
static void check_layout_names(Tally *t);
static bool layouts_json_accepted(const std::string &layouts);
static void check_layout_files(Tally *t);
static void check_layout_dialogs(Tally *t, const rmwork::ConfigPaths &paths);
static void check_later_settings(Tally *t, const std::string &tmp);
static std::string make_socket(const std::string &path);
static void check_health(Tally *t, const rmwork::ConfigPaths &paths, const std::string &tmp);
static void check_rebinding(Tally *t, const rmwork::ConfigPaths &paths);
static void check_key_resets(Tally *t);
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

// The last frame shows preset `preset` in mode `m` (check_layout_tree).
static void check_layout(Tally *t, const rmwork::App &app, rmwork::Mode m, int preset) {
    const rmwork::Preset &p = rmwork::presets()[static_cast<size_t>(preset)];
    check_layout_tree(t, app, m, p.root, p.name);
}

// The last frame shows exactly the layout's tools in mode `m`, each group in one dock node of the
// mode's dockspace, in order, its selected tab shown, at the layout's split ratios.
static void check_layout_tree(
    Tally *t, const rmwork::App &app, rmwork::Mode m, const rmwork::LayoutNode &layout, const std::string &label) {
    using namespace rmwork;
    const std::string at = std::string(mode_title(m)) + " / " + label + ": ";
    const ToolFlags expected = layout_tools(layout);
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
    const std::vector<const LayoutNode *> groups = layout_groups(layout);
    const std::vector<std::pair<ImVec2, ImVec2>> rects = layout_group_rects(layout, root->Pos, root->Size);
    std::vector<const ImGuiDockNode *> nodes;
    for (size_t g = 0; g < groups.size(); ++g) {
        const std::vector<Tool> &tools = groups[g]->tools;
        const ImGuiWindow *first = tool_window(m, tools.front());
        const ImGuiWindow *shown = tool_window(m, tools[static_cast<size_t>(groups[g]->selected)]);
        const ImGuiDockNode *node = first ? first->DockNode : nullptr;
        const std::string gname = at + "group of " + tool_info(tools.front()).title;
        check(t, node != nullptr, gname + " is not docked");
        if (!node) continue;
        check(t, ImGui::DockNodeGetRootNode(const_cast<ImGuiDockNode *>(node))->ID == root->ID,
            gname + " is docked outside the mode's dockspace");
        for (const ImGuiDockNode *other : nodes) check(t, other != node, gname + " shares a node with another group");
        nodes.push_back(node);
        std::vector<const ImGuiWindow *> want;
        for (Tool tool : tools) {
            const ImGuiWindow *w = tool_window(m, tool);
            want.push_back(w);
            check(t, w && w->DockNode == node, at + tool_info(tool).title + " is not in its group's tab node");
        }
        std::vector<const ImGuiWindow *> tabs;
        if (node->TabBar)
            for (const ImGuiTabItem &tab : node->TabBar->Tabs)
                if (tab.Window && window_active(tab.Window)) tabs.push_back(tab.Window);
        check(t, tabs == want, gname + ": tabs not in the layout's order:" + tab_names(node));
        check(t, node->VisibleWindow == shown, gname + ": the selected tab is not the one shown");
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

// The bindings act in their mode only and write a status message without a game; Ctrl+O opens the
// file picker and Ctrl+, Settings, Esc closes them and the error dialog.
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
    run_frames(app, 1);
    check(t,
        app.file_picker().is_open() && app.file_picker().purpose() == App::PICK_REPLAY &&
            app.file_picker().filter() == FileFilter::DECISION_LOGS,
        "Ctrl+O does not open the file picker for a decision log");
    press(app, ImGuiKey_Escape);
    check(t, !app.file_picker().is_open(), "Esc does not close the file picker");
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
        check_min_contrast(t, at, "text", p.TEXT,
            {{"seat A chip", p.SEAT_A}, {"seat B chip", p.SEAT_B}, {"counter", p.COUNTER}}, 4.5f);
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

// Each preset built in each mode reads back (capture_layout) as itself: same-direction splits
// flattened, the same groups, tabs, selected tabs and ratios.
static void check_preset_capture(Tally *t, rmwork::App &app) {
    using namespace rmwork;
    for (int mi = 0; mi < MODE_COUNT; ++mi) {
        const Mode m = static_cast<Mode>(mi);
        app.set_mode(m);
        for (size_t p = 0; p < presets().size(); ++p) {
            app.request_preset(static_cast<int>(p));
            run_frames(app, 4);
            const LayoutNode got = capture_layout(m, app.settings().tools[mi]);
            check(t, layouts_match(got, presets()[p].root, 0.01f),
                std::string(mode_title(m)) + " / " + presets()[p].name + " reads back as " + describe_layout(got) +
                    ", not " + describe_layout(presets()[p].root));
        }
    }
    app.set_mode(Mode::REPLAY);
    app.request_preset(default_preset(Mode::REPLAY));
    run_frames(app, 3);
}

// Two groups with an unusual split and a tab other than the first shown in each.
static rmwork::LayoutNode custom_layout() {
    using namespace rmwork;
    LayoutNode left;
    left.tools = {Tool::PLAYERS, Tool::MODEL};
    left.selected = 1;
    LayoutNode top;
    top.tools = {Tool::TABLE};
    LayoutNode bottom;
    bottom.tools = {Tool::LOG, Tool::TREE, Tool::CHOICES};
    bottom.selected = 2;
    LayoutNode right;
    right.kind = LayoutNode::Kind::COL;
    right.sizes = {0.6f, 0.4f};
    right.children = {top, bottom};
    LayoutNode root;
    root.kind = LayoutNode::Kind::ROW;
    root.sizes = {0.25f, 0.75f};
    root.children = {left, right};
    return root;
}

// A saved layout survives settings.json, rebuilds exactly (groups, tab order, the shown tabs,
// ratios) and reads back as itself; the live dockspace saved under a name comes back after another
// layout; a name is taken only with overwrite; the modes keep their own lists.
static void check_named_layouts(Tally *t, const rmwork::ConfigPaths &paths) {
    using namespace rmwork;
    Settings s = default_settings();
    const LayoutNode custom = custom_layout();
    s.layouts[mode_index(Mode::REPLAY)].push_back({"Custom", custom});
    Settings back;
    std::string err;
    check(t, settings_from_json(settings_to_json(s), &back, &err) && settings_to_json(back) == settings_to_json(s),
        "a saved layout does not survive settings.json: " + err);
    check(t,
        back.layouts[mode_index(Mode::REPLAY)].size() == 1 &&
            layouts_match(back.layouts[mode_index(Mode::REPLAY)][0].root, custom, 0.0001f),
        "a saved layout reads back changed");
    App app(back, paths, false);
    run_frames(app, 3);
    check(t, app.request_named_layout("custom"), "a saved layout is not found by name, case aside");
    check(t, !app.request_named_layout("nothing"), "a missing saved layout is found");
    run_frames(app, 4);
    check_layout_tree(t, app, Mode::REPLAY, custom, "Custom");
    const LayoutNode live = capture_layout(Mode::REPLAY, app.settings().tools[mode_index(Mode::REPLAY)]);
    check(t, layouts_match(live, custom, 0.01f),
        "the rebuilt saved layout reads back as " + describe_layout(live) + ", not " + describe_layout(custom));

    // The dockspace as it is now: a tool closed, another tab shown.
    app.set_tool_open(Mode::REPLAY, Tool::MODEL, false);
    run_frames(app, 2);
    ImGui::SetWindowFocus(tool_window_name(Mode::REPLAY, Tool::TREE).c_str());
    run_frames(app, 2);
    std::string why;
    check(t, app.save_current_layout("Mine", false, &why) == SaveResult::SAVED, "saving the current layout: " + why);
    const LayoutNode mine = app.settings().layouts[mode_index(Mode::REPLAY)].back().root;
    const std::vector<const LayoutNode *> groups = layout_groups(mine);
    check(t,
        groups.size() == 3 && groups[0]->tools == std::vector<Tool>{Tool::PLAYERS} &&
            groups[2]->tools[static_cast<size_t>(groups[2]->selected)] == Tool::TREE,
        "the saved current layout is " + describe_layout(mine) + " (Model's Read closed, State Tree shown expected)");
    app.request_preset(find_preset("analysis"));
    run_frames(app, 3);
    check(t, app.request_named_layout("Mine"), "the saved current layout is not found");
    run_frames(app, 4);
    check_layout_tree(t, app, Mode::REPLAY, mine, "Mine");
    check(t, layouts_match(capture_layout(Mode::REPLAY, app.settings().tools[mode_index(Mode::REPLAY)]), mine, 0.01f),
        "the saved current layout does not rebuild as saved");

    check(t, app.save_current_layout("MINE", false, &why) == SaveResult::EXISTS && !why.empty(),
        "saving over a layout of the same name does not ask");
    check(t,
        app.save_current_layout("MINE", true, &why) == SaveResult::SAVED &&
            app.settings().layouts[mode_index(Mode::REPLAY)].size() == 2 &&
            app.settings().layouts[mode_index(Mode::REPLAY)][1].name == "MINE",
        "overwriting a layout does not replace it in place under the new spelling");
    check(t, app.save_current_layout("Analysis", false, &why) == SaveResult::INVALID, "a preset's name is accepted");
    app.set_mode(Mode::PLAY);
    run_frames(app, 3);
    check(t, app.settings().layouts[mode_index(Mode::PLAY)].empty() && !app.request_named_layout("Custom"),
        "Play sees Replay's saved layouts");
    check(t, app.save_current_layout("Custom", false, &why) == SaveResult::SAVED,
        "Play cannot save a name Replay uses: " + why);
    check(t, app.settings().layouts[mode_index(Mode::REPLAY)].size() == 2, "saving in Play changed Replay's layouts");
}

// Names: rename and delete, and the names refused.
static void check_layout_names(Tally *t) {
    using namespace rmwork;
    NamedLayouts all = {{"One", custom_layout()}, {"Two", custom_layout()}};
    std::string why;
    check(t, !rename_named_layout(&all, "One", "two", &why) && !why.empty(), "a rename onto a taken name is accepted");
    check(t, rename_named_layout(&all, "One", "one", &why) && all[0].name == "one", "a rename of case only is refused");
    check(t, rename_named_layout(&all, "one", "Three", &why) && find_named_layout(all, "three") == 0,
        "a rename is refused: " + why);
    check(t, !rename_named_layout(&all, "Nine", "Ten", &why), "renaming a missing layout succeeds");
    check(t, delete_named_layout(&all, "TWO") && all.size() == 1 && !delete_named_layout(&all, "Two"),
        "delete does not remove exactly the named layout");
    for (const char *bad :
        {"", "   ", "Analysis", "table", "tab\there", "0123456789012345678901234567890123456789012345678"})
        check(t, !layout_name_valid(bad, &why), std::string("the layout name \"") + bad + "\" is accepted");
    check(t, layout_name_valid("My analysis (wide)", &why), "a plain layout name is refused: " + why);
}

// The settings with `layouts` as their "layouts" value load.
static bool layouts_json_accepted(const std::string &layouts) {
    rmwork::Settings s;
    std::string err;
    return rmwork::settings_from_json("{\"layouts\": " + layouts + "}", &s, &err);
}

// Malformed saved layouts refuse the settings file.
static void check_layout_files(Tally *t) {
    const std::string group = "{\"tabs\": [\"table\"], \"selected\": 0}";
    check(t, layouts_json_accepted("{\"replay\": {\"A\": " + group + "}, \"play\": {}}"),
        "a one-group layout is refused");
    const struct {
            const char *what;
            std::string json;
    } bad[] = {
        {"an unknown tool", "{\"replay\": {\"A\": {\"tabs\": [\"teapot\"]}}}"},
        {"a tool twice", "{\"replay\": {\"A\": {\"split\": \"row\", \"sizes\": [1, 1], \"children\": [" + group + ", " +
                             group + "]}}}"},
        {"a one-child split",
            "{\"replay\": {\"A\": {\"split\": \"col\", \"sizes\": [1], \"children\": [" + group + "]}}}"},
        {"sizes not matching children", "{\"replay\": {\"A\": {\"split\": \"row\", \"sizes\": [1], \"children\": [" +
                                            group + ", {\"tabs\": [\"log\"]}]}}}"},
        {"a negative size", "{\"replay\": {\"A\": {\"split\": \"row\", \"sizes\": [1, -1], \"children\": [" + group +
                                ", {\"tabs\": [\"log\"]}]}}}"},
        {"an unknown split", "{\"replay\": {\"A\": {\"split\": \"diagonal\", \"sizes\": [1, 1], \"children\": []}}}"},
        {"a selected tab out of range", "{\"replay\": {\"A\": {\"tabs\": [\"table\"], \"selected\": 3}}}"},
        {"a preset's name", "{\"play\": {\"Analysis\": " + group + "}}"},
        {"a name twice", "{\"play\": {\"A\": " + group + ", \"a\": " + group + "}}"},
        {"a list for a mode", "{\"play\": []}"},
    };
    for (const auto &b : bad)
        check(t, !layouts_json_accepted(b.json), std::string("a layout with ") + b.what + " is accepted");
}

// Save Layout As: Enter saves under the name typed and closes; the same name again asks to
// overwrite and changes nothing until it is answered; Manage Layouts opens.
static void check_layout_dialogs(Tally *t, const rmwork::ConfigPaths &paths) {
    using namespace rmwork;
    App app(default_settings(), paths, false);
    app.request_preset(default_preset(Mode::REPLAY));
    run_frames(app, 4);
    app.layout_dialogs().open_save(Mode::REPLAY);
    run_frames(app, 2);
    check(t, app.layout_dialogs().save_open(), "Save Layout As does not open");
    ImGui::GetIO().AddInputCharactersUTF8("Wide table");
    run_frames(app, 1);
    press(app, ImGuiKey_Enter);
    const NamedLayouts &saved = app.settings().layouts[mode_index(Mode::REPLAY)];
    check(t, saved.size() == 1 && saved[0].name == "Wide table" && !app.layout_dialogs().save_open(),
        "Enter in Save Layout As does not save and close");
    check(t, saved.size() == 1 && layouts_match(saved[0].root, presets()[default_preset(Mode::REPLAY)].root, 0.01f),
        "Save Layout As does not save the dockspace as it is: " +
            (saved.empty() ? std::string("nothing") : describe_layout(saved[0].root)));
    app.layout_dialogs().open_save(Mode::REPLAY);
    run_frames(app, 2);
    ImGui::GetIO().AddInputCharactersUTF8("wide TABLE");
    run_frames(app, 1);
    press(app, ImGuiKey_Enter);
    check(t, app.layout_dialogs().save_open() && saved.size() == 1 && saved[0].name == "Wide table",
        "Save Layout As over a taken name does not stop to ask");
    press(app, ImGuiKey_Escape);
    check(t, !app.layout_dialogs().save_open() && saved.size() == 1 && saved[0].name == "Wide table",
        "Esc does not cancel the overwrite");
    app.layout_dialogs().open_manage(Mode::REPLAY);
    run_frames(app, 2);
    check(t, app.layout_dialogs().manage_open(), "Manage Layouts does not open");
    press(app, ImGuiKey_Escape);
    check(t, !app.layout_dialogs().manage_open(), "Esc does not close Manage Layouts");
}

// The later stages' settings: defaults from the install, path resolution, a round trip of every
// field, and the refusals.
static void check_later_settings(Tally *t, const std::string &tmp) {
    using namespace rmwork;
    const Settings def = default_settings();
    const std::string bin = repo_root() + "/bin/" + build_config() + "/";
    check(t, parent_dir(executable_path()) + "/" == bin, "the repository is not found from the rmwork binary: " + bin);
    const std::string engine = engine_binary_path(def);
    check(t, engine == bin + "robomage_nn" || engine == bin + "robomage", "the default engine binary is " + engine);
    check(t, engine == bin + "robomage" || is_executable(bin + "robomage_nn"),
        "the default engine is robomage_nn though there is none");
    check(t, engine_workdir_path(def) == repo_root() + "/bin" && run_dir_path(def) == repo_root() + "/runs",
        "the default working directory or run directory is wrong");
    check(t, socket_path(def).empty(), "a socket is configured by default");
    const char *old_cache = std::getenv("XDG_CACHE_HOME");
    const std::string saved_cache = old_cache ? old_cache : "";
    setenv("XDG_CACHE_HOME", tmp.c_str(), 1);
    check(t, image_cache_path(def) == tmp + "/robomage/scryfall", "the image cache ignores $XDG_CACHE_HOME");
    unsetenv("XDG_CACHE_HOME");
    check(t, image_cache_path(def) == std::string(std::getenv("HOME")) + "/.cache/robomage/scryfall",
        "the image cache is not under ~/.cache without $XDG_CACHE_HOME");
    if (old_cache) setenv("XDG_CACHE_HOME", saved_cache.c_str(), 1);
    check(t,
        resolve_path("~/m.pt2", "x") == std::string(std::getenv("HOME")) + "/m.pt2" &&
            resolve_path("runs/a", "x") == repo_root() + "/runs/a" && resolve_path("/abs", "x") == "/abs" &&
            resolve_path("", "fallback") == "fallback",
        "resolve_path does not resolve ~/, relative and absolute paths and the default");
    const StopSettings stops = default_stops();
    int own = 0;
    for (int i = 0; i < STOP_STEP_COUNT; ++i) own += stops.own[static_cast<size_t>(i)] ? 1 : 0;
    check(t,
        own == 5 && stops.own[1] && stops.own[3] && stops.own[5] && stops.own[6] && stops.own[10] &&
            stops.opponent == stops.own && def.stops.own == stops.own,
        "the default stops are not upkeep, main 1, declare attackers, declare blockers and end step");

    Settings s = def;
    s.engine = {"bin/release/robomage", "/tmp/work"};
    s.models.socket = "~/rmserve.sock";
    s.models.run_dir = "runs/league";
    std::string why;
    check(t, add_policy(&s.models, "main", "runs/a/model.pt2", &why) && add_policy(&s.models, "b.2-x_", "/m.pt", &why),
        "add_policy refuses a valid row: " + why);
    check(t,
        !add_policy(&s.models, "main", "x", &why) && !add_policy(&s.models, "a=b", "x", &why) &&
            !add_policy(&s.models, "", "x", &why) && !add_policy(&s.models, "c", "", &why) &&
            s.models.policies.size() == 2,
        "add_policy accepts a taken, malformed or empty id or an empty model");
    s.stops.own.fill(false);
    s.stops.opponent[0] = s.stops.opponent[11] = true;
    s.image_cache = "/cache/images";
    Settings back;
    std::string err;
    check(t, settings_from_json(settings_to_json(s), &back, &err) && settings_to_json(back) == settings_to_json(s),
        "the engine, models, stops and image settings do not round-trip: " + err);
    check(t,
        back.models.policies.size() == 2 && back.models.policies[1].id == "b.2-x_" && !back.stops.own[1] &&
            back.stops.opponent[11] && back.engine.workdir == "/tmp/work",
        "the read-back later-stage settings differ");
    const char *refused[] = {"{\"stops\": {\"own\": [\"upkeep\", \"second_breakfast\"]}}",
        "{\"stops\": {\"own\": \"upkeep\"}}", "{\"stops\": []}", "{\"engine\": {\"binary\": 3}}",
        "{\"engine\": \"robomage\"}", "{\"models\": {\"socket\": []}}",
        "{\"models\": {\"policies\": [{\"id\": \"a\", \"model\": \"x\"}, {\"id\": \"a\", \"model\": \"y\"}]}}",
        "{\"models\": {\"policies\": [{\"id\": \"a b\", \"model\": \"x\"}]}}",
        "{\"models\": {\"policies\": [{\"id\": \"a\"}]}}", "{\"images\": {\"cache\": false}}",
        "{\"keys\": {\"replay.next\": \"Ctrl+O\"}}"};
    for (const char *text : refused) check(t, !settings_from_json(text, &back, &err), std::string("accepted: ") + text);
    check(t, settings_from_json("{\"keys\": {\"play.skip_turn\": \"RightArrow\"}}", &back, &err),
        "a Play key equal to a Replay key is refused: " + err);
}

// A listening UNIX socket at `path` (its descriptor kept open); "" or the error.
static std::string make_socket(const std::string &path) {
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return "socket() failed";
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof addr.sun_path, "%s", path.c_str());
    if (bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof addr) != 0) return "bind() failed";
    return "";
}

// The status bar says whether the configured engine binary exists and whether the socket is there.
static void check_health(Tally *t, const rmwork::ConfigPaths &paths, const std::string &tmp) {
    using namespace rmwork;
    Settings s = default_settings();
    s.engine.binary = tmp + "/no-engine";
    s.models.socket = tmp + "/no.sock";
    {
        App app(s, paths, false);
        run_frames(app, 1);
        check(t, app.engine_health() == "Engine: missing" && app.inference_health() == "Inference: not connected",
            "health with a missing engine and socket: " + app.engine_health() + ", " + app.inference_health());
    }
    const std::string sock = tmp + "/rm.sock";
    const std::string made = make_socket(sock);
    check(t, made.empty(), "cannot make a test socket: " + made);
    s.engine.binary = "/bin/sh";
    s.models.socket = sock;
    App app(s, paths, false);
    run_frames(app, 1);
    check(t, app.engine_health() == "Engine: found" && app.inference_health() == "Inference: socket present",
        "health with an engine and a socket: " + app.engine_health() + ", " + app.inference_health());
}

// Settings → Keys: a captured chord rebinds, one taken where both work is refused with a message,
// Esc cancels the capture without closing the dialog.
static void check_rebinding(Tally *t, const rmwork::ConfigPaths &paths) {
    using namespace rmwork;
    KeyBindings b = default_bindings();
    std::string why;
    check(t,
        !rebind(&b, Action::REPLAY_NEXT, ImGuiMod_Ctrl | ImGuiKey_O, &why) &&
            why.find("Open Replay") != std::string::npos && b == default_bindings(),
        "a Replay key taken by an Anywhere action is not refused: " + why);
    check(t, rebind(&b, Action::PLAY_SKIP_TURN, ImGuiKey_RightArrow, &why),
        "a Play key equal to a Replay key is refused");
    check(t, !rebind(&b, Action::MAGNIFY, ImGuiKey_F2, &why), "an Anywhere key taken by a Play action is accepted");

    App app(default_settings(), paths, false);
    run_frames(app, 2);
    app.begin_key_capture(Action::REPLAY_NEXT);
    run_frames(app, 2);
    check(t, app.settings_open() && app.settings_dialog().capturing(), "Keys does not wait for a key");
    press(app, ImGuiMod_Ctrl | ImGuiKey_L);
    const ImGuiKeyChord want = ImGuiMod_Ctrl | ImGuiKey_L;
    check(t, app.settings().keys[action_index(Action::REPLAY_NEXT)] == want && !app.settings_dialog().capturing(),
        "a captured Ctrl+L is not Next decision's key");
    app.begin_key_capture(Action::REPLAY_PREV);
    run_frames(app, 1);
    press(app, ImGuiMod_Ctrl | ImGuiKey_O);
    check(t,
        app.settings().keys[action_index(Action::REPLAY_PREV)] == ImGuiKey_LeftArrow &&
            app.settings_dialog().key_message_is_error() &&
            app.settings_dialog().key_message().find("Open Replay") != std::string::npos,
        "Ctrl+O, Open Replay's key, is not refused for Previous decision: " + app.settings_dialog().key_message());
    app.begin_key_capture(Action::REPLAY_PREV);
    run_frames(app, 1);
    press(app, ImGuiKey_Escape);
    check(t,
        app.settings_open() && !app.settings_dialog().capturing() &&
            app.settings().keys[action_index(Action::REPLAY_PREV)] == ImGuiKey_LeftArrow,
        "Esc does not cancel a capture, or closes Settings");
    press(app, ImGuiKey_Escape);
    check(t, !app.settings_open(), "Esc after a capture does not close Settings");
    app.set_mode(Mode::REPLAY);
    run_frames(app, 1);
    app.status("");
    press(app, ImGuiMod_Ctrl | ImGuiKey_L);
    check(t, app.status_message() == "No game loaded: Next decision", "the rebound Ctrl+L does not act");
}

// A row's Reset and Reset All; a reset onto a key another action took is refused.
static void check_key_resets(Tally *t) {
    using namespace rmwork;
    Settings s = default_settings();
    SettingsDialog d;
    std::string why;
    check(t,
        rebind(&s.keys, Action::REPLAY_FIRST, ImGuiMod_Ctrl | ImGuiKey_L, &why) &&
            rebind(&s.keys, Action::MAGNIFY, ImGuiKey_Home, &why),
        "rebinding First decision then Magnify onto Home: " + why);
    check(t,
        !d.reset_key(&s, Action::REPLAY_FIRST) && d.key_message_is_error() &&
            s.keys[action_index(Action::REPLAY_FIRST)] == (ImGuiMod_Ctrl | ImGuiKey_L),
        "resetting First decision onto Magnify's Home is not refused");
    check(t, d.reset_key(&s, Action::MAGNIFY) && s.keys[action_index(Action::MAGNIFY)] == ImGuiKey_Q,
        "Magnify's Reset does not restore Q");
    check(t, d.reset_key(&s, Action::REPLAY_FIRST) && s.keys[action_index(Action::REPLAY_FIRST)] == ImGuiKey_Home,
        "First decision's Reset does not restore Home");
    rebind(&s.keys, Action::FIND_IN_LOG, ImGuiMod_Ctrl | ImGuiKey_G, &why);
    d.reset_all_keys(&s);
    check(t, s.keys == default_bindings(), "Reset All does not restore every key");
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
        check_preset_capture(&t, app);
        app.open_settings(SettingsDialog::APPEARANCE);
        run_frames(app, 3);
        check(&t, app.settings_open(), "Settings does not open at Appearance");
    }
    check_named_layouts(&t, paths);
    check_layout_dialogs(&t, paths);
    check_health(&t, paths, tmp);
    check_rebinding(&t, paths);
    check_ini_round_trip(&t, paths);
    check_app_persists(&t, paths);
    destroy_ui_context();
    check_settings_file(&t, paths, tmp);
    check_font_settings(&t);
    check_theme_settings(&t);
    check_layout_names(&t);
    check_layout_files(&t);
    check_later_settings(&t, tmp);
    check_key_resets(&t);
    check_table_units([&t](bool ok, const std::string &what) { check(&t, ok, "Table: " + what); });
    begin_headless();
    check_theme_contrast(&t);
    destroy_ui_context();

    remove_tree(tmp);
    std::printf("RMWORK-SELFTEST: %d checks, %d failures\n", t.checks, t.failures);
    return t.failures == 0 ? 0 : 1;
}

}  // namespace rmwork
