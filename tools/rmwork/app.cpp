#include "app.h"

#include <algorithm>

#include "files.h"
#include "imgui_internal.h"
#include "panels.h"
#include "theme.h"
#include "widgets.h"

namespace {

const char *const VIEW_NAMES[] = {"seat A", "seat B", "omniscient"};
// The toolbar's game combo box, in font heights (300 px at the default 14).
const float GAME_COMBO_EMS = 21.5f;

}  // namespace

namespace rmwork {

App::App(const Settings &settings, const ConfigPaths &paths, bool persist)
    : settings_(settings), paths_(paths), persist_(persist), saved_json_(settings_to_json(settings)) {
    for (int m = 0; m < MODE_COUNT; ++m) {
        for (int t = 0; t < TOOL_COUNT; ++t) {
            window_names_[m][t] = tool_window_name(static_cast<Mode>(m), static_cast<Tool>(t));
            submit_order_[m].push_back(static_cast<Tool>(t));
        }
    }
    status_ =
        "Drag a tab to move a tool; drag the gaps between tools to resize them. View lists every tool and the "
        "preset layouts.";
}

void App::begin_frame() {
    check_health();
    if (find_theme(settings_.theme) < 0) settings_.theme = DEFAULT_THEME;
    if (applied_theme_ != settings_.theme) {
        apply_theme(settings_.theme);
        applied_theme_ = settings_.theme;
    }
    settings_.font = sanitize_font_choice(settings_.font);
    if (font_applied_ && applied_font_ == settings_.font) return;
    apply_font(settings_.font);
    applied_font_ = settings_.font;
    font_applied_ = true;
}

void App::frame() {
    handle_keys();
    draw_menu_bar();
    draw_toolbar();
    draw_status_bar();
    draw_dock();
    draw_tools();
    settings_dialog_.draw(&settings_, paths_.settings);
    Mode show_mode = mode();
    std::string show_name;
    if (layout_dialogs_.draw(&settings_, &show_mode, &show_name)) {
        set_mode(show_mode);
        request_named_layout(show_name);
    }
    if (error_dialog_.draw()) status("Diagnostics copied to the clipboard");
    save_if_changed();
}

void App::set_mode(Mode m) {
    if (settings_.mode == m) return;
    settings_.mode = m;
    status(std::string("Mode: ") + mode_title(m));
}

void App::request_preset(int index) {
    request_layout(presets()[static_cast<size_t>(index)].root, presets()[static_cast<size_t>(index)].name);
}

bool App::request_named_layout(const std::string &name) {
    const NamedLayouts &all = settings_.layouts[mode_index(mode())];
    const int i = find_named_layout(all, name);
    if (i < 0) return false;
    request_layout(all[static_cast<size_t>(i)].root, all[static_cast<size_t>(i)].name);
    return true;
}

SaveResult App::save_current_layout(const std::string &name, bool overwrite, std::string *why) {
    const int mi = mode_index(mode());
    return save_named_layout(&settings_.layouts[mi], name, capture_layout(mode(), settings_.tools[mi]), overwrite, why);
}

void App::request_layout(const LayoutNode &root, const std::string &label) {
    const int mi = mode_index(mode());
    pending_[mi] = true;
    pending_root_[mi] = root;
    status("Layout: " + label);
}

void App::begin_key_capture(Action a) {
    settings_dialog_.begin_key_capture(a);
}

std::string App::engine_health() const {
    return engine_found_ ? "Engine: found" : "Engine: missing";
}

std::string App::inference_health() const {
    return socket_present_ ? "Inference: socket present" : "Inference: not connected";
}

// The configured engine binary and socket, looked at again when their paths change or a second
// has passed.
void App::check_health() {
    const std::string engine = engine_binary_path(settings_), socket = socket_path(settings_);
    const std::string key = engine + '\n' + socket;
    const double now = ImGui::GetTime();
    if (key == health_paths_ && health_checked_ >= 0 && now - health_checked_ < 1.0) return;
    health_paths_ = key;
    health_checked_ = now;
    engine_found_ = is_executable(engine);
    socket_present_ = !socket.empty() && path_kind(socket) == PathKind::SOCKET;
}

void App::set_tool_open(Mode m, Tool t, bool open) {
    settings_.tools[mode_index(m)][tool_index(t)] = open;
    if (open) dock_home_if_lost(m, t);
}

void App::set_font(const FontChoice &c) {
    settings_.font = sanitize_font_choice(c);
}

void App::set_theme(const std::string &id) {
    settings_.theme = find_theme(id) >= 0 ? id : DEFAULT_THEME;
}

void App::open_settings(SettingsDialog::Tab tab) {
    settings_dialog_.open(tab);
}

void App::report_error(const std::string &title, const std::string &message, const std::string &detail) {
    error_dialog_.open(title, message, detail);
}

void App::status(const std::string &message) {
    status_ = message;
}

// Esc closes menus and combo boxes (a modal dialog closes itself on Esc); the bound keys act
// outside text fields and popups.
void App::handle_keys() {
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) ImGui::ClosePopupsExceptModals();
    if (ImGui::GetIO().WantTextInput || ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId)) return;
    for (int a = 0; a < ACTION_COUNT; ++a)
        if (chord_pressed(settings_.keys[a], action_info(static_cast<Action>(a)).repeat))
            run_action(static_cast<Action>(a));
}

void App::run_action(Action a) {
    const ActionInfo &info = action_info(a);
    if ((info.scope == Scope::REPLAY && mode() != Mode::REPLAY) || (info.scope == Scope::PLAY && mode() != Mode::PLAY))
        return;
    switch (a) {
        case Action::REPLAY_FIRST:
        case Action::REPLAY_PREV:
        case Action::REPLAY_NEXT:
        case Action::REPLAY_LAST:
        case Action::PLAY_TO_END_OF_TURN:
        case Action::PLAY_SKIP_TURN:
            status(std::string("No game loaded: ") + info.label);
            break;
        case Action::MAGNIFY:
            break;
        case Action::OPEN_SETTINGS:
            open_settings();
            break;
        case Action::OPEN_REPLAY:
        case Action::SAVE_DECISION_LOG:
        case Action::NEW_GAME:
        case Action::FIND_IN_LOG:
            not_built(info.label);
            break;
    }
}

void App::not_built(const char *item) {
    status(std::string("not built yet: ") + item);
}

std::string App::shortcut(Action a) const {
    return chord_label(settings_.keys[action_index(a)]);
}

void App::draw_menu_bar() {
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(9, 5));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, tok().BTN);
    ImGui::PushStyleColor(ImGuiCol_Header, tok().BTN);
    const bool open = ImGui::BeginMainMenuBar();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar();
    if (!open) return;
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float bar_h = ImGui::GetCurrentWindow()->MenuBarHeight;
    draw_icon(ImGui::GetWindowDrawList(), Icon::LOGO, ImVec2(p.x, ImGui::GetWindowPos().y + (bar_h - 18) / 2));
    ImGui::Dummy(ImVec2(18, 1));
    ImGui::SameLine(0, 4);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 5));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 4));
    if (ImGui::BeginMenu("File")) {
        draw_file_menu();
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Edit")) {
        if (ImGui::MenuItem("Copy Board")) not_built("Copy Board");
        if (ImGui::MenuItem("Copy Decision")) not_built("Copy Decision");
        if (ImGui::MenuItem("Find in Log…", shortcut(Action::FIND_IN_LOG).c_str())) not_built("Find in Log");
        ImGui::Separator();
        if (ImGui::MenuItem("Settings…", shortcut(Action::OPEN_SETTINGS).c_str())) open_settings();
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
        draw_view_menu();
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Game")) {
        if (ImGui::MenuItem("New Game vs Model…", shortcut(Action::NEW_GAME).c_str())) not_built("New Game vs Model");
        if (ImGui::MenuItem("Sideboard…")) not_built("Sideboard");
        ImGui::Separator();
        if (ImGui::MenuItem("Concede Game", "-2")) not_built("Concede Game");
        if (ImGui::MenuItem("Concede Match", "-3")) not_built("Concede Match");
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Analysis")) {
        if (ImGui::MenuItem("Re-evaluate With Checkpoint…")) not_built("Re-evaluate With Checkpoint");
        if (ImGui::MenuItem("Compare Policies…")) not_built("Compare Policies");
        if (ImGui::MenuItem("Biggest Swings")) {
            set_tool_open(mode(), Tool::SWINGS, true);
            focus_tool_ = tool_index(Tool::SWINGS);
        }
        if (ImGui::MenuItem("Batch Over Episodes…")) not_built("Batch Over Episodes");
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Help")) {
        if (ImGui::MenuItem("About Robomage…")) not_built("About Robomage");
        ImGui::EndMenu();
    }
    ImGui::PopStyleVar(2);
    const char *title = "Robomage Workbench";
    const float title_x = (ImGui::GetWindowWidth() - ImGui::CalcTextSize(title).x) / 2;
    if (title_x > ImGui::GetCursorPosX() + 20) {
        ImGui::SetCursorPosX(title_x);
        ImGui::TextDisabled("%s", title);
    }
    ImGui::EndMainMenuBar();
}

void App::draw_file_menu() {
    if (ImGui::MenuItem("Open Replay (.rmlog)…", shortcut(Action::OPEN_REPLAY).c_str())) not_built("Open Replay");
    if (ImGui::MenuItem("Open Trajectory Episode…")) not_built("Open Trajectory Episode");
    if (ImGui::MenuItem("Save Decision Log…", shortcut(Action::SAVE_DECISION_LOG).c_str()))
        not_built("Save Decision Log");
    ImGui::Separator();
    if (ImGui::MenuItem("Exit")) quit_ = true;
}

void App::draw_view_menu() {
    const Mode m = mode();
    // Each section has its own ID scope: the Play preset and the Play mode share a label.
    ImGui::TextDisabled("Tools");
    ImGui::PushID("tools");
    for (int t = 0; t < TOOL_COUNT; ++t) {
        const bool open = settings_.tools[mode_index(m)][t];
        if (ImGui::MenuItem(tool_info(static_cast<Tool>(t)).title, nullptr, open))
            set_tool_open(m, static_cast<Tool>(t), !open);
    }
    ImGui::PopID();
    ImGui::Separator();
    ImGui::TextDisabled("Layout for %s", mode_title(m));
    ImGui::PushID("layout");
    const std::vector<Preset> &all = presets();
    for (size_t i = 0; i < all.size(); ++i)
        if (ImGui::MenuItem(all[i].name)) request_preset(static_cast<int>(i));
    const NamedLayouts &saved = settings_.layouts[mode_index(m)];
    if (!saved.empty()) {
        ImGui::Separator();
        ImGui::TextDisabled("Saved layouts");
        ImGui::PushID("saved");
        for (size_t i = 0; i < saved.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            if (ImGui::MenuItem(saved[i].name.c_str())) request_named_layout(saved[i].name);
            ImGui::PopID();
        }
        ImGui::PopID();
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Save Layout As…")) layout_dialogs_.open_save(m);
    if (ImGui::MenuItem("Manage Layouts…")) layout_dialogs_.open_manage(m);
    if (ImGui::MenuItem("Reset Layout")) request_preset(default_preset(m));
    ImGui::PopID();
    ImGui::Separator();
    ImGui::TextDisabled("Mode");
    ImGui::PushID("mode");
    if (ImGui::MenuItem("Play", nullptr, m == Mode::PLAY)) set_mode(Mode::PLAY);
    if (ImGui::MenuItem("Replay", nullptr, m == Mode::REPLAY)) set_mode(Mode::REPLAY);
    ImGui::PopID();
}

void App::draw_toolbar() {
    ImGuiViewport *vp = ImGui::GetMainViewport();
    const float height = ImGui::GetFrameHeight() + 9;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6, 4));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(3, 4));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, tok().MENU);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNavFocus;
    if (ImGui::BeginViewportSideBar("##rmwork.toolbar", vp, ImGuiDir_Up, height, flags)) {
        if (icon_button("##open", Icon::OPEN, "Open replay or trajectory")) not_built("Open Replay");
        ImGui::SameLine();
        if (icon_button("##save", Icon::SAVE, "Save decision log")) not_built("Save Decision Log");
        toolbar_separator();
        const Segment modes[] = {{"Play", true, Icon::MODE_PLAY, "A live game against a model"},
            {"Replay", true, Icon::MODE_REPLAY, "A recorded game"}};
        int sel = mode() == Mode::PLAY ? 0 : 1;
        if (segmented("##mode", modes, 2, &sel)) set_mode(sel == 0 ? Mode::PLAY : Mode::REPLAY);
        ImGui::SameLine(0, 6);
        ImGui::SetNextItemWidth(GAME_COMBO_EMS * ImGui::GetFontSize());
        ImGui::PushStyleColor(ImGuiCol_Button, tok().BTN_HOVER);
        if (ImGui::BeginCombo("##game", "No game loaded")) {
            ImGui::TextDisabled("No games yet: File → Open Replay arrives with the replay data (plan §2.2).");
            ImGui::EndCombo();
        }
        ImGui::PopStyleColor();
        toolbar_separator();
        if (mode() == Mode::REPLAY) {
            const struct {
                    const char *id;
                    Icon icon;
                    const char *what;
                    Action action;
                    bool keyed;
            } nav[] = {{"##first", Icon::FIRST, "First decision", Action::REPLAY_FIRST, true},
                {"##prev", Icon::PREV, "Previous decision", Action::REPLAY_PREV, true},
                {"##play", Icon::PLAY, "Play through", Action::REPLAY_NEXT, false},
                {"##next", Icon::NEXT, "Next decision", Action::REPLAY_NEXT, true},
                {"##last", Icon::LAST, "Last decision", Action::REPLAY_LAST, true},
                {"##swing", Icon::SWING, "Next big swing", Action::REPLAY_NEXT, false}};
            for (const auto &n : nav) {
                const std::string tip =
                    n.keyed ? std::string(n.what) + " (" + shortcut(n.action) + ")" : std::string(n.what);
                icon_button(n.id, n.icon, tip.c_str(), false);
                ImGui::SameLine();
            }
            toolbar_separator();
            ImGui::AlignTextToFramePadding();
            ImGui::TextDisabled("View");
            ImGui::SameLine(0, 6);
            const Segment views[] = {{"A", false, Icon::LOGO, "Seat A's frames"},
                {"B", false, Icon::LOGO, "Seat B's frames"}, {"Omni", false, Icon::LOGO, "The omniscient frames"}};
            if (segmented("##view", views, 3, &view_)) status(std::string("View: ") + VIEW_NAMES[view_]);
        } else {
            const std::string f2 = shortcut(Action::PLAY_TO_END_OF_TURN) + " To end of turn";
            const std::string f6 = shortcut(Action::PLAY_SKIP_TURN) + " Skip turn";
            if (ImGui::Button(f2.c_str())) run_action(Action::PLAY_TO_END_OF_TURN);
            ImGui::SetItemTooltip("Pass to the end of the turn; stops if something goes on the stack");
            ImGui::SameLine();
            if (ImGui::Button(f6.c_str())) run_action(Action::PLAY_SKIP_TURN);
            ImGui::SetItemTooltip("Pass to the end of the turn no matter what; first choice at every decision");
            ImGui::SameLine();
            if (ImGui::Button("Concede…")) not_built("Concede");
            ImGui::SetItemTooltip("Concede (-2 game, -3 match)");
        }
        bottom_rule(tok().LINE);
    }
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(3);
}

void App::draw_status_bar() {
    ImGuiViewport *vp = ImGui::GetMainViewport();
    const float height = ImGui::GetTextLineHeight() + 7;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 3));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, tok().STATUS);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNavFocus;
    if (ImGui::BeginViewportSideBar("##rmwork.status", vp, ImGuiDir_Down, height, flags)) {
        const ImGuiWindow *w = ImGui::GetCurrentWindow();
        ImGui::GetWindowDrawList()->AddLine(w->Pos, ImVec2(w->Pos.x + w->Size.x, w->Pos.y), tok().LINE);
        const std::string mode_text =
            mode() == Mode::REPLAY ? std::string("Replay · view ") + VIEW_NAMES[view_] : std::string("Play · no game");
        status_segment(mode_text.c_str());
        status_segment("Decision —");
        status_segment("Turn —");
        status_segment("V —");
        const std::string engine_text = engine_health(), server_text = inference_health();
        const char *engine = engine_text.c_str();
        const char *server = server_text.c_str();
        const float right_w = ImGui::CalcTextSize(engine).x + ImGui::CalcTextSize(server).x + 12 * 2 + 21 + 2;
        const ImVec2 msg_pos = ImGui::GetCursorScreenPos();
        const float msg_end = w->Pos.x + w->Size.x - 8 - right_w - 12;
        if (msg_end > msg_pos.x) {
            ImGui::PushStyleColor(ImGuiCol_Text, tok().DIM);
            ImGui::RenderTextEllipsis(ImGui::GetWindowDrawList(), msg_pos,
                ImVec2(msg_end, msg_pos.y + ImGui::GetTextLineHeight()), msg_end, status_.c_str(), nullptr, nullptr);
            ImGui::PopStyleColor();
        }
        ImGui::SetCursorScreenPos(ImVec2(std::max(msg_pos.x, w->Pos.x + w->Size.x - 8 - right_w), msg_pos.y));
        health_indicator(engine, engine_found_ ? tok().GOOD : tok().BAD);
        ImGui::SameLine(0, 10);
        status_rule();
        health_indicator(server, socket_present_ ? tok().ACCENT : tok().DIM);
    }
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

void App::draw_dock() {
    const Mode m = mode();
    const int mi = mode_index(m);
    ImGuiViewportP *vp = static_cast<ImGuiViewportP *>(ImGui::GetMainViewport());
    const ImRect area = vp->GetBuildWorkRect();
    const ImGuiID ds = dockspace_id(m);
    if (!layout_checked_[mi]) {
        layout_checked_[mi] = true;
        if (!ImGui::DockBuilderGetNode(ds) && !pending_[mi]) {
            pending_[mi] = true;
            pending_root_[mi] = presets()[static_cast<size_t>(default_preset(m))].root;
        }
    }
    if (pending_[mi]) {
        apply_layout(m, pending_root_[mi], area.GetSize(), &settings_.tools[mi]);
        submit_order_[mi] = layout_order(pending_root_[mi]);
        applied_root_[mi] = pending_root_[mi];
        tab_order_frames_[mi] = 3;
        pending_[mi] = false;
        rehome_[mi].fill(false);
    } else if (tab_order_frames_[mi] > 0) {
        --tab_order_frames_[mi];
        order_layout_tabs(m, applied_root_[mi]);
    }
    ImGui::SetNextWindowPos(area.Min);
    ImGui::SetNextWindowSize(area.GetSize());
    ImGui::SetNextWindowViewport(vp->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(3, 3));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, tok().BG);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                                   ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking |
                                   ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollbar |
                                   ImGuiWindowFlags_NoScrollWithMouse;
    ImGui::Begin(dockspace_host_name(m), nullptr, flags);
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
    // The gaps between groups are the background until hovered (the splitters' colour).
    ImGui::PushStyleColor(ImGuiCol_Separator, tok().BG);
    ImGui::DockSpace(ds, ImVec2(0, 0), ImGuiDockNodeFlags_NoWindowMenuButton);
    ImGui::PopStyleColor();
    bool any = false;
    for (bool open : settings_.tools[mi]) any = any || open;
    if (!any) {
        const char *text = "Every tool is closed. Open one from the View menu, or pick a layout under View.";
        const ImVec2 size = ImGui::CalcTextSize(text);
        ImGui::GetWindowDrawList()->AddText(
            ImVec2(area.Min.x + (area.GetWidth() - size.x) / 2, area.Min.y + (area.GetHeight() - size.y) / 2),
            tok().BG_TEXT, text);
    }
    ImGui::End();
}

void App::draw_tools() {
    const int mi = mode_index(mode());
    for (Tool tool : submit_order_[mi]) {
        const int t = tool_index(tool);
        if (!settings_.tools[mi][t]) continue;
        if (rehome_[mi][t]) {
            rehome_[mi][t] = false;
            if (ImGuiDockNode *root = ImGui::DockBuilderGetNode(dockspace_id(mode())))
                ImGui::SetNextWindowDockID(first_leaf(root)->ID, ImGuiCond_Always);
        }
        if (focus_tool_ == t) {
            ImGui::SetNextWindowFocus();
            focus_tool_ = -1;
        }
        // A docked group has one close button, at the right of its tab strip; the tabs have none.
        ImGuiWindowClass tab_class;
        tab_class.TabItemFlagsOverrideSet = ImGuiTabItemFlags_NoCloseButton;
        ImGui::SetNextWindowClass(&tab_class);
        const ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoFocusOnAppearing;
        if (ImGui::Begin(window_names_[mi][t].c_str(), &settings_.tools[mi][t], flags)) draw_tool_body(tool);
        ImGui::End();
    }
}

// A tool reopened with no home in the mode's dockspace (never docked, or its node is gone) is
// docked into the dockspace's first tab group at its next Begin; a tool the user floated stays
// floating.
void App::dock_home_if_lost(Mode m, Tool t) {
    const ImGuiID id = ImHashStr(window_names_[mode_index(m)][tool_index(t)].c_str());
    bool known = false;
    ImGuiID dock = 0;
    if (const ImGuiWindow *w = ImGui::FindWindowByID(id)) {
        known = true;
        dock = w->DockId;
    } else if (const ImGuiWindowSettings *s = ImGui::FindWindowSettingsByID(id)) {
        known = true;
        dock = s->DockId;
    }
    if (known && dock == 0) return;
    ImGuiDockNode *node = dock ? ImGui::DockBuilderGetNode(dock) : nullptr;
    if (node && ImGui::DockNodeGetRootNode(node)->ID == dockspace_id(m)) return;
    rehome_[mode_index(m)][tool_index(t)] = true;
}

void App::save_if_changed() {
    const std::string json = settings_to_json(settings_);
    if (json == saved_json_) return;
    saved_json_ = json;
    if (!persist_) return;
    std::string err;
    if (!save_settings(paths_.settings, settings_, &err)) status("Could not save settings: " + err);
}

// The first leaf under `node` that holds a window, else the first leaf.
ImGuiDockNode *App::first_leaf(ImGuiDockNode *node) {
    if (node->IsLeafNode()) return node;
    for (ImGuiDockNode *child : node->ChildNodes) {
        if (!child) continue;
        ImGuiDockNode *leaf = first_leaf(child);
        if (!leaf->Windows.empty()) return leaf;
    }
    return node->ChildNodes[0] ? first_leaf(node->ChildNodes[0]) : node;
}

// One status bar field and the rule after it.
void App::status_segment(const char *text) {
    ImGui::TextUnformatted(text);
    ImGui::SameLine(0, 10);
    status_rule();
}

// A vertical rule between status bar fields, then the next field's spacing.
void App::status_rule() {
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float h = ImGui::GetTextLineHeight();
    ImGui::GetWindowDrawList()->AddLine(ImVec2(p.x, p.y - 1), ImVec2(p.x, p.y + h + 1), tok().LINE);
    ImGui::Dummy(ImVec2(1, h));
    ImGui::SameLine(0, 10);
}

// A connection indicator: a dot and its label.
void App::health_indicator(const char *label, ImU32 dot) {
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float h = ImGui::GetTextLineHeight();
    ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(p.x + 4, p.y + h / 2), 3.5f, dot);
    ImGui::Dummy(ImVec2(12, h));
    ImGui::SameLine(0, 0);
    ImGui::TextUnformatted(label);
}

// A one-pixel line along the bottom of the current window.
void App::bottom_rule(ImU32 color) {
    const ImGuiWindow *w = ImGui::GetCurrentWindow();
    ImGui::GetWindowDrawList()->AddLine(
        ImVec2(w->Pos.x, w->Pos.y + w->Size.y - 1), ImVec2(w->Pos.x + w->Size.x, w->Pos.y + w->Size.y - 1), color);
}

}  // namespace rmwork
