#include "app.h"

#include <algorithm>

#include "episodes.h"
#include "files.h"
#include "imgui_internal.h"
#include "panels.h"
#include "paths.h"
#include "replay_cache.h"
#include "theme.h"
#include "widgets.h"

namespace {

// The toolbar's game combo box, in font heights (300 px at the default 14).
const float GAME_COMBO_EMS = 21.5f;
// The status bar's load progress bar, in font heights.
const float PROGRESS_EMS = 9.0f;

}  // namespace

static std::string file_name(const std::string &path);
static std::string seconds_text(double s);

static std::string file_name(const std::string &path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

static std::string seconds_text(double s) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f s", s);
    return buf;
}

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
    poll_loader();
    replay_.tick(ImGui::GetTime());
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
    std::string picked;
    if (picker_.draw(&picked)) pick_done(picked);
    std::string run_log;
    if (run_browser_.draw(&run_log)) open_replay(run_log);
    save_if_changed();
}

bool App::open_replay(const std::string &log, const std::vector<std::string> &episodes) {
    if (loader_.busy()) {
        status("Still opening a game: wait, or cancel it in the status bar");
        return false;
    }
    LoadRequest req;
    req.log_path = log;
    req.episodes = episodes;
    opened_episodes_.clear();
    for (const std::string &e : episodes) opened_episodes_.push_back(absolute_path(e));
    req.engine = replay_engine();
    req.workdir = engine_workdir_path(settings_);
    req.cache_root = replay_cache_root();
    replay_.set_playing(false, ImGui::GetTime());
    load_started_ = ImGui::GetTime();
    loader_.start(req);
    set_mode(Mode::REPLAY);
    status("Opening " + file_name(log) + " …");
    return true;
}

bool App::open_episode(const std::string &path) {
    std::string err;
    const std::string log = absolute_path(log_for_episode(path, engine_workdir_path(settings_), &err));
    if (log.empty()) {
        report_error("No decision log for the episode",
            "The episode " + file_name(path) +
                " joins a replay of its decision log, which was not found: open the log with File → Open Replay "
                "first, or keep the episode beside it.",
            err + "\n");
        return false;
    }
    if (replay_.loaded() && replay_.match()->log_path == log && target_decision_ < 0)
        target_decision_ = static_cast<int>(replay_.cursor());
    return open_replay(log, {path});
}

void App::open_run(const std::string &dir) {
    run_browser_.open(absolute_path(dir));
    add_recent(&settings_.recent, absolute_path(dir));
    set_mode(Mode::REPLAY);
}

void App::open_file_picker(PickPurpose purpose) {
    std::string start = settings_.recent.empty() ? std::string() : parent_dir(settings_.recent.front());
    if (replay_.loaded()) start = parent_dir(replay_.match()->log_path);
    if (purpose == PICK_RUN) start = run_dir_path(settings_);
    if (start.empty() || path_kind(start) != PathKind::DIRECTORY) start = repo_root();
    const char *titles[] = {"Open Replay", "Open Trajectory Episode", "Open Run Directory"};
    const FileFilter filters[] = {FileFilter::DECISION_LOGS, FileFilter::EPISODES, FileFilter::DIRECTORIES};
    picker_.open(titles[purpose], purpose, filters[purpose], start, settings_.recent);
}

void App::pick_done(const std::string &path) {
    if (path_kind(path) == PathKind::DIRECTORY) {
        open_run(path);
        return;
    }
    const bool episode = path.size() > 7 && path.compare(path.size() - 7, 7, ".rmtraj") == 0;
    if (episode) {
        open_episode(path);
    } else {
        open_replay(path);
    }
}

std::string App::load_phase(double *fraction) const {
    *fraction = -1;
    if (!loader_.busy()) return "";
    std::string phase;
    loader_.progress(&phase, fraction);
    return phase;
}

void App::set_load_target(int decision, int view) {
    target_decision_ = decision;
    target_view_ = view;
}

std::string App::replay_engine() const {
    return replay_engine_binary(engine_override_.empty() ? engine_binary_path(settings_) : engine_override_);
}

// Takes a finished load: the match into the session, or the error dialog.
void App::poll_loader() {
    if (!loader_.busy() || !loader_.finished()) return;
    LoadOutcome o;
    if (!loader_.take(&o)) return;
    install(&o);
}

void App::install(LoadOutcome *o) {
    const double took = ImGui::GetTime() - load_started_;
    load_notes_ = o->notes;
    if (o->status == LoadOutcome::Status::CANCELLED) {
        status("Opening cancelled");
        target_decision_ = target_view_ = -1;
        return;
    }
    if (o->status == LoadOutcome::Status::FAILED) {
        status(o->title);
        report_error(o->title, o->message, o->detail);
        target_decision_ = target_view_ = -1;
        return;
    }
    Match &m = *o->match;
    add_recent(&settings_.recent, m.log_path);
    for (const std::string &e : opened_episodes_) add_recent(&settings_.recent, e);
    std::string msg = "Loaded " + file_name(m.log_path) + ": " + std::to_string(m.decisions.size()) + " decisions";
    size_t games = 0;
    for (const GameSpan &g : m.games) games += g.sideboarding ? 0 : 1;
    if (m.bo3()) msg += ", " + std::to_string(games) + " games";
    msg += ", " + m.result.text + (o->replayed ? " (replayed in " + seconds_text(took) + ")" : " (from the cache)");
    if (!m.episode_files.empty())
        msg += "; " + std::to_string(m.episode_files.size()) + " episode" + (m.episode_files.size() == 1 ? "" : "s") +
               ", " + std::to_string(m.episode_steps.size()) + " decisions joined";
    const size_t cursor = target_decision_ >= 0 ? static_cast<size_t>(target_decision_) : 0;
    replay_.set_match(std::move(o->match), cursor);
    if (target_view_ >= 0) replay_.set_view(target_view_);
    target_decision_ = target_view_ = -1;
    status(msg);
    if (!o->title.empty()) report_error(o->title, o->message, o->detail);
}

std::string App::status_mode_text() const {
    if (mode() == Mode::PLAY) return "Play · no game";
    return std::string("Replay · view ") + view_name(replay_.view());
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
    const bool nav =
        a == Action::REPLAY_FIRST || a == Action::REPLAY_PREV || a == Action::REPLAY_NEXT || a == Action::REPLAY_LAST;
    if (nav && !replay_.loaded()) {
        status(std::string("No game loaded: ") + info.label);
        return;
    }
    if (nav) replay_.set_playing(false, ImGui::GetTime());
    switch (a) {
        case Action::REPLAY_FIRST:
            replay_.first();
            break;
        case Action::REPLAY_PREV:
            replay_.step(-1);
            break;
        case Action::REPLAY_NEXT:
            replay_.step(1);
            break;
        case Action::REPLAY_LAST:
            replay_.last();
            break;
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
            open_file_picker(PICK_REPLAY);
            break;
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
    if (ImGui::MenuItem("Open Replay (.rmlog)…", shortcut(Action::OPEN_REPLAY).c_str())) open_file_picker(PICK_REPLAY);
    if (ImGui::MenuItem("Open Trajectory Episode…")) open_file_picker(PICK_EPISODE);
    if (ImGui::MenuItem("Open Run Directory…")) open_file_picker(PICK_RUN);
    if (ImGui::BeginMenu("Open Recent", !settings_.recent.empty())) {
        for (size_t i = 0; i < settings_.recent.size(); ++i) {
            const std::string path = settings_.recent[i];
            ImGui::PushID(static_cast<int>(i));
            const bool exists = path_kind(path) != PathKind::MISSING;
            if (ImGui::MenuItem(file_name(path).c_str(), nullptr, false, exists)) pick_done(path);
            ImGui::SetItemTooltip("%s", path.c_str());
            ImGui::PopID();
        }
        ImGui::EndMenu();
    }
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
        if (icon_button("##open", Icon::OPEN, "Open a replay (.rmlog)")) open_file_picker(PICK_REPLAY);
        ImGui::SameLine();
        if (icon_button("##save", Icon::SAVE, "Save decision log")) not_built("Save Decision Log");
        toolbar_separator();
        const Segment modes[] = {{"Play", true, Icon::MODE_PLAY, "A live game against a model"},
            {"Replay", true, Icon::MODE_REPLAY, "A recorded game"}};
        int sel = mode() == Mode::PLAY ? 0 : 1;
        if (segmented("##mode", modes, 2, &sel)) set_mode(sel == 0 ? Mode::PLAY : Mode::REPLAY);
        ImGui::SameLine(0, 6);
        draw_game_combo();
        toolbar_separator();
        if (mode() == Mode::REPLAY) {
            draw_replay_nav();
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

// Replay: the match's games and sideboarding phases (choosing one goes to its first decision);
// Play: no game yet.
void App::draw_game_combo() {
    const bool replay = mode() == Mode::REPLAY;
    const std::string preview = replay ? (loader_.busy() ? "Opening…" : replay_.game_label()) : "No game loaded";
    ImGui::SetNextItemWidth(GAME_COMBO_EMS * ImGui::GetFontSize());
    ImGui::PushStyleColor(ImGuiCol_Button, tok().BTN_HOVER);
    if (open_game_combo_ && replay && replay_.loaded()) {
        ImGui::OpenPopupEx(ImHashStr("##ComboPopup", 0, ImGui::GetID("##game")));
        open_game_combo_ = false;
    }
    if (ImGui::BeginCombo("##game", preview.c_str())) {
        if (!replay || !replay_.loaded()) {
            ImGui::TextDisabled(replay ? "No game loaded: File → Open Replay" : "No game: Play arrives in plan §2.3.");
        } else {
            const Match &m = *replay_.match();
            const int cur = m.span_of(replay_.cursor());
            ImGui::TextDisabled("%s", m.title().c_str());
            for (size_t g = 0; g < m.games.size(); ++g) {
                const GameSpan &span = m.games[g];
                std::string label = span.label + "  (" + std::to_string(span.count) + " decisions";
                if (!span.sideboarding && static_cast<size_t>(span.game) < m.result.game_winners.size()) {
                    const int w = m.result.game_winners[static_cast<size_t>(span.game)];
                    label += w == 0 ? ", Player A wins" : w == 1 ? ", Player B wins" : ", a draw";
                }
                label += ")";
                ImGui::PushID(static_cast<int>(g));
                if (ImGui::Selectable(label.c_str(), static_cast<int>(g) == cur)) {
                    replay_.set_playing(false, ImGui::GetTime());
                    replay_.go_to(span.first);
                }
                ImGui::PopID();
            }
            ImGui::Separator();
            ImGui::TextDisabled("Result: %s", m.result.text.c_str());
        }
        ImGui::EndCombo();
    }
    ImGui::PopStyleColor();
    if (replay && replay_.loaded()) ImGui::SetItemTooltip("%s", replay_.match()->log_path.c_str());
}

// First, previous, play / pause, next, last, next swing (waits for evaluations, plan §2.7); the viewer.
void App::draw_replay_nav() {
    const bool loaded = replay_.loaded();
    const double now = ImGui::GetTime();
    const std::string first = "First decision (" + shortcut(Action::REPLAY_FIRST) + ")";
    const std::string prev = "Previous decision (" + shortcut(Action::REPLAY_PREV) + ")";
    const std::string next = "Next decision (" + shortcut(Action::REPLAY_NEXT) + ")";
    const std::string last = "Last decision (" + shortcut(Action::REPLAY_LAST) + ")";
    if (icon_button("##first", Icon::FIRST, first.c_str(), loaded)) run_action(Action::REPLAY_FIRST);
    ImGui::SameLine();
    if (icon_button("##prev", Icon::PREV, prev.c_str(), loaded)) run_action(Action::REPLAY_PREV);
    ImGui::SameLine();
    const bool playing = replay_.playing();
    if (icon_button("##play", playing ? Icon::PAUSE : Icon::PLAY,
            playing ? "Pause" : "Play through, a decision at a time", loaded))
        replay_.set_playing(!playing, now);
    ImGui::SameLine();
    if (icon_button("##next", Icon::NEXT, next.c_str(), loaded)) run_action(Action::REPLAY_NEXT);
    ImGui::SameLine();
    if (icon_button("##last", Icon::LAST, last.c_str(), loaded)) run_action(Action::REPLAY_LAST);
    ImGui::SameLine();
    icon_button("##swing", Icon::SWING, "Next big swing (needs evaluations, plan §2.7)", false);
    ImGui::SameLine();
    toolbar_separator();
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled("View");
    ImGui::SameLine(0, 6);
    const Segment views[] = {{"A", false, Icon::LOGO, "Seat A's frames and its own log"},
        {"B", false, Icon::LOGO, "Seat B's frames and its own log"},
        {"Omni", false, Icon::LOGO, "The omniscient frames and the narrative"}};
    int view = replay_.view();
    if (segmented("##view", views, 3, &view)) {
        replay_.set_view(view);
        status(std::string("View: ") + view_name(view));
    }
}

// While a load runs: its phase, a progress bar and Cancel, ending at `right_edge`.
void App::draw_load_progress(float right_edge) {
    double fraction = -1;
    const std::string phase = load_phase(&fraction);
    const float bar_w = PROGRESS_EMS * ImGui::GetFontSize();
    const float cancel_w = ImGui::CalcTextSize("Cancel").x + 10;
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float h = ImGui::GetTextLineHeight();
    const float text_end = right_edge - bar_w - cancel_w - 16;
    if (text_end > p.x) {
        ImGui::PushStyleColor(ImGuiCol_Text, tok().TEXT);
        ImGui::RenderTextEllipsis(
            ImGui::GetWindowDrawList(), p, ImVec2(text_end, p.y + h), text_end, phase.c_str(), nullptr, nullptr);
        ImGui::PopStyleColor();
    }
    ImGui::SetCursorScreenPos(ImVec2(std::max(p.x, text_end + 8), p.y));
    const float shown = fraction < 0 ? static_cast<float>(-ImGui::GetTime()) : static_cast<float>(fraction);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 0));
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, tok().ACCENT);
    ImGui::ProgressBar(shown, ImVec2(bar_w, h), fraction < 0 ? "" : nullptr);
    ImGui::PopStyleColor();
    ImGui::SameLine(0, 6);
    if (ImGui::SmallButton("Cancel")) cancel_load();
    ImGui::PopStyleVar();
    ImGui::SameLine(0, 10);
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
        status_segment(status_mode_text().c_str());
        if (mode() == Mode::REPLAY) {
            status_segment(replay_.decision_text().c_str());
            status_segment(replay_.turn_text().c_str());
            status_segment(replay_.decider_text().c_str());
        } else {
            status_segment("Decision —");
            status_segment("Turn —");
        }
        status_segment("V —");
        const std::string engine_text = engine_health(), server_text = inference_health();
        const char *engine = engine_text.c_str();
        const char *server = server_text.c_str();
        const float right_w = ImGui::CalcTextSize(engine).x + ImGui::CalcTextSize(server).x + 12 * 2 + 21 + 2;
        if (loader_.busy()) draw_load_progress(w->Pos.x + w->Size.x - 8 - right_w - 12);
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
    links_.set_context(replay_.generation());
    links_.new_frame();
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
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoFocusOnAppearing;
        if (tool == Tool::TABLE) flags |= ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
        ToolContext ctx;
        ctx.mode = mode();
        ctx.replay = mode() == Mode::REPLAY ? &replay_ : nullptr;
        ctx.log = &game_log_;
        ctx.table = &table_;
        ctx.links = &links_;
        ctx.settings = &settings_;
        double fraction = -1;
        ctx.loading = load_phase(&fraction);
        if (ImGui::Begin(window_names_[mi][t].c_str(), &settings_.tools[mi][t], flags)) draw_tool_body(tool, ctx);
        ImGui::End();
    }
    const std::string table_status = table_.take_status();
    if (!table_status.empty()) status(table_status);
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
