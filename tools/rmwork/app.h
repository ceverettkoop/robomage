#ifndef RMWORK_APP_H
#define RMWORK_APP_H

#include <array>
#include <string>
#include <vector>

#include "dialogs.h"
#include "imgui.h"
#include "layout.h"
#include "settings.h"

// ── The application shell: menu bar, toolbar, the mode's dockspace and tools, status bar ────────
//
// begin_frame() goes before ImGui::NewFrame() (it applies a changed theme or font setting, which
// only take effect between frames) and frame() draws one UI frame between ImGui::NewFrame() and
// ImGui::Render(); the App owns no window or renderer, so the self-test runs it headless. Settings
// changes are written back as they happen (when `persist`).

struct ImGuiDockNode;

namespace rmwork {

class App {
    public:
        App(const Settings &settings, const ConfigPaths &paths, bool persist);

        void begin_frame();
        void frame();
        bool quit_requested() const { return quit_; }

        Mode mode() const { return settings_.mode; }
        void set_mode(Mode m);
        // Builds preset `index` on the current mode's dockspace at the next dock pass.
        void request_preset(int index);
        bool tool_open(Mode m, Tool t) const { return settings_.tools[mode_index(m)][tool_index(t)]; }
        // Opens or closes a tool as View does; a tool opened with no home in the mode's dockspace is
        // docked into its first tab group.
        void set_tool_open(Mode m, Tool t, bool open);
        const Settings &settings() const { return settings_; }
        // The UI font from the next frame on (sanitized, as Settings → Appearance sets it).
        void set_font(const FontChoice &c);
        // The colour theme from the next frame on (the default for an unknown id).
        void set_theme(const std::string &id);

        void open_settings(SettingsDialog::Tab tab = SettingsDialog::TABLE);
        bool settings_open() const { return settings_dialog_.is_open(); }
        bool error_open() const { return error_dialog_.is_open(); }
        // Shows the error dialog (for the later stages: an engine exit, a frame that fails to
        // decode, a lost inference server).
        void report_error(const std::string &title, const std::string &message, const std::string &detail);
        // The status bar's message.
        void status(const std::string &message);
        const std::string &status_message() const { return status_; }

    private:
        void handle_keys();
        void run_action(Action a);
        void not_built(const char *item);
        void draw_menu_bar();
        void draw_file_menu();
        void draw_view_menu();
        void draw_toolbar();
        void draw_status_bar();
        void draw_dock();
        void draw_tools();
        void dock_home_if_lost(Mode m, Tool t);
        void save_if_changed();
        std::string shortcut(Action a) const;
        static ImGuiDockNode *first_leaf(ImGuiDockNode *node);
        static void status_segment(const char *text);
        static void status_rule();
        static void health_indicator(const char *label, ImU32 dot);
        static void bottom_rule(ImU32 color);

        Settings settings_;
        ConfigPaths paths_;
        bool persist_;
        std::string saved_json_;  // the settings as last written (or read)
        std::string status_;
        FontChoice applied_font_;  // the font in effect, once font_applied_
        bool font_applied_ = false;
        std::string applied_theme_;  // the theme in effect (empty before the first frame)
        bool quit_ = false;
        int view_ = 0;         // Replay's viewer: 0 seat A, 1 seat B, 2 omniscient
        int focus_tool_ = -1;  // a tool to bring to the front at its next Begin
        std::array<int, MODE_COUNT> pending_preset_{{-1, -1}};
        std::array<int, MODE_COUNT> applied_preset_{{-1, -1}};  // the preset whose tabs are being ordered
        std::array<int, MODE_COUNT> tab_order_frames_{{0, 0}};  // frames left to order them
        std::array<bool, MODE_COUNT> layout_checked_{{false, false}};
        std::array<std::array<bool, TOOL_COUNT>, MODE_COUNT> rehome_{};
        std::array<std::array<std::string, TOOL_COUNT>, MODE_COUNT> window_names_;
        std::array<std::vector<Tool>, MODE_COUNT> submit_order_;  // the order draw_tools begins the windows
        SettingsDialog settings_dialog_;
        ErrorDialog error_dialog_;
};

}  // namespace rmwork

#endif /* RMWORK_APP_H */
