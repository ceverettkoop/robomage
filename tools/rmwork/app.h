#ifndef RMWORK_APP_H
#define RMWORK_APP_H

#include <array>
#include <string>
#include <vector>

#include "dialogs.h"
#include "file_picker.h"
#include "game_log.h"
#include "imgui.h"
#include "layout.h"
#include "layout_dialogs.h"
#include "links.h"
#include "loader.h"
#include "replay_session.h"
#include "run_browser.h"
#include "settings.h"
#include "settings_dialog.h"
#include "table_view.h"

// ── The application shell: menu bar, toolbar, the mode's dockspace and tools, status bar ────────
//
// begin_frame() goes before ImGui::NewFrame() (it applies a changed theme or font setting, which
// only take effect between frames) and frame() draws one UI frame between ImGui::NewFrame() and
// ImGui::Render(); the App owns no window or renderer, so the self-test runs it headless. Settings
// changes are written back as they happen (when `persist`). The status bar's health indicators
// look at the configured engine binary and server socket at most once a second.
//
// Replay's data: File → Open Replay (a decision log), Open Trajectory Episode and Open Run Directory
// go through the file picker (file_picker.h) or the run's list (run_browser.h); a log is replayed
// and indexed on the loader's thread (loader.h) with its progress and a Cancel in the status bar, and
// the match it hands over becomes the ReplaySession the toolbar, the keys, the status bar and the
// tools read. A failure opens the error dialog.

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
        // The same for the current mode's saved layout `name`; false when there is none.
        bool request_named_layout(const std::string &name);
        // Saves the current mode's dockspace as it is now under `name` (as Save Layout As does).
        SaveResult save_current_layout(const std::string &name, bool overwrite, std::string *why);
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
        // The error dialog's title and message (the last one opened).
        const std::string &error_title() const { return error_dialog_.title(); }
        const std::string &error_message() const { return error_dialog_.message(); }
        SettingsDialog &settings_dialog() { return settings_dialog_; }
        LayoutDialogs &layout_dialogs() { return layout_dialogs_; }
        // Settings, at the Keys tab, waiting for `a`'s new key.
        void begin_key_capture(Action a);
        // The status bar's health texts ("Engine: found", "Inference: not connected").
        std::string engine_health() const;
        std::string inference_health() const;
        // Shows the error dialog (for the later stages: an engine exit, a frame that fails to
        // decode, a lost inference server).
        void report_error(const std::string &title, const std::string &message, const std::string &detail);
        // The status bar's message.
        void status(const std::string &message);
        const std::string &status_message() const { return status_; }

        // Opens decision log `log` in Replay (replayed and indexed on the loader's thread); the
        // episodes named join it with those found by name. False when a load is running.
        bool open_replay(const std::string &log, const std::vector<std::string> &episodes = {});
        // Opens an episode: its decision log (the `rmlog` attribute, else by name) with it; when
        // that log is the one loaded, it is opened again at the same decision. False after an error.
        bool open_episode(const std::string &path);
        // Lists a run directory's decision logs (run_browser.h).
        void open_run(const std::string &dir);
        // The file picker, for a decision log, an episode or a run directory.
        enum PickPurpose {
            PICK_REPLAY,
            PICK_EPISODE,
            PICK_RUN
        };
        void open_file_picker(PickPurpose purpose);
        bool loading() const { return loader_.busy(); }
        // A game is being opened or a run directory read (a screenshot waits for both).
        bool working() const { return loader_.busy() || (run_browser_.is_open() && !run_browser_.ready()); }
        void cancel_load() { loader_.cancel(); }
        // The loader's phase and fraction (-1 unknown), "" when idle.
        std::string load_phase(double *fraction) const;
        ReplaySession &replay() { return replay_; }
        const ReplaySession &replay() const { return replay_; }
        // The decision (0-based) and viewer the next finished load shows (-1: the first, the current).
        void set_load_target(int decision, int view);
        // The engine binary used for replays instead of the setting's (for the command line and the
        // data self-test); "" goes back to the setting.
        void set_engine_override(const std::string &path) { engine_override_ = path; }
        // The binary a replay runs (replay_engine_binary of the override or the setting).
        std::string replay_engine() const;
        FilePicker &file_picker() { return picker_; }
        RunBrowser &run_browser() { return run_browser_; }
        GameLogView &game_log() { return game_log_; }
        TableView &table() { return table_; }
        LinkState &links() { return links_; }
        // Opens the toolbar's game combo box at the next frame.
        void open_game_combo() { open_game_combo_ = true; }
        // The status bar's fields, as drawn.
        std::string status_mode_text() const;
        // What the last finished load said (the notes), for the self-test.
        const std::vector<std::string> &load_notes() const { return load_notes_; }

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
        void draw_game_combo();
        void draw_replay_nav();
        void draw_load_progress(float right_edge);
        void poll_loader();
        void install(LoadOutcome *o);
        void pick_done(const std::string &path);
        void dock_home_if_lost(Mode m, Tool t);
        void save_if_changed();
        void check_health();
        void request_layout(const LayoutNode &root, const std::string &label);
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
        ReplaySession replay_;
        Loader loader_;
        GameLogView game_log_;
        TableView table_;
        LinkState links_;
        FilePicker picker_;
        RunBrowser run_browser_;
        std::string engine_override_;
        int target_decision_ = -1, target_view_ = -1;
        bool open_game_combo_ = false;
        double load_started_ = 0;
        std::vector<std::string> load_notes_;
        std::vector<std::string> opened_episodes_;              // the episodes the running load was asked to join
        int focus_tool_ = -1;                                   // a tool to bring to the front at its next Begin
        std::array<bool, MODE_COUNT> pending_{{false, false}};  // a layout waits for the next dock pass
        std::array<LayoutNode, MODE_COUNT> pending_root_;
        std::array<LayoutNode, MODE_COUNT> applied_root_;       // the layout whose tabs are being ordered
        std::array<int, MODE_COUNT> tab_order_frames_{{0, 0}};  // frames left to order them
        std::array<bool, MODE_COUNT> layout_checked_{{false, false}};
        std::array<std::array<bool, TOOL_COUNT>, MODE_COUNT> rehome_{};
        std::array<std::array<std::string, TOOL_COUNT>, MODE_COUNT> window_names_;
        std::array<std::vector<Tool>, MODE_COUNT> submit_order_;  // the order draw_tools begins the windows
        SettingsDialog settings_dialog_;
        LayoutDialogs layout_dialogs_;
        ErrorDialog error_dialog_;
        double health_checked_ = -1.0;
        std::string health_paths_;  // the engine and socket paths last checked
        bool engine_found_ = false;
        bool socket_present_ = false;
};

}  // namespace rmwork

#endif /* RMWORK_APP_H */
