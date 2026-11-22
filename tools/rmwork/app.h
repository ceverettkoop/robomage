#ifndef RMWORK_APP_H
#define RMWORK_APP_H

#include <array>
#include <deque>
#include <string>
#include <vector>

#include "card_images.h"
#include "dialogs.h"
#include "file_picker.h"
#include "game_log.h"
#include "image_renderer.h"
#include "inference_server.h"
#include "imgui.h"
#include "layout.h"
#include "layout_dialogs.h"
#include "links.h"
#include "loader.h"
#include "new_game_dialog.h"
#include "play_dialogs.h"
#include "play_session.h"
#include "replay_session.h"
#include "replay_tools.h"
#include "run_browser.h"
#include "settings.h"
#include "settings_dialog.h"
#include "start_page.h"
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
// tools read (the tools beside the Table through ReplayTools, replay_tools.h). A failure opens the
// error dialog.
//
// Play: Game → New Game vs Model (new_game_dialog.h) starts a PlaySession (play_session.h), whose
// live match is the session the tools read in Play (only the human seat's frames ever reach it);
// Choices submits rows at the human's decision, F2 / F6 and the stops pass, Game → Concede asks
// first (play_dialogs.h), the match's end opens the result screen, and File → Save Decision Log
// copies the game's log. The human's sideboarding phase brings the Sideboard tool to the front (and
// the next game the Table), and the tool sends its confirmed plan after each poll
// (sideboard_view.h). The engine's failures reach the error dialog; quitting kills the engine.
//
// The start page (start_page.h) takes the place of the mode's dockspace and tools while it is shown,
// so neither mode's layout changes under it: at start when its setting is on and nothing was opened,
// from File → Start Page, and when File → Close closes the open replay or a game that is over.
// Opening a replay, starting a game, the Start tab's ✕, the toolbar's mode switch and View's tools and
// layouts go back to the dockspace. Its lists are scanned on its thread when it is shown and every
// few seconds while it is; its status blocks look at the paths at most once a second.

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
        // Settings → Models → Start the server with the workbench, as its checkbox sets it.
        void set_start_server(bool on) { settings_.models.start_server = on; }
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
            PICK_RUN,
            PICK_SAVE_LOG
        };
        void open_file_picker(PickPurpose purpose);
        bool loading() const { return loader_.busy(); }
        // A game is being opened or a run directory read, or Play's engine has not asked yet or
        // auto_play is still answering (a screenshot waits for each).
        bool working() const;
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
        // The card images (Settings → Card Printings) the Table and the zone windows draw with.
        CardImages &card_images() { return card_images_; }
        // Cards drawn as their images (else as text), as Settings → Card Printings sets it.
        void set_card_faces(bool images) { settings_.card_images = images; }
        LinkState &links() { return links_; }
        ReplayTools &tools() { return tools_; }
        // Brings tool `t`'s tab to the front of its group once the layout has settled.
        void focus_tool(Tool t) { focus_tools_.push_back(tool_index(t)); }
        // Goes to the next swing the viewer sees (the toolbar's button); false when there is none.
        bool next_swing();
        // Opens the toolbar's game combo box at the next frame.
        void open_game_combo() { open_game_combo_ = true; }
        // The status bar's fields, as drawn.
        std::string status_mode_text() const;
        // What the last finished load said (the notes), for the self-test.
        const std::vector<std::string> &load_notes() const { return load_notes_; }

        // ── Play ──
        // The session the tools read: Replay's loaded match, or Play's live one.
        ReplaySession &session() { return mode() == Mode::PLAY ? play_.view() : replay_; }
        PlaySession &play() { return play_; }
        // Game → New Game vs Model, with the last choices.
        void open_new_game();
        NewGameDialog &new_game_dialog() { return new_game_; }
        // Starts a game with `g` (kept as the next game's choices) and switches to Play; false after
        // the error dialog. `setup`: the engine's test-setup flags (PlayConfig::setup), for the checks.
        bool start_game(const NewGameSettings &g, const std::vector<std::string> &setup = {});
        // The engine binary a game against `o` runs ("" with `why` when there is none).
        std::string play_engine(Opponent o, std::string *why) const;
        // Sends `row` at the human's pending decision; false (with a status message) when none is.
        bool play_reply(int row);
        // F2 / F6.
        void play_pass(PassMode mode);
        // Space: passes priority once at the human's pending priority decision (its PASS_PRIORITY row).
        void play_pass_priority();
        // The concession's confirmation (Game → Concede Game / Match).
        void open_concede(bool match);
        ConcedeDialog &concede_dialog() { return concede_; }
        GameOverDialog &game_over() { return game_over_; }
        // Answers the human's next `decisions` decisions (-1: every one) with a seeded random row
        // until `until` holds ("stack": the stack is not empty; "combat": an attacker or blocker
        // declaration; "sideboard": a sideboarding decision; "target": a target or a point of divided
        // damage to choose; "mana": a cost to pay with a mana ability; "pick": a card picker's decision;
        // "mulligan": keep or mulligan; "bottom": the London mulligan's bottom pick; "end": never, so to
        // the match's end; "" none): screenshots and checks.
        void auto_play(long decisions, const std::string &until);
        // Answers the human's next decisions with `rows`, one each, before auto_play's rule: a scripted
        // line for screenshots (with a sculpted game's setup flags).
        void script_rows(const std::vector<int> &rows) { script_.assign(rows.begin(), rows.end()); }
        // Applies a Sideboard tool plan spec (SideboardView::apply_spec) at the human's first sideboarding
        // decision: the screenshot flags'.
        void set_sideboard_spec(const std::string &spec);
        bool auto_playing() const { return (auto_left_ != 0 || !script_.empty()) && play_.running(); }
        // File → Save Decision Log: copies the session's decision log to `path`.
        bool save_log_to(const std::string &path, std::string *err);
        // Whether Choices submits at this frame: Play, the human's pending decision at the cursor.
        bool choosing() const;
        // The status bar's Play field: "Your decision", "Opponent is thinking…", "Match over", …
        std::string play_state_text() const;

        // ── The start page ──
        // Shows it in place of the mode's dockspace (its lists scanned again), or the dockspace.
        void show_start_page(bool on);
        bool start_page_shown() const { return start_shown_; }
        StartPage &start_page() { return start_page_; }
        // Does what the page asked for: an action, a recent game or run, a Settings tab, its ✕.
        void run_start_request(const StartRequest &r);
        // File → Close: Replay's log (a load cancelled), or Play's game once it is not running; then
        // the start page when its setting is on. False (a status message) while Play's game runs.
        bool close_session();
        // The page's status blocks (engine, card images, inference server, files) as they are now.
        std::vector<StartStatus> start_status();

    private:
        void update_card_images();
        void handle_keys();
        void run_action(Action a);
        void not_built(const char *item);
        void draw_menu_bar();
        void draw_file_menu();
        void draw_view_menu();
        void view_menu_tool(Tool t);
        void draw_toolbar();
        void draw_status_bar();
        void draw_dock();
        void draw_tools();
        void draw_start_page();
        void refresh_start_page();
        StartContext start_context();
        StartScanRequest start_scan_request() const;
        void show_tool(Tool t);
        void draw_game_combo();
        void draw_replay_nav();
        void draw_load_progress(float right_edge);
        void poll_loader();
        void poll_play();
        void step_auto_play();
        void follow_sideboarding();
        void apply_sideboard_spec();
        void draw_play_nav();
        void draw_play_dialogs();
        void open_save_picker();
        void install(LoadOutcome *o);
        void pick_done(const std::string &path);
        void dock_home_if_lost(Mode m, Tool t);
        void save_if_changed();
        void check_health();
        void run_server();
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
        CardImages card_images_;
        ImageCardRenderer image_renderer_{&card_images_};
        TableView table_;
        LinkState links_;
        ReplayTools tools_;
        FilePicker picker_;
        RunBrowser run_browser_;
        std::string engine_override_;
        int target_decision_ = -1, target_view_ = -1;
        bool open_game_combo_ = false;
        double load_started_ = 0;
        std::vector<std::string> load_notes_;
        std::vector<std::string> opened_episodes_;              // the episodes the running load was asked to join
        std::vector<int> focus_tools_;                          // tools to bring to the front at their next Begin
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
        PlaySession play_;
        NewGameDialog new_game_;
        ConcedeDialog concede_;
        GameOverDialog game_over_;
        bool new_game_pressed_ = false;  // Play's empty Table's button
        bool play_dealt_ = false;        // the live game's first decision has arrived
        long auto_left_ = 0;             // auto_play: decisions left (-1 unbounded)
        std::string auto_until_;
        std::deque<int> script_;  // script_rows' rows still to send
        int sideboard_phase_ = -1;      // the game of the sideboarding phase the human last decided in, -1 none
        bool sideboard_front_ = false;  // the Sideboard tool was brought to the front for it
        std::string sideboard_spec_;
        std::array<bool, MODE_COUNT> homed_{{false, false}};
        uint64_t auto_rng_ = 0;
        double health_checked_ = -1.0;
        std::string health_paths_;  // the engine and socket paths last checked
        bool engine_found_ = false;
        bool socket_present_ = false;
        InferenceServer server_;  // Settings → Models → Start the server (interactive sessions only)
        StartPage start_page_;
        bool start_shown_ = false;
        double start_scanned_at_ = -1;  // when the page's last scan was asked for
        std::string start_recent_;      // the recent list it was asked with
        double start_status_at_ = -1;
        std::vector<StartStatus> start_status_;
};

}  // namespace rmwork

#endif /* RMWORK_APP_H */
