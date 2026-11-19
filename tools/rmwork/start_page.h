#ifndef RMWORK_START_PAGE_H
#define RMWORK_START_PAGE_H

#include <atomic>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "start_data.h"
#include "widgets.h"

// ── The start page (plan §4.1, item 1): what the workbench shows when nothing is open ────────────
//
// A page over the whole work area in place of the mode's dockspace (app.h decides when): a "Start"
// tab whose ✕ goes to the mode's layout; on the left the actions (New Game, Open Replay, Open
// Trajectory Episode, Open Run, each with its key), the keys to know and "show this page when
// nothing is open"; in the middle Play's recent games, the recent replays and the recent runs (a
// double-click opens one: a game or replay in Replay, a run in the run browser); on the right the
// status blocks (engine, card images, inference server, files), each with the Settings tab that
// fixes it. The lists come from scan_start (start_data.h) on the page's own thread; refresh() asks
// for another scan, and the page shows the last one finished.

namespace rmwork {

// What the page asks the App to do.
struct StartRequest {
        enum class Kind {
            NONE,
            NEW_GAME,
            OPEN_REPLAY,   // the file picker on decision logs
            OPEN_EPISODE,  // the file picker on episodes
            PICK_RUN,      // the file picker on run directories
            OPEN_LOG,      // `path` in Replay
            OPEN_RUN,      // `path` in the run browser
            SETTINGS,      // Settings at tab `settings_tab`
            HIDE           // the Start tab's ✕: the mode's layout
        };
        Kind kind = Kind::NONE;
        std::string path;
        int settings_tab = 0;
};

// A status block and the Settings tab that changes what it shows (-1 none).
struct StartStatus {
        StatusBlock block;
        int settings_tab = -1;
        std::string settings_label;  // "Settings → Engine"
};

// One of the page's actions: its icon, title, key (the bound chord, "" none) and what it does.
struct StartAction {
        StartRequest::Kind kind;
        Icon icon;
        std::string title, key, text;
};

// What the page shows besides the scan.
struct StartContext {
        std::vector<StartAction> actions;
        std::vector<std::pair<std::string, std::string>> keys;  // (the chord, what it does)
        std::vector<StartStatus> status;
        std::string games_dir;
        bool *show_at_start = nullptr;  // the setting
};

class StartPage {
    public:
        StartPage() = default;
        StartPage(const StartPage &) = delete;
        StartPage &operator=(const StartPage &) = delete;
        ~StartPage();

        // Starts a scan with `r` unless one is running.
        void refresh(const StartScanRequest &r);
        // What is shown may be out of date: fresh() waits for a scan started after this.
        void invalidate() { stale_ = true; }
        // Takes a finished scan (the UI thread, once a frame); true when it took one.
        bool poll();
        bool scanning() const { return running_; }
        // A scan has finished since the page was made.
        bool scanned() const { return scanned_; }
        // The lists shown come from a scan started after the last invalidate(), and none is running.
        bool fresh() const { return scanned_ && !stale_ && !running_; }
        const StartData &data() const { return data_; }
        // Draws the page filling the current window; what was asked for this frame.
        StartRequest draw(const StartContext &c);

    private:
        void draw_left(const StartContext &c, StartRequest *req);
        void draw_middle(StartRequest *req);
        void draw_right(const StartContext &c, StartRequest *req);
        void draw_games(float height, StartRequest *req);
        void draw_replays(float height, StartRequest *req);
        void draw_runs(float height, StartRequest *req);
        static bool action_card(const StartAction &a, float width);
        static void section_header(const char *title, const char *subtitle, const char *hint);
        static void status_box(const StartStatus &s, float width, StartRequest *req);
        static void empty_note(const char *text);

        std::thread thread_;
        std::atomic<bool> done_{false};
        bool running_ = false;
        bool scanned_ = false;
        bool stale_ = true;
        StartData result_;  // the thread's
        StartData data_;    // shown
        int selected_game_ = -1, selected_replay_ = -1, selected_run_ = -1;
};

}  // namespace rmwork

#endif /* RMWORK_START_PAGE_H */
