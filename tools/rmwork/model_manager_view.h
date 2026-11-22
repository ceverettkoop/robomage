#ifndef RMWORK_MODEL_MANAGER_VIEW_H
#define RMWORK_MODEL_MANAGER_VIEW_H

#include <string>
#include <vector>

#include "model_manager.h"
#include "settings.h"

// ── The Models tool: the policies by date, their compatibility, the inference server ─────────────
//
// A strip with the inference server's state (the status bar's text), the socket and Start / Restart /
// Stop, the same as Settings → Models' checkbox and Restart server, and a link to that tab; a filter
// field (run or name), a Run combo box and Show all (the hidden rows); the policy table (date, run,
// name, learner steps, formats, compatible), sorted by a click on a header, date descending every
// time the tool opens, `latest` first under any sort. An incompatible row is greyed and names
// `rmtrain retable <run>`; hovering a row shows its files and what its compatibility rests on. The
// selected row is played against (New Game with it as the opponent's model; also a double-click),
// served (added to Settings → Models' policies, the server started) or its path copied: requests the
// App carries out. The runs directory is scanned again each time the tool opens and on Refresh; the
// compatibility is checked on CompatChecker's thread (model_manager.h).

namespace rmwork {

struct ModelRequest {
        enum class Kind {
            NONE,
            PLAY_AGAINST,  // New Game with `path` as the opponent's model
            SERVE,         // `path` added to the served policies, the server started
            COPY_PATH,     // `path` to the clipboard
            START_SERVER,
            RESTART_SERVER,
            STOP_SERVER,
            SETTINGS  // Settings → Models
        };
        Kind kind = Kind::NONE;
        std::string path;
};

struct ModelManagerContext {
        const Settings *settings = nullptr;
        std::string engine;        // the engine binary whose tables the rows are checked against
        std::string server_state;  // "Inference: server running" (the status bar's)
        std::string server_error;  // the server's failure ("" none)
        bool server_bad = false;   // the state is a failure
};

class ModelManagerView {
    public:
        // Whether the tool is open in the shown mode, every frame: opening it scans the runs
        // directory again and sorts by date, newest first.
        void set_open(bool open);
        // The tool's body; what the user asked for.
        ModelRequest draw(const ModelManagerContext &ctx);
        // Scans the runs directory and the models directory again (Refresh: the compatibility too).
        void rescan(const Settings &s, bool recheck);

        const std::vector<PolicyRow> &rows() const { return rows_; }
        // The rows shown at the last draw, in order (indices into rows()).
        const std::vector<size_t> &shown() const { return shown_; }
        ModelFilter &filter() { return filter_; }
        ModelColumn sort_column() const { return column_; }
        bool sort_ascending() const { return ascending_; }
        // Selects the shown row whose path to play (or checkpoint) is `path`; false when none is.
        bool select_path(const std::string &path);
        // The selected row (nullptr none).
        const PolicyRow *selected() const;
        CompatChecker &checker() { return checker_; }
        bool busy() const { return checker_.busy(); }

    private:
        void draw_server_strip(const ModelManagerContext &ctx, ModelRequest *req);
        void draw_filters(const ModelManagerContext &ctx);
        void draw_actions(ModelRequest *req);
        void draw_table(ModelRequest *req);
        void draw_row(size_t index, ModelRequest *req);
        static std::string row_key(const PolicyRow &r);
        static std::string row_path(const PolicyRow &r);
        static std::string row_tooltip(const PolicyRow &r, const CompatVerdict &v);

        bool open_ = false;
        bool scan_ = false;    // scan at the next draw
        int generation_ = 0;   // the table's ID: a new one each time the tool opens (the default sort)
        double requested_at_ = -1;
        std::vector<PolicyRow> rows_;
        std::vector<CompatVerdict> verdicts_;  // per row, this frame's
        std::vector<size_t> shown_;
        std::vector<std::string> runs_;
        ModelFilter filter_;
        char filter_text_[128] = {0};
        ModelColumn column_ = ModelColumn::DATE;
        bool ascending_ = false;
        std::string selected_;  // the selected row's key
        CompatChecker checker_;
        EngineTables tables_;
};

}  // namespace rmwork

#endif /* RMWORK_MODEL_MANAGER_VIEW_H */
