#ifndef RMWORK_RUN_BROWSER_H
#define RMWORK_RUN_BROWSER_H

#include <atomic>
#include <map>
#include <string>
#include <thread>
#include <vector>

// ── A run directory's decision logs (a minimal list; the Episode Browser is plan §3) ─────────────
//
// `runs/<name>/rmlog/<purpose>/*.rmlog` (selfplay, eval, …), or the logs directly in the directory
// given, each with the games its episodes cover (`episodes/<purpose>/<stem>.g<n>.rmtraj`, or beside
// the log): a purpose selector, a name filter, "with episodes only", the list in natural order.
// Choosing a log opens it, and its episodes join it (episodes.h). The directory is read on a
// background thread (a run can hold hundreds of thousands of logs).

namespace rmwork {

struct RunLog {
        std::string path;
        std::string name;        // the file name without .rmlog
        std::vector<int> games;  // the games with an episode, ascending
};

struct RunPurpose {
        std::string name;  // "selfplay", or "." for logs directly in the directory
        std::vector<RunLog> logs;
        size_t with_episodes = 0;
};

// Reads `dir` (what the browser's thread does).
std::vector<RunPurpose> scan_run_directory(const std::string &dir);

class RunBrowser {
    public:
        RunBrowser() = default;
        RunBrowser(const RunBrowser &) = delete;
        RunBrowser &operator=(const RunBrowser &) = delete;
        ~RunBrowser();
        // Starts reading `dir` and opens the dialog.
        void open(const std::string &dir);
        // Draws it while open (at the frame's top level); true in the frame a log was chosen.
        bool draw(std::string *log_path);
        bool is_open() const { return open_ || want_open_; }
        // The directory has been read.
        bool ready() const { return done_.load(); }
        const std::vector<RunPurpose> &purposes() const { return purposes_; }

    private:
        void collect();
        std::string dir_;
        bool want_open_ = false, open_ = false;
        std::thread thread_;
        std::atomic<bool> done_{false};
        bool collected_ = false;
        std::vector<RunPurpose> scanned_;   // the thread's result
        std::vector<RunPurpose> purposes_;  // shown
        int purpose_ = 0;
        char filter_[128] = "";
        bool episodes_only_ = false;
        std::vector<size_t> shown_;
        std::string shown_key_;
        int selected_ = -1;
};

}  // namespace rmwork

#endif /* RMWORK_RUN_BROWSER_H */
