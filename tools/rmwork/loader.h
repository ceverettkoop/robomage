#ifndef RMWORK_LOADER_H
#define RMWORK_LOADER_H

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "match.h"

// ── Opening a decision log, on a background thread ──────────────────────────────────────────────
//
// start() reads and checks the log, then on its own thread: finds the replay in the cache
// (replay_cache.h) or runs the engine (`<engine> --replay LOG --rmstate-dump DUMP --narrative`, in
// the engine's working directory, stdout and stderr into the cache entry's files) and caches it,
// reads the entry's index (match_index.h) or indexes the dump (match.h) and caches that, and joins
// the log's episodes (episodes.h) and any opened with it. The UI thread polls: progress() while
// it runs, take() once finished() — the thread hands the Match over whole, so the UI never sees
// one half built. cancel() kills the engine or stops the indexing; the destructor cancels and joins.

namespace rmwork {

// The engine that replays: `configured`, except that a robomage_nn with a robomage beside it gives
// that robomage (a replay needs no learned policy, and the plain engine is the one `make` keeps
// current).
std::string replay_engine_binary(const std::string &configured);

struct LoadRequest {
        std::string log_path;               // the decision log (made absolute)
        std::vector<std::string> episodes;  // episodes opened with it, joined after the log's own
        bool find_episodes = true;          // join the episodes that correspond to the log by name
        std::string engine;                 // the engine binary (replay_engine_binary applied)
        std::string workdir;                // its working directory (bin/)
        std::string cache_root;             // replay_cache_root()
};

struct LoadOutcome {
        enum class Status {
            OK,
            FAILED,
            CANCELLED
        } status = Status::FAILED;
        std::unique_ptr<Match> match;        // OK: the match
        bool replayed = false;               // the engine ran (not a cache hit)
        std::string title, message, detail;  // FAILED; with OK, a problem joining an episode opened with it
        std::vector<std::string> notes;      // what the status bar may mention
};

class Loader {
    public:
        Loader() = default;
        Loader(const Loader &) = delete;
        Loader &operator=(const Loader &) = delete;
        ~Loader();

        // False when a load is already running.
        bool start(const LoadRequest &req);
        // Started and not yet taken.
        bool busy() const { return running_; }
        bool finished() const { return done_.load(); }
        void cancel() { cancel_.store(true); }
        // What it is doing ("Replaying a0-m12.rmlog") and how far (0–1, -1 unknown).
        void progress(std::string *phase, double *fraction) const;
        // Once finished: the outcome (the thread joined); false otherwise.
        bool take(LoadOutcome *out);

    private:
        void run();
        void set_phase(const std::string &phase, double fraction);
        bool replay(const std::string &log_bytes, Match *m, LoadOutcome *out);
        bool index(const std::string &log_bytes, Match *m, LoadOutcome *out, std::string *err);
        void join_episodes(Match *m, LoadOutcome *out);
        void fail(LoadOutcome *out, const std::string &title, const std::string &message, const std::string &detail);

        std::thread thread_;
        std::atomic<bool> cancel_{false}, done_{false};
        bool running_ = false;
        mutable std::mutex mu_;
        std::string phase_;
        double fraction_ = -1;
        LoadRequest req_;
        LoadOutcome out_;
        std::string index_path_;  // the cache entry's index (match_index.h)
};

}  // namespace rmwork

#endif /* RMWORK_LOADER_H */
