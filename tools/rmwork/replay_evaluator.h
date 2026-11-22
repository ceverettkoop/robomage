#ifndef RMWORK_REPLAY_EVALUATOR_H
#define RMWORK_REPLAY_EVALUATOR_H

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../../src/featurize/episode.h"
#include "../../src/featurize/featurizer.h"
#include "../../src/featurize/setup.h"
#include "match.h"

// ── The Replay evaluator: a recorded match's seat frames evaluated through the inference server ──
//
// One thread, shared by every client in Replay: re-evaluation (reevaluation.h) and Attribution
// (docs/workbench_plan.md §2.8). A client submits a job: a match's frames (EvalSource, copied from
// the Match so the thread never touches the Match the UI holds), the served policies to evaluate
// with (rmserve's numeric --policy ids, in order), a priority, and its items: (decision, seat)
// pairs, each with an optional change to the decision's tensors (Attribution's perturbations).
//
// For each item the thread reads that seat's own STATE frame at the decision from the dump and
// featurizes it with featurize_seat (the actor entry point, which refuses the omniscient frame: the
// omniscient viewer's frames are never read), applies the item's change to a copy of the tensors,
// and evaluates it. Items go to the server in requests no larger than the server's batch (its
// WELCOME's max_batch, at most Config::batch_cap) whose tensors fit the request area. Between
// requests the thread takes the queued job of the highest priority (equal priorities in the order
// submitted), so a higher-priority job overtakes a long one at its next request. Results are
// handed back per request (take()), so a client can show them as they arrive.
//
// The featurizer's inputs (eval_connection.h: the engine's cards file and the oracle-text table) are
// loaded once per engine binary (its path and modification time) and working directory; one connection is kept, made
// again when a job asks for other policies, the configuration changed, or a request on a kept connection fails (the
// server was restarted: the request is sent once more on the new one). A job of no items always makes a fresh
// handshake, so its versions are those the server serves now. The last few decisions' featurized tensors are kept (an
// LRU), so a client changing one decision many ways featurizes it once.
//
// Failures fail the job they happen in, never the evaluator: no socket at the configured path
// (`no_server`), the inputs not loading, a refused handshake (a policy not served, a spec hash or
// tables mismatch), the server going away or a failing request (the connection is dropped and the
// next job connects again), a frame that does not decode or featurize. Cancelling a job drops its
// items not yet sent; a request in flight finishes and its results are still handed back.

namespace rmwork {

// A match's frames as the evaluator's thread reads them.
struct EvalSource {
        std::string dump_path;
        uint64_t log_hash = 0;
        std::array<std::vector<FrameRef>, 2> match_frames;  // per seat its once-per-match frames, in dump order
        std::vector<std::array<FrameRef, 2>> frames;        // per decision the seats' STATE frames
};
// `m`'s source: of Play's live match only the human seat's frames so far (its stream file, which
// grows; the other seat's decisions have no frame and an item naming one fails its job).
std::shared_ptr<const EvalSource> eval_source(const Match &m);

struct EvalItem {
        size_t decision = 0;  // the match's decision index
        int seat = 0;         // whose frame: 0 A, 1 B
        // A change to the featurized decision before it is evaluated (run on the evaluator's thread,
        // on a copy); empty for the decision as featurized. False with `err` fails the job.
        std::function<bool(featurize::Tensors *, std::string *)> change;
};

struct EvalJob {
        std::shared_ptr<const EvalSource> source;
        std::vector<uint32_t> policies;  // served ids, in order (non-empty)
        int priority = 0;                // higher first
        // Evaluated in this order; none: the job only connects (its status then names the versions).
        std::vector<EvalItem> items;
};

struct EvalResult {
        size_t item = 0;  // the job's item
        size_t decision = 0;
        int seat = 0;
        std::vector<featurize::PolicyEvalRecord> policies;  // per job policy in order, `id` the served id
};

struct EvalJobStatus {
        enum class State {
            QUEUED,
            RUNNING,
            DONE,
            CANCELLED,
            FAILED,
            UNKNOWN  // no such job (never submitted, or forgotten)
        };
        State state = State::UNKNOWN;
        size_t done = 0, total = 0;      // items evaluated (or dropped by a cancel: not counted), items
        std::string error;               // FAILED: why
        bool no_server = false;          // FAILED because no socket is at the configured path
        std::vector<uint64_t> versions;  // per job policy the version the server named, once connected
        bool finished() const { return state == State::DONE || state == State::CANCELLED || state == State::FAILED; }
};

class ReplayEvaluator {
    public:
        struct Config {
                std::string socket;      // the inference server's
                std::string engine;      // writes the cards file
                std::string workdir;     // the engine's working directory (its resources/)
                std::string cards_path;  // where the cards file goes
                double timeout_s = 60;   // the longest wait for the handshake or a request
                size_t batch_cap = 64;   // the largest request (also when the server names no max_batch)
                std::string client = "rmwork Replay evaluator";
        };
        struct Stats {
                size_t requests = 0;    // requests answered
                size_t decisions = 0;   // decisions in them
                size_t largest = 0;     // the largest request's decisions
                size_t featurized = 0;  // frames featurized (an LRU hit is not one)
                size_t max_batch = 0;   // the connected server's
        };

        ReplayEvaluator();
        ReplayEvaluator(const ReplayEvaluator &) = delete;
        ReplayEvaluator &operator=(const ReplayEvaluator &) = delete;
        ~ReplayEvaluator();

        // The server and the featurizer's inputs for jobs from now on (the connection is made again at
        // the next job when they changed).
        void configure(const Config &c);
        // Queues `job` (starting the thread); its id (> 0).
        uint64_t submit(EvalJob job);
        // Drops the job's items not yet sent; CANCELLED once a request in flight returns.
        void cancel(uint64_t job);
        EvalJobStatus status(uint64_t job) const;
        // The job's results not taken before, in the order evaluated.
        std::vector<EvalResult> take(uint64_t job);
        // Forgets a job (cancelling it first): its status becomes UNKNOWN.
        void forget(uint64_t job);
        // A job is queued or running.
        bool busy() const;
        Stats stats() const;
        // Stops the thread (a request in flight is waited for) and drops the connection.
        void stop();
        // The featurizer's inputs the thread loaded last (nullptr before a job loaded them): what a client
        // featurizes a frame with to read the same tensors the thread evaluates (Attribution's units).
        std::shared_ptr<const featurize::Featurization> features() const;

    private:
        struct JobState;
        struct Thread;
        void run();
        std::shared_ptr<JobState> next_job();
        // One request of `j`'s next items (or only its connection, for a job of none); the job ends
        // when its last item returns, it fails or it was cancelled.
        void step(const std::shared_ptr<JobState> &j);
        void fail(const std::shared_ptr<JobState> &j, const std::string &why, bool no_server);

        mutable std::mutex mu_;
        std::condition_variable cv_;
        std::thread thread_;
        std::atomic<bool> stop_{false};
        Config config_;
        unsigned config_gen_ = 0;
        uint64_t next_id_ = 1;
        std::map<uint64_t, std::shared_ptr<JobState>> jobs_;
        Stats stats_;
        std::shared_ptr<const featurize::Featurization> features_;  // the thread's, once loaded
        std::unique_ptr<Thread> t_;                                 // what the thread owns
};

}  // namespace rmwork

#endif /* RMWORK_REPLAY_EVALUATOR_H */
