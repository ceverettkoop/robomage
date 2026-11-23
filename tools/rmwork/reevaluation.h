#ifndef RMWORK_REEVALUATION_H
#define RMWORK_REEVALUATION_H

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "eval_cache.h"
#include "evaluations.h"
#include "match.h"
#include "replay_evaluator.h"
#include "settings.h"

// ── Re-evaluation: a recorded match's decisions evaluated again with the served policies ─────────
//
// Each chosen served policy evaluates each chosen seat's decisions in that seat's own frame (the
// deciding seat's frame, which carries the menu; sideboarding decisions too) through the Replay
// evaluator (replay_evaluator.h). Per (policy, seat) the result is merged into the match's
// evaluations (merge_evaluations: the decisions' episode steps, a step of its own where a decision
// has none) under the policy's re-evaluated id (evaluations.h: "re p0"), beside the recorded ones,
// so the Value Graph, Swings and seat disagreement, the Inspector, Choices, the Model's Read and the
// status bar read it as they read an episode's. What a viewer sees is unchanged: a seat view the
// evaluations of its own decisions, the omniscient view both seats'.
//
// The cache (eval_cache.h) is read first: an entry that fits the match is merged at once, with no
// server needed. What is missing is evaluated, one evaluator job per seat with that seat's missing
// policies. When entries came from the cache and the server is to be asked, a job of no items (a
// handshake, ahead of the others) compares the versions the server serves with the entries', and a
// stale entry is evaluated again.
// A seat job's results are written to the cache and merged when the whole job has returned, so a
// cancelled or failed re-evaluation leaves neither a partial entry nor a partial series.

namespace rmwork {

// A served policy: its id on the server (rmserve's --policy ID=MODEL) and the model it serves.
struct ServedPolicy {
        uint32_t id = 0;
        std::string model;  // as Settings → Models names it (a file, or `<archive>.zip!<member>`)
};

// Settings → Models' policies with a numeric id (the ones rmserve serves), in their order.
std::vector<ServedPolicy> served_policies(const ModelSettings &m);

// Sets policy `id`'s record at each listed decision (its index, the record; the record's id is set
// to `id`), replacing the policy's record there; a decision with no episode step gets one (seat its
// deciding seat, file REEVAL_STEP_FILE, evaluations.h). Bumps Match::evaluations_changed.
void merge_evaluations(
    Match *m, int64_t id, const std::vector<std::pair<size_t, featurize::PolicyEvalRecord>> &records);

struct ReevalRequest {
        std::vector<ServedPolicy> policies;
        std::array<bool, 2> seats{{true, true}};
        std::string cache_root;  // eval_cache_root() unless a check names another
        // False: the cache alone (no server is there): what it lacks stays missing.
        bool use_server = true;
        int priority = 0;  // the evaluator jobs'
};

class Reevaluation {
    public:
        enum class State {
            IDLE,
            RUNNING,
            DONE,       // everything asked for is merged
            CANCELLED,  // nothing more was merged after the cache
            FAILED      // error(): a job failed (no_server(): no socket at the configured path)
        };
        Reevaluation() = default;
        Reevaluation(const Reevaluation &) = delete;
        Reevaluation &operator=(const Reevaluation &) = delete;
        ~Reevaluation();

        // Starts on `m` (a running re-evaluation cancelled first): the cache's fitting entries merged
        // now, the rest submitted to `ev`. `m` must not be Play's live match.
        void start(Match *m, ReplayEvaluator *ev, const ReevalRequest &r);
        // Takes what the jobs returned; a finished seat job's records written to the cache and merged
        // into `m`. A match other than the one started on cancels it. Call it every frame.
        void poll(Match *m);
        void cancel();
        State state() const { return state_; }
        bool running() const { return state_ == State::RUNNING; }
        const std::string &error() const { return error_; }
        bool no_server() const { return no_server_; }
        // Decisions evaluated of those to evaluate, over the seat jobs (-1 while only verifying).
        double fraction() const;
        // "Re-evaluating: 120 of 549 decisions", "Re-evaluation: checking the cached versions".
        std::string phase() const;
        // What it did: "Re-evaluated with re p0 v12, re p1 v3 (A, B): 2 from the cache, 2 evaluated".
        std::string summary() const;
        // Per (policy, seat) whether it came from the cache (true) or the server (false), once merged.
        const std::map<std::pair<uint32_t, int>, bool> &merged() const { return merged_; }

    private:
        struct SeatJob {
                int seat = 0;
                uint64_t id = 0;
                std::vector<ServedPolicy> policies;
                std::vector<std::vector<std::pair<size_t, featurize::PolicyEvalRecord>>> records;  // per policy
                size_t total = 0;
        };
        void submit_seat(int seat, const std::vector<ServedPolicy> &policies);
        void finish_seat(Match *m, SeatJob *j);
        void finish_verify();
        void end(State s);

        ReplayEvaluator *ev_ = nullptr;
        const Match *match_ = nullptr;
        std::string dump_;
        std::shared_ptr<const EvalSource> source_;
        ReevalRequest req_;
        State state_ = State::IDLE;
        std::string error_;
        bool no_server_ = false;
        std::vector<SeatJob> jobs_;
        uint64_t verify_ = 0;                        // the version check's job, 0 none
        std::vector<ServedPolicy> verify_policies_;  // its policies, in its order
        // What the cache gave, per (policy, seat): its version, to compare with the server's.
        std::map<std::pair<uint32_t, int>, uint64_t> cached_;
        std::map<std::pair<uint32_t, int>, bool> merged_;
        std::map<uint32_t, uint64_t> versions_;  // per policy the version merged
        std::vector<std::string> notes_;         // cache entries not written
};

}  // namespace rmwork

#endif /* RMWORK_REEVALUATION_H */
