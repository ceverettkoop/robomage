#ifndef RMWORK_MODEL_READ_H
#define RMWORK_MODEL_READ_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../../src/featurize/episode.h"

// ── Model's Read while playing: the human seat's decisions evaluated through rmserve ─────────────
//
// On its own thread: the featurizer's inputs (the engine's whole-vocab cards file, written by
// `<engine> --cards-frame FILE` in the engine's working directory, and the oracle-text table under
// its resources/embeddings/), then a connection to the inference server at the configured socket
// (the torch-free remote evaluator, src/actor/remote_evaluator.h, over src/inference/protocol.h),
// asking for the policies Settings → Models lists by a numeric id (rmserve's --policy ID=MODEL),
// else policy 0. The session feeds it every record of the human seat's stream in order (feed()); it
// featurizes each decision's STATE frame with featurize_seat (the actor entry point, which refuses
// any frame but a seat's), evaluates it and hands back per decision each policy's value and log π
// over the menu rows (take()). It never sees another viewer's frame. The featurizer runs without
// recent-event history; a server whose model wants it refuses the handshake (a spec hash mismatch).
//
// A socket path with no socket behind it is "not connected" (the game goes on without it); a
// refused handshake, a server that goes away or a failing evaluation is a failure the session
// reports, after which nothing more is evaluated.

namespace rmwork {

struct ModelReadConfig {
        std::string socket;              // the inference server's socket
        std::vector<uint32_t> policies;  // the policy ids to ask for (policy 0 when empty)
        std::string engine;              // writes the cards file
        std::string workdir;             // the engine's working directory (its resources/)
        std::string cards_path;          // where the cards file goes
        double timeout_s = 15;
};

// The policy ids a server is asked for: the numeric ids of `ids` in order, else policy 0.
std::vector<uint32_t> model_read_policies(const std::vector<std::string> &ids);

struct ModelReadResult {
        size_t decision = 0;  // the live match's decision index
        std::vector<featurize::PolicyEvalRecord> policies;
};

class ModelReader {
    public:
        enum class State {
            OFF,
            CONNECTING,
            READY,
            NOT_CONNECTED,  // no server at the socket
            FAILED
        };

        ModelReader();
        ModelReader(const ModelReader &) = delete;
        ModelReader &operator=(const ModelReader &) = delete;
        ~ModelReader();

        // Starts the thread (connecting); a running reader is stopped first.
        void start(const ModelReadConfig &c);
        // One record of the human seat's stream, in order; `decision` names a STATE frame with a
        // menu that is to be evaluated, -1 for any other record.
        void feed(const std::string &payload, long decision);
        // Stops and joins the thread (the connection closed).
        void stop();
        State state() const { return state_.load(); }
        // Evaluating, or decisions waiting.
        bool thinking() const;
        // The failure or why it is not connected ("" otherwise).
        std::string error() const;
        // The results since the last take.
        std::vector<ModelReadResult> take();
        // "Model's Read: connected (2 policies)", "Model's Read: not connected", "Model's Read: off".
        std::string status_text() const;

    private:
        struct Item {
                std::string payload;
                long decision = -1;
        };
        void run();
        bool connect(std::string *why, bool *no_server);
        void fail(State s, const std::string &why);

        ModelReadConfig config_;
        std::thread thread_;
        std::atomic<State> state_{State::OFF};
        std::atomic<bool> stop_{false};
        std::atomic<bool> busy_{false};
        mutable std::mutex mu_;
        std::condition_variable cv_;
        std::deque<Item> queue_;
        std::vector<ModelReadResult> results_;
        std::string error_;
        struct Impl;
        std::unique_ptr<Impl> impl_;
};

}  // namespace rmwork

#endif /* RMWORK_MODEL_READ_H */
