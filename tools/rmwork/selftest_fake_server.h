#ifndef RMWORK_SELFTEST_FAKE_SERVER_H
#define RMWORK_SELFTEST_FAKE_SERVER_H

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../../src/featurize/featurizer.h"

// ── A stand-in inference server for the self-tests (re-evaluation's and Attribution's checks) ────
//
// One connection at a time on its own thread, speaking RMIP v2 (the handshake, the shared region,
// requests and responses). A policy's value of a decision is a function of the decision's tensors
// (value_of: its token count and a weighted sum of its numeric, categorical and card columns), and its
// logits a function of the row and of what the row's menu token relates to (logits_of), so the checks
// know what each decision's own tensors, changed or not, must give.

namespace rmwork {

class FakeServer {
    public:
        struct Options {
                uint64_t max_batch = 7;
                std::map<uint32_t, uint64_t> versions;  // per policy id (1 for the others)
                int delay_ms = 0;                       // before each answer
                bool refuse = false;                    // an ERROR to every HELLO
                long lose_after = -1;                   // requests answered before the connection drops
        };
        struct Counts {
                size_t hellos = 0, requests = 0, decisions = 0, largest = 0;
                std::vector<uint32_t> last_policies;  // the last HELLO's
        };

        ~FakeServer() { stop(); }
        bool start(const std::string &path, const Options &o);
        void stop();
        void set(const Options &o);
        Counts counts() const;
        // A decision's value and logits by policy `policy`.
        static float value_of(const featurize::Tensors &t, uint32_t policy);
        static std::vector<float> logits_of(const featurize::Tensors &t, uint32_t policy);

    private:
        void run();
        void serve(int c);

        int fd_ = -1;
        std::string path_;
        std::thread thread_;
        std::atomic<bool> stop_{false};
        mutable std::mutex mu_;
        Options opt_;
        Counts counts_;
        long answered_ = 0;
};

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_FAKE_SERVER_H */
