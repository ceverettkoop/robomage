#ifndef RMSERVE_SERVER_H
#define RMSERVE_SERVER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../../src/actor/evaluator.h"
#include "../../src/featurize/feature_spec.h"
#include "../../src/inference/protocol.h"

// ── rmserve's server: the inference protocol over the part-1 evaluator (docs/inference_protocol.md)
//
// One thread: a poll loop over the listening socket, the connected clients and a self-pipe the
// SIGTERM / SIGINT handler writes. Each policy is an in-process evaluator (actor/evaluator.h:
// TorchScript or .pt2, CPU) of one model; every model must be an export of the same tables (spec_hash,
// vocab_hash, text_model, layout_hash, vocab manifest), and a client's handshake must name them:
// featurize::check_compatible, exact, so an export of other tables is refused with the rebuild
// message. A client's requests queue in arrival
// order; a batch is the queued requests whose decisions fit max_batch (a larger request alone),
// evaluated when the queue reaches max_batch decisions, when every connected client has a request
// queued, or deadline_us after the oldest arrived (§5 of the protocol). Each policy runs once per
// batch over the decisions that asked for it, padded by the evaluator (featurize/batch.h). The
// spec's widths come from the first handshake (the server never reads the spec document).

namespace rmserve {

struct PolicySpec {
        uint32_t id = 0;
        std::string model;
        bool has_version = false;
        uint64_t version = 0;
};

struct Options {
        std::string socket;
        std::vector<PolicySpec> policies;
        size_t max_batch = 256;
        long deadline_us = 2000;
        int threads = 1;
        long pause_after = -1;  // tests: stop serving after this many responses (-1 never)
        bool verbose = false;
};

class Server {
    public:
        Server();
        ~Server();
        // Loads the models and opens the socket; false with `err`.
        bool start(const Options &opt, std::string *err);
        // Serves until SIGTERM or SIGINT; the exit status.
        int run();

    private:
        struct Policy {
                uint32_t id = 0;
                uint64_t version = 0;
                std::unique_ptr<actor::Evaluator> evaluator;
        };
        struct Client {
                int fd = -1;
                uint64_t number = 0;
                std::string name, inbuf;
                bool welcomed = false, pending = false;
                inference::SharedRegion region;
                size_t request_bytes = 0, response_bytes = 0;
                uint64_t spec_hash = 0;
                std::vector<uint32_t> policies;
                std::vector<size_t> policy_index;  // into policies_
                size_t requests = 0;
        };
        struct Queued {
                Client *client = nullptr;
                inference::Request request;
                double arrived = 0;
        };

        void accept_clients();
        // Reads what the client sent and handles each whole message; false when it was dropped.
        bool read_client(Client *c);
        bool handle(Client *c, const inference::Message &m);
        bool hello(Client *c, const std::string &body);
        bool eval(Client *c, const std::string &body);
        // Sends ERROR, logs it and drops the client; false.
        bool refuse(Client *c, const std::string &code, const std::string &message);
        void drop(Client *c);
        bool ready(double now) const;
        // Evaluates the next batch of queued requests and answers them.
        void flush();
        void answer(Queued &q, std::vector<std::vector<actor::Evaluation>> &by_policy, std::vector<size_t> &cursor,
            uint32_t batch, uint32_t compute_us);
        const Policy *find_policy(uint32_t id, size_t *index) const;
        double next_timeout_ms(double now) const;
        void log(const std::string &line) const;
        void print_stats() const;

        Options opt_;
        featurize::FeatureSpec spec_;  // only the widths (F, C, K, Q), from the first handshake
        bool have_dims_ = false;
        featurize::Manifest tables_;  // the models' (the first policy's)
        std::vector<Policy> policies_;
        int listen_fd_ = -1;
        std::vector<std::unique_ptr<Client>> clients_;
        std::vector<Queued> queue_;
        uint64_t next_client_ = 0;
        bool paused_ = false;
        // Totals for the RMSERVE-STATS line.
        size_t clients_seen_ = 0, requests_ = 0, decisions_ = 0, batches_ = 0, max_seen_ = 0, refused_ = 0;
        double compute_s_ = 0, started_ = 0;
};

}  // namespace rmserve

#endif /* RMSERVE_SERVER_H */
