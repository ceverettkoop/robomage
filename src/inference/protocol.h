#ifndef INFERENCE_PROTOCOL_H
#define INFERENCE_PROTOCOL_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../featurize/featurizer.h"

// ── The inference protocol's wire code (docs/inference_protocol.md, RMIP v1) ──────────────────
//
// Standard library and POSIX only (sockets, shm_open, mmap; no exceptions): what the remote
// evaluator (actor/remote_evaluator.h), rmserve (tools/rmserve/) and the gate's protocol checks
// share. The control messages over a Unix stream socket (send_message / recv_message, and
// take_message for a server's buffered reads), the handshake's JSON (Hello, Welcome, the ERROR
// body), the shared region (SharedRegion) and the request and response layouts in it
// (encode_request / decode_request, encode_response / decode_response).

namespace inference {

constexpr uint32_t PROTOCOL_VERSION = 1;
// A control message's body is at most this long.
constexpr uint32_t MAX_BODY = 1u << 20;

enum MessageType : uint16_t {
    MSG_HELLO = 1,
    MSG_WELCOME = 2,
    MSG_EVAL = 3,
    MSG_DONE = 4,
    MSG_ERROR = 5,
    MSG_BYE = 6
};

struct Message {
        uint16_t type = 0;
        std::string body;
};

enum class RecvStatus {
    OK,
    CLOSED,     // the peer closed the connection (or reset it)
    TIMEOUT,    // nothing complete within the timeout
    MALFORMED,  // not a message of this protocol
    FAILED      // a socket error (`err` says which)
};

// The spec's widths a request's tensors have.
struct Dims {
        uint32_t F = 0, C = 0, K = 0, Q = 0;
        bool operator==(const Dims &o) const { return F == o.F && C == o.C && K == o.K && Q == o.Q; }
};

struct Hello {
        uint32_t protocol = PROTOCOL_VERSION;
        std::string client;
        std::string spec_hash, vocab_hash, text_model;
        Dims dims;
        std::string shm;
        uint64_t request_bytes = 0, response_bytes = 0;
        std::vector<uint32_t> policies;
};

struct WelcomePolicy {
        uint32_t id = 0;
        uint64_t version = 0;
};

struct Welcome {
        uint32_t protocol = PROTOCOL_VERSION;
        std::string server;
        uint64_t max_batch = 0;
        std::vector<WelcomePolicy> policies;
};

struct Request {
        uint64_t spec_hash = 0, seq = 0;
        Dims dims;
        std::vector<uint32_t> policies;
        std::vector<featurize::Tensors> decisions;
};

// One policy's results over a request's decisions: value [D], logits and log_pi over the decisions'
// menu rows concatenated.
struct PolicyResult {
        uint32_t id = 0;
        uint64_t version = 0;
        std::vector<float> value, logits, log_pi;
};

struct Response {
        uint64_t spec_hash = 0, seq = 0;
        uint32_t batch = 0, compute_us = 0;
        std::vector<PolicyResult> policies;
};

// ── control messages ──

// One message, the whole of it (MSG_NOSIGNAL: a closed peer is false, not SIGPIPE).
bool send_message(int fd, uint16_t type, const std::string &body, std::string *err);
// One message within `timeout_s` seconds (<= 0: no limit).
RecvStatus recv_message(int fd, double timeout_s, Message *out, std::string *err);
// The first whole message of `buffer` (bytes read so far), removed from it: true with it, false
// when it holds none yet (`malformed` set when it never will: a bad magic or length).
bool take_message(std::string *buffer, Message *out, bool *malformed);

// The JSON bodies. parse_* are false with `err` on a body that is not the layout.
std::string hello_json(const Hello &h);
bool parse_hello(const std::string &body, Hello *out, std::string *err);
std::string welcome_json(const Welcome &w);
bool parse_welcome(const std::string &body, Welcome *out, std::string *err);
std::string error_json(const std::string &code, const std::string &message);
// An ERROR body's code and message (the raw body as the message when it is not JSON).
void parse_error(const std::string &body, std::string *code, std::string *message);
// EVAL's and DONE's body: (seq, length).
std::string seq_body(uint64_t seq, uint64_t length);
bool parse_seq_body(const std::string &body, uint64_t *seq, uint64_t *length);

// ── Unix sockets ──

// A connected stream socket to `path`; -1 with `err`.
int connect_unix(const std::string &path, std::string *err);
// A listening socket at `path`: bound under a temporary name and renamed onto `path` once
// listening, so a client never finds the path before it accepts; -1 with `err`.
int listen_unix(const std::string &path, int backlog, std::string *err);

// ── the shared region ──

class SharedRegion {
    public:
        SharedRegion() = default;
        SharedRegion(const SharedRegion &) = delete;
        SharedRegion &operator=(const SharedRegion &) = delete;
        ~SharedRegion();
        // A new region of `size` bytes under a fresh name (O_EXCL) starting with `prefix`.
        bool create(const std::string &prefix, size_t size, std::string *err);
        // The region `name`, mapped read-write; false when it is smaller than `size`.
        bool open(const std::string &name, size_t size, std::string *err);
        // Removes the name (the mapping stays).
        void unlink();
        void close();
        uint8_t *data() const { return data_; }
        size_t size() const { return size_; }
        const std::string &name() const { return name_; }

    private:
        uint8_t *data_ = nullptr;
        size_t size_ = 0;
        std::string name_;
        bool linked_ = false;
};

// ── requests and responses ──

// The bytes a request of `batch` takes.
size_t request_size(const Dims &dims, size_t policies, const std::vector<const featurize::Tensors *> &batch);
// Writes a request at dst[0, cap); false with `err` when it does not fit or a decision's tensors
// are not of `dims`.
bool encode_request(uint8_t *dst, size_t cap, uint64_t spec_hash, uint64_t seq, const Dims &dims,
    const std::vector<uint32_t> &policies, const std::vector<const featurize::Tensors *> &batch, size_t *length,
    std::string *err);
// Reads a request from src[0, cap) (copies: the decisions are its own); false with `err` naming
// what is not the layout (docs/inference_protocol.md §4, §6 `request`).
bool decode_request(const uint8_t *src, size_t cap, Request *out, std::string *err);

// The bytes a response takes, for decisions of `menu_rows` total rows.
size_t response_size(size_t decisions, size_t policies, size_t menu_rows);
// Writes a response at dst[0, cap) for decisions with `menu_rows` rows each.
bool encode_response(uint8_t *dst, size_t cap, const Response &r, const std::vector<size_t> &menu_rows, size_t *length,
    std::string *err);
// Reads a response for decisions with `menu_rows` rows each from src[0, cap).
bool decode_response(
    const uint8_t *src, size_t cap, const std::vector<size_t> &menu_rows, Response *out, std::string *err);

// log-softmax of `logits` over its entries (in double, as the in-process evaluator computes it).
void log_softmax(const float *logits, size_t n, float *out);

}  // namespace inference

#endif /* INFERENCE_PROTOCOL_H */
