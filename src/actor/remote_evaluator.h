#ifndef ACTOR_REMOTE_EVALUATOR_H
#define ACTOR_REMOTE_EVALUATOR_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "evaluator.h"

// ── The remote evaluator: the inference protocol's client (docs/inference_protocol.md) ────────
//
// An Evaluator whose evaluations come from an inference server (rmserve, the training plan's GPU
// server) at a Unix socket: connect, a shared region of its own (POSIX shared memory, unlinked once
// the server has it), the handshake (the featurizer's spec_hash, vocab_hash, text_model and widths,
// the policies to evaluate), then per evaluate() one request: the batch's unpadded tensors written
// into the region, EVAL, and the server's response read back. Each Evaluation holds the requested
// policies in the requested order with each one's version as the response gives it (so weights
// swapped between decisions show), and `value` is the first's. Torch-free (the protocol code,
// inference/protocol.h, is standard library and POSIX).
//
// Every failure is fatal for the caller: an ERROR from the server (a refused handshake, a spec
// hash mismatch, a failed model), the server closing the connection or going away, a response that
// is not the request's, or no answer within the timeout come back as false with a message naming
// the endpoint; there is no fallback.

namespace actor {

struct RemoteOptions {
        std::string endpoint;               // the server's socket path
        std::vector<uint32_t> policies{0};  // the policy ids to evaluate, the first the one sampled from
        double timeout_s = 60;              // the longest wait for the handshake's or a request's answer
        size_t request_bytes = 32u << 20;   // the region's request area
        std::string client;                 // the name the server logs
};

// Connects to the server and completes the handshake for decisions of `f`; nullptr with `err`.
std::unique_ptr<Evaluator> connect_remote_evaluator(
    const RemoteOptions &opt, const featurize::Featurization &f, std::string *err);

}  // namespace actor

#endif /* ACTOR_REMOTE_EVALUATOR_H */
