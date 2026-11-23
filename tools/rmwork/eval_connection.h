#ifndef RMWORK_EVAL_CONNECTION_H
#define RMWORK_EVAL_CONNECTION_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../../src/actor/evaluator.h"
#include "../../src/featurize/setup.h"

// ── What an evaluation through the inference server needs: the featurizer's inputs, a connection ──
//
// Shared by Model's Read (model_read.h), the Replay evaluator (replay_evaluator.h) and the Models
// tool's compatibility check (model_manager.h), all off the UI thread.
//   - The featurizer's inputs from the engine: its whole-vocab cards file, written by
//     `<engine> --cards-frame FILE` in the engine's working directory, and the oracle-text table
//     under that directory's resources/embeddings/, loaded into a featurize::Featurization.
//   - A connection: the torch-free remote evaluator (src/actor/remote_evaluator.h) to the server at
//     a Unix socket, for a list of numeric policy ids (rmserve's --policy ID=MODEL). A socket path
//     with no socket behind it is "no server", told apart from a refused handshake or another
//     failure.

namespace rmwork {

// The oracle-text table under an engine working directory.
std::string oracle_text_table(const std::string &workdir);

// Writes the cards file at `cards_path` (its stderr beside it, `.err`) and loads it with the text
// table into `out`. `stop` (may be null) abandons the wait for the engine. False with `why` ("the
// cards file: …", "the featurizer's inputs: …"; "" when stopped).
bool load_engine_features(const std::string &engine, const std::string &workdir, const std::string &cards_path,
    const std::atomic<bool> *stop, featurize::Featurization *out, std::string *why);

// "" when a socket is at `socket`, else why not ("no server socket is configured (…)", "no server at …").
std::string server_absent(const std::string &socket);

// A connection to the server at `socket` for `policies` (in order) and decisions of `features`;
// nullptr with `why` (the server's refusal, a spec hash mismatch, no answer within `timeout_s`).
std::unique_ptr<actor::Evaluator> connect_server(const std::string &socket, const std::vector<uint32_t> &policies,
    double timeout_s, const std::string &client, const featurize::Featurization &features, std::string *why);

}  // namespace rmwork

#endif /* RMWORK_EVAL_CONNECTION_H */
