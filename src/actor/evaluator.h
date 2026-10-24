#ifndef ACTOR_EVALUATOR_H
#define ACTOR_EVALUATOR_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../featurize/compat.h"
#include "../featurize/featurizer.h"
#include "../featurize/setup.h"

// ── The evaluator: featurized decisions → per decision its policies' logits and a value ───────
//
// docs/featurizer.md §6. An Evaluator takes a batch of featurized decisions (featurizer.h, from one
// game or many) and returns, per decision in batch order, an Evaluation: for each policy it serves
// (policies(); policy 0 is the one an actor samples from) the logits over the decision's menu rows,
// their log-softmax over those rows (log π) and the policy's version, and the value of policy 0.
// The in-process implementation serves one policy; the remote one (remote_evaluator.h,
// docs/inference_protocol.md) serves the policies its spec names, e.g. the training plan's current
// and regularisation policies in the same request, filling more entries, so callers index policies
// and never assume one.
//
// The in-process evaluator (load_model_evaluator) runs a model of the network contract on the CPU:
// the batch padded and stacked by the padding batcher (featurize/batch.h), one forward
//
//   forward(tok_type, tok_num, tok_cat, tok_card, tok_ability, tok_mask, rel, rel_mask, act_tok,
//           act_mask) -> (logits [B, A], value [B])
//
// and each decision's rows taken back out. A model is either export of tools/nn/
// make_reference_policy.py's kind, told apart by the file's contents (both are zip archives): an
// AOTInductor package (`.pt2`, an `archive_format` record) runs through
// torch::inductor::AOTIModelPackageLoader, a TorchScript module (`.ts.pt`, a `constants.pkl`
// record) through torch::jit::load. Its metadata, the strings spec_hash, vocab_hash, text_model,
// layout_hash, tables_digest, vocab (the vocab manifest, a JSON array of [id, name]:
// featurize/compat.h) and
// policy_version (an unsigned integer; "0" when absent), is the TorchScript module's string
// attributes of those names, or the package's AOTInductor metadata (the export's inductor config
// `aot_inductor.metadata`): no sidecar file. check_model refuses a model that is not of the
// featurizer's tables exactly (check_compatible, exact: an export bakes the card rows in, so it is
// rebuilt from its checkpoint for every vocab and tables).
//
// This header is torch-free (the model lives behind the implementation); the implementation is the
// only TU that includes libtorch. Every libtorch call is caught inside it: a C++ exception never
// reaches the engine's -fno-exceptions frames, failures come back as false with a message.
// Evaluation runs on one intra-op thread (actors are one process per core).

namespace actor {

// What a model carries.
struct ModelInfo {
        std::string path;
        std::string format;            // "torchscript" | "aoti"
        featurize::Manifest manifest;  // no registry (an export carries none)
        uint64_t policy_version = 0;
};

// One policy's output for one decision, over its menu rows.
struct PolicyEval {
        uint64_t version = 0;
        float value = 0;  // the policy's value, the deciding seat's view
        std::vector<float> logits;
        std::vector<float> log_pi;  // log-softmax of logits over the menu rows
};

struct Evaluation {
        float value = 0;                   // policy 0's value, the deciding seat's view
        std::vector<PolicyEval> policies;  // policy 0 first
};

class Evaluator {
    public:
        virtual ~Evaluator() = default;
        // The decisions' evaluations into `out`, in batch order. False with `err`: an actor fails
        // loudly, never falls back to a default action.
        virtual bool evaluate(
            const std::vector<const featurize::Tensors *> &batch, std::vector<Evaluation> *out, std::string *err) = 0;
        // Policies in each Evaluation.
        virtual size_t policies() const = 0;
        virtual const ModelInfo &info() const = 0;
        // Extra ` key=value` fields for the nn: agent's NN-STATS line ("" for none).
        virtual std::string stats() const { return ""; }
};

// The in-process evaluator of the model at `path` for decisions of `spec`; nullptr with `err`.
std::unique_ptr<Evaluator> load_model_evaluator(
    const std::string &path, const featurize::FeatureSpec &spec, std::string *err);

// False with `err` (naming the path and what differs) when the model is not an export of the
// featurizer's tables (featurize::check_compatible, exact).
bool check_model(const ModelInfo &model, const featurize::Featurization &f, std::string *err);

}  // namespace actor

#endif /* ACTOR_EVALUATOR_H */
