#include "evaluator.h"

#include <ATen/Parallel.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/script.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <utility>

#include "../featurize/batch.h"
#include "../featurize/rmtb.h"
#include "../inference/protocol.h"

namespace {

// The batcher's pad multiple (featurize/batch.h): no padded dimension is 0 or 1.
const size_t PAD_MULTIPLE = 8;
const char *const META_KEYS[] = {
    "spec_hash", "vocab_hash", "text_model", "policy_version", "layout_hash", "tables_digest", "vocab"};
// The keys an export made before model manifests lacks.
const char *const OPTIONAL_KEYS[] = {"policy_version", "layout_hash", "tables_digest", "vocab"};

// The in-process evaluator: one TorchScript module or AOTInductor package on the CPU.
class ModelEvaluator : public actor::Evaluator {
    public:
        ModelEvaluator(actor::ModelInfo info, const featurize::FeatureSpec &spec)
            : info_(std::move(info)), spec_(spec) {}
        bool load(std::string *err);
        bool evaluate(const std::vector<const featurize::Tensors *> &batch, std::vector<actor::Evaluation> *out,
            std::string *err) override;
        size_t policies() const override { return 1; }
        const actor::ModelInfo &info() const override { return info_; }

    private:
        // The model's outputs for the padded batch (logits [B, A], value [B], float, contiguous).
        void forward(torch::Tensor *logits, torch::Tensor *value);
        bool read_metadata(const std::unordered_map<std::string, std::string> &meta, std::string *err);

        actor::ModelInfo info_;
        const featurize::FeatureSpec &spec_;
        std::unique_ptr<torch::jit::Module> module_;
        std::unique_ptr<torch::inductor::AOTIModelPackageLoader> package_;
        featurize::PaddedBatch padded_;
};

}  // namespace

static void single_thread();
static std::string model_format(const std::string &path, std::string *err);
static torch::Tensor long_tensor(const std::vector<int64_t> &v, std::initializer_list<int64_t> shape);
static torch::Tensor bool_tensor(const std::vector<uint8_t> &v, std::initializer_list<int64_t> shape);

// One intra-op and one inter-op thread, set once per process (the inter-op count can only be set
// before any inter-op work; a second call throws, which is ignored).
static void single_thread() {
    at::set_num_threads(1);
    try {
        at::set_num_interop_threads(1);
    } catch (const std::exception &) {
    }
}

// "aoti" or "torchscript" by the archive's records; "" with `err` for anything else.
static std::string model_format(const std::string &path, std::string *err) {
    try {
        caffe2::serialize::PyTorchStreamReader reader(path);
        if (reader.hasRecord("archive_format")) return "aoti";
        if (reader.hasRecord("constants.pkl")) return "torchscript";
        *err = path + ": a zip archive that is neither an AOTInductor package nor a TorchScript module";
    } catch (const std::exception &e) {
        *err = path + ": not a model archive (" + std::string(e.what()).substr(0, 200) + ")";
    }
    return "";
}

static torch::Tensor long_tensor(const std::vector<int64_t> &v, std::initializer_list<int64_t> shape) {
    return torch::from_blob(const_cast<int64_t *>(v.data()), shape, torch::kInt64);
}

static torch::Tensor bool_tensor(const std::vector<uint8_t> &v, std::initializer_list<int64_t> shape) {
    return torch::from_blob(const_cast<uint8_t *>(v.data()), shape, torch::kUInt8).to(torch::kBool);
}

bool ModelEvaluator::read_metadata(const std::unordered_map<std::string, std::string> &meta, std::string *err) {
    for (const char *key : META_KEYS) {
        const std::string k = key;
        auto it = meta.find(key);
        if (it == meta.end()) {
            if (std::find(std::begin(OPTIONAL_KEYS), std::end(OPTIONAL_KEYS), k) != std::end(OPTIONAL_KEYS)) continue;
            *err = info_.path + ": the model carries no " + key;
            return false;
        }
        featurize::Manifest &m = info_.manifest;
        if (k == "spec_hash") m.spec_hash = it->second;
        if (k == "vocab_hash") m.vocab_hash = it->second;
        if (k == "text_model") m.text_model = it->second;
        if (k == "layout_hash") m.layout_hash = it->second;
        if (k == "tables_digest") m.tables_digest = it->second;
        if (k == "vocab" && !featurize::parse_vocab_json(it->second, &m.vocab, err)) {
            *err = info_.path + ": its vocab metadata: " + *err;
            return false;
        }
        if (k == "policy_version") {
            char *end = nullptr;
            info_.policy_version = std::strtoull(it->second.c_str(), &end, 10);
            if (it->second.empty() || *end != '\0') {
                *err = info_.path + ": policy_version '" + it->second + "' is not an unsigned integer";
                return false;
            }
        }
    }
    return true;
}

bool ModelEvaluator::load(std::string *err) {
    single_thread();
    info_.format = model_format(info_.path, err);
    if (info_.format.empty()) return false;
    std::unordered_map<std::string, std::string> meta;
    try {
        if (info_.format == "aoti") {
            package_.reset(new torch::inductor::AOTIModelPackageLoader(info_.path, "model", true));
            meta = package_->get_metadata();
        } else {
            module_.reset(new torch::jit::Module(torch::jit::load(info_.path, torch::kCPU)));
            module_->eval();
            for (const char *key : META_KEYS)
                if (module_->hasattr(key)) meta[key] = module_->attr(key).toStringRef();
        }
    } catch (const std::exception &e) {
        *err = info_.path + ": " + std::string(e.what()).substr(0, 400);
        return false;
    }
    return read_metadata(meta, err);
}

void ModelEvaluator::forward(torch::Tensor *logits, torch::Tensor *value) {
    const featurize::PaddedBatch &p = padded_;
    const int64_t B = static_cast<int64_t>(p.B), N = static_cast<int64_t>(p.N), R = static_cast<int64_t>(p.R),
                  A = static_cast<int64_t>(p.A);
    std::vector<torch::Tensor> in = {long_tensor(p.tok_type, {B, N}),
        torch::from_blob(const_cast<float *>(p.tok_num.data()), {B, N, static_cast<int64_t>(p.F)}, torch::kFloat),
        long_tensor(p.tok_cat, {B, N, static_cast<int64_t>(p.C)}),
        long_tensor(p.tok_card, {B, N, static_cast<int64_t>(p.K)}),
        long_tensor(p.tok_ability, {B, N, static_cast<int64_t>(p.Q)}), bool_tensor(p.tok_mask, {B, N}),
        long_tensor(p.rel, {B, R, 3}), bool_tensor(p.rel_mask, {B, R}), long_tensor(p.act_tok, {B, A}),
        bool_tensor(p.act_mask, {B, A})};
    c10::InferenceMode guard;
    if (package_) {
        std::vector<torch::Tensor> out = package_->run(in);
        *logits = out.at(0).to(torch::kFloat).contiguous();
        *value = out.at(1).to(torch::kFloat).contiguous();
        return;
    }
    std::vector<torch::jit::IValue> args(in.begin(), in.end());
    const auto tuple = module_->forward(args).toTuple();
    *logits = tuple->elements().at(0).toTensor().to(torch::kFloat).contiguous();
    *value = tuple->elements().at(1).toTensor().to(torch::kFloat).contiguous();
}

bool ModelEvaluator::evaluate(
    const std::vector<const featurize::Tensors *> &batch, std::vector<actor::Evaluation> *out, std::string *err) {
    out->clear();
    if (batch.empty()) return true;
    featurize::pad_batch(spec_, batch, PAD_MULTIPLE, &padded_);
    torch::Tensor logits, value;
    try {
        forward(&logits, &value);
    } catch (const std::exception &e) {
        *err = info_.path + ": evaluation failed: " + std::string(e.what()).substr(0, 400);
        return false;
    }
    const int64_t B = static_cast<int64_t>(padded_.B), A = static_cast<int64_t>(padded_.A);
    if (logits.dim() != 2 || logits.size(0) != B || logits.size(1) != A || value.dim() != 1 || value.size(0) != B) {
        *err = info_.path + ": the model's outputs are not logits [B, A] and value [B]";
        return false;
    }
    const float *lg = logits.data_ptr<float>(), *vl = value.data_ptr<float>();
    out->resize(batch.size());
    for (size_t b = 0; b < batch.size(); ++b) {
        actor::Evaluation &e = (*out)[b];
        e.value = vl[b];
        e.policies.resize(1);
        actor::PolicyEval &pe = e.policies[0];
        pe.version = info_.policy_version;
        pe.value = e.value;
        const size_t rows = padded_.menu_rows[b];
        pe.logits.assign(lg + b * padded_.A, lg + b * padded_.A + rows);
        for (float l : pe.logits)
            if (!std::isfinite(l)) {
                *err = info_.path + ": a non-finite logit";
                return false;
            }
        pe.log_pi.resize(rows);
        inference::log_softmax(pe.logits.data(), rows, pe.log_pi.data());
    }
    return true;
}

namespace actor {

std::unique_ptr<Evaluator> load_model_evaluator(
    const std::string &path, const featurize::FeatureSpec &spec, std::string *err) {
    ModelInfo info;
    info.path = path;
    std::unique_ptr<ModelEvaluator> e(new ModelEvaluator(std::move(info), spec));
    if (!e->load(err)) return nullptr;
    return e;
}

bool check_model(const ModelInfo &model, const featurize::Featurization &f, std::string *err) {
    const featurize::Refusal r = featurize::check_compatible(model.manifest, f.manifest(), true);
    if (!r.refused()) return true;
    *err = model.path + ": " + r.message;
    return false;
}

}  // namespace actor
