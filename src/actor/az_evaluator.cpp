#include "az_evaluator.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>  // pull in std::mt19937 (used by the engine's game.h) before torch's ATen headers define at::mt19937
#include <sstream>

// Engine headers BEFORE torch: gamestate.h -> game.h needs std::mt19937 resolved
// in the ordinary std context; including torch first shadows it with at::mt19937.
#include "obs_builder.h"  // ACTOR_OBS_SIZE
#include "classes/gamestate.h"  // MAX_ACTIONS
#include "machine_io.h"         // N_CARD_TYPES

#include <ATen/Parallel.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/cuda.h>
#include <torch/script.h>

// Any libtorch error path funnels here: report + hard exit so no C++ exception
// unwinds into the -fno-exceptions engine frames that call the input provider.
[[noreturn]] static void torch_fatal(const std::string& msg) {
    std::fprintf(stderr, "FATAL (az_actor/torch): %s\n", msg.c_str());
    std::exit(1);
}

// Minimal scalar extractor for the flat `.ts.meta.json` handshake dict (values
// are plain integers). Avoids pulling in a JSON dependency for three fields.
static bool json_int_field(const std::string& text, const std::string& key, long& out) {
    std::string needle = "\"" + key + "\"";
    size_t p = text.find(needle);
    if (p == std::string::npos) return false;
    p = text.find(':', p + needle.size());
    if (p == std::string::npos) return false;
    p++;
    while (p < text.size() && (text[p] == ' ' || text[p] == '\t' || text[p] == '\n' || text[p] == '\r'))
        p++;
    size_t start = p;
    while (p < text.size() && (text[p] == '-' || (text[p] >= '0' && text[p] <= '9'))) p++;
    if (p == start) return false;
    out = std::strtol(text.substr(start, p - start).c_str(), nullptr, 10);
    return true;
}

struct AZEvaluator::Impl {
    torch::jit::Module module;
    torch::Device device{torch::kCPU};
    bool loaded = false;
};

// Move a [k, OBS] CPU batch to the eval device. On cuda the staging copy goes
// through pinned memory so the host->device transfer can be async; on cpu this
// is the identity (`.to(cpu)` on a cpu tensor returns the same tensor).
static torch::Tensor to_eval_device(const torch::Tensor& t, const torch::Device& device) {
    if (device.is_cpu()) return t;
    return t.pin_memory().to(device, /*non_blocking=*/true);
}

AZEvaluator::AZEvaluator() : impl_(std::make_unique<Impl>()) {}
AZEvaluator::~AZEvaluator() = default;

void AZEvaluator::load(const std::string& path, const std::string& device) {
    // Single-threaded ATen for bit-parity with the Python reference driver
    // (torch.set_num_threads(1)); multithreaded reductions are not bit-stable.
    at::set_num_threads(1);
    // set_num_interop_threads throws if the interop pool is already initialized;
    // best-effort so a second load() (or prior parallel work) is not fatal.
    try {
        at::set_num_interop_threads(1);
    } catch (const std::exception&) {
    }

    // Verify the sibling meta handshake first (cheap, and gives a clearer error
    // than a shape mismatch mid-forward).
    std::string meta_path = path;
    const std::string ts_suffix = ".ts.pt";
    if (meta_path.size() >= ts_suffix.size() &&
        meta_path.compare(meta_path.size() - ts_suffix.size(), ts_suffix.size(), ts_suffix) == 0) {
        meta_path = meta_path.substr(0, meta_path.size() - ts_suffix.size()) + ".ts.meta.json";
    } else {
        meta_path += ".meta.json";
    }
    std::ifstream mf(meta_path);
    if (mf) {
        std::stringstream ss;
        ss << mf.rdbuf();
        std::string text = ss.str();
        struct { const char* key; long expect; } checks[] = {
            {"OBS_SIZE", ACTOR_OBS_SIZE},
            {"MAX_ACTIONS", MAX_ACTIONS},
            {"N_CARD_TYPES", N_CARD_TYPES},
        };
        for (const auto& c : checks) {
            long got = 0;
            if (json_int_field(text, c.key, got) && got != c.expect) {
                torch_fatal(std::string("TorchScript meta mismatch: ") + c.key + "=" +
                            std::to_string(got) + " but engine build has " +
                            std::to_string(c.expect) + " (" + meta_path + ")");
            }
        }
    } else {
        std::fprintf(stderr,
                     "WARNING (az_actor): no meta sidecar %s — skipping layout handshake\n",
                     meta_path.c_str());
    }

    try {
        impl_->device = torch::Device(device);
    } catch (const std::exception& e) {
        torch_fatal(std::string("invalid --device '") + device + "': " + e.what());
    }
    if (!impl_->device.is_cpu()) {
        if (!impl_->device.is_cuda())
            torch_fatal("unsupported --device '" + device + "' (use cpu or cuda)");
        if (!torch::cuda::is_available())
            torch_fatal("--device " + device + " requested but torch reports no cuda/HIP "
                        "device (ROCm: is HSA_OVERRIDE_GFX_VERSION=10.3.0 exported, and is "
                        "the venv torch the rocm build?)");
        // The TorchScript NNC/TensorExpr fuser's IRSimplifier grinds for minutes
        // (looks like a hang; gdb-verified) recompiling this module's gather-heavy
        // graph for the cuda device. The net is GEMM-dominated, so elementwise
        // fusion buys nothing here — skip the fuser entirely on the GPU path.
        // CPU loads never reach this branch, so the bit-parity envelope of the
        // default path is untouched.
        torch::jit::setTensorExprFuserEnabled(false);
    }

    try {
        impl_->module = torch::jit::load(path);
        impl_->module.eval();
        if (!impl_->device.is_cpu()) impl_->module.to(impl_->device);
        impl_->loaded = true;
    } catch (const std::exception& e) {
        torch_fatal(std::string("failed to load TorchScript module ") + path + ": " + e.what());
    }
}

AZEvalResult AZEvaluator::evaluate(const float* obs, int num_choices) {
    if (!impl_->loaded) torch_fatal("evaluate() before a successful load()");
    if (num_choices < 1 || num_choices > MAX_ACTIONS)
        torch_fatal("evaluate(): num_choices out of range: " + std::to_string(num_choices));

    AZEvalResult result;
    try {
        torch::NoGradGuard no_grad;
        // from_blob does not copy; the buffer outlives the forward call here.
        auto obs_t = torch::from_blob(const_cast<float*>(obs), {1, ACTOR_OBS_SIZE},
                                      torch::kFloat32);
        auto mask_t = torch::zeros({1, MAX_ACTIONS}, torch::kBool);
        auto mask_a = mask_t.accessor<bool, 2>();
        for (int i = 0; i < num_choices; i++) mask_a[0][i] = true;

        std::vector<torch::jit::IValue> inputs{to_eval_device(obs_t, impl_->device),
                                               to_eval_device(mask_t, impl_->device)};
        auto out = impl_->module.forward(inputs).toTuple();
        // [1, MAX_ACTIONS] / [1]; copied back to CPU so the prior math below is
        // device-independent (a no-op when the forward ran on CPU).
        torch::Tensor logits = out->elements()[0].toTensor().to(torch::kCPU);
        torch::Tensor value = out->elements()[1].toTensor().to(torch::kCPU);

        // softmax over the first num_choices logits (mirrors AZEvaluator in az_net.py).
        auto slice = logits.index({0, torch::indexing::Slice(0, num_choices)});
        auto probs = torch::softmax(slice, -1).contiguous();
        auto pa = probs.accessor<float, 1>();
        result.priors.resize(static_cast<size_t>(num_choices));
        for (int i = 0; i < num_choices; i++) result.priors[static_cast<size_t>(i)] = pa[i];
        result.value = value.item<float>();
    } catch (const std::exception& e) {
        torch_fatal(std::string("forward() failed: ") + e.what());
    }
    return result;
}

// Mirror train/az_net.py::AZEvaluator.evaluate for one decision: run the net,
// softmax the first num_choices logits in float32, widen to float64, renormalize
// by the float64 sum (uniform fallback on a non-finite/zero total). The value is
// the float32 tanh head output widened to double.
static AZEvalResultD normalize_double(const float* probs32, int num_choices, float value32) {
    AZEvalResultD r;
    r.priors.resize(static_cast<size_t>(num_choices));
    double total = 0.0;
    for (int i = 0; i < num_choices; i++) {
        double p = static_cast<double>(probs32[static_cast<size_t>(i)]);
        r.priors[static_cast<size_t>(i)] = p;
        total += p;
    }
    if (!std::isfinite(total) || total <= 0.0) {
        double u = 1.0 / static_cast<double>(num_choices);
        for (int i = 0; i < num_choices; i++) r.priors[static_cast<size_t>(i)] = u;
    } else {
        for (int i = 0; i < num_choices; i++) r.priors[static_cast<size_t>(i)] /= total;
    }
    r.value = static_cast<double>(value32);
    return r;
}

AZEvalResultD AZEvaluator::evaluate_double(const float* obs, int num_choices) {
    AZEvalResult f = evaluate(obs, num_choices);  // float32 softmax + value
    return normalize_double(f.priors.data(), num_choices, f.value);
}

std::vector<AZEvalResultD> AZEvaluator::evaluate_double_batch(
    const float* obs, const std::vector<int>& num_choices) {
    if (!impl_->loaded) torch_fatal("evaluate_double_batch() before a successful load()");
    const int k = static_cast<int>(num_choices.size());
    std::vector<AZEvalResultD> out;
    out.reserve(static_cast<size_t>(k));
    if (k == 0) return out;

    try {
        torch::NoGradGuard no_grad;
        auto obs_t = torch::from_blob(const_cast<float*>(obs), {k, ACTOR_OBS_SIZE},
                                      torch::kFloat32);
        auto mask_t = torch::zeros({k, MAX_ACTIONS}, torch::kBool);
        auto mask_a = mask_t.accessor<bool, 2>();
        for (int b = 0; b < k; b++) {
            int nc = num_choices[static_cast<size_t>(b)];
            if (nc < 1 || nc > MAX_ACTIONS)
                torch_fatal("evaluate_double_batch(): num_choices out of range");
            for (int i = 0; i < nc; i++) mask_a[b][i] = true;
        }
        std::vector<torch::jit::IValue> inputs{to_eval_device(obs_t, impl_->device),
                                               to_eval_device(mask_t, impl_->device)};
        auto res = impl_->module.forward(inputs).toTuple();
        // [k, MAX_ACTIONS] / [k], copied back to CPU (no-op for a CPU forward).
        torch::Tensor logits = res->elements()[0].toTensor().to(torch::kCPU);
        torch::Tensor value = res->elements()[1].toTensor().to(torch::kCPU);
        auto va = value.contiguous();
        auto vacc = va.accessor<float, 1>();
        for (int b = 0; b < k; b++) {
            int nc = num_choices[static_cast<size_t>(b)];
            auto slice = logits.index({b, torch::indexing::Slice(0, nc)});
            auto probs = torch::softmax(slice, -1).contiguous();
            auto pa = probs.accessor<float, 1>();
            std::vector<float> row(static_cast<size_t>(nc));
            for (int i = 0; i < nc; i++) row[static_cast<size_t>(i)] = pa[i];
            out.push_back(normalize_double(row.data(), nc, vacc[b]));
        }
    } catch (const std::exception& e) {
        torch_fatal(std::string("evaluate_double_batch forward() failed: ") + e.what());
    }
    return out;
}

int AZEvaluator::argmax_action(const float* obs, int num_choices) {
    AZEvalResult r = evaluate(obs, num_choices);
    int best = 0;
    float best_v = r.priors.empty() ? 0.0f : r.priors[0];
    for (int i = 1; i < num_choices; i++) {
        if (r.priors[static_cast<size_t>(i)] > best_v) {
            best_v = r.priors[static_cast<size_t>(i)];
            best = i;
        }
    }
    return best;
}
