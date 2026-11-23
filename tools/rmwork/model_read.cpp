#include "model_read.h"

#include <cstdlib>

#include "../../src/agents/seat_frames.h"
#include "../../src/featurize/featurizer.h"
#include "../../src/featurize/setup.h"
#include "eval_connection.h"

namespace rmwork {

// What the thread owns: the featurizer's inputs, the connection and the seat's stream.
struct ModelReader::Impl {
        featurize::Featurization features;
        std::unique_ptr<actor::Evaluator> evaluator;
        agents::SeatStream stream;
        featurize::Tensors tensors;
};

std::vector<uint32_t> model_read_policies(const std::vector<std::string> &ids) {
    std::vector<uint32_t> out;
    for (const std::string &id : ids) {
        if (id.empty() || id.size() > 10 || id.find_first_not_of("0123456789") != std::string::npos) continue;
        const unsigned long long v = std::strtoull(id.c_str(), nullptr, 10);
        if (v > 0xFFFFFFFFull) continue;
        const uint32_t p = static_cast<uint32_t>(v);
        bool seen = false;
        for (uint32_t q : out) seen = seen || q == p;
        if (!seen) out.push_back(p);
    }
    if (out.empty()) out.push_back(0);
    return out;
}

ModelReader::ModelReader() = default;

ModelReader::~ModelReader() {
    stop();
}

void ModelReader::start(const ModelReadConfig &c) {
    stop();
    config_ = c;
    if (config_.policies.empty()) config_.policies.push_back(0);
    {
        std::lock_guard<std::mutex> lock(mu_);
        queue_.clear();
        results_.clear();
        error_.clear();
    }
    stop_.store(false);
    state_.store(State::CONNECTING);
    thread_ = std::thread([this] { run(); });
}

void ModelReader::feed(const std::string &payload, long decision) {
    const State s = state_.load();
    if (s != State::CONNECTING && s != State::READY) return;
    std::lock_guard<std::mutex> lock(mu_);
    queue_.push_back({payload, decision});
    cv_.notify_one();
}

void ModelReader::stop() {
    stop_.store(true);
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    impl_.reset();
    const State s = state_.load();
    if (s == State::CONNECTING || s == State::READY) state_.store(State::OFF);
}

bool ModelReader::thinking() const {
    if (busy_.load()) return true;
    std::lock_guard<std::mutex> lock(mu_);
    for (const Item &i : queue_)
        if (i.decision >= 0) return true;
    return false;
}

std::string ModelReader::error() const {
    std::lock_guard<std::mutex> lock(mu_);
    return error_;
}

std::vector<ModelReadResult> ModelReader::take() {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<ModelReadResult> out;
    out.swap(results_);
    return out;
}

std::string ModelReader::status_text() const {
    switch (state_.load()) {
        case State::OFF:
            return "Model's Read: off";
        case State::CONNECTING:
            return "Model's Read: connecting";
        case State::READY:
            return "Model's Read: connected";
        case State::NOT_CONNECTED:
            return "Model's Read: not connected";
        case State::FAILED:
            return "Model's Read: failed";
    }
    return "";
}

void ModelReader::fail(State s, const std::string &why) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        error_ = why;
        queue_.clear();
    }
    state_.store(s);
}

// The cards file from the engine, the featurizer's inputs, the server's handshake. `no_server`:
// nothing listens at the socket.
bool ModelReader::connect(std::string *why, bool *no_server) {
    *no_server = false;
    const std::string absent = server_absent(config_.socket);
    if (!absent.empty()) {
        *no_server = true;
        *why = absent;
        return false;
    }
    impl_.reset(new Impl());
    if (!load_engine_features(config_.engine, config_.workdir, config_.cards_path, &stop_, &impl_->features, why))
        return false;
    impl_->evaluator = connect_server(
        config_.socket, config_.policies, config_.timeout_s, "rmwork Model's Read", impl_->features, why);
    return impl_->evaluator != nullptr;
}

void ModelReader::run() {
    std::string why;
    bool no_server = false;
    if (!connect(&why, &no_server)) {
        if (!stop_.load()) fail(no_server ? State::NOT_CONNECTED : State::FAILED, why);
        return;
    }
    state_.store(State::READY);
    while (true) {
        Item item;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this] { return stop_.load() || !queue_.empty(); });
            if (stop_.load()) return;
            item = std::move(queue_.front());
            queue_.pop_front();
            busy_.store(item.decision >= 0);
        }
        Impl &im = *impl_;
        std::string err;
        const uint8_t kind = item.payload.size() > 6 ? static_cast<uint8_t>(item.payload[6]) : 0;
        if (kind != static_cast<uint8_t>(rmstate::FrameKind::STATE)) {
            if (!im.stream.feed_match_frame(item.payload, &err)) {
                busy_.store(false);
                fail(State::FAILED, "a once-per-match frame: " + err);
                return;
            }
            continue;
        }
        if (item.decision < 0) continue;
        rmstate::DecodedFrame frame;
        std::vector<actor::Evaluation> evals;
        if (!im.stream.decode_state(item.payload, &frame, &err) ||
            !featurize::featurize_seat(im.features.spec, im.features.tables, frame, &im.tensors, &err) ||
            !im.evaluator->evaluate({&im.tensors}, &evals, &err) || evals.size() != 1) {
            busy_.store(false);
            fail(State::FAILED, "decision " + std::to_string(item.decision + 1) + ": " + err);
            return;
        }
        ModelReadResult r;
        r.decision = static_cast<size_t>(item.decision);
        for (size_t p = 0; p < evals[0].policies.size(); ++p) {
            const actor::PolicyEval &pe = evals[0].policies[p];
            featurize::PolicyEvalRecord rec;
            rec.id = p < config_.policies.size() ? static_cast<int64_t>(config_.policies[p]) : static_cast<int64_t>(p);
            rec.version = pe.version;
            rec.value = pe.value;
            rec.log_pi = pe.log_pi;
            r.policies.push_back(rec);
        }
        std::lock_guard<std::mutex> lock(mu_);
        results_.push_back(std::move(r));
        busy_.store(false);
    }
}

}  // namespace rmwork
