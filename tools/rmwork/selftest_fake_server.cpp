#include "selftest_fake_server.h"

#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>

#include "../../src/inference/protocol.h"

static double content_sum(const featurize::Tensors &t);

// A weighted sum of every token's numeric, categorical and card columns, in token order.
static double content_sum(const featurize::Tensors &t) {
    double s = 0;
    for (size_t i = 0; i < t.tok_num.size(); ++i)
        s += static_cast<double>(t.tok_num[i]) * (1 + static_cast<double>(i % 7)) * 1e-2;
    for (size_t i = 0; i < t.tok_cat.size(); ++i) s += static_cast<double>(t.tok_cat[i]) * 1e-4;
    for (size_t i = 0; i < t.tok_card.size(); ++i) s += static_cast<double>(t.tok_card[i] + 1) * 1e-3;
    return s;
}

namespace rmwork {

bool FakeServer::start(const std::string &path, const Options &o) {
    std::string err;
    fd_ = inference::listen_unix(path, 4, &err);
    if (fd_ < 0) return false;
    path_ = path;
    set(o);
    stop_.store(false);
    thread_ = std::thread([this] { run(); });
    return true;
}

void FakeServer::stop() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
    if (fd_ >= 0) close(fd_);
    fd_ = -1;
    if (!path_.empty()) unlink(path_.c_str());
    path_.clear();
}

void FakeServer::set(const Options &o) {
    std::lock_guard<std::mutex> lock(mu_);
    opt_ = o;
    answered_ = 0;
}

FakeServer::Counts FakeServer::counts() const {
    std::lock_guard<std::mutex> lock(mu_);
    return counts_;
}

float FakeServer::value_of(const featurize::Tensors &t, uint32_t policy) {
    return static_cast<float>(0.9 * std::sin(0.37 * static_cast<double>(t.n()) + 1.3 * policy + 0.05 * content_sum(t)));
}

// Per menu row: its index and the policy, plus the features of the tokens its menu token relates to
// and a share of the whole decision's sum.
std::vector<float> FakeServer::logits_of(const featurize::Tensors &t, uint32_t policy) {
    const size_t F = t.n() ? t.tok_num.size() / t.n() : 0, K = t.n() ? t.tok_card.size() / t.n() : 0;
    const double whole = content_sum(t);
    std::vector<float> out;
    for (size_t a = 0; a < t.a(); ++a) {
        double psi = 0;
        for (size_t r = 0; r < t.r(); ++r) {
            if (t.rel[3 * r] != t.act_tok[a]) continue;
            const size_t d = static_cast<size_t>(t.rel[3 * r + 1]);
            psi += 0.1;
            for (size_t c = 0; c < F; ++c) psi += 0.05 * t.tok_num[d * F + c];
            for (size_t c = 0; c < K; ++c) psi += 0.001 * static_cast<double>(t.tok_card[d * K + c] + 1);
        }
        out.push_back(static_cast<float>(0.25 * static_cast<double>(a % 5) + 0.1 * policy + 0.3 * std::sin(psi) +
                                         0.02 * whole * static_cast<double>(a % 3)));
    }
    return out;
}

void FakeServer::run() {
    while (!stop_.load()) {
        pollfd p = {fd_, POLLIN, 0};
        if (poll(&p, 1, 50) <= 0) continue;
        const int c = accept(fd_, nullptr, nullptr);
        if (c < 0) continue;
        serve(c);
        close(c);
    }
}

// One connection until it closes, BYE, a refusal or a drop.
void FakeServer::serve(int c) {
    inference::Hello h;
    inference::SharedRegion region;
    while (!stop_.load()) {
        pollfd p = {c, POLLIN, 0};
        if (poll(&p, 1, 50) <= 0) continue;
        inference::Message msg;
        std::string err;
        if (inference::recv_message(c, 5.0, &msg, &err) != inference::RecvStatus::OK) return;
        Options o;
        {
            std::lock_guard<std::mutex> lock(mu_);
            o = opt_;
        }
        if (msg.type == inference::MSG_BYE) return;
        if (msg.type == inference::MSG_HELLO) {
            if (!inference::parse_hello(msg.body, &h, &err)) return;
            {
                std::lock_guard<std::mutex> lock(mu_);
                ++counts_.hellos;
                counts_.last_policies = h.policies;
            }
            if (o.refuse) {
                inference::send_message(c, inference::MSG_ERROR,
                    inference::error_json(
                        "policy", "policy " + std::to_string(h.policies[0]) + " is not served (the stand-in refuses)"),
                    &err);
                return;
            }
            if (!region.open(h.shm, h.request_bytes + h.response_bytes, &err)) return;
            inference::Welcome w;
            w.server = "stand-in";
            w.max_batch = o.max_batch;
            for (uint32_t id : h.policies) {
                auto it = o.versions.find(id);
                w.policies.push_back({id, it == o.versions.end() ? 1 : it->second});
            }
            if (!inference::send_message(c, inference::MSG_WELCOME, inference::welcome_json(w), &err)) return;
            continue;
        }
        if (msg.type != inference::MSG_EVAL) return;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (o.lose_after >= 0 && answered_ >= o.lose_after) return;
        }
        inference::Request q;
        if (!inference::decode_request(region.data(), h.request_bytes, &q, &err)) return;
        if (o.delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(o.delay_ms));
        inference::Response r;
        r.spec_hash = q.spec_hash;
        r.seq = q.seq;
        r.batch = static_cast<uint32_t>(q.decisions.size());
        std::vector<size_t> rows;
        for (const featurize::Tensors &t : q.decisions) rows.push_back(t.a());
        for (uint32_t id : q.policies) {
            inference::PolicyResult pr;
            pr.id = id;
            auto it = o.versions.find(id);
            pr.version = it == o.versions.end() ? 1 : it->second;
            for (const featurize::Tensors &t : q.decisions) {
                pr.value.push_back(value_of(t, id));
                const std::vector<float> logits = logits_of(t, id);
                std::vector<float> lp(t.a());
                inference::log_softmax(logits.data(), logits.size(), lp.data());
                pr.logits.insert(pr.logits.end(), logits.begin(), logits.end());
                pr.log_pi.insert(pr.log_pi.end(), lp.begin(), lp.end());
            }
            r.policies.push_back(std::move(pr));
        }
        size_t length = 0;
        if (!inference::encode_response(region.data() + h.request_bytes, h.response_bytes, r, rows, &length, &err))
            return;
        {
            std::lock_guard<std::mutex> lock(mu_);
            ++counts_.requests;
            ++answered_;
            counts_.decisions += q.decisions.size();
            counts_.largest = std::max(counts_.largest, q.decisions.size());
        }
        if (!inference::send_message(c, inference::MSG_DONE, inference::seq_body(q.seq, length), &err)) return;
    }
}

}  // namespace rmwork
