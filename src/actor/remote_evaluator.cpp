#include "remote_evaluator.h"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>

#include "../featurize/rmtb.h"
#include "../inference/protocol.h"

namespace {

// The client of one connection.
class RemoteEvaluator : public actor::Evaluator {
    public:
        explicit RemoteEvaluator(actor::RemoteOptions opt) : opt_(std::move(opt)) {}
        ~RemoteEvaluator() override;
        // Connects and completes the handshake.
        bool connect(const featurize::Featurization &f, std::string *err);
        bool evaluate(const std::vector<const featurize::Tensors *> &batch, std::vector<actor::Evaluation> *out,
            std::string *err) override;
        size_t policies() const override { return opt_.policies.size(); }
        const actor::ModelInfo &info() const override { return info_; }
        std::string stats() const override;

    private:
        // `what` as the evaluator's error message (naming the endpoint).
        bool fail(const std::string &what, std::string *err) const;
        // The answer to the message just sent: a message of `want` (true) or a failure.
        bool await(uint16_t want, inference::Message *msg, std::string *err);
        bool fill(const std::vector<const featurize::Tensors *> &batch, const inference::Response &r,
            std::vector<actor::Evaluation> *out, std::string *err) const;

        actor::RemoteOptions opt_;
        actor::ModelInfo info_;
        uint64_t spec_hash_ = 0;
        inference::Dims dims_;
        int fd_ = -1;
        inference::SharedRegion region_;
        size_t response_bytes_ = 0;
        uint64_t seq_ = 0;
        std::vector<size_t> menu_rows_;
        inference::Response response_;
        // Requests answered, the decisions in the server batches they were in, and the server's
        // compute time attributed to them (each batch's time shared by its decisions).
        size_t requests_ = 0;
        double batch_sum_ = 0, server_s_ = 0;
};

}  // namespace

static size_t response_area(size_t request_bytes);

// The response area beside a request area: a request's results are far smaller than its tensors.
static size_t response_area(size_t request_bytes) {
    return std::max<size_t>(1u << 20, request_bytes / 16) & ~static_cast<size_t>(7);
}

RemoteEvaluator::~RemoteEvaluator() {
    if (fd_ >= 0) {
        std::string ignored;
        inference::send_message(fd_, inference::MSG_BYE, "", &ignored);
        close(fd_);
    }
}

bool RemoteEvaluator::fail(const std::string &what, std::string *err) const {
    *err = "remote evaluator " + opt_.endpoint + ": " + what;
    return false;
}

bool RemoteEvaluator::await(uint16_t want, inference::Message *msg, std::string *err) {
    std::string why;
    switch (inference::recv_message(fd_, opt_.timeout_s, msg, &why)) {
        case inference::RecvStatus::OK:
            break;
        case inference::RecvStatus::CLOSED:
            return fail("the server closed the connection (it went away)", err);
        case inference::RecvStatus::TIMEOUT:
            return fail("no answer within " + std::to_string(opt_.timeout_s) + " s (timeout=)", err);
        case inference::RecvStatus::MALFORMED:
            return fail("the server's answer is not an RMIP message: " + why, err);
        case inference::RecvStatus::FAILED:
            return fail(why, err);
    }
    if (msg->type == inference::MSG_ERROR) {
        std::string code, message;
        inference::parse_error(msg->body, &code, &message);
        return fail("the server refused: " + (code.empty() ? "" : code + ": ") + message, err);
    }
    if (msg->type != want) return fail("an unexpected message of type " + std::to_string(msg->type), err);
    return true;
}

bool RemoteEvaluator::connect(const featurize::Featurization &f, std::string *err) {
    std::string why;
    info_.path = opt_.endpoint;
    info_.format = "remote";
    info_.spec_hash = featurize::hex16(f.spec.hash);
    info_.vocab_hash = featurize::hex16(f.spec.vocab_hash);
    info_.text_model = f.text_model;
    spec_hash_ = f.spec.hash;
    dims_ = {static_cast<uint32_t>(f.spec.F), static_cast<uint32_t>(f.spec.C), static_cast<uint32_t>(f.spec.K),
        static_cast<uint32_t>(f.spec.Q)};
    const size_t request_bytes = (opt_.request_bytes + 7) & ~static_cast<size_t>(7);
    response_bytes_ = response_area(request_bytes);
    fd_ = inference::connect_unix(opt_.endpoint, &why);
    if (fd_ < 0) return fail(why, err);
    if (!region_.create("/rmip", request_bytes + response_bytes_, &why)) return fail(why, err);
    inference::Hello h;
    h.client = opt_.client.empty() ? "robomage_nn pid " + std::to_string(getpid()) : opt_.client;
    h.spec_hash = info_.spec_hash;
    h.vocab_hash = info_.vocab_hash;
    h.text_model = info_.text_model;
    h.dims = dims_;
    h.shm = region_.name();
    h.request_bytes = request_bytes;
    h.response_bytes = response_bytes_;
    h.policies = opt_.policies;
    if (!inference::send_message(fd_, inference::MSG_HELLO, inference::hello_json(h), &why)) {
        region_.unlink();
        return fail("the handshake: " + why, err);
    }
    inference::Message msg;
    const bool answered = await(inference::MSG_WELCOME, &msg, err);
    region_.unlink();
    if (!answered) return false;
    inference::Welcome w;
    if (!inference::parse_welcome(msg.body, &w, &why)) return fail(why, err);
    if (w.protocol != inference::PROTOCOL_VERSION)
        return fail("the server speaks protocol " + std::to_string(w.protocol), err);
    if (w.policies.size() != opt_.policies.size())
        return fail("the server's WELCOME lists " + std::to_string(w.policies.size()) + " policies, not the " +
                        std::to_string(opt_.policies.size()) + " asked for",
            err);
    for (size_t i = 0; i < w.policies.size(); ++i)
        if (w.policies[i].id != opt_.policies[i]) return fail("the server's WELCOME lists other policies", err);
    info_.policy_version = w.policies[0].version;
    return true;
}

bool RemoteEvaluator::fill(const std::vector<const featurize::Tensors *> &batch, const inference::Response &r,
    std::vector<actor::Evaluation> *out, std::string *err) const {
    out->assign(batch.size(), actor::Evaluation());
    for (size_t p = 0; p < r.policies.size(); ++p) {
        const inference::PolicyResult &pr = r.policies[p];
        size_t row = 0;
        for (size_t b = 0; b < batch.size(); ++b) {
            actor::Evaluation &e = (*out)[b];
            e.policies.resize(r.policies.size());
            actor::PolicyEval &pe = e.policies[p];
            const size_t rows = menu_rows_[b];
            pe.version = pr.version;
            pe.value = pr.value[b];
            pe.logits.assign(pr.logits.begin() + static_cast<std::ptrdiff_t>(row),
                pr.logits.begin() + static_cast<std::ptrdiff_t>(row + rows));
            pe.log_pi.assign(pr.log_pi.begin() + static_cast<std::ptrdiff_t>(row),
                pr.log_pi.begin() + static_cast<std::ptrdiff_t>(row + rows));
            row += rows;
            if (!std::isfinite(pe.value)) return fail("a non-finite value", err);
            for (size_t i = 0; i < rows; ++i)
                if (!std::isfinite(pe.logits[i]) || !std::isfinite(pe.log_pi[i]))
                    return fail("a non-finite logit or log pi", err);
            if (p == 0) e.value = pe.value;
        }
    }
    return true;
}

bool RemoteEvaluator::evaluate(
    const std::vector<const featurize::Tensors *> &batch, std::vector<actor::Evaluation> *out, std::string *err) {
    out->clear();
    if (batch.empty()) return true;
    if (fd_ < 0) return fail("not connected", err);
    std::string why;
    size_t length = 0;
    const size_t request_bytes = region_.size() - response_bytes_;
    if (!inference::encode_request(
            region_.data(), request_bytes, spec_hash_, ++seq_, dims_, opt_.policies, batch, &length, &why))
        return fail(why + " (shm=<MiB> sets the request area)", err);
    if (!inference::send_message(fd_, inference::MSG_EVAL, inference::seq_body(seq_, length), &why))
        return fail("the server went away: " + why, err);
    inference::Message msg;
    if (!await(inference::MSG_DONE, &msg, err)) return false;
    uint64_t seq = 0, rlen = 0;
    if (!inference::parse_seq_body(msg.body, &seq, &rlen) || seq != seq_)
        return fail(
            "DONE is not this request's (seq " + std::to_string(seq) + ", sent " + std::to_string(seq_) + ")", err);
    menu_rows_.clear();
    for (const featurize::Tensors *t : batch) menu_rows_.push_back(t->a());
    if (!inference::decode_response(region_.data() + request_bytes, response_bytes_, menu_rows_, &response_, &why))
        return fail(why, err);
    if (response_.seq != seq_ || response_.spec_hash != spec_hash_ || response_.policies.size() != opt_.policies.size())
        return fail("the response is not this request's (seq, spec hash or policies)", err);
    for (size_t p = 0; p < opt_.policies.size(); ++p)
        if (response_.policies[p].id != opt_.policies[p])
            return fail("the response's policies are not the request's", err);
    ++requests_;
    batch_sum_ += response_.batch;
    if (response_.batch > 0)
        server_s_ += 1e-6 * response_.compute_us * static_cast<double>(batch.size()) / response_.batch;
    info_.policy_version = response_.policies[0].version;
    return fill(batch, response_, out, err);
}

std::string RemoteEvaluator::stats() const {
    char buf[160];
    std::snprintf(buf, sizeof buf, " requests=%zu server_s=%.3f batch_mean=%.2f", requests_, server_s_,
        requests_ ? batch_sum_ / static_cast<double>(requests_) : 0.0);
    return buf;
}

namespace actor {

std::unique_ptr<Evaluator> connect_remote_evaluator(
    const RemoteOptions &opt, const featurize::Featurization &f, std::string *err) {
    std::unique_ptr<RemoteEvaluator> e(new RemoteEvaluator(opt));
    if (!e->connect(f, err)) return nullptr;
    return e;
}

}  // namespace actor
