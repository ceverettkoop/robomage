#include "replay_evaluator.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <list>

#include "../../src/actor/remote_evaluator.h"
#include "../../src/agents/seat_frames.h"
#include "../../src/inference/protocol.h"
#include "eval_connection.h"
#include "files.h"

namespace {

// How many featurized decisions the thread keeps.
const size_t TENSOR_LRU = 8;

}  // namespace

static bool read_at(int fd, const rmwork::FrameRef &ref, std::string *out, std::string *err);

// A frame's payload from the dump (pread, so it needs no seek).
static bool read_at(int fd, const rmwork::FrameRef &ref, std::string *out, std::string *err) {
    out->assign(ref.size, '\0');
    size_t done = 0;
    while (done < ref.size) {
        const ssize_t n = pread(fd, &(*out)[done], ref.size - done, static_cast<off_t>(ref.offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            *err = n < 0 ? std::string(std::strerror(errno)) : std::string("the dump is truncated");
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

namespace rmwork {

// A submitted job and its progress (shared by the UI and the thread; guarded by the evaluator's mutex
// but for `job`, which nobody changes after submit).
struct ReplayEvaluator::JobState {
        uint64_t id = 0;
        EvalJob job;
        EvalJobStatus status;
        size_t next = 0;  // the next item to send
        bool cancel = false;
        std::vector<EvalResult> results;  // not taken yet
};

// What only the thread touches: the featurizer's inputs, the connection, the open source, the LRU.
struct ReplayEvaluator::Thread {
        std::shared_ptr<featurize::Featurization> features;  // shared with the clients (features())
        bool have_features = false;
        std::string features_key;  // engine + workdir they were loaded for
        std::unique_ptr<actor::Evaluator> evaluator;
        std::vector<uint32_t> conn_policies;
        unsigned conn_gen = 0;
        std::shared_ptr<const EvalSource> source;
        int fd = -1;
        std::array<std::unique_ptr<agents::SeatStream>, 2> streams;
        // (decision, seat) → tensors of `source`, most recent first.
        std::list<std::pair<std::pair<size_t, int>, std::shared_ptr<const featurize::Tensors>>> lru;

        ~Thread() { close_source(); }
        // A fresh connection for `policies` (the old one closed); false with `err`.
        bool connect(const Config &c, unsigned gen, const std::vector<uint32_t> &policies, std::string *err) {
            evaluator.reset();
            evaluator = connect_server(c.socket, policies, c.timeout_s, c.client, *features, err);
            if (!evaluator) return false;
            conn_policies = policies;
            conn_gen = gen;
            return true;
        }
        void close_source() {
            if (fd >= 0) close(fd);
            fd = -1;
            source.reset();
            lru.clear();
            streams[0].reset();
            streams[1].reset();
        }
        // `s` opened: its dump and each seat's stream fed its once-per-match frames.
        bool open_source(const std::shared_ptr<const EvalSource> &s, std::string *err) {
            if (source == s && fd >= 0) return true;
            close_source();
            fd = ::open(s->dump_path.c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0) {
                *err = "cannot open " + s->dump_path + ": " + std::strerror(errno);
                return false;
            }
            for (int seat = 0; seat < 2; ++seat) {
                streams[static_cast<size_t>(seat)].reset(new agents::SeatStream());
                for (const FrameRef &ref : s->match_frames[static_cast<size_t>(seat)]) {
                    std::string payload, why;
                    if (!read_at(fd, ref, &payload, &why) ||
                        !streams[static_cast<size_t>(seat)]->feed_match_frame(payload, &why)) {
                        *err = std::string(seat == 0 ? "seat A" : "seat B") + "'s once-per-match frame: " + why;
                        close_source();
                        return false;
                    }
                }
            }
            source = s;
            return true;
        }
        // The decision's tensors in `seat`'s own frame, featurized or from the LRU.
        std::shared_ptr<const featurize::Tensors> tensors(
            size_t decision, int seat, size_t *featurized, std::string *err) {
            const std::pair<size_t, int> key(decision, seat);
            for (auto it = lru.begin(); it != lru.end(); ++it) {
                if (it->first != key) continue;
                lru.splice(lru.begin(), lru, it);
                return lru.front().second;
            }
            if (decision >= source->frames.size() || seat < 0 || seat > 1) {
                *err = "no decision " + std::to_string(decision + 1) + " of seat " + std::to_string(seat);
                return nullptr;
            }
            const FrameRef &ref = source->frames[decision][static_cast<size_t>(seat)];
            if (!ref.valid()) {
                *err = "decision " + std::to_string(decision + 1) + " has no frame of " +
                       (seat == 0 ? "seat A" : "seat B");
                return nullptr;
            }
            std::string payload;
            rmstate::DecodedFrame frame;
            std::shared_ptr<featurize::Tensors> t(new featurize::Tensors());
            if (!read_at(fd, ref, &payload, err) ||
                !streams[static_cast<size_t>(seat)]->decode_state(payload, &frame, err) ||
                !featurize::featurize_seat(features->spec, features->tables, frame, t.get(), err)) {
                *err = "decision " + std::to_string(decision + 1) + " (" + (seat == 0 ? "seat A" : "seat B") +
                       "'s frame): " + *err;
                return nullptr;
            }
            ++*featurized;
            lru.emplace_front(key, t);
            if (lru.size() > TENSOR_LRU) lru.pop_back();
            return t;
        }
};

std::shared_ptr<const EvalSource> eval_source(const Match &m) {
    std::shared_ptr<EvalSource> s(new EvalSource());
    s->dump_path = m.dump_path;
    s->log_hash = m.log_hash;
    for (const std::pair<int, FrameRef> &f : m.match_frames())
        if (f.first == VIEW_A || f.first == VIEW_B) s->match_frames[static_cast<size_t>(f.first)].push_back(f.second);
    s->frames.reserve(m.decisions.size());
    for (const DecisionInfo &d : m.decisions) s->frames.push_back({{d.frames[VIEW_A], d.frames[VIEW_B]}});
    return s;
}

ReplayEvaluator::ReplayEvaluator() = default;

ReplayEvaluator::~ReplayEvaluator() {
    stop();
}

void ReplayEvaluator::configure(const Config &c) {
    std::lock_guard<std::mutex> lock(mu_);
    const bool same = c.socket == config_.socket && c.engine == config_.engine && c.workdir == config_.workdir &&
                      c.cards_path == config_.cards_path && c.timeout_s == config_.timeout_s &&
                      c.batch_cap == config_.batch_cap && c.client == config_.client;
    if (same) return;
    config_ = c;
    ++config_gen_;
}

uint64_t ReplayEvaluator::submit(EvalJob job) {
    std::shared_ptr<JobState> j(new JobState());
    {
        std::lock_guard<std::mutex> lock(mu_);
        j->id = next_id_++;
        j->status.state = EvalJobStatus::State::QUEUED;
        j->status.total = job.items.size();
        j->job = std::move(job);
        jobs_[j->id] = j;
    }
    if (!thread_.joinable()) {
        stop_.store(false);
        t_.reset(new Thread());
        thread_ = std::thread([this] { run(); });
    }
    cv_.notify_all();
    return j->id;
}

void ReplayEvaluator::cancel(uint64_t job) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = jobs_.find(job);
    if (it == jobs_.end() || it->second->status.finished()) return;
    JobState &j = *it->second;
    j.cancel = true;
    if (j.status.state == EvalJobStatus::State::QUEUED) j.status.state = EvalJobStatus::State::CANCELLED;
}

EvalJobStatus ReplayEvaluator::status(uint64_t job) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = jobs_.find(job);
    return it == jobs_.end() ? EvalJobStatus() : it->second->status;
}

std::vector<EvalResult> ReplayEvaluator::take(uint64_t job) {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<EvalResult> out;
    auto it = jobs_.find(job);
    if (it != jobs_.end()) out.swap(it->second->results);
    return out;
}

void ReplayEvaluator::forget(uint64_t job) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = jobs_.find(job);
    if (it == jobs_.end()) return;
    it->second->cancel = true;
    jobs_.erase(it);
}

bool ReplayEvaluator::busy() const {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto &kv : jobs_)
        if (!kv.second->status.finished()) return true;
    return false;
}

std::shared_ptr<const featurize::Featurization> ReplayEvaluator::features() const {
    std::lock_guard<std::mutex> lock(mu_);
    return features_;
}

ReplayEvaluator::Stats ReplayEvaluator::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
}

void ReplayEvaluator::stop() {
    stop_.store(true);
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    t_.reset();
    std::lock_guard<std::mutex> lock(mu_);
    for (auto &kv : jobs_) {
        EvalJobStatus &s = kv.second->status;
        if (!s.finished()) s.state = EvalJobStatus::State::CANCELLED;
    }
}

// The unfinished job of the highest priority (the earliest of equals), nullptr when none; called
// with the mutex held.
std::shared_ptr<ReplayEvaluator::JobState> ReplayEvaluator::next_job() {
    std::shared_ptr<JobState> best;
    for (const auto &kv : jobs_) {
        const std::shared_ptr<JobState> &j = kv.second;
        if (j->status.finished()) continue;
        if (!best || j->job.priority > best->job.priority) best = j;
    }
    return best;
}

void ReplayEvaluator::run() {
    while (true) {
        std::shared_ptr<JobState> j;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this, &j] {
                if (stop_.load()) return true;
                j = next_job();
                return j != nullptr;
            });
            if (stop_.load()) return;
            if (j->cancel) {
                j->status.state = EvalJobStatus::State::CANCELLED;
                continue;
            }
            j->status.state = EvalJobStatus::State::RUNNING;
        }
        step(j);
    }
}

void ReplayEvaluator::fail(const std::shared_ptr<JobState> &j, const std::string &why, bool no_server) {
    std::lock_guard<std::mutex> lock(mu_);
    j->status.state = EvalJobStatus::State::FAILED;
    j->status.error = why;
    j->status.no_server = no_server;
}

void ReplayEvaluator::step(const std::shared_ptr<JobState> &j) {
    Thread &t = *t_;
    Config c;
    unsigned gen = 0;
    {
        std::lock_guard<std::mutex> lock(mu_);
        c = config_;
        gen = config_gen_;
    }
    const EvalJob &job = j->job;
    std::string err;
    if (!job.source || job.policies.empty()) {
        fail(j, !job.source ? "no match to evaluate" : "no policy to evaluate with", false);
        return;
    }
    const std::string absent = server_absent(c.socket);
    if (!absent.empty()) {
        t.evaluator.reset();
        fail(j, absent, true);
        return;
    }
    // A rebuilt engine may have other tables: its modification time is part of the key.
    const std::string features_key = c.engine + '\n' + std::to_string(file_stamp(c.engine).mtime_ns) + '\n' + c.workdir;
    if (!t.have_features || t.features_key != features_key) {
        t.evaluator.reset();
        t.lru.clear();
        t.features.reset(new featurize::Featurization());
        t.have_features = load_engine_features(c.engine, c.workdir, c.cards_path, &stop_, t.features.get(), &err);
        if (!t.have_features) {
            if (!stop_.load()) fail(j, err, false);
            return;
        }
        t.features_key = features_key;
        std::lock_guard<std::mutex> lock(mu_);
        features_ = t.features;
    }
    // A job of no items asks what the server serves now: always a fresh handshake.
    bool fresh = false;
    if (!t.evaluator || t.conn_policies != job.policies || t.conn_gen != gen || job.items.empty()) {
        if (!t.connect(c, gen, job.policies, &err)) {
            fail(j, err, false);
            return;
        }
        fresh = true;
    }
    const actor::ModelInfo &info = t.evaluator->info();
    {
        std::lock_guard<std::mutex> lock(mu_);
        j->status.versions = info.welcome_versions;
        stats_.max_batch = static_cast<size_t>(info.max_batch);
        if (job.items.empty()) {
            j->status.state = EvalJobStatus::State::DONE;
            return;
        }
    }
    if (!t.open_source(job.source, &err)) {
        fail(j, err, false);
        return;
    }
    size_t limit = std::max<size_t>(1, c.batch_cap);
    if (info.max_batch > 0) limit = std::min<size_t>(limit, static_cast<size_t>(info.max_batch));
    const size_t area = actor::RemoteOptions().request_bytes;
    const featurize::FeatureSpec &spec = t.features->spec;
    const inference::Dims dims = {static_cast<uint32_t>(spec.F), static_cast<uint32_t>(spec.C),
        static_cast<uint32_t>(spec.K), static_cast<uint32_t>(spec.Q)};
    // The batch: the job's next items, featurized (and changed), while they fit.
    size_t first = 0;
    {
        std::lock_guard<std::mutex> lock(mu_);
        first = j->next;
    }
    std::vector<std::shared_ptr<const featurize::Tensors>> held;
    std::vector<const featurize::Tensors *> batch;
    size_t featurized = 0;
    size_t k = first;
    while (k < job.items.size() && batch.size() < limit) {
        const EvalItem &item = job.items[k];
        std::shared_ptr<const featurize::Tensors> base = t.tensors(item.decision, item.seat, &featurized, &err);
        if (!base) {
            fail(j, err, false);
            return;
        }
        std::shared_ptr<const featurize::Tensors> use = base;
        if (item.change) {
            std::shared_ptr<featurize::Tensors> changed(new featurize::Tensors(*base));
            if (!item.change(changed.get(), &err)) {
                fail(j, "decision " + std::to_string(item.decision + 1) + ": the change: " + err, false);
                return;
            }
            use = changed;
        }
        batch.push_back(use.get());
        if (batch.size() > 1 && inference::request_size(dims, job.policies.size(), batch) > area) {
            batch.pop_back();
            break;
        }
        held.push_back(use);
        ++k;
    }
    std::vector<actor::Evaluation> evals;
    bool ok = t.evaluator->evaluate(batch, &evals, &err) && evals.size() == batch.size();
    // A connection kept from an earlier job may be to a server since restarted: connect once more.
    if (!ok && !fresh && t.connect(c, gen, job.policies, &err))
        ok = t.evaluator->evaluate(batch, &evals, &err) && evals.size() == batch.size();
    if (!ok) {
        t.evaluator.reset();
        fail(j, err.empty() ? "the server's answer has another number of decisions" : err, false);
        return;
    }
    std::vector<EvalResult> results;
    for (size_t b = 0; b < evals.size(); ++b) {
        const EvalItem &item = job.items[first + b];
        EvalResult r;
        r.item = first + b;
        r.decision = item.decision;
        r.seat = item.seat;
        for (size_t p = 0; p < evals[b].policies.size(); ++p) {
            const actor::PolicyEval &pe = evals[b].policies[p];
            featurize::PolicyEvalRecord rec;
            rec.id = p < job.policies.size() ? static_cast<int64_t>(job.policies[p]) : static_cast<int64_t>(p);
            rec.version = pe.version;
            rec.value = pe.value;
            rec.log_pi = pe.log_pi;
            r.policies.push_back(std::move(rec));
        }
        results.push_back(std::move(r));
    }
    std::lock_guard<std::mutex> lock(mu_);
    stats_.requests += 1;
    stats_.decisions += batch.size();
    stats_.largest = std::max(stats_.largest, batch.size());
    stats_.featurized += featurized;
    for (EvalResult &r : results) j->results.push_back(std::move(r));
    j->next = k;
    j->status.done = k;
    if (j->cancel) {
        j->status.state = EvalJobStatus::State::CANCELLED;
    } else if (k >= job.items.size()) {
        j->status.state = EvalJobStatus::State::DONE;
    }
}

}  // namespace rmwork
