#include "reevaluation.h"

#include <algorithm>
#include <cstdlib>

#include "model_read.h"

static std::string seats_text(const std::array<bool, 2> &seats);

// "A, B", "A", "B".
static std::string seats_text(const std::array<bool, 2> &seats) {
    if (seats[0] && seats[1]) return "A, B";
    return seats[0] ? "A" : "B";
}

namespace rmwork {

std::vector<ServedPolicy> served_policies(const ModelSettings &m) {
    std::vector<ServedPolicy> out;
    for (const PolicyEntry &p : m.policies) {
        const std::vector<uint32_t> ids = model_read_policies({p.id});
        // model_read_policies gives policy 0 for an id that is not a number: keep only real ones.
        if (p.id != std::to_string(ids[0])) continue;
        bool seen = false;
        for (const ServedPolicy &q : out) seen = seen || q.id == ids[0];
        if (!seen) out.push_back({ids[0], policy_model_path(p)});
    }
    return out;
}

void merge_evaluations(
    Match *m, int64_t id, const std::vector<std::pair<size_t, featurize::PolicyEvalRecord>> &records) {
    for (const std::pair<size_t, featurize::PolicyEvalRecord> &rec : records) {
        if (rec.first >= m->decisions.size()) continue;
        DecisionInfo &d = m->decisions[rec.first];
        if (d.episode_step < 0 || static_cast<size_t>(d.episode_step) >= m->episode_steps.size()) {
            EpisodeStepInfo e;
            e.file = REEVAL_STEP_FILE;
            e.seat = d.decider;
            // The log's -1 is the menu's last row (a declaration's confirm); a concession or no reply is none.
            e.chosen = d.reply >= 0 ? d.reply : d.reply == -1 && d.menu_rows > 0 ? d.menu_rows - 1 : -1;
            m->episode_steps.push_back(std::move(e));
            d.episode_step = static_cast<int>(m->episode_steps.size()) - 1;
        }
        std::vector<featurize::PolicyEvalRecord> &ps = m->episode_steps[static_cast<size_t>(d.episode_step)].policies;
        featurize::PolicyEvalRecord r = rec.second;
        r.id = id;
        auto it = std::find_if(ps.begin(), ps.end(), [id](const featurize::PolicyEvalRecord &x) { return x.id == id; });
        if (it != ps.end()) {
            *it = std::move(r);
        } else {
            // After the recorded ones, the re-evaluated ones in id order.
            auto at = std::find_if(ps.begin(), ps.end(),
                [id](const featurize::PolicyEvalRecord &x) { return is_reevaluated(x.id) && x.id > id; });
            ps.insert(at, std::move(r));
        }
    }
    ++m->evaluations_changed;
}

Reevaluation::~Reevaluation() {
    cancel();
}

void Reevaluation::start(Match *m, ReplayEvaluator *ev, const ReevalRequest &r) {
    cancel();
    ev_ = ev;
    match_ = m;
    dump_ = m->dump_path;
    req_ = r;
    if (req_.cache_root.empty()) req_.cache_root = eval_cache_root();
    error_.clear();
    no_server_ = false;
    jobs_.clear();
    verify_ = 0;
    verify_policies_.clear();
    cached_.clear();
    merged_.clear();
    versions_.clear();
    notes_.clear();
    source_ = m->live ? nullptr : eval_source(*m);
    if (!source_ || req_.policies.empty() || (!req_.seats[0] && !req_.seats[1])) {
        error_ = !source_                ? "Play's live game is re-evaluated once it is opened in Replay"
                 : req_.policies.empty() ? "no served policy to re-evaluate with (Settings \xE2\x86\x92 Models)"
                                         : "no seat to re-evaluate";
        state_ = State::FAILED;
        return;
    }
    std::array<std::vector<ServedPolicy>, 2> missing;
    for (const ServedPolicy &p : req_.policies) {
        const std::string identity = model_identity(p.model);
        for (int seat = 0; seat < 2; ++seat) {
            if (!req_.seats[static_cast<size_t>(seat)]) continue;
            CachedEvals c;
            std::string why;
            const std::string path = eval_cache_path(req_.cache_root, m->log_hash, identity, seat);
            if (!read_eval_cache(path, &c, &why) || c.identity != identity || !eval_cache_fits(c, *m, seat, &why)) {
                missing[static_cast<size_t>(seat)].push_back(p);
                continue;
            }
            std::vector<std::pair<size_t, featurize::PolicyEvalRecord>> recs;
            recs.reserve(c.decisions.size());
            for (CachedDecision &d : c.decisions) {
                featurize::PolicyEvalRecord rec;
                rec.version = c.version;
                rec.value = d.value;
                rec.log_pi = std::move(d.log_pi);
                recs.emplace_back(d.index, std::move(rec));
            }
            merge_evaluations(m, reevaluated_id(p.id), recs);
            cached_[{p.id, seat}] = c.version;
            merged_[{p.id, seat}] = true;
            versions_[p.id] = c.version;
        }
    }
    state_ = State::RUNNING;
    if (!req_.use_server) {
        const size_t n = missing[0].size() + missing[1].size();
        if (n) notes_.push_back(std::to_string(n) + " not in the cache (no server to evaluate them)");
        end(State::DONE);
        return;
    }
    if (!cached_.empty()) {
        // A handshake (first) to compare the versions the server serves with the cached entries'.
        EvalJob job;
        job.source = source_;
        job.priority = req_.priority + 1;
        for (const ServedPolicy &p : req_.policies)
            if (cached_.count({p.id, 0}) || cached_.count({p.id, 1})) {
                job.policies.push_back(p.id);
                verify_policies_.push_back(p);
            }
        verify_ = ev_->submit(std::move(job));
    }
    for (int seat = 0; seat < 2; ++seat)
        if (!missing[static_cast<size_t>(seat)].empty()) submit_seat(seat, missing[static_cast<size_t>(seat)]);
}

void Reevaluation::submit_seat(int seat, const std::vector<ServedPolicy> &policies) {
    SeatJob j;
    j.seat = seat;
    j.policies = policies;
    j.records.resize(policies.size());
    EvalJob job;
    job.source = source_;
    job.priority = req_.priority;
    for (const ServedPolicy &p : policies) job.policies.push_back(p.id);
    for (size_t i = 0; i < match_->decisions.size(); ++i)
        if (match_->decisions[i].decider == seat) job.items.push_back({i, seat, nullptr});
    j.total = job.items.size();
    j.id = ev_->submit(std::move(job));
    jobs_.push_back(std::move(j));
}

void Reevaluation::poll(Match *m) {
    if (state_ != State::RUNNING) return;
    if (m != match_ || !m || m->dump_path != dump_) {
        cancel();
        return;
    }
    if (verify_) finish_verify();
    bool all = verify_ == 0;
    for (SeatJob &j : jobs_) {
        if (!j.id) continue;
        for (EvalResult &r : ev_->take(j.id))
            for (size_t p = 0; p < r.policies.size() && p < j.records.size(); ++p)
                j.records[p].emplace_back(r.decision, std::move(r.policies[p]));
        const EvalJobStatus st = ev_->status(j.id);
        if (st.state == EvalJobStatus::State::FAILED) {
            error_ = st.error;
            no_server_ = st.no_server;
            end(State::FAILED);
            return;
        }
        if (st.state == EvalJobStatus::State::CANCELLED) {
            end(State::CANCELLED);
            return;
        }
        if (st.state != EvalJobStatus::State::DONE) {
            all = false;
            continue;
        }
        ev_->forget(j.id);
        j.id = 0;
        finish_seat(m, &j);
    }
    if (all) end(State::DONE);
}

// A seat job that returned whole: per policy its entry written and its records merged.
void Reevaluation::finish_seat(Match *m, SeatJob *j) {
    for (size_t p = 0; p < j->policies.size(); ++p) {
        std::vector<std::pair<size_t, featurize::PolicyEvalRecord>> &recs = j->records[p];
        if (recs.size() != j->total) continue;
        const ServedPolicy &sp = j->policies[p];
        CachedEvals c;
        c.log_hash = m->log_hash;
        c.identity = model_identity(sp.model);
        c.seat = j->seat;
        c.policy = sp.id;
        c.version = recs.empty() ? 0 : recs.front().second.version;
        for (const std::pair<size_t, featurize::PolicyEvalRecord> &r : recs) {
            const DecisionInfo &d = m->decisions[r.first];
            CachedDecision cd;
            cd.index = static_cast<uint32_t>(r.first);
            cd.game = d.game;
            cd.sideboarding = d.sideboarding;
            cd.decision = d.decision;
            cd.value = r.second.value;
            cd.log_pi = r.second.log_pi;
            c.decisions.push_back(std::move(cd));
        }
        const std::string path = eval_cache_path(req_.cache_root, m->log_hash, c.identity, j->seat);
        std::string err;
        if (write_eval_cache(path, c, &err)) {
            prune_eval_cache(req_.cache_root, EVAL_CACHE_MAX_BYTES, path);
        } else {
            notes_.push_back("not cached: " + err);
        }
        merge_evaluations(m, reevaluated_id(sp.id), recs);
        merged_[{sp.id, j->seat}] = false;
        versions_[sp.id] = c.version;
    }
}

// The version check, once returned: the cached entries of a version the server no longer serves are
// evaluated again. A check that could not be made leaves the cache's results standing.
void Reevaluation::finish_verify() {
    const EvalJobStatus st = ev_->status(verify_);
    if (!st.finished()) return;
    ev_->forget(verify_);
    verify_ = 0;
    if (st.state == EvalJobStatus::State::FAILED) {
        if (!st.no_server) notes_.push_back("the cached versions were not checked: " + st.error);
        return;
    }
    std::array<std::vector<ServedPolicy>, 2> stale;
    for (size_t p = 0; p < verify_policies_.size() && p < st.versions.size(); ++p)
        for (int seat = 0; seat < 2; ++seat) {
            auto it = cached_.find({verify_policies_[p].id, seat});
            if (it != cached_.end() && it->second != st.versions[p])
                stale[static_cast<size_t>(seat)].push_back(verify_policies_[p]);
        }
    for (int seat = 0; seat < 2; ++seat)
        if (!stale[static_cast<size_t>(seat)].empty()) submit_seat(seat, stale[static_cast<size_t>(seat)]);
}

void Reevaluation::end(State s) {
    for (SeatJob &j : jobs_)
        if (j.id) {
            ev_->forget(j.id);
            j.id = 0;
        }
    if (verify_) ev_->forget(verify_);
    verify_ = 0;
    state_ = s;
}

void Reevaluation::cancel() {
    if (state_ == State::RUNNING) end(State::CANCELLED);
}

double Reevaluation::fraction() const {
    if (state_ != State::RUNNING || verify_) return -1;
    size_t done = 0, total = 0;
    for (const SeatJob &j : jobs_) {
        total += j.total;
        done += j.id ? ev_->status(j.id).done : j.total;
    }
    return total ? static_cast<double>(done) / static_cast<double>(total) : -1;
}

std::string Reevaluation::phase() const {
    if (state_ != State::RUNNING) return "";
    if (verify_) return "Re-evaluation: checking the cached versions";
    size_t done = 0, total = 0;
    for (const SeatJob &j : jobs_) {
        total += j.total;
        done += j.id ? ev_->status(j.id).done : j.total;
    }
    return "Re-evaluating: " + std::to_string(done) + " of " + std::to_string(total) + " decisions";
}

std::string Reevaluation::summary() const {
    std::string pols;
    for (const ServedPolicy &p : req_.policies) {
        auto v = versions_.find(p.id);
        if (v == versions_.end()) continue;
        pols += (pols.empty() ? "" : ", ") + policy_label(reevaluated_id(p.id), v->second);
    }
    size_t cached = 0, evaluated = 0;
    for (const auto &kv : merged_) (kv.second ? cached : evaluated) += 1;
    std::string out;
    switch (state_) {
        case State::IDLE:
            return "";
        case State::RUNNING:
            return phase();
        case State::DONE:
            out = pols.empty()
                      ? std::string("Re-evaluation: nothing to evaluate")
                      : "Re-evaluated with " + pols + " (" + seats_text(req_.seats) + "): " + std::to_string(cached) +
                            " from the cache, " + std::to_string(evaluated) + " evaluated";
            break;
        case State::CANCELLED:
            out = "Re-evaluation cancelled" +
                  (cached ? " (" + std::to_string(cached) + " from the cache)" : std::string());
            break;
        case State::FAILED:
            out = "Re-evaluation failed: " + error_;
            break;
    }
    for (const std::string &n : notes_) out += "; " + n;
    return out;
}

}  // namespace rmwork
