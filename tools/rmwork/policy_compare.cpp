#include "policy_compare.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

// The smallest log π a row of the second distribution counts as in KL.
const double LOG_FLOOR = -30.0;

}  // namespace

static bool sees(int view, int seat);
static int band_of(const std::vector<rmwork::TurnBand> &bands, size_t index, size_t *from);

static bool sees(int view, int seat) {
    return view == rmwork::VIEW_OMNI || view == seat;
}

// The band holding decision `index`, searching from `*from` on (decisions come in order); -1 none.
static int band_of(const std::vector<rmwork::TurnBand> &bands, size_t index, size_t *from) {
    while (*from < bands.size() && bands[*from].last < index) ++*from;
    if (*from < bands.size() && bands[*from].first <= index && index <= bands[*from].last)
        return static_cast<int>(*from);
    return -1;
}

namespace rmwork {

const char *compare_metric_name(CompareMetric m) {
    return m == CompareMetric::KL ? "KL" : "TV";
}

double kl_divergence(const std::vector<float> &log_p, const std::vector<float> &log_q) {
    if (log_p.size() != log_q.size()) return 0;
    double sum = 0;
    for (size_t i = 0; i < log_p.size(); ++i) {
        const double lp = log_p[i];
        const double p = std::exp(lp);
        if (p <= 0) continue;
        sum += p * (lp - std::max(LOG_FLOOR, static_cast<double>(log_q[i])));
    }
    return std::max(0.0, sum);
}

double total_variation(const std::vector<float> &log_p, const std::vector<float> &log_q) {
    if (log_p.size() != log_q.size()) return 0;
    double sum = 0;
    for (size_t i = 0; i < log_p.size(); ++i)
        sum += std::fabs(std::exp(static_cast<double>(log_p[i])) - std::exp(static_cast<double>(log_q[i])));
    return sum / 2;
}

int argmax_row(const std::vector<float> &log_p) {
    int best = -1;
    for (size_t i = 0; i < log_p.size(); ++i)
        if (best < 0 || log_p[i] > log_p[static_cast<size_t>(best)]) best = static_cast<int>(i);
    return best;
}

std::vector<int64_t> comparable_policies(const MatchValues &v, int view) {
    std::vector<int64_t> out;
    for (int64_t p : v.policies)
        for (int seat = 0; seat < 2; ++seat)
            if (sees(view, seat) && v.find(seat, p)) {
                out.push_back(p);
                break;
            }
    return out;
}

int64_t default_reference(const std::vector<int64_t> &policies) {
    return policies.empty() ? -1 : policies.back();
}

bool compare_decision(
    const Match &m, size_t index, const std::vector<int64_t> &policies, int64_t ref, DecisionCompare *out) {
    *out = DecisionCompare();
    if (index >= m.decisions.size()) return false;
    const DecisionInfo &d = m.decisions[index];
    if (d.episode_step < 0 || static_cast<size_t>(d.episode_step) >= m.episode_steps.size()) return false;
    const EpisodeStepInfo &e = m.episode_steps[static_cast<size_t>(d.episode_step)];
    const featurize::PolicyEvalRecord *r = find_policy(e, ref);
    if (!r || d.menu_rows <= 0 || r->log_pi.size() != static_cast<size_t>(d.menu_rows)) return false;
    out->index = index;
    out->seat = d.decider;
    out->game = d.game;
    out->sideboarding = d.sideboarding;
    out->turn = d.turn;
    out->rows = static_cast<size_t>(d.menu_rows);
    out->chosen = d.reply >= 0 && d.reply < d.menu_rows ? d.reply : -1;
    for (int64_t id : policies) {
        PolicyAt p;
        p.policy = id;
        const featurize::PolicyEvalRecord *q = find_policy(e, id);
        if (q && q->id == id && q->log_pi.size() == out->rows) {
            p.has = true;
            p.version = q->version;
            p.value = q->value;
            p.pi = policy_pi(*q);
            p.argmax = argmax_row(q->log_pi);
            p.kl = kl_divergence(r->log_pi, q->log_pi);
            p.tv = total_variation(r->log_pi, q->log_pi);
            p.agrees = p.argmax == argmax_row(r->log_pi);
        }
        if (id == ref) {
            out->ref = static_cast<int>(out->at.size());
            p.kl = p.tv = 0;
            p.agrees = true;
        } else if (p.has) {
            out->max_kl = std::max(out->max_kl, p.kl);
            out->max_tv = std::max(out->max_tv, p.tv);
            out->differs = out->differs || !p.agrees;
        }
        out->at.push_back(std::move(p));
    }
    return out->ref >= 0;
}

void Agreement::add(const PolicyAt &p) {
    ++n;
    agree += p.agrees ? 1 : 0;
    kl += p.kl;
    tv += p.tv;
}

const DecisionCompare *MatchComparison::find(size_t index) const {
    auto it = std::lower_bound(
        decisions.begin(), decisions.end(), index, [](const DecisionCompare &d, size_t i) { return d.index < i; });
    return it != decisions.end() && it->index == index ? &*it : nullptr;
}

int MatchComparison::position(int64_t policy) const {
    for (size_t i = 0; i < policies.size(); ++i)
        if (policies[i] == policy) return static_cast<int>(i);
    return -1;
}

MatchComparison compare_match(const Match &m, int view, const std::vector<int64_t> &policies, int64_t ref) {
    MatchComparison c;
    c.match = &m;
    c.view = view;
    c.policies = policies;
    c.ref = std::find(policies.begin(), policies.end(), ref) != policies.end() ? ref : -1;
    if (c.ref < 0) return c;
    c.bands = turn_bands(m);
    for (int64_t p : policies) {
        if (p == ref) continue;
        PolicyAgreement a;
        a.policy = p;
        a.bands.resize(c.bands.size());
        c.agreement.push_back(std::move(a));
    }
    size_t band_from = 0;
    for (size_t i = 0; i < m.decisions.size(); ++i) {
        if (!sees(view, m.decisions[i].decider)) continue;
        DecisionCompare d;
        if (!compare_decision(m, i, policies, ref, &d)) continue;
        const int band = band_of(c.bands, i, &band_from);
        size_t k = 0;
        for (const PolicyAt &p : d.at) {
            if (p.policy == ref) continue;
            PolicyAgreement &a = c.agreement[k++];
            if (!p.has) continue;
            a.all.add(p);
            if (d.seat == 0 || d.seat == 1) a.seat[static_cast<size_t>(d.seat)].add(p);
            if (band >= 0) a.bands[static_cast<size_t>(band)].add(p);
        }
        c.decisions.push_back(std::move(d));
    }
    return c;
}

std::vector<RankedDecision> rank_disagreements(
    const MatchComparison &c, int64_t policy, CompareMetric metric, bool argmax_only, size_t n) {
    std::vector<RankedDecision> out;
    for (const DecisionCompare &d : c.decisions) {
        const PolicyAt *best = nullptr;
        double best_score = 0;
        for (const PolicyAt &p : d.at) {
            if (p.policy == c.ref || !p.has || (policy >= 0 && p.policy != policy)) continue;
            if (argmax_only && p.agrees) continue;
            const double score = metric == CompareMetric::KL ? p.kl : p.tv;
            if (!best || score > best_score) {
                best = &p;
                best_score = score;
            }
        }
        if (!best) continue;
        const PolicyAt &r = d.at[static_cast<size_t>(d.ref)];
        RankedDecision x;
        x.index = d.index;
        x.policy = best->policy;
        x.score = best_score;
        x.kl = best->kl;
        x.tv = best->tv;
        x.differs = !best->agrees;
        x.ref_argmax = r.argmax;
        x.argmax = best->argmax;
        if (best->argmax >= 0 && static_cast<size_t>(best->argmax) < r.pi.size())
            x.shift = best->pi[static_cast<size_t>(best->argmax)] - r.pi[static_cast<size_t>(best->argmax)];
        out.push_back(x);
    }
    std::stable_sort(
        out.begin(), out.end(), [](const RankedDecision &a, const RankedDecision &b) { return a.score > b.score; });
    if (out.size() > n) out.resize(n);
    return out;
}

std::string metric_text(double v) {
    if (v > 0 && v < 0.0005) return "<0.001";
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.3f", v);
    return buf;
}

std::string agreement_text(const Agreement &a) {
    if (!a.n) return "\xE2\x80\x94";
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.0f%% (%zu/%zu)", 100.0 * a.rate(), a.agree, a.n);
    return buf;
}

}  // namespace rmwork
