#include "evaluations.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

static size_t span_end(const rmwork::Match &m, size_t index);

// One past the last decision of the games entry holding decision `index`.
static size_t span_end(const rmwork::Match &m, size_t index) {
    const int s = m.span_of(index);
    if (s < 0) return m.decisions.size();
    const rmwork::GameSpan &g = m.games[static_cast<size_t>(s)];
    return g.first + g.count;
}

namespace rmwork {

const char *const REEVAL_STEP_FILE = "Re-evaluation";

const featurize::PolicyEvalRecord *find_policy(const EpisodeStepInfo &e, int64_t id) {
    if (e.policies.empty()) return nullptr;
    if (id < 0) return &e.policies.front();
    for (const featurize::PolicyEvalRecord &r : e.policies)
        if (r.id == id) return &r;
    return nullptr;
}

std::vector<double> policy_pi(const featurize::PolicyEvalRecord &r) {
    std::vector<double> p;
    p.reserve(r.log_pi.size());
    for (float l : r.log_pi) p.push_back(std::exp(static_cast<double>(l)));
    return p;
}

double entropy(const std::vector<double> &p) {
    double h = 0;
    for (double x : p)
        if (x > 0) h -= x * std::log(x);
    return h;
}

std::string policy_name(int64_t id) {
    if (is_reevaluated(id)) return "re p" + std::to_string(id - REEVAL_ID_BASE);
    return "p" + std::to_string(id);
}

std::string policy_label(int64_t id, uint64_t version) {
    return policy_name(id) + " v" + std::to_string(version);
}

std::string signed_text(double v, int decimals) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.*f", decimals, std::fabs(v));
    const double unit = std::pow(10.0, -decimals) / 2;
    if (std::fabs(v) < unit) return buf;
    return std::string(v > 0 ? "+" : "\xE2\x88\x92") + buf;
}

const ValueSeries *MatchValues::find(int seat, int64_t policy) const {
    for (const ValueSeries &s : series)
        if (s.seat == seat && s.policy == policy) return &s;
    return nullptr;
}

int64_t MatchValues::resolve(int64_t id) const {
    if (policies.empty()) return -1;
    return std::find(policies.begin(), policies.end(), id) != policies.end() ? id : policies.front();
}

MatchValues match_values(const Match &m) {
    MatchValues out;
    std::map<std::pair<int, int64_t>, ValueSeries> by_key;
    for (size_t i = 0; i < m.decisions.size(); ++i) {
        const int step = m.decisions[i].episode_step;
        if (step < 0 || static_cast<size_t>(step) >= m.episode_steps.size()) continue;
        const EpisodeStepInfo &e = m.episode_steps[static_cast<size_t>(step)];
        for (const featurize::PolicyEvalRecord &r : e.policies) {
            ValueSeries &s = by_key[{e.seat, r.id}];
            s.seat = e.seat;
            s.policy = r.id;
            s.version = r.version;
            s.points.push_back({i, static_cast<double>(r.value)});
            out.versions[r.id] = r.version;
        }
    }
    for (auto &kv : by_key) {
        out.series.push_back(kv.second);
        if (std::find(out.policies.begin(), out.policies.end(), kv.first.second) == out.policies.end())
            out.policies.push_back(kv.first.second);
    }
    std::sort(out.policies.begin(), out.policies.end());
    return out;
}

std::string turn_label(const DecisionInfo &d) {
    if (d.sideboarding) return "SB";
    if (d.phase == "Opening hand") return "Pre";
    return "T" + std::to_string(d.turn + 1);
}

std::vector<TurnBand> turn_bands(const Match &m) {
    std::vector<TurnBand> out;
    for (size_t i = 0; i < m.decisions.size(); ++i) {
        const DecisionInfo &d = m.decisions[i];
        const std::string label = turn_label(d);
        if (!out.empty() && out.back().game == d.game && out.back().label == label && out.back().last + 1 == i) {
            out.back().last = i;
            continue;
        }
        TurnBand b;
        b.first = b.last = i;
        b.game = d.game;
        b.label = label;
        out.push_back(b);
    }
    return out;
}

double game_result(const Match &m, int game, int seat) {
    if (game < 0 || static_cast<size_t>(game) >= m.result.game_winners.size()) return 0;
    const int w = m.result.game_winners[static_cast<size_t>(game)];
    if (w < 0) return 0;
    return w == seat ? 1.0 : -1.0;
}

std::vector<Swing> top_swings(const Match &m, const MatchValues &v, int seat, int64_t policy, size_t n) {
    std::vector<Swing> all;
    for (const ValueSeries &s : v.series) {
        if (s.policy != policy || (seat >= 0 && s.seat != seat)) continue;
        for (size_t k = 0; k < s.points.size(); ++k) {
            const ValuePoint &p = s.points[k];
            const DecisionInfo &d = m.decisions[p.decision];
            if (d.sideboarding) continue;
            Swing w;
            w.seat = s.seat;
            w.policy = s.policy;
            w.from = p.decision;
            w.v0 = p.value;
            w.game = d.game;
            // The seat's next evaluated decision of the same game, else the game's result.
            const ValuePoint *next = nullptr;
            for (size_t j = k + 1; j < s.points.size() && !next; ++j) {
                const DecisionInfo &e = m.decisions[s.points[j].decision];
                if (e.sideboarding) continue;
                if (e.game == d.game) next = &s.points[j];
                break;
            }
            if (next) {
                w.to = next->decision;
                w.v1 = next->value;
            } else {
                if (static_cast<size_t>(d.game) >= m.result.game_winners.size()) continue;
                w.to_result = true;
                w.v1 = game_result(m, d.game, s.seat);
            }
            w.dv = w.v1 - w.v0;
            all.push_back(w);
        }
    }
    std::stable_sort(all.begin(), all.end(), [](const Swing &a, const Swing &b) {
        if (std::fabs(a.dv) != std::fabs(b.dv)) return std::fabs(a.dv) > std::fabs(b.dv);
        return a.from < b.from;
    });
    if (all.size() > n) all.resize(n);
    return all;
}

long next_swing_target(const std::vector<Swing> &swings, size_t cursor) {
    long best = -1;
    for (const Swing &s : swings) {
        const size_t t = swing_target(s);
        if (t > cursor && (best < 0 || t < static_cast<size_t>(best))) best = static_cast<long>(t);
    }
    return best;
}

std::vector<Disagreement> top_disagreements(const Match &m, const MatchValues &v, int64_t policy, size_t n) {
    std::vector<Disagreement> all;
    std::vector<double> value(m.decisions.size(), 0);
    std::vector<int> seat(m.decisions.size(), -1);
    for (const ValueSeries &s : v.series) {
        if (s.policy != policy) continue;
        for (const ValuePoint &p : s.points) {
            value[p.decision] = p.value;
            seat[p.decision] = s.seat;
        }
    }
    for (size_t i = 0; i + 1 < m.decisions.size(); ++i) {
        const DecisionInfo &d = m.decisions[i], &e = m.decisions[i + 1];
        if (seat[i] < 0 || seat[i + 1] < 0 || seat[i] == seat[i + 1]) continue;
        if (d.game != e.game || d.sideboarding || e.sideboarding) continue;
        Disagreement g;
        g.a = seat[i] == 0 ? i : i + 1;
        g.b = seat[i] == 0 ? i + 1 : i;
        g.va = value[g.a];
        g.vb_on_a = -value[g.b];
        g.gap = std::fabs(g.va - g.vb_on_a);
        g.game = d.game;
        all.push_back(g);
    }
    std::stable_sort(all.begin(), all.end(), [](const Disagreement &a, const Disagreement &b) {
        if (a.gap != b.gap) return a.gap > b.gap;
        return std::min(a.a, a.b) < std::min(b.a, b.b);
    });
    if (all.size() > n) all.resize(n);
    return all;
}

std::string between_text(const Match &m, int view, size_t from, bool to_result, size_t to, size_t max_lines) {
    const size_t last = to_result ? span_end(m, from) - 1 : to;
    std::vector<std::string> lines;
    for (const LogLine &l : m.logs[static_cast<size_t>(view)]) {
        if (l.decision <= static_cast<int>(from) || l.decision > static_cast<int>(last)) continue;
        if (l.kind == LogLine::Kind::HEADER) continue;
        // The later decision's own chosen row is not between the two.
        if (l.kind == LogLine::Kind::CHOICE && !to_result && l.decision == static_cast<int>(last)) continue;
        lines.push_back(l.text);
    }
    std::string out;
    for (size_t i = 0; i < lines.size() && i < max_lines; ++i) out += (i ? " \xC2\xB7 " : "") + lines[i];
    if (lines.size() > max_lines) out += " \xE2\x80\xA6 +" + std::to_string(lines.size() - max_lines);
    return out;
}

}  // namespace rmwork
