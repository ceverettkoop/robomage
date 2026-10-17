// The belief layer's likelihood and posterior (belief.h, docs/knowledge_store.md §10.6).

#include "belief.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <map>
#include <set>

namespace knowledge {

namespace {

// The backward sweep of one network (belief.h). Card classes are the cards some pin or fixed card
// names (cards_, a dense index); a demand is a count per class.
class LikelihoodSweep {
    public:
        LikelihoodSweep(const Network &net, uint64_t budget) : net_(net), budget_(budget) {}
        Likelihood run();

    private:
        using Key = std::vector<int16_t>;
        // A place on a junction's in side a demanded card can come from, besides its known slots: an
        // EXACT root (its slots a shuffle of `counts`) or a carried in node (its demand joins its
        // producer's in the state).
        struct Target {
                int size = 0;
                std::vector<int> counts;  // a root's contents per class; empty for a carried node
                size_t slot = 0;          // a carried node: its producer's place in the next state
        };
        // One junction's in side and out side, the same for every state.
        struct Step {
                std::vector<int> base;   // per class: the out side's pinned nodes and fixed cards
                int out_at = -1;         // the junction's place in the state (its carried outs' demand)
                std::vector<int> known;  // per class: the in side's pinned nodes and fixed cards
                std::vector<Target> targets;
                int n = 0;  // units through the junction
        };
        bool prepare();
        bool order_junctions();
        int class_of(int card) const;
        bool apply(uint32_t j);
        void split(const Step &st, size_t c, size_t stage, int left, std::vector<int> &known_used,
            std::vector<std::vector<int>> &d, std::vector<int> &room, double lw, double w, const Key &kept,
            std::map<Key, double> *next);
        void leaf(const Step &st, const std::vector<int> &known_used, const std::vector<std::vector<int>> &d, double lw,
            double w, const Key &kept, std::map<Key, double> *next);

        const Network &net_;
        uint64_t budget_;
        uint64_t steps_ = 0;
        bool over_ = false;
        bool impossible_ = false;
        std::vector<int> cards_;  // the classes: every card a pin or a fixed card names, ascending
        std::vector<int> producer_, consumer_;
        std::vector<int> pin_class_;   // per node: its pin's class, else -1
        std::vector<uint32_t> order_;  // junctions in dependency order
        // The junctions whose carried outs' summed demand the state holds, in key order: a junction
        // reads only the sum of its outs' demands, so its outs share one entry.
        std::vector<uint32_t> frontier_;
        std::map<Key, double> states_;
        double scale_ = 0;  // log of the factor the state weights were divided by
};

}  // namespace

static double log_factorial(int n);
static double log_falling(int a, int k);

// log n!, from a table of cumulative logs (deterministic: the same sums in the same order).
static double log_factorial(int n) {
    static const std::vector<double> table = [] {
        std::vector<double> t(65537, 0.0);
        for (size_t i = 2; i < t.size(); ++i) t[i] = t[i - 1] + std::log(static_cast<double>(i));
        return t;
    }();
    if (n <= 1) return 0.0;
    if (static_cast<size_t>(n) < table.size()) return table[static_cast<size_t>(n)];
    return std::lgamma(static_cast<double>(n) + 1.0);
}

// log (a)_k = log a! / (a - k)!; -inf when k > a.
static double log_falling(int a, int k) {
    if (k > a) return -INFINITY;
    return log_factorial(a) - log_factorial(a - k);
}

bool LikelihoodSweep::order_junctions() {
    const size_t nj = net_.junction_count();
    std::vector<int> waiting(nj, 0);
    for (size_t j = 0; j < nj; ++j)
        for (NodeId m : net_.junction(static_cast<uint32_t>(j)).in)
            if (producer_[m] >= 0) ++waiting[j];
    std::vector<bool> done(nj, false);
    while (order_.size() < nj) {
        bool progress = false;
        for (size_t j = 0; j < nj; ++j) {
            if (done[j] || waiting[j] > 0) continue;
            done[j] = true;
            progress = true;
            order_.push_back(static_cast<uint32_t>(j));
            for (NodeId m : net_.junction(static_cast<uint32_t>(j)).out)
                if (consumer_[m] >= 0) --waiting[static_cast<size_t>(consumer_[m])];
        }
        if (!progress) return false;
    }
    return true;
}

int LikelihoodSweep::class_of(int card) const {
    const auto it = std::lower_bound(cards_.begin(), cards_.end(), card);
    return it != cards_.end() && *it == card ? static_cast<int>(it - cards_.begin()) : -1;
}

// The network's shape: producers, consumers, the classes, pins, the junction order. False for a
// network the model does not cover (a root that is not EXACT, a node no junction makes that is no
// root, a cycle); a pin on a node of several units, or a root whose contents do not fit its size
// or pin, makes every world impossible.
bool LikelihoodSweep::prepare() {
    const size_t n = net_.node_count();
    producer_.assign(n, -1);
    consumer_.assign(n, -1);
    pin_class_.assign(n, -1);
    std::set<int> cards;
    for (size_t j = 0; j < net_.junction_count(); ++j) {
        const Junction &jn = net_.junction(static_cast<uint32_t>(j));
        for (NodeId m : jn.in) consumer_[m] = static_cast<int>(j);
        for (NodeId m : jn.out) producer_[m] = static_cast<int>(j);
        for (const auto &c : jn.in_fixed) cards.insert(c.first);
        for (const auto &c : jn.out_fixed) cards.insert(c.first);
    }
    for (size_t i = 0; i < n; ++i) {
        const Node &node = net_.node(static_cast<NodeId>(i));
        if (node.pin >= 0) cards.insert(node.pin);
        if (producer_[i] < 0 && node.root < 0) return false;
    }
    cards_.assign(cards.begin(), cards.end());
    for (size_t i = 0; i < n; ++i) {
        const Node &node = net_.node(static_cast<NodeId>(i));
        if (node.pin < 0) continue;
        pin_class_[i] = class_of(node.pin);
        if (node.size != 1) impossible_ = true;
    }
    for (const Root &r : net_.roots()) {
        if (r.kind != RootKind::EXACT || producer_[r.node] >= 0) return false;
        const Node &node = net_.node(r.node);
        int total = 0, pinned = 0;
        for (const auto &c : r.counts) {
            total += c.second;
            if (c.first == node.pin) pinned += c.second;
        }
        if (total != node.size || (node.pin >= 0 && pinned != 1)) impossible_ = true;
    }
    return order_junctions();
}

// The weight of one way the junction's demand is met (split): the demanded out slots are mapped to
// distinct in slots, every injection equally likely; `d` and `known_used` say how many of each
// class land on each target and on the known slots.
void LikelihoodSweep::leaf(const Step &st, const std::vector<int> &known_used, const std::vector<std::vector<int>> &d,
    double lw, double w, const Key &kept, std::map<Key, double> *next) {
    if (++steps_ > budget_) {
        over_ = true;
        return;
    }
    const size_t K = cards_.size();
    double l = lw;
    for (size_t k = 0; k < K; ++k) l += log_falling(st.known[k], known_used[k]) - log_factorial(known_used[k]);
    Key key = kept;  // the junctions already in the state, then the in nodes' producers
    for (size_t t = 0; t < st.targets.size(); ++t) {
        const Target &tg = st.targets[t];
        int total = 0;
        for (size_t k = 0; k < K; ++k) {
            total += d[t][k];
            l -= log_factorial(d[t][k]);
            if (!tg.counts.empty()) l += log_falling(tg.counts[k], d[t][k]);
        }
        if (tg.counts.empty()) {
            l += log_falling(tg.size, total);
            for (size_t k = 0; k < K; ++k) key[tg.slot * K + k] = static_cast<int16_t>(key[tg.slot * K + k] + d[t][k]);
        }
    }
    if (!std::isfinite(l)) return;
    (*next)[key] += w * std::exp(l);
}

// Every split of the demand among the in side, class by class: `stage` 0 is the class's known slots,
// stage s the target s - 1; `left` is what is left of class c. `room` is each target's slots not yet
// demanded.
void LikelihoodSweep::split(const Step &st, size_t c, size_t stage, int left, std::vector<int> &known_used,
    std::vector<std::vector<int>> &d, std::vector<int> &room, double lw, double w, const Key &kept,
    std::map<Key, double> *next) {
    if (over_) return;
    const size_t K = cards_.size(), T = st.targets.size();
    if (c == K) {
        leaf(st, known_used, d, lw, w, kept, next);
        return;
    }
    if (stage == 0) {
        if (left == 0) {
            split(st, c + 1, 0, c + 1 < K ? st.base[c + 1] : 0, known_used, d, room, lw, w, kept, next);
            return;
        }
        const int top = std::min(left, st.known[c]);
        for (int a = 0; a <= top && !over_; ++a) {
            known_used[c] = a;
            split(st, c, 1, left - a, known_used, d, room, lw, w, kept, next);
        }
        known_used[c] = 0;
        return;
    }
    const size_t t = stage - 1;
    if (t == T) {
        if (left == 0) split(st, c + 1, 0, c + 1 < K ? st.base[c + 1] : 0, known_used, d, room, lw, w, kept, next);
        return;
    }
    const int cap = st.targets[t].counts.empty() ? room[t] : std::min(room[t], st.targets[t].counts[c]);
    // The last target takes the rest.
    const int lo = t + 1 == T ? left : 0;
    const int hi = std::min(left, cap);
    for (int v = lo; v <= hi && !over_; ++v) {
        d[t][c] = v;
        room[t] -= v;
        split(st, c, stage + 1, left - v, known_used, d, room, lw, w, kept, next);
        room[t] += v;
    }
    d[t][c] = 0;
}

// Junction j applied to every state, backwards: the demand on its out side (its pinned outs and fixed
// cards, and the carried outs' demands from the state) is met by its in side, every injection of the
// demanded slots into the in slots equally likely; a demand landing on a carried in node becomes that
// node's demand. Then the weights are rescaled so the largest is 1.
bool LikelihoodSweep::apply(uint32_t j) {
    const Junction &jn = net_.junction(j);
    const size_t K = cards_.size();
    Step st;
    st.base.assign(K, 0);
    st.known.assign(K, 0);
    bool carried_out = false;
    for (NodeId m : jn.out) {
        if (pin_class_[m] >= 0)
            ++st.base[static_cast<size_t>(pin_class_[m])];
        else if (consumer_[m] >= 0)
            carried_out = true;
    }
    const auto at = std::find(frontier_.begin(), frontier_.end(), j);
    if (at != frontier_.end()) st.out_at = static_cast<int>(at - frontier_.begin());
    if (carried_out && st.out_at < 0) return false;
    // The next state's junctions: the others already there, then the producers of the carried in
    // nodes.
    std::vector<uint32_t> frontier;
    for (uint32_t f : frontier_)
        if (f != j) frontier.push_back(f);
    for (const auto &c : jn.out_fixed) st.base[static_cast<size_t>(class_of(c.first))] += c.second;
    for (const auto &c : jn.in_fixed) {
        st.known[static_cast<size_t>(class_of(c.first))] += c.second;
        st.n += c.second;
    }
    for (NodeId m : jn.in) {
        const Node &node = net_.node(m);
        st.n += node.size;
        if (pin_class_[m] >= 0) {
            ++st.known[static_cast<size_t>(pin_class_[m])];
            continue;
        }
        Target t;
        t.size = node.size;
        if (node.root >= 0) {
            t.counts.assign(K, 0);
            for (const auto &c : net_.roots()[static_cast<size_t>(node.root)].counts) {
                const int k = class_of(c.first);
                if (k >= 0) t.counts[static_cast<size_t>(k)] += c.second;
            }
        } else {
            const uint32_t pj = static_cast<uint32_t>(producer_[m]);
            const auto f = std::find(frontier.begin(), frontier.end(), pj);
            t.slot = static_cast<size_t>(f - frontier.begin());
            if (f == frontier.end()) frontier.push_back(pj);
        }
        st.targets.push_back(t);
    }

    std::map<Key, double> next;
    Step cur = st;
    std::vector<int> known_used(K, 0), room(st.targets.size(), 0);
    std::vector<std::vector<int>> d(st.targets.size(), std::vector<int>(K, 0));
    for (const auto &state : states_) {
        cur.base = st.base;
        if (st.out_at >= 0)
            for (size_t k = 0; k < K; ++k) cur.base[k] += state.first[static_cast<size_t>(st.out_at) * K + k];
        int total = 0;
        double lw = 0;
        for (size_t k = 0; k < K; ++k) {
            total += cur.base[k];
            lw += log_factorial(cur.base[k]);
        }
        if (total > st.n) continue;
        lw -= log_falling(st.n, total);
        Key kept(frontier.size() * K, 0);
        for (size_t f = 0, g = 0; f < frontier_.size(); ++f) {
            if (static_cast<int>(f) == st.out_at) continue;
            std::copy(state.first.begin() + static_cast<std::ptrdiff_t>(f * K),
                state.first.begin() + static_cast<std::ptrdiff_t>((f + 1) * K),
                kept.begin() + static_cast<std::ptrdiff_t>(g * K));
            ++g;
        }
        for (size_t t = 0; t < st.targets.size(); ++t) room[t] = st.targets[t].size;
        split(cur, 0, 0, K > 0 ? cur.base[0] : 0, known_used, d, room, lw, state.second, kept, &next);
        if (over_) return false;
    }
    frontier_.swap(frontier);
    double top = 0;
    for (const auto &s : next) top = std::max(top, s.second);
    if (top > 0) {
        for (auto &s : next) s.second /= top;
        scale_ += std::log(top);
    }
    states_.swap(next);
    return true;
}

Likelihood LikelihoodSweep::run() {
    Likelihood res;
    if (!prepare()) return res;
    res.finished = true;
    if (impossible_) return res;
    states_[Key()] = 1.0;
    for (size_t o = order_.size(); o-- > 0;) {
        ++steps_;
        if (!apply(order_[o])) {
            res.finished = false;
            res.steps = steps_;
            return res;
        }
        if (states_.empty()) break;
    }
    res.steps = steps_;
    double total = 0;
    for (const auto &s : states_) total += s.second;
    res.possible = total > 0;
    if (res.possible) res.log_p = std::log(total) + scale_;
    return res;
}

Likelihood network_likelihood(const Network &net, uint64_t budget) {
    LikelihoodSweep s(net, budget);
    return s.run();
}

double seen_log_likelihood(const MetaList &list, const std::map<int, uint8_t> &seen) {
    std::map<int, int> t;
    int n = 0;
    for (const auto &c : list.main) {
        t[c.first] += c.second;
        n += c.second;
    }
    for (const auto &c : list.side) {
        t[c.first] += c.second;
        n += c.second;
    }
    double ll = 0;
    int m = 0;
    for (const auto &s : seen) {
        const int k = s.second, have = t.count(s.first) ? t[s.first] : 0;
        if (k == 0) continue;
        if (have < k) return -INFINITY;
        ll += log_falling(have, k);
        m += k;
    }
    if (n < m) return -INFINITY;
    return ll - log_falling(n, m);
}

std::vector<double> posterior(const std::vector<double> &log_w, const std::vector<uint8_t> &support) {
    std::vector<double> out(log_w.size(), 0.0);
    double top = -INFINITY;
    for (size_t i = 0; i < log_w.size(); ++i)
        if (support[i] && std::isfinite(log_w[i])) top = std::max(top, log_w[i]);
    if (!std::isfinite(top)) return out;
    double total = 0;
    for (size_t i = 0; i < log_w.size(); ++i)
        if (support[i] && std::isfinite(log_w[i])) {
            out[i] = std::exp(log_w[i] - top);
            total += out[i];
        }
    for (double &v : out) v /= total;
    return out;
}

std::vector<uint32_t> fixed_point_posterior(const std::vector<double> &log_w, const std::vector<uint8_t> &support) {
    const size_t n = log_w.size();
    std::vector<uint32_t> out(n, 0);
    std::vector<size_t> members;
    for (size_t i = 0; i < n; ++i)
        if (support[i]) members.push_back(i);
    if (members.empty()) return out;
    // A member whose weight is not finite (none should be) shares the floor of one unit only.
    std::vector<double> p = posterior(log_w, support);
    const uint64_t spare = PROB_SCALE - members.size();
    std::vector<std::pair<double, size_t>> frac;
    uint64_t given = 0;
    for (size_t i : members) {
        const double q = p[i] * static_cast<double>(spare);
        uint64_t f = static_cast<uint64_t>(std::floor(q));
        if (f > spare) f = spare;
        out[i] = static_cast<uint32_t>(1 + f);
        given += f;
        frac.emplace_back(q - static_cast<double>(f), i);
    }
    // Largest remainder first, ties to the lower index.
    std::sort(frac.begin(), frac.end(), [](const std::pair<double, size_t> &a, const std::pair<double, size_t> &b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    });
    while (given < spare)
        for (size_t r = 0; r < frac.size() && given < spare; ++r) {
            ++out[frac[r].second];
            ++given;
        }
    while (given > spare)
        for (size_t r = frac.size(); r-- > 0 && given > spare;)
            if (out[frac[r].second] > 1) {
                --out[frac[r].second];
                --given;
            }
    return out;
}

int32_t loglik_column(double log_p) {
    if (!std::isfinite(log_p)) return log_p > 0 ? INT32_MAX : INT32_MIN;
    const double v = std::round(log_p * 1000.0);
    if (v <= static_cast<double>(INT32_MIN + 1)) return INT32_MIN + 1;
    if (v >= static_cast<double>(INT32_MAX)) return INT32_MAX;
    return static_cast<int32_t>(v);
}

}  // namespace knowledge
