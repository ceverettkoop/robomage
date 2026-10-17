#include "enumerate.h"

#include <algorithm>
#include <map>
#include <set>

namespace rmknow_test {

namespace {

// A frontier state: the value (count per class) of every slot (a node a later junction consumes or
// a group reads), as one flat vector in slot order; a slot not on the frontier holds zeros.
using State = std::vector<int>;

// The sweep (enumerate.h).
class Sweep {
    public:
        Sweep(const EnumSpec &spec, const std::vector<std::vector<int>> &groups, uint64_t cap);
        EnumResult run();

    private:
        void node_options(int n, std::vector<std::vector<int>> *out) const;
        bool order_junctions(std::vector<int> *order) const;
        void place(State &st, int n, const std::vector<int> &v) const;
        void canonical(State &st) const;
        void divide(
            const std::vector<int> &total, const std::vector<int> &outs, size_t i, State &st, std::set<State> *out);

        const EnumSpec &spec_;
        const std::vector<std::vector<int>> &groups_;
        uint64_t cap_;
        int K_;
        std::vector<int> producer_, consumer_;
        std::vector<std::vector<int>> node_groups_;
        std::vector<int> slot_;       // per node: its slot, -1 when nothing reads it later
        std::vector<int> slot_node_;  // per slot: its node
        std::vector<int> twin_key_;   // per slot: slots with one key have the same size, consumer, groups
        std::vector<bool> live_;      // per slot: on the frontier (made, not yet consumed)
        bool over_ = false;
};

}  // namespace

static void vectors_of_size(
    int size, const std::vector<int> *limit, std::vector<int> &cur, size_t k, std::vector<std::vector<int>> *out);
static double factorial(int n);
static void weighted_divide(const EnumSpec &spec, const std::vector<int> &rem, const std::vector<int> &outs, size_t i,
    const std::vector<int> &slot, int K, double w, State &st, std::map<State, double> *out);

// Every count vector of `size` over cur.size() classes (at most `limit` per class when given).
static void vectors_of_size(
    int size, const std::vector<int> *limit, std::vector<int> &cur, size_t k, std::vector<std::vector<int>> *out) {
    if (k + 1 == cur.size()) {
        if (limit && size > (*limit)[k]) return;
        cur[k] = size;
        out->push_back(cur);
        return;
    }
    const int top = limit ? std::min(size, (*limit)[k]) : size;
    for (int c = 0; c <= top; ++c) {
        cur[k] = c;
        vectors_of_size(size - c, limit, cur, k + 1, out);
    }
}

Sweep::Sweep(const EnumSpec &spec, const std::vector<std::vector<int>> &groups, uint64_t cap)
    : spec_(spec), groups_(groups), cap_(cap), K_(spec.K) {
    const size_t n = spec.nodes.size();
    producer_.assign(n, -1);
    consumer_.assign(n, -1);
    for (size_t j = 0; j < spec.junctions.size(); ++j) {
        for (int m : spec.junctions[j].in) consumer_[static_cast<size_t>(m)] = static_cast<int>(j);
        for (int m : spec.junctions[j].out) producer_[static_cast<size_t>(m)] = static_cast<int>(j);
    }
    node_groups_.resize(n);
    for (size_t g = 0; g < groups.size(); ++g)
        for (int m : groups[g]) node_groups_[static_cast<size_t>(m)].push_back(static_cast<int>(g));
    slot_.assign(n, -1);
    std::map<std::vector<int>, int> keys;
    for (size_t i = 0; i < n; ++i) {
        if (consumer_[i] < 0 && node_groups_[i].empty()) continue;
        slot_[i] = static_cast<int>(slot_node_.size());
        slot_node_.push_back(static_cast<int>(i));
        std::vector<int> key = {spec.nodes[i].size, consumer_[i]};
        key.insert(key.end(), node_groups_[i].begin(), node_groups_[i].end());
        const auto it = keys.emplace(key, static_cast<int>(keys.size())).first;
        twin_key_.push_back(it->second);
    }
    live_.assign(slot_node_.size(), false);
}

// The values node `n` may take when it is made: its pin, and its root's exact contents or lower
// bounds.
void Sweep::node_options(int n, std::vector<std::vector<int>> *out) const {
    const EnumSpec::Node &node = spec_.nodes[static_cast<size_t>(n)];
    std::vector<int> cur(static_cast<size_t>(K_), 0);
    std::vector<std::vector<int>> all;
    vectors_of_size(node.size, nullptr, cur, 0, &all);
    for (const std::vector<int> &v : all) {
        bool ok = node.pin < 0 || v[static_cast<size_t>(node.pin)] == 1;
        for (const EnumSpec::Root &r : spec_.roots)
            if (r.node == n)
                for (size_t k = 0; k < v.size(); ++k)
                    if ((r.kind == 0 && v[k] != r.counts[k]) || (r.kind == 1 && v[k] < r.counts[k])) ok = false;
        if (ok) out->push_back(v);
    }
}

// The junctions in an order where each comes after the junctions that made its in nodes.
bool Sweep::order_junctions(std::vector<int> *order) const {
    const size_t nj = spec_.junctions.size();
    std::vector<int> waiting(nj, 0);
    for (size_t j = 0; j < nj; ++j)
        for (int m : spec_.junctions[j].in)
            if (producer_[static_cast<size_t>(m)] >= 0) ++waiting[j];
    std::vector<bool> done(nj, false);
    while (order->size() < nj) {
        bool progress = false;
        for (size_t j = 0; j < nj; ++j) {
            if (done[j] || waiting[j] > 0) continue;
            done[j] = true;
            progress = true;
            order->push_back(static_cast<int>(j));
            for (int m : spec_.junctions[j].out) {
                const int c = consumer_[static_cast<size_t>(m)];
                if (c >= 0) --waiting[static_cast<size_t>(c)];
            }
        }
        if (!progress) return false;
    }
    return true;
}

// Writes node `n`'s value into its slot of `st` (a node nothing reads later has none).
void Sweep::place(State &st, int n, const std::vector<int> &v) const {
    const int s = slot_[static_cast<size_t>(n)];
    if (s < 0) return;
    std::copy(v.begin(), v.end(), st.begin() + static_cast<std::ptrdiff_t>(s) * K_);
}

// Sorts the values within each set of interchangeable frontier slots (same size, same consumer,
// same groups): states that differ only by such a permutation have the same futures.
void Sweep::canonical(State &st) const {
    std::map<int, std::vector<int>> by_key;
    for (size_t s = 0; s < twin_key_.size(); ++s)
        if (live_[s]) by_key[twin_key_[s]].push_back(static_cast<int>(s));
    for (const auto &e : by_key) {
        if (e.second.size() < 2) continue;
        std::vector<std::vector<int>> vals;
        for (int s : e.second)
            vals.emplace_back(
                st.begin() + static_cast<std::ptrdiff_t>(s) * K_, st.begin() + static_cast<std::ptrdiff_t>(s + 1) * K_);
        std::sort(vals.begin(), vals.end());
        for (size_t i = 0; i < e.second.size(); ++i)
            std::copy(vals[i].begin(), vals[i].end(), st.begin() + static_cast<std::ptrdiff_t>(e.second[i]) * K_);
    }
}

// Every way to give the out nodes `outs[i..]` values of their sizes summing to `total` (each
// allowed by its pin), each written into `st` and collected (canonical) into `out`.
void Sweep::divide(
    const std::vector<int> &total, const std::vector<int> &outs, size_t i, State &st, std::set<State> *out) {
    if (over_) return;
    if (i == outs.size()) {
        for (int c : total)
            if (c != 0) return;
        State c = st;
        canonical(c);
        out->insert(c);
        if (out->size() > cap_) over_ = true;
        return;
    }
    const int n = outs[i];
    const EnumSpec::Node &node = spec_.nodes[static_cast<size_t>(n)];
    std::vector<std::vector<int>> vs;
    if (i + 1 == outs.size()) {
        int sum = 0;
        for (int c : total) sum += c;
        if (sum != node.size) return;
        vs.push_back(total);
    } else {
        std::vector<int> cur(static_cast<size_t>(K_), 0);
        vectors_of_size(node.size, &total, cur, 0, &vs);
    }
    std::vector<int> rest(total.size());
    for (const std::vector<int> &v : vs) {
        if (node.pin >= 0 && v[static_cast<size_t>(node.pin)] != 1) continue;
        for (size_t k = 0; k < v.size(); ++k) rest[k] = total[k] - v[k];
        place(st, n, v);
        divide(rest, outs, i + 1, st, out);
    }
}

EnumResult Sweep::run() {
    EnumResult res;
    const size_t n = spec_.nodes.size();
    std::vector<int> order;
    if (!order_junctions(&order)) return res;

    // The nodes no junction makes (roots) start the frontier, every combination of their options.
    std::set<State> states = {State(slot_node_.size() * static_cast<size_t>(K_), 0)};
    for (size_t i = 0; i < n; ++i) {
        if (producer_[i] >= 0) continue;
        std::vector<std::vector<int>> opts;
        node_options(static_cast<int>(i), &opts);
        if (slot_[i] >= 0) live_[static_cast<size_t>(slot_[i])] = true;
        std::set<State> next;
        for (const State &st : states)
            for (const std::vector<int> &v : opts) {
                State c = st;
                place(c, static_cast<int>(i), v);
                canonical(c);
                next.insert(c);
                if (next.size() > cap_) return res;
            }
        states.swap(next);
    }
    // Each junction consumes its in nodes and makes its out nodes: every division of the in nodes'
    // total plus its in-side fixed cards less its out-side fixed cards among the out nodes (made
    // with their pins).
    std::vector<int> total(static_cast<size_t>(K_));
    for (int j : order) {
        const EnumSpec::Junction &jn = spec_.junctions[static_cast<size_t>(j)];
        for (int m : jn.in) live_[static_cast<size_t>(slot_[static_cast<size_t>(m)])] = false;
        for (int m : jn.out)
            if (slot_[static_cast<size_t>(m)] >= 0) live_[static_cast<size_t>(slot_[static_cast<size_t>(m)])] = true;
        std::set<State> next;
        for (const State &st : states) {
            for (size_t k = 0; k < total.size(); ++k) total[k] = jn.in_fixed[k] - jn.out_fixed[k];
            for (int m : jn.in) {
                const size_t s = static_cast<size_t>(slot_[static_cast<size_t>(m)]);
                for (size_t k = 0; k < total.size(); ++k) total[k] += st[s * total.size() + k];
            }
            bool ok = true;
            for (int c : total) ok = ok && c >= 0;
            if (!ok) continue;
            State c = st;
            for (int m : jn.in) {
                const std::ptrdiff_t s = slot_[static_cast<size_t>(m)];
                std::fill(c.begin() + s * K_, c.begin() + (s + 1) * K_, 0);
            }
            divide(total, jn.out, 0, c, &next);
            if (over_) return res;
        }
        states.swap(next);
    }
    res.complete = true;
    res.worlds = states.size();
    res.lo.assign(groups_.size(), std::vector<int>(static_cast<size_t>(K_), 1 << 20));
    res.hi.assign(groups_.size(), std::vector<int>(static_cast<size_t>(K_), -1));
    for (const State &st : states)
        for (size_t g = 0; g < groups_.size(); ++g)
            for (size_t k = 0; k < static_cast<size_t>(K_); ++k) {
                int c = 0;
                for (int m : groups_[g])
                    c += st[static_cast<size_t>(slot_[static_cast<size_t>(m)]) * static_cast<size_t>(K_) + k];
                res.lo[g][k] = std::min(res.lo[g][k], c);
                res.hi[g][k] = std::max(res.hi[g][k], c);
            }
    return res;
}

static double factorial(int n) {
    double f = 1;
    for (int i = 2; i <= n; ++i) f *= i;
    return f;
}

// Every division of `rem` among the out nodes `outs[i..]` (their sizes and pins), each adding
// w / Π_o Π_c x_oc! to `out` (the rest of the division's probability is the caller's).
static void weighted_divide(const EnumSpec &spec, const std::vector<int> &rem, const std::vector<int> &outs, size_t i,
    const std::vector<int> &slot, int K, double w, State &st, std::map<State, double> *out) {
    if (i == outs.size()) {
        for (int c : rem)
            if (c != 0) return;
        (*out)[st] += w;
        return;
    }
    const EnumSpec::Node &node = spec.nodes[static_cast<size_t>(outs[i])];
    std::vector<std::vector<int>> vs;
    std::vector<int> cur(static_cast<size_t>(K), 0);
    vectors_of_size(node.size, &rem, cur, 0, &vs);
    std::vector<int> rest(rem.size());
    for (const std::vector<int> &v : vs) {
        if (node.pin >= 0 && v[static_cast<size_t>(node.pin)] != 1) continue;
        double f = 1;
        for (size_t k = 0; k < v.size(); ++k) {
            rest[k] = rem[k] - v[k];
            f *= factorial(v[k]);
        }
        const int s = slot[static_cast<size_t>(outs[i])];
        if (s >= 0) std::copy(v.begin(), v.end(), st.begin() + static_cast<std::ptrdiff_t>(s) * K);
        weighted_divide(spec, rest, outs, i + 1, slot, K, w / f, st, out);
    }
    const int s = slot[static_cast<size_t>(outs[i])];
    if (s >= 0)
        std::fill(
            st.begin() + static_cast<std::ptrdiff_t>(s) * K, st.begin() + static_cast<std::ptrdiff_t>(s + 1) * K, 0);
}

double weighted_world_probability(const EnumSpec &spec, uint64_t cap, bool *complete) {
    *complete = false;
    const size_t n = spec.nodes.size();
    const int K = spec.K;
    std::vector<int> producer(n, -1), consumer(n, -1);
    for (size_t j = 0; j < spec.junctions.size(); ++j) {
        for (int m : spec.junctions[j].in) consumer[static_cast<size_t>(m)] = static_cast<int>(j);
        for (int m : spec.junctions[j].out) producer[static_cast<size_t>(m)] = static_cast<int>(j);
    }
    // Junction order: each after the junctions that made its in nodes.
    std::vector<int> order;
    std::vector<bool> done(spec.junctions.size(), false);
    while (order.size() < spec.junctions.size()) {
        bool progress = false;
        for (size_t j = 0; j < spec.junctions.size(); ++j) {
            if (done[j]) continue;
            bool ready = true;
            for (int m : spec.junctions[j].in) {
                const int pj = producer[static_cast<size_t>(m)];
                if (pj >= 0 && !done[static_cast<size_t>(pj)]) ready = false;
            }
            if (!ready) continue;
            done[j] = true;
            progress = true;
            order.push_back(static_cast<int>(j));
        }
        if (!progress) return 0;
    }
    // A slot per node a junction consumes.
    std::vector<int> slot(n, -1);
    int slots = 0;
    for (size_t i = 0; i < n; ++i)
        if (consumer[i] >= 0) slot[i] = slots++;
    State start(static_cast<size_t>(slots) * static_cast<size_t>(K), 0);
    for (const EnumSpec::Root &r : spec.roots) {
        if (r.kind != 0) return 0;
        const EnumSpec::Node &node = spec.nodes[static_cast<size_t>(r.node)];
        int total = 0;
        for (int c : r.counts) total += c;
        if (total != node.size || (node.pin >= 0 && r.counts[static_cast<size_t>(node.pin)] != 1)) {
            *complete = true;
            return 0;
        }
        const int s = slot[static_cast<size_t>(r.node)];
        if (s >= 0) std::copy(r.counts.begin(), r.counts.end(), start.begin() + static_cast<std::ptrdiff_t>(s) * K);
    }
    std::map<State, double> states = {{start, 1.0}};
    for (int j : order) {
        const EnumSpec::Junction &jn = spec.junctions[static_cast<size_t>(j)];
        std::map<State, double> next;
        for (const auto &e : states) {
            std::vector<int> y(jn.in_fixed);
            for (int m : jn.in) {
                const size_t s = static_cast<size_t>(slot[static_cast<size_t>(m)]);
                for (size_t k = 0; k < y.size(); ++k) y[k] += e.first[s * static_cast<size_t>(K) + k];
            }
            std::vector<int> rem(y.size());
            int units = 0;
            bool ok = true;
            double w = e.second;
            for (size_t k = 0; k < y.size(); ++k) {
                units += y[k];
                rem[k] = y[k] - jn.out_fixed[k];
                if (rem[k] < 0) ok = false;
                w *= factorial(y[k]);
            }
            if (!ok) continue;
            w /= factorial(units);
            for (int m : jn.out) w *= factorial(spec.nodes[static_cast<size_t>(m)].size);
            State st = e.first;
            for (int m : jn.in) {
                const std::ptrdiff_t s = slot[static_cast<size_t>(m)];
                std::fill(st.begin() + s * K, st.begin() + (s + 1) * K, 0);
            }
            weighted_divide(spec, rem, jn.out, 0, slot, K, w, st, &next);
            if (next.size() > cap) return 0;
        }
        states.swap(next);
    }
    *complete = true;
    double total = 0;
    for (const auto &e : states) total += e.second;
    return total;
}

EnumSpec spec_of(const knowledge::Network &net, const knowledge::Universe &uni) {
    EnumSpec s;
    s.K = static_cast<int>(uni.classes());
    const auto classes = [&uni, &s](const knowledge::CardCounts &c) {
        std::vector<int> v(static_cast<size_t>(s.K), 0);
        for (const auto &e : c) v[uni.index_of(e.first)] += e.second;
        return v;
    };
    for (size_t i = 0; i < net.node_count(); ++i) {
        const knowledge::Node &n = net.node(static_cast<knowledge::NodeId>(i));
        EnumSpec::Node e;
        e.size = n.size;
        e.pin = n.pin >= 0 ? static_cast<int>(uni.index_of(n.pin)) : -1;
        s.nodes.push_back(e);
    }
    for (const knowledge::Root &r : net.roots()) {
        EnumSpec::Root e;
        e.node = static_cast<int>(r.node);
        e.kind = r.kind == knowledge::RootKind::EXACT ? 0 : r.kind == knowledge::RootKind::LOWER ? 1 : 2;
        e.counts = classes(r.counts);
        s.roots.push_back(e);
    }
    for (size_t j = 0; j < net.junction_count(); ++j) {
        const knowledge::Junction &jn = net.junction(static_cast<uint32_t>(j));
        EnumSpec::Junction e;
        e.in.assign(jn.in.begin(), jn.in.end());
        e.out.assign(jn.out.begin(), jn.out.end());
        e.in_fixed = classes(jn.in_fixed);
        e.out_fixed = classes(jn.out_fixed);
        s.junctions.push_back(e);
    }
    return s;
}

EnumResult enumerate_worlds(const EnumSpec &spec, const std::vector<std::vector<int>> &groups, uint64_t cap) {
    Sweep s(spec, groups, cap);
    return s.run();
}

}  // namespace rmknow_test
