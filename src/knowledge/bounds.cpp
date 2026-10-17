#include "bounds.h"

#include <algorithm>

#include "serial.h"

namespace knowledge {

size_t Universe::index_of(int card) const {
    const auto it = std::lower_bound(cards.begin(), cards.end(), card);
    return it != cards.end() && *it == card ? static_cast<size_t>(it - cards.begin()) : cards.size();
}

bool Universe::contains(int card) const {
    return std::binary_search(cards.begin(), cards.end(), card);
}

bool Universe::add(int card) {
    const auto it = std::lower_bound(cards.begin(), cards.end(), card);
    if (it != cards.end() && *it == card) return false;
    cards.insert(it, card);
    return true;
}

void BoundStore::reset() {
    classes_ = 0;
    lo_.clear();
    hi_.clear();
    nodes_done_ = junctions_done_ = pins_done_ = 0;
    contradiction_.clear();
    budget_hit_ = false;
}

int BoundStore::lo(const Network &net, NodeId n, size_t k) const {
    (void)net;
    return n < nodes_done_ ? lo_[n * classes_ + k] : 0;
}

int BoundStore::hi(const Network &net, NodeId n, size_t k) const {
    return n < nodes_done_ ? hi_[n * classes_ + k] : net.node(n).size;
}

void BoundStore::fail(const std::string &what) {
    if (contradiction_.empty()) contradiction_ = what;
}

// Narrows class `k` of node `n` to [lo, hi] ∩ its interval. True when it changed; an empty
// result is a contradiction (recorded, the interval kept).
bool BoundStore::narrow(NodeId n, size_t k, int lo, int hi) {
    const size_t at = n * classes_ + k;
    const int old_lo = lo_[at], old_hi = hi_[at];
    const int new_lo = std::max(old_lo, lo), new_hi = std::min(old_hi, hi);
    if (new_lo > new_hi) {
        fail("node " + std::to_string(n) + " class " + std::to_string(k) + ": [" + std::to_string(old_lo) + "," +
             std::to_string(old_hi) + "] ∩ [" + std::to_string(lo) + "," + std::to_string(hi) + "] is empty");
        return false;
    }
    if (new_lo == old_lo && new_hi == old_hi) return false;
    lo_[at] = static_cast<uint8_t>(new_lo);
    hi_[at] = static_cast<uint8_t>(new_hi);
    return true;
}

void BoundStore::init_node(const Network &net, NodeId n) {
    const Node &node = net.node(n);
    const size_t base = n * classes_;
    for (size_t k = 0; k < classes_; ++k) {
        lo_[base + k] = 0;
        hi_[base + k] = static_cast<uint8_t>(std::min<int>(node.size, 255));
    }
    if (node.pin >= 0) apply_pin(net, n);
    if (node.root >= 0 && use_roots_) {
        const Root &r = net.roots()[static_cast<size_t>(node.root)];
        if (r.kind == RootKind::EXACT) {
            std::vector<int> exact(classes_, 0);
            for (const auto &c : r.counts) exact[universe.index_of(c.first)] += c.second;
            for (size_t k = 0; k < classes_; ++k) narrow(n, k, exact[k], exact[k]);
        } else if (r.kind == RootKind::LOWER) {
            for (const auto &c : r.counts) narrow(n, universe.index_of(c.first), c.second, 255);
        }
    }
    tighten_size(net, n);
}

void BoundStore::apply_pin(const Network &net, NodeId n) {
    const Node &node = net.node(n);
    if (node.pin < 0) return;
    const size_t pk = universe.index_of(node.pin);
    for (size_t k = 0; k < classes_; ++k) narrow(n, k, k == pk ? 1 : 0, k == pk ? 1 : 0);
}

// A node's classes sum to its size: each class is at least the size less the others' maxima and
// at most the size less the others' minima. Repeated until stable. True when anything changed.
bool BoundStore::tighten_size(const Network &net, NodeId n) {
    const int size = net.node(n).size;
    const size_t base = n * classes_;
    bool any = false;
    for (int pass = 0; pass < 8; ++pass) {
        int sum_lo = 0, sum_hi = 0;
        for (size_t k = 0; k < classes_; ++k) {
            sum_lo += lo_[base + k];
            sum_hi += hi_[base + k];
        }
        if (sum_lo > size || sum_hi < size) {
            fail("node " + std::to_string(n) + " of size " + std::to_string(size) + " has class bounds summing to [" +
                 std::to_string(sum_lo) + "," + std::to_string(sum_hi) + "]");
            return any;
        }
        bool changed = false;
        for (size_t k = 0; k < classes_; ++k) {
            const int l = lo_[base + k], h = hi_[base + k];
            changed |= narrow(n, k, size - (sum_hi - h), size - (sum_lo - l));
        }
        if (!changed) break;
        any = true;
    }
    return any;
}

void BoundStore::enqueue(uint32_t j) {
    if (j == NO_JUNCTION || queued_[j]) return;
    queued_[j] = 1;
    queue_.push_back(j);
}

void BoundStore::node_changed(const Network &net, NodeId n) {
    tighten_size(net, n);
    const Node &node = net.node(n);
    enqueue(node.in);
    enqueue(node.out);
    if (node.loc != Loc::GONE || node.root >= 0) cut_queued_ = true;
}

// Per class of the universe, the copies `c` lists.
void BoundStore::class_counts(const CardCounts &c, std::vector<int> *out) const {
    out->assign(classes_, 0);
    for (const auto &e : c) (*out)[universe.index_of(e.first)] += e.second;
}

// The in nodes' units plus `in_fixed` are the out nodes' units plus `out_fixed`, per class: each
// member's interval is narrowed to what the other side's sum leaves after its own side's other
// members and fixed cards.
void BoundStore::eval_relation(const Network &net, const std::vector<NodeId> &in, const std::vector<NodeId> &out,
    const std::vector<int> &in_fixed, const std::vector<int> &out_fixed) {
    changed_.clear();
    for (size_t k = 0; k < classes_; ++k) {
        int in_lo = in_fixed[k], in_hi = in_fixed[k], out_lo = out_fixed[k], out_hi = out_fixed[k];
        for (NodeId n : in) in_lo += lo_[n * classes_ + k], in_hi += hi_[n * classes_ + k];
        for (NodeId n : out) out_lo += lo_[n * classes_ + k], out_hi += hi_[n * classes_ + k];
        if (in_lo > out_hi || out_lo > in_hi) {
            fail("a junction's sides cannot balance in class " + std::to_string(k) + ": in [" + std::to_string(in_lo) +
                 "," + std::to_string(in_hi) + "], out [" + std::to_string(out_lo) + "," + std::to_string(out_hi) +
                 "]");
            continue;
        }
        for (NodeId n : in) {
            const int l = lo_[n * classes_ + k], h = hi_[n * classes_ + k];
            if (narrow(n, k, out_lo - (in_hi - h), out_hi - (in_lo - l)) && !is_changed_[n]) {
                is_changed_[n] = 1;
                changed_.push_back(n);
            }
        }
        for (NodeId n : out) {
            const int l = lo_[n * classes_ + k], h = hi_[n * classes_ + k];
            if (narrow(n, k, in_lo - (out_hi - h), in_hi - (out_lo - l)) && !is_changed_[n]) {
                is_changed_[n] = 1;
                changed_.push_back(n);
            }
        }
    }
    const std::vector<NodeId> changed = changed_;
    for (NodeId n : changed) {
        is_changed_[n] = 0;
        node_changed(net, n);
    }
}

uint64_t BoundStore::propagate(const Network &net, const std::vector<NodeId> &live, uint64_t budget) {
    if (classes_ != universe.classes()) {
        reset();
        classes_ = universe.classes();
    }
    const size_t n_nodes = net.node_count();
    lo_.resize(n_nodes * classes_, 0);
    hi_.resize(n_nodes * classes_, 0);
    queued_.assign(net.junction_count(), 0);
    is_changed_.assign(n_nodes, 0);
    queue_.clear();
    cut_queued_ = true;
    live_ = &live;
    roots_.clear();
    for (const Root &r : net.roots()) roots_.push_back(r.node);

    for (size_t n = nodes_done_; n < n_nodes; ++n) init_node(net, static_cast<NodeId>(n));
    for (size_t n = nodes_done_; n < n_nodes; ++n) {
        enqueue(net.node(static_cast<NodeId>(n)).in);
        enqueue(net.node(static_cast<NodeId>(n)).out);
    }
    const std::vector<NodeId> &pins = net.pin_log();
    for (size_t i = pins_done_; i < pins.size(); ++i) {
        if (pins[i] >= nodes_done_) continue;  // initialised with its pin above
        apply_pin(net, pins[i]);
        node_changed(net, pins[i]);
    }
    for (size_t j = junctions_done_; j < net.junction_count(); ++j) enqueue(static_cast<uint32_t>(j));
    cut_in_fixed_.assign(classes_, 0);
    cut_out_fixed_.assign(classes_, 0);
    for (size_t j = 0; j < net.junction_count(); ++j) {
        const Junction &jn = net.junction(static_cast<uint32_t>(j));
        for (const auto &e : jn.in_fixed) cut_in_fixed_[universe.index_of(e.first)] += e.second;
        for (const auto &e : jn.out_fixed) cut_out_fixed_[universe.index_of(e.first)] += e.second;
    }
    nodes_done_ = n_nodes;
    pins_done_ = pins.size();
    junctions_done_ = net.junction_count();

    uint64_t steps = 0;
    size_t head = 0;
    while (head < queue_.size() || cut_queued_) {
        if (steps >= budget) {
            budget_hit_ = true;
            break;
        }
        ++steps;
        if (head < queue_.size()) {
            const uint32_t j = queue_[head++];
            queued_[j] = 0;
            const Junction &jn = net.junction(j);
            class_counts(jn.in_fixed, &fixed_in_);
            class_counts(jn.out_fixed, &fixed_out_);
            eval_relation(net, jn.in, jn.out, fixed_in_, fixed_out_);
        } else {
            cut_queued_ = false;
            eval_relation(net, roots_, live, cut_in_fixed_, cut_out_fixed_);
        }
        if (head > 4096 && head * 2 > queue_.size()) {
            queue_.erase(queue_.begin(), queue_.begin() + static_cast<std::ptrdiff_t>(head));
            head = 0;
        }
    }
    queue_.clear();
    live_ = nullptr;
    return steps;
}

void BoundStore::serialize(std::string &out) const {
    put_u8(out, use_roots_ ? 1 : 0);
    put_u32(out, universe.cards.size());
    for (int c : universe.cards) put_i32(out, c);
    put_u32(out, classes_);
    put_u64(out, nodes_done_);
    put_u64(out, junctions_done_);
    put_u64(out, pins_done_);
    put_str(out, contradiction_);
    put_u8(out, budget_hit_ ? 1 : 0);
    put_str(out, std::string(lo_.begin(), lo_.end()));
    put_str(out, std::string(hi_.begin(), hi_.end()));
}

bool BoundStore::deserialize(const std::string &in, size_t *at) {
    ByteReader r(in, *at);
    use_roots_ = r.u8() != 0;
    const uint64_t n_cards = r.u32();
    if (!r.count(n_cards, in.size())) return false;
    universe.cards.clear();
    for (uint64_t i = 0; i < n_cards; ++i) universe.cards.push_back(r.i32());
    classes_ = static_cast<size_t>(r.u32());
    nodes_done_ = static_cast<size_t>(r.u64());
    junctions_done_ = static_cast<size_t>(r.u64());
    pins_done_ = static_cast<size_t>(r.u64());
    contradiction_ = r.str();
    budget_hit_ = r.u8() != 0;
    const std::string lo = r.str(), hi = r.str();
    lo_.assign(lo.begin(), lo.end());
    hi_.assign(hi.begin(), hi.end());
    *at = r.at;
    return r.ok && lo_.size() == nodes_done_ * classes_ && hi_.size() == lo_.size();
}

}  // namespace knowledge
