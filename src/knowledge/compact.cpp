#include "compact.h"

#include <algorithm>
#include <map>

namespace knowledge {

namespace {

constexpr uint32_t NONE = 0xFFFFFFFFu;
constexpr int NO_ROOT = -1;

// A node of the network being compacted. `root_kind` is the RootKind of a root (as an int), else
// NO_ROOT; `root` the old root index (-1 for a root compaction made).
struct WNode {
        uint16_t size = 0;
        int32_t pin = -1;
        int root_kind = NO_ROOT;
        int32_t root = -1;
        uint32_t prod = NONE;
        uint32_t cons = NONE;
        bool keep = false;
        bool alive = true;
};

struct WJunction {
        std::vector<uint32_t> in, out;
        std::map<int, int> in_fixed, out_fixed;
        bool alive = true;
};

struct Work {
        std::vector<WNode> nodes;
        std::vector<WJunction> js;
};

}  // namespace

static Work load(const Network &net, const std::vector<uint8_t> &keep);
static bool eliminable(const WNode &n);
static void erase_one(std::vector<uint32_t> &v, uint32_t x);
static bool pin_to_fixed(Work &w, uint32_t n);
static bool contract(Work &w, uint32_t n);
static bool drop_free_root(Work &w, uint32_t n);
static bool merge_parallel(Work &w);
static bool tidy(Work &w, uint32_t j);
static CardCounts to_counts(const std::map<int, int> &m);

static Work load(const Network &net, const std::vector<uint8_t> &keep) {
    Work w;
    w.nodes.resize(net.node_count());
    for (size_t i = 0; i < net.node_count(); ++i) {
        const Node &n = net.node(static_cast<NodeId>(i));
        WNode &x = w.nodes[i];
        x.size = n.size;
        x.pin = n.pin;
        x.prod = n.in;
        x.cons = n.out;
        x.keep = i < keep.size() && keep[i];
        if (n.root >= 0) {
            x.root = n.root;
            x.root_kind = static_cast<int>(net.roots()[static_cast<size_t>(n.root)].kind);
        }
    }
    w.js.resize(net.junction_count());
    for (size_t j = 0; j < net.junction_count(); ++j) {
        const Junction &jn = net.junction(static_cast<uint32_t>(j));
        WJunction &x = w.js[j];
        x.in.assign(jn.in.begin(), jn.in.end());
        x.out.assign(jn.out.begin(), jn.out.end());
        for (const auto &c : jn.in_fixed) x.in_fixed[c.first] += c.second;
        for (const auto &c : jn.out_fixed) x.out_fixed[c.first] += c.second;
    }
    return w;
}

// Neither kept, pinned nor a root: a node only its two junctions constrain.
static bool eliminable(const WNode &n) {
    return n.alive && !n.keep && n.pin < 0 && n.root_kind == NO_ROOT;
}

static void erase_one(std::vector<uint32_t> &v, uint32_t x) {
    const auto it = std::find(v.begin(), v.end(), x);
    if (it != v.end()) v.erase(it);
}

// A pinned node nothing else needs (not kept; a FREE root at most) becomes its card on both of its
// junctions.
static bool pin_to_fixed(Work &w, uint32_t n) {
    WNode &x = w.nodes[n];
    if (!x.alive || x.keep || x.pin < 0) return false;
    if (x.root_kind != NO_ROOT && x.root_kind != static_cast<int>(RootKind::FREE)) return false;
    if (x.prod != NONE) {
        erase_one(w.js[x.prod].out, n);
        w.js[x.prod].out_fixed[x.pin] += 1;
    }
    if (x.cons != NONE) {
        erase_one(w.js[x.cons].in, n);
        w.js[x.cons].in_fixed[x.pin] += 1;
    }
    x.alive = false;
    return true;
}

// Series elimination: an eliminable node that is the only member of its producer's out side or of
// its consumer's in side; the two junctions become one. A node on both sides of one junction drops.
static bool contract(Work &w, uint32_t n) {
    WNode &x = w.nodes[n];
    if (!eliminable(x) || x.prod == NONE || x.cons == NONE) return false;
    const uint32_t p = x.prod, c = x.cons;
    if (p == c) {
        erase_one(w.js[p].in, n);
        erase_one(w.js[p].out, n);
        x.alive = false;
        return true;
    }
    WJunction &P = w.js[p];
    WJunction &C = w.js[c];
    const bool sole_out = P.out.size() == 1 && P.out_fixed.empty();
    const bool sole_in = C.in.size() == 1 && C.in_fixed.empty();
    if (!sole_out && !sole_in) return false;
    erase_one(P.out, n);
    erase_one(C.in, n);
    for (uint32_t m : C.in) {
        P.in.push_back(m);
        w.nodes[m].cons = p;
    }
    for (uint32_t m : C.out) {
        P.out.push_back(m);
        w.nodes[m].prod = p;
    }
    for (const auto &f : C.in_fixed) P.in_fixed[f.first] += f.second;
    for (const auto &f : C.out_fixed) P.out_fixed[f.first] += f.second;
    C = WJunction{};
    C.alive = false;
    x.alive = false;
    return true;
}

// A FREE root that is alone on its consumer's in side (no fixed card there either): the consumer
// says nothing (the root can be anything), so it drops and its out nodes become FREE roots; its
// out side's fixed cards go with it.
static bool drop_free_root(Work &w, uint32_t n) {
    WNode &x = w.nodes[n];
    if (!x.alive || x.keep || x.pin >= 0 || x.root_kind != static_cast<int>(RootKind::FREE)) return false;
    if (x.cons == NONE) {
        x.alive = false;
        return true;
    }
    WJunction &C = w.js[x.cons];
    if (C.in.size() != 1 || !C.in_fixed.empty()) return false;
    for (uint32_t m : C.out) {
        WNode &y = w.nodes[m];
        y.prod = NONE;
        if (y.root_kind == NO_ROOT) {
            y.root_kind = static_cast<int>(RootKind::FREE);
            y.root = -1;
        }
    }
    C = WJunction{};
    C.alive = false;
    x.alive = false;
    return true;
}

// Eliminable nodes with the same producer and the same consumer are one node of their total size,
// and so are unpinned FREE roots (not kept) with the same consumer.
static bool merge_parallel(Work &w) {
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> first;
    bool any = false;
    for (uint32_t n = 0; n < w.nodes.size(); ++n) {
        WNode &x = w.nodes[n];
        const bool free_source = x.alive && !x.keep && x.pin < 0 && x.root_kind == static_cast<int>(RootKind::FREE) &&
                                 x.prod == NONE && x.cons != NONE;
        const bool internal = eliminable(x) && x.prod != NONE && x.cons != NONE && x.prod != x.cons;
        if (!free_source && !internal) continue;
        const auto key = std::make_pair(x.prod, x.cons);
        const auto it = first.find(key);
        if (it == first.end()) {
            first[key] = n;
            continue;
        }
        WNode &keepn = w.nodes[it->second];
        keepn.size = static_cast<uint16_t>(keepn.size + x.size);
        if (x.prod != NONE) erase_one(w.js[x.prod].out, n);
        erase_one(w.js[x.cons].in, n);
        x.alive = false;
        any = true;
    }
    return any;
}

// Cancels cards fixed on both sides; drops a junction with no node whose fixed cards balance.
static bool tidy(Work &w, uint32_t j) {
    WJunction &J = w.js[j];
    if (!J.alive) return false;
    bool any = false;
    for (auto it = J.in_fixed.begin(); it != J.in_fixed.end();) {
        const auto o = J.out_fixed.find(it->first);
        if (o == J.out_fixed.end()) {
            ++it;
            continue;
        }
        const int both = std::min(it->second, o->second);
        it->second -= both;
        o->second -= both;
        any = true;
        if (o->second == 0) J.out_fixed.erase(o);
        if (it->second == 0)
            it = J.in_fixed.erase(it);
        else
            ++it;
    }
    if (J.in.empty() && J.out.empty() && J.in_fixed.empty() && J.out_fixed.empty()) {
        J.alive = false;
        any = true;
    }
    return any;
}

static CardCounts to_counts(const std::map<int, int> &m) {
    CardCounts out;
    for (const auto &e : m)
        if (e.second > 0) out.emplace_back(e.first, e.second);
    return out;
}

Network compact_network(const Network &net, const std::vector<uint8_t> &keep, std::vector<NodeId> *remap) {
    Work w = load(net, keep);
    bool changed = true;
    while (changed) {
        changed = false;
        for (uint32_t n = 0; n < w.nodes.size(); ++n) {
            changed |= pin_to_fixed(w, n);
            changed |= contract(w, n);
            changed |= drop_free_root(w, n);
        }
        changed |= merge_parallel(w);
        for (uint32_t j = 0; j < w.js.size(); ++j) changed |= tidy(w, j);
    }

    Network out;
    remap->assign(net.node_count(), NO_NODE);
    for (uint32_t n = 0; n < w.nodes.size(); ++n) {
        const WNode &x = w.nodes[n];
        if (!x.alive) continue;
        const Node &old = net.node(n);
        const NodeId id = out.add_node(x.size, x.keep ? old.loc : Loc::GONE);
        Node &nn = out.node(id);
        nn.pin = x.pin;
        if (x.keep) {
            nn.drawn_turn = old.drawn_turn;
            nn.handle = old.handle;
            nn.draw_seq = old.draw_seq;
        }
        (*remap)[n] = id;
        if (x.root_kind == NO_ROOT) continue;
        const RootKind kind = static_cast<RootKind>(x.root_kind);
        if (x.root < 0) {
            out.add_root_record(id, kind, CardCounts{});
            continue;
        }
        const Root &old_root = net.roots()[static_cast<size_t>(x.root)];
        out.add_root_record(id, kind, old_root.counts, old_root.part);
    }
    for (const WJunction &J : w.js) {
        if (!J.alive) continue;
        Junction nj;
        for (uint32_t m : J.in) nj.in.push_back((*remap)[m]);
        for (uint32_t m : J.out) nj.out.push_back((*remap)[m]);
        nj.in_fixed = to_counts(J.in_fixed);
        nj.out_fixed = to_counts(J.out_fixed);
        out.add_junction(nj);
    }
    return out;
}

}  // namespace knowledge
