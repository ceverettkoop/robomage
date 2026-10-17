#include "solver.h"

#include <algorithm>
#include <numeric>

#include "compact.h"
#include "serial.h"

namespace knowledge {

namespace {

constexpr int BIG = 1 << 20;

// One junction as an equation over variables: Σ in-side variables + cin = Σ out-side variables +
// cout, per class (fixed members and fixed cards folded into the constants).
struct Row {
        std::vector<uint32_t> vars;
        std::vector<int8_t> side;  // +1 in, -1 out
        std::vector<int> cin, cout;
};

// One player's network as a system over variables (the nodes not determined by their bounds).
struct System {
        size_t K = 0;
        std::vector<uint16_t> size;   // per variable
        std::vector<int16_t> lo, hi;  // per variable × class: the seed
        std::vector<Row> rows;
        std::vector<std::vector<uint32_t>> gvars;  // per group: its variables
        std::vector<std::vector<int>> gconst;      // per group × class: its determined members' cards
        std::vector<int> gsize;
        bool contradiction = false;
};

// One component of a System, in local indices.
struct Problem {
        size_t K = 0;
        std::vector<uint16_t> size;
        std::vector<int16_t> lo, hi;
        std::vector<Row> rows;
        std::vector<std::vector<uint32_t>> groups;
};

enum class LeafKind {
    NOT_LEAF,
    EMPTY,
    WORLDS
};

// An arc of a single-card flow relaxation (Search::flow_bound): flow in [lo, hi] at `cost` a unit.
struct FlowArc {
        int from = 0, to = 0;
        int lo = 0, hi = 0;
        int cost = 0;
};

// The search of one component (solver.h, steps 3-4).
class Search {
    public:
        // `any_world`: with no answers to settle, still search for one leaf with worlds (a
        // feasibility check), which `infeasible` then reports.
        Search(const Problem &p, uint64_t budget, bool any_world = false);
        void run();

        bool finished = false;
        bool infeasible = false;
        uint64_t steps = 0;
        std::vector<int> res_lo, res_hi;  // per group × class
        std::vector<uint8_t> settled;     // per group × class: both ends exact

    private:
        struct Term {
                uint32_t cell;
                int8_t sign;
        };
        struct Con {
                uint32_t begin, end;
                int rhs;
        };
        struct State {
                std::vector<int16_t> lo, hi;
                std::vector<uint8_t> pristine;  // per variable: never branched on
        };

        uint32_t vcell(uint32_t v, size_t k) const { return static_cast<uint32_t>(v * K_ + k); }
        uint32_t gcell(uint32_t g, size_t k) const { return static_cast<uint32_t>((nv_ + g) * K_ + k); }
        void add_con(const std::vector<Term> &terms, int rhs);
        bool tick();
        bool propagate(State &s, int64_t changed);
        bool eval(State &s, const Con &c);
        bool unfixed(const State &s, uint32_t v) const;
        bool slack_ok(uint32_t v) const;
        LeafKind leaf(const State &s);
        bool pick(const State &s, uint32_t *var, uint32_t *cell) const;
        bool reached() const;
        bool dfs(State &s, int64_t changed);
        bool flow_bound(const State &s, uint32_t g, size_t k, bool most, int *bound);

        const Problem &p_;
        size_t K_, nv_, ng_;
        uint64_t budget_;
        bool any_world_;
        bool aborted_ = false;
        bool any_leaf_ = false;
        int64_t target_ = -1;  // the answer (group × class) being searched for, -1 none
        bool target_max_ = false;
        std::vector<Term> terms_;
        std::vector<Con> cons_;
        std::vector<std::vector<uint32_t>> cell_cons_;
        std::vector<std::vector<uint32_t>> var_rows_;
        std::vector<uint8_t> in_group_;
        std::vector<int> gsize_;
        std::vector<int> best_lo_, best_hi_, proven_lo_, proven_hi_;
        std::vector<uint32_t> queue_;
        std::vector<uint8_t> queued_;
        std::vector<uint32_t> touched_;
        // leaf scratch
        std::vector<int> glo_, ghi_;
};

}  // namespace

static void class_vec(const BoundStore &store, const CardCounts &c, std::vector<int> *out);
static System build_system(const Network &net, const BoundStore &store, const std::vector<NodeId> &origin,
    const std::vector<std::vector<NodeId>> &groups);
static void components(const System &sys, std::vector<std::vector<uint32_t>> *comps, std::vector<int> *comp_of,
    std::vector<std::vector<uint32_t>> *comp_rows);
static Problem component_problem(
    const System &sys, const std::vector<uint32_t> &vars, const std::vector<uint32_t> &rows, std::vector<int> &local);
static std::string problem_key(const Problem &p);
static uint32_t find_root(std::vector<uint32_t> &parent, uint32_t x);
static bool min_cost_flow(
    int nodes, const std::vector<int> &balance, const std::vector<FlowArc> &arcs, int64_t *cost, uint64_t *steps);

static void class_vec(const BoundStore &store, const CardCounts &c, std::vector<int> *out) {
    out->assign(store.universe.classes(), 0);
    for (const auto &e : c) (*out)[store.universe.index_of(e.first)] += e.second;
}

// Step 1: the variables (nodes whose bounds leave more than one content), their seed intervals
// (the store's bounds of the node each came from, `origin`, when it is that node unchanged, met
// with pins and roots), every junction as a Row, every group as its variables plus its determined
// members' cards. A LOWER root is its lower bounds plus a free variable of the remaining size: in
// its consumer's row, or (a live root) in a row of its own.
static System build_system(const Network &net, const BoundStore &store, const std::vector<NodeId> &origin,
    const std::vector<std::vector<NodeId>> &groups) {
    System sys;
    const size_t K = store.universe.classes();
    sys.K = K;
    const size_t n_nodes = net.node_count();
    std::vector<int> var(n_nodes, -1);             // node → variable
    std::vector<int> free_part(n_nodes, -1);       // LOWER root node → its free variable
    std::vector<std::vector<int>> value(n_nodes);  // a determined node's cards per class
    std::vector<std::vector<int>> lower(n_nodes);
    std::vector<int> lo(K), hi(K), counts;
    const bool seeded = store.classes() == K;

    const auto add_var = [&sys, K](uint16_t size, const std::vector<int> &l, const std::vector<int> &h) {
        sys.size.push_back(size);
        for (size_t k = 0; k < K; ++k) {
            sys.lo.push_back(static_cast<int16_t>(l[k]));
            sys.hi.push_back(static_cast<int16_t>(h[k]));
        }
        return static_cast<int>(sys.size.size() - 1);
    };

    for (size_t i = 0; i < n_nodes; ++i) {
        const NodeId n = static_cast<NodeId>(i);
        const Node &node = net.node(n);
        const NodeId o = origin[i];
        const bool known = seeded && o != NO_NODE && store.knows(o);
        for (size_t k = 0; k < K; ++k) {
            lo[k] = known ? store.lo_of(o, k) : 0;
            hi[k] = known ? store.hi_of(o, k) : node.size;
        }
        if (node.pin >= 0) {
            const size_t pk = store.universe.index_of(node.pin);
            for (size_t k = 0; k < K; ++k) {
                lo[k] = std::max(lo[k], k == pk ? 1 : 0);
                hi[k] = std::min(hi[k], k == pk ? 1 : 0);
            }
        }
        bool is_lower = false;
        if (node.root >= 0) {
            const Root &r = net.roots()[static_cast<size_t>(node.root)];
            class_vec(store, r.counts, &counts);
            if (r.kind == RootKind::EXACT)
                for (size_t k = 0; k < K; ++k) lo[k] = std::max(lo[k], counts[k]), hi[k] = std::min(hi[k], counts[k]);
            if (r.kind == RootKind::LOWER) {
                for (size_t k = 0; k < K; ++k) lo[k] = std::max(lo[k], counts[k]);
                is_lower = true;
                lower[i] = counts;
            }
        }
        int sum_lo = 0, sum_hi = 0;
        bool determined = true;
        for (size_t k = 0; k < K; ++k) {
            if (lo[k] > hi[k]) sys.contradiction = true;
            sum_lo += lo[k];
            sum_hi += hi[k];
            determined = determined && lo[k] == hi[k];
        }
        if (sum_lo > node.size || sum_hi < node.size) sys.contradiction = true;
        if (determined) {
            value[i] = lo;
            continue;
        }
        if (is_lower) {
            int l = 0;
            std::vector<int> flo(K), fhi(K);
            for (size_t k = 0; k < K; ++k) {
                l += lower[i][k];
                flo[k] = std::max(0, lo[k] - lower[i][k]);
                fhi[k] = hi[k] - lower[i][k];
            }
            free_part[i] = add_var(static_cast<uint16_t>(node.size - l), flo, fhi);
            if (node.out == NO_JUNCTION) {
                // A live LOWER root: it is its lower bounds plus the free part.
                var[i] = add_var(node.size, lo, hi);
                Row r;
                r.vars = {static_cast<uint32_t>(free_part[i]), static_cast<uint32_t>(var[i])};
                r.side = {1, -1};
                r.cin = lower[i];
                r.cout.assign(K, 0);
                sys.rows.push_back(r);
            }
            continue;
        }
        var[i] = add_var(node.size, lo, hi);
    }

    std::vector<int> fixed_in, fixed_out;
    for (size_t j = 0; j < net.junction_count(); ++j) {
        const Junction &jn = net.junction(static_cast<uint32_t>(j));
        Row r;
        class_vec(store, jn.in_fixed, &fixed_in);
        class_vec(store, jn.out_fixed, &fixed_out);
        r.cin = fixed_in;
        r.cout = fixed_out;
        for (int s = 0; s < 2; ++s) {
            const std::vector<NodeId> &members = s == 0 ? jn.in : jn.out;
            std::vector<int> &c = s == 0 ? r.cin : r.cout;
            for (NodeId n : members) {
                if (free_part[n] >= 0) {
                    r.vars.push_back(static_cast<uint32_t>(free_part[n]));
                    r.side.push_back(static_cast<int8_t>(s == 0 ? 1 : -1));
                    for (size_t k = 0; k < K; ++k) c[k] += lower[n][k];
                } else if (var[n] >= 0) {
                    r.vars.push_back(static_cast<uint32_t>(var[n]));
                    r.side.push_back(static_cast<int8_t>(s == 0 ? 1 : -1));
                } else {
                    for (size_t k = 0; k < K; ++k) c[k] += value[n][k];
                }
            }
        }
        if (r.vars.empty()) {
            if (r.cin != r.cout) sys.contradiction = true;
            continue;
        }
        sys.rows.push_back(r);
    }

    for (const std::vector<NodeId> &g : groups) {
        std::vector<uint32_t> vars;
        std::vector<int> c(K, 0);
        int size = 0;
        for (NodeId n : g) {
            size += net.node(n).size;
            if (var[n] >= 0)
                vars.push_back(static_cast<uint32_t>(var[n]));
            else
                for (size_t k = 0; k < K; ++k) c[k] += value[n][k];
        }
        sys.gvars.push_back(vars);
        sys.gconst.push_back(c);
        sys.gsize.push_back(size);
    }
    return sys;
}

static uint32_t find_root(std::vector<uint32_t> &parent, uint32_t x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

// The least Σ cost·flow over integer flows on `arcs` (each within its [lo, hi]) with inflow −
// outflow = balance[j] at every node j; false when no flow exists. Lower bounds are shifted out,
// a feasible flow is found by augmenting paths from a super source to a super sink, then
// negative-cost cycles of the residual graph (Bellman-Ford) are cancelled until none is left, which
// is optimal. `steps` counts augmenting paths and Bellman-Ford passes.
static bool min_cost_flow(
    int nodes, const std::vector<int> &balance, const std::vector<FlowArc> &arcs, int64_t *cost, uint64_t *steps) {
    struct Edge {
            int to, rev, cap, cost;
    };
    const int src = nodes, dst = nodes + 1, n = nodes + 2;
    std::vector<std::vector<Edge>> g(static_cast<size_t>(n));
    const auto add = [&g](int u, int v, int cap, int c) {
        g[static_cast<size_t>(u)].push_back({v, static_cast<int>(g[static_cast<size_t>(v)].size()), cap, c});
        g[static_cast<size_t>(v)].push_back({u, static_cast<int>(g[static_cast<size_t>(u)].size()) - 1, 0, -c});
    };
    std::vector<int> b(balance);
    int64_t base = 0;
    std::vector<std::pair<int, size_t>> forward;  // per arc: its edge's node and index
    for (const FlowArc &a : arcs) {
        if (a.hi < a.lo) return false;
        base += static_cast<int64_t>(a.cost) * a.lo;
        b[static_cast<size_t>(a.to)] -= a.lo;
        b[static_cast<size_t>(a.from)] += a.lo;
        forward.emplace_back(a.from, g[static_cast<size_t>(a.from)].size());
        add(a.from, a.to, a.hi - a.lo, a.cost);
    }
    int need = 0;
    for (int j = 0; j < nodes; ++j) {
        if (b[static_cast<size_t>(j)] > 0) {
            add(j, dst, b[static_cast<size_t>(j)], 0);
            need += b[static_cast<size_t>(j)];
        } else if (b[static_cast<size_t>(j)] < 0) {
            add(src, j, -b[static_cast<size_t>(j)], 0);
        }
    }
    // A feasible flow: augmenting paths (breadth first) from src to dst.
    int sent = 0;
    std::vector<std::pair<int, int>> pred(static_cast<size_t>(n));
    while (sent < need) {
        ++*steps;
        std::fill(pred.begin(), pred.end(), std::make_pair(-1, -1));
        std::vector<int> queue = {src};
        pred[static_cast<size_t>(src)] = {src, -1};
        for (size_t h = 0; h < queue.size() && pred[static_cast<size_t>(dst)].first < 0; ++h) {
            const int u = queue[h];
            for (size_t e = 0; e < g[static_cast<size_t>(u)].size(); ++e) {
                const Edge &ed = g[static_cast<size_t>(u)][e];
                if (ed.cap <= 0 || pred[static_cast<size_t>(ed.to)].first >= 0) continue;
                pred[static_cast<size_t>(ed.to)] = {u, static_cast<int>(e)};
                queue.push_back(ed.to);
            }
        }
        if (pred[static_cast<size_t>(dst)].first < 0) return false;
        int push = need - sent;
        for (int v = dst; v != src; v = pred[static_cast<size_t>(v)].first)
            push = std::min(push, g[static_cast<size_t>(pred[static_cast<size_t>(v)].first)]
                                   [static_cast<size_t>(pred[static_cast<size_t>(v)].second)]
                                       .cap);
        for (int v = dst; v != src; v = pred[static_cast<size_t>(v)].first) {
            Edge &ed = g[static_cast<size_t>(pred[static_cast<size_t>(v)].first)]
                        [static_cast<size_t>(pred[static_cast<size_t>(v)].second)];
            ed.cap -= push;
            g[static_cast<size_t>(ed.to)][static_cast<size_t>(ed.rev)].cap += push;
        }
        sent += push;
    }
    // The super source and sink's flows stay as they are.
    for (int u : {src, dst}) {
        for (Edge &ed : g[static_cast<size_t>(u)]) {
            ed.cap = 0;
            g[static_cast<size_t>(ed.to)][static_cast<size_t>(ed.rev)].cap = 0;
        }
    }
    // Cancel negative cycles.
    std::vector<int64_t> dist(static_cast<size_t>(n));
    for (;;) {
        std::fill(dist.begin(), dist.end(), 0);
        std::fill(pred.begin(), pred.end(), std::make_pair(-1, -1));
        int last = -1;
        for (int pass = 0; pass < n; ++pass) {
            ++*steps;
            last = -1;
            for (int u = 0; u < n; ++u)
                for (size_t e = 0; e < g[static_cast<size_t>(u)].size(); ++e) {
                    const Edge &ed = g[static_cast<size_t>(u)][e];
                    if (ed.cap > 0 && dist[static_cast<size_t>(u)] + ed.cost < dist[static_cast<size_t>(ed.to)]) {
                        dist[static_cast<size_t>(ed.to)] = dist[static_cast<size_t>(u)] + ed.cost;
                        pred[static_cast<size_t>(ed.to)] = {u, static_cast<int>(e)};
                        last = ed.to;
                    }
                }
            if (last < 0) break;
        }
        if (last < 0) break;
        // `last` was relaxed in the n-th pass: walking back n predecessors lands on the cycle.
        int v = last;
        for (int i = 0; i < n; ++i) v = pred[static_cast<size_t>(v)].first;
        int push = 1 << 30;
        int u = v;
        do {
            const auto &pr = pred[static_cast<size_t>(u)];
            push = std::min(push, g[static_cast<size_t>(pr.first)][static_cast<size_t>(pr.second)].cap);
            u = pr.first;
        } while (u != v);
        u = v;
        do {
            const auto &pr = pred[static_cast<size_t>(u)];
            Edge &ed = g[static_cast<size_t>(pr.first)][static_cast<size_t>(pr.second)];
            ed.cap -= push;
            g[static_cast<size_t>(ed.to)][static_cast<size_t>(ed.rev)].cap += push;
            u = pr.first;
        } while (u != v);
    }
    // The cost: the shifted lower bounds' plus each arc's flow (its reverse edge's capacity).
    int64_t total = base;
    for (size_t i = 0; i < arcs.size(); ++i) {
        const Edge &ed = g[static_cast<size_t>(forward[i].first)][forward[i].second];
        total += static_cast<int64_t>(arcs[i].cost) * g[static_cast<size_t>(ed.to)][static_cast<size_t>(ed.rev)].cap;
    }
    *cost = total;
    return true;
}

// The memo key: everything the search of a Problem reads.
static std::string problem_key(const Problem &p) {
    std::string out;
    put_u32(out, p.K);
    put_u32(out, p.size.size());
    for (size_t v = 0; v < p.size.size(); ++v) put_u16(out, p.size[v]);
    for (size_t i = 0; i < p.lo.size(); ++i) {
        put_u16(out, static_cast<uint16_t>(p.lo[i]));
        put_u16(out, static_cast<uint16_t>(p.hi[i]));
    }
    put_u32(out, p.rows.size());
    for (const Row &r : p.rows) {
        put_u32(out, r.vars.size());
        for (size_t i = 0; i < r.vars.size(); ++i) {
            put_u32(out, r.vars[i]);
            put_u8(out, static_cast<uint8_t>(r.side[i]));
        }
        for (size_t k = 0; k < p.K; ++k) {
            put_i32(out, r.cin[k]);
            put_i32(out, r.cout[k]);
        }
    }
    put_u32(out, p.groups.size());
    for (const std::vector<uint32_t> &g : p.groups) put_u32s(out, g);
    return out;
}

// ── Search ────────────────────────────────────────────────────────────────────────────────────

Search::Search(const Problem &p, uint64_t budget, bool any_world)
    : p_(p), K_(p.K), nv_(p.size.size()), ng_(p.groups.size()), budget_(budget), any_world_(any_world) {
    cell_cons_.resize((nv_ + ng_) * K_);
    var_rows_.resize(nv_);
    in_group_.assign(nv_, 0);
    std::vector<Term> t;
    for (uint32_t v = 0; v < nv_; ++v) {
        t.clear();
        for (size_t k = 0; k < K_; ++k) t.push_back({vcell(v, k), 1});
        add_con(t, p.size[v]);
    }
    for (size_t r = 0; r < p.rows.size(); ++r) {
        const Row &row = p.rows[r];
        for (uint32_t v : row.vars) var_rows_[v].push_back(static_cast<uint32_t>(r));
        for (size_t k = 0; k < K_; ++k) {
            t.clear();
            for (size_t i = 0; i < row.vars.size(); ++i) t.push_back({vcell(row.vars[i], k), row.side[i]});
            add_con(t, row.cout[k] - row.cin[k]);
        }
    }
    for (uint32_t g = 0; g < ng_; ++g) {
        int size = 0;
        for (uint32_t v : p.groups[g]) {
            size += p.size[v];
            in_group_[v] = 1;
        }
        gsize_.push_back(size);
        for (size_t k = 0; k < K_; ++k) {
            t.clear();
            for (uint32_t v : p.groups[g]) t.push_back({vcell(v, k), 1});
            t.push_back({gcell(g, k), -1});
            add_con(t, 0);
        }
        t.clear();
        for (size_t k = 0; k < K_; ++k) t.push_back({gcell(g, k), 1});
        add_con(t, size);
    }
    queued_.assign(cons_.size(), 0);
    const size_t n = ng_ * K_;
    best_lo_.assign(n, BIG);
    best_hi_.assign(n, -BIG);
    proven_lo_.assign(n, 0);
    proven_hi_.assign(n, 0);
    for (uint32_t g = 0; g < ng_; ++g)
        for (size_t k = 0; k < K_; ++k) {
            int l = 0, h = 0;
            for (uint32_t v : p.groups[g]) l += p.lo[vcell(v, k)], h += p.hi[vcell(v, k)];
            proven_lo_[g * K_ + k] = l;
            proven_hi_[g * K_ + k] = std::min(h, gsize_[g]);
        }
}

void Search::add_con(const std::vector<Term> &terms, int rhs) {
    Con c;
    c.begin = static_cast<uint32_t>(terms_.size());
    terms_.insert(terms_.end(), terms.begin(), terms.end());
    c.end = static_cast<uint32_t>(terms_.size());
    c.rhs = rhs;
    for (const Term &t : terms) cell_cons_[t.cell].push_back(static_cast<uint32_t>(cons_.size()));
    cons_.push_back(c);
}

bool Search::tick() {
    if (++steps > budget_) aborted_ = true;
    return !aborted_;
}

// Narrows every term of `c` (Σ sign·cell = rhs) to what the others' intervals leave. False when the
// equation cannot hold. Cells that changed go to touched_.
bool Search::eval(State &s, const Con &c) {
    int mn = 0, mx = 0;
    for (uint32_t i = c.begin; i < c.end; ++i) {
        const Term &t = terms_[i];
        if (t.sign > 0)
            mn += s.lo[t.cell], mx += s.hi[t.cell];
        else
            mn -= s.hi[t.cell], mx -= s.lo[t.cell];
    }
    if (mn > c.rhs || mx < c.rhs) return false;
    for (uint32_t i = c.begin; i < c.end; ++i) {
        const Term &t = terms_[i];
        const int lo = s.lo[t.cell], hi = s.hi[t.cell];
        // This term's contribution lies in [rhs - (mx - its max), rhs - (mn - its min)].
        int nlo, nhi;
        if (t.sign > 0) {
            nlo = std::max(lo, c.rhs - (mx - hi));
            nhi = std::min(hi, c.rhs - (mn - lo));
        } else {
            nlo = std::max(lo, -(c.rhs - (mn + hi)));
            nhi = std::min(hi, -(c.rhs - (mx + lo)));
        }
        if (nlo > nhi) return false;
        if (nlo != lo || nhi != hi) {
            s.lo[t.cell] = static_cast<int16_t>(nlo);
            s.hi[t.cell] = static_cast<int16_t>(nhi);
            touched_.push_back(t.cell);
        }
    }
    return true;
}

// Bound propagation to a fixpoint from the constraints of cell `changed` (every constraint when
// negative). False when the state holds no world, or the budget ran out (aborted_).
bool Search::propagate(State &s, int64_t changed) {
    queue_.clear();
    if (changed < 0) {
        for (uint32_t c = 0; c < cons_.size(); ++c) {
            queue_.push_back(c);
            queued_[c] = 1;
        }
    } else {
        for (uint32_t c : cell_cons_[static_cast<size_t>(changed)]) {
            queue_.push_back(c);
            queued_[c] = 1;
        }
    }
    bool ok = true;
    size_t head = 0;
    while (head < queue_.size()) {
        const uint32_t c = queue_[head++];
        queued_[c] = 0;
        if (!tick()) {
            ok = false;
            break;
        }
        touched_.clear();
        if (!eval(s, cons_[c])) {
            ok = false;
            break;
        }
        for (uint32_t cell : touched_)
            for (uint32_t d : cell_cons_[cell])
                if (!queued_[d]) {
                    queued_[d] = 1;
                    queue_.push_back(d);
                }
    }
    for (size_t i = head; i < queue_.size(); ++i) queued_[queue_[i]] = 0;
    return ok;
}

// A variable only one row and no group reads: in a row, only its sum with others like it matters.
bool Search::slack_ok(uint32_t v) const {
    return !in_group_[v] && var_rows_[v].size() == 1;
}

bool Search::unfixed(const State &s, uint32_t v) const {
    for (size_t k = 0; k < K_; ++k)
        if (s.lo[vcell(v, k)] != s.hi[vcell(v, k)]) return true;
    return false;
}

// Whether `s` is a leaf (solver.h, step 3); for a leaf with worlds, the groups' intervals over it
// update the answers.
LeafKind Search::leaf(const State &s) {
    std::vector<uint8_t> free_var(nv_, 0);
    for (uint32_t v = 0; v < nv_; ++v) {
        free_var[v] = unfixed(s, v) ? 1 : 0;
        if (free_var[v] && !s.pristine[v]) return LeafKind::NOT_LEAF;
    }
    std::vector<uint8_t> active(p_.rows.size(), 0);
    for (size_t r = 0; r < p_.rows.size(); ++r)
        for (uint32_t v : p_.rows[r].vars)
            if (free_var[v]) active[r] = 1;
    for (uint32_t v = 0; v < nv_; ++v) {
        if (!free_var[v]) continue;
        int n = 0;
        for (uint32_t r : var_rows_[v]) n += active[r];
        if (n >= 2) return LeafKind::NOT_LEAF;
    }
    // Per active row: the free side (+1 / -1); the free variables on the other side, if any, are its
    // slack (each in no group and no other row, so only their sum matters).
    std::vector<int8_t> free_side(p_.rows.size(), 0);
    for (size_t r = 0; r < p_.rows.size(); ++r) {
        if (!active[r]) continue;
        const Row &row = p_.rows[r];
        int n_in = 0, n_out = 0;
        bool slack_in = true, slack_out = true;
        for (size_t i = 0; i < row.vars.size(); ++i) {
            const uint32_t v = row.vars[i];
            if (!free_var[v]) continue;
            const bool ok = slack_ok(v);
            if (row.side[i] > 0)
                ++n_in, slack_in = slack_in && ok;
            else
                ++n_out, slack_out = slack_out && ok;
        }
        if (n_in == 0 || (n_out > 0 && slack_in))
            free_side[r] = -1;
        else if (n_out == 0 || slack_out)
            free_side[r] = 1;
        else
            return LeafKind::NOT_LEAF;
    }

    // The groups' intervals over this leaf.
    glo_.assign(ng_ * K_, 0);
    ghi_.assign(ng_ * K_, 0);
    std::vector<int> row_of(nv_, -1);
    for (size_t r = 0; r < p_.rows.size(); ++r)
        if (active[r])
            for (uint32_t v : p_.rows[r].vars)
                if (free_var[v]) row_of[v] = static_cast<int>(r);
    for (uint32_t g = 0; g < ng_; ++g)
        for (uint32_t v : p_.groups[g]) {
            if (free_var[v] && row_of[v] >= 0) continue;  // a transportation row's (below)
            for (size_t k = 0; k < K_; ++k) {
                glo_[g * K_ + k] += s.lo[vcell(v, k)];
                ghi_[g * K_ + k] += s.hi[vcell(v, k)];
            }
        }
    std::vector<int> T(K_);
    std::vector<int> in_s(ng_);
    for (size_t r = 0; r < p_.rows.size(); ++r) {
        if (!active[r]) continue;
        const Row &row = p_.rows[r];
        const int8_t fs = free_side[r];
        // T: what the free side holds besides the slack: the other side's constants minus its own.
        for (size_t k = 0; k < K_; ++k) T[k] = fs < 0 ? row.cin[k] - row.cout[k] : row.cout[k] - row.cin[k];
        int N = 0, slack_size = 0;
        std::fill(in_s.begin(), in_s.end(), 0);
        for (size_t i = 0; i < row.vars.size(); ++i) {
            const uint32_t v = row.vars[i];
            if (free_var[v]) {
                (row.side[i] == fs ? N : slack_size) += p_.size[v];
                continue;
            }
            // A fixed member: on the free side it takes from T, on the other side it adds.
            const int dir = row.side[i] == fs ? -1 : 1;
            for (size_t k = 0; k < K_; ++k) T[k] += dir * s.lo[vcell(v, k)];
        }
        int sum_d = 0, need = 0, sum_t = 0;
        for (size_t k = 0; k < K_; ++k) {
            sum_d += std::max(0, T[k]);
            need += std::max(0, -T[k]);
            sum_t += T[k];
        }
        if (need > slack_size || sum_t + slack_size != N || sum_d > N) return LeafKind::EMPTY;
        const int spare = N - sum_d;
        for (uint32_t g = 0; g < ng_; ++g) {
            int sz = 0;
            for (uint32_t v : p_.groups[g])
                if (row_of[v] == static_cast<int>(r)) sz += p_.size[v];
            in_s[g] = sz;
        }
        for (uint32_t g = 0; g < ng_; ++g) {
            if (in_s[g] == 0) continue;
            for (size_t k = 0; k < K_; ++k) {
                const int d = std::max(0, T[k]);
                glo_[g * K_ + k] += std::max(0, in_s[g] + d - N);
                ghi_[g * K_ + k] += std::min(in_s[g], d + spare);
            }
        }
    }
    any_leaf_ = true;
    for (size_t i = 0; i < ng_ * K_; ++i) {
        best_lo_[i] = std::min(best_lo_[i], glo_[i]);
        best_hi_[i] = std::max(best_hi_[i], ghi_[i]);
    }
    return LeafKind::WORLDS;
}

// The variable and cell to branch on in a state that is not a leaf: a variable already branched on,
// else one in two junctions with free variables, else, in a junction with free variables on both
// sides and no slack, the smallest variable of the side with fewer. Its cell of the card an answer is
// being searched for when open, else its narrowest open cell.
bool Search::pick(const State &s, uint32_t *var, uint32_t *cell) const {
    int64_t chosen = -1;
    std::vector<uint8_t> free_var(nv_, 0);
    for (uint32_t v = 0; v < nv_; ++v) free_var[v] = unfixed(s, v) ? 1 : 0;
    for (uint32_t v = 0; v < nv_ && chosen < 0; ++v)
        if (free_var[v] && !s.pristine[v]) chosen = v;
    if (chosen < 0) {
        std::vector<uint8_t> active(p_.rows.size(), 0);
        for (size_t r = 0; r < p_.rows.size(); ++r)
            for (uint32_t v : p_.rows[r].vars)
                if (free_var[v]) active[r] = 1;
        for (uint32_t v = 0; v < nv_ && chosen < 0; ++v) {
            if (!free_var[v]) continue;
            int n = 0;
            for (uint32_t r : var_rows_[v]) n += active[r];
            if (n >= 2) chosen = v;
        }
        for (size_t r = 0; r < p_.rows.size() && chosen < 0; ++r) {
            if (!active[r]) continue;
            const Row &row = p_.rows[r];
            int n_in = 0, n_out = 0;
            bool slack_in = true, slack_out = true;
            for (size_t i = 0; i < row.vars.size(); ++i) {
                const uint32_t v = row.vars[i];
                if (!free_var[v]) continue;
                (row.side[i] > 0 ? n_in : n_out)++;
                (row.side[i] > 0 ? slack_in : slack_out) = (row.side[i] > 0 ? slack_in : slack_out) && slack_ok(v);
            }
            if (n_in == 0 || n_out == 0 || slack_in || slack_out) continue;
            const int8_t side = n_in <= n_out ? 1 : -1;
            for (size_t i = 0; i < row.vars.size(); ++i) {
                const uint32_t v = row.vars[i];
                if (!free_var[v] || row.side[i] != side) continue;
                if (chosen < 0 || p_.size[v] < p_.size[static_cast<size_t>(chosen)]) chosen = v;
            }
        }
    }
    if (chosen < 0) return false;
    const uint32_t v = static_cast<uint32_t>(chosen);
    *var = v;
    // The card an answer is being searched for first: its interval is what decides the search.
    if (target_ >= 0) {
        const size_t tk = static_cast<size_t>(target_) % K_;
        if (s.hi[vcell(v, tk)] > s.lo[vcell(v, tk)]) {
            *cell = vcell(v, tk);
            return true;
        }
    }
    int best_width = BIG;
    for (size_t k = 0; k < K_; ++k) {
        const int w = s.hi[vcell(v, k)] - s.lo[vcell(v, k)];
        if (w > 0 && w < best_width) {
            best_width = w;
            *cell = vcell(v, k);
        }
    }
    return best_width < BIG;
}

// Depth-first search below `s` (after propagating cell `changed`, every constraint when negative)
// for a leaf with worlds that reaches the answer being searched for (target_; any leaf when it is
// negative). A leaf ends its branch either way: its intervals are exact for its region.
bool Search::dfs(State &s, int64_t changed) {
    if (aborted_) return false;
    if (!propagate(s, changed) || !tick()) return false;
    if (target_ >= 0) {
        // The searched-for card's flow bound below this state: no world here reaches the answer
        // when it falls short.
        const size_t i = static_cast<size_t>(target_);
        int fb = 0;
        if (flow_bound(s, static_cast<uint32_t>(i / K_), i % K_, target_max_, &fb) &&
            (target_max_ ? fb < proven_hi_[i] : fb > proven_lo_[i]))
            return false;
        if (aborted_) return false;
    }
    const LeafKind lk = leaf(s);
    if (lk == LeafKind::WORLDS) return target_ < 0 || reached();
    if (lk == LeafKind::EMPTY) return false;
    uint32_t v = 0, cell = 0;
    if (!pick(s, &v, &cell)) return false;
    // The two halves partition the state; the upper one first when looking for the most copies of
    // this cell's card.
    const int mid = (s.lo[cell] + s.hi[cell]) / 2;
    const bool upper_first = target_max_ && target_ >= 0 && cell % K_ == static_cast<size_t>(target_) % K_;
    State other = s;
    (upper_first ? s.lo[cell] : s.hi[cell]) = static_cast<int16_t>(upper_first ? mid + 1 : mid);
    (upper_first ? other.hi[cell] : other.lo[cell]) = static_cast<int16_t>(upper_first ? mid : mid + 1);
    s.pristine[v] = 0;
    other.pristine[v] = 0;
    if (dfs(s, cell)) return true;
    if (aborted_) return false;
    return dfs(other, cell);
}

// Whether the answer being searched for (target_, target_max_) is reached by some leaf.
bool Search::reached() const {
    const size_t i = static_cast<size_t>(target_);
    return target_max_ ? best_hi_[i] >= proven_hi_[i] : best_lo_[i] <= proven_lo_[i];
}

// A bound on group `g`'s copies of card `k` (the most when `most`, else the least) from the card's
// flow alone: per card, the junctions conserve copies and every variable leaves at most one
// junction and enters at most one, so the cells of `s` form a flow network (a junction a node, the
// outside world one more) whose optimum, the node sizes coupling the cards dropped, bounds every
// world's. False when the budget ran out or no flow exists.
bool Search::flow_bound(const State &s, uint32_t g, size_t k, bool most, int *bound) {
    const int outside = static_cast<int>(p_.rows.size());
    std::vector<int> balance(p_.rows.size() + 1, 0);
    for (size_t r = 0; r < p_.rows.size(); ++r) {
        balance[r] = p_.rows[r].cout[k] - p_.rows[r].cin[k];
        balance[static_cast<size_t>(outside)] -= balance[r];
    }
    std::vector<int> from(nv_, outside), to(nv_, outside);
    for (size_t r = 0; r < p_.rows.size(); ++r)
        for (size_t i = 0; i < p_.rows[r].vars.size(); ++i)
            (p_.rows[r].side[i] > 0 ? to : from)[p_.rows[r].vars[i]] = static_cast<int>(r);
    std::vector<uint8_t> member(nv_, 0);
    for (uint32_t v : p_.groups[g]) member[v] = 1;
    std::vector<FlowArc> arcs;
    int loose = 0;  // variables in no junction: their own interval
    for (uint32_t v = 0; v < nv_; ++v) {
        const int lo = s.lo[vcell(v, k)], hi = s.hi[vcell(v, k)];
        if (from[v] == outside && to[v] == outside) {
            if (member[v]) loose += most ? hi : lo;
            continue;
        }
        FlowArc a;
        a.from = from[v];
        a.to = to[v];
        a.lo = lo;
        a.hi = hi;
        a.cost = member[v] ? (most ? -1 : 1) : 0;
        arcs.push_back(a);
    }
    int64_t cost = 0;
    uint64_t used = 0;
    const bool ok = min_cost_flow(outside + 1, balance, arcs, &cost, &used);
    steps += used;
    if (steps > budget_) aborted_ = true;
    if (!ok || aborted_) return false;
    *bound = loose + static_cast<int>(most ? -cost : cost);
    return true;
}

void Search::run() {
    State root;
    root.lo.assign((nv_ + ng_) * K_, 0);
    root.hi.assign((nv_ + ng_) * K_, 0);
    root.pristine.assign(nv_, 1);
    for (uint32_t v = 0; v < nv_; ++v)
        for (size_t k = 0; k < K_; ++k) {
            root.lo[vcell(v, k)] = p_.lo[vcell(v, k)];
            root.hi[vcell(v, k)] = p_.hi[vcell(v, k)];
        }
    for (uint32_t g = 0; g < ng_; ++g)
        for (size_t k = 0; k < K_; ++k) root.hi[gcell(g, k)] = static_cast<int16_t>(gsize_[g]);
    bool ok = propagate(root, -1);
    const bool has_answers = ng_ > 0 && K_ > 1;
    if (ok && (has_answers || any_world_)) {
        for (uint32_t g = 0; g < ng_; ++g)
            for (size_t k = 0; k < K_; ++k) {
                proven_lo_[g * K_ + k] = root.lo[gcell(g, k)];
                proven_hi_[g * K_ + k] = root.hi[gcell(g, k)];
            }
        // Any leaf first: the component has worlds.
        State s = root;
        target_ = -1;
        ok = dfs(s, -2) || aborted_;
        // Then each answer not yet settled: first its card's flow bound tightens the proven bound;
        // then a leaf whose interval reaches the proven bound settles it, and when no leaf below the
        // state constrained to the bound does, the bound is one tighter.
        for (uint32_t g = 0; g < ng_ && ok && !aborted_; ++g)
            for (size_t k = 0; k + 1 < K_ && !aborted_; ++k)
                for (int end = 0; end < 2 && !aborted_; ++end) {
                    const size_t i = g * K_ + k;
                    const uint32_t cell = gcell(g, k);
                    target_ = static_cast<int64_t>(i);
                    target_max_ = end == 1;
                    int fb = 0;
                    if (!reached() && flow_bound(root, g, k, target_max_, &fb)) {
                        if (target_max_)
                            proven_hi_[i] = std::min(proven_hi_[i], fb);
                        else
                            proven_lo_[i] = std::max(proven_lo_[i], fb);
                    }
                    while (!reached() && !aborted_) {
                        State t = root;
                        if (target_max_)
                            t.lo[cell] = static_cast<int16_t>(proven_hi_[i]);
                        else
                            t.hi[cell] = static_cast<int16_t>(proven_lo_[i]);
                        if (dfs(t, cell) || aborted_) break;
                        if (target_max_)
                            --proven_hi_[i];
                        else
                            ++proven_lo_[i];
                    }
                }
    }
    finished = !aborted_;
    infeasible = !ok && !aborted_;
    const size_t n = ng_ * K_;
    res_lo.assign(n, 0);
    res_hi.assign(n, 0);
    settled.assign(n, 0);
    for (uint32_t g = 0; g < ng_; ++g)
        for (size_t k = 0; k < K_; ++k) {
            const size_t i = g * K_ + k;
            const bool other = k + 1 == K_;
            const bool lo_ok = !other && any_leaf_ && best_lo_[i] <= proven_lo_[i];
            const bool hi_ok = !other && any_leaf_ && best_hi_[i] >= proven_hi_[i];
            res_lo[i] = lo_ok ? best_lo_[i] : proven_lo_[i];
            res_hi[i] = hi_ok ? best_hi_[i] : proven_hi_[i];
            settled[i] = (lo_ok && hi_ok) || other || !has_answers ? 1 : 0;
        }
}

// ── Solver ────────────────────────────────────────────────────────────────────────────────────

// Step 2: the components of `sys` (variables joined by a row), each variable's component, and each
// component's rows.
static void components(const System &sys, std::vector<std::vector<uint32_t>> *comps, std::vector<int> *comp_of,
    std::vector<std::vector<uint32_t>> *comp_rows) {
    const size_t nv = sys.size.size();
    std::vector<uint32_t> parent(nv);
    std::iota(parent.begin(), parent.end(), 0u);
    for (const Row &r : sys.rows)
        for (size_t i = 1; i < r.vars.size(); ++i) {
            const uint32_t a = find_root(parent, r.vars[0]), b = find_root(parent, r.vars[i]);
            if (a != b) parent[std::max(a, b)] = std::min(a, b);
        }
    comps->clear();
    comp_of->assign(nv, -1);
    for (uint32_t v = 0; v < nv; ++v) {
        const uint32_t r = find_root(parent, v);
        if ((*comp_of)[r] < 0) {
            (*comp_of)[r] = static_cast<int>(comps->size());
            comps->emplace_back();
        }
        (*comp_of)[v] = (*comp_of)[r];
        (*comps)[static_cast<size_t>((*comp_of)[v])].push_back(v);
    }
    comp_rows->assign(comps->size(), {});
    for (uint32_t r = 0; r < sys.rows.size(); ++r)
        (*comp_rows)[static_cast<size_t>((*comp_of)[sys.rows[r].vars[0]])].push_back(r);
}

// One component of `sys` as a Problem in local indices (`local` receives each of its variables'
// local index), without groups.
static Problem component_problem(
    const System &sys, const std::vector<uint32_t> &vars, const std::vector<uint32_t> &rows, std::vector<int> &local) {
    const size_t K = sys.K;
    Problem p;
    p.K = K;
    for (size_t i = 0; i < vars.size(); ++i) {
        const uint32_t v = vars[i];
        local[v] = static_cast<int>(i);
        p.size.push_back(sys.size[v]);
        for (size_t k = 0; k < K; ++k) {
            p.lo.push_back(sys.lo[v * K + k]);
            p.hi.push_back(sys.hi[v * K + k]);
        }
    }
    for (uint32_t r : rows) {
        Row row = sys.rows[r];
        for (uint32_t &v : row.vars) v = static_cast<uint32_t>(local[v]);
        p.rows.push_back(row);
    }
    return p;
}

// Memo entries kept before the memo is cleared (an unchanged component reuses its entry).
static constexpr size_t MEMO_LIMIT = 512;

Solution Solver::solve(const Network &net, const BoundStore &store, const std::vector<std::vector<NodeId>> &groups,
    uint64_t budget) const {
    Solution out;
    // Only the groups' projections are asked for: compact a copy keeping just their nodes (exact
    // for those projections, compact.h). A node it keeps unchanged keeps its store bounds.
    std::vector<uint8_t> keep(net.node_count(), 0);
    for (const std::vector<NodeId> &g : groups)
        for (NodeId n : g) keep[n] = 1;
    std::vector<NodeId> remap;
    const Network work = compact_network(net, keep, &remap);
    std::vector<NodeId> origin(work.node_count(), NO_NODE);
    for (size_t i = 0; i < remap.size(); ++i)
        if (remap[i] != NO_NODE && work.node(remap[i]).size == net.node(static_cast<NodeId>(i)).size)
            origin[remap[i]] = static_cast<NodeId>(i);
    std::vector<std::vector<NodeId>> wgroups;
    for (const std::vector<NodeId> &g : groups) {
        wgroups.emplace_back();
        for (NodeId n : g) wgroups.back().push_back(remap[n]);
    }
    const System sys = build_system(work, store, origin, wgroups);
    const size_t K = sys.K;
    const size_t nv = sys.size.size();
    // Every group starts at its determined members' cards; components add their intervals.
    std::vector<std::vector<int>> glo(groups.size()), ghi(groups.size());
    for (size_t g = 0; g < groups.size(); ++g) {
        glo[g] = sys.gconst[g];
        ghi[g] = sys.gconst[g];
    }
    out.contradiction = sys.contradiction;
    bool exact = !sys.contradiction;

    // Step 2: components.
    std::vector<std::vector<uint32_t>> comps, comp_rows;
    std::vector<int> comp_of;
    components(sys, &comps, &comp_of, &comp_rows);

    uint64_t used = 0;
    std::vector<int> local(nv, -1);
    for (size_t c = 0; c < comps.size(); ++c) {
        // Its Problem, in local indices, and the groups that meet it.
        Problem p = component_problem(sys, comps[c], comp_rows[c], local);
        std::vector<size_t> group_ids;
        for (size_t g = 0; g < groups.size(); ++g) {
            std::vector<uint32_t> members;
            for (uint32_t v : sys.gvars[g])
                if (comp_of[v] == static_cast<int>(c)) members.push_back(static_cast<uint32_t>(local[v]));
            if (members.empty()) continue;
            p.groups.push_back(members);
            group_ids.push_back(g);
        }
        if (p.groups.empty()) continue;  // no group reads it

        const uint64_t left = budget > used ? budget - used : 0;
        const std::string key = problem_key(p);
        auto it = memo_.find(key);
        const bool reuse =
            it != memo_.end() && (it->second.finished ? it->second.steps <= left : it->second.budget == left);
        if (!reuse) {
            Search search(p, left);
            search.run();
            Memo m;
            m.finished = search.finished;
            m.infeasible = search.infeasible;
            m.steps = search.steps;
            m.budget = left;
            m.lo.assign(search.res_lo.begin(), search.res_lo.end());
            m.hi.assign(search.res_hi.begin(), search.res_hi.end());
            m.settled = search.settled;
            if (memo_.size() >= MEMO_LIMIT) memo_.clear();
            it = memo_.insert_or_assign(key, m).first;
        }
        const Memo &m = it->second;
        used += std::min<uint64_t>(m.steps, left);
        if (m.infeasible) out.contradiction = true;
        for (size_t lg = 0; lg < group_ids.size(); ++lg) {
            const size_t g = group_ids[lg];
            for (size_t k = 0; k < K; ++k) {
                glo[g][k] += m.lo[lg * K + k];
                ghi[g][k] += m.hi[lg * K + k];
                if (k + 1 < K && !m.settled[lg * K + k]) exact = false;
            }
        }
    }

    out.steps = used;
    out.exact = exact && !out.contradiction;
    for (size_t g = 0; g < groups.size(); ++g) {
        GroupBounds gb;
        for (size_t k = 0; k < K; ++k) {
            const int hi = std::min(ghi[g][k], sys.gsize[g]);
            gb.lo.push_back(static_cast<uint8_t>(std::max(0, std::min(glo[g][k], 255))));
            gb.hi.push_back(static_cast<uint8_t>(std::max(0, std::min(hi, 255))));
        }
        out.groups.push_back(gb);
    }
    return out;
}

Feasibility Solver::feasible(const Network &net, const BoundStore &store, uint64_t budget) const {
    Feasibility out;
    if (!store.contradiction().empty()) {
        out.feasible = false;
        out.exact = true;
        return out;
    }
    // No group: compaction keeps no node (exact for whether a world exists). A node it keeps
    // unchanged keeps its store bounds.
    std::vector<NodeId> remap;
    const Network work = compact_network(net, std::vector<uint8_t>(net.node_count(), 0), &remap);
    std::vector<NodeId> origin(work.node_count(), NO_NODE);
    for (size_t i = 0; i < remap.size(); ++i)
        if (remap[i] != NO_NODE && work.node(remap[i]).size == net.node(static_cast<NodeId>(i)).size)
            origin[remap[i]] = static_cast<NodeId>(i);
    const System sys = build_system(work, store, origin, {});
    if (sys.contradiction) {
        out.feasible = false;
        out.exact = true;
        return out;
    }
    std::vector<std::vector<uint32_t>> comps, comp_rows;
    std::vector<int> comp_of;
    components(sys, &comps, &comp_of, &comp_rows);
    std::vector<int> local(sys.size.size(), -1);
    bool finished = true;
    for (size_t c = 0; c < comps.size(); ++c) {
        const Problem p = component_problem(sys, comps[c], comp_rows[c], local);
        const uint64_t left = budget > out.steps ? budget - out.steps : 0;
        const std::string key = problem_key(p);
        auto it = feasible_memo_.find(key);
        const bool reuse =
            it != feasible_memo_.end() && (it->second.finished ? it->second.steps <= left : it->second.budget == left);
        if (!reuse) {
            Search search(p, left, true);
            search.run();
            Memo m;
            m.finished = search.finished;
            m.infeasible = search.infeasible;
            m.steps = search.steps;
            m.budget = left;
            if (feasible_memo_.size() >= MEMO_LIMIT) feasible_memo_.clear();
            it = feasible_memo_.insert_or_assign(key, m).first;
        }
        const Memo &m = it->second;
        out.steps += std::min<uint64_t>(m.steps, left);
        if (m.infeasible) {
            out.feasible = false;
            out.exact = true;
            return out;
        }
        finished = finished && m.finished;
    }
    out.exact = finished;
    return out;
}

}  // namespace knowledge
