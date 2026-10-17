// Archetype inference: reference lists as alternative roots (templates.h).

#include "templates.h"

#include <algorithm>
#include <map>

namespace knowledge {

static int total_of(const CardCounts &c);
static CardCounts sum_counts(const CardCounts &a, const CardCounts &b);
static int overlap(const CardCounts &a, const CardCounts &b);

static int total_of(const CardCounts &c) {
    int n = 0;
    for (const auto &e : c) n += e.second;
    return n;
}

static CardCounts sum_counts(const CardCounts &a, const CardCounts &b) {
    std::map<int, int> m;
    for (const auto &e : a) m[e.first] += e.second;
    for (const auto &e : b) m[e.first] += e.second;
    return CardCounts(m.begin(), m.end());
}

// Σ over cards of the lesser of the two counts (both ascending by card).
static int overlap(const CardCounts &a, const CardCounts &b) {
    int n = 0;
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i].first < b[j].first) {
            ++i;
        } else if (b[j].first < a[i].first) {
            ++j;
        } else {
            n += std::min(a[i].second, b[j].second);
            ++i;
            ++j;
        }
    }
    return n;
}

void registered_counts(const Decklists &lists, int player, CardCounts *main, CardCounts *side) {
    std::map<int, int> m, s;
    for (const DeckEntry &e : lists.cards) {
        if (e.player != player || e.card == NO_CARD) continue;
        if (e.main > 0) m[e.card] += e.main;
        if (e.side > 0) s[e.card] += e.side;
    }
    main->assign(m.begin(), m.end());
    side->assign(s.begin(), s.end());
}

bool same_list(const MetaList &l, const CardCounts &main, const CardCounts &side) {
    return l.main == main && l.side == side;
}

std::vector<int> matching_lists(const Metagame &m, const CardCounts &main, const CardCounts &side) {
    std::vector<int> out;
    for (size_t i = 0; i < m.lists.size(); ++i)
        if (same_list(m.lists[i], main, side)) out.push_back(static_cast<int>(i));
    return out;
}

int classify_list(const Metagame &m, const std::string &name, const CardCounts &main, const CardCounts &side) {
    for (size_t i = 0; i < m.lists.size(); ++i)
        if (!name.empty() && m.lists[i].name == name) return static_cast<int>(i);
    int best = -1, best_overlap = 0;
    for (size_t i = 0; i < m.lists.size(); ++i) {
        const int o = overlap(m.lists[i].main, main) + overlap(m.lists[i].side, side);
        if (o > best_overlap) {
            best = static_cast<int>(i);
            best_overlap = o;
        }
    }
    return best;
}

bool hypothesis_network(const Network &net, const CardCounts &main, const CardCounts &side, Network *out) {
    Network h;
    for (size_t i = 0; i < net.node_count(); ++i) {
        const Node &old = net.node(static_cast<NodeId>(i));
        Node &n = h.node(h.add_node(old.size, old.loc));
        n.pin = old.pin;
        n.drawn_turn = old.drawn_turn;
        n.handle = old.handle;
        n.draw_seq = old.draw_seq;
    }
    std::map<int, int> net_fixed;
    for (size_t j = 0; j < net.junction_count(); ++j) {
        const Junction &jn = net.junction(static_cast<uint32_t>(j));
        h.add_junction(jn);
        for (const auto &c : jn.in_fixed) net_fixed[c.first] += c.second;
        for (const auto &c : jn.out_fixed) net_fixed[c.first] -= c.second;
    }
    const CardCounts all = sum_counts(main, side);
    std::vector<NodeId> set_aside, fresh;
    int set_aside_units = 0, fresh_units = 0;
    for (const Root &r : net.roots()) {
        const uint16_t size = net.node(r.node).size;
        if (r.part == RootPart::NONE) {
            fresh.push_back(r.node);
            fresh_units += size;
            continue;
        }
        const CardCounts &t = r.part == RootPart::MAIN ? main : r.part == RootPart::SIDE ? side : all;
        const int units = total_of(t);
        if (units < size || units > 0xFFFF) return false;
        Junction j;
        j.in = {h.add_root(static_cast<uint16_t>(units), Loc::GONE, RootKind::EXACT, t)};
        j.out = {r.node};
        if (units > size) {
            const NodeId a = h.add_node(static_cast<uint16_t>(units - size), Loc::GONE);
            j.out.push_back(a);
            set_aside.push_back(a);
            set_aside_units += units - size;
        }
        h.add_junction(j);
    }
    Junction from_none;
    int fixed_out = 0, fixed_in = 0;
    for (const auto &c : net_fixed) {
        if (c.second > 0) {
            from_none.out_fixed.emplace_back(c.first, c.second);
            fixed_out += c.second;
        } else if (c.second < 0) {
            from_none.in_fixed.emplace_back(c.first, -c.second);
            fixed_in += -c.second;
        }
    }
    if (!fresh.empty() || fixed_out > 0 || fixed_in > 0) {
        const int rest = set_aside_units + fixed_in - fresh_units - fixed_out;
        if (rest < 0) return false;
        from_none.in = set_aside;
        from_none.out = fresh;
        if (rest > 0) from_none.out.push_back(h.add_node(static_cast<uint16_t>(rest), Loc::GONE));
        h.add_junction(from_none);
    }
    *out = h;
    return true;
}

Universe network_universe(const Network &net) {
    Universe u;
    for (size_t i = 0; i < net.node_count(); ++i) {
        const int pin = net.node(static_cast<NodeId>(i)).pin;
        if (pin >= 0) u.add(pin);
    }
    for (size_t j = 0; j < net.junction_count(); ++j) {
        const Junction &jn = net.junction(static_cast<uint32_t>(j));
        for (const auto &c : jn.in_fixed) u.add(c.first);
        for (const auto &c : jn.out_fixed) u.add(c.first);
    }
    return u;
}

}  // namespace knowledge
