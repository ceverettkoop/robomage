#include "network.h"

#include "serial.h"

namespace knowledge {

NodeId Network::add_node(uint16_t size, Loc loc) {
    Node n;
    n.size = size;
    n.loc = loc;
    nodes_.push_back(n);
    return static_cast<NodeId>(nodes_.size() - 1);
}

NodeId Network::add_root(uint16_t size, Loc loc, RootKind kind, const std::vector<std::pair<int, int>> &counts) {
    const NodeId n = add_node(size, loc);
    Root r;
    r.node = n;
    r.kind = kind;
    r.counts = counts;
    nodes_[n].root = static_cast<int32_t>(roots_.size());
    roots_.push_back(r);
    return n;
}

std::vector<NodeId> Network::split(NodeId parent, const std::vector<uint16_t> &sizes) {
    if (sizes.size() <= 1) return {parent};
    const uint32_t j = static_cast<uint32_t>(junctions_.size());
    junctions_.emplace_back();
    const Loc loc = nodes_[parent].loc;
    const int16_t drawn = nodes_[parent].drawn_turn;
    std::vector<NodeId> parts;
    for (uint16_t s : sizes) {
        const NodeId c = add_node(s, loc);
        nodes_[c].in = j;
        nodes_[c].drawn_turn = drawn;
        parts.push_back(c);
    }
    nodes_[parent].out = j;
    nodes_[parent].loc = Loc::GONE;
    nodes_[parent].handle = 0;
    junctions_[j].in = {parent};
    junctions_[j].out = parts;
    return parts;
}

NodeId Network::merge(const std::vector<NodeId> &parts, Loc loc) {
    if (parts.size() == 1) {
        nodes_[parts[0]].loc = loc;
        return parts[0];
    }
    const uint32_t j = static_cast<uint32_t>(junctions_.size());
    junctions_.emplace_back();
    uint32_t size = 0;
    int16_t drawn = parts.empty() ? -1 : nodes_[parts[0]].drawn_turn;
    for (NodeId p : parts) {
        size += nodes_[p].size;
        if (nodes_[p].drawn_turn != drawn) drawn = -1;
        nodes_[p].out = j;
        nodes_[p].loc = Loc::GONE;
        nodes_[p].handle = 0;
    }
    const NodeId m = add_node(static_cast<uint16_t>(size), loc);
    nodes_[m].in = j;
    nodes_[m].drawn_turn = drawn;
    junctions_[j].in = parts;
    junctions_[j].out = {m};
    return m;
}

bool Network::pin(NodeId n, int card) {
    Node &node = nodes_[n];
    if (node.pin == card) return true;
    if (node.pin >= 0 || node.size != 1) return false;
    node.pin = card;
    pin_log_.push_back(n);
    return true;
}

void Network::serialize(std::string &out) const {
    put_u32(out, nodes_.size());
    for (const Node &n : nodes_) {
        put_u16(out, n.size);
        put_i32(out, n.pin);
        put_u32(out, n.in);
        put_u32(out, n.out);
        put_i32(out, n.root);
        put_u8(out, static_cast<uint8_t>(n.loc));
        put_i32(out, n.drawn_turn);
        put_u32(out, n.handle);
        put_u32(out, n.draw_seq);
    }
    put_u32(out, junctions_.size());
    for (const Junction &j : junctions_) {
        put_u32s(out, j.in);
        put_u32s(out, j.out);
    }
    put_u32(out, roots_.size());
    for (const Root &r : roots_) {
        put_u32(out, r.node);
        put_u8(out, static_cast<uint8_t>(r.kind));
        put_u32(out, r.counts.size());
        for (const auto &c : r.counts) {
            put_i32(out, c.first);
            put_i32(out, c.second);
        }
    }
    put_u32s(out, pin_log_);
}

bool Network::deserialize(const std::string &in, size_t *at) {
    ByteReader r(in, *at);
    *this = Network{};
    const uint64_t n_nodes = r.u32();
    if (!r.count(n_nodes, in.size())) return false;
    nodes_.resize(n_nodes);
    for (Node &n : nodes_) {
        n.size = static_cast<uint16_t>(r.u16());
        n.pin = r.i32();
        n.in = static_cast<uint32_t>(r.u32());
        n.out = static_cast<uint32_t>(r.u32());
        n.root = r.i32();
        n.loc = static_cast<Loc>(r.u8());
        n.drawn_turn = static_cast<int16_t>(r.i32());
        n.handle = static_cast<uint32_t>(r.u32());
        n.draw_seq = static_cast<uint32_t>(r.u32());
    }
    const uint64_t n_junctions = r.u32();
    if (!r.count(n_junctions, in.size())) return false;
    junctions_.resize(n_junctions);
    for (Junction &j : junctions_) {
        j.in = r.u32s();
        j.out = r.u32s();
    }
    const uint64_t n_roots = r.u32();
    if (!r.count(n_roots, in.size())) return false;
    roots_.resize(n_roots);
    for (Root &root : roots_) {
        root.node = static_cast<NodeId>(r.u32());
        root.kind = static_cast<RootKind>(r.u8());
        const uint64_t n_counts = r.u32();
        if (!r.count(n_counts, in.size())) return false;
        for (uint64_t i = 0; i < n_counts; ++i) {
            const int card = r.i32();
            const int copies = r.i32();
            root.counts.emplace_back(card, copies);
        }
    }
    pin_log_ = r.u32s();
    *at = r.at;
    return r.ok;
}

}  // namespace knowledge
