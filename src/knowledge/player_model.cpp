#include "player_model.h"

#include <algorithm>
#include <map>

#include "compact.h"
#include "serial.h"

namespace knowledge {

static void put_nodes(std::string &out, const std::vector<NodeId> &v);

static void put_nodes(std::string &out, const std::vector<NodeId> &v) {
    put_u32s(out, v);
}

bool loc_matches(Loc loc, Zn zone) {
    switch (loc) {
        case Loc::LIBRARY:
            return zone == Zn::LIBRARY;
        case Loc::HAND:
            return zone == Zn::HAND;
        case Loc::SIDEBOARD:
            return zone == Zn::SIDEBOARD;
        case Loc::SET_ASIDE:
            return zone == Zn::NONE;
        case Loc::FACE_DOWN:
        case Loc::PUBLIC:
            return is_public_zone(zone);
        case Loc::GONE:
            return false;
    }
    return false;
}

size_t PlayerModel::library_size() const {
    size_t n = 0;
    for (NodeId s : library) n += net.node(s).size;
    return n;
}

size_t PlayerModel::locate(size_t index, size_t *offset) const {
    size_t start = 0;
    for (size_t seg = 0; seg < library.size(); ++seg) {
        const size_t len = net.node(library[seg]).size;
        if (index < start + len) {
            *offset = index - start;
            return seg;
        }
        start += len;
    }
    *offset = 0;
    return library.size();
}

size_t PlayerModel::run_start(size_t seg) const {
    size_t start = 0;
    for (size_t s = 0; s < seg && s < library.size(); ++s) start += net.node(library[s]).size;
    return start;
}

// Makes a run start at top index `index` (cutting the run that spans it); returns that run's index
// in `library` (library.size() at the end).
size_t PlayerModel::cut_before(size_t index) {
    size_t off = 0;
    const size_t seg = locate(index, &off);
    if (seg == library.size() || off == 0) return seg;
    const NodeId run = library[seg];
    const uint16_t len = net.node(run).size;
    const std::vector<NodeId> parts = net.split(run, {static_cast<uint16_t>(off), static_cast<uint16_t>(len - off)});
    library[seg] = parts[0];
    library.insert(library.begin() + static_cast<std::ptrdiff_t>(seg) + 1, parts[1]);
    return seg + 1;
}

NodeId PlayerModel::take_library(size_t index) {
    size_t off = 0;
    const size_t seg = locate(index, &off);
    if (seg == library.size()) return NO_NODE;
    const NodeId run = library[seg];
    const uint16_t len = net.node(run).size;
    if (len == 1) {
        erase_library_run(seg);
        return run;
    }
    // The run's order is unknown, so the unit at the index is any of its units: one unit splits
    // off and the rest keeps the run's (now one shorter) span.
    const std::vector<NodeId> parts = net.split(run, {1, static_cast<uint16_t>(len - 1)});
    library[seg] = parts[1];
    return parts[0];
}

void PlayerModel::erase_library_run(size_t seg) {
    library.erase(library.begin() + static_cast<std::ptrdiff_t>(seg));
    rejoin(seg);
}

// Runs `seg - 1` and `seg`, adjacent now that what was between them left, merge back into one when
// nothing tells them apart: two unknown parts of one split (a run cut around a position the
// viewer saw, as a searched card is located in the shuffled library before it leaves) are the
// same as their union. Exact: their only constraint is the split they came from.
void PlayerModel::rejoin(size_t seg) {
    if (seg == 0 || seg >= library.size()) return;
    const Node &a = net.node(library[seg - 1]);
    const Node &b = net.node(library[seg]);
    if (a.in == NO_JUNCTION || a.in != b.in || net.junction(a.in).in.size() != 1) return;
    if (a.pin >= 0 || b.pin >= 0 || a.handle != 0 || b.handle != 0) return;
    const NodeId m = net.merge({library[seg - 1], library[seg]}, Loc::LIBRARY);
    library[seg - 1] = m;
    library.erase(library.begin() + static_cast<std::ptrdiff_t>(seg));
}

void PlayerModel::insert_library(size_t index, NodeId n) {
    const size_t seg = cut_before(index);
    library.insert(library.begin() + static_cast<std::ptrdiff_t>(seg), n);
    net.node(n).loc = Loc::LIBRARY;
    net.node(n).drawn_turn = -1;
}

// One three-way split (the units above, the unit, the units below), so the parts around the unit
// are siblings that rejoin once it leaves.
NodeId PlayerModel::isolate_library(size_t index) {
    size_t off = 0;
    const size_t seg = locate(index, &off);
    if (seg == library.size()) return NO_NODE;
    const NodeId run = library[seg];
    const uint16_t len = net.node(run).size;
    if (len == 1) return run;
    std::vector<uint16_t> sizes;
    if (off > 0) sizes.push_back(static_cast<uint16_t>(off));
    sizes.push_back(1);
    if (len - off - 1 > 0) sizes.push_back(static_cast<uint16_t>(len - off - 1));
    const std::vector<NodeId> parts = net.split(run, sizes);
    library.erase(library.begin() + static_cast<std::ptrdiff_t>(seg));
    library.insert(library.begin() + static_cast<std::ptrdiff_t>(seg), parts.begin(), parts.end());
    return parts[off > 0 ? 1 : 0];
}

NodeId PlayerModel::merge_library_range(size_t a, size_t b) {
    if (a >= b) return NO_NODE;
    const size_t sa = cut_before(a);
    const size_t sb = cut_before(b);
    const std::vector<NodeId> parts(
        library.begin() + static_cast<std::ptrdiff_t>(sa), library.begin() + static_cast<std::ptrdiff_t>(sb));
    if (parts.empty()) return NO_NODE;
    const NodeId m = net.merge(parts, Loc::LIBRARY);
    library.erase(library.begin() + static_cast<std::ptrdiff_t>(sa), library.begin() + static_cast<std::ptrdiff_t>(sb));
    library.insert(library.begin() + static_cast<std::ptrdiff_t>(sa), m);
    return m;
}

NodeId PlayerModel::merge_library() {
    if (library.empty()) return NO_NODE;
    for (NodeId s : library) net.node(s).handle = 0;
    const NodeId m = net.merge(library, Loc::LIBRARY);
    library.assign(1, m);
    return m;
}

NodeId PlayerModel::split_unit(std::vector<NodeId> &zone, size_t at) {
    const NodeId n = zone[at];
    const uint16_t size = net.node(n).size;
    if (size == 1) {
        zone.erase(zone.begin() + static_cast<std::ptrdiff_t>(at));
        return n;
    }
    const std::vector<NodeId> parts = net.split(n, {1, static_cast<uint16_t>(size - 1)});
    zone[at] = parts[1];
    return parts[0];
}

std::vector<NodeId> PlayerModel::live() const {
    std::vector<NodeId> out(library);
    for (const std::vector<NodeId> *z : {&hand, &sideboard, &set_aside, &face_down, &publics})
        out.insert(out.end(), z->begin(), z->end());
    return out;
}

std::vector<NodeId> *PlayerModel::zone_list(Loc loc) {
    switch (loc) {
        case Loc::HAND:
            return &hand;
        case Loc::SIDEBOARD:
            return &sideboard;
        case Loc::SET_ASIDE:
            return &set_aside;
        case Loc::FACE_DOWN:
            return &face_down;
        case Loc::PUBLIC:
            return &publics;
        case Loc::LIBRARY:
        case Loc::GONE:
            return nullptr;
    }
    return nullptr;
}

const std::vector<NodeId> *PlayerModel::zone_list(Loc loc) const {
    return const_cast<PlayerModel *>(this)->zone_list(loc);
}

// Two live nodes only their common producer constrains (each appears in it once, as a part of a
// sum): any split of their union between them is as possible as any other, so they can be one node.
bool PlayerModel::interchangeable(NodeId a, NodeId b, const std::vector<uint8_t> &protect) const {
    const Node &x = net.node(a);
    const Node &y = net.node(b);
    if (x.in == NO_JUNCTION || x.in != y.in) return false;
    if (x.pin >= 0 || y.pin >= 0 || x.handle != 0 || y.handle != 0 || x.root >= 0 || y.root >= 0) return false;
    return !protect[a] && !protect[b];
}

// Merges the interchangeable sub-pools of a set zone that share a drawn-this-turn turn; the merged
// node takes the place of the group's first.
void PlayerModel::merge_set_zone(std::vector<NodeId> &zone, const std::vector<uint8_t> &protect) {
    std::vector<NodeId> out;
    std::vector<bool> used(zone.size(), false);
    for (size_t i = 0; i < zone.size(); ++i) {
        if (used[i]) continue;
        std::vector<NodeId> group = {zone[i]};
        for (size_t k = i + 1; k < zone.size(); ++k)
            if (!used[k] && interchangeable(zone[i], zone[k], protect) &&
                net.node(zone[i]).drawn_turn == net.node(zone[k]).drawn_turn) {
                group.push_back(zone[k]);
                used[k] = true;
            }
        out.push_back(group.size() == 1 ? group[0] : net.merge(group, net.node(zone[i]).loc));
    }
    zone = out;
}

void PlayerModel::compact(const std::vector<NodeId> &protect, std::vector<NodeId> *remap) {
    std::vector<uint8_t> prot(net.node_count(), 0);
    for (NodeId n : protect)
        if (n < prot.size()) prot[n] = 1;
    for (size_t seg = 1; seg < library.size();) {
        if (!interchangeable(library[seg - 1], library[seg], prot)) {
            ++seg;
            continue;
        }
        const NodeId m = net.merge({library[seg - 1], library[seg]}, Loc::LIBRARY);
        prot.push_back(0);
        library[seg - 1] = m;
        library.erase(library.begin() + static_cast<std::ptrdiff_t>(seg));
    }
    for (std::vector<NodeId> *z : {&hand, &sideboard, &set_aside}) {
        merge_set_zone(*z, prot);
        prot.resize(net.node_count(), 0);
    }

    std::vector<uint8_t> keep(net.node_count(), 0);
    for (NodeId n : live()) keep[n] = 1;
    for (NodeId n : protect)
        if (n < keep.size()) keep[n] = 1;
    net = compact_network(net, keep, remap);
    for (std::vector<NodeId> *z : {&library, &hand, &sideboard, &set_aside, &face_down, &publics})
        for (NodeId &n : *z) n = (*remap)[n];
    full.reset();
    seen.reset();
    compacted_nodes = net.node_count();
}

void PlayerModel::serialize(std::string &out) const {
    net.serialize(out);
    full.serialize(out);
    seen.serialize(out);
    for (const std::vector<NodeId> *z : {&library, &hand, &sideboard, &set_aside, &face_down, &publics})
        put_nodes(out, *z);
    put_i32(out, dtt_unknown_turn);
    put_u64(out, compacted_nodes);
}

bool PlayerModel::deserialize(const std::string &in, size_t *at) {
    if (!net.deserialize(in, at) || !full.deserialize(in, at) || !seen.deserialize(in, at)) return false;
    ByteReader r(in, *at);
    for (std::vector<NodeId> *z : {&library, &hand, &sideboard, &set_aside, &face_down, &publics}) *z = r.u32s();
    dtt_unknown_turn = r.i32();
    compacted_nodes = static_cast<size_t>(r.u64());
    *at = r.at;
    if (!r.ok) return false;
    for (const NodeId n : live())
        if (n >= net.node_count()) return false;
    return true;
}

}  // namespace knowledge
