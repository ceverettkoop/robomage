// The k_* tables and the text summary, built from the model (docs/knowledge_store.md §6).

#include <algorithm>

#include "knowledge.h"

namespace knowledge {

static LibPos top_range(size_t pos, size_t span);
static std::string card_word(const std::map<int, std::string> &names, int card);

static LibPos top_range(size_t pos, size_t span) {
    LibPos p;
    p.anchor = static_cast<uint8_t>(rmstate::LibAnchor::TOP);
    p.pos = static_cast<uint16_t>(pos);
    p.span = static_cast<uint16_t>(span);
    return p;
}

static std::string card_word(const std::map<int, std::string> &names, int card) {
    const auto it = names.find(card);
    return it != names.end() ? it->second : "#" + std::to_string(card);
}

// The pools the tables may report for player `p`, in table order: the opponent's hand; the library
// runs top to bottom, then the blocks of runs from the top (runs 0..j, 0 < j < last) and from the
// bottom (runs i..last, 0 < i < last); the opponent's sideboard; face-down cards by handle; the
// set-aside units. The viewer's own hand and sideboard are in plain view (snapshot objects).
std::vector<Knowledge::PoolSpec> Knowledge::pool_specs(int p) const {
    const PlayerModel &m = players_[p];
    const bool opponent = p != viewer_;
    std::vector<PoolSpec> out;
    if (opponent && !m.hand.empty()) {
        PoolSpec s;
        s.kind = rmstate::PoolKind::HAND;
        s.nodes = m.hand;
        out.push_back(s);
    }
    std::vector<size_t> start;
    size_t at = 0;
    for (size_t i = 0; i < m.library.size(); ++i) {
        PoolSpec s;
        s.kind = rmstate::PoolKind::LIBRARY_RANGE;
        s.nodes = {m.library[i]};
        s.range = top_range(at, m.net.node(m.library[i]).size);
        s.first = s.last = static_cast<int>(i);
        out.push_back(s);
        start.push_back(at);
        at += m.net.node(m.library[i]).size;
    }
    const int last = static_cast<int>(m.library.size()) - 1;
    const auto block = [&](int a, int b) {
        PoolSpec s;
        s.kind = rmstate::PoolKind::LIBRARY_RANGE;
        s.nodes.assign(m.library.begin() + a, m.library.begin() + b + 1);
        const size_t end = b == last ? at : start[static_cast<size_t>(b) + 1];
        s.range = top_range(start[static_cast<size_t>(a)], end - start[static_cast<size_t>(a)]);
        s.first = a;
        s.last = b;
        out.push_back(s);
    };
    for (int j = 1; j < last; ++j) block(0, j);
    for (int i = last - 1; i > 0; --i) block(i, last);
    if (opponent && !m.sideboard.empty()) {
        PoolSpec s;
        s.kind = rmstate::PoolKind::SIDEBOARD;
        s.nodes = m.sideboard;
        out.push_back(s);
    }
    std::vector<NodeId> face_down = m.face_down;
    std::sort(face_down.begin(), face_down.end(),
        [&m](NodeId a, NodeId b) { return m.net.node(a).handle < m.net.node(b).handle; });
    for (NodeId n : face_down) {
        PoolSpec s;
        s.kind = rmstate::PoolKind::FACE_DOWN;
        s.nodes = {n};
        s.obj = m.net.node(n).handle;
        out.push_back(s);
    }
    if (!m.set_aside.empty()) {
        PoolSpec s;
        s.kind = rmstate::PoolKind::SET_ASIDE;
        s.nodes = m.set_aside;
        out.push_back(s);
    }
    return out;
}

// The solver's bounds of `specs[spec]` (per class of the universe); the propagated bounds' sums
// when the solver had no answer for it (a contradiction, or a state the solve did not see).
GroupBounds Knowledge::pool_bounds(int p, size_t spec, const std::vector<PoolSpec> &specs) const {
    const PlayerModel &m = players_[p];
    const Solution &sol = solved_[p];
    const size_t K = m.full.universe.classes();
    if (!sol.contradiction && sol.groups.size() == specs.size() && sol.groups[spec].lo.size() == K)
        return sol.groups[spec];
    GroupBounds b;
    int size = 0;
    for (NodeId n : specs[spec].nodes) size += m.net.node(n).size;
    for (size_t k = 0; k < K; ++k) {
        int lo = 0, hi = 0;
        for (NodeId n : specs[spec].nodes) {
            lo += m.full.lo(m.net, n, k);
            hi += m.full.hi(m.net, n, k);
        }
        b.lo.push_back(static_cast<uint8_t>(std::min(lo, 255)));
        b.hi.push_back(static_cast<uint8_t>(std::min(std::min(hi, size), 255)));
    }
    return b;
}

// One k_pool row for `spec` and its k_pool_card rows: per card of the universe with a maximum
// above 0, the least and most copies. Under a closed decklist a card outside the universe (never
// seen) can be in the pool too.
void Knowledge::pool_rows(int p, KRows &out, const PoolSpec &spec, const GroupBounds &b) const {
    const PlayerModel &m = players_[p];
    KPoolRow row;
    row.owner = p;
    row.kind = spec.kind;
    row.range = spec.range;
    row.obj = spec.obj;
    int size = 0;
    for (NodeId n : spec.nodes) size += m.net.node(n).size;
    row.size = static_cast<uint16_t>(size);
    const uint16_t id = static_cast<uint16_t>(out.pools.size());
    out.pools.push_back(row);
    for (size_t k = 0; k < m.full.universe.cards.size(); ++k) {
        if (b.hi[k] == 0) continue;
        KPoolCardRow c;
        c.pool = id;
        c.card = m.full.universe.cards[k];
        c.min = b.lo[k];
        c.max = b.hi[k];
        out.pool_cards.push_back(c);
    }
}

// Player `p`'s k_pool, k_pool_card and k_position rows. A library run whose card is determined
// (one card's minimum is its size) is a k_position row per position and no pool; any other run is
// a pool, and a one-position run also a k_position row naming it. A block of runs is a pool only
// when it says more than its parts: some card's least or most copies in it is sharper than the
// block one run shorter (or that run alone) together with the run it adds.
void Knowledge::player_rows(int p, KRows &out) const {
    const PlayerModel &m = players_[p];
    const std::vector<PoolSpec> specs = pool_specs(p);
    std::vector<GroupBounds> b;
    for (size_t i = 0; i < specs.size(); ++i) b.push_back(pool_bounds(p, i, specs));
    const size_t n_cards = m.full.universe.cards.size();
    std::map<std::pair<int, int>, size_t> spec_of;  // (first, last) → spec index
    for (size_t i = 0; i < specs.size(); ++i)
        if (specs[i].first >= 0) spec_of[std::make_pair(specs[i].first, specs[i].last)] = i;
    const int last = static_cast<int>(m.library.size()) - 1;
    for (size_t i = 0; i < specs.size(); ++i) {
        const PoolSpec &spec = specs[i];
        int size = 0;
        for (NodeId n : spec.nodes) size += m.net.node(n).size;
        if (spec.first >= 0 && spec.first == spec.last) {
            int card = NO_CARD;
            for (size_t k = 0; k < n_cards; ++k)
                if (b[i].lo[k] == size) card = m.full.universe.cards[k];
            if (card != NO_CARD) {
                for (int pos = 0; pos < size; ++pos) {
                    KPositionRow r;
                    r.owner = p;
                    r.at = top_range(spec.range.pos + static_cast<size_t>(pos), 1);
                    r.card = card;
                    out.positions.push_back(r);
                }
                continue;
            }
            if (size == 1) {
                KPositionRow r;
                r.owner = p;
                r.at = spec.range;
                r.pool = static_cast<uint16_t>(out.pools.size());
                out.positions.push_back(r);
            }
        } else if (spec.first >= 0) {
            // A block: compare with the block one run shorter plus the run it adds.
            const bool from_top = spec.first == 0;
            const std::pair<int, int> shorter =
                from_top ? std::make_pair(0, spec.last - 1) : std::make_pair(spec.first + 1, last);
            const int added = from_top ? spec.last : spec.first;
            const GroupBounds &a = b[spec_of.at(shorter)];
            const GroupBounds &r = b[spec_of.at(std::make_pair(added, added))];
            bool sharper = false;
            for (size_t k = 0; k < n_cards && !sharper; ++k)
                sharper = b[i].lo[k] > a.lo[k] + r.lo[k] || b[i].hi[k] < std::min(a.hi[k] + r.hi[k], size);
            if (!sharper) continue;
        }
        pool_rows(p, out, spec, b[i]);
    }
}

KRows Knowledge::rows() const {
    KRows out;
    out.meta.level = level_;
    out.meta.exact = level_ != rmstate::KnowledgeLevel::OFF && viewer_ != NO_SEAT &&
                     (!game_active_ || (solved_[0].exact && solved_[1].exact));
    out.meta.steps = static_cast<uint32_t>(std::min<uint64_t>(steps(), 0xFFFFFFFFu));
    if (level_ == rmstate::KnowledgeLevel::OFF || viewer_ == NO_SEAT) return out;
    const int order[2] = {viewer_, 1 - viewer_};

    for (int p : order) {
        std::map<int, std::pair<uint8_t, uint8_t>> seen;
        for (const auto &g : seen_game_[p]) seen[g.first].first = g.second;
        for (const auto &mt : seen_match_[p]) seen[mt.first].second = mt.second;
        for (const auto &s : seen) {
            if (s.second.first == 0 && s.second.second == 0) continue;
            KSeenRow r;
            r.player = p;
            r.card = s.first;
            r.game = s.second.first;
            r.match = s.second.second;
            out.seen.push_back(r);
        }
    }
    if (!game_active_) return out;

    for (int p : order) player_rows(p, out);

    // Definitely in the opponent's hand: the hand pool's least copies.
    const int opp = 1 - viewer_;
    const PlayerModel &om = players_[opp];
    const std::vector<PoolSpec> specs = pool_specs(opp);
    if (!specs.empty() && specs[0].kind == rmstate::PoolKind::HAND) {
        const GroupBounds hand = pool_bounds(opp, 0, specs);
        for (size_t k = 0; k < om.full.universe.cards.size(); ++k) {
            if (hand.lo[k] == 0) continue;
            KKnownHandRow r;
            r.card = om.full.universe.cards[k];
            r.copies = hand.lo[k];
            out.known_hand.push_back(r);
        }
    }
    return out;
}

void Knowledge::write_tables(rmstate::FrameWriter &w) const {
    write_k_tables(w, rows(), codes_, viewer_);
}

bool Knowledge::table_blocks(std::string *blocks, uint32_t *n) const {
    blocks->clear();
    *n = 0;
    if (viewer_ == NO_SEAT) return false;
    rmstate::FrameWriter w(
        rmstate::FrameKind::STATE, viewer_ == 0 ? rmstate::Viewer::A : rmstate::Viewer::B, schema_.hash);
    write_tables(w);
    std::string head;
    if (!rmstate::split_frame_blocks(w.finish(), static_cast<uint16_t>(rmstate::TableId::K_META), &head, blocks, n)
            .ok())
        return false;
    // The k_* tables have no text, so the frame they join keeps its own string pool: the rest is
    // that pool alone, holding only the empty string.
    return head.size() == rmstate::FRAME_HEADER_SIZE + rmstate::TABLE_HEADER_SIZE + rmstate::ROW_ALIGN &&
           head[rmstate::FRAME_HEADER_SIZE + 8] == 1;
}

// One line per player: hand, library runs, sideboard, face-down cards, then what was seen.
std::string Knowledge::text() const {
    if (viewer_ == NO_SEAT) return "(no viewer yet)\n";
    std::string out;
    const KRows r = rows();
    const int order[2] = {viewer_, 1 - viewer_};
    for (int p : order) {
        const PlayerModel &m = players_[p];
        std::string line = std::string("  ") + (p == 0 ? "A" : "B") + (p == viewer_ ? " (self)" : " (opp)") + ":";
        if (game_active_) {
            size_t hand = 0;
            for (NodeId n : m.hand) hand += m.net.node(n).size;
            line += " hand " + std::to_string(hand);
            std::string known;
            for (const KKnownHandRow &k : r.known_hand)
                if (p != viewer_)
                    known += (known.empty() ? "" : ", ") + std::to_string(k.copies) + "x " + card_word(names_, k.card);
            if (p == viewer_)
                for (NodeId n : m.hand)
                    if (m.net.node(n).pin >= 0)
                        known += (known.empty() ? "" : ", ") + card_word(names_, m.net.node(n).pin);
            if (!known.empty()) line += " {" + known + "}";
            line += "; library " + std::to_string(m.library_size()) + ":";
            for (size_t i = 0; i < r.pools.size(); ++i) {
                const KPoolRow &pool = r.pools[i];
                if (pool.owner != p) continue;
                if (pool.kind == rmstate::PoolKind::HAND) continue;
                std::string cards;
                size_t listed = 0, n_cards = 0;
                for (const KPoolCardRow &c : r.pool_cards) {
                    if (c.pool != i) continue;
                    ++n_cards;
                    const bool narrow = c.min > 0 || pool.size <= 4;
                    if (!narrow || listed >= 6) continue;
                    ++listed;
                    cards += (cards.empty() ? "" : ", ") + card_word(names_, c.card) + " " + std::to_string(c.min) +
                             "-" + std::to_string(c.max);
                }
                std::string where;
                if (pool.kind == rmstate::PoolKind::LIBRARY_RANGE)
                    where = " T" + std::to_string(pool.range.pos) +
                            (pool.size > 1 ? "-" + std::to_string(pool.range.pos + pool.size - 1) : "");
                else if (pool.kind == rmstate::PoolKind::SIDEBOARD)
                    where = " sideboard";
                else if (pool.kind == rmstate::PoolKind::FACE_DOWN)
                    where = " face-down h" + std::to_string(pool.obj);
                else
                    where = " set-aside";
                line += where + "(" + std::to_string(pool.size) + ": " + std::to_string(n_cards) + " cand" +
                        (cards.empty() ? "" : "; " + cards) + ")";
            }
            for (const KPositionRow &pos : r.positions)
                if (pos.owner == p && pos.card != NO_CARD)
                    line += " T" + std::to_string(pos.at.pos) + "=" + card_word(names_, pos.card);
        }
        std::string seen;
        for (const KSeenRow &s : r.seen)
            if (s.player == p)
                seen += (seen.empty() ? "" : ", ") + card_word(names_, s.card) + " " + std::to_string(s.game) + "/" +
                        std::to_string(s.match);
        if (!seen.empty()) line += "; seen game/match: " + seen;
        out += line + "\n";
    }
    return out;
}

}  // namespace knowledge
