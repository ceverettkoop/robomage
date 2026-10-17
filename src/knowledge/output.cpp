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

// The card every unit of node `n` is (a pin, or bounds that leave one card), else NO_CARD.
int Knowledge::single_card(int p, NodeId n) const {
    const PlayerModel &m = players_[p];
    const Node &node = m.net.node(n);
    if (node.pin >= 0) return node.pin;
    for (size_t k = 0; k < m.full.universe.cards.size(); ++k)
        if (m.full.lo(m.net, n, k) == node.size) return m.full.universe.cards[k];
    return NO_CARD;
}

// One k_pool row for `nodes` taken together (their sizes summed), and its k_pool_card rows: per
// card of the universe, the sum of the nodes' minima and of their maxima (capped at the size).
// Under a closed decklist a card outside the universe (never seen) can be in the pool too.
void Knowledge::pool_rows(int p, KRows &out, rmstate::PoolKind kind, const std::vector<NodeId> &nodes,
    const LibPos &range, uint32_t obj) const {
    const PlayerModel &m = players_[p];
    KPoolRow row;
    row.owner = p;
    row.kind = kind;
    row.range = range;
    row.obj = obj;
    int size = 0;
    for (NodeId n : nodes) size += m.net.node(n).size;
    row.size = static_cast<uint16_t>(size);
    const uint16_t id = static_cast<uint16_t>(out.pools.size());
    out.pools.push_back(row);
    for (size_t k = 0; k < m.full.universe.cards.size(); ++k) {
        int lo = 0, hi = 0;
        for (NodeId n : nodes) {
            lo += m.full.lo(m.net, n, k);
            hi += m.full.hi(m.net, n, k);
        }
        hi = std::min(hi, size);
        if (hi <= 0) continue;
        KPoolCardRow c;
        c.pool = id;
        c.card = m.full.universe.cards[k];
        c.min = static_cast<uint8_t>(std::min(lo, 255));
        c.max = static_cast<uint8_t>(std::min(hi, 255));
        out.pool_cards.push_back(c);
    }
}

KRows Knowledge::rows() const {
    KRows out;
    out.meta.level = level_;
    out.meta.exact = false;
    out.meta.steps = static_cast<uint32_t>(std::min<uint64_t>(steps_, 0xFFFFFFFFu));
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

    for (int p : order) {
        const PlayerModel &m = players_[p];
        const bool opponent = p != viewer_;
        // The viewer's own hand and sideboard are in plain view (snapshot objects), not pools.
        if (opponent && !m.hand.empty()) pool_rows(p, out, rmstate::PoolKind::HAND, m.hand, LibPos{}, 0);
        size_t at = 0;
        for (NodeId run : m.library) {
            const size_t len = m.net.node(run).size;
            const int card = single_card(p, run);
            if (card != NO_CARD) {
                for (size_t i = 0; i < len; ++i) {
                    KPositionRow pos;
                    pos.owner = p;
                    pos.at = top_range(at + i, 1);
                    pos.card = card;
                    out.positions.push_back(pos);
                }
            } else {
                const uint16_t pool = static_cast<uint16_t>(out.pools.size());
                pool_rows(p, out, rmstate::PoolKind::LIBRARY_RANGE, {run}, top_range(at, len), 0);
                if (len == 1) {
                    KPositionRow pos;
                    pos.owner = p;
                    pos.at = top_range(at, 1);
                    pos.pool = pool;
                    out.positions.push_back(pos);
                }
            }
            at += len;
        }
        if (opponent && !m.sideboard.empty()) pool_rows(p, out, rmstate::PoolKind::SIDEBOARD, m.sideboard, LibPos{}, 0);
        std::vector<NodeId> face_down = m.face_down;
        std::sort(face_down.begin(), face_down.end(),
            [&m](NodeId a, NodeId b) { return m.net.node(a).handle < m.net.node(b).handle; });
        for (NodeId n : face_down) pool_rows(p, out, rmstate::PoolKind::FACE_DOWN, {n}, LibPos{}, m.net.node(n).handle);
        if (!m.set_aside.empty()) pool_rows(p, out, rmstate::PoolKind::SET_ASIDE, m.set_aside, LibPos{}, 0);
    }

    const PlayerModel &opp = players_[1 - viewer_];
    for (size_t k = 0; k < opp.full.universe.cards.size(); ++k) {
        int lo = 0;
        for (NodeId n : opp.hand) lo += opp.full.lo(opp.net, n, k);
        if (lo <= 0) continue;
        KKnownHandRow r;
        r.card = opp.full.universe.cards[k];
        r.copies = static_cast<uint8_t>(std::min(lo, 255));
        out.known_hand.push_back(r);
    }
    return out;
}

void Knowledge::write_tables(rmstate::FrameWriter &w) const {
    write_k_tables(w, rows(), codes_, viewer_);
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
