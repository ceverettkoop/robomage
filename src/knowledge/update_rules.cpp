// The update rule of every event kind (docs/knowledge_store.md §5), and the steps they share.

#include <algorithm>

#include "knowledge.h"

namespace knowledge {

static std::vector<std::pair<int, int>> list_counts(
    const std::vector<DeckEntry> &entries, int player, bool main, bool side);
static int count_total(const std::vector<std::pair<int, int>> &counts);
static uint16_t clamp_size(long v);

static std::vector<std::pair<int, int>> list_counts(
    const std::vector<DeckEntry> &entries, int player, bool main, bool side) {
    std::map<int, int> m;
    for (const DeckEntry &e : entries) {
        if (e.player != player || e.card == NO_CARD) continue;
        const int n = (main ? e.main : 0) + (side ? e.side : 0);
        if (n > 0) m[e.card] += n;
    }
    return std::vector<std::pair<int, int>>(m.begin(), m.end());
}

static int count_total(const std::vector<std::pair<int, int>> &counts) {
    int n = 0;
    for (const auto &c : counts) n += c.second;
    return n;
}

static uint16_t clamp_size(long v) {
    return static_cast<uint16_t>(std::max<long>(0, std::min<long>(v, 255)));
}

// ── game_start ────────────────────────────────────────────────────────────────────────────────

// New networks for both players (match memory stays). The library holds the stated library count
// of units from the root; the rest of the root not in the library or sideboard is set aside (a
// test-harness preset or a companion, which arrive as moves from zone none); the sideboard holds
// the stated sideboard count. Hands start empty.
void Knowledge::on_game_start(const Event &ev) {
    game_active_ = true;
    game_ = ev.game;
    turn_ = ev.turn;
    own_deck_.clear();
    if (frame_own_deck_) own_deck_ = *frame_own_deck_;
    handles_.clear();
    draws_.clear();
    for (int p = 0; p < 2; ++p) {
        players_[p] = PlayerModel{};
        solved_[p] = Solution{};
        seen_game_[p].clear();
        for (int s = 0; s < 2; ++s) reported_contradiction_[p][s] = false;
    }
    bool built[2] = {false, false};
    for (const EventPlayer &ep : ev.players) {
        if (ep.player == NO_SEAT || built[ep.player]) continue;
        built[ep.player] = true;
        build_roots(ep.player, ep.library, ep.sideboard);
    }
    if (!built[0] || !built[1]) error("game_start lacks a player's library and sideboard counts");
}

// Makes a root of `counts` (its size their total, or the parts' total for a FREE root) split into
// parts of `sizes` (zero-size parts are left out); returns the parts, NO_NODE for a left-out one.
std::vector<NodeId> Knowledge::add_root_parts(
    int p, RootKind kind, const std::vector<std::pair<int, int>> &counts, const std::vector<uint16_t> &sizes) {
    PlayerModel &m = players_[p];
    int total = 0;
    for (uint16_t s : sizes) total += s;
    std::vector<NodeId> out(sizes.size(), NO_NODE);
    if (total == 0) return out;
    const NodeId root = m.net.add_root(clamp_size(total), Loc::SET_ASIDE, kind, counts);
    std::vector<uint16_t> nonzero;
    for (uint16_t s : sizes)
        if (s > 0) nonzero.push_back(s);
    const std::vector<NodeId> parts = m.net.split(root, nonzero);
    size_t next = 0;
    for (size_t i = 0; i < sizes.size(); ++i)
        if (sizes[i] > 0) out[i] = parts[next++];
    return out;
}

void Knowledge::build_roots(int p, uint16_t library, uint16_t sideboard) {
    PlayerModel &m = players_[p];
    const bool own = p == viewer_;
    std::vector<DeckEntry> own_list;
    for (const DeckEntry &e : own_deck_)
        if (e.player == p) own_list.push_back(e);
    // The viewer's own current deck (own_deck) after sideboarding; game 1 is the registered list,
    // which (unlike own_deck) also holds the test-harness presets.
    const bool use_own = own && (game_ > 0 || !lists_.present[p]) && !own_list.empty();
    closed_[p] = !use_own && !lists_.present[p];
    std::vector<NodeId> parts;  // library, sideboard, set aside...
    if (closed_[p]) {
        // Unknown cards from the vocab: a free root, at least the copies seen earlier this match.
        std::vector<std::pair<int, int>> lower;
        if (game_ > 0)
            for (const auto &s : seen_match_[p])
                if (s.second > 0) lower.emplace_back(s.first, s.second);
        const RootKind kind = lower.empty() ? RootKind::FREE : RootKind::LOWER;
        const std::vector<NodeId> got = add_root_parts(p, kind, lower, {library, sideboard});
        parts = {got[0], got[1], NO_NODE, NO_NODE};
        for (const auto &c : lower) m.full.universe.add(c.first);
    } else {
        const std::vector<DeckEntry> &source = use_own ? own_list : lists_.cards;
        const std::vector<std::pair<int, int>> main = list_counts(source, p, true, false);
        const std::vector<std::pair<int, int>> side = list_counts(source, p, false, true);
        const int n_main = count_total(main), n_side = count_total(side);
        if (!own && game_ > 0) {
            // The registered 75, split between deck and sideboard in a way the viewer cannot see.
            const std::vector<std::pair<int, int>> all = list_counts(source, p, true, true);
            const long rest = static_cast<long>(n_main + n_side) - library - sideboard;
            if (rest < 0) error("game_start counts exceed the registered 75");
            const std::vector<NodeId> got =
                add_root_parts(p, RootKind::EXACT, all, {library, sideboard, clamp_size(rest)});
            parts = {got[0], got[1], got[2], NO_NODE};
        } else {
            const long main_rest = static_cast<long>(n_main) - library;
            const long side_rest = static_cast<long>(n_side) - sideboard;
            if (main_rest < 0 || side_rest < 0) error("game_start counts exceed the registered lists");
            const std::vector<NodeId> got_main =
                add_root_parts(p, RootKind::EXACT, main, {library, clamp_size(main_rest)});
            const std::vector<NodeId> got_side =
                add_root_parts(p, RootKind::EXACT, side, {sideboard, clamp_size(side_rest)});
            parts = {got_main[0], got_side[0], got_main[1], got_side[1]};
        }
        for (const auto &c : list_counts(source, p, true, true)) m.full.universe.add(c.first);
    }
    if (parts[0] != NO_NODE) {
        m.net.node(parts[0]).loc = Loc::LIBRARY;
        m.library.push_back(parts[0]);
    }
    if (parts[1] != NO_NODE) {
        m.net.node(parts[1]).loc = Loc::SIDEBOARD;
        m.sideboard.push_back(parts[1]);
    }
    for (size_t i = 2; i < parts.size(); ++i)
        if (parts[i] != NO_NODE) {
            m.net.node(parts[i]).loc = Loc::SET_ASIDE;
            m.set_aside.push_back(parts[i]);
        }
    m.seen.universe = m.full.universe;
}

// ── game_end ──────────────────────────────────────────────────────────────────────────────────

// The game's last events are already counted into match memory (each was settled); the networks
// stay until the next game_start replaces them.
void Knowledge::on_game_end(const Event &ev) {
    (void)ev;
    game_active_ = false;
}

// ── draw ──────────────────────────────────────────────────────────────────────────────────────

// A unit leaves the top of the drawer's library for their hand (each card drawn is its own
// event). A card the event names is pinned (the drawer's own draw, a card in the viewer's live
// window); a unit the model already knew stays known. Unless it is an opening-hand draw it is
// drawn this turn (Node::drawn_turn) until it leaves the hand or the turn ends. The player's first
// such draw of the turn is the one a draw link may name later (Miracle: the first card drawn this
// turn), so it is remembered for the turn.
void Knowledge::on_draw(const Event &ev) {
    const int p = ev.player;
    if (p == NO_SEAT || ev.cards.empty()) {
        error("a draw without its drawer or card");
        return;
    }
    for (const EventCard &c : ev.cards) {
        const NodeId n = take_unit(p, Zn::LIBRARY, c, ev);
        if (n == NO_NODE) continue;
        PlayerModel &m = players_[p];
        Node &node = m.net.node(n);
        node.loc = Loc::HAND;
        node.drawn_turn = ev.opening ? -1 : static_cast<int16_t>(ev.turn);
        node.draw_seq = ev.seq;
        m.hand.push_back(n);
        if (!ev.opening && !drew_this_turn(p, ev.turn)) {
            DrawRef d;
            d.player = p;
            d.node = n;
            d.turn = ev.turn;
            draws_[ev.seq] = d;
        }
        bind_handle(p, n, c.handle_after);
    }
}

bool Knowledge::drew_this_turn(int p, uint16_t turn) const {
    for (const auto &d : draws_)
        if (d.second.player == p && d.second.turn == turn) return true;
    return false;
}

// ── zone_move, cast_from_hidden, play_from_hidden ─────────────────────────────────────────────

// One card moves: its unit leaves the source (pinned there when the event names it: a card cast
// from hand proves the hand held it) and enters the destination. A token or spell copy is no
// unit.
void Knowledge::on_move(const Event &ev) {
    for (const EventCard &c : ev.cards) {
        if (c.token) {
            drop_handle(c.handle_before);
            continue;
        }
        const int from = ev.zone == Zn::NONE ? ev.to_owner : ev.zone_owner;
        const int to = ev.to_zone == Zn::NONE ? from : ev.to_owner;
        if (from == NO_SEAT || to == NO_SEAT) {
            error("a zone move without its owner");
            continue;
        }
        if (from != to) error("a card moves between two players' zones");
        const NodeId n = take_unit(from, ev.zone, c, ev);
        if (n == NO_NODE) continue;
        put_unit(from, ev.to_zone, n, c, ev);
    }
}

// The unit an event card moves out of `zone` of player `p`, removed from it (and pinned to the
// card the event names). A unit the viewer holds a handle for is exactly that unit; otherwise the
// zone's rule picks it: a library position, range or anywhere; a set zone's sub-pools that can
// hold the card.
NodeId Knowledge::take_unit(int p, Zn zone, const EventCard &c, const Event &ev) {
    PlayerModel &m = players_[p];
    int hp = NO_SEAT;
    NodeId n = handle_node(c.handle_before, &hp);
    if (n != NO_NODE && hp == p) {
        const Node &node = m.net.node(n);
        if (!loc_matches(node.loc, zone))
            error("h" + std::to_string(c.handle_before) + " moves out of zone " +
                  std::to_string(static_cast<int>(zone)) + " but the model holds it elsewhere");
        if (!remove_live(p, n)) error("h" + std::to_string(c.handle_before) + " is not live in the model");
        drop_handle(c.handle_before);
    } else {
        if (c.handle_before && is_public_zone(zone))
            error("h" + std::to_string(c.handle_before) + " leaves a public zone, but the model has no unit for it");
        drop_handle(c.handle_before);
        n = NO_NODE;
        switch (zone) {
            case Zn::LIBRARY:
                n = take_library_at(p, c.from);
                break;
            case Zn::HAND:
                n = take_from_set(p, m.hand, c.card,
                    c.hand_subset == static_cast<uint8_t>(rmstate::HandSubset::DRAWN_THIS_TURN), ev.turn);
                break;
            case Zn::SIDEBOARD:
                n = take_from_set(p, m.sideboard, c.card, false, ev.turn);
                break;
            case Zn::NONE:
                if (!m.set_aside.empty())
                    n = take_from_set(p, m.set_aside, c.card, false, ev.turn);
                else {
                    if (!closed_[p]) error("a card is created from zone none with nothing set aside");
                    n = fresh_unit(p, Loc::SET_ASIDE);
                }
                break;
            default:
                if (!c.handle_before) error("a card leaves a public zone without a handle");
                n = fresh_unit(p, Loc::PUBLIC);
                break;
        }
    }
    if (n == NO_NODE) return NO_NODE;
    if (c.card != NO_CARD) pin_unit(p, n, c.card);
    return n;
}

// Where unit `n` (already removed from its source) lands.
void Knowledge::put_unit(int p, Zn zone, NodeId n, const EventCard &c, const Event &ev) {
    PlayerModel &m = players_[p];
    Node &node = m.net.node(n);
    node.drawn_turn = -1;
    switch (zone) {
        case Zn::LIBRARY: {
            const size_t size = m.library_size();
            if (ev.batch) {
                // Order hidden: the unit goes to the batch's end now and the batch's library_place
                // makes the batch's positions one unordered run.
                const auto it = batch_place_.find(ev.batch);
                const bool bottom = it != batch_place_.end() && it->second.bottom();
                m.insert_library(bottom ? size : 0, n);
                auto &batch = batch_nodes_[ev.batch];
                batch.first = p;
                batch.second.push_back(n);
                return;
            }
            if (c.to.in_library() && c.to.span == 1 && c.to.pos <= size) {
                m.insert_library(c.to.bottom() ? size - c.to.pos : c.to.pos, n);
                return;
            }
            // No exact destination: anywhere in the library.
            if (c.to.in_library() && c.to.span != 0)
                error("a card enters a library at " + std::to_string(c.to.pos) + "+" + std::to_string(c.to.span) +
                      ", not an exact position");
            m.insert_library(0, n);
            m.merge_library();
            return;
        }
        case Zn::HAND:
            node.loc = Loc::HAND;
            m.hand.push_back(n);
            bind_handle(p, n, c.handle_after);
            return;
        case Zn::SIDEBOARD:
            node.loc = Loc::SIDEBOARD;
            m.sideboard.push_back(n);
            bind_handle(p, n, c.handle_after);
            return;
        case Zn::BATTLEFIELD:
        case Zn::STACK:
        case Zn::GRAVEYARD:
        case Zn::EXILE: {
            // A public object; one whose identity the viewer cannot see (exiled face down, a
            // face-down permanent) is a face-down unit keyed by its handle.
            const bool face_down = c.face_down || (c.card == NO_CARD && node.pin < 0);
            node.loc = face_down ? Loc::FACE_DOWN : Loc::PUBLIC;
            (face_down ? m.face_down : m.publics).push_back(n);
            if (!c.handle_after) error("a card enters a public zone without a handle");
            bind_handle(p, n, c.handle_after);
            return;
        }
        case Zn::NONE:
        case Zn::OTHER:
            error("a card (not a token) moves to zone none or an unknown zone");
            node.loc = Loc::PUBLIC;
            m.publics.push_back(n);
            return;
    }
}

// A unit of no known origin (a recovery after an error, or a closed decklist's card created from
// zone none): a free root of one unit.
NodeId Knowledge::fresh_unit(int p, Loc loc) {
    return players_[p].net.add_root(1, loc, RootKind::FREE, {});
}

// The unit at a library source position: an exact position is that unit; a range first merges its
// positions into one unordered run (which looked-at card went where is unknown) and the unit is
// one of it; anywhere merges the whole library.
NodeId Knowledge::take_library_at(int p, const LibPos &from) {
    PlayerModel &m = players_[p];
    const size_t size = m.library_size();
    if (size == 0) {
        error("a card leaves an empty library");
        return fresh_unit(p, Loc::LIBRARY);
    }
    if (from.in_library() && from.span == 1) {
        size_t at = 0;
        if (!library_index(p, from, size, &at)) return fresh_unit(p, Loc::LIBRARY);
        return m.take_library(at);
    }
    if (from.in_library() && from.span > 1) {
        size_t a = 0, b = 0;
        if (!library_range(p, from, &a, &b)) return fresh_unit(p, Loc::LIBRARY);
        m.merge_library_range(a, b);
        return m.take_library(a);
    }
    m.merge_library();
    return m.take_library(0);
}

// One unit of a set zone (hand, sideboard, set-aside) for a card leaving it without a handle: any
// unit the viewer cannot single out by handle (a handled unit would have moved with its handle),
// among the drawn-this-turn ones when the instruction may only pick those. For a known card, a
// unit already known to be it if there is one (copies of a card are interchangeable), else the
// sub-pools that can hold it; for an unknown card, any. Several candidates are merged first (the
// viewer does not know which one gave the card).
NodeId Knowledge::take_from_set(int p, std::vector<NodeId> &zone, int card, bool drawn_this_turn, uint16_t turn) {
    PlayerModel &m = players_[p];
    std::vector<size_t> base;
    for (size_t i = 0; i < zone.size(); ++i) {
        const Node &node = m.net.node(zone[i]);
        if (node.handle != 0) continue;
        if (drawn_this_turn && m.dtt_unknown_turn != turn && node.drawn_turn != static_cast<int16_t>(turn)) continue;
        base.push_back(i);
    }
    std::vector<size_t> cands;
    if (card != NO_CARD) {
        for (size_t i : base)
            if (m.net.node(zone[i]).pin == card) return m.split_unit(zone, i);
        for (size_t i : base)
            if (m.net.node(zone[i]).pin < 0 && can_hold(p, zone[i], card)) cands.push_back(i);
    } else {
        cands = base;
    }
    if (cands.empty()) {
        error("no unit of the zone can be the card " + std::to_string(card) + " that leaves it");
        for (size_t i = 0; i < zone.size(); ++i)
            if (m.net.node(zone[i]).pin < 0 || m.net.node(zone[i]).pin == card) cands.push_back(i);
        if (cands.empty()) {
            if (zone.empty()) return fresh_unit(p, Loc::GONE);
            cands.push_back(0);
        }
    }
    const size_t at = cands.size() == 1 ? cands[0] : merge_candidates(p, zone, cands, turn);
    return m.split_unit(zone, at);
}

// Merges the nodes at `at` of `zone` into one (placed last in the zone); returns its index. A
// hand merge mixing this turn's draws with other cards loses which units were drawn this turn.
size_t Knowledge::merge_candidates(int p, std::vector<NodeId> &zone, const std::vector<size_t> &at, uint16_t turn) {
    PlayerModel &m = players_[p];
    std::vector<NodeId> parts;
    bool any_now = false, all_now = true;
    for (size_t i : at) {
        parts.push_back(zone[i]);
        const bool now = m.net.node(zone[i]).drawn_turn == static_cast<int16_t>(turn);
        any_now = any_now || now;
        all_now = all_now && now;
    }
    const Loc loc = m.net.node(parts[0]).loc;
    if (loc == Loc::HAND && any_now && !all_now) m.dtt_unknown_turn = turn;
    std::vector<NodeId> rest;
    for (size_t i = 0; i < zone.size(); ++i)
        if (std::find(at.begin(), at.end(), i) == at.end()) rest.push_back(zone[i]);
    const NodeId merged = m.net.merge(parts, loc);
    rest.push_back(merged);
    zone = rest;
    return zone.size() - 1;
}

bool Knowledge::can_hold(int p, NodeId n, int card) const {
    const PlayerModel &m = players_[p];
    const Node &node = m.net.node(n);
    if (node.pin >= 0) return node.pin == card;
    if (!m.full.knows(n)) return true;
    return m.full.hi(m.net, n, m.full.universe.index_of(card)) > 0;
}

void Knowledge::pin_unit(int p, NodeId n, int card) {
    PlayerModel &m = players_[p];
    if (m.net.node(n).size != 1) {
        error("a pin on a node of " + std::to_string(m.net.node(n).size) + " units");
        return;
    }
    if (m.full.universe.add(card)) {
        // A card new to the universe (a closed decklist's first sight of it): the stores restart.
        m.seen.universe = m.full.universe;
        m.full.reset();
        m.seen.reset();
    }
    if (!m.net.pin(n, card))
        error("the model knows this unit as card " + std::to_string(m.net.node(n).pin) + ", the event says " +
              std::to_string(card));
}

// ── library_place ─────────────────────────────────────────────────────────────────────────────

// The batch's units occupy the stated range; their order is hidden, so the range becomes one
// unordered run holding them.
void Knowledge::on_library_place(const Event &ev) {
    const auto it = batch_nodes_.find(ev.batch);
    if (it == batch_nodes_.end()) {
        error("library_place for batch " + std::to_string(ev.batch) + " with no card in it");
        return;
    }
    const int p = it->second.first;
    PlayerModel &m = players_[p];
    size_t a = 0, b = 0;
    if (!library_range(p, ev.range, &a, &b)) return;
    if (b - a != it->second.second.size())
        error("library_place covers " + std::to_string(b - a) + " position(s) for a batch of " +
              std::to_string(it->second.second.size()));
    for (NodeId n : it->second.second) {
        const auto seg = std::find(m.library.begin(), m.library.end(), n);
        const size_t start = seg == m.library.end() ? b : m.run_start(static_cast<size_t>(seg - m.library.begin()));
        if (start < a || start >= b) error("a batch card is not in its library_place range");
    }
    m.merge_library_range(a, b);
    batch_nodes_.erase(it);
}

// ── reveal_start, look ────────────────────────────────────────────────────────────────────────

// Every identity the event shows is pinned where it is: a library card at its position (the
// position becomes a run of its own), a hand or sideboard card on a unit of that zone, a face-down
// card on its unit. A card shown as the one a draw brought in (a draw link: Miracle) pins that
// draw's unit, and through it the library position it was drawn from. Shown cards keep the event's
// handles while their windows are open. A look by the other player shows no identity: nothing.
void Knowledge::on_reveal(const Event &ev) {
    const int p = ev.zone_owner;
    if (p == NO_SEAT) return;
    PlayerModel &m = players_[p];
    std::vector<const EventCard *> set_cards;
    for (const EventCard &c : ev.cards) {
        if (c.card == NO_CARD) continue;
        switch (ev.zone) {
            case Zn::LIBRARY: {
                size_t at = 0;
                if (!library_index(p, c.from, m.library_size(), &at)) break;
                const NodeId n = m.isolate_library(at);
                pin_unit(p, n, c.card);
                bind_handle(p, n, c.handle_after);
                break;
            }
            case Zn::HAND:
            case Zn::SIDEBOARD: {
                if (c.draw_seq != 0) {
                    const auto d = draws_.find(c.draw_seq);
                    if (d == draws_.end() || d->second.player != p) {
                        error(
                            "a draw link to draw " + std::to_string(c.draw_seq) + ", which the model has no unit for");
                    } else {
                        const NodeId n = d->second.node;
                        pin_unit(p, n, c.card);
                        const Node &node = m.net.node(n);
                        if (node.loc == Loc::HAND && node.handle == 0) {
                            bind_handle(p, n, c.handle_after);
                            break;
                        }
                    }
                }
                set_cards.push_back(&c);
                break;
            }
            default: {
                int hp = NO_SEAT;
                const NodeId n = handle_node(c.handle_after, &hp);
                if (n != NO_NODE && hp == p && m.net.node(n).loc == Loc::FACE_DOWN) pin_unit(p, n, c.card);
                break;
            }
        }
    }
    if (!set_cards.empty()) assign_revealed(p, ev.zone == Zn::HAND ? m.hand : m.sideboard, set_cards, ev.turn);
}

// Shown cards of a set zone, each on a distinct unit: the one its handle already names, else an
// unhandled unit already known to be that card (copies are interchangeable), else one unit split
// off the unknown sub-pools that can hold it (merged first when several can).
void Knowledge::assign_revealed(
    int p, std::vector<NodeId> &zone, const std::vector<const EventCard *> &cards, uint16_t turn) {
    PlayerModel &m = players_[p];
    for (const EventCard *c : cards) {
        int hp = NO_SEAT;
        const NodeId held = handle_node(c->handle_after, &hp);
        if (held != NO_NODE && hp == p && std::find(zone.begin(), zone.end(), held) != zone.end()) {
            pin_unit(p, held, c->card);
            continue;
        }
        bool done = false;
        for (NodeId n : zone)
            if (m.net.node(n).pin == c->card && m.net.node(n).handle == 0) {
                bind_handle(p, n, c->handle_after);
                done = true;
                break;
            }
        if (done) continue;
        std::vector<size_t> cands;
        for (size_t i = 0; i < zone.size(); ++i) {
            const Node &node = m.net.node(zone[i]);
            if (node.handle == 0 && node.pin < 0 && can_hold(p, zone[i], c->card)) cands.push_back(i);
        }
        if (cands.empty()) {
            error("a shown card " + std::to_string(c->card) + " is not among the units that can be in its zone");
            continue;
        }
        const size_t at = cands.size() == 1 ? cands[0] : merge_candidates(p, zone, cands, turn);
        const int16_t drawn = m.net.node(zone[at]).drawn_turn;
        const NodeId u = m.split_unit(zone, at);
        m.net.node(u).drawn_turn = drawn;
        zone.push_back(u);
        pin_unit(p, u, c->card);
        bind_handle(p, u, c->handle_after);
    }
}

// ── reveal_end ────────────────────────────────────────────────────────────────────────────────

// Visibility ends, knowledge stays: the handles retire, the pins remain.
void Knowledge::on_reveal_end(const Event &ev) {
    for (const EventCard &c : ev.cards) drop_handle(c.handle_before);
}

// ── shuffle ───────────────────────────────────────────────────────────────────────────────────

// Every position of the library merges into one unordered run: position facts are gone, contents
// facts stay (the merged run's contents are its parts').
void Knowledge::on_shuffle(const Event &ev) {
    const int p = ev.zone_owner != NO_SEAT ? ev.zone_owner : ev.player;
    if (p == NO_SEAT) return;
    PlayerModel &m = players_[p];
    for (NodeId n : m.library) drop_handle(m.net.node(n).handle);
    m.merge_library();
}

// ── reorder ───────────────────────────────────────────────────────────────────────────────────

// The range becomes one unordered run, unless the viewer saw the result (then each position is
// pinned).
void Knowledge::on_reorder(const Event &ev) {
    const int p = ev.zone_owner != NO_SEAT ? ev.zone_owner : ev.player;
    if (p == NO_SEAT) return;
    PlayerModel &m = players_[p];
    size_t a = 0, b = 0;
    if (!library_range(p, ev.range, &a, &b)) return;
    if (!ev.order_known) {
        m.merge_library_range(a, b);
        return;
    }
    for (const EventCard &c : ev.cards) {
        size_t at = 0;
        if (c.card == NO_CARD || !library_index(p, c.from, m.library_size(), &at)) continue;
        pin_unit(p, m.isolate_library(at), c.card);
    }
}

// ── face_down, face_up ────────────────────────────────────────────────────────────────────────

// Face up pins the face-down unit and makes it a public card; face down keeps the unit (and what
// the viewer knew of it) as a face-down unit keyed by its handle.
void Knowledge::on_face_turn(const Event &ev) {
    const bool up = ev.kind == rmstate::EventKind::FACE_UP;
    for (const EventCard &c : ev.cards) {
        int p = NO_SEAT;
        const NodeId n = handle_node(c.handle_before ? c.handle_before : c.handle_after, &p);
        if (n == NO_NODE) {
            if (!c.token) error("a face turn of an object the model has no unit for");
            continue;
        }
        PlayerModel &m = players_[p];
        remove_live(p, n);
        drop_handle(m.net.node(n).handle);
        if (up && c.card != NO_CARD) pin_unit(p, n, c.card);
        m.net.node(n).loc = up ? Loc::PUBLIC : Loc::FACE_DOWN;
        (up ? m.publics : m.face_down).push_back(n);
        bind_handle(p, n, c.handle_after ? c.handle_after : c.handle_before);
    }
}

}  // namespace knowledge
