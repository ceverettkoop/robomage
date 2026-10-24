#include "knowledge.h"

#include <algorithm>

#include "serial.h"

namespace knowledge {

// Constraint evaluations one propagation may use per player and store before it stops (still
// sound, possibly looser).
static constexpr uint64_t PROPAGATION_BUDGET = 4000000;
// Steps the exact solver may use per player and frame before it reports the propagated bounds
// for what it has not settled (still sound; k_meta.exact 0).
static constexpr uint64_t SOLVER_BUDGET = 1000000;
static constexpr uint32_t STATE_MAGIC = 0x4E4B4D52u;  // "RMKN"
static constexpr uint16_t STATE_VERSION = 5;

static const char *zone_word(Zn z);
static const char *loc_word(Loc l);
static void put_seen(std::string &out, const std::map<int, uint8_t> &m);
static bool get_seen(ByteReader &r, std::map<int, uint8_t> *m);
static void put_entries(std::string &out, const std::vector<DeckEntry> &v);
static bool get_entries(ByteReader &r, std::vector<DeckEntry> *v);
static std::vector<DeckEntry> entries_of(const std::vector<DeckEntry> &v, int player);
static bool same_entries(const std::vector<DeckEntry> &a, const std::vector<DeckEntry> &b);
static void put_solution(std::string &out, const Solution &s);
static bool get_solution(ByteReader &r, Solution *s);

Knowledge::Knowledge(rmstate::KnowledgeLevel level) : level_(level) {}

void Knowledge::error(const std::string &what) {
    std::string line = "ERROR: knowledge[";
    line += viewer_ == 0 ? "A" : viewer_ == 1 ? "B" : "?";
    line += "] game " + std::to_string(frame_game_) + " decision " + std::to_string(frame_decision_);
    if (frame_seq_) line += " seq " + std::to_string(frame_seq_);
    line += ": " + what;
    errors_.push_back(line);
}

std::vector<std::string> Knowledge::take_errors() {
    std::vector<std::string> out;
    out.swap(errors_);
    return out;
}

// ── Stream input ──────────────────────────────────────────────────────────────────────────────

bool Knowledge::feed(const std::string &payload, rmstate::DecodedFrame *decoded) {
    rmstate::FrameKind kind;
    rmstate::Viewer viewer;
    frame_seq_ = 0;
    if (!peek_frame_header(payload, &kind, &viewer)) {
        error("a record is not an RMSTATE frame");
        return false;
    }
    if (viewer == rmstate::Viewer::OMNISCIENT) {
        error("the knowledge sidecar runs for a seat, not the omniscient viewer");
        return false;
    }
    const int seat = viewer == rmstate::Viewer::A ? 0 : 1;
    if (viewer_ != NO_SEAT && seat != viewer_) {
        error("a frame for the other seat in this viewer's stream");
        return false;
    }
    if (kind == rmstate::FrameKind::SCHEMA) {
        rmstate::Schema schema;
        const rmstate::DecodeResult r = rmstate::decode_schema_frame(payload, &schema);
        if (!r.ok()) {
            error("the SCHEMA frame does not decode: " + r.message);
            return false;
        }
        Codes codes;
        std::string err;
        if (!read_codes(schema, &codes, &err)) {
            error(err);
            return false;
        }
        schema_ = schema;
        schema_payload_ = payload;
        codes_ = codes;
        have_schema_ = true;
        viewer_ = seat;
        return false;
    }
    if (!have_schema_) {
        error("a frame before the SCHEMA frame");
        return false;
    }
    // Level off reads nothing from a STATE frame: its tables are k_meta alone.
    if (kind == rmstate::FrameKind::STATE && level_ == rmstate::KnowledgeLevel::OFF) return false;
    rmstate::DecodedFrame frame;
    const rmstate::DecodeResult r = rmstate::decode_frame(payload, schema_, &frame);
    if (!r.ok()) {
        error("a frame does not decode: " + r.message);
        return false;
    }
    std::string err;
    if (kind == rmstate::FrameKind::CARDS) {
        if (!read_card_names(frame, &names_, &err)) error(err);
        return false;
    }
    if (kind == rmstate::FrameKind::DECKLISTS) {
        Decklists lists;
        if (!read_decklists(frame, codes_, &lists, &err)) {
            error(err);
            return false;
        }
        begin_match(seat, codes_, lists);
        return false;
    }
    Snapshot snap;
    if (!read_snapshot(frame, codes_, &snap, &err)) {
        error(err);
        return false;
    }
    apply_frame(snap);
    if (!decoded) return false;
    *decoded = std::move(frame);
    return true;
}

void Knowledge::begin_match(int viewer, const Codes &codes, const Decklists &lists) {
    solve_stale_ = true;
    viewer_ = viewer;
    codes_ = codes;
    lists_ = lists;
    seen_match_[0].clear();
    seen_match_[1].clear();
    last_seq_ = 0;
    game_active_ = false;
}

// The model is a function of the events alone: every event is applied and then settled (settle),
// so a frame only checks the snapshot against the model and solves the current state. The frame's
// own_deck table is read only by a game_start event in it (the viewer's own root in games 2-3): the
// live 75 changes only while sideboarding, before game_start, so every frame of a game shows the
// list the game started with, and check() confirms it.
void Knowledge::apply_frame(const Snapshot &snap) {
    frame_game_ = snap.game;
    frame_decision_ = snap.decision;
    frame_seq_ = 0;
    if (level_ == rmstate::KnowledgeLevel::OFF) return;
    frame_own_deck_ = &snap.own_deck;
    apply_events(snap.events);
    frame_own_deck_ = nullptr;
    frame_seq_ = 0;
    solve_steps_ = 0;
    if (!game_active_) return;
    check(snap);
    // The solver's answers depend on the model alone, which only events change.
    if (solve_stale_) {
        solve();
        solve_stale_ = false;
    } else {
        solve_steps_ = solved_[0].steps + solved_[1].steps;
    }
}

void Knowledge::apply_events(const std::vector<Event> &events) {
    // An order-hidden batch closes with its library_place in the same frame (one instruction puts
    // its cards into the library with no decision in between); its range says at which end the
    // batch's cards were put, which the cards' own moves do not.
    batch_place_.clear();
    batch_nodes_.clear();
    for (const Event &ev : events)
        if (ev.kind == rmstate::EventKind::LIBRARY_PLACE) batch_place_[ev.batch] = ev.range;
    for (const Event &ev : events) {
        apply_event(ev);
        if (game_active_) settle();
    }
    for (const auto &b : batch_nodes_)
        if (batch_place_.find(b.first) == batch_place_.end())
            error("batch " + std::to_string(b.first) + " has no library_place in its frame");
}

// After every event of a live game: the networks that changed are compacted (not while a batch
// waits for its library_place: its units are named by id until then), the bounds are brought to
// their fixpoint and the copies seen updated. Every later event's rule therefore sees the same
// model whichever frames the events arrived in.
void Knowledge::settle() {
    if (batch_nodes_.empty()) compact();
    propagate();
    update_seen();
}

void Knowledge::apply_event(const Event &ev) {
    using rmstate::EventKind;
    solve_stale_ = true;
    frame_seq_ = ev.seq;
    turn_ = std::max(turn_, ev.turn);
    if (last_seq_ != 0 && ev.seq != last_seq_ + 1)
        error("event seq " + std::to_string(ev.seq) + " follows " + std::to_string(last_seq_) + " (a lost event)");
    last_seq_ = ev.seq;
    if (ev.kind == EventKind::GAME_START) {
        on_game_start(ev);
        return;
    }
    if (ev.kind == EventKind::MATCH_END) return;
    if (!game_active_) {
        if (ev.kind != EventKind::GAME_END) error("an event outside a game");
        return;
    }
    switch (ev.kind) {
        case EventKind::GAME_END:
            on_game_end(ev);
            break;
        case EventKind::DRAW:
            on_draw(ev);
            break;
        case EventKind::ZONE_MOVE:
        case EventKind::CAST_FROM_HIDDEN:
        case EventKind::PLAY_FROM_HIDDEN:
            on_move(ev);
            break;
        case EventKind::LIBRARY_PLACE:
            on_library_place(ev);
            break;
        case EventKind::REVEAL_START:
        case EventKind::LOOK:
            on_reveal(ev);
            break;
        case EventKind::REVEAL_END:
            on_reveal_end(ev);
            break;
        case EventKind::SHUFFLE:
            on_shuffle(ev);
            break;
        case EventKind::REORDER:
            on_reorder(ev);
            break;
        case EventKind::SEARCH:
            // Nothing by itself: the look at the searched zone and the found cards' moves are
            // events of their own (knowledge_store §5).
            break;
        case EventKind::FACE_DOWN:
        case EventKind::FACE_UP:
            on_face_turn(ev);
            break;
        case EventKind::GAME_START:
        case EventKind::MATCH_END:
            break;
    }
}

// ── Compaction, propagation, solving and match memory ─────────────────────────────────────────

// Compacts every network that changed since its last compaction. A draw link only names a draw of
// the turn it is in, so earlier turns' draws are forgotten first and this turn's are kept.
// Handles and draw links follow their units to the new ids.
void Knowledge::compact() {
    if (!compaction_) return;
    for (auto it = draws_.begin(); it != draws_.end();)
        if (it->second.turn != turn_)
            it = draws_.erase(it);
        else
            ++it;
    for (int p = 0; p < 2; ++p) {
        PlayerModel &m = players_[p];
        if (!m.needs_compaction()) continue;
        std::vector<NodeId> protect;
        for (const auto &d : draws_)
            if (d.second.player == p) protect.push_back(d.second.node);
        std::vector<NodeId> remap;
        m.compact(protect, &remap);
        for (auto it = handles_.begin(); it != handles_.end();) {
            if (it->second.first == p) {
                const NodeId n = remap[it->second.second];
                if (n == NO_NODE) {
                    it = handles_.erase(it);
                    continue;
                }
                it->second.second = n;
            }
            ++it;
        }
        for (auto &d : draws_)
            if (d.second.player == p) d.second.node = remap[d.second.node];
    }
}

std::vector<std::vector<NodeId>> Knowledge::pool_groups(int p) const {
    std::vector<std::vector<NodeId>> groups;
    for (const PoolSpec &spec : pool_specs(p)) groups.push_back(spec.nodes);
    return groups;
}

// The exact solver over every pool the tables may report (pool_specs), per player.
void Knowledge::solve() {
    solve_steps_ = 0;
    for (int p = 0; p < 2; ++p) {
        const PlayerModel &m = players_[p];
        solved_[p] = solver_.solve(m.net, m.full, pool_groups(p), SOLVER_BUDGET);
        solve_steps_ += solved_[p].steps;
        if (solved_[p].contradiction && m.full.contradiction().empty() && !reported_contradiction_[p][0]) {
            reported_contradiction_[p][0] = true;
            error(std::string("player ") + (p == 0 ? "A" : "B") + ": the exact solver finds no world");
        }
    }
}

void Knowledge::propagate() {
    steps_ = 0;
    for (int p = 0; p < 2; ++p) {
        PlayerModel &m = players_[p];
        const std::vector<NodeId> live = m.live();
        BoundStore *stores[2] = {&m.full, &m.seen};
        for (int s = 0; s < 2; ++s) {
            steps_ += stores[s]->propagate(m.net, live, PROPAGATION_BUDGET);
            if (!stores[s]->contradiction().empty() && !reported_contradiction_[p][s]) {
                reported_contradiction_[p][s] = true;
                error(std::string("player ") + (p == 0 ? "A" : "B") + (s == 0 ? " deduction" : " observation") +
                      " bounds are contradictory: " + stores[s]->contradiction());
            }
        }
    }
}

// Copies seen: in the observation store (roots free), the fewest copies of each card the player
// can own given everything observed this game: at least the live nodes' minimum (they are
// disjoint) and the roots'. Per match, the most seen in any game of it.
void Knowledge::update_seen() {
    for (int p = 0; p < 2; ++p) {
        const PlayerModel &m = players_[p];
        const std::vector<NodeId> live = m.live();
        for (size_t k = 0; k < m.seen.universe.cards.size(); ++k) {
            int live_lo = 0, root_lo = 0;
            for (NodeId n : live) live_lo += m.seen.lo(m.net, n, k);
            for (const Root &r : m.net.roots()) root_lo += m.seen.lo(m.net, r.node, k);
            const int s = std::min(255, std::max(live_lo, root_lo));
            if (s == 0) continue;
            const int card = m.seen.universe.cards[k];
            uint8_t &g = seen_game_[p][card];
            g = static_cast<uint8_t>(std::max<int>(g, s));
            uint8_t &mt = seen_match_[p][card];
            mt = std::max(mt, g);
        }
    }
}

// ── Consistency with the snapshot ─────────────────────────────────────────────────────────────

void Knowledge::check(const Snapshot &snap) {
    frame_seq_ = 0;
    if (snap.sideboard_phase || snap.terminal) return;
    for (const SnapPlayer &sp : snap.players) {
        if (sp.player == NO_SEAT) continue;
        const PlayerModel &m = players_[sp.player];
        const char *who = sp.player == 0 ? "A" : "B";
        size_t hand = 0, sideboard = 0;
        for (NodeId n : m.hand) hand += m.net.node(n).size;
        for (NodeId n : m.sideboard) sideboard += m.net.node(n).size;
        if (hand != sp.hand)
            error(std::string(who) + "'s hand holds " + std::to_string(sp.hand) + " card(s), the model " +
                  std::to_string(hand));
        if (m.library_size() != sp.library)
            error(std::string(who) + "'s library holds " + std::to_string(sp.library) + " card(s), the model " +
                  std::to_string(m.library_size()));
        if (sideboard != sp.sideboard)
            error(std::string(who) + "'s sideboard holds " + std::to_string(sp.sideboard) + " card(s), the model " +
                  std::to_string(sideboard));
    }

    // The viewer's own deck is the one the game started with (the root of the viewer's own model).
    if (viewer_ != NO_SEAT && !own_deck_.empty() && !snap.own_deck.empty() &&
        !same_entries(entries_of(own_deck_, viewer_), entries_of(snap.own_deck, viewer_)))
        error("the viewer's own_deck differs from the one the game started with");

    // Every object the snapshot shows: public and face-down objects are the model's public and
    // face-down units (same handle, same card); the viewer's own hand and sideboard are exactly what
    // the model holds there.
    std::map<uint32_t, const SnapObject *> shown;
    std::map<int, int> own_hand, own_sideboard;
    for (const SnapObject &o : snap.objects) {
        shown[o.handle] = &o;
        if (o.owner == NO_SEAT) continue;
        if (o.zone == Zn::HAND || o.zone == Zn::SIDEBOARD || o.zone == Zn::LIBRARY) {
            if (o.owner == viewer_ && o.zone == Zn::HAND) ++own_hand[o.card];
            if (o.owner == viewer_ && o.zone == Zn::SIDEBOARD) ++own_sideboard[o.card];
            continue;
        }
        if (o.token) continue;
        int p = NO_SEAT;
        const NodeId n = handle_node(o.handle, &p);
        // An ability or spell copy on the stack is no card: the model holds no unit for it.
        const bool non_card = std::binary_search(snap.stack_non_cards.begin(), snap.stack_non_cards.end(), o.handle);
        if (n == NO_NODE) {
            if (!non_card)
                error(
                    std::string("no model unit for the ") + zone_word(o.zone) + " object h" + std::to_string(o.handle));
            continue;
        }
        if (non_card) {
            error("the stack object h" + std::to_string(o.handle) +
                  " is an ability or spell copy, the model holds a unit for it");
            continue;
        }
        const Node &node = players_[p].net.node(n);
        if (p != o.owner || node.loc != (o.face_down ? Loc::FACE_DOWN : Loc::PUBLIC))
            error("object h" + std::to_string(o.handle) + " is a " + loc_word(node.loc) + " unit in the model");
        if (o.card != NO_CARD && node.pin != o.card)
            error("object h" + std::to_string(o.handle) + " shows card " + std::to_string(o.card) + ", the model has " +
                  std::to_string(node.pin));
    }
    for (int p = 0; p < 2; ++p)
        for (const std::vector<NodeId> *z : {&players_[p].publics, &players_[p].face_down})
            for (NodeId n : *z) {
                const uint32_t h = players_[p].net.node(n).handle;
                if (shown.find(h) == shown.end())
                    error(std::string("the model's ") + loc_word(players_[p].net.node(n).loc) + " unit h" +
                          std::to_string(h) + " is not in the snapshot");
            }

    if (viewer_ != NO_SEAT) {
        const PlayerModel &own = players_[viewer_];
        std::map<int, int> model_hand;
        for (NodeId n : own.hand) {
            const Node &node = own.net.node(n);
            if (node.pin < 0 || node.size != 1)
                error("the viewer's own hand has a unit the model cannot name");
            else
                ++model_hand[node.pin];
        }
        if (model_hand != own_hand) error("the viewer's own hand differs from the model's");
        for (const auto &c : own_sideboard) {
            const size_t k = own.full.universe.index_of(c.first);
            int lo = 0, hi = 0;
            for (NodeId n : own.sideboard) lo += own.full.lo(own.net, n, k), hi += own.full.hi(own.net, n, k);
            if (c.second < lo || c.second > hi)
                error("the viewer's sideboard holds " + std::to_string(c.second) + " of card " +
                      std::to_string(c.first) + ", the model says " + std::to_string(lo) + "-" + std::to_string(hi));
        }
    }

    // Every reveal window agrees with the model: a library window's card is the unit at its
    // position, a hand window's card is the unit its handle names.
    for (const SnapReveal &rv : snap.reveals) {
        if (rv.owner == NO_SEAT || rv.card == NO_CARD) continue;
        const PlayerModel &m = players_[rv.owner];
        if (rv.zone == Zn::LIBRARY) {
            size_t off = 0;
            const size_t seg = m.locate(rv.pos, &off);
            if (seg == m.library.size() || m.net.node(m.library[seg]).size != 1) {
                error("a reveal window shows library position " + std::to_string(rv.pos) +
                      ", which the model does not hold as a known position");
                continue;
            }
            const Node &node = m.net.node(m.library[seg]);
            if (node.pin != rv.card)
                error("a reveal window shows card " + std::to_string(rv.card) + " at library position " +
                      std::to_string(rv.pos) + ", the model has " + std::to_string(node.pin));
            continue;
        }
        int p = NO_SEAT;
        const NodeId n = handle_node(rv.handle, &p);
        if (n == NO_NODE) {
            if (rv.zone == Zn::HAND || rv.zone == Zn::SIDEBOARD)
                error(std::string("no model unit for the ") + zone_word(rv.zone) + " reveal h" +
                      std::to_string(rv.handle));
            continue;
        }
        if (players_[p].net.node(n).pin != rv.card)
            error("reveal h" + std::to_string(rv.handle) + " shows card " + std::to_string(rv.card) +
                  ", the model has " + std::to_string(players_[p].net.node(n).pin));
    }
}

// ── Handles ───────────────────────────────────────────────────────────────────────────────────

NodeId Knowledge::handle_node(uint32_t h, int *player) const {
    if (h == 0) return NO_NODE;
    const auto it = handles_.find(h);
    if (it == handles_.end()) return NO_NODE;
    const Node &node = players_[it->second.first].net.node(it->second.second);
    if (node.handle != h || node.loc == Loc::GONE) return NO_NODE;
    *player = it->second.first;
    return it->second.second;
}

void Knowledge::bind_handle(int p, NodeId n, uint32_t h) {
    if (h == 0) return;
    players_[p].net.node(n).handle = h;
    handles_[h] = std::make_pair(p, n);
}

void Knowledge::drop_handle(uint32_t h) {
    const auto it = handles_.find(h);
    if (it == handles_.end()) return;
    Node &node = players_[it->second.first].net.node(it->second.second);
    if (node.handle == h) node.handle = 0;
    handles_.erase(it);
}

// ── Words for messages ────────────────────────────────────────────────────────────────────────

static const char *zone_word(Zn z) {
    switch (z) {
        case Zn::LIBRARY:
            return "library";
        case Zn::BATTLEFIELD:
            return "battlefield";
        case Zn::HAND:
            return "hand";
        case Zn::STACK:
            return "stack";
        case Zn::GRAVEYARD:
            return "graveyard";
        case Zn::EXILE:
            return "exile";
        case Zn::SIDEBOARD:
            return "sideboard";
        case Zn::NONE:
            return "none";
        case Zn::OTHER:
            return "other";
    }
    return "?";
}

static const char *loc_word(Loc l) {
    switch (l) {
        case Loc::GONE:
            return "consumed";
        case Loc::LIBRARY:
            return "library";
        case Loc::HAND:
            return "hand";
        case Loc::SIDEBOARD:
            return "sideboard";
        case Loc::SET_ASIDE:
            return "set-aside";
        case Loc::FACE_DOWN:
            return "face-down";
        case Loc::PUBLIC:
            return "public";
    }
    return "?";
}

// Removes live node `n` of player `p` from wherever it is (a library run or a set zone). False
// when it is not live there.
bool Knowledge::remove_live(int p, NodeId n) {
    PlayerModel &m = players_[p];
    const Loc loc = m.net.node(n).loc;
    if (loc == Loc::LIBRARY) {
        const auto it = std::find(m.library.begin(), m.library.end(), n);
        if (it == m.library.end()) return false;
        m.erase_library_run(static_cast<size_t>(it - m.library.begin()));
        return true;
    }
    std::vector<NodeId> *z = m.zone_list(loc);
    if (!z) return false;
    const auto it = std::find(z->begin(), z->end(), n);
    if (it == z->end()) return false;
    z->erase(it);
    return true;
}

bool Knowledge::library_index(int p, const LibPos &pos, size_t size, size_t *out) {
    (void)p;
    if (!pos.in_library() || pos.span != 1 || pos.pos >= size) {
        error("library position " + std::to_string(pos.pos) + " (anchor " + std::to_string(pos.anchor) + ", span " +
              std::to_string(pos.span) + ") is not an exact position in a library of " + std::to_string(size));
        return false;
    }
    *out = pos.bottom() ? size - 1 - pos.pos : pos.pos;
    return true;
}

bool Knowledge::library_range(int p, const LibPos &pos, size_t *a, size_t *b) {
    const size_t size = players_[p].library_size();
    if (!pos.in_library() || pos.span == 0 || static_cast<size_t>(pos.pos) + pos.span > size) {
        error("library range " + std::to_string(pos.pos) + "+" + std::to_string(pos.span) + " (anchor " +
              std::to_string(pos.anchor) + ") does not fit a library of " + std::to_string(size));
        return false;
    }
    *a = pos.bottom() ? size - pos.pos - pos.span : pos.pos;
    *b = *a + pos.span;
    return true;
}

// ── Serialisation ─────────────────────────────────────────────────────────────────────────────

static void put_seen(std::string &out, const std::map<int, uint8_t> &m) {
    put_u32(out, m.size());
    for (const auto &e : m) {
        put_i32(out, e.first);
        put_u8(out, e.second);
    }
}

static bool get_seen(ByteReader &r, std::map<int, uint8_t> *m) {
    m->clear();
    const uint64_t n = r.u32();
    if (!r.count(n, r.in.size())) return false;
    for (uint64_t i = 0; i < n; ++i) {
        const int card = r.i32();
        (*m)[card] = static_cast<uint8_t>(r.u8());
    }
    return r.ok;
}

static void put_entries(std::string &out, const std::vector<DeckEntry> &v) {
    put_u32(out, v.size());
    for (const DeckEntry &e : v) {
        put_i32(out, e.player);
        put_i32(out, e.card);
        put_i32(out, e.main);
        put_i32(out, e.side);
    }
}

static bool get_entries(ByteReader &r, std::vector<DeckEntry> *v) {
    v->clear();
    const uint64_t n = r.u32();
    if (!r.count(n, r.in.size())) return false;
    for (uint64_t i = 0; i < n; ++i) {
        DeckEntry e;
        e.player = r.i32();
        e.card = r.i32();
        e.main = r.i32();
        e.side = r.i32();
        v->push_back(e);
    }
    return r.ok;
}

static std::vector<DeckEntry> entries_of(const std::vector<DeckEntry> &v, int player) {
    std::vector<DeckEntry> out;
    for (const DeckEntry &e : v)
        if (e.player == player) out.push_back(e);
    return out;
}

static bool same_entries(const std::vector<DeckEntry> &a, const std::vector<DeckEntry> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].card != b[i].card || a[i].main != b[i].main || a[i].side != b[i].side) return false;
    return true;
}

static void put_solution(std::string &out, const Solution &s) {
    put_u8(out, s.exact ? 1 : 0);
    put_u8(out, s.contradiction ? 1 : 0);
    put_u64(out, s.steps);
    put_u32(out, s.groups.size());
    for (const GroupBounds &g : s.groups) {
        put_str(out, std::string(g.lo.begin(), g.lo.end()));
        put_str(out, std::string(g.hi.begin(), g.hi.end()));
    }
}

static bool get_solution(ByteReader &r, Solution *s) {
    *s = Solution{};
    s->exact = r.u8() != 0;
    s->contradiction = r.u8() != 0;
    s->steps = r.u64();
    const uint64_t n = r.u32();
    if (!r.count(n, r.in.size())) return false;
    for (uint64_t i = 0; i < n; ++i) {
        GroupBounds g;
        const std::string lo = r.str(), hi = r.str();
        g.lo.assign(lo.begin(), lo.end());
        g.hi.assign(hi.begin(), hi.end());
        if (g.lo.size() != g.hi.size()) return false;
        s->groups.push_back(g);
    }
    return r.ok;
}

void Knowledge::serialize(std::string &out) const {
    put_u32(out, STATE_MAGIC);
    put_u16(out, STATE_VERSION);
    put_u8(out, static_cast<uint8_t>(level_));
    put_i32(out, viewer_);
    put_u8(out, have_schema_ ? 1 : 0);
    put_str(out, schema_payload_);
    put_u32(out, codes_.zone_of.size());
    for (Zn z : codes_.zone_of) put_u8(out, static_cast<uint8_t>(z));
    put_u8(out, codes_.seat_value[0]);
    put_u8(out, codes_.seat_value[1]);
    put_u8(out, codes_.seat_none);
    put_u8(out, lists_.present[0] ? 1 : 0);
    put_u8(out, lists_.present[1] ? 1 : 0);
    put_entries(out, lists_.cards);
    for (int p = 0; p < 2; ++p) put_seen(out, seen_match_[p]);
    put_u32(out, last_seq_);
    put_u8(out, game_active_ ? 1 : 0);
    put_u8(out, game_);
    for (int p = 0; p < 2; ++p) put_u8(out, closed_[p] ? 1 : 0);
    put_entries(out, own_deck_);
    for (int p = 0; p < 2; ++p) players_[p].serialize(out);
    put_u32(out, handles_.size());
    for (const auto &e : handles_) {
        put_u32(out, e.first);
        put_u8(out, static_cast<uint8_t>(e.second.first));
        put_u32(out, e.second.second);
    }
    put_u32(out, draws_.size());
    for (const auto &e : draws_) {
        put_u32(out, e.first);
        put_u8(out, static_cast<uint8_t>(e.second.player));
        put_u32(out, e.second.node);
        put_u16(out, e.second.turn);
    }
    for (int p = 0; p < 2; ++p) put_seen(out, seen_game_[p]);
    for (int p = 0; p < 2; ++p)
        for (int s = 0; s < 2; ++s) put_u8(out, reported_contradiction_[p][s] ? 1 : 0);
    put_u64(out, steps_);
    put_u16(out, turn_);
    for (int p = 0; p < 2; ++p) put_solution(out, solved_[p]);
    put_u64(out, solve_steps_);
    for (int p = 0; p < 2; ++p) put_str(out, lists_.name[p]);
}

bool Knowledge::deserialize(const std::string &in) {
    solve_stale_ = true;
    ByteReader r(in, 0);
    if (r.u32() != STATE_MAGIC || r.u16() != STATE_VERSION) return false;
    level_ = static_cast<rmstate::KnowledgeLevel>(r.u8());
    viewer_ = r.i32();
    const bool had_schema = have_schema_;
    have_schema_ = r.u8() != 0;
    std::string payload = r.str();
    if (!r.ok) return false;
    // A restore into the instance that saved the state (an engine snapshot) finds the same SCHEMA
    // frame already decoded.
    const bool same_schema = had_schema && payload == schema_payload_;
    schema_payload_ = std::move(payload);
    if (have_schema_ && !same_schema) {
        schema_ = rmstate::Schema{};
        if (!rmstate::decode_schema_frame(schema_payload_, &schema_).ok()) return false;
    }
    const uint64_t n_zones = r.u32();
    if (!r.count(n_zones, in.size())) return false;
    codes_.zone_of.clear();
    for (uint64_t i = 0; i < n_zones; ++i) codes_.zone_of.push_back(static_cast<Zn>(r.u8()));
    codes_.seat_value[0] = static_cast<uint8_t>(r.u8());
    codes_.seat_value[1] = static_cast<uint8_t>(r.u8());
    codes_.seat_none = static_cast<uint8_t>(r.u8());
    lists_.present[0] = r.u8() != 0;
    lists_.present[1] = r.u8() != 0;
    if (!get_entries(r, &lists_.cards)) return false;
    for (int p = 0; p < 2; ++p)
        if (!get_seen(r, &seen_match_[p])) return false;
    last_seq_ = static_cast<uint32_t>(r.u32());
    game_active_ = r.u8() != 0;
    game_ = static_cast<uint8_t>(r.u8());
    for (int p = 0; p < 2; ++p) closed_[p] = r.u8() != 0;
    if (!get_entries(r, &own_deck_) || !r.ok) return false;
    size_t at = r.at;
    for (int p = 0; p < 2; ++p)
        if (!players_[p].deserialize(in, &at)) return false;
    ByteReader r2(in, at);
    handles_.clear();
    const uint64_t n_handles = r2.u32();
    if (!r2.count(n_handles, in.size())) return false;
    for (uint64_t i = 0; i < n_handles; ++i) {
        const uint32_t key = static_cast<uint32_t>(r2.u32());
        const int p = static_cast<int>(r2.u8());
        const NodeId node = static_cast<NodeId>(r2.u32());
        if (p > 1 || node >= players_[p].net.node_count()) return false;
        handles_[key] = std::make_pair(p, node);
    }
    draws_.clear();
    const uint64_t n_draws = r2.u32();
    if (!r2.count(n_draws, in.size())) return false;
    for (uint64_t i = 0; i < n_draws; ++i) {
        const uint32_t key = static_cast<uint32_t>(r2.u32());
        DrawRef d;
        d.player = static_cast<int>(r2.u8());
        d.node = static_cast<NodeId>(r2.u32());
        d.turn = static_cast<uint16_t>(r2.u16());
        if (d.player > 1 || d.node >= players_[d.player].net.node_count()) return false;
        draws_[key] = d;
    }
    for (int p = 0; p < 2; ++p)
        if (!get_seen(r2, &seen_game_[p])) return false;
    for (int p = 0; p < 2; ++p)
        for (int s = 0; s < 2; ++s) reported_contradiction_[p][s] = r2.u8() != 0;
    steps_ = r2.u64();
    turn_ = static_cast<uint16_t>(r2.u16());
    for (int p = 0; p < 2; ++p)
        if (!get_solution(r2, &solved_[p])) return false;
    solve_steps_ = r2.u64();
    for (int p = 0; p < 2; ++p) lists_.name[p] = r2.str();
    return r2.ok && r2.at == in.size();
}

}  // namespace knowledge
