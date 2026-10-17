// Unit tests of the knowledge sidecar's update rules (docs/knowledge_store.md §5) on small
// synthetic event streams: each test builds a two-player match with tiny registered lists, feeds
// hand-written events through Knowledge::apply_frame, and asserts on the k_* rows. Viewer A; the
// opponent is B (player 1).

#include "selftest.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "../../src/knowledge/knowledge.h"
#include "test_stream.h"

using knowledge::Event;
using knowledge::EventCard;
using knowledge::Knowledge;
using knowledge::KRows;
using knowledge::LibPos;
using knowledge::NO_CARD;
using knowledge::Snapshot;
using knowledge::Zn;
using rmstate::EventKind;

namespace {

// Cards of the synthetic lists.
constexpr int FOW = 101, DAZE = 102, XCARD = 103, FILL = 104, BOLT = 105, MIRACLE = 106, PONDER_A = 107, PONDER_B = 108,
              PONDER_C = 109;
constexpr int A = 0, B = 1;
using rmknow_test::Stream;

struct Result {
        int failures = 0;
        std::string first;
};

}  // namespace

static void expect(Result &r, bool ok, const std::string &what);
static void begin(Knowledge &k, const std::vector<std::pair<int, int>> &a, const std::vector<std::pair<int, int>> &b,
    bool closed = false);
static const knowledge::KPositionRow *position(const KRows &r, int owner, int pos);
static std::pair<int, int> range_of(const KRows &r, uint16_t pool, int card);
static int library_pool(const KRows &r, int owner, int pos);
static bool positions_are(const KRows &r, int owner, int from, int to, int card);
static int known_in_hand(const KRows &r, int card);
static int seen(const KRows &r, int player, int card, bool match);
static void no_errors(Result &r, Knowledge &k);
static Result test_brainstorm_put_back();
static Result test_cut_accounting();
static Result test_scry_range();
static Result test_batch();
static Result test_drawn_this_turn();
static Result test_draw_link();
static Result test_own_rearrange();
static Result test_search_rejoin();
static Result test_mulligan();
static Result test_closed_match_memory();
static Result test_serialize();
static Result test_compaction_bounded();
static Result test_reference_lists();
static const knowledge::KTemplateRow *template_row(const KRows &r, int player, int list);
static const knowledge::KArchetypeRow *archetype_row(const KRows &r, int player, int archetype);

static void expect(Result &r, bool ok, const std::string &what) {
    if (ok) return;
    if (r.failures++ == 0) r.first = what;
}

// A match where A and B registered `a` and `b` (card, main copies); B's list is absent under
// `closed`.
static void begin(
    Knowledge &k, const std::vector<std::pair<int, int>> &a, const std::vector<std::pair<int, int>> &b, bool closed) {
    knowledge::Decklists lists;
    lists.present[A] = true;
    lists.present[B] = !closed;
    for (const auto &c : a) lists.cards.push_back({A, c.first, c.second, 0});
    if (!closed)
        for (const auto &c : b) lists.cards.push_back({B, c.first, c.second, 0});
    knowledge::Codes codes;
    k.begin_match(A, codes, lists);
}

// The k_position row of `owner`'s top index `pos`, or nullptr.
static const knowledge::KPositionRow *position(const KRows &r, int owner, int pos) {
    for (const knowledge::KPositionRow &p : r.positions)
        if (p.owner == owner && p.at.pos == pos) return &p;
    return nullptr;
}

// The [min, max] of `card` in pool `pool` ([0, 0] when it has no row).
static std::pair<int, int> range_of(const KRows &r, uint16_t pool, int card) {
    for (const knowledge::KPoolCardRow &c : r.pool_cards)
        if (c.pool == pool && c.card == card) return {c.min, c.max};
    return {0, 0};
}

// The k_pool row of `owner`'s library run starting at top index `pos`.
static int library_pool(const KRows &r, int owner, int pos) {
    for (size_t i = 0; i < r.pools.size(); ++i)
        if (r.pools[i].owner == owner && r.pools[i].kind == rmstate::PoolKind::LIBRARY_RANGE &&
            r.pools[i].range.pos == pos)
            return static_cast<int>(i);
    return -1;
}

// Whether `owner`'s top positions [from, to) are all known to be `card`.
static bool positions_are(const KRows &r, int owner, int from, int to, int card) {
    for (int pos = from; pos < to; ++pos) {
        const knowledge::KPositionRow *p = position(r, owner, pos);
        if (!p || p->card != card) return false;
    }
    return true;
}

static int known_in_hand(const KRows &r, int card) {
    for (const knowledge::KKnownHandRow &k : r.known_hand)
        if (k.card == card) return k.copies;
    return 0;
}

static int seen(const KRows &r, int player, int card, bool match) {
    for (const knowledge::KSeenRow &s : r.seen)
        if (s.player == player && s.card == card) return match ? s.match : s.game;
    return 0;
}

static void no_errors(Result &r, Knowledge &k) {
    for (const std::string &e : k.take_errors()) expect(r, false, "unexpected " + e);
}

// knowledge_store §4.2 / §7: two cards put back from a hand known to be {Force of Will, Daze, X},
// then X cast: the top two are exactly Force of Will and Daze. The two put-back units come out of
// the same hand pool, so nothing tells them apart: they are one run of the top two (each position
// one of its cards), holding one of each, with X and the filler excluded.
static Result test_brainstorm_put_back() {
    Result r;
    Knowledge k;
    begin(k, {{FILL, 15}}, {{FOW, 1}, {DAZE, 1}, {XCARD, 1}, {FILL, 12}});
    Stream s(k);
    s.game_start(15, 15);
    for (int i = 0; i < 3; ++i) s.draw(B, NO_CARD, 0, true, 0);
    s.shown(EventKind::REVEAL_START, B, Zn::HAND,
        {Stream::card(FOW, 0, 11), Stream::card(DAZE, 0, 12), Stream::card(XCARD, 0, 13)});
    s.shown(
        EventKind::REVEAL_END, B, Zn::HAND, {Stream::card(FOW, 11), Stream::card(DAZE, 12), Stream::card(XCARD, 13)});
    for (int i = 0; i < 2; ++i)
        s.move(EventKind::ZONE_MOVE, B, Zn::HAND, Zn::LIBRARY, Stream::card(NO_CARD, 0, 0, LibPos{}, Stream::top(0)));
    s.move(EventKind::CAST_FROM_HIDDEN, B, Zn::HAND, Zn::STACK, Stream::card(XCARD, 0, 20));
    s.frame();
    no_errors(r, k);
    const KRows rows = k.rows();
    for (int pos = 0; pos < 2; ++pos) {
        const knowledge::KPositionRow *p = position(rows, B, pos);
        expect(r, !p || p->card == NO_CARD, "top " + std::to_string(pos) + " is not a known card");
    }
    expect(r, positions_are(rows, B, 2, 14, FILL), "the rest of the library is the 12 fillers");
    int block = -1;
    for (size_t i = 0; i < rows.pools.size(); ++i)
        if (rows.pools[i].owner == B && rows.pools[i].kind == rmstate::PoolKind::LIBRARY_RANGE &&
            rows.pools[i].range.pos == 0 && rows.pools[i].size == 2)
            block = static_cast<int>(i);
    expect(r, block >= 0, "a pool for the top two");
    if (block >= 0) {
        const uint16_t b = static_cast<uint16_t>(block);
        expect(r, range_of(rows, b, FOW) == std::make_pair(1, 1), "exactly one Force of Will in the top two");
        expect(r, range_of(rows, b, DAZE) == std::make_pair(1, 1), "exactly one Daze in the top two");
        expect(r, range_of(rows, b, XCARD).second == 0 && range_of(rows, b, FILL).second == 0,
            "nothing else in the top two");
    }
    expect(r, rows.meta.exact, "the solver is exact");
    return r;
}

// One of four Force of Will cast from an unknown hand: at most three remain in the library (the
// current cut: the roots equal the live nodes), and one copy is seen.
static Result test_cut_accounting() {
    Result r;
    Knowledge k;
    begin(k, {{FILL, 15}}, {{FOW, 4}, {FILL, 11}});
    Stream s(k);
    s.game_start(15, 15);
    for (int i = 0; i < 7; ++i) s.draw(B, NO_CARD, 0, true, 0);
    s.move(EventKind::CAST_FROM_HIDDEN, B, Zn::HAND, Zn::STACK, Stream::card(FOW, 0, 30));
    s.frame();
    no_errors(r, k);
    const KRows rows = k.rows();
    const int lib = library_pool(rows, B, 0);
    expect(r, lib >= 0 && range_of(rows, static_cast<uint16_t>(lib), FOW) == std::make_pair(0, 3),
        "library Force of Will 0-3");
    expect(r, seen(rows, B, FOW, false) == 1 && seen(rows, B, FOW, true) == 1, "one Force of Will seen");
    return r;
}

// The opponent scries 2 and bottoms one: "a card from top 0-1 to the bottom". When the card left on
// top is then revealed as the deck's only Force of Will, the bottomed card cannot be one.
static Result test_scry_range() {
    Result r;
    Knowledge k;
    begin(k, {{FILL, 15}}, {{FOW, 1}, {FILL, 14}});
    Stream s(k);
    s.game_start(15, 15);
    s.shown(EventKind::LOOK, B, Zn::LIBRARY,
        {Stream::card(NO_CARD, 0, 0, Stream::top(0)), Stream::card(NO_CARD, 0, 0, Stream::top(1))});
    s.move(EventKind::ZONE_MOVE, B, Zn::LIBRARY, Zn::LIBRARY,
        Stream::card(NO_CARD, 0, 0, Stream::top(0, 2), Stream::bottom(0)));
    s.shown(EventKind::REVEAL_START, B, Zn::LIBRARY, {Stream::card(FOW, 0, 40, Stream::top(0))});
    s.frame();
    no_errors(r, k);
    const KRows rows = k.rows();
    const knowledge::KPositionRow *t0 = position(rows, B, 0);
    expect(r, t0 && t0->card == FOW, "top 0 is Force of Will");
    const knowledge::KPositionRow *bot = position(rows, B, 14);
    expect(r, bot && bot->card == FILL, "the bottomed card is a filler");
    return r;
}

// Dig: the opponent looks at the top 3, takes one to hand (a pick among the three) and puts the rest
// on the bottom in an order the viewer does not see (a batch closed by library_place).
static Result test_batch() {
    Result r;
    Knowledge k;
    begin(k, {{FILL, 15}}, {{FOW, 2}, {FILL, 13}});
    Stream s(k);
    s.game_start(15, 15);
    s.shown(EventKind::LOOK, B, Zn::LIBRARY,
        {Stream::card(NO_CARD, 0, 0, Stream::top(0)), Stream::card(NO_CARD, 0, 0, Stream::top(1)),
            Stream::card(NO_CARD, 0, 0, Stream::top(2))});
    s.move(EventKind::ZONE_MOVE, B, Zn::LIBRARY, Zn::HAND, Stream::card(NO_CARD, 0, 0, Stream::top(0, 3)));
    s.move(EventKind::ZONE_MOVE, B, Zn::LIBRARY, Zn::LIBRARY, Stream::card(NO_CARD, 0, 0, Stream::top(0, 2)), 1);
    s.move(EventKind::ZONE_MOVE, B, Zn::LIBRARY, Zn::LIBRARY, Stream::card(NO_CARD, 0, 0, Stream::top(0)), 1);
    s.library_place(B, 1, Stream::bottom(0, 2));
    s.frame();
    no_errors(r, k);
    const KRows rows = k.rows();
    const int bottom_run = library_pool(rows, B, 12);
    expect(r, bottom_run >= 0 && rows.pools[static_cast<size_t>(bottom_run)].size == 2, "the bottom two are one run");
    expect(r, k.player(B).library_size() == 14, "library of 14");
    return r;
}

// Sylvan Library: the put-back may only pick cards drawn this turn, so a card known in hand from
// before stays known.
static Result test_drawn_this_turn() {
    Result r;
    Knowledge k;
    begin(k, {{FILL, 15}}, {{BOLT, 2}, {FILL, 13}});
    Stream s(k);
    s.game_start(15, 15);
    for (int i = 0; i < 7; ++i) s.draw(B, NO_CARD, 0, true, 0);
    std::vector<EventCard> hand = {Stream::card(BOLT, 0, 50)};
    for (int i = 0; i < 6; ++i) hand.push_back(Stream::card(FILL, 0, 51 + static_cast<uint32_t>(i)));
    s.shown(EventKind::REVEAL_START, B, Zn::HAND, hand, 0);
    std::vector<EventCard> end;
    for (const EventCard &c : hand) end.push_back(Stream::card(c.card, c.handle_after));
    s.shown(EventKind::REVEAL_END, B, Zn::HAND, end, 0);
    for (int i = 0; i < 3; ++i) s.draw(B, NO_CARD, 0, false, 2);
    for (int i = 0; i < 2; ++i) {
        EventCard c = Stream::card(NO_CARD, 0, 0, LibPos{}, Stream::top(0));
        c.hand_subset = static_cast<uint8_t>(rmstate::HandSubset::DRAWN_THIS_TURN);
        s.move(EventKind::ZONE_MOVE, B, Zn::HAND, Zn::LIBRARY, c, 0, 2);
    }
    s.frame();
    no_errors(r, k);
    const KRows rows = k.rows();
    expect(r, known_in_hand(rows, BOLT) == 1, "the revealed Lightning Bolt is still known in hand");
    expect(r, known_in_hand(rows, FILL) == 6, "the six revealed fillers are still known in hand");
    return r;
}

// Miracle: the reveal names the card the turn's first draw brought in, so that drawn unit is the
// miracle card, and the library holds no other copy.
static Result test_draw_link() {
    Result r;
    Knowledge k;
    begin(k, {{FILL, 15}}, {{MIRACLE, 1}, {FILL, 14}});
    Stream s(k);
    s.game_start(15, 15);
    for (int i = 0; i < 7; ++i) s.draw(B, NO_CARD, 0, true, 0);
    const uint32_t seq = s.draw(B, NO_CARD, 0, false, 2);
    EventCard c = Stream::card(MIRACLE, 0, 60);
    c.draw_seq = seq;
    c.draw_idx = 0;
    s.shown(EventKind::REVEAL_START, B, Zn::HAND, {c}, 2);
    s.frame();
    no_errors(r, k);
    const KRows rows = k.rows();
    expect(r, known_in_hand(rows, MIRACLE) == 1, "the miracle card is known in hand");
    expect(r, positions_are(rows, B, 0, 7, FILL), "no miracle card left in the library");
    return r;
}

// The viewer's own Ponder: look at the top three, put them back in a chosen order; every position
// is known exactly.
static Result test_own_rearrange() {
    Result r;
    Knowledge k;
    begin(k, {{PONDER_A, 1}, {PONDER_B, 1}, {PONDER_C, 1}, {FILL, 12}}, {{FILL, 15}});
    Stream s(k);
    s.game_start(15, 15);
    s.shuffle(A);
    s.shown(EventKind::LOOK, A, Zn::LIBRARY,
        {Stream::card(PONDER_A, 0, 1, Stream::top(0)), Stream::card(PONDER_B, 0, 2, Stream::top(1)),
            Stream::card(PONDER_C, 0, 3, Stream::top(2))});
    s.move(EventKind::ZONE_MOVE, A, Zn::LIBRARY, Zn::LIBRARY,
        Stream::card(PONDER_B, 2, 0, Stream::top(1), Stream::top(0)));
    s.move(EventKind::ZONE_MOVE, A, Zn::LIBRARY, Zn::LIBRARY,
        Stream::card(PONDER_A, 1, 0, Stream::top(1), Stream::top(0)));
    s.move(EventKind::ZONE_MOVE, A, Zn::LIBRARY, Zn::LIBRARY,
        Stream::card(PONDER_C, 3, 0, Stream::top(2), Stream::top(0)));
    s.frame();
    no_errors(r, k);
    const KRows rows = k.rows();
    const int want[3] = {PONDER_C, PONDER_A, PONDER_B};
    for (int pos = 0; pos < 3; ++pos) {
        const knowledge::KPositionRow *p = position(rows, A, pos);
        expect(r, p && p->card == want[pos], "own top " + std::to_string(pos));
    }
    expect(r, positions_are(rows, A, 3, 15, FILL), "the rest is exactly the 12 fillers");
    return r;
}

// A search finds a card, the library is shuffled, then the found card is located at a position and
// leaves: the rest is one run again (the positions around the found card tell nothing).
static Result test_search_rejoin() {
    Result r;
    Knowledge k;
    begin(k, {{BOLT, 1}, {FILL, 14}}, {{FILL, 15}});
    Stream s(k);
    s.game_start(15, 15);
    s.shuffle(A);
    s.shown(EventKind::LOOK, A, Zn::LIBRARY, {Stream::card(BOLT, 0, 5, Stream::top(6))});
    s.move(EventKind::ZONE_MOVE, A, Zn::LIBRARY, Zn::GRAVEYARD, Stream::card(BOLT, 5, 6, Stream::top(6)));
    s.frame();
    no_errors(r, k);
    expect(r, k.player(A).library.size() == 1, "one run after the found card leaves");
    return r;
}

// London mulligan: the hand goes back, the library is shuffled, seven are drawn and one is put on
// the bottom; the sizes and the snapshot agree.
static Result test_mulligan() {
    Result r;
    Knowledge k;
    begin(k, {{FILL, 15}}, {{FOW, 2}, {FILL, 13}});
    Stream s(k);
    s.game_start(15, 15);
    for (int i = 0; i < 7; ++i) s.draw(B, NO_CARD, 0, true, 0);
    for (int i = 0; i < 7; ++i)
        s.move(EventKind::ZONE_MOVE, B, Zn::HAND, Zn::LIBRARY, Stream::card(NO_CARD, 0, 0, LibPos{}, Stream::top(0)), 0,
            0);
    s.shuffle(B);
    for (int i = 0; i < 7; ++i) s.draw(B, NO_CARD, 0, true, 0);
    s.move(
        EventKind::ZONE_MOVE, B, Zn::HAND, Zn::LIBRARY, Stream::card(NO_CARD, 0, 0, LibPos{}, Stream::bottom(0)), 0, 0);
    Snapshot snap;
    snap.players.push_back({A, 0, 15, 0, 0});
    snap.players.push_back({B, 6, 9, 0, 0});
    // The frame's events and the snapshot check together.
    s.frame();
    no_errors(r, k);
    expect(r, k.player(B).library_size() == 9, "library of 9");
    size_t hand = 0;
    for (knowledge::NodeId n : k.player(B).hand) hand += k.player(B).net.node(n).size;
    expect(r, hand == 6, "hand of 6");
    k.apply_frame(snap);
    no_errors(r, k);
    return r;
}

// Under a closed decklist the opponent's root is free; cards seen in game 1 become the game-2
// root's lower bounds, and copies seen per match persist.
static Result test_closed_match_memory() {
    Result r;
    Knowledge k;
    begin(k, {{FILL, 15}}, {}, true);
    Stream s(k);
    s.game_start(15, 15);
    for (int i = 0; i < 7; ++i) s.draw(B, NO_CARD, 0, true, 0);
    s.move(EventKind::CAST_FROM_HIDDEN, B, Zn::HAND, Zn::STACK, Stream::card(FOW, 0, 70));
    s.move(EventKind::CAST_FROM_HIDDEN, B, Zn::HAND, Zn::STACK, Stream::card(FOW, 0, 71));
    s.game_end();
    s.frame();
    s.game_start(15, 15, 1);
    s.frame();
    no_errors(r, k);
    const KRows rows = k.rows();
    expect(r, seen(rows, B, FOW, true) == 2 && seen(rows, B, FOW, false) == 0,
        "two Force of Will seen this match, none this game");
    const knowledge::PlayerModel &m = k.player(B);
    expect(r, m.net.roots().size() == 1 && m.net.roots()[0].kind == knowledge::RootKind::LOWER,
        "the game-2 root has lower bounds");
    return r;
}

// serialize → deserialize → serialize is the identity, and the restored instance gives the same
// rows and carries on identically.
static Result test_serialize() {
    Result r;
    Knowledge k;
    begin(k, {{FILL, 15}}, {{FOW, 1}, {DAZE, 1}, {XCARD, 1}, {FILL, 12}});
    Stream s(k);
    s.game_start(15, 15);
    for (int i = 0; i < 3; ++i) s.draw(B, NO_CARD, 0, true, 0);
    s.shown(EventKind::REVEAL_START, B, Zn::HAND,
        {Stream::card(FOW, 0, 11), Stream::card(DAZE, 0, 12), Stream::card(XCARD, 0, 13)});
    s.frame();
    std::string blob;
    k.serialize(blob);
    Knowledge copy;
    expect(r, copy.deserialize(blob), "the blob deserializes");
    std::string again;
    copy.serialize(again);
    expect(r, again == blob, "serialize after deserialize is byte-identical");
    knowledge::Codes codes;
    const auto bytes = [&codes](const Knowledge &kk) {
        rmstate::FrameWriter w(rmstate::FrameKind::STATE, rmstate::Viewer::A, 0);
        knowledge::write_k_tables(w, kk.rows(), codes, A);
        return w.finish();
    };
    expect(r, bytes(k) == bytes(copy), "the restored rows are identical");
    for (Knowledge *kk : {&k, &copy}) {
        Stream t(*kk, s.seq());
        t.move(EventKind::CAST_FROM_HIDDEN, B, Zn::HAND, Zn::STACK, Stream::card(XCARD, 13, 20));
        t.frame();
    }
    expect(r, bytes(k) == bytes(copy), "both continue identically");
    no_errors(r, k);
    no_errors(r, copy);
    return r;
}

// Forty rounds of the opponent drawing, revealing, putting a card back and shuffling: compaction
// keeps the network a handful of nodes (it would otherwise grow by every operation), and the
// knowledge stays exact (the shuffled-away reveals are still counted).
static Result test_compaction_bounded() {
    Result r;
    Knowledge k;
    begin(k, {{FILL, 15}}, {{FOW, 2}, {DAZE, 2}, {FILL, 11}});
    Stream s(k);
    s.game_start(15, 15);
    for (int i = 0; i < 3; ++i) s.draw(B, NO_CARD, 0, true, 0);
    s.frame();
    size_t most = 0;
    for (int round = 1; round <= 40; ++round) {
        const uint16_t turn = static_cast<uint16_t>(round);
        s.draw(B, NO_CARD, 0, false, turn);
        s.shown(EventKind::REVEAL_START, B, Zn::LIBRARY, {Stream::card(FILL, 0, 500, Stream::top(0))}, turn);
        s.shown(EventKind::REVEAL_END, B, Zn::LIBRARY, {Stream::card(FILL, 500)}, turn);
        s.move(EventKind::ZONE_MOVE, B, Zn::HAND, Zn::LIBRARY, Stream::card(NO_CARD, 0, 0, LibPos{}, Stream::top(0)), 0,
            turn);
        s.shuffle(B);
        s.frame();
        most = std::max(most, k.player(B).net.node_count());
    }
    no_errors(r, k);
    expect(r, most < 20, "the network stays small (" + std::to_string(most) + " nodes at most)");
    expect(r, k.rows().meta.exact, "the solver is exact");
    return r;
}

static const knowledge::KTemplateRow *template_row(const KRows &r, int player, int list) {
    for (const knowledge::KTemplateRow &t : r.templates)
        if (t.player == player && t.list == list) return &t;
    return nullptr;
}

static const knowledge::KArchetypeRow *archetype_row(const KRows &r, int player, int archetype) {
    for (const knowledge::KArchetypeRow &a : r.archetypes)
        if (a.player == player && a.archetype == archetype) return &a;
    return nullptr;
}

// Reference lists (knowledge_store §10). Four lists: 0 (label 0) 2 Force of Will in the main deck and
// 3 Daze in the sideboard; 1 (label 1) the reverse; 2 (label 0) one Force of Will and a Bolt in the
// main deck; 3 (label 2) a 2-card sideboard. Viewer A registered list 2 under another name; B
// registered list 0, hidden (closed decklist). The game starts with B's 12-card library and 3-card
// sideboard (list 3 cannot be B's). A Force of Will cast from B's hand in game 1 excludes list 1 (its
// copies are in the sideboard), a second excludes list 2 (one copy), and list 1 stays excluded in
// game 2, where the 75 alone would allow it. With B's list open and named nothing in the metagame,
// it is classified by the cards in common. The state survives serialisation.
static Result test_reference_lists() {
    Result r;
    knowledge::Metagame meta;
    meta.present = true;
    meta.archetypes = {"t0", "t1", "t2", "unknown"};
    const auto list = [](const char *name, int arch, std::vector<std::pair<int, int>> main,
                          std::vector<std::pair<int, int>> side) {
        knowledge::MetaList l;
        l.name = name;
        l.archetype = arch;
        l.main = main;
        l.side = side;
        return l;
    };
    meta.lists = {list("zero", 0, {{FOW, 2}, {FILL, 10}}, {{DAZE, 3}}),
        list("one", 1, {{DAZE, 2}, {FILL, 10}}, {{FOW, 3}}),
        list("two", 0, {{FOW, 1}, {FILL, 10}, {BOLT, 1}}, {{DAZE, 3}}), list("three", 2, {{FILL, 12}}, {{FILL, 2}})};
    const auto start = [&meta](Knowledge &k, bool closed) {
        knowledge::Decklists lists;
        lists.present[A] = true;
        lists.name[A] = "mine";
        for (const auto &c : meta.lists[2].main) lists.cards.push_back({A, c.first, c.second, 0});
        for (const auto &c : meta.lists[2].side) lists.cards.push_back({A, c.first, 0, c.second});
        if (!closed) {
            lists.present[B] = true;
            lists.name[B] = "theirs";
            lists.cards.push_back({B, FOW, 1, 0});
            lists.cards.push_back({B, DAZE, 2, 0});
            lists.cards.push_back({B, FILL, 9, 2});
        }
        knowledge::Codes codes;
        k.begin_match(A, codes, lists);
        k.begin_metagame(meta);
    };
    const auto consistent = [](const KRows &rows, int p, int l) {
        const knowledge::KTemplateRow *t = template_row(rows, p, l);
        return t && t->consistent;
    };
    Knowledge k;
    start(k, true);
    Stream s(k);
    s.game_start(12, 12, 0, 0, 3);
    for (int i = 0; i < 7; ++i) s.draw(B, NO_CARD, 0, true, 0);
    s.frame();
    KRows rows = k.rows();
    expect(r, rows.templates.size() == 8 && rows.archetypes.size() == 8, "a row per player and list, and label");
    expect(r, consistent(rows, B, 0) && consistent(rows, B, 1) && consistent(rows, B, 2), "lists 0-2 are possible");
    expect(r, !consistent(rows, B, 3) && template_row(rows, B, 3)->exact, "list 3's sideboard is too small");
    expect(r,
        archetype_row(rows, B, 0)->possible && archetype_row(rows, B, 1)->possible &&
            !archetype_row(rows, B, 0)->exact && !archetype_row(rows, B, 2)->possible,
        "labels 0 and 1 are possible, neither exact");
    expect(r, consistent(rows, A, 2) && template_row(rows, A, 2)->registered && !consistent(rows, A, 0),
        "the viewer's own list is list 2 (by its cards: no list has its name)");
    for (const knowledge::KTemplateRow &t : rows.templates)
        expect(r, t.player == A || !t.registered, "a hidden list is classified as nothing");
    s.move(EventKind::CAST_FROM_HIDDEN, B, Zn::HAND, Zn::STACK, Stream::card(FOW, 0, 70));
    s.frame();
    rows = k.rows();
    expect(r, consistent(rows, B, 0) && !consistent(rows, B, 1) && consistent(rows, B, 2),
        "a Force of Will from the main deck excludes list 1");
    expect(r, archetype_row(rows, B, 0)->exact && !archetype_row(rows, B, 1)->possible, "label 0 is exact");
    expect(r, template_row(rows, B, 1)->exact, "list 1's exclusion is exact");

    std::string blob;
    k.serialize(blob);
    Knowledge copy;
    expect(r, copy.deserialize(blob), "the state deserializes");
    knowledge::Codes codes;
    const auto bytes = [&codes](const Knowledge &kk) {
        rmstate::FrameWriter w(rmstate::FrameKind::STATE, rmstate::Viewer::A, 0);
        knowledge::write_k_tables(w, kk.rows(), codes, A);
        return w.finish();
    };
    expect(r, bytes(k) == bytes(copy), "the restored rows are identical");

    s.move(EventKind::CAST_FROM_HIDDEN, B, Zn::HAND, Zn::STACK, Stream::card(FOW, 0, 71));
    s.game_end();
    s.frame();
    rows = k.rows();
    expect(r, consistent(rows, B, 0) && !consistent(rows, B, 2), "a second Force of Will excludes list 2");
    s.game_start(12, 12, 1, 0, 3);
    for (int i = 0; i < 7; ++i) s.draw(B, NO_CARD, 0, true, 0);
    s.frame();
    rows = k.rows();
    expect(r, consistent(rows, B, 0) && !consistent(rows, B, 1) && !consistent(rows, B, 2),
        "game 2: lists 1 and 2 stay excluded");
    no_errors(r, k);

    Knowledge open;
    start(open, false);
    Stream o(open);
    o.game_start(12, 12, 0, 0, 2);
    o.frame();
    rows = open.rows();
    expect(r, template_row(rows, B, 1)->registered, "the open list is classified as list 1 (the most cards in common)");
    for (int l = 0; l < 4; ++l)
        expect(r, !consistent(rows, B, l) && template_row(rows, B, l)->exact, "no reference list is the open list");
    no_errors(r, open);
    return r;
}

int run_selftest() {
    const std::vector<std::pair<const char *, std::function<Result()>>> tests = {
        {"brainstorm_put_back", test_brainstorm_put_back},
        {"cut_accounting", test_cut_accounting},
        {"scry_range", test_scry_range},
        {"batch", test_batch},
        {"drawn_this_turn", test_drawn_this_turn},
        {"draw_link", test_draw_link},
        {"own_rearrange", test_own_rearrange},
        {"search_rejoin", test_search_rejoin},
        {"mulligan", test_mulligan},
        {"closed_match_memory", test_closed_match_memory},
        {"serialize", test_serialize},
        {"compaction_bounded", test_compaction_bounded},
        {"reference_lists", test_reference_lists},
    };
    int failed = 0;
    for (const auto &t : tests) {
        const Result r = t.second();
        if (r.failures == 0) {
            std::printf("  OK   %s\n", t.first);
        } else {
            ++failed;
            std::printf("ERROR: rmknow selftest %s: %d failure(s), first: %s\n", t.first, r.failures, r.first.c_str());
        }
    }
    std::printf("RMKNOW_SELFTEST: tests=%zu failed=%d\n", tests.size(), failed);
    return failed == 0 ? 0 : 1;
}
