// Exactness tests of the knowledge solver and compaction (docs/knowledge_store.md §7) against the
// independent enumerator (enumerate.h), on small decks (at most 15 cards):
//   - synthetic networks: random histories of library and hand operations on one player's model
//     (PlayerModel), dealt from a hidden true deck so every pin is consistent. At checkpoints the
//     raw network's worlds are enumerated and compared with the solver on the raw network, the
//     compacted network's worlds (projected onto the same units) and the solver on the compacted
//     network;
//   - random event streams: a simulated two-player game (the opponent's draws, hand reveals,
//     put-backs, casts, top reveals, scries, mills and shuffles, the viewer's own draws and looks)
//     fed to the sidecar as frames, with compaction on and off. After every frame each player's
//     reported pools are compared with the enumeration of that sidecar's own network, and with the
//     simulated truth;
//   - sculpted cases: the §4.2 put-back example, a put-back from a partly known hand, a game 2
//     under a closed decklist (lower-bound root) and with open decklists (the registered 75 split).
// A check passes when the solver's bounds contain the enumeration's (soundness) and, when the
// solver reports exact, equal them on every card. `rmknow --exactness N` runs N synthetic histories,
// N compacted streams and every fourth seed's uncompacted stream (deterministic seeds) plus the
// sculpted cases.

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "../../src/knowledge/knowledge.h"
#include "../../src/knowledge/solver.h"
#include "../../src/knowledge/templates.h"
#include "enumerate.h"
#include "selftest.h"
#include "test_stream.h"

using knowledge::BoundStore;
using knowledge::LibPos;
using knowledge::Loc;
using knowledge::NO_NODE;
using knowledge::NodeId;
using knowledge::PlayerModel;
using knowledge::RootKind;
using knowledge::Solution;
using knowledge::Solver;
using knowledge::Zn;
using rmknow_test::EnumResult;
using rmknow_test::Stream;
using rmstate::EventKind;

namespace {

constexpr uint64_t ENUM_CAP = 200000;  // enumeration states at a time
constexpr uint64_t SOLVER_STEPS = 1000000;
constexpr int A = 0, B = 1;

// A small deterministic generator (the same sequence on every platform).
class Rng {
    public:
        explicit Rng(uint64_t seed) : s_(seed * 0x9E3779B97F4A7C15ull + 1) {}
        uint32_t next() {
            s_ ^= s_ << 13;
            s_ ^= s_ >> 7;
            s_ ^= s_ << 17;
            return static_cast<uint32_t>(s_ >> 11);
        }
        int below(int n) { return n <= 0 ? 0 : static_cast<int>(next() % static_cast<uint32_t>(n)); }
        bool chance(int percent) { return below(100) < percent; }

    private:
        uint64_t s_;
};

struct Tally {
        int checks = 0;       // solver results compared with the enumeration
        int exact = 0;        // of which the solver reported exact
        int compactions = 0;  // compacted networks whose worlds were compared with the raw network's
        int skipped = 0;      // enumeration past its cap
        int failures = 0;
        double seconds = 0;
        std::string first;
        void fail(const std::string &what) {
            if (failures++ == 0) first = what;
        }
};

}  // namespace

static Solution solve_fresh(const knowledge::Network &net, const knowledge::Universe &uni,
    const std::vector<NodeId> &live, const std::vector<std::vector<NodeId>> &groups);
static std::vector<std::vector<int>> as_int(const std::vector<std::vector<NodeId>> &groups);
static void compare(Tally &t, const Solution &sol, const EnumResult &e, size_t K, const std::string &where);
static void compare_enum(Tally &t, const EnumResult &a, const EnumResult &b, const std::string &where);
static void synthetic_check(Tally &t, const PlayerModel &m, const knowledge::Universe &uni, const std::vector<int> &lib,
    const std::map<NodeId, std::vector<int>> &held, const std::string &where);
static void synthetic_case(Tally &t, uint64_t seed);
static void stream_case(Tally &t, uint64_t seed, bool compaction);
static void check_knowledge(Tally &t, const knowledge::Knowledge &k, const std::vector<int> *truth_lib,
    const std::vector<int> *truth_hand, int p, const std::string &where);
static void sculpted(Tally &t);
static void template_case(Tally &t, uint64_t seed);

// The solver on `net` with a store propagated from scratch.
static Solution solve_fresh(const knowledge::Network &net, const knowledge::Universe &uni,
    const std::vector<NodeId> &live, const std::vector<std::vector<NodeId>> &groups) {
    BoundStore store(true);
    store.universe = uni;
    store.propagate(net, live, 4000000);
    Solver solver;
    return solver.solve(net, store, groups, SOLVER_STEPS);
}

static std::vector<std::vector<int>> as_int(const std::vector<std::vector<NodeId>> &groups) {
    std::vector<std::vector<int>> out;
    for (const auto &g : groups) out.emplace_back(g.begin(), g.end());
    return out;
}

// The solver's bounds contain the enumeration's on every card (OTHER, the last class, is never
// reported) and equal them when the solver says exact.
static void compare(Tally &t, const Solution &sol, const EnumResult &e, size_t K, const std::string &where) {
    ++t.checks;
    if (sol.exact) ++t.exact;
    if (e.worlds == 0) {
        t.fail(where + ": the enumeration finds no world");
        return;
    }
    if (sol.contradiction) {
        t.fail(where + ": the solver finds no world");
        return;
    }
    for (size_t g = 0; g < e.lo.size(); ++g)
        for (size_t k = 0; k + 1 < K; ++k) {
            const int lo = sol.groups[g].lo[k], hi = sol.groups[g].hi[k];
            const int elo = e.lo[g][k], ehi = e.hi[g][k];
            if (lo > elo || hi < ehi)
                t.fail(where + ": group " + std::to_string(g) + " class " + std::to_string(k) + " reported [" +
                       std::to_string(lo) + "," + std::to_string(hi) + "], worlds hold [" + std::to_string(elo) + "," +
                       std::to_string(ehi) + "] (unsound)");
            else if (sol.exact && (lo != elo || hi != ehi))
                t.fail(where + ": group " + std::to_string(g) + " class " + std::to_string(k) + " reported [" +
                       std::to_string(lo) + "," + std::to_string(hi) + "] as exact, worlds hold [" +
                       std::to_string(elo) + "," + std::to_string(ehi) + "]");
        }
}

// Two enumerations of the same groups (the raw and the compacted network) agree on every class.
static void compare_enum(Tally &t, const EnumResult &a, const EnumResult &b, const std::string &where) {
    ++t.compactions;
    if (a.lo != b.lo || a.hi != b.hi || (a.worlds == 0) != (b.worlds == 0))
        t.fail(where + ": the compacted network's worlds project differently");
}

// ── Synthetic networks ────────────────────────────────────────────────────────────────────────

// One checkpoint of a synthetic history. The groups of the raw model (each library run, the blocks
// of runs from the top, the hand, each hand node, each public unit) are enumerated, checked against
// the true deal and compared with the solver on the raw network. The model is then compacted; the
// compacted model's groups (each run, the blocks from the top, the hand, each public unit) and the
// raw units at the same library positions and zones are enumerated and compared, and the solver
// on the compacted network is compared with the raw enumeration.
static void synthetic_check(Tally &t, const PlayerModel &m, const knowledge::Universe &uni, const std::vector<int> &lib,
    const std::map<NodeId, std::vector<int>> &held, const std::string &where) {
    const std::vector<NodeId> live = m.live();
    // Groups over the raw model.
    std::vector<std::vector<NodeId>> groups;
    for (NodeId n : m.library) groups.push_back({n});
    for (size_t j = 1; j + 1 < m.library.size(); ++j) groups.emplace_back(m.library.begin(), m.library.begin() + j + 1);
    if (!m.hand.empty()) groups.push_back(m.hand);
    for (NodeId n : m.hand) groups.push_back({n});
    for (NodeId n : m.publics) groups.push_back({n});
    const rmknow_test::EnumSpec spec = rmknow_test::spec_of(m.net, uni);
    const EnumResult raw = rmknow_test::enumerate_worlds(spec, as_int(groups), ENUM_CAP);
    if (!raw.complete) {
        ++t.skipped;
        return;
    }
    // The true deal is a world: its counts lie within the enumeration.
    {
        size_t at = 0;
        std::vector<std::vector<int>> truth;
        for (NodeId n : m.library) {
            std::vector<int> c(uni.classes(), 0);
            for (size_t i = 0; i < m.net.node(n).size; ++i) ++c[uni.index_of(lib[at + i])];
            at += m.net.node(n).size;
            truth.push_back(c);
        }
        for (size_t g = 0; g < m.library.size(); ++g)
            for (size_t k = 0; k < uni.classes(); ++k)
                if (truth[g][k] < raw.lo[g][k] || truth[g][k] > raw.hi[g][k])
                    t.fail(where + ": the true deal is not among the enumerated worlds");
        for (NodeId n : m.hand) {
            const auto it = held.find(n);
            if (it == held.end()) t.fail(where + ": a hand node without a true content");
        }
    }
    compare(t, solve_fresh(m.net, uni, live, groups), raw, uni.classes(), where + " raw");

    // Compacted: the same units, grouped by library position (compaction may merge runs) and by
    // zone.
    PlayerModel c = m;
    std::vector<NodeId> remap;
    c.compact({}, &remap);
    std::vector<std::vector<NodeId>> cg, rg;  // compacted groups, the raw groups of the same units
    std::vector<size_t> raw_start;
    size_t at = 0;
    for (NodeId n : m.library) {
        raw_start.push_back(at);
        at += m.net.node(n).size;
    }
    const auto raw_runs = [&](size_t a, size_t b) {
        std::vector<NodeId> out;
        for (size_t i = 0; i < m.library.size(); ++i)
            if (raw_start[i] >= a && raw_start[i] < b) out.push_back(m.library[i]);
        return out;
    };
    at = 0;
    std::vector<size_t> c_start;
    for (NodeId n : c.library) {
        c_start.push_back(at);
        cg.push_back({n});
        rg.push_back(raw_runs(at, at + c.net.node(n).size));
        at += c.net.node(n).size;
    }
    for (size_t j = 1; j + 1 < c.library.size(); ++j) {
        cg.emplace_back(c.library.begin(), c.library.begin() + j + 1);
        rg.push_back(raw_runs(0, c_start[j + 1]));
    }
    if (!c.hand.empty()) {
        cg.push_back(c.hand);
        rg.push_back(m.hand);
    }
    for (size_t i = 0; i < m.publics.size(); ++i) {
        cg.push_back({c.publics[i]});
        rg.push_back({m.publics[i]});
    }
    const EnumResult raw_by_units = rmknow_test::enumerate_worlds(spec, as_int(rg), ENUM_CAP);
    const EnumResult comp = rmknow_test::enumerate_worlds(rmknow_test::spec_of(c.net, uni), as_int(cg), ENUM_CAP);
    if (!raw_by_units.complete || !comp.complete) {
        ++t.skipped;
        return;
    }
    compare_enum(t, raw_by_units, comp, where + " compaction");
    compare(t, solve_fresh(c.net, uni, c.live(), cg), raw_by_units, uni.classes(), where + " compacted");
}

// One random history on one player's model: a deck of 8-14 cards of 2-4 kinds under an EXACT root,
// or of 6-9 cards of 2-3 kinds under a LOWER or FREE root, then 10-20 operations, checked after
// every few.
static void synthetic_case(Tally &t, uint64_t seed) {
    Rng rng(seed);
    const int kind_roll = rng.below(10);
    const RootKind kind = kind_roll < 6 ? RootKind::EXACT : kind_roll < 8 ? RootKind::LOWER : RootKind::FREE;
    // A root that does not fix the contents has many more worlds: smaller decks.
    const int kinds = 2 + rng.below(kind == RootKind::EXACT ? 3 : 2);
    const int size = kind == RootKind::EXACT ? 8 + rng.below(7) : 6 + rng.below(4);
    std::vector<int> lib;
    for (int i = 0; i < size; ++i) lib.push_back(1 + (i < kinds ? i : rng.below(kinds)));
    for (size_t i = lib.size(); i > 1; --i)
        std::swap(lib[i - 1], lib[static_cast<size_t>(rng.below(static_cast<int>(i)))]);
    std::map<int, int> counts;
    for (int c : lib) ++counts[c];
    knowledge::CardCounts rc;
    for (const auto &c : counts) {
        if (kind == RootKind::EXACT) rc.emplace_back(c.first, c.second);
        if (kind == RootKind::LOWER && c.second > 1) rc.emplace_back(c.first, c.second - 1);
    }
    knowledge::Universe uni;
    for (const auto &c : counts)
        if (kind != RootKind::FREE || rng.chance(70)) uni.add(c.first);
    PlayerModel m;
    const NodeId root = m.net.add_root(static_cast<uint16_t>(size), Loc::LIBRARY, kind, rc);
    m.library.push_back(root);
    std::map<NodeId, std::vector<int>> held;  // true cards of each live non-library node

    const auto pin = [&](NodeId n, int card) {
        uni.add(card);
        m.net.pin(n, card);
    };
    const auto take_hand_unit = [&](size_t at) {  // a random unit of hand node `at`, split off
        const NodeId h = m.hand[at];
        std::vector<int> cards = held[h];
        held.erase(h);
        const size_t pick = static_cast<size_t>(rng.below(static_cast<int>(cards.size())));
        const int card = cards[pick];
        cards.erase(cards.begin() + static_cast<std::ptrdiff_t>(pick));
        const NodeId u = m.split_unit(m.hand, at);
        if (!cards.empty()) held[m.hand[at]] = cards;
        return std::make_pair(u, card);
    };
    const int ops = 10 + rng.below(11);
    for (int op = 0; op < ops; ++op) {
        const int r = rng.below(100);
        const size_t lsize = m.library_size();
        if (r < 25 && lsize > 0) {  // draw
            const NodeId n = m.take_library(0);
            m.net.node(n).loc = Loc::HAND;
            m.hand.push_back(n);
            held[n] = {lib[0]};
            lib.erase(lib.begin());
        } else if (r < 33 && lsize > 0) {  // reveal a library position
            const size_t at = static_cast<size_t>(rng.below(static_cast<int>(std::min<size_t>(lsize, 3))));
            pin(m.isolate_library(at), lib[at]);
        } else if (r < 41 && !m.hand.empty()) {  // reveal a hand card
            const size_t at = static_cast<size_t>(rng.below(static_cast<int>(m.hand.size())));
            if (m.net.node(m.hand[at]).pin >= 0) continue;
            const auto u = take_hand_unit(at);
            m.net.node(u.first).loc = Loc::HAND;
            m.hand.push_back(u.first);
            held[u.first] = {u.second};
            pin(u.first, u.second);
        } else if (r < 53 && !m.hand.empty()) {  // a card leaves the hand: cast (known) or put back (unknown)
            // The viewer does not know which hand node it came from: merge a random set holding it.
            const size_t at = static_cast<size_t>(rng.below(static_cast<int>(m.hand.size())));
            std::vector<NodeId> parts = {m.hand[at]};
            for (NodeId n : m.hand)
                if (n != m.hand[at] && m.net.node(n).pin < 0 && rng.chance(60)) parts.push_back(n);
            std::vector<int> cards;
            for (NodeId n : parts) {
                cards.insert(cards.end(), held[n].begin(), held[n].end());
                held.erase(n);
            }
            std::vector<NodeId> rest;
            for (NodeId n : m.hand)
                if (std::find(parts.begin(), parts.end(), n) == parts.end()) rest.push_back(n);
            const NodeId merged = m.net.merge(parts, Loc::HAND);
            rest.push_back(merged);
            m.hand = rest;
            held[merged] = cards;
            const auto u = take_hand_unit(m.hand.size() - 1);
            if (r < 47) {
                pin(u.first, u.second);
                m.net.node(u.first).loc = Loc::PUBLIC;
                m.publics.push_back(u.first);
            } else {
                const bool bottom = rng.chance(30);
                m.insert_library(bottom ? m.library_size() : 0, u.first);
                lib.insert(bottom ? lib.end() : lib.begin(), u.second);
            }
        } else if (r < 60 && lsize > 1) {  // shuffle
            m.merge_library();
            for (size_t i = lib.size(); i > 1; --i)
                std::swap(lib[i - 1], lib[static_cast<size_t>(rng.below(static_cast<int>(i)))]);
        } else if (r < 68 && lsize > 1) {  // a range whose order becomes unknown
            const size_t span = std::min<size_t>(lsize, 2 + static_cast<size_t>(rng.below(2)));
            m.merge_library_range(0, span);
        } else if (r < 76 && m.hand.size() > 1) {  // the hand's sub-pools become one
            std::vector<int> cards;
            for (NodeId n : m.hand) {
                cards.insert(cards.end(), held[n].begin(), held[n].end());
                held.erase(n);
            }
            const NodeId merged = m.net.merge(m.hand, Loc::HAND);
            m.hand = {merged};
            held[merged] = cards;
        } else if (r < 86 && lsize > 0) {  // a library card leaves to a public zone, known
            const size_t at = static_cast<size_t>(rng.below(static_cast<int>(std::min<size_t>(lsize, 4))));
            const NodeId n = m.take_library(at);
            pin(n, lib[at]);
            lib.erase(lib.begin() + static_cast<std::ptrdiff_t>(at));
            m.net.node(n).loc = Loc::PUBLIC;
            m.publics.push_back(n);
        } else if (lsize > 0) {  // the top card goes to the bottom, unseen
            const NodeId n = m.take_library(0);
            m.insert_library(m.library_size(), n);
            lib.push_back(lib.front());
            lib.erase(lib.begin());
        }
        if (op % 3 == 2 || op + 1 == ops)
            synthetic_check(t, m, uni, lib, held, "synthetic " + std::to_string(seed) + " op " + std::to_string(op));
    }
}

// ── Random event streams through the sidecar ──────────────────────────────────────────────────

// Player `p`'s pools in sidecar `k` against the enumeration of its network and against the true
// library (top first) and hand, when given.
static void check_knowledge(Tally &t, const knowledge::Knowledge &k, const std::vector<int> *truth_lib,
    const std::vector<int> *truth_hand, int p, const std::string &where) {
    const PlayerModel &m = k.player(p);
    const std::vector<std::vector<NodeId>> groups = k.pool_groups(p);
    const Solution &sol = k.solution(p);
    if (sol.groups.size() != groups.size()) {
        t.fail(where + ": the solution does not match the pools");
        return;
    }
    const EnumResult e =
        rmknow_test::enumerate_worlds(rmknow_test::spec_of(m.net, m.full.universe), as_int(groups), ENUM_CAP);
    if (!e.complete) {
        ++t.skipped;
        return;
    }
    compare(t, sol, e, m.full.universe.classes(), where);
    if (!truth_lib) return;
    // The truth lies within each library pool's and the hand's bounds.
    std::map<NodeId, size_t> start;
    size_t at = 0;
    for (NodeId n : m.library) {
        start[n] = at;
        at += m.net.node(n).size;
    }
    for (size_t g = 0; g < groups.size(); ++g) {
        std::vector<int> c(m.full.universe.classes(), 0);
        bool library = true, hand = true;
        for (NodeId n : groups[g]) {
            library = library && start.count(n);
            hand = hand && std::find(m.hand.begin(), m.hand.end(), n) != m.hand.end();
        }
        if (library)
            for (NodeId n : groups[g])
                for (size_t i = 0; i < m.net.node(n).size; ++i)
                    ++c[m.full.universe.index_of((*truth_lib)[start[n] + i])];
        else if (hand && truth_hand && groups[g].size() == m.hand.size())
            for (int card : *truth_hand) ++c[m.full.universe.index_of(card)];
        else
            continue;
        for (size_t kk = 0; kk + 1 < c.size(); ++kk)
            if (c[kk] < sol.groups[g].lo[kk] || c[kk] > sol.groups[g].hi[kk])
                t.fail(where + ": the truth is outside pool " + std::to_string(g) + " class " + std::to_string(kk));
    }
}

// A simulated bo1 game of 10-12-card decks (2-4 kinds; uncompacted, whose raw networks enumerate
// slowly: 9-10 cards of 2-3 kinds and a 3-card opening hand) between viewer A and opponent B, as
// events.
static void stream_case(Tally &t, uint64_t seed, bool compaction) {
    Rng rng(seed);
    const int kinds = 2 + rng.below(compaction ? 3 : 2);
    const int size = compaction ? 10 + rng.below(3) : 9 + rng.below(2);
    const int opening = compaction ? 4 : 3;
    std::vector<int> deck_b, deck_a;
    for (int i = 0; i < size; ++i) deck_b.push_back(201 + (i < kinds ? i : rng.below(kinds)));
    for (int i = 0; i < size; ++i) deck_a.push_back(301 + (i < 2 ? i : rng.below(2)));
    knowledge::Knowledge k;
    k.set_compaction(compaction);
    knowledge::Decklists lists;
    lists.present[A] = lists.present[B] = true;
    std::map<int, int> ca, cb;
    for (int c : deck_a) ++ca[c];
    for (int c : deck_b) ++cb[c];
    for (const auto &c : ca) lists.cards.push_back({A, c.first, c.second, 0});
    for (const auto &c : cb) lists.cards.push_back({B, c.first, c.second, 0});
    knowledge::Codes codes;
    k.begin_match(A, codes, lists);
    const auto shuffle = [&rng](std::vector<int> &v) {
        for (size_t i = v.size(); i > 1; --i)
            std::swap(v[i - 1], v[static_cast<size_t>(rng.below(static_cast<int>(i)))]);
    };
    shuffle(deck_a);
    shuffle(deck_b);
    std::vector<int> lib_b = deck_b, lib_a = deck_a;
    // B's hand: (card, the viewer's handle while it has one).
    std::vector<std::pair<int, uint32_t>> hand_b, hand_a;
    uint32_t next_handle = 1000;
    Stream s(k);
    s.game_start(size, size);
    for (int i = 0; i < opening; ++i) {
        s.draw(B, knowledge::NO_CARD, 0, true, 0);
        hand_b.emplace_back(lib_b.front(), 0);
        lib_b.erase(lib_b.begin());
        const uint32_t h = next_handle++;
        s.draw(A, lib_a.front(), h, true, 0);
        hand_a.emplace_back(lib_a.front(), h);
        lib_a.erase(lib_a.begin());
    }
    s.frame();
    const std::string tag = "stream " + std::to_string(seed) + (compaction ? "" : " (uncompacted)");
    const auto check = [&](int step) {
        for (const std::string &e : k.take_errors()) t.fail(tag + ": " + e);
        std::vector<int> hb;
        for (const auto &c : hand_b) hb.push_back(c.first);
        check_knowledge(t, k, &lib_b, &hb, B, tag + " step " + std::to_string(step) + " B");
        check_knowledge(t, k, &lib_a, nullptr, A, tag + " step " + std::to_string(step) + " A");
    };
    check(0);
    const int steps = 8 + rng.below(8);
    for (int step = 1; step <= steps; ++step) {
        const uint16_t turn = static_cast<uint16_t>(step);
        const int r = rng.below(100);
        if (r < 18 && !lib_b.empty()) {  // B draws
            s.draw(B, knowledge::NO_CARD, 0, false, turn);
            hand_b.emplace_back(lib_b.front(), 0);
            lib_b.erase(lib_b.begin());
        } else if (r < 26 && !hand_b.empty()) {  // B's hand is revealed
            std::vector<knowledge::EventCard> shown, ended;
            for (auto &c : hand_b) {
                c.second = next_handle++;
                shown.push_back(Stream::card(c.first, 0, c.second));
                ended.push_back(Stream::card(c.first, c.second));
                c.second = 0;
            }
            s.shown(EventKind::REVEAL_START, B, Zn::HAND, shown, turn);
            s.shown(EventKind::REVEAL_END, B, Zn::HAND, ended, turn);
        } else if (r < 40 && !hand_b.empty()) {  // B puts a card from hand on top or bottom, unseen
            const size_t at = static_cast<size_t>(rng.below(static_cast<int>(hand_b.size())));
            const bool bottom = rng.chance(30);
            s.move(EventKind::ZONE_MOVE, B, Zn::HAND, Zn::LIBRARY,
                Stream::card(knowledge::NO_CARD, 0, 0, LibPos{}, bottom ? Stream::bottom(0) : Stream::top(0)), 0, turn);
            lib_b.insert(bottom ? lib_b.end() : lib_b.begin(), hand_b[at].first);
            hand_b.erase(hand_b.begin() + static_cast<std::ptrdiff_t>(at));
        } else if (r < 54 && !hand_b.empty()) {  // B casts a card from hand
            const size_t at = static_cast<size_t>(rng.below(static_cast<int>(hand_b.size())));
            s.move(EventKind::CAST_FROM_HIDDEN, B, Zn::HAND, Zn::STACK,
                Stream::card(hand_b[at].first, 0, next_handle++), 0, turn);
            hand_b.erase(hand_b.begin() + static_cast<std::ptrdiff_t>(at));
        } else if (r < 62 && !lib_b.empty()) {  // B's top card is revealed
            const uint32_t h = next_handle++;
            s.shown(EventKind::REVEAL_START, B, Zn::LIBRARY, {Stream::card(lib_b.front(), 0, h, Stream::top(0))}, turn);
            s.shown(EventKind::REVEAL_END, B, Zn::LIBRARY, {Stream::card(lib_b.front(), h)}, turn);
        } else if (r < 70 && lib_b.size() > 1) {  // B scries 2 and bottoms one
            s.shown(EventKind::LOOK, B, Zn::LIBRARY,
                {Stream::card(knowledge::NO_CARD, 0, 0, Stream::top(0)),
                    Stream::card(knowledge::NO_CARD, 0, 0, Stream::top(1))},
                turn);
            s.move(EventKind::ZONE_MOVE, B, Zn::LIBRARY, Zn::LIBRARY,
                Stream::card(knowledge::NO_CARD, 0, 0, Stream::top(0, 2), Stream::bottom(0)), 0, turn);
            const size_t at = static_cast<size_t>(rng.below(2));
            lib_b.push_back(lib_b[at]);
            lib_b.erase(lib_b.begin() + static_cast<std::ptrdiff_t>(at));
        } else if (r < 76 && !lib_b.empty()) {  // B mills the top card
            s.move(EventKind::ZONE_MOVE, B, Zn::LIBRARY, Zn::GRAVEYARD,
                Stream::card(lib_b.front(), 0, next_handle++, Stream::top(0)), 0, turn);
            lib_b.erase(lib_b.begin());
        } else if (r < 82 && lib_b.size() > 1) {  // B shuffles
            s.shuffle(B);
            shuffle(lib_b);
        } else if (r < 90 && !lib_a.empty()) {  // A draws
            const uint32_t h = next_handle++;
            s.draw(A, lib_a.front(), h, false, turn);
            hand_a.emplace_back(lib_a.front(), h);
            lib_a.erase(lib_a.begin());
        } else if (r < 96 && lib_a.size() > 1) {  // A looks at the top two
            s.shown(EventKind::LOOK, A, Zn::LIBRARY,
                {Stream::card(lib_a[0], 0, 0, Stream::top(0)), Stream::card(lib_a[1], 0, 0, Stream::top(1))}, turn);
        } else if (!hand_a.empty()) {  // A puts a card from hand on top
            const size_t at = static_cast<size_t>(rng.below(static_cast<int>(hand_a.size())));
            s.move(EventKind::ZONE_MOVE, A, Zn::HAND, Zn::LIBRARY,
                Stream::card(hand_a[at].first, hand_a[at].second, 0, LibPos{}, Stream::top(0)), 0, turn);
            lib_a.insert(lib_a.begin(), hand_a[at].first);
            hand_a.erase(hand_a.begin() + static_cast<std::ptrdiff_t>(at));
        }
        if (rng.chance(70) || step == steps) {
            s.frame();
            check(step);
        }
    }
}

// ── Sculpted cases ────────────────────────────────────────────────────────────────────────────

static void sculpted(Tally &t) {
    constexpr int FOW = 101, DAZE = 102, XCARD = 103, FILL = 104;
    const auto begin = [](knowledge::Knowledge &k, const std::vector<std::pair<int, int>> &a,
                           const std::vector<std::pair<int, int>> &b, bool closed, int b_side) {
        knowledge::Decklists lists;
        lists.present[A] = true;
        lists.present[B] = !closed;
        for (const auto &c : a) lists.cards.push_back({A, c.first, c.second, 0});
        if (!closed)
            for (const auto &c : b) lists.cards.push_back({B, c.first, c.second, c.first == FILL ? b_side : 0});
        knowledge::Codes codes;
        k.begin_match(A, codes, lists);
    };
    const auto check_all = [&t](const knowledge::Knowledge &k, const std::string &name, bool want_exact) {
        check_knowledge(t, k, nullptr, nullptr, B, name + " B");
        check_knowledge(t, k, nullptr, nullptr, A, name + " A");
        if (want_exact && !k.rows().meta.exact) t.fail(name + ": not exact");
    };

    // §4.2 / §7: two cards put back from a hand known to be {Force of Will, Daze, X} plus two unknown
    // cards, then X cast.
    {
        knowledge::Knowledge k;
        begin(k, {{FILL, 15}}, {{FOW, 1}, {DAZE, 1}, {XCARD, 1}, {FILL, 12}}, false, 0);
        Stream s(k);
        s.game_start(15, 15);
        for (int i = 0; i < 3; ++i) s.draw(B, knowledge::NO_CARD, 0, true, 0);
        s.shown(EventKind::REVEAL_START, B, Zn::HAND,
            {Stream::card(FOW, 0, 11), Stream::card(DAZE, 0, 12), Stream::card(XCARD, 0, 13)});
        s.shown(EventKind::REVEAL_END, B, Zn::HAND,
            {Stream::card(FOW, 11), Stream::card(DAZE, 12), Stream::card(XCARD, 13)});
        for (int i = 0; i < 2; ++i) s.draw(B, knowledge::NO_CARD, 0, false, 1);
        for (int i = 0; i < 2; ++i)
            s.move(EventKind::ZONE_MOVE, B, Zn::HAND, Zn::LIBRARY,
                Stream::card(knowledge::NO_CARD, 0, 0, LibPos{}, Stream::top(0)));
        s.move(EventKind::CAST_FROM_HIDDEN, B, Zn::HAND, Zn::STACK, Stream::card(XCARD, 0, 20));
        s.frame();
        check_all(k, "sculpted put-back from a partly known hand", true);
    }
    // A game 2 under a closed decklist: two Force of Will seen in game 1 bound the root from below.
    {
        knowledge::Knowledge k;
        begin(k, {{FILL, 12}}, {}, true, 0);
        Stream s(k);
        s.game_start(12, 12);
        for (int i = 0; i < 4; ++i) s.draw(B, knowledge::NO_CARD, 0, true, 0);
        s.move(EventKind::CAST_FROM_HIDDEN, B, Zn::HAND, Zn::STACK, Stream::card(FOW, 0, 70));
        s.move(EventKind::CAST_FROM_HIDDEN, B, Zn::HAND, Zn::STACK, Stream::card(FOW, 0, 71));
        s.game_end();
        s.frame();
        s.game_start(12, 12, 1);
        for (int i = 0; i < 4; ++i) s.draw(B, knowledge::NO_CARD, 0, true, 0);
        s.move(EventKind::CAST_FROM_HIDDEN, B, Zn::HAND, Zn::STACK, Stream::card(FOW, 0, 72));
        s.frame();
        check_all(k, "sculpted closed-decklist game 2", true);
    }
    // A game 2 with open decklists: the registered 15 split between a 12-card deck and a 3-card
    // sideboard; a Daze cast proves it is in the deck.
    {
        knowledge::Knowledge k;
        begin(k, {{FILL, 12}}, {{FOW, 2}, {DAZE, 1}, {FILL, 9}}, false, 3);
        Stream s(k);
        s.game_start(12, 12);
        s.game_end();
        s.frame();
        s.game_start(12, 12, 1, 0, 3);
        for (int i = 0; i < 4; ++i) s.draw(B, knowledge::NO_CARD, 0, true, 0);
        s.move(EventKind::CAST_FROM_HIDDEN, B, Zn::HAND, Zn::STACK, Stream::card(DAZE, 0, 80));
        s.shown(EventKind::REVEAL_START, B, Zn::LIBRARY, {Stream::card(FOW, 0, 81, Stream::top(0))});
        s.frame();
        check_all(k, "sculpted open-decklist game 2", true);
    }
}

// ── Reference lists as alternative roots (templates.h) ──────────────────────────────────────

// A simulated game of opponent B under a closed decklist, B's registered list one of 3-4 small
// reference lists (main 8-10 cards, sideboard 2-3, of 4 kinds): B draws, casts, reveals its top
// card or its hand, puts cards back, scries, mills, shuffles, fetches a sideboard card (a wish) and,
// in some cases, puts a companion into hand from zone none. After every frame each list's k_template
// row of viewer A is compared with two independent answers: the enumeration of the hypothesis
// network (hypothesis_network on the sidecar's template network: does it have a world?), and a
// sidecar fed the same events with that list as B's open registered list (it reports no error and
// its network's enumeration finds a world). An exact row must equal both; a row not exact must say
// consistent. The archetype rows must follow from the template rows.
static void template_case(Tally &t, uint64_t seed) {
    constexpr int KINDS = 4, BASE = 201, OWN = 301;
    Rng rng(seed);
    knowledge::Metagame meta;
    meta.present = true;
    meta.archetypes = {"t0", "t1", "unknown"};
    const int n_lists = 3 + rng.below(2);
    const auto counts_of = [](const std::vector<int> &cards) {
        std::map<int, int> m;
        for (int c : cards) ++m[c];
        return std::vector<std::pair<int, int>>(m.begin(), m.end());
    };
    std::vector<std::vector<int>> mains, sides;
    for (int i = 0; i < n_lists; ++i) {
        std::vector<int> main, side;
        const int nm = 8 + rng.below(3), ns = 2 + rng.below(2);
        for (int c = 0; c < nm; ++c) main.push_back(BASE + rng.below(KINDS));
        for (int c = 0; c < ns; ++c) side.push_back(BASE + rng.below(KINDS));
        knowledge::MetaList l;
        l.name = "list" + std::to_string(i);
        l.archetype = rng.below(2);
        l.main = counts_of(main);
        l.side = counts_of(side);
        meta.lists.push_back(l);
        mains.push_back(main);
        sides.push_back(side);
    }
    const int truth = rng.below(n_lists);
    std::vector<int> lib_b = mains[static_cast<size_t>(truth)], side_b = sides[static_cast<size_t>(truth)];
    for (size_t i = lib_b.size(); i > 1; --i)
        std::swap(lib_b[i - 1], lib_b[static_cast<size_t>(rng.below(static_cast<int>(i)))]);
    // A companion: the true sideboard's last card, set aside (outside the sideboard at the start).
    int companion = knowledge::NO_CARD;
    if (rng.chance(30)) {
        companion = side_b.back();
        side_b.pop_back();
    }

    // Viewer A's sidecar (B's list hidden), then one sidecar per list with it as B's open list.
    const size_t n = meta.lists.size();
    std::vector<knowledge::Knowledge> ks(n + 1);
    knowledge::Codes codes;
    for (size_t i = 0; i <= n; ++i) {
        knowledge::Decklists lists;
        lists.present[A] = true;
        lists.cards.push_back({A, OWN, 10, 0});
        if (i > 0) {
            lists.present[B] = true;
            for (const auto &c : meta.lists[i - 1].main) lists.cards.push_back({B, c.first, c.second, 0});
            for (const auto &c : meta.lists[i - 1].side) lists.cards.push_back({B, c.first, 0, c.second});
        }
        ks[i].begin_match(A, codes, lists);
        ks[i].begin_metagame(meta);
    }
    std::vector<Stream> ss;
    for (size_t i = 0; i <= n; ++i) ss.emplace_back(ks[i]);
    const auto all = [&ss](const std::function<void(Stream &)> &op) {
        for (Stream &s : ss) op(s);
    };
    // B's hand: (card, the viewer's handle for it while it has one: a card put into hand openly).
    std::vector<std::pair<int, uint32_t>> hand_b;
    uint32_t next_handle = 1000;
    all([&](Stream &s) { s.game_start(10, static_cast<int>(lib_b.size()), 0, 0, static_cast<int>(side_b.size())); });
    for (int i = 0; i < 3; ++i) {
        all([](Stream &s) { s.draw(B, knowledge::NO_CARD, 0, true, 0); });
        hand_b.emplace_back(lib_b.front(), 0);
        lib_b.erase(lib_b.begin());
    }
    const std::string tag = "template " + std::to_string(seed);
    std::vector<bool> oracle_failed(n, false);
    const auto check = [&](int step) {
        const std::string where = tag + " step " + std::to_string(step);
        for (const std::string &e : ks[0].take_errors()) t.fail(where + ": " + e);
        const knowledge::KRows rows = ks[0].rows();
        std::vector<int> consistent_labels;
        for (size_t l = 0; l < n; ++l) {
            const knowledge::KTemplateRow *row = nullptr;
            for (const knowledge::KTemplateRow &r : rows.templates)
                if (r.player == B && r.list == l) row = &r;
            if (!row) {
                t.fail(where + ": no k_template row for list " + std::to_string(l));
                continue;
            }
            if (row->consistent) consistent_labels.push_back(meta.lists[l].archetype);
            if (row->registered) t.fail(where + ": a hidden list is marked registered");
            // The hypothesis network's enumeration.
            bool have_direct = false, direct = false;
            if (ks[0].template_network_active()) {
                knowledge::Network h;
                if (!knowledge::hypothesis_network(
                        ks[0].template_network(), meta.lists[l].main, meta.lists[l].side, &h)) {
                    have_direct = true;
                } else {
                    const EnumResult e = rmknow_test::enumerate_worlds(
                        rmknow_test::spec_of(h, knowledge::network_universe(h)), {}, ENUM_CAP);
                    have_direct = e.complete;
                    direct = e.worlds > 0;
                    if (!e.complete) ++t.skipped;
                }
            }
            // The sidecar with the list as B's open list.
            if (!ks[l + 1].take_errors().empty()) oracle_failed[l] = true;
            bool have_open = false, open = false;
            if (oracle_failed[l]) {
                have_open = true;
            } else {
                const PlayerModel &m = ks[l + 1].player(B);
                const EnumResult e =
                    rmknow_test::enumerate_worlds(rmknow_test::spec_of(m.net, m.full.universe), {}, ENUM_CAP);
                have_open = e.complete;
                open = e.worlds > 0;
                if (!e.complete) ++t.skipped;
            }
            for (int which = 0; which < 2; ++which) {
                const bool have = which == 0 ? have_direct : have_open;
                const bool want = which == 0 ? direct : open;
                if (!have) continue;
                ++t.checks;
                if (row->exact) ++t.exact;
                const std::string what =
                    where + " list " + std::to_string(l) + (which == 0 ? " (hypothesis network)" : " (open list)");
                if (!row->consistent && want)
                    t.fail(what + ": reported inconsistent, a world exists (unsound)");
                else if (row->exact && row->consistent != want)
                    t.fail(what + ": reported consistent as exact, no world exists");
            }
            if (static_cast<int>(l) == truth && !row->consistent) t.fail(where + ": the true list is excluded");
        }
        for (const knowledge::KArchetypeRow &a : rows.archetypes) {
            if (a.player != B) continue;
            const long c =
                std::count(consistent_labels.begin(), consistent_labels.end(), static_cast<int>(a.archetype));
            if (a.possible != (c > 0) || a.exact != (c > 0 && c == static_cast<long>(consistent_labels.size())))
                t.fail(where + ": archetype " + std::to_string(a.archetype) + " does not follow from the lists");
        }
    };
    all([](Stream &s) { s.frame(); });
    check(0);
    const int steps = 8 + rng.below(7);
    for (int step = 1; step <= steps; ++step) {
        const uint16_t turn = static_cast<uint16_t>(step);
        const int r = rng.below(100);
        if (r < 18 && !lib_b.empty()) {  // B draws
            all([turn](Stream &s) { s.draw(B, knowledge::NO_CARD, 0, false, turn); });
            hand_b.emplace_back(lib_b.front(), 0);
            lib_b.erase(lib_b.begin());
        } else if (r < 34 && !hand_b.empty()) {  // B casts a card from hand
            const size_t at = static_cast<size_t>(rng.below(static_cast<int>(hand_b.size())));
            const uint32_t h = next_handle++;
            const std::pair<int, uint32_t> card = hand_b[at];
            all([&](Stream &s) {
                s.move(EventKind::CAST_FROM_HIDDEN, B, Zn::HAND, Zn::STACK, Stream::card(card.first, card.second, h), 0,
                    turn);
            });
            hand_b.erase(hand_b.begin() + static_cast<std::ptrdiff_t>(at));
        } else if (r < 42 && !lib_b.empty()) {  // B's top card is revealed
            const uint32_t h = next_handle++;
            const int card = lib_b.front();
            all([&](Stream &s) {
                s.shown(EventKind::REVEAL_START, B, Zn::LIBRARY, {Stream::card(card, 0, h, Stream::top(0))}, turn);
                s.shown(EventKind::REVEAL_END, B, Zn::LIBRARY, {Stream::card(card, h)}, turn);
            });
        } else if (r < 48 && !hand_b.empty()) {  // B's hand is revealed
            std::vector<knowledge::EventCard> shown, ended;
            for (const auto &c : hand_b) {
                const uint32_t h = c.second ? c.second : next_handle++;
                shown.push_back(Stream::card(c.first, c.second, h));
                ended.push_back(Stream::card(c.first, h, c.second));
            }
            all([&](Stream &s) {
                s.shown(EventKind::REVEAL_START, B, Zn::HAND, shown, turn);
                s.shown(EventKind::REVEAL_END, B, Zn::HAND, ended, turn);
            });
        } else if (r < 58 && !hand_b.empty()) {  // B puts a card from hand on top or bottom, unseen
            const size_t at = static_cast<size_t>(rng.below(static_cast<int>(hand_b.size())));
            const bool bottom = rng.chance(30);
            const std::pair<int, uint32_t> card = hand_b[at];
            all([&](Stream &s) {
                s.move(EventKind::ZONE_MOVE, B, Zn::HAND, Zn::LIBRARY,
                    Stream::card(card.second ? card.first : knowledge::NO_CARD, card.second, 0, LibPos{},
                        bottom ? Stream::bottom(0) : Stream::top(0)),
                    0, turn);
            });
            lib_b.insert(bottom ? lib_b.end() : lib_b.begin(), card.first);
            hand_b.erase(hand_b.begin() + static_cast<std::ptrdiff_t>(at));
        } else if (r < 64 && lib_b.size() > 1) {  // B scries 2 and bottoms one
            const size_t at = static_cast<size_t>(rng.below(2));
            all([&](Stream &s) {
                s.shown(EventKind::LOOK, B, Zn::LIBRARY,
                    {Stream::card(knowledge::NO_CARD, 0, 0, Stream::top(0)),
                        Stream::card(knowledge::NO_CARD, 0, 0, Stream::top(1))},
                    turn);
                s.move(EventKind::ZONE_MOVE, B, Zn::LIBRARY, Zn::LIBRARY,
                    Stream::card(knowledge::NO_CARD, 0, 0, Stream::top(0, 2), Stream::bottom(0)), 0, turn);
            });
            lib_b.push_back(lib_b[at]);
            lib_b.erase(lib_b.begin() + static_cast<std::ptrdiff_t>(at));
        } else if (r < 72 && !lib_b.empty()) {  // B mills the top card
            const uint32_t h = next_handle++;
            const int card = lib_b.front();
            all([&](Stream &s) {
                s.move(EventKind::ZONE_MOVE, B, Zn::LIBRARY, Zn::GRAVEYARD, Stream::card(card, 0, h, Stream::top(0)), 0,
                    turn);
            });
            lib_b.erase(lib_b.begin());
        } else if (r < 80 && lib_b.size() > 1) {  // B shuffles
            all([](Stream &s) { s.shuffle(B); });
            for (size_t i = lib_b.size(); i > 1; --i)
                std::swap(lib_b[i - 1], lib_b[static_cast<size_t>(rng.below(static_cast<int>(i)))]);
        } else if (r < 88 && !side_b.empty()) {  // B fetches a sideboard card into hand (a wish)
            const size_t at = static_cast<size_t>(rng.below(static_cast<int>(side_b.size())));
            const uint32_t h = next_handle++;
            const int card = side_b[at];
            all([&](Stream &s) {
                s.move(EventKind::ZONE_MOVE, B, Zn::SIDEBOARD, Zn::HAND, Stream::card(card, 0, h), 0, turn);
            });
            side_b.erase(side_b.begin() + static_cast<std::ptrdiff_t>(at));
            hand_b.emplace_back(card, h);
        } else if (companion != knowledge::NO_CARD) {  // B puts its companion into hand
            const uint32_t h = next_handle++;
            const int card = companion;
            all([&](Stream &s) {
                s.move(EventKind::ZONE_MOVE, B, Zn::NONE, Zn::HAND, Stream::card(card, 0, h), 0, turn);
            });
            hand_b.emplace_back(card, h);
            companion = knowledge::NO_CARD;
        }
        if (rng.chance(70) || step == steps) {
            all([](Stream &s) { s.frame(); });
            check(step);
        }
    }
}

int run_exactness(int n) {
    Tally synth, streams, raw_streams, sculpt, templates;
    const std::clock_t t0 = std::clock();
    for (int i = 1; i <= n; ++i) synthetic_case(synth, static_cast<uint64_t>(i));
    const std::clock_t t1 = std::clock();
    for (int i = 1; i <= n; ++i) stream_case(streams, static_cast<uint64_t>(i), true);
    const std::clock_t tm = std::clock();
    for (int i = 1; i <= n; i += 4) stream_case(raw_streams, static_cast<uint64_t>(i), false);
    const std::clock_t t2 = std::clock();
    sculpted(sculpt);
    const std::clock_t t3 = std::clock();
    for (int i = 1; i <= n; ++i) template_case(templates, static_cast<uint64_t>(i));
    const std::clock_t t4 = std::clock();
    synth.seconds = static_cast<double>(t1 - t0) / CLOCKS_PER_SEC;
    streams.seconds = static_cast<double>(tm - t1) / CLOCKS_PER_SEC;
    raw_streams.seconds = static_cast<double>(t2 - tm) / CLOCKS_PER_SEC;
    sculpt.seconds = static_cast<double>(t3 - t2) / CLOCKS_PER_SEC;
    templates.seconds = static_cast<double>(t4 - t3) / CLOCKS_PER_SEC;
    int failed = 0;
    for (const auto &p : {std::make_pair("synthetic", &synth), std::make_pair("streams", &streams),
             std::make_pair("raw-stream", &raw_streams), std::make_pair("sculpted", &sculpt),
             std::make_pair("templates", &templates)}) {
        const Tally &tl = *p.second;
        std::printf("  %-10s checks=%d exact=%d compactions=%d skipped=%d failures=%d cpu=%.1fs\n", p.first, tl.checks,
            tl.exact, tl.compactions, tl.skipped, tl.failures, tl.seconds);
        if (tl.failures) {
            ++failed;
            std::printf(
                "ERROR: rmknow exactness %s: %d failure(s), first: %s\n", p.first, tl.failures, tl.first.c_str());
        }
    }
    Tally all;
    for (const Tally *tl : {&synth, &streams, &raw_streams, &sculpt, &templates}) {
        all.checks += tl->checks;
        all.exact += tl->exact;
        all.compactions += tl->compactions;
        all.skipped += tl->skipped;
    }
    std::printf("RMKNOW_EXACTNESS: cases=%d checks=%d exact=%d compactions=%d skipped=%d failed=%d\n", n, all.checks,
        all.exact, all.compactions, all.skipped, failed);
    return failed == 0 ? 0 : 1;
}
