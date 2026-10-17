#include "shadow_knowledge.h"

#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../card_vocab.h"
#include "../classes/game.h"
#include "../classes/gamestate.h"
#include "../game_driver.h"
#include "../machine_io.h"
#include "stream.h"

namespace {

// One mismatch as found at a decision: its key (kind, direction, owner, the two sides' values, the
// card) and, for top, the positions it covers.
struct Mismatch {
        std::string kind;
        std::string direction;
        int owner = 0;  // player index of the library or hand
        int card = -1;
        std::string old_value;
        std::string new_value;
        std::string truth;
        std::vector<int> at;
        int left = 0;     // hand: copies of the card the frame's moves show leaving the hand
        int entered = 0;  // hand: copies of the card the frame's moves show entering it
        int slots = 0;    // hand: copies of the card in the vector's opp_known_hand slots
};

struct Stats {
        size_t decisions = 0;
        size_t compared = 0;
        size_t mismatches = 0;
        size_t shown = 0;
};

}  // namespace

static Stats stats;
// Per seat, every mismatch key present at its previous decision, with its position count.
static std::map<std::string, size_t> previous[2];

static std::string card_text(int card);
static bool is_move(rmstate::EventKind kind);
static const char *event_kind_name(rmstate::EventKind kind);
static const char *zone_name(knowledge::Zn zone);
static std::string seat_text(int seat, int viewer);
static std::string events_text(const knowledge::Snapshot &snap, int viewer);
static std::string mismatch_key(const Mismatch &m);
static void compare_top(int owner, const int *old_top, const knowledge::KRows &rows, const knowledge::Truth &truth,
    std::vector<Mismatch> *out);
static void compare_hand(int viewer, const GameState &gs, const knowledge::KRows &rows, const knowledge::Truth &truth,
    const knowledge::Snapshot &snap, std::vector<Mismatch> *out);
static void compare_seen(int viewer, const GameState &gs, const knowledge::KRows &rows, std::vector<Mismatch> *out);
static void add_count_mismatch(
    const char *kind, int owner, int card, int old_n, int new_n, const std::string &truth, std::vector<Mismatch> *out);

static std::string card_text(int card) {
    if (card < 0) return "-";
    const char *name = card_index_to_name(card);
    return name ? name : "#" + std::to_string(card);
}

// An event that moves cards between zones.
static bool is_move(rmstate::EventKind kind) {
    return kind == rmstate::EventKind::ZONE_MOVE || kind == rmstate::EventKind::DRAW ||
           kind == rmstate::EventKind::CAST_FROM_HIDDEN || kind == rmstate::EventKind::PLAY_FROM_HIDDEN;
}

static const char *event_kind_name(rmstate::EventKind kind) {
#define SHADOW_EVENT_NAME(name) #name,
    static const char *const names[] = {RMSTATE_EVENT_KIND(SHADOW_EVENT_NAME)};
#undef SHADOW_EVENT_NAME
    const size_t i = static_cast<size_t>(kind);
    return i < sizeof(names) / sizeof(names[0]) ? names[i] : "?";
}

static const char *zone_name(knowledge::Zn zone) {
    switch (zone) {
        case knowledge::Zn::LIBRARY:
            return "library";
        case knowledge::Zn::BATTLEFIELD:
            return "battlefield";
        case knowledge::Zn::HAND:
            return "hand";
        case knowledge::Zn::STACK:
            return "stack";
        case knowledge::Zn::GRAVEYARD:
            return "graveyard";
        case knowledge::Zn::EXILE:
            return "exile";
        case knowledge::Zn::SIDEBOARD:
            return "sideboard";
        case knowledge::Zn::NONE:
            return "none";
        case knowledge::Zn::OTHER:
            return "other";
    }
    return "?";
}

// A seat relative to the viewer: self, opp, or - for none.
static std::string seat_text(int seat, int viewer) {
    if (seat == knowledge::NO_SEAT) return "-";
    return seat == viewer ? "self" : "opp";
}

// The frame's events, distinct ones in order with a repeat count.
static std::string events_text(const knowledge::Snapshot &snap, int viewer) {
    std::vector<std::pair<std::string, size_t>> parts;
    for (const knowledge::Event &ev : snap.events) {
        std::string s = event_kind_name(ev.kind);
        for (char &ch : s) ch = static_cast<char>(ch >= 'A' && ch <= 'Z' ? ch - 'A' + 'a' : ch);
        s += std::string(":") + zone_name(ev.zone) + "." + seat_text(ev.zone_owner, viewer);
        if (ev.to_zone != knowledge::Zn::NONE || ev.kind == rmstate::EventKind::ZONE_MOVE)
            s += std::string(">") + zone_name(ev.to_zone) + "." + seat_text(ev.to_owner, viewer);
        if (ev.source_card != knowledge::NO_CARD) s += "@" + card_text(ev.source_card);
        if (!parts.empty() && parts.back().first == s)
            ++parts.back().second;
        else
            parts.push_back({s, 1});
    }
    if (parts.empty()) return "-";
    std::string out;
    for (const auto &p : parts) {
        if (!out.empty()) out += ";";
        out += p.first;
        if (p.second > 1) out += " ×" + std::to_string(p.second);
    }
    return out;
}

// A mismatch's identity across decisions: its kind, direction, owner and card (a change of the
// values while it persists is not a new mismatch).
static std::string mismatch_key(const Mismatch &m) {
    return m.kind + "|" + m.direction + "|" + std::to_string(m.owner) + "|" + std::to_string(m.card);
}

// The known top positions of `owner`'s library: the old vector's `old_top` against the k_position
// rows with a card. Mismatches with the same two values are one, listing their positions.
static void compare_top(int owner, const int *old_top, const knowledge::KRows &rows, const knowledge::Truth &truth,
    std::vector<Mismatch> *out) {
    int new_top[KNOWN_TOP_LIBRARY_SIZE];
    for (int i = 0; i < KNOWN_TOP_LIBRARY_SIZE; ++i) new_top[i] = -1;
    for (const knowledge::KPositionRow &r : rows.positions) {
        if (r.owner != owner || r.card == knowledge::NO_CARD || r.at.bottom()) continue;
        if (r.at.pos < KNOWN_TOP_LIBRARY_SIZE) new_top[r.at.pos] = r.card;
    }
    std::map<std::string, size_t> index;
    for (int i = 0; i < KNOWN_TOP_LIBRARY_SIZE; ++i) {
        ++stats.compared;
        const int o = old_top[i];
        const int n = new_top[i];
        if (o == n) continue;
        Mismatch m;
        m.kind = "top";
        m.owner = owner;
        m.direction = o < 0 ? "new_more" : n < 0 ? "old_more" : "conflict";
        m.card = o < 0 ? n : o;
        m.old_value = card_text(o);
        m.new_value = card_text(n);
        const std::vector<int> &lib = truth.library[owner];
        m.truth = o < 0 ? "?" : (static_cast<size_t>(i) < lib.size() && lib[i] == o) ? "1" : "0";
        const std::string key = mismatch_key(m) + "|" + m.old_value + "|" + m.new_value + "|" + m.truth;
        auto it = index.find(key);
        if (it == index.end()) {
            index[key] = out->size();
            m.at.push_back(i);
            out->push_back(m);
        } else {
            (*out)[it->second].at.push_back(i);
        }
    }
}

static void add_count_mismatch(
    const char *kind, int owner, int card, int old_n, int new_n, const std::string &truth, std::vector<Mismatch> *out) {
    Mismatch m;
    m.kind = kind;
    m.owner = owner;
    m.card = card;
    m.direction = old_n > new_n ? "old_more" : "new_more";
    m.old_value = std::to_string(old_n);
    m.new_value = std::to_string(new_n);
    m.truth = truth;
    out->push_back(m);
}

// The opponent's cards definitely in hand: opp_known_hand's slots against k_known_hand, per card.
static void compare_hand(int viewer, const GameState &gs, const knowledge::KRows &rows, const knowledge::Truth &truth,
    const knowledge::Snapshot &snap, std::vector<Mismatch> *out) {
    const int opp = 1 - viewer;
    std::map<int, int> old_n, new_n, slots;
    for (int card : opp_known_hand_cards(viewer == 0 ? Zone::PLAYER_A : Zone::PLAYER_B)) ++old_n[card];
    for (int i = 0; i < MAX_HAND_SLOTS; ++i)
        if (gs.opp_known_hand[i] >= 0) ++slots[gs.opp_known_hand[i]];
    for (const knowledge::KKnownHandRow &r : rows.known_hand)
        if (r.copies > 0) new_n[r.card] += r.copies;
    std::set<int> cards;
    for (const auto &kv : old_n) cards.insert(kv.first);
    for (const auto &kv : new_n) cards.insert(kv.first);
    for (int card : cards) {
        ++stats.compared;
        const int o = old_n.count(card) ? old_n[card] : 0;
        const int n = new_n.count(card) ? new_n[card] : 0;
        if (o == n) continue;
        const auto h = truth.hand[opp].find(card);
        const int held = h == truth.hand[opp].end() ? 0 : h->second;
        add_count_mismatch("hand", opp, card, o, n, held >= o ? "1" : "0", out);
        out->back().slots = slots.count(card) ? slots[card] : 0;
        for (const knowledge::Event &ev : snap.events) {
            if (!is_move(ev.kind)) continue;
            for (const knowledge::EventCard &c : ev.cards) {
                if (c.card != card) continue;
                if (ev.zone == knowledge::Zn::HAND && ev.zone_owner == opp) ++out->back().left;
                if (ev.to_zone == knowledge::Zn::HAND && ev.to_owner == opp) ++out->back().entered;
            }
        }
    }
}

// The opponent's cards seen this match: the registered-decklist revealed bits against k_seen's
// match counts, per card.
static void compare_seen(int viewer, const GameState &gs, const knowledge::KRows &rows, std::vector<Mismatch> *out) {
    const int opp = 1 - viewer;
    std::set<int> old_seen, new_seen;
    for (int i = 0; i < DECKLIST_MAIN_SLOTS; ++i)
        if (gs.opp_deck_main_id[i] >= 0 && gs.opp_deck_main_revealed[i]) old_seen.insert(gs.opp_deck_main_id[i]);
    for (int i = 0; i < DECKLIST_SIDE_SLOTS; ++i)
        if (gs.opp_deck_side_id[i] >= 0 && gs.opp_deck_side_revealed[i]) old_seen.insert(gs.opp_deck_side_id[i]);
    for (const knowledge::KSeenRow &r : rows.seen)
        if (r.player == opp && r.match > 0) new_seen.insert(r.card);
    std::set<int> cards = old_seen;
    cards.insert(new_seen.begin(), new_seen.end());
    for (int card : cards) {
        ++stats.compared;
        const int o = old_seen.count(card) ? 1 : 0;
        const int n = new_seen.count(card) ? 1 : 0;
        if (o != n) add_count_mismatch("seen", opp, card, o, n, "?", out);
    }
}

void shadow_knowledge_seat(Zone::Ownership viewer, const knowledge::KRows &rows, const knowledge::Snapshot &snap,
    const knowledge::Truth &truth) {
    if (viewer != Zone::PLAYER_A && viewer != Zone::PLAYER_B) return;
    const int v = viewer == Zone::PLAYER_A ? 0 : 1;
    if (v == 0) ++stats.decisions;
    static GameState gs;
    populate_gamestate(&gs, viewer);
    std::vector<Mismatch> found;
    compare_top(v, gs.known_top_library_self, rows, truth, &found);
    compare_top(1 - v, gs.known_top_library_opp, rows, truth, &found);
    compare_hand(v, gs, rows, truth, snap, &found);
    compare_seen(v, gs, rows, &found);

    std::map<std::string, size_t> now;
    std::string events;
    for (const Mismatch &m : found) {
        const std::string key = mismatch_key(m);
        const size_t n = m.at.empty() ? 1 : m.at.size();
        now[key] += n;
        stats.mismatches += n;
        const auto it = previous[v].find(key);
        if (it != previous[v].end() && it->second >= now[key]) continue;
        if (events.empty()) events = events_text(snap, v);
        std::string at;
        for (int p : m.at) at += (at.empty() ? " at=" : ",") + std::to_string(p);
        if (m.kind == "hand")
            at = " left=" + std::to_string(m.left) + " entered=" + std::to_string(m.entered) +
                 " slots=" + std::to_string(m.slots);
        ++stats.shown;
        std::fflush(stdout);
        std::fprintf(stderr,
            "SHADOW: %s %s viewer=%s owner=%s card=%s old=%s new=%s%s truth=%s game=%d turn=%d decision=%u "
            "sideboarding=%d events=%s\n",
            m.kind.c_str(), m.direction.c_str(), v == 0 ? "A" : "B", m.owner == v ? "self" : "opp",
            card_text(m.card).c_str(), m.old_value.c_str(), m.new_value.c_str(), at.c_str(), m.truth.c_str(),
            match_game_number < 0 ? 0 : match_game_number, static_cast<int>(cur_game.turn_state.turn),
            rmstate_decision_number(), snap.sideboard_phase ? 1 : 0, events.c_str());
    }
    previous[v] = now;
}

void shadow_knowledge_match_over() {
    std::fflush(stdout);
    std::fprintf(stderr, "SHADOW-KNOWLEDGE: decisions=%zu compared=%zu mismatches=%zu shown=%zu\n", stats.decisions,
        stats.compared, stats.mismatches, stats.shown);
    std::fflush(stderr);
}
