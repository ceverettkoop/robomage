#include "sideboard_plan.h"

#include <algorithm>
#include <cctype>

#include "../../src/agents/decision_view.h"
#include "../../src/featurize/json.h"
#include "../../src/str_util.h"
#include "files.h"

static int seat_index(uint64_t seat);
static void add_count(std::map<int, int> *m, int card, int n);
static int count_of(const std::map<int, int> &m, int card);
static int total_of(const std::map<int, int> &m);
static bool read_card_counts(const featurize::Json &v, std::vector<std::pair<std::string, int>> *out);
static std::string signed_lines(const std::vector<std::pair<std::string, int>> &cards, const char *sign);

// The state format's seat value (1 A, 2 B, 0 none) as 0, 1 or -1.
static int seat_index(uint64_t seat) {
    return seat == 1 ? 0 : seat == 2 ? 1 : -1;
}

// Adds `n` copies of `card` (a count that reaches 0 is removed).
static void add_count(std::map<int, int> *m, int card, int n) {
    const int v = count_of(*m, card) + n;
    if (v == 0) {
        m->erase(card);
    } else {
        (*m)[card] = v;
    }
}

static int count_of(const std::map<int, int> &m, int card) {
    const auto it = m.find(card);
    return it == m.end() ? 0 : it->second;
}

static int total_of(const std::map<int, int> &m) {
    int n = 0;
    for (const auto &kv : m) n += kv.second;
    return n;
}

// A guide's {"Card": copies} object, in the file's order.
static bool read_card_counts(const featurize::Json &v, std::vector<std::pair<std::string, int>> *out) {
    if (!v.is_object()) return false;
    for (const auto &m : v.members()) {
        if (!m.second.is_integer() || m.second.as_int() < 1) return false;
        out->emplace_back(m.first, static_cast<int>(m.second.as_int()));
    }
    return true;
}

// "−1 Swamp, +1 Pernicious Deed" pieces: "<sign><n> <card>" joined.
static std::string signed_lines(const std::vector<std::pair<std::string, int>> &cards, const char *sign) {
    std::string out;
    for (const auto &c : cards)
        out += (out.empty() ? "" : ", ") + std::string(sign) + std::to_string(c.second) + " " + c.first;
    return out;
}

namespace rmwork {

int DeckCounts::main_of(int card) const {
    return count_of(main, card);
}

int DeckCounts::side_of(int card) const {
    return count_of(side, card);
}

int DeckCounts::main_total() const {
    return total_of(main);
}

int DeckCounts::side_total() const {
    return total_of(side);
}

SideboardFacts read_sideboard_facts(const agents::SeatStream &s, const rmstate::DecodedFrame &f, int seat) {
    SideboardFacts out;
    const agents::Table header(f, s.schema(), "header");
    if (header.rows()) {
        out.sideboarding = header.u(0, header.col("sideboarding")) != 0;
        out.game = static_cast<int>(header.u(0, header.col("game")));
    }
    const agents::Table match(f, s.schema(), "match");
    if (match.rows()) {
        out.plays_first = seat_index(match.u(0, match.col("plays_first")));
        out.swaps = static_cast<int>(match.u(0, match.col("sb_swaps")));
        out.drift = static_cast<int>(match.s(0, match.col("sb_drift")));
    }
    const agents::Table pending(f, s.schema(), "pending");
    if (out.drift != 0 && pending.rows()) out.pending_card = pending.card(0, pending.col("source_card"));
    const agents::Table deck(f, s.schema(), "own_deck");
    const int c_player = deck.col("player"), c_card = deck.col("card"), c_main = deck.col("main"),
              c_side = deck.col("side");
    for (size_t r = 0; r < deck.rows(); ++r) {
        if (seat_index(deck.u(r, c_player)) != seat) continue;
        const int card = deck.card(r, c_card);
        if (card < 0) continue;
        out.has_deck = true;
        add_count(&out.deck.main, card, static_cast<int>(deck.u(r, c_main)));
        add_count(&out.deck.side, card, static_cast<int>(deck.u(r, c_side)));
    }
    const agents::DecisionView v(s, f, seat);
    for (size_t i = 0; i < v.n(); ++i) {
        SideboardRow row;
        row.category = v.category(i);
        row.card = v.menu_card_id(i);
        out.menu.push_back(row);
    }
    return out;
}

Decklists read_decklists(const agents::SeatStream &s) {
    Decklists out;
    const rmstate::DecodedFrame &f = s.decklists_frame();
    const agents::Table lists(f, s.schema(), "decklist");
    for (size_t r = 0; r < lists.rows(); ++r) {
        const int seat = seat_index(lists.u(r, lists.col("player")));
        if (seat < 0) continue;
        out.known[static_cast<size_t>(seat)] = true;
        out.name[static_cast<size_t>(seat)] = lists.text(r, lists.col("name"));
    }
    const agents::Table cards(f, s.schema(), "decklist_card");
    for (size_t r = 0; r < cards.rows(); ++r) {
        const int seat = seat_index(cards.u(r, cards.col("player")));
        const int card = cards.card(r, cards.col("card"));
        if (seat < 0 || card < 0) continue;
        DeckCounts &d = out.lists[static_cast<size_t>(seat)];
        add_count(&d.main, card, static_cast<int>(cards.u(r, cards.col("main"))));
        add_count(&d.side, card, static_cast<int>(cards.u(r, cards.col("side"))));
    }
    return out;
}

// ── The plan ──

bool SideboardPlan::may_bring_in(const SideboardState &s, int card, std::string *why) const {
    if (s.now.side_of(card) - planned_in(card) < 1) {
        *why = "no copy left in the sideboard";
        return false;
    }
    if (s.out_locked(card)) {
        *why = "cut this phase: the engine does not bring it back";
        return false;
    }
    if (swaps_after(s) + 1 > SIDEBOARD_SWAP_CAP) {
        *why = "the plan already makes " + std::to_string(SIDEBOARD_SWAP_CAP) + " swaps, the most a phase allows";
        return false;
    }
    return true;
}

bool SideboardPlan::may_cut(const SideboardState &s, int card, std::string *why) const {
    if (s.now.main_of(card) - planned_out(card) < 1) {
        *why = "no copy left in the deck";
        return false;
    }
    if (s.in_locked(card)) {
        *why = "brought in this phase: the engine does not cut it again";
        return false;
    }
    if (outs() + 1 > ins() + s.drift) {
        *why = "nothing to balance: bring a sideboard card in first";
        return false;
    }
    return true;
}

bool SideboardPlan::click_side(const SideboardState &s, int card, std::string *why) {
    if (planned_out(card) > 0) {
        add_count(&out_, card, -1);
        history_.emplace_back(false, -card - 1);
        return true;
    }
    if (!may_bring_in(s, card, why)) return false;
    add_count(&in_, card, 1);
    history_.emplace_back(true, card);
    return true;
}

bool SideboardPlan::click_main(const SideboardState &s, int card, std::string *why) {
    if (planned_in(card) > 0) {
        add_count(&in_, card, -1);
        history_.emplace_back(true, -card - 1);
        return true;
    }
    if (!may_cut(s, card, why)) return false;
    add_count(&out_, card, 1);
    history_.emplace_back(false, card);
    return true;
}

// A history entry is (in, card) for a copy planned, (in, -card - 1) for a planned copy cancelled.
bool SideboardPlan::undo() {
    if (history_.empty()) return false;
    const std::pair<bool, int> h = history_.back();
    history_.pop_back();
    std::map<int, int> &m = h.first ? in_ : out_;
    if (h.second >= 0) {
        add_count(&m, h.second, -1);
    } else {
        add_count(&m, -h.second - 1, 1);
    }
    return true;
}

void SideboardPlan::clear() {
    in_.clear();
    out_.clear();
    history_.clear();
}

bool SideboardPlan::set_target(const SideboardState &s, const std::map<int, int> &target, std::string *why) {
    std::map<int, int> in, out;
    std::map<int, int> all = s.now.main;
    for (const auto &kv : target) all[kv.first] += 0;
    for (const auto &kv : all) {
        const int card = kv.first, have = s.now.main_of(card), want = count_of(target, card);
        if (want > have) {
            if (s.out_locked(card)) {
                *why = "card #" + std::to_string(card) + " was cut this phase and cannot come back";
                return false;
            }
            if (s.now.side_of(card) < want - have) {
                *why = "card #" + std::to_string(card) + " has " + std::to_string(s.now.side_of(card)) +
                       " copies in the sideboard, the plan brings in " + std::to_string(want - have);
                return false;
            }
            in[card] = want - have;
        } else if (want < have) {
            if (s.in_locked(card)) {
                *why = "card #" + std::to_string(card) + " was brought in this phase and cannot be cut";
                return false;
            }
            out[card] = have - want;
        }
    }
    if (total_of(out) != total_of(in) + s.drift) {
        *why = "the target deck is not the deck's size";
        return false;
    }
    if (s.swaps + s.drift + total_of(in) > SIDEBOARD_SWAP_CAP) {
        *why = "it takes more than the " + std::to_string(SIDEBOARD_SWAP_CAP) + " swaps a phase allows";
        return false;
    }
    in_ = in;
    out_ = out;
    history_.clear();
    for (const auto &kv : in_)
        for (int i = 0; i < kv.second; ++i) history_.emplace_back(true, kv.first);
    for (const auto &kv : out_)
        for (int i = 0; i < kv.second; ++i) history_.emplace_back(false, kv.first);
    return true;
}

int SideboardPlan::ins() const {
    return total_of(in_);
}

int SideboardPlan::outs() const {
    return total_of(out_);
}

int SideboardPlan::planned_in(int card) const {
    return count_of(in_, card);
}

int SideboardPlan::planned_out(int card) const {
    return count_of(out_, card);
}

DeckCounts SideboardPlan::result(const SideboardState &s) const {
    DeckCounts d = s.now;
    for (const auto &kv : in_) {
        add_count(&d.main, kv.first, kv.second);
        add_count(&d.side, kv.first, -kv.second);
    }
    for (const auto &kv : out_) {
        add_count(&d.main, kv.first, -kv.second);
        add_count(&d.side, kv.first, kv.second);
    }
    return d;
}

int SideboardPlan::swaps_after(const SideboardState &s) const {
    return s.swaps + s.drift + ins();
}

bool SideboardPlan::balanced(const SideboardState &s) const {
    return outs() == ins() + s.drift;
}

// ── Sending ──

SideboardStep next_sideboard_step(
    const SideboardFacts &f, const std::map<int, int> &target, const std::function<std::string(int)> &names) {
    SideboardStep out;
    if (!f.sideboarding) {
        out.error = "this decision is not a sideboarding decision";
        return out;
    }
    std::vector<int> want;  // the cards to move: out while a cut is owed, else in
    if (f.drift == 1) {
        for (const auto &kv : f.deck.main)
            if (kv.second > count_of(target, kv.first)) want.push_back(kv.first);
        for (size_t i = 0; i < f.menu.size(); ++i)
            if (f.menu[i].category == "SIDEBOARD_OUT" &&
                std::find(want.begin(), want.end(), f.menu[i].card) != want.end()) {
                out.row = static_cast<int>(i);
                out.what = "out: " + names(f.menu[i].card);
                return out;
            }
        out.error = "a cut is owed, but the menu offers no cut of ";
    } else if (f.drift == 0) {
        if (main_matches(f.deck, target)) {
            for (size_t i = 0; i < f.menu.size(); ++i)
                if (f.menu[i].category == "SIDEBOARD_DONE") {
                    out.row = static_cast<int>(i);
                    out.done = true;
                    out.what = "Done";
                    return out;
                }
            out.error = "the deck is as planned, but the menu offers no Done";
            return out;
        }
        for (const auto &kv : target)
            if (kv.second > f.deck.main_of(kv.first)) want.push_back(kv.first);
        for (size_t i = 0; i < f.menu.size(); ++i)
            if (f.menu[i].category == "SIDEBOARD_IN" &&
                std::find(want.begin(), want.end(), f.menu[i].card) != want.end()) {
                out.row = static_cast<int>(i);
                out.what = "in: " + names(f.menu[i].card);
                return out;
            }
        out.error = "the menu offers no card the plan brings in: ";
    } else {
        out.error = "the deck is off its size by " + std::to_string(f.drift);
        return out;
    }
    std::string list;
    for (int c : want) list += (list.empty() ? "" : ", ") + names(c);
    out.error += list.empty() ? "(none: the plan does not lead from this deck)" : list;
    return out;
}

bool main_matches(const DeckCounts &d, const std::map<int, int> &target) {
    for (const auto &kv : d.main)
        if (kv.second != count_of(target, kv.first)) return false;
    for (const auto &kv : target)
        if (kv.second != d.main_of(kv.first)) return false;
    return true;
}

// ── The guides ──

std::string card_key(const std::string &name) {
    const std::string folded = ascii_fold_card_name(name);
    std::string out;
    for (char c : folded)
        if (std::isalnum(static_cast<unsigned char>(c)))
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return out;
}

bool SideboardGuides::load(const std::string &path, std::string *err) {
    *this = SideboardGuides();
    path_ = path;
    std::string text;
    if (!read_file(path, &text)) {
        *err = "cannot read " + path;
        return false;
    }
    featurize::Json doc;
    if (!featurize::Json::parse(text, &doc, err)) {
        *err = path + ": " + *err;
        return false;
    }
    const featurize::Json *lands = doc.get("draw_land"), *guides = doc.get("guides"), *hold = doc.get("holdout");
    if (!lands || !lands->is_object() || !guides || !guides->is_object() || !hold || !hold->is_array()) {
        *err = path + ": no draw_land, guides or holdout";
        return false;
    }
    for (const auto &m : lands->members()) draw_lands_[m.first] = m.second.is_string() ? m.second.as_string() : "";
    for (const auto &own : guides->members()) {
        if (!own.second.is_object()) {
            *err = path + ": guides." + own.first + " is not an object";
            return false;
        }
        for (const auto &opp : own.second.members()) {
            GuideRow row;
            const featurize::Json *in = opp.second.get("in"), *out = opp.second.get("out"),
                                  *draw = opp.second.get("draw");
            if (!in || !out || !read_card_counts(*in, &row.in) || !read_card_counts(*out, &row.out)) {
                *err = path + ": guides." + own.first + "." + opp.first + " has no in / out lists";
                return false;
            }
            if (draw && draw->is_string()) row.draw = draw->as_string();
            rows_[std::make_pair(own.first, opp.first)] = row;
        }
    }
    for (const featurize::Json &p : hold->items())
        if (p.is_array() && p.items().size() == 2 && p.items()[0].is_string() && p.items()[1].is_string())
            holdout_.emplace_back(p.items()[0].as_string(), p.items()[1].as_string());
    loaded_ = true;
    return true;
}

const GuideRow *SideboardGuides::row(const std::string &own, const std::string &opp) const {
    const auto it = rows_.find(std::make_pair(own, opp));
    return it == rows_.end() ? nullptr : &it->second;
}

std::string SideboardGuides::draw_land(const std::string &own) const {
    const auto it = draw_lands_.find(own);
    return it == draw_lands_.end() ? "" : it->second;
}

bool SideboardGuides::holdout(const std::string &own, const std::string &opp) const {
    return std::find(holdout_.begin(), holdout_.end(), std::make_pair(own, opp)) != holdout_.end();
}

std::string league_stem(const std::string &deck_name) {
    const std::string prefix = "league/";
    if (deck_name.compare(0, prefix.size(), prefix) != 0 || deck_name.size() == prefix.size()) return "";
    return deck_name.substr(prefix.size());
}

GuidePlan guide_plan(
    const SideboardGuides &g, const Decklists &d, int seat, bool on_play, const agents::CardNames &names) {
    GuidePlan p;
    p.on_play = on_play;
    if (!g.loaded()) {
        p.why = "the guides file was not read";
        return p;
    }
    if (seat < 0 || seat > 1 || !d.known[static_cast<size_t>(seat)] || !d.known[static_cast<size_t>(1 - seat)]) {
        p.why = "the decklists are not known";
        return p;
    }
    const std::string own_name = d.name[static_cast<size_t>(seat)], opp_name = d.name[static_cast<size_t>(1 - seat)];
    p.own = league_stem(own_name);
    p.opp = league_stem(opp_name);
    if (p.own.empty() || !g.guided(p.own)) {
        p.why = own_name + " has no guide (the guides cover the ten guided league decks)";
        return p;
    }
    if (p.opp.empty() || !g.guided(p.opp)) {
        p.why = "the guides have no row against " + opp_name;
        return p;
    }
    const GuideRow *row = g.row(p.own, p.opp);
    if (!row) {
        p.why = "no guide row for " + p.own + " vs " + p.opp;
        return p;
    }
    p.found = true;
    p.holdout = g.holdout(p.own, p.opp);
    p.in = row->in;
    p.out = row->out;
    const std::string land = g.draw_land(p.own);
    std::vector<std::pair<std::string, int>> draw_in, draw_out;
    if (!land.empty() && !row->draw.empty()) {
        draw_in.emplace_back(row->draw, 1);
        draw_out.emplace_back(land, 1);
    }
    if (!on_play) {
        p.in.insert(p.in.end(), draw_in.begin(), draw_in.end());
        p.out.insert(p.out.end(), draw_out.begin(), draw_out.end());
    }
    if (!draw_in.empty())
        p.other =
            on_play ? "On the draw also: " + signed_lines(draw_out, "\xE2\x88\x92") + ", " + signed_lines(draw_in, "+")
                    : "On the play without the draw swap (" + signed_lines(draw_out, "\xE2\x88\x92") + ", " +
                          signed_lines(draw_in, "+") + ")";
    // The seat's own cards by key: front, back and both faces joined.
    const DeckCounts &reg = d.lists[static_cast<size_t>(seat)];
    std::map<std::string, int> by_key;
    std::map<int, int> all = reg.main;
    for (const auto &kv : reg.side) all[kv.first] += 0;
    for (const auto &kv : all) {
        const std::string front = names.name(kv.first), back = names.name(kv.first, true);
        by_key.emplace(card_key(front), kv.first);
        if (back != front) {
            by_key.emplace(card_key(back), kv.first);
            by_key.emplace(card_key(front + back), kv.first);
        }
    }
    for (const auto *lines : {&p.in, &p.out})
        for (const auto &c : *lines) {
            const auto it = by_key.find(card_key(c.first));
            if (it == by_key.end()) {
                if (p.unresolved.empty()) p.unresolved = c.first;
                continue;
            }
            p.ids[c.first] = it->second;
        }
    if (!p.unresolved.empty()) return p;
    // The target: the registered main deck minus the outs plus the ins.
    p.target = reg.main;
    for (const auto &c : p.out) add_count(&p.target, p.ids[c.first], -c.second);
    for (const auto &c : p.in) add_count(&p.target, p.ids[c.first], c.second);
    return p;
}

}  // namespace rmwork
