#include "history.h"

#include <map>

using Row = std::vector<rmstate::Value>;
using Links = std::map<uint64_t, uint64_t>;

static uint64_t frame_turn(const featurize::HistoryPlan &h, const rmstate::DecodedFrame &state,
    const std::deque<featurize::HistoryEvent> &events);
static Links handle_links(const featurize::HistoryPlan &h, const std::deque<featurize::HistoryEvent> &events);
static uint64_t follow(const Links &links, uint64_t handle);
static void forward_cell(const Links &links, Row *row, int col);

// The frame's turn (the turn table's), else the newest event's.
static uint64_t frame_turn(const featurize::HistoryPlan &h, const rmstate::DecodedFrame &state,
    const std::deque<featurize::HistoryEvent> &events) {
    const rmstate::DecodedTable *t = state.table(h.turn_table);
    if (t && !t->rows.empty()) return t->rows[0][static_cast<size_t>(h.turn_turn)].bits;
    return events.empty() ? 0 : events.back().turn;
}

// Every old → new handle link of the kept events' cards. A handle is never issued again, so a link
// applies to any event that names the old handle.
static Links handle_links(const featurize::HistoryPlan &h, const std::deque<featurize::HistoryEvent> &events) {
    Links links;
    for (const featurize::HistoryEvent &e : events)
        for (const Row &c : e.cards) {
            const uint64_t before = c[static_cast<size_t>(h.card_before)].bits;
            const uint64_t after = c[static_cast<size_t>(h.card_after)].bits;
            if (before != 0 && after != 0 && before != after) links[before] = after;
        }
    return links;
}

// The handle `handle` became after every link (bounded: the links form chains).
static uint64_t follow(const Links &links, uint64_t handle) {
    for (size_t steps = 0; handle != 0 && steps < links.size(); ++steps) {
        const auto it = links.find(handle);
        if (it == links.end()) break;
        handle = it->second;
    }
    return handle;
}

static void forward_cell(const Links &links, Row *row, int col) {
    rmstate::Value &v = (*row)[static_cast<size_t>(col)];
    v.bits = follow(links, v.bits);
}

namespace featurize {

void EventHistory::observe(const FeatureSpec &spec, const rmstate::DecodedFrame &state) {
    const HistoryPlan &h = spec.history;
    if (!h.on()) return;
    const rmstate::DecodedTable *ev = state.table(h.event_table);
    if (ev && !ev->rows.empty()) {
        // Each frame row's place in events_ (-1: before a later game_start of the same frame).
        std::vector<int64_t> at(ev->rows.size(), -1);
        for (size_t r = 0; r < ev->rows.size(); ++r) {
            const Row &row = ev->rows[r];
            if (row[static_cast<size_t>(h.event_kind)].bits == h.game_start_kind) {
                events_.clear();
                for (size_t q = 0; q < r; ++q) at[q] = -1;
            }
            at[r] = static_cast<int64_t>(events_.size());
            events_.emplace_back();
            events_.back().turn = row[static_cast<size_t>(h.event_turn)].bits;
            events_.back().row = row;
        }
        const rmstate::DecodedTable *cards = state.table(h.card_table);
        if (cards)
            for (const Row &c : cards->rows) {
                const uint64_t r = c[static_cast<size_t>(h.card_event)].bits;
                if (r < at.size() && at[r] >= 0) events_[static_cast<size_t>(at[r])].cards.push_back(c);
            }
    }
    const uint64_t turn = frame_turn(h, state, events_);
    while (!events_.empty() && events_.front().turn + 1 < turn) events_.pop_front();
}

void select_history(
    const FeatureSpec &spec, const EventHistory &history, const rmstate::DecodedFrame &state, HistorySelection *out) {
    out->rows.clear();
    out->events.clear();
    out->cards.clear();
    out->prev_turn.clear();
    out->more_cards.clear();
    const HistoryPlan &h = spec.history;
    if (!h.on()) return;
    const std::deque<HistoryEvent> &events = history.events();
    const uint64_t turn = frame_turn(h, state, events);
    const Links links = handle_links(h, events);
    std::vector<const HistoryEvent *> picked;
    for (auto it = events.rbegin(); it != events.rend() && picked.size() < h.events; ++it)
        if (it->turn + 1 >= turn) picked.push_back(&*it);
    for (size_t k = 0; k < picked.size(); ++k) {
        const HistoryEvent &e = *picked[k];
        out->rows.push_back(e.row);
        forward_cell(links, &out->rows.back(), h.event_source);
        out->events.push_back(&out->rows.back());
        out->prev_turn.push_back(e.turn < turn ? 1 : 0);
        out->more_cards.push_back(
            e.cards.size() > h.cards_per_event ? static_cast<uint32_t>(e.cards.size() - h.cards_per_event) : 0);
    }
    for (size_t k = 0; k < picked.size(); ++k) {
        const std::vector<Row> &cards = picked[k]->cards;
        for (size_t c = 0; c < cards.size() && c < h.cards_per_event; ++c) {
            out->rows.push_back(cards[c]);
            Row &row = out->rows.back();
            forward_cell(links, &row, h.card_before);
            forward_cell(links, &row, h.card_after);
            row[static_cast<size_t>(h.card_event)].bits = k;
            out->cards.push_back(&row);
        }
    }
}

}  // namespace featurize
