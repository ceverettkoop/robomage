#ifndef RMKNOW_TEST_STREAM_H
#define RMKNOW_TEST_STREAM_H

#include <cstdint>
#include <vector>

#include "../../src/knowledge/knowledge.h"

// Synthetic event streams for the knowledge sidecar's unit tests (selftest.cpp, exactness.cpp):
// events with consecutive seqs, applied to a Knowledge as STATE frames.

namespace rmknow_test {

using knowledge::Event;
using knowledge::EventCard;
using knowledge::Knowledge;
using knowledge::LibPos;
using knowledge::NO_CARD;
using knowledge::Snapshot;
using knowledge::Zn;
using rmstate::EventKind;

// Writes events with consecutive seqs and builds frames from them.
class Stream {
    public:
        explicit Stream(Knowledge &k, uint32_t seq = 0) : know_(k), seq_(seq) {}
        uint32_t seq() const { return seq_; }
        Event ev(EventKind kind, uint16_t turn = 1) {
            Event e;
            e.seq = ++seq_;
            e.kind = kind;
            e.turn = turn;
            return e;
        }
        void game_start(int lib_a, int lib_b, uint8_t game = 0, int sb_a = 0, int sb_b = 0) {
            Event e = ev(EventKind::GAME_START, 0);
            e.game = game;
            e.players.push_back({0, static_cast<uint16_t>(lib_a), static_cast<uint16_t>(sb_a), 0});
            e.players.push_back({1, static_cast<uint16_t>(lib_b), static_cast<uint16_t>(sb_b), 0});
            add(e);
        }
        void game_end() { add(ev(EventKind::GAME_END)); }
        uint32_t draw(int p, int card = NO_CARD, uint32_t handle = 0, bool opening = false, uint16_t turn = 1) {
            Event e = ev(EventKind::DRAW, turn);
            e.player = p;
            e.zone = Zn::LIBRARY;
            e.to_zone = Zn::HAND;
            e.zone_owner = e.to_owner = p;
            e.count = 1;
            e.opening = opening;
            EventCard c;
            c.card = card;
            c.handle_after = handle;
            c.from = top(0);
            e.cards.push_back(c);
            add(e);
            return e.seq;
        }
        void move(EventKind kind, int p, Zn from, Zn to, EventCard c, uint16_t batch = 0, uint16_t turn = 1) {
            Event e = ev(kind, turn);
            e.zone = from;
            e.to_zone = to;
            e.zone_owner = e.to_owner = p;
            e.count = 1;
            e.batch = batch;
            e.cards.push_back(c);
            add(e);
        }
        void shown(EventKind kind, int p, Zn zone, const std::vector<EventCard> &cards, uint16_t turn = 1) {
            Event e = ev(kind, turn);
            e.zone = zone;
            e.zone_owner = p;
            e.count = static_cast<uint16_t>(cards.size());
            e.cards = cards;
            add(e);
        }
        void shuffle(int p) {
            Event e = ev(EventKind::SHUFFLE);
            e.player = e.zone_owner = p;
            e.zone = Zn::LIBRARY;
            add(e);
        }
        void library_place(int p, uint16_t batch, LibPos range) {
            Event e = ev(EventKind::LIBRARY_PLACE);
            e.player = e.zone_owner = p;
            e.zone = Zn::LIBRARY;
            e.batch = batch;
            e.range = range;
            e.count = range.span;
            add(e);
        }
        // Applies the events written since the last frame as one STATE frame (snapshot checks off:
        // a terminal frame).
        void frame() {
            Snapshot s;
            s.terminal = true;
            s.events.swap(pending_);
            know_.apply_frame(s);
        }
        static LibPos top(int pos, int span = 1) { return pos_of(rmstate::LibAnchor::TOP, pos, span); }
        static LibPos bottom(int pos, int span = 1) { return pos_of(rmstate::LibAnchor::BOTTOM, pos, span); }
        static EventCard card(int id, uint32_t hb = 0, uint32_t ha = 0, LibPos from = LibPos{}, LibPos to = LibPos{}) {
            EventCard c;
            c.card = id;
            c.handle_before = hb;
            c.handle_after = ha;
            c.from = from;
            c.to = to;
            return c;
        }

    private:
        static LibPos pos_of(rmstate::LibAnchor anchor, int pos, int span) {
            LibPos p;
            p.anchor = static_cast<uint8_t>(anchor);
            p.pos = static_cast<uint16_t>(pos);
            p.span = static_cast<uint16_t>(span);
            return p;
        }
        void add(const Event &e) { pending_.push_back(e); }
        Knowledge &know_;
        uint32_t seq_ = 0;
        std::vector<Event> pending_;
};

}  // namespace rmknow_test

#endif /* RMKNOW_TEST_STREAM_H */
