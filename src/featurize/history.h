#ifndef FEATURIZE_HISTORY_H
#define FEATURIZE_HISTORY_H

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "../state_format/decode.h"
#include "feature_spec.h"

// ── Recent-event history: one seat's observation events across its frames (docs/featurizer.md §3.7) ─
//
// Standard library and the decoder only. A STATE frame carries the events since the viewer's
// previous frame (docs/state_format.md §5); the featurizer reads one frame, so a spec with history
// types (FeatureOptions::history_events > 0) is fed by an EventHistory the caller keeps per seat and
// match: observe() every frame of the seat, in order (frames without a menu too), before
// featurizing it. What it holds depends only on the events received, not on how they were batched
// into frames, so a dump's frames, the RMSTATE stream's and the in-process feed's give the same
// tokens at a decision.
//
// It keeps the events since the last `game_start` (inclusive) whose turn is at least the latest
// frame's turn less one, each with its event_card rows; event_player rows are not kept.
// select_history picks a frame's history tokens (feature_spec.h): the events of the frame's turn
// and the previous one, newest first, at most HistoryPlan::events, each with its first
// HistoryPlan::cards_per_event cards in frame order. The rows are copies whose handle columns hold
// the handle the object has now (each old → new link of the later events followed; a handle that
// was retired matches no token) and whose event_card.event is its event's token index.

namespace featurize {

struct HistoryEvent {
        uint64_t turn = 0;
        std::vector<rmstate::Value> row;                 // the event row
        std::vector<std::vector<rmstate::Value>> cards;  // its event_card rows, in frame order
};

class EventHistory {
    public:
        // Appends the frame's events (a no-op for a spec without history).
        void observe(const FeatureSpec &spec, const rmstate::DecodedFrame &state);
        void clear() { events_.clear(); }
        const std::deque<HistoryEvent> &events() const { return events_; }

    private:
        std::deque<HistoryEvent> events_;
};

// One frame's history tokens: per event token its row, its turn offset (0 or 1) and the cards past
// the cap; per card token its row.
struct HistorySelection {
        std::deque<std::vector<rmstate::Value>> rows;  // the copies
        std::vector<const std::vector<rmstate::Value> *> events, cards;
        std::vector<uint8_t> prev_turn;
        std::vector<uint32_t> more_cards;
};

void select_history(
    const FeatureSpec &spec, const EventHistory &history, const rmstate::DecodedFrame &state, HistorySelection *out);

}  // namespace featurize

#endif /* FEATURIZE_HISTORY_H */
