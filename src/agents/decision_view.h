#ifndef AGENTS_DECISION_VIEW_H
#define AGENTS_DECISION_VIEW_H

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "seat_frames.h"

// ── One decision as an in-process agent sees it: the deciding seat's decoded STATE frame ──────
//
// The frame is the one the seat would receive on the RMSTATE stream (knowledge tables included),
// read with the seat's SeatStream. Menu row i is the reply i selects. Besides raw table access
// (frame(), table()) the view answers what the ported agents ask:
//
// - the explore tier (agents/explore.h) reads, like python/rm_agent.py's FrameView: each row's
//   category name, its card (card(i): the action's object's name as it is now, a stack object's
//   card, a back face played or cast as it, a named card; none for a player or a card-less
//   choice), the own hand's card names and the CARDS frame's mana values;
// - the --play resolver (agents/play_spec.h) reads each row's card id and controller
//   (menu_card_id / menu_controller) and its description.

namespace agents {

class DecisionView {
    public:
        DecisionView(const SeatStream &stream, const rmstate::DecodedFrame &frame, int seat);

        const SeatStream &stream() const { return stream_; }
        const rmstate::DecodedFrame &frame() const { return frame_; }
        Table table(const char *name) const { return Table(frame_, stream_.schema(), name); }
        // The deciding seat: 0 = Player A, 1 = Player B.
        int seat() const { return seat_; }

        // ── the menu ──
        size_t n() const { return menu_.rows(); }
        const std::string &category(size_t i) const;
        const std::string &description(size_t i) const;
        // The action's card name (explore's view; none for a player or a card-less choice).
        std::optional<std::string> card(size_t i) const;
        // The action's card id (-1 none): its source object's face as it is now on the
        // battlefield or the stack (object_face_id / stack_face_id), else the row's source card
        // (its back face for a back-face play or cast); none for a player source.
        int menu_card_id(size_t i) const;
        // The action's controller: 1 the viewer, 0 the opponent, -1 none (no object or player
        // source, or an object outside the battlefield, hand, stack, graveyard and exile).
        int menu_controller(size_t i) const;

        // ── objects ──
        // The own hand's card names (face-down cards left out), in object-table order.
        std::vector<std::string> hand() const;
        const CardNames &cards() const { return stream_.cards(); }
        const Table &objects() const { return object_; }
        const Table &stack() const { return stack_; }
        // The stack's rows, top (pos 0) first.
        const std::vector<size_t> &stack_order() const { return stack_by_pos_; }
        // An object row's card name as it is now: a token's card, a copy's copied card, a
        // transformed card's back face; none when its card is hidden.
        std::optional<std::string> object_name(size_t row) const;
        // A stack row's card name (explore's view): a spell's card, an ability's source as it is
        // now; none for a trigger with no source of its own.
        std::optional<std::string> stack_card(size_t row) const;

    private:
        // The card id of an object as it is now (a token's id, a copy's copied card, a transformed
        // card's back face) and of a stack row.
        int object_face_id(size_t row) const;
        int stack_face_id(size_t row) const;
        // The object row of handle `h` (-1 when the frame has none).
        int object_row(uint64_t h) const;
        int stack_row(uint64_t h) const;

        const SeatStream &stream_;
        const rmstate::DecodedFrame &frame_;
        int seat_;
        Table menu_, object_, stack_;
        // Columns.
        int m_category_, m_source_, m_source_card_, m_source_player_rel_, m_variant_, m_description_;
        int o_handle_, o_zone_, o_owner_rel_, o_controller_rel_, o_card_, o_token_, o_transformed_, o_copy_of_;
        int s_handle_, s_pos_, s_kind_, s_source_, s_source_card_, s_ab_card_, s_ab_key_, s_variant_;
        // Enum values.
        int z_battlefield_, z_stack_, z_hand_, z_graveyard_, z_exile_;
        int k_triggered_;
        uint64_t back_face_bit_;
        std::map<uint64_t, size_t> objects_;  // handle -> object row
        std::vector<size_t> stack_by_pos_;    // stack rows, top (pos 0) first
};

}  // namespace agents

#endif /* AGENTS_DECISION_VIEW_H */
