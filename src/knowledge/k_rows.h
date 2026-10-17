#ifndef KNOWLEDGE_K_ROWS_H
#define KNOWLEDGE_K_ROWS_H

#include <cstdint>
#include <vector>

#include "../state_format/schema.h"
#include "frame_view.h"

// ── The knowledge tables as rows (docs/knowledge_store.md §6, state_format Appendix A.6) ──────
//
// Knowledge::rows() builds these from the model; write_k_tables appends them to a frame being
// built (tables k_meta … k_known_hand, ids 90-95). Seats are player indices (0 = A, 1 = B), written
// as the seat pair (absolute and relative to the viewer); cards are vocab ids or NO_CARD.

namespace knowledge {

struct KMetaRow {
        rmstate::KnowledgeLevel level = rmstate::KnowledgeLevel::OFF;
        bool exact = false;
        uint32_t steps = 0;
};

struct KSeenRow {
        int player = NO_SEAT;
        int card = NO_CARD;
        uint8_t game = 0;
        uint8_t match = 0;
};

struct KPoolRow {
        int owner = NO_SEAT;
        rmstate::PoolKind kind = rmstate::PoolKind::HAND;
        uint16_t size = 0;
        LibPos range;      // a library range; none otherwise
        uint32_t obj = 0;  // the face-down object's handle
};

struct KPoolCardRow {
        uint16_t pool = 0;  // row of k_pool
        int card = NO_CARD;
        uint8_t min = 0;
        uint8_t max = 0;
};

struct KPositionRow {
        int owner = NO_SEAT;
        LibPos at;
        int card = NO_CARD;               // the card there, when exact
        uint16_t pool = rmstate::NONE16;  // else the k_pool row of its candidates
};

struct KKnownHandRow {
        int card = NO_CARD;
        uint8_t copies = 0;
};

struct KRows {
        KMetaRow meta;  // always one row
        std::vector<KSeenRow> seen;
        std::vector<KPoolRow> pools;
        std::vector<KPoolCardRow> pool_cards;
        std::vector<KPositionRow> positions;
        std::vector<KKnownHandRow> known_hand;
};

// Appends `rows` to `w` for viewer seat `viewer`, with `codes` giving the seat enum values.
void write_k_tables(rmstate::FrameWriter &w, const KRows &rows, const Codes &codes, int viewer);

}  // namespace knowledge

#endif /* KNOWLEDGE_K_ROWS_H */
