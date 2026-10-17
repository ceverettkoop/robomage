#ifndef KNOWLEDGE_K_ROWS_H
#define KNOWLEDGE_K_ROWS_H

#include <cstdint>
#include <string>
#include <vector>

#include "../state_format/decode.h"
#include "../state_format/schema.h"
#include "frame_view.h"

// ── The knowledge tables as rows (docs/knowledge_store.md §6, state_format Appendix A.6) ──────
//
// Knowledge::rows() builds these from the model; write_k_tables appends them to a frame being
// built (tables k_meta … k_archetype, ids 90-97, and the belief layer's b_list and b_archetype,
// 98-99). Seats are player indices (0 = A, 1 = B), written as the seat pair (absolute and relative
// to the viewer); cards are vocab ids or NO_CARD.

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

// One reference list (a METAGAME meta_list row) for one player.
struct KTemplateRow {
        int player = NO_SEAT;
        uint16_t list = 0;
        bool consistent = false;
        bool exact = false;       // `consistent` is proven
        bool registered = false;  // the list is the classification of the player's registered list
};

// One archetype label (a METAGAME meta_archetype row) for one player.
struct KArchetypeRow {
        int player = NO_SEAT;
        uint8_t archetype = 0;
        bool possible = false;  // some consistent list has this label
        bool exact = false;     // possible, and every consistent list has it
};

// The belief layer (belief.h; level expert+belief): one reference list's posterior for one player.
struct BListRow {
        int player = NO_SEAT;
        uint16_t list = 0;
        uint32_t prob = 0;   // units of 10^-9 (PROB_SCALE)
        int32_t loglik = 0;  // log-likelihood, milli-nats; INT32_MIN when the list is excluded
        bool exact = false;  // the likelihood is the exact weighted count in every game of the match
};

// One archetype label's posterior (the sum of its lists') for one player.
struct BArchetypeRow {
        int player = NO_SEAT;
        uint8_t archetype = 0;
        uint32_t prob = 0;
};

struct KRows {
        KMetaRow meta;  // always one row
        std::vector<KSeenRow> seen;
        std::vector<KPoolRow> pools;
        std::vector<KPoolCardRow> pool_cards;
        std::vector<KPositionRow> positions;
        std::vector<KKnownHandRow> known_hand;
        std::vector<KTemplateRow> templates;
        std::vector<KArchetypeRow> archetypes;
        std::vector<BListRow> belief_lists;
        std::vector<BArchetypeRow> belief_archetypes;
};

// Appends `rows` to `w` for viewer seat `viewer`, with `codes` giving the seat enum values.
void write_k_tables(rmstate::FrameWriter &w, const KRows &rows, const Codes &codes, int viewer);

// The k_* tables a decoded STATE frame carries, read back into rows (the inverse of
// write_k_tables); false (with `err`) when a table lacks a column or the frame has no k_meta row.
bool read_k_tables(const rmstate::DecodedFrame &frame, const Codes &codes, KRows *out, std::string *err);

}  // namespace knowledge

#endif /* KNOWLEDGE_K_ROWS_H */
