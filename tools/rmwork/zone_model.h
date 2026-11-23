#ifndef RMWORK_ZONE_MODEL_H
#define RMWORK_ZONE_MODEL_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "../../src/agents/seat_frames.h"
#include "card_book.h"

// ── A player's hand, library, graveyard and exile as one viewer knows them ─────────────────────
//
// Built from one viewer's STATE frame (and its stream's CARDS frame through a CardBook) only:
//   - graveyard and exile: the frame's objects in that zone, newest first (the `pos` column, 0 the
//     top); a face-down exiled card is listed without a name; an exiled card's linked source
//     (`object_link`) is noted by the name the frame gives it;
//   - library: the omniscient viewer's every card in order from the top; a seat's only the cards it
//     knows — the library objects its frame lists (a card it looks at or that is revealed) and its
//     knowledge's exact positions (`k_position` rows naming a card) — each at its position from the
//     top, the rest a count;
//   - hand: the omniscient viewer's and a seat's own hand in full; the opponent's in a seat view only
//     the cards the frame shows (revealed) and those its knowledge holds as certainly there
//     (`k_known_hand`, the least copies), the rest a count.
// Every count is the `player` row's zone size.

namespace rmwork {

enum class ZoneKind {
    HAND,
    LIBRARY,
    GRAVEYARD,
    EXILE
};
inline constexpr int ZONE_KINDS = 4;

// "Hand", "Library", "Graveyard", "Exile"; "hand", … for keys.
const char *zone_title(ZoneKind z);
const char *zone_key(ZoneKind z);

struct ZoneCard {
        uint64_t handle = 0;  // the viewer's handle; 0 for a card known only to its knowledge
        int card = -1;        // -1: not identified (a face-down exiled card)
        bool back = false;
        std::string name;  // "" when not identified
        bool token = false, face_down = false;
        uint8_t colors = 0;  // WUBRG bits
        bool land = false;
        std::vector<std::string> subtypes;  // a land's, for its basic land types
        int pos = -1;                       // a library card's position from the top (0 the top), else -1
        int copies = 1;                     // a known card of the opponent's hand: its copies
        std::string note;                   // "exiled with Ephemerate", "known", "revealed", "seen on top"
        std::vector<std::string> counters;  // "Time 3"
};

struct ZoneList {
        int seat = 0;
        ZoneKind kind = ZoneKind::HAND;
        int size = 0;  // the player row's count
        std::vector<ZoneCard> cards;
        int hidden = 0;          // cards the viewer cannot name: size minus the listed copies
        bool all_known = false;  // every card is listed (the cards in order where the zone has one)
        // Cards listed (with their copies).
        int listed() const;
};

struct ZoneModel {
        std::array<std::array<ZoneList, ZONE_KINDS>, 2> zones;
        const ZoneList &get(int seat, ZoneKind z) const {
            return zones[static_cast<size_t>(seat)][static_cast<size_t>(z)];
        }
};

// The zones of `frame`, a STATE frame of viewer `viewer` (0 A, 1 B, 2 omniscient) read with
// `stream`, whose CARDS frame `book` was built from.
ZoneModel build_zone_model(
    const agents::SeatStream &stream, const CardBook &book, const rmstate::DecodedFrame &frame, int viewer);

}  // namespace rmwork

#endif /* RMWORK_ZONE_MODEL_H */
