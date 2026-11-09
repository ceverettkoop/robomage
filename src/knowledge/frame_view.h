#ifndef KNOWLEDGE_FRAME_VIEW_H
#define KNOWLEDGE_FRAME_VIEW_H

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../state_format/decode.h"
#include "../state_format/schema.h"

// ── The knowledge sidecar's view of one viewer's RMSTATE frames (docs/knowledge_store.md §1) ───
//
// Decoded frames (state_format/decode.h) read into the plain structs the model consumes: the
// events of a STATE frame and the few snapshot facts the per-frame consistency check compares
// against, plus the DECKLISTS and CARDS facts the model's roots and text rendering need. Columns
// are found by name in the frame's schema, and the engine's enums (zones, seats) by their names
// in the SCHEMA frame's string tables, so nothing here depends on an engine header.
//
// Seats are player indices (0 = A, 1 = B, NO_SEAT = none); card ids are the vocab id or NO_CARD.

namespace knowledge {

constexpr int NO_CARD = -1;
constexpr int NO_SEAT = -1;

// Zones as the model names them. NONE is zone none (0xFF: created, ceased to exist); OTHER is a
// zone value the model does not know.
enum class Zn : uint8_t {
    LIBRARY,
    BATTLEFIELD,
    HAND,
    STACK,
    GRAVEYARD,
    EXILE,
    SIDEBOARD,
    NONE,
    OTHER,
};

bool is_public_zone(Zn z);

// The engine enum values the frames use, resolved by name from a schema.
struct Codes {
        std::vector<Zn> zone_of;         // zone enum value → Zn
        uint8_t seat_value[2] = {1, 2};  // `seat` enum values of PLAYER_A, PLAYER_B
        uint8_t seat_none = 0;           // the `seat` enum value of UNKNOWN
};

// Library position (state_format §5): anchor (rmstate::LibAnchor), pos from that end, span (1
// exact, n one of n, 0 anywhere). anchor NONE8 = not in a library.
struct LibPos {
        uint8_t anchor = rmstate::NONE8;
        uint16_t pos = rmstate::NONE16;
        uint16_t span = 0;
        bool in_library() const { return anchor != rmstate::NONE8; }
        bool bottom() const { return anchor == static_cast<uint8_t>(rmstate::LibAnchor::BOTTOM); }
};

struct EventCard {
        uint32_t handle_before = 0;
        uint32_t handle_after = 0;
        int card = NO_CARD;
        LibPos from;
        LibPos to;
        bool face_down = false;
        bool token = false;
        uint8_t hand_subset = rmstate::NONE8;
        uint32_t draw_seq = 0;
        uint16_t draw_idx = rmstate::NONE16;
};

struct EventPlayer {
        int player = NO_SEAT;
        uint16_t library = 0;
        uint16_t sideboard = 0;
        uint8_t wins = 0;
};

struct Event {
        uint32_t seq = 0;
        rmstate::EventKind kind = rmstate::EventKind::ZONE_MOVE;
        int player = NO_SEAT;
        Zn zone = Zn::NONE;
        Zn to_zone = Zn::NONE;
        int zone_owner = NO_SEAT;
        int to_owner = NO_SEAT;
        uint16_t count = 0;
        LibPos range;
        bool order_known = false;
        uint16_t batch = 0;
        int to = NO_SEAT;
        uint32_t source = 0;
        int source_card = NO_CARD;
        uint16_t turn = 0;
        bool opening = false;
        uint8_t game = 0;
        std::vector<EventCard> cards;
        std::vector<EventPlayer> players;
};

// One registered or current list entry: a card's main and side copies for a player.
struct DeckEntry {
        int player = NO_SEAT;
        int card = NO_CARD;
        int main = 0;
        int side = 0;
};

// The DECKLISTS frame: the registered lists present (only the viewer's own under
// --closed-decklist), with their deck names.
struct Decklists {
        bool present[2] = {false, false};
        std::string name[2];
        std::vector<DeckEntry> cards;
};

struct SnapPlayer {
        int player = NO_SEAT;
        uint16_t hand = 0;
        uint16_t library = 0;
        uint16_t exile = 0;
        uint16_t sideboard = 0;
};

struct SnapObject {
        uint32_t handle = 0;
        Zn zone = Zn::NONE;
        int owner = NO_SEAT;
        uint16_t pos = rmstate::NONE16;
        int card = NO_CARD;
        bool token = false;
        bool face_down = false;
};

struct SnapReveal {
        uint32_t handle = 0;
        Zn zone = Zn::NONE;
        int owner = NO_SEAT;
        uint16_t pos = rmstate::NONE16;
        int card = NO_CARD;
};

// What one STATE frame tells the model: its events and the snapshot facts the consistency check
// reads.
struct Snapshot {
        uint8_t game = 0;
        uint32_t decision = 0;
        bool terminal = false;
        bool sideboard_phase = false;
        bool closed_decklist = false;
        std::vector<SnapPlayer> players;
        std::vector<SnapObject> objects;
        std::vector<SnapReveal> reveals;
        std::vector<uint32_t> stack_non_cards;  // stack handles that are abilities or spell copies (sorted)
        std::vector<DeckEntry> own_deck;
        std::vector<Event> events;
};

// Frame header fields of a payload without decoding it (false when it is too short or not RMSF).
bool peek_frame_header(const std::string &payload, rmstate::FrameKind *kind, rmstate::Viewer *viewer);

// The engine enum values of `schema`; false (with `err`) when an enum the model needs is missing.
bool read_codes(const rmstate::Schema &schema, Codes *out, std::string *err);
// A STATE frame's events and snapshot facts; false (with `err`) when a table lacks a column the
// model reads.
bool read_snapshot(const rmstate::DecodedFrame &frame, const Codes &codes, Snapshot *out, std::string *err);
bool read_decklists(const rmstate::DecodedFrame &frame, const Codes &codes, Decklists *out, std::string *err);
// Card names from a CARDS frame (the front face's name per card id).
bool read_card_names(const rmstate::DecodedFrame &frame, std::map<int, std::string> *out, std::string *err);

}  // namespace knowledge

#endif /* KNOWLEDGE_FRAME_VIEW_H */
