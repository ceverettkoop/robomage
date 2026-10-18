#ifndef AGENTS_SEAT_FRAMES_H
#define AGENTS_SEAT_FRAMES_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../state_format/decode.h"
#include "../state_format/schema.h"

// ── One seat's RMSTATE frames, decoded, as the in-process agents read them ────────────────────
//
// Standard library and the state format's decoder only (like src/knowledge/): an agent sees
// nothing but the frames its seat receives. Tables, columns and enum values are found by name in
// the stream's SCHEMA frame, cards by id in its CARDS frame.
//
// Relative seat columns (`<seat>_rel`) read 1 for the viewer, 2 for the opponent, 0 for none.

namespace agents {

constexpr int REL_NONE = 0;
constexpr int REL_SELF = 1;
constexpr int REL_OPP = 2;

// One table of a decoded frame with its columns found by name (a table the frame does not carry
// has no rows; a column the schema lacks reads 0, or "" as text).
class Table {
    public:
        Table() = default;
        Table(const rmstate::DecodedFrame &frame, const rmstate::Schema &schema, const char *name);
        size_t rows() const { return table_ ? table_->rows.size() : 0; }
        // The column's index, or -1.
        int col(const char *name) const;
        uint64_t u(size_t row, int col) const;
        int64_t s(size_t row, int col) const;
        const std::string &text(size_t row, int col) const;
        // A `card` column's value as an id, -1 for none.
        int card(size_t row, int col) const;
        // Whether the cell holds its column's none value (all ones of its width: a card, index,
        // row: or nullable enum column); true for a column the schema lacks.
        bool none(size_t row, int col) const;

    private:
        const rmstate::DecodedTable *table_ = nullptr;
        const rmstate::TableDef *def_ = nullptr;
};

// The CARDS frame's facts the agents read: per card id its front and back face names (the first
// row of each face) and its back face's card id (the last non-front row's face_card), per face
// card id its face's name (front rows first), and per face name the first row with that name's
// mana value.
class CardNames {
    public:
        void read(const rmstate::DecodedFrame &cards, const rmstate::Schema &schema);
        // The name of card `cid`'s front (or back) face; "#<cid>" for an id the frame does not list.
        // `cid` must not be negative.
        std::string name(int cid, bool back = false) const;
        // The card id of `cid`'s back face (`cid` itself when it has no back face; -1 when that
        // face has no card id).
        int back_face(int cid) const;
        // The name of the face whose own card id (face_card) is `face_card`; "#<id>" when no row has it.
        std::string face_name(int face_card) const;
        // The printed mana value of the first face row named `name` (0 when there is none).
        double mana_value(const std::string &name) const;

    private:
        std::map<int, std::string> front_;
        std::map<int, std::string> back_;
        std::map<int, int> back_card_;
        std::map<int, std::string> face_name_;
        std::map<std::string, double> mana_value_;
};

// One seat's frame stream: the schema and the once-per-match frames, which every STATE frame is
// read with. Not copyable: the kept frames' tables point into its schema.
class SeatStream {
    public:
        SeatStream() = default;
        SeatStream(const SeatStream &) = delete;
        SeatStream &operator=(const SeatStream &) = delete;
        // Takes one of the seat's once-per-match frames (SCHEMA first). False (with `err`) on a frame
        // that does not decode.
        bool feed_match_frame(const std::string &payload, std::string *err);
        // Decodes one of the seat's STATE frames into `out`.
        bool decode_state(const std::string &payload, rmstate::DecodedFrame *out, std::string *err) const;
        bool ready() const { return have_schema_; }
        const rmstate::Schema &schema() const { return schema_; }
        const CardNames &cards() const { return cards_; }
        // The latest CARDS, DECKLISTS and METAGAME frames, decoded (no tables before the first).
        const rmstate::DecodedFrame &cards_frame() const { return cards_frame_; }
        const rmstate::DecodedFrame &decklists_frame() const { return decklists_frame_; }
        const rmstate::DecodedFrame &metagame_frame() const { return metagame_frame_; }
        // The value of `name` in enum `enum_name`, or -1.
        int enum_value(const char *enum_name, const char *name) const;
        // The name of value `v` of enum `enum_name` ("?" when out of range).
        const std::string &enum_name(const char *enum_name, uint64_t v) const;
        // The string table of enum `enum_name` (empty when the schema has none).
        const std::vector<std::string> &enum_values(const char *enum_name) const;

    private:
        bool have_schema_ = false;
        rmstate::Schema schema_;
        CardNames cards_;
        rmstate::DecodedFrame cards_frame_, decklists_frame_, metagame_frame_;
};

}  // namespace agents

#endif /* AGENTS_SEAT_FRAMES_H */
