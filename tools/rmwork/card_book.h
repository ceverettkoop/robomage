#ifndef RMWORK_CARD_BOOK_H
#define RMWORK_CARD_BOOK_H

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../../src/agents/seat_frames.h"

// ── The printed faces of one viewer's CARDS frame ────────────────────────────────────────────────
//
// Per card id and face (front, back) what the card renderer prints: name, mana cost, type line,
// oracle text, printed P/T and loyalty, colours, and whether it is a land, creature, planeswalker
// or token. Read from one viewer's stream only, so a face is known exactly when that viewer's
// CARDS frame lists it (a CARDS frame lists both registered decklists' cards and their tokens,
// which every viewer is shown; it names no object).

namespace rmwork {

// Colour bits in WUBRG order.
inline constexpr uint8_t COLOR_W = 1, COLOR_U = 2, COLOR_B = 4, COLOR_R = 8, COLOR_G = 16;

struct CardFace {
        int card = -1;  // the physical card's id (a double-faced card's front face)
        bool back = false;
        std::string name;
        std::string mana_cost;  // "{2}{U}{U}", "" for a land
        std::string type_line;  // "Legendary Creature — Human Wizard"
        std::string oracle;
        int power = 0, toughness = 0;
        bool has_pt = false;  // a creature, or a Vehicle with printed P/T
        int loyalty = 0;      // printed; 0 for none
        uint8_t colors = 0;   // WUBRG bits
        bool land = false, creature = false, planeswalker = false, token = false;
        std::vector<std::string> subtypes;
};

// Maps the schema's enums to the renderer's terms; built from one stream's SCHEMA frame.
struct TypeBits {
        uint32_t land = 0, creature = 0, planeswalker = 0;
        // Per colour enum value 0..: its WUBRG bit (0 for colourless and the others).
        std::vector<uint8_t> color_bit;
        // The `mask:color` value as WUBRG bits.
        uint8_t wubrg(uint64_t mask) const;
        void read(const agents::SeatStream &s);
};

class CardBook {
    public:
        // Reads `s`'s CARDS frame (and its SCHEMA frame's enums); a stream with no CARDS frame
        // gives an empty book.
        void build(const agents::SeatStream &s);
        // Face `back` of card `card`, nullptr when the CARDS frame has no such row.
        const CardFace *face(int card, bool back) const;
        size_t size() const { return faces_.size(); }
        const TypeBits &bits() const { return bits_; }
        // The stream the book was built from (nullptr before build).
        const agents::SeatStream *stream() const { return stream_; }

    private:
        std::map<std::pair<int, bool>, CardFace> faces_;
        TypeBits bits_;
        const agents::SeatStream *stream_ = nullptr;
};

// "Legendary Artifact Creature — Equipment" from the masks and subtype names, in printed order
// (supertypes, then the card types Kindred, Enchantment, Artifact, Land, Creature, Planeswalker, …).
std::string type_line_of(
    const agents::SeatStream &s, uint64_t supertypes, uint64_t types, const std::vector<std::string> &subtypes);

// A mana cost's symbols: "{2}{U}{U}" -> {"2", "U", "U"}; "{W/U}" -> {"W/U"}.
std::vector<std::string> mana_symbols(const std::string &cost);

}  // namespace rmwork

#endif /* RMWORK_CARD_BOOK_H */
