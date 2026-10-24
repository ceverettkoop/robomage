#ifndef FEATURIZE_CARD_TABLES_H
#define FEATURIZE_CARD_TABLES_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../state_format/decode.h"
#include "registry.h"
#include "rmtb.h"

// ── The card, ability and part tables (docs/featurizer.md §4), standard library and the decoder only
//
// Whole-vocab tables, the same for every match, built from a whole-vocab CARDS frame (the engine's
// --cards-frame FILE) by the tag rules of columns.h, every column and row at its layout registry
// position (registry.h; a reserved position is 0):
//
//   card     [card_rows, Fc]: one row per face vocab id (the `card` row whose face_card is the id,
//            front rows first): the row's scalar columns, its masks bit by bit and its enums one-hot;
//            per (card, face) child table of CARDS (card_subtype, card_keyword, card_pip) the count of
//            the face's rows per value of each enum column; n_abilities (log1p of the face's
//            card_ability rows); present. card_rows is the text table's row count (N_CARD_TYPES + 1),
//            so the last row is the none row (all zeros) and nothing here reads an engine header;
//   ability  [Nab, Fa]: card id c's ability of index i at row c·A + i (A = the registry's
//            abilities_per_card; a row no ability has is all zeros), then the key band from key_base
//            = (card_rows - 1)·A: per fixed key (ab_keys) its registry row in the band; the columns by
//            the tag rules (the row's card and index are its address, not features) and `keyed` (1
//            for a key row). A key row that is also some card_ability row's key (a keyword-built
//            ability: Equip, Cycling) has the features of the first such row in (card, index) order;
//            the others (an engine-built definition no frame describes) only `keyed`. A card with an
//            ability index ≥ A is refused (a layout bump);
//   part     [Npart, Fp]: one row per card_ability_part row, in (card, ability, part) order, its
//            columns by the tag rules (card, ability and part are its address) and the ability row
//            it belongs to (`part_ability`, not a feature).
//
// The fixed key list is every distinct non-empty card_ability key of the frame, plus the key forms
// Appendix A.1 documents for engine-built definitions that the SCHEMA frame enumerates:
// `keyword:<k>` for each `keyword` value (keyword-derived triggers), `subtype_mana:<c>` for each
// `color` value (a basic land type's mana ability) and `LandType`; in key-row order. Other keys an
// abref can carry (`granted_trigger:<script>`, `enchant:<filter>`, the engine's delayed-trigger
// keys) have no row: the tensors' ab_key categorical still hashes them.
//
// Files (RMTB, §4.4): kind `card` (`features`, `back_face`), `ability` (`features`, `ab_card`,
// `ab_index`, `key_rows` whose columns are the keys; attrs abilities_per_card and key_base) and `part`
// (`features`, `ability`), each with vocab_hash and spec_hash.
//
// Beside the tables, the printed rows the object deltas compare with (docs/featurizer.md §3.6):
// per (card, face) child table of CARDS and face vocab id, the rows of the face (the columns after
// card and face), sorted. They are not exported: build_printed_rows makes them from the CARDS frame
// the tables were built from.

namespace featurize {

// A (card, face) child table's printed rows per face vocab id (card_subtype: its subtypes).
struct PrintedRows {
        std::string table;                              // card_subtype, card_keyword, card_pip
        std::vector<std::string> columns;               // the columns after card and face
        std::vector<uint64_t> offsets;                  // [card_rows + 1]: face f's rows are rows[offsets[f] …)
        std::vector<std::vector<rmstate::Value>> rows;  // per face sorted by printed_row_less; a text
                                                        // column's bits (a string-pool offset) are 0
};

// The order of a printed row's values: column by column, text by its bytes, then by bits.
bool printed_row_less(const std::vector<rmstate::Value> &a, const std::vector<rmstate::Value> &b);

struct CardTables {
        uint64_t vocab_hash = 0;
        uint64_t card_rows = 0;  // N_CARD_TYPES + 1
        uint64_t text_rows = 0;  // the text table's shape (0 when not given)
        uint64_t text_width = 0;
        std::vector<std::string> card_columns;
        std::vector<float> card;         // [card_rows, card_columns.size()]
        std::vector<int64_t> back_face;  // [card_rows]: a physical card's back face id (itself with none, -1: no id)
        std::vector<std::string> ability_columns;
        std::vector<float> ability;          // [ability_rows(), ability_columns.size()]
        std::vector<int64_t> ability_card;   // per row: its card (-1 for a key row or an unused row)
        std::vector<int64_t> ability_index;  // per row: its index in the card's list (-1 likewise)
        uint32_t abilities_per_card = 0;     // A: card c's ability i is row c·A + i
        uint64_t key_base = 0;               // the key band's first row
        std::vector<std::string> ab_keys;    // in row order
        std::vector<int64_t> key_rows;       // per key: its ability row
        std::vector<std::string> part_columns;
        std::vector<float> part;            // [part_rows(), part_columns.size()]
        std::vector<int64_t> part_ability;  // per part row: its ability row
        // Per face vocab id whether a card row describes it, and the printed child rows.
        std::vector<uint8_t> present;
        std::vector<PrintedRows> printed;
        // Lookup (rebuilt by build_index): per key its ability row.
        std::map<std::string, int64_t> key_index;

        size_t ability_rows() const { return ability_card.size(); }
        size_t part_rows() const { return part_ability.size(); }
        // The ability row of (card, index), or -1.
        int64_t ability_row(uint64_t card, uint64_t index) const;
        // The ability row of key `key`, or -1.
        int64_t key_row(const std::string &key) const;
        // The printed rows of child table `table`, or nullptr.
        const PrintedRows *printed_rows(const std::string &table) const;
        // Rebuilds key_index; false when a row's card and index are not its address, a card's
        // indices are not 0 … n-1 or a key row is outside the key band.
        bool build_index();
};

// The three tables' RMTB files.
struct CardTableFiles {
        RmtbFile card, ability, part;
};

// Builds the tables from a whole-vocab CARDS frame decoded with `schema`, at the positions of
// `registry` (with `append`, adding what it lacks). False with `err` when the frame does not have the
// CARDS tables, a card's ability rows are not 0 … n-1 or past abilities_per_card, or the registry
// lacks an entry or a capacity.
bool build_card_tables(const rmstate::Schema &schema, const rmstate::DecodedFrame &cards, uint64_t card_rows,
    uint64_t vocab_hash, Registry *registry, CardTables *out, std::string *err);
// The same at the compiled-in registry's positions.
bool build_card_tables(const rmstate::Schema &schema, const rmstate::DecodedFrame &cards, uint64_t card_rows,
    uint64_t vocab_hash, CardTables *out, std::string *err);

// The presence flags and printed rows of `out` (card_rows set) from the whole-vocab CARDS frame;
// build_card_tables calls it, a reader of the exported files calls it with the cards file's frame.
void build_printed_rows(const rmstate::Schema &schema, const rmstate::DecodedFrame &cards, CardTables *out);

// The tables as RMTB files, stamped with `spec_hash`.
CardTableFiles card_table_files(const CardTables &t, uint64_t spec_hash);

// The tables read back from their files (text_rows / text_width are left 0, the printed rows
// empty: build_printed_rows). False with `err` on a file of another kind, a missing array or a
// shape that does not agree.
bool read_card_tables(const CardTableFiles &files, CardTables *out, std::string *err);

}  // namespace featurize

#endif /* FEATURIZE_CARD_TABLES_H */
