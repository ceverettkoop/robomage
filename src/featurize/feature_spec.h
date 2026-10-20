#ifndef FEATURIZE_FEATURE_SPEC_H
#define FEATURIZE_FEATURE_SPEC_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "../state_format/decode.h"
#include "card_tables.h"
#include "columns.h"

// ── FEATURE_SPEC: what the per-decision tensors hold (docs/featurizer.md §3, §6) ───────────────
//
// Standard library and the decoder only. Built from a SCHEMA frame (every table, column, tag and
// enum), a METAGAME frame (its tables' row counts) and the card tables (their widths, column names
// and fixed key list), so it is the same for every match. Every STATE table with id 20-48 or 90-99
// that no fold takes whole is a token type, in ascending id, then the folds' synthetic types; each
// column becomes slots by its tag (columns.h, §3.2):
//
//   bool, count, signed, prob   one numeric slot (x, log1p, sign·log1p|x|, x / 10^9)
//   index                       one numeric slot (log1p, 0 at none) and its none flag
//   mask:<e>                    one numeric slot per value of e
//   enum:<e>, player-rel        one categorical slot of e's values (an absolute enum:seat column
//                               beside its <name>_rel partner is dropped: the tensors are
//                               seat-symmetric)
//   card                        one tok_card slot and a numeric none flag
//   abref (ab_card, ab_index,   one tok_ability slot, and ab_key hashed (below)
//     ab_key)
//   handle                      a relation from the row's token to the token owning the handle
//                               (the lowest-id table whose first column is a handle named `handle`:
//                               object, then stack), typed by (table, column); none when no token
//                               owns it or the row owns it itself
//   row:<t>                     a relation to row v of t's tokens; into a METAGAME table, a
//                               categorical slot of that table's rows; else dropped
//   text                        dropped
//
// A categorical slot of n values takes ids offset … offset + n of one global id space: offset + 0
// is the cell's none (all ones, or a value out of range), offset + 1 + v value v; id 0 is the pad.
// The overrides (§3.4, the only per-column rules) are listed in the spec: object.card's second card
// slot (the current face), the hashed text columns (FNV-1a 64 modulo HASH_BUCKETS, + 1; empty text
// is the none id), menu.description dropped, and the knowledge tables' METAGAME row indices
// (k_template.list, k_archetype.archetype, b_list.list, b_archetype.archetype: the key of their
// pivot, below).
//
// The folds (§3.6) are listed in the spec too; they make tokens otherwise than one per row:
//
//   TABLE      one token per row of its table (every type the folds leave alone)
//   FILTERED   k_pool_card: the rows the ledger does not fold (pools other than a HAND, a SIDEBOARD
//              and a player's only LIBRARY_RANGE pool when it is a run)
//   DELTA      object_subtype / object_keyword: per object the rows where it differs from its current
//              face's printed rows (card_subtype / card_keyword), a printed row it lacks with
//              `removed` set; the full set for an object whose face has no card row
//   LEDGER     `ledger`: one token per (player, card) folding own_deck, k_seen, k_known_hand and the
//              folded k_pool_card rows (LedgerSlot)
//   PIVOT      `lists` (k_template, k_archetype) and `b_lists` (b_list, b_archetype): one token per
//              player, a numeric slot per (METAGAME row of the key, other column)
//
// Synthetic types (LEDGER, PIVOT) follow the table types; the folded tables have no type of their
// own. Token order is canonical: the ledger by (player_rel, card), a pivot by player_rel, a delta
// type by object row, then removed, then value (folds.h).
//
// Recent-event history (FeatureOptions::history_events > 0; off by default, and then the spec and
// every tensor are those of a spec built without it): two HISTORY types after the synthetic ones,
// `hist_event` (the `event` table's columns) and `hist_card` (`event_card`'s), whose tokens come
// from the seat's EventHistory (history.h) rather than the frame: the events of the frame's turn
// and the previous one, newest first, at most history_events, each with at most
// history_cards_per_event of its cards. The columns that identify the game rather than describe
// the event are dropped (HistoryPlan::dropped: seq, the absolute turn, the game, the batch id, the
// draw link's seq and index); a token's turn offset (0 this turn, 1 the previous), its age (its
// newest-first rank) and the cards past the cap take three numeric slots. Handle columns relate a
// token to the object that holds the handle now (followed through the later events' moves), and
// hist_card.event to its event's token. The spec document gains a `history` key.
//
// spec_json() writes the document with its keys in a fixed order; spec_hash is FNV-1a 64 of its
// bytes.

namespace featurize {

const int SPEC_VERSION = 2;
// Buckets of a hashed text column.
const uint32_t HASH_BUCKETS = 256;
// The library depth of the learner targets (targets.h).
const uint32_t LIBRARY_DEPTH = 10;

struct NumSlot {
        std::string name, column;
        Transform transform = Transform::BOOL;
        uint32_t bit = 0;  // MASK_BIT's bit
};

struct CatSlot {
        std::string name, column;
        std::string source;  // enum:<e>, player-rel, hashed, row:<metagame table>
        int64_t offset = 0;  // the slot's global id of none; values follow
        uint32_t size = 0;   // ids, none included
};

struct CardSlot {
        std::string name, column;
        bool current_face = false;  // the face as it is now (object.card's second slot)
};

// How a token type's tokens are made (header comment).
enum class TypeKind : uint8_t {
    TABLE,
    FILTERED,
    DELTA,
    LEDGER,
    PIVOT,
    HISTORY
};

const char *type_kind_name(TypeKind k);

// The card ledger's numeric slots, in order (docs/featurizer.md §3.6).
enum LedgerSlot : uint32_t {
    L_DECK,        // the card has an own_deck row
    L_MAIN,        // own_deck.main
    L_SIDE,        // own_deck.side
    L_SEEN_GAME,   // k_seen.game
    L_SEEN_MATCH,  // k_seen.match
    L_KNOWN_HAND,  // k_known_hand.copies (the opponent's)
    L_HAND_MIN,    // the HAND pool's k_pool_card min / max
    L_HAND_MAX,
    L_LIBRARY_MIN,  // Σ over the library runs' k_pool_card min / max, plus the exact k_position rows
    L_LIBRARY_MAX,
    L_SIDEBOARD_MIN,  // the SIDEBOARD pool's k_pool_card min / max
    L_SIDEBOARD_MAX,
    L_SLOTS,
};

// One table a PIVOT type folds: per row, the columns but the seat pair and the key take the slots
// first_slot + key · cols.size() + i.
struct PivotSource {
        std::string table, key, meta;  // the table, its key column, the METAGAME table the key indexes
        uint16_t table_id = 0;
        uint16_t rel_col = 0, key_col = 0;
        uint64_t keys = 0;  // the METAGAME table's rows
        std::vector<uint16_t> cols;
        std::vector<Transform> transforms;
        std::vector<rmstate::Storage> storages;
        uint32_t first_slot = 0;
};

// The columns the ledger reads (-1: the table or column is not in the schema).
struct LedgerPlan {
        int32_t type = -1;  // the ledger's token type
        int32_t own_deck = -1, k_seen = -1, k_known_hand = -1, k_pool = -1, k_pool_card = -1, k_position = -1;
        int deck_rel = -1, deck_card = -1, deck_main = -1, deck_side = -1;
        int seen_rel = -1, seen_card = -1, seen_game = -1, seen_match = -1;
        int known_card = -1, known_copies = -1;
        int pool_rel = -1, pool_kind = -1, pool_anchor = -1, pool_pos = -1, pool_span = -1;
        int pc_pool = -1, pc_card = -1, pc_min = -1, pc_max = -1;
        int pos_rel = -1, pos_anchor = -1, pos_pos = -1, pos_card = -1;
        rmstate::Storage pos_card_storage = rmstate::Storage::U16;
        uint64_t kind_hand = UINT64_MAX, kind_library = UINT64_MAX, kind_sideboard = UINT64_MAX;
        uint64_t opponent_rel = 2;  // k_known_hand's player
};

struct Fold {
        std::string table, into, rows;
};

struct RelationType {
        std::string table, column;
        std::string target;  // "handle", or the table a row: column points into
};

// One column's work on a token row, planned when the spec is built.
enum class OpKind : uint8_t {
    NUMERIC,      // num = cell_value(transform)
    INDEX,        // num = index_log1p, flag = none
    MASK,         // num … num + width - 1 = the bits
    CATEGORICAL,  // cat = offset + (v < width ? v + 1 : 0)
    HASHED,       // cat = offset + (text empty ? 0 : 1 + fnv1a64(text) % HASH_BUCKETS)
    CARD,         // card = v (-1 at none or past the card table), flag = none; face: the current face
    ABILITY,      // ability = the ability row of (ab_card, ab_index), else of ab_key, else -1
    HANDLE,       // relation to the handle's owner
    ROW,          // relation to row v of the target type
};

struct ColumnOp {
        OpKind kind = OpKind::NUMERIC;
        uint16_t col = 0;
        rmstate::Storage storage = rmstate::Storage::U8;
        Transform transform = Transform::BOOL;
        uint32_t slot = 0;       // numeric (first), categorical, card or ability slot
        int32_t flag = -1;       // INDEX, CARD: the none flag's numeric slot
        uint32_t width = 0;      // MASK: bits; CATEGORICAL: values
        int64_t offset = 0;      // CATEGORICAL, HASHED
        int32_t face_slot = -1;  // CARD: the current face's card slot (-1 none)
        uint16_t aux1 = 0;       // CARD with a face slot: the `transformed` column; ABILITY: ab_index
        uint16_t aux2 = 0;       // ABILITY: ab_key
        rmstate::Storage aux1_storage = rmstate::Storage::U8;
        int32_t relation = -1;  // HANDLE, ROW
        int32_t target = -1;    // ROW: the target token type
};

struct TokenType {
        TypeKind kind = TypeKind::TABLE;
        uint16_t table = 0;    // TABLE, FILTERED, DELTA: its table
        uint16_t columns = 0;  // and its column count
        std::string name;
        bool owns_handles = false;  // first column a handle named `handle`
        uint16_t handle_col = 0;
        std::vector<NumSlot> numeric;
        std::vector<CatSlot> categorical;
        std::vector<CardSlot> cards;
        std::vector<std::string> abilities;
        std::vector<int32_t> relations;
        std::vector<ColumnOp> ops;  // column order
        // DELTA: the printed table compared with, the obj column, per printed column its column here,
        // and the `removed` numeric slot.
        std::string printed;
        uint16_t obj_col = 0;
        std::vector<uint16_t> value_cols;
        std::vector<uint8_t> value_text;  // per printed column: a text column (compared by its text)
        int32_t removed_slot = -1;
        // PIVOT: the tables folded; LEDGER and PIVOT: the player_rel categorical's id offset and values.
        std::vector<PivotSource> pivot;
        int64_t rel_offset = 0;
        uint32_t rel_values = 0;
        bool synthetic() const { return kind == TypeKind::LEDGER || kind == TypeKind::PIVOT; }
};

struct Override {
        std::string table, column, treatment;
};

// What a spec is built with. Every option's default leaves the spec as it is without options.
struct FeatureOptions {
        uint32_t history_events = 0;           // the recent-event tokens' cap (0: no history types)
        uint32_t history_cards_per_event = 8;  // each event's card tokens' cap
};

// The history types' plan (header comment); `events` 0 when the spec has none.
struct HistoryPlan {
        uint32_t events = 0, cards_per_event = 0;
        int32_t event_type = -1, card_type = -1;  // the token types
        uint16_t event_table = 0, card_table = 0, turn_table = 0;
        int event_kind = -1, event_turn = -1, event_source = -1;  // the event table's columns
        int card_event = -1, card_before = -1, card_after = -1;   // event_card's `event` and handles
        int turn_turn = -1;                                       // the turn table's `turn`
        uint64_t game_start_kind = UINT64_MAX;
        // Numeric slots of hist_event: the turn offset, the age, the cards past the cap.
        uint32_t offset_slot = 0, age_slot = 0, more_slot = 0;
        std::vector<std::string> dropped;  // "<table>.<column>"
        bool on() const { return events > 0; }
};

struct FeatureSpec {
        uint64_t schema_hash = 0;
        uint64_t vocab_hash = 0;
        uint32_t hash_buckets = HASH_BUCKETS;
        uint32_t library_depth = LIBRARY_DEPTH;
        // The rows of each METAGAME table a categorical slot counts (table name, rows), in first use.
        std::vector<std::pair<std::string, uint64_t>> metagame_rows;
        size_t F = 0, C = 0, K = 0, Q = 0;
        int64_t categorical_size = 0;  // ids 0 … categorical_size - 1
        std::vector<TokenType> types;
        std::vector<RelationType> relations;
        std::vector<Override> overrides;
        std::vector<Fold> folds;
        LedgerPlan ledger;
        HistoryPlan history;
        // The object table's token type and the columns a delta reads (-1: none).
        int32_t object_type = -1;
        int object_handle = -1, object_card = -1, object_transformed = -1;
        rmstate::Storage object_card_storage = rmstate::Storage::U16;
        // The card tables' shapes and column names (card_tables.h).
        uint64_t card_rows = 0, text_rows = 0, text_width = 0, ability_rows = 0, part_rows = 0;
        std::vector<std::string> card_columns, ability_columns, part_columns, ab_keys;
        // Lookups: per table id its token type (-1 none: not a token table, or folded); the menu's
        // token type; the header's.
        std::vector<int32_t> type_of_table;
        int32_t menu_type = -1, header_type = -1;
        int32_t header_game = -1, header_decision = -1, header_sideboarding = -1;
        std::string json;
        uint64_t hash = 0;
};

// Builds the spec. False with `err` when the schema lacks the menu or header table (or, with history,
// the event tables).
bool build_spec(const rmstate::Schema &schema, const rmstate::DecodedFrame &metagame, const CardTables &tables,
    const FeatureOptions &options, FeatureSpec *out, std::string *err);

// The spec document (FeatureSpec::json), keys in a fixed order.
std::string spec_json(const FeatureSpec &spec);

}  // namespace featurize

#endif /* FEATURIZE_FEATURE_SPEC_H */
