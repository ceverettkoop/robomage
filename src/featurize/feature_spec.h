#ifndef FEATURIZE_FEATURE_SPEC_H
#define FEATURIZE_FEATURE_SPEC_H

#include <cstdint>
#include <string>
#include <vector>

#include "../state_format/decode.h"
#include "card_tables.h"
#include "columns.h"
#include "registry.h"

// ── FEATURE_SPEC: what the per-decision tensors hold (docs/featurizer.md §3, §6) ───────────────
//
// Standard library and the decoder only. Built from a SCHEMA frame (every table, column, tag and
// enum), the card tables (their widths, column names and fixed key list) and the layout registry
// (registry.h: every position below is the registry's, by name), so it is the same for every
// match. Every STATE table with id 20-48 or 90-95 that no fold takes whole is a token type, as are
// the folds' synthetic type and the history types; a type's id (its tok_type) is its registry id,
// and tokens come in id order. Each column becomes slots by its tag (columns.h, §3.2):
//
//   bool, count, signed         one numeric slot (x, log1p, sign·log1p|x|)
//   index                       one numeric slot (log1p, 0 at none) and its none flag
//   mask:<e>                    a numeric block of e's capacity: value v at the block's start
//                               plus v's registry index
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
//   row:<t>                     a relation to row v of t's tokens; dropped when t has no token
//                               type
//   text                        dropped
//
// Positions: a type's numeric slots are columns of its own block (num_at … num_at + num_capacity
// of the G columns every type has side by side; within a token's tok_num row, column `at` of F);
// a categorical slot is a tok_cat column with a fixed id range offset … offset + size - 1 of one
// global id space: offset + 0 is the cell's none (all ones, or a value out of range), offset + 1 +
// i the value of registry index i; id 0 is the pad. Card and ability slots are tok_card /
// tok_ability columns; relation types have registry ids. Every position past what the schema uses
// is reserved and reads 0 (-1 for card and ability slots). The overrides (§3.4, the only
// per-column rules) are listed in the spec: object.card's second card slot (the current face), the
// hashed text columns (FNV-1a 64 modulo HASH_BUCKETS, + 1; empty text is the none id) and
// menu.description dropped.
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
//
// The folded tables have no type of their own; a registry type the schema does not make (or a
// history type without the option) is ABSENT: it has no tokens and is not in the document.
// Token order is canonical: the ledger by (player_rel, card), a delta type by object row, then
// removed, then value (folds.h).
//
// Recent-event history (FeatureOptions::history_events > 0; off by default, and then the spec and
// every tensor are those of a spec built without it): two HISTORY types, `hist_event` (the `event`
// table's columns) and `hist_card` (`event_card`'s), whose tokens come
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
// bytes, layout_hash the registry's (registry.h): equal for every spec of one layout, whatever was
// appended to it.

namespace featurize {

const int SPEC_VERSION = 4;
// Buckets of a hashed text column.
const uint32_t HASH_BUCKETS = 256;
// The library depth of the learner targets (targets.h).
const uint32_t LIBRARY_DEPTH = 10;

struct NumSlot {
        std::string name, column;
        Transform transform = Transform::BOOL;
        uint32_t bit = 0;  // MASK_BIT's bit
        uint32_t at = 0;   // the slot's column in the type's block (and the token's tok_num row)
};

struct CatSlot {
        std::string name, column;
        std::string source;  // enum:<e>, player-rel, hashed
        int64_t offset = 0;  // the slot's global id of none; values follow
        uint32_t size = 0;   // ids, none included
        uint32_t col = 0;    // its tok_cat column
};

struct CardSlot {
        std::string name, column;
        bool current_face = false;  // the face as it is now (object.card's second slot)
        uint32_t col = 0;           // its tok_card column
};

struct AbilitySlot {
        std::string name;
        uint32_t col = 0;  // its tok_ability column
};

// How a token type's tokens are made (header comment).
enum class TypeKind : uint8_t {
    ABSENT,  // a registry id the spec does not use
    TABLE,
    FILTERED,
    DELTA,
    LEDGER,
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
        int32_t enum_map = -1;   // MASK, CATEGORICAL of an enum: FeatureSpec::enum_maps' entry
        int32_t face_slot = -1;  // CARD: the current face's card slot (-1 none)
        uint16_t aux1 = 0;       // CARD with a face slot: the `transformed` column; ABILITY: ab_index
        uint16_t aux2 = 0;       // ABILITY: ab_key
        rmstate::Storage aux1_storage = rmstate::Storage::U8;
        int32_t relation = -1;  // HANDLE, ROW
        int32_t target = -1;    // ROW: the target token type
};

struct TokenType {
        TypeKind kind = TypeKind::ABSENT;
        uint16_t table = 0;    // TABLE, FILTERED, DELTA: its table
        uint16_t columns = 0;  // and its column count
        std::string name;
        bool owns_handles = false;  // first column a handle named `handle`
        uint16_t handle_col = 0;
        // The type's numeric block in G and its capacity (≤ F); per kind of slot the tok_cat /
        // tok_card / tok_ability columns it uses (the highest used + 1).
        uint32_t num_at = 0, num_capacity = 0;
        uint32_t cat_width = 0, card_width = 0, ability_width = 0;
        std::vector<NumSlot> numeric;  // the slots the schema makes (not the reserved positions)
        std::vector<CatSlot> categorical;
        std::vector<CardSlot> cards;
        std::vector<AbilitySlot> abilities;
        std::vector<int32_t> relations;
        std::vector<ColumnOp> ops;  // column order
        // DELTA: the printed table compared with, the obj column, per printed column its column here,
        // and the `removed` numeric slot.
        std::string printed;
        uint16_t obj_col = 0;
        std::vector<uint16_t> value_cols;
        std::vector<uint8_t> value_text;  // per printed column: a text column (compared by its text)
        int32_t removed_slot = -1;        // its column
        // LEDGER: the player_rel categorical's id offset, values and tok_cat column, the card's
        // tok_card column.
        int64_t rel_offset = 0;
        uint32_t rel_values = 0, rel_col = 0, card_col = 0;
        bool synthetic() const { return kind == TypeKind::LEDGER; }
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
        // Numeric slots (columns) of hist_event: the turn offset, the age, the cards past the cap.
        uint32_t offset_slot = 0, age_slot = 0, more_slot = 0;
        std::vector<std::string> dropped;  // "<table>.<column>"
        bool on() const { return events > 0; }
};

struct FeatureSpec {
        uint64_t schema_hash = 0;
        uint64_t vocab_hash = 0;
        uint64_t layout_hash = 0;
        int layout_version = 0;
        uint32_t hash_buckets = HASH_BUCKETS;
        uint32_t library_depth = LIBRARY_DEPTH;
        // The registry's dims: per token F numeric, C categorical, K card, Q ability slots; G the
        // numeric columns of every type's block; the capacities of the token types (tok_type ids),
        // the categorical ids (ids 0 … categorical_size - 1) and the relation types.
        size_t F = 0, C = 0, K = 0, Q = 0, G = 0;
        int64_t categorical_size = 0;
        uint32_t type_capacity = 0, relation_capacity = 0;
        std::vector<TokenType> types;         // indexed by type id (ABSENT for an unused id)
        std::vector<RelationType> relations;  // indexed by relation id ("" table: unused)
        // Per enum a MASK or CATEGORICAL column reads, the engine value → registry index map and
        // the registry's values (index order).
        std::vector<std::string> enum_names;
        std::vector<std::vector<int32_t>> enum_maps;
        std::vector<std::vector<std::string>> enum_values;
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
        uint32_t abilities_per_card = 0;
        uint64_t key_base = 0;
        std::vector<std::string> card_columns, ability_columns, part_columns, ab_keys;
        std::vector<int64_t> key_rows;
        // Lookups: per table id its token type (-1 none: not a token table, or folded); the menu's
        // token type; the header's.
        std::vector<int32_t> type_of_table;
        int32_t menu_type = -1, header_type = -1;
        int32_t header_game = -1, header_decision = -1, header_sideboarding = -1;
        std::string json;
        uint64_t hash = 0;
};

// Builds the spec at the positions of `registry` (with `append`, adding what it lacks). False with
// `err` when the schema lacks the menu or header table (or, with history, the event tables), or the
// registry lacks an entry or a capacity (Registry::problems).
bool build_spec(const rmstate::Schema &schema, const CardTables &tables, const FeatureOptions &options,
    Registry *registry, FeatureSpec *out, std::string *err);
// The same at the compiled-in registry's positions.
bool build_spec(const rmstate::Schema &schema, const CardTables &tables, const FeatureOptions &options,
    FeatureSpec *out, std::string *err);

// The spec document (FeatureSpec::json), keys in a fixed order.
std::string spec_json(const FeatureSpec &spec);

}  // namespace featurize

#endif /* FEATURIZE_FEATURE_SPEC_H */
