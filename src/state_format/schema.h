#ifndef STATE_FORMAT_SCHEMA_H
#define STATE_FORMAT_SCHEMA_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// ── RMSTATE: the single definition of the state format (docs/state_format.md §9, Appendix A) ───
//
// Everything a frame can hold is defined here: the frame and table-block layout, every table and
// its columns (name, storage type, semantic tag, value range), the enums the format adds, and the
// writer that lays rows out. The SCHEMA frame is generated from these definitions, and the schema
// hash is computed from the SCHEMA frame, so a change to any definition changes the hash.
//
// Standard library only: the decoder (state_format/decode.*) and the knowledge sidecar
// (src/knowledge/) include this header, and neither may see an engine header. Enums the engine
// defines (steps, zones, action categories, …) reach the SCHEMA frame as EnumTable lists the
// engine side builds from its own enums (state_format/enum_names.h); this header never repeats
// their values.

namespace rmstate {

constexpr uint32_t FRAME_MAGIC = 0x46534D52u;  // "RMSF", little-endian
constexpr uint16_t FORMAT_VERSION = 2;
constexpr size_t FRAME_HEADER_SIZE = 24;
constexpr size_t TABLE_HEADER_SIZE = 16;
constexpr size_t ROW_ALIGN = 8;  // each table block's rows are zero-padded to a multiple of this

// None values (Appendix A.1): handle 0, text 0, and all ones of the column's width for card,
// index, row: and nullable enum columns.
constexpr uint8_t NONE8 = 0xFF;
constexpr uint16_t NONE16 = 0xFFFF;
constexpr uint32_t NONE32 = 0xFFFFFFFFu;

// ── Enums the format adds (Appendix A.7) ──────────────────────────────────────────────────────
// Each list is an X-macro, so the C++ enum and its SCHEMA string table come from one list.

#define RMSTATE_FRAME_KIND(X) X(SCHEMA) X(CARDS) X(DECKLISTS) X(STATE)
#define RMSTATE_VIEWER(X) X(A) X(B) X(OMNISCIENT)
#define RMSTATE_STORAGE(X) X(U8) X(U16) X(U32) X(U64) X(I8) X(I16) X(I32) X(I64) X(F32) X(TEXT)
#define RMSTATE_HAND_SUBSET(X) X(ANY) X(DRAWN_THIS_TURN)
#define RMSTATE_EVENT_KIND(X)                                                                                     \
    X(ZONE_MOVE)                                                                                                  \
    X(DRAW)                                                                                                       \
    X(LIBRARY_PLACE)                                                                                              \
    X(REVEAL_START) X(REVEAL_END) X(LOOK) X(SHUFFLE) X(REORDER) X(SEARCH) X(CAST_FROM_HIDDEN) X(PLAY_FROM_HIDDEN) \
        X(FACE_DOWN) X(FACE_UP) X(GAME_START) X(GAME_END) X(MATCH_END)
#define RMSTATE_LIB_ANCHOR(X) X(TOP) X(BOTTOM)
#define RMSTATE_FACE(X) X(FRONT) X(BACK)
#define RMSTATE_LAYOUT(X) X(NORMAL) X(TRANSFORM) X(MODAL_DFC) X(SPLIT)
#define RMSTATE_ABILITY_KIND(X) X(SPELL) X(ACTIVATED) X(TRIGGERED) X(STATIC) X(REPLACEMENT) X(CHAPTER) X(OPENING_HAND)
#define RMSTATE_TIMING(X) X(ANY) X(SORCERY) X(INSTANT)
#define RMSTATE_PIP_KIND(X) X(HYBRID) X(TWOBRID) X(PHYREXIAN) X(SNOW)
#define RMSTATE_STACK_KIND(X) X(SPELL) X(ACTIVATED) X(TRIGGERED) X(COPY)
#define RMSTATE_TARGET_KIND(X) X(OBJECT) X(PLAYER)
#define RMSTATE_ATTACK_TARGET_KIND(X) X(PLAYER) X(PLANESWALKER)
#define RMSTATE_DELAYED_STATE(X) X(WAITING) X(ON_STACK)
#define RMSTATE_COST_PART(F) F(MANA) F(ALTERNATIVE) F(ADDITIONAL) F(UNLESS) F(X) F(ACTIVATION)
#define RMSTATE_LAYER(X) \
    X(COPY) X(CONTROL) X(TEXT) X(TYPE) X(COLOR) X(ABILITY) X(PT_CDA) X(PT_SET) X(PT_MODIFY) X(PT_COUNTERS) X(PT_SWITCH)
#define RMSTATE_DURATION(X) X(STATIC) X(END_OF_TURN) X(UNTIL_YOUR_NEXT_TURN) X(UNTIL_HOST_LEAVES) X(PERMANENT)
#define RMSTATE_LINK_KIND(X) X(EXILED_WITH) X(RETURNABLE)
#define RMSTATE_PLAYER_EFFECT_KIND(X) \
    X(PROTECTION_FROM_EVERYTHING)     \
    X(CANT_GAIN_LIFE)                 \
    X(HEXPROOF_FROM) X(SPELLS_CANT_BE_COUNTERED) X(CAST_AS_FLASH) X(SORCERY_SPEED_ONLY) X(FLOATING_TRIGGER)
#define RMSTATE_PERMISSION_KIND(X) X(FLASHBACK) X(ESCAPE) X(GRAVEYARD_CAST) X(GRAVEYARD_LAND) X(EXILE_GRANT)
#define RMSTATE_PERMISSION_RESOURCE(X) X(MANA) X(ENERGY) X(LIFE) X(FREE)
#define RMSTATE_SHIELD_KIND(X) X(COMBAT_DAMAGE)
#define RMSTATE_CAST_VARIANT(X) \
    X(FLASHBACK)                \
    X(EVOKE)                    \
    X(ESCAPE) X(OFFSPRING) X(IMPENDING) X(WARP) X(BACK_FACE) X(ALT_COST) X(IMPULSE) X(COMPANION) X(SUSPEND) X(CANCEL)
#define RMSTATE_KNOWLEDGE_LEVEL(X) X(OFF) X(EXPERT)
#define RMSTATE_POOL_KIND(X) X(HAND) X(LIBRARY_RANGE) X(SIDEBOARD) X(FACE_DOWN) X(SET_ASIDE)
#define RMSTATE_AMOUNT_KIND(X) X(NONE) X(FIXED) X(VARIABLE)
#define RMSTATE_TARGET_CLASS(X) \
    X(CREATURE)                 \
    X(PLANESWALKER)             \
    X(ARTIFACT)                 \
    X(ENCHANTMENT) X(LAND) X(BATTLE) X(PERMANENT_OTHER) X(PLAYER) X(SPELL) X(ABILITY) X(CARD_GRAVEYARD) X(CARD_OTHER)
#define RMSTATE_PART_KIND(X) X(SUB) X(MODE)

#define RMSTATE_ENUM_VALUE(name) name,
enum class FrameKind : uint8_t {
    RMSTATE_FRAME_KIND(RMSTATE_ENUM_VALUE)
};
enum class Viewer : uint8_t {
    RMSTATE_VIEWER(RMSTATE_ENUM_VALUE)
};
enum class Storage : uint8_t {
    RMSTATE_STORAGE(RMSTATE_ENUM_VALUE)
};
enum class HandSubset : uint8_t {
    RMSTATE_HAND_SUBSET(RMSTATE_ENUM_VALUE)
};
enum class EventKind : uint8_t {
    RMSTATE_EVENT_KIND(RMSTATE_ENUM_VALUE)
};
enum class LibAnchor : uint8_t {
    RMSTATE_LIB_ANCHOR(RMSTATE_ENUM_VALUE)
};
enum class Face : uint8_t {
    RMSTATE_FACE(RMSTATE_ENUM_VALUE)
};
enum class Layout : uint8_t {
    RMSTATE_LAYOUT(RMSTATE_ENUM_VALUE)
};
enum class AbilityKind : uint8_t {
    RMSTATE_ABILITY_KIND(RMSTATE_ENUM_VALUE)
};
enum class Timing : uint8_t {
    RMSTATE_TIMING(RMSTATE_ENUM_VALUE)
};
enum class PipKind : uint8_t {
    RMSTATE_PIP_KIND(RMSTATE_ENUM_VALUE)
};
enum class StackKind : uint8_t {
    RMSTATE_STACK_KIND(RMSTATE_ENUM_VALUE)
};
enum class TargetKind : uint8_t {
    RMSTATE_TARGET_KIND(RMSTATE_ENUM_VALUE)
};
enum class AttackTargetKind : uint8_t {
    RMSTATE_ATTACK_TARGET_KIND(RMSTATE_ENUM_VALUE)
};
enum class DelayedState : uint8_t {
    RMSTATE_DELAYED_STATE(RMSTATE_ENUM_VALUE)
};
enum class CostPart : uint8_t {
    RMSTATE_COST_PART(RMSTATE_ENUM_VALUE)
};
enum class Layer : uint8_t {
    RMSTATE_LAYER(RMSTATE_ENUM_VALUE)
};
enum class Duration : uint8_t {
    RMSTATE_DURATION(RMSTATE_ENUM_VALUE)
};
enum class LinkKind : uint8_t {
    RMSTATE_LINK_KIND(RMSTATE_ENUM_VALUE)
};
enum class PlayerEffectKind : uint8_t {
    RMSTATE_PLAYER_EFFECT_KIND(RMSTATE_ENUM_VALUE)
};
enum class PermissionKind : uint8_t {
    RMSTATE_PERMISSION_KIND(RMSTATE_ENUM_VALUE)
};
enum class PermissionResource : uint8_t {
    RMSTATE_PERMISSION_RESOURCE(RMSTATE_ENUM_VALUE)
};
enum class ShieldKind : uint8_t {
    RMSTATE_SHIELD_KIND(RMSTATE_ENUM_VALUE)
};
enum class CastVariant : uint8_t {
    RMSTATE_CAST_VARIANT(RMSTATE_ENUM_VALUE)
};
enum class KnowledgeLevel : uint8_t {
    RMSTATE_KNOWLEDGE_LEVEL(RMSTATE_ENUM_VALUE)
};
enum class PoolKind : uint8_t {
    RMSTATE_POOL_KIND(RMSTATE_ENUM_VALUE)
};
enum class AmountKind : uint8_t {
    RMSTATE_AMOUNT_KIND(RMSTATE_ENUM_VALUE)
};
enum class TargetClass : uint8_t {
    RMSTATE_TARGET_CLASS(RMSTATE_ENUM_VALUE)
};
enum class PartKind : uint8_t {
    RMSTATE_PART_KIND(RMSTATE_ENUM_VALUE)
};

// ── Tables ────────────────────────────────────────────────────────────────────────────────────
// Ids are grouped by frame kind (Appendix A.2): SCHEMA 2-4, CARDS 9-14, DECKLISTS 15-16, STATE
// snapshot 20-48, events 80-82, knowledge 90-95. The strings pool (1) belongs to every frame.
enum class TableId : uint16_t {
    STRINGS = 1,
    SCHEMA_TABLE = 2,
    SCHEMA_COLUMN = 3,
    SCHEMA_ENUM = 4,
    CARD_ABILITY_PART = 9,
    CARD = 10,
    CARD_SUBTYPE = 11,
    CARD_KEYWORD = 12,
    CARD_ABILITY = 13,
    CARD_PIP = 14,
    DECKLIST = 15,
    DECKLIST_CARD = 16,
    HEADER = 20,
    TURN = 21,
    PLAYER = 22,
    PLAYER_COUNTER = 23,
    OBJECT = 24,
    OBJECT_SUBTYPE = 25,
    OBJECT_KEYWORD = 26,
    OBJECT_COUNTER = 27,
    OBJECT_ABILITY = 28,
    OBJECT_PROVENANCE = 29,
    OBJECT_LINK = 30,
    COMBAT = 31,
    STACK = 32,
    STACK_MODE = 33,
    STACK_TARGET = 34,
    DELAYED_TRIGGER = 35,
    DELAYED_SUBJECT = 36,
    EMBLEM = 37,
    PLAYER_EFFECT = 38,
    PLAY_PERMISSION = 39,
    SHIELD = 40,
    LKI = 41,
    REVEAL = 42,
    PENDING = 43,
    PENDING_COST = 44,
    PENDING_COST_PIP = 45,
    MENU = 46,
    MATCH = 47,
    OWN_DECK = 48,
    EVENT = 80,
    EVENT_CARD = 81,
    EVENT_PLAYER = 82,
    K_META = 90,
    K_SEEN = 91,
    K_POOL = 92,
    K_POOL_CARD = 93,
    K_POSITION = 94,
    K_KNOWN_HAND = 95,
};

struct ColumnDef {
        std::string name;
        Storage storage = Storage::U8;
        std::string tag;  // semantic tag (§9, Appendix A.1): enum:<e>, count, bool, signed, handle, …
        bool has_range = false;
        int64_t min = 0;
        int64_t max = 0;
        uint32_t offset = 0;  // byte offset in the row
};

struct TableDef {
        TableId id = TableId::STRINGS;
        std::string name;
        FrameKind frame_kind = FrameKind::STATE;  // STRINGS lists STATE; it belongs to every frame
        std::vector<ColumnDef> columns;
        uint32_t row_width = 0;
};

// Every table, in ascending id.
const std::vector<TableDef> &table_defs();
// The table `id` (every TableId has one).
const TableDef &table_def(TableId id);
// The table with numeric id `id`, or nullptr (a decoder skips unknown ids, Appendix A.2).
const TableDef *find_table_def(uint16_t id);
// Bytes one value of `storage` takes.
uint32_t storage_width(Storage storage);

// One enum's string table: value i is named values[i].
struct EnumTable {
        std::string name;
        std::vector<std::string> values;
};
// The enums this header defines (A.7), in a fixed order.
std::vector<EnumTable> format_enums();

// ── Frames ────────────────────────────────────────────────────────────────────────────────────

// One table's rows while a frame is being built.
struct TableBuf {
        std::string bytes;
        uint32_t rows = 0;
        // Text cells awaiting their pool offset: byte position in `bytes`, the string.
        std::vector<std::pair<size_t, std::string>> texts;
};

// Builds one frame: rows are appended per table (each table's rows in the order written), text is
// pooled at finish() in table, row and column order (Appendix A.1), and blocks are laid out in
// ascending table id with empty tables omitted (A.2). In debug builds every value written is
// checked against its column's storage type and every row against the table's column count, so
// a row can never disagree with the schema it claims.
class FrameWriter {
    public:
        FrameWriter(FrameKind kind, Viewer viewer, uint64_t schema_hash);
        FrameWriter(const FrameWriter &) = delete;
        FrameWriter &operator=(const FrameWriter &) = delete;

        // One row being written; values go in column order. The row is complete when the Row
        // object is destroyed.
        class Row {
            public:
                Row(FrameWriter &w, TableId id);
                ~Row();
                Row(const Row &) = delete;
                Row &operator=(const Row &) = delete;
                Row &u8(uint64_t v);
                Row &u16(uint64_t v);
                Row &u32(uint64_t v);
                Row &u64(uint64_t v);
                Row &i8(int64_t v);
                Row &i16(int64_t v);
                Row &i32(int64_t v);
                Row &i64(int64_t v);
                Row &f32(float v);
                Row &text(const std::string &s);

            private:
                void put(Storage s, uint64_t bits);
                TableBuf *buf_;
                const TableDef *def_;
                size_t at_ = 0;  // the row's first byte in buf_->bytes (sized for the whole row)
                size_t col_ = 0;
        };

        Row row(TableId id) { return Row(*this, id); }
        // The frame payload (frame header + table blocks). Call once.
        std::string finish();
        // Rows written so far to `id`.
        uint32_t row_count(TableId id) const;

    private:
        friend class Row;
        TableBuf &buf(TableId id);
        FrameKind kind_;
        Viewer viewer_;
        uint64_t schema_hash_;
        std::map<uint16_t, TableBuf> tables_;
        // tables_'s entries by id (a map's nodes stay put), filled as rows are first written.
        static constexpr uint16_t BUF_CACHE = 128;
        TableBuf *buf_cache_[BUF_CACHE] = {};
};

// 64-bit FNV-1a.
uint64_t fnv1a64(const void *data, size_t len);

// The SCHEMA frame for `viewer` over `enums` (the format's own enums plus the engine's); its
// header carries its own hash, which every other frame's header repeats. `hash_out` receives it.
std::string build_schema_frame(Viewer viewer, const std::vector<EnumTable> &enums, uint64_t *hash_out);

// Wraps a payload in the stream framing: "RMSTATE <nbytes>\n" + payload (§9).
std::string frame_record(const std::string &payload);

}  // namespace rmstate

#endif /* STATE_FORMAT_SCHEMA_H */
