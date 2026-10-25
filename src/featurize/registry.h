#ifndef FEATURIZE_REGISTRY_H
#define FEATURIZE_REGISTRY_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// ── FEATURE_REGISTRY: the network's layout, append-only (docs/featurizer.md §6.1) ──────────────
//
// Standard library only. The committed file src/featurize/registry.json (compiled into every
// binary that featurizes: builtin_registry) gives every layout-defining value a permanent position
// and a reserved capacity, so that appending a card, an enum value, a column or a token type moves
// nothing a trained network's parameters are indexed by:
//
//   dims          the global sizes: F (numeric slots per token), C, K, Q (categorical, card and
//                 ability slots per token), G (the numeric columns of every type side by side),
//                 the token types', categorical ids', relation types' and fixed keys' capacities,
//                 abilities_per_card (A: the ability rows of card id c are c·A … c·A + A - 1) and
//                 the hashed text columns' bucket count;
//   enums         per enum read by a MASK, ENUM or child-count column: its values by name (index =
//                 position) and its capacity (closed enums their size, open ones room to grow);
//   token_types   name → id, and the type's numeric block in G (num_at, num_capacity ≤ F);
//   numeric       per (type, slot name) its column within the type's block; a MASK column one
//                 entry of its enum's capacity (value v at at + v's index);
//   categorical   per (type, slot) its tok_cat column, its first id (the none) and its id count;
//   cards, abilities  per (type, slot) its tok_card / tok_ability column;
//   relations     per (table, column, target) its relation type id;
//   tables        the card, ability and part tables' column capacities, and `columns`: per (table,
//                 column) its position (an enum one-hot, mask or child count one entry of its
//                 enum's capacity, like a MASK column);
//   ab_keys       per fixed ability key its row in the key band (the rows after the card band).
//
// Entries are matched by name, never by an engine index, so an engine enum may be reordered without
// moving a column; an entry whose value or column the schema no longer has is a hole (reads 0).
// layout_version is raised for a deliberate change that is not an append (a capacity, a moved
// entry: raise_capacities, rmfeat --registry-raise); layout_hash (FNV-1a 64 over layout_version, the
// dims, the tables' capacities and the card rows) is then different, and only then.
//
// Lookups (the featurizer's build_spec and build_card_tables) find an entry or record a problem:
// strict (the default) a missing entry is a problem naming `make registry`; with `append` set a
// missing entry is added at the next free position (an enum value at the end of its enum, a slot
// at its type's end, a categorical range at the id space's end, …), which `make registry` (rmfeat
// --registry-update) does and writes back. Exceeding a capacity is a problem either way: a layout
// bump, never a silent move. finish() places the types and tables an append made (their capacity
// from their use); before that their positions are relative to them.

namespace featurize {

const int REGISTRY_FORMAT = 1;

struct RegistryDims {
        uint32_t F = 0, C = 0, K = 0, Q = 0, G = 0;
        uint32_t types = 0, categorical = 0, relations = 0, ab_keys = 0;
        uint32_t abilities_per_card = 0, hash_buckets = 0;
};

struct RegEnum {
        std::string name;
        uint32_t capacity = 0;
        bool closed = false;
        std::vector<std::string> values;
};

struct RegType {
        std::string name;
        uint32_t id = 0, num_at = 0, num_capacity = 0;
};

// A numeric slot of a type, or a column of a card table: scalar (width 1), or an enum's block
// (mask, onehot, count: width = the enum's capacity).
struct RegSlot {
        std::string owner;  // the token type or the table
        std::string name, kind, enum_name;
        uint32_t at = 0, width = 1;
};

struct RegCat {
        std::string type, name, source;  // source: enum:<e>, player-rel, hashed
        uint32_t col = 0, offset = 0, size = 0;
};

struct RegCol {
        std::string type, name;
        uint32_t col = 0;
};

struct RegRelation {
        uint32_t id = 0;
        std::string table, column, target;
};

struct RegTable {
        std::string name;
        uint32_t capacity = 0;
};

struct RegKey {
        uint32_t row = 0;
        std::string key;
};

class Registry {
    public:
        int layout_version = 1;
        RegistryDims dims;
        std::vector<RegEnum> enums;
        std::vector<RegType> types;
        std::vector<RegSlot> numeric;
        std::vector<RegCat> categorical;
        std::vector<RegCol> cards, abilities;
        std::vector<RegRelation> relations;
        std::vector<RegTable> tables;
        std::vector<RegSlot> columns;
        std::vector<RegKey> ab_keys;

        // Lookups add missing entries (make registry) instead of recording them as problems.
        bool append = false;
        // What the lookups found missing or past a capacity, and (append) what they added.
        std::vector<std::string> problems, added;

        // An empty registry with the default dims, for a first `make registry`: its finish() sizes
        // F, G and the categorical id space from their use.
        static Registry fresh();
        // Parses and validates (no overlap, every position within its capacity) a registry file's
        // text. False with `err`.
        static bool parse(const std::string &text, Registry *out, std::string *err);
        // The file's text: keys in a fixed order, one entry per line.
        std::string json() const;

        // The registry index of `value` of enum `name` (an enum first seen gets its capacity:
        // closed enums their size `n`, open ones max(8, the power of two ≥ 2n)); -1 with a
        // problem.
        int32_t enum_index(const std::string &name, const std::string &value, size_t n);
        const RegEnum *find_enum(const std::string &name) const;
        // Per engine value of enum `name` (its string table `values`) its registry index (-1: none).
        std::vector<int32_t> enum_map(const std::string &name, const std::vector<std::string> &values);

        // A token type's id; -1 with a problem.
        int32_t type_id(const std::string &name);
        const RegType *find_type(const std::string &name) const;
        // A numeric slot's column within type `type`: kind scalar (enum_name ""), or mask with
        // the enum's block; -1 with a problem.
        int32_t numeric_at(
            const std::string &type, const std::string &name, const std::string &kind, const std::string &enum_name);
        // A categorical slot of `size` ids (none included) into `out`; false with a problem.
        bool categorical_slot(
            const std::string &type, const std::string &name, const std::string &source, uint32_t size, RegCat *out);
        int32_t card_col(const std::string &type, const std::string &name);
        int32_t ability_col(const std::string &type, const std::string &name);
        int32_t relation_id(const std::string &table, const std::string &column, const std::string &target);
        // A card table column's position (kinds as numeric_at, plus onehot and count); -1 with a
        // problem.
        int32_t column_at(
            const std::string &table, const std::string &name, const std::string &kind, const std::string &enum_name);
        uint32_t table_capacity(const std::string &table) const;
        // Every position's name of table `table`: an entry's name, an enum block's `<name><sep><value>`
        // (sep ':' for a mask, '=' otherwise) for each registered value and `<name>#<i>` past them,
        // `#<position>` past every entry.
        std::vector<std::string> column_names(const std::string &table) const;
        // A fixed ability key's row in the key band; -1 with a problem.
        int32_t key_row(const std::string &key);

        // Places what an append added (header comment); a problem past a capacity.
        void finish();

        // A layout bump's widening (rmfeat --registry-raise): each enum of `capacities` raised to
        // its capacity and made open. Every block of a raised enum (numeric masks, table columns,
        // categorical ranges) widens to it, every position after a widened block in the same space
        // (a type's numeric block, G, a table, the categorical ids) moves by the growth before it,
        // and a space whose entries no longer fit grows to them plus the spare room it had (rounded
        // up: a type's block to 8, G to 128, a table to 64, the categorical ids to 512). Every
        // entry keeps its name, order and value indices; layout_version is the caller's to raise
        // once per bump. False with `err` (an unknown enum, a smaller capacity, a type past F).
        bool raise_capacities(const std::map<std::string, uint32_t> &capacities, std::string *err);
        bool ok() const { return problems.empty(); }

    private:
        void index();
        uint32_t enum_width(const std::string &enum_name) const;
        void problem(const std::string &what);
        static std::string join(const std::string &a, const std::string &b);
        static std::string join(const std::string &a, const std::string &b, const std::string &c);

        bool fresh_ = false;
        std::map<std::string, size_t> enum_of_, type_of_, numeric_of_, cat_of_, card_of_, ability_of_, relation_of_,
            table_of_, column_of_, key_of_;
        std::vector<std::map<std::string, uint32_t>> value_of_;  // per enum
        std::map<std::string, uint32_t> numeric_end_, cat_cols_, card_cols_, ability_cols_, column_end_;
        std::vector<std::string> new_types_, new_tables_;  // placed by finish()
};

// The registry this binary was built with (src/featurize/registry.json) and its file text.
const Registry &builtin_registry();
const std::string &builtin_registry_text();

// The registry's problems as one error message (the first few and how many more).
std::string registry_error(const Registry &registry);

// layout_hash: FNV-1a 64 of layout_version, the dims, the tables' capacities, the card rows and
// the targets' library depth (header comment).
uint64_t layout_hash(const Registry &r, uint64_t card_rows, uint32_t library_depth);

// The problems that make `now` not an append to `base` (an entry of base missing or moved in now,
// a capacity or a dim changed); empty when now extends base.
std::vector<std::string> registry_moves(const Registry &base, const Registry &now);

}  // namespace featurize

#endif /* FEATURIZE_REGISTRY_H */
