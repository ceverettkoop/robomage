#ifndef FEATURIZE_PERTURB_H
#define FEATURIZE_PERTURB_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../state_format/decode.h"
#include "card_tables.h"
#include "feature_spec.h"
#include "featurizer.h"
#include "folds.h"
#include "rmtb.h"

// ── Perturbations of one decision's tensors: Attribution's occlusion units (docs/workbench_plan.md §2.8)
//
// Standard library and the decoder only. A unit is one input of a decision changed; it changes the
// featurized tensors (featurizer.h), never the frame:
//
//   TOKEN      one token removed: as if its frame row were not in the frame. A menu token is never
//              removed (that would change the menu)
//   GROUP      several tokens removed at once: one player's zone (B's lands, A's hand, the stack, a
//              player's knowledge or card ledger), a player with the rows that are the player's, or a
//              table's rows that belong to no object or player
//   CARD       every tok_card slot holding the card set to -1 in every token, its none flag set (read the
//              way a face-down card is), and the tok_ability slots of that card's abilities set to -1:
//              the card unknown, its properties included
//   SLOT       one slot of one token type in every token of that type: the columns its source column
//              makes (a numeric slot, an index's value and none flag, a mask block, a categorical, a card
//              slot with its current face and none flag, an ability slot) set to the baseline
//   CELL       one slot of one token
//   RELATION   every relation of one relation type removed
//
// Units change the tensors they were listed from (list_units): token indices are that decision's. A
// list of units is applied at once (apply_units): the slot, cell, card and relation changes first, then
// every removed token goes (its relations with it, the remaining tokens renumbered in order, act_tok
// and every relation renumbered), so units compose (Shapley's coalitions). The empty list changes
// nothing.
//
// Baselines for a slot or a cell: NONE, the featurizer's own encoding of absence (0, a none flag 1, the
// categorical's none id, -1); MEAN, each numeric column's mean over a reference sample of the type's
// tokens and each categorical, card and ability column's most frequent value; RESAMPLE, the columns of
// one token of the type drawn from the reference sample (`draw` picks which). The reference sample is
// the slot statistics (SlotStats, `rmfeat --slot-stats`): per token type the means, the modes and a
// reservoir of up to SLOT_SAMPLES tokens, keyed by the spec's hash. Without statistics for a type, MEAN
// and RESAMPLE fall back to NONE.
//
// Each unit carries what it maps to in the frame: its tokens' frame rows (table id, row), the object or
// stack handle a token is (or belongs to: a child row's first handle), the player (absolute seat), the
// zone group, and labels at the state level ("Ragavan (B, battlefield)", "Player A · life",
// "object.tapped (all 23 objects)", "Counterspell ×2") and at the tensors level ("object[12]",
// "num[17] object.power (signed_log1p)", "tok_card[0] card 412", "combat.blocker").

namespace featurize {

enum class UnitKind : uint8_t {
    TOKEN,
    GROUP,
    CARD,
    SLOT,
    CELL,
    RELATION
};
// "token", "group", "card", "slot", "cell", "relation".
const char *unit_kind_name(UnitKind k);

enum class Baseline : uint8_t {
    NONE,
    MEAN,
    RESAMPLE
};
// "none", "mean", "resample"; the inverse is false for another name.
const char *baseline_name(Baseline b);
bool parse_baseline(const std::string &name, Baseline *out);

// One slot of a token type: the tensor columns one source column makes.
struct SlotDef {
        std::string column;             // the source column ("power", "card"; a synthetic slot's name)
        std::vector<uint32_t> num;      // tok_num columns
        std::vector<uint8_t> num_flag;  // per num column: a none flag (1 at none)
        std::vector<uint32_t> cat;      // tok_cat columns
        std::vector<int64_t> cat_none;  // per cat column: its none id
        std::vector<uint32_t> card;     // tok_card columns
        std::vector<uint32_t> ability;  // tok_ability columns
        std::string kind;               // "signed_log1p", "mask", "index", "enum:zone", "card", "ability", …
        std::string label;              // "num[17] object.power (signed_log1p)"
};

// Per token type its slots, in the order the spec lists the type's slots (empty for an absent type).
std::vector<std::vector<SlotDef>> type_slots(const FeatureSpec &spec);

// A token as the frame has it.
struct TokenInfo {
        int32_t type = 0;
        int32_t table = -1;  // its frame table's id (a delta token's: its object's table), -1 none
        int64_t row = -1;    // that table's row, -1 none (a ledger token)
        int64_t sub = 0;
        int64_t rank = 0;     // its index among its type's tokens
        uint64_t handle = 0;  // the object or stack handle it is or belongs to (0 none)
        bool owns = false;    // `handle` is its own
        int seat = -1;        // the player (0 A, 1 B) it is or belongs to, -1 none
        int64_t card = -1;    // its first card (-1 none)
        std::string zone;     // an object's zone ("battlefield"), "" otherwise
        std::string group;    // its zone group's key ("A:battlefield:lands", "stack", "B:player"), "" none
        std::string label;    // "Ragavan (B, battlefield)", "Player A", "turn", "k_pool_card row 3"
        std::string tensors;  // "object[12]"
        bool menu = false;    // a menu token
        bool single = false;  // the only row of a single-row table (header, turn, pending, …)
};

struct Unit {
        UnitKind kind = UnitKind::TOKEN;
        std::string key;              // unique within the decision ("tok:24:3:0", "grp:A:hand", "slot:24:power")
        std::vector<int64_t> tokens;  // TOKEN, GROUP: removed; CELL: its token
        int64_t card = -1;            // CARD
        int32_t type = -1;            // SLOT, CELL: the token type
        int32_t slot = -1;            // SLOT, CELL: index into type_slots(spec)[type]
        int32_t relation = -1;        // RELATION
        // What it maps to in the frame.
        int32_t table = -1;  // TOKEN, CELL: its token's frame table id
        int64_t row = -1;
        uint64_t handle = 0;  // TOKEN, CELL: the object or stack handle
        bool owns = false;    // TOKEN, CELL: its token is the object (or stack object) `handle` names
        int seat = -1;        // the player it is or belongs to
        std::string column;   // SLOT, CELL: the source column; RELATION: "<table>.<column>"
        std::string group;    // TOKEN: its group's key; GROUP: its own key
        std::string label;    // state level
        std::string tensors;  // tensors level
        bool coarse = false;  // in the first pass (a group, a player, a single-row token, a card)
};

// A decision's tokens and units.
struct UnitList {
        std::vector<TokenInfo> tokens;  // per token of the tensors
        std::vector<Unit> units;        // the coarse units, every token (but menu tokens), every slot of
                                        // the types present, every relation type present
        size_t n = 0;                   // the tensors' token count
        // The unit of key `key`, or -1.
        int find(const std::string &key) const;
};

// The tokens and units of a seat's decision: `t` and `sources` from featurize_traced over `frame`
// (decoded with `schema`, whose enums name the zones). False with `err` when the sources do not
// describe `t`.
bool list_units(const FeatureSpec &spec, const CardTables &tables, const rmstate::Schema &schema,
    const rmstate::DecodedFrame &frame, const Tensors &t, const std::vector<TokenSource> &sources, UnitList *out,
    std::string *err);

// A unit's children, as units: a token's cells (one per slot of its type); a group's tokens; a card's
// holders (the tokens with a tok_card slot holding it); a slot's cells in every token of its type.
// None for a cell or a relation type. `slots` is type_slots(spec).
std::vector<Unit> unit_children(const FeatureSpec &spec, const std::vector<std::vector<SlotDef>> &slots,
    const UnitList &list, const Tensors &t, const Unit &unit);

inline constexpr uint32_t SLOT_SAMPLES = 32;

// The reference sample of the slot baselines (header comment).
struct SlotStats {
        uint64_t spec_hash = 0;
        size_t types = 0, F = 0, C = 0, K = 0, Q = 0;
        uint64_t decisions = 0;
        std::vector<uint64_t> count;        // [types] tokens seen
        std::vector<float> num_mean;        // [types, F]
        std::vector<int64_t> cat_mode;      // [types, C]
        std::vector<int64_t> card_mode;     // [types, K]
        std::vector<int64_t> ability_mode;  // [types, Q]
        std::vector<uint32_t> samples;      // [types] tokens kept (≤ SLOT_SAMPLES)
        std::vector<float> sample_num;      // [types, SLOT_SAMPLES, F]
        std::vector<int64_t> sample_cat, sample_card, sample_ability;
        bool has(int32_t type) const {
            return type >= 0 && static_cast<size_t>(type) < count.size() && count[static_cast<size_t>(type)] > 0;
        }
};

// Accumulates decisions into a SlotStats: the reservoir drawn with a fixed generator, so the same
// decisions in the same order give the same file.
class SlotStatsBuilder {
    public:
        explicit SlotStatsBuilder(const FeatureSpec &spec);
        void add(const Tensors &t);
        SlotStats finish() const;

    private:
        const FeatureSpec &spec_;
        SlotStats s_;
        std::vector<double> sum_;
        std::vector<std::map<int64_t, uint64_t>> cat_, card_, ability_;  // per (type, column) value counts
        uint64_t rng_ = 0x9E3779B97F4A7C15ull;
};

// The statistics as an RMTB file of kind `slot_stats`, stamped with the spec's hash; and back (false
// with `err` for another kind or malformed arrays).
RmtbFile slot_stats_file(const SlotStats &s);
bool parse_slot_stats(const RmtbFile &f, SlotStats *out, std::string *err);

struct BaselineChoice {
        Baseline kind = Baseline::NONE;
        const SlotStats *stats = nullptr;  // MEAN, RESAMPLE (nullptr: NONE)
        uint64_t draw = 0;                 // RESAMPLE: which sampled token (per type, modulo its sample)
};

// Applies `units` (listed from `t`) to `t` at once (header comment); `slots` is type_slots(spec). False
// with `err` when a unit names a token, slot or relation `t` does not have, or a menu token to remove.
bool apply_units(const FeatureSpec &spec, const std::vector<std::vector<SlotDef>> &slots,
    const std::vector<const Unit *> &units, const BaselineChoice &b, Tensors *t, std::string *err);

// "" when `t` agrees with the spec and the tables: every tensor's width, act_tok one entry per menu row
// (`menu_rows`) and each a menu token, every token type present, every tok_cat column in its slot's
// range or the pad, tok_card and tok_ability within the tables and only at the type's slots, a numeric
// value other than 0 only at a slot's column, every relation's tokens in range and its type's table the
// source token's; else what is wrong.
std::string tensor_problems(const FeatureSpec &spec, const CardTables &tables, const Tensors &t, size_t menu_rows);

// The checks of `rmfeat --perturb --check` on one decision: the empty change byte-identical; every
// unit's change passing tensor_problems; every removed token's relations gone and the rest renumbered
// in order (each relation of the change one of the decision's between kept tokens, and each of those
// kept); every unit naming a row of `frame`. Returns the problems (empty when none); `units` counts the
// units changed.
std::vector<std::string> check_perturbations(const FeatureSpec &spec, const CardTables &tables,
    const rmstate::DecodedFrame &frame, const Tensors &t, const UnitList &list, const BaselineChoice &b, size_t *units);

}  // namespace featurize

#endif /* FEATURIZE_PERTURB_H */
