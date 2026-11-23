#include "perturb.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace {

// The single-row tables (docs/featurizer.md §3.1): one token each.
const char *const SINGLE_ROW_TABLES[] = {"header", "turn", "pending", "pending_cost", "match", "k_meta"};

// What list_units reads of one frame table.
struct TableInfo {
        const rmstate::DecodedTable *table = nullptr;
        int handle = -1;  // its first handle column
        int seat = -1;    // its first absolute seat column (enum:seat)
        int rel = -1;     // its first player-rel column
        int card = -1;    // its first card column
};

// An object of the frame's object table.
struct ObjectInfo {
        int seat = -1;
        int64_t card = -1;
        std::string zone, label, group;
};

}  // namespace

static int first_column(const rmstate::TableDef &def, const char *tag_prefix);
static TableInfo table_info(const rmstate::DecodedTable *t);
static int abs_seat(const std::vector<rmstate::Value> &row, const TableInfo &ti, int viewer);
static std::string seat_letter(int seat);
static std::string player_name(int seat);
static std::string lower(const std::string &s);
static std::string card_name(const featurize::CardTables &tables, int64_t card);
static int64_t card_cell(const featurize::CardTables &tables, const rmstate::Value &v, rmstate::Storage storage);
static std::string slot_label(const featurize::TokenType &type, const featurize::SlotDef &s);
static std::string list_text(const char *tensor, const std::vector<uint32_t> &cols);
static void set_slot(const featurize::FeatureSpec &spec, const featurize::SlotDef &s, int32_t type, size_t tok,
    const featurize::BaselineChoice &b, featurize::Tensors *t);
static void remove_tokens(
    const featurize::FeatureSpec &spec, const std::vector<uint8_t> &removed, featurize::Tensors *t);
static featurize::Unit token_unit(const featurize::UnitList &list, int64_t token);
static featurize::Unit cell_unit(const std::vector<std::vector<featurize::SlotDef>> &slots,
    const featurize::UnitList &list, int64_t token, int32_t slot);
static std::string removal_problem(
    const featurize::Tensors &before, const featurize::Tensors &after, const std::vector<int64_t> &removed);
static std::string frame_problem(const rmstate::DecodedFrame &frame, const featurize::Unit &u);
static uint64_t next_random(uint64_t *state);
template <typename T>
static T mode_of(const std::map<T, uint64_t> &counts, T fallback);

static int first_column(const rmstate::TableDef &def, const char *tag_prefix) {
    const size_t n = std::char_traits<char>::length(tag_prefix);
    for (size_t c = 0; c < def.columns.size(); ++c)
        if (def.columns[c].tag.compare(0, n, tag_prefix) == 0) return static_cast<int>(c);
    return -1;
}

static TableInfo table_info(const rmstate::DecodedTable *t) {
    TableInfo ti;
    ti.table = t;
    if (!t || !t->def) return ti;
    ti.handle = first_column(*t->def, "handle");
    ti.seat = first_column(*t->def, "enum:seat");
    ti.rel = first_column(*t->def, "player-rel");
    ti.card = first_column(*t->def, "card");
    return ti;
}

// The row's player (0 A, 1 B): its first absolute seat column (1 A, 2 B), else its first player-rel
// column (1 the viewer, 2 the opponent) read from a seat's frame; -1 none.
static int abs_seat(const std::vector<rmstate::Value> &row, const TableInfo &ti, int viewer) {
    if (ti.seat >= 0) {
        const uint64_t v = row[static_cast<size_t>(ti.seat)].bits;
        if (v == 1 || v == 2) return static_cast<int>(v) - 1;
    }
    if (ti.rel >= 0 && viewer >= 0 && viewer <= 1) {
        const uint64_t v = row[static_cast<size_t>(ti.rel)].bits;
        if (v == 1) return viewer;
        if (v == 2) return 1 - viewer;
    }
    return -1;
}

static std::string seat_letter(int seat) {
    return seat == 0 ? "A" : seat == 1 ? "B" : "?";
}

static std::string player_name(int seat) {
    return "Player " + seat_letter(seat);
}

static std::string lower(const std::string &s) {
    std::string out = s;
    for (char &c : out)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return out;
}

static std::string card_name(const featurize::CardTables &tables, int64_t card) {
    if (card < 0) return "a hidden card";
    if (static_cast<size_t>(card) < tables.names.size() && !tables.names[static_cast<size_t>(card)].empty())
        return tables.names[static_cast<size_t>(card)];
    return "card " + std::to_string(card);
}

static int64_t card_cell(const featurize::CardTables &tables, const rmstate::Value &v, rmstate::Storage storage) {
    if (rmstate::value_is_all_ones(v, storage) || v.bits + 1 >= tables.card_rows) return -1;
    return static_cast<int64_t>(v.bits);
}

// "num[17]", "num[40-52]", "num[3,9]".
static std::string list_text(const char *tensor, const std::vector<uint32_t> &cols) {
    std::string out = std::string(tensor) + "[";
    std::vector<uint32_t> c = cols;
    std::sort(c.begin(), c.end());
    bool run = c.size() > 2;
    for (size_t i = 1; i < c.size(); ++i) run = run && c[i] == c[i - 1] + 1;
    if (run) return out + std::to_string(c.front()) + "-" + std::to_string(c.back()) + "]";
    for (size_t i = 0; i < c.size(); ++i) out += (i ? "," : "") + std::to_string(c[i]);
    return out + "]";
}

// "num[17] object.power (signed_log1p)", "cat[3] object.zone (enum:zone)", "tok_card[0,1] num[33] object.card".
static std::string slot_label(const featurize::TokenType &type, const featurize::SlotDef &s) {
    std::string out;
    const auto add = [&out](const std::string &part) { out += (out.empty() ? "" : " ") + part; };
    if (!s.card.empty()) add(list_text("tok_card", s.card));
    if (!s.ability.empty()) add(list_text("tok_ability", s.ability));
    if (!s.cat.empty()) add(list_text("cat", s.cat));
    if (!s.num.empty()) add(list_text("num", s.num));
    return out + " " + type.name + "." + s.column + (s.kind.empty() ? "" : " (" + s.kind + ")");
}

// Token `tok`'s columns of slot `s` set to baseline `b` of type `type`.
static void set_slot(const featurize::FeatureSpec &spec, const featurize::SlotDef &s, int32_t type, size_t tok,
    const featurize::BaselineChoice &b, featurize::Tensors *t) {
    const featurize::SlotStats *st = b.kind == featurize::Baseline::NONE ? nullptr : b.stats;
    const size_t ty = static_cast<size_t>(type);
    bool have = st && st->has(type) && st->F == spec.F && st->C == spec.C && st->K == spec.K && st->Q == spec.Q;
    size_t sample = 0;
    if (have && b.kind == featurize::Baseline::RESAMPLE) {
        have = st->samples[ty] > 0;
        if (have) sample = ty * featurize::SLOT_SAMPLES + static_cast<size_t>(b.draw % st->samples[ty]);
    }
    const bool resample = have && b.kind == featurize::Baseline::RESAMPLE;
    for (size_t i = 0; i < s.num.size(); ++i) {
        const size_t c = s.num[i];
        float v = s.num_flag[i] ? 1.0f : 0.0f;
        if (have) v = resample ? st->sample_num[sample * spec.F + c] : st->num_mean[ty * spec.F + c];
        t->tok_num[tok * spec.F + c] = v;
    }
    for (size_t i = 0; i < s.cat.size(); ++i) {
        const size_t c = s.cat[i];
        int64_t v = s.cat_none[i];
        if (have) v = resample ? st->sample_cat[sample * spec.C + c] : st->cat_mode[ty * spec.C + c];
        t->tok_cat[tok * spec.C + c] = v;
    }
    for (const uint32_t c : s.card) {
        int64_t v = -1;
        if (have) v = resample ? st->sample_card[sample * spec.K + c] : st->card_mode[ty * spec.K + c];
        t->tok_card[tok * spec.K + c] = v;
    }
    for (const uint32_t c : s.ability) {
        int64_t v = -1;
        if (have) v = resample ? st->sample_ability[sample * spec.Q + c] : st->ability_mode[ty * spec.Q + c];
        t->tok_ability[tok * spec.Q + c] = v;
    }
}

// Drops the removed tokens: the rest renumbered in order, every relation touching a removed token
// dropped, the others and act_tok renumbered.
static void remove_tokens(
    const featurize::FeatureSpec &spec, const std::vector<uint8_t> &removed, featurize::Tensors *t) {
    const size_t n = t->n();
    std::vector<int64_t> at(n, -1);
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) {
        if (removed[i]) continue;
        at[i] = static_cast<int64_t>(k);
        if (k != i) {
            t->tok_type[k] = t->tok_type[i];
            t->tok_mask[k] = t->tok_mask[i];
            std::copy_n(&t->tok_num[i * spec.F], spec.F, &t->tok_num[k * spec.F]);
            std::copy_n(&t->tok_cat[i * spec.C], spec.C, &t->tok_cat[k * spec.C]);
            std::copy_n(&t->tok_card[i * spec.K], spec.K, &t->tok_card[k * spec.K]);
            std::copy_n(&t->tok_ability[i * spec.Q], spec.Q, &t->tok_ability[k * spec.Q]);
        }
        ++k;
    }
    t->tok_type.resize(k);
    t->tok_mask.resize(k);
    t->tok_num.resize(k * spec.F);
    t->tok_cat.resize(k * spec.C);
    t->tok_card.resize(k * spec.K);
    t->tok_ability.resize(k * spec.Q);
    size_t r = 0;
    for (size_t i = 0; i < t->r(); ++i) {
        const int64_t s = at[static_cast<size_t>(t->rel[3 * i])], d = at[static_cast<size_t>(t->rel[3 * i + 1])];
        if (s < 0 || d < 0) continue;
        t->rel[3 * r] = s;
        t->rel[3 * r + 1] = d;
        t->rel[3 * r + 2] = t->rel[3 * i + 2];
        ++r;
    }
    t->rel.resize(3 * r);
    for (int64_t &a : t->act_tok) a = at[static_cast<size_t>(a)];
}

static featurize::Unit token_unit(const featurize::UnitList &list, int64_t token) {
    const featurize::TokenInfo &ti = list.tokens[static_cast<size_t>(token)];
    featurize::Unit u;
    u.kind = featurize::UnitKind::TOKEN;
    u.key = "tok:" + std::to_string(ti.type) + ":" + std::to_string(ti.table) + ":" + std::to_string(ti.row) + ":" +
            std::to_string(ti.sub) + ":" + std::to_string(ti.rank);
    u.tokens = {token};
    u.table = ti.table;
    u.row = ti.row;
    u.handle = ti.handle;
    u.owns = ti.owns;
    u.seat = ti.seat;
    u.group = ti.group;
    u.label = ti.label;
    u.tensors = ti.tensors;
    u.coarse = ti.single;
    return u;
}

static featurize::Unit cell_unit(const std::vector<std::vector<featurize::SlotDef>> &slots,
    const featurize::UnitList &list, int64_t token, int32_t slot) {
    const featurize::TokenInfo &ti = list.tokens[static_cast<size_t>(token)];
    const featurize::SlotDef &s = slots[static_cast<size_t>(ti.type)][static_cast<size_t>(slot)];
    featurize::Unit u = token_unit(list, token);
    u.kind = featurize::UnitKind::CELL;
    u.key = "cell:" + u.key.substr(4) + ":" + s.column;
    u.type = ti.type;
    u.slot = slot;
    u.column = s.column;
    u.label = ti.label + " \xC2\xB7 " + s.column;
    u.tensors = ti.tensors + " " + s.label;
    u.coarse = false;
    return u;
}

// "" when `after` is `before` with the tokens `removed` dropped (ascending), the rest in order, every
// relation between kept tokens kept (renumbered, in order) and none other, act_tok renumbered.
static std::string removal_problem(
    const featurize::Tensors &before, const featurize::Tensors &after, const std::vector<int64_t> &removed) {
    std::vector<int64_t> at(before.n(), 0);
    for (const int64_t r : removed) at[static_cast<size_t>(r)] = -1;
    int64_t k = 0;
    for (int64_t &a : at) a = a < 0 ? -1 : k++;
    if (after.n() != static_cast<size_t>(k))
        return "the change has " + std::to_string(after.n()) + " tokens, not " + std::to_string(k);
    std::vector<int64_t> want;
    for (size_t i = 0; i < before.n(); ++i)
        if (at[i] >= 0 && after.tok_type[static_cast<size_t>(at[i])] != before.tok_type[i])
            return "a kept token moved out of order";
    for (size_t i = 0; i < before.r(); ++i) {
        const int64_t s = at[static_cast<size_t>(before.rel[3 * i])],
                      d = at[static_cast<size_t>(before.rel[3 * i + 1])];
        if (s >= 0 && d >= 0) want.insert(want.end(), {s, d, before.rel[3 * i + 2]});
    }
    if (want != after.rel) return "the relations are not the decision's between kept tokens, renumbered";
    for (size_t a = 0; a < before.a(); ++a)
        if (after.act_tok[a] != at[static_cast<size_t>(before.act_tok[a])]) return "act_tok is not renumbered";
    return "";
}

// "" when the unit names rows `frame` holds: its table and row, its object or stack handle, its player.
static std::string frame_problem(const rmstate::DecodedFrame &frame, const featurize::Unit &u) {
    if (u.table >= 0) {
        const rmstate::DecodedTable *t = frame.table(static_cast<uint16_t>(u.table));
        if (!t || u.row < 0 || static_cast<size_t>(u.row) >= t->rows.size())
            return "names table " + std::to_string(u.table) + " row " + std::to_string(u.row) + ", not in the frame";
    }
    if (u.handle) {
        bool found = false;
        for (const rmstate::DecodedTable &t : frame.tables) {
            if (!t.def || t.def->columns.empty() || t.def->columns[0].name != "handle" ||
                t.def->columns[0].tag != "handle")
                continue;
            for (const std::vector<rmstate::Value> &row : t.rows) found = found || row[0].bits == u.handle;
        }
        if (!found) return "names handle " + std::to_string(u.handle) + ", which no object or stack row holds";
    }
    if (u.seat >= 0) {
        bool found = false;
        for (const rmstate::DecodedTable &t : frame.tables) {
            if (!t.def || t.def->name != "player") continue;
            const int c = first_column(*t.def, "enum:seat");
            for (const std::vector<rmstate::Value> &row : t.rows)
                found = found || (c >= 0 && row[static_cast<size_t>(c)].bits == static_cast<uint64_t>(u.seat) + 1);
        }
        if (!found) return "names player " + seat_letter(u.seat) + ", whom the frame's player table lacks";
    }
    return "";
}

static uint64_t next_random(uint64_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    return *state;
}

template <typename T>
static T mode_of(const std::map<T, uint64_t> &counts, T fallback) {
    T best = fallback;
    uint64_t n = 0;
    for (const auto &kv : counts)
        if (kv.second > n) {
            best = kv.first;
            n = kv.second;
        }
    return best;
}

namespace featurize {

const char *unit_kind_name(UnitKind k) {
    switch (k) {
        case UnitKind::TOKEN:
            return "token";
        case UnitKind::GROUP:
            return "group";
        case UnitKind::CARD:
            return "card";
        case UnitKind::SLOT:
            return "slot";
        case UnitKind::CELL:
            return "cell";
        case UnitKind::RELATION:
            return "relation";
    }
    return "?";
}

const char *baseline_name(Baseline b) {
    return b == Baseline::MEAN ? "mean" : b == Baseline::RESAMPLE ? "resample" : "none";
}

bool parse_baseline(const std::string &name, Baseline *out) {
    for (const Baseline b : {Baseline::NONE, Baseline::MEAN, Baseline::RESAMPLE})
        if (name == baseline_name(b)) {
            *out = b;
            return true;
        }
    return false;
}

std::vector<std::vector<SlotDef>> type_slots(const FeatureSpec &spec) {
    std::vector<std::vector<SlotDef>> out(spec.types.size());
    for (size_t ty = 0; ty < spec.types.size(); ++ty) {
        const TokenType &type = spec.types[ty];
        if (type.kind == TypeKind::ABSENT) continue;
        std::vector<SlotDef> &slots = out[ty];
        std::map<std::string, size_t> at;
        // A synthetic type's slots are its own (the ledger's numeric slots share source columns).
        const auto slot_of = [&](const std::string &name, const std::string &column) -> SlotDef & {
            const std::string key = column.empty() || type.kind == TypeKind::LEDGER ? name : column;
            auto it = at.find(key);
            if (it != at.end()) return slots[it->second];
            at[key] = slots.size();
            slots.emplace_back();
            slots.back().column = key;
            return slots.back();
        };
        for (const NumSlot &s : type.numeric) {
            SlotDef &d = slot_of(s.name, s.column);
            d.num.push_back(s.at);
            d.num_flag.push_back(s.transform == Transform::NONE_FLAG ? 1 : 0);
            if (d.kind.empty() || d.kind == transform_name(Transform::NONE_FLAG))
                d.kind = s.transform == Transform::MASK_BIT ? "mask" : transform_name(s.transform);
        }
        for (const CatSlot &s : type.categorical) {
            SlotDef &d = slot_of(s.name, s.column);
            d.cat.push_back(s.col);
            d.cat_none.push_back(s.offset);
            d.kind = s.source;
        }
        for (const CardSlot &s : type.cards) {
            SlotDef &d = slot_of(s.name, s.column);
            d.card.push_back(s.col);
            d.kind = "card";
        }
        for (const AbilitySlot &s : type.abilities) {
            SlotDef &d = slot_of(s.name, "");
            d.ability.push_back(s.col);
            d.kind = "ability";
        }
        for (SlotDef &d : slots) d.label = slot_label(type, d);
    }
    return out;
}

int UnitList::find(const std::string &key) const {
    for (size_t i = 0; i < units.size(); ++i)
        if (units[i].key == key) return static_cast<int>(i);
    return -1;
}

bool list_units(const FeatureSpec &spec, const CardTables &tables, const rmstate::Schema &schema,
    const rmstate::DecodedFrame &frame, const Tensors &t, const std::vector<TokenSource> &sources, UnitList *out,
    std::string *err) {
    if (sources.size() != t.n()) {
        *err = "the token sources (" + std::to_string(sources.size()) + ") are not the tensors' " +
               std::to_string(t.n()) + " tokens";
        return false;
    }
    const int viewer = frame.header.viewer <= 1 ? static_cast<int>(frame.header.viewer) : -1;
    const std::vector<std::vector<SlotDef>> slots = type_slots(spec);
    *out = UnitList();
    out->n = t.n();
    // The frame's tables, and its objects by handle.
    std::map<uint16_t, TableInfo> tables_by_id;
    for (const rmstate::DecodedTable &tab : frame.tables) tables_by_id[tab.id] = table_info(&tab);
    std::map<uint64_t, ObjectInfo> objects;
    const std::vector<std::string> &zones = enum_values(schema, "zone");
    const std::vector<std::string> &card_types = enum_values(schema, "card_type");
    int64_t land_bit = -1;
    for (size_t b = 0; b < card_types.size(); ++b)
        if (card_types[b] == "LAND") land_bit = static_cast<int64_t>(b);
    for (const auto &kv : tables_by_id) {
        const TableInfo &ti = kv.second;
        if (!ti.table->def || ti.table->def->name != "object") continue;
        const rmstate::TableDef &def = *ti.table->def;
        const int zone = column_index(def, "zone"), owner = column_index(def, "owner"),
                  controller = column_index(def, "controller"), card = column_index(def, "card"),
                  types = column_index(def, "types");
        for (size_t r = 0; r < ti.table->rows.size(); ++r) {
            const std::vector<rmstate::Value> &row = ti.table->rows[r];
            ObjectInfo o;
            const uint64_t zv = zone >= 0 ? row[static_cast<size_t>(zone)].bits : UINT64_MAX;
            const std::string z = zv < zones.size() ? lower(zones[static_cast<size_t>(zv)]) : std::string();
            o.zone = z;
            const bool in_play = z == "battlefield" || z == "stack";
            const int seat_col = in_play ? controller : owner;
            if (seat_col >= 0) {
                const uint64_t v = row[static_cast<size_t>(seat_col)].bits;
                o.seat = v == 1 || v == 2 ? static_cast<int>(v) - 1 : -1;
            }
            if (card >= 0)
                o.card =
                    card_cell(tables, row[static_cast<size_t>(card)], def.columns[static_cast<size_t>(card)].storage);
            const bool land = land_bit >= 0 && types >= 0 && ((row[static_cast<size_t>(types)].bits >> land_bit) & 1u);
            if (z == "stack") {
                o.group = "stack";
            } else {
                o.group = seat_letter(o.seat) + ":" + (z.empty() ? "zone" : z) +
                          (z == "battlefield" ? (land ? ":lands" : ":permanents") : "");
            }
            o.label = card_name(tables, o.card) + " (" + seat_letter(o.seat) + ", " + z + ")";
            objects[row[0].bits] = o;
        }
    }
    // The tokens.
    std::vector<int64_t> rank(spec.types.size(), 0);
    out->tokens.resize(t.n());
    for (size_t i = 0; i < t.n(); ++i) {
        const TokenSource &src = sources[i];
        TokenInfo &ti = out->tokens[i];
        ti.type = src.type;
        ti.table = src.table;
        ti.row = src.table >= 0 ? src.row : -1;
        ti.sub = src.sub;
        ti.rank = rank[static_cast<size_t>(src.type)]++;
        const TokenType &type = spec.types[static_cast<size_t>(src.type)];
        ti.menu = src.type == spec.menu_type;
        ti.tensors = type.name + "[" + std::to_string(ti.rank) + "]";
        for (size_t k = 0; k < spec.K && ti.card < 0; ++k) ti.card = t.tok_card[i * spec.K + k];
        if (type.kind == TypeKind::LEDGER) {
            const int64_t rel = t.tok_cat[i * spec.C + type.rel_col] - type.rel_offset - 1;
            ti.seat = rel == 1 ? viewer : rel == 2 && viewer >= 0 ? 1 - viewer : -1;
            ti.card = t.tok_card[i * spec.K + type.card_col];
            ti.group = seat_letter(ti.seat) + ":ledger";
            ti.label = "ledger: " + card_name(tables, ti.card) + " (" + seat_letter(ti.seat) + ")";
            continue;
        }
        auto tab = src.table >= 0 ? tables_by_id.find(static_cast<uint16_t>(src.table)) : tables_by_id.end();
        if (tab == tables_by_id.end() || ti.row < 0 || static_cast<size_t>(ti.row) >= tab->second.table->rows.size() ||
            !tab->second.table->def) {
            ti.label = ti.tensors;
            ti.table = -1;
            ti.row = -1;
            continue;
        }
        const TableInfo &info = tab->second;
        const std::string &name = info.table->def->name;
        const std::vector<rmstate::Value> &row = info.table->rows[static_cast<size_t>(ti.row)];
        for (const char *s : SINGLE_ROW_TABLES) ti.single = ti.single || name == s;
        if (name == "object") {
            ti.handle = row[0].bits;
            ti.owns = type.table == info.table->id;
            const ObjectInfo &o = objects[ti.handle];
            ti.seat = o.seat;
            ti.zone = o.zone;
            ti.group = o.group;
            ti.label = ti.owns ? o.label : o.label + " \xC2\xB7 " + type.name;
            if (ti.owns) ti.card = o.card;
            continue;
        }
        if (name == "stack") {
            ti.handle = row[0].bits;
            ti.owns = objects.find(ti.handle) == objects.end();
            ti.seat = abs_seat(row, info, viewer);
            ti.group = "stack";
            const int64_t c = info.card >= 0 ? card_cell(tables, row[static_cast<size_t>(info.card)],
                                                   info.table->def->columns[static_cast<size_t>(info.card)].storage)
                                             : -1;
            ti.label = "stack: " + card_name(tables, c) + " (" + seat_letter(ti.seat) + ")";
            continue;
        }
        ti.seat = abs_seat(row, info, viewer);
        // A row with no player of its own takes the player of the row its first row: column names.
        for (size_t c = 0; ti.seat < 0 && c < info.table->def->columns.size(); ++c) {
            const rmstate::ColumnDef &col = info.table->def->columns[c];
            if (col.tag.compare(0, 4, "row:") != 0 || rmstate::value_is_all_ones(row[c], col.storage)) continue;
            for (const auto &kv : tables_by_id) {
                const TableInfo &target = kv.second;
                if (!target.table->def || target.table->def->name != col.tag.substr(4) ||
                    row[c].bits >= target.table->rows.size())
                    continue;
                ti.seat = abs_seat(target.table->rows[static_cast<size_t>(row[c].bits)], target, viewer);
            }
            break;
        }
        if (name.compare(0, 2, "k_") == 0) {
            if (!ti.single) ti.group = seat_letter(ti.seat) + ":knowledge";
            ti.label = "Knowledge " + name + " row " + std::to_string(ti.row) +
                       (ti.seat >= 0 ? " (" + seat_letter(ti.seat) + ")" : "");
            continue;
        }
        if (name == "player" || name.compare(0, 7, "player_") == 0) {
            ti.group = seat_letter(ti.seat) + ":player";
            ti.label = name == "player" ? player_name(ti.seat)
                                        : player_name(ti.seat) + " \xC2\xB7 " + name + " row " + std::to_string(ti.row);
            continue;
        }
        // A row of an object or a stack object (its first handle column), else of the table.
        const uint64_t h = info.handle >= 0 ? row[static_cast<size_t>(info.handle)].bits : 0;
        auto obj = objects.find(h);
        if (h && obj != objects.end() && name != "menu") {
            ti.handle = h;
            ti.seat = obj->second.seat;
            ti.group = obj->second.group;
            ti.label = obj->second.label + " \xC2\xB7 " + name;
            continue;
        }
        if (ti.menu) {
            ti.label = "menu row " + std::to_string(ti.row);
            continue;
        }
        if (h && (name == "stack_target" || name == "stack_mode")) ti.group = "stack";
        if (!ti.single && ti.group.empty()) ti.group = "table:" + name;
        ti.label = ti.single ? name : name + " row " + std::to_string(ti.row);
    }
    // The coarse units: the groups (players among them), the single-row tokens, the cards.
    std::vector<std::string> order;
    std::map<std::string, std::vector<int64_t>> groups;
    for (size_t i = 0; i < t.n(); ++i) {
        const TokenInfo &ti = out->tokens[i];
        if (ti.menu || ti.group.empty()) continue;
        if (groups.find(ti.group) == groups.end()) order.push_back(ti.group);
        groups[ti.group].push_back(static_cast<int64_t>(i));
    }
    for (const std::string &g : order) {
        Unit u;
        u.kind = UnitKind::GROUP;
        u.key = "grp:" + g;
        u.tokens = groups[g];
        u.group = g;
        u.coarse = true;
        const size_t colon = g.find(':');
        const std::string head = colon == std::string::npos ? g : g.substr(0, colon);
        const std::string rest = colon == std::string::npos ? "" : g.substr(colon + 1);
        if (g == "stack") {
            u.label = "the stack";
        } else if (head == "table") {
            u.label = "table " + rest;
        } else if (rest == "player") {
            u.label = player_name(head == "A" ? 0 : head == "B" ? 1 : -1);
            u.seat = head == "A" ? 0 : head == "B" ? 1 : -1;
        } else {
            std::string what = rest;
            std::replace(what.begin(), what.end(), ':', ' ');
            if (what == "battlefield lands") what = "lands";
            if (what == "battlefield permanents") what = "permanents";
            if (what == "ledger") what = "card ledger";
            u.label = "Player " + head + " \xC2\xB7 " + what;
            u.seat = head == "A" ? 0 : head == "B" ? 1 : -1;
        }
        u.tensors = std::to_string(u.tokens.size()) + " tokens (" + g + ")";
        out->units.push_back(u);
    }
    std::map<int64_t, std::pair<size_t, size_t>> cards;  // card → (object tokens, tokens) holding it
    for (size_t i = 0; i < t.n(); ++i) {
        std::set<int64_t> here;
        for (size_t k = 0; k < spec.K; ++k)
            if (t.tok_card[i * spec.K + k] >= 0) here.insert(t.tok_card[i * spec.K + k]);
        for (const int64_t c : here) {
            std::pair<size_t, size_t> &n = cards[c];
            ++n.second;
            if (out->tokens[i].owns && !out->tokens[i].zone.empty()) ++n.first;
        }
    }
    for (const auto &kv : cards) {
        Unit u;
        u.kind = UnitKind::CARD;
        u.key = "card:" + std::to_string(kv.first);
        u.card = kv.first;
        const size_t n = kv.second.first ? kv.second.first : kv.second.second;
        u.label = card_name(tables, kv.first) + (n > 1 ? " \xC3\x97" + std::to_string(n) : "");
        u.tensors = "card " + std::to_string(kv.first) + " in " + std::to_string(kv.second.second) + " tokens";
        u.coarse = true;
        out->units.push_back(u);
    }
    // Every token but the menu's (the single-row ones coarse), every slot of the types present, every
    // relation type present.
    std::vector<size_t> per_type(spec.types.size(), 0);
    for (size_t i = 0; i < t.n(); ++i) {
        ++per_type[static_cast<size_t>(t.tok_type[i])];
        if (!out->tokens[i].menu) out->units.push_back(token_unit(*out, static_cast<int64_t>(i)));
    }
    for (size_t ty = 0; ty < spec.types.size(); ++ty) {
        if (!per_type[ty]) continue;
        for (size_t s = 0; s < slots[ty].size(); ++s) {
            Unit u;
            u.kind = UnitKind::SLOT;
            u.type = static_cast<int32_t>(ty);
            u.slot = static_cast<int32_t>(s);
            u.column = slots[ty][s].column;
            u.key = "slot:" + std::to_string(ty) + ":" + u.column;
            u.label = spec.types[ty].name + "." + u.column + " (all " + std::to_string(per_type[ty]) + " " +
                      spec.types[ty].name + (per_type[ty] == 1 ? ")" : " tokens)");
            u.tensors = slots[ty][s].label;
            out->units.push_back(u);
        }
    }
    std::map<int64_t, size_t> rels;
    for (size_t r = 0; r < t.r(); ++r) ++rels[t.rel[3 * r + 2]];
    for (const auto &kv : rels) {
        Unit u;
        u.kind = UnitKind::RELATION;
        u.relation = static_cast<int32_t>(kv.first);
        const RelationType &rt = static_cast<size_t>(kv.first) < spec.relations.size()
                                     ? spec.relations[static_cast<size_t>(kv.first)]
                                     : RelationType();
        u.column = rt.table + "." + rt.column;
        u.key = "rel:" + std::to_string(kv.first);
        u.label = u.column + " (" + std::to_string(kv.second) + (kv.second == 1 ? " relation)" : " relations)");
        u.tensors = "rel type " + std::to_string(kv.first) + " " + u.column;
        out->units.push_back(u);
    }
    return true;
}

std::vector<Unit> unit_children(const FeatureSpec &spec, const std::vector<std::vector<SlotDef>> &slots,
    const UnitList &list, const Tensors &t, const Unit &unit) {
    std::vector<Unit> out;
    switch (unit.kind) {
        case UnitKind::TOKEN: {
            const int32_t ty = list.tokens[static_cast<size_t>(unit.tokens[0])].type;
            for (size_t s = 0; s < slots[static_cast<size_t>(ty)].size(); ++s)
                out.push_back(cell_unit(slots, list, unit.tokens[0], static_cast<int32_t>(s)));
            break;
        }
        case UnitKind::GROUP:
            for (const int64_t tok : unit.tokens) out.push_back(token_unit(list, tok));
            break;
        case UnitKind::CARD:
            for (size_t i = 0; i < t.n() && i < list.tokens.size(); ++i) {
                if (list.tokens[i].menu) continue;
                for (size_t k = 0; k < spec.K; ++k)
                    if (t.tok_card[i * spec.K + k] == unit.card) {
                        out.push_back(token_unit(list, static_cast<int64_t>(i)));
                        break;
                    }
            }
            break;
        case UnitKind::SLOT:
            for (size_t i = 0; i < t.n() && i < list.tokens.size(); ++i)
                if (t.tok_type[i] == unit.type)
                    out.push_back(cell_unit(slots, list, static_cast<int64_t>(i), unit.slot));
            break;
        case UnitKind::CELL:
        case UnitKind::RELATION:
            break;
    }
    return out;
}

bool apply_units(const FeatureSpec &spec, const std::vector<std::vector<SlotDef>> &slots,
    const std::vector<const Unit *> &units, const BaselineChoice &b, Tensors *t, std::string *err) {
    const size_t n = t->n();
    std::vector<uint8_t> removed(n, 0);
    std::vector<uint8_t> drop_rel;
    bool any_removed = false;
    const auto slot_ok = [&](int32_t type, int32_t slot) {
        return type >= 0 && static_cast<size_t>(type) < slots.size() && slot >= 0 &&
               static_cast<size_t>(slot) < slots[static_cast<size_t>(type)].size();
    };
    for (const Unit *u : units) {
        switch (u->kind) {
            case UnitKind::TOKEN:
            case UnitKind::GROUP:
                for (const int64_t tok : u->tokens) {
                    if (tok < 0 || static_cast<size_t>(tok) >= n) {
                        *err = u->key + ": token " + std::to_string(tok) + " of " + std::to_string(n);
                        return false;
                    }
                    if (t->tok_type[static_cast<size_t>(tok)] == spec.menu_type) {
                        *err = u->key + ": a menu token is never removed";
                        return false;
                    }
                    removed[static_cast<size_t>(tok)] = 1;
                    any_removed = true;
                }
                break;
            case UnitKind::CARD: {
                const uint64_t apc = spec.abilities_per_card;
                for (size_t i = 0; i < n; ++i) {
                    const size_t ty = static_cast<size_t>(t->tok_type[i]);
                    for (size_t k = 0; k < spec.K; ++k) {
                        if (t->tok_card[i * spec.K + k] != u->card) continue;
                        t->tok_card[i * spec.K + k] = -1;
                        if (ty >= slots.size()) continue;
                        for (const SlotDef &s : slots[ty]) {
                            if (std::find(s.card.begin(), s.card.end(), static_cast<uint32_t>(k)) == s.card.end())
                                continue;
                            for (size_t f = 0; f < s.num.size(); ++f)
                                if (s.num_flag[f]) t->tok_num[i * spec.F + s.num[f]] = 1.0f;
                        }
                    }
                    for (size_t q = 0; q < spec.Q && apc; ++q) {
                        const int64_t a = t->tok_ability[i * spec.Q + q];
                        if (a >= 0 && static_cast<uint64_t>(a) < spec.key_base &&
                            static_cast<int64_t>(static_cast<uint64_t>(a) / apc) == u->card)
                            t->tok_ability[i * spec.Q + q] = -1;
                    }
                }
                break;
            }
            case UnitKind::SLOT:
                if (!slot_ok(u->type, u->slot)) {
                    *err = u->key + ": no such slot";
                    return false;
                }
                for (size_t i = 0; i < n; ++i)
                    if (t->tok_type[i] == u->type)
                        set_slot(
                            spec, slots[static_cast<size_t>(u->type)][static_cast<size_t>(u->slot)], u->type, i, b, t);
                break;
            case UnitKind::CELL: {
                const int64_t tok = u->tokens.empty() ? -1 : u->tokens[0];
                if (tok < 0 || static_cast<size_t>(tok) >= n || t->tok_type[static_cast<size_t>(tok)] != u->type ||
                    !slot_ok(u->type, u->slot)) {
                    *err = u->key + ": no such token or slot";
                    return false;
                }
                set_slot(spec, slots[static_cast<size_t>(u->type)][static_cast<size_t>(u->slot)], u->type,
                    static_cast<size_t>(tok), b, t);
                break;
            }
            case UnitKind::RELATION:
                if (u->relation < 0) {
                    *err = u->key + ": no relation type";
                    return false;
                }
                if (drop_rel.size() <= static_cast<size_t>(u->relation))
                    drop_rel.resize(static_cast<size_t>(u->relation) + 1, 0);
                drop_rel[static_cast<size_t>(u->relation)] = 1;
                break;
        }
    }
    if (!drop_rel.empty()) {
        size_t r = 0;
        for (size_t i = 0; i < t->r(); ++i) {
            const int64_t ty = t->rel[3 * i + 2];
            if (ty >= 0 && static_cast<size_t>(ty) < drop_rel.size() && drop_rel[static_cast<size_t>(ty)]) continue;
            std::copy_n(&t->rel[3 * i], 3, &t->rel[3 * r]);
            ++r;
        }
        t->rel.resize(3 * r);
    }
    if (any_removed) remove_tokens(spec, removed, t);
    return true;
}

SlotStatsBuilder::SlotStatsBuilder(const FeatureSpec &spec) : spec_(spec) {
    const size_t T = spec.types.size();
    s_.spec_hash = spec.hash;
    s_.types = T;
    s_.F = spec.F;
    s_.C = spec.C;
    s_.K = spec.K;
    s_.Q = spec.Q;
    s_.count.assign(T, 0);
    s_.samples.assign(T, 0);
    s_.sample_num.assign(T * SLOT_SAMPLES * spec.F, 0.0f);
    s_.sample_cat.assign(T * SLOT_SAMPLES * spec.C, 0);
    s_.sample_card.assign(T * SLOT_SAMPLES * spec.K, -1);
    s_.sample_ability.assign(T * SLOT_SAMPLES * spec.Q, -1);
    sum_.assign(T * spec.F, 0.0);
    cat_.assign(T * spec.C, {});
    card_.assign(T * spec.K, {});
    ability_.assign(T * spec.Q, {});
}

void SlotStatsBuilder::add(const Tensors &t) {
    const size_t F = spec_.F, C = spec_.C, K = spec_.K, Q = spec_.Q;
    ++s_.decisions;
    for (size_t i = 0; i < t.n(); ++i) {
        const int64_t type = t.tok_type[i];
        if (type < 0 || static_cast<size_t>(type) >= s_.types) continue;
        const size_t ty = static_cast<size_t>(type);
        const uint64_t seen = ++s_.count[ty];
        for (size_t c = 0; c < F; ++c) sum_[ty * F + c] += t.tok_num[i * F + c];
        for (size_t c = 0; c < C; ++c) ++cat_[ty * C + c][t.tok_cat[i * C + c]];
        for (size_t c = 0; c < K; ++c) ++card_[ty * K + c][t.tok_card[i * K + c]];
        for (size_t c = 0; c < Q; ++c) ++ability_[ty * Q + c][t.tok_ability[i * Q + c]];
        size_t slot = SLOT_SAMPLES;
        if (s_.samples[ty] < SLOT_SAMPLES) {
            slot = s_.samples[ty]++;
        } else {
            const uint64_t j = next_random(&rng_) % seen;
            if (j < SLOT_SAMPLES) slot = static_cast<size_t>(j);
        }
        if (slot == SLOT_SAMPLES) continue;
        const size_t at = ty * SLOT_SAMPLES + slot;
        std::copy_n(&t.tok_num[i * F], F, &s_.sample_num[at * F]);
        std::copy_n(&t.tok_cat[i * C], C, &s_.sample_cat[at * C]);
        std::copy_n(&t.tok_card[i * K], K, &s_.sample_card[at * K]);
        std::copy_n(&t.tok_ability[i * Q], Q, &s_.sample_ability[at * Q]);
    }
}

SlotStats SlotStatsBuilder::finish() const {
    SlotStats s = s_;
    const size_t T = s.types;
    s.num_mean.assign(T * s.F, 0.0f);
    s.cat_mode.assign(T * s.C, 0);
    s.card_mode.assign(T * s.K, -1);
    s.ability_mode.assign(T * s.Q, -1);
    for (size_t ty = 0; ty < T; ++ty) {
        if (!s.count[ty]) continue;
        for (size_t c = 0; c < s.F; ++c)
            s.num_mean[ty * s.F + c] = static_cast<float>(sum_[ty * s.F + c] / static_cast<double>(s.count[ty]));
        for (size_t c = 0; c < s.C; ++c) s.cat_mode[ty * s.C + c] = mode_of<int64_t>(cat_[ty * s.C + c], 0);
        for (size_t c = 0; c < s.K; ++c) s.card_mode[ty * s.K + c] = mode_of<int64_t>(card_[ty * s.K + c], -1);
        for (size_t c = 0; c < s.Q; ++c) s.ability_mode[ty * s.Q + c] = mode_of<int64_t>(ability_[ty * s.Q + c], -1);
    }
    return s;
}

RmtbFile slot_stats_file(const SlotStats &s) {
    RmtbFile f;
    f.kind = "slot_stats";
    f.has_spec_hash = true;
    f.spec_hash = s.spec_hash;
    f.set_attr("decisions", static_cast<int64_t>(s.decisions));
    f.set_attr("sample", static_cast<int64_t>(SLOT_SAMPLES));
    const uint64_t T = s.types, S = SLOT_SAMPLES;
    std::vector<int64_t> count(s.count.begin(), s.count.end()), samples(s.samples.begin(), s.samples.end());
    f.arrays.push_back(rmtb_ints("count", Dtype::I64, {T}, count));
    f.arrays.push_back(rmtb_floats("num_mean", Dtype::F32, {T, s.F}, s.num_mean));
    f.arrays.push_back(rmtb_ints("cat_mode", Dtype::I64, {T, s.C}, s.cat_mode));
    f.arrays.push_back(rmtb_ints("card_mode", Dtype::I64, {T, s.K}, s.card_mode));
    f.arrays.push_back(rmtb_ints("ability_mode", Dtype::I64, {T, s.Q}, s.ability_mode));
    f.arrays.push_back(rmtb_ints("samples", Dtype::I64, {T}, samples));
    f.arrays.push_back(rmtb_floats("sample_num", Dtype::F32, {T, S, s.F}, s.sample_num));
    f.arrays.push_back(rmtb_ints("sample_cat", Dtype::I64, {T, S, s.C}, s.sample_cat));
    f.arrays.push_back(rmtb_ints("sample_card", Dtype::I64, {T, S, s.K}, s.sample_card));
    f.arrays.push_back(rmtb_ints("sample_ability", Dtype::I64, {T, S, s.Q}, s.sample_ability));
    return f;
}

bool parse_slot_stats(const RmtbFile &f, SlotStats *out, std::string *err) {
    if (f.kind != "slot_stats" || !f.has_spec_hash) {
        *err = "not a slot statistics file (kind " + f.kind + ")";
        return false;
    }
    const RmtbArray *count = f.find("count"), *mean = f.find("num_mean");
    const RmtbAttr *sample = f.attr("sample");
    if (!count || !mean || count->shape.size() != 1 || mean->shape.size() != 2 || !sample || !sample->is_int ||
        sample->i != SLOT_SAMPLES) {
        *err = "the slot statistics file lacks its arrays or has another sample size";
        return false;
    }
    SlotStats s;
    s.spec_hash = f.spec_hash;
    s.types = count->shape[0];
    s.F = mean->shape[1];
    const RmtbAttr *dec = f.attr("decisions");
    s.decisions = dec && dec->is_int ? static_cast<uint64_t>(dec->i) : 0;
    std::vector<int64_t> ints;
    const auto ints_of = [&](const char *name, size_t width, std::vector<int64_t> *v) {
        const RmtbArray *a = f.find(name);
        return a && a->shape.size() >= 2 && a->shape[0] == s.types && rmtb_values(*a, v) &&
               v->size() == s.types * width * (a->shape.size() == 3 ? SLOT_SAMPLES : 1);
    };
    const RmtbArray *cat = f.find("cat_mode"), *card = f.find("card_mode"), *ab = f.find("ability_mode");
    if (!cat || !card || !ab || cat->shape.size() != 2 || card->shape.size() != 2 || ab->shape.size() != 2) {
        *err = "the slot statistics file's modes are malformed";
        return false;
    }
    s.C = cat->shape[1];
    s.K = card->shape[1];
    s.Q = ab->shape[1];
    bool ok = rmtb_values(*count, &ints) && ints.size() == s.types;
    if (ok) s.count.assign(ints.begin(), ints.end());
    ok = ok && rmtb_values(*mean, &s.num_mean) && s.num_mean.size() == s.types * s.F;
    ok = ok && ints_of("cat_mode", s.C, &s.cat_mode) && ints_of("card_mode", s.K, &s.card_mode) &&
         ints_of("ability_mode", s.Q, &s.ability_mode);
    const RmtbArray *samples = f.find("samples"), *snum = f.find("sample_num");
    ok = ok && samples && rmtb_values(*samples, &ints) && ints.size() == s.types;
    if (ok) s.samples.assign(ints.begin(), ints.end());
    ok = ok && snum && rmtb_values(*snum, &s.sample_num) && s.sample_num.size() == s.types * SLOT_SAMPLES * s.F;
    ok = ok && ints_of("sample_cat", s.C, &s.sample_cat) && ints_of("sample_card", s.K, &s.sample_card) &&
         ints_of("sample_ability", s.Q, &s.sample_ability);
    for (size_t ty = 0; ok && ty < s.types; ++ty) ok = s.samples[ty] <= SLOT_SAMPLES;
    if (!ok) {
        *err = "the slot statistics file's arrays are malformed";
        return false;
    }
    *out = std::move(s);
    return true;
}

std::string tensor_problems(const FeatureSpec &spec, const CardTables &tables, const Tensors &t, size_t menu_rows) {
    const size_t n = t.n();
    if (t.tok_num.size() != n * spec.F || t.tok_cat.size() != n * spec.C || t.tok_card.size() != n * spec.K ||
        t.tok_ability.size() != n * spec.Q || t.tok_mask.size() != n || t.rel.size() % 3 != 0)
        return "a tensor's width is not the spec's";
    if (t.a() != menu_rows || t.act_mask.size() != menu_rows) return "act_tok has not one entry per menu row";
    // Per token type which columns are its slots'.
    struct Use {
            std::vector<uint8_t> num, card, ability;
            std::vector<const CatSlot *> cat;
    };
    std::vector<Use> uses(spec.types.size());
    for (size_t i = 0; i < spec.types.size(); ++i) {
        const TokenType &type = spec.types[i];
        Use &u = uses[i];
        u.num.assign(spec.F, 0);
        u.cat.assign(spec.C, nullptr);
        u.card.assign(spec.K, 0);
        u.ability.assign(spec.Q, 0);
        for (const NumSlot &x : type.numeric)
            if (x.at < spec.F) u.num[x.at] = 1;
        for (const CatSlot &x : type.categorical)
            if (x.col < spec.C) u.cat[x.col] = &x;
        for (const CardSlot &x : type.cards)
            if (x.col < spec.K) u.card[x.col] = 1;
        for (const AbilitySlot &x : type.abilities)
            if (x.col < spec.Q) u.ability[x.col] = 1;
    }
    for (size_t i = 0; i < n; ++i) {
        const int64_t ty = t.tok_type[i];
        if (ty < 0 || static_cast<size_t>(ty) >= spec.types.size() ||
            spec.types[static_cast<size_t>(ty)].kind == TypeKind::ABSENT)
            return "tok_type out of range";
        const TokenType &type = spec.types[static_cast<size_t>(ty)];
        // Every column of a kind of slot is one of the type's slots (within its range) or reserved
        // (0 / -1); a numeric value other than 0 only at a slot's column.
        const Use &use = uses[static_cast<size_t>(ty)];
        for (size_t c = 0; c < spec.C; ++c) {
            const int64_t id = t.tok_cat[i * spec.C + c];
            const CatSlot *slot = use.cat[c];
            if (slot ? id < slot->offset || id >= slot->offset + slot->size : id != 0)
                return type.name + ": tok_cat column " + std::to_string(c) + " out of its slot's range";
        }
        for (size_t k = 0; k < spec.K; ++k) {
            const int64_t c = t.tok_card[i * spec.K + k];
            if (c < -1 || c >= static_cast<int64_t>(tables.card_rows) || (!use.card[k] && c != -1))
                return type.name + ": tok_card out of the card table";
        }
        for (size_t q = 0; q < spec.Q; ++q) {
            const int64_t a = t.tok_ability[i * spec.Q + q];
            if (a < -1 || a >= static_cast<int64_t>(tables.ability_rows()) || (!use.ability[q] && a != -1))
                return type.name + ": tok_ability out of the ability table";
        }
        for (size_t c = 0; c < spec.F; ++c)
            if (t.tok_num[i * spec.F + c] != 0.0f && !use.num[c])
                return type.name + ": tok_num column " + std::to_string(c) + " is no slot's but not 0";
    }
    for (size_t r = 0; r < t.r(); ++r) {
        const int64_t s = t.rel[3 * r], d = t.rel[3 * r + 1], ty = t.rel[3 * r + 2];
        if (s < 0 || d < 0 || static_cast<size_t>(s) >= n || static_cast<size_t>(d) >= n || ty < 0 ||
            static_cast<size_t>(ty) >= spec.relations.size() || s == d)
            return "a relation's token or type is out of range";
        const RelationType &rt = spec.relations[static_cast<size_t>(ty)];
        if (spec.types[static_cast<size_t>(t.tok_type[static_cast<size_t>(s)])].name != rt.table)
            return "a relation of type " + rt.table + "." + rt.column + " from another table's token";
    }
    for (const int64_t a : t.act_tok)
        if (a < 0 || static_cast<size_t>(a) >= n || t.tok_type[static_cast<size_t>(a)] != spec.menu_type)
            return "act_tok is not a menu token";
    return "";
}

std::vector<std::string> check_perturbations(const FeatureSpec &spec, const CardTables &tables,
    const rmstate::DecodedFrame &frame, const Tensors &t, const UnitList &list, const BaselineChoice &b,
    size_t *units) {
    std::vector<std::string> problems;
    const std::vector<std::vector<SlotDef>> slots = type_slots(spec);
    std::string err;
    Tensors same = t;
    if (!apply_units(spec, slots, {}, b, &same, &err) ||
        rmtb_encode(tensors_file(spec, {same})) != rmtb_encode(tensors_file(spec, {t})))
        problems.push_back("the empty change is not byte-identical");
    // Every listed unit, and the cells of each type's first token.
    std::vector<Unit> all = list.units;
    std::vector<uint8_t> first(spec.types.size(), 1);
    for (size_t i = 0; i < t.n(); ++i) {
        const size_t ty = static_cast<size_t>(t.tok_type[i]);
        if (!first[ty]) continue;
        first[ty] = 0;
        Unit tok;
        tok.kind = UnitKind::TOKEN;
        tok.tokens = {static_cast<int64_t>(i)};
        for (const Unit &c : unit_children(spec, slots, list, t, tok)) all.push_back(c);
    }
    *units = 0;
    for (const Unit &u : all) {
        ++*units;
        Tensors changed = t;
        const std::string at = u.key + " (" + u.label + "): ";
        if (!apply_units(spec, slots, {&u}, b, &changed, &err)) {
            problems.push_back(at + err);
            continue;
        }
        const std::string shape = tensor_problems(spec, tables, changed, t.a());
        if (!shape.empty()) problems.push_back(at + shape);
        if (u.kind == UnitKind::TOKEN || u.kind == UnitKind::GROUP) {
            std::vector<int64_t> removed = u.tokens;
            std::sort(removed.begin(), removed.end());
            removed.erase(std::unique(removed.begin(), removed.end()), removed.end());
            const std::string p = removal_problem(t, changed, removed);
            if (!p.empty()) problems.push_back(at + p);
        }
        const std::string f = frame_problem(frame, u);
        if (!f.empty()) problems.push_back(at + f);
    }
    return problems;
}

}  // namespace featurize
