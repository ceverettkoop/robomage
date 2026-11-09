#include "featurize_layout.h"

#include <algorithm>
#include <map>

#include "../../src/featurize/columns.h"
#include "../../src/featurize/featurizer.h"
#include "../../src/featurize/registry.h"

using featurize::Featurization;
using featurize::Tensors;
using rmcheck::LayoutVariants;

namespace {

const char *const FAKE_KEYWORD = "FakeKeyword";
const char *const KEYWORD_ENUM = "keyword";
// The first token card id: the vocab's cards are below it (src/card_vocab.h).
const uint64_t TOKEN_BAND = 900;

}  // namespace

static bool registry_with_fake(const featurize::Registry &base, bool key, featurize::Registry *out, std::string *err);
static bool pick_cards(const Featurization &f, uint64_t *source, uint64_t *fresh, std::string *err);
static void append_card(const rmstate::Schema &schema, rmstate::DecodedFrame *cards, uint64_t source, uint64_t fresh);
static void shift_enum(const rmstate::Schema &schema, rmstate::DecodedFrame *frame, const std::string &enum_name);
static bool build_variant(const Featurization &f, const rmstate::Schema &schema, const rmstate::DecodedFrame &cards,
    const featurize::Registry &reg, uint64_t vocab_hash, Featurization *out, std::string *err);
static std::string same_layout(const featurize::FeatureSpec &a, const featurize::FeatureSpec &b);
static std::string same_columns(const std::vector<std::string> &a, const std::vector<std::string> &b, size_t *renamed);
static std::vector<float> row_of(const std::vector<float> &values, size_t width, size_t r);
static std::map<int64_t, std::vector<std::vector<float>>> parts_by_ability(const featurize::CardTables &t);
static std::string check_appended_tables(const Featurization &f, const LayoutVariants &v);
static std::string check_renamed_tables(const Featurization &f, const LayoutVariants &v);
static std::string expect_refusal(bool built, const std::string &err, const std::string &cause);
static std::string check_capacities(const Featurization &f, const LayoutVariants &v);

// `base` with the fake keyword value appended to the keyword enum (and, with `key`, its
// `keyword:` ability key).
static bool registry_with_fake(const featurize::Registry &base, bool key, featurize::Registry *out, std::string *err) {
    *out = base;
    out->append = true;
    out->enum_index(KEYWORD_ENUM, FAKE_KEYWORD, 0);
    if (key) out->key_row(std::string("keyword:") + FAKE_KEYWORD);
    out->append = false;
    out->added.clear();
    if (out->ok()) return true;
    *err = featurize::registry_error(*out);
    return false;
}

// The vocab card with the most ability rows and no back face (lowest id on a tie), and the next free
// card id after every vocab card.
static bool pick_cards(const Featurization &f, uint64_t *source, uint64_t *fresh, std::string *err) {
    const rmstate::Schema &schema = f.source.schema;
    const rmstate::TableDef *card = featurize::table_named(schema, "card");
    const rmstate::TableDef *ab = featurize::table_named(schema, "card_ability");
    const rmstate::DecodedTable *cards = card ? f.source.cards.table(static_cast<uint16_t>(card->id)) : nullptr;
    const rmstate::DecodedTable *abilities = ab ? f.source.cards.table(static_cast<uint16_t>(ab->id)) : nullptr;
    const int c_card = card ? featurize::column_index(*card, "card") : -1;
    const int c_face = card ? featurize::column_index(*card, "face") : -1;
    const int a_card = ab ? featurize::column_index(*ab, "card") : -1;
    if (!cards || !abilities || c_card < 0 || c_face < 0 || a_card < 0) {
        *err = "the cards frame has no card or card_ability rows";
        return false;
    }
    std::map<uint64_t, size_t> count;
    std::map<uint64_t, bool> back;
    uint64_t top = 0;
    for (const std::vector<rmstate::Value> &row : cards->rows) {
        const uint64_t id = row[static_cast<size_t>(c_card)].bits;
        if (id >= TOKEN_BAND) continue;
        top = std::max(top, id);
        count[id];
        if (row[static_cast<size_t>(c_face)].bits != 0) back[id] = true;
    }
    for (const std::vector<rmstate::Value> &row : abilities->rows) {
        auto it = count.find(row[static_cast<size_t>(a_card)].bits);
        if (it != count.end()) ++it->second;
    }
    size_t best = 0;
    for (const auto &c : count)
        if (!back[c.first] && c.second > best) {
            best = c.second;
            *source = c.first;
        }
    *fresh = top + 1;
    if (best == 0 || *fresh >= TOKEN_BAND || *fresh + 1 >= f.tables.card_rows) {
        *err = "no single-faced vocab card with abilities, or no free card id below the token band";
        return false;
    }
    return true;
}

// Every CARDS row of card `source` (a table whose first column is the card) copied to card `fresh`
// (its face_card too).
static void append_card(const rmstate::Schema &schema, rmstate::DecodedFrame *cards, uint64_t source, uint64_t fresh) {
    for (rmstate::DecodedTable &t : cards->tables) {
        const rmstate::TableDef *def = schema.find_table(t.id);
        if (!def || def->frame_kind != rmstate::FrameKind::CARDS || def->columns.empty() ||
            def->columns[0].name != "card")
            continue;
        const int face_card = def->name == "card" ? featurize::column_index(*def, "face_card") : -1;
        const size_t n = t.rows.size();
        for (size_t r = 0; r < n; ++r) {
            if (t.rows[r][0].bits != source) continue;
            std::vector<rmstate::Value> copy = t.rows[r];
            copy[0].bits = fresh;
            if (face_card >= 0 && copy[static_cast<size_t>(face_card)].bits == source)
                copy[static_cast<size_t>(face_card)].bits = fresh;
            t.rows.push_back(copy);
        }
    }
}

// Every column of `frame` over enum `enum_name` as if one value were inserted before its first: an
// enum cell (not none) one more, a mask cell shifted a bit up.
static void shift_enum(const rmstate::Schema &schema, rmstate::DecodedFrame *frame, const std::string &enum_name) {
    for (rmstate::DecodedTable &t : frame->tables) {
        const rmstate::TableDef *def = schema.find_table(t.id);
        if (!def) continue;
        for (size_t c = 0; c < def->columns.size(); ++c) {
            const rmstate::ColumnDef &col = def->columns[c];
            const bool is_enum = col.tag == "enum:" + enum_name, is_mask = col.tag == "mask:" + enum_name;
            if (!is_enum && !is_mask) continue;
            for (std::vector<rmstate::Value> &row : t.rows) {
                rmstate::Value &v = row[c];
                if (is_mask)
                    v.bits <<= 1;
                else if (!rmstate::value_is_all_ones(v, col.storage))
                    v.bits += 1;
            }
        }
    }
}

static bool build_variant(const Featurization &f, const rmstate::Schema &schema, const rmstate::DecodedFrame &cards,
    const featurize::Registry &reg, uint64_t vocab_hash, Featurization *out, std::string *err) {
    out->source = f.source;
    out->source.schema = schema;
    out->source.cards = cards;
    out->registry = reg;
    out->options = f.options;
    out->text_model = f.text_model;
    if (!featurize::build_card_tables(
            out->source.schema, out->source.cards, f.tables.card_rows, vocab_hash, &out->registry, &out->tables, err))
        return false;
    out->tables.text_rows = f.tables.text_rows;
    out->tables.text_width = f.tables.text_width;
    return featurize::build_spec(out->source.schema, out->tables, out->options, &out->registry, &out->spec, err);
}

// "" when `b` lays every slot, range, relation and type out where `a` does (its tables' columns
// aside: same_columns).
static std::string same_layout(const featurize::FeatureSpec &a, const featurize::FeatureSpec &b) {
    if (a.layout_hash != b.layout_hash) return "another layout_hash";
    if (a.F != b.F || a.C != b.C || a.K != b.K || a.Q != b.Q || a.G != b.G ||
        a.categorical_size != b.categorical_size || a.type_capacity != b.type_capacity ||
        a.relation_capacity != b.relation_capacity || a.types.size() != b.types.size() ||
        a.relations.size() != b.relations.size())
        return "other dims";
    for (size_t i = 0; i < a.types.size(); ++i) {
        const featurize::TokenType &x = a.types[i], &y = b.types[i];
        if (x.name != y.name || x.kind != y.kind || x.num_at != y.num_at || x.num_capacity != y.num_capacity ||
            x.numeric.size() != y.numeric.size() || x.categorical.size() != y.categorical.size() ||
            x.cards.size() != y.cards.size() || x.abilities.size() != y.abilities.size() || x.relations != y.relations)
            return "token type " + x.name + " differs";
        for (size_t s = 0; s < x.numeric.size(); ++s)
            if (x.numeric[s].name != y.numeric[s].name || x.numeric[s].at != y.numeric[s].at)
                return "numeric slot " + x.name + "." + x.numeric[s].name + " moved";
        for (size_t s = 0; s < x.categorical.size(); ++s)
            if (x.categorical[s].name != y.categorical[s].name || x.categorical[s].col != y.categorical[s].col ||
                x.categorical[s].offset != y.categorical[s].offset || x.categorical[s].size != y.categorical[s].size)
                return "categorical slot " + x.name + "." + x.categorical[s].name + " moved";
        for (size_t s = 0; s < x.cards.size(); ++s)
            if (x.cards[s].name != y.cards[s].name || x.cards[s].col != y.cards[s].col)
                return "card slot " + x.name + "." + x.cards[s].name + " moved";
        for (size_t s = 0; s < x.abilities.size(); ++s)
            if (x.abilities[s].col != y.abilities[s].col) return "ability slot of " + x.name + " moved";
    }
    for (size_t i = 0; i < a.relations.size(); ++i)
        if (a.relations[i].table != b.relations[i].table || a.relations[i].column != b.relations[i].column ||
            a.relations[i].target != b.relations[i].target)
            return "relation type " + std::to_string(i) + " moved";
    if (a.abilities_per_card != b.abilities_per_card || a.key_base != b.key_base || a.ability_rows != b.ability_rows ||
        a.card_rows != b.card_rows)
        return "the ability rows' layout differs";
    for (size_t k = 0; k < a.ab_keys.size(); ++k) {
        auto it = std::find(b.ab_keys.begin(), b.ab_keys.end(), a.ab_keys[k]);
        if (it == b.ab_keys.end() || b.key_rows[static_cast<size_t>(it - b.ab_keys.begin())] != a.key_rows[k])
            return "ability key " + a.ab_keys[k] + " moved";
    }
    return "";
}

// "" when the column names are the same but where `a` names a reserved column (…#i) that `b` names
// after the fake keyword (counted in `renamed`).
static std::string same_columns(const std::vector<std::string> &a, const std::vector<std::string> &b, size_t *renamed) {
    if (a.size() != b.size()) return "another width";
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i]) continue;
        const size_t hash = a[i].rfind('#');
        const std::string tail = std::string("=") + FAKE_KEYWORD;
        if (hash == std::string::npos || b[i].size() < tail.size() ||
            b[i].compare(b[i].size() - tail.size(), tail.size(), tail) != 0 ||
            a[i].compare(0, hash, b[i], 0, b[i].size() - tail.size()) != 0)
            return "column " + std::to_string(i) + " is " + a[i] + ", then " + b[i];
        ++*renamed;
    }
    return "";
}

static std::vector<float> row_of(const std::vector<float> &values, size_t width, size_t r) {
    return std::vector<float>(
        values.begin() + static_cast<long>(r * width), values.begin() + static_cast<long>((r + 1) * width));
}

// Per ability row its part rows' features, sorted.
static std::map<int64_t, std::vector<std::vector<float>>> parts_by_ability(const featurize::CardTables &t) {
    std::map<int64_t, std::vector<std::vector<float>>> out;
    const size_t width = t.part_columns.size();
    for (size_t r = 0; r < t.part_rows(); ++r) out[t.part_ability[r]].push_back(row_of(t.part, width, r));
    for (auto &p : out) std::sort(p.second.begin(), p.second.end());
    return out;
}

// The appended card's rows are the copied card's, at its own card and ability rows; every other row
// (cards, abilities, keys, parts) is f's.
static std::string check_appended_tables(const Featurization &f, const LayoutVariants &v) {
    const featurize::CardTables &a = f.tables, &b = v.appended.tables;
    const size_t cw = a.card_columns.size(), aw = a.ability_columns.size();
    for (size_t c = 0; c < a.card_rows; ++c) {
        const size_t from = c == v.new_card ? static_cast<size_t>(v.source_card) : c;
        if (row_of(b.card, cw, c) != row_of(a.card, cw, from)) return "card row " + std::to_string(c) + " differs";
    }
    const uint64_t per = a.abilities_per_card;
    const auto moved = [&](size_t r) {
        return r / per == v.new_card && r < a.key_base ? static_cast<size_t>(v.source_card * per + r % per) : r;
    };
    for (size_t r = 0; r < a.ability_rows(); ++r) {
        const size_t from = moved(r);
        const bool fresh = from != r;
        if (row_of(b.ability, aw, r) != row_of(a.ability, aw, from) || b.ability_index[r] != a.ability_index[from] ||
            b.ability_card[r] !=
                (fresh && a.ability_card[from] >= 0 ? static_cast<int64_t>(v.new_card) : a.ability_card[from]))
            return "ability row " + std::to_string(r) + " differs";
    }
    for (uint64_t c = 0; c + 1 < a.card_rows; ++c)
        for (uint64_t i = 0; i < per; ++i)
            if (c != v.new_card && a.ability_row(c, i) != b.ability_row(c, i))
                return "card " + std::to_string(c) + "'s ability " + std::to_string(i) + " has another row";
    const std::map<int64_t, std::vector<std::vector<float>>> pa = parts_by_ability(a), pb = parts_by_ability(b);
    for (const auto &p : pb) {
        auto it = pa.find(static_cast<int64_t>(moved(static_cast<size_t>(p.first))));
        if (it == pa.end() || it->second != p.second)
            return "the parts of ability row " + std::to_string(p.first) + " differ";
    }
    for (const auto &p : pa) {
        auto it = pb.find(p.first);
        if (it == pb.end() || it->second != p.second)
            return "the parts of ability row " + std::to_string(p.first) + " moved";
    }
    return "";
}

// The renamed inputs' card table is f's; their ability rows f's but the fake keyword's key row.
static std::string check_renamed_tables(const Featurization &f, const LayoutVariants &v) {
    const featurize::CardTables &a = f.tables, &b = v.renamed.tables;
    if (a.card != b.card || a.back_face != b.back_face) return "the card table differs";
    const size_t aw = a.ability_columns.size();
    const int64_t fake = b.key_row(std::string("keyword:") + FAKE_KEYWORD);
    for (size_t r = 0; r < a.ability_rows(); ++r)
        if (static_cast<int64_t>(r) != fake && row_of(a.ability, aw, r) != row_of(b.ability, aw, r))
            return "ability row " + std::to_string(r) + " differs";
    if (fake < static_cast<int64_t>(a.key_base) || a.part != b.part || a.part_ability != b.part_ability)
        return "the fake keyword's key row or the parts differ";
    return "";
}

// "" when a build was refused with an error naming `cause`.
static std::string expect_refusal(bool built, const std::string &err, const std::string &cause) {
    if (built) return "built, expected a refusal naming \"" + cause + "\"";
    if (err.find(cause) == std::string::npos) return "refused with \"" + err + "\", expected \"" + cause + "\"";
    return "";
}

// A full enum, a card past abilities_per_card and a registry lacking a value are refused.
static std::string check_capacities(const Featurization &f, const LayoutVariants &v) {
    featurize::CardTables t;
    std::string err;
    // The keyword enum full: the renamed schema's fake keyword has no room.
    featurize::Registry full = f.registry;
    for (featurize::RegEnum &e : full.enums)
        if (e.name == KEYWORD_ENUM) e.capacity = static_cast<uint32_t>(e.values.size());
    full.append = true;
    std::string bad = expect_refusal(featurize::build_card_tables(v.renamed.source.schema, v.renamed.source.cards,
                                         f.tables.card_rows, f.tables.vocab_hash, &full, &t, &err),
        err, "is full");
    if (!bad.empty()) return "a full enum: " + bad;
    // The compiled-in registry lacks the renamed schema's fake keyword.
    featurize::Registry strict = f.registry;
    err.clear();
    bad = expect_refusal(featurize::build_card_tables(v.renamed.source.schema, v.renamed.source.cards,
                             f.tables.card_rows, f.tables.vocab_hash, &strict, &t, &err),
        err, std::string("has no value ") + FAKE_KEYWORD);
    if (!bad.empty()) return "a value the registry lacks: " + bad;
    // The appended card with abilities up to index abilities_per_card.
    rmstate::DecodedFrame cards = v.appended.source.cards;
    const rmstate::TableDef *ab = featurize::table_named(f.source.schema, "card_ability");
    const int a_card = featurize::column_index(*ab, "card"), a_index = featurize::column_index(*ab, "index");
    for (rmstate::DecodedTable &tab : cards.tables) {
        if (tab.id != static_cast<uint16_t>(ab->id)) continue;
        std::vector<rmstate::Value> model;
        uint64_t n = 0;
        for (const std::vector<rmstate::Value> &row : tab.rows)
            if (row[static_cast<size_t>(a_card)].bits == v.new_card) {
                model = row;
                ++n;
            }
        for (; n <= f.tables.abilities_per_card; ++n) {
            model[static_cast<size_t>(a_index)].bits = n;
            tab.rows.push_back(model);
        }
    }
    featurize::Registry reg = v.appended.registry;
    err.clear();
    bad = expect_refusal(
        featurize::build_card_tables(f.source.schema, cards, f.tables.card_rows, f.tables.vocab_hash, &reg, &t, &err),
        err, "abilities_per_card");
    return bad.empty() ? "" : "a card past abilities_per_card: " + bad;
}

namespace rmcheck {

bool build_layout_variants(const Featurization &f, LayoutVariants *out, std::string *err) {
    if (!pick_cards(f, &out->source_card, &out->new_card, err)) return false;
    featurize::Registry reg;
    rmstate::DecodedFrame cards = f.source.cards;
    append_card(f.source.schema, &cards, out->source_card, out->new_card);
    if (!registry_with_fake(f.registry, false, &reg, err) ||
        !build_variant(f, f.source.schema, cards, reg, f.tables.vocab_hash + 1, &out->appended, err)) {
        *err = "the appended inputs: " + *err;
        return false;
    }
    rmstate::Schema schema = f.source.schema;
    for (rmstate::EnumTable &e : schema.enums)
        if (e.name == KEYWORD_ENUM) e.values.insert(e.values.begin(), FAKE_KEYWORD);
    cards = f.source.cards;
    shift_enum(f.source.schema, &cards, KEYWORD_ENUM);
    if (!registry_with_fake(f.registry, true, &reg, err) ||
        !build_variant(f, schema, cards, reg, f.tables.vocab_hash, &out->renamed, err)) {
        *err = "the renamed inputs: " + *err;
        return false;
    }
    return true;
}

void featurize_layout_part(Report &rep, const Featurization &f, const LayoutVariants &v) {
    size_t renamed = 0, errors = 0;
    std::string bad = same_layout(f.spec, v.appended.spec);
    if (bad.empty()) bad = same_columns(f.spec.card_columns, v.appended.spec.card_columns, &renamed);
    if (bad.empty()) bad = same_columns(f.spec.ability_columns, v.appended.spec.ability_columns, &renamed);
    if (bad.empty()) bad = same_columns(f.spec.part_columns, v.appended.spec.part_columns, &renamed);
    if (bad.empty() && renamed != 1) bad = "the fake keyword names " + std::to_string(renamed) + " columns, not 1";
    if (bad.empty()) bad = check_appended_tables(f, v);
    if (!bad.empty() && ++errors) rep.error("featurize", "layout, one card and a keyword appended: " + bad);
    renamed = 0;
    bad = same_layout(f.spec, v.renamed.spec);
    if (bad.empty()) bad = same_columns(f.spec.card_columns, v.renamed.spec.card_columns, &renamed);
    if (bad.empty()) bad = check_renamed_tables(f, v);
    if (!bad.empty() && ++errors) rep.error("featurize", "layout, the keyword enum reordered: " + bad);
    bad = check_capacities(f, v);
    if (!bad.empty() && ++errors) rep.error("featurize", "layout capacities: " + bad);
    if (errors == 0)
        rep.info("OK   layout " + featurize::hex16(f.spec.layout_hash) + ": card " + std::to_string(v.new_card) +
                 " appended (a copy of card " + std::to_string(v.source_card) +
                 ") and a fake keyword: the same layout_hash, every slot, column, relation and key in place, every "
                 "existing card, ability and part row unchanged; the keyword enum reordered: the same positions "
                 "and card table; a full enum, a card past abilities_per_card and a missing value refused");
}

std::string check_layout_decision(
    const LayoutVariants &v, const rmstate::DecodedFrame &state, bool seat, bool renamed, const Tensors &t) {
    Tensors other;
    std::string err;
    const Featurization &a = v.appended;
    if (!(seat ? featurize::featurize_seat(a.spec, a.tables, state, &other, &err)
               : featurize::featurize_viewer(a.spec, a.tables, state, &other, &err)) ||
        !featurize::same_tensors(t, other))
        return "the inputs with a card and a keyword appended featurize it otherwise " + err;
    if (!renamed) return "";
    rmstate::DecodedFrame shifted = state;
    shift_enum(a.source.schema, &shifted, KEYWORD_ENUM);
    const Featurization &r = v.renamed;
    if (!(seat ? featurize::featurize_seat(r.spec, r.tables, shifted, &other, &err)
               : featurize::featurize_viewer(r.spec, r.tables, shifted, &other, &err)) ||
        !featurize::same_tensors(t, other))
        return "the inputs with the keyword enum reordered featurize it otherwise " + err;
    return "";
}

}  // namespace rmcheck
