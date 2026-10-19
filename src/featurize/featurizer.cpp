#include "featurizer.h"

#include <algorithm>
#include <utility>

static bool check_frame(const featurize::FeatureSpec &spec, const rmstate::DecodedFrame &state, std::string *err);
static void fill_row(const featurize::FeatureSpec &spec, const featurize::CardTables &tables,
    const featurize::TokenType &type, const std::vector<rmstate::Value> &row, int64_t tok,
    const std::vector<std::pair<uint64_t, int64_t>> &owners, const std::vector<int64_t> &base,
    const std::vector<const rmstate::DecodedTable *> &tabs, featurize::Tensors *out);
static int64_t owner_of(const std::vector<std::pair<uint64_t, int64_t>> &owners, uint64_t handle);
static int64_t card_value(const featurize::CardTables &tables, const rmstate::Value &v, rmstate::Storage storage);
static int64_t ability_value(
    const featurize::CardTables &tables, const featurize::ColumnOp &op, const std::vector<rmstate::Value> &row);
static bool featurize_frame(const featurize::FeatureSpec &spec, const featurize::CardTables &tables,
    const rmstate::DecodedFrame &state, featurize::Tensors *out, std::string *err);
static void read_decision(
    const featurize::FeatureSpec &spec, const rmstate::DecodedTable *header, featurize::Tensors *out);
template <typename T>
static void append_ragged(featurize::RmtbFile *file, const char *name, featurize::Dtype dtype, uint64_t width,
    const std::vector<featurize::Tensors> &list, const std::vector<T> featurize::Tensors::*member,
    size_t (featurize::Tensors::*rows)() const);

static bool check_frame(const featurize::FeatureSpec &spec, const rmstate::DecodedFrame &state, std::string *err) {
    if (state.header.kind != static_cast<uint8_t>(rmstate::FrameKind::STATE)) {
        *err = "the frame is not a STATE frame";
        return false;
    }
    if (state.header.schema_hash != spec.schema_hash) {
        *err = "the frame's schema hash " + featurize::hex16(state.header.schema_hash) + " is not the spec's " +
               featurize::hex16(spec.schema_hash);
        return false;
    }
    return true;
}

// The token owning `handle` (owners sorted by handle), or -1.
static int64_t owner_of(const std::vector<std::pair<uint64_t, int64_t>> &owners, uint64_t handle) {
    auto it = std::lower_bound(owners.begin(), owners.end(), std::make_pair(handle, static_cast<int64_t>(-1)));
    return it != owners.end() && it->first == handle ? it->second : -1;
}

// A card cell as a card-table row: -1 at none or past the table's cards.
static int64_t card_value(const featurize::CardTables &tables, const rmstate::Value &v, rmstate::Storage storage) {
    if (rmstate::value_is_all_ones(v, storage) || v.bits + 1 >= tables.card_rows) return -1;
    return static_cast<int64_t>(v.bits);
}

// An ability reference's ability row: (ab_card, ab_index), else ab_key, else -1.
static int64_t ability_value(
    const featurize::CardTables &tables, const featurize::ColumnOp &op, const std::vector<rmstate::Value> &row) {
    const rmstate::Value &card = row[op.col], &index = row[op.aux1];
    int64_t r = -1;
    if (!rmstate::value_is_all_ones(card, op.storage) && !rmstate::value_is_all_ones(index, op.aux1_storage))
        r = tables.ability_row(card.bits, index.bits);
    const std::string &key = row[op.aux2].text;
    if (r < 0 && !key.empty()) r = tables.key_row(key);
    return r;
}

static void fill_row(const featurize::FeatureSpec &spec, const featurize::CardTables &tables,
    const featurize::TokenType &type, const std::vector<rmstate::Value> &row, int64_t tok,
    const std::vector<std::pair<uint64_t, int64_t>> &owners, const std::vector<int64_t> &base,
    const std::vector<const rmstate::DecodedTable *> &tabs, featurize::Tensors *out) {
    const size_t t = static_cast<size_t>(tok);
    float *num = &out->tok_num[t * spec.F];
    int64_t *cat = spec.C ? &out->tok_cat[t * spec.C] : nullptr;
    int64_t *card = spec.K ? &out->tok_card[t * spec.K] : nullptr;
    int64_t *ab = spec.Q ? &out->tok_ability[t * spec.Q] : nullptr;
    for (const featurize::ColumnOp &op : type.ops) {
        const rmstate::Value &v = row[op.col];
        switch (op.kind) {
            case featurize::OpKind::NUMERIC:
                num[op.slot] = featurize::cell_value(op.transform, 0, v, op.storage);
                break;
            case featurize::OpKind::INDEX:
                num[op.slot] = featurize::cell_value(featurize::Transform::INDEX, 0, v, op.storage);
                num[op.flag] = featurize::cell_value(featurize::Transform::NONE_FLAG, 0, v, op.storage);
                break;
            case featurize::OpKind::MASK:
                for (uint32_t b = 0; b < op.width && b < 64; ++b) num[op.slot + b] = (v.bits >> b) & 1u ? 1.0f : 0.0f;
                break;
            case featurize::OpKind::CATEGORICAL:
                cat[op.slot] = op.offset + (v.bits < op.width ? static_cast<int64_t>(v.bits) + 1 : 0);
                break;
            case featurize::OpKind::HASHED:
                cat[op.slot] =
                    op.offset + (v.text.empty()
                                        ? 0
                                        : 1 + static_cast<int64_t>(
                                                  rmstate::fnv1a64(v.text.data(), v.text.size()) % spec.hash_buckets));
                break;
            case featurize::OpKind::CARD: {
                const int64_t c = card_value(tables, v, op.storage);
                card[op.slot] = c;
                num[op.flag] = rmstate::value_is_all_ones(v, op.storage) ? 1.0f : 0.0f;
                if (op.face_slot >= 0)
                    card[op.face_slot] =
                        c >= 0 && row[op.aux1].bits != 0 ? tables.back_face[static_cast<size_t>(c)] : c;
                break;
            }
            case featurize::OpKind::ABILITY:
                ab[op.slot] = ability_value(tables, op, row);
                break;
            case featurize::OpKind::HANDLE: {
                if (v.bits == 0) break;
                const int64_t dst = owner_of(owners, v.bits);
                if (dst >= 0 && dst != tok) out->rel.insert(out->rel.end(), {tok, dst, op.relation});
                break;
            }
            case featurize::OpKind::ROW: {
                const size_t tt = static_cast<size_t>(op.target);
                const size_t rows = tabs[tt] ? tabs[tt]->rows.size() : 0;
                if (!rmstate::value_is_all_ones(v, op.storage) && v.bits < rows)
                    out->rel.insert(out->rel.end(), {tok, base[tt] + static_cast<int64_t>(v.bits), op.relation});
                break;
            }
        }
    }
}

static void read_decision(
    const featurize::FeatureSpec &spec, const rmstate::DecodedTable *header, featurize::Tensors *out) {
    out->game = out->decision = 0;
    out->sideboarding = false;
    if (!header || header->rows.empty()) return;
    const std::vector<rmstate::Value> &row = header->rows[0];
    out->game = static_cast<int64_t>(row[static_cast<size_t>(spec.header_game)].bits);
    out->decision = static_cast<int64_t>(row[static_cast<size_t>(spec.header_decision)].bits);
    out->sideboarding = row[static_cast<size_t>(spec.header_sideboarding)].bits != 0;
}

static bool featurize_frame(const featurize::FeatureSpec &spec, const featurize::CardTables &tables,
    const rmstate::DecodedFrame &state, featurize::Tensors *out, std::string *err) {
    if (!check_frame(spec, state, err)) return false;
    const size_t n_types = spec.types.size();
    std::vector<const rmstate::DecodedTable *> tabs(n_types, nullptr);
    for (const rmstate::DecodedTable &t : state.tables) {
        const int32_t ty = spec.type_of_table[t.id];
        if (ty >= 0) tabs[static_cast<size_t>(ty)] = &t;
    }
    std::vector<int64_t> base(n_types + 1, 0);
    for (size_t i = 0; i < n_types; ++i)
        base[i + 1] = base[i] + static_cast<int64_t>(tabs[i] ? tabs[i]->rows.size() : 0);
    const size_t n = static_cast<size_t>(base[n_types]);
    out->viewer = state.header.viewer;
    read_decision(spec, tabs[static_cast<size_t>(spec.header_type)], out);
    out->tok_type.resize(n);
    out->tok_num.assign(n * spec.F, 0.0f);
    out->tok_cat.assign(n * spec.C, 0);
    out->tok_card.assign(n * spec.K, -1);
    out->tok_ability.assign(n * spec.Q, -1);
    out->tok_mask.assign(n, 1);
    out->rel.clear();
    // Handle owners in type order, so the lowest table id wins a handle two tables own.
    std::vector<std::pair<uint64_t, int64_t>> owners;
    for (size_t i = 0; i < n_types; ++i) {
        if (!spec.types[i].owns_handles || !tabs[i]) continue;
        for (size_t r = 0; r < tabs[i]->rows.size(); ++r)
            owners.emplace_back(tabs[i]->rows[r][spec.types[i].handle_col].bits, base[i] + static_cast<int64_t>(r));
    }
    std::stable_sort(owners.begin(), owners.end(),
        [](const std::pair<uint64_t, int64_t> &a, const std::pair<uint64_t, int64_t> &b) { return a.first < b.first; });
    owners.erase(std::unique(owners.begin(), owners.end(),
                     [](const std::pair<uint64_t, int64_t> &a, const std::pair<uint64_t, int64_t> &b) {
                         return a.first == b.first;
                     }),
        owners.end());
    for (size_t i = 0; i < n_types; ++i) {
        if (!tabs[i]) continue;
        for (size_t r = 0; r < tabs[i]->rows.size(); ++r) {
            const int64_t tok = base[i] + static_cast<int64_t>(r);
            out->tok_type[static_cast<size_t>(tok)] = static_cast<int64_t>(i);
            fill_row(spec, tables, spec.types[i], tabs[i]->rows[r], tok, owners, base, tabs, out);
        }
    }
    const size_t m = static_cast<size_t>(spec.menu_type);
    const size_t a = tabs[m] ? tabs[m]->rows.size() : 0;
    out->act_tok.resize(a);
    for (size_t i = 0; i < a; ++i) out->act_tok[i] = base[m] + static_cast<int64_t>(i);
    out->act_mask.assign(a, 1);
    return true;
}

// One tensor of every decision: the values concatenated (`width` per row, 0 for a tensor of rank
// 1; `rows` per decision) and the row offsets.
template <typename T>
static void append_ragged(featurize::RmtbFile *file, const char *name, featurize::Dtype dtype, uint64_t width,
    const std::vector<featurize::Tensors> &list, const std::vector<T> featurize::Tensors::*member,
    size_t (featurize::Tensors::*rows)() const) {
    std::vector<T> values;
    std::vector<int64_t> offsets = {0};
    for (const featurize::Tensors &t : list) {
        const std::vector<T> &v = t.*member;
        values.insert(values.end(), v.begin(), v.end());
        offsets.push_back(offsets.back() + static_cast<int64_t>((t.*rows)()));
    }
    std::vector<uint64_t> shape = {static_cast<uint64_t>(offsets.back())};
    if (width) shape.push_back(width);
    if (dtype == featurize::Dtype::F32) {
        file->arrays.push_back(
            featurize::rmtb_floats(name, dtype, shape, std::vector<float>(values.begin(), values.end())));
    } else {
        file->arrays.push_back(
            featurize::rmtb_ints(name, dtype, shape, std::vector<int64_t>(values.begin(), values.end())));
    }
    file->arrays.push_back(featurize::rmtb_ints(
        std::string(name) + "_offsets", featurize::Dtype::I64, {static_cast<uint64_t>(offsets.size())}, offsets));
}

namespace featurize {

bool featurize_seat(const FeatureSpec &spec, const CardTables &tables, const rmstate::DecodedFrame &state, Tensors *out,
    std::string *err) {
    if (state.header.viewer != static_cast<uint8_t>(rmstate::Viewer::A) &&
        state.header.viewer != static_cast<uint8_t>(rmstate::Viewer::B)) {
        *err = "the actor entry point takes a seat's frame, not viewer " + std::to_string(state.header.viewer);
        return false;
    }
    return featurize_frame(spec, tables, state, out, err);
}

bool featurize_viewer(const FeatureSpec &spec, const CardTables &tables, const rmstate::DecodedFrame &state,
    Tensors *out, std::string *err) {
    return featurize_frame(spec, tables, state, out, err);
}

RmtbFile tensors_file(const FeatureSpec &spec, const std::vector<Tensors> &list) {
    RmtbFile f;
    f.kind = "tensors";
    f.has_vocab_hash = f.has_spec_hash = true;
    f.vocab_hash = spec.vocab_hash;
    f.spec_hash = spec.hash;
    std::vector<int64_t> viewer, game, decision, sideboarding;
    for (const Tensors &t : list) {
        viewer.push_back(t.viewer);
        game.push_back(t.game);
        decision.push_back(t.decision);
        sideboarding.push_back(t.sideboarding ? 1 : 0);
    }
    const uint64_t d = list.size();
    f.arrays.push_back(rmtb_ints("viewer", Dtype::I64, {d}, viewer));
    f.arrays.push_back(rmtb_ints("game", Dtype::I64, {d}, game));
    f.arrays.push_back(rmtb_ints("decision", Dtype::I64, {d}, decision));
    f.arrays.push_back(rmtb_ints("sideboarding", Dtype::I64, {d}, sideboarding));
    append_ragged(&f, "tok_type", Dtype::I64, 0, list, &Tensors::tok_type, &Tensors::n);
    append_ragged(&f, "tok_num", Dtype::F32, spec.F, list, &Tensors::tok_num, &Tensors::n);
    append_ragged(&f, "tok_cat", Dtype::I64, spec.C, list, &Tensors::tok_cat, &Tensors::n);
    append_ragged(&f, "tok_card", Dtype::I64, spec.K, list, &Tensors::tok_card, &Tensors::n);
    append_ragged(&f, "tok_ability", Dtype::I64, spec.Q, list, &Tensors::tok_ability, &Tensors::n);
    append_ragged(&f, "tok_mask", Dtype::U8, 0, list, &Tensors::tok_mask, &Tensors::n);
    append_ragged(&f, "rel", Dtype::I64, 3, list, &Tensors::rel, &Tensors::r);
    append_ragged(&f, "act_tok", Dtype::I64, 0, list, &Tensors::act_tok, &Tensors::a);
    append_ragged(&f, "act_mask", Dtype::U8, 0, list, &Tensors::act_mask, &Tensors::a);
    return f;
}

}  // namespace featurize
