#include "folds.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <utility>

namespace {

using Row = std::vector<rmstate::Value>;

// One k_pool row as the ledger reads it.
struct Pool {
        uint64_t rel = 0, kind = 0, anchor = 0, pos = 0, span = 0;
        bool library = false;
        bool run = false;     // a library run: a library pool holding no other library pool or exact position
        bool folded = false;  // its k_pool_card rows fold into the ledger
};

// An exact k_position row: an owner's library position whose card is known.
struct Exact {
        uint64_t rel = 0, anchor = 0, pos = 0, card = 0;
};

using LedgerKey = std::pair<uint64_t, uint64_t>;  // (player_rel, card)
using LedgerVals = std::array<uint32_t, featurize::L_SLOTS>;
using LedgerEntry = std::pair<LedgerKey, LedgerVals>;

// An object row's handle and the face vocab id its printed rows are read from (-1: none).
struct ObjectFace {
        uint64_t handle = 0;
        int64_t face = -1;
};

}  // namespace

static const rmstate::DecodedTable *table_of(const rmstate::DecodedFrame &state, int32_t id);
static uint64_t cell(const Row &row, int col);
static std::vector<Pool> ledger_pools(const featurize::LedgerPlan &l, const rmstate::DecodedFrame &state);
static std::vector<Exact> exact_positions(const featurize::LedgerPlan &l, const rmstate::DecodedFrame &state);
static void mark_runs(std::vector<Pool> *pools, const std::vector<Exact> &exact);
static void merge_entries(std::vector<LedgerEntry> *entries);
static void fold_ledger(const featurize::FeatureSpec &spec, const featurize::CardTables &tables,
    const rmstate::DecodedFrame &state, bool want_sources, featurize::FoldedRows *out);
static int value_cmp(const Row &a, const featurize::TokenType &t, const Row &printed);
static int rows_cmp(const Row &a, const Row &b, const featurize::TokenType &t);
static int64_t printed_face(const featurize::FeatureSpec &spec, const featurize::CardTables &tables, const Row &obj);
static std::vector<ObjectFace> object_faces(const featurize::FeatureSpec &spec, const featurize::CardTables &tables,
    const std::vector<const rmstate::DecodedTable *> &tabs);
static void fold_delta(const featurize::FeatureSpec &spec, const featurize::CardTables &tables, size_t type,
    const std::vector<const rmstate::DecodedTable *> &tabs, const std::vector<ObjectFace> &objects, bool want_sources,
    featurize::FoldedRows *out);

static const rmstate::DecodedTable *table_of(const rmstate::DecodedFrame &state, int32_t id) {
    return id >= 0 ? state.table(static_cast<uint16_t>(id)) : nullptr;
}

// Column `col` of `row` (0 when the column is not in the schema).
static uint64_t cell(const Row &row, int col) {
    return col >= 0 ? row[static_cast<size_t>(col)].bits : 0;
}

// The k_pool rows, which of them are library runs and which fold (folds.h).
static std::vector<Pool> ledger_pools(const featurize::LedgerPlan &l, const rmstate::DecodedFrame &state) {
    std::vector<Pool> pools;
    const rmstate::DecodedTable *t = table_of(state, l.k_pool);
    if (!t) return pools;
    std::map<uint64_t, size_t> libraries;  // per owner its library pools
    for (const Row &row : t->rows) {
        Pool p;
        p.rel = cell(row, l.pool_rel);
        p.kind = cell(row, l.pool_kind);
        p.anchor = cell(row, l.pool_anchor);
        p.pos = cell(row, l.pool_pos);
        p.span = cell(row, l.pool_span);
        p.library = p.kind == l.kind_library;
        p.folded = p.kind == l.kind_hand || p.kind == l.kind_sideboard;
        if (p.library) ++libraries[p.rel];
        pools.push_back(p);
    }
    for (Pool &p : pools)
        if (p.library) p.folded = libraries[p.rel] == 1;
    return pools;
}

static std::vector<Exact> exact_positions(const featurize::LedgerPlan &l, const rmstate::DecodedFrame &state) {
    std::vector<Exact> out;
    const rmstate::DecodedTable *t = table_of(state, l.k_position);
    if (!t || l.pos_card < 0) return out;
    for (const Row &row : t->rows) {
        if (rmstate::value_is_all_ones(row[static_cast<size_t>(l.pos_card)], l.pos_card_storage)) continue;
        out.push_back({cell(row, l.pos_rel), cell(row, l.pos_anchor), cell(row, l.pos_pos), cell(row, l.pos_card)});
    }
    return out;
}

// A library pool is a run when no other library pool of its owner lies strictly inside its range
// and no exact position of its owner inside it (a block of runs holds at least two runs, each a pool
// or exact positions; ranges of one anchor compared).
static void mark_runs(std::vector<Pool> *pools, const std::vector<Exact> &exact) {
    for (Pool &p : *pools) {
        if (!p.library) continue;
        const uint64_t end = p.pos + p.span;
        p.run = true;
        for (const Pool &q : *pools)
            if (&q != &p && q.library && q.rel == p.rel && q.anchor == p.anchor && q.pos >= p.pos &&
                q.pos + q.span <= end && q.span < p.span)
                p.run = false;
        for (const Exact &e : exact)
            if (e.rel == p.rel && e.anchor == p.anchor && e.pos >= p.pos && e.pos < end) p.run = false;
    }
}

// The contributions summed per key, in key order (sorted through an index, the entries being wide).
static void merge_entries(std::vector<LedgerEntry> *entries) {
    std::vector<std::pair<LedgerKey, uint32_t>> order(entries->size());
    for (size_t r = 0; r < entries->size(); ++r) order[r] = {(*entries)[r].first, static_cast<uint32_t>(r)};
    std::sort(order.begin(), order.end());
    std::vector<LedgerEntry> merged;
    merged.reserve(order.size());
    for (const auto &o : order) {
        const LedgerEntry &e = (*entries)[o.second];
        if (merged.empty() || merged.back().first != e.first) {
            merged.push_back(e);
            continue;
        }
        for (size_t s = 0; s < featurize::L_SLOTS; ++s) merged.back().second[s] += e.second[s];
    }
    entries->swap(merged);
}

// The ledger's tokens and the k_pool_card rows it leaves as tokens (folds.h, featurizer.md §3.6).
static void fold_ledger(const featurize::FeatureSpec &spec, const featurize::CardTables &tables,
    const rmstate::DecodedFrame &state, bool want_sources, featurize::FoldedRows *out) {
    const featurize::LedgerPlan &l = spec.ledger;
    // One contribution per folded row (the reference is used before the next is added), merged below.
    std::vector<LedgerEntry> led;
    led.reserve(256);
    const auto at = [&led](uint64_t rel, uint64_t card) -> LedgerVals & {
        led.emplace_back(LedgerKey(rel, card), LedgerVals{});
        return led.back().second;
    };
    if (const rmstate::DecodedTable *t = table_of(state, l.own_deck))
        for (const Row &row : t->rows) {
            LedgerVals &v = at(cell(row, l.deck_rel), cell(row, l.deck_card));
            v[featurize::L_DECK] = 1;
            v[featurize::L_MAIN] += static_cast<uint32_t>(cell(row, l.deck_main));
            v[featurize::L_SIDE] += static_cast<uint32_t>(cell(row, l.deck_side));
        }
    if (const rmstate::DecodedTable *t = table_of(state, l.k_seen))
        for (const Row &row : t->rows) {
            LedgerVals &v = at(cell(row, l.seen_rel), cell(row, l.seen_card));
            v[featurize::L_SEEN_GAME] += static_cast<uint32_t>(cell(row, l.seen_game));
            v[featurize::L_SEEN_MATCH] += static_cast<uint32_t>(cell(row, l.seen_match));
        }
    if (const rmstate::DecodedTable *t = table_of(state, l.k_known_hand))
        for (const Row &row : t->rows)
            at(l.opponent_rel, cell(row, l.known_card))[featurize::L_KNOWN_HAND] +=
                static_cast<uint32_t>(cell(row, l.known_copies));
    std::vector<Pool> pools = ledger_pools(l, state);
    const std::vector<Exact> exact = exact_positions(l, state);
    mark_runs(&pools, exact);
    for (Pool &p : pools)
        if (p.library) p.folded = p.folded && p.run;
    for (const Exact &e : exact) {
        LedgerVals &v = at(e.rel, e.card);
        ++v[featurize::L_LIBRARY_MIN];
        ++v[featurize::L_LIBRARY_MAX];
    }
    // k_pool_card: the folded rows' bounds into the ledger, the others kept as FILTERED tokens.
    const int32_t pc_type = l.k_pool_card >= 0 ? spec.type_of_table[static_cast<uint16_t>(l.k_pool_card)] : -1;
    const rmstate::DecodedTable *pc = table_of(state, l.k_pool_card);
    if (pc && pc_type >= 0) {
        const size_t ty = static_cast<size_t>(pc_type);
        out->remap[ty].assign(pc->rows.size(), -1);
        for (size_t r = 0; r < pc->rows.size(); ++r) {
            const Row &row = pc->rows[r];
            const uint64_t p = cell(row, l.pc_pool);
            const Pool *pool = p < pools.size() ? &pools[static_cast<size_t>(p)] : nullptr;
            if (pool) {
                LedgerVals &v = at(pool->rel, cell(row, l.pc_card));
                const uint32_t lo = static_cast<uint32_t>(cell(row, l.pc_min)),
                               hi = static_cast<uint32_t>(cell(row, l.pc_max));
                if (pool->kind == l.kind_hand) {
                    v[featurize::L_HAND_MIN] += lo;
                    v[featurize::L_HAND_MAX] += hi;
                } else if (pool->kind == l.kind_sideboard) {
                    v[featurize::L_SIDEBOARD_MIN] += lo;
                    v[featurize::L_SIDEBOARD_MAX] += hi;
                } else if (pool->run) {
                    v[featurize::L_LIBRARY_MIN] += lo;
                    v[featurize::L_LIBRARY_MAX] += hi;
                }
                if (pool->folded) continue;
            }
            out->remap[ty][r] = static_cast<int64_t>(out->rows[ty].size());
            out->rows[ty].push_back(&row);
            if (want_sources) out->sources[ty].push_back({pc_type, l.k_pool_card, static_cast<int64_t>(r), 0});
        }
        out->count[ty] = out->rows[ty].size();
    }
    const size_t lt = static_cast<size_t>(l.type);
    const size_t width = spec.types[lt].numeric.size();
    merge_entries(&led);
    out->num[lt].assign(led.size() * width, 0.0f);
    size_t k = 0;
    for (const auto &e : led) {
        float *num = &out->num[lt][k * width];
        num[featurize::L_DECK] = e.second[featurize::L_DECK] ? 1.0f : 0.0f;
        for (size_t s = featurize::L_DECK + 1; s < featurize::L_SLOTS && s < width; ++s)
            num[s] = std::log1p(static_cast<float>(e.second[s]));
        out->rel[lt].push_back(e.first.first);
        const uint64_t card = e.first.second;
        out->card[lt].push_back(card + 1 < tables.card_rows ? static_cast<int64_t>(card) : -1);
        if (want_sources) out->sources[lt].push_back({l.type, -1, static_cast<int64_t>(k), 0});
        ++k;
    }
    out->count[lt] = led.size();
}

// A delta row's values against a printed row: text columns by their text, the others by bits.
static int value_cmp(const Row &a, const featurize::TokenType &t, const Row &printed) {
    for (size_t i = 0; i < t.value_cols.size(); ++i) {
        const rmstate::Value &x = a[t.value_cols[i]], &y = printed[i];
        if (t.value_text[i]) {
            const int c = x.text.compare(y.text);
            if (c) return c < 0 ? -1 : 1;
        } else if (x.bits != y.bits) {
            return x.bits < y.bits ? -1 : 1;
        }
    }
    return 0;
}

// Two rows of a delta type's table by the same order.
static int rows_cmp(const Row &a, const Row &b, const featurize::TokenType &t) {
    for (size_t i = 0; i < t.value_cols.size(); ++i) {
        const rmstate::Value &x = a[t.value_cols[i]], &y = b[t.value_cols[i]];
        if (t.value_text[i]) {
            const int c = x.text.compare(y.text);
            if (c) return c < 0 ? -1 : 1;
        } else if (x.bits != y.bits) {
            return x.bits < y.bits ? -1 : 1;
        }
    }
    return 0;
}

// The face vocab id an object row's printed rows are read from (its current face), or -1 when the
// viewer cannot see its card or no card row describes the face.
static int64_t printed_face(const featurize::FeatureSpec &spec, const featurize::CardTables &tables, const Row &obj) {
    if (spec.object_card < 0) return -1;
    const rmstate::Value &c = obj[static_cast<size_t>(spec.object_card)];
    if (rmstate::value_is_all_ones(c, spec.object_card_storage) || c.bits + 1 >= tables.card_rows) return -1;
    int64_t face = static_cast<int64_t>(c.bits);
    if (spec.object_transformed >= 0 && obj[static_cast<size_t>(spec.object_transformed)].bits != 0)
        face = tables.back_face[static_cast<size_t>(face)];
    if (face < 0 || static_cast<uint64_t>(face) >= tables.present.size() || !tables.present[static_cast<size_t>(face)])
        return -1;
    return face;
}

// Every object row's handle and printed face, in frame order.
static std::vector<ObjectFace> object_faces(const featurize::FeatureSpec &spec, const featurize::CardTables &tables,
    const std::vector<const rmstate::DecodedTable *> &tabs) {
    std::vector<ObjectFace> out;
    const rmstate::DecodedTable *objects =
        spec.object_type >= 0 && spec.object_handle >= 0 ? tabs[static_cast<size_t>(spec.object_type)] : nullptr;
    if (!objects) return out;
    out.reserve(objects->rows.size());
    for (const Row &obj : objects->rows)
        out.push_back({obj[static_cast<size_t>(spec.object_handle)].bits, printed_face(spec, tables, obj)});
    return out;
}

// A DELTA type's tokens (folds.h); its table may be absent (no rows), a visible object's printed rows
// then all removed.
static void fold_delta(const featurize::FeatureSpec &spec, const featurize::CardTables &tables, size_t type,
    const std::vector<const rmstate::DecodedTable *> &tabs, const std::vector<ObjectFace> &objects, bool want_sources,
    featurize::FoldedRows *out) {
    const featurize::TokenType &t = spec.types[type];
    const rmstate::DecodedTable *child = tabs[type];
    const featurize::PrintedRows *printed = tables.printed_rows(t.printed);
    // The child rows by (handle, value).
    std::vector<const Row *> sorted;
    if (child) {
        sorted.reserve(child->rows.size());
        for (const Row &row : child->rows) sorted.push_back(&row);
    }
    std::stable_sort(sorted.begin(), sorted.end(), [&t](const Row *a, const Row *b) {
        const uint64_t ha = (*a)[t.obj_col].bits, hb = (*b)[t.obj_col].bits;
        return ha != hb ? ha < hb : rows_cmp(*a, *b, t) < 0;
    });
    std::vector<uint8_t> used(sorted.size(), 0);
    const auto range = [&](uint64_t handle) {
        auto lo = std::lower_bound(
            sorted.begin(), sorted.end(), handle, [&t](const Row *r, uint64_t h) { return (*r)[t.obj_col].bits < h; });
        auto hi = lo;
        while (hi != sorted.end() && (**hi)[t.obj_col].bits == handle) ++hi;
        return std::make_pair(static_cast<size_t>(lo - sorted.begin()), static_cast<size_t>(hi - sorted.begin()));
    };
    std::vector<const Row *> &rows = out->rows[type];
    std::vector<uint8_t> &removed = out->removed[type];
    const auto emit = [&](const Row *row, bool gone, int32_t table, int64_t at, int64_t sub) {
        rows.push_back(row);
        removed.push_back(gone ? 1 : 0);
        if (want_sources) out->sources[type].push_back({static_cast<int32_t>(type), table, at, sub});
    };
    const int32_t object_table =
        spec.object_type >= 0 ? static_cast<int32_t>(spec.types[static_cast<size_t>(spec.object_type)].table) : -1;
    for (size_t r = 0; r < objects.size(); ++r) {
        const uint64_t handle = objects[r].handle;
        const std::pair<size_t, size_t> mine = range(handle);
        const int64_t face = printed ? objects[r].face : -1;
        size_t p = face >= 0 ? printed->offsets[static_cast<size_t>(face)] : 0;
        const size_t p_end = face >= 0 ? printed->offsets[static_cast<size_t>(face) + 1] : 0;
        std::vector<size_t> lacking;  // printed rows the object lacks
        int64_t sub = 0;
        for (size_t i = mine.first; i < mine.second; ++i) {
            used[i] = 1;
            int c = 1;
            while (p < p_end && (c = value_cmp(*sorted[i], t, printed->rows[p])) > 0) lacking.push_back(p++);
            if (p < p_end && c == 0) {
                ++p;
                continue;
            }
            emit(sorted[i], false, object_table, static_cast<int64_t>(r), sub++);
        }
        while (p < p_end) lacking.push_back(p++);
        for (const size_t q : lacking) {
            Row made(t.columns);
            made[t.obj_col].bits = handle;
            for (size_t i = 0; i < t.value_cols.size(); ++i) made[t.value_cols[i]] = printed->rows[q][i];
            out->owned.push_back(std::move(made));
            emit(&out->owned.back(), true, object_table, static_cast<int64_t>(r), sub++);
        }
    }
    for (size_t i = 0; i < sorted.size(); ++i)
        if (!used[i]) emit(sorted[i], false, -1, static_cast<int64_t>(rows.size()), 0);
    out->count[type] = rows.size();
}

namespace featurize {

void fold_rows(const FeatureSpec &spec, const CardTables &tables, const rmstate::DecodedFrame &state,
    const std::vector<const rmstate::DecodedTable *> &tabs, bool want_sources, FoldedRows *out) {
    const size_t n = spec.types.size();
    out->count.assign(n, 0);
    out->rows.assign(n, {});
    out->removed.assign(n, {});
    out->remap.assign(n, {});
    out->num.assign(n, {});
    out->rel.assign(n, {});
    out->card.assign(n, {});
    out->sources.assign(n, {});
    out->owned.clear();
    std::vector<ObjectFace> objects;
    bool faces_built = false;
    for (size_t i = 0; i < n; ++i) {
        const TokenType &t = spec.types[i];
        const bool whole = t.kind == TypeKind::TABLE || (t.kind == TypeKind::FILTERED && spec.ledger.type < 0);
        if (whole && tabs[i]) {
            out->rows[i].reserve(tabs[i]->rows.size());
            for (size_t r = 0; r < tabs[i]->rows.size(); ++r) {
                out->rows[i].push_back(&tabs[i]->rows[r]);
                if (want_sources)
                    out->sources[i].push_back(
                        {static_cast<int32_t>(i), static_cast<int32_t>(t.table), static_cast<int64_t>(r), 0});
            }
            out->count[i] = out->rows[i].size();
        } else if (t.kind == TypeKind::DELTA) {
            if (!faces_built) objects = object_faces(spec, tables, tabs);
            faces_built = true;
            fold_delta(spec, tables, i, tabs, objects, want_sources, out);
        }
    }
    if (spec.ledger.type >= 0) fold_ledger(spec, tables, state, want_sources, out);
}

}  // namespace featurize
