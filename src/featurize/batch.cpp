#include "batch.h"

#include <algorithm>

static size_t padded_size(size_t n, size_t multiple);
template <typename T>
static void place(std::vector<T> *dst, size_t at, const std::vector<T> &src);

// n rounded up to a multiple of `multiple`, and at least `multiple`.
static size_t padded_size(size_t n, size_t multiple) {
    const size_t m = multiple ? multiple : 1;
    return std::max(m, (n + m - 1) / m * m);
}

// src's values at dst[at …].
template <typename T>
static void place(std::vector<T> *dst, size_t at, const std::vector<T> &src) {
    std::copy(src.begin(), src.end(), dst->begin() + static_cast<std::ptrdiff_t>(at));
}

namespace featurize {

void pad_batch(const FeatureSpec &spec, const std::vector<const Tensors *> &batch, size_t multiple, PaddedBatch *out) {
    size_t n = 0, r = 0, a = 0;
    for (const Tensors *t : batch) {
        n = std::max(n, t->n());
        r = std::max(r, t->r());
        a = std::max(a, t->a());
    }
    PaddedBatch &p = *out;
    p.B = batch.size();
    p.N = padded_size(n, multiple);
    p.R = padded_size(r, multiple);
    p.A = padded_size(a, multiple);
    p.F = spec.F;
    p.C = spec.C;
    p.K = spec.K;
    p.Q = spec.Q;
    p.tok_type.assign(p.B * p.N, 0);
    p.tok_num.assign(p.B * p.N * p.F, 0.0f);
    p.tok_cat.assign(p.B * p.N * p.C, 0);
    p.tok_card.assign(p.B * p.N * p.K, -1);
    p.tok_ability.assign(p.B * p.N * p.Q, -1);
    p.tok_mask.assign(p.B * p.N, 0);
    p.rel.assign(p.B * p.R * 3, 0);
    p.rel_mask.assign(p.B * p.R, 0);
    p.act_tok.assign(p.B * p.A, 0);
    p.act_mask.assign(p.B * p.A, 0);
    p.menu_rows.assign(p.B, 0);
    for (size_t b = 0; b < p.B; ++b) {
        const Tensors &t = *batch[b];
        place(&p.tok_type, b * p.N, t.tok_type);
        place(&p.tok_num, b * p.N * p.F, t.tok_num);
        place(&p.tok_cat, b * p.N * p.C, t.tok_cat);
        place(&p.tok_card, b * p.N * p.K, t.tok_card);
        place(&p.tok_ability, b * p.N * p.Q, t.tok_ability);
        place(&p.tok_mask, b * p.N, t.tok_mask);
        place(&p.rel, b * p.R * 3, t.rel);
        std::fill_n(p.rel_mask.begin() + static_cast<std::ptrdiff_t>(b * p.R), t.r(), uint8_t{1});
        place(&p.act_tok, b * p.A, t.act_tok);
        place(&p.act_mask, b * p.A, t.act_mask);
        p.menu_rows[b] = t.a();
    }
}

}  // namespace featurize
