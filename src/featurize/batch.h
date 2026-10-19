#ifndef FEATURIZE_BATCH_H
#define FEATURIZE_BATCH_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "featurizer.h"

// ── The padding batcher: decisions → the network contract's [B, …] arrays (docs/featurizer.md §6)
//
// Standard library only. Stacks a batch of unpadded decisions (featurizer.h) into row-major arrays
// of the contract's shapes, every decision padded to the batch's largest N, R and A rounded up to
// a multiple of `multiple` (at least `multiple`, so no dimension is 0 or 1 whatever the batch):
//
//   tok_type [B,N] i64, tok_num [B,N,F] f32, tok_cat [B,N,C] i64, tok_card [B,N,K] i64,
//   tok_ability [B,N,Q] i64, tok_mask [B,N] u8, rel [B,R,3] i64, rel_mask [B,R] u8,
//   act_tok [B,A] i64, act_mask [B,A] u8
//
// Padding: tok_type, tok_num, tok_cat (the pad id) and act_tok 0, tok_card and tok_ability -1,
// rel (0, 0, 0), every mask 0. The arrays are reused from call to call (no reallocation once the
// largest batch has been seen).

namespace featurize {

struct PaddedBatch {
        size_t B = 0, N = 0, R = 0, A = 0;
        size_t F = 0, C = 0, K = 0, Q = 0;
        std::vector<int64_t> tok_type;
        std::vector<float> tok_num;
        std::vector<int64_t> tok_cat, tok_card, tok_ability;
        std::vector<uint8_t> tok_mask;
        std::vector<int64_t> rel;
        std::vector<uint8_t> rel_mask;
        std::vector<int64_t> act_tok;
        std::vector<uint8_t> act_mask;
        std::vector<size_t> menu_rows;  // per decision its A (its real menu rows)
};

// Pads and stacks `batch` (decisions of `spec`) into `out`.
void pad_batch(const FeatureSpec &spec, const std::vector<const Tensors *> &batch, size_t multiple, PaddedBatch *out);

}  // namespace featurize

#endif /* FEATURIZE_BATCH_H */
