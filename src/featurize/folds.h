#ifndef FEATURIZE_FOLDS_H
#define FEATURIZE_FOLDS_H

#include <cstdint>
#include <deque>
#include <vector>

#include "../state_format/decode.h"
#include "card_tables.h"
#include "feature_spec.h"

// ── The folds: one decoded STATE frame's tokens by type (docs/featurizer.md §3.6) ───────────────
//
// Standard library and the decoder only. fold_rows decides every token of a decision before the
// featurizer fills the tensors: per table-backed type (TABLE, FILTERED, DELTA) the rows that are
// tokens, in token order; for the synthetic type (LEDGER) each token's numeric slots, player_rel
// and card. A pure function of the frame, the spec and the card tables (their printed rows), with
// the token order canonical where the frame's row order is not meaningful:
//
//   TABLE      every row, in frame order
//   FILTERED   k_pool_card: the rows of the pools the ledger does not fold, in frame order; `remap`
//              gives each frame row its token row (-1: folded), for row: relations into the type
//   DELTA      per object row in frame order, the object's rows its current face's printed rows
//              lack (`removed` 0), then the printed rows it lacks (synthesized, `removed` 1), each
//              part sorted by value; an object whose face the viewer cannot see, or whose face has
//              no card row, keeps every row. Rows of a handle no object row has follow, by handle
//   LEDGER     one token per (player_rel, card) of own_deck, k_seen, k_known_hand, k_pool_card and
//              the exact k_position rows, ascending
//
// `sources` names each token's origin for the gate's row-order check.

namespace featurize {

// A token's origin: row `row` of STATE table `table` (and its `sub`-th token, for a delta object's
// rows: `table` is then the object table), or, with `table` -1, the `row`-th token of its type.
struct TokenSource {
        int32_t type = 0;
        int32_t table = -1;
        int64_t row = 0;
        int64_t sub = 0;
};

struct FoldedRows {
        // Per token type.
        std::vector<size_t> count;
        std::vector<std::vector<const std::vector<rmstate::Value> *>> rows;  // TABLE, FILTERED, DELTA
        std::vector<std::vector<uint8_t>> removed;                           // DELTA: per token
        std::vector<std::vector<int64_t>> remap;                             // FILTERED: per frame row
        std::vector<std::vector<float>> num;                                 // LEDGER: [count, numeric slots]
        std::vector<std::vector<uint64_t>> rel;                              // LEDGER: player_rel per token
        std::vector<std::vector<int64_t>> card;                              // LEDGER: the card per token
        std::vector<std::vector<TokenSource>> sources;                       // with `want_sources`
        std::deque<std::vector<rmstate::Value>> owned;                       // the synthesized rows
};

// The frame's tokens by type; `tabs` per token type its table in the frame (nullptr: none or a
// synthetic type).
void fold_rows(const FeatureSpec &spec, const CardTables &tables, const rmstate::DecodedFrame &state,
    const std::vector<const rmstate::DecodedTable *> &tabs, bool want_sources, FoldedRows *out);

}  // namespace featurize

#endif /* FEATURIZE_FOLDS_H */
