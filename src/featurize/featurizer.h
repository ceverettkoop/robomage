#ifndef FEATURIZE_FEATURIZER_H
#define FEATURIZE_FEATURIZER_H

#include <cstdint>
#include <string>
#include <vector>

#include "../state_format/decode.h"
#include "card_tables.h"
#include "feature_spec.h"
#include "folds.h"
#include "history.h"
#include "rmtb.h"

// ── The featurizer: one decoded STATE frame → the per-decision tensors (docs/featurizer.md §3) ─
//
// Standard library and the decoder only: a pure function of one viewer's decoded STATE frame
// (knowledge tables included), the spec and the card tables, reading no engine state and no frame
// bytes. Tokens are those the folds decide (folds.h: the rows of a table type's table in frame order,
// a folded type's in its canonical order), the types in spec order; a decision's tensors are
// unpadded:
//
//   tok_type [N] i64, tok_num [N, F] f32, tok_cat [N, C] i64, tok_card [N, K] i64,
//   tok_ability [N, Q] i64, tok_mask [N] (1), rel [R, 3] i64 (src, dst, type), act_tok [A] i64,
//   act_mask [A] (1)
//
// with unused numeric and categorical slots 0 and unused card / ability slots -1. Relations are
// emitted row by row in token order, each row's in column order. act_tok[a] is menu row a's token.
//
// featurize_seat is the actor entry point (what an nn: seat calls): it refuses a frame whose viewer
// is not seat A or B. featurize_viewer also takes the omniscient viewer's frame, for learner-side
// writers only (rmfeat, the training plan's trajectory writer). Both refuse a frame that is not a
// STATE frame of the spec's schema. Under a spec with recent-event history (feature_spec.h) the
// seat's EventHistory (history.h, observed up to this frame) gives the history tokens:
// featurize_seat refuses such a spec without one, featurize_viewer emits none without one (the
// omniscient frame).
//
// A list of decisions is written as an RMTB file of kind `tensors` (tensors_file): per decision its
// viewer, game, decision and sideboarding; per tensor its values over every decision concatenated
// along the first dimension and `<tensor>_offsets` (i64, decisions + 1 entries).

namespace featurize {

struct Tensors {
        uint8_t viewer = 0;  // rmstate::Viewer
        int64_t game = 0, decision = 0;
        bool sideboarding = false;
        std::vector<int64_t> tok_type;
        std::vector<float> tok_num;
        std::vector<int64_t> tok_cat, tok_card, tok_ability;
        std::vector<uint8_t> tok_mask;
        std::vector<int64_t> rel;
        std::vector<int64_t> act_tok;
        std::vector<uint8_t> act_mask;
        size_t n() const { return tok_type.size(); }
        size_t r() const { return rel.size() / 3; }
        size_t a() const { return act_tok.size(); }
};

// The actor entry point: a seat's STATE frame. False with `err` for the omniscient viewer's frame,
// a frame that is not a STATE frame or one of another schema.
bool featurize_seat(const FeatureSpec &spec, const CardTables &tables, const rmstate::DecodedFrame &state, Tensors *out,
    std::string *err);
bool featurize_seat(const FeatureSpec &spec, const CardTables &tables, const rmstate::DecodedFrame &state,
    const EventHistory *history, Tensors *out, std::string *err);

// The learner entry point: any viewer's STATE frame (the omniscient one included).
bool featurize_viewer(const FeatureSpec &spec, const CardTables &tables, const rmstate::DecodedFrame &state,
    Tensors *out, std::string *err);
bool featurize_viewer(const FeatureSpec &spec, const CardTables &tables, const rmstate::DecodedFrame &state,
    const EventHistory *history, Tensors *out, std::string *err);

// featurize_viewer with each token's origin (the gate's row-order check; a history token's origin
// is its rank in its type).
bool featurize_traced(const FeatureSpec &spec, const CardTables &tables, const rmstate::DecodedFrame &state,
    const EventHistory *history, Tensors *out, std::vector<TokenSource> *sources, std::string *err);

// The decisions as one RMTB tensors file, stamped with the spec's and vocab's hashes.
RmtbFile tensors_file(const FeatureSpec &spec, const std::vector<Tensors> &list);

}  // namespace featurize

#endif /* FEATURIZE_FEATURIZER_H */
