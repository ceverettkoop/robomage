#ifndef ROBOMAGE_CHECK_FEATURIZE_LAYOUT_H
#define ROBOMAGE_CHECK_FEATURIZE_LAYOUT_H

#include <string>

#include "../../src/featurize/featurizer.h"
#include "../../src/featurize/setup.h"
#include "gate.h"

// ── robomage_check: the layout registry's append-only property (docs/featurizer.md §6.1) ───────
//
// Part of the featurize tier, over the whole-vocab inputs `f` (the compiled-in registry). Two
// variants of them are built in memory:
//   - appended: the CARDS frame with one card appended at the next free id (the rows of the vocab
//     card with the most abilities copied to it) and the registry with a fake keyword value
//     appended (a hole: the schema has no such value);
//   - renamed: the schema's keyword enum with the fake keyword inserted first (an engine enum
//     reordered and grown), every keyword column of the CARDS frame shifted to match, and the
//     registry with the fake keyword value and its `keyword:` ability key appended.
// featurize_layout_part checks that both keep the layout: the same layout_hash and dims; every
// numeric slot, categorical range, card and ability slot, relation type and table column at the
// same position (the appended fake keyword only names its reserved column); every existing card's,
// ability's and part's rows unchanged (renamed: the card table identical), the appended card's rows
// those of the card it copies at its own ability rows (card id × abilities_per_card + index).
// check_layout_decision checks, per decision of the corpus, that the appended inputs featurize the
// decision to the same tensors and (sampled) that the renamed inputs featurize the decision with
// its keyword columns shifted to the same tensors. Then the capacities: the keyword enum full, a
// card with more abilities than abilities_per_card and a registry lacking the fake keyword are each
// refused with a message naming the cause.

namespace rmcheck {

struct LayoutVariants {
        featurize::Featurization appended, renamed;
        uint64_t new_card = 0, source_card = 0;
};

// The two variants of `f`; false with `err`.
bool build_layout_variants(const featurize::Featurization &f, LayoutVariants *out, std::string *err);

// The variants' layout against `f` and the capacities' refusals.
void featurize_layout_part(Report &rep, const featurize::Featurization &f, const LayoutVariants &v);

// "" when `t` (state featurized by `f`; `seat`: through the actor entry point) is what the appended
// inputs featurize and, with `renamed`, what the renamed inputs featurize from the state shifted.
std::string check_layout_decision(
    const LayoutVariants &v, const rmstate::DecodedFrame &state, bool seat, bool renamed, const featurize::Tensors &t);

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_FEATURIZE_LAYOUT_H */
