#ifndef ROBOMAGE_CHECK_FEATURIZE_HISTORY_H
#define ROBOMAGE_CHECK_FEATURIZE_HISTORY_H

#include <string>

#include "../../src/featurize/setup.h"
#include "gate.h"

// ── robomage_check: the recent-event history tokens (docs/featurizer.md §3.7) ─────────────────
//
// Part of the featurize tier. `off` is the featurizer built from the cards file `cards` without
// options; the part builds it again with the history on and checks:
//   - the spec: off has no history key and no history type; on has both, the off spec's types
//     first, another hash; rmfeat --spec --history writes the same document;
//   - one corpus game replayed with a dump and --trajectory-out --feature-history: every seat
//     decision of the dump featurized with history twice (each pass its own event histories) gives
//     the same tensors; without the history tokens they are the off tensors (types, slots, cards,
//     abilities, relations and menu), and the history tokens keep the caps (events, cards per
//     event), the turn offset and age slots, and each card token's one relation to its event;
//   - the episodes the engine wrote are those tensors, step by step (the in-process feed batches
//     events otherwise than the dump); rmfeat --featurize --history over the dump writes them too.

namespace rmcheck {

void featurize_history_part(
    const Options &opt, Report &rep, const featurize::Featurization &off, const std::string &cards);

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_FEATURIZE_HISTORY_H */
