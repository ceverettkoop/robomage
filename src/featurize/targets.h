#ifndef FEATURIZE_TARGETS_H
#define FEATURIZE_TARGETS_H

#include <cstdint>
#include <string>
#include <vector>

#include "../state_format/decode.h"
#include "feature_spec.h"
#include "rmtb.h"

// ── Learner targets (docs/featurizer.md §3.5), standard library and the decoder only ──────────
//
// Per decision of a seat, from the omniscient viewer's frames beside the seat's (learner-side data,
// never an actor's input):
//
//   opp_hand       the opponent's hand: the omniscient frame's object rows in the opponent's hand,
//                  their card ids ascending (a multiset)
//   own_library,   the seat's and the opponent's library from the top, as card ids, at most the
//   opp_library    spec's library_depth (the omniscient frame's object rows in a library, by `pos`)
//
// Written as an RMTB file of kind `targets`: per decision its seat, game, decision and sideboarding;
// the three lists as values plus `<name>_offsets`.

namespace featurize {

struct Targets {
        uint8_t seat = 0;  // rmstate::Viewer A or B
        int64_t game = 0, decision = 0;
        bool sideboarding = false;
        std::vector<int64_t> opp_hand;
        std::vector<int64_t> own_library, opp_library;
};

// The targets of the decision `seat_state` (a seat's STATE frame) from the omniscient frame of the same
// decision. False with `err` when the seat frame is not a seat's, the omniscient frame is not the
// omniscient viewer's or names another decision.
bool decision_targets(const FeatureSpec &spec, const rmstate::Schema &schema, const rmstate::DecodedFrame &seat_state,
    const rmstate::DecodedFrame &omniscient_state, Targets *out, std::string *err);

// The decisions' targets as one RMTB targets file.
RmtbFile targets_file(const FeatureSpec &spec, const std::vector<Targets> &list);

}  // namespace featurize

#endif /* FEATURIZE_TARGETS_H */
