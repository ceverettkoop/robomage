#ifndef STATE_FORMAT_SELFCHECK_FEATURIZE_H
#define STATE_FORMAT_SELFCHECK_FEATURIZE_H

#include <string>
#include <vector>

#include "../components/zone.h"

// ── --selfcheck-featurize: the featurizer under the state self-check (docs/featurizer.md §3.5) ─
//
// Implies --selfcheck-state. Each seat's featurizer inputs are built once per match from its own
// SCHEMA, CARDS and METAGAME frames (card tables of the match's cards, N_CARD_TYPES + 1 card rows;
// featurize/card_tables.h, feature_spec.h). Then:
//
//   scramble     at every decision the seat's frame, k_* tables included, featurized by the actor
//                entry point (featurize_seat) must give the same tensors as the frame re-encoded
//                with every hidden identity rotated (the scramble check's frame, with the same k_*
//                tables);
//   permutation  at the sampled decisions likewise for the frame encoded in the world with
//                everything hidden from the seat physically permuted;
//   isolation    the omniscient frame is refused by featurize_seat and featurized by
//                featurize_viewer.
//
// A failure is an ERROR: line of check `featurize` (selfcheck_state_mismatch); at match end one
// summary line on stderr: SELFCHECK-FEATURIZE: decisions=<n> compared=<n> refused=<n> errors=<n>.

extern bool selfcheck_featurize;  // --selfcheck-featurize

// The seat's match frames (SCHEMA, CARDS, DECKLISTS, METAGAME), once per match; ignored for the
// omniscient viewer.
void selfcheck_featurize_match_frames(Zone::Ownership viewer, const std::vector<std::string> &payloads);

// `frame` (a seat's STATE frame with its k_* tables) against `other` (the same decision's frame as
// a `check` re-encoded it, without k_* tables): equal tensors.
void selfcheck_featurize_compare(
    const char *check, Zone::Ownership viewer, const std::string &frame, const std::string &other);

// The omniscient viewer's STATE frame: refused by the actor entry point.
void selfcheck_featurize_omniscient(const std::string &frame);

// The summary line; the inputs are dropped for the next match.
void selfcheck_featurize_match_over();

#endif /* STATE_FORMAT_SELFCHECK_FEATURIZE_H */
