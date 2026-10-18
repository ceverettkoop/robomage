#ifndef ROBOMAGE_CHECK_KNOWLEDGE_META_H
#define ROBOMAGE_CHECK_KNOWLEDGE_META_H

#include <string>

#include "gate.h"

// ── robomage_check: the knowledge tier's reference-metagame parts ─────────────────────────────
//
// Archetype inference (docs/knowledge_store.md §10) and the belief calibration report (§10.6), on
// --machine-format rmstate sessions answered with random replies (frames.h). The reference lists'
// tags come from bin/resources/decks/archetypes.json (its `_archetypes` labels and deck keys).

namespace rmcheck {

// The archetype sessions (ARCHETYPE_GAMES): the METAGAME frame against archetypes.json, and per
// seat frame the k_template / k_archetype facts against the true lists and the cards seen.
void archetype_part(const Options &opt, Report &rep, const std::string &dir);
// The calibration sessions (every tagged deck against the one five places on, bo1, under
// --closed-decklist, both seats at expert+belief): log-loss, Brier, reliability, first-vs-last
// frame; fails when the list or archetype log-loss is not below the uniform prior's, or the true
// archetype's mean probability does not rise from a game's first frame to its last.
void calibration_part(const Options &opt, Report &rep, const std::string &dir);

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_KNOWLEDGE_META_H */
