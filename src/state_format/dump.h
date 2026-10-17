#ifndef STATE_FORMAT_DUMP_H
#define STATE_FORMAT_DUMP_H

#include <vector>

#include "../classes/action.h"

// ── --rmstate-dump FILE (docs/step1_handoff.md T5) ────────────────────────────────────────────
//
// Writes the state-format frames of every viewer (A, B, omniscient) to FILE as
// "RMSTATE <nbytes>\n<payload>" records while the normal protocol runs unchanged: for each viewer,
// at its first decision of the match its SCHEMA, CARDS and DECKLISTS frames, then a STATE frame
// at every decision. A "%p" in FILE is replaced by the engine's process id. Encoding only reads the game (it issues
// handles, which play never reads), so play is the same with or without the dump.

// Called as each decision is read (InputLogger::choose), with the menu as offered. No-op unless
// --rmstate-dump is given.
void rmstate_dump_decision(const std::vector<LegalAction> &actions);

#endif /* STATE_FORMAT_DUMP_H */
