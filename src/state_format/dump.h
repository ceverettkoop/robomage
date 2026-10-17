#ifndef STATE_FORMAT_DUMP_H
#define STATE_FORMAT_DUMP_H

#include <vector>

#include "../classes/action.h"

// ── --rmstate-dump FILE (docs/step1_handoff.md T5) ────────────────────────────────────────────
//
// Writes the state-format frames of every viewer (A, B, omniscient) to FILE as
// "RMSTATE <nbytes>\n<payload>" records while the normal protocol runs unchanged: for each viewer,
// at its first decision of the match its SCHEMA, CARDS, DECKLISTS and METAGAME frames, then a STATE frame
// at every decision, and once the match is over a terminal STATE frame (header.terminal = 1, no
// menu) with its last events. A seat's STATE frames end with the k_* tables of the dump's own
// knowledge instance for that seat (state_format/knowledge_feed.h). A "%p" in FILE is replaced by the engine's process id. Encoding only
// reads the game (it issues handles, which play never reads), so play is the same with or without
// the dump.
//
// --state-dump json: every record is also decoded (state_format/decode.h) and rendered as one
// JSON line (state_format/json.h) into FILE.jsonl, in the same order.

// Called as each decision is read (InputLogger::choose), after rmstate_begin_decision
// (state_format/stream.h), with the menu as offered. No-op unless --rmstate-dump is given.
void rmstate_dump_decision(const std::vector<LegalAction> &actions);

// The match is over: every viewer's terminal frame. No-op unless --rmstate-dump is given.
void rmstate_dump_match_over();

#endif /* STATE_FORMAT_DUMP_H */
