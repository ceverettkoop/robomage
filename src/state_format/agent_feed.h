#ifndef STATE_FORMAT_AGENT_FEED_H
#define STATE_FORMAT_AGENT_FEED_H

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../components/zone.h"

// ── The in-process frame feed: RMSTATE frames for the agents in the engine process ───────────
//
// A frame writer (ObsConsumer::AGENT) beside --rmstate-dump and the RMSTATE stream. It serves
// the seats an in-process agent decides (src/agents/, --agent-a / --agent-b, --play, --actions) and,
// under --trace-decisions, both seats. At a decision of a served seat it builds exactly what the
// stdout stream would send that seat (state_format/stream.h): the seat's SCHEMA, CARDS, DECKLISTS
// and METAGAME frames the first time it decides in the match, then its STATE frame with the menu as
// offered and its events, ending with the k_* tables of this writer's own knowledge instance for
// the seat (state_format/knowledge_feed.h). The bytes are the stream's for the same decision; they
// are handed over in memory instead of being written out. Like every writer it changes nothing
// play reads.

// Serves `seat` from now on (a seat, before the first game starts).
void agent_feed_serve(Zone::Ownership seat);
// Before a match (agents_install): no seat is served and no seat has had its once-per-match frames,
// so each seat served in the match gets them again the first time it decides.
void agent_feed_reset();
// Whether any seat is served.
bool agent_feed_enabled();
// Whether `viewer` is a served seat.
bool agent_feed_serves(Zone::Ownership viewer);

// The frames of served seat `seat` for this decision (`actions` as offered): its once-per-match
// frames, the first time it decides in the match, are appended to `match_frames`, and `state`
// receives its STATE frame.
void agent_feed_decision(Zone::Ownership seat, const std::vector<LegalAction> &actions,
    std::vector<std::string> *match_frames, std::string *state);
// The match is over: served seat `seat`'s terminal STATE frame (no menu, its last events; the
// stream's for the seat, state_format/stream.h) in `state`, after its once-per-match frames in
// `match_frames` when it never decided in the match.
void agent_feed_terminal(Zone::Ownership seat, std::vector<std::string> *match_frames, std::string *state);

#endif /* STATE_FORMAT_AGENT_FEED_H */
