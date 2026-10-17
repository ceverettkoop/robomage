#ifndef STATE_FORMAT_ENCODE_H
#define STATE_FORMAT_ENCODE_H

#include <cstdint>
#include <string>
#include <vector>

#include "../classes/action.h"
#include "../components/zone.h"
#include "events.h"

// ── Engine → RMSTATE frames (docs/state_format.md §4, §9, Appendix A) ─────────────────────────
//
// A viewer is a seat (Zone::PLAYER_A / PLAYER_B) or, for any other value, the omniscient viewer.
// Every frame is a self-contained payload (frame header + table blocks, schema.h); the caller
// adds the stream framing (rmstate::frame_record).
//
// The encoder only reads the game, through the shared queries, with one exception it shares with
// the handle tables: building a STATE frame issues the viewer's handles (issue_handles_canonical),
// which changes Game::handle_tables and nothing that play reads. Identity reaches a frame only
// through visible_card_id / card_identity (queries/visibility.h); an object's characteristics are
// read only when the viewer can see its identity.

// The schema hash every frame header carries (computed once from the SCHEMA frame).
uint64_t rmstate_schema_hash();

// The SCHEMA frame: tables, columns and enum string tables.
std::string encode_schema_frame(Zone::Ownership viewer);

// The CARDS frame: every card in the registered lists the viewer may see (both, or under
// --closed-decklist a seat's own only), each face a row, plus every token those cards can make.
std::string encode_cards_frame(Zone::Ownership viewer);

// The DECKLISTS frame: the registered lists with their deck names (a seat's own only under
// --closed-decklist).
std::string encode_decklists_frame(Zone::Ownership viewer);

// The STATE frame for one decision: the snapshot tables, the menu `actions` when the viewer is
// the one deciding (`decider`) or omniscient, and `events`, the viewer's observation events since
// their previous frame (take_obs_events, state_format/events.h), already projected for them. A
// menu description that would name a card the viewer cannot see is replaced by a redacted one.
// `terminal` marks the frame written once the match is over (header.terminal): it answers no
// decision, so it is built with no menu and no decider.
std::string encode_state_frame(Zone::Ownership viewer, const std::vector<LegalAction> &actions, Zone::Ownership decider,
    const std::vector<ObsEvent> &events, bool terminal = false);

// A STATE frame holding only `events` as `viewer`'s event tables (no snapshot tables): the rows an
// event projects to, for comparing two builds of one event (--selfcheck-state). Not a frame any
// stream carries.
std::string encode_events_frame(Zone::Ownership viewer, const std::vector<ObsEvent> &events);

#endif /* STATE_FORMAT_ENCODE_H */
