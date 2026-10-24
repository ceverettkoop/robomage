#ifndef STATE_FORMAT_KNOWLEDGE_FEED_H
#define STATE_FORMAT_KNOWLEDGE_FEED_H

#include <string>
#include <vector>

#include "../components/zone.h"
#include "decode.h"
#include "events.h"
#include "schema.h"

namespace knowledge {
class Knowledge;
}

// ── The knowledge sidecar in the engine (docs/knowledge_store.md §2, §9.6) ────────────────────
//
// Every RMSTATE frame writer (ObsConsumer: --rmstate-dump, the RMSTATE stream,
// --selfcheck-state, the in-process feed) has a knowledge::Knowledge instance of its own per seat, fed with exactly the
// frames that writer builds for that seat, in order: the seat's SCHEMA, CARDS and DECKLISTS
// frames, then each of its STATE frames as encoded (its events drained into it), before the k_*
// tables are appended to it. An instance is a pure function of its seat's frame stream from that
// writer, so `rmknow --isolation` run on the frames the writer emitted reproduces the appended
// tables byte for byte, and what one writer's frames carry never depends on whether another
// writer is on (the stream's frames are the same with or without a dump or the self-check). The
// model follows the seat's events, not the frames they were drained into, so every writer's
// frame for one seat and decision carries the same k_* tables.
//
// Level per seat: --knowledge-a / --knowledge-b off|expert (default expert). At level off the frame
// carries the k_meta row alone (level off) and the instance reads nothing but the match frames. The
// level changes no play: only the frames' k_* tables.
//
// Lifetime: the instances live outside Game, for the whole match (match memory persists across
// the per-game reset; each instance resets its networks itself at game_start), and are made at a
// writer's first frame for that seat (and again after knowledge_feed_reset). When the first is
// made, a snapshot hook is registered (snapshot_set_sidecar_hooks, snapshot.h) that saves every
// instance's state (Knowledge::serialize) with each engine snapshot and restores it with it, so a
// restored decision rebuilds the same frames, k_* tables included. With no RMSTATE writer on, no
// instance exists, no hook is registered and nothing here runs.

// Before a match (agents_install): every instance is dropped, so each writer's next frame for a
// seat makes a new one at the seat's current level.
void knowledge_feed_reset();

// `viewer`'s knowledge level (--knowledge-a / --knowledge-b; OFF for the omniscient viewer, which
// has no sidecar): what its STATE frames' header.knowledge says.
rmstate::KnowledgeLevel knowledge_level_of(Zone::Ownership viewer);

// Feeds one of `viewer`'s once-per-match frames (SCHEMA, CARDS, DECKLISTS), as `consumer` writes
// it, to that writer's instance for the seat. No-op for the omniscient viewer.
void knowledge_feed_match_frame(ObsConsumer consumer, Zone::Ownership viewer, const std::string &payload);

// Feeds `viewer`'s STATE frame `payload`, as `consumer` built it, to that writer's instance for the
// seat and appends the instance's k_* tables to it. No-op for the omniscient viewer. The
// instance's error lines (its per-frame consistency check, an event it cannot apply) go to `errors`
// when given, else they are printed as ERROR: lines. With `decoded`, the frame the instance decoded
// is handed out, its k_* tables added: what decoding the returned payload gives (true then; false,
// `decoded` untouched, when the instance decoded nothing: level off, a frame that does not decode).
bool knowledge_append_tables(ObsConsumer consumer, Zone::Ownership viewer, std::string *payload,
    std::vector<std::string> *errors = nullptr, rmstate::DecodedFrame *decoded = nullptr);

// `consumer`'s instance for seat `viewer`, or nullptr before its first frame (for checks:
// state_format/selfcheck_knowledge.h).
const knowledge::Knowledge *knowledge_instance(ObsConsumer consumer, Zone::Ownership viewer);

// Every instance's state, as the snapshot hook saves it (per writer and seat: a presence byte, then
// the serialised state with its length).
std::string knowledge_state_blob();

#endif /* STATE_FORMAT_KNOWLEDGE_FEED_H */
