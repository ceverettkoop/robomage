#ifndef STATE_FORMAT_KNOWLEDGE_FEED_H
#define STATE_FORMAT_KNOWLEDGE_FEED_H

#include <string>
#include <vector>

#include "../components/zone.h"
#include "events.h"
#include "schema.h"

// ── The knowledge sidecar in the engine (docs/knowledge_store.md §2, §9.6) ────────────────────
//
// Every RMSTATE frame writer (ObsConsumer: --rmstate-dump, --machine-format rmstate,
// --selfcheck-state) has a knowledge::Knowledge instance of its own per seat, fed with exactly the
// frames that writer builds for that seat, in order: the seat's SCHEMA, CARDS and DECKLISTS
// frames, then each of its STATE frames as encoded (its events drained into it), before the k_*
// tables are appended to it. An instance is a pure function of its seat's frame stream from that
// writer, so `rmknow --isolation` run on the frames the writer emitted reproduces the appended
// tables byte for byte, and what one writer's frames carry never depends on whether another
// writer is on (the stream's frames are the same with or without a dump or the self-check).
//
// Level per seat: --knowledge-a / --knowledge-b off|expert (default expert). At level off the
// frame carries the k_meta row alone (level off) and the instance reads nothing but the match
// frames. The level changes no play: only the frames' k_* tables.
//
// Lifetime: the instances live outside Game, for the whole match (match memory persists across
// the per-game reset; each instance resets its networks itself at game_start), and are made at a
// writer's first frame for that seat. When the first is made, a snapshot hook is registered
// (snapshot_set_sidecar_hooks, snapshot.h) that saves every instance's state (Knowledge::serialize)
// with each engine snapshot and restores it with it, so a restored decision rebuilds the same
// frames, k_* tables included. With no RMSTATE writer on, no instance exists, no hook is
// registered and nothing here runs.

// `viewer`'s knowledge level (--knowledge-a / --knowledge-b; OFF for the omniscient viewer, which
// has no sidecar): what its STATE frames' header.knowledge says.
rmstate::KnowledgeLevel knowledge_level_of(Zone::Ownership viewer);

// Feeds one of `viewer`'s once-per-match frames (SCHEMA, CARDS, DECKLISTS), as `consumer` writes
// it, to that writer's instance for the seat. No-op for the omniscient viewer.
void knowledge_feed_match_frame(ObsConsumer consumer, Zone::Ownership viewer, const std::string &payload);

// Feeds `viewer`'s STATE frame `payload`, as `consumer` built it, to that writer's instance for the
// seat and appends the instance's k_* tables to it. No-op for the omniscient viewer. The
// instance's error lines (its per-frame consistency check, an event it cannot apply) go to `errors`
// when given, else they are printed as ERROR: lines.
void knowledge_append_tables(
    ObsConsumer consumer, Zone::Ownership viewer, std::string *payload, std::vector<std::string> *errors = nullptr);

#endif /* STATE_FORMAT_KNOWLEDGE_FEED_H */
