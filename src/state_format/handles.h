#ifndef STATE_FORMAT_HANDLES_H
#define STATE_FORMAT_HANDLES_H

#include <cstdint>
#include <vector>

#include "../components/zone.h"
#include "../ecs/entity.h"

// ── Per-viewer handle tables (docs/state_format.md §6) ─────────────────────────────────────────
//
// A handle is a per-viewer opaque id (1, 2, 3, … per game and viewer; 0 = none) with no relation
// to entity ids. A viewer is a seat (Zone::PLAYER_A / PLAYER_B); any other value is the omniscient
// viewer, which has a table of its own. The tables live in Game::handle_tables, so a snapshot
// covers them and every game starts with fresh ones.
//
// Rules:
//   - A viewer holds a handle for an object only while they can individually identify it
//     (individually_identifiable, src/queries/visibility.h): a public-zone object, a face-down
//     one included; their own hand and sideboard; a card in one of their reveal windows.
//   - A handle names one object (CR 400.7). A zone change ends the old object's handles
//     (handles_begin_move); for each viewer who can identify the object it became, the move
//     issues the new object's handle at once and reports the old → new link (handles_end_move).
//   - Becoming unidentifiable retires the handle where it happens: at a zone change, as the
//     reveal window that showed it closes (visibility.cpp's close hook), or as the object ceases
//     to exist. A lookup or a canonical issue also retires any handle whose object is gone, as a
//     backstop. A retired handle is never issued again; an object identifiable again gets a fresh
//     one.
//   - Handles are issued lazily, the first time one is needed: in move order by the zone-change
//     links (the order the move events report), in the caller's order by handle_of (the event
//     constructors, in event order), and in canonical_object_order by issue_handles_canonical
//     (the snapshot encoder, at frame start). None of these orders depends on entity ids or on
//     anything hidden from the viewer.
//
// Nothing here changes the game: with no handle ever asked for, every table stays empty and the
// zone-change and window-close hooks do nothing.

// `viewer`'s handle for `e`, issued now if the viewer can identify `e` and holds none; 0 when the
// viewer cannot identify it (retiring a handle they still held for it).
uint32_t handle_of(Zone::Ownership viewer, Entity e);
// `viewer`'s live handle for `e`, or 0. Never issues or retires.
uint32_t handle_peek(Zone::Ownership viewer, Entity e);
// Retires every handle of `viewer` whose object is gone or no longer identifiable, then issues a
// handle to every object `viewer` can identify that has none, in canonical_object_order. The
// snapshot encoder calls this at the start of each frame; every identifiable object then has a
// handle.
void issue_handles_canonical(Zone::Ownership viewer);

// One live handle.
struct HandleRow {
        uint32_t handle = 0;
        Entity entity = 0;
};
// `viewer`'s live handles, in ascending handle order.
std::vector<HandleRow> handle_table(Zone::Ownership viewer);

// ── Zone changes ───────────────────────────────────────────────────────────────────────────────

// One object's handles across a zone change, per viewer (A, B, omniscient): `before` the handle
// of the object it was (0 = the viewer could not identify it), `after` the handle of the object it
// became (0 = they cannot identify it, or held no handle before). before != 0 && after == 0 is a
// move that hides the object from the viewer; both nonzero is the old → new link
// (`h5 → graveyard as h9`).
struct HandleMove {
        // Indexed like Game::handle_tables: [Zone::UNKNOWN] omniscient, [PLAYER_A], [PLAYER_B].
        uint32_t before_handle[3] = {0, 0, 0};
        uint32_t after_handle[3] = {0, 0, 0};
        uint32_t before(Zone::Ownership viewer) const { return before_handle[slot(viewer)]; }
        uint32_t after(Zone::Ownership viewer) const { return after_handle[slot(viewer)]; }
        static int slot(Zone::Ownership viewer) {
            return viewer == Zone::PLAYER_A || viewer == Zone::PLAYER_B ? static_cast<int>(viewer) : 0;
        }
};
// Called as `e` starts to leave its zone, before its reveal windows close and its identity stamp
// changes: records and retires each viewer's handle for it.
HandleMove handles_begin_move(Entity e);
// Called once `e` is in its new zone (stamped, face-down state and windows set): issues the new
// object's handle for each viewer that held one before and can identify it now.
void handles_end_move(HandleMove &move, Entity e);
// `e` ceases to exist (an ability or a spell copy leaving the stack): its handles retire.
void retire_object_handles(Entity e);
// Reveal windows on `cards` just closed: each viewer's handle for a card they can no longer
// identify retires (in handle order).
void retire_unidentifiable_handles(const std::vector<Entity> &cards);

// --trace-handles: issue canonically for every viewer and print each table as one HANDLES: line
// (game log). Called at each decision.
void trace_handle_tables();

#endif /* STATE_FORMAT_HANDLES_H */
