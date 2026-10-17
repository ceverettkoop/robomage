#ifndef QUERIES_VISIBILITY_H
#define QUERIES_VISIBILITY_H

#include <vector>

#include "../classes/game.h"
#include "../components/zone.h"
#include "../ecs/entity.h"

// ── What a viewer can see right now (CR 400.2, 401.2-3, 402.3, 406.3, 708.5) ──────────────────
//
// A viewer is a seat (Zone::PLAYER_A / PLAYER_B); any other value is the omniscient viewer, who
// sees everything. These answer what is explicit to the viewer at this instant, never what they
// remember: the engine's older knowledge bookkeeping (Game::KnownLibraryTop, Zone::identity_known,
// the match revealed bits) is separate and unaffected.

// Whether `viewer` can see `entity`'s identity: everything in a public zone except a face-down
// object (CR 400.2); their own hand and sideboard (CR 402.3); a face-down permanent or spell they
// control (CR 708.5); and any card in one of their reveal windows. Objects with no Zone (players)
// are always visible.
bool identity_visible(Entity entity, Zone::Ownership viewer);

// Whether `viewer` can see the order of the cards `owner` has in `zone`: the stack, graveyards and
// exile keep a public order; a library's order is hidden from everyone (CR 401.2), and hands, the
// battlefield and the sideboard have none. A library card in one of the viewer's reveal windows
// shows its own position (it is read from its Zone).
bool order_visible(Zone::ZoneValue zone, Zone::Ownership owner, Zone::Ownership viewer);

// Whether `viewer` can tell `entity` apart as one individual object right now, so it may hold a
// handle for it (docs/state_format.md §6): every object in a public zone, a face-down one included
// (CR 400.2, 708.6: face-down objects must stay distinguishable even with their identity
// withheld), plus every object whose identity the viewer can see (identity_visible: their own hand
// and sideboard, a reveal window). The omniscient viewer (any non-seat value) identifies every
// zoned object. Objects with no Zone (players) are never handle-bearing objects.
bool individually_identifiable(Entity entity, Zone::Ownership viewer);

// `objects` in an order `viewer` can derive from what they see right now, never from entity ids
// or hidden information (docs/state_format.md §6): zone (battlefield, stack, graveyards, exile,
// hands, libraries, sideboards); then the seat it belongs to, the viewer's own first (controller
// on the battlefield, owner elsewhere; none on the shared stack); then its position where the
// zone's order is visible (stack, graveyard and exile from the top, a library card in a window by
// its position); then its name when the viewer can see its identity (a face-down object sorts
// before named ones); then its battlefield timestamp; then its arrival order where the viewer saw
// it arrive (a public zone or their own hand: the zone-entry stamp). Objects equal on all of
// these are identical copies of one card in the same unordered place (two Islands in a revealed
// hand, two copies in a sideboard): they are tied by entity id. That tells the viewer nothing:
// swapping two such copies changes nothing the viewer sees, and the id order among copies of one
// card is fixed when the deck is built and unrelated to where the uniform shuffle put each copy,
// so to anything that happened to them since (their arrival order in a hidden hand would not be:
// it would say which copy was drawn when).
std::vector<Entity> canonical_object_order(std::vector<Entity> objects, Zone::Ownership viewer);

// One-word names of a zone and a seat for the trace lines (--trace-visibility, --trace-handles).
const char *trace_zone_word(Zone::ZoneValue zone);
const char *trace_seat_word(Zone::Ownership seat);

// ── Reveal windows (Game::reveal_windows) ─────────────────────────────────────────────────────

// One window as one viewer has it: the card, why the viewer sees it, until when, and who it is
// shown to (Zone::UNKNOWN = all players).
struct VisibleWindow {
        Entity card = 0;
        Game::RevealWindow::Kind kind = Game::RevealWindow::REVEAL;
        Game::RevealWindow::End until = Game::RevealWindow::RESOLUTION;
        Zone::Ownership audience = Zone::UNKNOWN;
};

// `viewer`'s reveal-window set: every open window shown to them, in the order the windows opened.
// A card may appear twice (a look and a reveal of it at once).
std::vector<VisibleWindow> reveal_windows_for(Zone::Ownership viewer);

// Whether one of `viewer`'s reveal windows shows `card` as the object it is now.
bool in_reveal_window(Entity card, Zone::Ownership viewer);

// Opening. Each opens a window per card that is in a hidden zone (a face-down exiled card
// included); a look a viewer needs no window for (their own hand or sideboard) opens none, and
// opening a window that is already open changes nothing, so a resumed effect can repeat its
// calls. Several cards open in zone and position order, so the order says nothing about entity ids.
//
// A reveal to all players while the current resolution needs it (CR 701.20a).
void reveal_card(Entity card);
void reveal_cards(const std::vector<Entity> &cards);
// A look by `looker` alone while the current resolution needs it (CR 701.20e).
void look_at_card(Entity card, Zone::Ownership looker);
void look_at_cards(const std::vector<Entity> &cards, Zone::Ownership looker);
// A reveal that lasts until `stack_object` leaves the stack (CR 701.20a, 702.94b: a miracle card
// stays revealed until its trigger resolves or otherwise leaves the stack).
void reveal_until_leaves_stack(Entity card, Entity stack_object);
// A reveal that lasts until the card leaves its zone (the chosen companion, revealed outside the
// game before it begins, CR 702.139a / 103.2b).
void reveal_continuously(Entity card);

// Closing. A window that ends because its card moved needs no other notice than the move.
//
// Who sees `card` in a hidden zone right now (a seat whose own hand it is in, or who has a window
// on it). Called before `card` moves, for the face-down exile rule below.
struct HiddenViewers {
        bool a = false;
        bool b = false;
};
HiddenViewers hidden_identity_viewers(Entity card);
// `card` is leaving its zone: every window on it ends (CR 400.7, 701.20d). Called before the move
// restamps its identity.
void close_windows_on_zone_change(Entity card);
// `card` was just exiled face down: each player in `saw` (who looked at it before) may keep
// looking at it until it leaves exile (CR 406.3).
void open_face_down_windows(Entity card, HiddenViewers saw);
// `card` was turned face up where it is: it is public, so its face-down look windows end.
void close_windows_on_face_up(Entity card);
// `owner`'s library was shuffled: windows on its cards end (CR 701.20d).
void close_windows_on_shuffle(Zone::Ownership owner);
// An effect finished: windows lasting for the resolution end, and so do windows tied to the
// resolving stack object `resolving` (0 for an effect resolved off the stack) or to a stack
// object that has since left the stack.
void close_resolution_windows(Entity resolving);

#endif /* QUERIES_VISIBILITY_H */
