#ifndef STATE_FORMAT_EVENTS_H
#define STATE_FORMAT_EVENTS_H

#include <cstdint>
#include <vector>

#include "../components/zone.h"
#include "../ecs/entity.h"
#include "../object_ref.h"
#include "schema.h"

// ── Observation events (docs/state_format.md §5) ──────────────────────────────────────────────
//
// Every piece of information that is explicit to a viewer only for a moment (a card moving into or
// out of a hidden zone, a reveal, a look, a shuffle) becomes an event in that viewer's queue, as
// that viewer saw it at that instant. A viewer is a seat (Zone::PLAYER_A / PLAYER_B) or, for any
// other value, the omniscient viewer, who has a queue of its own. The next STATE frame built for a
// viewer drains its queue into the event tables (state_format/encode.h).
//
// The engine calls the obs_* hooks below at its choke points (Orderer's zone moves, draws and
// shuffles, the reveal-window opens and closes in queries/visibility.cpp, searches, face turns,
// game boundaries). Each hook reads the game once per viewer through the visibility accessors
// (individually_identifiable, identity_visible, visible_card_id, library_position_seen_by) and the
// handle tables, and builds that viewer's projected event from those answers alone. The queued
// ObsEvent holds only those answers (handles, card ids, positions, counts): no entity id reaches a
// queue, so nothing reaches the encoded bytes unfiltered.
//
// Lifetime. The queues live in Game::obs_events, so a snapshot covers them. Each viewer's `seq`
// runs per MATCH (1 at match start, +1 per event, never reset within the match), so the queues
// (seq counters and any events not yet delivered) are carried over the per-game Game reset into
// the next game of a bo3 match (obs_events_for_next_game); the first game of a match starts fresh.
// Events still queued when a game ends are delivered in that viewer's next frame (the sideboarding
// frames, or the next game's), after the game_end event that follows them in the queue.
//
// Nothing here changes the game: with rmstate_events_enabled() false every hook returns at once,
// and with it true the hooks only read the game, append to the queues and issue handles
// (Game::handle_tables), which play never reads.

// Library position (§5): anchor (rmstate::LibAnchor), pos from that end (0 = the end card), span
// (1 exact, n = one of n consecutive positions from pos, 0 = anywhere). anchor NONE8 / pos NONE16 /
// span 0 = not in a library (or, for a destination, held back by an order-hidden batch).
struct ObsLibPos {
        uint8_t anchor = rmstate::NONE8;
        uint16_t pos = rmstate::NONE16;
        uint16_t span = 0;
};

// One card of an event, as its viewer saw it (event_card row).
struct ObsEventCard {
        uint32_t handle_before = 0;            // the viewer's handle for the object before the event (0 none)
        uint32_t handle_after = 0;             // the viewer's handle for the object after it (0 none)
        int card = -1;                         // card id when the viewer saw the identity, else -1
        ObsLibPos from;                        // position in a library before the event
        ObsLibPos to;                          // position in a library after it
        bool face_down = false;                // exiled face down
        bool token = false;                    // a token or a spell copy
        uint8_t hand_subset = rmstate::NONE8;  // rmstate::HandSubset for a card leaving a hand
        uint32_t draw_seq = 0;                 // reveal_start: seq of the draw that brought it (0 none)
        uint16_t draw_idx = rmstate::NONE16;   // its index among that draw's cards
};

// One player's part of a game-boundary event (event_player row).
struct ObsEventPlayer {
        Zone::Ownership player = Zone::UNKNOWN;
        uint16_t library = 0;
        uint16_t sideboard = 0;
        uint8_t wins = 0;
};

// One event as one viewer saw it (event row). Seats are absolute; the encoder adds the
// viewer-relative columns. Zones are Zone::ZoneValue, -1 = none (created / ceased to exist).
struct ObsEvent {
        uint32_t seq = 0;
        rmstate::EventKind kind = rmstate::EventKind::ZONE_MOVE;
        Zone::Ownership player = Zone::UNKNOWN;
        int zone = -1;
        int to_zone = -1;
        Zone::Ownership zone_owner = Zone::UNKNOWN;
        Zone::Ownership to_owner = Zone::UNKNOWN;
        uint16_t count = 0;
        ObsLibPos range;
        bool order_known = false;
        uint16_t batch = 0;                  // 0 = none
        Zone::Ownership to = Zone::UNKNOWN;  // audience of a reveal or look (UNKNOWN = all)
        uint32_t source = 0;                 // handle of the effect's source object (0 none)
        int source_card = -1;                // card id of the effect's source (-1 none)
        uint16_t turn = 0;
        bool opening = false;
        uint8_t game = 0;
        std::vector<ObsEventCard> cards;
        std::vector<ObsEventPlayer> players;
};

// The frame writers that drain the queues; each takes its own copy of every event, at its own
// pace (the dump at every decision, the stdout stream at that viewer's decisions).
enum class ObsConsumer : uint8_t {
    DUMP,    // --rmstate-dump (state_format/dump.h)
    STREAM,     // --machine-format rmstate (state_format/stream.h)
    SELFCHECK,  // --selfcheck-state (state_format/selfcheck.h)
};

// One viewer's queue.
struct ObsEventQueue {
        uint32_t next_seq = 1;
        std::vector<ObsEvent> events;         // not yet in a --rmstate-dump frame
        std::vector<ObsEvent> stream_events;     // not yet in a --machine-format rmstate frame (seats only)
        std::vector<ObsEvent> selfcheck_events;  // not yet in a --selfcheck-state frame
};

// The order-hidden batch an instruction is putting cards into a library with (§5): its cards'
// zone_moves carry `id` and no destination for every batched viewer, and the batch's
// library_place gives the range once the instruction is done (ObsLibraryBatch).
struct ObsBatch {
        bool active = false;
        uint16_t id = 0;
        Zone::Ownership library_owner = Zone::UNKNOWN;
        bool batched[3] = {false, false, false};  // per viewer, indexed like Game::handle_tables
        bool on_bottom = false;
        std::vector<ObjectRef> cards;  // the cards placed so far (their new objects)
};

// An object that left the stack while another stack object was there to refer to it (a spell's
// target), as each viewer saw it go: its last-known characteristics for the frame's lki rows
// (CR 608.2h; encode.cpp's stack targets). Recorded by obs_begin_move, kept until a stack object
// leaves an otherwise empty stack (nothing on it can refer to a departed object any more).
struct ObsStackDeparture {
        ObjectRef ref;  // the object as it was on the stack
        struct Seen {
                bool visible = false;  // the viewer saw its identity (a face-up spell: everyone)
                int card = -1;         // its card id as the viewer saw it
        } seen[3];                     // indexed like Game::handle_tables
        bool token = false;            // a token or a spell copy
        Zone::Ownership controller = Zone::UNKNOWN;
        int power = 0;
        int toughness = 0;
        uint32_t types = 0;  // rmstate card_type mask
        uint8_t colors = 0;  // rmstate color mask
};

// Everything the events channel keeps (Game::obs_events).
struct ObsEventState {
        ObsEventQueue queues[3];    // indexed like Game::handle_tables: omniscient, A, B
        bool game_started = false;  // game_start was queued for the game in progress
        uint16_t next_batch = 1;
        ObsBatch batch;
        // The hand subset the running instruction moves cards out of a hand from (ObsHandSubsetScope).
        uint8_t hand_subset = static_cast<uint8_t>(rmstate::HandSubset::ANY);
        uint32_t last_draw_seq[3] = {0, 0, 0};  // per viewer: seq of the latest draw event
        // The miracle card drawn first this turn and the seq of its draw event per viewer, for the
        // draw link on its reveal (CR 702.94a).
        ObjectRef miracle_card;
        uint32_t miracle_draw_seq[3] = {0, 0, 0};
        std::vector<ObsStackDeparture> stack_departed;
};

// The object `ref` named as it left the stack (ObsStackDeparture), with in `card` its card id as
// `viewer` saw it; null when it did not leave the stack while referable, or the viewer did not see
// its identity.
const ObsStackDeparture *obs_stack_departure(const ObjectRef &ref, Zone::Ownership viewer, int &card);

// Whether anyone consumes events (a frame writer: --rmstate-dump or --machine-format rmstate; the
// --selfcheck-state frames; or --trace-events). Every obs_* hook is a no-op otherwise.
//
// Under --selfcheck-state each seat's event is built twice as it is constructed, the second time
// in a world where every card hidden from that seat has another identity (ScrambledIdentities,
// state_format/selfcheck.h), and the two must project to the same rows: a hook that reads a hidden
// identity, by any route, shows up as an ERROR: line.
bool rmstate_events_enabled();

// ── Zone moves (Orderer) ─────────────────────────────────────────────────────────────────────

// What each viewer saw of an object just before it moves: taken by obs_begin_move, completed into
// the move's event by obs_end_move. Holds only per-viewer answers (no entity).
struct ObsMoveCapture {
        bool active = false;
        bool created = false;  // the object comes into existence with this move (a token)
        int origin = -1;
        Zone::Ownership origin_owner = Zone::UNKNOWN;
        bool origin_hidden = false;
        bool token = false;
        uint8_t hand_subset = rmstate::NONE8;
        struct Seen {
                bool visible = false;
                uint32_t handle = 0;
                int card = -1;
                ObsLibPos from;
        } before[3];
};

// How a move is reported (obs_end_move).
struct ObsMoveInfo {
        bool draw = false;       // a draw (perform_draw): a draw event, not a zone_move
        bool opening = false;    // an opening-hand or mulligan draw
        bool on_bottom = false;  // put on the bottom of a library (else on top, or at a depth)
};

// Called as `e` starts to leave its zone, before its reveal windows close and its handles retire
// (Orderer::add_to_zone, the cast move and its rewind).
ObsMoveCapture obs_begin_move(Entity e);
// Called once `e` is in its new zone with its handles linked (`after_handles[slot]`, indexed like
// Game::handle_tables, from handles_end_move): queues the move for every viewer.
void obs_end_move(const ObsMoveCapture &cap, Entity e, const uint32_t after_handles[3], const ObsMoveInfo &info);

// `e` was just created in its zone without a move: a test-harness preset (Orderer::place_*), a
// spell copy created on the stack. Reported as a zone_move from zone none.
void obs_object_created(Entity e);
// `e` is about to cease to exist (a token or a spell copy, CR 111.7 / 707.10a): a zone_move to
// zone none; its handles retire. `spell_copy` marks a copy whose Spell component is already gone.
void obs_object_ceased(Entity e, bool spell_copy);

// `owner`'s library was shuffled (after its reveal windows closed).
void obs_shuffle(Zone::Ownership owner);
// `e` was turned face up (or face down) where it is (CR 708.2).
void obs_face_turned(Entity e, bool face_up);
// `searcher` searched `owner`'s `zone` and found `found` cards in it (CR 701.23).
void obs_search(Zone::Ownership searcher, Zone::ZoneValue zone, Zone::Ownership owner, int found);
// The miracle card `card` was just drawn as its owner's first card this turn: its reveal links to
// that draw (obs_windows_opened's draw_link).
void obs_note_miracle_draw(Entity card);

// ── Reveal windows (queries/visibility.cpp) ──────────────────────────────────────────────────

// Windows just opened on `cards` (in the order they opened) for `audience` (Zone::UNKNOWN = all
// players): a reveal_start, or a look when `look`, per zone, for every viewer. `position_public`:
// the windows opened on library positions every player knows (the top N, the whole library);
// false for a card picked out by identity. `draw_link`: the reveal shows a card as the one just
// drawn (the miracle reveal).
void obs_windows_opened(
    const std::vector<Entity> &cards, Zone::Ownership audience, bool look, bool position_public, bool draw_link);
// One window that is closing.
struct ObsClosedWindow {
        Entity card = 0;
        Zone::Ownership audience = Zone::UNKNOWN;
};
// `windows` are closing for a reason other than their card moving or being turned face up: a
// reveal_end per zone and audience for every viewer each was shown to. Called while they are still
// open (their handles are still live).
void obs_windows_closed(const std::vector<ObsClosedWindow> &windows);

// ── Instruction context ──────────────────────────────────────────────────────────────────────

// While alive, the cards an instruction puts into `library_owner`'s library form one order-hidden
// batch (§5) for every seat other than `chooser`, the player who orders them (Zone::UNKNOWN: a
// random order, hidden from both seats); the omniscient viewer is never batched. close() (or the
// destruction) ends the batch and queues its library_place.
class ObsLibraryBatch {
    public:
        ObsLibraryBatch(Zone::Ownership library_owner, Zone::Ownership chooser);
        ~ObsLibraryBatch();
        void close();
        ObsLibraryBatch(const ObsLibraryBatch &) = delete;
        ObsLibraryBatch &operator=(const ObsLibraryBatch &) = delete;

    private:
        bool opened_ = false;
};

// While alive, cards leaving a hand are reported with hand subset `subset` (an instruction that
// may only choose among cards drawn this turn: Sylvan Library).
class ObsHandSubsetScope {
    public:
        explicit ObsHandSubsetScope(rmstate::HandSubset subset);
        ~ObsHandSubsetScope();
        ObsHandSubsetScope(const ObsHandSubsetScope &) = delete;
        ObsHandSubsetScope &operator=(const ObsHandSubsetScope &) = delete;

    private:
        uint8_t prev_;
};

// ── Game boundaries (game_driver.cpp) ────────────────────────────────────────────────────────

// The events state the next game starts with: the current one carried over (queues and seq
// counters, the per-game parts cleared) for a later game of a bo3 match, else a fresh one. Called
// just before the per-game Game reset, whose result takes it.
ObsEventState obs_events_for_next_game();
// The libraries are built and shuffled, before the opening draws: game_start.
void obs_game_start(bool player_a_goes_first);
// The game is over (`winner`: Zone::PLAYER_A / PLAYER_B, else a draw): game_end.
void obs_game_end(int winner);
// The match is over: match_end with the final score.
void obs_match_end(int winner, int wins_a, int wins_b);

// ── Draining ─────────────────────────────────────────────────────────────────────────────────

// `viewer`'s events queued for `consumer`, in order, removing them from that consumer's queue (a
// frame is built from them).
std::vector<ObsEvent> take_obs_events(Zone::Ownership viewer, ObsConsumer consumer);

#endif /* STATE_FORMAT_EVENTS_H */
