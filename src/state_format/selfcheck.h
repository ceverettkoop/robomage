#ifndef STATE_FORMAT_SELFCHECK_H
#define STATE_FORMAT_SELFCHECK_H

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../components/zone.h"
#include "../ecs/entity.h"
#include "events.h"

// ── --selfcheck-state: the state format's engine self-checks (docs/state_format.md §10) ───────
//
// At every decision the self-check builds every viewer's STATE frame (A, B, omniscient) from its
// own copy of the event queues (ObsConsumer::SELFCHECK) and checks:
//
//   roundtrip    decode → re-encode is byte-identical (state_format/decode.h), for every frame
//                and, once per match, each viewer's SCHEMA, CARDS and DECKLISTS frames;
//   scramble     each seat's frame is encoded again with every card hidden from that seat given
//                another identity (ScrambledIdentities): the bytes must be the same. Every event
//                is checked the same way as it is built (state_format/events.cpp);
//   complete     every object the viewer can identify has an object row and every public one
//                does; the player rows' zone counts are the zones' sizes; the reveal table lists
//                the viewer's windows; every handle a snapshot table names is an object row's
//                (event tables exempt: they may name objects that have since gone); every row:
//                reference is in range; the menu has one row per action as offered (the deciding
//                seat's and the omniscient frame) or none;
//
// and, sampled (every --selfcheck-state-sample'th decision):
//
//   permutation  for each seat, everything hidden from it is physically permuted (the
//                opponent's hand, library and sideboard cards dealt among those zones' slots;
//                the seat's own library reordered; the identities of all cards hidden from the
//                seat, face-down exiled cards included, swapped among cards of the same owner),
//                keeping every live reveal window fixed; the seat's frame is encoded in that
//                world and must be the same bytes. Then everything is put back exactly;
//   snapshot     the game is saved (snapshot.h) at a loop-safe decision, play goes on for at
//                least SNAPSHOT_K decisions, and at the next loop-safe decision the saved state is
//                restored, every viewer's frame encoded again and compared with the bytes of the
//                saved decision, and the current state restored.
//
// A failure prints an ERROR: line (non_fatal_error) naming the check, the viewer and the first
// differing table cell. None of the checks changes play: everything they change is put back, the
// RNG is never drawn, and nothing is written to stdout. The self-check is off under
// --search-server (whose cooperative unwind returns from InputLogger::choose before a decision is
// made) and for the in-process actor's decisions.

// Whether --selfcheck-state is in effect (the flag, and not --search-server).
bool selfcheck_state_enabled();

// Called as each decision is read (InputLogger::choose), after the RMSTATE writers'
// bookkeeping (rmstate_begin_decision) and frames, with the menu as offered.
void selfcheck_state_decision(const std::vector<LegalAction> &actions);

// The match is over (after its match_end event): every viewer's terminal frame is checked, and a
// summary line (SELFCHECK-STATE: …) goes to stderr.
void selfcheck_state_match_over();

// ── Event construction (state_format/events.cpp) ─────────────────────────────────────────────

// While alive, every card hidden from `viewer` (a seat) has another identity: the hidden cards'
// CardData are rotated among them so that each holds another card's (another name's whenever no
// name has more than half the hidden copies), and card_identity substitutes another card id for
// a hidden object without CardData. Which cards are hidden is decided from the visibility rules
// of docs/state_format.md §3 directly (zone, owner, controller, face-down state and the viewer's
// reveal windows), not by asking queries/visibility.h. `keep` lists cards left as they are (a
// menu choice that makes its card public). Destruction puts every identity back.
class ScrambledIdentities {
    public:
        explicit ScrambledIdentities(Zone::Ownership viewer, const std::vector<Entity> &keep = {});
        ~ScrambledIdentities();
        ScrambledIdentities(const ScrambledIdentities &) = delete;
        ScrambledIdentities &operator=(const ScrambledIdentities &) = delete;

    private:
        bool active_ = false;  // false for a nested scramble, which changes nothing
        // holders_[i] holds the identity holders_[from_[i]] had (ECS PermuteComponents).
        std::vector<Entity> holders_;
        std::vector<size_t> from_;
};

// One event, built normally and with ScrambledIdentities, for `viewer`: their rows must match.
void selfcheck_state_event(Zone::Ownership viewer, const ObsEvent &built, const ObsEvent &scrambled);

// Reports a failed check (an ERROR: line with the check, viewer and decision).
void selfcheck_state_mismatch(const char *check, Zone::Ownership viewer, const std::string &detail);

#endif /* STATE_FORMAT_SELFCHECK_H */
