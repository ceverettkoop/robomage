#ifndef KNOWLEDGE_TRUTH_H
#define KNOWLEDGE_TRUTH_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "bounds.h"
#include "frame_view.h"
#include "k_rows.h"

// ── Soundness: the k_* facts against the true world (docs/knowledge_store.md §7) ──────────────
//
// Every fact a seat's knowledge tables report must hold in the true world of that decision, with
// zero tolerance. Two drivers build the true world: `rmknow --truth` from the omniscient frames of a
// --rmstate-dump (truth_from_snapshot), and the engine's --selfcheck-knowledge from the engine state
// itself (state_format/selfcheck_knowledge.h). The comparison is shared: check_truth.
//
// Facts checked (one each):
//   - a k_pool row's size: the opponent's hand, a sideboard, a library range (or block of runs) and
//     a face-down card (when the truth knows face-down cards by the viewer's handle);
//   - a k_pool_card row: the pool's true copies of the card are within [min, max];
//   - per such pool, that no card it truly holds is missing from its rows while in the owner's
//     universe (outside it, under --closed-decklist, a card never seen has no row);
//   - a k_position row: the card at that position when exact, else one of the candidates of the
//     library-range pool it names, which covers the position;
//   - a k_known_hand row: the opponent's hand holds at least that many copies;
//   - a k_seen row: the player owns at least the copies seen this game, and the match count is at
//     least the game count;
// Set-aside pools (test-harness presets not yet created) have no counterpart in the true world.

namespace knowledge {

// The true world of one decision. Seats are player indices (0 = A, 1 = B); cards are vocab ids.
struct Truth {
        std::map<int, int> hand[2];
        std::map<int, int> sideboard[2];
        std::map<int, int> owned[2];  // every card object of the player (tokens and copies excluded)
        std::vector<int> library[2];  // top first
        // The card of every face-down object, by the viewer's handle; only when face_down_known (an
        // omniscient frame's handles are not the viewer's).
        std::map<uint32_t, int> face_down;
        bool face_down_known = false;
};

// The truth an omniscient STATE frame shows; false when its library positions are not 0..n-1.
bool truth_from_snapshot(const Snapshot &omni, Truth *out);

struct TruthCheck {
        size_t facts = 0;
        std::vector<std::string> errors;  // one line per violated fact
};

// Checks every fact of `rows`, viewer seat `viewer`'s k_* tables, against `truth`, adding to `out`.
// `universe[p]` is the card list player p's pools are reported over (Knowledge::player(p).full).
void check_truth(const KRows &rows, const Truth &truth, int viewer, const Universe *const universe[2], TruthCheck *out);

}  // namespace knowledge

#endif /* KNOWLEDGE_TRUTH_H */
