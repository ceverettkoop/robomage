#ifndef MATCH_CONTEXT_H
#define MATCH_CONTEXT_H

#include <string>
#include <unordered_set>

#include "../components/zone.h"
#include "deck.h"

// Value structs describing the state of a best-of-three MATCH between games.
// Deliberately hold NO ECS/entity members: everything here survives an
// init_ecs() teardown so a later MCTS search rooted at a sideboard prompt can
// snapshot the match, roll forward into the next game, and restore. Stage 1
// only introduces the shape (a pure refactor of play_bo3_match); nothing yet
// snapshots or restores these.

// Per-player sideboard-phase state. run_sideboard_phase reads/writes this so the
// phase is resumable in principle: its swap counter and one-shot sided-in/out
// bookkeeping live here rather than in call-scoped locals.
struct SideboardPhaseState {
    Zone::Ownership player = Zone::UNKNOWN;
    int sb_swaps = 0;
    // Each card is a one-shot decision per phase: once moved it cannot be moved
    // back (prevents oscillation, keeps the 15-swap cap easy to reason about).
    // Keyed by card name to mirror the original locals.
    std::unordered_set<std::string> sided_out_names;
    std::unordered_set<std::string> sided_in_names;
    // OUT-menu resumption point: -1 = at the IN menu; else the sideboard index of
    // the chosen IN card (re-derived via load_card(deck.sideboard[idx].second) on
    // re-entry so the OUT menu resumes with its pending-decision context intact).
    long pending_in_sb_idx = -1;
    // Maindeck drift from its size at phase start, constrained to {-1, 0, +1} and
    // serialized into the observation. Always 0 under the paired IN->OUT menu,
    // which applies a swap atomically and so never leaves the deck unbalanced at a
    // decision point; it becomes live when the balanced delta menu lands.
    int delta = 0;
};

// The whole match's between-game state, in one snapshottable value struct.
struct MatchContext {
    bool active = false;
    int game_num = 0;
    int wins_a = 0;
    int wins_b = 0;
    bool a_goes_first = true;
    unsigned int base_seed = 0;
    enum Stage { PLAY_GAME, SIDEBOARD_A, SIDEBOARD_B } stage = PLAY_GAME;
    Deck deck_a;
    Deck deck_b;
    SideboardPhaseState sb;
};

#endif /* MATCH_CONTEXT_H */
