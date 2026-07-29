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
// phase is resumable: the whole menu is a pure function of (deck, these one-shot
// name sets, delta, unpaired_name), so after a MATCH-scoped restore the phase
// re-derives the identical menu — which is what makes a sideboard prompt a
// loop-safe MCTS search root.
struct SideboardPhaseState {
    Zone::Ownership player = Zone::UNKNOWN;
    // Completed swaps (a +1 paired with a -1). Takebacks do not count.
    int sb_swaps = 0;
    // One-shot direction locks, keyed by card name: a name in sided_out_names can
    // no longer move INTO the maindeck, one in sided_in_names can no longer move
    // OUT of it. They prevent oscillation, and a takeback adds its card to the
    // opposing set — so each takeback permanently shrinks one pool and the phase
    // is guaranteed to terminate.
    std::unordered_set<std::string> sided_out_names;
    std::unordered_set<std::string> sided_in_names;
    // Maindeck drift from its size at phase start, constrained to {-1, 0, +1} and
    // serialized into the observation. Nonzero means a move is outstanding: the
    // menu offers only the balancing direction and Done is withheld, so the deck
    // can never be submitted off-size.
    int delta = 0;
    // The card whose unpaired move produced a nonzero delta ("" when balanced).
    // It is exempt from its own direction lock, which is what lets an exploratory
    // move be taken back in one ply instead of forcing a matching move.
    std::string unpaired_name;
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
