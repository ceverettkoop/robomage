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
// name sets, delta, unpaired_name, force_done), so after a MATCH-scoped restore
// the phase re-derives the identical menu — which is what makes a sideboard
// prompt a loop-safe MCTS search root.
struct SideboardPhaseState {
    Zone::Ownership player = Zone::UNKNOWN;
    // Completed swaps (an IN paired with the OUT that balances it). The forced
    // cut that resolves a stranded IN counts as one.
    int sb_swaps = 0;
    // One-shot direction locks, keyed by card name: a name in sided_out_names can
    // no longer move INTO the maindeck, one in sided_in_names can no longer move
    // OUT of it. They prevent oscillation, and every completed swap permanently
    // shrinks both pools — so the phase is guaranteed to terminate.
    std::unordered_set<std::string> sided_out_names;
    std::unordered_set<std::string> sided_in_names;
    // Maindeck drift from its size at phase start, constrained to {0, +1} and
    // serialized into the observation. Every swap opens with its IN half, so the
    // deck is only ever oversized; +1 means a cut is outstanding, the menu offers
    // only cuts and Done is withheld, so the deck can never be submitted off-size.
    int delta = 0;
    // The card whose unpaired IN produced delta == +1 ("" when balanced).
    std::string unpaired_name;
    // Set when a stranded +1 was resolved by the forced cut of main_deck[0]: the
    // next balanced menu offers ONLY Done, force-ending the phase.
    bool force_done = false;
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
