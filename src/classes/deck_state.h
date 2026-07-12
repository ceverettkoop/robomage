#ifndef DECK_STATE_H
#define DECK_STATE_H

#include <vector>

#include "../components/zone.h"

struct Deck;

// Persistent, match-scoped store of each player's STATIC decklist (name -> count)
// resolved to vocab indices. Lives outside the per-game ECS (like match_state) so
// the state serializer can read the opponent's configured list without threading
// the Deck struct through the call chains. Refreshed at game start (after any
// sideboard phase has already mutated the Deck structs), so game 2+ of a bo3 sees
// the post-board configuration.
struct DecklistEntry {
    int vocab_idx;  // card vocab index (>= 0)
    int count;      // copies of that card in the block
};

// Sorted ascending by vocab_idx, one entry per distinct vocab id. main + side per player.
extern std::vector<DecklistEntry> g_deck_main_a;
extern std::vector<DecklistEntry> g_deck_side_a;
extern std::vector<DecklistEntry> g_deck_main_b;
extern std::vector<DecklistEntry> g_deck_side_b;

// Convert `deck`'s main + sideboard card names to vocab indices and store them for
// `owner`, merged by vocab id and sorted ascending. Fatal (fatal_error) if a card
// name has no vocab index, or if a block has more distinct names than its
// serialized slot count (DECKLIST_MAIN_SLOTS / DECKLIST_SIDE_SLOTS) — loud, never a
// silent truncation. Call at game start for each player.
void deck_state_set(Zone::Ownership owner, const Deck &deck);

// Zero both players' lists (match start). Not required between bo3 games — the
// per-game refresh overwrites — but keeps single-game runs deterministic.
void deck_state_reset();

const std::vector<DecklistEntry> &deck_state_main(Zone::Ownership owner);
const std::vector<DecklistEntry> &deck_state_side(Zone::Ownership owner);

#endif /* DECK_STATE_H */
