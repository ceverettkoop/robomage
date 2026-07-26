#ifndef DECK_STATE_H
#define DECK_STATE_H

#include <vector>

#include "../components/zone.h"

struct Deck;

// Persistent, match-scoped stores of each player's decklist (name -> count)
// resolved to vocab indices. They live outside the per-game ECS (like
// match_state) so the state serializer can read a decklist without threading the
// Deck struct through the call chains.
//
// There are TWO stores, and the distinction is a rules/information one:
//
//   REGISTERED — the 75 as registered at MATCH start. Frozen for the whole
//     match: a sideboard swap does not touch it and a game start does not
//     refresh it. This is the OPPONENT-VISIBLE list under this engine's
//     open-decklist ruleset — you know your opponent's registered maindeck and
//     sideboard, but NOT which 60 of the 75 they boarded into for game 2+.
//     Serialized into the opp_deck_main / opp_deck_side observation blocks (see
//     the machine_io.h tail). This matches the MCTS determinizer, which already
//     models the opponent's post-board split as hidden information by mixing
//     their sideboard into the exchange pool for game 2+ (see
//     `opp_sideboard_hidden` in search_server.cpp).
//
//   LIVE — each player's CURRENT configuration: refreshed at every game start
//     and again as each sideboard swap lands, so it tracks the mutating Deck
//     struct. This is what a player legitimately knows about their OWN deck.
//     Not serialized yet; it exists as the seam for a future self-decklist
//     observation block, and is covered by the deck-state tests.
struct DecklistEntry {
    int vocab_idx;  // card vocab index (>= 0)
    int count;      // copies of that card in the block
};

// Set the REGISTERED (frozen, opponent-visible) list for `owner`. Call ONCE per
// match, before game 1 — calling it again mid-match would leak the post-board
// configuration, which is exactly what this store exists to prevent.
void deck_state_set_registered(Zone::Ownership owner, const Deck &deck);

// Set the LIVE (own-view) list for `owner` from the current Deck struct. Called
// at every game start and after every completed sideboard swap.
void deck_state_set_live(Zone::Ownership owner, const Deck &deck);

// Zero both stores for both players. Call at match start (and before a single
// game) so a previous match's lists can never bleed through.
void deck_state_reset();

// Registered (frozen) accessors — what the OPPONENT of `owner` is allowed to see.
const std::vector<DecklistEntry> &deck_state_registered_main(Zone::Ownership owner);
const std::vector<DecklistEntry> &deck_state_registered_side(Zone::Ownership owner);

// Live accessors — `owner`'s own current configuration.
const std::vector<DecklistEntry> &deck_state_live_main(Zone::Ownership owner);
const std::vector<DecklistEntry> &deck_state_live_side(Zone::Ownership owner);

// Whole-store copy for the match-scoped game snapshot (see snapshot.cpp): a
// sideboard-rooted MCTS sim mutates the LIVE store (each simulated swap) and
// rolls forward into game k+1, whose play_single_game start refreshes it again —
// so the snapshot must capture at the sideboard root and restore wholesale on
// rollback. The registered store is frozen and so is carried for completeness
// only (restoring it is a no-op on the real line).
struct DeckStateSnapshot {
    std::vector<DecklistEntry> reg_main_a, reg_side_a, reg_main_b, reg_side_b;
    std::vector<DecklistEntry> live_main_a, live_side_a, live_main_b, live_side_b;
};
void deck_state_capture(DeckStateSnapshot &out);
void deck_state_restore(const DeckStateSnapshot &in);

#endif /* DECK_STATE_H */
