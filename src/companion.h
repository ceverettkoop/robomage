#ifndef COMPANION_H
#define COMPANION_H

#include <memory>
#include <string>

#include "classes/deck.h"

class Orderer;

// Companion (CR 702.139). A companion begins the game in the sideboard ("outside the game"). If
// the chosen companion's deckbuilding restriction is satisfied by the starting deck, its
// controller may, once per game at sorcery speed, pay {3} to put it from the sideboard into their
// hand (a special action — see the COMPANION block in determine_legal_actions and the
// companion_to_hand branch in process_action).

// Evaluate a Forge companion restriction token against a deck. Currently supports the
// "DeckSizePlus<N>" family (Yorion: "DeckSizePlus20" — the starting deck must contain at least N
// cards more than the minimum deck size). Unknown tokens fail closed (companion not eligible).
bool deck_meets_companion_restriction(const std::string &restriction, const Deck &deck);

// At game start, for each player, find a Companion-keyworded card in their sideboard whose
// deckbuilding restriction the starting deck meets, instantiate it as a SIDEBOARD-zone entity if
// it is not already one, and record it on that Player as their chosen companion. Safe to call once
// per game after the libraries and any preset sideboard entities are in place.
void setup_companions(const Deck &deck_a, const Deck &deck_b,
                      const std::shared_ptr<Orderer> &orderer);

#endif /* COMPANION_H */
