#ifndef DECK_CHECK_H
#define DECK_CHECK_H

#include <string>
#include <vector>

// ── --check-decks: every deck card implemented (plays no game) ───────────────────────────────
//
// Reads every .dk file of the given directories (relative to resources/decks/; "" is the
// top-level directory; a path ending in ".dk" is that one deck; default: the top-level decks,
// league/ and heldout/, the playable decks — heldout/ may hold none) and checks each distinct card
// it names, main deck and sideboard: the card's script loads (load_card's resolution, a
// double-faced card by its combined name included), and the card the script defines is in the
// card vocabulary (src/card_vocab.h) by its front-face name. Prints one line per failing card,
// most-played first:
//
//   check-decks: MISSING <name>: <not in src/card_vocab.h | no script> (decks <n>, copies <m>[, next index <i>])
//
// (the next free vocab index is suggested for a card with a script, in order), then a summary
// line "check-decks: <d> deck(s), <c> distinct card(s), <k> missing", and returns the process
// exit status: 0 when nothing is missing, 1 otherwise (a directory with no .dk file, reported as
// "check-decks: ERROR no .dk files in <dir>", fails too).
int check_decks(const std::vector<std::string> &dirs);

#endif /* DECK_CHECK_H */
