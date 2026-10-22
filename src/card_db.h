#ifndef CARD_DB_H
#define CARD_DB_H

#include <string>
#include <cstdint>
#include <unordered_map>
struct Deck;

#include "components/carddata.h"

extern std::unordered_map<std::string, Entity> card_db;
extern std::string RESOURCE_DIR;

//cards are loaded into db on demand
Entity load_card(std::string card_name);

// The printed card `card_name` names (a face's name, as load_card resolves it), read from the
// process-wide parse cache: no entity is created and no game state is touched, so it can look at
// cards no deck holds (the whole vocab). nullptr when the card has no script.
const CardData *card_data_by_name(const std::string &card_name);

// The printed card of card_vocab_entries[`pos`] from the parse cache (card_data_by_name), when that
// entry is its card's front-face name; nullptr for a back face's own vocab entry (not a separate
// card), another alias of the card's id (a split card's "Gone" / "Dead/Gone") or a card with no
// script, so each card is one entry. The names are compared ASCII-folded (ascii_fold_card_name),
// so an accented script `Name:` still matches its ASCII vocab entry.
const CardData *vocab_front_card_data(int pos);

// The raw text of the script file load_card reads for `card_name` (a double-faced card's combined
// script), or "" when there is none. `RESOURCE_DIR/tokenscripts/<stem>.txt` for a token script
// stem through token_script_text.
std::string card_script_text(const std::string &card_name);
std::string token_script_text(const std::string &stem);

//True when `card_name` names the back face of `cd` (a double-faced card loaded by
//its back-face name) rather than its front face.
bool names_back_face(const std::string& card_name, const CardData& cd);

#endif /* CARD_H */

