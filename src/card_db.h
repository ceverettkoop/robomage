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

//True when `card_name` names the back face of `cd` (a double-faced card loaded by
//its back-face name) rather than its front face.
bool names_back_face(const std::string& card_name, const CardData& cd);

#endif /* CARD_H */

