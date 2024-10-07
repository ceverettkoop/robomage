#ifndef CARD_DB_H
#define CARD_DB_H

#include <string>
#include <cstdint>
#include <unordered_map>

#include "card.h"

extern std::unordered_map<std::string, Card> card_db;
extern std::string RESOURCE_DIR;

//cards are loaded into db on demand
const Card& load_card(std::string card_name);

#endif /* CARD_H */

