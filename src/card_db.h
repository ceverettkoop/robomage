#ifndef CARD_DB_H
#define CARD_DB_H

#include <string>
#include <cstdint>
#include <unordered_map>

//this is the underlying card, a token or copy would not have these traits
struct Card{
    std::string uid;
    std::string name;
    std::string oracle_text;
};

extern std::unordered_map<std::string, Card> card_db;

//cards are loaded into db on demand
void load_card(std::string card_name);

#endif /* CARD_H */
