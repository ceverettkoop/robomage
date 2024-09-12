#ifndef CARD_DB_H
#define CARD_DB_H

#include <string>
#include <cstdint>
#include <unordered_map>

//this is the underlying card, a token or copy would not have these traits
struct Card{
    std::string name;
    std::string oracle_text;
    uint32_t uid;
};

extern std::unordered_map<uint32_t, Card> card_db;

#endif /* CARD_H */
