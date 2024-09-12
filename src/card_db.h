#ifndef CARD_DB_H
#define CARD_DB_H

#include <string>
#include <cstdint>

//this is the underlying card, a token or copy would not have these traits
class Card{
    std::string name;
    std::string oracle_text;
    uint32_t uid;
};

#endif /* CARD_H */
