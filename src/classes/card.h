#ifndef CARD_H
#define CARD_H

#include <string>

//this is the underlying card, a token or copy would not have these traits
struct Card{
    std::string uid;
    std::string name;
    std::string oracle_text;
};

#endif /* CARD_H */
