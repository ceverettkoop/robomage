#ifndef DECK_H
#define DECK_H

#include <set>
#include <string>
#include <utility>
#include <cstddef>

const Deck DEFAULT_DECK_ONE;
const Deck DEFAULT_DECK_TWO;

struct Deck{
    std::set<std::pair<size_t, std::string>> cards;    
};

#endif /* DECK_H */
