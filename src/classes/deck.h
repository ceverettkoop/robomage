#ifndef DECK_H
#define DECK_H

#include <set>
#include <string>
#include <utility>
#include <cstddef>

struct Deck;

extern const Deck DEFAULT_DECK_ONE;
extern const Deck DEFAULT_DECK_TWO;

struct Deck{
    Deck(std::string path);
    std::set<std::pair<size_t, std::string>> main_deck;
    std::set<std::pair<size_t, std::string>> sideboard;
};

#endif /* DECK_H */
