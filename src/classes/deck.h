#ifndef DECK_H
#define DECK_H

#include <cstddef>
#include <set>
#include <string>
#include <utility>

struct Deck;

struct Deck {
        Deck(){};
        Deck(std::string path);
        std::set<std::pair<size_t, std::string>> main_deck;
        std::set<std::pair<size_t, std::string>> sideboard;
};

#endif /* DECK_H */
