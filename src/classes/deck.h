#ifndef DECK_H
#define DECK_H

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

struct Deck;

struct Deck {
        Deck(){};
        Deck(std::string path);
        std::vector<std::pair<size_t, std::string>> main_deck;
        std::vector<std::pair<size_t, std::string>> sideboard;
};

#endif /* DECK_H */
