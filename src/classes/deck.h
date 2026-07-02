#ifndef DECK_H
#define DECK_H

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

struct Deck;

struct Deck {
        Deck() {};
        Deck(std::string path);
        // Build a deck from raw decklist text (same format as a .dk file). Used by
        // replay to reconstruct the decks embedded in an RMLOG v2 decision log.
        static Deck from_string(const std::string& text);
        std::vector<std::pair<size_t, std::string>> main_deck;
        std::vector<std::pair<size_t, std::string>> sideboard;
        // Verbatim decklist file content (including SIDEBOARD: section); embedded in
        // v2 decision logs so a replay is self-contained.
        std::string raw_text;

       private:
        void parse_text(const std::string& text);
};

#endif /* DECK_H */
