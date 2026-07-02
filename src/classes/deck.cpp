#include "deck.h"

#include <fstream>
#include <sstream>

#include "../card_db.h"
#include "../error.h"

extern std::string RESOURCE_DIR;

Deck::Deck(std::string path) {
    auto stream = std::fstream(path);
    if (!stream.is_open()) {
        fatal_error("Failed to open decklist at " + path);
        return;
    }

    std::stringstream buffer;
    buffer << stream.rdbuf();
    raw_text = buffer.str();
    parse_text(raw_text);
}

Deck Deck::from_string(const std::string &text) {
    Deck deck;
    deck.raw_text = text;
    deck.parse_text(text);
    return deck;
}

void Deck::parse_text(const std::string &text) {
    std::istringstream stream(text);
    std::string line;
    bool in_sideboard = false;

    while (std::getline(stream, line)) {
        if (line.empty()) continue;

        if (line == "SIDEBOARD:") {
            in_sideboard = true;
            continue;
        }

        size_t space_pos = line.find(' ');
        if (space_pos == std::string::npos) continue;

        size_t count = std::stoul(line.substr(0, space_pos));
        std::string card_name = line.substr(space_pos + 1);

        if (in_sideboard) {
            sideboard.push_back({count, card_name});
        } else {
            main_deck.push_back({count, card_name});
        }
    }
}
