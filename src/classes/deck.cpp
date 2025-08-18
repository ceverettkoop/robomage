#include "deck.h"
#include <fstream>
#include "../error.h"

Deck::Deck(std::string path) {
    auto stream = std::fstream(path);
    if(!stream.is_open()) non_fatal_error("Failed to open decklist at " + path);
    
}