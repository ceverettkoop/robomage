#include "deck.h"
#include <fstream>
#include "../error.h"
#include "../card_db.h"

extern std::string RESOURCE_DIR;

Deck::Deck(std::string path) {
    auto stream = std::fstream(path);
    if(!stream.is_open()) non_fatal_error("Failed to open decklist at " + path);
    
    std::string line;
    bool in_sideboard = false;
    
    while(std::getline(stream, line)) {
        if(line.empty()) continue;
        
        if(line == "SIDEBOARD:") {
            in_sideboard = true;
            continue;
        }
        
        size_t space_pos = line.find(' ');
        if(space_pos == std::string::npos) continue;
        
        size_t count = std::stoul(line.substr(0, space_pos));
        std::string card_name = line.substr(space_pos + 1);
        
        if(in_sideboard) {
            sideboard.insert({count, card_name});
        } else {
            main_deck.insert({count, card_name});
        }
    }
}