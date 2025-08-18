#include "card_db.h"
#include "parse.h"
#include "error.h"

std::unordered_map<std::string, Entity> card_db;

Entity load_card(std::string card_name) {
    //search for card based on normalized name string
    auto uid = name_to_uid(card_name);
    auto itr = card_db.find(uid);
    //check if already loaded
    if(itr != card_db.end()) return itr->second;
    //load script
    std::string path = RESOURCE_DIR + "/" + uid[0] + "/" + uid + ".txt";
    Entity parsed_card = parse_card_script(path);
    if(parsed_card < 0){
        non_fatal_error("Failed to parse card " + card_name);
    }
    return parsed_card;
}
