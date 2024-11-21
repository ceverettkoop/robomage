#include "card_db.h"
#include "parse.h"

std::unordered_map<std::string, Card> card_db;

const Card& load_card(std::string card_name) {
    auto uid = name_to_uid(card_name);
    auto itr = card_db.find(uid);
    //check if already loaded
    if(itr != card_db.end()) return itr->second;
    //load script
    std::string path = RESOURCE_DIR + "/" + uid[0] + "/" + uid + ".txt";
    Card card_data;
    if(parse_card_script(card_data, path)){

    }

}
