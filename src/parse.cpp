#include "parse.h"

#include <string>
#include <algorithm>
#include <cctype>
#include <vector>
#include <fstream>
#include "error.h"

const size_t SCRIPT_MAX_LEN = 10000;

static std::string value_from_script(std::string script, std::string key);

//all to lowercase, spaces to underscores, other characters removed
std::string name_to_uid(std::string name) {
    auto len = name.size();
    std::vector<size_t> to_rm;

    for (size_t i = 0; i < name.size(); i++){
        char value = name[i];
        if(std::isalpha(value)){
            name[i] = std::tolower(value);
        }else if(value == ' '){
            name[i] = '_';
        }else{
            to_rm.push_back(i);
        }
    }
    for (size_t i = 0; i < to_rm.size(); i++){
        auto index_to_rm = to_rm[i] - i;
        name.erase(index_to_rm);
    }
    return std::string();
}

int parse_card_script(Card& card, std::string path) {
    Card ret_val;
    std::string script_data;
    auto stream = std::ifstream(path);
    if(!stream.is_open()) return FAILED_TO_OPEN_STREAM;
    for (size_t i = 0; true; i++){
        if(i > SCRIPT_MAX_LEN) return SCRIPT_TOO_LONG;
        char c = stream.get();
        if(c == stream.eof()) break;
        script_data += c;
    }
    card.name = value_from_script(script_data, "Name");
    card.uid = name_to_uid(card.name);
    card.mana_cost = parse_mana_cost(value_from_script(script_data, "ManaCost"));
    card.types = parse_types(value_from_script(script_data, "Types");

    return PARSE_SUCCESS;
}

//private util functions
static std::string value_from_script(std::string script, std::string key){
    auto pos = script.find(key);

}