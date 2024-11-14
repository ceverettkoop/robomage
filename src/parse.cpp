#include "parse.h"

#include <string>
#include <algorithm>
#include <cctype>
#include <vector>

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

Card parse_card_script(std::__1::ifstream stream) {
    return Card();
}
