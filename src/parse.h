#ifndef PARSE_H
#define PARSE_H

#include <string>
#include "card.h"

std::string name_to_uid(std::string name);
int parse_card_script(Card& card, std::string path);

#endif /* PARSE_H */
