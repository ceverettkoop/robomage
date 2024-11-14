#ifndef PARSE_H
#define PARSE_H

#include <string>
#include <fstream>
#include "card.h"

std::string name_to_uid(std::string name);
Card parse_card_script(std::__1::ifstream stream);

#endif /* PARSE_H */k
