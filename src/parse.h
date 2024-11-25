#ifndef PARSE_H
#define PARSE_H

#include <string>
#include "ecs/entity.h"

std::string name_to_uid(std::string name);
Entity parse_card_script(std::string path);

#endif /* PARSE_H */
