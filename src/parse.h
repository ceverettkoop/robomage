#ifndef PARSE_H
#define PARSE_H

#include <string>
#include "ecs/entity.h"
#include "components/token.h"

std::string name_to_uid(std::string name);
Entity parse_card_script(std::string path);
Token parse_token_script(const std::string &script_name);

#endif /* PARSE_H */
