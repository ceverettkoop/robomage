#ifndef PARSE_H
#define PARSE_H

#include <string>
#include "ecs/entity.h"
#include "components/ability.h"
#include "components/token.h"

std::string name_to_uid(std::string name);
Entity parse_card_script(std::string path);
Token parse_token_script(const std::string &script_name);

// Parse a single activated/spell ability body line (e.g. "AB$ Mana | Cost$ T | Produced$ C")
// into an Ability. Used to materialize an AddAbility$ static's granted ability (Petrified
// Hamlet) at grant time, so the same Forge ability grammar (Cost$/Produced$/...) is honoured
// as for a printed A: ability. No SVar table is available here, so SVar-referencing params
// inside the body are left as their literal token (the granted bodies in use are self-contained).
Ability parse_ability_body(const std::string &body, Ability::AbilityType type = Ability::ACTIVATED);

#endif /* PARSE_H */
