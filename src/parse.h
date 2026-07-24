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

// Parse a TRIGGERED ability granted by a Continuous static's AddTrigger$/AddSVar$ (The Tabernacle
// at Pendrell Vale). `trigger_line` is the resolved Mode$ ... trigger body; `svar_name`/`svar_body`
// are the Execute$ SVar it references (so the trigger's Execute can be reparsed at grant time).
// Returns a TRIGGERED Ability (trigger_on == 0 if unrecognised). The layer-6 grant pass attaches a
// per-recipient copy to each affected permanent's ability list.
Ability parse_granted_trigger(const std::string &trigger_line, const std::string &svar_name,
                              const std::string &svar_body);

// Parse a Ward cost argument — the text after "Ward:" in a K: line or a granted keyword
// (CR 702.21). Sets `cost` and `is_life` for the two supported shapes: "PayLife<N>"
// (Ward—Pay N life) and a plain numeric "N" ({N} generic mana). A missing or malformed arg
// degrades to a {1} mana ward rather than crashing card load (with -fno-exceptions a stray
// std::stoi would abort). Shared by the printed-ward K: parse (parse.cpp) and the
// granted-ward keyword scan (collect_ward_instances in action_processor.cpp) so both paths
// enforce the same cost shapes.
void parse_ward_cost(const std::string &arg, int &cost, bool &is_life);

#endif /* PARSE_H */
