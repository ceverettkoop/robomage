#include "parse.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>

#include "classes/colors.h"
#include "components/types.h"
#include "components/ability.h"
#include "components/carddata.h"
#include "ecs/coordinator.h"
#include "error.h"
#include "type_constants.h"

const size_t SCRIPT_MAX_LEN = 10000;

static std::string value_from_script(std::string script, std::string key);
static std::vector<std::string> multi_values_from_script(std::string script, std::string key);
static std::multiset<Colors> parse_mana_cost(std::string value);
static std::set<Type> parse_types(std::string value);
static std::set<Entity> parse_abilities(std::vector<std::string> lines, const std::set<Type>& types);
static uint32_t parse_power(std::string value);
static uint32_t parse_toughness(std::string value);

// all to lowercase, spaces to underscores, other characters removed
std::string name_to_uid(std::string name) {
    std::vector<size_t> to_rm;

    for (size_t i = 0; i < name.size(); i++) {
        char value = name[i];
        if (std::isalpha(value)) {
            name[i] = std::tolower(value);
        } else if( (value == ' ') && (i != name.size() - 1) )   { // we will excise up to 1 trailing space, rest to underscores
                name[i] = '_';
        } else {
            to_rm.push_back(i);
        }
    }
    for (size_t i = 0; i < to_rm.size(); i++) {
        auto index_to_rm = to_rm[i] - i;
        name.erase(index_to_rm);
    }

    return name;
}

Entity parse_card_script(std::string path) {
    auto id = global_coordinator.CreateEntity();
    std::string script_data;
    auto stream = std::ifstream(path);
    assert(stream.is_open());
    for (size_t i = 0; true; i++) {
        if (i > SCRIPT_MAX_LEN) fatal_error("Script too long");
        char c = stream.get();
        if (stream.eof()) break;
        script_data += c;
    }
    CardData card;
    card.name = value_from_script(script_data, "Name");
    card.uid = name_to_uid(card.name);
    card.mana_cost = parse_mana_cost(value_from_script(script_data, "ManaCost"));
    card.types = parse_types(value_from_script(script_data, "Types"));
    // TODO optimize
    card.power = parse_power(value_from_script(script_data, "PT"));
    card.toughness = parse_toughness(value_from_script(script_data, "PT"));
    // register abilities associated with this card as entities unique to this card
    card.abilities = parse_abilities(multi_values_from_script(script_data, "A"), card.types);

    // no error handling here
    global_coordinator.AddComponent(id, card);

    return id;
}

// private util functions
static std::string value_from_script(std::string script, std::string key) {
    auto pos = script.find(key);
    if (pos == std::string::npos) {
        return "";
    }
    // advance for key itself and ':'
    pos += key.length() + 1;
    auto end_pos = script.find("\n", pos);
    return script.substr(pos, (end_pos - pos));  // omit linebreak at end
}

static std::vector<std::string> multi_values_from_script(std::string script, std::string key) {
    std::vector<std::string> ret_val;
    auto pos = script.find(key);
    while (pos != std::string::npos) {
        // advance for key itself and ':'
        pos += key.length() + 1;
        auto end_pos = script.find("\n", pos);
        ret_val.push_back(script.substr(pos, (end_pos - pos - 1)));  // omit linebreak at end
        pos = script.find(key, end_pos);                             // find next instance
    }
    return ret_val;
}

static std::multiset<Colors> parse_mana_cost(std::string value) {
    auto len = value.length();
    std::multiset<Colors> ret_val;
    if (value == "no cost") return ret_val;
    for (size_t i = 0; i < len; i++) {
        switch (value[i]) {
            case 'W':
                ret_val.emplace(WHITE);
                break;
            case 'U':
                ret_val.emplace(BLUE);
                break;
            case 'B':
                ret_val.emplace(BLACK);
                break;
            case 'R':
                ret_val.emplace(RED);
                break;
            case 'G':
                ret_val.emplace(GREEN);
                break;
            case 'C':
                ret_val.emplace(COLORLESS);
                break;
            default:
                if (std::isdigit(value[i])) {
                    for (size_t j = 0; j < value[i] - '0'; j++) {
                        ret_val.emplace(GENERIC);
                    }
                }
                break;
        }
    }
    return ret_val;
}

static std::set<Type> parse_types(std::string value) {
    std::set<Type> ret_val;
    std::vector<std::string> tokens;
    std::string token;
    std::string delimiter = " ";
    Type found;
    size_t pos = 0;
    while ((pos = value.find(delimiter)) != std::string::npos) {
        token = value.substr(0, pos);
        tokens.push_back(token);
        value.erase(0, pos + delimiter.length());
    }
    for (auto &&i : tokens) {
        found.name = i;
        // subtypes before types as bandaid for weird types in my list due to... unset cards?
        if (all_subtypes.find(i) != all_subtypes.end()) {
            found.kind = SUBTYPE;
            goto EMPLACE;
        }
        if (all_types.find(i) != all_types.end()) {
            found.kind = TYPE;
            goto EMPLACE;
        }
        if (all_supertypes.find(i) != all_supertypes.end()) {
            found.kind = SUPERTYPE;
            goto EMPLACE;
        }
        non_fatal_error("UNRECOGNIZED TYPE TOKEN: " + i + " registering as subtype");
        found.kind = SUBTYPE;
    EMPLACE:
        ret_val.emplace(found);
    }
    return ret_val;
}

static uint32_t parse_power(std::string value) {
    if (value == "") return 0;
    auto slash_pos = value.find("/");
    std::string pow_string = value.substr(0, slash_pos);
    return std::stoi(pow_string);
}

static uint32_t parse_toughness(std::string value) {
    if (value == "") return 0;
    auto slash_pos = value.find("/");
    std::string tough_string = value.substr(slash_pos + 1);
    return std::stoi(tough_string);
}

// fed each ability line
static std::set<Entity> parse_abilities(std::vector<std::string> lines, const std::set<Type>& types) {
    size_t pos = 0;
    std::set<Entity> ret_val;
    for (auto &&line : lines) {
        pos = 0;
        // TODO: ONLY CAN DEAL WITH SPELL ABILITY RN
        Ability ability;
        pos = line.find("SP$", pos);
        if (pos == std::string::npos) continue;
        // confirmed spell ability, future switch goes here
        ability.ability_type = Ability::AbilityType::SPELL;
        pos += 4;
        // Check if we're past the end of the string
        if (pos >= line.length()) continue;

        // Extract category (before first space or pipe)
        size_t category_end = line.find_first_of(" |", pos);
        if (category_end == std::string::npos) category_end = line.length();

        // Check if category_end is valid
        if (category_end <= pos) continue;

        std::string category = line.substr(pos, category_end - pos);

        if (category == "DealDamage") {
            ability.category = category;

            // Parse pipe-delimited parameters
            size_t param_pos = line.find("|", pos);
            while (param_pos != std::string::npos) {
                if(param_pos >= line.size()) break;
                param_pos++;  // Skip '|'

                // Skip whitespace
                while (param_pos < line.length() && line[param_pos] == ' ') param_pos++;

                // Find end of this parameter (next '|' or end of line)
                size_t param_end = line.find("|", param_pos);
                if (param_end == std::string::npos) param_end = line.length();

                std::string param = line.substr(param_pos, param_end - param_pos);

                // Parse key-value pair (format: "Key$ Value")
                size_t dollar_pos = param.find("$");
                if (dollar_pos != std::string::npos) {
                    std::string key = param.substr(0, dollar_pos);
                    std::string value = param.substr(dollar_pos + 1);

                    // Trim whitespace from both
                    size_t key_start = key.find_first_not_of(" ");
                    size_t key_end = key.find_last_not_of(" ");
                    if (key_start != std::string::npos) {
                        key = key.substr(key_start, key_end - key_start + 1);
                    }

                    size_t value_start = value.find_first_not_of(" ");
                    size_t value_end = value.find_last_not_of(" ");
                    if (value_start != std::string::npos) {
                        value = value.substr(value_start, value_end - value_start + 1);
                    }

                    // Extract relevant parameters; run only num dmg is found
                    if (key == "NumDmg") {
                        ability.amount = static_cast<size_t>(std::stoi(value));
                    }
                    // ValidTgts handled during targeting, not during parse
                }

                param_pos = param_end;
            }

            auto id = global_coordinator.CreateEntity();
            ability.source = id;
            global_coordinator.AddComponent(id, ability);
            ret_val.emplace(id);
        }
    }

    // basic lands having mana abilities is by virtue of their type
    // Check if this is a basic land and add mana ability
    bool is_basic = false;
    bool is_land = false;
    std::string land_subtype;

    for (auto& type : types) {
        if (type.kind == SUPERTYPE && type.name == "Basic") {
            is_basic = true;
        }
        if (type.kind == TYPE && type.name == "Land") {
            is_land = true;
        }
        if (type.kind == SUBTYPE && (type.name == "Mountain" || type.name == "Forest" ||
                                     type.name == "Plains" || type.name == "Island" ||
                                     type.name == "Swamp")) {
            land_subtype = type.name;
        }
    }

    if (is_basic && is_land && !land_subtype.empty()) {
        Ability mana_ability;
        mana_ability.ability_type = Ability::ACTIVATED;
        mana_ability.category = "AddMana";

        // Determine mana color from subtype (using amount field to store color)
        if (land_subtype == "Mountain") {
            mana_ability.amount = static_cast<size_t>(RED);
        } else if (land_subtype == "Forest") {
            mana_ability.amount = static_cast<size_t>(GREEN);
        } else if (land_subtype == "Plains") {
            mana_ability.amount = static_cast<size_t>(WHITE);
        } else if (land_subtype == "Island") {
            mana_ability.amount = static_cast<size_t>(BLUE);
        } else if (land_subtype == "Swamp") {
            mana_ability.amount = static_cast<size_t>(BLACK);
        }

        auto ability_id = global_coordinator.CreateEntity();
        mana_ability.source = ability_id;
        global_coordinator.AddComponent(ability_id, mana_ability);
        ret_val.emplace(ability_id);
    }

    return ret_val;
}