#include "parse.h"

#include <cctype>
#include <fstream>
#include <map>
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
static std::map<std::string, std::string> parse_svars(const std::string& script);
static void apply_param_to_ability(Ability& ability, const std::string& key, const std::string& value);
static std::vector<Ability> parse_abilities(std::vector<std::string> lines, const std::set<Type>& types,
                                            const std::map<std::string, std::string>& svars);
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
    // parse ability templates; entities are only created when abilities go on the stack
    auto svars = parse_svars(script_data);
    card.abilities = parse_abilities(multi_values_from_script(script_data, "A"), card.types, svars);

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
    if (!value.empty()) tokens.push_back(value);
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

// Extracts all SVar:name:content entries from a card script into a name→content map.
static std::map<std::string, std::string> parse_svars(const std::string& script) {
    std::map<std::string, std::string> svars;
    const std::string prefix = "SVar:";
    size_t pos = 0;
    while ((pos = script.find(prefix, pos)) != std::string::npos) {
        pos += prefix.size();
        size_t colon = script.find(':', pos);
        if (colon == std::string::npos) break;
        std::string name = script.substr(pos, colon - pos);
        pos = colon + 1;
        size_t end = script.find('\n', pos);
        if (end == std::string::npos) end = script.size();
        std::string value = script.substr(pos, end - pos);
        while (!value.empty() && (value.back() == '\r' || value.back() == ' '))
            value.pop_back();
        svars[name] = value;
        pos = end;
    }
    return svars;
}

// Applies a single key/value parameter to an ability struct.
static void apply_param_to_ability(Ability& ability, const std::string& key, const std::string& value) {
    if (key == "NumDmg") {
        ability.amount = static_cast<size_t>(std::stoi(value));
    } else if (key == "NumCards" || key == "ChangeNum") {
        ability.amount = static_cast<size_t>(std::stoi(value));
    } else if (key == "Produced") {
        for (char c : value) {
            if      (c == 'W') { ability.color = WHITE;     break; }
            else if (c == 'U') { ability.color = BLUE;      break; }
            else if (c == 'B') { ability.color = BLACK;     break; }
            else if (c == 'R') { ability.color = RED;       break; }
            else if (c == 'G') { ability.color = GREEN;     break; }
            else if (c == 'C') { ability.color = COLORLESS; break; }
        }
        ability.amount = 1;
    } else if (key == "ValidTgts") {
        ability.valid_tgts = value;
    } else if (key == "ChangeType") {
        ability.change_type = value;
    } else if (key == "Origin") {
        if (value == "Library")        ability.origin = Zone::LIBRARY;
        else if (value == "Hand")      ability.origin = Zone::HAND;
        else if (value == "Graveyard") ability.origin = Zone::GRAVEYARD;
        else if (value == "Exile")     ability.origin = Zone::EXILE;
    } else if (key == "Destination") {
        if (value == "Battlefield")    ability.destination = Zone::BATTLEFIELD;
        else if (value == "Library")   ability.destination = Zone::LIBRARY;
        else if (value == "Hand")      ability.destination = Zone::HAND;
        else if (value == "Graveyard") ability.destination = Zone::GRAVEYARD;
        else if (value == "Exile")     ability.destination = Zone::EXILE;
    } else if (key == "Mandatory") {
        ability.mandatory = (value == "True");
    } else if (key == "Cost") {
        size_t tok_pos = 0;
        while (tok_pos < value.size()) {
            size_t tok_end = value.find(' ', tok_pos);
            if (tok_end == std::string::npos) tok_end = value.size();
            std::string tok = value.substr(tok_pos, tok_end - tok_pos);
            if (tok == "T") {
                ability.tap_cost = true;
            } else if (tok.rfind("PayLife<", 0) == 0) {
                size_t angle = tok.find('<');
                size_t close = tok.find('>');
                if (angle != std::string::npos && close != std::string::npos && close > angle + 1)
                    ability.life_cost = std::stoi(tok.substr(angle + 1, close - angle - 1));
            } else if (tok.rfind("Sac<", 0) == 0 && tok.find("CARDNAME") != std::string::npos) {
                ability.sac_self = true;
            }
            tok_pos = (tok_end < value.size()) ? tok_end + 1 : tok_end;
        }
    }
}

// Parses a SVar's DB$ content string into an Ability. The ability_type is inherited from the parent.
static Ability parse_svar_ability(const std::string& content, Ability::AbilityType ability_type) {
    Ability sub;
    sub.ability_type = ability_type;
    size_t db_pos = content.find("DB$");
    if (db_pos == std::string::npos) return sub;
    size_t p = db_pos + 4;  // skip "DB$ "
    size_t cat_end = content.find_first_of(" |", p);
    if (cat_end == std::string::npos) cat_end = content.length();
    if (cat_end > p)
        sub.category = content.substr(p, cat_end - p);

    size_t param_pos = content.find("|", p);
    while (param_pos != std::string::npos) {
        if (param_pos >= content.size()) break;
        param_pos++;
        while (param_pos < content.length() && content[param_pos] == ' ') param_pos++;
        size_t param_end = content.find("|", param_pos);
        if (param_end == std::string::npos) param_end = content.length();
        std::string param = content.substr(param_pos, param_end - param_pos);

        size_t dollar_pos = param.find("$");
        if (dollar_pos != std::string::npos) {
            std::string key = param.substr(0, dollar_pos);
            std::string value = param.substr(dollar_pos + 1);
            size_t ks = key.find_first_not_of(" "), ke = key.find_last_not_of(" ");
            if (ks != std::string::npos) key = key.substr(ks, ke - ks + 1);
            size_t vs = value.find_first_not_of(" "), ve = value.find_last_not_of(" ");
            if (vs != std::string::npos) value = value.substr(vs, ve - vs + 1);
            apply_param_to_ability(sub, key, value);
        }
        param_pos = param_end;
    }
    return sub;
}

// fed each ability line
static std::vector<Ability> parse_abilities(std::vector<std::string> lines, const std::set<Type>& types,
                                            const std::map<std::string, std::string>& svars) {
    size_t pos = 0;
    std::vector<Ability> ret_val;
    for (auto &&line : lines) {
        pos = 0;
        Ability ability;
        size_t sp_pos = line.find("SP$");
        size_t ab_pos = line.find("AB$");
        bool is_sp = (sp_pos != std::string::npos);
        bool is_ab = (ab_pos != std::string::npos);
        if (!is_sp && !is_ab) continue;
        if (is_sp && (!is_ab || sp_pos < ab_pos)) {
            ability.ability_type = Ability::AbilityType::SPELL;
            pos = sp_pos + 4;  // skip "SP$ "
        } else {
            ability.ability_type = Ability::AbilityType::ACTIVATED;
            pos = ab_pos + 4;  // skip "AB$ "
        }
        // Check if we're past the end of the string
        if (pos >= line.length()) continue;

        // Extract category (before first space or pipe)
        size_t category_end = line.find_first_of(" |", pos);
        if (category_end == std::string::npos) category_end = line.length();

        // Check if category_end is valid
        if (category_end <= pos) continue;

        std::string category = line.substr(pos, category_end - pos);
        // Normalize script category names to internal names used throughout the engine
        if (category == "Mana") category = "AddMana";
        ability.category = category;

        // Parse pipe-delimited parameters — applies to all ability categories
        size_t param_pos = line.find("|", pos);
        while (param_pos != std::string::npos) {
            if (param_pos >= line.size()) break;
            param_pos++;  // Skip '|'

            while (param_pos < line.length() && line[param_pos] == ' ') param_pos++;

            size_t param_end = line.find("|", param_pos);
            if (param_end == std::string::npos) param_end = line.length();

            std::string param = line.substr(param_pos, param_end - param_pos);

            size_t dollar_pos = param.find("$");
            if (dollar_pos != std::string::npos) {
                std::string key = param.substr(0, dollar_pos);
                std::string value = param.substr(dollar_pos + 1);

                size_t key_start = key.find_first_not_of(" ");
                size_t key_end = key.find_last_not_of(" ");
                if (key_start != std::string::npos)
                    key = key.substr(key_start, key_end - key_start + 1);

                size_t value_start = value.find_first_not_of(" ");
                size_t value_end = value.find_last_not_of(" ");
                if (value_start != std::string::npos)
                    value = value.substr(value_start, value_end - value_start + 1);

                if (key == "SubAbility") {
                    auto it = svars.find(value);
                    if (it != svars.end())
                        ability.subabilities.push_back(parse_svar_ability(it->second, ability.ability_type));
                } else {
                    apply_param_to_ability(ability, key, value);
                }
            }

            param_pos = param_end;
        }
        ret_val.push_back(ability);
    }

    return ret_val;
}