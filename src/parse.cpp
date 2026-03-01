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
#include "components/static_ability.h"
#include "ecs/coordinator.h"
#include "ecs/events.h"
#include "error.h"
#include "type_constants.h"

const size_t SCRIPT_MAX_LEN = 10000;

static std::string value_from_script(std::string script, std::string key);
static std::vector<std::string> multi_values_from_script(std::string script, std::string key);
static std::multiset<Colors> parse_mana_cost(std::string value);
static std::set<Type> parse_types(std::string value);
static std::map<std::string, std::string> parse_svars(const std::string& script);
static std::string normalize_category(std::string category);
static void apply_param_to_ability(Ability& ability, const std::string& key, const std::string& value);
static std::vector<Ability> parse_abilities(std::vector<std::string> lines, const std::set<Type>& types,
                                            const std::map<std::string, std::string>& svars);
static std::vector<Ability> parse_triggered_abilities(const std::string& script,
                                                      const std::map<std::string, std::string>& svars);
static std::vector<StaticAbility> parse_static_abilities(const std::string& script);
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

    // Split at ALTERNATE marker for DFCs
    std::string front_script = script_data;
    std::string back_script;
    size_t alt_pos = script_data.find("\nALTERNATE");
    if (alt_pos != std::string::npos) {
        front_script = script_data.substr(0, alt_pos);
        size_t back_start = alt_pos + 1;  // skip initial '\n'
        back_start = script_data.find('\n', back_start);  // skip "ALTERNATE" line
        if (back_start != std::string::npos) {
            back_start++;
            back_script = script_data.substr(back_start);
        }
    }

    CardData card;
    card.name = value_from_script(front_script, "Name");
    card.uid = name_to_uid(card.name);
    card.mana_cost = parse_mana_cost(value_from_script(front_script, "ManaCost"));
    card.types = parse_types(value_from_script(front_script, "Types"));
    // TODO optimize
    card.power = parse_power(value_from_script(front_script, "PT"));
    card.toughness = parse_toughness(value_from_script(front_script, "PT"));
    // parse ability templates; entities are only created when abilities go on the stack
    auto svars = parse_svars(front_script);
    card.abilities = parse_abilities(multi_values_from_script(front_script, "A"), card.types, svars);
    // parse triggered abilities from T: lines
    for (auto &trig : parse_triggered_abilities(front_script, svars))
        card.abilities.push_back(trig);

    // Parse S: lines for alternate costs
    for (auto& line : multi_values_from_script(front_script, "S")) {
        if (line.find("AlternativeCost") == std::string::npos) continue;
        size_t cost_pos = line.find("Cost$");
        if (cost_pos == std::string::npos) continue;
        cost_pos += 5;
        while (cost_pos < line.size() && line[cost_pos] == ' ') cost_pos++;
        size_t cost_end = line.find('|', cost_pos);
        if (cost_end == std::string::npos) cost_end = line.size();
        std::string cost_str = line.substr(cost_pos, cost_end - cost_pos);
        while (!cost_str.empty() && cost_str.back() == ' ') cost_str.pop_back();
        AltCost ac;
        ac.has_alt_cost = true;
        size_t pl = cost_str.find("PayLife<");
        if (pl != std::string::npos) {
            size_t close = cost_str.find('>', pl);
            ac.life_cost = std::stoi(cost_str.substr(pl + 8, close - pl - 8));
        }
        size_t ef = cost_str.find("ExileFromHand<");
        if (ef != std::string::npos) {
            size_t slash = cost_str.find('/', ef);
            ac.exile_blue_from_hand = std::stoi(cost_str.substr(ef + 14, slash - ef - 14));
        }
        size_t rf = cost_str.find("Return<");
        if (rf != std::string::npos) {
            size_t slash = cost_str.find('/', rf);
            size_t close = cost_str.find('>', rf);
            ac.return_to_hand_count = std::stoi(cost_str.substr(rf + 7, slash - rf - 7));
            ac.return_to_hand_subtype = cost_str.substr(slash + 1, close - slash - 1);
        }
        card.alt_cost = ac;
        break;
    }

    // Parse S: lines for static abilities (Continuous, MustAttack, etc.)
    card.static_abilities = parse_static_abilities(front_script);

    // Parse K: keyword lines
    for (auto& kw_line : multi_values_from_script(front_script, "K")) {
        size_t pos = 0;
        while (pos < kw_line.size()) {
            size_t comma = kw_line.find(',', pos);
            if (comma == std::string::npos) comma = kw_line.size();
            std::string kw = kw_line.substr(pos, comma - pos);
            size_t s = kw.find_first_not_of(" ");
            size_t e = kw.find_last_not_of(" ");
            if (s != std::string::npos)
                card.keywords.push_back(kw.substr(s, e - s + 1));
            pos = (comma < kw_line.size()) ? comma + 1 : comma;
        }
    }

    // Parse backside for DFCs
    if (!back_script.empty()) {
        auto backside = std::make_shared<CardData>();
        backside->name = value_from_script(back_script, "Name");
        backside->uid  = name_to_uid(backside->name);
        backside->types = parse_types(value_from_script(back_script, "Types"));
        backside->power     = parse_power(value_from_script(back_script, "PT"));
        backside->toughness = parse_toughness(value_from_script(back_script, "PT"));
        for (auto& kw_line : multi_values_from_script(back_script, "K")) {
            size_t pos = 0;
            while (pos < kw_line.size()) {
                size_t comma = kw_line.find(',', pos);
                if (comma == std::string::npos) comma = kw_line.size();
                std::string kw = kw_line.substr(pos, comma - pos);
                size_t s = kw.find_first_not_of(" "), e = kw.find_last_not_of(" ");
                if (s != std::string::npos) backside->keywords.push_back(kw.substr(s, e - s + 1));
                pos = (comma < kw_line.size()) ? comma + 1 : comma;
            }
        }
        card.backside = backside;
    }

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
        std::string line = script.substr(pos, (end_pos - pos));  // end_pos is at \n, so no -1 needed
        if (!line.empty() && line.back() == '\r') line.pop_back();  // strip \r for Windows line endings
        ret_val.push_back(line);
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
    } else if (key == "MayShuffle") {
        ability.may_shuffle = (value == "True");
    } else if (key == "UnlessCost") {
        ability.unless_generic_cost = static_cast<size_t>(std::stoi(value));
    } else if (key == "LifeAmount" || key == "Amount") {
        ability.amount = static_cast<size_t>(std::stoi(value));
    } else if (key == "TargetType") {
        if (value == "Spell") ability.target_type = "Spell";
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

// Normalizes script category names to the internal names used throughout the engine.
static std::string normalize_category(std::string category) {
    if (category == "Mana") category = "AddMana";
    return category;
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
        sub.category = normalize_category(content.substr(p, cat_end - p));

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

        ability.category = normalize_category(line.substr(pos, category_end - pos));

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

// Finds all lines that start with "T:" (trigger lines) in the card script.
static std::vector<std::string> find_trigger_lines(const std::string &script) {
    std::vector<std::string> result;
    size_t pos = 0;
    // Check if the script itself starts with "T:"
    if (script.size() >= 2 && script[0] == 'T' && script[1] == ':') {
        size_t end = script.find('\n', 0);
        if (end == std::string::npos) end = script.size();
        std::string line = script.substr(2, end - 2);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        result.push_back(line);
        pos = end;
    }
    while ((pos = script.find("\nT:", pos)) != std::string::npos) {
        pos += 3;  // skip "\nT:"
        size_t end = script.find('\n', pos);
        if (end == std::string::npos) end = script.size();
        std::string line = script.substr(pos, end - pos);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        result.push_back(line);
        pos = end;
    }
    return result;
}

// Parses a single T: trigger line and its Execute$ SVar into a triggered Ability.
// Returns a default Ability with trigger_on == 0 if the trigger is unrecognised.
static Ability parse_one_trigger(const std::string &line, const std::map<std::string, std::string> &svars) {
    Ability ability;
    ability.ability_type = Ability::TRIGGERED;

    std::string execute_svar;
    bool mode_changes_zone = false;
    bool dest_is_battlefield = false;
    bool valid_card_creature = false;
    bool mode_is_phase = false;
    bool phase_is_upkeep = false;
    bool valid_player_is_you = false;
    bool mode_is_spell_cast = false;
    bool valid_card_non_creature = false;

    // Walk pipe-delimited params
    size_t param_pos = 0;
    while (param_pos <= line.size()) {
        size_t param_end = line.find('|', param_pos);
        if (param_end == std::string::npos) param_end = line.size();
        std::string param = line.substr(param_pos, param_end - param_pos);

        // trim whitespace
        size_t ks = param.find_first_not_of(" ");
        size_t ke = param.find_last_not_of(" ");
        if (ks != std::string::npos) param = param.substr(ks, ke - ks + 1);

        size_t dollar = param.find('$');
        if (dollar != std::string::npos) {
            std::string key = param.substr(0, dollar);
            std::string value = param.substr(dollar + 1);
            size_t vs = value.find_first_not_of(" ");
            size_t ve = value.find_last_not_of(" ");
            if (vs != std::string::npos) value = value.substr(vs, ve - vs + 1);
            size_t ks2 = key.find_first_not_of(" ");
            size_t ke2 = key.find_last_not_of(" ");
            if (ks2 != std::string::npos) key = key.substr(ks2, ke2 - ks2 + 1);

            if (key == "Mode") {
                if (value == "ChangesZone") mode_changes_zone = true;
                else if (value == "Phase") mode_is_phase = true;
                else if (value == "SpellCast") mode_is_spell_cast = true;
            } else if (key == "Phase") {
                if (value == "Upkeep") phase_is_upkeep = true;
            } else if (key == "ValidPlayer" || key == "ValidActivatingPlayer") {
                if (value == "You") valid_player_is_you = true;
            } else if (key == "Destination") {
                if (value == "Battlefield") dest_is_battlefield = true;
            } else if (key == "ValidCard") {
                if (value.find("Creature") != std::string::npos) valid_card_creature = true;
                if (value.find("nonCreature") != std::string::npos) valid_card_non_creature = true;
                if (value.find(".Other") != std::string::npos) ability.trigger_self_excluded = true;
            } else if (key == "Execute") {
                execute_svar = value;
            }
        }

        if (param_end >= line.size()) break;
        param_pos = param_end + 1;
    }

    // Map trigger condition to event ID.
    // TODO: Issue generic zone-change events matching the T: line template fields
    //   (Mode$ ChangesZone, Origin$, Destination$, ValidCard$, ValidTgts$) so that
    //   triggers beyond creature ETB can be wired up without hardcoding each case.
    //   Each distinct (Origin, Destination, ValidCard) combination would correspond
    //   to a unique EventId, and SendEvent would carry the entering/leaving entity
    //   so that per-trigger ValidCard filters (type, controller, .Other, etc.) can
    //   be evaluated at check time rather than baked into the event ID.
    if (mode_changes_zone && dest_is_battlefield && valid_card_creature) {
        ability.trigger_on = Events::CREATURE_ENTERED;
    }

    if (mode_is_phase && phase_is_upkeep) {
        ability.trigger_on = Events::UPKEEP_BEGAN;
        ability.trigger_valid_player_is_controller = valid_player_is_you;
    }

    if (mode_is_spell_cast && valid_card_non_creature) {
        ability.trigger_on = Events::NONCREATURE_SPELL_CAST;
        ability.trigger_valid_player_is_controller = valid_player_is_you;
    }

    // Resolve effect from Execute$ SVar
    if (!execute_svar.empty()) {
        auto it = svars.find(execute_svar);
        if (it != svars.end()) {
            Ability effect = parse_svar_ability(it->second, Ability::TRIGGERED);
            ability.category = effect.category;
            ability.amount = effect.amount;
            ability.subabilities = effect.subabilities;
        }
    }

    return ability;
}

static std::vector<Ability> parse_triggered_abilities(const std::string &script,
                                                      const std::map<std::string, std::string> &svars) {
    std::vector<Ability> result;
    for (const auto &line : find_trigger_lines(script)) {
        Ability ab = parse_one_trigger(line, svars);
        if (ab.trigger_on != 0)  // only keep recognised triggers
            result.push_back(ab);
    }
    return result;
}

static std::vector<StaticAbility> parse_static_abilities(const std::string &script) {
    std::vector<StaticAbility> result;
    for (const auto &line : multi_values_from_script(script, "S")) {
        // Skip alt cost lines (handled separately) and garbage matches
        if (line.find("AlternativeCost") != std::string::npos) continue;
        if (line.find("Mode$") == std::string::npos) continue;

        StaticAbility sa;
        size_t param_pos = 0;
        while (param_pos <= line.size()) {
            size_t param_end = line.find('|', param_pos);
            if (param_end == std::string::npos) param_end = line.size();
            std::string param = line.substr(param_pos, param_end - param_pos);

            // trim whitespace
            size_t ks = param.find_first_not_of(" ");
            size_t ke = param.find_last_not_of(" ");
            if (ks != std::string::npos) param = param.substr(ks, ke - ks + 1);

            size_t dollar = param.find('$');
            if (dollar != std::string::npos) {
                std::string key = param.substr(0, dollar);
                std::string value = param.substr(dollar + 1);
                size_t vs = value.find_first_not_of(" ");
                size_t ve = value.find_last_not_of(" ");
                if (vs != std::string::npos) value = value.substr(vs, ve - vs + 1);
                size_t ks2 = key.find_first_not_of(" ");
                size_t ke2 = key.find_last_not_of(" ");
                if (ks2 != std::string::npos) key = key.substr(ks2, ke2 - ks2 + 1);

                if (key == "Mode") {
                    sa.category = value;
                } else if (key == "Condition") {
                    sa.condition = value;
                } else if (key == "AddPower") {
                    sa.add_power = std::stoi(value);
                } else if (key == "AddToughness") {
                    sa.add_toughness = std::stoi(value);
                } else if (key == "AddKeyword") {
                    sa.add_keyword = value;
                }
            }

            if (param_end >= line.size()) break;
            param_pos = param_end + 1;
        }

        if (!sa.category.empty()) result.push_back(sa);
    }
    return result;
}