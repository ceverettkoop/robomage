#include "parse.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <cassert>
#include <cstring>

#include "classes/colors.h"
#include "components/types.h"
#include "components/ability.h"
#include "components/carddata.h"
#include "components/effect.h"
#include "components/token.h"
#include "components/static_ability.h"
#include "effects/effects.h"
#include "ecs/coordinator.h"
#include "ecs/events.h"
#include "error.h"
#include "str_util.h"
#include "type_constants.h"

extern std::string RESOURCE_DIR;

const size_t SCRIPT_MAX_LEN = 10000;

static std::string value_from_script(std::string script, std::string key);
static std::vector<std::string> multi_values_from_script(std::string script, std::string key);
static std::multiset<Colors> parse_mana_cost(std::string value, std::vector<Colors> *phyrexian_out = nullptr,
                                             std::vector<HybridPip> *hybrid_out = nullptr);
static void parse_alt_cost_tokens(const std::string& cost_str, AltCost& ac);
static std::set<Type> parse_types(std::string value);
static std::set<Colors> parse_colors_field(const std::string &colors_field);
static std::map<std::string, std::string> parse_svars(const std::string& script);
static std::string normalize_category(std::string category);
static void apply_param_to_ability(Ability& ability, const std::string& key, const std::string& value,
                                   const std::string& card_name = "");
static std::vector<Ability> parse_abilities(std::vector<std::string> lines, const std::set<Type>& types,
                                            const std::map<std::string, std::string>& svars,
                                            const std::string& card_name = "");
static std::vector<Ability> parse_triggered_abilities(const std::string& script,
                                                      const std::map<std::string, std::string>& svars,
                                                      const std::string& card_name = "");
static std::vector<StaticAbility> parse_static_abilities(const std::string& script, const std::map<std::string, std::string>& svars);
static StaticAbility parse_one_static_ability(const std::string& line, const std::map<std::string, std::string>& svars);
// Parses one T:/trigger SVar line into a TRIGGERED Ability (trigger_on == 0 if unrecognised).
// Forward-declared so parse_svar_ability can build a DB$ Effect | Triggers$ <SVar> floating
// triggered ability from the named trigger SVar.
static Ability parse_one_trigger(const std::string& line, const std::map<std::string, std::string>& svars,
                                 const std::string& card_name);
static std::vector<Effect::Replacement> parse_replacement_effects(const std::string& script,
                                                                   const std::map<std::string, std::string>& svars);
static uint32_t parse_power(std::string value);
static uint32_t parse_toughness(std::string value);
static std::vector<std::string> find_trigger_lines(const std::string &script);
static Ability parse_one_trigger(const std::string &line, const std::map<std::string, std::string> &svars,
                                 const std::string& card_name = "");
static void split_keywords(const std::string& kw_line, std::vector<std::string>& out);
static bool next_param(const std::string& line, size_t& pos, std::string& key, std::string& value);
static void parse_card_face(const std::string& front_script, CardData& card);
// Forward-declared so the K: keyword pass can parse a Gift keyword's GiftAbility SVar into the
// card's gift effect (Into the Flood Maw's tapped-Fish token).
static Ability parse_svar_ability(const std::string& content, Ability::AbilityType ability_type,
                                  const std::map<std::string, std::string>& svars,
                                  const std::string& card_name);

// Split a comma-separated K: keyword list into trimmed keywords appended to out.
static void split_keywords(const std::string& kw_line, std::vector<std::string>& out) {
    size_t pos = 0;
    while (pos < kw_line.size()) {
        size_t comma = kw_line.find(',', pos);
        if (comma == std::string::npos) comma = kw_line.size();
        std::string kw = kw_line.substr(pos, comma - pos);
        size_t s = kw.find_first_not_of(" ");
        size_t e = kw.find_last_not_of(" ");
        if (s != std::string::npos)
            out.push_back(kw.substr(s, e - s + 1));
        pos = (comma < kw_line.size()) ? comma + 1 : comma;
    }
}

// Walk a '|'-delimited Forge parameter line ("Key$Value | Key$Value | ...").
// Starting at *pos*, find the next "Key$Value" chunk, split it on the first '$',
// trim surrounding spaces from key and value, and advance *pos* past the chunk.
// Chunks lacking a '$' are skipped. A leading '|' at *pos* is consumed, so the
// helper works whether the walk starts at the first param (pos == 0) or at a '|'
// separator following a prefix (e.g. an ability category). Returns false once
// the line is exhausted.
static bool next_param(const std::string& line, size_t& pos, std::string& key, std::string& value) {
    while (pos < line.size()) {
        if (line[pos] == '|') pos++;
        while (pos < line.size() && line[pos] == ' ') pos++;
        size_t end = line.find('|', pos);
        if (end == std::string::npos) end = line.size();
        std::string param = line.substr(pos, end - pos);
        pos = end;
        size_t dollar = param.find('$');
        if (dollar == std::string::npos) continue;
        key = param.substr(0, dollar);
        value = param.substr(dollar + 1);
        size_t ks = key.find_first_not_of(" "), ke = key.find_last_not_of(" ");
        if (ks != std::string::npos) key = key.substr(ks, ke - ks + 1);
        size_t vs = value.find_first_not_of(" "), ve = value.find_last_not_of(" ");
        if (vs != std::string::npos) value = value.substr(vs, ve - vs + 1);
        return true;
    }
    return false;
}

// Parse a Ward cost argument (the text after "Ward:") into its amount and payment kind
// (CR 702.21). "PayLife<N>" is a life payment; a plain numeric arg is {N} generic mana.
// Parse the amount defensively: with -fno-exceptions a std::stoi on a missing '>' or
// non-numeric body would abort, so validate digits first and degrade to a {1} mana ward
// on a malformed or missing arg rather than crashing card load.
void parse_ward_cost(const std::string &arg, int &cost, bool &is_life) {
    cost = 1;
    is_life = false;
    if (arg.rfind("PayLife<", 0) == 0) {
        size_t close = arg.find('>');
        std::string n = (close != std::string::npos && close > 8) ? arg.substr(8, close - 8)
                                                                  : std::string();
        if (!n.empty() && n.find_first_not_of("0123456789") == std::string::npos) {
            cost = std::stoi(n);
            is_life = true;
        }
    } else if (!arg.empty() && arg.find_first_not_of("0123456789") == std::string::npos) {
        cost = std::stoi(arg);
    }
}

// all to lowercase, spaces to underscores, other characters removed
std::string name_to_uid(std::string name) {
    std::vector<size_t> to_rm;

    for (size_t i = 0; i < name.size(); i++) {
        char value = name[i];
        if (std::isalpha(value)) {
            name[i] = std::tolower(value);
        } else if( ((value == '-') || (value == ' ')) && (i != name.size() - 1) )   { // we will excise up to 1 trailing space, rest to underscores
                name[i] = '_';
        } else {
            to_rm.push_back(i);
        }
    }
    for (size_t i = 0; i < to_rm.size(); i++) {
        auto index_to_rm = to_rm[i] - i;
        name.erase(index_to_rm, 1);
    }

    return name;
}

// Parses the cost portion of an alternate cost string (the value after Cost$ / after
// "Evoke:") into an AltCost. Recognises "0" (free), PayLife<N>, ExileFromHand<n/filter>
// (pitch), Return<n/Type>; anything else is treated as a mana cost (e.g. Evoke:R).
// Condition fields (CheckSVar etc.) are parsed separately by the caller.
static void parse_alt_cost_tokens(const std::string& cost_str, AltCost& ac) {
    ac.has_alt_cost = true;
    bool matched_special = false;
    // Cost$ 0 means free
    if (cost_str == "0") {
        ac.is_free = true;
        return;
    }
    size_t pl = cost_str.find("PayLife<");
    if (pl != std::string::npos) {
        size_t close = cost_str.find('>', pl);
        ac.life_cost = std::stoi(cost_str.substr(pl + 8, close - pl - 8));
        matched_special = true;
    }
    size_t pe = cost_str.find("PayEnergy<");
    if (pe != std::string::npos) {
        size_t close = cost_str.find('>', pe);
        ac.energy_cost = std::stoi(cost_str.substr(pe + 10, close - pe - 10));
        matched_special = true;
    }
    size_t ef = cost_str.find("ExileFromHand<");
    if (ef != std::string::npos) {
        size_t slash = cost_str.find('/', ef);
        ac.exile_from_hand_count = std::stoi(cost_str.substr(ef + 14, slash - ef - 14));
        // Extract color from filter e.g. "Card.Blue+Other" → BLUE
        std::string filter = cost_str.substr(slash + 1);
        size_t close = filter.find('>');
        if (close != std::string::npos) filter = filter.substr(0, close);
        if (filter.find("Blue") != std::string::npos) ac.exile_from_hand_color = BLUE;
        else if (filter.find("Green") != std::string::npos) ac.exile_from_hand_color = GREEN;
        else if (filter.find("Red") != std::string::npos) ac.exile_from_hand_color = RED;
        else if (filter.find("White") != std::string::npos) ac.exile_from_hand_color = WHITE;
        else if (filter.find("Black") != std::string::npos) ac.exile_from_hand_color = BLACK;
        matched_special = true;
    }
    size_t rf = cost_str.find("Return<");
    if (rf != std::string::npos) {
        size_t slash = cost_str.find('/', rf);
        size_t close = cost_str.find('>', rf);
        ac.return_to_hand_count = std::stoi(cost_str.substr(rf + 7, slash - rf - 7));
        ac.return_to_hand_type = cost_str.substr(slash + 1, close - slash - 1);
        matched_special = true;
    }
    // ExileFromGrave<X/<filter>/<label>> (Escape: Nethergoyf): exile any number of other
    // cards from your graveyard, constrained so the chosen set collectively has at least N
    // distinct card types. The "N" is encoded in the filter as "withTypesGE<N>" (CR 702.139).
    size_t eg = cost_str.find("ExileFromGrave<");
    if (eg != std::string::npos) {
        size_t ge = cost_str.find("withTypesGE", eg);
        if (ge != std::string::npos) {
            size_t num_start = ge + strlen("withTypesGE");
            size_t num_end = num_start;
            while (num_end < cost_str.size() &&
                   std::isdigit(static_cast<unsigned char>(cost_str[num_end])))
                num_end++;
            if (num_end > num_start)
                ac.exile_grave_min_types = std::stoi(cost_str.substr(num_start, num_end - num_start));
        } else {
            // Literal-count form ExileFromGrave<N/<filter>/<label>> (Escape: Uro — "Exile five
            // other cards from your graveyard"): the leading token before the first '/' is the
            // fixed number of OTHER graveyard cards to exile. Only digits qualify; a non-numeric
            // leading token (e.g. Nethergoyf's "X") is handled by the withTypesGE branch above.
            size_t num_start = eg + strlen("ExileFromGrave<");
            size_t num_end = num_start;
            while (num_end < cost_str.size() &&
                   std::isdigit(static_cast<unsigned char>(cost_str[num_end])))
                num_end++;
            if (num_end > num_start)
                ac.exile_grave_count = std::stoi(cost_str.substr(num_start, num_end - num_start));
        }
        matched_special = true;
    }
    // No special cost token: the whole string is a mana cost (e.g. Evoke:R, Evoke:2 R)
    if (!matched_special && !cost_str.empty()) {
        ac.mana_cost = parse_mana_cost(cost_str);
    }
}

// Parses a space-separated activation-cost string (the value after Cost$, or after a
// Cycling:/Flashback: keyword) into the cost fields of `ability`. Recognises the tap
// symbol `T`, PayLife<N>, Sac<.../...> (CARDNAME → sacrifice self), Discard<0/Hand or
// CARDNAME>, Return<N/Type>, and bare mana symbols. Single source for the cost-token
// grammar so every cost-bearing keyword honours the same tokens as Cost$ (previously
// Cycling/Flashback open-coded partial copies that silently dropped tokens).
static void parse_activation_cost(const std::string &cost_str, Ability &ability) {
    size_t tok_pos = 0;
    while (tok_pos < cost_str.size()) {
        size_t tok_end = cost_str.find(' ', tok_pos);
        if (tok_end == std::string::npos) tok_end = cost_str.size();
        std::string tok = cost_str.substr(tok_pos, tok_end - tok_pos);
        if (tok == "T") {
            ability.tap_cost = true;
        } else if (tok.rfind("PayLife<", 0) == 0) {
            size_t angle = tok.find('<');
            size_t close = tok.find('>');
            if (angle != std::string::npos && close != std::string::npos && close > angle + 1) {
                std::string arg = tok.substr(angle + 1, close - angle - 1);
                if (!arg.empty() && std::isdigit(static_cast<unsigned char>(arg[0])))
                    ability.life_cost = std::stoi(arg);
                else
                    // PayLife<X> — a VARIABLE life cost: the life paid is X (Count$xPaid),
                    // chosen as an additional cost while casting (Toxic Deluge). Don't stoi("X").
                    ability.life_cost_is_x = true;
            }
        } else if (tok == "X") {
            // A bare {X} in an activation cost (Candelabra of Tawnos: Cost$ X T). parse_mana_cost
            // drops X (it has no fixed value); record that the activator chooses X, so the cost
            // path can prompt for it and add X generic mana. X = Count$xPaid.
            ability.activation_has_x = true;
        } else if (tok.rfind("PayEnergy<", 0) == 0) {
            // PayEnergy<N> — pay N energy ({E}) as part of the cost (CR 122.1c). Used on
            // Guide of Souls' AttackersDeclared ImmediateTrigger ("you may pay {E}{E}{E}").
            size_t angle = tok.find('<');
            size_t close = tok.find('>');
            if (angle != std::string::npos && close != std::string::npos && close > angle + 1)
                ability.energy_cost = std::stoi(tok.substr(angle + 1, close - angle - 1));
        } else if (tok.rfind("Sac<", 0) == 0) {
            // Consume additional tokens if '>' not found (label may contain spaces)
            while (tok.find('>') == std::string::npos && tok_pos < cost_str.size()) {
                tok_end = cost_str.find(' ', tok_pos);
                if (tok_end == std::string::npos) tok_end = cost_str.size();
                tok += " " + cost_str.substr(tok_pos, tok_end - tok_pos);
                tok_pos = (tok_end < cost_str.size()) ? tok_end + 1 : tok_end;
            }
            size_t slash = tok.find('/');
            size_t close = tok.find('>');
            if (slash != std::string::npos && close != std::string::npos && close > slash + 1) {
                std::string spec = tok.substr(slash + 1, close - slash - 1);
                // Remove second slash and label (e.g. "Forest;Plains/Forest or Plains" → "Forest;Plains")
                size_t spec_slash = spec.find('/');
                if (spec_slash != std::string::npos) spec = spec.substr(0, spec_slash);
                if (spec == "CARDNAME") {
                    ability.sac_self = true;
                } else {
                    ability.sac_cost_spec = spec;
                }
            }
        } else if (tok.rfind("Discard<", 0) == 0) {
            // Discard<0/Hand> — discard entire hand as activation cost (Lion's Eye Diamond)
            if (tok.find("0/Hand") != std::string::npos) {
                ability.discard_hand_cost = true;
            } else if (tok.find("CARDNAME") != std::string::npos) {
                ability.discard_self_cost = true;
            }
        } else if (tok.rfind("Return<", 0) == 0) {
            // Return<1/Forest> — bounce a land of given subtype
            size_t slash = tok.find('/');
            size_t close = tok.find('>');
            if (slash != std::string::npos && close != std::string::npos) {
                ability.return_cost_count = std::stoi(tok.substr(7, slash - 7));
                ability.return_cost_type = tok.substr(slash + 1, close - slash - 1);
            }
        } else if ((tok.rfind("AddCounter<", 0) == 0 || tok.rfind("SubCounter<", 0) == 0) &&
                   tok.find("/LOYALTY>") != std::string::npos) {
            // Loyalty ability cost (606.4): AddCounter<N/LOYALTY> adds, SubCounter<N/LOYALTY> removes.
            size_t angle = tok.find('<');
            size_t slash = tok.find('/');
            std::string amt = tok.substr(angle + 1, slash - angle - 1);
            bool is_sub = (tok[0] == 'S');
            if (!amt.empty() && std::isdigit(static_cast<unsigned char>(amt[0]))) {
                int n = std::stoi(amt);
                ability.loyalty_cost = is_sub ? -n : n;
            } else {
                // SubCounter<X/LOYALTY> — a VARIABLE loyalty cost (Chandra, Flamecaller's [-X]
                // ultimate). The amount X is chosen at activation; loyalty_cost holds only the sign.
                // Don't stoi("X") (it would throw/abort — see PayLife<X> above).
                ability.loyalty_cost_is_x = true;
                ability.loyalty_cost = is_sub ? -1 : 1;
            }
        } else {
            // Remaining tokens are mana symbols (e.g. "4", "1", "W", "2 B")
            auto mana = parse_mana_cost(tok);
            ability.activation_mana_cost.insert(mana.begin(), mana.end());
        }
        tok_pos = (tok_end < cost_str.size()) ? tok_end + 1 : tok_end;
    }
}

Entity parse_card_script(std::string path) {
    auto id = global_coordinator.CreateEntity();
    std::string script_data;
    auto stream = std::ifstream(path);
    if (!stream.is_open()) {
        fprintf(stderr, "parse_card_script: failed to open '%s'\n", path.c_str());
        assert(false);
    }
    for (size_t i = 0; true; i++) {
        if (i > SCRIPT_MAX_LEN) fatal_error("Script too long");
        char c = stream.get();
        if (stream.eof()) break;
        if (c == '\r') continue;
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
    parse_card_face(front_script, card);

    // Parse the DFC back face as a complete second face so a transformed permanent
    // (Delver -> Insectile Aberration, Ajani Pariah -> Avenger) has its own name,
    // types, P/T, loyalty, abilities, and triggers -- not just a P/T swap.
    if (!back_script.empty()) {
        auto backside = std::make_shared<CardData>();
        parse_card_face(back_script, *backside);
        card.backside = backside;
    }

    // no error handling here
    global_coordinator.AddComponent(id, card);

    return id;
}

// Parses one card face (front or DFC back) into `card`: mana cost, types, colors,
// oracle text, P/T, starting loyalty, activated/spell abilities, triggered abilities,
// alternate costs, static abilities, replacement effects, and keywords. Shared by
// both faces so each face of a DFC is a fully-functional permanent definition.
static void parse_card_face(const std::string& front_script, CardData& card) {
    card.name = value_from_script(front_script, "Name");
    card.uid = name_to_uid(card.name);
    std::string mana_cost_str = value_from_script(front_script, "ManaCost");
    card.mana_cost = parse_mana_cost(mana_cost_str, &card.phyrexian_mana, &card.hybrid_mana);
    card.has_x_cost = (mana_cost_str.find('X') != std::string::npos);
    card.types = parse_types(value_from_script(front_script, "Types"));
    // AlternateMode:Modal marks a MODAL double-faced card (MDFC, CR 712.x) — both faces are
    // playable from hand (front spell OR back face). Only the front face carries this line; the
    // back face (parsed from the ALTERNATE block) leaves it false. Distinct from a transform DFC.
    card.is_modal_dfc = (value_from_script(front_script, "AlternateMode") == "Modal");
    // Parse explicit Colors: override (e.g. Dryad Arbor which is a land/creature with green identity)
    card.explicit_colors = parse_colors_field(value_from_script(front_script, "Colors"));
    card.oracle_text = value_from_script(front_script, "Oracle");
    // Expand literal \n escape sequences to real newlines for word-wrap rendering
    for (size_t i = 0; i + 1 < card.oracle_text.size(); ++i) {
        if (card.oracle_text[i] == '\\' && card.oracle_text[i + 1] == 'n') {
            card.oracle_text.replace(i, 2, "\n");
        }
    }
    // TODO optimize
    card.power = parse_power(value_from_script(front_script, "PT"));
    card.toughness = parse_toughness(value_from_script(front_script, "PT"));
    // Loyalty: line — printed loyalty a planeswalker enters with (306.5b)
    {
        std::string loy = value_from_script(front_script, "Loyalty");
        if (!loy.empty()) card.starting_loyalty = std::stoi(loy);
    }
    // parse ability templates; entities are only created when abilities go on the stack
    auto svars = parse_svars(front_script);
    card.abilities = parse_abilities(multi_values_from_script(front_script, "A"), card.types, svars, card.name);
    // Detect "shuffle into library" pattern: SVar with DB$ ChangeZone from Stack to Library + Defined$ Parent
    // (e.g. Green Sun's Zenith) — sets a flag so stack manager moves to library instead of graveyard.
    // Strip the sub-ability since the stack manager handles it via the flag.
    for (auto &sv : svars) {
        if (sv.second.find("DB$ ChangeZone") != std::string::npos &&
            sv.second.find("Origin$ Stack") != std::string::npos &&
            sv.second.find("Destination$ Library") != std::string::npos &&
            sv.second.find("Defined$ Parent") != std::string::npos) {
            card.shuffle_into_library = true;
            // Remove the sub-ability from all spell abilities so it doesn't resolve as a ChangeZone
            for (auto &ab : card.abilities) {
                ab.subabilities.erase(
                    std::remove_if(ab.subabilities.begin(), ab.subabilities.end(),
                        [](const Ability &sub) {
                            return sub.category == "ChangeZone" &&
                                   sub.origin == Zone::STACK &&
                                   sub.destination == Zone::LIBRARY;
                        }),
                    ab.subabilities.end());
            }
            break;
        }
    }
    // parse triggered abilities from T: lines
    for (auto &trig : parse_triggered_abilities(front_script, svars, card.name))
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
        parse_alt_cost_tokens(cost_str, ac);
        // Parse CheckSVar$ and SVarCompare$ conditions
        // Walk remaining pipe-separated params for condition fields
        size_t pp = cost_end;
        std::string key, value;
        while (next_param(line, pp, key, value)) {
            if (key == "CheckSVar") {
                auto it = svars.find(value);
                ac.condition_svar = (it != svars.end()) ? it->second : value;
            } else if (key == "SVarCompare") {
                ac.condition_compare = value;
            } else if (key == "Condition" && value == "NotPlayerTurn") {
                ac.condition_not_your_turn = true;
            } else if (key == "IsPresent") {
                ac.condition_is_present = value;
            }
        }
        card.alt_cost = ac;
        break;
    }

    // Parse S: lines for static abilities (Continuous, MustAttack, etc.)
    card.static_abilities = parse_static_abilities(front_script, svars);

    // Parse R: lines for replacement effects (e.g. enters tapped)
    card.replacement_effects = parse_replacement_effects(front_script, svars);

    // Parse K: keyword lines
    for (auto& kw_line : multi_values_from_script(front_script, "K")) {
        // K:Delve
        if (kw_line == "Delve" || kw_line.rfind("Delve", 0) == 0) {
            card.has_delve = true;
            card.keywords.push_back("Delve");
            continue;
        }
        // K:Improvise — your artifacts can help cast this spell; each untapped artifact you
        // tap after activating mana abilities pays for {1} of the generic cost (CR 702.126).
        // A cast-time generic cost reduction, mirroring Delve but tapping battlefield
        // artifacts instead of exiling graveyard cards.
        if (kw_line == "Improvise" || kw_line.rfind("Improvise", 0) == 0) {
            card.has_improvise = true;
            card.keywords.push_back("Improvise");
            continue;
        }
        // K:Companion:<grouping>:<restriction>:<desc> — the Companion keyword (CR 702.139). Forge
        // encodes the deckbuilding restriction as a token in the 3rd colon field (Yorion:
        // "Companion:Special:DeckSizePlus20:..."). Store the restriction token structured so
        // setup_companions can evaluate it against the starting deck; the trailing prose is display.
        if (kw_line.rfind("Companion:", 0) == 0) {
            card.is_companion = true;
            std::vector<std::string> parts = split(kw_line, ':');
            if (parts.size() >= 3) card.companion_restriction = parts[2];
            card.keywords.push_back("Companion");
            continue;
        }
        // K:Enchant:<ValidTgts>[:<prompt>] — an Aura's enchant restriction (CR 303.4). The
        // middle field is a target filter (e.g. "Creature.YouCtrl") for the object this Aura can
        // be attached to. Stored on the card so the cast path targets a matching object and the
        // resolved Aura attaches to it (sets equipped_to). The trailing human prompt is ignored.
        if (kw_line.rfind("Enchant:", 0) == 0) {
            std::string rest = kw_line.substr(8);  // strip "Enchant:"
            size_t colon = rest.find(':');
            card.enchant_filter = (colon != std::string::npos) ? rest.substr(0, colon) : rest;
            card.keywords.push_back("Enchant");
            continue;
        }
        // K:Ward:N — "Whenever this permanent becomes the target of a spell or ability an
        // opponent controls, counter that spell or ability unless that player pays {N}."
        // (CR 702.21). Stored as the keyword + a numeric cost; the becomes-targeted trigger
        // is synthesized when a targeting spell/ability is put on the stack.
        if (kw_line.rfind("Ward", 0) == 0) {
            size_t colon = kw_line.find(':');
            // K:Ward without a cost arg defaults to a {1} mana ward inside parse_ward_cost.
            std::string ward_arg = (colon != std::string::npos) ? kw_line.substr(colon + 1)
                                                                : std::string();
            parse_ward_cost(ward_arg, card.ward_cost, card.ward_is_life);
            card.keywords.push_back("Ward");
            continue;
        }
        // K:Affinity:Artifact — this spell costs {1} less to cast for each artifact you
        // control (CR 702.41). A generic cost reduction applied at cast time in
        // effective_base_cost(); only the artifact variant is supported.
        if (kw_line.rfind("Affinity", 0) == 0) {
            if (kw_line.find("Artifact") != std::string::npos) card.affinity_artifact = true;
            card.keywords.push_back("Affinity");
            continue;
        }
        // K:ETBReplacement:Other:ChooseCT — choose creature type on ETB (Cavern of Souls)
        if (kw_line.find("ETBReplacement") != std::string::npos &&
            kw_line.find("ChooseCT") != std::string::npos) {
            card.has_etb_choose_creature_type = true;
            continue;
        }
        // K:ETBReplacement:Other:DBNameCard — choose a card name on ETB (Disruptor Flute)
        if (kw_line.find("ETBReplacement") != std::string::npos &&
            kw_line.find("NameCard") != std::string::npos) {
            card.has_etb_name_card = true;
            continue;
        }
        // K:etbCounter:P1P1:X:... — "this card enters with counters"
        // Parsed as a static ability; counters applied in apply_permanent_components on ETB.
        if (kw_line.rfind("etbCounter", 0) == 0) {
            // K:etbCounter:<TYPE>:<count>  where <count> is either a literal number or a
            // SVar key resolving to a Count$ expression (e.g. Chalice's "X" → Count$xPaid).
            std::string sub = kw_line.substr(strlen("etbCounter"));
            std::string counter_type_str = "P1P1";
            bool from_delve = false;
            bool from_xpaid = false;
            std::string delve_filter = "";
            int literal_count = 0;
            if (!sub.empty() && sub[0] == ':') {
                size_t c1 = sub.find(':', 1);
                if (c1 != std::string::npos) {
                    counter_type_str = sub.substr(1, c1 - 1);
                    size_t c2 = sub.find(':', c1 + 1);
                    std::string count_tok = (c2 != std::string::npos)
                        ? sub.substr(c1 + 1, c2 - c1 - 1)
                        : sub.substr(c1 + 1);
                    // The count is either a literal number (etbCounter:M1M1:6 → 6) or an SVar
                    // key resolving to a Count$ expression (delve / X paid at cast).
                    if (!count_tok.empty() &&
                        std::all_of(count_tok.begin(), count_tok.end(),
                                    [](unsigned char ch) { return std::isdigit(ch); })) {
                        literal_count = std::stoi(count_tok);
                    } else {
                        auto svar_it = svars.find(count_tok);
                        if (svar_it != svars.end()) {
                            if (svar_it->second.find("ExiledWithSource") != std::string::npos) {
                                from_delve = true;
                                // Capture the Count$ValidExile printed-characteristics filter
                                // (Murktide Regent: "Instant.ExiledWithSource,
                                // Sorcery.ExiledWithSource") so the ETB counter count is
                                // restricted to the matching delve exiles — Delve itself may
                                // exile ANY card (CR 702.66a). The ExiledWithSource qualifier
                                // is implied by membership in cur_game.delve_exiled, so strip
                                // it; the remainder ("Instant,Sorcery") is a card_matches_any
                                // spec.
                                const std::string ve_prefix = "Count$ValidExile ";
                                size_t vp = svar_it->second.find(ve_prefix);
                                if (vp != std::string::npos) {
                                    delve_filter =
                                        svar_it->second.substr(vp + ve_prefix.size());
                                    for (const char *qual :
                                         {".ExiledWithSource", "+ExiledWithSource"}) {
                                        size_t qp;
                                        while ((qp = delve_filter.find(qual)) !=
                                               std::string::npos)
                                            delve_filter.erase(qp, strlen(qual));
                                    }
                                }
                            }
                            // Count$xPaid — the count equals the X value paid at cast time
                            // (Chalice of the Void enters with X charge counters).
                            else if (svar_it->second.find("xPaid") != std::string::npos)
                                from_xpaid = true;
                        }
                    }
                }
            }
            StaticAbility sa;
            sa.category = "EtbCounter";
            sa.counter_type = counter_type_str;
            sa.counter_count = literal_count;
            sa.counter_count_from_delve = from_delve;
            sa.counter_count_delve_filter = delve_filter;
            sa.counter_count_from_xpaid = from_xpaid;
            card.static_abilities.push_back(sa);
            continue;
        }
        // K:Equip:1 R  (equip cost after "Equip:")
        if (kw_line.rfind("Equip", 0) == 0) {
            card.is_equipment = true;
            size_t colon = kw_line.find(':');
            if (colon != std::string::npos) {
                card.equip_cost = parse_mana_cost(kw_line.substr(colon + 1));
            }
            card.keywords.push_back("Equip");
            continue;
        }
        // K:Reconfigure:2  (CR 702.151) — an Equipment keyword on a creature card. Parsed like
        // Equip (the cost grants an attach ability and shares the equip-attach machinery), plus the
        // reconfigure-specific behaviour flagged by is_reconfigure: attach only to a creature you
        // control, an unattach ability while attached, and "while attached this isn't a creature".
        if (kw_line.rfind("Reconfigure", 0) == 0) {
            card.is_equipment = true;
            card.is_reconfigure = true;
            size_t colon = kw_line.find(':');
            if (colon != std::string::npos) {
                card.equip_cost = parse_mana_cost(kw_line.substr(colon + 1));
            }
            card.keywords.push_back("Reconfigure");
            continue;
        }
        // K:Impending:<N>:<mana> — Impending (CR 702.175). An alternative casting cost: the spell
        // may be cast for <mana> instead of its normal mana cost; if so the permanent enters with N
        // time counters and isn't a creature until the last is removed (CR 702.175d-e). Encoded on
        // the shared AltCost (mana portion = parse_mana_cost(<mana>), is_impending + impending_count
        // flag the impending-specific entry/shed behaviour). The format mirrors Reconfigure's
        // colon-split (Equip/Reconfigure), with an extra leading count field: "Impending:5:1 B".
        if (kw_line.rfind("Impending", 0) == 0) {
            std::string rest = kw_line.substr(strlen("Impending"));
            if (!rest.empty() && rest[0] == ':') rest = rest.substr(1);  // "5:1 B"
            size_t colon = rest.find(':');
            if (colon != std::string::npos) {
                AltCost ac;
                ac.has_alt_cost = true;
                ac.is_impending = true;
                ac.impending_count = std::stoi(rest.substr(0, colon));
                ac.mana_cost = parse_mana_cost(rest.substr(colon + 1));
                card.alt_cost = ac;
            }
            card.keywords.push_back("Impending");
            continue;
        }
        // K:Chapter:<final>:<svar1>,<svar2>,...,<svarN> — a Saga's chapter abilities (CR 714). The
        // first field is the Saga's final chapter number (= the number of chapter slots, CR 714.2d);
        // each subsequent comma-separated entry is an SVar naming the DB$ ability run when the Saga's
        // lore counters reach that chapter (CR 714.2b/714.3). Multiple chapters may name the SAME
        // SVar (Summon: Bahamut I & II both DBDestroy) — each becomes its own chapter slot, so two
        // independent triggers fire at lore 1 and lore 2. Parsed 1-indexed into card.saga_chapters;
        // the Saga lifecycle (lore counters, chapter triggers, sacrifice SBA) lives in src/saga.cpp.
        if (kw_line.rfind("Chapter:", 0) == 0) {
            std::vector<std::string> parts = split(kw_line, ':');
            if (parts.size() >= 3) {
                for (const std::string &name : split(parts[2], ',', /*skip_empty=*/true)) {
                    auto it = svars.find(name);
                    if (it != svars.end())
                        card.saga_chapters.push_back(
                            parse_svar_ability(it->second, Ability::TRIGGERED, svars, card.name));
                    else
                        card.saga_chapters.push_back(Ability{});  // keep chapter indexing aligned
                }
            }
            card.keywords.push_back("Chapter");
            continue;
        }
        // K:Prowess — keyword stored; triggered ability applied by apply_keyword_abilities
        if (kw_line == "Prowess" || kw_line.rfind("Prowess", 0) == 0) {
            card.keywords.push_back("Prowess");
            continue;
        }
        // K:Dredge:N — replacement effect: while in graveyard, may replace a draw by
        // milling N cards and returning this card to hand. Value stored on CardData;
        // the replacement is offered in Orderer::draw.
        if (kw_line.rfind("Dredge:", 0) == 0) {
            card.dredge = std::stoi(kw_line.substr(strlen("Dredge:")));
            card.keywords.push_back("Dredge");
            continue;
        }
        // K:Landwalk:Swamp / Forest / Island / Mountain / Plains
        if (kw_line.rfind("Landwalk:", 0) == 0) {
            std::string land_type = kw_line.substr(strlen("Landwalk:"));
            card.keywords.push_back(land_type + "walk");
            continue;
        }
        // K:Cycling:<cost> — activated ability from hand: pay cost, discard this card, draw a card
        if (kw_line.rfind("Cycling:", 0) == 0) {
            std::string cost_str = kw_line.substr(strlen("Cycling:"));
            Ability ab;
            ab.ability_type = Ability::ACTIVATED;
            ab.category = "Draw";
            ab.amount = 1;
            ab.activation_zone = Zone::HAND;
            // Shared Cost$ token grammar (PayLife, Sac, Discard, Return, tap, mana).
            parse_activation_cost(cost_str, ab);
            card.abilities.push_back(ab);
            card.keywords.push_back("Cycling");
            continue;
        }
        // K:Ninjutsu:<cost> (CR 702.49) — a hand-activated ability usable only during the
        // declare-blockers step, after blockers are declared, while you control an unblocked
        // attacker. Pay <cost> and return that unblocked attacker to its owner's hand, then put
        // this card from your hand onto the battlefield tapped and attacking. Modeled as a
        // hand-activated ability flagged is_ninjutsu; process_ninjutsu handles the bespoke cost
        // (return attacker) and effect (enter tapped + attacking). General over any K:Ninjutsu.
        if (kw_line.rfind("Ninjutsu:", 0) == 0) {
            std::string cost_str = kw_line.substr(strlen("Ninjutsu:"));
            Ability ab;
            ab.ability_type = Ability::ACTIVATED;
            ab.category = "Ninjutsu";
            ab.is_ninjutsu = true;
            ab.activation_zone = Zone::HAND;
            // Only the mana portion of the cost is parsed here; the return-an-unblocked-attacker
            // cost is intrinsic to ninjutsu and paid by process_ninjutsu.
            parse_activation_cost(cost_str, ab);
            card.abilities.push_back(ab);
            card.keywords.push_back("Ninjutsu");
            continue;
        }
        // K:TypeCycling:<Subtype>:<cost> — typecycling (CR 702.29f). Like Cycling, an
        // activated ability usable from hand whose cost is the given mana plus discarding
        // this card; but instead of drawing, it searches the library for a card of the
        // named subtype, reveals it, puts it into hand, then shuffles. General over the
        // subtype (Islandcycling/Swampcycling/Plainscycling/...). The discard-this-card
        // cost is the auto-consume that fires for any hand-activated ability (the source
        // goes to the graveyard at activation); the effect is a Library→Hand search.
        if (kw_line.rfind("TypeCycling:", 0) == 0) {
            std::string rest = kw_line.substr(strlen("TypeCycling:"));
            size_t colon = rest.find(':');
            std::string subtype = (colon != std::string::npos) ? rest.substr(0, colon) : rest;
            std::string cost_str = (colon != std::string::npos) ? rest.substr(colon + 1) : "";
            Ability ab;
            ab.ability_type = Ability::ACTIVATED;
            ab.category = "ChangeZone";
            ab.activation_zone = Zone::HAND;
            ab.origin = Zone::LIBRARY;
            ab.destination = Zone::HAND;
            ab.change_type = subtype;       // subtype filter (search_zone matches card subtypes)
            ab.mandatory = false;           // searches may fail to find (CR 701.19c)
            // Shared Cost$ token grammar (the mana portion of the cycling cost).
            parse_activation_cost(cost_str, ab);
            card.abilities.push_back(ab);
            card.keywords.push_back(subtype + "cycling");
            continue;
        }
        // K:Flashback:<cost> — cast from graveyard for flashback cost, then exile
        if (kw_line.rfind("Flashback:", 0) == 0) {
            std::string cost_str = kw_line.substr(strlen("Flashback:"));
            card.has_flashback = true;
            // Shared Cost$ token grammar, then map onto the flashback cost fields the
            // cast path consumes (mana + life). Deep Analysis is "1 U PayLife<3>" — both
            // mana and life — which the token-by-token grammar handles in one pass.
            Ability fb;
            parse_activation_cost(cost_str, fb);
            card.flashback_mana_cost = fb.activation_mana_cost;
            card.flashback_alt_cost.life_cost = fb.life_cost;
            // Flashback—Sacrifice a creature (Cabal Therapy): Sac<1/Creature> in the
            // flashback cost. Carry the sac filter so the cast path pays it.
            card.flashback_alt_cost.sac_cost_spec = fb.sac_cost_spec;
            card.keywords.push_back("Flashback");
            continue;
        }
        // K:Unearth:<cost> — Unearth (CR 702.84): an activated ability usable only from the
        // graveyard, at sorcery speed, that returns this card to the battlefield. The returned
        // permanent gains haste, is exiled at the beginning of the next end step (a delayed
        // triggered ability, CR 603.7b), and is exiled instead if it would leave the battlefield.
        // Modeled as a synthetic graveyard-activated ChangeZone (Graveyard -> Battlefield, Defined$
        // Self); is_unearth flags it so the resolution marks the permanent unearthed (haste +
        // delayed exile + leaves-the-battlefield replacement). General over any K:Unearth:<cost>.
        if (kw_line.rfind("Unearth:", 0) == 0) {
            std::string cost_str = kw_line.substr(strlen("Unearth:"));
            Ability ab;
            ab.ability_type = Ability::ACTIVATED;
            ab.category = "ChangeZone";
            ab.activation_zone = Zone::GRAVEYARD;
            ab.origin = Zone::GRAVEYARD;
            ab.destination = Zone::BATTLEFIELD;
            ab.defined_self = true;        // returns its own source from the graveyard
            ab.sorcery_speed_only = true;  // "Unearth only as a sorcery." (CR 702.84a)
            ab.is_unearth = true;
            // Shared Cost$ token grammar (the mana portion of the unearth cost).
            parse_activation_cost(cost_str, ab);
            card.abilities.push_back(ab);
            card.keywords.push_back("Unearth");
            continue;
        }
        // K:Escape:<mana> [<additional cost>] — cast this card from your graveyard for the
        // escape cost (CR 702.139). The mana portion (e.g. "2 B") precedes any additional cost
        // token (e.g. ExileFromGrave<.../withTypesGE4/...> for Nethergoyf). Mana is parsed from
        // the leading mana symbols; the additional cost is parsed by the shared alt-cost grammar.
        if (kw_line.rfind("Escape:", 0) == 0) {
            std::string cost_str = kw_line.substr(strlen("Escape:"));
            card.has_escape = true;
            // The mana portion runs up to the first additional-cost keyword (ExileFromGrave/
            // PayLife/Sac/Return...); take the substring before "ExileFromGrave" (the only
            // additional cost currently in the vocab) as mana, the remainder as the alt cost.
            std::string mana_part = cost_str;
            std::string alt_part;
            size_t eg = cost_str.find("ExileFromGrave");
            if (eg != std::string::npos) {
                mana_part = cost_str.substr(0, eg);
                alt_part = cost_str.substr(eg);
            }
            // Trim trailing space from the mana part.
            size_t mend = mana_part.find_last_not_of(' ');
            mana_part = (mend == std::string::npos) ? "" : mana_part.substr(0, mend + 1);
            if (!mana_part.empty()) card.escape_mana_cost = parse_mana_cost(mana_part);
            if (!alt_part.empty()) parse_alt_cost_tokens(alt_part, card.escape_alt_cost);
            card.keywords.push_back("Escape");
            continue;
        }
        // K:Evoke:<cost> — alternate cost; when paid, the creature sacrifices itself as it
        // enters. The cost may be a pitch (ExileFromHand), mana (e.g. R), or life. The
        // self-sacrifice is a synthetic ETB self-trigger gated on Permanent::evoked, which
        // is set only when the spell was cast for its evoke cost.
        if (kw_line.rfind("Evoke", 0) == 0) {
            size_t colon = kw_line.find(':');
            std::string cost_str = (colon != std::string::npos) ? kw_line.substr(colon + 1) : "";
            AltCost ac;
            parse_alt_cost_tokens(cost_str, ac);
            ac.is_evoke = true;
            card.alt_cost = ac;
            card.keywords.push_back("Evoke");

            Ability sac;
            sac.ability_type = Ability::TRIGGERED;
            sac.category = "ChangeZone";
            sac.trigger_on = Events::CARD_CHANGED_ZONE;
            sac.trigger_zone_destination = Zone::BATTLEFIELD;
            sac.trigger_only_self = true;
            sac.is_evoke_sacrifice = true;
            sac.defined_self = true;          // moves its own source (no targeting)
            sac.valid_tgts = "N_A";
            sac.origin = Zone::BATTLEFIELD;
            sac.destination = Zone::GRAVEYARD;
            sac.mandatory = true;
            card.abilities.push_back(sac);
            continue;
        }
        // K:Offspring:<cost> — an optional additional cost (CR 702.171). You may pay the
        // offspring cost in addition to the spell's mana cost as you cast it; if you do,
        // when this creature enters, create a 1/1 token that's a copy of it. Modeled as a
        // second cast option (paying base + offspring) that sets Permanent::entered_with_offspring,
        // gating a synthetic ETB self-trigger that creates the 1/1 token copy.
        if (kw_line.rfind("Offspring", 0) == 0) {
            size_t colon = kw_line.find(':');
            if (colon != std::string::npos)
                card.offspring_cost = parse_mana_cost(kw_line.substr(colon + 1));
            card.has_offspring = true;
            card.keywords.push_back("Offspring");

            Ability tok;
            tok.ability_type = Ability::TRIGGERED;
            tok.category = "CopyPermanent";
            tok.trigger_on = Events::CARD_CHANGED_ZONE;
            tok.trigger_zone_destination = Zone::BATTLEFIELD;
            tok.trigger_only_self = true;
            tok.is_offspring_token = true;
            tok.defined_self = true;          // copies its own source (no targeting)
            tok.valid_tgts = "N_A";
            tok.mandatory = true;
            card.abilities.push_back(tok);
            continue;
        }
        // K:Kicker:<cost1>[:<cost2>...] — one or more OPTIONAL ADDITIONAL costs (CR 702.33).
        // Forge encodes "Kicker [A] and/or [B]" as two colon-separated costs (CR 702.33b:
        // it means "Kicker [A], kicker [B]" — two independent kickers). Each segment is a mana
        // cost paid in addition to the spell's cost as it's cast; paying it makes the spell
        // "kicked with its Nth kicker". Stored as a list so the model is multikicker-ready and
        // the linked "if it was kicked with its [N] kicker" triggers index into it.
        if (kw_line.rfind("Kicker:", 0) == 0) {
            std::string rest = kw_line.substr(strlen("Kicker:"));
            for (const std::string &seg : split(rest, ':', /*skip_empty=*/true))
                card.kicker_costs.push_back(parse_mana_cost(seg));
            card.keywords.push_back("Kicker");
            continue;
        }
        // K:Replicate:<cost> — an OPTIONAL ADDITIONAL cost (CR 702.x) that may be paid any
        // number of times as the spell is cast. Each payment copies the spell once on cast
        // (the copies may choose new targets). Stored as a single per-instance mana cost; the
        // count paid is decided at cast time (see action_processor) and recorded per-Spell.
        if (kw_line.rfind("Replicate:", 0) == 0) {
            std::string rest = kw_line.substr(strlen("Replicate:"));
            card.replicate_cost = parse_mana_cost(rest);
            card.has_replicate = true;
            card.keywords.push_back("Replicate");
            continue;
        }
        // K:Devoid — the object is colorless (CR 702.114a). Forge cards with Devoid omit a
        // Colors: line and rely on the keyword for their colorlessness, so apply it here as a
        // general color override (e.g. an Eldrazi printed with colored mana symbols is still
        // colorless). explicit_colors = {COLORLESS} marks the card colorless for ColorIdentity.
        if (kw_line == "Devoid") {
            card.explicit_colors.clear();
            card.explicit_colors.insert(COLORLESS);
            card.keywords.push_back("Devoid");
            continue;
        }
        // K:Gift — the Gift keyword (CR 702.176). As the spell is cast its controller MAY promise
        // the gift to an opponent (an optional choice, not a cost); if promised, the opponent
        // receives the gift as the spell resolves, before its other effects. The gift effect is
        // held in the card's GiftAbility SVar (a DB$ Token making the gift token). Parse it into
        // card.gift_abilities; the cast path (action_processor) offers the promise choice and the
        // resolving spell runs these when Spell::gift_promised is set (Ability::resolve).
        if (kw_line == "Gift" || kw_line.rfind("Gift", 0) == 0) {
            card.has_gift = true;
            card.keywords.push_back("Gift");
            auto git = svars.find("GiftAbility");
            if (git != svars.end()) {
                card.gift_abilities.push_back(
                    parse_svar_ability(git->second, Ability::SPELL, svars, card.name));
                size_t gd = git->second.find("GiftDescription$");
                if (gd != std::string::npos) {
                    gd += strlen("GiftDescription$");
                    while (gd < git->second.size() && git->second[gd] == ' ') gd++;
                    size_t ge = git->second.find('|', gd);
                    if (ge == std::string::npos) ge = git->second.size();
                    card.gift_description = git->second.substr(gd, ge - gd);
                    while (!card.gift_description.empty() && card.gift_description.back() == ' ')
                        card.gift_description.pop_back();
                }
            }
            continue;
        }
        // K:MayEffectFromOpeningHand:<SVar>[:!PlayFirst] — "If this card is in your opening
        // hand, you may [effect]" (CR 103.6b; Leyline of the Void's begin-the-game-on-the-
        // battlefield). The colon field names the SVar holding the effect body (Leyline:
        // DB$ ChangeZone | Defined$ Self | Origin$ Hand | Destination$ Battlefield); an optional
        // !PlayFirst field (Gemstone Caverns) limits the offer to the player NOT going first.
        // The offer itself happens after mulligans in Orderer::do_opening_hand_actions.
        if (kw_line.rfind("MayEffectFromOpeningHand", 0) == 0) {
            card.keywords.push_back("MayEffectFromOpeningHand");
            std::vector<std::string> parts = split(kw_line, ':');
            if (parts.size() >= 2) {
                auto oit = svars.find(parts[1]);
                if (oit != svars.end())
                    card.opening_hand_abilities.push_back(
                        parse_svar_ability(oit->second, Ability::SPELL, svars, card.name));
            }
            for (size_t pi = 2; pi < parts.size(); pi++)
                if (parts[pi] == "!PlayFirst") card.opening_hand_not_first = true;
            continue;
        }
        // K:Storm — Storm (CR 702.40). A triggered ability that functions on the stack: "When
        // you cast this spell, copy it for each spell cast before it this turn. You may choose new
        // targets for the copies." Synthesize the self-cast SPELL_CAST trigger here (general over
        // any Storm card); the copy count is locked in when the trigger fires
        // (state_manager_triggers) and the copies are put on the stack at resolution
        // (effects::storm). The trigger itself takes no target — each copy chooses its own.
        if (kw_line == "Storm") {
            card.keywords.push_back("Storm");
            Ability st;
            st.ability_type = Ability::TRIGGERED;
            st.category = "Storm";
            st.trigger_on = Events::SPELL_CAST;
            st.trigger_only_self = true;  // ValidCard$ Card.Self — fires for the cast spell itself
            st.valid_tgts = "N_A";
            st.mandatory = true;
            card.abilities.push_back(st);
            continue;
        }
        // K:Annihilator:N — Annihilator N (CR 702.85). "Whenever this creature attacks, defending
        // player sacrifices N permanents." Synthesize the self-attack trigger here (general over any
        // Annihilator card): a TRIGGERED Sacrifice that fires once per declared attack of this
        // creature (CREATURE_ATTACKED). defined_each_opponent routes the edict to the defending
        // player (the controller's opponent in the two-player engine), who chooses and sacrifices N
        // of their own permanents one at a time (sac_count). It resolves like any triggered ability,
        // i.e. before blockers are declared. SacValid$ Permanent = any permanent they control.
        if (kw_line.rfind("Annihilator", 0) == 0) {
            size_t colon = kw_line.find(':');
            int n = (colon != std::string::npos) ? std::stoi(kw_line.substr(colon + 1)) : 1;
            card.keywords.push_back(kw_line);
            Ability ab;
            ab.ability_type = Ability::TRIGGERED;
            ab.category = "Sacrifice";
            ab.trigger_on = Events::CREATURE_ATTACKED;
            ab.trigger_only_self = true;  // ValidCard$ Card.Self — only this creature's own attack
            ab.valid_tgts = "N_A";
            ab.mandatory = true;
            ab.defined_each_opponent = true;  // the defending player sacrifices (CR 702.85b)
            ab.sac_valid = "Permanent";       // any permanent the defending player controls
            ab.sac_count = static_cast<size_t>(n);
            card.abilities.push_back(ab);
            continue;
        }
        // K:Protection:<quality>:<desc> — structured Protection keyword (CR 702.16). The middle
        // field is the quality. Emrakul uses Protection:Spell.nonColorless ("protection from
        // colored spells"): a one-or-more-colors SPELL can't target it (702.16b/e). Modeled as a
        // creature keyword consulted in has_protection_from. A structured single-color quality is
        // normalized to the literal "Protection from <color>" form the color-protection path
        // already understands (the common color-protection cards spell that form out directly).
        if (kw_line.rfind("Protection:", 0) == 0) {
            std::vector<std::string> parts = split(kw_line, ':');
            std::string spec = parts.size() > 1 ? parts[1] : "";
            if (spec.rfind("Spell", 0) == 0 && spec.find("nonColorless") != std::string::npos) {
                card.keywords.push_back("Protection from colored spells");
            } else {
                std::string color = spec;
                std::transform(color.begin(), color.end(), color.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                card.keywords.push_back("Protection from " + color);
            }
            continue;
        }
        split_keywords(kw_line, card.keywords);
    }
}

Token parse_token_script(const std::string &script_name) {
    Token tok;
    std::string path = RESOURCE_DIR + "/tokenscripts/" + script_name + ".txt";
    std::ifstream stream(path);
    if (!stream.is_open()) {
        non_fatal_error("Could not open token script: " + path);
        return tok;
    }
    std::string script_data;
    char buffer[SCRIPT_MAX_LEN];
    while (stream.getline(buffer, SCRIPT_MAX_LEN)) {
        std::string line(buffer);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        script_data += line;
        script_data += "\n";
    }
    stream.close();

    tok.name = value_from_script(script_data, "Name");
    // Forge token scripts usually name the token "<Name> Token" ("Cat Warrior Token").
    // Store the clean name once here — display code (entity_name's " token" tag,
    // GameState::token_name) adds its own token marker, so keeping the suffix doubled
    // every log line ("Cat Warrior Token token"). Per CR 111.4 the token's real name is
    // the part without "Token" anyway.
    static const std::string kTokenSuffix = " Token";
    if (tok.name.size() > kTokenSuffix.size() &&
        tok.name.compare(tok.name.size() - kTokenSuffix.size(), kTokenSuffix.size(),
                         kTokenSuffix) == 0)
        tok.name.erase(tok.name.size() - kTokenSuffix.size());
    tok.types = parse_types(value_from_script(script_data, "Types"));
    tok.explicit_colors = parse_colors_field(value_from_script(script_data, "Colors"));

    std::string pt = value_from_script(script_data, "PT");
    tok.power = parse_power(pt);
    tok.toughness = parse_toughness(pt);

    // Parse K: keyword lines — keyword stored; triggered ability applied by apply_keyword_abilities
    for (auto &kw_line : multi_values_from_script(script_data, "K")) {
        split_keywords(kw_line, tok.keywords);
    }

    // Parse T: triggered abilities, then A: activated/spell abilities. A token can carry an
    // intrinsic activated ability (e.g. the Eldrazi Spawn token's "Sacrifice this creature:
    // Add {C}.", c_0_1_eldrazi_spawn_sac) — parse_abilities honours the same cost/category
    // grammar as a real card's A: line, so the sac-for-mana ability resolves identically to
    // Lotus Petal's.
    auto svars = parse_svars(script_data);
    tok.abilities = parse_triggered_abilities(script_data, svars, tok.name);
    for (auto &ab : parse_abilities(multi_values_from_script(script_data, "A"), tok.types,
                                    svars, tok.name))
        tok.abilities.push_back(ab);
    // S: lines — continuous static abilities (e.g. the Construct token's "+1/+1 for each
    // artifact you control" self-buff). Applied via the Permanent once bootstrapped.
    tok.static_abilities = parse_static_abilities(script_data, svars);

    return tok;
}

// private util functions
static std::string value_from_script(std::string script, std::string key) {
    // Match the key only as a line-start field header ("Key:value"), never as a substring inside
    // a later line. Top-level fields are one per line, so the key must begin the script or follow
    // a '\n' AND be immediately followed by ':'. Without this, a short key like "PT" would match
    // inside an ability line (e.g. Karn's "AILogic$ PTByCMC"), returning garbage that downstream
    // numeric parsers (parse_power) then crash on.
    size_t search = 0;
    while (true) {
        auto pos = script.find(key, search);
        if (pos == std::string::npos) return "";
        bool at_line_start = (pos == 0 || script[pos - 1] == '\n');
        size_t after = pos + key.length();
        bool followed_by_colon = (after < script.size() && script[after] == ':');
        if (at_line_start && followed_by_colon) {
            size_t valstart = after + 1;  // skip the ':'
            auto end_pos = script.find("\n", valstart);
            return script.substr(valstart, (end_pos - valstart));  // omit linebreak at end
        }
        search = pos + 1;
    }
}

static std::vector<std::string> multi_values_from_script(std::string script, std::string key) {
    // Match the key only as a line-start field header ("Key:value"), same rule as
    // value_from_script above. The old bare substring find leaked SVar bodies into the "A"
    // scan: on Urza's Saga, the 'A' inside "SVar:ABMana:AB$ Mana | ..." matched, the tail of
    // the line ("Mana:AB$ Mana | Cost$ T | ...") was returned as an ability line, and the Saga
    // got its chapter-granted activated abilities at parse time — available from ETB instead
    // of only after the granting chapter ability resolved.
    std::vector<std::string> ret_val;
    size_t search = 0;
    while (true) {
        auto pos = script.find(key, search);
        if (pos == std::string::npos) break;
        bool at_line_start = (pos == 0 || script[pos - 1] == '\n');
        size_t after = pos + key.length();
        bool followed_by_colon = (after < script.size() && script[after] == ':');
        if (!at_line_start || !followed_by_colon) {
            search = pos + 1;
            continue;
        }
        size_t valstart = after + 1;  // skip the ':'
        auto end_pos = script.find("\n", valstart);
        std::string line = script.substr(valstart, (end_pos - valstart));  // end_pos is at \n, so no -1 needed
        if (!line.empty() && line.back() == '\r') line.pop_back();  // strip \r for Windows line endings
        ret_val.push_back(line);
        if (end_pos == std::string::npos) break;
        search = end_pos + 1;
    }
    return ret_val;
}

// Map a single mana-cost color letter to its color, or NO_COLOR for a non-color char.
static Colors mana_letter_color(char c) {
    switch (c) {
        case 'W': return WHITE;
        case 'U': return BLUE;
        case 'B': return BLACK;
        case 'R': return RED;
        case 'G': return GREEN;
        case 'C': return COLORLESS;
        default:  return NO_COLOR;
    }
}

// Recognize a HYBRID mana token (CR 107.4b/107.4e) within a single space-separated ManaCost
// token and, if matched, append the pip to *hybrid_out and return true. Two Forge encodings:
//   - color hybrid: two adjacent color letters ("WU") or a slashed pair ("W/U") → one pip
//     payable by either color, MV 1.
//   - monocolored hybrid / "twobrid": "<N>/<color>" (e.g. "2/W") → one pip payable by N generic
//     OR one mana of the color, MV N.
// Phyrexian ("WP"/"W/P") is NOT handled here — it is left to the phyrexian path. A token that is
// not a hybrid (single color, generic number, X, phyrexian) returns false so the caller can fall
// back to the per-character parse. Color letters here exclude COLORLESS ('C') for color hybrids,
// as Forge has no {C/x} color-hybrid pip.
static bool parse_hybrid_token(const std::string &tok, std::vector<HybridPip> *hybrid_out) {
    if (!hybrid_out) return false;
    auto is_color_letter = [](char c) {
        return c == 'W' || c == 'U' || c == 'B' || c == 'R' || c == 'G';
    };
    // Two adjacent color letters: "WU" (color hybrid, no slash).
    if (tok.size() == 2 && is_color_letter(tok[0]) && is_color_letter(tok[1])) {
        HybridPip pip;
        pip.colors = {mana_letter_color(tok[0]), mana_letter_color(tok[1])};
        pip.generic_alt = 0;
        pip.mana_value = 1;
        hybrid_out->push_back(pip);
        return true;
    }
    // Slashed forms: "<A>/<B>".
    size_t slash = tok.find('/');
    if (slash != std::string::npos && slash > 0 && slash + 1 < tok.size()) {
        std::string lhs = tok.substr(0, slash);
        std::string rhs = tok.substr(slash + 1);
        // Phyrexian ("W/P") is handled elsewhere — not a hybrid pip.
        if (rhs == "P") return false;
        // Twobrid "<N>/<color>": generic alternative N, single color option.
        bool lhs_num = !lhs.empty() &&
                       std::all_of(lhs.begin(), lhs.end(),
                                   [](char c) { return std::isdigit(static_cast<unsigned char>(c)); });
        if (lhs_num && rhs.size() == 1 && is_color_letter(rhs[0])) {
            HybridPip pip;
            pip.colors = {mana_letter_color(rhs[0])};
            pip.generic_alt = std::stoi(lhs);
            pip.mana_value = pip.generic_alt;
            hybrid_out->push_back(pip);
            return true;
        }
        // Slashed color hybrid "W/U".
        if (lhs.size() == 1 && rhs.size() == 1 && is_color_letter(lhs[0]) && is_color_letter(rhs[0])) {
            HybridPip pip;
            pip.colors = {mana_letter_color(lhs[0]), mana_letter_color(rhs[0])};
            pip.generic_alt = 0;
            pip.mana_value = 1;
            hybrid_out->push_back(pip);
            return true;
        }
    }
    return false;
}

static std::multiset<Colors> parse_mana_cost(std::string value, std::vector<Colors> *phyrexian_out,
                                             std::vector<HybridPip> *hybrid_out) {
    std::multiset<Colors> ret_val;
    if (value == "no cost") return ret_val;
    // Hybrid pips are space-separated tokens in Forge's ManaCost encoding ("3 WU WU", "2/B 2/R").
    // Pull those out first so a two-color token isn't mis-read as two separate colored pips by the
    // per-character scan below; everything else falls through to the original char-by-char parse.
    if (hybrid_out) {
        std::string rest;
        for (const auto &tok : split(value, ' ', /*skip_empty=*/true)) {
            if (parse_hybrid_token(tok, hybrid_out)) continue;
            if (!rest.empty()) rest += ' ';
            rest += tok;
        }
        value = rest;
    }
    auto len = value.length();
    for (size_t i = 0; i < len; i++) {
        // Check for Phyrexian mana: XP where X is a color letter
        bool is_phyrexian = false;
        if (i + 1 < len && value[i + 1] == 'P') {
            Colors phyrexian_color = NO_COLOR;
            switch (value[i]) {
                case 'W': phyrexian_color = WHITE; break;
                case 'U': phyrexian_color = BLUE; break;
                case 'B': phyrexian_color = BLACK; break;
                case 'R': phyrexian_color = RED; break;
                case 'G': phyrexian_color = GREEN; break;
                default: break;
            }
            if (phyrexian_color != NO_COLOR) {
                if (phyrexian_out) phyrexian_out->push_back(phyrexian_color);
                i++;  // skip the 'P'
                is_phyrexian = true;
            }
        }
        if (is_phyrexian) continue;
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
            case 'X':
                // X is variable; handled separately by has_x_cost flag
                break;
            default:
                if (std::isdigit(static_cast<unsigned char>(value[i]))) {
                    // Consume the entire run of digits so a multi-digit generic cost
                    // (e.g. "10") parses as one number, not one generic per digit.
                    size_t j = i;
                    while (j < len && std::isdigit(static_cast<unsigned char>(value[j]))) j++;
                    int generic = std::stoi(value.substr(i, j - i));
                    for (int g = 0; g < generic; g++) ret_val.emplace(GENERIC);
                    i = j - 1;  // for-loop ++ advances past the last digit
                }
                break;
        }
    }
    return ret_val;
}

// Parse a Forge "Colors:" field (space-separated color words) into an explicit color set.
// Shared by card and token parsing so both compute colorlessness identically: an empty set
// (no Colors: line) leaves the object's color to be derived from its mana cost (CR 105.2),
// which for a costless token means colorless.
static std::set<Colors> parse_colors_field(const std::string &colors_field) {
    std::set<Colors> ret;
    if (colors_field.empty()) return ret;
    size_t cp = 0;
    while (cp <= colors_field.size()) {
        size_t sp = colors_field.find(' ', cp);
        if (sp == std::string::npos) sp = colors_field.size();
        std::string ctok = colors_field.substr(cp, sp - cp);
        if      (ctok == "white")    ret.insert(WHITE);
        else if (ctok == "blue")     ret.insert(BLUE);
        else if (ctok == "black")    ret.insert(BLACK);
        else if (ctok == "red")      ret.insert(RED);
        else if (ctok == "green")    ret.insert(GREEN);
        else if (ctok == "colorless")ret.insert(COLORLESS);
        cp = sp + 1;
    }
    return ret;
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
    if (pow_string.find('*') != std::string::npos) return 0;  // characteristic-defining; base 0
    return std::stoi(pow_string);
}

static uint32_t parse_toughness(std::string value) {
    if (value == "") return 0;
    auto slash_pos = value.find("/");
    std::string tough_string = value.substr(slash_pos + 1);
    // "1+*" → base is the numeric prefix (1); * is characteristic-defining
    size_t star_pos = tough_string.find('*');
    if (star_pos != std::string::npos) {
        if (star_pos == 0) return 0;
        std::string prefix = tough_string.substr(0, star_pos);
        // Strip trailing '+' from "1+"
        while (!prefix.empty() && (prefix.back() == '+' || prefix.back() == '-'))
            prefix.pop_back();
        return prefix.empty() ? 0 : static_cast<uint32_t>(std::stoi(prefix));
    }
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
static void apply_param_to_ability(Ability& ability, const std::string& key, const std::string& value,
                                   const std::string& card_name) {
    if (key == "NumCards" || key == "ChangeNum" || key == "Amount" ||
        key == "TokenAmount" || key == "ScryNum" || key == "Num" || key == "NumTurns") {
        if (value == "DamageAmount" || value == "TriggerCount$DamageAmount") {
            ability.amount_from_damage = true;
        } else if (key == "ChangeNum" && value == "Any") {
            // "Any" = the player may take any number of the looked-at cards (Dig/Fateseal).
            ability.change_num_any = true;
        } else if (!value.empty() && std::isdigit(static_cast<unsigned char>(value[0]))) {
            ability.amount = static_cast<size_t>(std::stoi(value));
            // For Dig, ChangeNum$ is the exact take count and 0 is meaningful ("take nothing"),
            // which amount==0 (the unset default) can't express — record it explicitly.
            if (key == "ChangeNum") ability.change_num = std::stoi(value);
        } else if (!value.empty()) {
            // Non-numeric value is a SVar key — store for runtime resolution
            ability.amount_svar = value;
        }
    } else if (key == "ValidTgts") {
        ability.valid_tgts = value;
    } else if (key == "Mandatory") {
        ability.mandatory = (value == "True");
    } else if (key == "UnlessCost") {
        // UnlessCost$ PayEnergy<N> (Wrath of the Skies' DestroyAll) is an energy unless-cost,
        // not the generic {N} mana unless-cost: route the energy amount (an SVar) to the
        // DestroyAll params. A numeric value is the legacy generic-mana unless-cost.
        size_t pe = value.find("PayEnergy<");
        size_t pd = value.find("Discard<");
        if (pe != std::string::npos) {
            size_t close = value.find('>', pe);
            std::string n = (close != std::string::npos)
                                ? value.substr(pe + 10, close - (pe + 10)) : "";
            if (ability.category == "DestroyAll") {
                effect_params<DestroyAllParams>(ability).energy_unless_expr = n;  // SVar token; resolved post-parse
            } else {
                // UnlessCost$ PayEnergy<N> on any other effect (Static Prison's DB$ Sacrifice):
                // pay N energy ({E}) to prevent the effect. The count is a literal here; route it
                // through the generic unless-cost count + an energy flag so run_unless_loop pays it.
                ability.unless_cost_is_energy = true;
                bool numeric = !n.empty() &&
                               n.find_first_not_of("0123456789") == std::string::npos;
                ability.unless_generic_cost = numeric ? static_cast<size_t>(std::stoi(n)) : 1;
            }
        } else if (pd != std::string::npos) {
            // UnlessCost$ Discard<N/Card> (Reality Smasher): the payer discards N card(s) from
            // hand to prevent the counter. The N count rides on unless_generic_cost; the discard
            // flag selects the discard payment path in run_unless_loop. The "/Card" filter is the
            // (only) supported discard-any-card filter today.
            std::string inner = value.substr(pd + 8);  // after "Discard<"
            size_t slash = inner.find('/');
            std::string n = (slash != std::string::npos) ? inner.substr(0, slash) : inner;
            ability.unless_generic_cost = static_cast<size_t>(std::stoi(n));
            ability.unless_cost_is_discard = true;
        } else {
            ability.unless_generic_cost = static_cast<size_t>(std::stoi(value));
        }
    } else if (key == "UnlessPayer") {
        // UnlessPayer$ You — the controller is the payer of the unless-cost (the only payer we
        // model for the energy unless-cost). Cosmetic given the energy is paid by the controller.
        // UnlessPayer$ TriggeredSourceSAController — the payer is the controller of the spell that
        // targeted the source (Reality Smasher's opponent), bound at trigger-fire time.
        if (value == "TriggeredSourceSAController")
            ability.unless_payer_is_triggered_source_sa_ctrl = true;
    } else if (key == "UnlessSwitched") {
        // UnlessSwitched$ True inverts the normal "do unless paid" into "do only if paid"
        // (Wrath of the Skies: destroy only if the energy was paid).
        effect_params<DestroyAllParams>(ability).energy_unless_switched = (value == "True");
    } else if (key == "LifeAmount") {
        if (!value.empty() && std::isdigit(static_cast<unsigned char>(value[0]))) {
            ability.amount = static_cast<size_t>(std::stoi(value));
        } else if (!value.empty()) {
            ability.amount_svar = value;
        }
    } else if (key == "Shuffle") {
        // Shuffle$ True on a ChangeZoneAll into Library (Emrakul's death trigger: "shuffle their
        // graveyard into their library"): shuffle the destination library after the move. For the
        // sameName search/move path (Extirpate/Surgical) the destination is Exile, so this flag is
        // inert there — the change_zone_same_name handler doesn't consult it.
        ability.shuffle_after = (value == "True");
    } else if (key == "TargetType") {
        ability.target_type = value;  // "Spell", "Activated,Triggered", etc.
    } else if (key == "Optional") {
        ability.optional_choice = (value == "True");
    } else if (key == "Defined" || key == "DefinedPlayer") {
        // Keep the raw token verbatim (CR 608.2c) so sub-ability target binding can read the
        // script's stated intent; the specific bools below remain authoritative per effect.
        ability.defined = value;
        if (value == "Remembered") ability.defined_remembered = true;
        // Defined$ DelayTriggerRememberedLKI — the objects a DB$ DelayedTrigger captured at
        // registration (RememberObjects$ RememberedLKI). delayed_trigger() restores them into
        // cur_game.remembered_entities before the fire ability resolves, so this resolves
        // exactly like Defined$ Remembered (Flickerwisp / Phelia return the exiled card).
        else if (value == "DelayTriggerRememberedLKI") ability.defined_remembered = true;
        // Defined$ TriggeredSpellAbility — the effect acts on the spell that fired this
        // trigger (Chalice of the Void: "counter that spell"). Set at trigger fire time.
        else if (value == "TriggeredSpellAbility" || value == "TriggeredSpell")
            ability.defined_triggered_spell = true;
        // Defined$ TriggeredSourceSA — the effect acts on the spell/ability that targeted the
        // source (Reality Smasher: "counter that spell"). Bound at trigger fire time from the
        // BECAME_TARGET event's ENTITY (the targeting object).
        else if (value == "TriggeredSourceSA")
            ability.defined_triggered_source_sa = true;
        // Defined$ TriggeredAttacker(LKICopy) — the effect acts on the creature whose attack
        // fired this trigger (Tamiyo, Seasoned Scholar: the attacking creature gets -1/-0).
        // Bound as the ability's target at trigger-fire time from the CREATURE_ATTACKED event.
        else if (value == "TriggeredAttacker" || value == "TriggeredAttackerLKICopy")
            ability.defined_triggered_attacker_lki = true;
        else if (value == "TargetedController") ability.defined_targeted_controller = true;
        // Defined$ TriggeredActivator — the effect acts on the player who caused the trigger
        // (the caster of the triggering spell). The actual player is bound at trigger-fire
        // time from the event's PLAYER param. CR 603.x.
        else if (value == "TriggeredActivator") ability.defined_triggered_activator = true;
        else if (value == "Self") ability.defined_self = true;
        // Defined$ You — the effect's player is the source's controller (CR 109.5). Used by
        // self-pain riders like Ancient Tomb's "deals 2 damage to you" sub-ability.
        else if (value == "You") ability.defined_you = true;
        // Defined$ Player.Opponent — the effect's player is "each opponent" (no chosen
        // target). CR 109.5 / 102.1: in a two-player game this is the single opponent.
        else if (value == "Player.Opponent" || value == "Opponent") ability.defined_each_opponent = true;
        // Defined$ Valid <filter> (CopyPermanent's "for each token you control that entered
        // this turn") — store the filter spec the same place ValidCards$ writes it, so the
        // effect can scan the battlefield for matches.
        else if (value.rfind("Valid ", 0) == 0) ability.valid_cards_filter = value.substr(6);
    } else if (key == "Chooser") {
        // Chooser$ You on a search/move ChangeZone over another player's hidden zone: the
        // ABILITY'S CONTROLLER makes the selection, not the searched zone's owner (Thought-Knot
        // Seer — you choose a nonland card from the targeted opponent's revealed hand to exile).
        // Any other Chooser value (e.g. the sameName cosmetic) leaves the zone owner choosing.
        ability.chooser_is_controller = (value == "You");
    } else if (key == "Condition" && value == "Blessing") {
        ability.condition_city_blessing = true;  // CopyPermanent gated on the city's blessing
    } else if (key == "RememberTargets") {
        ability.remember_targeted = (value == "True");
    } else if (key == "RememberObjects") {
        // RememberObjects$ Targeted — remember the spell's target(s) for later
        // Remembered.sameName subabilities (Surgical Extraction).
        if (value.find("Targeted") != std::string::npos) ability.remember_targeted = true;
        // RememberObjects$ Self — a DB$ Effect that tracks its own source (Kappa Cannoneer's
        // can't-be-blocked effect remembers the creature it applies to).
        if (value.find("Self") != std::string::npos) ability.effect_remember_self = true;
        // RememberObjects$ RememberedLKI — a DB$ DelayedTrigger snapshots the objects the
        // preceding RememberChanged$ ChangeZone just moved, so its Execute$ ability can act on
        // those same objects when it fires later (Flickerwisp / Phelia exile-and-return).
        if (value.find("RememberedLKI") != std::string::npos)
            effect_params<DelayedTriggerParams>(ability).remember_objects_lki = true;
    } else if (key == "StaticAbilities") {
        // DB$ Effect | StaticAbilities$ <name> — names the continuous static the transient
        // effect grants (e.g. Unblockable). Stored for the Effect handler to interpret.
        ability.effect_static_ability = value;
    } else if (key == "TgtZone") {
        if (value == "Graveyard") ability.target_in_graveyard = true;
    } else if (key == "RememberRevealed") {
        // RememberRevealed$ True (Cloak and Dagger): the revealed hand becomes the remembered
        // candidate set for a later Defined$ Remembered exile (handled in effect_reveal_hand).
        ability.remember_revealed = (value == "True");
    } else if (key == "RememberPumped") {
        // RememberPumped$ True (Cloak and Dagger): the optionally-chosen creature is appended
        // to the remembered candidate set (handled in effect_pump).
        ability.remember_pumped = (value == "True");
    } else if (key == "ClearRemembered") {
        ability.clear_remembered = (value == "True");
    } else if (key == "ClearChosenCard") {
        ability.clear_chosen = (value == "True");
    } else if (key == "ChooseEach") {
        ability.choose_each = value;
    } else if (key == "TargetMin") {
        // A numeric minimum (TargetMin$ 0 = optional, TargetMin$ 2 = at least 2) is used
        // directly. A non-numeric value is an SVar key (TargetMin$ X, X = Count$xPaid →
        // "exactly X targets"): stash the token; parse_abilities' post-pass resolves it and,
        // when it is Count$xPaid, sets target_min_from_xpaid so select_target requires X targets.
        if (!value.empty() && std::isdigit(static_cast<unsigned char>(value[0])))
            ability.target_min = std::stoi(value);
        else
            ability.target_min_svar = value;
    } else if (key == "TargetMax") {
        // A numeric cap (TargetMax$ 3) is used directly. A count-SVar cap means there is no
        // fixed upper bound, so fall back to "effectively unlimited" (MAX_ENTITIES); the
        // multi-target selection loop stops on its own once no further legal targets remain
        // (Mindbreak Trap). The post-pass resolves the stashed token: a Count$xPaid cap
        // (TargetMax$ X) sets target_max_from_xpaid so the loop clamps to X (Kozilek's Command;
        // with the matching TargetMin$ X this yields EXACTLY-X targeting).
        if (!value.empty() && std::isdigit(static_cast<unsigned char>(value[0]))) {
            ability.target_max = std::stoi(value);
        } else {
            ability.target_max_svar = value;
            ability.target_max = MAX_ENTITIES;
        }
    } else if (key == "ActivationZone") {
        if (value == "Hand") ability.activation_zone = Zone::HAND;
    } else if (key == "Activation") {
        // Activation$ <condition> — "activate only if <condition>" gate (CR 602.5). The
        // named condition (e.g. "Metalcraft") is evaluated against the activator at
        // activation-legality time by activation_condition_met(); kept general so other
        // gated activations name their condition here without retagging.
        ability.activation_condition = value;
    } else if (key == "ActivationLimit") {
        ability.activation_limit = std::stoi(value);
    } else if (key == "ReduceCost") {
        // ReduceCost$ on an activated ability (Eiganjo Channel). Store the raw value
        // verbatim: a literal integer ("1") is used as-is; a single SVar key ("X") is
        // resolved to its Count$/dynamic expression in parse_abilities' post-pass. The
        // generic mana portion of the activation cost is reduced by the resolved amount at
        // activation time (CR 601.2f).
        ability.reduce_cost_expr = value;
    } else if (key == "ChangeNum") {
        ability.amount = static_cast<size_t>(std::stoi(value));
    } else if (key == "RestrictValid") {
        if (value.find("Creature") != std::string::npos &&
            value.find("ChosenType") != std::string::npos) {
            ability.restrict_to_chosen_type_creature = true;
        } else if (value.find("Eldrazi") != std::string::npos &&
                   value.find("Colorless") != std::string::npos) {
            // RestrictValid$ Spell.Eldrazi+Colorless,... — mana usable only to cast a
            // colorless Eldrazi spell (Eldrazi Temple). The trailing Activated.Eldrazi…
            // clause (activate abilities of colorless Eldrazi) is folded into the same
            // restriction. CR 106.7.
            ability.restrict_to_colorless_eldrazi = true;
        } else if (value.find("Creature") != std::string::npos) {
            // RestrictValid$ Spell.Creature — mana usable only to cast a creature spell
            // (any creature, no subtype constraint), e.g. Abundant Countryside. CR 106.7.
            ability.restrict_to_creature = true;
        }
    } else if (key == "AddsNoCounter") {
        ability.adds_no_counter = (value == "True");
    } else if (key == "InstantSpeed") {
        ability.instant_speed = (value == "True");
    } else if (key == "SorcerySpeed") {
        // SorcerySpeed$ True — this activated ability can be activated only any time its
        // controller could cast a sorcery (main phase, their turn, empty stack). Used by the
        // activated form of Earthbend (Ba Sing Se). Gated in the legal-action enumeration.
        ability.sorcery_speed_only = (value == "True");
    } else if (key == "ETB") {
        // ETB$ True on a DB$ Tap (Ba Sing Se's LandTapped replacement SVar): the tap happens as
        // the permanent enters the battlefield. The conditional "enters tapped" is realized via
        // the ENTERS_TAPPED replacement; this flag marks the resolve-time Tap as an ETB tap.
        ability.tap_on_etb = (value == "True");
    } else if (key == "Planeswalker") {
        ability.is_loyalty_ability = (value == "True");
    } else if (key == "Cost") {
        parse_activation_cost(value, ability);
    } else if (key == "ConditionCheckSVar") {
        // ConditionCheckSVar$ <SVar> on a top-level A: ability (Veil of Summer's SP$ Draw gate).
        // Store the raw SVar name; parse_abilities' post-pass resolves it to its Count$ expression
        // and defaults the comparator (ConditionSVarCompare) to GE1 when none is given.
        ability.condition_check_svar = value;
    } else if (key == "ConditionSVarCompare") {
        ability.condition_svar_compare = value;
    } else if (key == "ConditionPresent") {
        ability.condition_present = value;
    } else if (key == "ConditionNotPresent") {
        // Inverted intervening-if (CR 603.4-style): the gated body runs only when the filter is
        // NOT present (Uro's TrigSac: DB$ Sacrifice | ConditionNotPresent$ Card.Self+escaped —
        // sacrifice unless the permanent escaped). Mark it as an intervening-if so it is
        // re-checked when the trigger goes on the stack AND at resolution, and negate the result.
        ability.condition_present = value;
        ability.condition_negate = true;
        ability.intervening_if = true;
    } else if (key == "ConditionDefined") {
        // "Targeted" → the condition is evaluated against the chosen target at
        // resolution, so it must not gate cast-time legality.
        ability.condition_on_target = (value == "Targeted");
        // "Remembered" → condition_present is counted over the remembered cards at
        // resolution (Birthing Ritual: only dig if a creature was sacrificed).
        // "Imprinted" → the same evaluation, over the imprinted card. For the exile-and-return
        // cards (Phelia) the imprinted card IS the returned card already held in
        // cur_game.remembered_entities (RememberObjects$ RememberedLKI / Defined$
        // DelayTriggerRememberedLKI), so it reuses the remembered-set condition path; the
        // redundant Imprint$ True on the preceding ChangeZone is ignored.
        // "RememberedLKI" → the same remembered-set condition path, but the remembered card is
        // now off the battlefield (Boomerang Basics bounced it to hand); its controller qualifier
        // is resolved from last-known information at resolution. See evaluate_present_condition.
        ability.condition_on_remembered = (value == "Remembered" || value == "Imprinted" ||
                                           value == "RememberedLKI");
        // "TriggeredCard" → condition_present is a property check on the ability's source/
        // triggering card (Amped Raptor: Card.wasCastFromYourHandByYou), evaluated at
        // resolution against that card's permanent state.
        ability.condition_on_triggered_card = (value == "TriggeredCard");
    } else if (key == "ConditionCompare") {
        ability.condition_compare = value;
    } else if (key == "ValidCards" && ability.category == "NameCard") {
        // SP$/DB$ NameCard ValidCards$ <filter> — the filter restricting the nameable card
        // set (Petrified Hamlet's Land, Cabal Therapy's Card.nonLand). The name_card handler
        // passes it to build_name_card_choices, which applies it to each candidate via the
        // unified matcher (card_matches_any).
        ability.valid_cards_filter = value;
    } else if (key == "VoteCard") {
        // SP$/AB$ Vote VoteCard$ <filter> (Council's Judgment): the permanent filter the vote
        // chooses among. In the two-player engine the vote handler offers the controller every
        // battlefield permanent matching this filter to choose one. See effect_vote.cpp.
        ability.vote_card_filter = value;
    } else if (key == "SacValid") {
        ability.sac_valid = value;            // DB$ Sacrifice — what may be sacrificed
    } else if (key == "RememberLKI") {
        // RememberLKI$ True on a ChangeZone (Boomerang Basics) — stash the moved object so a
        // paired ConditionDefined$ RememberedLKI gate can read its last-known controller.
        ability.remember_lki = (value == "True");
    } else if (key == "RememberSacrificed") {
        ability.remember_sacrificed = (value == "True");
    } else if (key == "RepeatPlayers") {
        ability.repeat_players = value;       // RepeatEach over players (Price of Progress)
    } else if (key == "Types" && ability.category == "Animate") {
        // DB$ Animate | Types$ Angel [Cleric ...] — the type/subtype list the animated permanent
        // gains "in addition to its other types" (Guide of Souls: "Angel"; The Fantasticar:
        // "Creature,Artifact"). Forge separates this list with COMMAS (unlike the printed Types:
        // line, which is space-separated), so normalize commas to spaces before classifying each
        // token (TYPE/SUBTYPE/SUPERTYPE) the same way the printed Types: line is parsed.
        std::string normalized = value;
        std::replace(normalized.begin(), normalized.end(), ',', ' ');
        for (const auto &t : parse_types(normalized)) ability.animate_types.push_back(t);
    } else if (key == "Duration" && ability.category == "Animate") {
        ability.animate_duration_permanent = (value == "Permanent");
        // Duration$ UntilYourNextTurn (Karn +1): a longer-than-EOT continuous effect that
        // reverts at the start of the animating player's next turn (see effect_animate.cpp).
        ability.animate_duration_until_your_next_turn = (value == "UntilYourNextTurn");
    } else if ((key == "Power" || key == "Toughness") && ability.category == "Animate") {
        // AB$ Animate | Power$/Toughness$ — the animated creature's base P/T. A bare integer is
        // the literal base; any other token is an SVar key resolved post-parse (Karn: Power$ X,
        // X = Targeted$CardManaCost → the target's mana value).
        ability.animate_has_pt = true;
        int *base = (key == "Power") ? &ability.animate_base_power : &ability.animate_base_toughness;
        std::string *tok = (key == "Power") ? &ability.animate_power_token : &ability.animate_toughness_token;
        if (!value.empty() && std::isdigit(static_cast<unsigned char>(value[0])))
            *base = std::stoi(value);
        else
            *tok = value;
    } else if (key == "Duration" && value == "UntilHostLeavesPlay") {
        // Duration$ UntilHostLeavesPlay on a ChangeZone | Destination$ Exile (CR 603.6e): the
        // exiled card(s) return when the ability's host leaves the battlefield. See
        // effects::register_exile_until_host_leaves.
        ability.duration_until_host_leaves = true;
    } else if (key == "Duration" && value == "UntilYourNextTurn") {
        // Generic "until your next turn" duration on a non-Animate effect (The One Ring's ETB Pump
        // granting the controller protection from everything). Reverted at the controller's untap.
        ability.duration_until_your_next_turn = true;
    } else if (effects::apply_parse_hook(ability, key, value)) {
        // Consumed by an effect-specific parse hook co-located with its handler.
    } else {
        static const std::set<std::string> ignored_keys = {
            "SpellDescription", "AILogic", "AINoRecursiveCheck", "TgtPrompt", "StackDescription",
            "ConditionDescription",
            // AB$ Effect emblem (Kaito's [+1]): Name$ is the emblem's display name and Image$ its
            // art — both cosmetic. Duration$ is read load-bearingly from the raw line in
            // parse_abilities (Duration$ Permanent ⇒ the Effect makes a permanent emblem); the
            // Animate/UntilHostLeavesPlay/UntilYourNextTurn Duration forms are consumed by their
            // own branches above, so any Duration reaching here is already handled or cosmetic.
            "Name", "Image", "Duration",
            // SelectPrompt$ — the prose prompt shown when a ChangeZone effect asks the player to
            // pick which permanents to move (Yorion: "Select any number of other nonland
            // permanents..."). Purely cosmetic, like TgtPrompt; the selection is driven by
            // ChangeType$/Origin$/Destination$.
            "SelectPrompt",
            // ValidTgtsDesc$ — the prose name of a target restriction shown to the player (e.g.
            // Cloak and Dagger's "creature controlled by the targeted opponent"). Purely cosmetic;
            // the load-bearing restriction is ValidTgts$, parsed above.
            "ValidTgtsDesc",
            // PrecostDesc$ — the reminder-text prefix Forge prints before an activated
            // ability's cost (e.g. "Metalcraft —" on Mox Opal). Purely cosmetic; the
            // load-bearing gate is Activation$ (parsed above).
            "PrecostDesc",
            // TriggerDescription$ — reminder/Oracle prose on an AB$ ImmediateTrigger's reflexive
            // ability (Guide of Souls). Purely cosmetic, like SpellDescription/StackDescription.
            "TriggerDescription",
            // sameName search/move (Surgical Extraction, Extirpate, ...): these refine who
            // chooses or how the search is hidden, but the change_zone_same_name handler
            // already derives the full behavior from ChangeType/Origin/Destination/Defined.
            // Hidden/ForgetOtherTargets are cosmetic given the "move the maximum" simplification.
            // (Chooser$ is parsed above into chooser_is_controller — the search-based ChangeZone
            // honors Chooser$ You. Shuffle$ is handled in apply_param_to_ability above.)
            "Hidden", "ForgetOtherTargets",
            // ForgetOnMoved$ Exile (Ugin, Eye of the Storms' -11 Effect): tells Forge to drop a
            // remembered object from the effect once it leaves the named zone. Bookkeeping for the
            // transient free-cast grant only; the grant itself is a no-op here, so this is cosmetic.
            "ForgetOnMoved",
            // ChooseCard ChooseEach (Ajani -4): the per-type breakdown is the load-bearing
            // ChooseEach$; Choices$ (the umbrella pool), ControlledByPlayer$ Chooser, and
            // Reveal$ are captured by / cosmetic to the choose_each handler.
            "Choices", "ControlledByPlayer", "Reveal",
            // Ultimate$ True is informational: ultimate legality is already covered by the
            // minus-loyalty cost check, so the flag is unused.
            "Ultimate",
            // AB$ Effect | Triggers$ <SVar> (Tamiyo, Seasoned Scholar's +2): the floating triggered
            // ability is resolved in the parse_abilities post-loop (where svars are in scope), not
            // in apply_param_to_ability — so suppress the spurious "unrecognized" warning here.
            "Triggers",
            // Stackable$ False on an emblem-making Effect (Tamiyo's ultimate): tells Forge not to
            // create a second identical emblem. We don't model emblem de-duplication; cosmetic.
            "Stackable",
            // ForgetOtherRemembered$ True on a Defined$ Remembered ChangeZone (Tamiyo's front
            // exile-and-return-transformed): drops the OTHER remembered objects. Only the single
            // returned card is ever remembered here, so this is a no-op — cosmetic.
            "ForgetOtherRemembered",
            // DamageMap$ True (RepeatEach DealDamage, e.g. Price of Progress): a Forge
            // bookkeeping flag that the per-iteration damage is collected into one
            // simultaneous damage event. Our resolution deals each player's damage in the
            // repeat loop; the simultaneity is cosmetic for a one-shot instant.
            "DamageMap",
            // Announce$ X (Kozilek's Command): declares the X to announce while casting. The
            // X cost is already auto-detected from a ManaCost containing X (has_x_cost), so
            // the announce is prompted regardless; the tag is informational here.
            "Announce",
            // SP$ NameCard ValidCards$ Card.nonLand / ValidDescription$ nonland (Cabal
            // Therapy): the name_card handler already restricts the candidate set to nonland
            // vocab cards, so the filter spec and its prose are informational here.
            "ValidCards", "ValidDescription",
            // Imprint$ True on an exile-and-return ChangeZone (Phelia): the returned card is
            // already tracked in cur_game.remembered_entities (RememberObjects$ RememberedLKI),
            // which the paired ConditionDefined$ Imprinted gate reads, so the imprint is redundant.
            // ClearImprinted$ True on the paired DB$ Cleanup is likewise redundant — the imprint
            // set is the remembered set, cleared by the same Cleanup's ClearRemembered$ True.
            "Imprint", "ClearImprinted",
            // SP$/AB$ Vote VoteMessage$ <text> (Council's Judgment): the prose shown to voters
            // ("for a nonland permanent you don't control"). Purely cosmetic — the load-bearing
            // VoteCard$ filter and VoteSubAbility$ are parsed above.
            "VoteMessage",
            // DB$ Sacrifice SacMessage$ <text> (Pick Your Poison): the prose naming what is
            // sacrificed ("creature with flying"). Purely cosmetic — the load-bearing SacValid$
            // filter is parsed above.
            "SacMessage",
            // ChangeTypeDesc$ <text> (Prismatic Vista / Price of Freedom): the prose name of the
            // searched-for type ("basic land"). Purely cosmetic — the load-bearing ChangeType$
            // filter is parsed above.
            "ChangeTypeDesc",
            // ShuffleNonMandatory$ True (Price of Freedom): marks the post-search shuffle as
            // optional on a fail-to-find. The search-based ChangeZone already shuffles after a
            // library fetch (the found case, which is all this card does in practice), so the
            // flag adds nothing the handler doesn't already do.
            "ShuffleNonMandatory",
            // ── AI-only / cosmetic params — no rules impact, safely ignored ──────────────
            // AITgts$ <filter> (Into the Flood Maw, Pyroblast, Hydroblast, Fatal Push) and
            // AIXMax$ <svar> (Green Sun's Zenith): hints that steer Forge's own AI (which
            // target to pick / how big an X to pay). Our engine picks targets and X itself,
            // so these are advisory only and irrelevant to the modeled rules.
            "AITgts", "AIXMax",
            // GiftDescription$ <text> (Into the Flood Maw) and ChangeValidDesc$ <text>
            // (Once Upon a Time): prose describing the gift offered / the ChangeZone filter.
            // Purely cosmetic — the load-bearing filters are parsed from the other params.
            "GiftDescription", "ChangeValidDesc",
            // ForceRevealToController$ True (Once Upon a Time): reveals the looked-at card to
            // its own controller. Informational in a perfect-information engine.
            "ForceRevealToController",
            // ── Params for mechanics NOT YET MODELED — currently no-ops, tracked in todo.md ─
            // Suppressed here to keep the log clean; each still needs a real handler (see the
            // "Unrecognized ability params suppressed but unimplemented" section of todo.md):
            //   Reorder$ True (Brainstorm) — let the player order the cards put back on top.
            //   TriggerAmount$ / RememberOriginalTokens$ (Ajani, Nacatl Avenger) — the token
            //     count carried to the transform trigger, and tracking the original tokens.
            //   LockTokenScript$ True (Into the Flood Maw) — pin the gifted Fish token's script.
            //   ExileOnMoved$ Battlefield (Manifold Key) — exile the permanent when it moves.
            "Reorder", "TriggerAmount", "RememberOriginalTokens", "LockTokenScript",
            "ExileOnMoved"
        };
        if (ignored_keys.find(key) == ignored_keys.end()) {
            std::string msg = "Unrecognized ability param: " + key + "$ " + value;
            if (!card_name.empty()) msg += " (card: " + card_name + ")";
            warning(msg);
        }
    }
}

// Normalizes script category names to the internal names used throughout the engine.
static std::string normalize_category(std::string category) {
    if (category == "Mana") category = "AddMana";
    return category;
}

// Resolves a TargetMin$/TargetMax$ that was given as an SVar key (stashed during param parsing)
// to its runtime meaning. When the SVar resolves to Count$xPaid the bound equals the X paid at
// cast/activation (CR 601.2b chooses X before targets, so x_paid is known when targets are
// selected). Setting BOTH target_min_from_xpaid and target_max_from_xpaid yields EXACTLY-X
// targeting (Candelabra of Tawnos, Hide on the Ceiling); a lone TargetMax$ X gives "up to X"
// (Kozilek's Command). Other count-SVar caps keep the "effectively unlimited" fallback already
// stored by apply_param_to_ability. Shared by the top-level and sub-ability parse paths.
static void resolve_xpaid_target_counts(Ability& ability,
                                        const std::map<std::string, std::string>& svars) {
    if (!ability.target_min_svar.empty()) {
        auto it = svars.find(ability.target_min_svar);
        if (it != svars.end()) {
            if (it->second.find("xPaid") != std::string::npos)
                ability.target_min_from_xpaid = true;
            else if (it->second.find("Count$") != std::string::npos)
                // A non-xPaid count-SVar minimum (Into the Flood Maw: X = Count$PromisedGift.0.1).
                // select_target evaluates it at cast and stamps target_min; default it to 0 now so
                // cast-time legality treats it as optional rather than over-requiring a target.
                { ability.target_min_count_expr = it->second; ability.target_min = 0; }
        }
    }
    if (!ability.target_max_svar.empty()) {
        auto it = svars.find(ability.target_max_svar);
        if (it != svars.end()) {
            if (it->second.find("xPaid") != std::string::npos)
                ability.target_max_from_xpaid = true;
            else if (it->second.find("Count$") != std::string::npos)
                // A non-xPaid count-SVar cap; select_target evaluates it at cast and stamps the
                // real max (0 → the ability targets nothing and does nothing).
                ability.target_max_count_expr = it->second;
        }
    }
}

// Resolves a Pump/PumpAll NumAtt$/NumDef$ count-SVar key (e.g. "X" or "-X" → att_expr/def_expr
// "X") to its runtime Count$ expression (e.g. Count$xPaid), so the pump effect can evaluate the
// signed magnitude at resolution (Toxic Deluge's -X/-X; Eldrazi Linebreaker's +X). The sign was
// captured separately (att_sign/def_sign) by parse_pump_amount. Shared by both parse paths.
static void resolve_pump_exprs(Ability& ability,
                               const std::map<std::string, std::string>& svars) {
    if (auto *pp = std::get_if<PumpParams>(&ability.params)) {
        for (std::string *expr : {&pp->att_expr, &pp->def_expr}) {
            if (expr->empty()) continue;
            auto it = svars.find(*expr);
            if (it != svars.end()) *expr = it->second;
        }
    }
}

// Forward declaration so parse_svar_ability can recurse via SubAbility$.
static Ability parse_svar_ability(const std::string& content, Ability::AbilityType ability_type,
                                  const std::map<std::string, std::string>& svars,
                                  const std::string& card_name = "");

// Parses a SVar's DB$ content string into an Ability. Resolves SubAbility$ chains.
static Ability parse_svar_ability(const std::string& content, Ability::AbilityType ability_type,
                                  const std::map<std::string, std::string>& svars,
                                  const std::string& card_name) {
    Ability sub;
    sub.ability_type = ability_type;
    // An Execute$/SubAbility$ SVar normally holds a DB$ ability, but some hold an AB$
    // (e.g. Guide of Souls' TrigImmediateTrig: "AB$ ImmediateTrigger | Cost$ PayEnergy<3>").
    // All three prefixes (DB$/AB$/SP$) are 3 chars + a space, so the category offset is the
    // same; accept whichever leads the content.
    size_t db_pos = content.find("DB$");
    if (db_pos == std::string::npos) db_pos = content.find("AB$");
    if (db_pos == std::string::npos) db_pos = content.find("SP$");
    if (db_pos == std::string::npos) return sub;
    size_t p = db_pos + 4;  // skip the "XX$ " prefix
    size_t cat_end = content.find_first_of(" |", p);
    if (cat_end == std::string::npos) cat_end = content.length();
    if (cat_end > p)
        sub.category = normalize_category(content.substr(p, cat_end - p));

    // Earthbend (CR keyword action) inherently targets a land the controller controls; the
    // Forge scripts carry no ValidTgts$, so default it here (overridden if the script ever
    // states one explicitly).
    if (sub.category == "Earthbend") sub.valid_tgts = "Land.YouCtrl";

    size_t param_pos = content.find("|", p);
    std::string key, value;
    while (next_param(content, param_pos, key, value)) {
        if (key == "SubAbility") {
            auto it = svars.find(value);
            if (it != svars.end())
                sub.subabilities.push_back(parse_svar_ability(it->second, ability_type, svars, card_name));
        } else if (key == "Choices") {
            // Charm modal in DB$ context (e.g. Knight of Autumn ETB)
            size_t cpos = 0;
            while (cpos < value.size()) {
                size_t comma = value.find(',', cpos);
                if (comma == std::string::npos) comma = value.size();
                std::string svar_name = value.substr(cpos, comma - cpos);
                auto cit = svars.find(svar_name);
                if (cit != svars.end()) {
                    Ability choice = parse_svar_ability(cit->second, ability_type, svars, card_name);
                    std::string desc;
                    size_t sd = cit->second.find("SpellDescription$");
                    if (sd != std::string::npos) {
                        sd += 17;
                        while (sd < cit->second.size() && cit->second[sd] == ' ') sd++;
                        size_t de = cit->second.find('|', sd);
                        if (de == std::string::npos) de = cit->second.size();
                        desc = cit->second.substr(sd, de - sd);
                        while (!desc.empty() && desc.back() == ' ') desc.pop_back();
                    }
                    sub.charm_choices.push_back(choice);
                    sub.charm_choice_descriptions.push_back(desc);
                }
                cpos = comma + 1;
            }
        } else if (key == "CharmNum") {
            sub.charm_num = std::stoi(value);
        } else if (key == "Abilities" && sub.category == "Animate") {
            // DB$ Animate | Abilities$ <svar>[,<svar>...] — the activated ability(ies) the Animate
            // grants to the target permanent (Urza's Saga: ABMana "{T}: Add {C}." / ABToken
            // "{2},{T}: Create a Construct"). Each named SVar is a self-contained AB$ ability;
            // parse it as an ACTIVATED ability and store it for the Animate handler to attach.
            for (const std::string &svar_name : split(value, ',', /*skip_empty=*/true)) {
                auto it = svars.find(svar_name);
                if (it != svars.end())
                    sub.animate_granted_abilities.push_back(
                        parse_svar_ability(it->second, Ability::ACTIVATED, svars, card_name));
            }
        } else if (key == "Execute") {
            // Execute$ references an SVar containing the ability to fire (delayed triggers)
            effect_params<DelayedTriggerParams>(sub).execute_svar = value;
            auto it = svars.find(value);
            if (it != svars.end()) {
                Ability exec = parse_svar_ability(it->second, ability_type, svars, card_name);
                exec.from_delayed_execute = true;  // delayed_trigger() fires this one
                sub.subabilities.push_back(exec);
            }
        } else if (key == "Triggers") {
            // DB$ Effect | Triggers$ <SVar>[,<SVar>...] — a transient until-end-of-turn floating
            // triggered ability (Forth Eorlingas!). Each named SVar holds a trigger line
            // (Mode$ ... | Execute$ ...); parse it like a card's T: line so it carries the same
            // trigger metadata and its Execute$ effect, and store it on the Effect to be
            // registered (controller-bound) into cur_game.floating_triggers at resolution.
            for (const std::string &svar_name : split(value, ',', /*skip_empty=*/true)) {
                auto it = svars.find(svar_name);
                if (it != svars.end()) {
                    Ability trig = parse_one_trigger(it->second, svars, card_name);
                    if (trig.trigger_on != 0) sub.effect_floating_triggers.push_back(trig);
                }
            }
        } else if (key == "ReplacementEffects") {
            // DB$ Effect | ReplacementEffects$ <SVar> (Veil of Summer: AntiMagic =
            // "Event$ Counter | ValidSA$ Spell.YouCtrl | Layer$ CantHappen"). A turn-long
            // "spells you control can't be countered" grant. Detect the CantHappen counter
            // replacement on the controller's spells and flag it; the GrantCast handler records
            // the controller in cur_game.cant_counter_spells_of for the rest of the turn.
            auto it = svars.find(value);
            if (it != svars.end()) {
                const std::string &body = it->second;
                if (body.find("Event$ Counter") != std::string::npos &&
                    body.find("CantHappen") != std::string::npos &&
                    body.find("YouCtrl") != std::string::npos)
                    sub.effect_spells_uncounterable_this_turn = true;
            }
        } else if (key == "StaticAbilities") {
            // DB$ Effect | StaticAbilities$ <name>. The value may be a literal keyword
            // (Unblockable) or a named SVar holding a continuous static-ability line. Keep the
            // raw value (the Unblockable path reads it), and additionally resolve a named SVar to
            // detect the "may cast those exiled cards without paying their mana costs" grant
            // (Ugin -11: MayPlay$ True + MayPlayWithoutManaCost$ True + AffectedZone$ Exile),
            // which the GrantCast handler turns into free cast-from-exile permissions.
            sub.effect_static_ability = value;
            auto it = svars.find(value);
            if (it != svars.end() &&
                it->second.find("MayPlayWithoutManaCost$ True") != std::string::npos &&
                it->second.find("AffectedZone$ Exile") != std::string::npos)
                sub.effect_grant_free_cast_from_exile = true;
        } else if (key == "ConditionCheckSVar") {
            // Resolve SVar reference to its expression (e.g. "X" → "Count$ResolvedThisTurn")
            auto it = svars.find(value);
            sub.condition_check_svar = (it != svars.end()) ? it->second : value;
        } else if (key == "ConditionSVarCompare") {
            sub.condition_svar_compare = value;
        } else if (key == "TargetMax" && !value.empty() &&
                   !std::isdigit(static_cast<unsigned char>(value[0]))) {
            // A count-SVar cap. If it resolves to Count$xPaid (Kozilek's Command:
            // "up to X target cards"), the true cap is the X paid at cast — flag it so
            // select_target can clamp the multi-target loop to x_paid at resolution.
            // Other count-SVar caps fall back to "any number" (MAX_ENTITIES).
            auto it = svars.find(value);
            if (it != svars.end() && it->second.find("xPaid") != std::string::npos)
                sub.target_max_from_xpaid = true;
            sub.target_max = MAX_ENTITIES;
            // Stash the SVar key so resolve_xpaid_target_counts can also resolve a non-xPaid
            // count-SVar cap (Into the Flood Maw's DBChangeZone: TargetMax$ Y = Count$PromisedGift)
            // into target_max_count_expr, evaluated at cast (0 → targets nothing).
            sub.target_max_svar = value;
        } else {
            apply_param_to_ability(sub, key, value, card_name);
        }
    }
    // DB$ Effect | StaticAbilities$ <SVar> | Duration$ Permanent — an emblem-making SUB-ability
    // (Tamiyo, Seasoned Scholar's ultimate DBEmblem: "You get an emblem with 'You have no maximum
    // hand size.'"). Mirror the top-level emblem resolution in parse_abilities so a sub-ability
    // Effect also carries its permanent continuous static into effect_emblem_statics, which the
    // GrantCast handler turns into a player-owned emblem. General over any emblem-making sub-Effect.
    if (sub.category == "Effect" && !sub.effect_static_ability.empty() &&
        content.find("Duration$ Permanent") != std::string::npos) {
        auto it = svars.find(sub.effect_static_ability);
        if (it != svars.end() && it->second.find("Mode$ Continuous") != std::string::npos) {
            StaticAbility est = parse_one_static_ability(it->second, svars);
            if (!est.category.empty()) sub.effect_emblem_statics.push_back(est);
        }
    }
    // Resolve amount_svar through SVars map (same logic as parse_abilities)
    if (!sub.amount_svar.empty()) {
        auto it = svars.find(sub.amount_svar);
        if (it != svars.end()) {
            const std::string &sv = it->second;
            // TriggerCount$DamageAmount → use combat damage trigger's damage amount at runtime
            if (sv == "TriggerCount$DamageAmount") {
                sub.amount_from_damage = true;
                sub.amount_svar = "";
                return sub;
            }
            // Delirium-conditional value: Count$Delirium.<yes>.<no> or Count$...GE4.<yes>.<no>.
            size_t delirium_pos = sv.find("Count$Delirium");
            size_t ge_pos = sv.find("GE");
            size_t scale_pos = delirium_pos != std::string::npos
                                   ? delirium_pos + std::string("Count$Delirium").size()
                                   : (ge_pos != std::string::npos ? ge_pos + 2 : std::string::npos);
            if (scale_pos != std::string::npos) {
                std::string rest = sv.substr(scale_pos);
                size_t d1 = rest.find('.');
                if (d1 != std::string::npos) {
                    size_t d2 = rest.find('.', d1 + 1);
                    if (d2 != std::string::npos) {
                        sub.amount = static_cast<size_t>(std::stoi(rest.substr(d1 + 1, d2 - d1 - 1)));
                        DamageParams &dp = effect_params<DamageParams>(sub);
                        dp.delirium_amount = static_cast<size_t>(std::stoi(rest.substr(d2 + 1)));
                        dp.is_delirium_scale = true;
                    }
                }
            } else if (sv.find("Count$Valid") != std::string::npos ||
                       sv.find("Targeted$") != std::string::npos ||
                       sv.find("Count$InYourLibrary") != std::string::npos ||
                       sv.find("Count$YourLifeTotal") != std::string::npos ||
                       // Remembered$Valid <filter> — count of remembered (e.g. just-moved by a
                       // RememberChanged$ ChangeZoneAll) cards matching the filter (Canoptek
                       // Scarab Swarm: TokenAmount$ X, X = Remembered$Valid Land,Artifact).
                       sv.find("Remembered$") != std::string::npos ||
                       // Count$xPaid — amount equals the X paid at cast (Kozilek's Command:
                       // TokenAmount$/ScryNum$/TargetMax$ all = X = Count$xPaid).
                       sv.find("xPaid") != std::string::npos ||
                       // Count$CardCounters.<TYPE> — counters on the source permanent (The One
                       // Ring: NumCards$/LifeAmount$ X = Count$CardCounters.BURDEN). Evaluated at
                       // resolution against the source by evaluate_dynamic_amount.
                       sv.find("Count$CardCounters") != std::string::npos) {
                sub.dynamic_amount_expr = sv;
            }
        }
        sub.amount_svar = "";
    }
    // Resolve a Pump NumAtt$/NumDef$ given as a count-SVar (e.g. "+X", X = Count$Valid
    // Eldrazi.YouCtrl) to its runtime Count$ expression (Eldrazi Linebreaker), and a
    // TargetMin$/TargetMax$ SVar to its exactly-X / up-to-X meaning.
    resolve_pump_exprs(sub, svars);
    resolve_xpaid_target_counts(sub, svars);
    // Resolve dig_num_expr SVar reference (e.g. "X" → "Count$Devotion.Blue")
    if (!sub.dig_num_expr.empty()) {
        auto it = svars.find(sub.dig_num_expr);
        if (it != svars.end()) {
            sub.dig_num_expr = it->second;
        }
    }
    // ChooseNumber Max$ and a dynamic CounterNum$ both stash a raw SVar token (Wrath of the
    // Skies: Max$ Max → Count$YourCountersEnergy; CounterNum$ X → Count$xPaid). Resolve those
    // SVar references to their runtime Count$ expressions so the effect can evaluate them at
    // resolution.
    if (sub.category == "ChooseNumber" && !sub.dynamic_amount_expr.empty()) {
        auto it = svars.find(sub.dynamic_amount_expr);
        if (it != svars.end()) sub.dynamic_amount_expr = it->second;
    }
    if (auto *cp = std::get_if<CounterParams>(&sub.params)) {
        if (!cp->count_expr.empty()) {
            auto it = svars.find(cp->count_expr);
            if (it != svars.end()) cp->count_expr = it->second;
        }
    }
    // TokenPower$/TokenToughness$ given as an SVar token (Skyclave Apparition: "X" →
    // Remembered$CardManaCost): resolve the reference to its runtime expression so the Token
    // effect can size the created token's P/T at creation time.
    if (auto *tkp = std::get_if<TokenParams>(&sub.params)) {
        for (std::string *expr : {&tkp->power_expr, &tkp->toughness_expr}) {
            if (expr->empty()) continue;
            auto it = svars.find(*expr);
            if (it != svars.end()) *expr = it->second;
        }
    }
    // DestroyAll with a dynamic mana-value bound and/or an energy unless-cost (Wrath of the
    // Skies): resolve the "cmcLE<SVar>" threshold from the ValidCards$ filter and the
    // PayEnergy<SVar> amount into their runtime Count$ expressions. The numeric/cmcLEX legacy
    // paths are unchanged; this only fires when the SVar is non-numeric (e.g. cmcLEY, Y =
    // Count$ChosenNumber).
    if (sub.category == "DestroyAll") {
        auto &dp = effect_params<DestroyAllParams>(sub);
        if (dp.cmc_expr.empty() && !sub.valid_cards_filter.empty()) {
            for (const char *op : {"cmcEQ", "cmcLE", "cmcGE", "cmcLT", "cmcGT", "cmcNE"}) {
                size_t pos = sub.valid_cards_filter.find(op);
                if (pos == std::string::npos) continue;
                std::string svar_ref = sub.valid_cards_filter.substr(pos + 5);
                size_t end = 0;
                while (end < svar_ref.size() &&
                       (std::isalnum(static_cast<unsigned char>(svar_ref[end])) || svar_ref[end] == '_'))
                    end++;
                svar_ref = svar_ref.substr(0, end);
                // A pure-numeric or "X" bound stays on the legacy path (handled at resolution);
                // only a named SVar resolving to a Count$ expression routes here.
                if (svar_ref.empty() || svar_ref == "X") break;
                auto it = svars.find(svar_ref);
                if (it != svars.end()) {
                    dp.cmc_expr = it->second;
                    dp.cmc_op = std::string(op + 3);  // "cmcLE" → "LE"
                }
                break;
            }
        }
        if (!dp.energy_unless_expr.empty()) {
            auto it = svars.find(dp.energy_unless_expr);
            if (it != svars.end()) dp.energy_unless_expr = it->second;
        }
    }
    // Resolve a cmcLE<SVar> threshold inside ChangeValid$ (Birthing Ritual: "Creature.cmcLEX")
    // into dynamic_amount_expr, evaluated by the Dig effect at resolution.
    if (sub.dynamic_amount_expr.empty() && !sub.change_valid.empty()) {
        size_t lex = sub.change_valid.find("cmcLE");
        if (lex != std::string::npos) {
            std::string svar_ref = sub.change_valid.substr(lex + 5);
            size_t end = 0;
            while (end < svar_ref.size() &&
                   (std::isalpha(static_cast<unsigned char>(svar_ref[end])) || svar_ref[end] == '_'))
                end++;
            svar_ref = svar_ref.substr(0, end);
            auto it = svars.find(svar_ref);
            if (it != svars.end()) sub.dynamic_amount_expr = it->second;
        }
    }
    // Resolve ConditionSVarCompare$ when RHS is an SVar reference (e.g. "LEX" where X = "Count$Devotion.Blue")
    if (sub.condition_svar_compare.size() >= 3) {
        std::string rhs_str = sub.condition_svar_compare.substr(2);
        // If RHS is not a pure integer, it might be an SVar reference
        if (!rhs_str.empty() && !std::isdigit(rhs_str[0]) && rhs_str[0] != '-') {
            auto it = svars.find(rhs_str);
            if (it != svars.end()) {
                sub.condition_compare_svar_expr = it->second;
                sub.condition_svar_compare = sub.condition_svar_compare.substr(0, 2);  // keep just "LE"
            }
        }
    }
    return sub;
}

// Public entry: parse one activated/spell ability body (the resolved RHS of an SVar, e.g.
// "AB$ Mana | Cost$ T | Produced$ C") into an Ability, honouring the full Forge ability
// grammar via parse_svar_ability. Used to materialize an AddAbility$ static's granted
// ability (Petrified Hamlet). No SVar table is available at the grant site, so an empty map
// is passed; the granted bodies in use are self-contained (no SVar references).
Ability parse_ability_body(const std::string &body, Ability::AbilityType type) {
    static const std::map<std::string, std::string> kNoSvars;
    return parse_svar_ability(body, type, kNoSvars, "");
}

// Resolves an additive SVar chain (e.g. "SVar$Z1/Plus.Z2") into the list of
// runtime Count$ expressions to be summed at resolution. Each "SVar$<name>"
// token is looked up in `svars`; if its value is itself an SVar$ chain it is
// resolved recursively, otherwise the raw Count$ expression is collected as a
// term. A non-SVar$ expression is returned as a single term. Used for the
// conditional take-count of Flow State (Y = Z1 + Z2, each a capped yard count).
static void resolve_additive_svar(const std::string& expr, const std::map<std::string, std::string>& svars,
                                  std::vector<std::string>& terms) {
    // A leaf runtime expression (not an SVar$ reference) is collected as one term.
    if (expr.rfind("SVar$", 0) != 0) {
        terms.push_back(expr);
        return;
    }
    // Head term: the name between "SVar$" and the first '/' (or end of string).
    size_t head_end = expr.find('/', 5);
    std::string head = (head_end == std::string::npos) ? expr.substr(5) : expr.substr(5, head_end - 5);
    auto hit = svars.find(head);
    if (hit != svars.end()) resolve_additive_svar(hit->second, svars, terms);
    // Each subsequent "/Plus.<name>" segment adds another (bare) SVar term.
    size_t plus = (head_end == std::string::npos) ? std::string::npos : expr.find("/Plus.", head_end);
    while (plus != std::string::npos) {
        size_t name_start = plus + 6;  // skip "/Plus."
        size_t name_end = expr.find('/', name_start);
        std::string name = (name_end == std::string::npos) ? expr.substr(name_start)
                                                           : expr.substr(name_start, name_end - name_start);
        auto it = svars.find(name);
        if (it != svars.end()) resolve_additive_svar(it->second, svars, terms);
        plus = (name_end == std::string::npos) ? std::string::npos : expr.find("/Plus.", name_end);
    }
}

// fed each ability line
static std::vector<Ability> parse_abilities(std::vector<std::string> lines, const std::set<Type>& types,
                                            const std::map<std::string, std::string>& svars,
                                            const std::string& card_name) {
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

        // Earthbend (CR keyword action) inherently targets a land the controller controls; the
        // activated form (Ba Sing Se) carries no ValidTgts$, so default it here.
        if (ability.category == "Earthbend") ability.valid_tgts = "Land.YouCtrl";

        // Parse pipe-delimited parameters — applies to all ability categories
        size_t param_pos = line.find("|", pos);
        std::string key, value;
        while (next_param(line, param_pos, key, value)) {
            if (key == "SubAbility" || key == "RepeatSubAbility" || key == "VoteSubAbility") {
                // RepeatSubAbility$ (RepeatEach) resolves the same way as SubAbility$: the
                // value names an SVar holding a DB$ ability. For RepeatEach the parsed
                // sub-ability is the per-iteration body the handler resolves once per player.
                // VoteSubAbility$ (Vote, Council's Judgment) likewise names an SVar holding the
                // DB$ ChangeZone applied to the voted-for permanent; the vote handler remembers
                // the chosen permanent so this Defined$ Remembered sub-ability exiles it.
                auto it = svars.find(value);
                if (it != svars.end())
                    ability.subabilities.push_back(parse_svar_ability(it->second, ability.ability_type, svars, card_name));
            } else if (key == "Choices") {
                // Charm modal: resolve comma-separated SVar names into sub-abilities
                size_t cpos = 0;
                while (cpos < value.size()) {
                    size_t comma = value.find(',', cpos);
                    if (comma == std::string::npos) comma = value.size();
                    std::string svar_name = value.substr(cpos, comma - cpos);
                    auto it = svars.find(svar_name);
                    if (it != svars.end()) {
                        Ability choice = parse_svar_ability(it->second, ability.ability_type, svars, card_name);
                        // Extract SpellDescription from the choice for display
                        std::string desc;
                        size_t sd = it->second.find("SpellDescription$");
                        if (sd != std::string::npos) {
                            sd += 17; // skip "SpellDescription$"
                            while (sd < it->second.size() && it->second[sd] == ' ') sd++;
                            size_t de = it->second.find('|', sd);
                            if (de == std::string::npos) de = it->second.size();
                            desc = it->second.substr(sd, de - sd);
                            while (!desc.empty() && desc.back() == ' ') desc.pop_back();
                        }
                        ability.charm_choices.push_back(choice);
                        ability.charm_choice_descriptions.push_back(desc);
                    }
                    cpos = comma + 1;
                }
            } else if (key == "CharmNum") {
                ability.charm_num = std::stoi(value);
            } else {
                apply_param_to_ability(ability, key, value, card_name);
            }
        }
        // Fatal Push pattern: ConditionPresent "Creature.cmcLE<SVar>" references an SVar
        // (X = Count$Revolt.4.2) for the cmc threshold but sets no Amount/NumDmg, so
        // amount_svar would be empty and the revolt-scaled threshold never resolves.
        // Wire the referenced SVar into amount_svar so the block below resolves it into
        // dynamic_amount_expr (evaluated at resolution by effects::destroy).
        if (ability.amount_svar.empty()) {
            size_t lex = ability.condition_present.find("cmcLE");
            if (lex != std::string::npos) {
                std::string svar_ref = ability.condition_present.substr(lex + 5);
                size_t end = 0;
                while (end < svar_ref.size() &&
                       (std::isalpha(static_cast<unsigned char>(svar_ref[end])) || svar_ref[end] == '_'))
                    end++;
                svar_ref = svar_ref.substr(0, end);
                if (!svar_ref.empty() && svars.find(svar_ref) != svars.end())
                    ability.amount_svar = svar_ref;
            }
        }
        // Aether Vial pattern: a ChangeType search filter whose mana-value bound is dynamic,
        // e.g. "Creature.cmcEQX+YouCtrl" with SVar:X:Count$CardCounters.CHARGE. Resolve the
        // "cmcEQ<svar>"/"cmcLE<svar>" SVar reference to its runtime Count$ expression and stash
        // it (with the comparator) so the ChangeZone search can gate hand cards by mana value
        // == the source's charge-counter count at resolution time.
        if (ability.change_type_cmc_expr.empty() && !ability.change_type.empty()) {
            for (const char *op : {"cmcEQ", "cmcLE", "cmcGE", "cmcLT", "cmcGT", "cmcNE"}) {
                size_t pos = ability.change_type.find(op);
                if (pos == std::string::npos) continue;
                std::string svar_ref = ability.change_type.substr(pos + 5);
                size_t end = 0;
                while (end < svar_ref.size() &&
                       (std::isalpha(static_cast<unsigned char>(svar_ref[end])) || svar_ref[end] == '_'))
                    end++;
                svar_ref = svar_ref.substr(0, end);
                auto it = svars.find(svar_ref);
                if (it != svars.end()) {
                    ability.change_type_cmc_expr = it->second;
                    ability.change_type_cmc_op = std::string(op + 3);  // "cmcEQ" → "EQ"
                }
                break;
            }
        }
        // Resolve a top-level ConditionCheckSVar$ reference (Veil of Summer's SP$ Draw: "X" →
        // "Count$ThisTurnCast_Card.OppCtrl+Blue,Card.OppCtrl+Black") to its Count$ expression, and
        // default the comparator to GE1 (Forge's default for a bare ConditionCheckSVar with no
        // ConditionSVarCompare — the value must be >= 1). Sub-ability ConditionCheckSVars are
        // resolved separately in parse_svar_ability and keep their existing behavior.
        if (!ability.condition_check_svar.empty()) {
            auto it = svars.find(ability.condition_check_svar);
            if (it != svars.end()) ability.condition_check_svar = it->second;
            if (ability.condition_svar_compare.empty()) ability.condition_svar_compare = "GE1";
        }
        // Resolve amount_svar for delirium-conditional damage (Unholy Heat pattern).
        // SVar:X:Count$Compare Y GE4.6.2 where Y resolves to a graveyard card-type count.
        // Also handles runtime SVar expressions: Count$Valid ..., Targeted$CardPower
        if (!ability.amount_svar.empty()) {
            auto it = svars.find(ability.amount_svar);
            if (it != svars.end()) {
                const std::string &sv = it->second;
                // Generalized conditional amount (Flow State): "Count$Compare <Var>
                // <op><n>.<t>.<f>" where <Var> is an SVar that resolves to a sum of
                // (capped) runtime counts. The effective amount is <t> when the summed
                // counts satisfy the compare, else <f>. Distinguished from the delirium
                // GE form below by <Var> being a nested SVar$ chain (e.g. SVar$Z1/Plus.Z2)
                // rather than a direct Count$ expression.
                bool handled_conditional = false;
                if (sv.rfind("Count$Compare ", 0) == 0) {
                    std::string rest = sv.substr(14);  // "Y GE2.2.1"
                    size_t sp = rest.find(' ');
                    if (sp != std::string::npos) {
                        std::string var = rest.substr(0, sp);    // "Y"
                        std::string tail = rest.substr(sp + 1);  // "GE2.2.1"
                        auto vit = svars.find(var);
                        if (vit != svars.end() && vit->second.rfind("SVar$", 0) == 0) {
                            size_t d2 = tail.rfind('.');
                            size_t d1 = (d2 == std::string::npos || d2 == 0)
                                            ? std::string::npos : tail.rfind('.', d2 - 1);
                            if (d1 != std::string::npos) {
                                std::vector<std::string> terms;
                                resolve_additive_svar(vit->second, svars, terms);
                                if (!terms.empty()) {
                                    ability.cond_amount_active = true;
                                    ability.cond_amount_exprs = terms;
                                    ability.cond_amount_compare = tail.substr(0, d1);  // "GE2"
                                    ability.cond_amount_if_true =
                                        static_cast<size_t>(std::stoi(tail.substr(d1 + 1, d2 - d1 - 1)));
                                    ability.amount = static_cast<size_t>(std::stoi(tail.substr(d2 + 1)));
                                    handled_conditional = true;
                                }
                            }
                        }
                    }
                }
                // Delirium-conditional value. Two equivalent Forge spellings, both
                // meaning "<yes> if the caster has delirium, else <no>":
                //   Count$Delirium.<yes>.<no>           (compact form, e.g. Unholy Heat)
                //   Count$Compare Y GE4.<yes>.<no>      (explicit GE form)
                size_t delirium_pos = handled_conditional ? std::string::npos : sv.find("Count$Delirium");
                size_t ge_pos = handled_conditional ? std::string::npos : sv.find("GE");
                size_t scale_pos = delirium_pos != std::string::npos
                                       ? delirium_pos + std::string("Count$Delirium").size()
                                       : (ge_pos != std::string::npos ? ge_pos + 2 : std::string::npos);
                if (handled_conditional) {
                    // already routed to the conditional-amount fields above
                } else if (scale_pos != std::string::npos) {
                    std::string rest = sv.substr(scale_pos);
                    size_t d1 = rest.find('.');
                    if (d1 != std::string::npos) {
                        size_t d2 = rest.find('.', d1 + 1);
                        if (d2 != std::string::npos) {
                            size_t delirium_amt = static_cast<size_t>(std::stoi(rest.substr(d1 + 1, d2 - d1 - 1)));
                            size_t default_amt  = static_cast<size_t>(std::stoi(rest.substr(d2 + 1)));
                            ability.amount = default_amt;
                            DamageParams &dp = effect_params<DamageParams>(ability);
                            dp.delirium_amount = delirium_amt;
                            dp.is_delirium_scale = true;
                        }
                    }
                } else if (sv.find("Count$Valid") != std::string::npos ||
                           sv.find("Targeted$") != std::string::npos ||
                           sv.find("Count$InYourLibrary") != std::string::npos ||
                           sv.find("Count$YourLifeTotal") != std::string::npos ||
                           sv.find("Count$Revolt") != std::string::npos ||
                           // Count$Threshold.<hi>.<lo> — graveyard-threshold ritual scaling
                           // (Cabal Ritual: Amount$ X, X = Count$Threshold.5.3 → BBBBB if the
                           // caster has 7+ cards in their graveyard, else BBB). Preserved here
                           // and evaluated at activation by evaluate_dynamic_amount.
                           sv.find("Count$Threshold") != std::string::npos ||
                           // Count$UrzaLands.<hi>.<lo> — the "Tron" mana lands (Urza's Mine/Power
                           // Plant/Tower): hi colorless mana if the controller controls a complete
                           // set of all three, else lo. Preserved here and evaluated at activation
                           // by evaluate_dynamic_amount (mana-ability path via eval_mana_amount).
                           sv.find("Count$UrzaLands") != std::string::npos ||
                           // Count$CardCounters.<TYPE> — counters on the source permanent (The One
                           // Ring's burden-counter scaling). Evaluated at resolution by
                           // evaluate_dynamic_amount against the ability's source.
                           sv.find("Count$CardCounters") != std::string::npos ||
                           // Count$xPaid — amount equals the X paid at cast (Forth Eorlingas!:
                           // TokenAmount$ X, X = Count$xPaid → X 2/2 Human Knight tokens). Mirrors
                           // the sub-ability path (parse_svar_ability) so a top-level SP$/AB$
                           // ability scales by X too, instead of falling back to the count==1
                           // single-token default.
                           sv.find("xPaid") != std::string::npos) {
                    // Runtime expression — preserve for evaluation at activation/resolve time
                    ability.dynamic_amount_expr = sv;
                }
            }
            ability.amount_svar = "";
        }

        // Resolve an activated-ability ReduceCost$ SVar reference (Eiganjo's Channel:
        // ReduceCost$ X, X = Count$Valid Creature.Legendary+YouCtrl) into its runtime Count$
        // expression. A literal integer (e.g. "1") is kept verbatim; a single SVar key is
        // expanded to its Count$/dynamic expression for evaluation at activation time. The
        // generic mana portion is reduced by the resolved amount (CR 601.2f).
        if (!ability.reduce_cost_expr.empty() &&
            !std::isdigit(static_cast<unsigned char>(ability.reduce_cost_expr[0]))) {
            auto it = svars.find(ability.reduce_cost_expr);
            if (it != svars.end()) ability.reduce_cost_expr = it->second;
        }

        // Resolve a dynamic PutCounter CounterNum$ SVar reference (Wrath of the Skies:
        // CounterNum$ X, X = Count$xPaid) into its runtime Count$ expression so the
        // put_counter effect can evaluate the count at resolution.
        if (auto *cp = std::get_if<CounterParams>(&ability.params)) {
            if (!cp->count_expr.empty()) {
                auto it = svars.find(cp->count_expr);
                if (it != svars.end()) cp->count_expr = it->second;
            }
        }

        // Resolve a top-level Pump/PumpAll NumAtt$/NumDef$ count-SVar (Toxic Deluge: -X/-X →
        // Count$xPaid) and a TargetMin$/TargetMax$ SVar (exactly-X / up-to-X targeting). The
        // sub-ability path resolves these in parse_svar_ability; do the same for primary SP$/AB$
        // abilities whose effect/targeting scales by X (Toxic Deluge, Candelabra, Hide on the Ceiling).
        resolve_pump_exprs(ability, svars);
        resolve_xpaid_target_counts(ability, svars);

        // Resolve AB$ Animate Power$/Toughness$ SVar tokens (Karn: Power$ X, X =
        // Targeted$CardManaCost) into their runtime dynamic_amount expression, evaluated against
        // the animate target at resolution. A token that is not an SVar key is left as no dynamic
        // expr (the numeric base, parsed above, stands).
        for (int which = 0; which < 2; which++) {
            std::string &tok = which == 0 ? ability.animate_power_token : ability.animate_toughness_token;
            std::string &expr = which == 0 ? ability.animate_power_expr : ability.animate_toughness_expr;
            if (tok.empty()) continue;
            auto it = svars.find(tok);
            if (it != svars.end()) expr = it->second;
            tok.clear();
        }

        // Emblem (CR 114): an AB$ Effect that grants a permanent continuous static to its
        // controller — Kaito's "[+1]: You get an emblem with 'Ninjas you control get +1/+1.'"
        // (StaticAbilities$ <SVar> + Duration$ Permanent). Resolve the named continuous static
        // SVar into a StaticAbility and store it on the ability; the Effect handler creates a
        // player-owned emblem carrying it at resolution. Distinguished from the transient
        // StaticAbilities$ Effects (Unblockable, Ugin's MayPlay) by Duration$ Permanent — those
        // are EOT/until-leaves and handled by their own flags. General over any emblem-making
        // Effect that names a permanent-duration continuous-static SVar.
        if (ability.category == "Effect" && !ability.effect_static_ability.empty() &&
            line.find("Duration$ Permanent") != std::string::npos) {
            auto it = svars.find(ability.effect_static_ability);
            if (it != svars.end() && it->second.find("Mode$ Continuous") != std::string::npos) {
                StaticAbility est = parse_one_static_ability(it->second, svars);
                if (!est.category.empty()) ability.effect_emblem_statics.push_back(est);
            }
        }

        // AB$ Effect | Triggers$ <SVar>[,<SVar>...] — a transient floating triggered ability
        // hosted on a command-zone Effect (Tamiyo, Seasoned Scholar's +2: "until your next turn,
        // whenever a creature an opponent controls attacks you or a planeswalker you control, it
        // gets -1/-0"). The sub-ability path (parse_svar_ability) resolves Triggers$ for DB$
        // Effects; do the same here for a TOP-LEVEL AB$ Effect, where svars are in scope. The
        // Duration$ (UntilYourNextTurn, parsed onto the ability above) is applied to the registered
        // floating trigger by the GrantCast handler. General over any AB$ Effect naming a Triggers$.
        if (ability.category == "Effect" && ability.effect_floating_triggers.empty()) {
            size_t tp = line.find("Triggers$");
            if (tp != std::string::npos) {
                size_t vstart = tp + strlen("Triggers$");
                while (vstart < line.size() && line[vstart] == ' ') vstart++;
                size_t vend = line.find('|', vstart);
                std::string tval = line.substr(vstart,
                    (vend == std::string::npos ? line.size() : vend) - vstart);
                while (!tval.empty() && (tval.back() == ' ' || tval.back() == '\r')) tval.pop_back();
                for (const std::string &svar_name : split(tval, ',', /*skip_empty=*/true)) {
                    auto it = svars.find(svar_name);
                    if (it != svars.end()) {
                        Ability trig = parse_one_trigger(it->second, svars, card_name);
                        if (trig.trigger_on != 0) ability.effect_floating_triggers.push_back(trig);
                    }
                }
            }
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
static Ability parse_one_trigger(const std::string &line, const std::map<std::string, std::string> &svars,
                                 const std::string& card_name) {
    Ability ability;
    ability.ability_type = Ability::TRIGGERED;

    std::string execute_svar;
    bool mode_changes_zone = false;
    bool mode_changes_zone_all = false;
    bool dest_is_battlefield = false;
    bool dest_is_graveyard = false;
    bool origin_is_battlefield = false;
    bool origin_is_graveyard = false;
    bool valid_card_creature = false;
    bool valid_card_self = false;
    bool mode_is_phase = false;
    bool phase_is_upkeep = false;
    bool phase_is_end_step = false;
    bool phase_is_draw = false;
    bool phase_is_begin_combat = false;
    bool phase_is_first_main = false;
    bool phase_is_second_main = false;
    bool phase_is_cleanup = false;
    bool trigger_zone_is_graveyard = false;
    bool valid_player_is_you = false;
    bool mode_is_spell_cast = false;
    bool mode_is_damage_done = false;
    bool mode_is_damage_all = false;
    bool valid_source_creature_youctrl = false;
    bool damage_combat_only = false;
    bool valid_card_non_creature = false;
    bool valid_card_instant = false;
    bool valid_card_sorcery = false;
    bool valid_card_owner_you = false;
    bool valid_card_land = false;
    bool valid_card_artifact = false;
    bool valid_card_colorless = false;
    bool valid_card_non_token = false;
    bool valid_card_untapped = false;
    bool valid_card_permanent = false;
    bool mode_is_drawn = false;
    bool mode_is_attacks = false;
    bool mode_is_attackers_declared = false;
    bool mode_is_taps_for_mana = false;
    bool mode_is_becomes_target = false;
    bool mode_is_become_monstrous = false;
    bool source_is_spell = false;
    bool source_opp_ctrl = false;
    bool valid_target_self = false;
    bool trigger_static = false;
    bool attacking_player_is_you = false;
    bool valid_card_opp_own = false;
    bool valid_card_opp_ctrl = false;
    bool exclude_first_draw_step = false;
    bool trigger_optional_local = false;
    size_t draw_number_eq = 0;          // Number$ N on a Mode$ Drawn trigger (Nth-draw gate)
    bool attacked_defender_you = false; // Attacked$ You,Planeswalker.YouCtrl (the attack hits you/your PW)
    std::string valid_card_subtype;
    size_t activator_this_turn_cast_eq = 0;
    int kicked_index = 0;  // ValidCard$ ...+kicked N — fires only if the Nth kicker was paid

    // Walk pipe-delimited params
    size_t param_pos = 0;
    std::string key, value;
    while (next_param(line, param_pos, key, value)) {
        if (key == "Mode") {
            // ChangesZone fires once per matching card. ChangesZoneAll ("whenever one or more
            // cards ...") is a single batch trigger (CR 603.2c): it fires exactly ONCE for a
            // group of simultaneous zone changes, no matter how many cards matched. Both reuse the
            // same per-card origin/destination/ValidCards filters; the _all form additionally sets
            // trigger_batch_zone_all so the trigger scan dedupes it to a single firing per batch
            // (Moonshadow: milling 3 permanent cards removes ONE -1/-1 counter, not three).
            if (value == "ChangesZone" || value == "ChangesZoneAll") mode_changes_zone = true;
            if (value == "ChangesZoneAll") mode_changes_zone_all = true;
            else if (value == "Phase") mode_is_phase = true;
            else if (value == "SpellCast") mode_is_spell_cast = true;
            else if (value == "DamageDone") mode_is_damage_done = true;
            else if (value == "DamageAll") mode_is_damage_all = true;
            else if (value == "Drawn") mode_is_drawn = true;
            else if (value == "Attacks") mode_is_attacks = true;
            else if (value == "AttackersDeclared") mode_is_attackers_declared = true;
            else if (value == "TapsForMana") mode_is_taps_for_mana = true;
            else if (value == "BecomesTarget") mode_is_becomes_target = true;
            else if (value == "BecomeMonstrous") mode_is_become_monstrous = true;
        } else if (key == "ValidSource") {
            // Mode$ BecomesTarget | ValidSource$ Spell.OppCtrl — the targeting object must be a
            // SPELL (not an ability) controlled by an opponent of the source's controller.
            if (value.rfind("Spell", 0) == 0) source_is_spell = true;
            if (value.find("OppCtrl") != std::string::npos) source_opp_ctrl = true;
            // Mode$ DamageAll | ValidSource$ Creature.YouCtrl — the damaging creature must be one
            // this trigger's controller controls (Forth Eorlingas!'s floating monarch trigger).
            if (value.find("Creature") != std::string::npos && value.find("YouCtrl") != std::string::npos)
                valid_source_creature_youctrl = true;
        } else if (key == "ValidTarget") {
            // ValidTarget$ Card.Self — the permanent that became a target must be this source.
            if (value == "Card.Self") valid_target_self = true;
        } else if (key == "Activator") {
            // Mode$ TapsForMana | Activator$ You — only the source controller tapping a
            // permanent for mana fires this ("whenever YOU tap ...").
            if (value == "You") valid_player_is_you = true;
        } else if (key == "Static") {
            // Static$ True on a TapsForMana trigger: it is a mana-additional effect that does
            // not use the stack (CR 605.1a) — resolved immediately by the mana system.
            if (value == "True") trigger_static = true;
        } else if (key == "AttackingPlayer") {
            // Mode$ AttackersDeclared | AttackingPlayer$ You — the trigger fires only when
            // the player who declared attackers is this ability's controller ("whenever you attack").
            if (value == "You") attacking_player_is_you = true;
        } else if (key == "Phase") {
            // A Phase trigger may list several phases comma-separated (Carpet of Flowers:
            // Phase$ Main1,Main2 — "at the beginning of each of your main phases"). Split and set
            // each phase flag so the trigger can bind to every listed phase's event.
            size_t tok_pos = 0;
            while (tok_pos <= value.size()) {
                size_t comma = value.find(',', tok_pos);
                std::string tok = value.substr(tok_pos, comma == std::string::npos
                                                            ? std::string::npos : comma - tok_pos);
                if (tok == "Upkeep")   phase_is_upkeep   = true;
                // Forge writes the end step as either "EndStep" or "End of Turn".
                if (tok == "EndStep" || tok == "End of Turn")  phase_is_end_step = true;
                if (tok == "Draw")     phase_is_draw     = true;
                if (tok == "BeginCombat") phase_is_begin_combat = true;
                // Forge writes the (pre-combat) first main phase as "Main1", the post-combat one as "Main2".
                if (tok == "Main1")    phase_is_first_main = true;
                if (tok == "Main2")    phase_is_second_main = true;
                if (tok == "Cleanup")  phase_is_cleanup = true;
                if (comma == std::string::npos) break;
                tok_pos = comma + 1;
            }
        } else if (key == "TriggerZones") {
            // The zone(s) the source must be in for this triggered ability to function
            // (CR 113.6 / 603.6). Arclight Phoenix's combat trigger functions from the
            // graveyard, so the trigger scan must look at graveyard cards, not just the
            // battlefield.
            if (value.find("Graveyard") != std::string::npos) trigger_zone_is_graveyard = true;
        } else if (key == "ValidPlayer" || key == "ValidActivatingPlayer") {
            if (value == "You") valid_player_is_you = true;
        } else if (key == "Origin") {
            if (value == "Battlefield") origin_is_battlefield = true;
            if (value == "Graveyard")   origin_is_graveyard   = true;
        } else if (key == "Destination") {
            if (value == "Battlefield") dest_is_battlefield = true;
            if (value == "Graveyard")   dest_is_graveyard   = true;
        } else if (key == "ValidCard" || key == "ValidCards") {
            if (value.find("Creature")    != std::string::npos) valid_card_creature     = true;
            if (value.find("nonCreature") != std::string::npos) valid_card_non_creature = true;
            if (value.find(".Other")      != std::string::npos) ability.trigger_self_excluded = true;
            if (value.rfind("Card.Self", 0) == 0)                valid_card_self         = true;
            // The Self qualifier may also be a trailing token (e.g. "Card.wasCastByYou+Self",
            // The One Ring) rather than the head — match the delimited ".Self"/"+Self" form.
            if (value.find(".Self") != std::string::npos ||
                value.find("+Self") != std::string::npos)        valid_card_self         = true;
            // wasCastByYou — "if you cast it" cast-condition on an ETB trigger (The One Ring): the
            // source must have entered by being cast (Permanent::entered_by_cast).
            if (value.find("wasCastByYou") != std::string::npos)
                ability.trigger_requires_entered_by_cast = true;
            // Kicker-linked condition (CR 702.33f): "Card.Self+kicked N" — fires only when the
            // Nth kicker was paid. Parse the 1-based index after "kicked " (a missing number
            // defaults to the first kicker). General over any "+kicked N" SpellCast trigger.
            {
                size_t kp = value.find("kicked");
                if (kp != std::string::npos) {
                    size_t np = kp + strlen("kicked");
                    while (np < value.size() && value[np] == ' ') np++;
                    int n = 0;
                    while (np < value.size() && isdigit((unsigned char)value[np]))
                        n = n * 10 + (value[np++] - '0');
                    kicked_index = (n > 0) ? n : 1;
                }
            }
            if (value.find("Instant")     != std::string::npos) valid_card_instant      = true;
            if (value.find("Sorcery")     != std::string::npos) valid_card_sorcery      = true;
            if (value.find(".YouOwn")     != std::string::npos) valid_card_owner_you    = true;
            if (value.find(".OppOwn")     != std::string::npos) valid_card_opp_own      = true;
            if (value.find("OppCtrl")     != std::string::npos) valid_card_opp_ctrl     = true;
            if (value.find("Land")        != std::string::npos) valid_card_land         = true;
            if (value.find("Artifact")    != std::string::npos) valid_card_artifact     = true;
            if (value.find("Colorless")   != std::string::npos) valid_card_colorless    = true;
            if (value.find("!token")      != std::string::npos) valid_card_non_token    = true;
            // "+untapped"/".untapped" qualifier — the changing card must be untapped when the
            // trigger checks it (Mystic Sanctuary: ValidCard$ Card.Self+untapped, "enters
            // untapped"). Matched delimited so a plain "tapped" token can't set it.
            if (value.find("+untapped")   != std::string::npos ||
                value.find(".untapped")   != std::string::npos) valid_card_untapped     = true;
            // ValidCard$ Permanent (head token) — restrict to permanent card types. Matched
            // on the leading token so a subtype merely named within isn't misread.
            if (value.substr(0, value.find_first_of(".+")) == "Permanent")
                valid_card_permanent = true;
            // YouCtrl may be the first ('.YouCtrl') or a later ('+YouCtrl') qualifier.
            if (value.find("YouCtrl")     != std::string::npos) valid_player_is_you     = true;
            // Dynamic mana-value filter on the cast spell (Chalice of the Void:
            // "Card.cmcEQY", Y = Count$CardCounters.CHARGE). Resolve the cmc<op><svar>
            // qualifier to its runtime Count$ expression + comparison op, mirroring the
            // ChangeType cmcEQ handling used by Aether Vial.
            for (const char *op : {"cmcEQ", "cmcLE", "cmcGE", "cmcLT", "cmcGT", "cmcNE"}) {
                size_t p = value.find(op);
                if (p == std::string::npos) continue;
                std::string svar_key = value.substr(p + strlen(op));
                size_t end = svar_key.find_first_of(".+");
                if (end != std::string::npos) svar_key = svar_key.substr(0, end);
                auto it = svars.find(svar_key);
                if (it != svars.end()) {
                    ability.trigger_cmc_expr = it->second;
                    ability.trigger_cmc_op = std::string(op + 3);  // "cmcEQ" → "EQ"
                }
                break;
            }
            // Leading token before '.'/'+' that isn't a recognized card type is a
            // subtype filter (e.g. "Cat.Other+YouCtrl" -> subtype "Cat").
            std::string head = value.substr(0, value.find_first_of(".+"));
            static const std::set<std::string> known_types = {
                "Creature", "Land", "Instant", "Sorcery", "Card", "Permanent",
                "Artifact", "Enchantment", "Planeswalker"};
            if (!head.empty() && known_types.find(head) == known_types.end() &&
                head.rfind("cmc", 0) != 0)
                valid_card_subtype = head;
        } else if (key == "OptionalDecider") {
            // Any named decider ("You" / "TriggeredCardController" / "Controller") makes the
            // whole triggered ability optional ("you may ...") for that player — the controller
            // of the source, which is who the engine prompts in every supported case.
            if (value.find("You") != std::string::npos ||
                value.find("Controller") != std::string::npos)
                trigger_optional_local = true;
        } else if (key == "FirstCardInDrawStep") {
            if (value == "False") exclude_first_draw_step = true;
        } else if (key == "Number") {
            // Number$ N on a Mode$ Drawn trigger (Tamiyo, Inquisitive Student: "your THIRD card
            // in a turn"). Fire only on the Nth card the player draws this turn.
            if (!value.empty() && isdigit((unsigned char)value[0]))
                draw_number_eq = static_cast<size_t>(std::stoi(value));
        } else if (key == "Attacked") {
            // Attacked$ You,Planeswalker.YouCtrl — the attack must be against the trigger's
            // controller or a planeswalker they control (Tamiyo, Seasoned Scholar's +2 trigger).
            if (value.find("You") != std::string::npos) attacked_defender_you = true;
        } else if (key == "CombatDamage") {
            if (value == "True") damage_combat_only = true;
        } else if (key == "ActivatorThisTurnCast") {
            if (value.rfind("EQ", 0) == 0) {
                activator_this_turn_cast_eq = static_cast<size_t>(std::stoi(value.substr(2)));
            }
        } else if (key == "IsPresent") {
            // Intervening-if (603.4): "..., if you control a <thing>, ...". Checked both
            // when the trigger would go on the stack and again on resolution.
            ability.condition_present = value;
            ability.intervening_if = true;
        } else if (key == "PresentCompare") {
            ability.condition_compare = value;  // e.g. "GE2"; empty defaults to ">= 1"
        } else if (key == "CheckSVar") {
            auto it = svars.find(value);
            const std::string svdef = (it != svars.end()) ? it->second : std::string();
            if (svdef.rfind("Number$", 0) == 0) {
                // Per-permanent stored-SVar gate (Carpet of Flowers: CheckSVar$ CarpetX, where
                // SVar:CarpetX:Number$0 is a scratch int latched by DB$ StoreSVar — "if you
                // haven't added mana with this ability this turn"). Read the SOURCE permanent's
                // stored_svars[name] at trigger time AND resolution, NOT a board-presence count.
                ability.stored_svar_gate_name = value;
            } else {
                // Intervening-if (603.4) gated on an SVar count rather than a board presence,
                // e.g. Ocelot Pride's "if you gained life this turn" (CheckSVar$ YouLifeGained →
                // Count$LifeYouGainedThisTurn). Resolve the SVar to its Count$ expression and store
                // it as the intervening-if condition so the whole trigger fizzles when false.
                ability.condition_present = (it != svars.end()) ? it->second : value;
                ability.intervening_if = true;
            }
        } else if (key == "SVarCompare") {
            // SVarCompare follows CheckSVar on the line; route it to whichever gate CheckSVar set up.
            if (!ability.stored_svar_gate_name.empty())
                ability.stored_svar_gate_compare = value;  // per-permanent stored-SVar latch compare
            else
                ability.condition_compare = value;  // explicit compare for the CheckSVar count gate
        } else if (key == "Execute") {
            execute_svar = value;
        }
    }

    // Map trigger condition to event ID.

    // OptionalDecider$ You ("At the beginning of your upkeep, you may ...") makes the whole
    // triggered ability optional at resolution, independent of the trigger mode (Aether
    // Vial's upkeep charge-counter trigger is a Phase trigger, not a zone-change trigger).
    ability.trigger_optional = trigger_optional_local;

    // All ChangesZone triggers use CARD_CHANGED_ZONE; origin/destination/type filters applied at match time.
    if (mode_changes_zone) {
        ability.trigger_on = Events::CARD_CHANGED_ZONE;
        if (origin_is_battlefield)       ability.trigger_zone_origin      = Zone::BATTLEFIELD;
        else if (origin_is_graveyard)    ability.trigger_zone_origin      = Zone::GRAVEYARD;
        if (dest_is_battlefield)         ability.trigger_zone_destination = Zone::BATTLEFIELD;
        else if (dest_is_graveyard)      ability.trigger_zone_destination = Zone::GRAVEYARD;
        ability.trigger_valid_card_is_creature            = valid_card_creature;
        ability.trigger_valid_card_is_instant_or_sorcery  = valid_card_instant || valid_card_sorcery;
        ability.trigger_valid_card_is_land                = valid_card_land;
        ability.trigger_valid_card_is_artifact            = valid_card_artifact;
        ability.trigger_valid_card_colorless              = valid_card_colorless;
        ability.trigger_valid_card_non_token              = valid_card_non_token;
        ability.trigger_valid_card_untapped               = valid_card_untapped;
        ability.trigger_valid_card_is_permanent           = valid_card_permanent;
        ability.trigger_batch_zone_all                    = mode_changes_zone_all;
        ability.trigger_valid_card_subtype                = valid_card_subtype;
        ability.trigger_valid_player_is_controller        = valid_card_owner_you || valid_player_is_you;
        if (valid_card_self) ability.trigger_only_self = true;
        // A Destination$ Battlefield trigger gated by IsPresent$ Card.Self (the source must
        // already be on the battlefield) is Forge's idiom for "Whenever ANOTHER permanent
        // enters" — the source's own entry must not satisfy it (Kappa Cannoneer's Oracle text
        // reads "another artifact you control"). Exclude the source from this trigger.
        if (dest_is_battlefield && ability.condition_present == "Card.Self")
            ability.trigger_self_excluded = true;
    }

    if (mode_is_phase && phase_is_upkeep) {
        ability.trigger_on = Events::UPKEEP_BEGAN;
        ability.trigger_valid_player_is_controller = valid_player_is_you;
    }

    if (mode_is_phase && phase_is_end_step) {
        ability.trigger_on = Events::END_STEP_BEGAN;
        ability.trigger_valid_player_is_controller = valid_player_is_you;
    }

    if (mode_is_phase && phase_is_draw) {
        ability.trigger_on = Events::DRAW_STEP_BEGAN;
        ability.trigger_valid_player_is_controller = valid_player_is_you;
    }

    if (mode_is_phase && phase_is_begin_combat) {
        ability.trigger_on = Events::BEGIN_COMBAT_BEGAN;
        ability.trigger_valid_player_is_controller = valid_player_is_you;
    }

    if (mode_is_phase && phase_is_first_main) {
        ability.trigger_on = Events::FIRST_MAIN_BEGAN;
        ability.trigger_valid_player_is_controller = valid_player_is_you;
    }

    if (mode_is_phase && phase_is_second_main) {
        // "at the beginning of [your] second/postcombat main phase" / Phase$ Main2. If Main1 already
        // claimed trigger_on (Phase$ Main1,Main2 — "each of your main phases", Carpet of Flowers),
        // bind SECOND_MAIN_BEGAN as an additional event so the one trigger fires on both phases.
        if (ability.trigger_on == 0) ability.trigger_on = Events::SECOND_MAIN_BEGAN;
        else                         ability.trigger_on_extra.push_back(Events::SECOND_MAIN_BEGAN);
        ability.trigger_valid_player_is_controller = valid_player_is_you;
    }

    if (mode_is_phase && phase_is_cleanup) {
        // "at the beginning of the cleanup step" / Phase$ Cleanup (Carpet of Flowers' Static$ True
        // reset). With no ValidPlayer$ You it fires at every cleanup (re-arming the latch).
        ability.trigger_on = Events::CLEANUP_BEGAN;
        ability.trigger_valid_player_is_controller = valid_player_is_you;
    }

    // Static$ True on a phase or zone-change trigger (Carpet of Flowers' cleanup / leave-battlefield
    // resets) is a bookkeeping trigger that resolves immediately off the stack — it never uses the
    // stack like a normal triggered ability (CR 605.1a-style). The TapsForMana static path has its
    // own dedicated flag (trigger_taps_for_mana_static) and inline mana-system handling, so it is
    // excluded here. General over any Static$ True phase/ChangesZone trigger.
    if (trigger_static && !mode_is_taps_for_mana)
        ability.trigger_static_offstack = true;

    // TriggerZones$ Graveyard — the ability functions while its source is in the graveyard.
    ability.trigger_from_graveyard = trigger_zone_is_graveyard;

    if (mode_is_spell_cast && valid_card_non_creature) {
        ability.trigger_on = Events::NONCREATURE_SPELL_CAST;
        ability.trigger_valid_player_is_controller = valid_player_is_you;
    }

    // "Whenever you cast a colorless spell, ..." — Glaring Fleshraker
    // (Mode$ SpellCast | ValidCard$ Card.Colorless | ValidActivatingPlayer$ You). A plain
    // SpellCast with a colorless filter on the cast spell; matched at trigger time against the
    // spell's colorlessness (CR 105.2c). Keyed on the general Colorless tag, not this card.
    if (mode_is_spell_cast && valid_card_colorless) {
        ability.trigger_on = Events::SPELL_CAST;
        ability.trigger_valid_card_colorless = true;
        ability.trigger_valid_player_is_controller = valid_player_is_you;
    }

    // "whenever you cast your Nth spell" — Cori-Steel Cutter; or "your Nth NONCREATURE spell each
    // turn" — The Fantasticar. Bind to SPELL_CAST (fired AFTER the per-cast spell counters bump,
    // unlike NONCREATURE_SPELL_CAST which fires before) so the count gate sees the current cast.
    if (mode_is_spell_cast && activator_this_turn_cast_eq > 0) {
        ability.trigger_on = Events::SPELL_CAST;
        ability.trigger_valid_player_is_controller = valid_player_is_you;
        ability.trigger_spell_count_eq = activator_this_turn_cast_eq;
        if (valid_card_non_creature) {
            // Count only noncreature spells, and only fire on a noncreature cast (the SPELL_CAST
            // event carries every spell, so filter the triggering card to noncreature too).
            ability.trigger_valid_card_non_creature = true;
            ability.trigger_spell_count_noncreature = true;
        }
    }

    // "Whenever a player casts a spell with mana value equal to ..." — Chalice of the Void
    // (Mode$ SpellCast | ValidCard$ Card.cmcEQY | ValidActivatingPlayer$ Player). A dynamic
    // mana-value filter on any player's spell. The cmc match is checked at trigger time.
    if (mode_is_spell_cast && !ability.trigger_cmc_expr.empty()) {
        ability.trigger_on = Events::SPELL_CAST;
        ability.trigger_valid_player_is_controller = valid_player_is_you;
    }

    // "When you cast this spell, [if it was kicked with its [N] kicker,] ..." — Wastescape
    // Battlemage (Mode$ SpellCast | ValidCard$ Card.Self[+kicked N]). A linked self-cast
    // trigger that fires while the spell is on the stack (CR 702.33e/f). trigger_only_self
    // restricts it to the source spell; trigger_kicked_index (>0) additionally gates on the
    // Nth kicker having been paid. Handled by the dedicated self-cast SPELL_CAST scan.
    if (mode_is_spell_cast && valid_card_self) {
        ability.trigger_on = Events::SPELL_CAST;
        ability.trigger_only_self = true;
        ability.trigger_kicked_index = kicked_index;
    }

    // General "whenever you cast a spell, ..." — Paradox Engine
    // (Mode$ SpellCast | ValidCard$ Card | ValidActivatingPlayer$ You). A plain, unfiltered
    // SpellCast trigger that fires on EVERY spell the source's controller casts. None of the
    // specialized SpellCast bindings above (noncreature/colorless/count/cmc/self) applied, so
    // bind to SPELL_CAST and gate on the caster being this source's controller. The general
    // battlefield trigger scan fires it (no extra ValidCard$ filter ⇒ any spell). Keyed on the
    // bare SpellCast mode, not this card.
    if (mode_is_spell_cast && ability.trigger_on == 0) {
        ability.trigger_on = Events::SPELL_CAST;
        ability.trigger_valid_player_is_controller = valid_player_is_you;
    }

    // "Whenever CARDNAME deals combat damage to a player" — Barrowgoyf
    if (mode_is_damage_done && damage_combat_only) {
        ability.trigger_on = Events::COMBAT_DAMAGE_TO_PLAYER;
        ability.trigger_only_self = true;  // ValidSource$ Card.Self
    }

    // "Whenever one or more creatures you control deal combat damage to one or more players" —
    // Forth Eorlingas!'s floating monarch trigger (Mode$ DamageAll | ValidSource$ Creature.YouCtrl
    // | ValidTarget$ Player | CombatDamage$ True). Fires on COMBAT_DAMAGE_TO_PLAYER when the
    // damaging creature is controlled by this trigger's controller (matched at fire time, since
    // the floating trigger has no source permanent to self-reference).
    if (mode_is_damage_all && damage_combat_only) {
        ability.trigger_on = Events::COMBAT_DAMAGE_TO_PLAYER;
        ability.trigger_damage_source_youctrl = valid_source_creature_youctrl;
    }

    // "whenever a player draws a card" — Orcish Bowmasters (Mode$ Drawn)
    if (mode_is_drawn) {
        ability.trigger_on = Events::PLAYER_DREW_CARD;
        ability.trigger_valid_card_opp_own = valid_card_opp_own;
        ability.trigger_exclude_first_draw_step = exclude_first_draw_step;
        ability.trigger_draw_number_eq = draw_number_eq;
        // ValidCard$ Card.YouCtrl on a Drawn trigger ("whenever YOU draw ...", Tamiyo): the drawer
        // must be the source's controller. Reuse the controller-is-event-player gate (the
        // PLAYER_DREW_CARD event's PLAYER is the drawer), set by YouCtrl in the ValidCard parse.
        ability.trigger_valid_player_is_controller = valid_player_is_you;
    }

    // "Whenever CARDNAME attacks, ..." — Phelia (Mode$ Attacks | ValidCard$ Card.Self). Fires
    // once for this creature each time it is declared as an attacker (CR 508.2). ValidCard$
    // Card.Self → trigger_only_self matches the attacking ENTITY against the source.
    if (mode_is_attacks) {
        ability.trigger_on = Events::CREATURE_ATTACKED;
        if (valid_card_self) ability.trigger_only_self = true;
        // ValidCard$ Creature.OppCtrl | Attacked$ You,Planeswalker.YouCtrl (Tamiyo, Seasoned
        // Scholar's +2 hosted trigger): an opponent's creature attacking you/your planeswalker.
        // Matched at fire time against the attacker's controller (the trigger has no source perm).
        ability.trigger_attacker_opp_ctrl = valid_card_opp_ctrl;
        ability.trigger_attacked_defender_you = attacked_defender_you;
    }

    // "Whenever you attack" — Guide of Souls (Mode$ AttackersDeclared | AttackingPlayer$ You).
    // Fires once per combat when the source's controller declares one or more attackers.
    if (mode_is_attackers_declared) {
        ability.trigger_on = Events::ATTACKERS_DECLARED;
        ability.trigger_valid_player_is_controller = attacking_player_is_you;
    }

    // "Whenever you tap a creature for mana, add an additional {G}." — Badgermole Cub
    // (Mode$ TapsForMana | ValidCard$ Creature | Activator$ You | Static$ True). A
    // mana-additional triggered ability resolved immediately by the mana system (off-stack,
    // CR 605.1a) rather than placed on the stack.
    if (mode_is_taps_for_mana) {
        ability.trigger_on = Events::TAPPED_FOR_MANA;
        ability.trigger_valid_card_is_creature = valid_card_creature;
        ability.trigger_valid_player_is_controller = valid_player_is_you;
        ability.trigger_taps_for_mana_static = trigger_static;
    }

    // "Whenever CARDNAME becomes the target of a spell an opponent controls, ..." — Reality
    // Smasher (Mode$ BecomesTarget | ValidSource$ Spell.OppCtrl | ValidTarget$ Card.Self). Fires
    // when this permanent becomes the target of a matching spell (CR 603.2c). ValidTarget$
    // Card.Self reuses trigger_only_self (the targeted permanent must be the source).
    if (mode_is_becomes_target) {
        ability.trigger_on = Events::BECAME_TARGET;
        ability.trigger_source_must_be_spell = source_is_spell;
        ability.trigger_source_opp_ctrl = source_opp_ctrl;
        if (valid_target_self) ability.trigger_only_self = true;
    }

    // "When CARDNAME becomes monstrous, ..." — Mode$ BecomeMonstrous (CR 701.37). Fired by the
    // resolving Monstrosity$ ability (effect_put_counter.cpp) with ENTITY = the permanent that
    // became monstrous, so ValidCard$ Card.Self reuses the standard trigger_only_self ENTITY check.
    // TriggerZones$ Battlefield is the default functioning zone; no extra handling needed.
    if (mode_is_become_monstrous) {
        ability.trigger_on = Events::BECAME_MONSTROUS;
        if (valid_card_self) ability.trigger_only_self = true;
    }

    // Resolve effect from Execute$ SVar
    if (!execute_svar.empty()) {
        auto it = svars.find(execute_svar);
        if (it != svars.end()) {
            // Check for Sylvan Library pattern: ChooseCard with DrawnThisTurn
            if (it->second.find("ChooseCard") != std::string::npos &&
                it->second.find("DrawnThisTurn") != std::string::npos) {
                ability.category = "SylvanLibrary";
            } else {
                Ability effect = parse_svar_ability(it->second, Ability::TRIGGERED, svars, card_name);
                // Take the effect's full configuration, then restore the trigger
                // metadata computed above from the T: line. Previously this copied
                // only a hand-picked subset of effect fields, which silently dropped
                // Origin$/Destination$/ValidTgts$/TargetMin$/TargetMax$ etc. — e.g.
                // Endurance's "bottom target player's graveyard into their library"
                // became a "dump the whole library onto the battlefield", spawning a
                // landfall trigger storm.
                effect.ability_type                             = ability.ability_type;
                effect.trigger_on                               = ability.trigger_on;
                effect.trigger_on_extra                         = ability.trigger_on_extra;
                effect.trigger_static_offstack                  = ability.trigger_static_offstack;
                effect.stored_svar_gate_name                    = ability.stored_svar_gate_name;
                effect.stored_svar_gate_compare                 = ability.stored_svar_gate_compare;
                effect.trigger_zone_origin                      = ability.trigger_zone_origin;
                effect.trigger_zone_destination                 = ability.trigger_zone_destination;
                effect.trigger_valid_card_is_creature           = ability.trigger_valid_card_is_creature;
                effect.trigger_valid_card_is_instant_or_sorcery = ability.trigger_valid_card_is_instant_or_sorcery;
                effect.trigger_valid_card_is_land               = ability.trigger_valid_card_is_land;
                effect.trigger_valid_card_is_artifact           = ability.trigger_valid_card_is_artifact;
                effect.trigger_valid_card_colorless             = ability.trigger_valid_card_colorless;
                effect.trigger_valid_card_non_token             = ability.trigger_valid_card_non_token;
                effect.trigger_valid_card_untapped              = ability.trigger_valid_card_untapped;
                effect.trigger_valid_card_is_permanent          = ability.trigger_valid_card_is_permanent;
                effect.trigger_batch_zone_all                   = ability.trigger_batch_zone_all;
                effect.trigger_valid_card_subtype               = ability.trigger_valid_card_subtype;
                effect.trigger_optional                         = ability.trigger_optional;
                effect.trigger_valid_card_opp_own               = ability.trigger_valid_card_opp_own;
                effect.trigger_exclude_first_draw_step          = ability.trigger_exclude_first_draw_step;
                effect.trigger_draw_number_eq                   = ability.trigger_draw_number_eq;
                effect.trigger_attacker_opp_ctrl                = ability.trigger_attacker_opp_ctrl;
                effect.trigger_attacked_defender_you            = ability.trigger_attacked_defender_you;
                effect.trigger_valid_player_is_controller       = ability.trigger_valid_player_is_controller;
                effect.trigger_only_self                        = ability.trigger_only_self;
                effect.trigger_self_excluded                    = ability.trigger_self_excluded;
                effect.trigger_spell_count_eq                   = ability.trigger_spell_count_eq;
                effect.trigger_spell_count_noncreature          = ability.trigger_spell_count_noncreature;
                effect.trigger_valid_card_non_creature          = ability.trigger_valid_card_non_creature;
                effect.trigger_kicked_index                     = ability.trigger_kicked_index;
                effect.trigger_cmc_expr                         = ability.trigger_cmc_expr;
                effect.trigger_cmc_op                           = ability.trigger_cmc_op;
                effect.trigger_from_graveyard                   = ability.trigger_from_graveyard;
                effect.trigger_taps_for_mana_static             = ability.trigger_taps_for_mana_static;
                effect.trigger_source_must_be_spell             = ability.trigger_source_must_be_spell;
                effect.trigger_source_opp_ctrl                  = ability.trigger_source_opp_ctrl;
                effect.trigger_damage_source_youctrl            = ability.trigger_damage_source_youctrl;
                // 603.4 intervening-if lives on the trigger line, not the Execute SVar — carry
                // it onto the resolved ability so it is re-checked at resolution. OR (don't
                // clobber) any intervening-if the Execute SVar itself declared, e.g. Uro's
                // TrigSac ConditionNotPresent$ Card.Self+escaped, which sets effect.intervening_if
                // (and condition_present/condition_negate) during SVar parse.
                effect.intervening_if                           = effect.intervening_if || ability.intervening_if;
                if (ability.intervening_if) {
                    effect.condition_present = ability.condition_present;
                    effect.condition_compare = ability.condition_compare;
                }
                ability = effect;
            }
        }
    }

    return ability;
}

static std::vector<Ability> parse_triggered_abilities(const std::string &script,
                                                      const std::map<std::string, std::string> &svars,
                                                      const std::string& card_name) {
    std::vector<Ability> result;
    for (const auto &line : find_trigger_lines(script)) {
        Ability ab = parse_one_trigger(line, svars, card_name);
        if (ab.trigger_on != 0)  // only keep recognised triggers
            result.push_back(ab);
    }
    return result;
}

// Parse a single static-ability line (the "Mode$ ... | ..." body of an S: line or a continuous
// static SVar) into a StaticAbility. Factored out of parse_static_abilities so a named continuous
// static referenced elsewhere — e.g. an AB$ Effect emblem's StaticAbilities$ SVar — can be parsed
// with the same grammar. Returns a StaticAbility with an empty category if the line is not a Mode$
// static (the caller skips it).
static StaticAbility parse_one_static_ability(const std::string &line,
                                              const std::map<std::string, std::string> &svars) {
    StaticAbility sa;
    if (line.find("Mode$") == std::string::npos) return sa;

    bool cant_attack_targeted = false;  // CantAttack with a Target$ player restriction (handled below)
    size_t param_pos = 0;
    std::string key, value;
    while (next_param(line, param_pos, key, value)) {
            if (key == "Mode") {
                sa.category = value;
            } else if (key == "Condition") {
                sa.condition = value;
            } else if (key == "AddPower") {
                if (!value.empty() && (std::isdigit(static_cast<unsigned char>(value[0])) || value[0] == '-'))
                    sa.add_power = std::stoi(value);
                else if (!value.empty()) {
                    auto it = svars.find(value);
                    sa.add_power_svar = (it != svars.end()) ? it->second : value;
                }
            } else if (key == "AddToughness") {
                if (!value.empty() && (std::isdigit(static_cast<unsigned char>(value[0])) || value[0] == '-'))
                    sa.add_toughness = std::stoi(value);
                else if (!value.empty()) {
                    auto it = svars.find(value);
                    sa.add_toughness_svar = (it != svars.end()) ? it->second : value;
                }
            } else if (key == "AddKeyword") {
                sa.add_keyword = value;
            } else if (key == "SetMaxHandSize") {
                // SetMaxHandSize$ Unlimited (Tamiyo, Seasoned Scholar's emblem: "You have no
                // maximum hand size.") → -1 (no maximum); a numeric value sets the maximum to N.
                if (value == "Unlimited") sa.set_max_hand_size = -1;
                else if (!value.empty() && isdigit((unsigned char)value[0]))
                    sa.set_max_hand_size = std::stoi(value);
            } else if (key == "AddAbility") {
                // AddAbility$ <SVarName> (Petrified Hamlet): a continuous static that grants
                // a full activated ability to every Affected$ permanent (CR 613.1f, layer 6).
                // Resolve the named SVar to its ability body now (e.g. "AB$ Mana | Cost$ T |
                // Produced$ C"); the layer-6 grant pass parses it to an Ability per recipient.
                auto it = svars.find(value);
                sa.add_ability = (it != svars.end()) ? it->second : value;
            } else if (key == "Affected") {
                sa.affected = value;
                // Also store as affected_subtype for untap prevention (Choke: Affected$ Island)
                if (sa.category == "Continuous" && value.find("EquippedBy") == std::string::npos) {
                    sa.affected_subtype = value;
                }
                // Per-source counter gate (Kaito: Affected$ Permanent.Self+counters_GE1_LOYALTY).
                // Forge spells the qualifier "counters_<CMP><N>_<TYPE>" (e.g. counters_GE1_LOYALTY =
                // "the source has 1 or more LOYALTY counters"). Extract compare ("GE1") and counter
                // type ("LOYALTY") so gather_active_statics can AND it into the static's condition.
                {
                    size_t cp = value.find("counters_");
                    if (cp != std::string::npos) {
                        std::string rest = value.substr(cp + 9);  // after "counters_"
                        size_t end = rest.find_first_of(".+");    // stop at the next qualifier
                        if (end != std::string::npos) rest = rest.substr(0, end);
                        size_t us = rest.find('_');               // split "<CMP><N>_<TYPE>"
                        if (us != std::string::npos) {
                            sa.self_counter_compare = rest.substr(0, us);
                            sa.self_counter_type = rest.substr(us + 1);
                        }
                    }
                }
            } else if (key == "Amount") {
                // Used by RaiseCost / ReduceCost (generic mana added to / removed from cost)
                // and SetCost (the minimum total mana value a cost floor raises spells up to).
                if (!value.empty() && std::isdigit(static_cast<unsigned char>(value[0]))) {
                    if (sa.category == "ReduceCost")
                        sa.reduce_cost = std::stoi(value);
                    else if (sa.category == "SetCost")
                        sa.set_cost_min = std::stoi(value);
                    else
                        sa.raise_cost = std::stoi(value);
                } else if (sa.category == "RaiseCost" && !value.empty()) {
                    // Non-numeric Amount$ — an SVar reference (Damping Sphere: Amount$ X with
                    // X = Count$ThisTurnCast_Card.YouCtrl). A "spells you cast this turn" count is
                    // the per-cast relative surcharge; resolve the SVar and flag it so the cost
                    // computation adds the caster's spells-cast-this-turn count (CR 601.2f).
                    auto it = svars.find(value);
                    const std::string body = (it != svars.end()) ? it->second : value;
                    if (body.find("ThisTurnCast") != std::string::npos)
                        sa.raise_cost_per_spell_cast = true;
                }
            } else if (key == "RaiseTo") {
                // SetCost RaiseTo$ True (Trinisphere): the Amount$ is a FLOOR — raise a sub-Amount
                // total up to Amount, never lower a cost that is already at/above it.
                if (sa.category == "SetCost") sa.set_cost_raise_to = (value == "True");
            } else if (key == "Activator") {
                // ReduceCost Activator$ You (Eye of Ugin): the reduction applies only to
                // spells cast by the source's controller, not to everyone's spells.
                if (sa.category == "ReduceCost" && value == "You")
                    sa.reduce_cost_you_only = true;
            } else if (key == "ValidCard") {
                // Card.NamedCard restricts the static to the source's chosen card name
                // (RaiseCost / CantBeActivated on Disruptor Flute).
                if (value.find("NamedCard") != std::string::npos)
                    sa.match_named_card = true;
                if (sa.category == "RaiseCost") {
                    if (value.find("nonCreature") != std::string::npos)
                        sa.raise_cost_filter = "nonCreature";
                } else if (sa.category == "ReduceCost") {
                    // Full ValidCard$ filter spec (e.g. "Eldrazi.Colorless"); matched against
                    // each spell's card characteristics when computing its cast cost.
                    sa.reduce_cost_filter = value;
                } else if (sa.category == "CantAttack") {
                    // The creatures the can't-attack restriction applies to (Ensnaring Bridge:
                    // "Creature.powerGTX"). A dynamic 'X' in a power/toughness qualifier references
                    // SVar X; resolve and store it for evaluation against the source's controller.
                    sa.cant_attack_filter = value;
                    if (value.find('X') != std::string::npos) {
                        auto it = svars.find("X");
                        if (it != svars.end()) sa.cant_attack_x_svar = it->second;
                    }
                } else if (sa.category == "CantBeActivated") {
                    // Store the full type list (e.g. "Artifact" for Null Rod, or
                    // "Artifact,Creature,Planeswalker" for Clarion Conqueror). The
                    // NamedCard variant (Disruptor Flute) is handled via match_named_card
                    // above and leaves this filter empty.
                    if (!sa.match_named_card)
                        sa.cant_activate_card_filter = value;
                } else if (sa.category == "CantBeCast") {
                    sa.cant_cast_filter = value;
                } else if (sa.category == "SetCost") {
                    // The spells the cost floor applies to (Trinisphere: ValidCard$ Card = every
                    // spell). A bare "Card" is left empty (matches all) to skip a useless filter run.
                    if (value != "Card") sa.set_cost_filter = value;
                }
            } else if (key == "NumLimitEachTurn") {
                sa.cant_cast_limit_per_turn = std::stoi(value);
            } else if (key == "Caster") {
                // CantBeCast Caster$ Opponent (Voice of Victory): the restriction applies to
                // the source controller's opponents, not the controller themselves.
                if (value == "Opponent") sa.cant_cast_by_opponent = true;
            } else if (key == "Origin") {
                // CantBeCast Origin$ Graveyard,Library (Grafdigger's Cage): restrict casting
                // spells from these zones (e.g. flashback).
                if (sa.category == "CantBeCast") {
                    if (value.find("Graveyard") != std::string::npos) sa.cant_cast_from_graveyard = true;
                    if (value.find("Library")   != std::string::npos) sa.cant_cast_from_library   = true;
                }
            } else if (key == "AddHiddenKeyword") {
                sa.hidden_keyword = value;
            } else if (key == "ValidCause") {
                sa.disable_triggers_cause = value;
            } else if (key == "ValidMode") {
                sa.disable_triggers_mode = value;
            } else if (key == "CharacteristicDefining") {
                sa.characteristic_defining = (value == "True");
            } else if (key == "SetPower") {
                auto it = svars.find(value);
                sa.set_power_svar = (it != svars.end()) ? it->second : value;
            } else if (key == "SetToughness") {
                auto it = svars.find(value);
                std::string resolved = (it != svars.end()) ? it->second : value;
                // Resolve SVar$<name>/Plus.<N> pattern at parse time
                // e.g. "SVar$X/Plus.1" → resolve X from svars, append "/Plus.1"
                if (resolved.rfind("SVar$", 0) == 0) {
                    size_t slash = resolved.find('/');
                    std::string ref_name = (slash != std::string::npos)
                        ? resolved.substr(5, slash - 5) : resolved.substr(5);
                    std::string suffix = (slash != std::string::npos)
                        ? resolved.substr(slash) : "";
                    auto ref_it = svars.find(ref_name);
                    if (ref_it != svars.end())
                        resolved = ref_it->second + suffix;
                }
                sa.set_toughness_svar = resolved;
            } else if (key == "AddType") {
                sa.add_type = value;
            } else if (key == "SetColor") {
                // Layer-5 color-changing static (Mycosynth Lattice). The colorset spec ("Colorless"
                // or a White/Blue/... list) is resolved in setcolor_override_for.
                sa.set_color = value;
            } else if (key == "AffectedZone") {
                sa.affected_zone = value;
            } else if (key == "ManaConversion") {
                // ManaConvert static (Mycosynth Lattice): AnyType->AnyColor lets any mana pay any
                // colored pip. Read by any_mana_as_any_color_active during mana payment.
                sa.mana_conversion = value;
            } else if (key == "RemoveLandTypes") {
                sa.remove_land_types = (value == "True");
            } else if (key == "RemoveCardTypes") {
                // RemoveCardTypes$ True (Kaito's self type-add): drop the source's original printed
                // card types when the static turns it into a creature. The layer-4 self-animate
                // applier preserves Planeswalker (CR 306 — "that's still a planeswalker").
                sa.remove_card_types = (value == "True");
            } else if (key == "RemoveAllAbilities") {
                sa.remove_all_abilities = (value == "True");
            } else if (key == "AdjustLandPlays") {
                if (!value.empty() && std::isdigit(static_cast<unsigned char>(value[0])))
                    sa.adjust_land_plays = std::stoi(value);
            } else if (key == "MayPlay") {
                if (value == "True") sa.may_play_from_graveyard = true;
            } else if (key == "CheckSVar") {
                auto it = svars.find(value);
                sa.check_svar_expr = (it != svars.end()) ? it->second : value;
            } else if (key == "SVarCompare") {
                sa.svar_compare = value;
            } else if (key == "IsPresent") {
                // General present-count gate for a continuous static (Elvish Reclaimer:
                // +2/+2 while 3+ land cards are in your graveyard). Counted at SBA time in
                // gather_active_statics against PresentZone$/PresentCompare$.
                sa.present_filter = value;
            } else if (key == "PresentZone") {
                sa.present_zone = value;
            } else if (key == "PresentCompare") {
                sa.present_compare = value;
            } else if (key == "Target") {
                // A CantAttack with Target$ (e.g. "Target$ You" — "can't attack you") restricts
                // WHICH player can't be attacked rather than forbidding attacking outright. Only
                // the blanket form (no Target$) is implemented; flag the targeted form so it is
                // not mistaken for a global can't-attack below.
                if (sa.category == "CantAttack") cant_attack_targeted = true;
            }
        }

    // The CantAttack query treats a non-empty cant_attack_filter as a blanket "can't attack"
    // restriction; drop it for the targeted ("can't attack you") variant that isn't handled.
    if (sa.category == "CantAttack" && cant_attack_targeted) sa.cant_attack_filter.clear();

    return sa;
}

static std::vector<StaticAbility> parse_static_abilities(const std::string &script, const std::map<std::string, std::string> &svars) {
    std::vector<StaticAbility> result;
    for (const auto &line : multi_values_from_script(script, "S")) {
        // Skip alt cost lines (handled separately) and garbage matches
        if (line.find("AlternativeCost") != std::string::npos) continue;
        StaticAbility sa = parse_one_static_ability(line, svars);
        if (!sa.category.empty()) result.push_back(sa);
    }
    return result;
}

// Parses R: replacement-effect lines from a card script.
// Only the ETB-tapped pattern is recognised for now:
//   Event$ Moved | ValidCard$ Card.Self | Destination$ Battlefield | ReplaceWith$ ETBTapped
static std::vector<Effect::Replacement> parse_replacement_effects(const std::string& script,
                                                                   const std::map<std::string, std::string>& svars) {
    std::vector<Effect::Replacement> result;

    // Collect all R: lines
    std::vector<std::string> lines;
    size_t pos = 0;
    if (script.size() >= 2 && script[0] == 'R' && script[1] == ':') {
        size_t end = script.find('\n', 0);
        if (end == std::string::npos) end = script.size();
        std::string line = script.substr(2, end - 2);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines.push_back(line);
        pos = end;
    }
    while ((pos = script.find("\nR:", pos)) != std::string::npos) {
        pos += 3;
        size_t end = script.find('\n', pos);
        if (end == std::string::npos) end = script.size();
        std::string line = script.substr(pos, end - pos);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines.push_back(line);
        pos = end;
    }

    for (const auto& line : lines) {
        bool event_is_moved       = false;
        bool event_is_counter     = false;
        bool event_is_untap       = false;
        bool event_is_produce_mana = false;       // Event$ ProduceMana (Damping Sphere)
        std::string produce_valid_type;           // ValidCard$ <type> for a ProduceMana replacement ("Land")
        int produce_min_amount    = 1;            // ManaAmount$ GEN — minimum produced amount
        std::string untap_valid_subtype;  // ValidCard$ <subtype> for an Untap-prevention (Choke: Island)
        bool valid_card_self      = false;
        std::string valid_sa_filter;  // ValidSA$ spec on an R: line (e.g. "Spell.YouCtrl")
        bool dest_is_battlefield  = false;
        bool dest_is_graveyard_r  = false;
        bool replace_with_etb_tapped = false;
        bool replace_with_exile   = false;
        bool layer_cant_happen    = false;
        bool active_zones_battlefield = false;
        bool valid_card_opp_non_token = false;
        bool valid_card_uncast_creature = false;  // Containment Priest: Creature.!token+!wasCast
        bool prevent_true         = false;        // Prevent$ True — the event simply doesn't happen
        bool valid_lki_creature   = false;        // ValidLKI$ Creature.* — the moving card is (last known) a creature
        bool origin_graveyard     = false;        // Origin$ includes Graveyard
        bool origin_library       = false;        // Origin$ includes Library
        std::string replace_with_svar;  // the SVar named by ReplaceWith$ (e.g. "Exile"), used to inspect the actual zone-change effect

        size_t param_pos = 0;
        std::string key, value;
        while (next_param(line, param_pos, key, value)) {
            if      (key == "Event"       && value == "Moved")       event_is_moved          = true;
            else if (key == "Event"       && value == "Counter")    event_is_counter        = true;
            else if (key == "Event"       && value == "Untap")      event_is_untap          = true;
            else if (key == "Event"       && value == "ProduceMana") event_is_produce_mana  = true;
            else if (key == "ManaAmount"  && value.rfind("GE", 0) == 0) {
                // ManaAmount$ GEN — applies when a source is tapped for >= N mana (Damping Sphere: GE2).
                std::string n = value.substr(2);
                produce_min_amount = n.empty() ? 1 : std::stoi(n);
            }
            else if (key == "ValidSA")    valid_sa_filter          = value;
            else if (key == "ValidCard"   && value == "Card.Self")   valid_card_self         = true;
            else if (key == "ValidCard"   && value.find('.') == std::string::npos) {
                untap_valid_subtype = value;  // a bare subtype filter (Choke: ValidCard$ Island)
                produce_valid_type  = value;  // a bare type filter (Damping Sphere: ValidCard$ Land)
            }
            else if (key == "ValidCard"   && value.find("OppOwn") != std::string::npos &&
                     (value.find("!token") != std::string::npos ||
                      value.find("nonToken") != std::string::npos)) valid_card_opp_non_token = true;
            // Containment Priest: a non-token creature that wasn't cast (Creature.!token+!wasCast).
            else if (key == "ValidCard"   && value.find("Creature") != std::string::npos &&
                     value.find("!wasCast") != std::string::npos &&
                     (value.find("!token") != std::string::npos ||
                      value.find("nonToken") != std::string::npos)) valid_card_uncast_creature = true;
            else if (key == "Destination" && value == "Battlefield") dest_is_battlefield     = true;
            else if (key == "Destination" && value == "Graveyard")   dest_is_graveyard_r     = true;
            else if (key == "ReplaceWith" && value == "ETBTapped")   replace_with_etb_tapped = true;
            else if (key == "ReplaceWith") { replace_with_svar = value; if (value == "Exile") replace_with_exile = true; }
            else if (key == "Layer"       && value == "CantHappen") layer_cant_happen        = true;
            else if (key == "ActiveZones" && value == "Battlefield") active_zones_battlefield = true;
            else if (key == "Prevent"     && value == "True")        prevent_true             = true;
            else if (key == "ValidLKI"    && value.find("Creature") != std::string::npos)
                valid_lki_creature = true;
            else if (key == "Origin") {
                if (value.find("Graveyard") != std::string::npos) origin_graveyard = true;
                if (value.find("Library")   != std::string::npos) origin_library   = true;
            }
        }

        // Does the replacement's zone-change effect attach a void counter (Dauthi Voidwalker:
        // WithCountersType$ VOID)? Leyline of the Void omits it — a plain exile. We read this
        // off the SVar named by ReplaceWith$ rather than retagging the identical R: lines.
        bool replace_with_void_counter = false;
        // Conditional "enters tapped" (Ba Sing Se): ReplaceWith$ <SVar> where the SVar is a
        // DB$ Tap | ETB$ True with a ConditionPresent$/ConditionCompare$ gate ("enters tapped
        // unless you control a basic land"). Detected here off the named SVar's body so the
        // identical R: line isn't retagged. An ETBTapped token (above) is the unconditional form.
        bool replace_with_etb_tapped_conditional = false;
        std::string tapped_cond_filter, tapped_cond_compare;
        int tapped_unless_life = 0;  // UnlessCost$ PayLife<N> — pay N life to enter untapped instead
        // ProduceMana replacement (Damping Sphere): the ReplaceWith$ SVar is a DB$ ReplaceMana
        // whose ReplaceMana$ names the single color all the produced mana is converted to.
        bool replace_with_produce_mana = false;
        Colors produce_replacement_color = COLORLESS;
        if (!replace_with_svar.empty()) {
            auto sv = svars.find(replace_with_svar);
            if (sv != svars.end()) {
                const std::string &body = sv->second;
                if (body.find("VOID") != std::string::npos) replace_with_void_counter = true;
                if (body.find("DB$ ReplaceMana") != std::string::npos) {
                    replace_with_produce_mana = true;
                    size_t pp = 0; std::string k, v;
                    while (next_param(body, pp, k, v)) {
                        if (k != "ReplaceMana" || v.empty()) continue;
                        switch (v[0]) {
                            case 'W': produce_replacement_color = WHITE;     break;
                            case 'U': produce_replacement_color = BLUE;      break;
                            case 'B': produce_replacement_color = BLACK;     break;
                            case 'R': produce_replacement_color = RED;       break;
                            case 'G': produce_replacement_color = GREEN;     break;
                            default:  produce_replacement_color = COLORLESS; break;
                        }
                    }
                }
                if (body.find("DB$ Tap") != std::string::npos &&
                    body.find("ETB$ True") != std::string::npos) {
                    replace_with_etb_tapped_conditional = true;
                    // Pull ConditionPresent$ / ConditionCompare$ / UnlessCost$ out of the SVar body.
                    size_t pp = 0; std::string k, v;
                    while (next_param(body, pp, k, v)) {
                        if (k == "ConditionPresent") {
                            // Drop a "+YouCtrl" qualifier — the condition is always evaluated
                            // controller-relative (the permanent's controller as it enters).
                            std::string f = v;
                            size_t plus = f.find("+YouCtrl");
                            if (plus != std::string::npos) f.erase(plus);
                            tapped_cond_filter = f;
                        } else if (k == "ConditionCompare") {
                            tapped_cond_compare = v;
                        } else if (k == "UnlessCost" && v.rfind("PayLife<", 0) == 0) {
                            // "...unless you pay N life" — the controller may pay N life as the
                            // permanent enters to have it enter untapped (Witch-Blessed Meadow,
                            // shock lands). Parse the N out of PayLife<N>.
                            size_t lt = v.find('<'), gt = v.find('>');
                            if (lt != std::string::npos && gt != std::string::npos && gt > lt + 1)
                                tapped_unless_life = std::stoi(v.substr(lt + 1, gt - lt - 1));
                        }
                    }
                }
            }
        }

        if (event_is_moved && valid_card_self && dest_is_battlefield && replace_with_etb_tapped) {
            Effect::Replacement r;
            r.kind = Effect::Replacement::ENTERS_TAPPED;
            r.applies_to_self_only = true;
            result.push_back(r);
        }
        // Conditional "enters tapped" (Ba Sing Se): an ENTERS_TAPPED replacement whose
        // application is gated on the controller's board (e.g. "unless you control a basic land").
        if (event_is_moved && valid_card_self && dest_is_battlefield &&
            replace_with_etb_tapped_conditional) {
            Effect::Replacement r;
            r.kind = Effect::Replacement::ENTERS_TAPPED;
            r.applies_to_self_only = true;
            r.tapped_condition_filter = tapped_cond_filter;
            r.tapped_condition_compare = tapped_cond_compare;
            r.tapped_unless_life = tapped_unless_life;
            result.push_back(r);
        }
        if (event_is_counter && valid_card_self && layer_cant_happen) {
            Effect::Replacement r;
            r.kind = Effect::Replacement::CANT_BE_COUNTERED;
            r.applies_to_self_only = true;
            result.push_back(r);
        }
        // Hexing Squelcher: "Spells you control can't be countered." A continuous, battlefield-active
        // (ActiveZones$ Battlefield) can't-be-countered replacement scoped by a ValidSA$ filter
        // (e.g. Spell.YouCtrl) rather than the source spell itself. Consulted at counter-resolution
        // time against every spell on the stack (614.13/CantHappen).
        if (event_is_counter && layer_cant_happen && active_zones_battlefield && !valid_card_self &&
            !valid_sa_filter.empty()) {
            Effect::Replacement r;
            r.kind = Effect::Replacement::CANT_BE_COUNTERED;
            r.applies_to_self_only = false;
            r.from_battlefield = true;
            r.valid_sa_filter = valid_sa_filter;
            result.push_back(r);
        }
        // Opponent's non-token cards go to exile instead of graveyard (Dauthi Voidwalker exiles
        // with a void counter; Leyline of the Void plain-exiles — distinguished by the SVar above).
        if (event_is_moved && dest_is_graveyard_r && replace_with_exile &&
            valid_card_opp_non_token && active_zones_battlefield) {
            Effect::Replacement r;
            r.kind = Effect::Replacement::EXILE_INSTEAD_OF_GRAVEYARD;
            r.applies_to_self_only = false;
            r.with_void_counter = replace_with_void_counter;
            result.push_back(r);
        }
        // Containment Priest: a non-token creature that wasn't cast is exiled instead of
        // entering the battlefield (614.1a). The replacement applies to any such creature
        // entering the battlefield while this permanent is on the battlefield.
        if (event_is_moved && dest_is_battlefield && replace_with_exile &&
            valid_card_uncast_creature && active_zones_battlefield) {
            Effect::Replacement r;
            r.kind = Effect::Replacement::EXILE_INSTEAD_OF_ETB;
            r.applies_to_self_only = false;
            result.push_back(r);
        }
        // Grafdigger's Cage: creature cards in graveyards and libraries can't enter the
        // battlefield (614.13). This is a prevention (Prevent$ True) — the moving card simply
        // doesn't enter and stays in its origin zone, distinct from Containment Priest's
        // exile-instead replacement above.
        if (event_is_moved && dest_is_battlefield && prevent_true && valid_lki_creature &&
            active_zones_battlefield && (origin_graveyard || origin_library)) {
            Effect::Replacement r;
            r.kind = Effect::Replacement::PREVENT_ETB_FROM_ZONES;
            r.applies_to_self_only = false;
            r.prevent_from_graveyard = origin_graveyard;
            r.prevent_from_library = origin_library;
            result.push_back(r);
        }
        // Choke: matching lands don't untap during their controllers' untap steps (614.1d).
        if (event_is_untap && layer_cant_happen && !untap_valid_subtype.empty()) {
            Effect::Replacement r;
            r.kind = Effect::Replacement::SKIP_UNTAP;
            r.applies_to_self_only = false;
            r.valid_subtype = untap_valid_subtype;
            result.push_back(r);
        }
        // Damping Sphere: "If a land is tapped for two or more mana, it produces {C} instead of
        // any other type and amount." A ProduceMana replacement (614.1) — when a permanent of the
        // named type is tapped for >= ManaAmount mana, replace its production with that much of the
        // ReplaceMana color.
        if (event_is_produce_mana && replace_with_produce_mana && active_zones_battlefield &&
            !produce_valid_type.empty()) {
            Effect::Replacement r;
            r.kind = Effect::Replacement::PRODUCE_MANA;
            r.applies_to_self_only = false;
            r.produce_valid_type = produce_valid_type;
            r.produce_min_amount = produce_min_amount;
            r.produce_replacement_color = produce_replacement_color;
            result.push_back(r);
        }
        // Grim Monolith: "This artifact doesn't untap during your untap step." (614.1d) — a
        // self-referential untap-prevention (ValidCard$ Card.Self). ValidStepTurnToController$ You
        // is implicit in a 2-player game (a permanent only untaps during its controller's untap
        // step), so it needs no extra gating here.
        if (event_is_untap && layer_cant_happen && valid_card_self && untap_valid_subtype.empty()) {
            Effect::Replacement r;
            r.kind = Effect::Replacement::SKIP_UNTAP;
            r.applies_to_self_only = true;
            result.push_back(r);
        }
    }

    return result;
}