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
#include "type_constants.h"

extern std::string RESOURCE_DIR;

const size_t SCRIPT_MAX_LEN = 10000;

static std::string value_from_script(std::string script, std::string key);
static std::vector<std::string> multi_values_from_script(std::string script, std::string key);
static std::multiset<Colors> parse_mana_cost(std::string value, std::vector<Colors> *phyrexian_out = nullptr);
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
            if (angle != std::string::npos && close != std::string::npos && close > angle + 1)
                ability.life_cost = std::stoi(tok.substr(angle + 1, close - angle - 1));
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
            int n = std::stoi(tok.substr(angle + 1, slash - angle - 1));
            ability.loyalty_cost = (tok[0] == 'S') ? -n : n;
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
    card.mana_cost = parse_mana_cost(mana_cost_str, &card.phyrexian_mana);
    card.has_x_cost = (mana_cost_str.find('X') != std::string::npos);
    card.types = parse_types(value_from_script(front_script, "Types"));
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
        // K:Ward:N — "Whenever this permanent becomes the target of a spell or ability an
        // opponent controls, counter that spell or ability unless that player pays {N}."
        // (CR 702.21). Stored as the keyword + a numeric cost; the becomes-targeted trigger
        // is synthesized when a targeting spell/ability is put on the stack.
        if (kw_line.rfind("Ward", 0) == 0) {
            size_t colon = kw_line.find(':');
            if (colon != std::string::npos)
                card.ward_cost = std::stoi(kw_line.substr(colon + 1));
            else
                card.ward_cost = 1;  // K:Ward without a cost defaults to {1}
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
            if (!sub.empty() && sub[0] == ':') {
                size_t c1 = sub.find(':', 1);
                if (c1 != std::string::npos) {
                    counter_type_str = sub.substr(1, c1 - 1);
                    size_t c2 = sub.find(':', c1 + 1);
                    std::string svar_key = (c2 != std::string::npos)
                        ? sub.substr(c1 + 1, c2 - c1 - 1)
                        : sub.substr(c1 + 1);
                    auto svar_it = svars.find(svar_key);
                    if (svar_it != svars.end()) {
                        if (svar_it->second.find("ExiledWithSource") != std::string::npos)
                            from_delve = true;
                        // Count$xPaid — the count equals the X value paid at cast time
                        // (Chalice of the Void enters with X charge counters).
                        else if (svar_it->second.find("xPaid") != std::string::npos)
                            from_xpaid = true;
                    }
                }
            }
            StaticAbility sa;
            sa.category = "EtbCounter";
            sa.counter_type = counter_type_str;
            sa.counter_count_from_delve = from_delve;
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
            card.keywords.push_back("Flashback");
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

    return tok;
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

static std::multiset<Colors> parse_mana_cost(std::string value, std::vector<Colors> *phyrexian_out) {
    auto len = value.length();
    std::multiset<Colors> ret_val;
    if (value == "no cost") return ret_val;
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
        key == "TokenAmount" || key == "ScryNum") {
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
        ability.unless_generic_cost = static_cast<size_t>(std::stoi(value));
    } else if (key == "LifeAmount") {
        if (!value.empty() && std::isdigit(static_cast<unsigned char>(value[0]))) {
            ability.amount = static_cast<size_t>(std::stoi(value));
        } else if (!value.empty()) {
            ability.amount_svar = value;
        }
    } else if (key == "TargetType") {
        ability.target_type = value;  // "Spell", "Activated,Triggered", etc.
    } else if (key == "Optional") {
        ability.optional_choice = (value == "True");
    } else if (key == "Defined" || key == "DefinedPlayer") {
        if (value == "Remembered") ability.defined_remembered = true;
        // Defined$ TriggeredSpellAbility — the effect acts on the spell that fired this
        // trigger (Chalice of the Void: "counter that spell"). Set at trigger fire time.
        else if (value == "TriggeredSpellAbility" || value == "TriggeredSpell")
            ability.defined_triggered_spell = true;
        else if (value == "TargetedController") ability.defined_targeted_controller = true;
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
    } else if (key == "StaticAbilities") {
        // DB$ Effect | StaticAbilities$ <name> — names the continuous static the transient
        // effect grants (e.g. Unblockable). Stored for the Effect handler to interpret.
        ability.effect_static_ability = value;
    } else if (key == "TgtZone") {
        if (value == "Graveyard") ability.target_in_graveyard = true;
    } else if (key == "ClearRemembered") {
        ability.clear_remembered = (value == "True");
    } else if (key == "ClearChosenCard") {
        ability.clear_chosen = (value == "True");
    } else if (key == "ChooseEach") {
        ability.choose_each = value;
    } else if (key == "TargetMin") {
        ability.target_min = std::stoi(value);
    } else if (key == "TargetMax") {
        // A numeric cap (TargetMax$ 3) is used directly. A count-SVar cap
        // (TargetMax$ MaxTgts, where MaxTgts = Count$ValidStack Card) means "any
        // number of targets" — there is no fixed upper bound, so treat it as
        // effectively unlimited; the multi-target selection loop stops on its own
        // once no further legal targets remain (Mindbreak Trap: exile any number of
        // target spells).
        if (!value.empty() && std::isdigit(static_cast<unsigned char>(value[0])))
            ability.target_max = std::stoi(value);
        else
            ability.target_max = MAX_ENTITIES;
    } else if (key == "ActivationZone") {
        if (value == "Hand") ability.activation_zone = Zone::HAND;
    } else if (key == "ActivationLimit") {
        ability.activation_limit = std::stoi(value);
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
    } else if (key == "Planeswalker") {
        ability.is_loyalty_ability = (value == "True");
    } else if (key == "Cost") {
        parse_activation_cost(value, ability);
    } else if (key == "ConditionPresent") {
        ability.condition_present = value;
    } else if (key == "ConditionDefined") {
        // "Targeted" → the condition is evaluated against the chosen target at
        // resolution, so it must not gate cast-time legality.
        ability.condition_on_target = (value == "Targeted");
        // "Remembered" → condition_present is counted over the remembered cards at
        // resolution (Birthing Ritual: only dig if a creature was sacrificed).
        ability.condition_on_remembered = (value == "Remembered");
    } else if (key == "ConditionCompare") {
        ability.condition_compare = value;
    } else if (key == "SacValid") {
        ability.sac_valid = value;            // DB$ Sacrifice — what may be sacrificed
    } else if (key == "RememberSacrificed") {
        ability.remember_sacrificed = (value == "True");
    } else if (key == "RepeatPlayers") {
        ability.repeat_players = value;       // RepeatEach over players (Price of Progress)
    } else if (effects::apply_parse_hook(ability, key, value)) {
        // Consumed by an effect-specific parse hook co-located with its handler.
    } else {
        static const std::set<std::string> ignored_keys = {
            "SpellDescription", "AILogic", "AINoRecursiveCheck", "TgtPrompt", "StackDescription",
            "ConditionDescription",
            // sameName search/move (Surgical Extraction, Extirpate, ...): these refine who
            // chooses or how the search is hidden, but the change_zone_same_name handler
            // already derives the full behavior from ChangeType/Origin/Destination/Defined.
            // Shuffle$ is inferred from a Library origin; Chooser/Hidden/ForgetOtherTargets
            // are cosmetic given the "move the maximum" simplification.
            "Chooser", "Hidden", "Shuffle", "ForgetOtherTargets", "RememberRevealed",
            // ChooseCard ChooseEach (Ajani -4): the per-type breakdown is the load-bearing
            // ChooseEach$; Choices$ (the umbrella pool), ControlledByPlayer$ Chooser, and
            // Reveal$ are captured by / cosmetic to the choose_each handler.
            "Choices", "ControlledByPlayer", "Reveal",
            // Ultimate$ True is informational: ultimate legality is already covered by the
            // minus-loyalty cost check, so the flag is unused.
            "Ultimate",
            // DamageMap$ True (RepeatEach DealDamage, e.g. Price of Progress): a Forge
            // bookkeeping flag that the per-iteration damage is collected into one
            // simultaneous damage event. Our resolution deals each player's damage in the
            // repeat loop; the simultaneity is cosmetic for a one-shot instant.
            "DamageMap",
            // Announce$ X (Kozilek's Command): declares the X to announce while casting. The
            // X cost is already auto-detected from a ManaCost containing X (has_x_cost), so
            // the announce is prompted regardless; the tag is informational here.
            "Announce"
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
    size_t db_pos = content.find("DB$");
    if (db_pos == std::string::npos) return sub;
    size_t p = db_pos + 4;  // skip "DB$ "
    size_t cat_end = content.find_first_of(" |", p);
    if (cat_end == std::string::npos) cat_end = content.length();
    if (cat_end > p)
        sub.category = normalize_category(content.substr(p, cat_end - p));

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
        } else if (key == "Execute") {
            // Execute$ references an SVar containing the ability to fire (delayed triggers)
            effect_params<DelayedTriggerParams>(sub).execute_svar = value;
            auto it = svars.find(value);
            if (it != svars.end())
                sub.subabilities.push_back(parse_svar_ability(it->second, ability_type, svars, card_name));
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
        } else {
            apply_param_to_ability(sub, key, value, card_name);
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
                       // Count$xPaid — amount equals the X paid at cast (Kozilek's Command:
                       // TokenAmount$/ScryNum$/TargetMax$ all = X = Count$xPaid).
                       sv.find("xPaid") != std::string::npos) {
                sub.dynamic_amount_expr = sv;
            }
        }
        sub.amount_svar = "";
    }
    // Resolve a Pump NumAtt$/NumDef$ given as a count-SVar (e.g. "+X", X = Count$Valid
    // Eldrazi.YouCtrl): turn the stored SVar key into its runtime Count$ expression so the
    // pump effect can evaluate the magnitude at resolution (Eldrazi Linebreaker).
    if (auto *pp = std::get_if<PumpParams>(&sub.params)) {
        for (std::string *expr : {&pp->att_expr, &pp->def_expr}) {
            if (expr->empty()) continue;
            auto it = svars.find(*expr);
            if (it != svars.end()) *expr = it->second;
        }
    }
    // Resolve dig_num_expr SVar reference (e.g. "X" → "Count$Devotion.Blue")
    if (!sub.dig_num_expr.empty()) {
        auto it = svars.find(sub.dig_num_expr);
        if (it != svars.end()) {
            sub.dig_num_expr = it->second;
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

        // Parse pipe-delimited parameters — applies to all ability categories
        size_t param_pos = line.find("|", pos);
        std::string key, value;
        while (next_param(line, param_pos, key, value)) {
            if (key == "SubAbility" || key == "RepeatSubAbility") {
                // RepeatSubAbility$ (RepeatEach) resolves the same way as SubAbility$: the
                // value names an SVar holding a DB$ ability. For RepeatEach the parsed
                // sub-ability is the per-iteration body the handler resolves once per player.
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
                           sv.find("Count$Revolt") != std::string::npos) {
                    // Runtime expression — preserve for evaluation at activation/resolve time
                    ability.dynamic_amount_expr = sv;
                }
            }
            ability.amount_svar = "";
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
    bool trigger_zone_is_graveyard = false;
    bool valid_player_is_you = false;
    bool mode_is_spell_cast = false;
    bool mode_is_damage_done = false;
    bool damage_combat_only = false;
    bool valid_card_non_creature = false;
    bool valid_card_instant = false;
    bool valid_card_sorcery = false;
    bool valid_card_owner_you = false;
    bool valid_card_land = false;
    bool valid_card_artifact = false;
    bool valid_card_colorless = false;
    bool mode_is_drawn = false;
    bool valid_card_opp_own = false;
    bool exclude_first_draw_step = false;
    bool trigger_optional_local = false;
    std::string valid_card_subtype;
    size_t activator_this_turn_cast_eq = 0;

    // Walk pipe-delimited params
    size_t param_pos = 0;
    std::string key, value;
    while (next_param(line, param_pos, key, value)) {
        if (key == "Mode") {
            // ChangesZoneAll ("one or more ... ") is matched per changing card like
            // ChangesZone; the once-per-batch nuance is elided (it differs only when
            // multiple matching cards change zone simultaneously).
            if (value == "ChangesZone" || value == "ChangesZoneAll") mode_changes_zone = true;
            else if (value == "Phase") mode_is_phase = true;
            else if (value == "SpellCast") mode_is_spell_cast = true;
            else if (value == "DamageDone") mode_is_damage_done = true;
            else if (value == "Drawn") mode_is_drawn = true;
        } else if (key == "Phase") {
            if (value == "Upkeep")   phase_is_upkeep   = true;
            // Forge writes the end step as either "EndStep" or "End of Turn".
            if (value == "EndStep" || value == "End of Turn")  phase_is_end_step = true;
            if (value == "Draw")     phase_is_draw     = true;
            if (value == "BeginCombat") phase_is_begin_combat = true;
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
            if (value == "Card.Self")                            valid_card_self         = true;
            if (value.find("Instant")     != std::string::npos) valid_card_instant      = true;
            if (value.find("Sorcery")     != std::string::npos) valid_card_sorcery      = true;
            if (value.find(".YouOwn")     != std::string::npos) valid_card_owner_you    = true;
            if (value.find(".OppOwn")     != std::string::npos) valid_card_opp_own      = true;
            if (value.find("Land")        != std::string::npos) valid_card_land         = true;
            if (value.find("Artifact")    != std::string::npos) valid_card_artifact     = true;
            if (value.find("Colorless")   != std::string::npos) valid_card_colorless    = true;
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
            // Intervening-if (603.4) gated on an SVar count rather than a board presence,
            // e.g. Ocelot Pride's "if you gained life this turn" (CheckSVar$ YouLifeGained →
            // Count$LifeYouGainedThisTurn). Resolve the SVar to its Count$ expression and store
            // it as the intervening-if condition so the whole trigger fizzles when false.
            auto it = svars.find(value);
            ability.condition_present = (it != svars.end()) ? it->second : value;
            ability.intervening_if = true;
        } else if (key == "SVarCompare") {
            ability.condition_compare = value;  // explicit compare for the CheckSVar gate
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

    // "whenever you cast your Nth spell" — Cori-Steel Cutter
    if (mode_is_spell_cast && activator_this_turn_cast_eq > 0) {
        ability.trigger_on = Events::SPELL_CAST;
        ability.trigger_valid_player_is_controller = valid_player_is_you;
        ability.trigger_spell_count_eq = activator_this_turn_cast_eq;
    }

    // "Whenever a player casts a spell with mana value equal to ..." — Chalice of the Void
    // (Mode$ SpellCast | ValidCard$ Card.cmcEQY | ValidActivatingPlayer$ Player). A dynamic
    // mana-value filter on any player's spell. The cmc match is checked at trigger time.
    if (mode_is_spell_cast && !ability.trigger_cmc_expr.empty()) {
        ability.trigger_on = Events::SPELL_CAST;
        ability.trigger_valid_player_is_controller = valid_player_is_you;
    }

    // "Whenever CARDNAME deals combat damage to a player" — Barrowgoyf
    if (mode_is_damage_done && damage_combat_only) {
        ability.trigger_on = Events::COMBAT_DAMAGE_TO_PLAYER;
        ability.trigger_only_self = true;  // ValidSource$ Card.Self
    }

    // "whenever a player draws a card" — Orcish Bowmasters (Mode$ Drawn)
    if (mode_is_drawn) {
        ability.trigger_on = Events::PLAYER_DREW_CARD;
        ability.trigger_valid_card_opp_own = valid_card_opp_own;
        ability.trigger_exclude_first_draw_step = exclude_first_draw_step;
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
                effect.trigger_zone_origin                      = ability.trigger_zone_origin;
                effect.trigger_zone_destination                 = ability.trigger_zone_destination;
                effect.trigger_valid_card_is_creature           = ability.trigger_valid_card_is_creature;
                effect.trigger_valid_card_is_instant_or_sorcery = ability.trigger_valid_card_is_instant_or_sorcery;
                effect.trigger_valid_card_is_land               = ability.trigger_valid_card_is_land;
                effect.trigger_valid_card_is_artifact           = ability.trigger_valid_card_is_artifact;
                effect.trigger_valid_card_colorless             = ability.trigger_valid_card_colorless;
                effect.trigger_valid_card_subtype               = ability.trigger_valid_card_subtype;
                effect.trigger_optional                         = ability.trigger_optional;
                effect.trigger_valid_card_opp_own               = ability.trigger_valid_card_opp_own;
                effect.trigger_exclude_first_draw_step          = ability.trigger_exclude_first_draw_step;
                effect.trigger_valid_player_is_controller       = ability.trigger_valid_player_is_controller;
                effect.trigger_only_self                        = ability.trigger_only_self;
                effect.trigger_self_excluded                    = ability.trigger_self_excluded;
                effect.trigger_spell_count_eq                   = ability.trigger_spell_count_eq;
                effect.trigger_cmc_expr                         = ability.trigger_cmc_expr;
                effect.trigger_cmc_op                           = ability.trigger_cmc_op;
                effect.trigger_from_graveyard                   = ability.trigger_from_graveyard;
                // 603.4 intervening-if lives on the trigger line, not the Execute SVar — carry
                // it onto the resolved ability so it is re-checked at resolution.
                effect.intervening_if                           = ability.intervening_if;
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

static std::vector<StaticAbility> parse_static_abilities(const std::string &script, const std::map<std::string, std::string> &svars) {
    std::vector<StaticAbility> result;
    for (const auto &line : multi_values_from_script(script, "S")) {
        // Skip alt cost lines (handled separately) and garbage matches
        if (line.find("AlternativeCost") != std::string::npos) continue;
        if (line.find("Mode$") == std::string::npos) continue;

        StaticAbility sa;
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
            } else if (key == "Affected") {
                sa.affected = value;
                // Also store as affected_subtype for untap prevention (Choke: Affected$ Island)
                if (sa.category == "Continuous" && value.find("EquippedBy") == std::string::npos) {
                    sa.affected_subtype = value;
                }
            } else if (key == "Amount") {
                // Used by RaiseCost
                if (!value.empty() && std::isdigit(static_cast<unsigned char>(value[0])))
                    sa.raise_cost = std::stoi(value);
            } else if (key == "ValidCard") {
                // Card.NamedCard restricts the static to the source's chosen card name
                // (RaiseCost / CantBeActivated on Disruptor Flute).
                if (value.find("NamedCard") != std::string::npos)
                    sa.match_named_card = true;
                if (sa.category == "RaiseCost") {
                    if (value.find("nonCreature") != std::string::npos)
                        sa.raise_cost_filter = "nonCreature";
                } else if (sa.category == "CantBeActivated") {
                    // Store the full type list (e.g. "Artifact" for Null Rod, or
                    // "Artifact,Creature,Planeswalker" for Clarion Conqueror). The
                    // NamedCard variant (Disruptor Flute) is handled via match_named_card
                    // above and leaves this filter empty.
                    if (!sa.match_named_card)
                        sa.cant_activate_card_filter = value;
                } else if (sa.category == "CantBeCast") {
                    sa.cant_cast_filter = value;
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
            } else if (key == "RemoveLandTypes") {
                sa.remove_land_types = (value == "True");
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
            }
        }

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
        std::string untap_valid_subtype;  // ValidCard$ <subtype> for an Untap-prevention (Choke: Island)
        bool valid_card_self      = false;
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
            else if (key == "ValidCard"   && value == "Card.Self")   valid_card_self         = true;
            else if (key == "ValidCard"   && value.find('.') == std::string::npos)
                untap_valid_subtype = value;  // a bare subtype filter (Choke: ValidCard$ Island)
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
        if (!replace_with_svar.empty()) {
            auto sv = svars.find(replace_with_svar);
            if (sv != svars.end() && sv->second.find("VOID") != std::string::npos)
                replace_with_void_counter = true;
        }

        if (event_is_moved && valid_card_self && dest_is_battlefield && replace_with_etb_tapped) {
            Effect::Replacement r;
            r.kind = Effect::Replacement::ENTERS_TAPPED;
            r.applies_to_self_only = true;
            result.push_back(r);
        }
        if (event_is_counter && valid_card_self && layer_cant_happen) {
            Effect::Replacement r;
            r.kind = Effect::Replacement::CANT_BE_COUNTERED;
            r.applies_to_self_only = true;
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
    }

    return result;
}