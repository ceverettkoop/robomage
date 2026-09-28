#include "parse.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <unordered_map>
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
#include "queries/filters.h"
#include "str_util.h"
#include "type_constants.h"

extern std::string RESOURCE_DIR;

const size_t SCRIPT_MAX_LEN = 10000;

static std::string value_from_script(const std::string &script, const std::string &key);
static std::vector<std::string> multi_values_from_script(const std::string &script, const std::string &key);
static std::multiset<Colors> parse_mana_cost(std::string value, std::vector<Colors> *phyrexian_out = nullptr,
                                             std::vector<HybridPip> *hybrid_out = nullptr);
static void parse_alt_cost_tokens(const std::string& cost_str, AltCost& ac);
static std::set<Type> parse_types(std::string value);
static std::set<Colors> parse_colors_field(const std::string &colors_field);
static std::map<std::string, std::string> parse_svars(const std::string& script);
static std::string normalize_category(std::string category, const std::string& card_name);
static void apply_param_to_ability(AbilityDef& ability, const std::string& key, const std::string& value,
                                   const std::string& card_name = "");
static std::vector<AbilityDef> parse_abilities(const std::vector<std::string>& lines,
                                               const std::map<std::string, std::string>& svars,
                                               const std::string& card_name);
static std::vector<AbilityDef> parse_triggered_abilities(const std::string& script,
                                                      const std::map<std::string, std::string>& svars,
                                                      const std::string& card_name = "");
static std::vector<StaticAbility> parse_static_abilities(const std::string& script, const std::map<std::string, std::string>& svars);
static StaticAbility parse_one_static_ability(const std::string& line, const std::map<std::string, std::string>& svars);
static std::vector<Effect::Replacement> parse_replacement_effects(const std::string& script,
                                                                   const std::map<std::string, std::string>& svars);
static bool take_direct_amount_expr(AbilityDef& ability);
static void warn_unresolved_amount_svar(const AbilityDef& ability, const std::string& card_name);
static uint32_t parse_power(std::string value);
static uint32_t parse_toughness(std::string value);
static AbilityDef parse_one_trigger(const std::string &line, const std::map<std::string, std::string> &svars,
                                 const std::string& card_name);
static void split_keywords(const std::string& kw_line, std::vector<std::string>& out);
static bool next_param(const std::string& line, size_t& pos, std::string& key, std::string& value);
static std::string param_value(const std::string& line, const std::string& want_key);
static std::string svar_key_param(const std::string& line, const std::string& want_key);
// A card face's ability definitions while it is being parsed (still editable); parse_card_face
// interns them into the CardData once the face is complete.
struct FaceAbilityDefs {
    std::vector<AbilityDef> abilities;
    std::vector<AbilityDef> gift_abilities;
    std::vector<AbilityDef> saga_chapters;
    std::vector<AbilityDef> opening_hand_abilities;
};
// The facts a trigger line (a T: line or trigger SVar) states that decide which event its
// ability fires on and how that event is filtered, read in one pass over its params.
struct TriggerLine {
    std::string mode;                 // Mode$
    std::vector<std::string> phases;  // Phase$ list (Mode$ Phase)
    int origin = -1;                  // Origin$ / Destination$ zone filter (Mode$ ChangesZone); -1 = any
    int destination = -1;
    bool player_is_you = false;       // the event's player is the source's controller (ValidPlayer$ You, ...)
    bool valid_card_creature = false;
    bool valid_card_self = false;
    bool valid_card_non_creature = false;
    bool valid_card_colorless = false;
    bool valid_card_untapped = false;
    bool valid_card_opp_own = false;
    bool valid_card_opp_ctrl = false;
    int kicked_index = 0;             // ValidCard$ ...+kicked N — fires only if the Nth kicker was paid
    bool source_is_spell = false;     // ValidSource$ Spell...
    bool source_opp_ctrl = false;     // ValidSource$ ...OppCtrl
    bool source_creature_youctrl = false;  // ValidSource$ Creature.YouCtrl
    bool target_self = false;         // ValidTarget$ Card.Self
    bool is_static = false;           // Static$ True
    bool attacking_player_is_you = false;
    bool exclude_first_draw_step = false;
    bool optional = false;            // OptionalDecider$
    bool combat_damage_only = false;  // CombatDamage$ True
    bool from_graveyard = false;      // TriggerZones$ Graveyard
    size_t draw_number_eq = 0;        // Number$ N on a Mode$ Drawn trigger (Nth-draw gate)
    size_t cast_count_eq = 0;         // ActivatorThisTurnCast$ EQN
    // 603.4 intervening-if (IsPresent$/PresentCompare$ or a CheckSVar$ count gate).
    bool intervening_if = false;
    std::string condition_present;
    std::string condition_compare;
};
static TriggerLine read_trigger_line(const std::string& line,
                                     const std::map<std::string, std::string>& svars,
                                     AbilityDef& ability);
static int trigger_zone_filter(const std::string& value);
static void read_mana_spent_filter(const std::string& value, AbilityDef& ability);
static void read_trigger_valid_card(const std::string& value,
                                    const std::map<std::string, std::string>& svars, TriggerLine& t,
                                    AbilityDef& ability);
static void bind_trigger_line(const TriggerLine& t, AbilityDef& ability);
static void bind_phase_trigger(const TriggerLine& t, AbilityDef& ability);
static void bind_spell_cast_trigger(const TriggerLine& t, AbilityDef& ability);
// The face a K: keyword line is parsed into.
struct KeywordContext {
    CardData &card;
    FaceAbilityDefs &face_defs;
    const std::map<std::string, std::string> &svars;
};
// One K: keyword kKeywordTable handles: a line matching `name` (the whole line, its start, or
// anywhere in it) is parsed by `handler`.
struct KeywordEntry {
    enum Match { EXACT, PREFIX, CONTAINS };
    const char *name;
    Match match;
    void (*handler)(const std::string &kw_line, KeywordContext &ctx);
};
static void parse_keyword_line(const std::string& kw_line, KeywordContext& ctx);
static std::string keyword_arg(const std::string& kw_line);
static void add_keyword_alt_cost(const std::string& kw_line, const char* name, bool AltCost::*flag,
                                 CardData& card);
static AbilityDef keyword_activated_ability(const std::string& category, Zone::ZoneValue zone,
                                            const std::string& cost);
static AbilityDef keyword_self_trigger(const std::string& category, EventId event);
static void kw_companion(const std::string& kw_line, KeywordContext& ctx);
static void kw_enchant(const std::string& kw_line, KeywordContext& ctx);
static void kw_ward(const std::string& kw_line, KeywordContext& ctx);
static void kw_affinity(const std::string& kw_line, KeywordContext& ctx);
static void kw_etb_replacement(const std::string& kw_line, KeywordContext& ctx);
static void kw_etb_counter(const std::string& kw_line, KeywordContext& ctx);
static void kw_equip(const std::string& kw_line, KeywordContext& ctx);
static void kw_reconfigure(const std::string& kw_line, KeywordContext& ctx);
static void kw_impending(const std::string& kw_line, KeywordContext& ctx);
static void kw_suspend(const std::string& kw_line, KeywordContext& ctx);
static void kw_chapter(const std::string& kw_line, KeywordContext& ctx);
static void kw_dredge(const std::string& kw_line, KeywordContext& ctx);
static void kw_landwalk(const std::string& kw_line, KeywordContext& ctx);
static void kw_cycling(const std::string& kw_line, KeywordContext& ctx);
static void kw_ninjutsu(const std::string& kw_line, KeywordContext& ctx);
static void kw_type_cycling(const std::string& kw_line, KeywordContext& ctx);
static void kw_flashback(const std::string& kw_line, KeywordContext& ctx);
static void kw_unearth(const std::string& kw_line, KeywordContext& ctx);
static void kw_escape(const std::string& kw_line, KeywordContext& ctx);
static void kw_evoke(const std::string& kw_line, KeywordContext& ctx);
static void kw_offspring(const std::string& kw_line, KeywordContext& ctx);
static void kw_kicker(const std::string& kw_line, KeywordContext& ctx);
static void kw_replicate(const std::string& kw_line, KeywordContext& ctx);
static void kw_devoid(const std::string& kw_line, KeywordContext& ctx);
static void kw_gift(const std::string& kw_line, KeywordContext& ctx);
static void kw_opening_hand(const std::string& kw_line, KeywordContext& ctx);
static void kw_storm(const std::string& kw_line, KeywordContext& ctx);
static void kw_annihilator(const std::string& kw_line, KeywordContext& ctx);
static void kw_protection(const std::string& kw_line, KeywordContext& ctx);
// The K: keywords with their own parse, tried in order (the first match handles the line).
static const KeywordEntry kKeywordTable[] = {
    {"Delve", KeywordEntry::PREFIX,
     [](const std::string &, KeywordContext &ctx) {
         ctx.card.has_delve = true;
         ctx.card.keywords.push_back("Delve");
     }},
    // K:Improvise — your artifacts can help cast this spell; each untapped artifact you
    // tap after activating mana abilities pays for {1} of the generic cost (CR 702.126).
    // A cast-time generic cost reduction, mirroring Delve but tapping battlefield
    // artifacts instead of exiling graveyard cards.
    {"Improvise", KeywordEntry::PREFIX,
     [](const std::string &, KeywordContext &ctx) {
         ctx.card.has_improvise = true;
         ctx.card.keywords.push_back("Improvise");
     }},
    {"Companion:", KeywordEntry::PREFIX, kw_companion},
    {"Enchant:", KeywordEntry::PREFIX, kw_enchant},
    {"Ward", KeywordEntry::PREFIX, kw_ward},
    {"Affinity", KeywordEntry::PREFIX, kw_affinity},
    {"ETBReplacement", KeywordEntry::CONTAINS, kw_etb_replacement},
    {"etbCounter", KeywordEntry::PREFIX, kw_etb_counter},
    {"Equip", KeywordEntry::PREFIX, kw_equip},
    {"Reconfigure", KeywordEntry::PREFIX, kw_reconfigure},
    {"Impending", KeywordEntry::PREFIX, kw_impending},
    {"Suspend", KeywordEntry::PREFIX, kw_suspend},
    // K:Spectacle:<cost> — Spectacle (CR 702.107). An alternative casting cost: the spell may
    // be cast for <cost> instead of its normal mana cost, but only if an opponent lost life
    // this turn (CR 702.107a). can_afford_alt gates the offering on the opponent's
    // life_lost_this_turn.
    {"Spectacle", KeywordEntry::PREFIX,
     [](const std::string &kw_line, KeywordContext &ctx) {
         add_keyword_alt_cost(kw_line, "Spectacle", &AltCost::is_spectacle, ctx.card);
     }},
    // K:Warp:<cost> — Warp (a 2025 keyword; not in the checked-in CR snapshot). An alternative
    // casting cost: the spell may be cast from hand for <cost> instead of its normal mana cost.
    // If cast this way the object is exiled at the beginning of the next end step and may then
    // be cast from exile later for its normal cost (see effect_warp.cpp / the cast-with-warp
    // markers). can_afford_alt gates it purely on affordability of the warp cost.
    {"Warp", KeywordEntry::PREFIX,
     [](const std::string &kw_line, KeywordContext &ctx) {
         add_keyword_alt_cost(kw_line, "Warp", &AltCost::is_warp, ctx.card);
     }},
    // K:Miracle:<cost> — Miracle (CR 702.94). An alternative casting cost: when this card is
    // drawn as the FIRST card its controller drew this turn, they may reveal it and cast it
    // for <cost> instead of its normal mana cost. The qualifying-draw gate lives in orderer.cpp
    // (arms Game::pending.miracle_reveal); miracle then runs as two mandatory-choice decisions —
    // a private reveal and an immediate cast/do-not-cast — rather than a priority-menu alt cost.
    {"Miracle", KeywordEntry::PREFIX,
     [](const std::string &kw_line, KeywordContext &ctx) {
         add_keyword_alt_cost(kw_line, "Miracle", &AltCost::is_miracle, ctx.card);
     }},
    {"Chapter:", KeywordEntry::PREFIX, kw_chapter},
    // K:Prowess — keyword stored; triggered ability applied by apply_keyword_abilities
    {"Prowess", KeywordEntry::PREFIX,
     [](const std::string &, KeywordContext &ctx) { ctx.card.keywords.push_back("Prowess"); }},
    {"Dredge:", KeywordEntry::PREFIX, kw_dredge},
    {"Landwalk:", KeywordEntry::PREFIX, kw_landwalk},
    {"Cycling:", KeywordEntry::PREFIX, kw_cycling},
    {"Ninjutsu:", KeywordEntry::PREFIX, kw_ninjutsu},
    {"TypeCycling:", KeywordEntry::PREFIX, kw_type_cycling},
    {"Flashback:", KeywordEntry::PREFIX, kw_flashback},
    {"Unearth:", KeywordEntry::PREFIX, kw_unearth},
    {"Escape:", KeywordEntry::PREFIX, kw_escape},
    {"Evoke", KeywordEntry::PREFIX, kw_evoke},
    {"Offspring", KeywordEntry::PREFIX, kw_offspring},
    {"Kicker:", KeywordEntry::PREFIX, kw_kicker},
    {"Replicate:", KeywordEntry::PREFIX, kw_replicate},
    {"Devoid", KeywordEntry::EXACT, kw_devoid},
    {"Gift", KeywordEntry::PREFIX, kw_gift},
    {"MayEffectFromOpeningHand", KeywordEntry::PREFIX, kw_opening_hand},
    {"Storm", KeywordEntry::EXACT, kw_storm},
    {"Annihilator", KeywordEntry::PREFIX, kw_annihilator},
    {"Protection:", KeywordEntry::PREFIX, kw_protection},
};
static void parse_card_face_body(const std::string& front_script, CardData& card,
                                 FaceAbilityDefs& face_defs);
static void parse_card_face(const std::string& front_script, CardData& card);
// Parses the card script at `path` into `card`; false when the file can't be opened.
static bool parse_card_file(const std::string &path, CardData &card);
// Parses token script `script_name` into `tok`; false when the file can't be opened.
static bool parse_token_file(const std::string &script_name, Token &tok);
static AbilityDef equip_keyword_ability(const std::string &kw_line, const std::string &category,
                                     const std::string &label);
static AbilityDef parse_ability_text(const std::string& text, size_t category_pos,
                                     AbilityDef::AbilityType type,
                                     const std::map<std::string, std::string>& svars,
                                     const std::string& card_name);
static AbilityDef parse_svar_ability(const std::string& content, AbilityDef::AbilityType ability_type,
                                     const std::map<std::string, std::string>& svars,
                                     const std::string& card_name);
static bool apply_composite_param(AbilityDef& ability, const std::string& key, const std::string& value,
                                  const std::map<std::string, std::string>& svars,
                                  const std::string& card_name);
static void resolve_ability_svars(AbilityDef& ability, const std::string& text,
                                  const std::map<std::string, std::string>& svars,
                                  const std::string& card_name);
static void resolve_effect_static_svars(AbilityDef& ability, const std::string& text,
                                        const std::map<std::string, std::string>& svars);
static void resolve_amount_svar(AbilityDef& ability, const std::map<std::string, std::string>& svars,
                                const std::string& card_name);
static bool resolve_conditional_amount(AbilityDef& ability, const std::string& sv,
                                       const std::map<std::string, std::string>& svars);
static bool is_runtime_amount_expr(const std::string& sv);
static void resolve_condition_svars(AbilityDef& ability,
                                    const std::map<std::string, std::string>& svars);
static void resolve_xpaid_target_counts(AbilityDef& ability,
                                        const std::map<std::string, std::string>& svars,
                                        const std::string& line);
static void resolve_pump_exprs(AbilityDef& ability, const std::map<std::string, std::string>& svars);
static void resolve_destroyall_svars(AbilityDef& ability, const std::map<std::string, std::string>& svars);
static void resolve_additive_svar(const std::string& expr, const std::map<std::string, std::string>& svars,
                                  std::vector<std::string>& terms);
static bool find_cmc_bound(const std::string& filter, const char* only_op, std::string& op,
                           std::string& bound);

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

// Return the value of the first `want_key$` param in a `|`-delimited Forge line/SVar body,
// or "" if absent. A convenience over next_param for the one-off "just fetch this field"
// lookups (e.g. the ValidCard$ filter of a StaticAbilities$ SVar).
static std::string param_value(const std::string& line, const std::string& want_key) {
    size_t pos = 0;
    std::string key, value;
    while (next_param(line, pos, key, value))
        if (key == want_key) return value;
    return "";
}

// The value of `want_key` in an ability line when it names an SVar (a non-numeric token such as
// "X"); empty when the param is absent or a literal number.
static std::string svar_key_param(const std::string& line, const std::string& want_key) {
    std::string value = param_value(line, want_key);
    if (value.empty() || std::isdigit(static_cast<unsigned char>(value[0])) || value[0] == '-')
        return "";
    return value;
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
        } else if( ((value == '-') || (value == ' ') || (value == '/')) && (i != name.size() - 1) )   { // we will excise up to 1 trailing space, rest to underscores
                // '/' is a separator too (CR 709 split cards): a combined "Front/Back" reference
                // maps to Forge's underscore-joined filename (e.g. "Dead/Gone" -> "dead_gone" ->
                // cardsfolder/d/dead_gone.txt), matching how space/dash are already normalized.
                name[i] = '_';
        } else {
            to_rm.push_back(i);
        }
    }
    for (size_t i = 0; i < to_rm.size(); i++) {
        auto index_to_rm = to_rm[i] - i;
        name.erase(index_to_rm, 1);
    }

    // Collapse consecutive underscores to one. Removing a non-alphanumeric character that sat
    // between two separators (e.g. the "& " in "Raph & Mikey" — space->_, '&' excised, space->_)
    // leaves a doubled "__"; Forge's filenames join such names with a SINGLE underscore
    // (raph_mikey_troublemakers.txt). Collapsing here aligns name_to_uid with Forge's convention.
    // Safe: no shipped Forge script has a "__" in its filename, so this can't shadow another card.
    std::string collapsed;
    collapsed.reserve(name.size());
    for (char c : name) {
        if (c == '_' && !collapsed.empty() && collapsed.back() == '_') continue;
        collapsed.push_back(c);
    }

    return collapsed;
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
    // Sac<N/Type> — an ALTERNATIVE casting cost paid by sacrificing N permanents matching Type
    // (CR 118.9, Fireblast: "sacrifice two Mountains"). Mirrors the Sac token grammar in
    // parse_activation_cost: the count precedes the first '/', the filter follows it (a trailing
    // ';'/'/label' is stripped so "2/Mountain" and "1/Forest;Plains/label" both parse).
    size_t sc = cost_str.find("Sac<");
    if (sc != std::string::npos) {
        size_t slash = cost_str.find('/', sc);
        size_t close = cost_str.find('>', sc);
        if (slash != std::string::npos && close != std::string::npos && close > slash + 1) {
            ac.sac_cost_count = std::stoi(cost_str.substr(sc + 4, slash - (sc + 4)));
            std::string spec = cost_str.substr(slash + 1, close - slash - 1);
            size_t spec_slash = spec.find('/');
            if (spec_slash != std::string::npos) spec = spec.substr(0, spec_slash);
            ac.sac_cost_spec = spec;
            matched_special = true;
        }
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
static void parse_activation_cost(const std::string &cost_str, AbilityDef &ability) {
    size_t tok_pos = 0;
    while (tok_pos < cost_str.size()) {
        size_t tok_end = cost_str.find(' ', tok_pos);
        if (tok_end == std::string::npos) tok_end = cost_str.size();
        std::string tok = cost_str.substr(tok_pos, tok_end - tok_pos);
        if (tok == "T") {
            ability.tap_cost = true;
        } else if (tok == "Mandatory") {
            // "Mandatory" prefix on a triggered-ability cost (Dark Depths' TrigToken:
            // Cost$ Mandatory Sac<1/CARDNAME>). The sacrifice is a mandatory part of the
            // triggered ability's resolution, not an optional cost the controller may decline —
            // record it and, crucially, do NOT let it fall through to parse_mana_cost (which
            // would read the word as a spurious mana symbol). CR 603.8 / the card's "sacrifice
            // it. If you do, ...".
            ability.mandatory = true;
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
            // COUNT the pips: a cost may repeat X (Blast Zone's "Cost$ X X T" = {X}{X}), and the
            // one chosen X is then owed once per pip.
            ability.activation_has_x = true;
            ability.activation_x_count++;
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

// One of the sorcery-speed activated abilities an equip-style keyword line (K:Equip:<cost>,
// K:Reconfigure:<cost>) represents, with the cost after the first ':' and `label` shown in the
// action menu. `category` "Attach" is "[Cost]: Attach this permanent to target creature you
// control" (CR 702.6a, 702.151a), activatable only while the Equipment can equip some creature
// its controller controls (Activation$ gate "CanEquip", CR 301.5c); "Unattach" is reconfigure's
// "[Cost]: Unattach this permanent", activatable only while it is attached (gate "Attached").
static AbilityDef equip_keyword_ability(const std::string &kw_line, const std::string &category,
                                     const std::string &label) {
    AbilityDef ab;
    ab.ability_type = AbilityDef::ACTIVATED;
    ab.category = category;
    ab.sorcery_speed_only = true;
    ab.keyword_label = label;
    if (category == "Attach") {
        ab.valid_tgts = "Creature.Other+YouCtrl";
        ab.activation_condition = "CanEquip";
    } else {
        ab.activation_condition = "Attached";
    }
    size_t colon = kw_line.find(':');
    if (colon != std::string::npos) parse_activation_cost(kw_line.substr(colon + 1), ab);
    return ab;
}

Entity parse_card_script(std::string path) {
    // A script parses the same way every time, so each is parsed once per process: its ability
    // definitions are interned once, and a later load (the next game's init_ecs) reuses them.
    static std::unordered_map<std::string, CardData> parsed;
    auto it = parsed.find(path);
    if (it == parsed.end()) {
        CardData card;
        if (!parse_card_file(path, card)) return 0;
        it = parsed.emplace(path, std::move(card)).first;
    }
    auto id = global_coordinator.CreateEntity();
    global_coordinator.AddComponent(id, it->second);
    return id;
}

static bool parse_card_file(const std::string &path, CardData &card) {
    auto stream = std::ifstream(path);
    if (!stream.is_open()) {
        fprintf(stderr, "parse_card_script: failed to open '%s'\n", path.c_str());
        return false;
    }
    std::string script_data;
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

    parse_card_face(front_script, card);

    // Parse the DFC back face as a complete second face so a transformed permanent
    // (Delver -> Insectile Aberration, Ajani Pariah -> Avenger) has its own name,
    // types, P/T, loyalty, abilities, and triggers -- not just a P/T swap.
    if (!back_script.empty()) {
        auto backside = std::make_shared<CardData>();
        parse_card_face(back_script, *backside);
        card.backside = backside;
    }

    return true;
}

// Parses one card face (front or DFC back) into `card`: mana cost, types, colors,
// oracle text, P/T, starting loyalty, activated/spell abilities, triggered abilities,
// alternate costs, static abilities, replacement effects, and keywords. Shared by
// both faces so each face of a DFC is a fully-functional permanent definition.
static void parse_card_face_body(const std::string& front_script, CardData& card,
                                 FaceAbilityDefs& face_defs) {
    card.name = value_from_script(front_script, "Name");
    card.uid = name_to_uid(card.name);
    std::string mana_cost_str = value_from_script(front_script, "ManaCost");
    card.mana_cost = parse_mana_cost(mana_cost_str, &card.phyrexian_mana, &card.hybrid_mana);
    card.x_pip_count = static_cast<int>(std::count(mana_cost_str.begin(), mana_cost_str.end(), 'X'));
    card.has_x_cost = (card.x_pip_count > 0);
    card.types = parse_types(value_from_script(front_script, "Types"));
    // AlternateMode:Modal marks a MODAL double-faced card (MDFC, CR 712.x) — both faces are
    // playable from hand (front spell OR back face). Only the front face carries this line; the
    // back face (parsed from the ALTERNATE block) leaves it false. Distinct from a transform DFC.
    card.is_modal_dfc = (value_from_script(front_script, "AlternateMode") == "Modal");
    // AlternateMode:Split marks a SPLIT card (CR 709) — both halves playable from hand, neither a
    // permanent. Only the front face carries this line; the back half (from the ALTERNATE block)
    // leaves it false. Distinct from a modal DFC so split cards skip MDFC permanent logic.
    card.is_split = (value_from_script(front_script, "AlternateMode") == "Split");
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
    face_defs.abilities = parse_abilities(multi_values_from_script(front_script, "A"), svars, card.name);
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
            for (auto &ab : face_defs.abilities) {
                ab.subabilities.erase(
                    std::remove_if(ab.subabilities.begin(), ab.subabilities.end(),
                        [](const AbilityDef &sub) {
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
        face_defs.abilities.push_back(trig);

    // Parse S: lines for alternate costs
    for (auto& line : multi_values_from_script(front_script, "S")) {
        if (line.find("AlternativeCost") == std::string::npos) continue;
        std::string cost_str = param_value(line, "Cost");
        if (cost_str.empty()) continue;
        AltCost ac;
        parse_alt_cost_tokens(cost_str, ac);
        // Parse CheckSVar$ and SVarCompare$ conditions
        size_t pp = 0;
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
    KeywordContext kw_ctx{card, face_defs, svars};
    for (const auto& kw_line : multi_values_from_script(front_script, "K"))
        parse_keyword_line(kw_line, kw_ctx);
}

// Parses one K: keyword line through kKeywordTable: the first entry whose name matches handles it;
// a line no entry matches is a plain comma-separated keyword list.
static void parse_keyword_line(const std::string &kw_line, KeywordContext &ctx) {
    for (const KeywordEntry &entry : kKeywordTable) {
        bool matches = entry.match == KeywordEntry::EXACT    ? kw_line == entry.name
                       : entry.match == KeywordEntry::PREFIX ? kw_line.rfind(entry.name, 0) == 0
                                                             : kw_line.find(entry.name) != std::string::npos;
        if (!matches) continue;
        entry.handler(kw_line, ctx);
        return;
    }
    split_keywords(kw_line, ctx.card.keywords);
}

// The text after a keyword line's first ':' ("Spectacle:1 R" → "1 R"); "" when it has none.
static std::string keyword_arg(const std::string &kw_line) {
    size_t colon = kw_line.find(':');
    return (colon != std::string::npos) ? kw_line.substr(colon + 1) : "";
}

// An alternative casting cost keyword (<name>:<cost>): the spell may be cast for <cost> instead of
// its normal mana cost, recorded on the shared AltCost with the keyword's `flag` set.
static void add_keyword_alt_cost(const std::string &kw_line, const char *name, bool AltCost::*flag,
                                 CardData &card) {
    AltCost ac;
    parse_alt_cost_tokens(keyword_arg(kw_line), ac);
    ac.*flag = true;
    card.alt_cost = ac;
    card.keywords.push_back(name);
}

// An activated ability a keyword grants (Cycling, Ninjutsu, typecycling, Unearth): activated from
// `zone` for `cost`, parsed with the shared Cost$ token grammar.
static AbilityDef keyword_activated_ability(const std::string &category, Zone::ZoneValue zone,
                                            const std::string &cost) {
    AbilityDef ab;
    ab.ability_type = AbilityDef::ACTIVATED;
    ab.category = category;
    ab.activation_zone = zone;
    parse_activation_cost(cost, ab);
    return ab;
}

// A mandatory, untargeted triggered ability a keyword synthesizes (Evoke, Offspring, Storm,
// Annihilator) that fires on `event` for the source itself.
static AbilityDef keyword_self_trigger(const std::string &category, EventId event) {
    AbilityDef ab;
    ab.ability_type = AbilityDef::TRIGGERED;
    ab.category = category;
    ab.trigger_on = event;
    ab.trigger_only_self = true;
    ab.valid_tgts = "N_A";
    ab.mandatory = true;
    return ab;
}

// K:Companion:<grouping>:<restriction>:<desc> — the Companion keyword (CR 702.139). Forge
// encodes the deckbuilding restriction as a token in the 3rd colon field (Yorion:
// "Companion:Special:DeckSizePlus20:..."). Store the restriction token structured so
// setup_companions can evaluate it against the starting deck; the trailing prose is display.
static void kw_companion(const std::string &kw_line, KeywordContext &ctx) {
    ctx.card.is_companion = true;
    std::vector<std::string> parts = split(kw_line, ':');
    if (parts.size() >= 3) ctx.card.companion_restriction = parts[2];
    ctx.card.keywords.push_back("Companion");
}

// K:Enchant:<ValidTgts>[:<prompt>] — an Aura's enchant restriction (CR 303.4). The
// middle field is a target filter (e.g. "Creature.YouCtrl") for the object this Aura can
// be attached to. Stored on the card so the cast path targets a matching object and the
// resolved Aura attaches to it (sets equipped_to). The trailing human prompt is ignored.
static void kw_enchant(const std::string &kw_line, KeywordContext &ctx) {
    std::string rest = kw_line.substr(8);  // strip "Enchant:"
    size_t colon = rest.find(':');
    ctx.card.enchant_filter = (colon != std::string::npos) ? rest.substr(0, colon) : rest;
    ctx.card.keywords.push_back("Enchant");
}

// K:Ward:N — "Whenever this permanent becomes the target of a spell or ability an
// opponent controls, counter that spell or ability unless that player pays {N}."
// (CR 702.21). Stored as the keyword + a numeric cost; the becomes-targeted trigger
// is synthesized when a targeting spell/ability is put on the stack.
static void kw_ward(const std::string &kw_line, KeywordContext &ctx) {
    // K:Ward without a cost arg defaults to a {1} mana ward inside parse_ward_cost.
    parse_ward_cost(keyword_arg(kw_line), ctx.card.ward_cost, ctx.card.ward_is_life);
    ctx.card.keywords.push_back("Ward");
}

// K:Affinity:Artifact — this spell costs {1} less to cast for each artifact you
// control (CR 702.41). A generic cost reduction applied at cast time in
// effective_base_cost(); only the artifact variant is supported.
static void kw_affinity(const std::string &kw_line, KeywordContext &ctx) {
    if (kw_line.find("Artifact") != std::string::npos) ctx.card.affinity_artifact = true;
    ctx.card.keywords.push_back("Affinity");
}

// K:ETBReplacement:Other:ChooseCT — choose creature type on ETB (Cavern of Souls);
// K:ETBReplacement:Other:DBNameCard — choose a card name on ETB (Disruptor Flute). Any other
// ETBReplacement is kept as a plain keyword.
static void kw_etb_replacement(const std::string &kw_line, KeywordContext &ctx) {
    if (kw_line.find("ChooseCT") != std::string::npos)
        ctx.card.has_etb_choose_creature_type = true;
    else if (kw_line.find("NameCard") != std::string::npos)
        ctx.card.has_etb_name_card = true;
    else
        split_keywords(kw_line, ctx.card.keywords);
}

// K:etbCounter:P1P1:X:... — "this card enters with counters"
// Parsed as a static ability; counters applied in apply_permanent_components on ETB.
static void kw_etb_counter(const std::string &kw_line, KeywordContext &ctx) {
    // K:etbCounter:<TYPE>:<count>  where <count> is either a literal number or a
    // SVar key resolving to a Count$ expression (e.g. Chalice's "X" → Count$xPaid).
    std::string sub = kw_line.substr(strlen("etbCounter"));
    StaticAbility sa;
    sa.category = "EtbCounter";
    sa.counter_type = "P1P1";
    if (!sub.empty() && sub[0] == ':') {
        size_t c1 = sub.find(':', 1);
        if (c1 != std::string::npos) {
            sa.counter_type = sub.substr(1, c1 - 1);
            size_t c2 = sub.find(':', c1 + 1);
            std::string count_tok = (c2 != std::string::npos)
                ? sub.substr(c1 + 1, c2 - c1 - 1)
                : sub.substr(c1 + 1);
            // The count is either a literal number (etbCounter:M1M1:6 → 6) or an SVar
            // key resolving to a Count$ expression (delve / X paid at cast).
            if (!count_tok.empty() &&
                std::all_of(count_tok.begin(), count_tok.end(),
                            [](unsigned char ch) { return std::isdigit(ch); })) {
                sa.counter_count = std::stoi(count_tok);
            } else {
                auto svar_it = ctx.svars.find(count_tok);
                if (svar_it != ctx.svars.end()) {
                    if (svar_it->second.find("ExiledWithSource") != std::string::npos) {
                        sa.counter_count_from_delve = true;
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
                            std::string &delve_filter = sa.counter_count_delve_filter;
                            delve_filter = svar_it->second.substr(vp + ve_prefix.size());
                            for (const char *qual : {".ExiledWithSource", "+ExiledWithSource"}) {
                                size_t qp;
                                while ((qp = delve_filter.find(qual)) != std::string::npos)
                                    delve_filter.erase(qp, strlen(qual));
                            }
                        }
                    }
                    // Count$xPaid — the count equals the X value paid at cast time
                    // (Chalice of the Void enters with X charge counters).
                    else if (svar_it->second.find("xPaid") != std::string::npos)
                        sa.counter_count_from_xpaid = true;
                }
            }
        }
    }
    ctx.card.static_abilities.push_back(sa);
}

// K:Equip:<cost> (CR 702.6a): "[Cost]: Attach this permanent to target creature you
// control. Activate only as a sorcery." Stored as an ordinary activated ability on the
// card (resolved by effects::attach), gated on the Equipment being able to equip some
// creature (CR 301.5c).
static void kw_equip(const std::string &kw_line, KeywordContext &ctx) {
    ctx.card.is_equipment = true;
    ctx.face_defs.abilities.push_back(equip_keyword_ability(kw_line, "Attach", "Equip"));
    ctx.card.keywords.push_back("Equip");
}

// K:Reconfigure:<cost> (CR 702.151a): an Equipment keyword on a creature card, two
// activated abilities: "[Cost]: Attach this permanent to another target creature you
// control. Activate only as a sorcery." and "[Cost]: Unattach this permanent. Activate
// only if this permanent is attached to a creature and only as a sorcery." It shares the
// equip-attach machinery; is_reconfigure additionally lets it equip while a creature
// (CR 301.5c) and makes it stop being a creature while attached (CR 702.151b).
static void kw_reconfigure(const std::string &kw_line, KeywordContext &ctx) {
    ctx.card.is_equipment = true;
    ctx.card.is_reconfigure = true;
    ctx.face_defs.abilities.push_back(equip_keyword_ability(kw_line, "Attach", "Reconfigure"));
    ctx.face_defs.abilities.push_back(equip_keyword_ability(kw_line, "Unattach", "Unattach"));
    ctx.card.keywords.push_back("Reconfigure");
}

// K:Impending:<N>:<mana> — Impending (CR 702.175). An alternative casting cost: the spell
// may be cast for <mana> instead of its normal mana cost; if so the permanent enters with N
// time counters and isn't a creature until the last is removed (CR 702.175d-e). Encoded on
// the shared AltCost (mana portion = parse_mana_cost(<mana>), is_impending + impending_count
// flag the impending-specific entry/shed behaviour). The format mirrors Reconfigure's
// colon-split (Equip/Reconfigure), with an extra leading count field: "Impending:5:1 B".
static void kw_impending(const std::string &kw_line, KeywordContext &ctx) {
    std::string rest = kw_line.substr(strlen("Impending"));
    if (!rest.empty() && rest[0] == ':') rest = rest.substr(1);  // "5:1 B"
    size_t colon = rest.find(':');
    if (colon != std::string::npos) {
        AltCost ac;
        ac.has_alt_cost = true;
        ac.is_impending = true;
        ac.impending_count = std::stoi(rest.substr(0, colon));
        ac.mana_cost = parse_mana_cost(rest.substr(colon + 1));
        ctx.card.alt_cost = ac;
    }
    ctx.card.keywords.push_back("Impending");
}

// K:Suspend:<N>:<cost> — Suspend (CR 702.62). NOT an alternative casting cost: it is a
// special action taken from the HAND. Its owner may pay <cost> and exile the card with N
// time counters on it (state_manager offers the action; action_processor performs the
// exile). The count/cost are stored on CardData (has_suspend/suspend_count/suspend_cost);
// the upkeep time-counter removal and free cast are driven from those. Format mirrors
// Impending's colon-split: "Suspend:1:R".
static void kw_suspend(const std::string &kw_line, KeywordContext &ctx) {
    std::string rest = kw_line.substr(strlen("Suspend"));
    if (!rest.empty() && rest[0] == ':') rest = rest.substr(1);  // "1:R"
    size_t colon = rest.find(':');
    if (colon != std::string::npos) {
        ctx.card.has_suspend = true;
        ctx.card.suspend_count = std::stoi(rest.substr(0, colon));
        ctx.card.suspend_cost = parse_mana_cost(rest.substr(colon + 1));
    }
    ctx.card.keywords.push_back("Suspend");
}

// K:Chapter:<final>:<svar1>,<svar2>,...,<svarN> — a Saga's chapter abilities (CR 714). The
// first field is the Saga's final chapter number (= the number of chapter slots, CR 714.2d);
// each subsequent comma-separated entry is an SVar naming the DB$ ability run when the Saga's
// lore counters reach that chapter (CR 714.2b/714.3). Multiple chapters may name the SAME
// SVar (Summon: Bahamut I & II both DBDestroy) — each becomes its own chapter slot, so two
// independent triggers fire at lore 1 and lore 2. Parsed 1-indexed into card.saga_chapters;
// the Saga lifecycle (lore counters, chapter triggers, sacrifice SBA) lives in src/saga.cpp.
static void kw_chapter(const std::string &kw_line, KeywordContext &ctx) {
    std::vector<std::string> parts = split(kw_line, ':');
    if (parts.size() >= 3) {
        for (const std::string &name : split(parts[2], ',', /*skip_empty=*/true)) {
            auto it = ctx.svars.find(name);
            AbilityDef chapter;  // an unknown SVar keeps chapter indexing aligned
            if (it != ctx.svars.end())
                chapter = parse_svar_ability(it->second, AbilityDef::TRIGGERED, ctx.svars, ctx.card.name);
            // A chapter ability is a triggered ability (CR 714.2b) the Saga lifecycle
            // tracks until it leaves the stack (CR 714.4).
            chapter.ability_type = AbilityDef::TRIGGERED;
            chapter.is_saga_chapter = true;
            ctx.face_defs.saga_chapters.push_back(chapter);
        }
    }
    ctx.card.keywords.push_back("Chapter");
}

// K:Dredge:N — replacement effect: while in graveyard, may replace a draw by
// milling N cards and returning this card to hand. Value stored on CardData;
// the replacement is offered in Orderer::draw.
static void kw_dredge(const std::string &kw_line, KeywordContext &ctx) {
    ctx.card.dredge = std::stoi(kw_line.substr(strlen("Dredge:")));
    ctx.card.keywords.push_back("Dredge");
}

// K:Landwalk:Swamp / Forest / Island / Mountain / Plains
static void kw_landwalk(const std::string &kw_line, KeywordContext &ctx) {
    ctx.card.keywords.push_back(kw_line.substr(strlen("Landwalk:")) + "walk");
}

// K:Cycling:<cost> — activated ability from hand: pay cost, discard this card, draw a card
static void kw_cycling(const std::string &kw_line, KeywordContext &ctx) {
    AbilityDef ab = keyword_activated_ability("Draw", Zone::HAND, kw_line.substr(strlen("Cycling:")));
    ab.amount = 1;
    ctx.face_defs.abilities.push_back(ab);
    ctx.card.keywords.push_back("Cycling");
}

// K:Ninjutsu:<cost> (CR 702.49a): "[Cost], Reveal this card from your hand, Return an
// unblocked attacking creature you control to its owner's hand: Put this card onto the
// battlefield from your hand tapped and attacking." A hand-activated ability whose cost is
// the ninjutsu mana plus the return (an ordinary return-to-hand cost over unblocked
// attackers), resolved by effects::ninjutsu. It moves its own source (Defined$ Self), so
// activating it doesn't consume the card from hand. General over any K:Ninjutsu.
static void kw_ninjutsu(const std::string &kw_line, KeywordContext &ctx) {
    AbilityDef ab = keyword_activated_ability(
        "Ninjutsu", Zone::HAND,
        kw_line.substr(strlen("Ninjutsu:")) + " Return<1/Creature.attacking+unblocked>");
    ab.is_ninjutsu = true;
    ab.defined_self = true;
    ctx.face_defs.abilities.push_back(ab);
    ctx.card.keywords.push_back("Ninjutsu");
}

// K:TypeCycling:<Subtype>:<cost> — typecycling (CR 702.29f). Like Cycling, an
// activated ability usable from hand whose cost is the given mana plus discarding
// this card; but instead of drawing, it searches the library for a card of the
// named subtype, reveals it, puts it into hand, then shuffles. General over the
// subtype (Islandcycling/Swampcycling/Plainscycling/...). The discard-this-card
// cost is the auto-consume that fires for any hand-activated ability (the source
// goes to the graveyard at activation); the effect is a Library→Hand search.
static void kw_type_cycling(const std::string &kw_line, KeywordContext &ctx) {
    std::string rest = kw_line.substr(strlen("TypeCycling:"));
    size_t colon = rest.find(':');
    std::string subtype = (colon != std::string::npos) ? rest.substr(0, colon) : rest;
    std::string cost_str = (colon != std::string::npos) ? rest.substr(colon + 1) : "";
    AbilityDef ab = keyword_activated_ability("ChangeZone", Zone::HAND, cost_str);
    ab.origin = Zone::LIBRARY;
    ab.destination = Zone::HAND;
    ab.change_type = subtype;       // subtype filter (search_zones matches card subtypes)
    ab.mandatory = false;           // searches may fail to find (CR 701.19c)
    ctx.face_defs.abilities.push_back(ab);
    ctx.card.keywords.push_back(subtype + "cycling");
}

// K:Flashback:<cost> — cast from graveyard for flashback cost, then exile
static void kw_flashback(const std::string &kw_line, KeywordContext &ctx) {
    ctx.card.has_flashback = true;
    // Shared Cost$ token grammar, then map onto the flashback cost fields the
    // cast path consumes (mana + life). Deep Analysis is "1 U PayLife<3>" — both
    // mana and life — which the token-by-token grammar handles in one pass.
    AbilityDef fb;
    parse_activation_cost(kw_line.substr(strlen("Flashback:")), fb);
    ctx.card.flashback_mana_cost = fb.activation_mana_cost;
    ctx.card.flashback_alt_cost.life_cost = fb.life_cost;
    // Flashback—Sacrifice a creature (Cabal Therapy): Sac<1/Creature> in the
    // flashback cost. Carry the sac filter so the cast path pays it.
    ctx.card.flashback_alt_cost.sac_cost_spec = fb.sac_cost_spec;
    ctx.card.keywords.push_back("Flashback");
}

// K:Unearth:<cost> — Unearth (CR 702.84): an activated ability usable only from the
// graveyard, at sorcery speed, that returns this card to the battlefield. The returned
// permanent gains haste, is exiled at the beginning of the next end step (a delayed
// triggered ability, CR 603.7b), and is exiled instead if it would leave the battlefield.
// Modeled as a synthetic graveyard-activated ChangeZone (Graveyard -> Battlefield, Defined$
// Self); is_unearth flags it so the resolution marks the permanent unearthed (haste +
// delayed exile + leaves-the-battlefield replacement). General over any K:Unearth:<cost>.
static void kw_unearth(const std::string &kw_line, KeywordContext &ctx) {
    AbilityDef ab = keyword_activated_ability("ChangeZone", Zone::GRAVEYARD,
                                              kw_line.substr(strlen("Unearth:")));
    ab.origin = Zone::GRAVEYARD;
    ab.destination = Zone::BATTLEFIELD;
    ab.defined_self = true;        // returns its own source from the graveyard
    ab.sorcery_speed_only = true;  // "Unearth only as a sorcery." (CR 702.84a)
    ab.is_unearth = true;
    ctx.face_defs.abilities.push_back(ab);
    ctx.card.keywords.push_back("Unearth");
}

// K:Escape:<mana> [<additional cost>] — cast this card from your graveyard for the
// escape cost (CR 702.139). The mana portion (e.g. "2 B") precedes any additional cost
// token (e.g. ExileFromGrave<.../withTypesGE4/...> for Nethergoyf). Mana is parsed from
// the leading mana symbols; the additional cost is parsed by the shared alt-cost grammar.
static void kw_escape(const std::string &kw_line, KeywordContext &ctx) {
    std::string cost_str = kw_line.substr(strlen("Escape:"));
    ctx.card.has_escape = true;
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
    if (!mana_part.empty()) ctx.card.escape_mana_cost = parse_mana_cost(mana_part);
    if (!alt_part.empty()) parse_alt_cost_tokens(alt_part, ctx.card.escape_alt_cost);
    ctx.card.keywords.push_back("Escape");
}

// K:Evoke:<cost> — alternate cost; when paid, the creature sacrifices itself as it
// enters. The cost may be a pitch (ExileFromHand), mana (e.g. R), or life. The
// self-sacrifice is a synthetic ETB self-trigger gated on Permanent::evoked, which
// is set only when the spell was cast for its evoke cost.
static void kw_evoke(const std::string &kw_line, KeywordContext &ctx) {
    add_keyword_alt_cost(kw_line, "Evoke", &AltCost::is_evoke, ctx.card);
    AbilityDef sac = keyword_self_trigger("ChangeZone", Events::CARD_CHANGED_ZONE);
    sac.trigger_zone_destination = Zone::BATTLEFIELD;
    sac.is_evoke_sacrifice = true;
    sac.defined_self = true;          // moves its own source (no targeting)
    sac.origin = Zone::BATTLEFIELD;
    sac.destination = Zone::GRAVEYARD;
    ctx.face_defs.abilities.push_back(sac);
}

// K:Offspring:<cost> — an optional additional cost (CR 702.171). You may pay the
// offspring cost in addition to the spell's mana cost as you cast it; if you do,
// when this creature enters, create a 1/1 token that's a copy of it. Modeled as a
// second cast option (paying base + offspring) that sets Permanent::entered_with_offspring,
// gating a synthetic ETB self-trigger that creates the 1/1 token copy.
static void kw_offspring(const std::string &kw_line, KeywordContext &ctx) {
    if (kw_line.find(':') != std::string::npos)
        ctx.card.offspring_cost = parse_mana_cost(keyword_arg(kw_line));
    ctx.card.has_offspring = true;
    ctx.card.keywords.push_back("Offspring");
    AbilityDef tok = keyword_self_trigger("CopyPermanent", Events::CARD_CHANGED_ZONE);
    tok.trigger_zone_destination = Zone::BATTLEFIELD;
    tok.is_offspring_token = true;
    tok.defined_self = true;          // copies its own source (no targeting)
    ctx.face_defs.abilities.push_back(tok);
}

// K:Kicker:<cost1>[:<cost2>...] — one or more OPTIONAL ADDITIONAL costs (CR 702.33).
// Forge encodes "Kicker [A] and/or [B]" as two colon-separated costs (CR 702.33b:
// it means "Kicker [A], kicker [B]" — two independent kickers). Each segment is a mana
// cost paid in addition to the spell's cost as it's cast; paying it makes the spell
// "kicked with its Nth kicker". Stored as a list so the model is multikicker-ready and
// the linked "if it was kicked with its [N] kicker" triggers index into it.
static void kw_kicker(const std::string &kw_line, KeywordContext &ctx) {
    for (const std::string &seg : split(kw_line.substr(strlen("Kicker:")), ':', /*skip_empty=*/true))
        ctx.card.kicker_costs.push_back(parse_mana_cost(seg));
    ctx.card.keywords.push_back("Kicker");
}

// K:Replicate:<cost> — an OPTIONAL ADDITIONAL cost (CR 702.x) that may be paid any
// number of times as the spell is cast. Each payment copies the spell once on cast
// (the copies may choose new targets). Stored as a single per-instance mana cost; the
// count paid is decided at cast time (see action_processor) and recorded per-Spell.
static void kw_replicate(const std::string &kw_line, KeywordContext &ctx) {
    ctx.card.replicate_cost = parse_mana_cost(kw_line.substr(strlen("Replicate:")));
    ctx.card.has_replicate = true;
    ctx.card.keywords.push_back("Replicate");
}

// K:Devoid — the object is colorless (CR 702.114a). Forge cards with Devoid omit a
// Colors: line and rely on the keyword for their colorlessness, so apply it here as a
// general color override (e.g. an Eldrazi printed with colored mana symbols is still
// colorless). explicit_colors = {COLORLESS} marks the card colorless (card_colors).
static void kw_devoid(const std::string &, KeywordContext &ctx) {
    ctx.card.explicit_colors.clear();
    ctx.card.explicit_colors.insert(COLORLESS);
    ctx.card.keywords.push_back("Devoid");
}

// K:Gift — the Gift keyword (CR 702.176). As the spell is cast its controller MAY promise
// the gift to an opponent (an optional choice, not a cost); if promised, the opponent
// receives the gift as the spell resolves, before its other effects. The gift effect is
// held in the card's GiftAbility SVar (a DB$ Token making the gift token). Parse it into
// card.gift_abilities; the cast path (action_processor) offers the promise choice and the
// resolving spell runs these when Spell::gift_promised is set (resolve_ability).
static void kw_gift(const std::string &, KeywordContext &ctx) {
    ctx.card.has_gift = true;
    ctx.card.keywords.push_back("Gift");
    auto git = ctx.svars.find("GiftAbility");
    if (git == ctx.svars.end()) return;
    ctx.face_defs.gift_abilities.push_back(
        parse_svar_ability(git->second, AbilityDef::SPELL, ctx.svars, ctx.card.name));
    ctx.card.gift_description = param_value(git->second, "GiftDescription");
}

// K:MayEffectFromOpeningHand:<SVar>[:!PlayFirst] — "If this card is in your opening
// hand, you may [effect]" (CR 103.6b; Leyline of the Void's begin-the-game-on-the-
// battlefield). The colon field names the SVar holding the effect body (Leyline:
// DB$ ChangeZone | Defined$ Self | Origin$ Hand | Destination$ Battlefield); an optional
// !PlayFirst field (Gemstone Caverns) limits the offer to the player NOT going first.
// The offer itself happens after mulligans in the pregame gate's
// OPENING_ACTIONS stage (game_driver.cpp).
static void kw_opening_hand(const std::string &kw_line, KeywordContext &ctx) {
    ctx.card.keywords.push_back("MayEffectFromOpeningHand");
    std::vector<std::string> parts = split(kw_line, ':');
    if (parts.size() >= 2) {
        auto oit = ctx.svars.find(parts[1]);
        if (oit != ctx.svars.end())
            ctx.face_defs.opening_hand_abilities.push_back(
                parse_svar_ability(oit->second, AbilityDef::SPELL, ctx.svars, ctx.card.name));
    }
    for (size_t pi = 2; pi < parts.size(); pi++)
        if (parts[pi] == "!PlayFirst") ctx.card.opening_hand_not_first = true;
}

// K:Storm — Storm (CR 702.40). A triggered ability that functions on the stack: "When
// you cast this spell, copy it for each spell cast before it this turn. You may choose new
// targets for the copies." Synthesize the self-cast SPELL_CAST trigger here (general over
// any Storm card); the copy count is locked in when the trigger fires
// (state_manager_triggers) and the copies are put on the stack at resolution
// (effects::storm). The trigger itself takes no target — each copy chooses its own.
static void kw_storm(const std::string &, KeywordContext &ctx) {
    ctx.card.keywords.push_back("Storm");
    // ValidCard$ Card.Self — fires for the cast spell itself
    ctx.face_defs.abilities.push_back(keyword_self_trigger("Storm", Events::SPELL_CAST));
}

// K:Annihilator:N — Annihilator N (CR 702.85). "Whenever this creature attacks, defending
// player sacrifices N permanents." Synthesize the self-attack trigger here (general over any
// Annihilator card): a TRIGGERED Sacrifice that fires once per declared attack of this
// creature (CREATURE_ATTACKED). defined_each_opponent routes the edict to the defending
// player (the controller's opponent in the two-player engine), who chooses and sacrifices N
// of their own permanents one at a time (sac_count). It resolves like any triggered ability,
// i.e. before blockers are declared. SacValid$ Permanent = any permanent they control.
static void kw_annihilator(const std::string &kw_line, KeywordContext &ctx) {
    size_t colon = kw_line.find(':');
    int n = (colon != std::string::npos) ? std::stoi(kw_line.substr(colon + 1)) : 1;
    ctx.card.keywords.push_back(kw_line);
    // ValidCard$ Card.Self — only this creature's own attack
    AbilityDef ab = keyword_self_trigger("Sacrifice", Events::CREATURE_ATTACKED);
    ab.defined_each_opponent = true;  // the defending player sacrifices (CR 702.85b)
    ab.sac_valid = "Permanent";       // any permanent the defending player controls
    ab.sac_count = static_cast<size_t>(n);
    ctx.face_defs.abilities.push_back(ab);
}

// K:Protection:<quality>:<desc> — structured Protection keyword (CR 702.16). The middle
// field is the quality. Emrakul uses Protection:Spell.nonColorless ("protection from
// colored spells"): a one-or-more-colors SPELL can't target it (702.16b/e). Modeled as a
// creature keyword consulted in has_protection_from. A structured single-color quality is
// normalized to the literal "Protection from <color>" form the color-protection path
// already understands (the common color-protection cards spell that form out directly).
static void kw_protection(const std::string &kw_line, KeywordContext &ctx) {
    std::vector<std::string> parts = split(kw_line, ':');
    std::string spec = parts.size() > 1 ? parts[1] : "";
    if (spec.rfind("Spell", 0) == 0 && spec.find("nonColorless") != std::string::npos)
        ctx.card.keywords.push_back("Protection from colored spells");
    else
        ctx.card.keywords.push_back("Protection from " + ascii_lower(spec));
}

static void parse_card_face(const std::string& front_script, CardData& card) {
    FaceAbilityDefs face_defs;
    parse_card_face_body(front_script, card, face_defs);
    card.abilities = intern_ability_defs(std::move(face_defs.abilities));
    card.gift_abilities = intern_ability_defs(std::move(face_defs.gift_abilities));
    card.saga_chapters = intern_ability_defs(std::move(face_defs.saga_chapters));
    card.opening_hand_abilities = intern_ability_defs(std::move(face_defs.opening_hand_abilities));
}

Token parse_token_script(const std::string &script_name) {
    // Parsed once per script per process, like a card script (parse_card_script).
    static std::unordered_map<std::string, Token> parsed;
    auto it = parsed.find(script_name);
    if (it != parsed.end()) return it->second;
    Token tok;
    if (!parse_token_file(script_name, tok)) return tok;
    parsed.emplace(script_name, tok);
    return tok;
}

static bool parse_token_file(const std::string &script_name, Token &tok) {
    tok.script_name = script_name;
    std::string path = RESOURCE_DIR + "/tokenscripts/" + script_name + ".txt";
    std::ifstream stream(path);
    if (!stream.is_open()) {
        non_fatal_error("Could not open token script: " + path);
        return false;
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
    std::vector<AbilityDef> tok_defs = parse_triggered_abilities(script_data, svars, tok.name);
    for (auto &ab : parse_abilities(multi_values_from_script(script_data, "A"), svars, tok.name))
        tok_defs.push_back(ab);
    tok.abilities = intern_ability_defs(std::move(tok_defs));
    // S: lines — continuous static abilities (e.g. the Construct token's "+1/+1 for each
    // artifact you control" self-buff). Applied via the Permanent once bootstrapped.
    tok.static_abilities = parse_static_abilities(script_data, svars);

    return true;
}

// private util functions
static std::string value_from_script(const std::string &script, const std::string &key) {
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

static std::vector<std::string> multi_values_from_script(const std::string &script, const std::string &key) {
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
    // Forge separates a multicolor indicator with commas ("Colors:green,blue"); accept spaces too.
    size_t cp = 0;
    while (cp <= colors_field.size()) {
        size_t sp = colors_field.find_first_of(" ,", cp);
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
static void apply_param_to_ability(AbilityDef& ability, const std::string& key, const std::string& value,
                                   const std::string& card_name) {
    if (key == "NumCards" || key == "ChangeNum" || key == "Amount" ||
        key == "TokenAmount" || key == "ScryNum" || key == "Num" || key == "NumTurns") {
        if (value == "DamageAmount" || value == "TriggerCount$DamageAmount") {
            ability.amount_from_damage = true;
        } else if (key == "ChangeNum" && value == "Any") {
            // "Any" = the player may take any number of the looked-at cards (Dig/Fateseal).
            ability.change_num_any = true;
        } else if (key == "ChangeNum" && value == "All") {
            // "All" = take every matching looked-at card, mandatorily and automatically
            // (no player choice / DIG_CHOICE prompt). Goblin Guide reveals the top card and
            // the defender takes it if it's a land.
            ability.change_num_all = true;
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
    } else if (key == "GainThisAbility") {
        // GainThisAbility$ True on an AB$ Clone (Thespian's Stage): the in-place copy retains this
        // very ability ("...except it has this ability."). CR 706.2. Consumed by the clone handler.
        ability.gain_this_ability = (value == "True");
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
        } else if (value.find_first_not_of("0123456789") == std::string::npos) {
            ability.unless_generic_cost = static_cast<size_t>(std::stoi(value));
        } else {
            // A colored/mixed unless-cost (Chain Lightning: UnlessCost$ R R = {R}{R}). Parse the
            // exact pips; the pip count drives the >0 gate and logging, the pips drive payment.
            ability.unless_cost_pips = parse_mana_cost(value);
            ability.unless_generic_cost = ability.unless_cost_pips.size();
        }
    } else if (key == "UnlessPayer") {
        // UnlessPayer$ You — the controller is the payer of the unless-cost (the only payer we
        // model for the energy unless-cost). Cosmetic given the energy is paid by the controller.
        // UnlessPayer$ TriggeredSourceSAController — the payer is the controller of the spell that
        // targeted the source (Reality Smasher's opponent), bound at trigger-fire time.
        if (value == "TriggeredSourceSAController")
            ability.unless_payer_is_triggered_source_sa_ctrl = true;
        // UnlessPayer$ TargetedOrController (Chain Lightning) — the targeted player, or the targeted
        // permanent's controller; derived from the target at resolution.
        else if (value == "TargetedOrController")
            ability.unless_payer_is_targeted_or_controller = true;
    } else if (key == "UnlessSwitched") {
        // UnlessSwitched$ True inverts the normal "do unless paid" into "do only if paid". For a
        // DestroyAll (Wrath of the Skies: destroy only if the energy was paid) it rides on the
        // DestroyAll params; for any other effect (Chain Lightning's CopySpellAbility: copy only if
        // {R}{R} was paid) it rides on the generic unless_switched flag.
        if (ability.category == "DestroyAll")
            effect_params<DestroyAllParams>(ability).energy_unless_switched = (value == "True");
        else
            ability.unless_switched = (value == "True");
    } else if (key == "MayChooseTarget") {
        // MayChooseTarget$ True (Chain Lightning): the copy's controller may choose new targets.
        // The shared copy machine (effect_copy_spell.cpp) re-runs target selection for every copy
        // (CR 707.10c), so the flag needs no field.
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
        // cur_game.resolution.memory.remembered before the fire ability resolves, so this resolves
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
        // Defined$ TriggeredDefendingPlayer — the effect acts on the defending player of the
        // attack that fired this trigger (Goblin Guide). Bound at trigger-fire time from the
        // CREATURE_ATTACKED event (the opponent of the attacker's controller in a 2-player game).
        else if (value == "TriggeredDefendingPlayer") ability.defined_triggered_defending_player = true;
        // Defined$ TriggeredPlayer — the effect acts on the player whose event fired this trigger
        // (Roiling Vortex: each player's upkeep damages THAT player). The actual player is bound at
        // trigger-fire time from the event's PLAYER param.
        else if (value == "TriggeredPlayer") ability.defined_triggered_player = true;
        // Defined$ TriggeredCardController — the effect's player is the controller of the card
        // whose zone change fired the (Mode$ ChangesZone delayed) trigger (Searing Blood: the
        // dead creature's controller). Bound from the watched card's last-known controller at
        // trigger-fire time (CR 608.2g); the card is already in the graveyard by then.
        else if (value == "TriggeredCardController") ability.defined_triggered_card_controller = true;
        // Defined$ TriggeredNewCard(LKICopy) / TriggeredCard(LKICopy) — the object that changed
        // zones and fired this trigger, referenced in its NEW zone. For a Card.Self ChangesZone
        // trigger (CR 603.6e) the triggering object IS the ability's own source (e.g. Triumph of
        // Saint Katherine's dies-trigger, which exiles itself from the graveyard as the head of a
        // "Descend"-style recursion), so bind it to the source. The LKICopy variant is a
        // last-known snapshot of that same card; moving the real card in its new zone is
        // equivalent here. General over self-referential graveyard-recursion cards.
        else if (value == "TriggeredNewCardLKICopy" || value == "TriggeredNewCard" ||
                 value == "TriggeredCardLKICopy" || value == "TriggeredCard")
            ability.defined_self = true;
        else if (value == "Self") ability.defined_self = true;
        // Defined$ ExiledWith — the effect acts on the card this permanent's Saga chapter I
        // exiled face down, tracked in Permanent::exiled_with on the source (The Creation of
        // Avacyn chapters II & III). Resolved to that entity at resolution by exiled_with_card().
        else if (value == "ExiledWith") ability.defined_exiled_with = true;
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
    } else if (key == "GainControl") {
        // GainControl$ True on a ChangeZone Destination$ Battlefield: the moved card enters under
        // the ability controller's control, not its owner's (Animate Dead's reanimation). CR 110.2a.
        ability.gain_control = (value == "True");
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
        // RememberObjects$ ChosenCard — a DB$ Effect applies to the card the preceding ChooseCard
        // chose (Dauthi Voidwalker's "You may play it this turn").
        if (value.find("ChosenCard") != std::string::npos) ability.effect_remember_chosen_card = true;
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
        // "exactly X targets"): stash the token; resolve_ability_svars resolves it and,
        // when it is Count$xPaid, sets target_min_from_xpaid so select_target requires X targets.
        if (!value.empty() && std::isdigit(static_cast<unsigned char>(value[0])))
            ability.target_min = std::stoi(value);
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
        // resolved to its Count$/dynamic expression by resolve_ability_svars. The
        // generic mana portion of the activation cost is reduced by the resolved amount at
        // activation time (CR 601.2f).
        ability.reduce_cost_expr = value;
    } else if (key == "RestrictValid") {
        if (filter_names_token(value, "Creature") &&
            value.find("ChosenType") != std::string::npos) {
            ability.restrict_to_chosen_type_creature = true;
        } else if (value.find("Eldrazi") != std::string::npos &&
                   value.find("Colorless") != std::string::npos) {
            // RestrictValid$ Spell.Eldrazi+Colorless,... — mana usable only to cast a
            // colorless Eldrazi spell (Eldrazi Temple). The trailing Activated.Eldrazi…
            // clause (activate abilities of colorless Eldrazi) is folded into the same
            // restriction. CR 106.7.
            ability.restrict_to_colorless_eldrazi = true;
        } else if (filter_names_token(value, "Creature")) {
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
        // the ENTERS_TAPPED replacement, so the flag needs no field.
    } else if (key == "Planeswalker") {
        ability.is_loyalty_ability = (value == "True");
    } else if (key == "Cost") {
        parse_activation_cost(value, ability);
    } else if (key == "ConditionCheckSVar") {
        // ConditionCheckSVar$ <SVar> (Veil of Summer's SP$ Draw gate). Store the raw SVar name;
        // resolve_condition_svars resolves it to its Count$ expression and defaults the
        // comparator (ConditionSVarCompare) to GE1 when none is given.
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
        // cur_game.resolution.memory.remembered (RememberObjects$ RememberedLKI / Defined$
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
        // "ExiledWith" → condition_present is a property check on the card the source Saga exiled
        // face down (The Creation of Avacyn: lose life / put onto battlefield only if it's a
        // Creature). Evaluated at resolution against that card's printed characteristics.
        ability.condition_on_exiled_with = (value == "ExiledWith");
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
    } else if (key == "RepeatTypesFrom") {
        // DB$ RepeatEach | RepeatTypesFrom$ ValidLibrary Card.IsImprinted (Atraxa, Grand Unifier):
        // loop the RepeatSubAbility once per distinct card type among the imprinted cards.
        ability.repeat_types_from = value;
    } else if (key == "RememberChosen") {
        // ChooseCard | RememberChosen$ True (Atraxa): append the chosen card to the remembered set.
        ability.remember_chosen = (value == "True");
    } else if (key == "ClearImprinted") {
        // DB$ Cleanup | ClearImprinted$ True (Atraxa): clear cur_game.resolution.memory.imprinted. (For the
        // exile-and-return cards whose "imprint" is actually the remembered set, memory.imprinted
        // is empty, so this is a harmless no-op there.)
        ability.clear_imprinted = (value == "True");
    } else if (key == "Choices" && ability.category == "ChooseCard") {
        // ChooseCard | Choices$ <filter>: the cards the choice is made from (Dauthi Voidwalker:
        // Card.OppOwn+counters_GE1_VOID). Choices$ Card.ChosenType+YouOwn+IsImprinted (Atraxa)
        // chooses one imprinted card of the current cur_game.resolution.memory.chosen_type. (The Ajani -4
        // umbrella Choices$ is superseded by its per-type ChooseEach$.)
        ability.choose_card_filter = value;
        ability.choose_imprinted = filter_names_token(value, "IsImprinted");
    } else if (key == "ChoiceZone" && ability.category == "ChooseCard") {
        // ChooseCard | ChoiceZone$ <zone>: the zone the Choices$ cards are chosen from. (Atraxa's
        // ChoiceZone$ Library is implied by its imprinted set, which stays in the library.)
        if (value == "Exile") ability.choose_card_zone = Zone::EXILE;
        else if (value == "Graveyard") ability.choose_card_zone = Zone::GRAVEYARD;
        else if (value == "Hand") ability.choose_card_zone = Zone::HAND;
        else if (value == "Library") ability.choose_card_zone = Zone::LIBRARY;
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
        if (!value.empty() && std::isdigit(static_cast<unsigned char>(value[0])))
            *base = std::stoi(value);
    } else if (key == "Duration" && value == "UntilHostLeavesPlay") {
        // Duration$ UntilHostLeavesPlay on a ChangeZone | Destination$ Exile (CR 603.6e): the
        // exiled card(s) return when the ability's host leaves the battlefield. See
        // effects::register_exile_until_host_leaves.
        ability.duration_until_host_leaves = true;
    } else if (key == "Duration" && value == "UntilYourNextTurn") {
        // Generic "until your next turn" duration on a non-Animate effect (The One Ring's ETB Pump
        // granting the controller protection from everything). Reverted at the controller's untap.
        ability.duration_until_your_next_turn = true;
    } else if (key == "Duration" && value == "UntilTheEndOfYourNextTurn") {
        // "Until the end of your next turn" (Light Up the Stage's play-permission Effect) — one
        // turn cycle longer than UntilYourNextTurn. Expired at the controller's next-turn cleanup.
        ability.duration_until_end_of_your_next_turn = true;
    } else if (effects::apply_parse_hook(ability, key, value)) {
        // Consumed by an effect-specific parse hook co-located with its handler.
    } else {
        static const std::set<std::string> ignored_keys = {
            "SpellDescription", "AILogic", "AINoRecursiveCheck", "TgtPrompt", "StackDescription",
            "ConditionDescription",
            // AB$ Effect emblem (Kaito's [+1]): Name$ is the emblem's display name and Image$ its
            // art — both cosmetic. Duration$ is read load-bearingly from the raw line in
            // resolve_effect_static_svars (Duration$ Permanent ⇒ the Effect makes a permanent emblem); the
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
            // ForgetOnMoved$ Exile (the play-from-exile Effects of Ugin -11, Light Up the Stage,
            // Dauthi Voidwalker): tells Forge to drop a remembered object from the effect once it
            // leaves the named zone. The exile-play permission already covers a card only while it
            // stays in exile (card_play_permission), so the tag needs no separate handling.
            "ForgetOnMoved",
            // ChooseCard ChooseEach (Ajani -4): the per-type breakdown is the load-bearing
            // ChooseEach$; Choices$ (the umbrella pool), ControlledByPlayer$ Chooser, and
            // Reveal$ are captured by / cosmetic to the choose_each handler.
            "Choices", "ControlledByPlayer", "Reveal",
            // ChoiceTitle$ <text> (Dauthi Voidwalker's ChooseCard): the prose prompt shown for the
            // choice. Purely cosmetic — the load-bearing Choices$ / ChoiceZone$ are parsed above.
            "ChoiceTitle",
            // Ultimate$ True is informational: ultimate legality is already covered by the
            // minus-loyalty cost check, so the flag is unused.
            "Ultimate",
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
            // already tracked in cur_game.resolution.memory.remembered (RememberObjects$ RememberedLKI),
            // which the paired ConditionDefined$ Imprinted gate reads, so the imprint is redundant.
            // (ClearImprinted$ True is parsed above into clear_imprinted — a no-op for Phelia,
            // whose memory.imprinted set is empty, but load-bearing for Atraxa.)
            "Imprint",
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
            // "Audit: ability-param keys the parser silently ignores" section of todo.md):
            //   Reorder$ True (Brainstorm) — let the player order the cards put back on top.
            //   TriggerAmount$ / RememberOriginalTokens$ (Ajani, Nacatl Avenger) — the token
            //     count carried to the transform trigger, and tracking the original tokens.
            //   LockTokenScript$ True (Into the Flood Maw) — pin the gifted Fish token's script.
            //   ExileOnMoved$ Battlefield (Manifold Key) — exile the permanent when it moves.
            "Reorder", "TriggerAmount", "RememberOriginalTokens", "LockTokenScript",
            "ExileOnMoved",
            // Controller$ <token> (Chain Lightning's DB$ CopySpellAbility: Controller$
            // TargetedOrController) — names who controls the resolving sub-ability. The
            // copy_spell_ability handler derives the copy's controller from UnlessPayer$
            // TargetedOrController directly, so this is informational here.
            "Controller",
            // DB$ Animate | Keywords$ / RemoveKeywords$ (Animate Dead): the keyword swap that
            // re-anchors the aura's Enchant restriction from "creature card in a graveyard" to
            // "the creature put onto the battlefield with this" is cosmetic here — the reanimation
            // + Attach chain already anchors the aura to the returned creature (equipped_to), and
            // the "unattached aura" state-based action reads that link, not the Enchant filter
            // string. So the live keyword text is never re-evaluated; ignoring the swap is a no-op.
            "Keywords", "RemoveKeywords"
        };
        if (ignored_keys.find(key) == ignored_keys.end()) {
            std::string msg = "Unrecognized ability param: " + key + "$ " + value;
            if (!card_name.empty()) msg += " (card: " + card_name + ")";
            warning(msg);
        }
    }
}

// Normalizes script category names to the internal names used throughout the engine. A category
// with no EffectKind (effect_kinds.def) would resolve as a no-op, so it is reported here.
static std::string normalize_category(std::string category, const std::string& card_name) {
    if (category == "Mana") category = "AddMana";
    EffectKind kind;
    if (!effect_kind_from_category(category, kind)) {
        std::string msg = "Unknown ability category: " + category + " (resolves as a no-op)";
        if (!card_name.empty()) msg += " (card: " + card_name + ")";
        warning(msg);
    }
    return category;
}

// Resolves a TargetMin$/TargetMax$ that was given as an SVar key (read back from the ability's
// script `line`) to its runtime meaning. When the SVar resolves to Count$xPaid the bound equals the X paid at
// cast/activation (CR 601.2b chooses X before targets, so x_paid is known when targets are
// selected). Setting BOTH target_min_from_xpaid and target_max_from_xpaid yields EXACTLY-X
// targeting (Candelabra of Tawnos, Hide on the Ceiling); a lone TargetMax$ X gives "up to X"
// (Kozilek's Command). Other count-SVar caps keep the "effectively unlimited" fallback already
// stored by apply_param_to_ability.
static void resolve_xpaid_target_counts(AbilityDef& ability,
                                        const std::map<std::string, std::string>& svars,
                                        const std::string& line) {
    const std::string min_key = svar_key_param(line, "TargetMin");
    const std::string max_key = svar_key_param(line, "TargetMax");
    if (!min_key.empty()) {
        auto it = svars.find(min_key);
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
    if (!max_key.empty()) {
        auto it = svars.find(max_key);
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
// captured separately (att_sign/def_sign) by parse_pump_amount.
static void resolve_pump_exprs(AbilityDef& ability,
                               const std::map<std::string, std::string>& svars) {
    if (auto *pp = std::get_if<PumpParams>(&ability.params)) {
        for (std::string *expr : {&pp->att_expr, &pp->def_expr}) {
            if (expr->empty()) continue;
            auto it = svars.find(*expr);
            if (it != svars.end()) *expr = it->second;
        }
    }
}

// A non-numeric amount param (NumCards$, LifeAmount$, ...) that contains '$' is a DIRECT dynamic
// expression rather than an SVar name (Kaito, Bane of Nightmares: NumCards$
// PlayerCountRegisteredOpponents$HasPropertyLostLifeThisTurn; The Creation of Avacyn:
// ExiledWith$CardManaCost). Keep it verbatim for evaluate_amount at resolution (CR 608.2c).
static bool take_direct_amount_expr(AbilityDef &ability) {
    if (ability.amount_svar.find('$') == std::string::npos) return false;
    ability.dynamic_amount_expr = ability.amount_svar;
    ability.amount_svar = "";
    return true;
}

// An amount SVar name with no SVar body cannot be evaluated; flag it instead of silently falling
// back to the effect's default amount.
static void warn_unresolved_amount_svar(const AbilityDef &ability, const std::string &card_name) {
    std::string msg = "Unresolved amount SVar: " + ability.amount_svar;
    if (!card_name.empty()) msg += " (card: " + card_name + ")";
    warning(msg);
}

// Resolve a DestroyAll ValidCards$ dynamic mana-value bound (Blast Zone:
// "Permanent.nonLand+cmcEQY", Y = Count$CardCounters.CHARGE) and an energy UnlessCost SVar into
// their runtime Count$ expressions + comparator on DestroyAllParams, so effect_destroy_all can
// gate candidates by mana value at resolution.
static void resolve_destroyall_svars(AbilityDef &ability,
                                     const std::map<std::string, std::string> &svars) {
    if (ability.category != "DestroyAll") return;
    auto &dp = effect_params<DestroyAllParams>(ability);
    std::string op, bound;
    // A pure-numeric or "X" bound stays on the legacy path (handled at resolution); only a named
    // SVar resolving to a Count$ expression routes here.
    if (dp.cmc_expr.empty() && find_cmc_bound(ability.valid_cards_filter, nullptr, op, bound) &&
        bound != "X") {
        auto it = svars.find(bound);
        if (it != svars.end()) {
            dp.cmc_expr = it->second;
            dp.cmc_op = op;
        }
    }
    if (!dp.energy_unless_expr.empty()) {
        auto it = svars.find(dp.energy_unless_expr);
        if (it != svars.end()) dp.energy_unless_expr = it->second;
    }
}

// The first "cmc<OP><bound>" mana-value qualifier in a card filter (e.g. "Creature.cmcLEX+YouCtrl"
// → "LE", "X"): its two-letter comparator and its bound, the SVar name or number that follows (a
// run of letters, digits and '_'). OP is tried in the order EQ, LE, GE, LT, GT, NE, or is only
// `only_op` when given. False when the filter has no such qualifier.
static bool find_cmc_bound(const std::string &filter, const char *only_op, std::string &op,
                           std::string &bound) {
    for (const char *cmp : {"EQ", "LE", "GE", "LT", "GT", "NE"}) {
        if (only_op && std::strcmp(cmp, only_op) != 0) continue;
        size_t pos = filter.find(std::string("cmc") + cmp);
        if (pos == std::string::npos) continue;
        size_t start = pos + 5, end = start;
        while (end < filter.size() &&
               (std::isalnum(static_cast<unsigned char>(filter[end])) || filter[end] == '_'))
            end++;
        op = cmp;
        bound = filter.substr(start, end - start);
        return true;
    }
    return false;
}

// Parses a spell, activated or DB$ ability body: the category at `category_pos` (just past its
// SP$/AB$/DB$ prefix), then every `| Key$ Value` param — the sub-ability chain, modes, granted
// bodies and plain params alike — then the SVar references those params name. Its sub-abilities
// and modes share its `type`. The one ability parser behind a card's A: lines, SVar bodies,
// trigger Execute$ bodies and granted ability bodies.
static AbilityDef parse_ability_text(const std::string &text, size_t category_pos,
                                     AbilityDef::AbilityType type,
                                     const std::map<std::string, std::string> &svars,
                                     const std::string &card_name) {
    AbilityDef ability;
    ability.ability_type = type;
    size_t cat_end = text.find_first_of(" |", category_pos);
    if (cat_end == std::string::npos) cat_end = text.length();
    if (cat_end > category_pos)
        ability.category = normalize_category(text.substr(category_pos, cat_end - category_pos), card_name);

    // Earthbend (CR keyword action) inherently targets a land the controller controls; the
    // Forge scripts carry no ValidTgts$, so default it here (overridden if the script ever
    // states one explicitly).
    if (ability.category == "Earthbend") ability.valid_tgts = "Land.YouCtrl";

    size_t param_pos = text.find('|', category_pos);
    std::string key, value;
    while (next_param(text, param_pos, key, value))
        if (!apply_composite_param(ability, key, value, svars, card_name))
            apply_param_to_ability(ability, key, value, card_name);
    resolve_ability_svars(ability, text, svars, card_name);
    return ability;
}

// Parses a SVar's content (normally a DB$ ability; an AB$ or SP$ body parses the same way, e.g.
// Guide of Souls' TrigImmediateTrig: "AB$ ImmediateTrigger | Cost$ PayEnergy<3>") into an
// ability of `ability_type`. Content with no ability prefix yields an empty ability.
static AbilityDef parse_svar_ability(const std::string &content, AbilityDef::AbilityType ability_type,
                                     const std::map<std::string, std::string> &svars,
                                     const std::string &card_name) {
    size_t prefix = content.find("DB$");
    if (prefix == std::string::npos) prefix = content.find("AB$");
    if (prefix == std::string::npos) prefix = content.find("SP$");
    if (prefix == std::string::npos) {
        AbilityDef empty;
        empty.ability_type = ability_type;
        return empty;
    }
    // All three prefixes are 3 chars + a space.
    return parse_ability_text(content, prefix + 4, ability_type, svars, card_name);
}

// The params whose value names SVars to parse as further abilities or bodies (the sub-ability
// chain, modes, granted abilities, triggers, replacement and static effects). Returns false for
// any other key, which apply_param_to_ability handles.
static bool apply_composite_param(AbilityDef &ability, const std::string &key, const std::string &value,
                                  const std::map<std::string, std::string> &svars,
                                  const std::string &card_name) {
    if (key == "SubAbility" || key == "RepeatSubAbility" || key == "VoteSubAbility") {
        // RepeatSubAbility$ (RepeatEach) / VoteSubAbility$ (Vote, Council's Judgment) resolve like
        // SubAbility$: the value names an SVar holding a DB$ ability, pushed as a sub-ability. For a
        // per-type RepeatEach (Atraxa) the RepeatSubAbility entries are the per-iteration body; count
        // them so repeat_each_types knows how many leading subs are the body vs. trailing links.
        auto it = svars.find(value);
        if (it == svars.end()) return true;
        ability.subabilities.push_back(parse_svar_ability(it->second, ability.ability_type, svars, card_name));
        if (key == "RepeatSubAbility") ability.repeat_sub_count++;
        return true;
    }
    if (key == "Choices" && ability.category != "ChooseCard") {
        // Charm modal: resolve comma-separated SVar names into modes. Excludes ChooseCard, whose
        // Choices$ is a card FILTER (Atraxa: Card.ChosenType+YouOwn+IsImprinted) consumed by
        // apply_param_to_ability, not a list of modal SVars.
        for (const std::string &svar_name : split(value, ',')) {
            auto it = svars.find(svar_name);
            if (it == svars.end()) continue;
            ability.charm_choices.push_back(parse_svar_ability(it->second, ability.ability_type, svars, card_name));
            // The mode's SpellDescription$, shown when the modes are offered.
            ability.charm_choice_descriptions.push_back(param_value(it->second, "SpellDescription"));
        }
        return true;
    }
    if (key == "CharmNum") {
        ability.charm_num = std::stoi(value);
        return true;
    }
    if (key == "Abilities" && ability.category == "Animate") {
        // DB$ Animate | Abilities$ <svar>[,<svar>...] — the activated ability(ies) the Animate
        // grants to the target permanent (Urza's Saga: ABMana "{T}: Add {C}." / ABToken
        // "{2},{T}: Create a Construct"). Each named SVar is a self-contained AB$ ability;
        // parse it as an ACTIVATED ability and store it for the Animate handler to attach.
        for (const std::string &svar_name : split(value, ',', /*skip_empty=*/true)) {
            auto it = svars.find(svar_name);
            if (it != svars.end())
                ability.animate_granted_abilities.push_back(
                    parse_svar_ability(it->second, AbilityDef::ACTIVATED, svars, card_name));
        }
        return true;
    }
    if (key == "Execute") {
        // Execute$ references an SVar containing the ability to fire (delayed triggers)
        effect_params<DelayedTriggerParams>(ability).execute_svar = value;
        auto it = svars.find(value);
        if (it != svars.end()) {
            AbilityDef exec = parse_svar_ability(it->second, ability.ability_type, svars, card_name);
            exec.from_delayed_execute = true;  // delayed_trigger() fires this one
            ability.subabilities.push_back(exec);
        }
        return true;
    }
    if (key == "Triggers") {
        // Effect | Triggers$ <SVar>[,<SVar>...] — a transient floating triggered ability (Forth
        // Eorlingas!'s monarch trigger; Tamiyo, Seasoned Scholar's +2, whose Duration$ the
        // GrantCast handler applies to it). Each named SVar holds a trigger line (Mode$ ... |
        // Execute$ ...); parse it like a card's T: line so it carries the same trigger metadata and
        // its Execute$ effect, and store it on the Effect to be registered (controller-bound) into
        // cur_game.resolved_effects.floating_triggers at resolution.
        for (const std::string &svar_name : split(value, ',', /*skip_empty=*/true)) {
            auto it = svars.find(svar_name);
            if (it != svars.end()) {
                AbilityDef trig = parse_one_trigger(it->second, svars, card_name);
                if (trig.trigger_on != 0) ability.effect_floating_triggers.push_back(trig);
            }
        }
        return true;
    }
    if (key == "ReplacementEffects") {
        // Effect | ReplacementEffects$ <SVar>[,<SVar>...] — one or more named replacement
        // effects the transient Effect carries. Two forms are recognized:
        //  * Veil of Summer's AntiMagic = "Event$ Counter | ValidSA$ Spell.YouCtrl | Layer$
        //    CantHappen" — a turn-long "spells you control can't be countered" grant; the
        //    GrantCast handler records the controller in cur_game.resolved_effects.cant_counter_spells_of.
        //  * Maze of Ith's RPrevent1/RPrevent2 = "Event$ DamageDone | Prevent$ True |
        //    IsCombat$ True | ValidSource$/ValidTarget$ Card.IsRemembered" — prevent all
        //    combat damage dealt by/to the remembered creature this turn (CR 615); the
        //    GrantCast handler registers a turn-scoped combat-damage prevention shield.
        for (const std::string &svar_name : split(value, ',', /*skip_empty=*/true)) {
            auto it = svars.find(svar_name);
            if (it == svars.end()) continue;
            const std::string &body = it->second;
            if (body.find("Event$ Counter") != std::string::npos &&
                body.find("CantHappen") != std::string::npos &&
                body.find("YouCtrl") != std::string::npos) {
                ability.effect_spells_uncounterable_this_turn = true;
            } else if (body.find("Event$ DamageDone") != std::string::npos &&
                       body.find("Prevent$ True") != std::string::npos &&
                       body.find("IsCombat$ True") != std::string::npos &&
                       body.find("IsRemembered") != std::string::npos) {
                if (body.find("ValidSource$") != std::string::npos)
                    ability.effect_prevent_combat_damage_by_remembered = true;
                if (body.find("ValidTarget$") != std::string::npos)
                    ability.effect_prevent_combat_damage_to_remembered = true;
            }
        }
        return true;
    }
    if (key == "StaticAbilities") {
        // Effect | StaticAbilities$ <name> — names the continuous static the transient effect
        // grants. The value may be a literal keyword (Unblockable) or a named SVar holding a
        // continuous static-ability line. Keep the raw value (the Unblockable path reads it), and
        // additionally resolve a named SVar to detect a MayPlay$ True grant over exiled cards
        // (Light Up the Stage, Ugin -11, Dauthi Voidwalker), which the GrantCast handler turns into
        // play-from-exile permissions: free with MayPlayWithoutManaCost$ True, lands included
        // unless Affected$ says nonLand.
        ability.effect_static_ability = value;
        auto it = svars.find(value);
        if (it != svars.end() && param_value(it->second, "MayPlay") == "True" &&
            param_value(it->second, "AffectedZone") == "Exile") {
            ability.effect_may_play_from_exile = true;
            ability.effect_may_play_free =
                param_value(it->second, "MayPlayWithoutManaCost") == "True";
            ability.effect_may_play_lands =
                !filter_names_token(param_value(it->second, "Affected"), "nonLand");
        }
        return true;
    }
    return false;
}

// Resolves the SVar references an ability's params named (amounts, counts, bounds, conditions and
// the statics an Effect grants) into the runtime expressions its effect evaluates. `text` is the
// ability's own script text.
static void resolve_ability_svars(AbilityDef &ability, const std::string &text,
                                  const std::map<std::string, std::string> &svars,
                                  const std::string &card_name) {
    resolve_effect_static_svars(ability, text, svars);
    std::string op, bound;  // a filter's cmc<OP><bound> qualifier (find_cmc_bound)
    // Fatal Push pattern: ConditionPresent "Creature.cmcLE<SVar>" references an SVar
    // (X = Count$Revolt.4.2) for the cmc threshold but sets no Amount/NumDmg, so
    // amount_svar would be empty and the revolt-scaled threshold never resolves.
    // Wire the referenced SVar into amount_svar so resolve_amount_svar resolves it into
    // dynamic_amount_expr (evaluated at resolution by effects::destroy).
    if (ability.amount_svar.empty() && find_cmc_bound(ability.condition_present, "LE", op, bound) &&
        svars.find(bound) != svars.end())
        ability.amount_svar = bound;
    resolve_amount_svar(ability, svars, card_name);
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
    // Aether Vial pattern: a ChangeType search filter whose mana-value bound is dynamic,
    // e.g. "Creature.cmcEQX+YouCtrl" with SVar:X:Count$CardCounters.CHARGE. Resolve the
    // "cmcEQ<svar>"/"cmcLE<svar>" SVar reference to its runtime Count$ expression and stash
    // it (with the comparator) so the ChangeZone search can gate hand cards by mana value
    // == the source's charge-counter count at resolution time.
    if (ability.change_type_cmc_expr.empty() && find_cmc_bound(ability.change_type, nullptr, op, bound)) {
        auto it = svars.find(bound);
        if (it != svars.end()) {
            ability.change_type_cmc_expr = it->second;
            ability.change_type_cmc_op = op;
        }
    }
    // DestroyAll with a dynamic mana-value bound and/or an energy unless-cost (Wrath of the
    // Skies, Blast Zone): resolve the "cmc<op><SVar>" threshold from the ValidCards$ filter and
    // the PayEnergy<SVar> amount into their runtime Count$ expressions.
    resolve_destroyall_svars(ability, svars);
    resolve_condition_svars(ability, svars);
    // Resolve a Pump NumAtt$/NumDef$ given as a count-SVar (Toxic Deluge: -X/-X → Count$xPaid;
    // Eldrazi Linebreaker: +X, X = Count$Valid Eldrazi.YouCtrl) to its runtime Count$ expression,
    // and a TargetMin$/TargetMax$ SVar to its exactly-X / up-to-X meaning (Candelabra, Hide on
    // the Ceiling, Kozilek's Command).
    resolve_pump_exprs(ability, svars);
    resolve_xpaid_target_counts(ability, svars, text);
    // Resolve AB$ Animate Power$/Toughness$ SVar tokens (Karn: Power$ X, X =
    // Targeted$CardManaCost) into their runtime dynamic_amount expression, evaluated against
    // the animate target at resolution. A token that is not an SVar key is left as no dynamic
    // expr (the numeric base, parsed by apply_param_to_ability, stands).
    if (ability.category == "Animate") {
        for (int which = 0; which < 2; which++) {
            const std::string tok = svar_key_param(text, which == 0 ? "Power" : "Toughness");
            std::string &expr = which == 0 ? ability.animate_power_expr : ability.animate_toughness_expr;
            if (tok.empty()) continue;
            auto it = svars.find(tok);
            if (it != svars.end()) expr = it->second;
        }
    }
    // Resolve dig_num_expr SVar reference (e.g. "X" → "Count$Devotion.Blue")
    if (!ability.dig_num_expr.empty()) {
        auto it = svars.find(ability.dig_num_expr);
        if (it != svars.end()) ability.dig_num_expr = it->second;
    }
    // ChooseNumber Max$ and a dynamic CounterNum$ both stash a raw SVar token (Wrath of the
    // Skies: Max$ Max → Count$YourCountersEnergy; CounterNum$ X → Count$xPaid). Resolve those
    // SVar references to their runtime Count$ expressions so the effect can evaluate them at
    // resolution.
    if (ability.category == "ChooseNumber" && !ability.dynamic_amount_expr.empty()) {
        auto it = svars.find(ability.dynamic_amount_expr);
        if (it != svars.end()) ability.dynamic_amount_expr = it->second;
    }
    if (auto *cp = std::get_if<CounterParams>(&ability.params)) {
        if (!cp->count_expr.empty()) {
            auto it = svars.find(cp->count_expr);
            if (it != svars.end()) cp->count_expr = it->second;
        }
    }
    // TokenPower$/TokenToughness$ given as an SVar token (Skyclave Apparition: "X" →
    // Remembered$CardManaCost): resolve the reference to its runtime expression so the Token
    // effect can size the created token's P/T at creation time.
    if (auto *tkp = std::get_if<TokenParams>(&ability.params)) {
        for (std::string *expr : {&tkp->power_expr, &tkp->toughness_expr}) {
            if (expr->empty()) continue;
            auto it = svars.find(*expr);
            if (it != svars.end()) *expr = it->second;
        }
    }
    // Resolve a cmcLE<SVar> threshold inside ChangeValid$ (Birthing Ritual: "Creature.cmcLEX")
    // into dynamic_amount_expr, evaluated by the Dig effect at resolution.
    if (ability.dynamic_amount_expr.empty() && find_cmc_bound(ability.change_valid, "LE", op, bound)) {
        auto it = svars.find(bound);
        if (it != svars.end()) ability.dynamic_amount_expr = it->second;
    }
}

// Resolves the statics an Effect's StaticAbilities$ SVar names into what the GrantCast handler
// registers: an emblem's permanent continuous static, a CantGainLife scope, a CastWithFlash
// permission.
static void resolve_effect_static_svars(AbilityDef &ability, const std::string &text,
                                        const std::map<std::string, std::string> &svars) {
    if (ability.category != "Effect" || ability.effect_static_ability.empty()) return;
    auto it = svars.find(ability.effect_static_ability);
    if (it == svars.end()) return;
    const std::string &body = it->second;
    // Emblem (CR 114): an Effect that grants a permanent continuous static to its controller —
    // Kaito's "[+1]: You get an emblem with 'Ninjas you control get +1/+1.'"; Tamiyo, Seasoned
    // Scholar's ultimate "You get an emblem with 'You have no maximum hand size.'"
    // (StaticAbilities$ <SVar> + Duration$ Permanent). Resolve the named continuous static SVar
    // into a StaticAbility and store it on the ability; the Effect handler creates a player-owned
    // emblem carrying it at resolution. Distinguished from the transient StaticAbilities$ Effects
    // (Unblockable, Ugin's MayPlay) by Duration$ Permanent — those are EOT/until-leaves and
    // handled by their own flags.
    if (text.find("Duration$ Permanent") != std::string::npos &&
        body.find("Mode$ Continuous") != std::string::npos) {
        StaticAbility est = parse_one_static_ability(body, svars);
        if (!est.category.empty()) ability.effect_emblem_statics.push_back(est);
    }
    // StaticAbilities$ <SVar(Mode$ CantGainLife | ValidPlayer$ ...)> — a turn-long life-gain
    // prohibition (CR 119.x, Roiling Vortex's {R}: "Your opponents can't gain life this turn.").
    // Classify its ValidPlayer scope relative to the effect's controller so the GrantCast
    // handler registers the right player(s).
    if (body.find("CantGainLife") != std::string::npos) {
        if (body.find("ValidPlayer$ Player.Opponent") != std::string::npos ||
            body.find("ValidPlayer$ Opponent") != std::string::npos)
            ability.effect_cant_gain_life = AbilityDef::CantGainLifeScope::OPPONENTS;
        else if (body.find("ValidPlayer$ You") != std::string::npos)
            ability.effect_cant_gain_life = AbilityDef::CantGainLifeScope::YOU;
        else
            ability.effect_cant_gain_life = AbilityDef::CantGainLifeScope::ALL;
    }
    // StaticAbilities$ <SVar(Mode$ CastWithFlash | ValidCard$ <filter> | Caster$ You)> (Teferi,
    // Time Raveler's +1): a cast-timing PERMISSION — the controller may cast matching (sorcery)
    // spells as though they had flash for the effect's Duration. The GrantCast handler records a
    // cur_game.resolved_effects.cast_with_flash_permissions entry for the effect's Duration
    // (duration_until_your_next_turn).
    if (body.find("CastWithFlash") != std::string::npos) {
        ability.effect_cast_with_flash = true;
        ability.effect_cast_with_flash_filter = param_value(body, "ValidCard");
    }
}

// Resolves a non-numeric amount param (NumCards$, NumDmg$, LifeAmount$, ...) that named an SVar:
// a direct dynamic expression is kept as is; otherwise the SVar body becomes the combat-damage
// amount, a conditional amount (Flow State), a delirium scale (Unholy Heat), or a runtime
// expression evaluated at activation/resolution.
static void resolve_amount_svar(AbilityDef &ability, const std::map<std::string, std::string> &svars,
                                const std::string &card_name) {
    if (ability.amount_svar.empty() || take_direct_amount_expr(ability)) return;
    auto it = svars.find(ability.amount_svar);
    if (it == svars.end()) warn_unresolved_amount_svar(ability, card_name);
    ability.amount_svar = "";
    if (it == svars.end()) return;
    const std::string &sv = it->second;
    // TriggerCount$DamageAmount → use combat damage trigger's damage amount at runtime
    if (sv == "TriggerCount$DamageAmount") {
        ability.amount_from_damage = true;
        return;
    }
    if (resolve_conditional_amount(ability, sv, svars)) return;
    // Delirium-conditional value. Two equivalent Forge spellings, both
    // meaning "<yes> if the caster has delirium, else <no>":
    //   Count$Delirium.<yes>.<no>           (compact form, e.g. Unholy Heat)
    //   Count$Compare Y GE4.<yes>.<no>      (explicit GE form)
    size_t delirium_pos = sv.find("Count$Delirium");
    size_t ge_pos = sv.find("GE");
    size_t scale_pos = delirium_pos != std::string::npos
                           ? delirium_pos + std::string("Count$Delirium").size()
                           : (ge_pos != std::string::npos ? ge_pos + 2 : std::string::npos);
    if (scale_pos != std::string::npos) {
        std::string rest = sv.substr(scale_pos);
        size_t d1 = rest.find('.');
        if (d1 == std::string::npos) return;
        size_t d2 = rest.find('.', d1 + 1);
        if (d2 == std::string::npos) return;
        ability.amount = static_cast<size_t>(std::stoi(rest.substr(d2 + 1)));
        DamageParams &dp = effect_params<DamageParams>(ability);
        dp.delirium_amount = static_cast<size_t>(std::stoi(rest.substr(d1 + 1, d2 - d1 - 1)));
        dp.is_delirium_scale = true;
        return;
    }
    if (is_runtime_amount_expr(sv)) ability.dynamic_amount_expr = sv;
}

// The generalized conditional amount (Flow State): "Count$Compare <Var> <op><n>.<t>.<f>" where
// <Var> is an SVar that resolves to a sum of (capped) runtime counts. The effective amount is <t>
// when the summed counts satisfy the compare, else <f>. Distinguished from the delirium GE form by
// <Var> being a nested SVar$ chain (e.g. SVar$Z1/Plus.Z2) rather than a direct Count$ expression.
// Returns true when `sv` is such an amount (now set on `ability`).
static bool resolve_conditional_amount(AbilityDef &ability, const std::string &sv,
                                       const std::map<std::string, std::string> &svars) {
    if (sv.rfind("Count$Compare ", 0) != 0) return false;
    std::string rest = sv.substr(14);  // "Y GE2.2.1"
    size_t sp = rest.find(' ');
    if (sp == std::string::npos) return false;
    std::string var = rest.substr(0, sp);    // "Y"
    std::string tail = rest.substr(sp + 1);  // "GE2.2.1"
    auto vit = svars.find(var);
    if (vit == svars.end() || vit->second.rfind("SVar$", 0) != 0) return false;
    size_t d2 = tail.rfind('.');
    size_t d1 = (d2 == std::string::npos || d2 == 0) ? std::string::npos : tail.rfind('.', d2 - 1);
    if (d1 == std::string::npos) return false;
    std::vector<std::string> terms;
    resolve_additive_svar(vit->second, svars, terms);
    if (terms.empty()) return false;
    ability.cond_amount_active = true;
    ability.cond_amount_exprs = terms;
    ability.cond_amount_compare = tail.substr(0, d1);  // "GE2"
    ability.cond_amount_if_true = static_cast<size_t>(std::stoi(tail.substr(d1 + 1, d2 - d1 - 1)));
    ability.amount = static_cast<size_t>(std::stoi(tail.substr(d2 + 1)));
    return true;
}

// An amount SVar body that is a runtime expression, kept verbatim for evaluate_amount at
// activation/resolution.
static bool is_runtime_amount_expr(const std::string &sv) {
    static const char *const kRuntimeMarkers[] = {
        "Count$Valid", "Targeted$", "Count$InYourLibrary", "Count$YourLifeTotal", "Count$Revolt",
        // Count$Threshold.<hi>.<lo> — graveyard-threshold ritual scaling (Cabal Ritual: Amount$ X,
        // X = Count$Threshold.5.3 → BBBBB if the caster has 7+ cards in their graveyard, else BBB).
        "Count$Threshold",
        // Count$UrzaLands.<hi>.<lo> — the "Tron" mana lands (Urza's Mine/Power Plant/Tower): hi
        // colorless mana if the controller controls a complete set of all three, else lo
        // (mana-ability path via eval_mana_amount).
        "Count$UrzaLands",
        // Count$CardCounters.<TYPE> — counters on the source permanent (The One Ring:
        // NumCards$/LifeAmount$ X = Count$CardCounters.BURDEN), evaluated against the source.
        "Count$CardCounters",
        // Count$Converge (Prismatic Ending) — the distinct colors of mana spent to cast the spell,
        // from current_converge(). Used as the cmcLEY exile threshold.
        "Count$Converge",
        // Remembered$Valid <filter> — count of remembered (e.g. just-moved by a RememberChanged$
        // ChangeZoneAll) cards matching the filter (Canoptek Scarab Swarm: TokenAmount$ X, X =
        // Remembered$Valid Land,Artifact). The RememberedLKI$ form reads the same remembered set,
        // but populated from a RememberLKI$ ChangeZone last-known-info snapshot (Reanimate:
        // LifeAmount$ X, X = RememberedLKI$CardManaCost = the reanimated creature's mana value).
        "Remembered$", "RememberedLKI$",
        // Count$xPaid — amount equals the X paid at cast (Kozilek's Command: TokenAmount$/ScryNum$
        // = X; Forth Eorlingas!: TokenAmount$ X → X 2/2 Human Knight tokens).
        "xPaid",
    };
    for (const char *marker : kRuntimeMarkers)
        if (sv.find(marker) != std::string::npos) return true;
    return false;
}

// Resolves a ConditionCheckSVar$ reference (Veil of Summer's SP$ Draw: "X" →
// "Count$ThisTurnCast_Card.OppCtrl+Blue,Card.OppCtrl+Black"; Thassa's Oracle) to its Count$
// expression, defaults a bare ConditionCheckSVar's comparator to Forge's GE1 (the value must be
// >= 1), and resolves a ConditionSVarCompare$ whose right-hand side is an SVar (e.g. "LEX" where
// X = "Count$Devotion.Blue").
static void resolve_condition_svars(AbilityDef &ability,
                                    const std::map<std::string, std::string> &svars) {
    if (!ability.condition_check_svar.empty()) {
        auto it = svars.find(ability.condition_check_svar);
        if (it != svars.end()) ability.condition_check_svar = it->second;
        if (ability.condition_svar_compare.empty()) ability.condition_svar_compare = "GE1";
    }
    if (ability.condition_svar_compare.size() >= 3) {
        std::string rhs_str = ability.condition_svar_compare.substr(2);
        // If RHS is not a pure integer, it might be an SVar reference
        if (!rhs_str.empty() && !std::isdigit(static_cast<unsigned char>(rhs_str[0])) && rhs_str[0] != '-') {
            auto it = svars.find(rhs_str);
            if (it != svars.end()) {
                ability.condition_compare_svar_expr = it->second;
                ability.condition_svar_compare = ability.condition_svar_compare.substr(0, 2);  // keep just "LE"
            }
        }
    }
}

// Public entry: parse one activated/spell ability body (the resolved RHS of an SVar, e.g.
// "AB$ Mana | Cost$ T | Produced$ C") into an Ability, honouring the full Forge ability
// grammar via parse_svar_ability. Used to materialize an AddAbility$ static's granted
// ability (Petrified Hamlet). No SVar table is available at the grant site, so an empty map
// is passed; the granted bodies in use are self-contained (no SVar references).
const AbilityDef *parse_ability_body(const std::string &body, AbilityDef::AbilityType type) {
    std::string key = "body:" + std::to_string(static_cast<int>(type)) + ":" + body;
    return keyed_ability_def(key, [&body, type] {
        static const std::map<std::string, std::string> kNoSvars;
        return parse_svar_ability(body, type, kNoSvars, "");
    });
}

const AbilityDef *parse_granted_trigger(const std::string &trigger_line,
                                        const std::string &svar_name,
                                        const std::string &svar_body) {
    std::string key = "granted_trigger:" + trigger_line + "\n" + svar_name + "\n" + svar_body;
    return keyed_ability_def(key, [&] {
        // Build the minimal svars table the trigger's Execute$ resolves against (its named execute
        // SVar), then run the shared trigger parser — so the granted trigger honours the full trigger
        // grammar (Mode$/Phase$/ValidPlayer$/Execute$ + the DB$ effect body) exactly as a printed T:.
        std::map<std::string, std::string> svars;
        if (!svar_name.empty()) svars[svar_name] = svar_body;
        AbilityDef d = parse_one_trigger(trigger_line, svars, "");
        d.ability_type = AbilityDef::TRIGGERED;
        return d;
    });
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

// Parses a card's (or token's) A: lines: "SP$ <category> | ..." is a spell ability, "AB$ <category>
// | ..." an activated ability (whichever prefix comes first names it). A line with neither, or with
// no category, is skipped.
static std::vector<AbilityDef> parse_abilities(const std::vector<std::string> &lines,
                                               const std::map<std::string, std::string> &svars,
                                               const std::string &card_name) {
    std::vector<AbilityDef> ret_val;
    for (const auto &line : lines) {
        size_t sp_pos = line.find("SP$");
        size_t ab_pos = line.find("AB$");
        if (sp_pos == std::string::npos && ab_pos == std::string::npos) continue;
        bool is_spell = sp_pos != std::string::npos && (ab_pos == std::string::npos || sp_pos < ab_pos);
        size_t pos = (is_spell ? sp_pos : ab_pos) + 4;  // skip "SP$ " / "AB$ "
        if (pos >= line.length()) continue;
        size_t category_end = line.find_first_of(" |", pos);
        if (category_end == std::string::npos) category_end = line.length();
        if (category_end <= pos) continue;
        ret_val.push_back(parse_ability_text(
            line, pos, is_spell ? AbilityDef::SPELL : AbilityDef::ACTIVATED, svars, card_name));
    }
    return ret_val;
}

// Parses a single T: trigger line and its Execute$ SVar into a triggered Ability.
// Returns a default Ability with trigger_on == 0 if the trigger is unrecognised.
static AbilityDef parse_one_trigger(const std::string &line, const std::map<std::string, std::string> &svars,
                                 const std::string& card_name) {
    // The Execute$ SVar supplies the effect; the trigger metadata parsed from the T: line below is
    // written directly onto it, so no trigger field has to be carried over by hand.
    AbilityDef ability;
    auto exec_it = svars.find(param_value(line, "Execute"));
    if (exec_it != svars.end()) {
        // Check for Sylvan Library pattern: ChooseCard with DrawnThisTurn
        if (exec_it->second.find("ChooseCard") != std::string::npos &&
            exec_it->second.find("DrawnThisTurn") != std::string::npos)
            ability.category = "SylvanLibrary";
        else
            ability = parse_svar_ability(exec_it->second, AbilityDef::TRIGGERED, svars, card_name);
    }
    ability.ability_type = AbilityDef::TRIGGERED;
    bind_trigger_line(read_trigger_line(line, svars, ability), ability);
    return ability;
}

// Reads a trigger line's params. Filters and gates that need no event binding are written straight
// onto `ability`; the facts the event binding depends on are returned.
static TriggerLine read_trigger_line(const std::string &line,
                                     const std::map<std::string, std::string> &svars,
                                     AbilityDef &ability) {
    TriggerLine t;
    size_t param_pos = 0;
    std::string key, value;
    while (next_param(line, param_pos, key, value)) {
        if (key == "Mode") {
            t.mode = value;
        } else if (key == "ValidSource") {
            // Mode$ BecomesTarget | ValidSource$ Spell.OppCtrl — the targeting object must be a
            // SPELL (not an ability) controlled by an opponent of the source's controller.
            if (value.rfind("Spell", 0) == 0) t.source_is_spell = true;
            if (value.find("OppCtrl") != std::string::npos) t.source_opp_ctrl = true;
            // Mode$ DamageAll | ValidSource$ Creature.YouCtrl — the damaging creature must be one
            // this trigger's controller controls (Forth Eorlingas!'s floating monarch trigger).
            if (filter_has_head(value, "Creature") && filter_names_token(value, "YouCtrl"))
                t.source_creature_youctrl = true;
        } else if (key == "ValidTarget") {
            // ValidTarget$ Card.Self — the permanent that became a target must be this source.
            if (value == "Card.Self") t.target_self = true;
        } else if (key == "Activator") {
            // Mode$ TapsForMana | Activator$ You — only the source controller tapping a
            // permanent for mana fires this ("whenever YOU tap ...").
            if (value == "You") t.player_is_you = true;
        } else if (key == "Static") {
            // Static$ True on a TapsForMana trigger: it is a mana-additional effect that does
            // not use the stack (CR 605.1a) — resolved immediately by the mana system.
            if (value == "True") t.is_static = true;
        } else if (key == "AttackingPlayer") {
            // Mode$ AttackersDeclared | AttackingPlayer$ You — the trigger fires only when
            // the player who declared attackers is this ability's controller ("whenever you attack").
            if (value == "You") t.attacking_player_is_you = true;
        } else if (key == "Phase") {
            // A Phase trigger may list several phases comma-separated (Carpet of Flowers:
            // Phase$ Main1,Main2 — "at the beginning of each of your main phases").
            t.phases = split(value, ',');
        } else if (key == "TriggerZones") {
            // The zone(s) the source must be in for this triggered ability to function
            // (CR 113.6 / 603.6). Arclight Phoenix's combat trigger functions from the
            // graveyard, so the trigger scan must look at graveyard cards, not just the
            // battlefield.
            if (value.find("Graveyard") != std::string::npos) t.from_graveyard = true;
        } else if (key == "ValidPlayer" || key == "ValidActivatingPlayer") {
            if (value == "You") t.player_is_you = true;
            // ValidActivatingPlayer$ Opponent (Lavinia, Azorius Renegade): the trigger fires only
            // when an OPPONENT of the source's controller is the acting player.
            if (value == "Opponent") ability.trigger_valid_player_is_opponent = true;
        } else if (key == "ValidSA") {
            read_mana_spent_filter(value, ability);
            if (value.find("YouCtrl") != std::string::npos) t.player_is_you = true;
        } else if (key == "Origin") {
            t.origin = trigger_zone_filter(value);
        } else if (key == "Destination") {
            t.destination = trigger_zone_filter(value);
        } else if (key == "ValidCard" || key == "ValidCards") {
            read_trigger_valid_card(value, svars, t, ability);
        } else if (key == "OptionalDecider") {
            // Any named decider ("You" / "TriggeredCardController" / "Controller") makes the
            // whole triggered ability optional ("you may ...") for that player — the controller
            // of the source, which is who the engine prompts in every supported case.
            if (value.find("You") != std::string::npos ||
                value.find("Controller") != std::string::npos)
                t.optional = true;
        } else if (key == "FirstCardInDrawStep") {
            if (value == "False") t.exclude_first_draw_step = true;
        } else if (key == "Number") {
            // Number$ N on a Mode$ Drawn trigger (Tamiyo, Inquisitive Student: "your THIRD card
            // in a turn"). Fire only on the Nth card the player draws this turn.
            if (!value.empty() && isdigit((unsigned char)value[0]))
                t.draw_number_eq = static_cast<size_t>(std::stoi(value));
        } else if (key == "Attacked") {
            // Attacked$ You,Planeswalker.YouCtrl (Tamiyo, Seasoned Scholar's +2 trigger): in a
            // two-player game an opponent's attacker can only be attacking the trigger's
            // controller or a planeswalker they control (CR 508.1), so it needs no gate.
        } else if (key == "CombatDamage") {
            if (value == "True") t.combat_damage_only = true;
        } else if (key == "ActivatorThisTurnCast") {
            if (value.rfind("EQ", 0) == 0) {
                t.cast_count_eq = static_cast<size_t>(std::stoi(value.substr(2)));
            }
        } else if (key == "IsPresent") {
            // Intervening-if (603.4): "..., if you control a <thing>, ...". Checked both
            // when the trigger would go on the stack and again on resolution.
            t.condition_present = value;
            t.intervening_if = true;
        } else if (key == "PresentCompare") {
            t.condition_compare = value;  // e.g. "GE2"; empty defaults to ">= 1"
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
                t.condition_present = (it != svars.end()) ? it->second : value;
                t.intervening_if = true;
            }
        } else if (key == "SVarCompare") {
            // SVarCompare follows CheckSVar on the line; route it to whichever gate CheckSVar set up.
            if (!ability.stored_svar_gate_name.empty())
                ability.stored_svar_gate_compare = value;  // per-permanent stored-SVar latch compare
            else
                t.condition_compare = value;  // explicit compare for the CheckSVar count gate
        }
    }
    return t;
}

// The zone a ChangesZone trigger's Origin$/Destination$ names (-1 = any zone; only the
// battlefield and graveyard filters are modeled).
static int trigger_zone_filter(const std::string &value) {
    if (value == "Battlefield") return Zone::BATTLEFIELD;
    if (value == "Graveyard") return Zone::GRAVEYARD;
    return -1;
}

// SpellCast trigger ValidSA$ Spell.ManaSpent <op><n> (Roiling Vortex: "if no mana was spent to
// cast that spell" = Spell.ManaSpent EQ0). Parse the ManaSpent comparison (op + integer) into the
// runtime filter; a ".YouCtrl"/".OppCtrl" restriction on the spell's controller is handled by the
// ValidActivatingPlayer path. General over any Spell.ManaSpent-gated SpellCast trigger.
static void read_mana_spent_filter(const std::string &value, AbilityDef &ability) {
    size_t mp = value.find("ManaSpent");
    if (mp == std::string::npos) return;
    size_t p = mp + strlen("ManaSpent");
    while (p < value.size() && value[p] == ' ') p++;
    for (const char *op : {"EQ", "NE", "LE", "GE", "LT", "GT"}) {
        if (value.compare(p, 2, op) != 0) continue;
        ability.trigger_mana_spent_op = op;
        p += 2;
        int n = 0;
        bool any = false;
        while (p < value.size() && isdigit((unsigned char)value[p])) {
            n = n * 10 + (value[p++] - '0');
            any = true;
        }
        if (any) ability.trigger_mana_spent_val = n;
        return;
    }
}

// A trigger's ValidCard$ / ValidCards$ filter. The filter itself is matched against the event's
// object at trigger time (zone_change_object_matches); only the tokens that select an event
// binding or an identity gate are read here, as whole filter tokens (so "nonCreature" is not read
// as "Creature", nor "nonLand" as "Land").
static void read_trigger_valid_card(const std::string &value,
                                    const std::map<std::string, std::string> &svars, TriggerLine &t,
                                    AbilityDef &ability) {
    ability.trigger_valid_card = value;
    if (filter_has_head(value, "Creature"))          t.valid_card_creature     = true;
    if (filter_names_token(value, "nonCreature"))    t.valid_card_non_creature = true;
    if (filter_names_token(value, "Other"))          ability.trigger_self_excluded = true;
    // Self may be the head qualifier (Card.Self) or a later one (The One Ring's
    // Card.wasCastByYou+Self).
    if (filter_names_token(value, "Self"))           t.valid_card_self         = true;
    // wasCastByYou — "if you cast it" cast-condition on an ETB trigger (The One Ring): the
    // source must have entered by being cast (Permanent::entered_by_cast).
    if (filter_names_token(value, "wasCastByYou"))
        ability.trigger_requires_entered_by_cast = true;
    // Kicker-linked condition (CR 702.33f): "Card.Self+kicked N" — fires only when the
    // Nth kicker was paid. Parse the 1-based index after "kicked " (a missing number
    // defaults to the first kicker). General over any "+kicked N" SpellCast trigger.
    size_t kp = value.find("kicked");
    if (kp != std::string::npos) {
        size_t np = kp + strlen("kicked");
        while (np < value.size() && value[np] == ' ') np++;
        int n = 0;
        while (np < value.size() && isdigit((unsigned char)value[np]))
            n = n * 10 + (value[np++] - '0');
        t.kicked_index = (n > 0) ? n : 1;
    }
    if (filter_names_token(value, "OppOwn"))         t.valid_card_opp_own      = true;
    if (filter_names_token(value, "OppCtrl"))        t.valid_card_opp_ctrl     = true;
    if (filter_names_token(value, "Colorless"))      t.valid_card_colorless    = true;
    // "untapped" qualifier — the changing card must be untapped when the trigger checks
    // it (Mystic Sanctuary: ValidCard$ Card.Self+untapped, "enters untapped").
    if (filter_names_token(value, "untapped"))       t.valid_card_untapped     = true;
    // YouCtrl on a Drawn / SpellCast trigger names the event's player (the drawer / caster).
    if (filter_names_token(value, "YouCtrl"))        t.player_is_you           = true;
    // Dynamic mana-value filter on the cast spell (Chalice of the Void:
    // "Card.cmcEQY", Y = Count$CardCounters.CHARGE). Resolve the cmc<op><svar>
    // qualifier to its runtime Count$ expression + comparison op, mirroring the
    // ChangeType cmcEQ handling used by Aether Vial.
    std::string op, bound;
    if (find_cmc_bound(value, nullptr, op, bound)) {
        auto it = svars.find(bound);
        if (it != svars.end()) {
            ability.trigger_cmc_expr = it->second;
            ability.trigger_cmc_op = op;
        } else if (!bound.empty() && bound.find_first_not_of("0123456789") == std::string::npos) {
            // A LITERAL numeric bound (Eidolon of the Great Revel: Card.cmcLE3).
            // evaluate_svar returns a plain integer literal as itself, so store
            // the number directly; without this the filter would be dropped and the
            // trigger would fire on every spell.
            ability.trigger_cmc_expr = bound;
            ability.trigger_cmc_op = op;
        }
    }
}

// Binds a read trigger line onto `ability`: its intervening-if and optionality, then the event its
// Mode$ fires on together with that mode's filters.
static void bind_trigger_line(const TriggerLine &t, AbilityDef &ability) {
    // 603.4 intervening-if from the trigger line (IsPresent$ / CheckSVar$). It replaces any
    // condition the Execute SVar declared; an SVar-only intervening-if (Uro's TrigSac
    // ConditionNotPresent$ Card.Self+escaped) is kept when the line declares none.
    if (t.intervening_if) {
        ability.intervening_if = true;
        ability.condition_present = t.condition_present;
        ability.condition_compare = t.condition_compare;
    }
    // OptionalDecider$ You ("At the beginning of your upkeep, you may ...") makes the whole
    // triggered ability optional at resolution, independent of the trigger mode (Aether
    // Vial's upkeep charge-counter trigger is a Phase trigger, not a zone-change trigger).
    ability.trigger_optional = t.optional;
    // Static$ True on a phase or zone-change trigger (Carpet of Flowers' cleanup / leave-battlefield
    // resets) is a bookkeeping trigger that resolves immediately off the stack — it never uses the
    // stack like a normal triggered ability (CR 605.1a-style). The TapsForMana static path has its
    // own dedicated flag (trigger_taps_for_mana_static) and inline mana-system handling, so it is
    // excluded here. General over any Static$ True phase/ChangesZone trigger.
    if (t.is_static && t.mode != "TapsForMana") ability.trigger_static_offstack = true;
    // TriggerZones$ Graveyard — the ability functions while its source is in the graveyard.
    ability.trigger_from_graveyard = t.from_graveyard;

    const std::string &mode = t.mode;
    if (mode == "ChangesZone" || mode == "ChangesZoneAll") {
        // All ChangesZone triggers use CARD_CHANGED_ZONE; origin/destination/type filters applied
        // at match time. ChangesZone fires once per matching card. ChangesZoneAll ("whenever one
        // or more cards ...") is a single batch trigger (CR 603.2c): it fires exactly ONCE for a
        // group of simultaneous zone changes, no matter how many cards matched, so the trigger
        // scan dedupes it to a single firing per batch (Moonshadow: milling 3 permanent cards
        // removes ONE -1/-1 counter, not three).
        ability.trigger_on = Events::CARD_CHANGED_ZONE;
        ability.trigger_zone_origin = t.origin;
        ability.trigger_zone_destination = t.destination;
        // ValidCard$ (types, control, ownership, …) is matched against the moving object itself
        // through ability.trigger_valid_card, not against the event's player.
        ability.trigger_valid_card_untapped = t.valid_card_untapped;
        ability.trigger_batch_zone_all = (mode == "ChangesZoneAll");
        if (t.valid_card_self) ability.trigger_only_self = true;
        // A Destination$ Battlefield trigger gated by IsPresent$ Card.Self (the source must
        // already be on the battlefield) is Forge's idiom for "Whenever ANOTHER permanent
        // enters" — the source's own entry must not satisfy it (Kappa Cannoneer's Oracle text
        // reads "another artifact you control"). Exclude the source from this trigger.
        if (t.destination == Zone::BATTLEFIELD && t.intervening_if && t.condition_present == "Card.Self")
            ability.trigger_self_excluded = true;
    } else if (mode == "Phase") {
        bind_phase_trigger(t, ability);
    } else if (mode == "SpellCast") {
        bind_spell_cast_trigger(t, ability);
    } else if (mode == "DamageDone") {
        // "Whenever CARDNAME deals combat damage to a player" — Barrowgoyf
        if (t.combat_damage_only) {
            ability.trigger_on = Events::COMBAT_DAMAGE_TO_PLAYER;
            ability.trigger_only_self = true;  // ValidSource$ Card.Self
        }
    } else if (mode == "DamageAll") {
        // "Whenever one or more creatures you control deal combat damage to one or more players" —
        // Forth Eorlingas!'s floating monarch trigger (Mode$ DamageAll | ValidSource$ Creature.YouCtrl
        // | ValidTarget$ Player | CombatDamage$ True). Fires on COMBAT_DAMAGE_TO_PLAYER when the
        // damaging creature is controlled by this trigger's controller (matched at fire time, since
        // the floating trigger has no source permanent to self-reference).
        if (t.combat_damage_only) {
            ability.trigger_on = Events::COMBAT_DAMAGE_TO_PLAYER;
            ability.trigger_damage_source_youctrl = t.source_creature_youctrl;
        }
    } else if (mode == "Drawn") {
        // "whenever a player draws a card" — Orcish Bowmasters (Mode$ Drawn)
        ability.trigger_on = Events::PLAYER_DREW_CARD;
        ability.trigger_valid_card_opp_own = t.valid_card_opp_own;
        ability.trigger_exclude_first_draw_step = t.exclude_first_draw_step;
        ability.trigger_draw_number_eq = t.draw_number_eq;
        // ValidCard$ Card.YouCtrl on a Drawn trigger ("whenever YOU draw ...", Tamiyo): the drawer
        // must be the source's controller. Reuse the controller-is-event-player gate (the
        // PLAYER_DREW_CARD event's PLAYER is the drawer), set by YouCtrl in the ValidCard parse.
        ability.trigger_valid_player_is_controller = t.player_is_you;
    } else if (mode == "Attacks") {
        // "Whenever CARDNAME attacks, ..." — Phelia (Mode$ Attacks | ValidCard$ Card.Self). Fires
        // once for this creature each time it is declared as an attacker (CR 508.2). ValidCard$
        // Card.Self → trigger_only_self matches the attacking ENTITY against the source.
        ability.trigger_on = Events::CREATURE_ATTACKED;
        if (t.valid_card_self) ability.trigger_only_self = true;
        // ValidCard$ Creature.OppCtrl | Attacked$ You,Planeswalker.YouCtrl (Tamiyo, Seasoned
        // Scholar's +2 hosted trigger): an opponent's creature attacking you/your planeswalker.
        // Matched at fire time against the attacker's controller (the trigger has no source perm).
        ability.trigger_attacker_opp_ctrl = t.valid_card_opp_ctrl;
    } else if (mode == "AttackersDeclared") {
        // "Whenever you attack" — Guide of Souls (Mode$ AttackersDeclared | AttackingPlayer$ You).
        // Fires once per combat when the source's controller declares one or more attackers.
        ability.trigger_on = Events::ATTACKERS_DECLARED;
        ability.trigger_valid_player_is_controller = t.attacking_player_is_you;
    } else if (mode == "TapsForMana") {
        // "Whenever you tap a creature for mana, add an additional {G}." — Badgermole Cub
        // (Mode$ TapsForMana | ValidCard$ Creature | Activator$ You | Static$ True). A
        // mana-additional triggered ability resolved immediately by the mana system (off-stack,
        // CR 605.1a) rather than placed on the stack.
        ability.trigger_on = Events::TAPPED_FOR_MANA;
        ability.trigger_valid_card_is_creature = t.valid_card_creature;
        ability.trigger_valid_player_is_controller = t.player_is_you;
        ability.trigger_taps_for_mana_static = t.is_static;
    } else if (mode == "BecomesTarget") {
        // "Whenever CARDNAME becomes the target of a spell an opponent controls, ..." — Reality
        // Smasher (Mode$ BecomesTarget | ValidSource$ Spell.OppCtrl | ValidTarget$ Card.Self). Fires
        // when this permanent becomes the target of a matching spell (CR 603.2c). ValidTarget$
        // Card.Self reuses trigger_only_self (the targeted permanent must be the source).
        ability.trigger_on = Events::BECAME_TARGET;
        ability.trigger_source_must_be_spell = t.source_is_spell;
        ability.trigger_source_opp_ctrl = t.source_opp_ctrl;
        if (t.target_self) ability.trigger_only_self = true;
    } else if (mode == "BecomeMonstrous") {
        // "When CARDNAME becomes monstrous, ..." — Mode$ BecomeMonstrous (CR 701.37). Fired by the
        // resolving Monstrosity$ ability (effect_put_counter.cpp) with ENTITY = the permanent that
        // became monstrous, so ValidCard$ Card.Self reuses the standard trigger_only_self ENTITY check.
        // TriggerZones$ Battlefield is the default functioning zone; no extra handling needed.
        ability.trigger_on = Events::BECAME_MONSTROUS;
        if (t.valid_card_self) ability.trigger_only_self = true;
    } else if (mode == "Always") {
        // Mode$ Always — a state-triggered ability (CR 603.8). Its trigger condition is a game STATE
        // (the IsPresent$ intervening-if bound above into condition_present/intervening_if), not a
        // game event, so it has no trigger_on; the dedicated state-trigger scan in
        // collect_triggered_abilities evaluates the condition each SBA pass and fires once when it
        // becomes true. Dark Depths: IsPresent$ Card.Self+counters_EQ0_ICE ("when this has no ice
        // counters on it"). trigger_only_self is set so the source is the permanent whose counters
        // are checked. parse_triggered_abilities keeps this ability despite trigger_on == 0.
        ability.trigger_state_condition = true;
        ability.trigger_only_self = true;
    }
}

// Mode$ Phase: "at the beginning of [your] <step>". The first listed phase (in the order below)
// binds trigger_on and any further one is an additional event the same trigger fires on (Carpet
// of Flowers' Phase$ Main1,Main2 — "each of your main phases").
static void bind_phase_trigger(const TriggerLine &t, AbilityDef &ability) {
    static const struct {
        const char *phase;
        EventId event;
    } kPhaseEvents[] = {
        {"Upkeep", Events::UPKEEP_BEGAN},
        // Forge writes the end step as either "EndStep" or "End of Turn".
        {"EndStep", Events::END_STEP_BEGAN},
        {"End of Turn", Events::END_STEP_BEGAN},
        {"Draw", Events::DRAW_STEP_BEGAN},
        {"BeginCombat", Events::BEGIN_COMBAT_BEGAN},
        // Forge writes the (pre-combat) first main phase as "Main1", the post-combat one as "Main2".
        {"Main1", Events::FIRST_MAIN_BEGAN},
        {"Main2", Events::SECOND_MAIN_BEGAN},
        // With no ValidPlayer$ You a Cleanup trigger fires at every cleanup (Carpet of Flowers'
        // Static$ True reset re-arming its latch).
        {"Cleanup", Events::CLEANUP_BEGAN},
    };
    for (const auto &pe : kPhaseEvents) {
        if (std::find(t.phases.begin(), t.phases.end(), pe.phase) == t.phases.end()) continue;
        if (ability.trigger_on == 0) ability.trigger_on = pe.event;
        else if (ability.trigger_on != pe.event &&
                 std::find(ability.trigger_on_extra.begin(), ability.trigger_on_extra.end(),
                           pe.event) == ability.trigger_on_extra.end())
            ability.trigger_on_extra.push_back(pe.event);
        ability.trigger_valid_player_is_controller = t.player_is_you;
    }
}

// Mode$ SpellCast: which cast event the trigger binds to, and its gates on the cast spell.
static void bind_spell_cast_trigger(const TriggerLine &t, AbilityDef &ability) {
    if (t.valid_card_non_creature) {
        ability.trigger_on = Events::NONCREATURE_SPELL_CAST;
        ability.trigger_valid_player_is_controller = t.player_is_you;
    }
    // "Whenever you cast a colorless spell, ..." — Glaring Fleshraker
    // (Mode$ SpellCast | ValidCard$ Card.Colorless | ValidActivatingPlayer$ You). A plain
    // SpellCast with a colorless filter on the cast spell; matched at trigger time against the
    // spell's colorlessness (CR 105.2c). Keyed on the general Colorless tag, not this card.
    if (t.valid_card_colorless) {
        ability.trigger_on = Events::SPELL_CAST;
        ability.trigger_valid_card_colorless = true;
        ability.trigger_valid_player_is_controller = t.player_is_you;
    }
    // "whenever you cast your Nth spell" — Cori-Steel Cutter; or "your Nth NONCREATURE spell each
    // turn" — The Fantasticar. Bind to SPELL_CAST (fired AFTER the per-cast spell counters bump,
    // unlike NONCREATURE_SPELL_CAST which fires before) so the count gate sees the current cast.
    if (t.cast_count_eq > 0) {
        ability.trigger_on = Events::SPELL_CAST;
        ability.trigger_valid_player_is_controller = t.player_is_you;
        ability.trigger_spell_count_eq = t.cast_count_eq;
        if (t.valid_card_non_creature) {
            // Count only noncreature spells, and only fire on a noncreature cast (the SPELL_CAST
            // event carries every spell, so filter the triggering card to noncreature too).
            ability.trigger_valid_card_non_creature = true;
            ability.trigger_spell_count_noncreature = true;
        }
    }
    // "Whenever a player casts a spell with mana value equal to ..." — Chalice of the Void
    // (Mode$ SpellCast | ValidCard$ Card.cmcEQY | ValidActivatingPlayer$ Player). A dynamic
    // mana-value filter on any player's spell. The cmc match is checked at trigger time.
    if (!ability.trigger_cmc_expr.empty()) {
        ability.trigger_on = Events::SPELL_CAST;
        ability.trigger_valid_player_is_controller = t.player_is_you;
    }
    // "When you cast this spell, [if it was kicked with its [N] kicker,] ..." — Wastescape
    // Battlemage (Mode$ SpellCast | ValidCard$ Card.Self[+kicked N]). A linked self-cast
    // trigger that fires while the spell is on the stack (CR 702.33e/f). trigger_only_self
    // restricts it to the source spell; trigger_kicked_index (>0) additionally gates on the
    // Nth kicker having been paid. Handled by the dedicated self-cast SPELL_CAST scan.
    if (t.valid_card_self) {
        ability.trigger_on = Events::SPELL_CAST;
        ability.trigger_only_self = true;
        ability.trigger_kicked_index = t.kicked_index;
    }
    // General "whenever you cast a spell, ..." — Paradox Engine
    // (Mode$ SpellCast | ValidCard$ Card | ValidActivatingPlayer$ You). A plain, unfiltered
    // SpellCast trigger that fires on EVERY spell the source's controller casts. None of the
    // specialized SpellCast bindings above (noncreature/colorless/count/cmc/self) applied, so
    // bind to SPELL_CAST and gate on the caster being this source's controller. The general
    // battlefield trigger scan fires it (no extra ValidCard$ filter ⇒ any spell). Keyed on the
    // bare SpellCast mode, not this card.
    if (ability.trigger_on == 0) {
        ability.trigger_on = Events::SPELL_CAST;
        ability.trigger_valid_player_is_controller = t.player_is_you;
    }
}

static std::vector<AbilityDef> parse_triggered_abilities(const std::string &script,
                                                      const std::map<std::string, std::string> &svars,
                                                      const std::string& card_name) {
    std::vector<AbilityDef> result;
    for (const auto &line : multi_values_from_script(script, "T")) {
        AbilityDef ab = parse_one_trigger(line, svars, card_name);
        // Keep event-driven triggers (trigger_on != 0) and state-triggered abilities
        // (Mode$ Always, CR 603.8), which have no event but are fired by the state-trigger scan.
        if (ab.trigger_on != 0 || ab.trigger_state_condition)
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
            } else if (key == "AddTrigger") {
                // AddTrigger$ <SVarName> (The Tabernacle): a continuous static that grants a full
                // TRIGGERED ability to every Affected$ permanent (CR 613.1f, layer 6). Resolve the
                // named SVar to the trigger line body now; the paired AddSVar$ supplies the
                // Execute$ SVar the layer-6 grant pass needs to reparse it (parse_granted_trigger).
                auto it = svars.find(value);
                sa.add_trigger = (it != svars.end()) ? it->second : value;
            } else if (key == "AddSVar") {
                // AddSVar$ <SVarName> — the Execute$ SVar the granted trigger references. Store both
                // its name (so the reparse's svars map is keyed correctly) and its resolved body.
                sa.add_trigger_svar_name = value;
                auto it = svars.find(value);
                sa.add_trigger_svar = (it != svars.end()) ? it->second : value;
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
            } else if (key == "OnlySorcerySpeed") {
                // CantBeCast OnlySorcerySpeed$ True (Teferi, Time Raveler): a TIMING restriction,
                // not a blanket prohibition — the affected caster may cast spells only when they
                // could cast a sorcery (CR 601.3a). Enforced in the cast-speed gate
                // (rules_mod::opponent_sorcery_speed_locked), not by cast_prohibited.
                if (sa.category == "CantBeCast") sa.only_sorcery_speed = (value == "True");
            } else if (key == "Origin") {
                // CantBeCast Origin$ Graveyard,Library (Grafdigger's Cage): restrict casting
                // spells from these zones (e.g. flashback).
                if (sa.category == "CantBeCast") {
                    if (value.find("Graveyard") != std::string::npos) sa.cant_cast_from_graveyard = true;
                    if (value.find("Library")   != std::string::npos) sa.cant_cast_from_library   = true;
                }
            } else if (key == "cmcGT") {
                // CantBeCast cmcGT$ Land (Lavinia, Azorius Renegade): a DYNAMIC mana-value bound.
                // The spell is prohibited when its mana value exceeds the number of lands the
                // caster controls. Enforced in rules_mod::cast_prohibited on the opponent path.
                if (sa.category == "CantBeCast" && value == "Land") sa.cant_cast_cmc_gt_land = true;
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

    for (const auto& line : multi_values_from_script(script, "R")) {
        bool event_is_moved       = false;
        bool event_is_counter     = false;
        bool event_is_untap       = false;
        bool event_is_produce_mana = false;       // Event$ ProduceMana (Damping Sphere)
        bool event_is_draw        = false;        // Event$ Draw (Jace, Wielder of Mysteries: win on empty-library draw)
        bool event_is_draw_cards  = false;        // Event$ DrawCards (Quantum Riddler: additive "draw that many plus one")
        std::string draw_check_svar;              // CheckSVar$ <var> — the count-gate SVar name (resolved to its Count$ body below)
        std::string draw_svar_compare;            // SVarCompare$ <cmp> — comparator for the count gate (e.g. "LE1")
        bool valid_player_you     = false;        // ValidPlayer$ You — the replacement applies to the source controller's own draws
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
        // Spell-mastery-style gate on a self CANT_BE_COUNTERED (Exquisite Firecraft): IsPresent$
        // comma-OR filter counted in PresentZone$ against PresentCompare$ (e.g. two+ instant/
        // sorcery cards in your graveyard). Left empty for an unconditional "can't be countered".
        std::string cant_counter_present, cant_counter_zone, cant_counter_compare;

        size_t param_pos = 0;
        std::string key, value;
        while (next_param(line, param_pos, key, value)) {
            if      (key == "Event"       && value == "Moved")       event_is_moved          = true;
            else if (key == "Event"       && value == "Counter")    event_is_counter        = true;
            else if (key == "Event"       && value == "Untap")      event_is_untap          = true;
            else if (key == "Event"       && value == "ProduceMana") event_is_produce_mana  = true;
            else if (key == "Event"       && value == "Draw")        event_is_draw           = true;
            else if (key == "Event"       && value == "DrawCards")   event_is_draw_cards     = true;
            else if (key == "ValidPlayer" && value == "You")         valid_player_you        = true;
            else if (key == "CheckSVar")  draw_check_svar          = value;
            else if (key == "SVarCompare") draw_svar_compare       = value;
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
            else if (key == "ValidCard"   && filter_has_head(value, "Creature") &&
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
            else if (key == "ValidLKI"    && filter_has_head(value, "Creature"))
                valid_lki_creature = true;
            else if (key == "Origin") {
                if (value.find("Graveyard") != std::string::npos) origin_graveyard = true;
                if (value.find("Library")   != std::string::npos) origin_library   = true;
            }
            else if (key == "IsPresent")     cant_counter_present = value;
            else if (key == "PresentZone")   cant_counter_zone    = value;
            else if (key == "PresentCompare") cant_counter_compare = value;
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
        // Mox Diamond / Chrome Mox "pay-a-discard-as-it-enters-else-to-graveyard" form: the
        // ReplaceWith$ SVar chain is an optional DB$ Discard (DiscardValid$ <filter>) whose
        // subability moves the card (Defined$ ReplacedCard) to the graveyard when nothing was
        // discarded. Walk the SubAbility$ chain of named SVars to recognize all three signals
        // (an optional discard, its DiscardValid$ filter, an "else to graveyard" move) rather than
        // retagging the R: line.
        bool replace_with_discard_else_grave = false;
        std::string discard_else_filter;
        if (!replace_with_svar.empty()) {
            std::string cur = replace_with_svar;
            bool saw_discard = false, saw_optional = false, saw_grave = false;
            for (int guard = 0; guard < 8 && !cur.empty(); guard++) {
                auto sv2 = svars.find(cur);
                if (sv2 == svars.end()) break;
                const std::string &b = sv2->second;
                if (b.find("DB$ Discard") != std::string::npos)          saw_discard = true;
                if (b.find("Optional$ True") != std::string::npos)       saw_optional = true;
                if (b.find("Destination$ Graveyard") != std::string::npos) saw_grave = true;
                std::string next_sub;
                size_t pp = 0; std::string k, v;
                while (next_param(b, pp, k, v)) {
                    if (k == "DiscardValid" && discard_else_filter.empty()) discard_else_filter = v;
                    else if (k == "SubAbility") next_sub = v;
                }
                cur = next_sub;
            }
            if (saw_discard && saw_optional && saw_grave) replace_with_discard_else_grave = true;
        }
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
        // Mox Diamond / Chrome Mox: as this card would enter, its owner may discard a matching
        // card (an additional cost); if they don't, it goes to its owner's graveyard instead
        // (614.1a self-replacement).
        if (event_is_moved && valid_card_self && dest_is_battlefield &&
            replace_with_discard_else_grave) {
            Effect::Replacement r;
            r.kind = Effect::Replacement::DISCARD_ELSE_GRAVEYARD;
            r.applies_to_self_only = true;
            r.discard_else_filter = discard_else_filter.empty() ? "Card" : discard_else_filter;
            result.push_back(r);
        }
        if (event_is_counter && valid_card_self && layer_cant_happen) {
            Effect::Replacement r;
            r.kind = Effect::Replacement::CANT_BE_COUNTERED;
            r.applies_to_self_only = true;
            // Spell-mastery gate (Exquisite Firecraft): carried onto the replacement and
            // re-evaluated at counter time. Empty = the unconditional self form.
            r.cant_counter_present = cant_counter_present;
            r.cant_counter_zone    = cant_counter_zone;
            r.cant_counter_compare = cant_counter_compare;
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
        // Jace, Wielder of Mysteries: "If you would draw a card while your library has no cards in
        // it, you win the game instead." (CR 104.3a/121.4). A draw-event replacement whose
        // ReplaceWith$ SVar is a DB$ WinsGame; it fires for the CONTROLLER's own empty-library draw
        // (checked at the empty-draw site in Orderer::perform_draw). We confirm the replacement's
        // effect is a game win via the named SVar body rather than retagging the R: line.
        {
            bool replace_with_win = false;
            if (!replace_with_svar.empty()) {
                auto it = svars.find(replace_with_svar);
                if (it != svars.end() && it->second.find("WinsGame") != std::string::npos)
                    replace_with_win = true;
            }
            if (event_is_draw && replace_with_win) {
                Effect::Replacement r;
                r.kind = Effect::Replacement::DRAW_EMPTY_WIN;
                r.applies_to_self_only = false;  // scoped to the controller at the draw site
                result.push_back(r);
            }
        }
        // Quantum Riddler: "As long as you have one or fewer cards in hand, if you would draw one
        // or more cards, you draw that many cards plus one instead." (CR 614.1). A MODIFYING draw
        // replacement — Event$ DrawCards, ValidPlayer$ You (the source controller's own draws),
        // whose ReplaceWith$ SVar is a DB$ Draw with NumCards$ ReplaceCount$Number/Plus.K: the draw
        // count is increased by K. The optional CheckSVar$/SVarCompare$ gate ("one or fewer cards in
        // hand") is carried onto the replacement and re-evaluated for the drawing player at draw
        // time (replacement::draw_count_bonus). General over any additive draw replacement.
        if (event_is_draw_cards && active_zones_battlefield && valid_player_you &&
            !replace_with_svar.empty()) {
            auto it = svars.find(replace_with_svar);
            int add = 0;
            if (it != svars.end() && it->second.find("DB$ Draw") != std::string::npos) {
                // Pull the additive count out of NumCards$ ReplaceCount$Number/Plus.K.
                size_t pp = 0; std::string k, v;
                while (next_param(it->second, pp, k, v)) {
                    if (k != "NumCards") continue;
                    size_t plus = v.find("/Plus.");
                    if (plus != std::string::npos) add = std::stoi(v.substr(plus + 6));
                    break;
                }
            }
            if (add > 0) {
                Effect::Replacement r;
                r.kind = Effect::Replacement::DRAW_ADD;
                r.applies_to_self_only = false;
                r.draw_add = add;
                // Resolve CheckSVar$ <var> to the SVar's Count$ body so the count gate can be
                // evaluated directly (mirrors the trigger-side CheckSVar handling). An absent
                // CheckSVar leaves the gate empty = the additive draw always applies.
                if (!draw_check_svar.empty()) {
                    auto cv = svars.find(draw_check_svar);
                    r.draw_condition_count_expr = (cv != svars.end()) ? cv->second : draw_check_svar;
                    r.draw_condition_compare = draw_svar_compare;
                }
                result.push_back(r);
            }
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