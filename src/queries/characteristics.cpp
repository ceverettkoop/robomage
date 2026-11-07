#include "characteristics.h"

#include "../card_vocab.h"
#include "../classes/game.h"
#include "../components/ability.h"
#include "../components/creature.h"
#include "../components/token.h"
#include "../svar_eval.h"
#include "battlefield.h"
#include "lki.h"
#include "players.h"

static int token_vocab_idx(Entity e);
static int last_known_vocab_idx(Entity e);
static bool named_vocab_idx(Entity e, int *idx);
static void append_ability_defs(std::vector<const AbilityDef *> &out, const std::vector<const AbilityDef *> &defs);
static void append_ability_defs(std::vector<const AbilityDef *> &out, const std::vector<Ability> &abilities);
static std::set<Colors> colors_before_layer5(Entity e);

// The effective_* accessors implement CR 608.2h: use the object's current information while it
// is in the zone it is expected to be in (the battlefield, for a permanent's continuous-effect-
// and counter-modified characteristics), otherwise its last-known information. They live here
// rather than inline in characteristics.h so the header does not need to depend on game.h (and the
// cur_game.last_known_info store).

ObjectCharacteristics printed_characteristics(Entity e, const CardData &cd) {
    const CardData &face = e != 0 ? card_characteristics(e, cd) : cd;
    ObjectCharacteristics c;
    c.types = &face.types;
    c.colors = card_colors(face);
    c.has_pt = true;
    c.power = static_cast<int>(face.power);
    c.toughness = static_cast<int>(face.toughness);
    c.counted_pt = e != 0 && face.has_counted_pt;
    return c;
}

PowerToughness card_power_toughness(Entity e, const CardData &cd) {
    const CardData &face = card_characteristics(e, cd);
    PowerToughness pt{static_cast<int>(face.power), static_cast<int>(face.toughness)};
    if (!face.has_counted_pt) return pt;
    const Zone::Ownership controller = source_controller(e);
    for (const StaticAbility &sa : face.static_abilities) {
        if (!sa.characteristic_defining) continue;
        if (!sa.set_power_svar.empty()) pt.power = evaluate_svar(sa.set_power_svar, controller, e);
        if (!sa.set_toughness_svar.empty()) pt.toughness = evaluate_svar(sa.set_toughness_svar, controller, e);
    }
    return pt;
}

ObjectCharacteristics card_object_characteristics(Entity e, const CardData &cd) {
    ObjectCharacteristics c = printed_characteristics(e, cd);
    if (e != 0) apply_color_statics(e, c.colors);
    return c;
}

ObjectCharacteristics permanent_characteristics(Entity e, const Permanent &perm) {
    ObjectCharacteristics c;
    c.types = &perm.types;
    c.colors = effective_colors(e);
    if (global_coordinator.entity_has_component<Creature>(e)) {
        c.has_pt = true;
        c.power = effective_power(e);
        c.toughness = effective_toughness(e);
    }
    return c;
}

const std::set<Type> &effective_types(Entity e) {
    static const std::set<Type> none;
    if (is_battlefield_permanent(e)) return global_coordinator.GetComponent<Permanent>(e).types;
    if (global_coordinator.entity_has_component<CardData>(e))
        return card_characteristics(e, global_coordinator.GetComponent<CardData>(e)).types;
    if (global_coordinator.entity_has_component<Token>(e)) return global_coordinator.GetComponent<Token>(e).types;
    return none;
}

static void append_ability_defs(std::vector<const AbilityDef *> &out, const std::vector<const AbilityDef *> &defs) {
    out.insert(out.end(), defs.begin(), defs.end());
}

static void append_ability_defs(std::vector<const AbilityDef *> &out, const std::vector<Ability> &abilities) {
    for (const Ability &ab : abilities) out.push_back(ab.def);
}

// Face selection happens here (a transformed permanent reads its back face), so a scan never
// needs a per-ability front-face skip.
std::vector<const AbilityDef *> object_abilities(Entity e) {
    std::vector<const AbilityDef *> defs;
    if (global_coordinator.entity_has_component<Permanent>(e)) {
        const Permanent &perm = global_coordinator.GetComponent<Permanent>(e);
        if (!perm.abilities_removed) {
            if (global_coordinator.entity_has_component<CardData>(e)) {
                const CardData &cd = global_coordinator.GetComponent<CardData>(e);
                append_ability_defs(defs, (perm.transformed && cd.backside) ? cd.backside->abilities : cd.abilities);
            }
            if (global_coordinator.entity_has_component<Token>(e))
                append_ability_defs(defs, global_coordinator.GetComponent<Token>(e).abilities);
        }
        append_ability_defs(defs, perm.abilities);
    } else if (global_coordinator.entity_has_component<CardData>(e)) {
        append_ability_defs(defs, card_characteristics(e, global_coordinator.GetComponent<CardData>(e)).abilities);
    } else if (global_coordinator.entity_has_component<Token>(e)) {
        append_ability_defs(defs, global_coordinator.GetComponent<Token>(e).abilities);
    }
    return defs;
}

const std::vector<Effect::Replacement> &permanent_replacement_effects(Entity e) {
    static const std::vector<Effect::Replacement> none;
    if (global_coordinator.entity_has_component<CardData>(e))
        return active_face(e, global_coordinator.GetComponent<CardData>(e)).replacement_effects;
    if (global_coordinator.entity_has_component<Token>(e))
        return global_coordinator.GetComponent<Token>(e).replacement_effects;
    return none;
}

std::set<Colors> card_colors(const CardData &cd) {
    std::set<Colors> result;
    if (!cd.explicit_colors.empty()) {
        for (Colors c : {WHITE, BLUE, BLACK, RED, GREEN})
            if (cd.explicit_colors.count(c)) result.insert(c);
        return result;
    }
    for (Colors c : {WHITE, BLUE, BLACK, RED, GREEN})
        if (cd.mana_cost.count(c)) result.insert(c);
    // Hybrid pips carry color too (CR 105.2/202.3f): {W/U} makes the card both white and blue,
    // {2/W} makes it white. hybrid_mana is kept out of mana_cost, so fold its colors in here.
    for (const auto &pip : cd.hybrid_mana)
        for (Colors c : pip.colors)
            if (c != COLORLESS && c != GENERIC) result.insert(c);
    // Phyrexian pips carry color too (CR 202.2d): {B/P} makes the card black no matter how the
    // cost is actually paid (mana or life). phyrexian_mana is kept out of mana_cost like hybrid,
    // so fold its colors in here — otherwise a Phyrexian-only cost (Surgical Extraction, {B/P}
    // {B/P}) would read as colorless.
    for (Colors c : cd.phyrexian_mana)
        if (c != COLORLESS && c != GENERIC) result.insert(c);
    return result;
}

int object_mana_value(Entity e, const CardData &cd) {
    if (e != 0 && global_coordinator.entity_has_component<Spell>(e)) {
        const CardData &face = active_face(e, cd);
        return card_mana_value(face) + face.x_pip_count * global_coordinator.GetComponent<Spell>(e).chosen.x;
    }
    if (e != 0 && cd.is_split && cd.backside) return card_mana_value(cd) + card_mana_value(*cd.backside);
    return card_mana_value(cd);
}

int effective_power(Entity e) {
    if (is_battlefield_permanent(e) && global_coordinator.entity_has_component<Creature>(e))
        return static_cast<int>(global_coordinator.GetComponent<Creature>(e).power);
    if (const LastKnownInfo *lki = lki_for(e)) return lki->power;
    if (global_coordinator.entity_has_component<CardData>(e))
        return card_power_toughness(e, global_coordinator.GetComponent<CardData>(e)).power;
    return 0;
}

int effective_toughness(Entity e) {
    if (is_battlefield_permanent(e) && global_coordinator.entity_has_component<Creature>(e))
        return static_cast<int>(global_coordinator.GetComponent<Creature>(e).toughness);
    if (const LastKnownInfo *lki = lki_for(e)) return lki->toughness;
    if (global_coordinator.entity_has_component<CardData>(e))
        return card_power_toughness(e, global_coordinator.GetComponent<CardData>(e)).toughness;
    return 0;
}

std::set<Colors> effective_colors(Entity e) {
    // Layer 5 (613.1e) goes on top of what the object has before it, in every zone a static reaches.
    std::set<Colors> colors = colors_before_layer5(e);
    apply_color_statics(e, colors);
    return colors;
}

static std::set<Colors> colors_before_layer5(Entity e) {
    // On the battlefield, the printed colors of the ACTIVE face — a transformed permanent has its
    // back face's colors (CR 712.8e: Ajani, Nacatl Avenger is red-white via its color indicator, not
    // its mono-white front; a face-up MDFC land back is colorless, not the front spell's color).
    if (is_battlefield_permanent(e) && global_coordinator.entity_has_component<CardData>(e))
        return card_colors(active_face(e, global_coordinator.GetComponent<CardData>(e)));
    if (const LastKnownInfo *lki = lki_for(e)) return lki->colors;
    if (global_coordinator.entity_has_component<CardData>(e))
        return card_colors(card_characteristics(e, global_coordinator.GetComponent<CardData>(e)));
    // A token (no CardData) is colored by its color indicator (Token::explicit_colors, CR 111.4),
    // not a mana cost, so a colored token reads correctly to the .Red/.Blue filter qualifiers (e.g.
    // Blue Elemental Blast's "target red permanent" can see a red Warrior token). Filtered to the
    // five real colors so a stray COLORLESS/GENERIC sentinel never leaks in.
    if (global_coordinator.entity_has_component<Token>(e)) {
        std::set<Colors> cols;
        const auto &tk = global_coordinator.GetComponent<Token>(e).explicit_colors;
        for (Colors c : {WHITE, BLUE, BLACK, RED, GREEN})
            if (tk.count(c)) cols.insert(c);
        return cols;
    }
    return {};
}

// The single shared display-name resolver (contract documented at the declaration in
// characteristics.h). Display-only: never consulted by rules logic, so the " token" tag and
// the ability-entity description cannot affect behavior.
std::string whole_card_name(const CardData &cd) {
    return cd.is_split && cd.backside ? cd.name + " // " + cd.backside->name : cd.name;
}

std::vector<std::string> card_names(const CardData &cd) {
    std::vector<std::string> names = {cd.name};
    if ((cd.is_split || cd.is_adventure || cd.is_prepare) && cd.backside) names.push_back(cd.backside->name);
    return names;
}

std::string entity_name(Entity e) {
    if (global_coordinator.entity_has_component<Permanent>(e)) {
        auto &perm = global_coordinator.GetComponent<Permanent>(e);
        return perm.is_token ? perm.name + " token" : perm.name;
    }
    if (global_coordinator.entity_has_component<CardData>(e))
        return active_face(e, global_coordinator.GetComponent<CardData>(e)).name;
    if (global_coordinator.entity_has_component<Token>(e))
        return global_coordinator.GetComponent<Token>(e).name + " token";
    if (global_coordinator.entity_has_component<Ability>(e)) {
        // A standalone ability entity (an activated/triggered ability on the stack, or a
        // delayed-trigger holder): describe it via its source card when the source still
        // carries a name, else via the ability's effect category.
        const auto &ab = global_coordinator.GetComponent<Ability>(e);
        const Entity src = ab.source.lki_entity();
        if (src != 0 && src != e &&
            (global_coordinator.entity_has_component<Permanent>(src) ||
             global_coordinator.entity_has_component<CardData>(src) ||
             global_coordinator.entity_has_component<Token>(src) ||
             !last_known_name(src).empty()))
            return entity_name(src) + "'s ability";
        if (!ab.def->category.empty()) return ab.def->category + " ability";
        return "an ability";
    }
    std::string lk = last_known_name(e);
    return lk.empty() ? "<unknown>" : lk;
}

// Vocab index for a token entity: its registered token-band index (keyed by the
// token SCRIPT stem — display names collide, stems don't), else TOKEN_SENTINEL.
static int token_vocab_idx(Entity e) {
    if (global_coordinator.entity_has_component<Token>(e)) {
        int idx = token_script_to_index(global_coordinator.GetComponent<Token>(e).script_name);
        if (idx >= 0) return idx;
    }
    return TOKEN_SENTINEL;
}

// Vocab index of an object that left the battlefield and ceased to exist (a token, CR 111.7),
// from its last-known information: a token's token-band index (else TOKEN_SENTINEL), a card's
// name index. -1 when no last-known information was captured.
static int last_known_vocab_idx(Entity e) {
    const LastKnownInfo *lki = lki_for(e);
    if (!lki || lki->name.empty()) return -1;
    if (!lki->is_token) return card_name_to_index(lki->name);
    int idx = token_script_to_index(lki->token_script);
    return idx >= 0 ? idx : TOKEN_SENTINEL;
}

// Sets *idx to the vocab index of a permanent by its current name (a token by its token index) or
// of a card by its CardData name; false when `e` has neither component.
static bool named_vocab_idx(Entity e, int *idx) {
    if (global_coordinator.entity_has_component<Permanent>(e)) {
        auto &perm = global_coordinator.GetComponent<Permanent>(e);
        *idx = perm.is_token ? token_vocab_idx(e) : card_name_to_index(perm.name);
        return true;
    }
    if (!global_coordinator.entity_has_component<CardData>(e)) return false;
    *idx = card_name_to_index(global_coordinator.GetComponent<CardData>(e).name);
    return true;
}

int source_vocab_idx(Entity e) {
    if (e == 0) return -1;
    int idx = -1;
    if (named_vocab_idx(e, &idx)) return idx;
    if (global_coordinator.entity_has_component<Ability>(e)) {
        Entity src = global_coordinator.GetComponent<Ability>(e).source.lki_entity();
        if (named_vocab_idx(src, &idx)) return idx;
        // An ability whose token source has already left play keeps no Permanent/CardData;
        // a lingering Token component still identifies it as a token.
        if (global_coordinator.entity_has_component<Token>(src)) return token_vocab_idx(src);
        // A source token that has ceased to exist entirely: its last-known identity.
        return last_known_vocab_idx(src);
    }
    // An object with no live components left (a token that ceased to exist, CR 111.7) is
    // identified by its last-known information.
    return last_known_vocab_idx(e);
}
