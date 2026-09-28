#include "characteristics.h"

#include "../classes/game.h"
#include "../components/ability.h"
#include "../components/creature.h"
#include "../components/token.h"
#include "battlefield.h"
#include "lki.h"

// The effective_* accessors implement CR 608.2h: use the object's current information while it
// is in the zone it is expected to be in (the battlefield, for a permanent's continuous-effect-
// and counter-modified characteristics), otherwise its last-known information. They live here
// rather than inline in characteristics.h so the header does not need to depend on game.h (and the
// cur_game.last_known_info store).

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
        return card_mana_value(face) + face.x_pip_count * global_coordinator.GetComponent<Spell>(e).x_paid;
    }
    if (e != 0 && cd.is_split && cd.backside) return card_mana_value(cd) + card_mana_value(*cd.backside);
    return card_mana_value(cd);
}

int effective_power(Entity e) {
    if (is_battlefield_permanent(e) && global_coordinator.entity_has_component<Creature>(e))
        return static_cast<int>(global_coordinator.GetComponent<Creature>(e).power);
    if (const LastKnownInfo *lki = lki_for(e)) return lki->power;
    if (global_coordinator.entity_has_component<CardData>(e))
        return static_cast<int>(global_coordinator.GetComponent<CardData>(e).power);
    return 0;
}

int effective_toughness(Entity e) {
    if (is_battlefield_permanent(e) && global_coordinator.entity_has_component<Creature>(e))
        return static_cast<int>(global_coordinator.GetComponent<Creature>(e).toughness);
    if (const LastKnownInfo *lki = lki_for(e)) return lki->toughness;
    if (global_coordinator.entity_has_component<CardData>(e))
        return static_cast<int>(global_coordinator.GetComponent<CardData>(e).toughness);
    return 0;
}

std::set<Colors> effective_colors(Entity e) {
    // Layer-5 (613.1e) global color-changing override (Mycosynth Lattice: every card colorless).
    // Checked first so it wins over the printed/last-known color in every zone the static reaches.
    std::set<Colors> override_colors;
    if (setcolor_override_for(e, override_colors)) return override_colors;
    // On the battlefield, effective color is the layer-5 (613.1e) result. With no active color-
    // changing static this reads the printed colors of the ACTIVE face — a transformed permanent
    // has its back face's colors (CR 712.8e: Ajani, Nacatl Avenger is red-white via its color
    // indicator, not its mono-white front; a face-up MDFC land back is colorless, not the front
    // spell's color). The override above is the single seam a color-changing static plugs into
    // without touching any consumer.
    if (is_battlefield_permanent(e) && global_coordinator.entity_has_component<CardData>(e))
        return card_colors(active_face(e, global_coordinator.GetComponent<CardData>(e)));
    if (const LastKnownInfo *lki = lki_for(e)) return lki->colors;
    if (global_coordinator.entity_has_component<CardData>(e))
        return card_colors(global_coordinator.GetComponent<CardData>(e));
    // A token (no CardData) is colored by its color indicator (Token::explicit_colors, CR 111.4),
    // not a mana cost. With no active SetColor override (checked first, above) this is its
    // effective color — so a colored token reads correctly to the .Red/.Blue filter qualifiers
    // (e.g. Blue Elemental Blast's "target red permanent" can see a red Warrior token). Filtered
    // to the five real colors so a stray COLORLESS/GENERIC sentinel never leaks in.
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
std::string entity_name(Entity e) {
    if (global_coordinator.entity_has_component<Permanent>(e)) {
        auto &perm = global_coordinator.GetComponent<Permanent>(e);
        return perm.is_token ? perm.name + " token" : perm.name;
    }
    if (global_coordinator.entity_has_component<CardData>(e))
        return global_coordinator.GetComponent<CardData>(e).name;
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
