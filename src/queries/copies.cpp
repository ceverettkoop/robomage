#include "copies.h"

#include <memory>
#include <vector>

#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../coverage.h"
#include "../ecs/coordinator.h"
#include "characteristics.h"
#include "keywords.h"
#include "types.h"

template <typename Values>
static void apply_copy_overlay(Values &values, const CopyOverlay &overlay);

// What `overlay` sets on the copied values: a token's or a card's.
template <typename Values>
static void apply_copy_overlay(Values &values, const CopyOverlay &overlay) {
    if (overlay.sets_pt) {
        values.power = overlay.power;
        values.toughness = overlay.toughness;
    }
    if (!overlay.add_keywords.empty()) coverage_row("copy_add_keywords");
    for (const std::string &kw : overlay.add_keywords) add_keyword_instance(values.keywords, kw);
    values.types.insert(overlay.add_types.begin(), overlay.add_types.end());
    if (!overlay.set_creature_types.empty()) {
        coverage_row("copy_set_creature_types");
        for (auto t = values.types.begin(); t != values.types.end();)
            t = t->kind == SUBTYPE && is_creature_subtype(t->name) ? values.types.erase(t) : std::next(t);
        values.types.insert(overlay.set_creature_types.begin(), overlay.set_creature_types.end());
    }
    if (!overlay.set_colors.empty()) values.explicit_colors = overlay.set_colors;
}

Token copied_token(Entity original, const CopyOverlay &overlay) {
    Token tok;
    if (global_coordinator.entity_has_component<Token>(original)) {
        tok = global_coordinator.GetComponent<Token>(original);
        apply_copy_overlay(tok, overlay);
        return tok;
    }
    if (!global_coordinator.entity_has_component<CardData>(original)) return tok;
    const CardData &card = global_coordinator.GetComponent<CardData>(original);
    bool transformed = global_coordinator.entity_has_component<Permanent>(original) &&
                       global_coordinator.GetComponent<Permanent>(original).transformed;
    // Being bestowed isn't a copiable value.
    const CardData &face = transformed ? active_face(original, card) : card;
    tok.name = face.name;
    tok.copied_card = face.name;
    tok.types = face.types;
    tok.explicit_colors = card_colors(face);
    tok.mana_value = card_mana_value(mana_value_face(card, transformed));
    tok.power = face.power;
    tok.toughness = face.toughness;
    tok.keywords = face.keywords;
    tok.abilities = face.abilities;
    tok.static_abilities = face.static_abilities;
    tok.replacement_effects = face.replacement_effects;
    if (card.is_prepare) tok.prepare_spell = card.backside;
    tok.enters_prepared = face.enters_prepared;
    apply_copy_overlay(tok, overlay);
    return tok;
}

void become_copy(Entity permanent, Entity original, const CopyOverlay &overlay) {
    // Captured BEFORE the overwrite, so the copy keeps "…except it has this ability." and can
    // re-clone later.
    std::vector<const AbilityDef *> retained;
    if (overlay.gains_this_ability) {
        for (const auto &a : global_coordinator.GetComponent<CardData>(permanent).abilities)
            if (a->kind == EffectKind::Clone) retained.push_back(a);
    }

    // Build the copy from the original's copiable characteristics (a value copy of its CardData),
    // then re-add the retained Clone ability(ies).
    CardData new_cd = global_coordinator.GetComponent<CardData>(original);
    apply_copy_overlay(new_cd, overlay);
    for (const auto &a : retained) new_cd.abilities.push_back(a);

    auto &perm = global_coordinator.GetComponent<Permanent>(permanent);
    if (!perm.printed_card)
        perm.printed_card =
            std::make_shared<const CardData>(global_coordinator.GetComponent<CardData>(permanent));
    global_coordinator.GetComponent<CardData>(permanent) = new_cd;
    std::string old_name = perm.name;
    perm.name = new_cd.name;
    perm.types = new_cd.types;
    perm.abilities.clear();         // re-derived from the new CardData next SBA pass
    perm.static_abilities.clear();  // re-derived from the new CardData next SBA pass
    game_log("%s becomes a copy of %s.\n", old_name.c_str(), new_cd.name.c_str());
}

void end_copy_effect(Entity permanent) {
    auto &perm = global_coordinator.GetComponent<Permanent>(permanent);
    if (!perm.printed_card) return;
    if (global_coordinator.entity_has_component<CardData>(permanent))
        global_coordinator.GetComponent<CardData>(permanent) = *perm.printed_card;
    perm.name = perm.printed_card->name;
    perm.types = perm.printed_card->types;
    perm.abilities.clear();
    perm.static_abilities.clear();
    perm.printed_card.reset();
}
