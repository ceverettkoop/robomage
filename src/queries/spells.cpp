#include "spells.h"

#include <algorithm>

#include "../classes/game.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../coverage.h"
#include "../svar_eval.h"
#include "battlefield.h"
#include "characteristics.h"
#include "filters.h"
#include "overlays.h"

static bool unfiltered_counter_protection_covers(const Effect::Replacement &r,
                                                 Zone::Ownership source_ctrl,
                                                 Zone::Ownership player);

SpellDestination spell_destination(Entity spell, Zone::ZoneValue proposed, SpellExit exit) {
    SpellDestination to;
    to.zone = proposed;
    if (exit == SpellExit::RESOLVED && global_coordinator.entity_has_component<CardData>(spell) &&
        global_coordinator.GetComponent<CardData>(spell).shuffle_into_library) {
        to.zone = Zone::LIBRARY;
        to.shuffled = true;
    } else if (global_coordinator.entity_has_component<Spell>(spell)) {
        const CastOption &how = global_coordinator.GetComponent<Spell>(spell).chosen.option;
        const SpellOverlayRow &overlay = spell_overlay(how.overlay);
        if (how.destination != Zone::GRAVEYARD) {
            to.zone = how.destination;
            to.by_cast = how.row;
        } else if (exit == SpellExit::RESOLVED && overlay.resolved_to != Zone::GRAVEYARD) {
            to.zone = overlay.resolved_to;
            to.by_cast = overlay.name;
        }
        if (to.zone == Zone::GRAVEYARD && spell_graveyard_replaced(spell)) {
            to.zone = std::get<PlayParams>(how.cast_by->params).graveyard_replaced_by;
            to.by_cast = "cast this way";
            to.graveyard_replaced = true;
            coverage_row("graveyard_replaced");
        }
    }
    return to;
}

bool spell_graveyard_replaced(Entity spell) {
    const AbilityDef *cast_by = global_coordinator.GetComponent<Spell>(spell).chosen.option.cast_by;
    if (cast_by == nullptr) return false;
    const PlayParams &pp = std::get<PlayParams>(cast_by->params);
    return pp.replaces_graveyard && card_matches_filter(spell, pp.graveyard_replaced_valid);
}

bool spell_bargained(Entity spell) {
    if (!global_coordinator.entity_has_component<Spell>(spell) ||
        !global_coordinator.entity_has_component<CardData>(spell) ||
        !active_face(spell, global_coordinator.GetComponent<CardData>(spell)).has_bargain)
        return false;
    const std::vector<ChosenCost::Moved> &moved = global_coordinator.GetComponent<Spell>(spell).chosen.moved;
    return std::any_of(moved.begin(), moved.end(),
                       [](const ChosenCost::Moved &m) { return m.kind == CostPart::SACRIFICE; });
}

FilterSpec spell_additional_sac_spec(const CardData &cd) {
    for (const auto &ab : cd.abilities)
        if (ab->ability_type == AbilityDef::SPELL && !ab->cost.filter(CostPart::SACRIFICE).empty())
            return ab->cost.filter(CostPart::SACRIFICE);
    return FilterSpec();
}

bool spell_reads_colors_spent(const CardData &cd) {
    if (std::find(cd.keywords.begin(), cd.keywords.end(), "Sunburst") != cd.keywords.end()) return true;
    for (const AbilityDef *def : cd.abilities)
        if (def->condition_present.text().rfind("Count$Adamant_", 0) == 0) return true;
    return false;
}

const CostParts &spell_additional_cost(const CardData &cd) {
    static const CostParts kNoCost;
    for (const auto &ab : cd.abilities)
        if (ab->ability_type == AbilityDef::SPELL && !ab->cost.filter(CostPart::SACRIFICE).empty()) return ab->cost;
    return kNoCost;
}

bool spell_has_variable_life_cost(const CardData &cd) {
    for (const auto &ab : cd.abilities)
        if (ab->ability_type == AbilityDef::SPELL && ab->cost.life_is_x())
            return true;
    return false;
}

int current_x_paid() {
    if (cur_game.pending.cast.active) return cur_game.pending.cast.chosen.x;
    if (cur_game.pending.activation.active) {
        const int x = cur_game.pending.activation.stack_ab.x_paid;
        return x < 0 ? 0 : x;
    }
    return cur_game.resolution.active ? cur_game.resolution.x_paid : 0;
}

int current_converge() { return cur_game.resolution.active ? cur_game.resolution.converge : 0; }

bool current_gift_promised() { return cur_game.pending.cast.chosen.gift_promised; }

const Ability &stack_object_ability(Entity stack_object) {
    const ResolutionFrame &fr = cur_game.resolution;
    if (fr.active && fr.stack_entity == stack_object && !fr.levels.empty())
        return fr.levels.front().work;
    return global_coordinator.GetComponent<Ability>(stack_object);
}

// True if `r` is a continuous CANT_BE_COUNTERED replacement whose ValidSA$ filter is the bare
// controller-scoped spell filter covering every spell `player` controls, given that the
// replacement's source is controlled by `source_ctrl`.
static bool unfiltered_counter_protection_covers(const Effect::Replacement &r,
                                                 Zone::Ownership source_ctrl,
                                                 Zone::Ownership player) {
    if (r.kind != Effect::Replacement::CANT_BE_COUNTERED || r.applies_to_self_only || r.protects_remembered) return false;
    if (r.valid_sa_filter == "Spell.YouCtrl") return source_ctrl == player;
    if (r.valid_sa_filter == "Spell.OppCtrl") return source_ctrl != player;
    return false;
}

bool counter_replacement_covers(const Effect::Replacement &r, Zone::Ownership source_ctrl, Entity object,
                                Zone::Ownership object_ctrl) {
    if (global_coordinator.entity_has_component<Spell>(object) &&
        unfiltered_counter_protection_covers(r, source_ctrl, object_ctrl))
        return true;
    if (!r.from_battlefield || !global_coordinator.entity_has_component<CardData>(object)) return false;
    Zone::Ownership spell_ctrl = global_coordinator.entity_has_component<Spell>(object)
                                     ? global_coordinator.GetComponent<Spell>(object).caster
                                     : Zone::UNKNOWN;
    // Controller scope is read from the spell's caster (the matcher can't read it for a
    // stack object). YouCtrl → caster is the source's controller; OppCtrl → it isn't.
    if (r.valid_sa_filter.text().find("YouCtrl") != std::string::npos) {
        if (spell_ctrl != source_ctrl) return false;
    } else if (r.valid_sa_filter.text().find("OppCtrl") != std::string::npos) {
        if (spell_ctrl == source_ctrl || spell_ctrl == Zone::UNKNOWN) return false;
    }
    // Any remaining type/characteristic qualifiers on the filter (head "Spell", color,
    // type) are checked against the spell's printed characteristics.
    MatchCtx ctx;
    ctx.controller = source_ctrl;
    return card_matches_filter(object, r.valid_sa_filter, ctx);
}

// Map a Forge PresentZone$ token to its zone value (only the non-battlefield zones a spell-
// mastery-style condition reads by printed characteristics). Defaults to the graveyard, which
// is what every "spell mastery" card counts.
static Zone::ZoneValue present_zone_value(const std::string &zone) {
    if (zone == "Exile")   return Zone::EXILE;
    if (zone == "Hand")    return Zone::HAND;
    if (zone == "Library") return Zone::LIBRARY;
    return Zone::GRAVEYARD;
}

bool spell_uncounterable_by_own_condition(Entity spell, const EntitySet &entities) {
    if (!global_coordinator.entity_has_component<CardData>(spell)) return false;
    if (!global_coordinator.entity_has_component<Spell>(spell)) return false;
    Zone::Ownership caster = global_coordinator.GetComponent<Spell>(spell).caster;
    if (caster == Zone::UNKNOWN) return false;
    const auto &cd = global_coordinator.GetComponent<CardData>(spell);
    for (const auto &r : cd.replacement_effects) {
        if (r.kind != Effect::Replacement::CANT_BE_COUNTERED || r.from_battlefield) continue;
        if (r.cant_counter_present.empty()) continue;  // unconditional self form is a cast-time stamp
        Zone::ZoneValue zone = present_zone_value(r.cant_counter_zone);
        MatchCtx mctx;
        mctx.controller = caster;  // the "you" reference for the filter's YouOwn qualifier
        int count = 0;
        for (auto e : entities) {
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            const auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location != zone || z.owner != caster) continue;
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            if (card_matches_filter(e, r.cant_counter_present, mctx)) count++;  // ',' = OR over filters
        }
        if (compare_svar(count, r.cant_counter_compare.empty() ? "GE1" : r.cant_counter_compare))
            return true;
    }
    return false;
}

bool effect_object_protects_spells_of(Zone::Ownership player) {
    for (const EffectObject &object : cur_game.resolved_effects.effect_objects)
        for (const Effect::Replacement &r : object.replacements)
            if (unfiltered_counter_protection_covers(r, object.controller, player)) return true;
    return false;
}

bool player_spells_cant_be_countered(Zone::Ownership player, const EntitySet &entities) {
    if (effect_object_protects_spells_of(player)) return true;
    for (auto e : battlefield_permanents(entities)) {
        const Permanent &perm = global_coordinator.GetComponent<Permanent>(e);
        if (perm.abilities_removed) continue;  // CR 613.1f: it generates no replacement effects
        for (const auto &r : permanent_replacement_effects(e))
            if (unfiltered_counter_protection_covers(r, perm.controller, player)) return true;
    }
    return false;
}
