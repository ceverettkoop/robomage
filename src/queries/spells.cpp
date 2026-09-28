#include "spells.h"

#include "../classes/game.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "battlefield.h"
#include "characteristics.h"
#include "filters.h"

static bool unfiltered_counter_protection_covers(const Effect::Replacement &r,
                                                 Zone::Ownership source_ctrl,
                                                 Zone::Ownership player);

std::string spell_additional_sac_spec(const CardData &cd) {
    for (const auto &ab : cd.abilities)
        if (ab->ability_type == AbilityDef::SPELL && !ab->sac_cost_spec.empty())
            return ab->sac_cost_spec;
    return "";
}

bool spell_has_variable_life_cost(const CardData &cd) {
    for (const auto &ab : cd.abilities)
        if (ab->ability_type == AbilityDef::SPELL && ab->life_cost_is_x)
            return true;
    return false;
}

int current_x_paid() {
    if (cur_game.pending.cast.active) return cur_game.pending.cast.x_paid;
    if (cur_game.pending.activation.active) {
        const int x = cur_game.pending.activation.stack_ab.x_paid;
        return x < 0 ? 0 : x;
    }
    return cur_game.resolution.active ? cur_game.resolution.x_paid : 0;
}

int current_converge() { return cur_game.resolution.active ? cur_game.resolution.converge : 0; }

bool current_gift_promised() { return cur_game.pending.cast.gift_promised; }

bool spell_uncounterable_by_static(Entity spell, const std::set<Entity> &entities) {
    if (!global_coordinator.entity_has_component<CardData>(spell)) return false;
    Zone::Ownership spell_ctrl = global_coordinator.entity_has_component<Spell>(spell)
                                     ? global_coordinator.GetComponent<Spell>(spell).caster
                                     : Zone::UNKNOWN;
    for (auto e : battlefield_permanents(entities)) {
        Zone::Ownership perm_ctrl = global_coordinator.GetComponent<Permanent>(e).controller;
        for (const auto &r : permanent_replacement_effects(e)) {
            if (r.kind != Effect::Replacement::CANT_BE_COUNTERED || !r.from_battlefield) continue;
            // Controller scope is read from the spell's caster (the matcher can't read it for a
            // stack object). YouCtrl → caster is the source's controller; OppCtrl → it isn't.
            if (r.valid_sa_filter.find("YouCtrl") != std::string::npos) {
                if (spell_ctrl != perm_ctrl) continue;
            } else if (r.valid_sa_filter.find("OppCtrl") != std::string::npos) {
                if (spell_ctrl == perm_ctrl || spell_ctrl == Zone::UNKNOWN) continue;
            }
            // Any remaining type/characteristic qualifiers on the filter (head "Spell", color,
            // type) are checked against the spell's printed characteristics.
            MatchCtx ctx;
            ctx.controller = perm_ctrl;
            if (card_matches_filter(spell, r.valid_sa_filter, ctx)) return true;
        }
    }
    return false;
}

// True if `r` is a battlefield CANT_BE_COUNTERED replacement whose ValidSA$ filter is the bare
// controller-scoped spell filter covering every spell `player` controls, given that the
// replacement's source is controlled by `source_ctrl`.
static bool unfiltered_counter_protection_covers(const Effect::Replacement &r,
                                                 Zone::Ownership source_ctrl,
                                                 Zone::Ownership player) {
    if (r.kind != Effect::Replacement::CANT_BE_COUNTERED || !r.from_battlefield) return false;
    if (r.valid_sa_filter == "Spell.YouCtrl") return source_ctrl == player;
    if (r.valid_sa_filter == "Spell.OppCtrl") return source_ctrl != player;
    return false;
}

bool player_spells_cant_be_countered(Zone::Ownership player, const std::set<Entity> &entities) {
    if (cur_game.resolved_effects.cant_counter_spells_of.count(player) > 0) return true;
    for (auto e : battlefield_permanents(entities)) {
        Zone::Ownership ctrl = global_coordinator.GetComponent<Permanent>(e).controller;
        for (const auto &r : permanent_replacement_effects(e))
            if (unfiltered_counter_protection_covers(r, ctrl, player)) return true;
    }
    return false;
}
