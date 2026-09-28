#include "ability.h"

#include <deque>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "../ecs/coordinator.h"
#include "player.h"

extern Coordinator global_coordinator;

// The process-wide definition store behind intern_ability_def (see ability.h).
static std::deque<AbilityDef> &ability_def_store();

// edge case of two identical abilities being applied from two sources not handled
bool Ability::identical_activated_ability(const AbilityDef *other) const {
    if (other == def) return true;
    if (other->category != def->category) return false;
    if (other->valid_tgts != def->valid_tgts) return false;
    if (other->amount != def->amount) return false;
    // A Metalcraft-/condition-gated ability and an otherwise-identical ungated one are
    // distinct abilities (Urza's Workshop: plain "{T}: Add {C}" vs the Metalcraft
    // "{T}: Add {C} for each Urza's land"). Likewise two mana abilities that produce
    // different dynamic amounts are distinct. Without these the second is wrongly deduped.
    if (other->activation_condition != def->activation_condition) return false;
    if (other->dynamic_amount_expr != def->dynamic_amount_expr) return false;
    if (other->tap_cost != def->tap_cost) return false;
    if (other->activation_mana_cost != def->activation_mana_cost) return false;
    if (other->sac_self != def->sac_self) return false;
    if (other->change_type != def->change_type) return false;
    if (other->origin != def->origin) return false;
    if (other->origin_any != def->origin_any) return false;
    if (other->destination != def->destination) return false;
    if (other->color != def->color) return false;
    if (other->mana_choices != def->mana_choices) return false;
    if (other->reflected_mana_filter != def->reflected_mana_filter) return false;
    if (other->restrict_to_chosen_type_creature != def->restrict_to_chosen_type_creature) return false;
    if (other->restrict_to_creature != def->restrict_to_creature) return false;
    if (other->adds_no_counter != def->adds_no_counter) return false;
    return true;
}

Ability::Ability(const AbilityDef *d)
    : def(d), target_min(d->target_min), target_max(d->target_max), color(d->color) {
    subabilities.reserve(d->subabilities.size());
    for (const AbilityDef &sub : d->subabilities) subabilities.emplace_back(&sub);
    charm_choices.reserve(d->charm_choices.size());
    for (const AbilityDef &mode : d->charm_choices) charm_choices.emplace_back(&mode);
}

// The definition store (see ability.h). A deque never moves its elements, so the pointers it
// hands out stay valid as it grows.
static std::deque<AbilityDef> &ability_def_store() {
    static std::deque<AbilityDef> store;
    return store;
}

const AbilityDef *intern_ability_def(AbilityDef def) {
    ability_def_store().push_back(std::move(def));
    return &ability_def_store().back();
}

std::vector<const AbilityDef *> intern_ability_defs(std::vector<AbilityDef> defs) {
    std::vector<const AbilityDef *> out;
    out.reserve(defs.size());
    for (AbilityDef &d : defs) out.push_back(intern_ability_def(std::move(d)));
    return out;
}

const AbilityDef *keyed_ability_def(const std::string &key,
                                    const std::function<AbilityDef()> &build) {
    static std::unordered_map<std::string, const AbilityDef *> by_key;
    auto it = by_key.find(key);
    if (it != by_key.end()) return it->second;
    const AbilityDef *d = intern_ability_def(build());
    by_key.emplace(key, d);
    return d;
}

const AbilityDef *derived_ability_def(const AbilityDef *base, const std::string &variant, int value,
                                      const std::function<void(AbilityDef &)> &edit) {
    static std::map<std::tuple<const AbilityDef *, std::string, int>, const AbilityDef *> derived;
    auto key = std::make_tuple(base, variant, value);
    auto it = derived.find(key);
    if (it != derived.end()) return it->second;
    AbilityDef copy = *base;
    edit(copy);
    const AbilityDef *d = intern_ability_def(std::move(copy));
    derived.emplace(key, d);
    return d;
}

const AbilityDef *triggered_effect_def(const std::string &category) {
    return keyed_ability_def("triggered:" + category, [&category] {
        AbilityDef d;
        d.ability_type = AbilityDef::TRIGGERED;
        d.category = category;
        return d;
    });
}

const AbilityDef *draw_one_trigger_def() {
    return keyed_ability_def("draw_one", [] {
        AbilityDef d;
        d.ability_type = AbilityDef::TRIGGERED;
        d.category = "Draw";
        d.amount = 1;
        return d;
    });
}

const AbilityDef *blank_ability_def() {
    static const AbilityDef *blank = intern_ability_def(AbilityDef{});
    return blank;
}

Entity Ability::player_target_for_subs() const {
    const Entity t = target.get();
    return (t != 0 && global_coordinator.entity_has_component<Player>(t)) ? t : targeted_player;
}
