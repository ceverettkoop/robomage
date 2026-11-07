#include "effects.h"

#include <vector>

#include "../action_processor.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/entry_info.h"
#include "../ecs/coordinator.h"
#include "../queries/characteristics.h"
#include "../queries/entry.h"
#include "../queries/players.h"
#include "../systems/state_manager_internal.h"
#include "../target_gate.h"
#include "../targeting.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Where the one target of a stack object sits: on an instruction, or on an Aura spell's entry.
struct SingleTarget {
    Ability *holder = nullptr;
    EntryInfo *entry = nullptr;
};

static SingleTarget single_target_of(Entity object);
// The target as a pick reads it, its ability's or the Aura's enchant ability.
static Ability pick_view(const SingleTarget &slot, Entity object);
static void set_target(const SingleTarget &slot, const ObjectRef &target);

static SingleTarget single_target_of(Entity object) {
    SingleTarget slot;
    if (global_coordinator.entity_has_component<Ability>(object))
        for (Ability *part : announced_instructions(global_coordinator.GetComponent<Ability>(object)))
            if (part->def->valid_tgts != "N_A" && (!part->target.empty() || !part->targets.empty())) slot.holder = part;
    if (!slot.holder) slot.entry = find_entry_info(object);
    return slot;
}

static Ability pick_view(const SingleTarget &slot, Entity object) {
    if (slot.holder) return *slot.holder;
    const CardData &card = global_coordinator.GetComponent<CardData>(object);
    Ability enchant = enchant_target_ability(object, active_face(object, card), source_controller(object));
    enchant.target = slot.entry->aura_target;
    return enchant;
}

static void set_target(const SingleTarget &slot, const ObjectRef &target) {
    if (!slot.holder) {
        slot.entry->aura_target = target;
        return;
    }
    slot.holder->target = target;
    if (!slot.holder->targets.empty()) slot.holder->targets[0] = target;
}

// DB$ ChangeTargets (Untimely Malfunction): "change the target of target spell or ability with a
// single target". As this resolves its controller changes that target to another legal target of
// the object, through the object's own legality and the pick's rows, and with none to change it to
// the target stays (CR 115.7a), illegal or not. Its division is unchanged (CR 115.7f) and its modes
// too (CR 115.8). A new target has become its target: Ward and becomes-target abilities see it.
HandlerResult change_targets(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    const Entity object = ab.target.get();
    if (object == 0 || chosen_target_count(object) != 1) return HandlerResult::DONE_RUN_SUBS;
    const SingleTarget slot = single_target_of(object);
    Ability view = pick_view(slot, object);
    view.target_min = 1;
    const ObjectRef was = view.targets.empty() ? view.target : view.targets[0];
    TargetPickGate gate;
    gate.changing_from = was.lki_entity();
    gate.changing_object = object;
    std::vector<Entity> pool = build_valid_targets(view, orderer, view.controller);
    apply_target_gate(pool, TargetPick{view, 0, &gate, orderer, view.controller});
    if (pool.empty()) {
        game_log("%s has no other legal target; its target is unchanged\n", entity_name(object).c_str());
        return HandlerResult::DONE_RUN_SUBS;
    }
    ResolutionTargetAsker asker(ctx, ab.controller);
    if (select_single_target(view, pool, false, asker) < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
    set_target(slot, view.target);
    game_log("%s now targets %s\n", entity_name(object).c_str(),
             target_display_name(cur_game, view.target.lki_entity()).c_str());
    fire_targeting_hooks(object, view.controller);
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
