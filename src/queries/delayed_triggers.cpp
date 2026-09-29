#include "delayed_triggers.h"

#include <vector>

#include "../classes/game.h"
#include "../components/player.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../machine_io.h"
#include "../transform.h"
#include "spells.h"

static DelayedTriggerLink::FireKind delayed_fire_kind(const DelayedTrigger &dt);
static Step delayed_fire_step(uint32_t fire_on);
static std::vector<ObjectRef> derive_delayed_subjects(const DelayedTrigger &dt);

static DelayedTriggerLink::FireKind delayed_fire_kind(const DelayedTrigger &dt) {
    if (dt.fire_on_leave_battlefield) return DelayedTriggerLink::FIRE_LEAVES_BATTLEFIELD;
    if (dt.fire_on == Events::UPKEEP_BEGAN) return DelayedTriggerLink::FIRE_UPKEEP;
    if (dt.fire_on == Events::END_STEP_BEGAN) return DelayedTriggerLink::FIRE_END_STEP;
    if (dt.fire_on == Events::END_OF_COMBAT_BEGAN) return DelayedTriggerLink::FIRE_END_OF_COMBAT;
    return DelayedTriggerLink::FIRE_OTHER;
}

// The step whose beginning fires a phase-based delayed trigger's event, or CLEANUP (never
// ahead of any step that fires one) for an event with no step.
static Step delayed_fire_step(uint32_t fire_on) {
    if (fire_on == Events::UPKEEP_BEGAN) return UPKEEP;
    if (fire_on == Events::DRAW_STEP_BEGAN) return DRAW;
    if (fire_on == Events::END_OF_COMBAT_BEGAN) return END_OF_COMBAT;
    if (fire_on == Events::END_STEP_BEGAN) return END_STEP;
    return CLEANUP;
}

static std::vector<ObjectRef> derive_delayed_subjects(const DelayedTrigger &dt) {
    if (!dt.remembered_objects.empty()) return dt.remembered_objects;
    std::vector<ObjectRef> targets;
    for (const ObjectRef &t : dt.ability.targets) {
        const Entity e = t.get();
        if (e != 0 && !global_coordinator.entity_has_component<Player>(e)) targets.push_back(t);
    }
    if (!targets.empty()) return targets;
    if (!dt.ability.restore_remembered_exiled_with.empty())
        return dt.ability.restore_remembered_exiled_with;
    if (!dt.watched.empty()) return {dt.watched};
    return {};
}

void register_delayed_trigger(DelayedTrigger dt, const ObjectRef &creator) {
    stamp_source_transforms(dt.ability);  // CR 701.27f: since the delayed trigger was created
    // CR 107.3n: a delayed trigger created by a resolving spell or ability uses that object's X.
    if (dt.ability.x_paid < 0 && cur_game.resolution.active)
        dt.ability.x_paid = current_x_paid();
    DelayedTriggerLink &link = dt.ability.delayed_link;
    link.seq = cur_game.next_delayed_seq++;
    link.creator = creator;
    link.creator_vocab_idx = action_card_vocab_idx(creator.lki_entity());
    if (link.subjects.empty()) link.subjects = derive_delayed_subjects(dt);
    link.subject_vocab_idx =
        link.subjects.empty() ? -1 : action_card_vocab_idx(link.subjects[0].lki_entity());
    link.fire_kind = delayed_fire_kind(dt);
    cur_game.delayed_triggers.push_back(std::move(dt));
}

bool is_waiting_delayed_trigger_subject(Entity e) {
    if (e == 0) return false;
    for (const auto &dt : cur_game.delayed_triggers) {
        if (dt.watched.get() == e) return true;
        if (refs_contain(dt.ability.delayed_link.subjects, e)) return true;
    }
    return false;
}

bool delayed_trigger_fires_this_turn(const DelayedTrigger &dt) {
    if (dt.fire_on_leave_battlefield) return false;
    if (dt.fire_on_turn > cur_game.turn_state.turn) return false;
    Entity active = cur_game.turn_state.player_a_turn ? cur_game.player_a_entity : cur_game.player_b_entity;
    if (dt.restrict_player != 0 && dt.restrict_player != active) return false;
    return cur_game.turn_state.step < delayed_fire_step(dt.fire_on);
}
