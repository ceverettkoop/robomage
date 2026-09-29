#include "effects.h"

#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../queries/battlefield.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static std::vector<Entity> attached_to(Entity host, bool phased_out_indirectly,
                                       const std::set<Entity> &entities);
static void remove_from_combat(Entity e, const std::set<Entity> &entities);
static void phase_out_as(Entity e, bool indirectly, const std::set<Entity> &entities);

// The battlefield Auras/Equipment attached to `host` (Permanent::equipped_to) whose phased-out
// state is `phased_out_indirectly`: the attachments still phased in when `host` phases out, or
// the ones that phased out along with it when `host` phases in.
static std::vector<Entity> attached_to(Entity host, bool phased_out_indirectly,
                                       const std::set<Entity> &entities) {
    std::vector<Entity> out;
    for (auto e : entities) {
        if (e == host || !global_coordinator.entity_has_component<Permanent>(e)) continue;
        if (global_coordinator.GetComponent<Zone>(e).location != Zone::BATTLEFIELD) continue;
        const auto &perm = global_coordinator.GetComponent<Permanent>(e);
        if (perm.equipped_to.get() != host) continue;
        bool indirect = perm.is_phased_out && perm.phased_out_indirectly;
        if (phased_out_indirectly ? indirect : !perm.is_phased_out) out.push_back(e);
    }
    return out;
}

// CR 506.4: a creature removed from combat stops being an attacking or blocking creature. An
// attacker it was blocking stays blocked (CR 509.1h), and a creature blocking it stays a blocking
// creature that blocks nothing. A planeswalker removed from combat stops being attacked; its
// attackers keep attacking and, if unblocked, deal no combat damage (CR 506.4c).
static void remove_from_combat(Entity e, const std::set<Entity> &entities) {
    cur_game.combat.damage_assignment.erase(e);
    if (!global_coordinator.entity_has_component<Creature>(e)) return;
    auto &cr = global_coordinator.GetComponent<Creature>(e);
    bool was_attacking = cr.is_attacking;
    cr.is_attacking = false;
    cr.attack_target = ObjectRef{};
    cr.is_blocked = false;
    cr.is_blocking = false;
    cr.blocking_target = ObjectRef{};
    if (!was_attacking) return;
    for (auto b : entities) {
        if (!global_coordinator.entity_has_component<Creature>(b)) continue;
        auto &bcr = global_coordinator.GetComponent<Creature>(b);
        if (bcr.is_blocking && bcr.blocking_target.get() == e) bcr.blocking_target = ObjectRef{};
    }
}

// CR 702.26b/g: phase `e` out (directly, or indirectly with the permanent it is attached to),
// remove it from combat, and phase out its own attachments indirectly.
static void phase_out_as(Entity e, bool indirectly, const std::set<Entity> &entities) {
    auto &perm = global_coordinator.GetComponent<Permanent>(e);
    perm.is_phased_out = true;
    perm.phased_out_indirectly = indirectly;
    remove_from_combat(e, entities);
    game_log("%s phases out\n", perm.name.c_str());
    for (auto att : attached_to(e, false, entities)) phase_out_as(att, true, entities);
}

namespace effects {

void phase_out(Entity e, const std::set<Entity> &entities) {
    if (!is_battlefield_permanent(e)) return;
    phase_out_as(e, false, entities);
}

void phase_in(Entity e, const std::set<Entity> &entities) {
    auto &perm = global_coordinator.GetComponent<Permanent>(e);
    perm.is_phased_out = false;
    perm.phased_out_indirectly = false;
    game_log("%s phases in\n", perm.name.c_str());
    for (auto att : attached_to(e, true, entities)) phase_in(att, entities);
}

HandlerResult phases(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)ctx;
    if (Entity t = ab.target.get()) phase_out(t, orderer->mEntities);
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
