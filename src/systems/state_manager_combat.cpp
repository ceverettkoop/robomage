#include "state_manager.h"
#include "state_manager_internal.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "../action_processor.h"
#include "../card_vocab.h"
#include "../classes/game.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/static_ability.h"
#include "../components/damage.h"
#include "../components/effect.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/token.h"
#include "../components/types.h"
#include "../type_constants.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../cli_output.h"
#include "../game_queries.h"
#include "../input_logger.h"
#include "../mana_system.h"
#include "../svar_eval.h"
#include "../systems/stack_manager.h"
#include "orderer.h"

// Should this creature deal damage during this combat damage step?
bool should_deal_damage(const Creature &cr, bool first_strike_only) {
    bool has_fs = creature_has_keyword(cr, "First Strike");
    bool has_ds = creature_has_keyword(cr, "Double Strike");
    if (first_strike_only) return creature_deals_first_strike_damage(cr);
    // Regular damage step: skip first-strikers (they already dealt), but double strikers hit again
    if (has_fs && !has_ds) return false;
    return true;
}

// Combat damage is a turn-based action (rule 510.2), not a state-based action
void StateManager::deal_combat_damage(Game &game, bool first_strike_only) {
    game_log("\n--- %sCombat Damage ---\n", first_strike_only ? "First Strike " : "");

    for (auto entity : mEntities) {
        if (!is_attacking_creature(entity)) continue;
        auto &cr = global_coordinator.GetComponent<Creature>(entity);
        // CR 510.1/702.7b: each creature deals damage in the step its own first/double
        // strike status dictates — attackers AND blockers independently. Visit every
        // attacker's combat in BOTH damage steps and gate each creature's damage on its
        // own should_deal_damage; skipping the whole combat when the attacker doesn't
        // deal this step would drop a first-striking BLOCKER's first-strike-step damage.
        bool attacker_deals = should_deal_damage(cr, first_strike_only);

        std::vector<Entity> blockers = blockers_of(entity, mEntities);
        // CR 506.4c: an attacker whose planeswalker was removed from combat (it left or phased
        // out) keeps attacking but, unblocked, deals no combat damage; no trample damage either.
        bool target_in_combat = global_coordinator.entity_has_component<Player>(cr.attack_target) ||
                                is_battlefield_permanent(cr.attack_target);

        if (!cr.is_blocked) {
            // Unblocked — deal damage to attack target
            if (attacker_deals && target_in_combat)
                deal_damage(entity, cr.attack_target, cr.power, true);
        } else {
            // Blocked — assign damage to blockers, blockers deal damage back.
            // T3.10: if the controller was prompted to divide damage (it couldn't kill every
            // blocker), apply their stored per-blocker assignment; otherwise auto-assign lethal
            // in order. Either way lethal accounts for damage already marked on the blocker
            // (the T3.11 fix) and treats deathtouch as lethal-1 (702.2c) — see lethal_needed_for_blocker.
            auto assign_it = game.combat_damage_assignment.find(entity);
            bool have_assignment = (assign_it != game.combat_damage_assignment.end());
            bool has_trample = creature_has_keyword(cr, "Trample");
            // An attacker that doesn't deal damage this step assigns nothing to its
            // blockers (and tramples nothing) — its blockers still deal their own damage.
            uint32_t remaining = attacker_deals ? cr.power : 0;
            for (auto blocker : blockers) {
                auto &bcr = global_coordinator.GetComponent<Creature>(blocker);

                // Blocker deals damage to attacker (only if the blocker qualifies for this step).
                if (should_deal_damage(bcr, first_strike_only))
                    deal_damage(blocker, entity, bcr.power, true);

                // Attacker deals damage to this blocker.
                uint32_t assigned = 0;
                if (have_assignment) {
                    auto bit = assign_it->second.find(blocker);
                    assigned = (bit != assign_it->second.end()) ? bit->second : 0u;
                    if (assigned > remaining) assigned = remaining;
                } else if (remaining > 0) {
                    // CR 510.1c: a blocked creature must assign ALL its combat damage. Auto-assign
                    // lethal to each blocker in order; the LAST blocker absorbs any leftover
                    // (harmless overkill) unless the attacker has trample, in which case the excess
                    // tramples over to the attack target below. So a single blocker receives the
                    // attacker's full power (Solitude [3/2] blocked by a [1/1] deals 3, lifelinking
                    // 3) — not just the blocker's lethal amount.
                    uint32_t needed = lethal_needed_for_blocker(entity, blocker);
                    bool is_last_blocker = (blocker == blockers.back());
                    if (is_last_blocker && !has_trample)
                        assigned = remaining;
                    else
                        assigned = (remaining >= needed) ? needed : remaining;
                }
                // Assignment (CR 510.1c) consumes `remaining` even when prevention (CR 615) then
                // stops the assigned damage from being dealt.
                deal_damage(entity, blocker, assigned, true);
                remaining -= assigned;
            }
            // Trample: excess damage goes to attack target
            if (has_trample && remaining > 0 && target_in_combat)
                deal_damage(entity, cr.attack_target, remaining, true);
        }
    }

    game.combat_damage_dealt = true;
    game_log("--- End %sCombat Damage ---\n\n", first_strike_only ? "First Strike " : "");
}

