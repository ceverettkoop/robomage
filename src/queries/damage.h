#ifndef QUERIES_DAMAGE_H
#define QUERIES_DAMAGE_H

#include <cstddef>
#include <cstdint>
#include "../components/damage.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"
#include "counters.h"

// ── Damage (CR 120) ───────────────────────────────────────────────────────────

// The characteristics a damage source deals damage with (CR 702.2d-e, 702.15b-d, 702.80b-c,
// 702.90d-e): these keywords work from any zone, and a source that changed zones before the effect
// has it deal damage uses its last-known information. A battlefield permanent reads its live
// keywords, a spell on the stack its printed ones, a card that left the battlefield its snapshot
// (departed_lki_for), anything else its printed card. The controller is the source's controller,
// or its owner if it has none (702.15b); a departed source's is its last-known controller.
// Defined in damage.cpp.
bool damage_source_has_keyword(Entity source, const char *kw);
Zone::Ownership damage_source_controller(Entity source);

// CR 702.16: is `player_entity` currently under a "protection from everything" grant
// (cur_game.resolved_effects.player_protection_from_everything)? Protection from everything is protection from ALL
// sources — including the protected player's OWN sources — so this returns true whenever the grant
// is active for that player, regardless of who controls `source`. True means damage from `source`
// to that player is prevented. Consulted by the shared damage path (deal_damage) so the prevention
// rule lives in one place.
bool player_protected_from_source(Entity player_entity, Entity source);

// CR 702.16d (damage facet): is `perm_target` a permanent with "protection from colored spells"
// (Emrakul) being dealt damage by a SOURCE that is a colored spell? True means that damage is
// prevented. Tightly gated: the source must be a spell object (entity_has_component<Spell>) — so
// combat damage (creature source) and ability damage are never prevented — and one or more colors
// (a colorless spell's damage is not prevented). Reuses has_protection_from_colored_spells and
// effective_colors so the targeting block and the damage block share one definition of the rule.
bool permanent_protected_from_colored_spell_source(Entity perm_target, Entity source);

// Combat damage already marked on an entity this turn (0 if it has no Damage component).
inline uint32_t marked_damage_on(Entity e) {
    if (global_coordinator.entity_has_component<Damage>(e))
        return static_cast<uint32_t>(global_coordinator.GetComponent<Damage>(e).damage_counters);
    return 0u;
}

// Damage to a planeswalker removes that many loyalty counters (306.8). The loyalty-0 SBA
// (704.5i) then moves it to the graveyard. Single source for both combat and noncombat damage.
inline void damage_planeswalker(Entity pw, size_t amount) {
    add_counters(pw, "LOYALTY", -static_cast<int>(amount));
}

#endif /* QUERIES_DAMAGE_H */
