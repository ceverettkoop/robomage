#ifndef DAMAGE_H
#define DAMAGE_H

#include <cstdint>
#include <cstddef>
#include "../ecs/entity.h"

struct Damage{
    size_t damage_counters;
    // 702.2b: any nonzero damage from a deathtouch source is lethal. Track whether
    // any damage marked on this creature came from such a source this turn.
    bool has_deathtouch_damage = false;
};

// CR 120.1/120.1a: `e` can be dealt damage — a player, a creature (it carries Damage) or a
// planeswalker permanent.
bool can_be_dealt_damage(Entity e);

// The one damage event (CR 120.4), shared by combat damage (CR 510.2) and every effect that deals
// damage. `amount` damage from `source` to `recipient`:
//   1. prevention (CR 615, 702.16e): a combat-damage shield (Maze of Ith) when `is_combat`, a
//      player's protection from everything, a creature's protection from the source, protection
//      from colored spells;
//   2. its results (CR 120.3): life loss, or poison counters from an infect source, for a player;
//      loyalty loss for a planeswalker; -1/-1 counters from a wither/infect source, else marked
//      damage, for a creature, flagged deathtouch damage from a deathtouch source (CR 702.2b);
//      and life gain for the source's controller from a lifelink source (CR 702.15b), combat or
//      not;
//   3. the damage event: COMBAT_DAMAGE_TO_PLAYER for combat damage to a player.
// Logs the damage (or its prevention) and any lifelink gain.
// The source's keywords and controller are read through damage_source_has_keyword /
// damage_source_controller (last-known information for a source that left play). Returns the
// damage actually dealt: 0 when it was prevented, `amount` was 0 (CR 120.8), or `recipient` can't
// be dealt damage.
size_t deal_damage(Entity source, Entity recipient, size_t amount, bool is_combat);

#endif /* DAMAGE_H */
