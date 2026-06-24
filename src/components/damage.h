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

bool deal_damage(Entity source, Entity target, size_t amount);

// Deal `amount` damage from `source` to a player: reduce that player's life and grant
// the source's controller lifelink if the source has it. The single non-combat
// damage-to-player path (burn/ability damage); combat damage batches lifelink
// separately so it stays simultaneous with damage taken (rule 119.3). Returns false if
// `player_entity` is not a player.
bool deal_damage_to_player(Entity source, Entity player_entity, size_t amount);

#endif /* DAMAGE_H */