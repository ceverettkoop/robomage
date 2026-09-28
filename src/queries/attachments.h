#ifndef QUERIES_ATTACHMENTS_H
#define QUERIES_ATTACHMENTS_H

#include <set>
#include <vector>
#include "../components/zone.h"
#include "../ecs/entity.h"
#include "battlefield.h"

// ── Equipment (CR 301.5) ──────────────────────────────────────────────────────

// Whether the Equipment `equipment` can equip `host` (CR 301.5c): `host` is a creature on the
// battlefield other than the Equipment itself, and the Equipment is not itself a creature unless
// it has reconfigure. Who controls `host` is not part of it (CR 301.5d). Shared by the equip
// ability's candidates and the 704.5n unattach check.
bool equipment_can_equip(Entity equipment, Entity host);

// Whether `host` is a legal target for the equip ability of `equipment` activated by `player`
// (CR 702.6a: "attach to target creature you control"), checked when the ability is offered and
// again as it resolves.
inline bool is_equip_candidate(Entity equipment, Entity host, Zone::Ownership player) {
    return is_battlefield_permanent(host, player) && equipment_can_equip(equipment, host);
}

// The creatures `player` controls that the Equipment `equipment` can be attached to by its equip
// ability. Shared by the equip legal-action gate and the creature menu offered when it is
// activated.
std::vector<Entity> equip_candidates(Entity equipment, Zone::Ownership player,
                                     const std::set<Entity> &entities);

#endif /* QUERIES_ATTACHMENTS_H */
