#ifndef ZONE_SEARCH_H
#define ZONE_SEARCH_H

#include <memory>
#include <string>
#include <vector>

#include "components/zone.h"
#include "ecs/entity.h"

class Orderer;
class FrameCtx;

// Search a zone for cards matching the comma-separated type list in change_type
// (empty change_type matches all cards in the zone).
// When mandatory=true, "fail to find" is suppressed unless the zone is empty.
// Returns the chosen Entity, or 0 if the player fails to find / zone is empty.
// reveal=true marks every offered card choice as public knowledge (revealed
// tutors), so observers may show the chosen card's name even into a hidden zone.
// cmc_bound (>= 0) plus cmc_op ("EQ"/"LE"/...) additionally gate candidate cards by
// mana value (Aether Vial: MV == charge-counter count, resolved by the caller); -1 = none.
// The pick asks through `ctx` (with `decision_source` as the pending-decision
// context — the calling handler's ab.source); if the ask suspends, `suspended`
// is set and 0 is returned mutating nothing — the caller must propagate
// SUSPENDED. The candidate scan and menu rebuild identically on resume.
// `chain_target` is the resolving chain's card target for a ChangeType `targetedBy` filter
// alternative (Cloak and Dagger, Entwined); 0 = none.
Entity search_zone(std::shared_ptr<Orderer> orderer, Zone::Ownership owner,
    Zone::ZoneValue zone, const std::string &change_type, bool mandatory,
    Zone::ZoneValue destination, bool reveal, int cmc_bound, const std::string &cmc_op,
    FrameCtx &ctx, Entity decision_source, bool &suspended, Entity chain_target = 0);
Entity search_multi_zone(std::shared_ptr<Orderer> orderer, Zone::Ownership owner,
    const std::vector<Zone::ZoneValue> &zones, const std::string &change_type, bool mandatory,
    Zone::ZoneValue destination, bool reveal,
    FrameCtx &ctx, Entity decision_source, bool &suspended, Entity chain_target = 0);

#endif /* ZONE_SEARCH_H */
