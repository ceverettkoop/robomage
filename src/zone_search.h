#ifndef ZONE_SEARCH_H
#define ZONE_SEARCH_H

#include <memory>
#include <string>
#include <vector>

#include "components/filter_spec.h"
#include "components/zone.h"
#include "ecs/entity.h"

class Orderer;
class FrameCtx;
struct ResolveCtx;

// What one pick of a search (CR 701.23) looks through and for.
struct ZoneSearch {
    Zone::Ownership owner = Zone::PLAYER_A;   // whose zones are searched
    std::vector<Zone::ZoneValue> zones;       // searched together (Doomsday: Graveyard,Library)
    // Comma-separated ChangeType$ filter alternatives; empty (or "Card") matches every card.
    FilterSpec change_type;
    // Mandatory$: "fail to find" is offered only when nothing matches.
    bool mandatory = false;
    Zone::ZoneValue destination = Zone::HAND;  // where the chosen card is headed (picks the menu kind)
    // Every offered card is public knowledge (a revealed tutor), so observers may show the chosen
    // card's name even into a hidden zone.
    bool reveal = false;
    // The value of the SVar a cmc<OP><SVar> qualifier of change_type reads (Aether Vial: its charge
    // counters); -1 = none.
    int cmc_bound = -1;
    // What a ChangeType `targetedBy` alternative reads its chain's target from (Cloak and Dagger,
    // Entwined); null = none.
    const ResolveCtx *resolve = nullptr;
    // Leave out the objects this resolution has already remembered (Doomsday choosing its five
    // cards one at a time).
    bool exclude_remembered = false;
};

// Searches `search.zones` for cards matching the comma-separated change_type string. Presents all
// matches plus a "fail to find" option (index 0).
// Returns the chosen Entity, or 0 for fail to find.
// 0 is a valid entity but will always be player a  so is never correct
// The pick asks through `ctx` (with `decision_source` as the pending-decision context — the
// calling handler's ab.source), on the priority seat the caller set to the choosing player; if
// the ask suspends, `suspended` is set and 0 is returned mutating nothing — the caller must
// propagate SUSPENDED. The candidate scan and menu rebuild identically on resume.
Entity search_zones(std::shared_ptr<Orderer> orderer, const ZoneSearch &search, FrameCtx &ctx,
                    Entity decision_source, bool &suspended);

#endif /* ZONE_SEARCH_H */
