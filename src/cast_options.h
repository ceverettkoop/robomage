#ifndef CAST_OPTIONS_H
#define CAST_OPTIONS_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "classes/action.h"
#include "components/cast_option.h"
#include "components/zone.h"
#include "ecs/entity.h"

class Orderer;

// What a resolving effect lets its controller cast a card for, as part of that resolution (CR 608.2g).
struct ResolutionCastTerms {
        CastGrant grant;
};

// Who is being offered casts, and whether this is their sorcery window (CR 307.1). `resolution` says
// the offer is a resolution's, on those terms: not at priority, the spell's timing ignored.
struct CastWindow {
        Zone::Ownership caster;
        bool sorcery_window;
        std::shared_ptr<Orderer> orderer;
        const ResolutionCastTerms *resolution = nullptr;
};

// The cast rows of a priority menu, in menu order, appended to `actions`: `hand` is the caster's
// hand, `routes` the cards they have a permission for (play_routes).
void offer_cast_options(std::vector<LegalAction> &actions, const CastWindow &window, const std::vector<Entity> &hand,
                        const std::vector<std::pair<Entity, unsigned>> &routes);

// Whether `owner` may cast the revealed miracle `card` from their hand for its miracle cost as
// its miracle trigger resolves (CR 702.94a, 608.2g): timing ignored, its targets and cast
// prohibitions (can_cast_now), and the miracle mana cost with any cost floor (CR 118.9d, 601.2f).
bool miracle_castable(Entity card, Zone::Ownership owner, std::shared_ptr<Orderer> orderer);

// The cast a miracle trigger offers, `name` being what the row calls the card.
LegalAction miracle_cast_action(Entity card, const std::string &name);

// The ways `caster` may cast `card` from the zone it is in as a resolution offers it (CR 608.2g), in
// the table's order: its targets and cast prohibitions (can_cast_now), the spell's timing ignored,
// and what `terms` asks of it with any cost floor (CR 118.9d, 601.2f).
std::vector<LegalAction> resolution_cast_rows(Entity card, Zone::Ownership caster, const ResolutionCastTerms &terms,
                                              std::shared_ptr<Orderer> orderer);

#endif /* CAST_OPTIONS_H */
