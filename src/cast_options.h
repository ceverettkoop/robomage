#ifndef CAST_OPTIONS_H
#define CAST_OPTIONS_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "classes/action.h"
#include "components/zone.h"
#include "ecs/entity.h"

class Orderer;

// Who is being offered casts, and whether this is their sorcery window (CR 307.1).
struct CastWindow {
        Zone::Ownership caster;
        bool sorcery_window;
        std::shared_ptr<Orderer> orderer;
};

// The cast rows of a priority menu, in menu order, appended to `actions`: `hand` is the caster's
// hand, `routes` the cards they have a permission for (play_routes).
void offer_cast_options(std::vector<LegalAction> &actions, const CastWindow &window, const std::vector<Entity> &hand,
                        const std::vector<std::pair<Entity, unsigned>> &routes);

// Whether `caster` may cast the exiled `card` now under its cast permission
// (Game::resolved_effects.impulse_cast_permission): the spell's timing — ignored for a permission granted for a
// cast during resolution (CR 608.2g) — its targets and cast prohibitions (can_cast_now), and the
// permission's alternative cost with any cost floor (CR 118.9d, 601.2f). Shared by the priority
// offer of granted exile casts and cast_during_resolution's offer.
bool exile_grant_castable(Entity card, Zone::Ownership caster, bool sorcery_window,
                          std::shared_ptr<Orderer> orderer);

// Whether `owner` may cast the revealed miracle `card` from their hand for its miracle cost as
// its miracle trigger resolves (CR 702.94a, 608.2g): timing ignored, its targets and cast
// prohibitions (can_cast_now), and the miracle mana cost with any cost floor (CR 118.9d, 601.2f).
bool miracle_castable(Entity card, Zone::Ownership owner, std::shared_ptr<Orderer> orderer);

// The cast a miracle trigger offers, and the cast from exile a resolution offers under its grant,
// `name` being what the row calls the card.
LegalAction miracle_cast_action(Entity card, const std::string &name);
LegalAction exile_grant_cast_action(Entity card, const std::string &name);

#endif /* CAST_OPTIONS_H */
