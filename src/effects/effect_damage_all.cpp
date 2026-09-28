#include "effects.h"

#include <string>
#include <vector>

#include "../components/damage.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../queries/filters.h"
#include "../queries/players.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;

namespace effects {

// DamageAll (Whipflare, Pyroclasm, ...): deal NumDmg$ damage (a number, or a dynamic amount
// evaluated at resolution like DealDamage's) to every battlefield permanent matching the
// ValidCards$ filter (e.g. "Creature.nonArtifact") and every player matching ValidPlayers$
// (e.g. "Player", "Player.Opponent"). The damage is dealt simultaneously (CR 120.4) through the
// shared damage path, so creatures, planeswalkers and players each get their own result (CR
// 120.3); the lethal-damage and loyalty state-based actions then act on the permanents.
HandlerResult damage_all(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    size_t dmg = ab.amount;
    if (!ab.dynamic_amount_expr.empty())
        dmg = evaluate_dynamic_amount(ab.dynamic_amount_expr, ab.controller, orderer, ab.target.get(),
                                      ab.source.lki_entity());
    std::vector<Entity> recipients;
    if (!ab.valid_cards_filter.empty())
        for (auto e : orderer->mEntities)
            if (permanent_matches_filter(e, ab.valid_cards_filter, MatchCtx{ab.controller, ab.source.lki_entity()}))
                recipients.push_back(e);
    const DamageParams *dp = std::get_if<DamageParams>(&ab.params);
    if (dp && !dp->valid_players.empty())
        for (Zone::Ownership seat : {Zone::PLAYER_A, Zone::PLAYER_B}) {
            Entity pe = get_player_entity(seat);
            if (player_matches_target_spec(dp->valid_players, pe, ab.controller))
                recipients.push_back(pe);
        }

    for (auto e : recipients) ::deal_damage(ab.source.lki_entity(), e, dmg, false);
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
