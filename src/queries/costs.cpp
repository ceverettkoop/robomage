#include "costs.h"

#include <cstddef>

#include "../components/player.h"
#include "../ecs/coordinator.h"
#include "../systems/orderer.h"
#include "activation.h"
#include "battlefield.h"
#include "characteristics.h"
#include "counters.h"
#include "filters.h"
#include "player_resources.h"
#include "players.h"
#include "zones.h"

extern Coordinator global_coordinator;

static bool part_payable(const CostPart &part, const CostCheck &check);
static bool controls_enough(const CostPart &part, const CostCheck &check, Entity except);
static bool pitch_covers(const CostPart &part, const CostCheck &check);
static bool hand_covers(const CostPart &part, const CostCheck &check);
static bool graveyard_covers(const CostPart &part, const CostCheck &check);
static bool mana_payable(const CostParts &cost, const CostCheck &check);

bool can_pay(const CostParts &cost, const CostCheck &check) {
    for (const CostPart &part : cost.parts())
        if (!(check.unchecked & CostParts::kind_bit(part.kind)) && !part_payable(part, check)) return false;
    return mana_payable(cost, check);
}

static bool part_payable(const CostPart &part, const CostCheck &check) {
    switch (part.kind) {
        case CostPart::TAP:
            return source_can_tap(check.source);
        case CostPart::LIFE:
            // CR 119.4: you can't pay life you don't have. A fetch land (Pay 1 life) at 1 life is
            // still legal (you pay down to 0, then die); only a cost of MORE life than you have fails.
            return can_pay_life(global_coordinator.GetComponent<Player>(get_player_entity(check.payer)), part.count);
        case CostPart::ENERGY:
            // CR 122.1c: you can't pay {E} you don't have.
            return player_energy(global_coordinator.GetComponent<Player>(get_player_entity(check.payer))) >= part.count;
        case CostPart::LOYALTY:
            // A fixed minus cost needs enough loyalty (606.6; equality is legal — it may go to exactly
            // 0 and die to the SBA). An X minus cost is legal at any loyalty: X is chosen 0..current.
            return part.count_is_x || part.count >= 0 || get_counters(check.source, "LOYALTY") >= -part.count;
        case CostPart::SACRIFICE:
            // A .Other in the filter excludes the source ("another creature").
            return part.is_self() || controls_enough(part, check, check.source);
        case CostPart::RETURN:
            return controls_enough(part, check, 0);
        case CostPart::DISCARD:
            return hand_covers(part, check);
        case CostPart::EXILE_FROM_HAND:
            return pitch_covers(part, check);
        case CostPart::EXILE_FROM_GRAVEYARD:
            return graveyard_covers(part, check);
        default:
            return true;
    }
}

// Does the payer control `part.count` permanents matching its filter, `except` left out of a .Other?
static bool controls_enough(const CostPart &part, const CostCheck &check, Entity except) {
    return controlled_permanents_matching(check.payer, part.filter, check.orderer->mEntities, except).size() >=
           static_cast<size_t>(part.count);
}

// Enough cards matching the pitch's filter in the payer's hand, the spell itself left out of its
// .Other.
static bool pitch_covers(const CostPart &part, const CostCheck &check) {
    MatchCtx ctx;
    ctx.source = check.source;
    int matching = 0;
    for (Entity e : check.orderer->get_hand(check.payer))
        if (card_matches_filter(e, part.filter, ctx)) matching++;
    return matching >= part.count;
}

// Enough cards in the payer's hand, other than the spell, for a Discard<N/...> `times` over: that many
// matching its filter, and that many left once the rest of the cost has taken its own.
static bool hand_covers(const CostPart &part, const CostCheck &check) {
    if (part.is_self() || part.whole_hand) return true;
    const int need = part.count * check.times;
    int hand = 0;
    int matching = 0;
    for (Entity e : check.orderer->get_hand(check.payer)) {
        if (e == check.source) continue;
        hand++;
        if (card_matches_filter(e, part.filter)) matching++;
    }
    return hand - check.hand_reserved >= need && matching >= need;
}

// Enough OTHER cards in the payer's graveyard: a fixed number of them (Uro's escape), or any number
// that collectively cover the card types the cost asks for (Nethergoyf's, CR 601.2f).
static bool graveyard_covers(const CostPart &part, const CostCheck &check) {
    if (part.is_self()) return true;
    const EntitySet &entities = check.orderer->mEntities;
    if (part.count_is_x) return graveyard_card_types(check.payer, entities, check.source) >= part.min_types;
    return graveyard_card_count(check.payer, entities, check.source) >= part.count;
}

// The mana owed. A {T} in the cost spends the source's tap, so its mana ability is NOT also available
// to pay with — exclude it, or a Blast Zone whose only other land is an Ancient Tomb reads as able to
// pay {3} off 2 mana plus its own {C}. The mana is paid before a PayLife cost, so a painful source
// may not spend the life that cost needs (life_reserve).
static bool mana_payable(const CostParts &cost, const CostCheck &check) {
    if (check.unchecked & CostParts::kind_bit(CostPart::MANA)) return true;
    const ManaValue &owed = check.mana ? *check.mana : cost.mana();
    if (check.spell) return can_pay_spell_mana(check.payer, owed, *check.spell, check.payee, check.orderer);
    if (check.hybrid)
        return resolve_hybrid_cost(check.payer, owed, *check.hybrid, check.payee, check.orderer, check.delve,
                                   check.improvise);
    if (owed.empty()) return true;
    const Entity exclude = cost.taps() ? check.source : 0;
    if (check.estimate_mana) return can_afford_with_sources(check.payer, owed, check.orderer, exclude);
    return can_pay_mana(check.payer, owed, check.payee, check.orderer, check.delve, check.improvise, exclude,
                        cost.life() + check.life_reserve);
}
