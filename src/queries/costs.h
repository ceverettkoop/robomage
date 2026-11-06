#ifndef QUERIES_COSTS_H
#define QUERIES_COSTS_H

#include <cstdint>
#include <memory>

#include "../classes/colors.h"
#include "../components/cost_part.h"
#include "../components/zone.h"
#include "../ecs/entity.h"
#include "../mana_system.h"

class Orderer;
struct CardData;

// Who a cost is being offered to and what for (can_pay).
struct CostCheck {
        CostCheck(Zone::Ownership who, Entity paying_for, const std::shared_ptr<Orderer> &orderer_)
            : payer(who), source(paying_for), orderer(orderer_), payee(paying_for) {}

        Zone::Ownership payer;
        Entity source;  // the ability's source or the card being cast: the object {T} taps and `Other` excludes
        const std::shared_ptr<Orderer> &orderer;
        ManaPayee payee;                  // what the mana pays for; a cast of `source` unless set
        const ManaValue *mana = nullptr;  // the mana owed after reductions and floors; null: the cost's own
        const CardData *spell = nullptr;  // a cast for its mana cost: the card's hybrid and Phyrexian pips too
        bool delve = false;               // the spell's Delve applies to `mana`
        bool improvise = false;           // the spell's Improvise applies to `mana`
        bool estimate_mana = false;       // a mana ability's own mana: can_afford_with_sources, not the payer
        // The kinds this path does not hold the player to (CostParts::kind_bit, ORed): what it offers
        // whether or not they can be paid.
        uint16_t unchecked = 0;
};

// Every kind but mana: a path that holds the player to a cost's mana alone.
constexpr uint16_t kAllButMana = static_cast<uint16_t>(~CostParts::kind_bit(CostPart::MANA));

// Can `check.payer` pay `cost` now? The one affordability check behind every offer of a cost: an
// activation, a mana ability, an alternative, flashback or escape cast, a regular cast's additional
// costs, suspend and a triggered ability's optional cost. Reads no decision and changes nothing.
//
// What it does not hold a player to, here or at the call sites' `unchecked`:
// - ExileFromHand<N/...> asks for one matching card, whatever N.
// - The rest of a cost that names the source, or a hand, is always payable: Sac<1/CARDNAME>,
//   Discard<1/CARDNAME>, Discard<0/Hand>.
bool can_pay(const CostParts &cost, const CostCheck &check);

#endif /* QUERIES_COSTS_H */
