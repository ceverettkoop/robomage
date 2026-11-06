#ifndef QUERIES_COSTS_H
#define QUERIES_COSTS_H

#include <cstdint>
#include <memory>
#include <vector>

#include "../classes/colors.h"
#include "../components/cost_part.h"
#include "../components/zone.h"
#include "../ecs/entity.h"
#include "../mana_system.h"

class Orderer;
struct CardData;
struct HybridPip;

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
        TapToPay tap_to_pay = TapToPay::NONE;  // what may be tapped toward `mana`, beside the cost's own
        // A cost added to a cast: the hybrid pips still to resolve on top of `mana`, which is the
        // mana the cast has committed to with this cost's.
        const std::vector<HybridPip> *hybrid = nullptr;
        int times = 1;          // the cost is paid this many times over: its Discard's count
        int hand_reserved = 0;  // cards of the payer's hand the rest of the cost takes
        int life_reserve = 0;   // life the rest of the cost takes, which a painful source may not spend
        bool estimate_mana = false;       // a mana ability's own mana: can_afford_with_sources, not the payer
        // The kinds this path does not hold the player to (CostParts::kind_bit, ORed): what it offers
        // whether or not they can be paid.
        uint16_t unchecked = 0;
};

// Every kind but mana: a path that holds the player to a cost's mana alone.
constexpr uint16_t kAllButMana = static_cast<uint16_t>(~CostParts::kind_bit(CostPart::MANA));

// Can `check.payer` pay `cost` now? The one affordability check behind every offer of a cost: an
// activation, a mana ability, an alternative, flashback or escape cast, a regular cast's additional
// costs, Kicker, Replicate and Escalate on top of what the cast has committed to, suspend and a
// triggered ability's optional cost. Reads no decision and changes nothing.
//
// What it does not hold a player to, here or at the call sites' `unchecked`:
// - The rest of a cost that names the source, or a hand, is always payable: Sac<1/CARDNAME>,
//   Discard<1/CARDNAME>, Discard<0/Hand>, Exile<1/CARDNAME>, ExileFromHand<1/CARDNAME>,
//   ExileFromGrave<1/CARDNAME>.
bool can_pay(const CostParts &cost, const CostCheck &check);

// Which of a spell's additional-cost forms can be paid (CR 601.2b): their indices, in order.
std::vector<size_t> payable_choices(const std::vector<CostParts> &choices, const CostCheck &check);

// The fewest cards of the payer's hand any payable form takes: what a cast's mana must leave there.
int fewest_hand_cards(const std::vector<CostParts> &choices, const CostCheck &check);

// Stop offering a spell or ability whose payment already failed twice (the payer bumps
// priority.payment_fail_counts on each cancelled payment), so an agent can't loop on it.
bool payment_blocked(Entity paid_for);

#endif /* QUERIES_COSTS_H */
