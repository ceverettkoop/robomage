#ifndef COST_PART_H
#define COST_PART_H

#include <cstdint>
#include <vector>

#include "../classes/colors.h"
#include "filter_spec.h"
#include "object_spec.h"

// One part of a cost (CR 118, 601.2f-h, 602.2b) as the script's cost string names it: "T", "2 R",
// "PayLife<2>", "Sac<1/Creature>". Parsed once, by parse_cost, onto the immutable definition that
// holds it; can_pay (queries/costs.h) reads it as a payment is offered.
struct CostPart {
    enum Kind : uint8_t {
        MANA,                  // plain pips, `mana` (WUBRGC and generic); one part per cost
        X_MANA,                // {X}: `count` pips, each owing the one chosen X; one part per cost
        TAP,                   // {T}: the source taps
        LIFE,                  // PayLife<count>; count_is_x: PayLife<X>, the life paid is X
        ENERGY,                // PayEnergy<count> (CR 122.1c)
        LOYALTY,               // AddCounter / SubCounter<count/LOYALTY>: `count` is signed; count_is_x:
                               // the chosen X, `count` holding only its sign
        SACRIFICE,             // Sac<count/filter>, or the source (`object`)
        DISCARD,               // Discard<count/filter>, the source (`object`), or `whole_hand`
        RETURN,                // Return<count/filter>: to its owner's hand
        EXILE_FROM_HAND,       // ExileFromHand<count/filter>
        EXILE_FROM_GRAVEYARD,  // ExileFromGrave<count/filter>; count_is_x: any number, covering `min_types`
        MANDATORY,             // not a payment: a triggered ability's cost is not optional (CR 603.8)
    };

    Kind kind = MANA;
    int count = 0;
    bool count_is_x = false;
    FilterSpec filter;
    ObjectSpec object;        // SELF for CARDNAME; DEFAULT when `filter` picks
    ManaValue mana;           // MANA's pips
    bool whole_hand = false;  // DISCARD: Discard<0/Hand>
    int min_types = 0;        // EXILE_FROM_GRAVEYARD: distinct card types among the cards exiled

    static CostPart of(Kind kind, int count = 0) {
        CostPart part;
        part.kind = kind;
        part.count = count;
        return part;
    }
    bool is_self() const { return object.is(ObjectSpec::SELF); }
};

// A cost: its parts in script order. Every typed read below is over the parts, so nothing can
// disagree with them; add is the only writer.
class CostParts {
    public:
        // Appends `part`. Mana merges into the cost's one MANA part and {X} pips into its one X_MANA
        // part; mana of no pips adds nothing.
        void add(const CostPart &part);
        // Cost$ 0 on an alternative cost: castable for nothing, as distinct from no cost named.
        void set_free() { free_ = true; }

        bool is_free() const { return free_; }
        bool empty() const { return parts_.empty(); }
        bool has(CostPart::Kind kind) const { return (kinds_ & kind_bit(kind)) != 0; }
        const std::vector<CostPart> &parts() const { return parts_; }
        // The first part of `kind`; null with none.
        const CostPart *find(CostPart::Kind kind) const;

        const ManaValue &mana() const;
        bool taps() const { return has(CostPart::TAP); }
        int x_pips() const { return count(CostPart::X_MANA); }
        int life() const { return count(CostPart::LIFE); }
        bool life_is_x() const { return variable(CostPart::LIFE); }
        int energy() const { return count(CostPart::ENERGY); }
        int loyalty() const { return count(CostPart::LOYALTY); }
        bool loyalty_is_x() const { return variable(CostPart::LOYALTY); }
        bool discards_hand() const;
        int exile_min_types() const;  // the distinct card types an ExileFromGrave<X/...> must cover

        // The first part of `kind`'s count; 0 with none.
        int count(CostPart::Kind kind) const;
        // Is the first part of `kind` variable?
        bool variable(CostPart::Kind kind) const;
        // Does a part of `kind` name the source itself (CARDNAME)?
        bool self(CostPart::Kind kind) const;
        // The filter of the first part of `kind` that has one; the empty filter with none.
        const FilterSpec &filter(CostPart::Kind kind) const;

        static constexpr uint16_t kind_bit(CostPart::Kind kind) { return static_cast<uint16_t>(1u << kind); }

    private:
        std::vector<CostPart> parts_;
        uint16_t kinds_ = 0;
        bool free_ = false;
};

#endif /* COST_PART_H */
