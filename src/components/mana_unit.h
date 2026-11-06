#ifndef MANA_UNIT_H
#define MANA_UNIT_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "../classes/colors.h"
#include "../object_ref.h"

// What a mana ability says its mana may be spent on (CR 106.6), from its RestrictValid$. On a
// definition `chosen_type` is empty: a chosen type is the source's, read as the mana is produced.
struct ManaRestriction {
        enum Kind : uint8_t {
            NONE,
            CHOSEN_TYPE_CREATURE,    // Spell.Creature+ChosenType
            CREATURE,                // Spell.Creature (any creature spell)
            COLORLESS_ELDRAZI,       // Spell.Eldrazi+Colorless (Eldrazi Temple)
            NO_NONARTIFACT_SPELLS,   // CantCastNonArtifactSpells (Powerstone tokens)
        };
        Kind kind = NONE;
        bool allows_activated = false;          // has an Activated. clause (Secluded Courtyard)
        bool activated_on_battlefield = false;  // that clause says inZoneBattlefield (Eldrazi Temple)
        std::string chosen_type;

        bool operator==(const ManaRestriction &o) const { return key() == o.key(); }
        bool operator!=(const ManaRestriction &o) const { return !(*this == o); }
        bool operator<(const ManaRestriction &o) const { return key() < o.key(); }

    private:
        std::tuple<Kind, bool, bool, const std::string &> key() const {
            return std::tie(kind, allows_activated, activated_on_battlefield, chosen_type);
        }
};

// `r` as an object that outlives the source and the game: a process-wide append-only store of
// values, like the ability definitions', so a unit's pointer stays valid across a snapshot restore.
const ManaRestriction *intern_mana_restriction(const ManaRestriction &r);

// What mana does to what it is spent on, by the ability that made it.
enum ManaRider : uint8_t {
    MANA_RIDER_NONE = 0,
    MANA_RIDER_CANT_BE_COUNTERED = 1 << 0,  // AddsNoCounter$ True (Cavern of Souls)
};

// One mana (CR 106.1): its color, what made it and what that said about it.
struct ManaUnit {
        ManaUnit(Colors c = NO_COLOR) : color(c) {}

        // The object that made it, as it stood once its mana ability's costs were paid: a source that
        // sacrificed or exiled itself is named where it went. Empty in a simulated payment.
        ObjectRef source;
        const ManaRestriction *restriction = nullptr;  // null: spendable on anything
        Colors color;
        uint8_t riders = MANA_RIDER_NONE;
};

// A player's mana pool (CR 106.4), or the copy a simulated payment works on. Its units stay in Colors
// order, oldest first within a color: what the pool spends first is what a multiset of colors would.
class ManaPool {
    public:
        using const_iterator = std::vector<ManaUnit>::const_iterator;

        const_iterator begin() const { return units.begin(); }
        const_iterator end() const { return units.end(); }
        bool empty() const { return units.empty(); }
        size_t size() const { return units.size(); }
        size_t count(Colors color) const;
        const_iterator find(Colors color) const;  // the first unit of `color`, else end()
        void insert(const ManaUnit &unit);
        void erase(const_iterator it) { units.erase(it); }
        void clear() { units.clear(); }

    private:
        std::vector<ManaUnit> units;
};

#endif /* MANA_UNIT_H */
