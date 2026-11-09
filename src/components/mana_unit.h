#ifndef MANA_UNIT_H
#define MANA_UNIT_H

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "../classes/colors.h"
#include "../ecs/entity.h"
#include "filter_spec.h"
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
struct ManaRider {
        enum Until : uint8_t {
            NO_DURATION,
            END_OF_TURN,  // AddsKeywordsUntil$ UntilEOT
        };
        bool cant_be_countered = false;  // AddsNoCounter$ True (Cavern of Souls)
        std::string keyword;             // AddsKeywords$: a spell it is spent on gains it (Arena of Glory)
        FilterSpec valid;                // AddsKeywordsValid$: the spells that keyword goes to
        Until until = NO_DURATION;

        bool operator<(const ManaRider &o) const { return key() < o.key(); }

    private:
        std::tuple<bool, const std::string &, const std::string &, Until> key() const {
            return std::tie(cant_be_countered, keyword, valid.text(), until);
        }
};

// `r` in the same kind of store as a restriction's: a definition and every unit it makes hold one pointer.
const ManaRider *intern_mana_rider(const ManaRider &r);

// One mana (CR 106.1): its color, what made it and what that said about it.
struct ManaUnit {
        ManaUnit(Colors c = NO_COLOR) : color(c) {}

        // The object that made it, as it stood once its mana ability's costs were paid: a source that
        // sacrificed or exiled itself is named where it went.
        ObjectRef source;
        const ManaRestriction *restriction = nullptr;  // null: spendable on anything
        const ManaRider *rider = nullptr;              // null: it does nothing to what it is spent on
        Colors color;
};

// The mana a cost spent, in the order it left the pool. Everything a card asks about the mana spent
// to cast a spell is a view of this list.
using ManaPaid = std::vector<ManaUnit>;

// How much of it was `color`: "if {R}{R} was spent to cast it" (Vibrance).
size_t mana_of_color(const ManaPaid &paid, Colors color);
// The distinct colors of it, WHITE..GREEN: colorless mana is not a color (CR 105.1).
std::set<Colors> colors_of(const ManaPaid &paid);
// How much of it `source` made: a ValidSA$ Spell.ManaFromCard reads this. The id survives the source
// moving, so a source that spent itself to make the mana still counts.
size_t mana_from(const ManaPaid &paid, Entity source);

// Does any of it say what it is spent on can't be countered?
bool mana_makes_uncounterable(const ManaPaid &paid);

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
