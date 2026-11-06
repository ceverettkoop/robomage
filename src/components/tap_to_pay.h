#ifndef TAP_TO_PAY_H
#define TAP_TO_PAY_H

#include <cstdint>

// A payment modifier: while the mana of a cost is paid, permanents may be tapped rather than that
// mana being paid (CR 702.126a Improvise, 702.51a Convoke). Each kind is a row of the rule table in mana_system.cpp,
// which says what may be tapped and which pips a tap pays. Spell keywords set it on the card; a
// cost that carries one holds it on its CostParts.
enum class TapToPay : uint8_t {
    NONE,
    IMPROVISE,
    CONVOKE,
};

#endif /* TAP_TO_PAY_H */
