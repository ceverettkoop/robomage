#ifndef EFFECT_H
#define EFFECT_H

#include "zone.h"
#include <set>
#include <string>
#include <vector>
#include "types.h"

// Replacement effects modify how an event occurs (e.g. entering tapped instead of untapped).
// Only the nested Replacement type is live — it is parsed from a card's R: lines and stored on
// CardData::replacement_effects. The Effect component itself is never instantiated.
struct Effect {
    struct Replacement {
        enum Kind {
            ENTERS_TAPPED,              // permanent enters the battlefield tapped
            CANT_BE_COUNTERED,          // this spell can't be countered
            EXILE_INSTEAD_OF_GRAVEYARD, // opponent's cards exiled with void counter instead of going to graveyard
            SKIP_UNTAP,                 // 614.1d — a matching permanent doesn't untap during its controller's untap step (Choke)
        };
        Kind kind = ENTERS_TAPPED;
        bool applies_to_self_only = false;  // only fires when the affected entity is the source itself
        std::string valid_subtype = "";     // SKIP_UNTAP: the (sub)type the untap-prevention applies to (e.g. "Island")
    };
};



#endif /* EFFECT_H */
