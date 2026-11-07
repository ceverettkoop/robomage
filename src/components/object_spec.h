#ifndef OBJECT_SPEC_H
#define OBJECT_SPEC_H

#include <cstdint>
#include <string>

#include "filter_spec.h"

// A reference to objects or players, as a script's Defined$ / DefinedPlayer$ value names them (CR
// 608.2c). Parsed once onto the immutable AbilityDef; resolved each time it is read, against a
// ResolveCtx (resolve_objects / resolve_players, queries/affected.h). Each name, what it resolves
// to and which abilities read it are one row of the table in queries/affected.cpp.
struct ObjectSpec {
    enum Kind : uint8_t {
        DEFAULT,  // the ability declared no Defined$
        UNREAD,   // a name without a row: no reader acts on it (--parse-report's *.Defined findings)

        SELF,                // the ability's source
        TRIGGERED_CARD,      // the object whose event fired it, else its source
        REMEMBERED,          // the objects the resolution remembers
        DELAY_REMEMBERED,    // the objects a delayed trigger captured, restored as remembered
        IMPRINTED,           // the objects the resolution imprinted (a filter's set; no Defined$ name)
        CHOSEN_CARDS,        // the cards the resolution chose
        TARGETED,            // the target its parent chose
        TARGETED_MATCHING,   // that target, while it matches `filter`
        PARENT_TARGET,       // the same; only ChangeZone tells the two apart
        TRIGGERED_ATTACKER,  // bound as the target when it triggered
        TRIGGERED_SPELL,     // bound as the target when it triggered
        TRIGGERED_SOURCE_SA, // bound as the target when it triggered
        EXILED_WITH,         // the card the source exiled
        ENCHANTED,           // the object the source Aura enchants
        EQUIPPED,            // the object the source Equipment is attached to
        REPLACED_CARD,       // the object a replacement effect's event acts on
        VALID,               // every permanent matching `filter`
        VALID_HAND,          // every card in a hand matching `filter`

        YOU,                         // the ability's controller
        OPPONENT,                    // that controller's single opponent
        EACH_PLAYER,                 // both players, active player first
        TARGETED_CONTROLLER,         // the last-known controller of the chosen target
        PARENT_TARGETED_CONTROLLER,  // the controller of the target before this one, as it is chosen
        TRIGGERED_ACTIVATOR,         // bound when it triggered
        TRIGGERED_PLAYER,            // bound when it triggered
        TRIGGERED_CARD_CONTROLLER,   // bound when it triggered
        TRIGGERED_CARD_OWNER,        // the owner of the card that triggered it
        TRIGGERED_DEFENDING_PLAYER,  // bound as the target when it triggered
        YOU_AND_PERMANENTS,          // the controller and their permanents
        REMEMBERED_PLAYERS,          // the players among the remembered objects
    };

    Kind kind = DEFAULT;
    FilterSpec filter;  // Valid / ValidHand / Targeted.<filter>'s filter; empty for every other kind

    ObjectSpec() = default;
    explicit ObjectSpec(Kind k) : kind(k) {}

    bool is(Kind k) const { return kind == k; }
    bool is_remembered() const { return kind == REMEMBERED || kind == DELAY_REMEMBERED; }
    bool operator==(const ObjectSpec &o) const { return kind == o.kind && filter == o.filter; }
    bool operator!=(const ObjectSpec &o) const { return !(*this == o); }
};

#endif /* OBJECT_SPEC_H */
