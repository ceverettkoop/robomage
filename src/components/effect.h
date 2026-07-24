#ifndef EFFECT_H
#define EFFECT_H

#include "zone.h"
#include <set>
#include <string>
#include <vector>
#include "types.h"
#include "../classes/colors.h"

// Replacement effects modify how an event occurs (e.g. entering tapped instead of untapped).
// Only the nested Replacement type is live — it is parsed from a card's R: lines and stored on
// CardData::replacement_effects. The Effect component itself is never instantiated.
struct Effect {
    struct Replacement {
        enum Kind {
            ENTERS_TAPPED,              // permanent enters the battlefield tapped
            CANT_BE_COUNTERED,          // this spell can't be countered
            EXILE_INSTEAD_OF_GRAVEYARD, // opponent's cards exiled instead of going to graveyard (Dauthi Voidwalker, Leyline of the Void)
            EXILE_INSTEAD_OF_ETB,       // 614.1a — a non-token creature that wasn't cast is exiled instead of entering the battlefield (Containment Priest)
            SKIP_UNTAP,                 // 614.1d — a matching permanent doesn't untap during its controller's untap step (Choke)
            PREVENT_ETB_FROM_ZONES,     // 614.13/CantHappen — a creature card moving from a restricted origin zone to the battlefield doesn't enter; it stays put (Grafdigger's Cage)
            PRODUCE_MANA,               // 614.1 — replaces the mana a matching permanent produces when tapped (Damping Sphere: a land tapped for 2+ produces that much {C})
            DISCARD_ELSE_GRAVEYARD,     // 614.1a self-replacement — as this card would enter, its owner MAY pay an additional discard cost (a matching card); if paid it enters, otherwise it goes to its owner's graveyard instead (Mox Diamond, Chrome Mox)
        };
        Kind kind = ENTERS_TAPPED;
        bool applies_to_self_only = false;  // only fires when the affected entity is the source itself
        // CANT_BE_COUNTERED battlefield form (Hexing Squelcher: "Spells you control can't be
        // countered"). The self form ("This spell can't be countered", applies_to_self_only) is
        // resolved at cast time by stamping Spell::cant_be_countered; this continuous form is a
        // battlefield-active replacement (ActiveZones$ Battlefield) consulted at counter-resolution
        // time against every spell on the stack. valid_sa_filter is the ValidSA$ spec it protects
        // (e.g. "Spell.YouCtrl") — the controller scope (YouCtrl/OppCtrl) is read from the spell's
        // caster relative to this permanent's controller, since a stack spell has no Permanent
        // controller for the filter matcher to read.
        bool from_battlefield = false;      // ActiveZones$ Battlefield — a continuous battlefield static
        std::string valid_sa_filter = "";   // ValidSA$ spec the replacement protects from being countered
        // CANT_BE_COUNTERED spell-mastery gate (Exquisite Firecraft: "this spell can't be
        // countered if there are two or more instant and/or sorcery cards in your graveyard").
        // When cant_counter_present is non-empty the SELF form is NOT stamped onto the spell at
        // cast; instead the condition is re-evaluated as the counter would resolve (CR 614.13 —
        // the replacement fires now), counting the caster-owned cards matching the comma-OR
        // filter `cant_counter_present` in zone `cant_counter_zone` against `cant_counter_compare`.
        // An empty filter is the unconditional self form ("This spell can't be countered", a
        // cast-time stamp like Long Goodbye).
        std::string cant_counter_present = "";   // IsPresent$ (e.g. "Instant.YouOwn,Sorcery.YouOwn")
        std::string cant_counter_zone = "";      // PresentZone$ (e.g. "Graveyard")
        std::string cant_counter_compare = "";   // PresentCompare$ (e.g. "GE2")
        std::string valid_subtype = "";     // SKIP_UNTAP: the (sub)type the untap-prevention applies to (e.g. "Island")
        // ENTERS_TAPPED conditional gating (Ba Sing Se: "enters tapped unless you control a
        // basic land"). The replacement applies only when the count of the controller's
        // permanents matching `tapped_condition_filter` satisfies `tapped_condition_compare`
        // (e.g. filter "Land.Basic", compare "EQ0" — tapped only with zero basic lands).
        // Empty filter = unconditional (the plain "enters tapped").
        std::string tapped_condition_filter = "";   // e.g. "Land.Basic" (controller-relative)
        std::string tapped_condition_compare = "";   // e.g. "EQ0", "GE1"
        // ENTERS_TAPPED "unless you pay N life" (the shock-land / Witch-Blessed Meadow pattern:
        // DB$ Tap | ETB$ True | UnlessCost$ PayLife<N>). When > 0, the controller may pay this
        // much life as the permanent enters to have it enter UNTAPPED instead; declining (or
        // being unable to pay) leaves it entering tapped (CR 614.1d self-replacement + 118.8).
        int tapped_unless_life = 0;
        bool with_void_counter = false;     // EXILE_INSTEAD_OF_GRAVEYARD: exile with a void counter (Dauthi Voidwalker) so the controller may later play it; plain exile (Leyline of the Void) leaves it false
        // PREVENT_ETB_FROM_ZONES: the set of origin zones whose creature cards can't enter the battlefield.
        bool prevent_from_graveyard = false;  // Origin$ includes Graveyard
        bool prevent_from_library = false;    // Origin$ includes Library
        // PRODUCE_MANA (Damping Sphere "R:Event$ ProduceMana | ValidCard$ Land | ManaAmount$ GE2 |
        // ReplaceWith$ ...ReplaceMana$ C"): when a permanent matching `produce_valid_type` is
        // tapped for at least `produce_min_amount` mana, replace all of that mana with the same
        // total amount of `produce_replacement_color` (614.1). produce_min_amount == 1 means it
        // applies to any production.
        std::string produce_valid_type = "";          // ValidCard$ type filter ("Land")
        int produce_min_amount = 1;                    // ManaAmount$ GEN — minimum produced amount that triggers
        Colors produce_replacement_color = COLORLESS;  // ReplaceMana$ color the production is converted to
        // DISCARD_ELSE_GRAVEYARD (Mox Diamond / Chrome Mox): the type filter of the card the owner
        // may discard as the additional cost to have this permanent enter (e.g. "Land"); if they
        // don't (or can't), it is put into its owner's graveyard instead.
        std::string discard_else_filter = "";
    };
};



#endif /* EFFECT_H */
