#ifndef ENTRY_INFO_H
#define ENTRY_INFO_H

#include "../object_ref.h"

// How a card will enter the battlefield: what a cast or an effect records about the card's next
// battlefield entry before its Permanent exists. These facts shape the entry itself (CR 614.1c-d,
// 614.12: tapped, transformed, attacking, attached) and the permanent's own "if it was cast / if
// it escaped / X" gates. Carried on the card as a component: added through entry_info()
// (queries/entry.h), each fact consumed as the entry is built (the ENTERS_BATTLEFIELD replacement
// dispatch, StateManager::apply_permanent_components) and the component removed once nothing is
// left (drop_entry_info_if_consumed). An entry that doesn't happen takes the record with it: the
// card entering any other zone (Orderer::add_to_zone), a prevented or impossible move onto the
// battlefield, a reversed cast (drop_entry_info).
struct EntryInfo {
    // Put onto the battlefield tapped by the effect that moves it (a ChangeZone Tapped$, ninjutsu,
    // CR 614.1d); applied by the ENTERS_BATTLEFIELD replacement dispatch.
    bool enters_tapped = false;
    // Enters showing its back face: put onto the battlefield transformed (CR 712.14a), a modal
    // DFC's back face played or cast (CR 712.8f), a daybound card entering at night (CR 702.145b),
    // a test preset named by its back face.
    bool enters_transformed = false;
    // Enters attacking this player or planeswalker (ninjutsu CR 702.49e, a dig's Attacking$,
    // CR 508.4). Consumed once the permanent has its Creature component, which a noncreature
    // ninja (Kaito) gets only later in the same pass. Empty = not entering attacking.
    ObjectRef enters_attacking;
    // A spell resolving onto the battlefield: it "was cast" (CR 614.12; Containment Priest lets
    // it through) → Permanent::entered_by_cast.
    bool cast = false;
    // Cast from its controller's own hand (a normal CR 601 hand cast) →
    // Permanent::cast_from_hand_by_controller (Amped Raptor's Card.wasCastFromYourHandByYou).
    bool cast_from_hand = false;
    bool evoked = false;     // cast for its evoke cost → Permanent::evoked
    bool offspring = false;  // cast with its Offspring cost → Permanent::entered_with_offspring
    bool escaped = false;    // cast with Escape → Permanent::cast_with_escape (Uro)
    bool impending = false;  // cast for its Impending cost (CR 702.175d) → enters with time counters
    bool warp = false;       // cast for its warp cost → the delayed end-step exile
    // The X its spell was cast with (> 0): an "enters with X counters" replacement (Chalice of
    // the Void) and Permanent::entered_x for its ETB abilities (CR 107.3m). 0 = none.
    int x_paid = 0;
    bool unearthed = false;  // returned by its unearth ability (CR 702.84)
    // An Equipment a DB$ Attach attached to it before its Permanent existed (reanimate-then-
    // attach, Pre-War Formalwear): the link is made as the Permanent is created.
    ObjectRef attach_equipment;
    // An Aura's enchant object, chosen before it enters: its cast target (CR 303.4a) or the object
    // chosen as it enters without being cast (CR 303.4f). Kept past the Permanent's creation while
    // that object is a graveyard card (Animate Dead) until its reanimation attaches the Aura.
    // Empty = none chosen.
    ObjectRef aura_target;

    bool consumed() const {
        return !enters_tapped && !enters_transformed && enters_attacking.empty() && !cast &&
               !cast_from_hand && !evoked && !offspring && !escaped && !impending && !warp &&
               x_paid == 0 && !unearthed && attach_equipment.empty() && aura_target.empty();
    }
};

#endif /* ENTRY_INFO_H */
