#ifndef CHOICE_LABELS_H
#define CHOICE_LABELS_H

#include <cstddef>
#include <string>
#include <vector>

#include "classes/action.h"
#include "classes/colors.h"
#include "components/zone.h"
#include "ecs/entity.h"

// Shared builders for the menu labels of binary player choices: the pay/decline menu of an
// unless-cost (CR 118.12) and the accept/decline menu of an optional choice. Every such prompt
// builds its entries here, so a label always states the real cost and what the choice causes,
// worded the same way at every site. Labels are display text only (the ML observation reads the
// category + option_ordinal), but they are what the semantic `desc:` resolver and the replay
// corpus read.

// How an unless-cost is paid: {N} generic mana or exact pips (default), N life (Ward—Pay life,
// CR 702.21), discard N card(s) from hand (Reality Smasher, CR 701.8), or N energy ({E},
// CR 122.1c — Static Prison's "unless you pay {E}").
enum class UnlessPayKind { MANA, LIFE, DISCARD, ENERGY };

// The effect an unless-cost governs, named from the resolving ability's category.
enum class UnlessEffect {
    COUNTER,    // Counter: "counter target spell unless its controller pays" (Daze, Mana Leak, Ward)
    DESTROY,    // Destroy: "destroy this creature unless you pay" (The Tabernacle at Pendrell Vale)
    SACRIFICE,  // Sacrifice: "sacrifice CARDNAME unless you pay {E}" (Static Prison)
    COPY,       // CopySpell: "that player may pay to copy this spell" (Chain Lightning)
};

// What an unless-cost prompt is about: the effect, the object it affects (the spell countered,
// the permanent destroyed or sacrificed, the spell copied), and whether paying ENABLES the effect
// (UnlessSwitched$ True, "copy only if paid") instead of preventing it.
struct UnlessSubject {
    UnlessEffect effect = UnlessEffect::COUNTER;
    Entity object = 0;
    bool happens_if_paid = false;
};

// "<object> is <effect>" / "<object> is not <effect>" for the outcome that follows when the
// payer pays (`paid`) or declines. The object is named through entity_name, which falls back to
// last-known information for a token that has ceased to exist.
std::string unless_outcome_text(const UnlessSubject &subject, bool paid);

// The cost as it is paid: "{1}", "{R}{R}", "3 life", "{E}{E}", "1 card".
std::string unless_cost_text(UnlessPayKind kind, size_t cost, const ManaValue *cost_pips);

// The PAY_UNLESS pay entry (option_ordinal 1): "Pay {R}{R} (Chain Lightning is copied)",
// "Discard 1 card (Counterspell is not countered)".
LegalAction unless_pay_action(const UnlessSubject &subject, UnlessPayKind kind, size_t cost,
                              const ManaValue *cost_pips);

// The PAY_UNLESS decline entry (option_ordinal 0): "Don't pay (Grizzly Bears is destroyed)",
// "Don't discard (Counterspell is countered)".
LegalAction unless_decline_action(const UnlessSubject &subject, UnlessPayKind kind);

// The two-entry OPTIONAL_YESNO menu: decline first (option_ordinal 0), accept second
// (option_ordinal 1). `accept_source` is the accept entry's source entity (0 = none).
std::vector<LegalAction> yesno_menu(const std::string &decline_label, const std::string &accept_label,
                                    Entity accept_source = 0);

// The optional-choice menu worded from one prompt: "Decline: <prompt>" / "Accept: <prompt>".
std::vector<LegalAction> optional_yesno_menu(const std::string &prompt);

// The "choose nothing (more)" entry label of an optional dig pick, worded from where a chosen
// card goes and where the unchosen rest go: "Take nothing (rest go to the bottom of library)",
// "Put nothing on the bottom of library (rest stay on top of library)" (Fateseal).
// `any_chosen` switches "nothing" to "no more" once a card has been picked.
std::string dig_decline_label(Zone::ZoneValue chosen_dest, bool chosen_on_bottom,
                              Zone::ZoneValue rest_dest, bool rest_on_bottom, bool any_chosen);

#endif /* CHOICE_LABELS_H */
