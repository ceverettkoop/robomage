#include "choice_labels.h"

#include "game_queries.h"

static const char *unless_effect_participle(UnlessEffect effect);
static std::string zone_destination_phrase(Zone::ZoneValue dest, bool on_bottom);

// The past participle naming what the effect does to its object.
static const char *unless_effect_participle(UnlessEffect effect) {
    switch (effect) {
        case UnlessEffect::COUNTER:   return "countered";
        case UnlessEffect::DESTROY:   return "destroyed";
        case UnlessEffect::SACRIFICE: return "sacrificed";
        case UnlessEffect::COPY:      return "copied";
    }
    return "affected";
}

// Where a moved card goes, as a prepositional phrase ("on the bottom of library", "into hand").
static std::string zone_destination_phrase(Zone::ZoneValue dest, bool on_bottom) {
    switch (dest) {
        case Zone::LIBRARY:     return on_bottom ? "on the bottom of library" : "on top of library";
        case Zone::HAND:        return "into hand";
        case Zone::BATTLEFIELD: return "onto the battlefield";
        case Zone::GRAVEYARD:   return "into the graveyard";
        case Zone::EXILE:       return "into exile";
        default:                return "";
    }
}

std::string unless_outcome_text(const UnlessSubject &subject, bool paid) {
    bool happens = subject.happens_if_paid ? paid : !paid;
    return entity_name(subject.object) + (happens ? " is " : " is not ") +
           unless_effect_participle(subject.effect);
}

std::string unless_cost_text(UnlessPayKind kind, size_t cost, const ManaValue *cost_pips) {
    switch (kind) {
        case UnlessPayKind::LIFE:
            return std::to_string(cost) + " life";
        case UnlessPayKind::DISCARD:
            return std::to_string(cost) + (cost == 1 ? " card" : " cards");
        case UnlessPayKind::ENERGY: {
            std::string out;
            for (size_t i = 0; i < cost; i++) out += "{E}";
            return out;
        }
        case UnlessPayKind::MANA:
            break;
    }
    if (cost_pips && !cost_pips->empty()) return mana_value_text(*cost_pips);
    return "{" + std::to_string(cost) + "}";
}

LegalAction unless_pay_action(const UnlessSubject &subject, UnlessPayKind kind, size_t cost,
                              const ManaValue *cost_pips) {
    const char *verb = (kind == UnlessPayKind::DISCARD) ? "Discard " : "Pay ";
    LegalAction pay(PASS_PRIORITY, verb + unless_cost_text(kind, cost, cost_pips) + " (" +
                                       unless_outcome_text(subject, /*paid=*/true) + ")");
    pay.category = ActionCategory::PAY_UNLESS;
    pay.option_ordinal = 1;  // 1 = pay
    return pay;
}

LegalAction unless_decline_action(const UnlessSubject &subject, UnlessPayKind kind) {
    const char *verb = (kind == UnlessPayKind::DISCARD) ? "Don't discard (" : "Don't pay (";
    LegalAction decline(PASS_PRIORITY, verb + unless_outcome_text(subject, /*paid=*/false) + ")");
    decline.category = ActionCategory::PAY_UNLESS;
    decline.option_ordinal = 0;  // 0 = don't pay
    return decline;
}

std::vector<LegalAction> yesno_menu(const std::string &decline_label, const std::string &accept_label,
                                    Entity accept_source) {
    std::vector<LegalAction> yn;
    LegalAction decline(PASS_PRIORITY, decline_label);
    decline.category = ActionCategory::OPTIONAL_YESNO;
    decline.option_ordinal = 0;  // 0 = decline
    yn.push_back(decline);
    LegalAction accept(PASS_PRIORITY, accept_source, accept_label);
    accept.category = ActionCategory::OPTIONAL_YESNO;
    accept.option_ordinal = 1;  // 1 = accept
    yn.push_back(accept);
    return yn;
}

std::vector<LegalAction> optional_yesno_menu(const std::string &prompt) {
    return yesno_menu("Decline: " + prompt, "Accept: " + prompt);
}

std::string dig_decline_label(Zone::ZoneValue chosen_dest, bool chosen_on_bottom,
                              Zone::ZoneValue rest_dest, bool rest_on_bottom, bool any_chosen) {
    const char *amount = any_chosen ? "no more" : "nothing";
    std::string label = (chosen_dest == Zone::HAND)
                            ? std::string("Take ") + amount
                            : std::string("Put ") + amount + " " +
                                  zone_destination_phrase(chosen_dest, chosen_on_bottom);
    if (rest_dest == Zone::LIBRARY)
        label += rest_on_bottom ? " (rest go to the bottom of library)" : " (rest stay on top of library)";
    else
        label += " (rest go " + zone_destination_phrase(rest_dest, rest_on_bottom) + ")";
    return label;
}
