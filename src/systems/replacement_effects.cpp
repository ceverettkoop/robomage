#include "replacement_effects.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/static_ability.h"
#include "../components/token.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../input_logger.h"
#include "../svar_eval.h"

// Internal description of a single applicable replacement effect for one dispatch.
namespace {

enum CandidateKind {
    SELF_TAPPED,    // 614.1d — this permanent enters tapped (self-replacement, 614.15)
    PENDING_TAPPED, // a resolving ability put this permanent onto the battlefield tapped
    ETB_COUNTERS,   // 614.1c — this permanent enters with +1/+1 counters (self-replacement)
    EXILE_INSTEAD,  // 614.1a — opponent's card is exiled (with a void counter) instead of going to graveyard
    EXILE_INSTEAD_OF_ETB, // 614.1a — a non-token creature that wasn't cast is exiled instead of entering (Containment Priest)
    SKIP_UNTAP,     // 614.1d — this permanent doesn't untap during its controller's untap step (Choke)
    PREVENT_ETB,    // 614.13 — a creature card from a restricted origin zone is prevented from entering (Grafdigger's Cage)
};

struct Candidate {
    Entity source = 0;            // object generating the replacement effect
    int kind = SELF_TAPPED;
    int index = 0;                // disambiguator within `source` (for the once-per-event applied set)
    bool self_replacement = false; // 614.15 — gates choice order under 616.1a
    std::string label;            // menu label when >1 applies
    int amount = 0;               // ETB_COUNTERS: counter count
    std::string counter_type = "P1P1"; // ETB_COUNTERS: kind of counter
    bool with_void_counter = false; // EXILE_INSTEAD: tag the exiled card with a void counter (Dauthi), else plain exile (Leyline)
};

// Number of cards in a player's library / graveyard scan helpers mirror the
// battlefield scan in Orderer (entity-id order, matching get_graveyard()).
size_t library_size(Zone::Ownership owner) {
    size_t n = 0;
    Entity max_e = global_coordinator.GetMaxIssuedEntity();
    for (Entity e = 0; e < max_e; e++) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location == Zone::LIBRARY && z.owner == owner) n++;
    }
    return n;
}

bool already_applied(const std::set<std::pair<Entity, int>> &applied, const Candidate &c) {
    return applied.count({c.source, c.index}) != 0;
}

// Evaluate a conditional "enters tapped" gate (Ba Sing Se: "enters tapped unless you control a
// basic land"). `filter` is a "<Type>[.<Supertype>]" spec evaluated controller-relative; the
// count of `controller`'s matching battlefield permanents is compared via `compare` (e.g. EQ0).
// An empty filter means unconditional (always tapped). The entering permanent (`entering`) is
// excluded — it isn't a live battlefield permanent yet during the ENTERS_BATTLEFIELD dispatch.
static bool tapped_condition_met(const std::string &filter, const std::string &compare,
                                 Zone::Ownership controller, Entity entering) {
    if (filter.empty()) return true;
    // Split "Land.Basic" into a main type and an optional Basic/nonBasic supertype qualifier.
    std::string type_filter = filter;
    bool require_basic = false, require_nonbasic = false;
    size_t dot = filter.find('.');
    if (dot != std::string::npos) {
        type_filter = filter.substr(0, dot);
        std::string qual = filter.substr(dot + 1);
        if (qual == "Basic") require_basic = true;
        else if (qual == "nonBasic") require_nonbasic = true;
    }
    int count = 0;
    Entity max_e = global_coordinator.GetMaxIssuedEntity();
    for (Entity e = 0; e < max_e; e++) {
        if (e == entering) continue;
        if (!is_battlefield_permanent(e, controller)) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(e);
        bool type_ok = type_filter.empty() || type_filter == "Permanent";
        if (!type_ok)
            for (const auto &t : perm.types)
                if (t.name == type_filter) { type_ok = true; break; }
        if (!type_ok) continue;
        if (require_basic && !has_basic_supertype(perm.types)) continue;
        if (require_nonbasic && has_basic_supertype(perm.types)) continue;
        count++;
    }
    std::string cmp = compare.empty() ? "GE1" : compare;
    return compare_svar(count, cmp);
}

// Gather every replacement effect currently applicable to `ev`, excluding any that
// have already been applied this dispatch (614.5 — one opportunity per event), and
// re-checking applicability against the live event state (616.1f / 616.2).
std::vector<Candidate> collect(const ReplacementEvent &ev,
                               const std::set<std::pair<Entity, int>> &applied) {
    std::vector<Candidate> out;

    if (ev.type == ReplacementEvent::ENTERS_BATTLEFIELD) {
        if (global_coordinator.entity_has_component<CardData>(ev.entity)) {
            auto &cd = global_coordinator.GetComponent<CardData>(ev.entity);
            // Self "enters tapped" replacement effect (614.1d / self-replacement 614.15).
            for (size_t i = 0; i < cd.replacement_effects.size(); i++) {
                const Effect::Replacement &r = cd.replacement_effects[i];
                if (r.kind != Effect::Replacement::ENTERS_TAPPED) continue;
                // Conditional "enters tapped" (Ba Sing Se): apply only when the controller's
                // board satisfies the gate (e.g. zero basic lands). The entering permanent's
                // controller is ev.affected_player (616.1 chooser).
                if (!tapped_condition_met(r.tapped_condition_filter, r.tapped_condition_compare,
                                          ev.affected_player, ev.entity))
                    continue;
                Candidate c;
                c.source = ev.entity;
                c.kind = SELF_TAPPED;
                c.index = static_cast<int>(i);
                c.self_replacement = true;
                c.label = "enters tapped";
                if (!already_applied(applied, c)) out.push_back(c);
            }
            // "Enters with N counters" replacement effect (614.1c) from an EtbCounter static
            // ability. Count from delve (Hangarback-style) or from X paid at cast (Chalice of
            // the Void's CHARGE counters); the counter kind is whatever the script declared.
            for (size_t i = 0; i < cd.static_abilities.size(); i++) {
                const StaticAbility &sa = cd.static_abilities[i];
                if (sa.category != "EtbCounter") continue;
                int n = 0;
                if (sa.counter_count_from_delve) n = static_cast<int>(cur_game.delve_exiled.size());
                else if (sa.counter_count_from_xpaid) {
                    auto it = cur_game.pending_etb_xpaid.find(ev.entity);
                    if (it != cur_game.pending_etb_xpaid.end()) n = it->second;
                }
                else n = sa.counter_count;  // literal count (etbCounter:M1M1:6 → 6)
                if (n <= 0) continue;
                Candidate c;
                c.source = ev.entity;
                c.kind = ETB_COUNTERS;
                c.index = -1000 - static_cast<int>(i);  // disjoint from replacement_effects indices
                c.self_replacement = true;
                c.amount = n;
                c.counter_type = sa.counter_type.empty() ? "P1P1" : sa.counter_type;
                c.label = "enters with counters";
                if (!already_applied(applied, c)) out.push_back(c);
            }
        }
        // A resolving ChangeZone effect may have requested this permanent enter tapped.
        if (cur_game.pending_enters_tapped.count(ev.entity)) {
            Candidate c;
            c.source = ev.entity;
            c.kind = PENDING_TAPPED;
            c.index = -1;
            c.self_replacement = false;
            c.label = "enters tapped";
            if (!already_applied(applied, c)) out.push_back(c);
        }
        return out;
    }

    if (ev.type == ReplacementEvent::MOVE_TO_ZONE) {
        // A move already prevented (614.13) admits no further replacements.
        if (ev.prevented) return out;
        Entity max_e = global_coordinator.GetMaxIssuedEntity();

        // A card about to enter the battlefield can meet two replacement effects, and both
        // are evaluated here — inside the zone move, *before* add_to_zone emits the
        // enters-the-battlefield CARD_CHANGED_ZONE event. Applying the exile-redirect later
        // (at the ENTERS_BATTLEFIELD dispatch) would be too late: the creature would already
        // be on the battlefield and other permanents' "whenever a creature enters" triggers
        // (and its own ETB) would have fired for a creature that is supposed to never enter.
        if (ev.destination == Zone::BATTLEFIELD) {
            bool nontoken_creature =
                !global_coordinator.entity_has_component<Token>(ev.entity) &&
                global_coordinator.entity_has_component<CardData>(ev.entity) &&
                is_creature_card(global_coordinator.GetComponent<CardData>(ev.entity));

            // Grafdigger's Cage (614.13): a creature card moving from a graveyard or library
            // onto the battlefield is prevented from entering — it stays in its origin zone.
            // This covers reanimation (graveyard → battlefield) and search-to-battlefield
            // (Green Sun's Zenith, library → battlefield). A card that can't enter at all
            // needn't also be exiled, so a prevention takes precedence over Containment
            // Priest's exile-redirect below.
            if (nontoken_creature &&
                (ev.origin == Zone::GRAVEYARD || ev.origin == Zone::LIBRARY)) {
                for (Entity e = 0; e < max_e; e++) {
                    if (!is_battlefield_permanent(e)) continue;
                    if (!global_coordinator.entity_has_component<CardData>(e)) continue;
                    auto &cd = global_coordinator.GetComponent<CardData>(e);
                    for (size_t i = 0; i < cd.replacement_effects.size(); i++) {
                        const Effect::Replacement &r = cd.replacement_effects[i];
                        if (r.kind != Effect::Replacement::PREVENT_ETB_FROM_ZONES) continue;
                        if (ev.origin == Zone::GRAVEYARD && !r.prevent_from_graveyard) continue;
                        if (ev.origin == Zone::LIBRARY && !r.prevent_from_library) continue;
                        Candidate c;
                        c.source = e;
                        c.kind = PREVENT_ETB;
                        c.index = static_cast<int>(i);
                        c.self_replacement = false;
                        c.label = global_coordinator.GetComponent<Permanent>(e).name +
                                  ": can't enter from that zone";
                        if (!already_applied(applied, c)) out.push_back(c);
                    }
                }
                if (!out.empty()) return out;
            }

            // Containment Priest (614.1a): a non-token creature that wasn't cast is exiled
            // instead of entering, regardless of which zone it would have entered from
            // (reanimation, blink out of exile, put onto the battlefield from hand, ...). A
            // creature spell resolving from the stack is in cast_to_battlefield and let through.
            if (nontoken_creature && cur_game.cast_to_battlefield.count(ev.entity) == 0) {
                for (Entity e = 0; e < max_e; e++) {
                    if (!is_battlefield_permanent(e)) continue;
                    if (!global_coordinator.entity_has_component<CardData>(e)) continue;
                    auto &cd = global_coordinator.GetComponent<CardData>(e);
                    for (size_t i = 0; i < cd.replacement_effects.size(); i++) {
                        if (cd.replacement_effects[i].kind != Effect::Replacement::EXILE_INSTEAD_OF_ETB)
                            continue;
                        Candidate c;
                        c.source = e;
                        c.kind = EXILE_INSTEAD_OF_ETB;
                        c.index = static_cast<int>(i);
                        c.self_replacement = false;
                        c.label = global_coordinator.GetComponent<Permanent>(e).name +
                                  ": exile instead of entering";
                        if (!already_applied(applied, c)) out.push_back(c);
                    }
                }
            }
            return out;
        }

        // Dauthi Voidwalker etc.: only a graveyard-bound, non-token, owned card is eligible.
        if (ev.destination != Zone::GRAVEYARD) return out;
        if (global_coordinator.entity_has_component<Token>(ev.entity)) return out;
        if (!global_coordinator.entity_has_component<CardData>(ev.entity)) return out;
        auto &tz = global_coordinator.GetComponent<Zone>(ev.entity);

        for (Entity e = 0; e < max_e; e++) {
            if (!is_battlefield_permanent(e)) continue;
            auto &perm = global_coordinator.GetComponent<Permanent>(e);
            // The replacement source must be controlled by the opponent of the card's owner.
            if (perm.controller == tz.owner) continue;
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            for (size_t i = 0; i < cd.replacement_effects.size(); i++) {
                if (cd.replacement_effects[i].kind != Effect::Replacement::EXILE_INSTEAD_OF_GRAVEYARD)
                    continue;
                Candidate c;
                c.source = e;
                c.kind = EXILE_INSTEAD;
                c.index = static_cast<int>(i);
                c.self_replacement = false;
                c.with_void_counter = cd.replacement_effects[i].with_void_counter;
                c.label = perm.name + ": exile instead of graveyard";
                if (!already_applied(applied, c)) out.push_back(c);
            }
        }
        return out;
    }

    if (ev.type == ReplacementEvent::UNTAP) {
        // Choke etc.: a battlefield permanent generates a "matching lands don't untap"
        // replacement. The untap step only processes the active player's permanents during
        // their own turn, so "during their controllers' untap steps" holds implicitly.
        if (!global_coordinator.entity_has_component<Permanent>(ev.entity)) return out;
        auto &subject = global_coordinator.GetComponent<Permanent>(ev.entity);

        Entity max_e = global_coordinator.GetMaxIssuedEntity();
        for (Entity e = 0; e < max_e; e++) {
            if (!is_battlefield_permanent(e)) continue;
            auto &perm = global_coordinator.GetComponent<Permanent>(e);
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            for (size_t i = 0; i < cd.replacement_effects.size(); i++) {
                const Effect::Replacement &r = cd.replacement_effects[i];
                if (r.kind != Effect::Replacement::SKIP_UNTAP) continue;
                bool matches = false;
                for (const auto &t : subject.types)
                    if (t.name == r.valid_subtype) { matches = true; break; }
                if (!matches) continue;
                Candidate c;
                c.source = e;
                c.kind = SKIP_UNTAP;
                c.index = static_cast<int>(i);
                c.self_replacement = false;
                c.label = perm.name + ": doesn't untap";
                if (!already_applied(applied, c)) out.push_back(c);
            }
        }
        return out;
    }

    return out;
}

// Apply one chosen replacement effect to `ev`, mutating its outcome fields and
// consuming any one-shot game state the effect depends on.
void apply_one(ReplacementEvent &ev, const Candidate &c) {
    switch (c.kind) {
        case SELF_TAPPED:
            ev.enters_tapped = true;
            break;
        case PENDING_TAPPED:
            ev.enters_tapped = true;
            cur_game.pending_enters_tapped.erase(ev.entity);
            break;
        case ETB_COUNTERS:
            ev.etb_p1p1 += c.amount;
            ev.etb_counter_type = c.counter_type;
            cur_game.delve_exiled.clear();
            cur_game.pending_etb_xpaid.erase(ev.entity);
            break;
        case EXILE_INSTEAD: {
            ev.destination = Zone::EXILE;
            std::string name = global_coordinator.GetComponent<CardData>(ev.entity).name;
            if (c.with_void_counter) {
                cur_game.void_countered.insert(ev.entity);
                game_log("%s is exiled with a void counter.\n", name.c_str());
            } else {
                game_log("%s is exiled instead of being put into a graveyard.\n", name.c_str());
            }
            break;
        }
        case EXILE_INSTEAD_OF_ETB: {
            ev.destination = Zone::EXILE;
            std::string name = global_coordinator.GetComponent<CardData>(ev.entity).name;
            game_log("%s is exiled instead of entering the battlefield.\n", name.c_str());
            break;
        }
        case SKIP_UNTAP:
            ev.skip_untap = true;
            break;
        case PREVENT_ETB: {
            ev.prevented = true;
            std::string name = global_coordinator.GetComponent<CardData>(ev.entity).name;
            game_log("%s can't enter the battlefield from that zone.\n", name.c_str());
            break;
        }
    }
}

// 616.1: the affected player chooses ONE of several applicable replacement effects.
// 616.1a gates the choice to self-replacement effects first when any apply. Returns
// the index into `cands` of the chosen effect.
size_t choose_one(Zone::Ownership chooser, const std::vector<Candidate> &cands) {
    // 616.1a — self-replacement effects (614.15) must be chosen before others.
    std::vector<size_t> eligible;
    for (size_t i = 0; i < cands.size(); i++)
        if (cands[i].self_replacement) eligible.push_back(i);
    if (eligible.empty())
        for (size_t i = 0; i < cands.size(); i++) eligible.push_back(i);

    if (eligible.size() == 1) return eligible[0];

    std::vector<LegalAction> choices;
    for (size_t i : eligible) {
        LegalAction la(PASS_PRIORITY, cands[i].source, cands[i].label);
        la.category = ActionCategory::CHOOSE_REPLACEMENT;
        choices.push_back(la);
    }
    game_log("%s chooses which replacement effect applies next (%zu applicable).\n",
             player_name(chooser).c_str(), eligible.size());
    int pick = InputLogger::instance().get_input(choices);
    return eligible[static_cast<size_t>(pick)];
}

// Dredge (702.52a / 614.1a): an optional, exclusive draw replacement. The drawing
// player may replace the draw with one dredge from their graveyard, or decline.
void dispatch_draw(ReplacementEvent &ev) {
    size_t lib = library_size(ev.affected_player);
    std::vector<Candidate> dredges;
    Entity max_e = global_coordinator.GetMaxIssuedEntity();
    for (Entity e = 0; e < max_e; e++) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location != Zone::GRAVEYARD || z.owner != ev.affected_player) continue;
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        auto &cd = global_coordinator.GetComponent<CardData>(e);
        if (cd.dredge > 0 && static_cast<size_t>(cd.dredge) <= lib) {
            Candidate c;
            c.source = e;
            c.kind = SELF_TAPPED;  // unused for dredge
            c.amount = cd.dredge;
            c.label = "Dredge " + cd.name + " (mill " + std::to_string(cd.dredge) + ")";
            dredges.push_back(c);
        }
    }
    if (dredges.empty()) return;

    // Option 0 is always "draw normally" so the default (index 0) keeps the draw.
    std::vector<LegalAction> actions;
    LegalAction draw_act(PASS_PRIORITY, std::string("Draw a card"));
    draw_act.category = ActionCategory::CHOOSE_REPLACEMENT;
    actions.push_back(draw_act);
    for (const Candidate &c : dredges) {
        LegalAction la(PASS_PRIORITY, c.source, c.label);
        la.category = ActionCategory::CHOOSE_REPLACEMENT;
        actions.push_back(la);
    }
    int choice = InputLogger::instance().get_input(actions);
    if (choice == 0) return;  // chose to draw normally

    const Candidate &chosen = dredges[static_cast<size_t>(choice - 1)];
    ev.draw_replaced = true;
    ev.dredge_source = chosen.source;
    ev.dredge_mill = chosen.amount;
}

}  // namespace

namespace replacement {

void dispatch(ReplacementEvent &ev) {
    if (ev.type == ReplacementEvent::DRAW_CARD) {
        dispatch_draw(ev);
        return;
    }

    // 616.1 / 616.1f: apply one applicable replacement at a time, re-evaluating the
    // remaining applicable effects each pass, until none are left.
    std::set<std::pair<Entity, int>> applied;
    while (true) {
        std::vector<Candidate> cands = collect(ev, applied);
        if (cands.empty()) break;
        size_t pick = (cands.size() == 1) ? 0 : choose_one(ev.affected_player, cands);
        apply_one(ev, cands[pick]);
        applied.insert({cands[pick].source, cands[pick].index});
    }
}

}  // namespace replacement
