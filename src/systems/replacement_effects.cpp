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
#include "../game_driver.h"
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
    PRODUCE_MANA,   // 614.1 — replaces the mana a tapped permanent produces (Damping Sphere)
    DISCARD_ELSE_GRAVEYARD, // 614.1a self-replacement — pay a discard cost as this card enters, else it goes to its owner's graveyard (Mox Diamond, Chrome Mox)
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
    int tapped_unless_life = 0;     // SELF_TAPPED: "enters tapped unless you pay N life" — pay N to enter untapped instead
    Colors produce_color = COLORLESS; // PRODUCE_MANA: color the production is converted to
    std::string discard_filter;     // DISCARD_ELSE_GRAVEYARD: type filter of the card the owner may discard
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
            // A permanent entering as its DFC back face (a transform DFC's transformed entry, or
            // a modal DFC's back face played from hand) carries the BACK face's self-replacement
            // effects, not the front's. Detect via the live Permanent's transformed flag (set
            // before this dispatch on a later pass) or the one-shot pending_enters_transformed
            // marker (set when the back face is put onto the battlefield, consumed at perm
            // creation). Otherwise the front face's effects apply.
            const std::vector<Effect::Replacement> *reps = &cd.replacement_effects;
            if (cd.backside &&
                ((global_coordinator.entity_has_component<Permanent>(ev.entity) &&
                  global_coordinator.GetComponent<Permanent>(ev.entity).transformed) ||
                 cur_game.pending_enters_transformed.count(ev.entity)))
                reps = &cd.backside->replacement_effects;
            // Self "enters tapped" replacement effect (614.1d / self-replacement 614.15).
            for (size_t i = 0; i < reps->size(); i++) {
                const Effect::Replacement &r = (*reps)[i];
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
                c.tapped_unless_life = r.tapped_unless_life;
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
                if (sa.counter_count_from_delve) {
                    // Delve may exile ANY card (CR 702.66a); the etbCounter rider counts only
                    // the exiles matching the script's ValidExile filter (Murktide Regent:
                    // "each instant and sorcery card exiled with it" → "Instant,Sorcery").
                    MatchCtx dctx;
                    dctx.controller = ev.affected_player;
                    dctx.source = ev.entity;
                    for (Entity ex : cur_game.delve_exiled)
                        if (sa.counter_count_delve_filter.empty() ||
                            card_matches_any(ex, sa.counter_count_delve_filter, dctx))
                            n++;
                }
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
            // Mox Diamond / Chrome Mox (614.1a self-replacement): as this card would enter, its
            // owner may pay an additional discard cost; if not, it goes to its owner's graveyard.
            // The source is the entering card itself (self-replacement), so it is read off ev.entity.
            if (global_coordinator.entity_has_component<CardData>(ev.entity)) {
                auto &cd = global_coordinator.GetComponent<CardData>(ev.entity);
                for (size_t i = 0; i < cd.replacement_effects.size(); i++) {
                    const Effect::Replacement &r = cd.replacement_effects[i];
                    if (r.kind != Effect::Replacement::DISCARD_ELSE_GRAVEYARD) continue;
                    Candidate c;
                    c.source = ev.entity;
                    c.kind = DISCARD_ELSE_GRAVEYARD;
                    c.index = static_cast<int>(i);
                    c.self_replacement = true;
                    c.discard_filter = r.discard_else_filter;
                    c.label = "discard to enter, or go to graveyard";
                    if (!already_applied(applied, c)) out.push_back(c);
                }
            }

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
                bool matches;
                if (r.applies_to_self_only) {
                    // Grim Monolith: "this artifact doesn't untap" — only when the source IS the
                    // permanent being untapped.
                    matches = (e == ev.entity);
                } else {
                    // Choke: subtype-filtered — the subject must have the named (sub)type.
                    matches = false;
                    for (const auto &t : subject.types)
                        if (t.name == r.valid_subtype) { matches = true; break; }
                }
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

    if (ev.type == ReplacementEvent::PRODUCE_MANA) {
        // Damping Sphere: a land (ValidCard$ type) tapped for >= ManaAmount mana produces that
        // much of the replacement color instead. Once one replacement has rewritten the
        // production the event is fully replaced — the result is the same regardless of how many
        // identical effects are present (idempotent), so further passes add nothing and we never
        // prompt for a choice (avoids an input read during mana-payment simulation).
        if (ev.mana_replaced) return out;
        if (!global_coordinator.entity_has_component<Permanent>(ev.entity)) return out;
        auto &src = global_coordinator.GetComponent<Permanent>(ev.entity);
        Entity max_e = global_coordinator.GetMaxIssuedEntity();
        for (Entity e = 0; e < max_e; e++) {
            if (!is_battlefield_permanent(e)) continue;
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            for (size_t i = 0; i < cd.replacement_effects.size(); i++) {
                const Effect::Replacement &r = cd.replacement_effects[i];
                if (r.kind != Effect::Replacement::PRODUCE_MANA) continue;
                if (ev.produced_amount < static_cast<size_t>(r.produce_min_amount)) continue;
                // The producing permanent must match the ValidCard$ type filter (e.g. "Land").
                bool type_ok = false;
                for (const auto &t : src.types)
                    if (t.name == r.produce_valid_type) { type_ok = true; break; }
                if (!type_ok) continue;
                Candidate c;
                c.source = e;
                c.kind = PRODUCE_MANA;
                c.index = static_cast<int>(i);
                c.self_replacement = false;
                c.produce_color = r.produce_replacement_color;
                c.label = global_coordinator.GetComponent<Permanent>(e).name + ": produces colorless";
                if (!already_applied(applied, c)) {
                    // Several identical ProduceMana replacements yield the same {C} production, so
                    // return just one — applying it sets mana_replaced and the next pass collects
                    // none. This keeps the dispatch choice-free (no input read in mana simulation).
                    out.push_back(c);
                    return out;
                }
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
            // "Enters tapped unless you pay N life" (Witch-Blessed Meadow / shock lands): give
            // the controller the choice to pay the life and enter untapped instead. A player can
            // pay life only if they have at least that much (CR 119.4 / 118.8); declining or being
            // unable to pay leaves it entering tapped.
            if (c.tapped_unless_life > 0) {
                Entity pe = (ev.affected_player == Zone::PLAYER_A)
                                ? cur_game.player_a_entity : cur_game.player_b_entity;
                auto &pl = global_coordinator.GetComponent<Player>(pe);
                std::string prompt = "pay " + std::to_string(c.tapped_unless_life) +
                                     " life so it enters untapped";
                if (pl.life_total >= c.tapped_unless_life) {
                    // Latched-answer site (tag SBE_LATCHED): the y/n is a loop-top
                    // pending decision. The dispatch runs inside the SBE
                    // apply_permanent_components pass BEFORE anything about the
                    // entering permanent is mutated, so on resume the re-run
                    // re-reaches the same entity, the re-dispatch re-derives this
                    // same single self-replacement candidate (616.1a admits only
                    // self-replacements, and no vocab card carries two), and the
                    // re-ask consumes the latch. The armed menu is byte-identical
                    // to request_optional_yesno's Decline/Accept pair.
                    uint64_t key = pq_key(SbeSite::ETB_TAPPED_LIFE,
                                          ev.affected_player == Zone::PLAYER_A,
                                          ev.entity, 2);
                    int choice = -1;
                    if (!pq_take_latched(key, &choice)) {
                        if (in_main_loop()) {
                            std::vector<LegalAction> yn;
                            LegalAction decline(PASS_PRIORITY, std::string("Decline: ") + prompt);
                            decline.category = ActionCategory::OPTIONAL_YESNO;
                            decline.option_ordinal = 0;  // 0 = decline
                            yn.push_back(decline);
                            LegalAction accept(PASS_PRIORITY, std::string("Accept: ") + prompt);
                            accept.category = ActionCategory::OPTIONAL_YESNO;
                            accept.option_ordinal = 1;  // 1 = accept
                            yn.push_back(accept);
                            // Park the choice and suspend mid-apply: dispatch()
                            // and its SBE caller early-return cooperatively
                            // before the Permanent is created.
                            pq_arm_sbe(key, std::move(yn), ev.affected_player,
                                       /*decision_source=*/0);
                            return;
                        }
                        // Blocking fallback for an SBE call outside the main loop
                        // (defensive only since the Batch 13 pregame gate).
                        choice = request_optional_yesno(ev.affected_player, prompt) ? 1 : 0;
                    }
                    if (choice == 1) {
                        pl.life_total -= c.tapped_unless_life;
                        pl.life_lost_this_turn += c.tapped_unless_life;  // CR 119.4: paying life is losing life
                        game_log("%s pays %d life.\n", player_name(ev.affected_player).c_str(),
                                 c.tapped_unless_life);
                        break;  // enters untapped
                    }
                }
            }
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
        case PRODUCE_MANA:
            ev.produced_color = c.produce_color;
            ev.mana_replaced = true;
            break;
        case PREVENT_ETB: {
            ev.prevented = true;
            std::string name = global_coordinator.GetComponent<CardData>(ev.entity).name;
            game_log("%s can't enter the battlefield from that zone.\n", name.c_str());
            break;
        }
        case DISCARD_ELSE_GRAVEYARD: {
            // Mox Diamond / Chrome Mox: offer the owner an optional additional cost — discard a
            // card matching the filter (a land) as this permanent enters. If they do, it enters
            // normally (the discard is performed by the caller via ev.pending_discard); if they
            // don't (or hold no matching card), it goes to its owner's graveyard instead.
            std::string self_name = global_coordinator.GetComponent<CardData>(ev.entity).name;
            MatchCtx dctx;
            dctx.controller = ev.affected_player;
            dctx.source = ev.entity;
            std::vector<Entity> discardable;
            Entity max_e = global_coordinator.GetMaxIssuedEntity();
            for (Entity e = 0; e < max_e; e++) {
                if (e == ev.entity) continue;  // the entering card is not in hand
                if (!global_coordinator.entity_has_component<Zone>(e)) continue;
                auto &z = global_coordinator.GetComponent<Zone>(e);
                if (z.location != Zone::HAND || z.owner != ev.affected_player) continue;
                if (!global_coordinator.entity_has_component<CardData>(e)) continue;
                if (card_matches_any(e, c.discard_filter, dctx)) discardable.push_back(e);
            }
            if (discardable.empty()) {
                // Can't pay the cost — put it into its owner's graveyard (614.1a).
                ev.destination = Zone::GRAVEYARD;
                game_log("%s has no %s to discard and is put into its owner's graveyard.\n",
                         self_name.c_str(), c.discard_filter.c_str());
                break;
            }
            std::vector<LegalAction> choices;
            for (Entity e : discardable) {
                auto &cd = global_coordinator.GetComponent<CardData>(e);
                LegalAction la(PASS_PRIORITY, e,
                               "Discard " + cd.name + " (put " + self_name + " onto the battlefield)");
                la.category = ActionCategory::CHOOSE_CARD;
                choices.push_back(la);
            }
            LegalAction decline(PASS_PRIORITY,
                                std::string("Don't discard (put ") + self_name + " into graveyard)");
            decline.category = ActionCategory::CHOOSE_CARD;
            choices.push_back(decline);
            // Blocking decision seated on the owner — this MOVE_TO_ZONE dispatch fires inside
            // Orderer::add_to_zone, the one remaining blocking replacement site (see choose_one).
            bool prev_priority = cur_game.player_a_has_priority;
            cur_game.player_a_has_priority = (ev.affected_player == Zone::PLAYER_A);
            int pick = InputLogger::instance().get_input(choices);
            cur_game.player_a_has_priority = prev_priority;
            if (pick >= 0 && pick < static_cast<int>(discardable.size())) {
                ev.pending_discard = discardable[static_cast<size_t>(pick)];  // caller discards it
                game_log("%s discards %s.\n", player_name(ev.affected_player).c_str(),
                         global_coordinator.GetComponent<CardData>(ev.pending_discard).name.c_str());
                // destination stays BATTLEFIELD — it enters normally
            } else {
                ev.destination = Zone::GRAVEYARD;
                game_log("%s is put into its owner's graveyard.\n", self_name.c_str());
            }
            break;
        }
    }
}

// Count of 616.1 multi-replacement prompts emitted (exposed via
// replacement::choose_one_prompt_count for reachability measurement).
size_t g_choose_one_prompts = 0;

// 616.1: the affected player chooses ONE of several applicable replacement effects.
// 616.1a gates the choice to self-replacement effects first when any apply. Returns
// the index into `cands` of the chosen effect.
//
// RESIDUAL (snapshot-safe project, Batch 14): this prompt is the one remaining
// blocking mid-flow decision family. It fires only when >= 2 replacement
// effects apply to ONE event simultaneously. Reachability against the current
// vocab/decks:
//   - MOVE_TO_ZONE -> GRAVEYARD: two EXILE_INSTEAD sources in play at once
//     (wrb_energy runs 4x Leyline of the Void; doomsday runs 2x Dauthi
//     Voidwalker). The outcome of every choice is identical (exile the card;
//     void_countered records no source), only the prompt itself differs.
//   - MOVE_TO_ZONE -> BATTLEFIELD: two PREVENT_ETB / EXILE_INSTEAD_OF_ETB
//     sources (wrb_energy runs 2x Containment Priest).
//   - UNTAP: two SKIP_UNTAP sources matching one permanent (gw_maverick runs
//     2x Choke; both choices are identical in outcome).
//   - ENTERS_BATTLEFIELD: needs >= 2 SELF-replacements on one entering card
//     (616.1a filters non-self out) — no vocab card has two, so unreachable.
// It was NOT converted because the MOVE_TO_ZONE dispatch fires inside
// Orderer::add_to_zone, which is called from every family (resolution
// handlers mid-mutation, SBE death sweeps, cost payments, combat damage) —
// suspending there requires threading a suspension result through every
// add_to_zone caller, far beyond a safe end-of-project change. Conversion
// recipe when wanted: (1) UNTAP context — a PendingUntapRT mini-frame over
// advance_step's untap loop, exactly like PendingDrawRT; (2) ENTERS/SBE
// context — SBE_LATCHED (pq_key over the entering entity + menu size);
// (3) MOVE_TO_ZONE — make add_to_zone return a "suspended" status and convert
// its callers family by family, starting with the SBE death sweep (latched)
// and resolution handlers (frame rt), leaving cost-payment moves (already
// inside PendingCast/PendingActivation flows) last. Until then the prompt
// blocks inline (safe=0), counted by g_choose_one_prompts.
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
    g_choose_one_prompts++;
    // Point priority at the chooser so the query routes/observes/records from
    // their perspective (they may not be the priority holder).
    bool prev_priority = cur_game.player_a_has_priority;
    cur_game.player_a_has_priority = (chooser == Zone::PLAYER_A);
    int pick = InputLogger::instance().get_input(choices);
    cur_game.player_a_has_priority = prev_priority;
    return eligible[static_cast<size_t>(pick)];
}

// Dredge (702.52a / 614.1a): an optional, exclusive draw replacement. The drawing
// player may replace the draw with one dredge from their graveyard, or decline.
// BLOCKING form, kept for the unconverted draw sites (mulligan redraws, where a
// dredge card cannot yet be in the graveyard); the turn-based draw and the
// resolution-time draw handlers ask the same menu through the suspension
// framework instead (resume_pending_draws / draw_n_with_replacements).
void dispatch_draw(ReplacementEvent &ev) {
    std::vector<replacement::DrawReplacementOption> opts;
    std::vector<LegalAction> actions =
        replacement::collect_draw_replacements(ev.affected_player, &opts);
    if (actions.empty()) return;

    // The drawing player makes the dredge decision — route/observe/record from
    // their perspective, which may differ from the current priority holder (a
    // draw can be forced by an opponent's effect).
    bool prev_priority = cur_game.player_a_has_priority;
    cur_game.player_a_has_priority = (ev.affected_player == Zone::PLAYER_A);
    int choice = InputLogger::instance().get_input(actions);
    cur_game.player_a_has_priority = prev_priority;
    if (choice == 0) return;  // chose to draw normally

    const replacement::DrawReplacementOption &chosen = opts[static_cast<size_t>(choice - 1)];
    ev.draw_replaced = true;
    ev.dredge_source = chosen.source;
    ev.dredge_mill = chosen.mill;
}

}  // namespace

namespace replacement {

std::vector<LegalAction> collect_draw_replacements(Zone::Ownership player,
                                                   std::vector<DrawReplacementOption> *opts) {
    opts->clear();
    std::vector<LegalAction> actions;
    size_t lib = library_size(player);
    Entity max_e = global_coordinator.GetMaxIssuedEntity();
    for (Entity e = 0; e < max_e; e++) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location != Zone::GRAVEYARD || z.owner != player) continue;
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        auto &cd = global_coordinator.GetComponent<CardData>(e);
        if (cd.dredge > 0 && static_cast<size_t>(cd.dredge) <= lib)
            opts->push_back({e, cd.dredge});
    }
    if (opts->empty()) return actions;

    // Option 0 is always "draw normally" so the default (index 0) keeps the draw.
    LegalAction draw_act(PASS_PRIORITY, std::string("Draw a card"));
    draw_act.category = ActionCategory::CHOOSE_REPLACEMENT;
    actions.push_back(draw_act);
    for (const DrawReplacementOption &o : *opts) {
        auto &cd = global_coordinator.GetComponent<CardData>(o.source);
        LegalAction la(PASS_PRIORITY, o.source,
                       "Dredge " + cd.name + " (mill " + std::to_string(o.mill) + ")");
        la.category = ActionCategory::CHOOSE_REPLACEMENT;
        actions.push_back(la);
    }
    return actions;
}

int draw_count_bonus(Zone::Ownership player) {
    int bonus = 0;
    Entity max_e = global_coordinator.GetMaxIssuedEntity();
    for (Entity e = 0; e < max_e; e++) {
        if (!is_battlefield_permanent(e, player)) continue;
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        const auto &cd = global_coordinator.GetComponent<CardData>(e);
        for (const auto &r : cd.replacement_effects) {
            if (r.kind != Effect::Replacement::DRAW_ADD) continue;
            // Optional CheckSVar condition ("one or fewer cards in hand"): evaluate the resolved
            // Count$ expression for the drawing player and test it against the comparator. An empty
            // expression means the additive draw is unconditional.
            if (!r.draw_condition_count_expr.empty()) {
                int val = evaluate_sa_svar(r.draw_condition_count_expr, player, e);
                if (!compare_svar(val, r.draw_condition_compare)) continue;
            }
            bonus += r.draw_add;
        }
    }
    return bonus;
}

size_t choose_one_prompt_count() { return g_choose_one_prompts; }

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
        // The tapped-unless-life y/n parked a pending decision mid-apply
        // (SBE_LATCHED): bail without recording anything — the caller discards
        // `ev`, and the resumed SBE pass re-runs this whole dispatch from
        // scratch and consumes the latch at the same ask. Gated on the query
        // being UNANSWERED (the arm we just created): an answered latch still
        // in flight belongs to a site downstream of a resumed pass and must
        // not truncate an unrelated dispatch.
        if (cur_game.pending_query.active && !cur_game.pending_query.answered) return;
        applied.insert({cands[pick].source, cands[pick].index});
    }
}

}  // namespace replacement
