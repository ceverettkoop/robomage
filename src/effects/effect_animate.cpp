#include "effects.h"

#include <algorithm>
#include <string>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/damage.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"

extern Coordinator global_coordinator;

namespace effects {

// Bootstrap a Creature (and Damage) component on a permanent that the Animate extension points
// have turned into a creature (animate_make_creature). The base P/T come from animate_*; the
// granted keywords (animate_added_keywords, e.g. Haste) are merged on. Idempotent: a permanent
// that already has a Creature keeps it (its base P/T are re-asserted to the animate values so a
// 0/0 land-creature stays 0/0 before counters). Reused by the SBA reapply pass so an animated
// land that somehow lost its Creature regains it. Returns nothing; refreshes cached P/T.
void apply_animate_creature_bootstrap(Entity e) {
    if (!global_coordinator.entity_has_component<Permanent>(e)) return;
    auto &perm = global_coordinator.GetComponent<Permanent>(e);
    if (!perm.animate_make_creature && !perm.animate_make_creature_eot) return;
    if (!global_coordinator.entity_has_component<Creature>(e)) {
        // Base P/T: an Animate that set them explicitly (Earthbend) wins; otherwise fall back to
        // the permanent's PRINTED P/T (a Vehicle such as The Fantasticar animates as its printed
        // 4/4; a true land animates as its printed 0/0).
        int base_p = 0, base_t = 0;
        if (perm.animate_set_pt) {
            base_p = perm.animate_power;
            base_t = perm.animate_toughness;
        } else if (global_coordinator.entity_has_component<CardData>(e)) {
            const auto &cd = global_coordinator.GetComponent<CardData>(e);
            base_p = static_cast<int>(cd.power);
            base_t = static_cast<int>(cd.toughness);
        }
        Creature cr;
        cr.base_power = base_p;
        cr.base_toughness = base_t;
        global_coordinator.AddComponent(e, cr);
        if (!global_coordinator.entity_has_component<Damage>(e)) {
            Damage dmg;
            dmg.damage_counters = 0;
            global_coordinator.AddComponent(e, dmg);
        }
    } else if (perm.animate_set_pt) {
        // Re-assert the animated base P/T (the SBA keyword/static rebuild leaves base_* alone,
        // but a freshly added Creature from is_creature_card would carry the printed P/T).
        auto &cr = global_coordinator.GetComponent<Creature>(e);
        cr.base_power = perm.animate_power;
        cr.base_toughness = perm.animate_toughness;
    }
    auto &cr = global_coordinator.GetComponent<Creature>(e);
    for (const auto &kw : perm.animate_added_keywords)
        if (std::find(cr.keywords.begin(), cr.keywords.end(), kw) == cr.keywords.end())
            cr.keywords.push_back(kw);
    // +1/+1 counters already on the permanent (earthbend's counters) sync the cached P/T.
    refresh_counter_pt(e);
}

// DB$ Animate (CR 613 continuous effects). Today's only exercised path is the Guide of Souls
// "it becomes an Angel in addition to its other types" rider: add the parsed Types$ subtypes
// to the targeted permanent for the effect's Duration. Duration$ Permanent (rest of the game)
// is baked onto the Permanent's animate_* fields so the layer system reapplies it every SBE
// pass (the granting ability's source may leave play). The handler is intentionally general —
// extension points for a later land-animation card (set base P/T, grant keywords, add a
// Creature component) are wired through the same animate_* fields and are no-ops until a card
// populates them.
bool animate(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // Defined$ Self (The Fantasticar: "have CARDNAME become an artifact creature") animates the
    // ability's own source; otherwise animate the chosen/inherited target (Guide of Souls).
    Entity tgt = ab.defined_self ? ab.source : ab.target;
    if (tgt == 0 || !is_battlefield_permanent(tgt)) return true;
    auto &perm = global_coordinator.GetComponent<Permanent>(tgt);

    // Duration (CR 613.7 + 514.2). Three variants:
    //   * Duration$ Permanent          — rest-of-game animate_* fields (Guide of Souls).
    //   * Duration$ UntilYourNextTurn  — persists past this cleanup, reverts at the animating
    //                                    player's next untap step (Karn +1). Uses the rest-of-game
    //                                    animate_* fields for persistence PLUS the until-turn marker
    //                                    so the untap revert knows what/when to lapse.
    //   * (default, until end of turn) — the *_eot fields, cleared by the CLEANUP step this turn.
    const bool permanent_dur = ab.animate_duration_permanent;
    const bool until_turn = ab.animate_duration_until_your_next_turn;
    const bool eot = !permanent_dur && !until_turn;
    const bool reverts = !permanent_dur;  // eot and until-turn both lapse later

    // Added types/subtypes (layer 4). Persist on the permanent for the layer reapply, and
    // also apply immediately so a same-resolution reader sees the new type (the SBE pass
    // re-derives them anyway). Skip duplicates so re-animating is idempotent. For a reverting
    // duration we only record/insert types the permanent does NOT already have, so the revert
    // erases exactly what this effect added and never a printed type.
    auto &types_bucket = eot       ? perm.animate_added_types_eot
                         : until_turn ? perm.animate_added_types_until_turn
                                      : perm.animate_added_types;
    for (const auto &t : ab.animate_types) {
        bool already_recorded = false;
        for (const auto &existing : types_bucket)
            if (existing.kind == t.kind && existing.name == t.name) { already_recorded = true; break; }
        const bool already_on_perm = perm.types.count(t) != 0;
        if (reverts) {
            // Record only genuinely-new types so the revert erases exactly the grant.
            if (!already_on_perm && !already_recorded) types_bucket.push_back(t);
        } else if (!already_recorded) {
            types_bucket.push_back(t);
        }
        perm.types.insert(t);
        game_log("%s becomes a%s %s in addition to its other types.\n",
                 perm.name.c_str(), (t.name.size() && std::string("AEIOU").find(t.name[0]) != std::string::npos) ? "n" : "",
                 t.name.c_str());
    }

    // Base P/T (CR 613, layer 7b). AB$ Animate | Power$/Toughness$ sets the animated creature's
    // base P/T — a numeric literal, or a snapshot of a dynamic expr evaluated against the target
    // now (Karn: Targeted$CardManaCost → the target's mana value at resolution). We snapshot at
    // resolution rather than tracking a live characteristic-defining value: mana value is a fixed
    // printed characteristic of a noncreature artifact, so the snapshot equals the CDA reading in
    // practice and matches how the existing animate effect stores base P/T as plain ints. Only
    // applied for persistent/until-turn durations (the rest-of-game animate_* fields persist; no
    // EOT-duration card needs a set P/T, and animate_set_pt is not cleared at cleanup).
    if (ab.animate_has_pt && !eot) {
        int base_p = ab.animate_base_power;
        int base_t = ab.animate_base_toughness;
        if (!ab.animate_power_expr.empty())
            base_p = static_cast<int>(evaluate_dynamic_amount(ab.animate_power_expr, ab.controller, orderer, tgt));
        if (!ab.animate_toughness_expr.empty())
            base_t = static_cast<int>(evaluate_dynamic_amount(ab.animate_toughness_expr, ab.controller, orderer, tgt));
        perm.animate_set_pt = true;
        perm.animate_power = base_p;
        perm.animate_toughness = base_t;
    }

    // Becoming a creature: a card whose Types$ adds Creature to a noncreature permanent (The
    // Fantasticar, a Vehicle; Karn's animated artifact) needs a Creature/Damage component
    // bootstrapped. Flag it on the permanent (permanent/until-turn use the rest-of-game flag so
    // the grant survives this cleanup; EOT uses its own flag so cleanup can revert it).
    bool adds_creature_type = false;
    for (const auto &t : ab.animate_types)
        if (t.kind == TYPE && t.name == "Creature") { adds_creature_type = true; break; }
    if (adds_creature_type) {
        if (eot) perm.animate_make_creature_eot = true;
        else perm.animate_make_creature = true;
    }

    // Record the until-your-next-turn marker so the animating player's untap step reverts exactly
    // this grant (types + creature + base P/T) on their next turn — see revert_until_turn_animates.
    if (until_turn) {
        perm.animate_until_my_turn = true;
        perm.animate_until_turn_controller = ab.controller;
    }

    // Land-animation extension (Earthbend): turn a noncreature permanent into a creature with
    // the baked base P/T, then merge the granted keywords (Haste). For a permanent that is
    // already a creature, just merge the keywords (the layer-6 reapply does the same each pass).
    if (perm.animate_make_creature || perm.animate_make_creature_eot) {
        apply_animate_creature_bootstrap(tgt);
    } else if (global_coordinator.entity_has_component<Creature>(tgt)) {
        // Granted keywords (layer 6) — extension point; populated by future Animate cards via
        // perm.animate_added_keywords (the layer-6 reapply pass merges them onto the rebuilt list).
        auto &cr = global_coordinator.GetComponent<Creature>(tgt);
        for (const auto &kw : perm.animate_added_keywords)
            if (std::find(cr.keywords.begin(), cr.keywords.end(), kw) == cr.keywords.end())
                cr.keywords.push_back(kw);
    }

    // DB$ Animate | Abilities$ <svar> (Urza's Saga chapters I & II): grant the parsed activated
    // ability(ies) to the permanent (CR 613.1f — a lasting "gains [activated ability]"). Only a
    // Duration$ Permanent grant attaches a rest-of-game ability here; the permanent's activated-
    // ability list isn't cleared between SBA passes (only subtype-derived mana abilities are), so a
    // once-added grant persists. Deduped against existing abilities so re-resolving is idempotent.
    if (!ab.animate_granted_abilities.empty() && permanent_dur) {
        for (Ability granted : ab.animate_granted_abilities) {
            granted.source = tgt;
            bool dup = false;
            for (auto &existing : perm.abilities)
                if (existing.identical_activated_ability(granted)) { dup = true; break; }
            if (dup) continue;
            perm.abilities.push_back(granted);
            game_log("%s gains an activated ability.\n", perm.name.c_str());
        }
    }
    return true;
}

// Lapse every "until your next turn" Animate (Karn, the Great Creator +1) created by
// `active_player`, called from that player's untap step (CR 514/613 — the grant ends as their
// next turn begins). Erases exactly the granted types, drops the snapshotted base P/T, and strips
// the bootstrapped Creature/Damage components unless the permanent is a creature by a permanent
// means (its printed face). General over any UntilYourNextTurn Animate, not just Karn.
void revert_until_turn_animates(Zone::Ownership active_player) {
    for (Entity e = 0; e < global_coordinator.GetMaxIssuedEntity(); ++e) {
        if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(e);
        if (!perm.animate_until_my_turn || perm.animate_until_turn_controller != active_player) continue;

        for (const auto &t : perm.animate_added_types_until_turn) perm.types.erase(t);
        perm.animate_added_types_until_turn.clear();
        perm.animate_until_my_turn = false;
        perm.animate_until_turn_controller = Zone::UNKNOWN;
        perm.animate_make_creature = false;
        perm.animate_set_pt = false;
        perm.animate_power = 0;
        perm.animate_toughness = 0;

        bool still_creature = false;
        if (global_coordinator.entity_has_component<CardData>(e))
            still_creature = is_creature_card(global_coordinator.GetComponent<CardData>(e));
        if (!still_creature) {
            if (global_coordinator.entity_has_component<Creature>(e))
                global_coordinator.RemoveComponent<Creature>(e);
            if (global_coordinator.entity_has_component<Damage>(e))
                global_coordinator.RemoveComponent<Damage>(e);
        }
        game_log("%s is no longer a creature.\n", perm.name.c_str());
    }
}

}  // namespace effects
