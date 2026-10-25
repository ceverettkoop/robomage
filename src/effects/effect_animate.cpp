#include "effects.h"

#include <algorithm>
#include <string>
#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/damage.h"
#include "../components/entry_info.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../queries/battlefield.h"
#include "../queries/counters.h"
#include "../queries/delayed_triggers.h"
#include "../queries/entry.h"
#include "../queries/players.h"
#include "../queries/types.h"
#include "../svar_eval.h"

extern Coordinator global_coordinator;

namespace effects {

static void register_sacrifice_at_end_step(const Ability &ab, const std::vector<ObjectRef> &animated);

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
    // Maintain the LIVE type line too (CR 613.1d layer 4): everything that enumerates
    // battlefield permanents by type — targeting (is_legal_target), sacrifice/return cost
    // candidates (controlled_permanents_matching), statics' Affected$ — matches through
    // permanent_matches_filter, which reads Permanent::types, not the Creature component.
    // An Earthbend-style caller sets the make-creature flag directly (no ab.animate_types
    // pass through animate()), so without this insert the animated land is invisible to
    // every "Creature" filter despite fighting in combat. Idempotent (set insert); re-run
    // every SBA pass while the flag holds, so it survives type-line rebuilds. The reverting
    // durations already erase "Creature" via their recorded type buckets (animate() records
    // it as genuinely-new on a noncreature), so nothing leaks past cleanup/until-turn.
    perm.types.insert(Type{TYPE, "Creature"});
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
HandlerResult animate(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    // Defined$ Self (The Fantasticar: "have CARDNAME become an artifact creature") animates the
    // ability's own source; Defined$ Remembered (Sneak Attack: the creature it just put onto the
    // battlefield) every remembered permanent; otherwise the chosen/inherited target (Guide of
    // Souls).
    std::vector<Entity> objects;
    if (ab.def->defined_self) {
        objects.push_back(ab.source.get());
    } else if (ab.def->defined_remembered) {
        for (const ObjectRef &r : cur_game.resolution.memory.remembered) objects.push_back(r.get());
    } else {
        objects.push_back(ab.target.get());
    }
    std::vector<ObjectRef> animated;
    for (Entity tgt : objects) {
        if (tgt == 0 || !global_coordinator.entity_has_component<Zone>(tgt)) continue;
        if (is_battlefield_permanent(tgt)) {
            animate_permanent(ab, tgt);
        } else if (global_coordinator.GetComponent<Zone>(tgt).location == Zone::BATTLEFIELD &&
                   !global_coordinator.entity_has_component<Permanent>(tgt)) {
            // Put onto the battlefield by this resolution, its Permanent not built yet: the grant
            // waits on its entry.
            entry_info(tgt).animates.push_back(ab);
        } else {
            continue;
        }
        animated.push_back(ObjectRef::of(tgt));
    }
    if (ab.def->animate_sacrifice_at_eot && !animated.empty()) register_sacrifice_at_end_step(ab, animated);
    return HandlerResult::DONE_RUN_SUBS;
}

// See declaration in effects.h.
void animate_permanent(const Ability &ab, Entity tgt) {
    auto &perm = global_coordinator.GetComponent<Permanent>(tgt);

    // Duration (CR 613.7 + 514.2). Three variants:
    //   * Duration$ Permanent          — rest-of-game animate_* fields (Guide of Souls).
    //   * Duration$ UntilYourNextTurn  — persists past this cleanup, reverts at the animating
    //                                    player's next untap step (Karn +1). Uses the rest-of-game
    //                                    animate_* fields for persistence PLUS the until-turn marker
    //                                    so the untap revert knows what/when to lapse.
    //   * (default, until end of turn) — the *_eot fields, cleared by the CLEANUP step this turn.
    const bool permanent_dur = ab.def->animate_duration_permanent;
    const bool until_turn = ab.def->animate_duration_until_your_next_turn;
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
    // The effect's timestamp (CR 613.7b), ordering it against static abilities' effects.
    (eot ? perm.animate_timestamp_eot : perm.animate_timestamp) = cur_game.timestamp++;
    for (const auto &t : ab.def->animate_types) {
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
    if (ab.def->animate_has_pt && !eot) {
        int base_p = ab.def->animate_base_power;
        int base_t = ab.def->animate_base_toughness;
        if (!ab.def->animate_power_expr.empty())
            base_p = static_cast<int>(evaluate_amount(ab.def->animate_power_expr, ab.controller, 0, tgt));
        if (!ab.def->animate_toughness_expr.empty())
            base_t = static_cast<int>(evaluate_amount(ab.def->animate_toughness_expr, ab.controller, 0, tgt));
        perm.animate_set_pt = true;
        perm.animate_power = base_p;
        perm.animate_toughness = base_t;
    }

    // Becoming a creature: a card whose Types$ adds Creature to a noncreature permanent (The
    // Fantasticar, a Vehicle; Karn's animated artifact) needs a Creature/Damage component
    // bootstrapped. Flag it on the permanent (permanent/until-turn use the rest-of-game flag so
    // the grant survives this cleanup; EOT uses its own flag so cleanup can revert it).
    bool adds_creature_type = false;
    for (const auto &t : ab.def->animate_types)
        if (t.kind == TYPE && t.name == "Creature") { adds_creature_type = true; break; }
    if (adds_creature_type) {
        if (eot) perm.animate_make_creature_eot = true;
        else perm.animate_make_creature = true;
    }

    // Keywords$ (layer 6): a Duration$ Permanent grant joins the rest-of-game keywords the
    // layer-6 rebuild merges every pass (Sneak Attack's Haste). No vocab card grants keywords for
    // a shorter duration.
    if (permanent_dur)
        for (const std::string &kw : ab.def->animate_keywords)
            if (std::find(perm.animate_added_keywords.begin(), perm.animate_added_keywords.end(), kw) ==
                perm.animate_added_keywords.end()) {
                perm.animate_added_keywords.push_back(kw);
                game_log("%s gains %s.\n", perm.name.c_str(), kw.c_str());
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
    // Duration$ Permanent grant attaches a rest-of-game ability here. The grant is recorded on
    // Permanent::animate_granted_abilities (the persistent store apply_permanent_components
    // re-merges into perm.abilities each SBE pass — so it survives the layer-6 ability-removal
    // strip and comes back once the remover leaves) and also pushed onto perm.abilities now so a
    // same-pass reader sees it immediately. Deduped so re-resolving is idempotent.
    if (!ab.def->animate_granted_abilities.empty() && permanent_dur) {
        for (const AbilityDef &granted_def : ab.def->animate_granted_abilities) {
            Ability granted(&granted_def);
            granted.source = ObjectRef::of(tgt);
            bool recorded = false;
            for (auto &existing : perm.animate_granted_abilities)
                if (existing.identical_activated_ability(&granted_def)) { recorded = true; break; }
            if (!recorded) perm.animate_granted_abilities.push_back(granted);
            bool dup = false;
            for (auto &existing : perm.abilities)
                if (existing.identical_activated_ability(&granted_def)) { dup = true; break; }
            if (dup) continue;
            perm.abilities.push_back(granted);
            game_log("%s gains an activated ability.\n", perm.name.c_str());
        }
    }
}

// AtEOT$ Sacrifice (Sneak Attack): a delayed trigger at the beginning of the next end step, whoever
// is active (CR 603.7), that sacrifices exactly the permanents `animated` still on the battlefield
// as the objects they were (CR 400.7: one that left and returned is a new object and stays).
static void register_sacrifice_at_end_step(const Ability &ab, const std::vector<ObjectRef> &animated) {
    Ability fire_ab(keyed_ability_def("animate_sacrifice_at_eot", [] {
        AbilityDef d;
        d.ability_type = AbilityDef::TRIGGERED;
        d.category = "ChangeZone";
        d.defined_remembered = true;
        d.origin = Zone::BATTLEFIELD;
        d.origins = {Zone::BATTLEFIELD};
        d.destination = Zone::GRAVEYARD;
        return d;
    }));
    fire_ab.restore_remembered_exiled_with = animated;  // resolve() seeds the remembered set to them
    fire_ab.source = ab.source;

    DelayedTrigger dt;
    dt.ability = fire_ab;
    dt.fire_on = Events::END_STEP_BEGAN;
    dt.owner_entity = get_player_entity(ab.controller);
    dt.fire_on_turn = cur_game.turn_state.turn;
    register_delayed_trigger(dt, ab.source);
}

// Lapse the "until your next turn" Animate (Karn, the Great Creator +1) on permanent `e` if
// `active_player` created it, called from that player's untap step (CR 514/613 — the grant ends as
// their next turn begins). Erases exactly the granted types, drops the snapshotted base P/T, and
// strips the bootstrapped Creature/Damage components unless the permanent is a creature by a
// permanent means (its printed face). General over any UntilYourNextTurn Animate, not just Karn.
void revert_until_turn_animate(Entity e, Zone::Ownership active_player) {
    auto &perm = global_coordinator.GetComponent<Permanent>(e);
    if (!perm.animate_until_my_turn || perm.animate_until_turn_controller != active_player) return;

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

}  // namespace effects
