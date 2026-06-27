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
    if (!perm.animate_make_creature) return;
    if (!global_coordinator.entity_has_component<Creature>(e)) {
        Creature cr;
        cr.base_power = perm.animate_set_pt ? perm.animate_power : 0;
        cr.base_toughness = perm.animate_set_pt ? perm.animate_toughness : 0;
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
    (void)orderer;
    Entity tgt = ab.target;
    if (tgt == 0 || !is_battlefield_permanent(tgt)) return true;
    auto &perm = global_coordinator.GetComponent<Permanent>(tgt);

    // Added types/subtypes (layer 4). Persist on the permanent for the layer reapply, and
    // also apply immediately so a same-resolution reader sees the new type (the SBE pass
    // re-derives them anyway). Skip duplicates so re-animating is idempotent.
    for (const auto &t : ab.animate_types) {
        bool already = false;
        for (const auto &existing : perm.animate_added_types)
            if (existing.kind == t.kind && existing.name == t.name) { already = true; break; }
        if (!already) perm.animate_added_types.push_back(t);
        perm.types.insert(t);
        game_log("%s becomes a%s %s in addition to its other types.\n",
                 perm.name.c_str(), (t.name.size() && std::string("AEIOU").find(t.name[0]) != std::string::npos) ? "n" : "",
                 t.name.c_str());
    }

    // Land-animation extension (Earthbend): turn a noncreature permanent into a creature with
    // the baked base P/T, then merge the granted keywords (Haste). For a permanent that is
    // already a creature, just merge the keywords (the layer-6 reapply does the same each pass).
    if (perm.animate_make_creature) {
        apply_animate_creature_bootstrap(tgt);
    } else if (global_coordinator.entity_has_component<Creature>(tgt)) {
        // Granted keywords (layer 6) — extension point; populated by future Animate cards via
        // perm.animate_added_keywords (the layer-6 reapply pass merges them onto the rebuilt list).
        auto &cr = global_coordinator.GetComponent<Creature>(tgt);
        for (const auto &kw : perm.animate_added_keywords)
            if (std::find(cr.keywords.begin(), cr.keywords.end(), kw) == cr.keywords.end())
                cr.keywords.push_back(kw);
    }
    return true;
}

}  // namespace effects
