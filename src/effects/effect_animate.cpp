#include "effects.h"

#include <algorithm>
#include <string>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"

extern Coordinator global_coordinator;

namespace effects {

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

    // Granted keywords (layer 6) — extension point; populated by future Animate cards via
    // perm.animate_added_keywords (the layer-6 reapply pass merges them onto the rebuilt list).
    if (global_coordinator.entity_has_component<Creature>(tgt)) {
        auto &cr = global_coordinator.GetComponent<Creature>(tgt);
        for (const auto &kw : perm.animate_added_keywords)
            if (std::find(cr.keywords.begin(), cr.keywords.end(), kw) == cr.keywords.end())
                cr.keywords.push_back(kw);
    }
    return true;
}

}  // namespace effects
