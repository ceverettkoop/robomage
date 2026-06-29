#include "saga.h"

#include <cstdint>

#include "cli_output.h"
#include "components/carddata.h"
#include "components/permanent.h"
#include "components/zone.h"
#include "ecs/coordinator.h"
#include "ecs/events.h"
#include "game_queries.h"

extern Coordinator global_coordinator;

bool card_is_saga(const CardData &cd) { return !cd.saga_chapters.empty(); }

void saga_add_lore_counters(Entity saga, int n) {
    if (n <= 0) return;
    if (!global_coordinator.entity_has_component<CardData>(saga)) return;
    if (!global_coordinator.entity_has_component<Permanent>(saga)) return;
    const auto &cd = global_coordinator.GetComponent<CardData>(saga);
    int final_chapter = static_cast<int>(cd.saga_chapters.size());
    if (final_chapter == 0) return;  // not a Saga

    // CR 714.2: add the lore counter(s) via the shared counter store, then fire each chapter the
    // count newly reaches (CR 714.2b: "if the number of lore counters was less than N and became
    // at least N"). add_counters returns the new total.
    int before = get_counters(saga, "LORE");
    int after = add_counters(saga, "LORE", n);
    game_log("%s gets %d lore counter%s (now %d).\n", cd.name.c_str(), n, n == 1 ? "" : "s", after);

    auto &perm = global_coordinator.GetComponent<Permanent>(saga);
    for (int chapter = before + 1; chapter <= after && chapter <= final_chapter; ++chapter) {
        // The chapter ability has triggered but not yet left the stack (CR 714.4): hold off the
        // sacrifice SBA until it resolves.
        perm.saga_chapters_in_flight++;
        Event ev(Events::SAGA_CHAPTER);
        ev.SetParam(Params::ENTITY, saga);
        ev.SetParam(Params::AMOUNT, static_cast<uint32_t>(chapter));
        global_coordinator.SendEvent(ev);
    }
}

void saga_put_precombat_lore_counters(Zone::Ownership active) {
    for (Entity e = 0; e < global_coordinator.GetMaxIssuedEntity(); ++e) {
        if (!is_battlefield_permanent(e, active)) continue;
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        if (!card_is_saga(global_coordinator.GetComponent<CardData>(e))) continue;
        saga_add_lore_counters(e, 1);
    }
}
