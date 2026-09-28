#ifndef QUERIES_BATTLEFIELD_H
#define QUERIES_BATTLEFIELD_H

#include <set>
#include <string>
#include <vector>
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"

struct Ability;

// ── Battlefield accessors ─────────────────────────────────────────────────────

// True if the entity currently sits on the battlefield.
inline bool on_battlefield(Entity e) {
    return global_coordinator.entity_has_component<Zone>(e) &&
           global_coordinator.GetComponent<Zone>(e).location == Zone::BATTLEFIELD;
}

// True if `e` is a *live* battlefield permanent: it carries a Permanent component,
// its Zone is BATTLEFIELD, and it is not phased out (702.26b — a phased-out permanent
// is treated as though it doesn't exist), optionally controlled by `ctrl` (UNKNOWN =
// any controller). This is the single source of "is this on the battlefield": prefer
// it (or battlefield_permanents() below) over open-coding the
// Permanent+Zone+BATTLEFIELD(+phased)(+controller) check, so the phasing rule lives in
// exactly one place. The only code that should read Permanent::is_phased_out directly
// is the phasing subsystem itself (the untap-step phase-in/skip in game.cpp), the
// rare loop that must still process phased-out permanents (e.g. resetting their cached
// P/T before skipping them when gathering static abilities), and the ML serialization
// (machine_io.cpp's populate_gamestate), which deliberately INCLUDES phased-out
// permanents in the observation with an is_phased_out flag instead of hiding them.
inline bool is_battlefield_permanent(Entity e, Zone::Ownership ctrl = Zone::UNKNOWN) {
    if (!global_coordinator.entity_has_component<Permanent>(e)) return false;
    if (!global_coordinator.entity_has_component<Zone>(e)) return false;
    if (global_coordinator.GetComponent<Zone>(e).location != Zone::BATTLEFIELD) return false;
    auto &perm = global_coordinator.GetComponent<Permanent>(e);
    if (perm.is_phased_out) return false;
    if (ctrl != Zone::UNKNOWN && perm.controller != ctrl) return false;
    return true;
}

// All live battlefield permanents (phased-out excluded), optionally only those
// controlled by `ctrl`. Pass the iterating system's mEntities (or orderer->mEntities).
// Prefer this over re-scanning entities inline when you need the whole set.
inline std::vector<Entity> battlefield_permanents(
    const std::set<Entity> &entities, Zone::Ownership ctrl = Zone::UNKNOWN) {
    std::vector<Entity> out;
    for (auto e : entities)
        if (is_battlefield_permanent(e, ctrl)) out.push_back(e);
    return out;
}

// battlefield_permanents for code that runs outside a system (an SVar evaluation, a count helper)
// and so has no mEntities: scans every issued entity id. Defined in battlefield.cpp.
std::vector<Entity> battlefield_permanents_scan(Zone::Ownership ctrl = Zone::UNKNOWN);

// True if the ability's source is still the object it was when the ability was created (CR
// 400.7) and is a battlefield permanent (phased-in). Defined in battlefield.cpp.
bool ability_source_on_battlefield(const Ability &ab);

// True if a permanent whose Permanent::entered_on_turn is `entered_on_turn` entered the
// battlefield during the current turn. The single "entered this turn" predicate: the
// ThisTurnEntered filter qualifier and the observation's per-permanent entered_this_turn
// flag both read it. Defined in battlefield.cpp (needs cur_game).
bool entered_battlefield_this_turn(long entered_on_turn);

// Count the battlefield permanents matching a Forge `Count$Valid <filter>` spec — the single
// shared implementation behind the SVar evaluator's Count$Valid (svar_eval.h) and presence
// conditions, so the `controller` ("you") reference, the
// `source` reference (for source-relative qualifiers like +Other / sameName), and the
// battlefield/phasing guard are identical on both. `filter_spec` is the bare filter (the text after
// "Count$Valid "); control/type/etc. qualifiers in it are enforced by permanent_matches_filter.
int count_battlefield_matching(const std::string &filter_spec, Zone::Ownership controller,
                               Entity source);

// Battlefield permanents controlled by `player` matching the ';'-delimited `spec` used by
// Sacrifice-a-<type> / Return-a-<type> activation costs (e.g. "Forest;Plains", "Creature",
// "Creature.Other", "Creature.Green"). Drives both Sacrifice-a-<type> and Return-a-<type>
// activation costs: non-empty == the cost is payable (legality), and the list itself is the
// player's choice menu (payment). `exclude_entity` is the cost's source, honoured by a
// `.Other` qualifier in the spec (e.g. Wight of the Reliquary's "Sacrifice another creature").
// Matching runs through the shared permanent_matches_filter so the full qualifier grammar
// (colors, P/T, subtypes, …) is available here too.
std::vector<Entity> controlled_permanents_matching(
Zone::Ownership player, const std::string &spec, const std::set<Entity> &entities,
Entity exclude_entity = 0);

// Strip the battlefield-state components (Permanent/Creature/Damage) from a card that is no
// longer on the battlefield — first unattaching every Equipment and Aura among `entities`
// attached to it (logging each non-Aura as it becomes unattached, CR 704.5n), so no link names
// the old object (CR 400.7) or a later reuse of its id. Shared
// by the state-based off-battlefield strip (apply_permanent_components) and by add_to_zone's battlefield-entry reset: a card that left
// and returned within a single resolution (same-resolution flicker, Ajani's exile-and-return
// transform) re-enters before the state-based pass could strip it, and per CR 400.7 the
// returning card is a NEW object that must not keep its stale tapped/summoning-sickness/
// counter/attachment state. Defined in battlefield.cpp.
void strip_permanent_components(Entity entity, const std::set<Entity> &entities);

// CR 702.131b: Ascend on a permanent is a static ability — "ANY TIME you control ten or
// more permanents and you don't have the city's blessing, you get the city's blessing for
// the rest of the game" (a one-way latch, never lost once gained, 702.131c). Because it
// applies "any time", the grant must be visible IMMEDIATELY when the tenth permanent
// arrives — in particular mid-resolution, between a Token sub-ability creating the 10th
// permanent and a later Condition$ Blessing gate reading the flag (Ocelot Pride's copy
// clause) — not only at the next state-based pass. Re-evaluates and latches the blessing
// for both players. The SBA preamble runs it every pass; any code that READS the blessing
// flag after possibly changing the permanent count should call it first. Pass the
// iterating system's mEntities (or orderer->mEntities). Defined in battlefield.cpp
// (needs cur_game's player entities and game_log).
void refresh_city_blessing(const std::set<Entity> &entities);

#endif /* QUERIES_BATTLEFIELD_H */
