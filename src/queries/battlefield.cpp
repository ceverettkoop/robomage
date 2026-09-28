#include "battlefield.h"

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/damage.h"
#include "../components/player.h"
#include "characteristics.h"
#include "filters.h"

bool ability_source_on_battlefield(const Ability &ab) {
    const Entity self = ab.source.get();
    return self != 0 && is_battlefield_permanent(self);
}

bool entered_battlefield_this_turn(long entered_on_turn) {
    return entered_on_turn == static_cast<long>(cur_game.turn_state.turn);
}

std::vector<Entity> battlefield_permanents_scan(Zone::Ownership ctrl) {
    std::vector<Entity> out;
    Entity max_e = global_coordinator.GetMaxIssuedEntity();
    for (Entity e = 0; e < max_e; ++e)
        if (is_battlefield_permanent(e, ctrl)) out.push_back(e);
    return out;
}

int count_battlefield_matching(const std::string &filter_spec, Zone::Ownership controller,
                               Entity source) {
    MatchCtx mctx;
    mctx.controller = controller;  // the "you" reference for YouCtrl/OppCtrl in the spec
    mctx.source = source;          // for source-relative qualifiers (e.g. +Other, sameName)
    int count = 0;
    // Control is enforced by the filter.
    for (Entity e : battlefield_permanents_scan())
        if (permanent_matches_filter(e, filter_spec, mctx)) count++;
    return count;
}

std::vector<Entity> controlled_permanents_matching(
Zone::Ownership player, const std::string &spec, const std::set<Entity> &entities,
Entity exclude_entity) {
    std::vector<Entity> out;
    MatchCtx ctx;
    ctx.controller = player;
    ctx.source = exclude_entity;
    for (auto e : entities)
        if (is_battlefield_permanent(e, player) && permanent_matches_filter(e, spec, ctx))
            out.push_back(e);
    return out;
}

// Strip Permanent/Creature/Damage from a card no longer on the battlefield (or re-entering it as
// a new object, CR 400.7). Every Equipment and Aura attached to it becomes unattached first, so no
// link survives to name the entity once it is a new object or its id is reused (an Equipment
// stays on the battlefield and is logged as unattached, 704.5n; an unattached Aura goes to the
// graveyard, 704.5m, which logs its own line). Contract documented at the declaration in
// battlefield.h.
void strip_permanent_components(Entity entity, const std::set<Entity> &entities) {
    if (global_coordinator.entity_has_component<Permanent>(entity)) {
        // Every attachment link, whatever the attached permanent's zone or phasing: this clears
        // the relationship itself rather than querying the battlefield.
        for (auto e : entities) {
            if (e == entity || !global_coordinator.entity_has_component<Permanent>(e)) continue;
            auto &attached = global_coordinator.GetComponent<Permanent>(e);
            if (attached.equipped_to.lki_entity() != entity) continue;
            attached.equipped_to = ObjectRef{};
            const bool is_aura = global_coordinator.entity_has_component<CardData>(e) &&
                                 !global_coordinator.GetComponent<CardData>(e).enchant_filter.empty();
            if (!is_aura) game_log("%s becomes unattached\n", entity_name(e).c_str());
        }
        global_coordinator.RemoveComponent<Permanent>(entity);
    }
    if (global_coordinator.entity_has_component<Creature>(entity))
        global_coordinator.RemoveComponent<Creature>(entity);
    if (global_coordinator.entity_has_component<Damage>(entity))
        global_coordinator.RemoveComponent<Damage>(entity);
}

// CR 702.131b: Ascend on a permanent — any time its controller controls ten or more
// permanents and doesn't yet have the city's blessing, they get the city's blessing
// for the rest of the game (a one-way latch; never lost once gained, 702.131c). The
// keyword lives on the source CardData, so this also covers non-creature permanents
// that have Ascend. Shared by the SBA preamble (every pass) and by mid-resolution
// readers of the flag (the Condition$ Blessing gate), since "any time" means the
// grant may not lag to the next state-based pass — see the header comment.
void refresh_city_blessing(const std::set<Entity> &entities) {
    bool ascend_a = false, ascend_b = false;
    int perms_a = 0, perms_b = 0;
    for (auto entity : entities) {
        if (!is_battlefield_permanent(entity)) continue;
        Zone::Ownership ctrl = global_coordinator.GetComponent<Permanent>(entity).controller;
        if (ctrl == Zone::PLAYER_A) ++perms_a;
        else if (ctrl == Zone::PLAYER_B) ++perms_b;
        if (global_coordinator.entity_has_component<CardData>(entity)) {
            auto &cd = global_coordinator.GetComponent<CardData>(entity);
            for (const auto &kw : cd.keywords)
                if (kw == "Ascend") {
                    if (ctrl == Zone::PLAYER_A) ascend_a = true;
                    else if (ctrl == Zone::PLAYER_B) ascend_b = true;
                }
        }
    }
    auto grant = [](Entity pe, int perms, const char *who) {
        auto &pl = global_coordinator.GetComponent<Player>(pe);
        if (!pl.has_city_blessing && perms >= 10) {
            pl.has_city_blessing = true;
            game_log("%s gets the city's blessing.\n", who);
        }
    };
    if (ascend_a) grant(cur_game.player_a_entity, perms_a, "Player A");
    if (ascend_b) grant(cur_game.player_b_entity, perms_b, "Player B");
}
