#include "permissions.h"

#include "../classes/game.h"
#include "../components/carddata.h"
#include "../ecs/coordinator.h"
#include "../systems/rules_modifying.h"
#include "types.h"

// What a row of the registry is asked about.
struct PermissionQuery {
        Entity card;
        const CardData &cd;
        Zone::Ownership player;
};

// A permission: the zone it reaches, whether only the card's owner has it, whether it covers the
// card for the player, and whether it lapses at this turn's cleanup.
struct PermissionRow {
        Permission kind;
        Zone::ZoneValue zone;
        bool owner_only;
        bool (*covers)(const PermissionQuery &q);
        bool (*expires)(const PermissionQuery &q);
};

static bool never_expires(const PermissionQuery &q);
static bool expires_at_cleanup(const PermissionQuery &q);
static bool has_flashback(const PermissionQuery &q);
static bool has_escape(const PermissionQuery &q);
static bool granted_graveyard_cast(const PermissionQuery &q);
static bool graveyard_land_static(const PermissionQuery &q);
static bool exile_grant_covers(const PermissionQuery &q);
static bool exile_grant_expires(const PermissionQuery &q);
static unsigned zones_with_rows();

static const PermissionRow kPermissionRows[] = {
    // kind                      zone             owner only  covers                  expires
    {Permission::FLASHBACK,      Zone::GRAVEYARD, true,       has_flashback,          never_expires},
    {Permission::ESCAPE,         Zone::GRAVEYARD, true,       has_escape,             never_expires},
    {Permission::GRAVEYARD_CAST, Zone::GRAVEYARD, true,       granted_graveyard_cast, expires_at_cleanup},
    {Permission::GRAVEYARD_LAND, Zone::GRAVEYARD, true,       graveyard_land_static,  never_expires},
    {Permission::EXILE_GRANT,    Zone::EXILE,     false,      exile_grant_covers,     exile_grant_expires},
};

static const unsigned kZonesWithRows = zones_with_rows();

static unsigned zones_with_rows() {
    unsigned zones = 0;
    for (const PermissionRow &row : kPermissionRows) zones |= 1u << row.zone;
    return zones;
}

static bool never_expires(const PermissionQuery &) { return false; }

static bool expires_at_cleanup(const PermissionQuery &) { return true; }

static bool has_flashback(const PermissionQuery &q) { return q.cd.has_flashback; }

static bool has_escape(const PermissionQuery &q) { return q.cd.has_escape; }

static bool granted_graveyard_cast(const PermissionQuery &q) {
    return !is_land_card(q.cd) && cur_game.resolved_effects.may_cast_this_turn.count(q.card);
}

static bool graveyard_land_static(const PermissionQuery &q) {
    return is_land_card(q.cd) && rules_mod::may_play_lands_from_graveyard(q.player);
}

static bool exile_grant_covers(const PermissionQuery &q) {
    const Game::ImpulseCastPermission *grant = cur_game.resolved_effects.impulse_cast_permission.find(q.card);
    if (grant == nullptr) return false;
    const Game::ImpulseCastPermission &g = *grant;
    if (g.caster != q.player) return false;
    // A permission for a cast made during a resolution (CR 608.2g) is used only there.
    if (g.during_resolution) return false;
    return !is_land_card(q.cd) || g.allow_land;
}

static bool exile_grant_expires(const PermissionQuery &q) {
    const Game::ImpulseCastPermission &g = *cur_game.resolved_effects.impulse_cast_permission.find(q.card);
    // Mirrors the cleanup expiry in game.cpp: a warp grant lasts while the card stays
    // in exile; an until-the-end-of-your-next-turn grant lapses at a later turn's cleanup
    // whose active player is its caster; every other grant lapses at this cleanup.
    Zone::Ownership active = cur_game.turn_state.player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B;
    return !g.warp && (!g.persist_until_end_of_next_turn ||
                       (g.caster == active && cur_game.turn_state.turn > g.grant_turn));
}

PlayPermissions play_permissions(Entity card, Zone::Ownership player) {
    PlayPermissions out;
    if (!global_coordinator.entity_has_component<Zone>(card)) return out;
    const Zone &zone = global_coordinator.GetComponent<Zone>(card);
    if (!(kZonesWithRows & (1u << zone.location))) return out;
    if (!global_coordinator.entity_has_component<CardData>(card)) return out;
    const PermissionQuery q{card, global_coordinator.GetComponent<CardData>(card), player};
    // A source that outlives this turn's cleanup clears this; it starts true and is only
    // reported when some source applies.
    bool all_expire = true;
    for (const PermissionRow &row : kPermissionRows) {
        if (row.zone != zone.location || (row.owner_only && zone.owner != player) || !row.covers(q)) continue;
        out.sources |= permission_bit(row.kind);
        if (!row.expires(q)) all_expire = false;
    }
    out.expires_this_turn = out.playable() && all_expire;
    return out;
}

std::vector<std::pair<Entity, unsigned>> play_routes(const EntitySet &entities, Zone::Ownership player) {
    std::vector<std::pair<Entity, unsigned>> routes;
    for (Entity e : entities) {
        const unsigned sources = play_permissions(e, player).sources;
        if (sources) routes.emplace_back(e, sources);
    }
    return routes;
}
