#include "damage.h"

#include <algorithm>

#include "../classes/game.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/spell.h"
#include "../components/token.h"
#include "battlefield.h"
#include "characteristics.h"
#include "keywords.h"
#include "lki.h"
#include "players.h"

static const LastKnownInfo *departed_damage_source_lki(Entity source);

// ── Damage-source characteristics (declared in damage.h) ──────────────
// The last-known snapshot a damage source deals damage with: set only when the source is neither
// on the battlefield nor a spell on the stack, i.e. it left the battlefield before the effect had
// it deal damage (CR 702.15c, 702.2e).
static const LastKnownInfo *departed_damage_source_lki(Entity source) {
    if (on_battlefield(source) || global_coordinator.entity_has_component<Spell>(source))
        return nullptr;
    return departed_lki_for(source);
}

bool damage_source_has_keyword(Entity source, const char *kw) {
    if (on_battlefield(source)) return permanent_has_keyword(source, kw);
    if (const LastKnownInfo *lki = departed_damage_source_lki(source))
        return std::find(lki->keywords.begin(), lki->keywords.end(), kw) != lki->keywords.end();
    if (global_coordinator.entity_has_component<CardData>(source)) {
        for (const auto &k : active_face(source, global_coordinator.GetComponent<CardData>(source)).keywords)
            if (k == kw) return true;
        return false;
    }
    if (global_coordinator.entity_has_component<Token>(source))
        for (const auto &k : global_coordinator.GetComponent<Token>(source).keywords)
            if (k == kw) return true;
    return false;
}

Zone::Ownership damage_source_controller(Entity source) {
    if (const LastKnownInfo *lki = departed_damage_source_lki(source))
        if (lki->controller != Zone::UNKNOWN) return lki->controller;
    return source_controller(source);
}

bool player_protected_from_source(Entity player_entity, Entity /*source*/) {
    if (cur_game.resolved_effects.player_protection_from_everything.empty()) return false;
    Zone::Ownership prot =
        (player_entity == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
    for (const auto &p : cur_game.resolved_effects.player_protection_from_everything) {
        if (p.player != prot) continue;
        // CR 702.16: "protection from everything" is protection from ALL sources, including the
        // protected player's OWN sources — not only an opponent's. So the grant prevents the
        // damage/targeting regardless of who controls the source.
        return true;
    }
    return false;
}

bool permanent_protected_from_colored_spell_source(Entity perm_target, Entity source) {
    if (!global_coordinator.entity_has_component<Creature>(perm_target)) return false;
    if (!has_protection_from_colored_spells(global_coordinator.GetComponent<Creature>(perm_target)))
        return false;
    // The source must be a spell on the stack (not a creature dealing combat damage, nor an
    // activated/triggered ability) and one or more colors (CR 702.16a: the quality is a colored
    // spell). A colorless spell's damage is not prevented.
    if (!global_coordinator.entity_has_component<Spell>(source)) return false;
    return !is_colorless(source);
}
