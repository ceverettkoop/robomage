#include "damage.h"
#include "creature.h"
#include "permanent.h"
#include "player.h"
#include "zone.h"
#include "../cli_output.h"
#include "../classes/game.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../mana_system.h"

extern Game cur_game;

static bool source_has_keyword(Entity source, const char *kw) {
    Coordinator &coord = Coordinator::global();
    if (!coord.entity_has_component<Creature>(source)) return false;
    auto &cr = coord.GetComponent<Creature>(source);
    for (const auto &k : cr.keywords)
        if (k == kw) return true;
    return false;
}

//returns "was damage dealt this way"
bool deal_damage(Entity source, Entity target, size_t amount) {

    Coordinator& coordinator = Coordinator::global();

    //does nothing if target cannot be damaged and/or no longer is valid
    if(!coordinator.entity_has_component<Damage>(target)){
        return false;
    }

    if (coordinator.entity_has_component<Creature>(target)) {
        if (has_protection_from(coordinator.GetComponent<Creature>(target), source))
            return false;
        // Protection from colored spells, damage facet (Emrakul, CR 702.16d): a permanent with
        // this protection isn't dealt damage by a colored spell source. Backstop at the damage
        // chokepoint — the targeted DealDamage effect short-circuits this before calling here, so
        // this covers any other path (a future non-targeted "deal damage to each creature" spell).
        // Combat damage has a creature source (not a spell), so it is never prevented here.
        if (permanent_protected_from_colored_spell_source(target, source)) return false;
    }

    auto &dmg = coordinator.GetComponent<Damage>(target);
    dmg.damage_counters += amount;
    // Deathtouch flag propagates per-creature until cleanup clears damage.
    if (amount > 0 && source_has_keyword(source, "Deathtouch"))
        dmg.has_deathtouch_damage = true;
    return true;
}

bool deal_damage_to_player(Entity source, Entity player_entity, size_t amount) {
    Coordinator &coord = Coordinator::global();
    if (!coord.entity_has_component<Player>(player_entity)) return false;

    // Protection from everything for a player (CR 702.16d; The One Ring): the protected player
    // isn't dealt damage by a source an opponent controls. Enforced here at the effect-damage
    // chokepoint (the combat-damage path checks the same shared predicate).
    if (amount > 0 && player_protected_from_source(player_entity, source)) {
        Zone::Ownership prot =
            (player_entity == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
        game_log("%s has protection from everything — %zu damage prevented\n",
                 player_name(prot).c_str(), amount);
        return false;
    }

    // Damage dealt to a player is a loss of that much life (CR 120.3); routed through the
    // shared helper so Spectacle's life_lost_this_turn tracker stays in sync with life_total.
    player_lose_life(player_entity, static_cast<int32_t>(amount));

    // Lifelink: the source's controller gains life equal to the damage dealt.
    if (amount > 0 && source_has_keyword(source, "Lifelink") &&
        coord.entity_has_component<Permanent>(source)) {
        Zone::Ownership ctrl = coord.GetComponent<Permanent>(source).controller;
        Entity ctrl_entity = get_player_entity(ctrl);
        if (coord.entity_has_component<Player>(ctrl_entity)) {
            player_gain_life(ctrl_entity, static_cast<int32_t>(amount));
            game_log("%s gains %zu life (lifelink)\n", player_name(ctrl).c_str(), amount);
        }
    }
    return true;
}