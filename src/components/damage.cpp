#include "damage.h"
#include "creature.h"
#include "permanent.h"
#include "player.h"
#include "zone.h"
#include "../cli_output.h"
#include "../classes/game.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../queries/characteristics.h"
#include "../queries/counters.h"
#include "../queries/damage.h"
#include "../queries/player_resources.h"
#include "../queries/players.h"
#include "../mana_system.h"

#include <string>

extern Game cur_game;

static std::string recipient_name(Entity recipient);
static bool damage_prevented(Entity source, Entity recipient, size_t amount, bool is_combat);
static void apply_damage_results(Entity source, Entity recipient, size_t amount, bool is_combat);

bool can_be_dealt_damage(Entity e) {
    Coordinator &coord = Coordinator::global();
    return coord.entity_has_component<Player>(e) || coord.entity_has_component<Damage>(e) ||
           is_planeswalker_permanent(e);
}

size_t deal_damage(Entity source, Entity recipient, size_t amount, bool is_combat) {
    Coordinator &coord = Coordinator::global();
    // CR 120.8: a source that would deal 0 damage deals no damage at all.
    if (amount == 0 || !can_be_dealt_damage(recipient)) return 0;
    // CR 120.4b: damage is dealt as modified by prevention effects.
    if (damage_prevented(source, recipient, amount, is_combat)) return 0;

    // CR 120.4c: the dealt damage is processed into its results.
    apply_damage_results(source, recipient, amount, is_combat);

    // CR 120.3f / 702.15b: damage from a lifelink source — combat or not, to any recipient —
    // causes the source's controller to gain that much life.
    if (damage_source_has_keyword(source, "Lifelink")) {
        Zone::Ownership ctrl = damage_source_controller(source);
        Entity ctrl_entity = get_player_entity(ctrl);
        if (coord.entity_has_component<Player>(ctrl_entity)) {
            player_gain_life(ctrl_entity, static_cast<int32_t>(amount));
            game_log("%s%s gains %zu life (lifelink)\n", is_combat ? "  " : "",
                     player_name(ctrl).c_str(), amount);
        }
    }

    // CR 120.4d: the damage event.
    if (is_combat && coord.entity_has_component<Player>(recipient)) {
        Event ev(Events::COMBAT_DAMAGE_TO_PLAYER);
        ev.SetParam(Params::ENTITY, source);
        ev.SetParam(Params::PLAYER, recipient);
        ev.SetParam(Params::AMOUNT, static_cast<uint32_t>(amount));
        coord.SendEvent(ev);
    }
    return amount;
}

// Display name of a damage recipient: a player's seat name, else the permanent's name.
static std::string recipient_name(Entity recipient) {
    if (Coordinator::global().entity_has_component<Player>(recipient))
        return player_name(seat_of_player(recipient));
    return entity_name(recipient);
}

// True (and logged) when a prevention effect stops all of this damage (CR 615, 702.16e).
static bool damage_prevented(Entity source, Entity recipient, size_t amount, bool is_combat) {
    Coordinator &coord = Coordinator::global();
    const char *indent = is_combat ? "  " : "";
    // CR 615: a combat-damage prevention shield (Maze of Ith) on the source or the recipient.
    if (is_combat && cur_game.combat_damage_prevented(source, recipient)) {
        game_log("%s%zu combat damage from %s to %s is prevented\n", indent, amount,
                 entity_name(source).c_str(), recipient_name(recipient).c_str());
        return true;
    }
    // CR 702.16e: a player with protection from everything (The One Ring) isn't dealt damage.
    if (coord.entity_has_component<Player>(recipient)) {
        if (player_protected_from_source(recipient, source)) {
            game_log("%s%s has protection from everything — %zu damage prevented\n", indent,
                     recipient_name(recipient).c_str(), amount);
            return true;
        }
        return false;
    }
    // CR 702.16e: a creature with protection from a quality of the source (Scryb Ranger's
    // protection from blue).
    if (coord.entity_has_component<Creature>(recipient) &&
        has_protection_from(coord.GetComponent<Creature>(recipient), source)) {
        game_log("%s%s has protection from %s — %zu damage prevented\n", indent,
                 recipient_name(recipient).c_str(), entity_name(source).c_str(), amount);
        return true;
    }
    // CR 702.16e, protection from colored spells (Emrakul): a colored spell source deals it no
    // damage. Combat and ability damage have a non-spell source, so they are never stopped here.
    if (permanent_protected_from_colored_spell_source(recipient, source)) {
        game_log("%s%s has protection from colored spells — %zu damage prevented\n", indent,
                 recipient_name(recipient).c_str(), amount);
        return true;
    }
    return false;
}

// The results of dealt damage by recipient kind (CR 120.3a-e), logged with the damage. A
// planeswalker that is also a creature gets both its loyalty loss and its creature result.
static void apply_damage_results(Entity source, Entity recipient, size_t amount, bool is_combat) {
    Coordinator &coord = Coordinator::global();
    const char *indent = is_combat ? "  " : "";
    const std::string src_name = entity_name(source);
    const std::string rcp_name = recipient_name(recipient);
    const bool infect = damage_source_has_keyword(source, "Infect");
    if (coord.entity_has_component<Player>(recipient)) {
        auto &pl = coord.GetComponent<Player>(recipient);
        if (infect) {
            // CR 120.3b / 702.90b: poison counters instead of life loss.
            pl.add_counters("POISON", static_cast<int>(amount));
            game_log("%s%s deals %zu damage to %s (poison now %d)\n", indent, src_name.c_str(),
                     amount, rcp_name.c_str(), pl.counter_count("POISON"));
        } else {
            // CR 120.3a: a loss of that much life, through the shared helper so Spectacle's
            // life_lost_this_turn stays in sync with life_total.
            player_lose_life(recipient, static_cast<int32_t>(amount));
            game_log("%s%s deals %zu damage to %s (now at %d life)\n", indent, src_name.c_str(),
                     amount, rcp_name.c_str(), pl.life_total);
        }
        return;
    }
    if (is_planeswalker_permanent(recipient)) {
        // CR 120.3c / 306.8: loyalty counters removed.
        damage_planeswalker(recipient, amount);
        game_log("%s%s deals %zu damage to %s (loyalty now %d)\n", indent, src_name.c_str(), amount,
                 rcp_name.c_str(), get_counters(recipient, "LOYALTY"));
    }
    if (!coord.entity_has_component<Damage>(recipient)) return;
    if (infect || damage_source_has_keyword(source, "Wither")) {
        // CR 120.3d / 702.80a / 702.90c: -1/-1 counters instead of marked damage.
        add_counters(recipient, "M1M1", static_cast<int>(amount));
        game_log("%s%s deals %zu damage to %s as -1/-1 counters\n", indent, src_name.c_str(),
                 amount, rcp_name.c_str());
    } else {
        // CR 120.3e: damage marked on the creature.
        coord.GetComponent<Damage>(recipient).damage_counters += amount;
        game_log("%s%s deals %zu damage to %s\n", indent, src_name.c_str(), amount,
                 rcp_name.c_str());
    }
    // CR 702.2b: a creature dealt damage by a deathtouch source is destroyed by the state-based
    // action whatever form the damage's result took. The flag lasts until cleanup clears damage.
    if (damage_source_has_keyword(source, "Deathtouch"))
        coord.GetComponent<Damage>(recipient).has_deathtouch_damage = true;
}
