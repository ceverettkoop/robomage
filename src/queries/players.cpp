#include "players.h"

#include <algorithm>
#include <vector>

#include "../classes/game.h"
#include "../components/ability.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/spell.h"
#include "../ecs/coordinator.h"
#include "../str_util.h"
#include "lki.h"

static bool player_target_alternative(const std::string &alt, bool &opponent_only, bool &you_only);

// ── Seats (declared in players.h) ──────────────────────────────────────
Entity get_player_entity(Zone::Ownership player) {
    return (player == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
}

bool revolt_this_turn(Zone::Ownership player) {
    return global_coordinator.GetComponent<Player>(get_player_entity(player))
        .permanent_left_battlefield_this_turn;
}

Zone::Ownership seat_of_player(Entity player_entity) {
    if (player_entity == cur_game.player_a_entity) return Zone::PLAYER_A;
    if (player_entity == cur_game.player_b_entity) return Zone::PLAYER_B;
    return Zone::UNKNOWN;
}

Zone::Ownership priority_seat() {
    return cur_game.priority.player_a_has_priority ? Zone::PLAYER_A : Zone::PLAYER_B;
}

Zone::Ownership active_seat() { return cur_game.turn_state.player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B; }

// ── Player targets (declared in players.h) ─────────────────────────────
// One comma-OR alternative of a ValidTgts$ spec, if it names a player: the seat restriction it
// places (UNKNOWN = either player, else the required relation to `you`). False when it names no
// player (a permanent/card alternative).
static bool player_target_alternative(const std::string &alt, bool &opponent_only, bool &you_only) {
    opponent_only = you_only = false;
    if (alt == "Any") return true;
    size_t sep = alt.find_first_of(".+");
    const std::string head = alt.substr(0, sep);
    if (head == "Opponent") { opponent_only = true; return true; }
    if (head != "Player") return false;
    if (sep == std::string::npos) return true;
    std::string quals = alt.substr(sep + 1);
    std::replace(quals.begin(), quals.end(), '.', '+');
    for (const auto &q : split(quals, '+', /*skip_empty=*/true)) {
        if (q == "Opponent") opponent_only = true;
        else if (q == "You") you_only = true;
    }
    return true;
}

bool target_spec_names_players(const std::string &valid_tgts) {
    bool opp_only = false, you_only = false;
    for (const auto &alt : split(valid_tgts, ',', /*skip_empty=*/true))
        if (player_target_alternative(alt, opp_only, you_only)) return true;
    return false;
}

bool player_matches_target_spec(const std::string &valid_tgts, Entity player, Zone::Ownership you) {
    Zone::Ownership seat = seat_of_player(player);
    if (seat == Zone::UNKNOWN) return false;
    for (const auto &alt : split(valid_tgts, ',', /*skip_empty=*/true)) {
        bool opp_only = false, you_only = false;
        if (!player_target_alternative(alt, opp_only, you_only)) continue;
        if (opp_only && seat != opponent_of(you)) continue;
        if (you_only && seat != you) continue;
        return true;
    }
    return false;
}

// ── Defined$ player resolution (declared in players.h) ─────────────────
Zone::Ownership source_controller(Entity source) {
    if (global_coordinator.entity_has_component<Permanent>(source))
        return global_coordinator.GetComponent<Permanent>(source).controller;
    // A spell's controller is the player who cast it (CR 110.2 / 608.2), not its owner. A spell
    // copy choosing its targets has no Zone yet (CR 707.10), only its Spell.
    if (global_coordinator.entity_has_component<Spell>(source) &&
        global_coordinator.GetComponent<Spell>(source).caster != Zone::UNKNOWN)
        return global_coordinator.GetComponent<Spell>(source).caster;
    if (global_coordinator.entity_has_component<Player>(source)) return seat_of_player(source);
    if (global_coordinator.entity_has_component<Zone>(source)) {
        const auto &z = global_coordinator.GetComponent<Zone>(source);
        // A card that has just entered the battlefield, before its Permanent is built, is
        // controlled by the player recorded on its Zone as it entered.
        if (z.location == Zone::BATTLEFIELD && z.controller != Zone::UNKNOWN) return z.controller;
        return z.owner;
    }
    // A token that ceased to exist (CR 111.7): its last-known controller.
    if (const LastKnownInfo *lki = lki_for(source)) return lki->controller;
    return Zone::UNKNOWN;
}

Zone::Ownership last_known_controller(Entity e) {
    if (global_coordinator.entity_has_component<Zone>(e)) {
        Zone::Ownership c = global_coordinator.GetComponent<Zone>(e).controller;
        if (c != Zone::UNKNOWN) return c;  // still on the battlefield (or wherever Zone records it)
    }
    if (global_coordinator.entity_has_component<Permanent>(e))
        return global_coordinator.GetComponent<Permanent>(e).controller;  // mid-resolution, pre-SBA strip
    if (const LastKnownInfo *lki = lki_for(e)) return lki->controller;     // already left the battlefield
    return Zone::UNKNOWN;
}

Zone::Ownership resolve_defined_player(const Ability &ab) {
    if (ab.defined_you)                 return ab.controller;
    if (ab.defined_each_opponent)       return opponent_of(ab.controller);
    if (ab.defined_targeted_controller)
        return !ab.target.empty() ? last_known_controller(ab.target.lki_entity()) : Zone::UNKNOWN;
    if (ab.defined_triggered_activator) return ab.triggered_activator;
    if (ab.defined_triggered_player)    return ab.triggered_player;
    // TriggeredCardController shares triggered_player storage; bound at fire time (see
    // collect_triggered_abilities' delayed-trigger leave-battlefield path).
    if (ab.defined_triggered_card_controller) return ab.triggered_player;
    return Zone::UNKNOWN;
}
