#include "effects.h"

#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// RepeatEach over players (Price of Progress): resolve the RepeatSubAbility once per
// player. Each iteration sets cur_game.remembered_entities to that player's entity (so a
// Defined$ Remembered / RememberedPlayerCtrl sub-ability resolves against that player) and
// resolves the parsed sub-ability with its target/controller pinned to that player.
//
// Players are processed in APNAP order (active player, then non-active — CR 608.2g/101.4).
// The card's effect ("deals damage to each player ... they control") is one simultaneous
// event in MTG; here it is dealt player-by-player, which is observationally identical for a
// one-shot instant since no player can respond between the two amounts.
bool repeat_each(Ability &ab, std::shared_ptr<Orderer> orderer) {
    if (ab.repeat_players.empty() || ab.subabilities.empty()) return false;

    Zone::Ownership active = cur_game.player_a_active ? Zone::PLAYER_A : Zone::PLAYER_B;
    Zone::Ownership nonactive = (active == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;
    std::vector<Zone::Ownership> order = {active, nonactive};

    std::vector<Entity> saved_remembered = cur_game.remembered_entities;
    for (Zone::Ownership p : order) {
        Entity pe = (p == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
        if (!global_coordinator.entity_has_component<Player>(pe)) continue;
        cur_game.remembered_entities.clear();
        cur_game.remembered_entities.push_back(pe);
        for (Ability sub : ab.subabilities) {
            sub.source = ab.source;
            sub.controller = ab.controller;
            // Defined$ Remembered points the sub-ability at the looped player.
            if (sub.defined_remembered) sub.target = pe;
            sub.resolve(orderer);
        }
    }
    cur_game.remembered_entities = saved_remembered;
    return false;  // sub-abilities already resolved per-player; suppress default chaining
}

}  // namespace effects
