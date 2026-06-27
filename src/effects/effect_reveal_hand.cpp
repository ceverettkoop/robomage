#include "effects.h"

#include <string>
#include <vector>

#include "../classes/game.h"
#include "../classes/match_state.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// DB$ RevealHand (Thought-Knot Seer; general over Duress/Thoughtseize-style "target player reveals
// their hand"): the targeted player reveals their entire hand to all players (CR 701.16). The
// reveal is a public game action — every revealed card's identity is recorded into the match-scoped
// belief state (mark_card_revealed) so the chooser of a chained selection (e.g. a Chooser$ You
// ChangeZone) sees the actual cards. This effect moves nothing on its own; the subsequent
// sub-ability acts on the now-public hand.
bool reveal_hand(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // Whose hand: the targeted Player entity (ValidTgts$ Opponent/Player). With no player target
    // the effect has nothing to reveal, so fall through (chaining subabilities) as a no-op.
    if (!global_coordinator.entity_has_component<Player>(ab.target)) return true;
    Zone::Ownership hand_owner =
        (ab.target == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;

    std::vector<Entity> hand = orderer->get_hand(hand_owner);
    if (hand.empty()) {
        game_log("%s reveals their hand: it is empty.\n", player_name(hand_owner).c_str());
        return true;
    }

    game_log("%s reveals their hand:\n", player_name(hand_owner).c_str());
    for (auto e : hand) {
        auto &cd = global_coordinator.GetComponent<CardData>(e);
        game_log("  %s\n", cd.name.c_str());
        // The whole hand is now public — record each card's identity in the belief state.
        mark_card_revealed(e, hand_owner);
    }
    return true;
}

}  // namespace effects
