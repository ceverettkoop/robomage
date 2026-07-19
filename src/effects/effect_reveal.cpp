#include "effects.h"

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../stable_rng.h"
#include "../components/carddata.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// AB$/DB$ Reveal with Random$ True (Urza's Bauble): the ability's controller looks at one
// card chosen uniformly at random from the target/defined player's HAND (CR 701.16 "look
// at" — the card is seen privately by the controller only; it is NOT a public reveal, so we
// do NOT record it in the belief state / mark_card_revealed, exactly as Mishra's Bauble's
// private library peek does). The look-at has no game-state effect, but it is observable in
// perfect-information / narrative mode via game_log_private, so it is no longer a silent
// no-op. Subabilities (the slowtrip DelayedTrigger draw) still chain afterward.
HandlerResult reveal(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    const PeekParams *pp = std::get_if<PeekParams>(&ab.params);
    if (!pp || !pp->random_from_hand) {
        // No other Reveal mode is modeled yet; fall through to subability chaining as a no-op.
        return HandlerResult::DONE_RUN_SUBS;
    }

    // Resolve which player's hand to look at: the targeted Player entity (ValidTgts$ Player),
    // else the ability's controller.
    Zone::Ownership hand_owner = ab.controller;
    if (global_coordinator.entity_has_component<Player>(ab.target)) {
        hand_owner = (ab.target == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
    }

    std::vector<Entity> hand = orderer->get_hand(hand_owner);
    if (hand.empty()) {
        game_log_private(ab.controller, "%s looks at %s's hand at random: hand is empty.\n",
            player_name(ab.controller).c_str(), player_name(hand_owner).c_str());
        return HandlerResult::DONE_RUN_SUBS;
    }

    // Choose one card at random using the game's seeded RNG (cur_game.gen) so replays are
    // deterministic — the same generator used for library shuffling in orderer.cpp,
    // drawn platform-stably (see stable_rng.h).
    Entity chosen = hand[stable_rand_below(cur_game.gen, hand.size())];
    auto &cd = global_coordinator.GetComponent<CardData>(chosen);
    game_log_private(ab.controller, "%s looks at a random card in %s's hand: %s\n",
        player_name(ab.controller).c_str(), player_name(hand_owner).c_str(), cd.name.c_str());
    return HandlerResult::DONE_RUN_SUBS;
}

bool parse_reveal(Ability &ab, const std::string &key, const std::string &value) {
    if (key == "Random") { effect_params<PeekParams>(ab).random_from_hand = (value == "True"); return true; }
    return false;
}

}  // namespace effects
