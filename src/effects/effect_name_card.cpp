#include "effects.h"

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../classes/gamestate.h"
#include "../cli_output.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../name_card_choices.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// SP$/DB$ NameCard (Cabal Therapy): the ability's controller chooses a card name
// (CR 201.4). The chosen name is recorded in cur_game.named_card so a chained
// Card.NamedCard sub-ability — here a RevealDiscardAll discard that makes a player discard
// every copy of the named card — can reference it.
//
// A player may name any card. We restrict the offered candidate set to the distinct
// nonland vocab cards owned by the spell's target player (the player who will discard) —
// their whole deck, not just their hand — which is the only sensible, decidable candidate
// set for the engine and mirrors the Disruptor Flute ETB name-a-card decision. The shared
// build_name_card_choices() helper (also used by Disruptor Flute in state_manager_statics.cpp)
// builds that menu; the ValidCards$ Card.nonLand filter is honored by passing exclude_lands=true.
bool name_card(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // The card name is being chosen for the benefit of the chained discard, which targets a
    // player. Offer names from that player's owned cards; fall back to the controller's
    // opponent if no target player is set.
    Zone::Ownership name_owner;
    if (ab.target != 0 && global_coordinator.entity_has_component<Player>(ab.target)) {
        name_owner = (ab.target == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
    } else {
        name_owner = (ab.controller == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;
    }

    std::vector<std::string> names;
    std::vector<LegalAction> name_choices =
        build_name_card_choices(orderer->mEntities, name_owner, /*exclude_lands=*/true, names);

    if (name_choices.empty()) {
        // No nameable card; the chained Card.NamedCard discard will find no match.
        cur_game.named_card = "";
        game_log("%s names no card (no eligible card to name).\n",
                 player_name(ab.controller).c_str());
        return true;
    }

    bool prev_priority = cur_game.player_a_has_priority;
    cur_game.player_a_has_priority = (ab.controller == Zone::PLAYER_A);
    game_log("%s names a card:\n", player_name(ab.controller).c_str());
    int choice = InputLogger::instance().get_input(name_choices);
    cur_game.player_a_has_priority = prev_priority;

    cur_game.named_card = names[static_cast<size_t>(choice)];
    game_log("%s names card: %s\n", player_name(ab.controller).c_str(), cur_game.named_card.c_str());
    return true;
}

}  // namespace effects
