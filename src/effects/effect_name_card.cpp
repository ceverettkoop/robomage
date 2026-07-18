#include "effects.h"

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../classes/gamestate.h"
#include "../cli_output.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../name_card_choices.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Prompt-and-record for a "name a card" decision (CR 201.4): temporarily hand priority to the
// chooser so they are the one queried, log the prompt, run the choice over the prepared candidate
// list, restore priority, and return the chosen name. The candidate list (names + parallel
// name_choices) is built separately by build_name_card_choices(); this helper is only the
// priority-swap + get_input + name lookup. Returns "" if there is no eligible card to name.
static std::string prompt_name_card(Zone::Ownership chooser, const std::vector<std::string> &names,
                                    const std::vector<LegalAction> &name_choices);

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
// builds that menu; the ValidCards$ filter (Card.nonLand / Land / …) is passed through and
// applied by the unified matcher inside the builder.
HandlerResult name_card(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    PendingDecisionScope pending_scope(ab.source);
    // Defined$ You + ValidCards$ Land (Petrified Hamlet's ETB "choose a land card name"):
    // the SOURCE's controller names a land card. The choice persists for the source's
    // continuous Card.NamedCard static, so it is recorded on the source permanent's
    // chosen_name (the same per-source state match_named_card statics read) rather than the
    // transient global cur_game.named_card (which is cleared after the ability resolves).
    bool defines_self_owner = ab.defined_you;
    bool only_lands = (ab.valid_cards_filter == "Land");
    if (defines_self_owner) {
        Zone::Ownership chooser = ab.controller;
        // Land-naming (Petrified Hamlet / Alpine Moon-style "name a land") offers lands present
        // in EITHER player's deck, not just the chooser's, so a land that exists only in the
        // opponent's deck is still nameable (CR 201.4 lets you name any card; the engine uses a
        // limited, context-driven set — see CLAUDE.md). Non-land self-named cards stay
        // chooser-scoped.
        NameCardScope scope = only_lands ? NameCardScope::BOTH_PLAYERS : NameCardScope::CHOOSER_ONLY;
        std::vector<std::string> names;
        std::vector<LegalAction> name_choices =
            build_name_card_choices(orderer->mEntities, chooser, ab.valid_cards_filter, names,
                                    scope);
        std::string chosen = prompt_name_card(chooser, names, name_choices);
        // Record on the source permanent so its continuous static can read it; also set the
        // global for any chained sub-ability resolving in the same call (cleared afterward).
        if (ab.source != 0 && global_coordinator.entity_has_component<Permanent>(ab.source))
            global_coordinator.GetComponent<Permanent>(ab.source).chosen_name = chosen;
        cur_game.named_card = chosen;
        if (chosen.empty())
            game_log("%s names no card (no eligible card to name).\n", player_name(chooser).c_str());
        else
            game_log("%s names card: %s\n", player_name(chooser).c_str(), chosen.c_str());
        return HandlerResult::DONE_RUN_SUBS;
    }

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
        build_name_card_choices(orderer->mEntities, name_owner, ab.valid_cards_filter, names);

    if (name_choices.empty()) {
        // No nameable card; the chained Card.NamedCard discard will find no match.
        cur_game.named_card = "";
        game_log("%s names no card (no eligible card to name).\n",
                 player_name(ab.controller).c_str());
        return HandlerResult::DONE_RUN_SUBS;
    }

    cur_game.named_card = prompt_name_card(ab.controller, names, name_choices);
    game_log("%s names card: %s\n", player_name(ab.controller).c_str(), cur_game.named_card.c_str());
    return HandlerResult::DONE_RUN_SUBS;
}

// See forward declaration at top of file.
static std::string prompt_name_card(Zone::Ownership chooser, const std::vector<std::string> &names,
                                    const std::vector<LegalAction> &name_choices) {
    if (name_choices.empty())
        return "";
    bool prev_priority = cur_game.player_a_has_priority;
    cur_game.player_a_has_priority = (chooser == Zone::PLAYER_A);
    game_log("%s names a card:\n", player_name(chooser).c_str());
    int choice = InputLogger::instance().get_input(name_choices);
    cur_game.player_a_has_priority = prev_priority;
    return names[static_cast<size_t>(choice)];
}

}  // namespace effects
