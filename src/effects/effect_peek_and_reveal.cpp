#include "effects.h"

#include <algorithm>
#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/damage.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/types.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../systems/orderer.h"
#include "../transform.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

HandlerResult peek_and_reveal(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    const PeekParams *pp = std::get_if<PeekParams>(&ab.params);
    if (pp && pp->no_reveal) {
        // Look at the top N cards of the target player's library privately, no reveal choice.
        // N = PeekAmount (Mishra's Bauble = 1; Birthing Ritual = 7, so the controller sees the
        // top 7 *before* the subsequent sacrifice decision, per "look at the top seven... Then
        // you may sacrifice"). No card movement here — a chained Dig does the actual selection.
        Zone::Ownership peek_owner = global_coordinator.entity_has_component<Player>(ab.target)
                                         ? (ab.target == cur_game.player_a_entity ? Zone::PLAYER_A : Zone::PLAYER_B)
                                         : ab.controller;
        int n = pp->peek_amount > 0 ? pp->peek_amount : 1;
        std::vector<Entity> top = orderer->get_library_top(peek_owner, static_cast<size_t>(n));
        if (top.empty()) {
            game_log("%s's library is empty — nothing to peek.\n", player_name(peek_owner).c_str());
        } else {
            for (auto e : top) {
                if (!global_coordinator.entity_has_component<CardData>(e)) continue;
                auto &cd = global_coordinator.GetComponent<CardData>(e);
                game_log_private(ab.controller, "%s looks at top of %s's library: %s\n",
                    player_name(ab.controller).c_str(), player_name(peek_owner).c_str(), cd.name.c_str());
            }
        }
        // fall through to subabilities (DelayedTrigger sub-ability fires next upkeep)
        return HandlerResult::DONE_RUN_SUBS;
    }

    // Delver of Secrets: peek own library top, optionally reveal
    if (!global_coordinator.entity_has_component<Permanent>(ab.source)) {
        game_log("%s fizzles\n", ab.category.c_str());
        return HandlerResult::DONE_NO_SUBS;
    }
    auto &src_perm = global_coordinator.GetComponent<Permanent>(ab.source);
    Entity top_card = 0;
    for (auto e : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location == Zone::LIBRARY && z.owner == ab.controller && z.distance_from_top == 0) {
            top_card = e;
            break;
        }
    }

    if (top_card == 0) {
        game_log("Library is empty — nothing to peek.\n");
        return HandlerResult::DONE_NO_SUBS;
    }
    auto &top_cd = global_coordinator.GetComponent<CardData>(top_card);
    // Arm-only peek line: the resume rebuilds the same menu (the top card is
    // pinned against determinize by collect_pending_pins) without re-logging.
    if (!ctx.resuming())
        game_log_private(ab.controller, "Top card of library: %s\n", top_cd.name.c_str());
    std::vector<LegalAction> reveal_actions = {
        LegalAction(PASS_PRIORITY, top_card, std::string("Don't reveal")),
        LegalAction(PASS_PRIORITY, top_card, std::string("Reveal")),
    };
    // The old inline get_input ran without a priority repoint — ambient priority
    // is the resolving controller here — so seating the ask on ab.controller is
    // a no-op swap, byte-identical to today.
    int reveal_choice = ctx.ask(std::move(reveal_actions), ab.controller, ab.source);
    if (reveal_choice < 0 && decision_suspended()) return HandlerResult::SUSPENDED;

    if (reveal_choice == 1) {
        game_log("Revealed: %s\n", top_cd.name.c_str());
        bool is_instant_or_sorcery = false;
        for (auto &t : top_cd.types) {
            if (t.kind == TYPE && (t.name == "Instant" || t.name == "Sorcery")) {
                is_instant_or_sorcery = true;
                break;
            }
        }
        if (is_instant_or_sorcery && global_coordinator.entity_has_component<CardData>(ab.source)) {
            auto &src_cd = global_coordinator.GetComponent<CardData>(ab.source);
            if (src_cd.backside && !src_perm.transformed) {
                // Flip to the back face through the shared transform subsystem so
                // Delver's creature->creature flip and Ajani's creature->planeswalker
                // flip travel the same code path.
                set_permanent_face(ab.source, true);
            }
        }
    }
    return HandlerResult::DONE_NO_SUBS;  // transform logic handled inline; skip subabilities loop
}

bool parse_peek_and_reveal(Ability &ab, const std::string &key, const std::string &value) {
    if (key == "NoReveal") { effect_params<PeekParams>(ab).no_reveal = (value == "True"); return true; }
    if (key == "PeekAmount") { effect_params<PeekParams>(ab).peek_amount = std::stoi(value); return true; }
    return false;
}

}  // namespace effects
