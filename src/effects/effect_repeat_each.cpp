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
//
// Suspendable (Batch 8): the (player, sub) loop progress persists in RepeatRt — including
// the saved outer remembered set and a per-player setup flag, so a resume never re-clears a
// remembered set a suspended sub-ability's earlier siblings may have grown — and each sub
// resolves as a persisted REPEAT_SUB FrameLevel (iter_index = the player index).
HandlerResult repeat_each(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    if (ab.repeat_players.empty() || ab.subabilities.empty()) return HandlerResult::DONE_NO_SUBS;

    Zone::Ownership active = cur_game.player_a_active ? Zone::PLAYER_A : Zone::PLAYER_B;
    Zone::Ownership nonactive = (active == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;
    std::vector<Zone::Ownership> order = {active, nonactive};

    RepeatRt local_rt;
    RepeatRt &rt = ctx.can_suspend() ? ctx.rt<RepeatRt>() : local_rt;
    if (!rt.init) {
        rt.saved_remembered = cur_game.remembered_entities;
        rt.init = true;
    }
    while (rt.player_idx < static_cast<int>(order.size())) {
        Zone::Ownership p = order[static_cast<size_t>(rt.player_idx)];
        Entity pe = (p == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
        if (global_coordinator.entity_has_component<Player>(pe)) {
            if (!rt.player_setup) {
                cur_game.remembered_entities.clear();
                cur_game.remembered_entities.push_back(pe);
                rt.player_setup = true;
            }
            for (; rt.sub_idx < static_cast<int>(ab.subabilities.size()); ++rt.sub_idx) {
                const Ability &sub_template = ab.subabilities[static_cast<size_t>(rt.sub_idx)];
                if (ctx.can_suspend()) {
                    Ability *parent = &ab;
                    auto bind = [parent, pe](Ability &sub) {
                        sub.source = parent->source;
                        sub.controller = parent->controller;
                        // Defined$ Remembered points the sub-ability at the looped player.
                        if (sub.defined_remembered) sub.target = pe;
                    };
                    if (ctx.resolve_child(sub_template, FrameLevel::REPEAT_SUB, rt.sub_idx,
                                          rt.player_idx, bind, orderer) ==
                        ResolveStatus::SUSPENDED)
                        return HandlerResult::SUSPENDED;
                } else {
                    Ability sub = sub_template;
                    sub.source = ab.source;
                    sub.controller = ab.controller;
                    // Defined$ Remembered points the sub-ability at the looped player.
                    if (sub.defined_remembered) sub.target = pe;
                    sub.resolve(orderer);
                }
            }
        }
        rt.player_idx++;
        rt.sub_idx = 0;
        rt.player_setup = false;
    }
    cur_game.remembered_entities = rt.saved_remembered;
    return HandlerResult::DONE_NO_SUBS;  // sub-abilities already resolved per-player; suppress default chaining
}

}  // namespace effects
