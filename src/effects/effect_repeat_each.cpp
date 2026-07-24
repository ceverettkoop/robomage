#include "effects.h"

#include <string>
#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Distinct card types (kind == TYPE, e.g. "Creature") present among the imprinted cards owned by
// `owner`, in the canonical CR 300.1 card-type order. Drives the per-type RepeatEach loop
// (Atraxa, Grand Unifier). Recomputed purely each entry so a resume re-derives the identical list
// (the imprinted cards stay in the library throughout the loop — the trailing DBChangeZone that
// moves the chosen ones runs only AFTER the loop). General over any "for each card type" effect.
static std::vector<std::string> imprinted_types(Zone::Ownership owner) {
    static const char *kCardTypes[] = {"Artifact", "Battle",       "Creature", "Enchantment",
                                       "Instant",  "Kindred",      "Land",     "Planeswalker",
                                       "Sorcery"};
    std::vector<std::string> present;
    for (const char *t : kCardTypes) {
        for (auto e : cur_game.imprinted_entities) {
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            if (global_coordinator.entity_has_component<Zone>(e) &&
                global_coordinator.GetComponent<Zone>(e).owner != owner)
                continue;
            if (card_has_type(global_coordinator.GetComponent<CardData>(e), t)) {
                present.push_back(t);
                break;
            }
        }
    }
    return present;
}

// RepeatEach over CARD TYPES (Atraxa, Grand Unifier): for each distinct card type among the
// imprinted cards, set cur_game.chosen_type and resolve the RepeatSubAbility body (a ChooseCard
// that takes one imprinted card of that type). The leading `repeat_sub_count` subabilities are the
// per-type body; the remaining subabilities are trailing SubAbility$ links resolved ONCE after the
// whole loop (Atraxa: the DBChangeZone that moves the chosen cards to hand). Suspendable: the type
// index / body-sub index / trailing-sub index persist in RepeatRt, and each child resolves as a
// persisted FrameLevel via resolve_child (the same machinery the per-player path uses).
static HandlerResult repeat_each_types(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    std::vector<std::string> types = imprinted_types(ab.controller);
    size_t body_count = ab.repeat_sub_count > 0 ? ab.repeat_sub_count : ab.subabilities.size();
    if (body_count > ab.subabilities.size()) body_count = ab.subabilities.size();

    RepeatRt local_rt;
    RepeatRt &rt = ctx.can_suspend() ? ctx.rt<RepeatRt>() : local_rt;

    // Per-type loop: player_idx is reused as the type index.
    while (rt.player_idx < static_cast<int>(types.size())) {
        cur_game.chosen_type = types[static_cast<size_t>(rt.player_idx)];
        for (; rt.sub_idx < static_cast<int>(body_count); ++rt.sub_idx) {
            const Ability &body = ab.subabilities[static_cast<size_t>(rt.sub_idx)];
            if (ctx.can_suspend()) {
                Ability *parent = &ab;
                auto bind = [parent](Ability &sub) {
                    sub.source = parent->source;
                    sub.controller = parent->controller;
                };
                if (ctx.resolve_child(body, FrameLevel::REPEAT_SUB, rt.sub_idx, rt.player_idx, bind,
                                      orderer) == ResolveStatus::SUSPENDED)
                    return HandlerResult::SUSPENDED;
            } else {
                Ability sub = body;
                sub.source = ab.source;
                sub.controller = ab.controller;
                sub.resolve(orderer);
            }
        }
        rt.player_idx++;
        rt.sub_idx = 0;
    }
    cur_game.chosen_type = "";

    // Trailing SubAbility$ links (the DBChangeZone → ShuffleRest → Cleanup chain) resolve once.
    for (; rt.trailing_idx < static_cast<int>(ab.subabilities.size() - body_count); ++rt.trailing_idx) {
        const Ability &tail = ab.subabilities[body_count + static_cast<size_t>(rt.trailing_idx)];
        if (ctx.can_suspend()) {
            Ability *parent = &ab;
            auto bind = [parent](Ability &sub) {
                sub.source = parent->source;
                sub.controller = parent->controller;
            };
            if (ctx.resolve_child(tail, FrameLevel::SUB, rt.trailing_idx,
                                  static_cast<int>(types.size()), bind, orderer) ==
                ResolveStatus::SUSPENDED)
                return HandlerResult::SUSPENDED;
        } else {
            Ability sub = tail;
            sub.source = ab.source;
            sub.controller = ab.controller;
            sub.resolve(orderer);
        }
    }
    return HandlerResult::DONE_NO_SUBS;  // body + trailing subs resolved here; suppress default chaining
}

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
    // RepeatTypesFrom$ — loop once per distinct card type among the imprinted cards (Atraxa).
    if (!ab.repeat_types_from.empty()) return repeat_each_types(ab, orderer, ctx);

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
