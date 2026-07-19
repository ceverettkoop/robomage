#include "state_manager.h"
#include "state_manager_internal.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "../action_processor.h"
#include "../card_vocab.h"
#include "../classes/game.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/color_identity.h"
#include "../components/creature.h"
#include "../components/static_ability.h"
#include "../components/damage.h"
#include "../components/effect.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/token.h"
#include "../components/types.h"
#include "../type_constants.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../cli_output.h"
#include "../error.h"
#include "../game_driver.h"
#include "../game_queries.h"
#include "../input_logger.h"
#include "../mana_system.h"
#include "../pending_query.h"
#include "../saga.h"
#include "../svar_eval.h"
#include "../systems/stack_manager.h"
#include "orderer.h"

// Rebuilt from scratch by gather_active_statics on every SBE pass; its ActiveStatic
// entries hold raw pointers into component arrays, so it can be left holding stale
// pointers across a snapshot restore (the ECS component storage moves). That is
// safe: nothing reads g_active_statics outside the SBE / legal-action machinery,
// which never runs at a sideboard prompt (the only cross-game restore root), and
// the very next SBE pass rebuilds it before any reader. It is deliberately NOT part
// of the game snapshot.
std::vector<ActiveStatic> g_active_statics;

void StateManager::init() {
    Signature signature;
    signature.set(global_coordinator.GetComponentType<Zone>());
    global_coordinator.SetSystemSignature<StateManager>(signature);
}

// Turn-based actions happen at the start of specific steps (rules 508, 509, 510, 514)
void StateManager::process_turn_based_actions(Game &game, std::shared_ptr<Orderer> orderer) {
    game.pending_choice = NONE;

    // First strike combat damage (rule 510.1)
    if (game.cur_step == FIRST_STRIKE_DAMAGE && !game.combat_damage_dealt) {
        // T3.10: let a controller divide damage among blockers it can't all kill before dealing.
        if (any_attacker_needs_damage_assignment(game, orderer, /*first_strike_only=*/true)) {
            game.pending_choice = ASSIGN_COMBAT_DAMAGE_CHOICE;
            return;
        }
        deal_combat_damage(game, true);
    }
    // Regular combat damage (rule 510.2)
    if (game.cur_step == COMBAT_DAMAGE && !game.combat_damage_dealt) {
        if (any_attacker_needs_damage_assignment(game, orderer, /*first_strike_only=*/false)) {
            game.pending_choice = ASSIGN_COMBAT_DAMAGE_CHOICE;
            return;
        }
        deal_combat_damage(game, false);
    }

    // Declare attackers (rule 508.1)
    if (game.cur_step == DECLARE_ATTACKERS && !game.attackers_declared) {
        game.pending_choice = DECLARE_ATTACKERS_CHOICE;
        return;
    }
    // Declare blockers (rule 509.1)
    if (game.cur_step == DECLARE_BLOCKERS && !game.blockers_declared) {
        game.pending_choice = DECLARE_BLOCKERS_CHOICE;
        return;
    }
    // Cleanup discard (rule 514.1)
    if (game.cur_step == CLEANUP) {
        Zone::Ownership active_player = game.player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B;
        size_t hand_size = 0;
        for (auto entity : mEntities) {
            if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
            auto &zone = global_coordinator.GetComponent<Zone>(entity);
            if (zone.location == Zone::HAND && zone.owner == active_player) hand_size++;
        }
        // Maximum hand size is 7 by default (CR 402.2), but a SetMaxHandSize continuous static
        // affecting the active player overrides it — Tamiyo, Seasoned Scholar's emblem grants
        // "no maximum hand size" (Unlimited → no cleanup discard). g_active_statics holds emblem
        // statics (gathered each SBA pass with the controller as owner); the most permissive
        // applicable override wins (Unlimited beats any finite cap).
        int max_hand_size = 7;
        bool unlimited = false;
        for (const auto &as : g_active_statics) {
            if (as.suppressed || !as.condition_met || !as.sa) continue;
            if (as.controller != active_player || as.sa->set_max_hand_size == 0) continue;
            if (as.sa->set_max_hand_size < 0) { unlimited = true; break; }
            if (as.sa->set_max_hand_size > max_hand_size) max_hand_size = as.sa->set_max_hand_size;
        }
        if (!unlimited && hand_size > static_cast<size_t>(max_hand_size)) {
            game.pending_choice = CLEANUP_DISCARD;
            return;
        }
    }
}

// State-based actions are checked simultaneously and loop until stable (rule 704.3)
void StateManager::state_based_effects(Game &game, std::shared_ptr<Orderer> orderer) {
    for (;;) {
        // Continuous effects define the game state that SBAs evaluate
        apply_permanent_components(game, orderer);
        // An ETB choice inside apply_permanent_components parked a loop-top
        // pending decision (tag SBE_LATCHED): suspend the whole SBE call by
        // early return (no trigger drain — it stays deferred until the resumed
        // run's scan). On resume the main loop re-enters state_based_effects
        // from scratch; already-processed permanents are idempotent no-ops and
        // the same mid-apply site re-finds the question and consumes the latch.
        // (An ACTIVE-and-ANSWERED query is that resume in flight — fall
        // through so the deriving scan below can reach its site.)
        if (game.pending_query.active && !game.pending_query.answered) return;
        apply_continuous_effects(game);
        update_city_blessing(game);  // 702.131: ascend grants the city's blessing at 10+ permanents

        bool any_applied = false;

        // 704.5a - player with 0 or less life loses
        auto &player_a = global_coordinator.GetComponent<Player>(game.player_a_entity);
        auto &player_b = global_coordinator.GetComponent<Player>(game.player_b_entity);
        if (player_a.life_total <= 0) {
            printf("\nPlayer A has %d life - Player B wins!\n", player_a.life_total);
            game.ended = true;
            game.winner = Zone::PLAYER_B;
            return;
        }
        if (player_b.life_total <= 0) {
            printf("\nPlayer B has %d life - Player A wins!\n", player_b.life_total);
            game.ended = true;
            game.winner = Zone::PLAYER_A;
            return;
        }

        // 704.5d - tokens in zones other than battlefield cease to exist
        // (handled by apply_permanent_components above)

        // 704.5f - creature with toughness 0 or less goes to graveyard
        // 704.5g - creature with lethal damage is destroyed
        std::vector<Entity> creatures_to_destroy;
        for (auto entity : mEntities) {
            if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
            if (!is_battlefield_permanent(entity)) continue;

            auto &creature = global_coordinator.GetComponent<Creature>(entity);
            if (creature.toughness == 0) {
                // 704.5f: a creature with toughness 0 or less is put into its owner's
                // graveyard. This is NOT a destroy — indestructible does not prevent it.
                creatures_to_destroy.push_back(entity);
            } else if (global_coordinator.entity_has_component<Damage>(entity) &&
                       !is_indestructible(entity)) {
                // 704.5g/h: lethal-damage / deathtouch destruction. 702.12b: an
                // indestructible creature ignores these state-based actions, so it is
                // excluded above (it keeps its marked damage but is not destroyed).
                auto &damage = global_coordinator.GetComponent<Damage>(entity);
                // 702.2b: any nonzero damage from a deathtouch source is lethal.
                bool deathtouched = damage.has_deathtouch_damage && damage.damage_counters > 0;
                if (deathtouched || damage.damage_counters >= creature.toughness) {
                    creatures_to_destroy.push_back(entity);
                }
            }
        }

        for (auto entity : creatures_to_destroy) {
            std::string name = entity_name(entity);
            auto &creature = global_coordinator.GetComponent<Creature>(entity);
            if (creature.toughness == 0)
                game_log("%s dies (zero toughness)\n", name.c_str());
            else
                game_log("%s is destroyed (lethal damage)\n", name.c_str());
            orderer->add_to_zone(false, entity, Zone::GRAVEYARD);
            any_applied = true;
        }

        // 704.5i - a planeswalker with 0 (or less) loyalty is put into its owner's graveyard
        for (auto entity : mEntities) {
            if (!is_battlefield_permanent(entity)) continue;
            auto &perm = global_coordinator.GetComponent<Permanent>(entity);
            if (!is_planeswalker(perm.types)) continue;
            if (get_counters(entity, "LOYALTY") <= 0) {
                game_log("%s dies (0 loyalty)\n", entity_name(entity).c_str());
                orderer->add_to_zone(false, entity, Zone::GRAVEYARD);
                any_applied = true;
            }
        }

        // 704.5n - an Aura attached to an illegal object, or not attached to anything, is put
        // into its owner's graveyard. Auras carry an enchant restriction (CardData::enchant_filter)
        // and track their attachment via Permanent::equipped_to (shared with equipment). We check
        // the structural part of "illegal": no attachment, or the enchanted object has left the
        // battlefield / is no longer a creature (the common fall-off when the enchanted creature
        // dies or is bounced).
        for (auto entity : mEntities) {
            if (!is_battlefield_permanent(entity)) continue;
            if (!global_coordinator.entity_has_component<CardData>(entity)) continue;
            const auto &cd = global_coordinator.GetComponent<CardData>(entity);
            if (cd.enchant_filter.empty()) continue;  // not an Aura
            auto &perm = global_coordinator.GetComponent<Permanent>(entity);
            Entity enchanted = perm.equipped_to;
            bool illegal = (enchanted == 0) || !is_battlefield_permanent(enchanted);
            if (!illegal && cd.enchant_filter.find("Creature") != std::string::npos &&
                !global_coordinator.entity_has_component<Creature>(enchanted))
                illegal = true;
            if (illegal) {
                game_log("%s is put into the graveyard (Aura not attached to a legal object)\n",
                         entity_name(entity).c_str());
                orderer->add_to_zone(false, entity, Zone::GRAVEYARD);
                any_applied = true;
            }
        }

        // CR 714.4 - Saga sacrifice. A Saga whose lore counters are >= its final chapter number,
        // and which isn't the source of a chapter ability that has triggered but not yet left the
        // stack (tracked by Permanent::saga_chapters_in_flight), is sacrificed by its controller.
        // This holds for a Saga that is also a creature (Summon: Bahamut) — the printed "Sacrifice
        // after IV" / the script's chapter count governs, independent of its other types. Note: the
        // checked-in CR 714.4 text has no creature-Saga exception, and both cards' Oracle text
        // ("Sacrifice after III/IV") confirms the creature Saga is sacrificed, so they agree.
        for (auto entity : mEntities) {
            if (!is_battlefield_permanent(entity)) continue;
            if (!global_coordinator.entity_has_component<CardData>(entity)) continue;
            const auto &cd = global_coordinator.GetComponent<CardData>(entity);
            if (!card_is_saga(cd)) continue;
            auto &perm = global_coordinator.GetComponent<Permanent>(entity);
            // Layer-6 ability removal: a Saga turned into a Mountain (Magus of the Moon,
            // CR 305.7) is no longer a Saga — 714.4 doesn't sacrifice it, even at full lore.
            if (perm.abilities_removed) continue;
            int final_chapter = static_cast<int>(cd.saga_chapters.size());
            if (get_counters(entity, "LORE") < final_chapter) continue;
            if (perm.saga_chapters_in_flight > 0) continue;  // a chapter ability is still on the stack
            game_log("%s is sacrificed (final chapter completed).\n", entity_name(entity).c_str());
            orderer->add_to_zone(false, entity, Zone::GRAVEYARD);
            any_applied = true;
        }

        // 704.5q - if a permanent has both a +1/+1 and a -1/-1 counter, N of each are
        // removed, where N is the smaller of the two counts (122.3 annihilation).
        for (auto entity : mEntities) {
            if (!is_battlefield_permanent(entity)) continue;
            int plus = get_counters(entity, "P1P1");
            int minus = get_counters(entity, "M1M1");
            int n = std::min(plus, minus);
            if (n <= 0) continue;
            add_counters(entity, "P1P1", -n);
            add_counters(entity, "M1M1", -n);
            game_log("%s: %d +1/+1 and %d -1/-1 counter(s) annihilate.\n",
                     entity_name(entity).c_str(), n, n);
            any_applied = true;
        }

        // 704.5j - legend rule: a player who controls two or more legendary permanents with
        // the same name chooses one to keep; the rest go to their owners' graveyards. Affected
        // players choose in APNAP order (active player first); one conflict is resolved per pass,
        // then the SBA loop re-evaluates.
        {
            Zone::Ownership legend_order[2] = {
                game.player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B,
                game.player_a_turn ? Zone::PLAYER_B : Zone::PLAYER_A};
            bool legend_applied = false;
            for (Zone::Ownership owner : legend_order) {
                if (legend_applied) break;
                std::map<std::string, std::vector<Entity>> by_name;
                for (auto entity : mEntities) {
                    if (!is_battlefield_permanent(entity, owner)) continue;
                    auto &perm = global_coordinator.GetComponent<Permanent>(entity);
                    if (!has_legendary_supertype(perm.types)) continue;
                    by_name[perm.name].push_back(entity);
                }
                for (auto &grp : by_name) {
                    if (grp.second.size() < 2) continue;
                    std::vector<LegalAction> choices;
                    for (auto e : grp.second) {
                        LegalAction la(PASS_PRIORITY, e, "Keep " + entity_name(e));
                        la.category = ActionCategory::KEEP_LEGEND;
                        choices.push_back(la);
                    }
                    // Latched-answer site (tag SBE_LATCHED): the keep choice is a
                    // loop-top pending decision. Re-finding is deterministic — the
                    // legend check is the LAST check in the pass body, so suspending
                    // here ≡ end-of-pass: on resume the re-run's earlier checks are
                    // silent no-ops on the frozen state, legend_order re-derives from
                    // the unchanged player_a_turn, the name-sorted by_name map yields
                    // the same first conflict, and legend_applied still limits the
                    // pass to that one conflict — so the key (owner + conflicting
                    // name + menu size) provably re-derives.
                    uint64_t key = pq_key(SbeSite::LEGEND_KEEP, owner == Zone::PLAYER_A,
                                          std::hash<std::string>{}(grp.first), choices.size());
                    int keep = -1;
                    if (!pq_take_latched(key, &keep)) {
                        game_log("Legend rule: %s controls %zu copies of %s; choose one to keep.\n",
                                 player_name(owner).c_str(), grp.second.size(), grp.first.c_str());
                        if (in_main_loop()) {
                            // Park the choice (priority persisted at the chooser) and
                            // suspend the whole SBE call — early return WITHOUT
                            // check_triggered_abilities, deferring the pass's event
                            // drain to the resumed run (the drain must stay inside
                            // the trigger scan).
                            pq_arm_sbe(key, std::move(choices), owner, /*decision_source=*/0);
                            return;
                        }
                        // Blocking fallback for an SBE call outside the main loop.
                        // Since the pregame gate (Batch 13) even the preplaced-preset
                        // SBE pass runs inside the loop, so this is defensive only.
                        // The controller of the duplicates chooses which to keep;
                        // point priority at them so the query routes/observes/records
                        // from their perspective (SBAs run regardless of who
                        // currently holds priority).
                        bool prev_priority = game.player_a_has_priority;
                        game.player_a_has_priority = (owner == Zone::PLAYER_A);
                        keep = InputLogger::instance().get_input(choices);
                        game.player_a_has_priority = prev_priority;
                    }
                    Entity kept = grp.second[static_cast<size_t>(keep)];
                    for (auto e : grp.second) {
                        if (e == kept) continue;
                        game_log("%s is put into the graveyard (legend rule)\n", entity_name(e).c_str());
                        orderer->add_to_zone(false, e, Zone::GRAVEYARD);
                    }
                    any_applied = true;
                    legend_applied = true;
                    break;
                }
            }
        }

        if (!any_applied) break;
    }

    // Latched-answer tripwire: a re-run entered with an answered SBE_LATCHED
    // query must have consumed it at the site that armed it (pass 1 re-derives
    // the question — the state is frozen while the answer is latched). Settling
    // without consuming means the re-run failed to re-find the question.
    if (game.pending_query.active && game.pending_query.answered &&
        game.pending_query.tag == PendingQuery::SBE_LATCHED)
        fatal_error("SBE re-run did not re-derive the latched question");

    // SBA loop settled; triggered abilities go on the stack (rule 704.3)
    check_triggered_abilities(game, orderer);
}

