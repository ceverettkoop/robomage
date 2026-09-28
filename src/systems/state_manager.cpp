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
#include "../components/creature.h"
#include "../components/entry_info.h"
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

namespace {
// The state-based actions one check finds on the battlefield (CR 704.3), performed together as a
// single event by perform_permanent_sbas.
struct SbaActions {
    std::vector<std::pair<Entity, std::string>> to_graveyard;  // object + narrative line (704.5f-j, m, s)
    std::vector<Entity> unattach;                               // 704.5n / 704.5p
    std::vector<std::pair<Entity, int>> annihilate;             // 704.5q: permanent + N

    bool empty() const { return to_graveyard.empty() && unattach.empty() && annihilate.empty(); }
    // Records `e` as put into its owner's graveyard, once however many actions apply to it.
    void put_into_graveyard(Entity e, const std::string &log_line) {
        for (const auto &entry : to_graveyard)
            if (entry.first == e) return;
        to_graveyard.emplace_back(e, log_line);
    }
};
}  // namespace

// `who`'s state-based loss condition (CR 704.5a 0 or less life, 704.5b a draw from an empty
// library, 704.5c ten or more poison counters), describing it in `reason`.
static bool player_loss_reason(Zone::Ownership who, const Player &player, std::string &reason);

// CR 704.5a-c for both players in one check (CR 704.3): ends the game when either loses — a draw
// when both do (CR 104.4a). Returns whether the game ended.
static bool check_player_losses(Game &game);

// Whether an Aura is attached to an illegal object or to nothing (CR 704.5m).
static bool aura_attached_illegally(Entity aura, const CardData &cd,
                                    const Permanent &perm);

// One pass over the battlefield collecting every permanent state-based action that applies to
// the current game state (CR 704.5f-i, m, n, p, q, s) into `out`.
static void find_permanent_sbas(Game &game, const std::set<Entity> &entities, SbaActions &out);

// The legend rule (CR 704.5j): the first conflict's controller chooses the one to keep, and the
// rest join `out`. Returns false when the choice parked as a loop-top pending decision.
static bool choose_legend_rule_keep(Game &game, const std::set<Entity> &entities,
                                    SbaActions &out);

// Perform one check's actions together (CR 704.3).
static void perform_permanent_sbas(std::shared_ptr<Orderer> orderer, const SbaActions &actions);

// Rebuilt from scratch by gather_active_statics on every SBE pass; its ActiveStatic
// entries are (source, static index) handles resolved at read time, so a snapshot
// restore can leave it describing the previous simulation's statics. That is safe for
// restores: the post-restore hook recomputes it before any reader. It is deliberately
// NOT part of the game snapshot. Across a bo3 game boundary the per-game init_ecs()
// destroys the previous game's components and entity ids are reissued, and the
// mulligan-decision observation reads it (via rules_mod::land_drops_remaining in
// populate_gamestate) BEFORE the first SBE pass of the new game rebuilds it — so
// StateManager::init() must clear it.
std::vector<ActiveStatic> g_active_statics;

void StateManager::init() {
    // Drop the previous game's registry: its handles name entities the ECS reset just
    // destroyed (see the comment on g_active_statics).
    g_active_statics.clear();
    Signature signature;
    signature.set(global_coordinator.GetComponentType<Zone>());
    global_coordinator.SetSystemSignature<StateManager>(signature);
}

// Turn-based actions happen at the start of specific steps (rules 508, 509, 510, 514)
void StateManager::process_turn_based_actions(Game &game, std::shared_ptr<Orderer> orderer) {
    game.pending.choice = NONE;

    // First strike combat damage (rule 510.1)
    if (game.turn_state.step == FIRST_STRIKE_DAMAGE && !game.combat.damage_dealt) {
        // T3.10: let a controller divide damage among blockers it can't all kill before dealing.
        if (any_attacker_needs_damage_assignment(game, orderer, /*first_strike_only=*/true)) {
            game.pending.choice = ASSIGN_COMBAT_DAMAGE_CHOICE;
            return;
        }
        deal_combat_damage(game, true);
    }
    // Regular combat damage (rule 510.2)
    if (game.turn_state.step == COMBAT_DAMAGE && !game.combat.damage_dealt) {
        if (any_attacker_needs_damage_assignment(game, orderer, /*first_strike_only=*/false)) {
            game.pending.choice = ASSIGN_COMBAT_DAMAGE_CHOICE;
            return;
        }
        deal_combat_damage(game, false);
    }

    // Declare attackers (rule 508.1)
    if (game.turn_state.step == DECLARE_ATTACKERS && !game.combat.attackers_declared) {
        game.pending.choice = DECLARE_ATTACKERS_CHOICE;
        return;
    }
    // Declare blockers (rule 509.1)
    if (game.turn_state.step == DECLARE_BLOCKERS && !game.combat.blockers_declared) {
        game.pending.choice = DECLARE_BLOCKERS_CHOICE;
        return;
    }
    // Cleanup discard (rule 514.1)
    if (game.turn_state.step == CLEANUP) {
        Zone::Ownership active_player = active_seat();
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
            if (as.suppressed || !as.condition_met) continue;
            if (as.controller != active_player || as.sa()->set_max_hand_size == 0) continue;
            if (as.sa()->set_max_hand_size < 0) { unlimited = true; break; }
            if (as.sa()->set_max_hand_size > max_hand_size) max_hand_size = as.sa()->set_max_hand_size;
        }
        if (!unlimited && hand_size > static_cast<size_t>(max_hand_size)) {
            game.pending.choice = CLEANUP_DISCARD;
            return;
        }
        // CR 514.2, after the discard: damage wears off and "until end of turn" / "this turn"
        // effects end.
        if (!game.turn_state.cleanup_effects_ended) {
            game.turn_state.cleanup_effects_ended = true;
            game.end_cleanup_effects();
        }
    }
}

// State-based actions are checked simultaneously and loop until stable (rule 704.3)
void StateManager::state_based_effects(Game &game, std::shared_ptr<Orderer> orderer) {
    // CR 104.1: once the game has ended, nothing further happens. No SBAs — the
    // loss checks below must not overturn a "wins the game" effect that
    // already decided it (first game-ending event wins) — and no trigger
    // collection/placement, which could otherwise park a decision under a
    // finished game (the loop-exit "still parked" fatal). A resolution can end
    // the game (deck-out, WinsGame) and still fall through to the post-advance
    // SBE call, so this gate is load-bearing, not just belt-and-braces.
    if (game.ended) return;
    for (;;) {
        // Continuous effects define the game state that SBAs evaluate
        apply_permanent_components(game, orderer);
        // An ETB choice inside apply_permanent_components parked a loop-top
        // pending decision (tag SBE_LATCHED): suspend the whole SBE call by
        // early return. On resume the main loop re-enters state_based_effects
        // from scratch; already-processed permanents are idempotent no-ops and
        // the same mid-apply site re-finds the question and consumes the latch.
        // (An ACTIVE-and-ANSWERED query is that resume in flight — fall
        // through so the deriving scan below can reach its site.)
        if (game.pending.query.active && !game.pending.query.answered) return;
        apply_continuous_effects(game);
        refresh_city_blessing(mEntities);  // 702.131: ascend grants the city's blessing at 10+ permanents

        // CR 704.5a-c: a player who meets a loss condition loses; both players' conditions are
        // checked together, so two simultaneous losers draw the game (CR 104.4a).
        if (check_player_losses(game)) return;

        // CR 603.2 / 603.3: abilities trigger when their events happen and wait to be put on the
        // stack. Record them before this check's actions move any object, so a source the
        // check removes (a creature that dealt combat damage and died of it) keeps the
        // abilities it already triggered.
        collect_triggered_abilities(game, orderer);

        // CR 704.3: check every permanent against the same game state, then perform all the
        // applicable actions together as a single event.
        SbaActions actions;
        find_permanent_sbas(game, mEntities, actions);
        if (!choose_legend_rule_keep(game, mEntities, actions)) return;  // keep choice parked
        if (actions.empty()) break;
        perform_permanent_sbas(orderer, actions);
        // CR 514.3a: an SBA performed during the cleanup step gives players priority.
        if (game.turn_state.step == CLEANUP) game.turn_state.cleanup_sba_performed = true;
    }

    // Latched-answer tripwire: a re-run entered with an answered SBE_LATCHED
    // query must have consumed it at the site that armed it (pass 1 re-derives
    // the question — the state is frozen while the answer is latched). Settling
    // without consuming means the re-run failed to re-find the question.
    if (game.pending.query.active && game.pending.query.answered &&
        game.pending.query.tag == PendingQuery::SBE_LATCHED)
        fatal_error("SBE re-run did not re-derive the latched question");

    // SBA loop settled; triggered abilities go on the stack (rule 704.3)
    place_waiting_triggers(game, orderer);
    // CR 603.3b: abilities that triggered while that batch was put on the stack (Ward on a
    // placed trigger's target) go on the stack before any player receives priority, after the
    // game checks state-based actions again.
    if (!game.waiting_triggers.empty() && !game.pending.trigger_placement.active)
        state_based_effects(game, orderer);
}

static bool player_loss_reason(Zone::Ownership who, const Player &player, std::string &reason) {
    const std::string name = player_name(who);
    if (player.life_total <= 0) {
        reason = name + " has " + std::to_string(player.life_total) + " life";
        return true;
    }
    if (player.attempted_draw_from_empty) {
        reason = name + " decked";
        return true;
    }
    if (player.counter_count("POISON") >= 10) {
        reason = name + " has " + std::to_string(player.counter_count("POISON")) +
                 " poison counters";
        return true;
    }
    return false;
}

static bool check_player_losses(Game &game) {
    // 704.5b's failed draw is recorded by Orderer::perform_draw, not acted on there (CR 120.3):
    // the resolving effect finishes first, so a "then if your library is empty, you win"
    // sub-ability (Jace, Wielder of Mysteries' -8) decides the game before this check runs.
    std::string reason_a, reason_b;
    bool a_loses = player_loss_reason(
        Zone::PLAYER_A, global_coordinator.GetComponent<Player>(game.player_a_entity), reason_a);
    bool b_loses = player_loss_reason(
        Zone::PLAYER_B, global_coordinator.GetComponent<Player>(game.player_b_entity), reason_b);
    if (!a_loses && !b_loses) return false;
    std::string reason = a_loses && b_loses ? reason_a + " and " + reason_b
                                            : (a_loses ? reason_a : reason_b);
    game.players_lose(a_loses, b_loses, reason);
    return true;
}

// An Aura (CardData::enchant_filter) attached to an illegal object or to nothing (CR 704.5m).
// The structural part of "illegal": no attachment, the enchanted object has left the battlefield
// / is no longer a creature (the common fall-off when the enchanted creature dies or is bounced),
// or it enchants itself or is a creature (303.4d).
static bool aura_attached_illegally(Entity aura, const CardData &cd,
                                    const Permanent &perm) {
    // Animate Dead-style aura (K:Enchant:Creature.inZoneGraveyard, CR 303.4) awaiting its
    // ETB reanimation: it entered unattached (its enchant target is still a graveyard card)
    // and its trigger has not yet returned+attached the creature. Its EntryInfo::aura_target
    // is retained (see state_manager_statics.cpp) to mark this window — skip the
    // unattached-aura check until the reanimation resolves and attaches it, for as long as
    // that card is still a legal object for it (a card that left the graveyard in
    // response is gone, and the aura goes to the graveyard, CR 704.5m).
    if (EntryInfo *entry = find_entry_info(aura); entry && !entry->aura_target.empty()) {
        if (pending_aura_target_legal(aura, perm.controller)) return false;
        entry->aura_target = ObjectRef{};
        drop_entry_info_if_consumed(aura);
    }
    Entity enchanted = perm.equipped_to.get();
    if (enchanted == 0 || enchanted == aura || !is_battlefield_permanent(enchanted) ||
        global_coordinator.entity_has_component<Creature>(aura))
        return true;
    return cd.enchant_filter.find("Creature") != std::string::npos &&
           !global_coordinator.entity_has_component<Creature>(enchanted);
}

static void find_permanent_sbas(Game &game, const std::set<Entity> &entities, SbaActions &out) {
    for (Entity entity : entities) {
        if (!is_battlefield_permanent(entity)) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(entity);
        const CardData *cd = global_coordinator.entity_has_component<CardData>(entity)
                                 ? &global_coordinator.GetComponent<CardData>(entity)
                                 : nullptr;
        const std::string name = entity_name(entity);

        if (global_coordinator.entity_has_component<Creature>(entity)) {
            auto &creature = global_coordinator.GetComponent<Creature>(entity);
            if (creature.toughness == 0) {
                // 704.5f: a creature with toughness 0 or less is put into its owner's
                // graveyard. This is NOT a destroy — indestructible does not prevent it.
                out.put_into_graveyard(entity, name + " dies (zero toughness)");
            } else if (global_coordinator.entity_has_component<Damage>(entity) &&
                       !is_indestructible(entity)) {
                // 704.5g/h: lethal-damage / deathtouch destruction. 702.12b: an
                // indestructible creature ignores these state-based actions, so it is
                // excluded above (it keeps its marked damage but is not destroyed).
                auto &damage = global_coordinator.GetComponent<Damage>(entity);
                // 702.2b: any nonzero damage from a deathtouch source is lethal.
                // The flag is set only by nonzero damage actually dealt, marked or as -1/-1
                // counters (wither/infect), so it alone decides.
                if (damage.has_deathtouch_damage || damage.damage_counters >= creature.toughness)
                    out.put_into_graveyard(entity, name + " is destroyed (lethal damage)");
            }
        }

        // 704.5i - a planeswalker with 0 (or less) loyalty is put into its owner's graveyard
        if (is_planeswalker(perm.types) && get_counters(entity, "LOYALTY") <= 0)
            out.put_into_graveyard(entity, name + " dies (0 loyalty)");

        // Attachments (Permanent::equipped_to, shared by Auras and Equipment):
        // 704.5m - an Aura attached to an illegal object, or not attached to anything, is put
        // into its owner's graveyard (aura_attached_illegally).
        // 704.5n / 704.5p - an Equipment attached to a permanent it can't equip (not a creature,
        // gone, itself, or the Equipment is a creature without reconfigure, 301.5c), and any
        // other non-Aura permanent attached to something, becomes unattached and stays on the
        // battlefield.
        if (cd && !cd->enchant_filter.empty()) {
            if (aura_attached_illegally(entity, *cd, perm))
                out.put_into_graveyard(
                    entity, name + " is put into the graveyard (Aura not attached to a legal object)");
        } else if (!perm.equipped_to.empty() &&
                   !(cd && cd->is_equipment && perm.equipped_to.get() != 0 &&
                     equipment_can_equip(entity, perm.equipped_to.get()))) {
            out.unattach.push_back(entity);
        }

        // CR 714.4 - Saga sacrifice. A Saga whose lore counters are >= its final chapter number,
        // and which isn't the source of a chapter ability that has triggered but not yet left the
        // stack (tracked by Permanent::saga_chapters_in_flight), is sacrificed by its controller.
        // This holds for a Saga that is also a creature (Summon: Bahamut) — the printed "Sacrifice
        // after IV" / the script's chapter count governs, independent of its other types. Note: the
        // checked-in CR 714.4 text has no creature-Saga exception, and both cards' Oracle text
        // ("Sacrifice after III/IV") confirms the creature Saga is sacrificed, so they agree.
        // Layer-6 ability removal: a Saga turned into a Mountain (Magus of the Moon,
        // CR 305.7) is no longer a Saga — 714.4 doesn't sacrifice it, even at full lore.
        if (cd && card_is_saga(*cd) && !perm.abilities_removed &&
            get_counters(entity, "LORE") >= static_cast<int>(cd->saga_chapters.size()) &&
            perm.saga_chapters_in_flight == 0)
            out.put_into_graveyard(entity, name + " is sacrificed (final chapter completed).");

        // 704.5q - if a permanent has both a +1/+1 and a -1/-1 counter, N of each are
        // removed, where N is the smaller of the two counts (122.3 annihilation).
        int n = std::min(get_counters(entity, "P1P1"), get_counters(entity, "M1M1"));
        if (n > 0) out.annihilate.emplace_back(entity, n);
    }
}

static bool choose_legend_rule_keep(Game &game, const std::set<Entity> &entities,
                                    SbaActions &out) {
    // 704.5j - legend rule: a player who controls two or more legendary permanents with
    // the same name chooses one to keep; the rest go to their owners' graveyards. Affected
    // players choose in APNAP order (active player first). One conflict is chosen per check and
    // performed with the check's other actions; the loop's next check finds any other conflict.
    Zone::Ownership legend_order[2] = {active_seat(), opponent_of(active_seat())};
    for (Zone::Ownership owner : legend_order) {
        std::map<std::string, std::vector<Entity>> by_name;
        for (auto entity : entities) {
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
            // Latched-answer site (tag SBE_LATCHED): the keep choice is a loop-top pending
            // decision, asked before any of this check's actions is performed. Re-finding is
            // deterministic — on resume the re-run finds the same actions on the frozen state,
            // legend_order re-derives from the unchanged player_a_turn, and the name-sorted
            // by_name map yields the same first conflict — so the key (owner + conflicting name +
            // menu size) provably re-derives.
            uint64_t key = pq_key(SbeSite::LEGEND_KEEP, owner == Zone::PLAYER_A,
                                  std::hash<std::string>{}(grp.first), choices.size());
            int keep = -1;
            if (!pq_take_latched(key, &keep)) {
                game_log("Legend rule: %s controls %zu copies of %s; choose one to keep.\n",
                         player_name(owner).c_str(), grp.second.size(), grp.first.c_str());
                if (in_main_loop()) {
                    // Park the choice (priority persisted at the chooser) and suspend the whole
                    // SBE call; the triggers collected so far wait in Game::waiting_triggers.
                    pq_arm_sbe(key, std::move(choices), owner, /*decision_source=*/0);
                    return false;
                }
                // Blocking fallback for an SBE call outside the main loop.
                // Since the pregame gate (Batch 13) even the preplaced-preset
                // SBE pass runs inside the loop, so this is defensive only.
                // The controller of the duplicates chooses which to keep;
                // point priority at them so the query routes/observes/records
                // from their perspective (SBAs run regardless of who
                // currently holds priority).
                bool prev_priority = game.priority.player_a_has_priority;
                game.priority.player_a_has_priority = (owner == Zone::PLAYER_A);
                keep = InputLogger::instance().get_input(choices);
                game.priority.player_a_has_priority = prev_priority;
            }
            Entity kept = grp.second[static_cast<size_t>(keep)];
            for (auto e : grp.second)
                if (e != kept)
                    out.put_into_graveyard(
                        e, entity_name(e) + " is put into the graveyard (legend rule)");
            return true;
        }
    }
    return true;
}

static void perform_permanent_sbas(std::shared_ptr<Orderer> orderer, const SbaActions &actions) {
    auto moving = [&](Entity e) {
        for (const auto &[m, line] : actions.to_graveyard)
            if (m == e) return true;
        return false;
    };
    for (auto [entity, n] : actions.annihilate) {
        if (moving(entity)) continue;
        add_counters(entity, "P1P1", -n);
        add_counters(entity, "M1M1", -n);
        game_log("%s: %d +1/+1 and %d -1/-1 counter(s) annihilate.\n",
                 entity_name(entity).c_str(), n, n);
    }
    for (Entity entity : actions.unattach) {
        if (moving(entity)) continue;
        game_log("%s becomes unattached\n", entity_name(entity).c_str());
        global_coordinator.GetComponent<Permanent>(entity).equipped_to = ObjectRef{};
    }
    for (const auto &[entity, line] : actions.to_graveyard) {
        game_log("%s\n", line.c_str());
        orderer->add_to_zone(false, entity, Zone::GRAVEYARD);
    }
}
