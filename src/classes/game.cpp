#include "game.h"

#include <algorithm>

#include "../cli_output.h"
#include "../day_night.h"
#include "../components/creature.h"
#include "../components/damage.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"
#include "../ecs/events.h"
#include "../effects/effects.h"
#include "../error.h"
#include "../queries/activation.h"
#include "../queries/battlefield.h"
#include "../queries/combat.h"
#include "../queries/counters.h"
#include "../queries/players.h"
#include "../queries/types.h"
#include "../mana_system.h"
#include "../saga.h"
#include "../systems/orderer.h"
#include "../systems/replacement_effects.h"
#include "../systems/stack_manager.h"
#include "../systems/state_manager.h"
#include "deck.h"

extern Coordinator global_coordinator;

static void known_top_remove(int *arr, int pos);
static void known_top_insert(int *arr, int pos, int card_vocab_idx);

bool Game::ready_to_resolve() {
    return priority.a_has_passed && priority.b_has_passed;
}

bool Game::combat_damage_prevented(Entity source, Entity target) const {
    for (const auto &shield : resolved_effects.combat_damage_prevention_shields) {
        const Entity creature = shield.creature.get();
        if (creature == 0) continue;
        if (shield.prevent_as_source && creature == source) return true;
        if (shield.prevent_as_target && creature == target) return true;
    }
    return false;
}

bool Game::combat_damage_shielded(Entity creature) const {
    if (creature == 0) return false;
    for (const auto &shield : resolved_effects.combat_damage_prevention_shields)
        if (shield.creature.get() == creature && (shield.prevent_as_source || shield.prevent_as_target))
            return true;
    return false;
}

void Game::generate_players(const Deck &deck_a, const Deck &deck_b) {
    player_a_entity = gen_player(deck_a);
    player_b_entity = gen_player(deck_b);
}

Entity Game::gen_player(const Deck &deck) {
    Entity player_entity = global_coordinator.CreateEntity();
    Player player;
    player.life_total = 20;
    player.lands_played_this_turn = 0;
    global_coordinator.AddComponent(player_entity, player);
    return player_entity;
}

void Game::set_monarch(Entity player_entity) {
    if (monarch_entity == player_entity) return;  // already the monarch — no change (725.3)
    monarch_entity = player_entity;  // the previous monarch ceases to be the monarch (725.3)
    game_log("%s becomes the monarch.\n",
             player_name(seat_of_player(player_entity)).c_str());
}

void Game::queue_trigger(const Ability &ab, const std::string &log_line) {
    PendingTriggerRT pt;
    pt.ab = ab;
    // Whatever queues it, it is a triggered ability (a reflexive one reuses a DB$ body).
    if (pt.ab.def->ability_type != AbilityDef::TRIGGERED)
        pt.ab.def = derived_ability_def(ab.def, "triggered", 0, [](AbilityDef &d) {
            d.ability_type = AbilityDef::TRIGGERED;
        });
    pt.controller = ab.controller;
    pt.source = ab.source.lki_entity();
    pt.log_line = log_line;
    pt.needs_target = (ab.def->valid_tgts != "N_A" && ab.target.empty() && ab.targets.empty());
    waiting_triggers.push_back(std::move(pt));
}

void Game::end_game(Zone::Ownership result, const std::string &reason) {
    if (ended) return;
    ended = true;
    winner = static_cast<int>(result);
    if (result == Zone::UNKNOWN)
        printf("\n%s - the game is a draw!\n", reason.c_str());
    else
        printf("\n%s - %s wins!\n", reason.c_str(), player_name(result).c_str());
}

void Game::player_loses(Zone::Ownership loser, const std::string &reason) {
    end_game(opponent_of(loser), reason);
}

void Game::players_lose(bool a_loses, bool b_loses, const std::string &reason) {
    if (a_loses && b_loses)
        end_game(Zone::UNKNOWN, reason);
    else if (a_loses)
        player_loses(Zone::PLAYER_A, reason);
    else if (b_loses)
        player_loses(Zone::PLAYER_B, reason);
}

Game::KnownLibraryTop &Game::known_top_library(Zone::Ownership library_owner) {
    return library_owner == Zone::PLAYER_A ? known_top_library_a : known_top_library_b;
}

const Game::KnownLibraryTop &Game::known_top_library(Zone::Ownership library_owner) const {
    return library_owner == Zone::PLAYER_A ? known_top_library_a : known_top_library_b;
}

const int *Game::known_top_library_seen_by(Zone::Ownership library_owner,
                                           Zone::Ownership viewer) const {
    const KnownLibraryTop &k = known_top_library(library_owner);
    return viewer == library_owner ? k.by_owner : k.by_opponent;
}

void Game::clear_known_top_library(Zone::Ownership library_owner) {
    known_top_library(library_owner) = KnownLibraryTop{};
}

void Game::known_top_library_push(Zone::Ownership library_owner, int owner_idx, int opp_idx) {
    KnownLibraryTop &k = known_top_library(library_owner);
    known_top_insert(k.by_owner, 0, owner_idx);
    known_top_insert(k.by_opponent, 0, opp_idx);
}

void Game::known_top_library_remove_pos(Zone::Ownership library_owner, int pos) {
    if (pos < 0 || pos >= KNOWN_TOP_LIBRARY_SIZE) return;
    KnownLibraryTop &k = known_top_library(library_owner);
    known_top_remove(k.by_owner, pos);
    known_top_remove(k.by_opponent, pos);
}

void Game::known_top_library_note(Zone::Ownership library_owner, int pos, int card_vocab_idx,
                                  Zone::Ownership knower) {
    if (pos < 0 || pos >= KNOWN_TOP_LIBRARY_SIZE) return;
    KnownLibraryTop &k = known_top_library(library_owner);
    (knower == library_owner ? k.by_owner : k.by_opponent)[pos] = card_vocab_idx;
}

void Game::known_top_library_move(Zone::Ownership library_owner, int from, int to) {
    KnownLibraryTop &k = known_top_library(library_owner);
    for (int *arr : {k.by_owner, k.by_opponent}) {
        int idx = (from >= 0 && from < KNOWN_TOP_LIBRARY_SIZE) ? arr[from] : -1;
        known_top_remove(arr, from);
        known_top_insert(arr, to, idx);
    }
}

// Removes the entry at `pos` of one known-top array, shifting the deeper ones up (the last
// becomes unknown). No-op outside the window.
static void known_top_remove(int *arr, int pos) {
    if (pos < 0 || pos >= KNOWN_TOP_LIBRARY_SIZE) return;
    for (int i = pos; i < KNOWN_TOP_LIBRARY_SIZE - 1; i++) arr[i] = arr[i + 1];
    arr[KNOWN_TOP_LIBRARY_SIZE - 1] = -1;
}

// Inserts `card_vocab_idx` at `pos` of one known-top array, shifting the entries at pos.. one
// deeper (the deepest falls off the window). No-op outside the window.
static void known_top_insert(int *arr, int pos, int card_vocab_idx) {
    if (pos < 0 || pos >= KNOWN_TOP_LIBRARY_SIZE) return;
    for (int i = KNOWN_TOP_LIBRARY_SIZE - 1; i > pos; i--) arr[i] = arr[i - 1];
    arr[pos] = card_vocab_idx;
}

void Game::pass_priority() {
    if (priority.player_a_has_priority) priority.a_has_passed = true;
    if (!priority.player_a_has_priority) priority.b_has_passed = true;
    priority.player_a_has_priority = !priority.player_a_has_priority;
}

void Game::take_action() {
    // When a player takes an action, reset the pass tracking
    priority.a_has_passed = false;
    priority.b_has_passed = false;
    priority.payment_fail_counts.clear();
}

void Game::end_cleanup_effects(const std::set<Entity> &entities) {
    Zone::Ownership active_player = active_seat();
    // Clear damage from all creatures; reset prowess bonus
    for (Entity entity : entities) {
        if (global_coordinator.entity_has_component<Damage>(entity)) {
            auto &damage = global_coordinator.GetComponent<Damage>(entity);
            damage.damage_counters = 0;
            damage.has_deathtouch_damage = false;
        }
        if (global_coordinator.entity_has_component<Creature>(entity)) {
            auto &cr = global_coordinator.GetComponent<Creature>(entity);
            if (cr.prowess_bonus != 0 || cr.eot_power_bonus != 0 ||
                cr.eot_toughness_bonus != 0) {
                cr.prowess_bonus = 0;
                cr.eot_power_bonus = 0;
                cr.eot_toughness_bonus = 0;
                recompute_pt(cr);
            }
            // Drop "until end of turn" keyword grants (e.g. Haste); the
            // static pass re-merges these onto cr.keywords each pass, so
            // clearing the bucket here lets them lapse at cleanup (514.2).
            cr.eot_keywords.clear();
            // "Can't be blocked this turn" (Kappa Cannoneer) lapses at cleanup.
            cr.cant_be_blocked_this_turn = false;
        }
        // "Loses <keyword> until end of turn" (Shadowspear's AB$ AnimateAll |
        // RemoveKeywords$) lapses at cleanup (514.2). On a permanent (any type),
        // so cleared outside the Creature branch above.
        if (global_coordinator.entity_has_component<Permanent>(entity)) {
            auto &perm = global_coordinator.GetComponent<Permanent>(entity);
            perm.removed_keywords_eot.clear();
            // "Until end of turn" Animate (CR 514.2) lapses now: erase the
            // EOT-added types, and if this EOT animate is what made a noncreature
            // permanent a creature (a crewed-by-trigger Vehicle like The
            // Fantasticar), strip its bootstrapped Creature/Damage components so it
            // stops being a creature — unless it is a creature by a permanent means.
            if (!perm.animate_added_types_eot.empty() ||
                perm.animate_make_creature_eot) {
                for (const auto &t : perm.animate_added_types_eot)
                    perm.types.erase(t);
                perm.animate_added_types_eot.clear();
                if (perm.animate_make_creature_eot) {
                    perm.animate_make_creature_eot = false;
                    bool still_creature = perm.animate_make_creature;
                    if (!still_creature &&
                        global_coordinator.entity_has_component<CardData>(entity))
                        still_creature = is_creature_card(
                            global_coordinator.GetComponent<CardData>(entity));
                    if (!still_creature) {
                        if (global_coordinator.entity_has_component<Creature>(entity))
                            global_coordinator.RemoveComponent<Creature>(entity);
                        if (global_coordinator.entity_has_component<Damage>(entity))
                            global_coordinator.RemoveComponent<Damage>(entity);
                    }
                }
            }
        }
    }

    // "You may cast that card this turn" grants (Emry) expire at cleanup (601.3e).
    resolved_effects.may_cast_this_turn.clear();
    // Floating "this turn" triggered abilities (Forth Eorlingas!'s become-monarch
    // trigger, CR 603.7e) last only their turn of creation; drop them at cleanup.
    // A Duration$ UntilYourNextTurn floating trigger (Tamiyo, Seasoned Scholar's +2)
    // survives cleanup — it is removed at its controller's next untap step instead.
    {
        auto &floating = resolved_effects.floating_triggers;
        floating.erase(std::remove_if(floating.begin(), floating.end(),
                                      [](const Ability &ft) { return !ft.def->duration_until_your_next_turn; }),
                       floating.end());
    }
    // Impulse-cast permissions (Amped Raptor / Ugin) last only "this turn" and are
    // cleared here. A persist_until_end_of_next_turn grant (Light Up the Stage's
    // "until the end of your next turn") survives this cleanup and is removed at the
    // caster's NEXT turn's cleanup instead — detected as a later cleanup (turn >
    // grant_turn) whose active player is the grant's caster.
    // A permission whose card left exile (cast, or moved by another effect) is gone with the
    // object it was granted to (CR 400.7). A warp recast permission persists across turns for as
    // long as the card remains in exile; it is never expired by the per-turn cleanup.
    resolved_effects.impulse_cast_permission.purge_stale();
    resolved_effects.impulse_cast_permission.erase_if([&](Entity, const ImpulseCastPermission &g) {
        if (g.warp) return false;
        return !g.persist_until_end_of_next_turn ||
               (g.caster == active_player && turn_state.turn > g.grant_turn);
    });
    // Turn-long continuous effects created by an instant/sorcery (Veil of Summer:
    // "Spells you control can't be countered this turn" + "hexproof from blue and
    // from black until end of turn") lapse at cleanup (CR 514.2).
    resolved_effects.cant_counter_spells_of.clear();
    // "Can't gain life this turn" (Roiling Vortex's {R}) lapses at cleanup (514.2).
    resolved_effects.cant_gain_life_this_turn.clear();
    // Combat-damage prevention shields (Maze of Ith, CR 615) are "this turn" and
    // lapse at cleanup (514.2).
    resolved_effects.combat_damage_prevention_shields.clear();
    resolved_effects.hexproof_from_colors_this_turn.clear();
    // An "until end of turn" player protection-from-everything grant lapses at
    // cleanup; an "until your next turn" grant persists (reverted at that player's
    // untap step instead — see the UNTAP case above).
    resolved_effects.player_protection_from_everything.erase(
        std::remove_if(resolved_effects.player_protection_from_everything.begin(),
                       resolved_effects.player_protection_from_everything.end(),
                       [](const PlayerProtectionFromEverything &p) {
                           return !p.until_your_next_turn;
                       }),
        resolved_effects.player_protection_from_everything.end());
    // An "until end of turn" cast-with-flash permission (a bare CastWithFlash
    // Effect) lapses at cleanup; an "until your next turn" grant (Teferi, Time
    // Raveler's +1) persists, reverted at that player's untap step instead.
    resolved_effects.cast_with_flash_permissions.erase(
        std::remove_if(resolved_effects.cast_with_flash_permissions.begin(),
                       resolved_effects.cast_with_flash_permissions.end(),
                       [](const CastWithFlashPermission &p) {
                           return !p.until_your_next_turn;
                       }),
        resolved_effects.cast_with_flash_permissions.end());
    // "This turn" leave-battlefield delayed triggers (Searing Blood's "when that
    // creature dies this turn") expire unfired at cleanup if the watched object
    // never left the battlefield (CR 603.7b). Reached after the end step, so any
    // death during this turn has already fired the trigger (and removed it).
    delayed_triggers.erase(
        std::remove_if(delayed_triggers.begin(), delayed_triggers.end(),
                       [](const DelayedTrigger &dt) { return dt.expires_end_of_turn; }),
        delayed_triggers.end());
}

// Begin the end of combat step (CR 511). "At end of combat" (CR 511.2): fire the end-of-combat
// step event so end-of-combat delayed triggers (Geist of Saint Traft's "exile that token at end
// of combat") can fire before the combat state is cleared as the step ends.
void Game::begin_end_of_combat_step(Entity active_player_entity) {
    turn_state.step = END_OF_COMBAT;
    Event end_of_combat_event(Events::END_OF_COMBAT_BEGAN);
    end_of_combat_event.SetParam(Params::PLAYER, active_player_entity);
    global_coordinator.SendEvent(end_of_combat_event);
}

// Begin a cleanup step (CR 514): turn_state.step becomes CLEANUP with no player holding priority, its
// 514.2 actions still to happen, and CLEANUP_BEGAN fired for "at the beginning of the cleanup
// step" abilities.
void Game::begin_cleanup_step(Entity active_player_entity) {
    turn_state.step = CLEANUP;
    turn_state.cleanup_effects_ended = false;
    turn_state.cleanup_sba_performed = false;
    turn_state.cleanup_priority_round = false;
    Event cleanup_event(Events::CLEANUP_BEGAN);
    cleanup_event.SetParam(Params::PLAYER, active_player_entity);
    global_coordinator.SendEvent(cleanup_event);
}

bool Game::advance_step(std::shared_ptr<StackManager> stack_manager, std::shared_ptr<Orderer> orderer) {
    // will advance step and return true if step advanced
    // otherwise will resove stack or pass priority as needed
    // CR 502.4: no player receives priority during the untap step. The step's
    // turn-based actions (phasing, day/night check, untapping) run in the UNTAP
    // case below and the step advances straight to upkeep without a decision
    // window, regardless of pass tracking — this also covers the start of the
    // game, where turn_state.step begins at UNTAP with neither player having passed.
    // Nothing is resolved off the stack here either: an ability that triggers
    // during the untap step waits until a player would receive priority during
    // the upkeep (CR 603.3b), so it stays on the stack for the normal upkeep
    // priority round after the step change.
    if (ready_to_resolve() || turn_state.step == UNTAP) {
        // CR 514.3a: a state-based action performed or a triggered ability put on the stack
        // during the cleanup step gives the active player priority (the step began with no
        // player holding it); another cleanup step follows once that priority round ends.
        if (turn_state.step == CLEANUP && !turn_state.cleanup_priority_round && !resolution.active &&
            (turn_state.cleanup_sba_performed || !stack_manager->is_empty())) {
            turn_state.cleanup_priority_round = true;
            priority.player_a_has_priority = turn_state.player_a_turn;
            priority.a_has_passed = false;
            priority.b_has_passed = false;
            return false;
        }
        if (!stack_manager->is_empty() && turn_state.step != UNTAP) {
            stack_manager->resolve_top(orderer);
            // A suspended resolution parked its decision for the loop top:
            // LEAVE the pass flags set, so the next iteration's advance_step
            // naturally re-enters resolve_top — the resume path. (Unreachable
            // until a handler is flipped suspendable.)
            if (resolution.active) return true;
            // reset pass tracking when something has resolved
            priority.a_has_passed = false;
            priority.b_has_passed = false;
            // remaining in current step
            return false;
        } else {
            // stack is empty and both players have passed
            //  step is changing
            Entity active_player_entity = turn_state.player_a_turn ? player_a_entity : player_b_entity;
            Zone::Ownership active_player = active_seat();

            switch (turn_state.step) {
                case UNTAP: {
                    // Lapse any "until your next turn" player protection-from-everything grant
                    // protecting this player (The One Ring) — its duration ends as the protected
                    // player's next turn begins.
                    resolved_effects.player_protection_from_everything.erase(
                        std::remove_if(resolved_effects.player_protection_from_everything.begin(),
                                       resolved_effects.player_protection_from_everything.end(),
                                       [active_player](const PlayerProtectionFromEverything &p) {
                                           return p.until_your_next_turn && p.player == active_player;
                                       }),
                        resolved_effects.player_protection_from_everything.end());
                    // Lapse any "until your next turn" cast-with-flash permission this player was
                    // granted (Teferi, Time Raveler's +1) — its duration ends as the controller's
                    // next turn begins (CR 611.2).
                    resolved_effects.cast_with_flash_permissions.erase(
                        std::remove_if(resolved_effects.cast_with_flash_permissions.begin(),
                                       resolved_effects.cast_with_flash_permissions.end(),
                                       [active_player](const CastWithFlashPermission &p) {
                                           return p.until_your_next_turn && p.controller == active_player;
                                       }),
                        resolved_effects.cast_with_flash_permissions.end());
                    // Lapse any "until your next turn" floating triggered ability this player
                    // created (Tamiyo, Seasoned Scholar's +2 "until your next turn, whenever ...")
                    // — its duration ends as the controller's next turn begins (CR 611.2).
                    {
                        auto &floating = resolved_effects.floating_triggers;
                        floating.erase(std::remove_if(floating.begin(), floating.end(),
                                                      [active_player](const Ability &ft) {
                                                          return ft.def->duration_until_your_next_turn &&
                                                                 ft.controller == active_player;
                                                      }),
                                       floating.end());
                    }
                    // Phase in phased-out permanents controlled by active player (CR 702.26a).
                    // An Aura or Equipment that phased out indirectly phases in only along with
                    // the permanent it is attached to (CR 702.26g), inside effects::phase_in.
                    {
                        std::vector<Entity> phasing_in;
                        for (auto entity : orderer->mEntities) {
                            if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
                            // Lapse any "until your next turn" Animate (Karn +1) this player created —
                            // its longer continuous-effect duration ends as their next turn begins.
                            effects::revert_until_turn_animate(entity, active_player);
                            auto &perm_phase = global_coordinator.GetComponent<Permanent>(entity);
                            if (perm_phase.controller == active_player && perm_phase.is_phased_out &&
                                !perm_phase.phased_out_indirectly)
                                phasing_in.push_back(entity);
                        }
                        for (auto entity : phasing_in) effects::phase_in(entity, orderer->mEntities);
                    }
                    // Second part of the untap step (CR 502.2 / 731.2): the day/night turn-based
                    // check, based on the turn that just ended. Runs after phasing, before untap.
                    day_night_untap_transition(orderer->mEntities);
                    // Untap all permanents controlled by active player. Once-each-turn activation
                    // gates (ActivationLimit$, CR 602.5b; loyalty, CR 606.3) reset for EVERY
                    // battlefield permanent: "each turn" includes the opponent's turns. A
                    // phased-out permanent is skipped; it phases in only at its controller's
                    // untap step, where this loop then resets it.
                    for (Entity entity : orderer->mEntities) {
                        if (!is_battlefield_permanent(entity)) continue;

                        auto &permanent = global_coordinator.GetComponent<Permanent>(entity);
                        reset_permanent_activations_this_turn(permanent);
                        if (permanent.controller == active_player) {
                            // Untap-prevention (Choke; rule 614.1d) is a replacement effect:
                            // dispatch an UNTAP event and skip untapping if it is replaced.
                            ReplacementEvent rev;
                            rev.type = ReplacementEvent::UNTAP;
                            rev.entity = entity;
                            rev.affected_player = active_player;
                            replacement::dispatch(rev);
                            // Stun counters (CR 122.1d): "If a permanent with a stun counter would
                            // become untapped, instead remove a stun counter from it." A tapped
                            // permanent with one or more STUN counters stays tapped and sheds one
                            // counter rather than untapping; otherwise it untaps normally.
                            if (!rev.skip_untap) {
                                // Counter type is stored verbatim from the script's CounterType$
                                // (Forge writes "Stun", CR 122.1d), so match that exact key.
                                if (permanent.is_tapped && get_counters(entity, "Stun") > 0) {
                                    add_counters(entity, "Stun", -1);
                                    game_log("%s has a stun counter removed instead of untapping.\n",
                                             permanent.name.c_str());
                                } else {
                                    permanent.is_tapped = false;
                                }
                            }
                            permanent.has_summoning_sickness = false;  // Clear summoning sickness
                        }
                    }
                    turn_state.step = UPKEEP;
                    {
                        Event upkeep_event(Events::UPKEEP_BEGAN);
                        upkeep_event.SetParam(Params::PLAYER, active_player_entity);
                        global_coordinator.SendEvent(upkeep_event);
                    }
                    break;
                }
                case UPKEEP:
                    turn_state.step = DRAW;
                    // Fire DRAW_STEP_BEGAN before drawing
                    {
                        Event draw_step_event(Events::DRAW_STEP_BEGAN);
                        draw_step_event.SetParam(Params::PLAYER, active_player_entity);
                        global_coordinator.SendEvent(draw_step_event);
                    }
                    // first turn first player skips draw!
                    if (turn_state.turn == 0) break;
                    // PLAYER_DREW_CARD is fired per-card inside the draw batch
                    // (with the first-card-in-draw-step flag), so no emit here.
                    // The turn-based draw runs as a resumable batch (pending.query
                    // tag TURN_DRAW): a dredge draw-replacement question (CR
                    // 702.52a) parks as a loop-top decision instead of blocking.
                    pending.draw.active = true;
                    pending.draw.player = active_player;
                    pending.draw.remaining = 1;
                    resume_pending_draws(*this, orderer);
                    // A dredge question parked the draw: return with the pass
                    // flags left true and the post-switch epilogue below DEFERRED
                    // — the parked query must be emitted against exactly the
                    // state the blocking prompt read (priority at the drawer,
                    // pass flags set, mana pools not yet emptied). The loop-top
                    // TURN_DRAW dispatch runs the epilogue via
                    // finish_suspended_turn_draw once the batch completes.
                    if (pending.draw.active) return true;
                    break;
                case DRAW:
                    turn_state.step = FIRST_MAIN;
                    {
                        Event first_main_event(Events::FIRST_MAIN_BEGAN);
                        first_main_event.SetParam(Params::PLAYER, active_player_entity);
                        global_coordinator.SendEvent(first_main_event);
                    }
                    // CR 714.3c turn-based action: as the active player's precombat main phase
                    // begins, put a lore counter on each Saga they control (firing the next
                    // chapter). Mirrors shed_impending_time_counters' built-in step hook.
                    saga_put_precombat_lore_counters(active_player, orderer->mEntities);
                    break;
                case FIRST_MAIN:
                    turn_state.step = BEGIN_COMBAT;
                    {
                        Event begin_combat_event(Events::BEGIN_COMBAT_BEGAN);
                        begin_combat_event.SetParam(Params::PLAYER, active_player_entity);
                        global_coordinator.SendEvent(begin_combat_event);
                    }
                    break;
                case BEGIN_COMBAT:
                    turn_state.step = DECLARE_ATTACKERS;
                    combat.attackers_declared = false;  // Reset for new combat
                    break;
                case DECLARE_ATTACKERS:
                    combat.blockers_declared = false;  // Reset for new combat
                    // CR 508.8: with no creature attacking (none declared or put onto the
                    // battlefield attacking), the declare blockers and combat damage steps are
                    // skipped.
                    if (std::none_of(orderer->mEntities.begin(), orderer->mEntities.end(),
                                     is_attacking_creature)) {
                        begin_end_of_combat_step(active_player_entity);
                        break;
                    }
                    turn_state.step = DECLARE_BLOCKERS;
                    break;
                case DECLARE_BLOCKERS: {
                    // Scan for first strikers / double strikers
                    combat.has_first_strikers = false;
                    for (auto e : orderer->mEntities) {
                        if (!is_attacking_creature(e) && !is_blocking_creature(e)) continue;
                        if (creature_deals_first_strike_damage(global_coordinator.GetComponent<Creature>(e))) {
                            combat.has_first_strikers = true;
                            break;
                        }
                    }
                    if (combat.has_first_strikers) {
                        turn_state.step = FIRST_STRIKE_DAMAGE;
                    } else {
                        turn_state.step = COMBAT_DAMAGE;
                    }
                    combat.damage_dealt = false;
                    break;
                }
                case FIRST_STRIKE_DAMAGE:
                    turn_state.step = COMBAT_DAMAGE;
                    combat.damage_dealt = false;
                    combat.damage_assignment.clear();  // T3.10: regular step re-decides for survivors
                    break;
                case COMBAT_DAMAGE:
                    begin_end_of_combat_step(active_player_entity);
                    break;
                case END_OF_COMBAT:
                    // Clear all combat state from creatures
                    for (Entity entity : orderer->mEntities) {
                        if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
                        auto &creature = global_coordinator.GetComponent<Creature>(entity);
                        creature.is_attacking = false;
                        creature.attack_target = ObjectRef{};
                        creature.is_blocking = false;
                        creature.blocking_target = ObjectRef{};
                        creature.is_blocked = false;
                    }
                    combat.damage_assignment.clear();  // T3.10: drop any per-attacker assignments
                    turn_state.step = SECOND_MAIN;
                    {
                        Event second_main_event(Events::SECOND_MAIN_BEGAN);
                        second_main_event.SetParam(Params::PLAYER, active_player_entity);
                        global_coordinator.SendEvent(second_main_event);
                    }
                    break;
                case SECOND_MAIN:
                    turn_state.step = END_STEP;
                    {
                        Event end_step_event(Events::END_STEP_BEGAN);
                        end_step_event.SetParam(Params::PLAYER, active_player_entity);
                        global_coordinator.SendEvent(end_step_event);
                    }
                    // CR 702.175e: the "remove a time counter" impending shed is a triggered
                    // ability (produced from END_STEP_BEGAN in collect_triggered_abilities and put
                    // on the stack), not a step side effect — so nothing is done inline here.
                    break;
                case END_STEP:
                    begin_cleanup_step(active_player_entity);
                    break;
                case CLEANUP:
                    // CR 514.3a: players received priority during this cleanup step, so once
                    // the stack is empty and all players pass, another cleanup step begins.
                    if (turn_state.cleanup_priority_round) {
                        begin_cleanup_step(active_player_entity);
                        break;
                    }
                    reset_turn_counters(active_player_entity);

                    // Empty mana pools
                    empty_mana_pool(Zone::PLAYER_A);
                    empty_mana_pool(Zone::PLAYER_B);

                    // End of turn, move to next turn. An extra turn (CR 500.7 / 720) takes
                    // priority over the normal active-player flip: if a player is owed an extra
                    // turn, that player (the most recently added — extra_turns is a LIFO stack)
                    // takes the next turn instead of passing to the opponent.
                    turn_state.step = UNTAP;
                    turn_state.turn++;
                    if (!turn_state.extra_turns.empty()) {
                        Zone::Ownership next_active = turn_state.extra_turns.back();
                        turn_state.extra_turns.pop_back();
                        turn_state.player_a_turn = (next_active == Zone::PLAYER_A);
                    } else {
                        turn_state.player_a_turn = !turn_state.player_a_turn;
                    }
                    break;
            }
            // if the new step is untap or cleanup, we pretend both players passed
            // hacky
            if (turn_state.step == UNTAP || turn_state.step == CLEANUP) {
                priority.a_has_passed = true;
                priority.b_has_passed = true;
            } else {
                // otherwise we now get active player priority
                priority.player_a_has_priority = turn_state.player_a_turn;
                // Reset pass tracking
                priority.a_has_passed = false;
                priority.b_has_passed = false;
            }
            // any case where we are returning true, mana pool is now emptied
            empty_mana_pool(Zone::PLAYER_A);
            empty_mana_pool(Zone::PLAYER_B);
            return true;
        }
    } else {
        // return false if not ready to resolve, meaning someone has priority
        return false;
    }
}

void Game::reset_turn_counters(Entity active_player_entity) {
    // Snapshot the ending turn's active player's spell count before it resets, for the next
    // turn's untap day/night check (CR 502.2 / 731.2). Both players' counts are reset every
    // turn, so it holds only the spells cast during this turn.
    turn_state.prev_turn_active_spell_count = static_cast<int>(
        global_coordinator.GetComponent<Player>(active_player_entity).spells_cast_this_turn);
    global_coordinator.GetComponent<Player>(player_a_entity).reset_turn_counters();
    global_coordinator.GetComponent<Player>(player_b_entity).reset_turn_counters();
    ability_resolution_counts.clear();
    // Miracle (CR 702.94) is a "first card drawn this turn" concept — the reveal opportunity
    // lapses at end of turn, so a never-answered pending reveal decision (e.g. the game ended
    // first) lapses each cleanup.
    pending.miracle_reveal = 0;
}

bool Game::is_mandatory_choice_pending() const {
    // A pending miracle reveal (CR 702.94) is a forced decision the drawing player must make
    // before proceeding, so it rides the mandatory-choice channel alongside pending_choice.
    return pending.choice != NONE || pending.miracle_reveal != 0;
}

void Game::finish_suspended_turn_draw() {
    // Exactly the post-switch epilogue for a priority-bearing step (turn_state.step
    // is DRAW here, never UNTAP/CLEANUP): active player gets priority, pass
    // tracking resets, mana pools empty across the step change.
    priority.player_a_has_priority = turn_state.player_a_turn;
    priority.a_has_passed = false;
    priority.b_has_passed = false;
    empty_mana_pool(Zone::PLAYER_A);
    empty_mana_pool(Zone::PLAYER_B);
}

void resume_pending_draws(Game &game, std::shared_ptr<Orderer> orderer) {
    Game::PendingDrawRT &pd = game.pending.draw;
    while (pd.active) {
        // A latched TURN_DRAW answer from the previous arm: apply it first —
        // draw normally (option 0) or one dredge — restoring the pre-arm
        // priority seat exactly as the blocking prompt's post-get_input
        // restore did.
        if (game.pending.query.active) {
            PendingQuery &pq = game.pending.query;
            if (pq.tag != PendingQuery::TURN_DRAW || !pq.answered)
                fatal_error("resume_pending_draws: foreign or unanswered pending query parked");
            std::vector<replacement::DrawReplacementOption> opts;
            std::vector<LegalAction> menu =
                replacement::collect_draw_replacements(pd.player, &opts);
            // Purity tripwire: nothing runs between suspend and resume, so the
            // re-derived menu must match the parked one.
            if (menu.size() != pq.menu.size())
                fatal_error("resume_pending_draws: menu size changed between arm and resume");
            int choice = pq.answer;
            game.priority.player_a_has_priority = pq.prev_priority;
            pq = PendingQuery{};
            if (choice == 0) {
                // The base draw plus any additive-draw-replacement bonus (CR 614.1/614.5,
                // Quantum Riddler) is applied inside the shared helper.
                orderer->perform_draw_with_bonus(pd.player);
            } else {
                orderer->apply_dredge(pd.player, opts[static_cast<size_t>(choice) - 1].source,
                                      opts[static_cast<size_t>(choice) - 1].mill);
            }
            pd.remaining--;
            continue;
        }
        // Per-draw ended bail, mirroring Orderer::draw's loop guard (a decked
        // draw ends the game mid-batch).
        if (pd.remaining <= 0 || game.ended) {
            pd = Game::PendingDrawRT{};
            return;
        }
        std::vector<replacement::DrawReplacementOption> opts;
        std::vector<LegalAction> menu = replacement::collect_draw_replacements(pd.player, &opts);
        if (menu.empty()) {
            // No dredge applies — the promptless common case, exactly today's
            // draw_one with an empty replacement dispatch. The additive draw replacement
            // (CR 614.1/614.5, Quantum Riddler) bonus is applied inside the helper.
            orderer->perform_draw_with_bonus(pd.player);
            pd.remaining--;
            continue;
        }
        // Park the dredge question for the loop top (tag TURN_DRAW): persist
        // priority at the drawing player (the blocking prompt's repoint) and
        // arm with the ambient pending-decision source (0: a turn-based draw
        // has no asking card; each dredge entry names its own card).
        PendingQuery &pq = game.pending.query;
        pq = PendingQuery{};
        pq.tag = PendingQuery::TURN_DRAW;
        pq.active = true;
        pq.menu = std::move(menu);
        pq.chooser_is_a = (pd.player == Zone::PLAYER_A);
        pq.decision_source = game.pending.decision_source;
        pq.prev_priority = game.priority.player_a_has_priority;
        game.priority.player_a_has_priority = pq.chooser_is_a;
        return;
    }
}
