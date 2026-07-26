#include "action_processor.h"

#include <algorithm>
#include <cstdio>

#include "classes/match_state.h"
#include "cli_output.h"
#include "components/ability.h"
#include "components/carddata.h"
#include "components/color_identity.h"
#include "components/creature.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/spell.h"
#include "components/token.h"
#include "components/zone.h"
#include "ecs/coordinator.h"
#include "ecs/entity.h"
#include "ecs/events.h"
#include "effects/effects.h"
#include "error.h"
#include "game_queries.h"
#include "input_logger.h"
#include "mana_system.h"
#include "parse.h"
#include "search_server.h"
#include "systems/orderer.h"
#include "systems/rules_modifying.h"
#include "systems/state_manager.h"
#include "systems/state_manager_internal.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static std::vector<LegalAction> permanent_choice_menu(const std::vector<Entity> &choices,
                                                      const char *verb, const char *suffix,
                                                      ActionCategory category);
static int select_single_target(Ability &ability, const std::vector<Entity> &valid_targets,
                                bool allow_done, TargetAsker &asker);
static std::string chosen_targets_display(const Ability &ab);
static void process_activate_ability(const LegalAction &action, Game &game, std::shared_ptr<Orderer> orderer);
static void run_activation_flow(Game::PendingActivation &pa, Game &game,
                                std::shared_ptr<Orderer> orderer, int resume_choice);
static std::vector<Entity> build_valid_targets(
    const Ability &ability, std::shared_ptr<Orderer> orderer, Zone::Ownership priority_player);
static void defer_alternate_cost(Game &game, const CardData &card_data, Zone::Ownership caster);
static void declare_attackers(Game &game, std::shared_ptr<Orderer> orderer);
static void park_combat_target_query(Game &game, PendingQuery::Tag tag,
                                     std::vector<LegalAction> &&menu, bool chooser_is_a,
                                     Entity chosen_creature);
static void resume_attack_target(Game &game);
static void resume_block_target(Game &game);
static bool player_controls_land_subtype(Zone::Ownership player, const std::string &subtype);
static std::string landwalk_subtype(const std::string &kw);
static std::vector<Entity> determine_blockable_attackers(Entity blocker, const std::vector<Entity> &attackers);
static void release_illegal_menace_blockers(const std::vector<Entity> &eligible,
                                            const std::vector<Entity> &attackers);
static void declare_blockers(Game &game, std::shared_ptr<Orderer> orderer);
static std::vector<Entity> collect_live_blockers(Entity attacker, std::shared_ptr<Orderer> orderer);
static bool attacker_needs_assignment(Entity attacker, std::shared_ptr<Orderer> orderer, bool first_strike_only);
static bool arm_damage_assign_query(Game &game);
static void finish_pending_attacker(Game &game);
static void run_damage_assignment(Game &game, std::shared_ptr<Orderer> orderer, int resume_choice);
static void assign_combat_damage(Game &game, std::shared_ptr<Orderer> orderer);
static void proc_miracle_reveal(Game &game, std::shared_ptr<Orderer> orderer);
static void proc_miracle_cast(Game &game, std::shared_ptr<Orderer> orderer);
// One Ward ability a permanent currently has (CR 702.21): an unless-cost (generic mana
// amount, or a life amount when is_life) the targeting player must pay or have the spell/
// ability countered. Collected from the printed ward (CardData::ward_cost) and from any
// granted "Ward:N" / "Ward:PayLife<N>" in the effective keyword list.
struct WardInstance {
    int cost;
    bool is_life;
};
static std::vector<WardInstance> collect_ward_instances(Entity e);
static void trigger_ward_for_targets(Entity targeting_entity, Zone::Ownership controller,
                                     const std::vector<Entity> &targets,
                                     std::shared_ptr<Orderer> orderer);
static void fire_became_target_events(Entity targeting_entity, Zone::Ownership controller,
                                      const std::vector<Entity> &targets);
static void fire_targeting_hooks(Entity targeting_entity, Zone::Ownership controller,
                                 const Ability &targeting_ab, std::shared_ptr<Orderer> orderer);
static std::vector<LegalAction> escape_exile_menu(Zone::Ownership caster, Entity spell_entity,
                                                  std::shared_ptr<Orderer> orderer);
static std::vector<const Ability *> spell_targeting_abilities(const Ability &primary);
static bool gift_mode_satisfiable(const std::vector<const Ability *> &targeting,
                                  std::shared_ptr<Orderer> orderer, Zone::Ownership caster,
                                  bool promised);
static bool charm_mode_choosable(Ability &candidate, std::shared_ptr<Orderer> orderer,
                                 Zone::Ownership caster);
static std::string charm_mode_desc(const Ability &ability, size_t idx);
static std::vector<LegalAction> build_charm_mode_menu(Ability &ability,
                                                      std::shared_ptr<Orderer> orderer,
                                                      Zone::Ownership caster,
                                                      const std::vector<bool> &taken,
                                                      std::vector<size_t> &mode_indices);
static void announce_charm_modes(Ability &ability, std::shared_ptr<Orderer> orderer,
                                 Zone::Ownership caster);
static std::vector<LegalAction> optional_yesno_menu(const std::string &prompt);
static void arm_flow_query(Game &game, PendingQuery::Tag tag, std::vector<LegalAction> &&menu,
                           Zone::Ownership chooser, Entity decision_source = 0);
static void arm_cast_query(Game &game, std::vector<LegalAction> &&menu, Zone::Ownership chooser,
                           Entity decision_source = 0);
static void run_cast_flow(Game::PendingCast &pc, Game &game, std::shared_ptr<Orderer> orderer,
                          int resume_choice);

// entity_name() is shared from the StateManager TUs via state_manager_internal.h.
// mana_symbol_str() is the canonical const-char* color symbol from classes/colors.h.

// Build the menu of permanents for a choose-one cost pick. `verb` and `suffix`
// frame the label, e.g. ("Sacrifice ", "") or ("Return ", " to hand"). menu[i]
// carries choices[i] as its source_entity, so a latched answer indexes straight
// back into the candidate list. (The menu-building half of the old blocking
// prompt_permanent_choice — the get_input half is now a parked ACTIVATION
// pending decision at each of its former call sites.)
static std::vector<LegalAction> permanent_choice_menu(const std::vector<Entity> &choices,
                                                      const char *verb, const char *suffix,
                                                      ActionCategory category) {
    std::vector<LegalAction> menu;
    for (auto e : choices) {
        std::string nm = global_coordinator.GetComponent<Permanent>(e).name;
        LegalAction la(PASS_PRIORITY, e, std::string(verb) + nm + suffix);
        la.category = category;
        menu.push_back(la);
    }
    return menu;
}

// Has `e` already been CHOSEN as a cost item for the in-flight cast? The pick loops
// re-derive their candidate menu each pass and used to rely on the previous pick having
// already left its zone to drop it from the next menu. With the moves deferred to
// PAY_APPLY the picks are still in place, so every multi-pick cost filters on this
// instead (pitch two cards, sacrifice two Mountains, exile five graveyard cards).
static bool already_chosen_as_cost(const Game::PendingCast &pc, Entity e) {
    for (const auto &r : pc.cost_removals)
        if (r.entity == e) return true;
    return false;
}

// Record a chosen non-mana cost item. `log` is the narrative line for it, formatted now
// (while the card is still where it is) and emitted when PAY_APPLY performs the move.
static void choose_cost_item(Game::PendingCast &pc, Entity e, Zone::ZoneValue dest,
                             std::string log) {
    pc.cost_removals.push_back({e, dest, std::move(log)});
}

// Drop already-chosen entities from a re-derived cost menu (see already_chosen_as_cost).
static void drop_chosen_cost_items(const Game::PendingCast &pc,
                                   std::vector<LegalAction> &menu) {
    menu.erase(std::remove_if(menu.begin(), menu.end(),
                              [&](const LegalAction &la) {
                                  return already_chosen_as_cost(pc, la.source_entity);
                              }),
               menu.end());
}

// Candidate menu for one Escape ExileFromGrave pick (CR 702.139 / 601.2f): the caster's
// remaining OTHER graveyard cards, in mEntities order. Re-derived from the live graveyard
// at each arm of the DEF_EXILE_TYPES / DEF_EXILE_COUNT steps (each exile drops that card
// from the next menu), exactly the per-iteration rebuild the old blocking loops did.
// menu[i].source_entity is the candidate, so the resumed apply indexes straight into it.
static std::vector<LegalAction> escape_exile_menu(Zone::Ownership caster, Entity spell_entity,
                                                  std::shared_ptr<Orderer> orderer) {
    std::vector<LegalAction> menu;
    for (auto e : orderer->mEntities) {
        if (e == spell_entity) continue;
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location != Zone::GRAVEYARD || z.owner != caster) continue;
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        std::string nm = global_coordinator.GetComponent<CardData>(e).name;
        LegalAction la(PASS_PRIORITY, e, "Exile " + nm + " from graveyard");
        la.category = ActionCategory::EXILE_FROM_YARD;
        menu.push_back(la);
    }
    return menu;
}

// Every chosen target of an ability, joined for the activation announcement. Targets are
// public information as soon as the ability is put on the stack (CR 601.2c), so a
// multi-target activation (e.g. Faerie Macabre's "up to two target cards") must announce
// all of its targets — naming only the first makes the transcript read as if the other
// cards were affected without ever being targeted.
static std::string chosen_targets_display(const Ability &ab) {
    if (ab.targets.empty()) return target_display_name(cur_game, ab.target);
    std::string out;
    for (size_t i = 0; i < ab.targets.size(); i++) {
        if (i > 0) out += (i + 1 == ab.targets.size()) ? " and " : ", ";
        out += target_display_name(cur_game, ab.targets[i]);
    }
    return out;
}

static void process_activate_ability(const LegalAction &action, Game &game, std::shared_ptr<Orderer> orderer) {
    Entity permanent_entity = action.source_entity;
    const Ability &ability = action.ability;

    // Initialize the persisted activation state machine (Game::PendingActivation) from the
    // consumed LegalAction and hand control to run_activation_flow — the extracted
    // ACTIVATE_ABILITY body. The branch's former locals (the ability, the targeted
    // stack_ab copy, the chosen X, the pre-payment equip/ninjutsu candidate lists) live
    // in pa; converted prompts suspend as loop-top pending decisions (tag ACTIVATION)
    // that the main loop emits and resume_activation_flow re-enters with the answer.
    Game::PendingActivation &pa = game.pending_activation;
    if (pa.active) fatal_error("ACTIVATE_ABILITY with an activation flow already in flight");

    // Ninjutsu (CR 702.49): bespoke cost (return an unblocked attacker) and effect (enter tapped
    // and attacking) — its own step pair (NINJA_PAY / NINJA_RETURN), separate from the generic
    // hand-activated-ability path.
    if (ability.is_ninjutsu) {
        if (!global_coordinator.entity_has_component<Zone>(permanent_entity)) return;
        Zone::Ownership ctrl = global_coordinator.GetComponent<Zone>(permanent_entity).owner;
        pa = Game::PendingActivation{};
        pa.active = true;
        pa.step = Game::PendingActivation::NINJA_PAY;
        pa.source_entity = permanent_entity;
        pa.activator_is_a = (ctrl == Zone::PLAYER_A);
        pa.ability = ability;
        pa.stack_ab = ability;
        run_activation_flow(pa, game, orderer, -1);
        return;
    }

    // ActivationZone$ Hand / Graveyard: card activated from a non-battlefield zone (no Permanent
    // component) — e.g. Cycling/Talon Gates from hand, or Unearth (CR 702.84) from the graveyard.
    // Same flow: select targets, pay the cost, push the ability onto the stack (the ZONE_TARGET /
    // ZONE_PAY steps, converging on the shared SECONDARY chain).
    if ((ability.activation_zone == Zone::HAND || ability.activation_zone == Zone::GRAVEYARD) &&
        !global_coordinator.entity_has_component<Permanent>(permanent_entity)) {
        auto &card_zone = global_coordinator.GetComponent<Zone>(permanent_entity);
        Zone::Ownership ctrl = card_zone.owner;
        pa = Game::PendingActivation{};
        pa.active = true;
        pa.step = Game::PendingActivation::ZONE_TARGET;
        pa.source_entity = permanent_entity;
        pa.activator_is_a = (ctrl == Zone::PLAYER_A);
        pa.zone_path = true;
        pa.ability = ability;
        pa.stack_ab = ability;
        run_activation_flow(pa, game, orderer, -1);
        return;
    }

    auto &permanent = global_coordinator.GetComponent<Permanent>(permanent_entity);
    Zone::Ownership controller = permanent.controller;

    // Activation$ gate (CR 602.5): refuse to activate an ability whose "activate only if
    // <condition>" gate (e.g. Mox Opal's Metalcraft) isn't met, so it can't be forced illegally.
    if (!activation_condition_met(ability, controller, orderer->mEntities, permanent_entity)) {
        game_log("Activation condition not met.\n");
        return;
    }

    // EQUIP: special activated ability — attach equipment to a creature (EQUIP_PAY freezes the
    // creature menu and pays; EQUIP_TARGET suspends on the menu).
    if (ability.category == "Equip") {
        pa = Game::PendingActivation{};
        pa.active = true;
        pa.step = Game::PendingActivation::EQUIP_PAY;
        pa.source_entity = permanent_entity;
        pa.activator_is_a = (controller == Zone::PLAYER_A);
        pa.ability = ability;
        pa.stack_ab = ability;
        run_activation_flow(pa, game, orderer, -1);
        return;
    }

    // UNATTACH: Reconfigure (CR 702.151) — pay the cost to detach this equipment from the creature
    // it is attached to. Clears the attach link; the continuous-effects pass restores its
    // creature-ness (a reconfigured permanent isn't a creature only while attached). No menu
    // prompt exists here — the only interactive read is the mana payment, which machine mode
    // auto-resolves with zero decisions — so this path stays synchronous (deliberately
    // unconverted, like the interactive payer itself).
    if (ability.category == "Unattach") {
        if (permanent.equipped_to == 0) {
            game_log("%s is not attached.\n", permanent.name.c_str());
            return;
        }
        ManaValue unattach_cost = effective_activation_mana_cost(ability, controller, orderer);
        if (!unattach_cost.empty()) {
            auto mana_snap = snapshot_mana_state(controller, orderer);
            if (!prompt_mana_payment(controller, unattach_cost, permanent_entity, orderer)) {
                restore_mana_state(controller, mana_snap, orderer);
                cur_game.payment_fail_counts[permanent_entity]++;
                game_log("Payment cancelled.\n");
                return;
            }
        }
        if (global_coordinator.entity_has_component<Permanent>(permanent.equipped_to)) {
            global_coordinator.GetComponent<Permanent>(permanent.equipped_to).equipped_by = 0;
        }
        game_log("%s unattaches.\n", permanent.name.c_str());
        permanent.equipped_to = 0;
        game.take_action();
        return;
    }

    // Generic battlefield activation (mana abilities included — they skip the X ladder and
    // target steps inside the flow and resolve off-stack at FINISH).
    pa = Game::PendingActivation{};
    pa.active = true;
    pa.step = Game::PendingActivation::X_LADDER;
    pa.source_entity = permanent_entity;
    pa.activator_is_a = (controller == Zone::PLAYER_A);
    pa.ability = ability;
    pa.stack_ab = ability;  // not used for mana ability
    run_activation_flow(pa, game, orderer, -1);
}

//  Build the list of legal targets for an ability.
//  Targets are sorted from the caster's perspective: opponent entities first (opponent
//  player, then opponent's permanents in entity-ID order), followed by own entities
//  (own player, then own permanents in entity-ID order).  This keeps action index 0
//  pointing at the opponent player for burn spells regardless of which player is casting,
//  which makes the action space symmetric and simplifies self-play training.
//
//  Legality of each candidate is decided by Ability::is_legal_target (the single source
//  of truth shared with resolution-time re-verification); this function only chooses the
//  candidate set and the order they are presented in.
static std::vector<Entity> build_valid_targets(
    const Ability &ability, std::shared_ptr<Orderer> orderer, Zone::Ownership priority_player) {
    std::vector<Entity> valid_targets;
    const std::string &vt = ability.valid_tgts;

    // Stack targets: spells (counterspells) or standalone abilities (Stifle)
    if (ability.target_type == "Spell" ||
        ability.target_type.find("Activated") != std::string::npos ||
        ability.target_type.find("Triggered") != std::string::npos) {
        for (auto e : orderer->get_stack()) {
            // A spell/ability can't target itself — e.g. a modal spell (Pyroblast/
            // Hydroblast) that picks its target at resolution is still on the stack.
            if (e == ability.source) continue;
            if (ability.is_legal_target(e, priority_player)) valid_targets.push_back(e);
        }
        return valid_targets;
    }

    Zone::Ownership opp = (priority_player == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;

    // Target cards in a graveyard (e.g. Faerie Macabre targeting any graveyard card,
    // Life from the Loam targeting Land.YouCtrl, or targeted reanimation graveyard→
    // battlefield like Lorehold Charm): opponent's graveyard first, then own.
    // is_legal_target applies the type/owner/MV filter, so YouOwn effects only keep the
    // caster's own cards. The destination is irrelevant to where the candidate sits, so
    // a graveyard-origin ChangeZone enumerates the graveyard regardless of destination.
    // target_in_graveyard covers spells that target a graveyard card via a non-ChangeZone
    // vehicle (Surgical Extraction's SP$ Pump with TgtZone$ Graveyard).
    if (ability.target_in_graveyard ||
        (ability.category == "ChangeZone" && ability.origin == Zone::GRAVEYARD)) {
        for (int pass = 0; pass < 2; pass++) {
            Zone::Ownership slot_owner = (pass == 0) ? opp : priority_player;
            for (auto e : orderer->mEntities) {
                if (!global_coordinator.entity_has_component<Zone>(e)) continue;
                if (global_coordinator.GetComponent<Zone>(e).owner != slot_owner) continue;
                if (ability.is_legal_target(e, priority_player)) valid_targets.push_back(e);
            }
        }
        return valid_targets;
    }

    bool any = (vt == "Any");
    bool opp_only = (vt == "Opponent");
    bool inc_players = any || opp_only || vt.find("Player") != std::string::npos;

    // Players: opponent first, self second
    if (inc_players) {
        if (ability.is_legal_target(get_player_entity(opp), priority_player))
            valid_targets.push_back(get_player_entity(opp));
        if (!opp_only && ability.is_legal_target(get_player_entity(priority_player), priority_player))
            valid_targets.push_back(get_player_entity(priority_player));
    }

    // Permanents: two passes — opponent's first, then own (entity-ID order within each group)
    for (int pass = 0; pass < 2; pass++) {
        Zone::Ownership slot_owner = (pass == 0) ? opp : priority_player;
        for (auto entity : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
            if (global_coordinator.GetComponent<Permanent>(entity).controller != slot_owner) continue;
            if (ability.is_legal_target(entity, priority_player)) valid_targets.push_back(entity);
        }
    }
    return valid_targets;
}

// TODO MAKE THIS GENERAL
// DETERMINE the mana + life portion of an alternate cast cost (CR 601.2f) and defer both
// into the payment phase, exactly as the flashback/escape/impulse branches do. Nothing is
// paid here: the alt cost's pick loops (pitch a card from hand, return a permanent, sacrifice)
// and its mana payment all run after targets are chosen, so a Force of Will's pitch and a
// Daze's bounce are no longer spent before the spell has a target — and a payment that then
// fails rewinds without having consumed them (see PendingCast::cost_removals).
static void defer_alternate_cost(Game &game, const CardData &card_data, Zone::Ownership caster) {
    // Mana portion of the alt cost (e.g. Evoke:R) with the active SetCost floor (Trinisphere)
    // folded in — CR 601.2f applies the floor AFTER the alternative cost is substituted for the
    // mana cost, so even a "free" alt cast ({0} Mindbreak Trap, Daze's pitch) pays up to the
    // floor. The non-mana parts of the cost below are unaffected.
    ManaValue alt_mana = floored_alt_mana_cost(card_data, card_data.alt_cost.mana_cost, caster);

    // Record the mana this alternative cost will pay (CR 106/601.2g). A pitch/life alt cost
    // with no mana component (Force of Will, Daze) leaves this 0, which a ValidSA$
    // Spell.ManaSpent EQ0 trigger (Roiling Vortex) reads at cast time. MANA_PAY recomputes it
    // from the deferred cost when there IS one, so the two agree.
    game.pending_cast.mana_spent = static_cast<int>(alt_mana.size());

    // Free alt cost (e.g. Once Upon a Time first spell), with no floor imposed on it
    if (card_data.alt_cost.is_free && alt_mana.empty()) {
        game_log("%s casts for free (alternate cost)\n", player_name(caster).c_str());
        return;
    }

    // Affordability is pre-verified by can_afford_alt (against the same floored cost),
    // so in machine mode the deferred payment always succeeds.
    if (!alt_mana.empty()) {
        game.pending_cast.deferred_mana_cost = alt_mana;
        game.pending_cast.deferred_mana_pending = true;
    }
    if (card_data.alt_cost.life_cost != 0)
        game.pending_cast.deferred_life_cost += card_data.alt_cost.life_cost;
}

// Park a combat target sub-prompt (attack target / block target) as a loop-top
// pending decision (pending_query.h). The chosen-but-uncommitted creature is
// persisted in pending_attacker/pending_blocker; the main loop emits the stored
// menu loop-safely (a legal SNAPSHOT/DETERMINIZE root) and dispatches the answer
// to resume_attack_target/resume_block_target. Priority already sits with the
// chooser at both call sites (advance_step seats the active player for declare
// attackers; declare_blockers repoints to the defender at entry), so the caller
// passes the live chooser and nothing needs restoring on resume.
static void park_combat_target_query(Game &game, PendingQuery::Tag tag,
                                     std::vector<LegalAction> &&menu, bool chooser_is_a,
                                     Entity chosen_creature) {
    if (tag == PendingQuery::ATTACK_TARGET)
        game.pending_attacker = chosen_creature;
    else
        game.pending_blocker = chosen_creature;
    PendingQuery &pq = game.pending_query;
    pq.tag = tag;
    pq.menu = std::move(menu);
    pq.chooser_is_a = chooser_is_a;
    // Byte-compat: today's inline sub-prompt runs with pending_decision_source
    // == 0 (no PendingDecisionScope wraps it), and the replay corpus decodes
    // that state field into a "Pending:" transcript line — a nonzero source
    // here drifts the corpus. Keep 0; enriching the observation with the
    // chosen creature is a deliberate (corpus re-record) change for later.
    pq.decision_source = 0;
    pq.answered = false;
    pq.answer = -1;
    pq.active = true;
}

// Commit a parked attack-target answer: exactly the post-get_input code the
// inline sub-prompt ran (attack flag + target + narrative). Declaration events
// (CREATURE_ATTACKED / ATTACKERS_DECLARED / exalted) still fire at confirm time
// in declare_attackers, from the is_attacking flags. The next loop iteration
// re-derives DECLARE_ATTACKERS_CHOICE (attackers_declared is still false) and
// re-enters declare_attackers to continue the declaration.
static void resume_attack_target(Game &game) {
    PendingQuery &pq = game.pending_query;
    Entity chosen_attacker = game.pending_attacker;
    auto &cr = global_coordinator.GetComponent<Creature>(chosen_attacker);
    cr.is_attacking = true;
    cr.attack_target = pq.menu[static_cast<size_t>(pq.answer)].source_entity;
    game_log("%s attacking %s.\n", entity_name(chosen_attacker).c_str(),
        target_display_name(game, cr.attack_target).c_str());
    game.pending_attacker = 0;
    pq = PendingQuery{};
}

// Commit a parked block-target answer: the post-get_input code of the inline
// sub-prompt (block flag + target + attacker's is_blocked + narrative). Menace
// legality is still resolved at confirm time (release_illegal_menace_blockers).
static void resume_block_target(Game &game) {
    PendingQuery &pq = game.pending_query;
    Entity chosen = game.pending_blocker;
    auto &cr = global_coordinator.GetComponent<Creature>(chosen);
    cr.is_blocking = true;
    cr.blocking_target = pq.menu[static_cast<size_t>(pq.answer)].source_entity;
    // Mark the attacker as blocked. It stays blocked for the rest of combat even if this
    // (and every other) blocker later leaves combat (509.1h), so it assigns no damage to
    // the player unless it has trample.
    if (global_coordinator.entity_has_component<Creature>(cr.blocking_target))
        global_coordinator.GetComponent<Creature>(cr.blocking_target).is_blocked = true;
    game_log("%s blocking %s.\n", entity_name(chosen).c_str(),
        entity_name(cr.blocking_target).c_str());
    game.pending_blocker = 0;
    pq = PendingQuery{};
}

// Loop-top dispatcher entry (game_driver.cpp) for both combat target tags.
void resume_combat_target_choice(Game &game) {
    if (game.pending_query.tag == PendingQuery::ATTACK_TARGET)
        resume_attack_target(game);
    else
        resume_block_target(game);
}

static void declare_attackers(Game &game, std::shared_ptr<Orderer> orderer) {
    Zone::Ownership active_player = game.player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B;
    Entity defending_entity = game.player_a_turn ? game.player_b_entity : game.player_a_entity;
    if (game.pending_attacker != 0)
        fatal_error("declare_attackers entered with an attack-target sub-prompt parked");

    // Collect eligible attackers with stable indices
    std::vector<Entity> eligible;
    for (auto entity : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
        auto &permanent = global_coordinator.GetComponent<Permanent>(entity);
        if (permanent.controller != active_player) continue;
        if (permanent.is_tapped) continue;
        if (permanent.has_summoning_sickness) {
            auto &cr = global_coordinator.GetComponent<Creature>(entity);
            bool has_haste = false;
            for (const auto &kw : cr.keywords) {
                if (kw == "Haste") { has_haste = true; break; }
            }
            if (!has_haste) continue;
        }
        // A creature a CantAttack static forbids from attacking (Ensnaring Bridge: power greater
        // than the controller's hand size) is never eligible (CR 509.1a) — not offered and not
        // forced by a "must attack" effect, since it isn't able to attack.
        if (rules_mod::attack_prohibited(entity)) continue;
        eligible.push_back(entity);
    }

    if (eligible.empty()) {
        game_log("No creatures eligible to attack.\n");
        game.attackers_declared = true;
        game.pending_choice = NONE;
        return;
    }

    // Build targets: defending player first, then the defending player's planeswalkers (rule 508.1).
    Zone::Ownership defending_owner = game.player_a_turn ? Zone::PLAYER_B : Zone::PLAYER_A;
    std::vector<Entity> targets;
    targets.push_back(defending_entity);
    for (auto e : orderer->mEntities) {
        if (!is_battlefield_permanent(e, defending_owner)) continue;
        auto &p = global_coordinator.GetComponent<Permanent>(e);
        if (is_planeswalker(p.types)) targets.push_back(e);
    }

    // Pre-declare must_attack creatures — they attack without player input
    for (auto entity : eligible) {
        auto &cr = global_coordinator.GetComponent<Creature>(entity);
        if (!cr.must_attack || cr.is_attacking) continue;
        cr.is_attacking = true;
        cr.attack_target = defending_entity;
        game_log("%s must attack and is declared as an attacker.\n",
            global_coordinator.GetComponent<Permanent>(entity).name.c_str());
    }

    // Selection loop — only un-declared creatures are offered each iteration.
    // Once a creature is declared as an attacker it cannot be removed.
    while (true) {
        // Build list of creatures not yet declared as attackers
        std::vector<Entity> not_yet_attacking;
        for (auto entity : eligible) {
            auto &cr = global_coordinator.GetComponent<Creature>(entity);
            if (!cr.is_attacking) not_yet_attacking.push_back(entity);
        }

        game_log("\n--- Declare Attackers (%s) ---\n", player_name(active_player).c_str());
        // Show already-declared attackers
        for (auto entity : eligible) {
            auto &cr = global_coordinator.GetComponent<Creature>(entity);
            if (!cr.is_attacking) continue;
            std::string ename = entity_name(entity);
            game_log("  [attacking] %s [%d/%d] -> %s\n", ename.c_str(), cr.power, cr.toughness,
                target_display_name(game, cr.attack_target).c_str());
        }
        // Build attacker selection actions
        std::vector<LegalAction> atk_actions;
        for (auto entity : not_yet_attacking) {
            std::string ename = entity_name(entity);
            auto &cr = global_coordinator.GetComponent<Creature>(entity);
            LegalAction la(PASS_PRIORITY, entity,
                ename + " [" + std::to_string(cr.power) + "/" + std::to_string(cr.toughness) + "]");
            la.category = ActionCategory::SELECT_ATTACKER;
            atk_actions.push_back(la);
        }
        {
            LegalAction confirm(PASS_PRIORITY, std::string("Confirm attackers"));
            confirm.category = ActionCategory::CONFIRM_ATTACKERS;
            atk_actions.push_back(confirm);
        }
        // Loop-safe: partial declarations live entirely in Creature components,
        // so a restored snapshot re-derives this same menu. The attack-target
        // sub-prompt below is loop-safe too: it is parked as a pending query
        // (the chosen attacker persisted in Game::pending_attacker) and emitted
        // at the main-loop top.
        search_set_loop_safe(true);
        int creature_choice = InputLogger::instance().get_input(atk_actions);
        search_set_loop_safe(false);

        if (creature_choice == static_cast<int>(atk_actions.size()) - 1) break;

        Entity chosen_attacker = not_yet_attacking[static_cast<size_t>(creature_choice)];
        auto &cr = global_coordinator.GetComponent<Creature>(chosen_attacker);
        std::string chosen_name = entity_name(chosen_attacker);

        std::vector<LegalAction> tgt_actions;
        for (auto t_entity : targets) {
            std::string label;
            if (global_coordinator.entity_has_component<Player>(t_entity)) {
                auto &player = global_coordinator.GetComponent<Player>(t_entity);
                Zone::Ownership t = (t_entity == game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
                label = player_name(t) + " (" + std::to_string(player.life_total) + " life)";
            } else {
                auto &p = global_coordinator.GetComponent<Permanent>(t_entity);
                label = p.name + " (loyalty " + std::to_string(get_counters(t_entity, "LOYALTY")) + ")";
            }
            LegalAction la(PASS_PRIORITY, t_entity, label);
            la.category = ActionCategory::ATTACK_TARGET;
            tgt_actions.push_back(la);
        }
        // Only prompt for a target when there is a real choice (the defending
        // player plus at least one planeswalker). With just the defending player
        // the target is forced, so skip the decision and auto-assign it.
        if (tgt_actions.size() > 1) {
            game_log("Select target for %s:\n", chosen_name.c_str());
            // Park the sub-prompt as a loop-top pending decision (loop-safe
            // search root); the resume commits the attack and the next
            // iteration re-enters this declaration. Nothing else is touched —
            // attackers_declared stays false, pending_choice re-derives.
            park_combat_target_query(game, PendingQuery::ATTACK_TARGET,
                std::move(tgt_actions), game.player_a_turn, chosen_attacker);
            return;
        }
        cr.is_attacking = true;
        cr.attack_target = tgt_actions[0].source_entity;
        game_log("%s attacking %s.\n", chosen_name.c_str(),
            target_display_name(game, cr.attack_target).c_str());
    }

    game_log("\nAttackers declared:\n");
    bool any = false;
    for (auto entity : eligible) {
        auto &cr = global_coordinator.GetComponent<Creature>(entity);
        if (cr.is_attacking) {
            any = true;
            std::string ename = entity_name(entity);
            game_log("  %s -> %s\n", ename.c_str(), target_display_name(game, cr.attack_target).c_str());

            // Tap the attacker, unless it has vigilance (702.21).
            auto &permanent = global_coordinator.GetComponent<Permanent>(entity);
            if (!creature_has_keyword(cr, "Vigilance"))
                permanent.is_tapped = true;

            // Fire a per-attacker "whenever this creature attacks" event (508.2 attack
            // declaration), so triggers like Mobilize go on the stack for each attacker.
            Entity actrl_entity = (active_player == Zone::PLAYER_A)
                                  ? game.player_a_entity : game.player_b_entity;
            Event attacked_ev(Events::CREATURE_ATTACKED);
            attacked_ev.SetParam(Params::ENTITY, entity);
            attacked_ev.SetParam(Params::PLAYER, actrl_entity);
            global_coordinator.SendEvent(attacked_ev);
        }
    }
    if (!any) game_log("  (none)\n");

    // "Whenever you attack" (Mode$ AttackersDeclared) — a player-level trigger that fires once
    // when one or more attackers are declared (508.2), independent of how many. Guide of Souls.
    if (any) {
        Entity actrl_entity = (active_player == Zone::PLAYER_A)
                              ? game.player_a_entity : game.player_b_entity;
        Event declared_ev(Events::ATTACKERS_DECLARED);
        declared_ev.SetParam(Params::PLAYER, actrl_entity);
        global_coordinator.SendEvent(declared_ev);
    }

    // Exalted: if exactly one creature is attacking, fire the event so triggers go on the stack
    int attacker_count = 0;
    Entity sole_attacker = 0;
    for (auto entity : eligible) {
        auto &cr = global_coordinator.GetComponent<Creature>(entity);
        if (cr.is_attacking) { attacker_count++; sole_attacker = entity; }
    }
    if (attacker_count == 1) {
        Entity ctrl_entity = (active_player == Zone::PLAYER_A)
                             ? game.player_a_entity : game.player_b_entity;
        Event exalted_ev(Events::CREATURE_ATTACKED_ALONE);
        exalted_ev.SetParam(Params::ENTITY, sole_attacker);
        exalted_ev.SetParam(Params::PLAYER, ctrl_entity);
        global_coordinator.SendEvent(exalted_ev);
    }

    game.attackers_declared = true;
    game.pending_choice = NONE;
}

static bool player_controls_land_subtype(Zone::Ownership player, const std::string &subtype) {
    Entity max_e = global_coordinator.GetMaxIssuedEntity();
    for (Entity e = 0; e < max_e; e++) {
        if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(e);
        if (perm.controller != player) continue;
        for (auto &t : perm.types) {
            if (t.kind == SUBTYPE && t.name == subtype) return true;
        }
    }
    return false;
}

static std::string landwalk_subtype(const std::string &kw) {
    // "Swampwalk" -> "Swamp", "Forestwalk" -> "Forest", etc.
    if (kw == "Swampwalk") return "Swamp";
    if (kw == "Forestwalk") return "Forest";
    if (kw == "Islandwalk") return "Island";
    if (kw == "Mountainwalk") return "Mountain";
    if (kw == "Plainswalk") return "Plains";
    return "";
}

static std::vector<Entity> determine_blockable_attackers(Entity blocker, const std::vector<Entity> &attackers) {
    auto &bcr = global_coordinator.GetComponent<Creature>(blocker);
    bool blocker_can_fly = false;
    bool blocker_has_shadow = false;
    for (auto &kw : bcr.keywords) {
        if (kw == "Flying" || kw == "Reach") blocker_can_fly = true;
        if (kw == "Shadow") blocker_has_shadow = true;
    }

    // Determine defending player from blocker's controller
    auto &blocker_perm = global_coordinator.GetComponent<Permanent>(blocker);
    Zone::Ownership defending_player = blocker_perm.controller;

    std::vector<Entity> result;
    for (auto atk : attackers) {
        auto &acr = global_coordinator.GetComponent<Creature>(atk);
        // "Can't be blocked this turn" (Kappa Cannoneer): no creature may block it (509.1b).
        if (acr.cant_be_blocked_this_turn) continue;
        bool atk_flying = false;
        bool atk_has_shadow = false;
        bool has_landwalk_evasion = false;
        for (auto &kw : acr.keywords) {
            if (kw == "Flying") atk_flying = true;
            if (kw == "Shadow") atk_has_shadow = true;
            std::string subtype = landwalk_subtype(kw);
            if (!subtype.empty() && player_controls_land_subtype(defending_player, subtype))
                has_landwalk_evasion = true;
        }

        // Shadow: creatures with shadow can only be blocked by shadow creatures,
        // and creatures without shadow cannot be blocked by shadow creatures (rule 702.28)
        if (atk_has_shadow != blocker_has_shadow) continue;

        if (atk_flying && !blocker_can_fly) continue;
        if (has_landwalk_evasion) continue;
        if (has_protection_from(acr, blocker)) continue;
        result.push_back(atk);
    }
    return result;
}

// 702.111b / 509.1b: a creature with menace can only be blocked by two or more creatures.
// After blocks are declared, any menace attacker blocked by exactly one creature has an
// illegal block; the legal resolution is that the lone blocker isn't blocking it. Release
// such lone blockers (clear is_blocking, and the attacker's is_blocked if it now has none).
static void release_illegal_menace_blockers(const std::vector<Entity> &eligible,
                                            const std::vector<Entity> &attackers) {
    for (auto atk : attackers) {
        if (!global_coordinator.entity_has_component<Creature>(atk)) continue;
        auto &acr = global_coordinator.GetComponent<Creature>(atk);
        if (!creature_has_keyword(acr, "Menace")) continue;
        std::vector<Entity> blockers;
        for (auto b : eligible) {
            auto &bcr = global_coordinator.GetComponent<Creature>(b);
            if (bcr.is_blocking && bcr.blocking_target == atk) blockers.push_back(b);
        }
        if (blockers.size() == 1) {
            auto &bcr = global_coordinator.GetComponent<Creature>(blockers[0]);
            bcr.is_blocking = false;
            bcr.blocking_target = 0;
            acr.is_blocked = false;  // no other blocker assigned this attacker
            game_log("%s cannot block %s alone (menace) — block released.\n",
                     entity_name(blockers[0]).c_str(), entity_name(atk).c_str());
        }
    }
}

static void declare_blockers(Game &game, std::shared_ptr<Orderer> orderer) {
    Zone::Ownership defending_player = game.player_a_turn ? Zone::PLAYER_B : Zone::PLAYER_A;
    // defending player declares blockers — priority must be theirs for the input routing to work correctly
    game.player_a_has_priority = !game.player_a_turn;
    if (game.pending_blocker != 0)
        fatal_error("declare_blockers entered with a block-target sub-prompt parked");

    // Collect attackers
    std::vector<Entity> attackers;
    for (auto entity : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
        auto &cr = global_coordinator.GetComponent<Creature>(entity);
        if (cr.is_attacking) attackers.push_back(entity);
    }

    if (attackers.empty()) {
        game_log("No attackers — skipping declare blockers.\n");
        game.blockers_declared = true;
        game.pending_choice = NONE;
        return;
    }

    // Collect eligible blockers: defending player's untapped creatures
    std::vector<Entity> eligible;
    for (auto entity : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
        auto &permanent = global_coordinator.GetComponent<Permanent>(entity);
        if (permanent.controller != defending_player) continue;
        if (permanent.is_tapped) continue;
        eligible.push_back(entity);
    }

    // Remove creatures that can't legally block any attacker (e.g. non-flyers vs all-flying attackers)
    eligible.erase(std::remove_if(eligible.begin(), eligible.end(),
                       [&](Entity blocker) { return determine_blockable_attackers(blocker, attackers).empty(); }),
        eligible.end());

    if (eligible.empty()) {
        game_log("No creatures eligible to block.\n");
        game.blockers_declared = true;
        game.pending_choice = NONE;
        return;
    }

    // Selection loop
    while (true) {
        // Only offer creatures not yet assigned to a blocker slot
        std::vector<Entity> unblocked;
        for (auto entity : eligible) {
            if (!global_coordinator.GetComponent<Creature>(entity).is_blocking) unblocked.push_back(entity);
        }

        game_log("\n--- Declare Blockers (%s) ---\n", player_name(defending_player).c_str());
        game_log("Attackers:\n");
        for (size_t i = 0; i < attackers.size(); i++) {
            std::string aname = entity_name(attackers[i]);
            auto &cr = global_coordinator.GetComponent<Creature>(attackers[i]);
            game_log("  %zu: %s [%d/%d]\n", i, aname.c_str(), cr.power, cr.toughness);
        }
        game_log("Your creatures:\n");
        for (auto entity : eligible) {
            auto &cr = global_coordinator.GetComponent<Creature>(entity);
            if (!cr.is_blocking) continue;
            std::string ename = entity_name(entity);
            std::string atk_name = entity_name(cr.blocking_target);
            game_log("  (assigned) %s [%d/%d] blocking %s\n", ename.c_str(), cr.power, cr.toughness, atk_name.c_str());
        }
        // Build blocker selection actions
        std::vector<LegalAction> blk_actions;
        for (auto entity : unblocked) {
            std::string ename = entity_name(entity);
            auto &cr = global_coordinator.GetComponent<Creature>(entity);
            LegalAction la(PASS_PRIORITY, entity,
                ename + " [" + std::to_string(cr.power) + "/" + std::to_string(cr.toughness) + "]");
            la.category = ActionCategory::SELECT_BLOCKER;
            blk_actions.push_back(la);
        }
        {
            LegalAction confirm(PASS_PRIORITY, std::string("Confirm blockers"));
            confirm.category = ActionCategory::CONFIRM_BLOCKERS;
            blk_actions.push_back(confirm);
        }
        // Loop-safe like the attacker selection: committed blocks live in
        // Creature components; the block-target sub-prompt below is parked as a
        // loop-top pending query (chosen blocker in Game::pending_blocker), so
        // it is loop-safe too.
        search_set_loop_safe(true);
        int blocker_choice = InputLogger::instance().get_input(blk_actions);
        search_set_loop_safe(false);

        if (blocker_choice == static_cast<int>(blk_actions.size()) - 1) {
            // 509.1b / 702.111b: a creature with menace can't be blocked except by two or
            // more creatures. A declaration leaving a menace attacker blocked by exactly one
            // creature is illegal; the only legal resolution is that the lone creature isn't
            // blocking it. Release any such lone blocker (it deals/takes no combat damage)
            // rather than reject the confirm, so the step can never deadlock when no second
            // blocker is available.
            release_illegal_menace_blockers(eligible, attackers);
            break;
        }

        Entity chosen = unblocked[static_cast<size_t>(blocker_choice)];
        std::string chosen_name = entity_name(chosen);

        auto legal_attackers = determine_blockable_attackers(chosen, attackers);
        game_log("Select attacker for %s to block:\n", chosen_name.c_str());
        std::vector<LegalAction> blk_tgt_actions;
        for (auto atk_entity : legal_attackers) {
            std::string aname = entity_name(atk_entity);
            auto &acr = global_coordinator.GetComponent<Creature>(atk_entity);
            LegalAction la(PASS_PRIORITY, atk_entity,
                aname + " [" + std::to_string(acr.power) + "/" + std::to_string(acr.toughness) + "]");
            la.category = ActionCategory::BLOCK_TARGET;
            blk_tgt_actions.push_back(la);
        }
        // Always park (no size-1 pre-collapse exists here — this prompt fires
        // even with a single legal attacker, and the loop-top emitter never
        // collapses single-entry menus). The resume commits the block; the next
        // iteration re-derives DECLARE_BLOCKERS_CHOICE and re-enters here.
        park_combat_target_query(game, PendingQuery::BLOCK_TARGET,
            std::move(blk_tgt_actions), !game.player_a_turn, chosen);
        return;
    }

    game_log("\nBlockers declared:\n");
    bool any = false;
    for (auto entity : eligible) {
        auto &cr = global_coordinator.GetComponent<Creature>(entity);
        if (cr.is_blocking) {
            any = true;
            game_log("  %s blocking %s\n", entity_name(entity).c_str(), entity_name(cr.blocking_target).c_str());
        }
    }
    if (!any) game_log("  (none)\n");

    game.blockers_declared = true;
    game.pending_choice = NONE;
}

// Perspective player for an ability's target search. Ownership-restricted targets
// (.YouOwn/.YouCtrl/.OppOwn — e.g. Emry's "target artifact card in YOUR graveyard") are
// relative to the activating/controlling player, so the existence check must use that player
// rather than a hardcoded placeholder. Derive it from the ability's source: a battlefield
// permanent's controller, else its owning zone, else the ability's stored controller.
static Zone::Ownership ability_perspective_player(const Ability &ability) {
    Entity src = ability.source;
    if (src != 0) {
        if (global_coordinator.entity_has_component<Permanent>(src))
            return global_coordinator.GetComponent<Permanent>(src).controller;
        if (global_coordinator.entity_has_component<Zone>(src))
            return global_coordinator.GetComponent<Zone>(src).owner;
    }
    return ability.controller;
}

bool has_legal_targets(const Ability &ability, std::shared_ptr<Orderer> orderer) {
    if (ability.valid_tgts == "N_A") return true;
    if (ability.target_min == 0) return true;  // optional targeting always has "legal targets"
    // Ordering doesn't affect existence for symmetric targets, but ownership-restricted
    // targets must be evaluated from the controlling player's perspective (see above), or a
    // ".YouOwn" ability could be offered with no legal target and crash on an empty target menu.
    return !build_valid_targets(ability, orderer, ability_perspective_player(ability)).empty();
}

// Effective minimum target count of an ability at cast/activation-legality time. A static
// TargetMin$ uses its literal value. An xPaid-driven min (Kozilek's Command "up to X target")
// is treated as 0 here — X is chosen later and may legally be 0, so it must not gate castability.
// A non-xPaid count-SVar min (Into the Flood Maw: TargetMin$ X = Count$PromisedGift.0.1) is
// evaluated now against the current game state (which reads the pending gift-promise flag, set by
// the caller for each reachable mode).
static int effective_target_min(const Ability &ab, Zone::Ownership perspective,
                                std::shared_ptr<Orderer> orderer) {
    if (ab.target_min_from_xpaid) return 0;
    if (!ab.target_min_count_expr.empty())
        return static_cast<int>(evaluate_dynamic_amount(ab.target_min_count_expr, perspective, orderer, 0));
    return ab.target_min;
}

// Gather a spell's targeting abilities: the primary spell ability plus any chained sub-ability
// that targets (each picks its own target as the spell is cast — CR 601.2c).
static std::vector<const Ability *> spell_targeting_abilities(const Ability &primary) {
    std::vector<const Ability *> targeting;
    if (primary.valid_tgts != "N_A") targeting.push_back(&primary);
    for (const Ability &sub : primary.subabilities)
        if (sub.valid_tgts != "N_A") targeting.push_back(&sub);
    return targeting;
}

// Is a Gift spell's <promised> mode satisfiable — does a legal target exist for every required
// target of its targeting abilities under that gift-promise state? (CR 601.2c). Toggles the
// pending gift-promise flag (which a Count$PromisedGift-driven target count reads) around the
// check and restores it.
static bool gift_mode_satisfiable(const std::vector<const Ability *> &targeting,
                                  std::shared_ptr<Orderer> orderer, Zone::Ownership caster,
                                  bool promised) {
    bool saved = cur_game.pending_gift_promised;
    cur_game.pending_gift_promised = promised;
    bool ok = true;
    for (const Ability *ab : targeting) {
        if (effective_target_min(*ab, caster, orderer) > 0 &&
            build_valid_targets(*ab, orderer, caster).empty()) {
            ok = false;
            break;
        }
    }
    cur_game.pending_gift_promised = saved;
    return ok;
}

// CR 601.2c: a spell can only be cast if a legal target can be chosen for every required target,
// for at least one reachable set of mode/cost choices. Most spells have a single targeting
// ability and this reduces to has_legal_targets. A Gift spell (Into the Flood Maw) switches which
// of its abilities actually requires a target on the gift promise — without the gift it bounces a
// creature (primary ability), with the gift it bounces a nonland permanent (sub-ability) — so it
// is castable iff a legal target exists for the not-promised OR the promised mode. General over
// any spell whose required-target counts depend on the Count$PromisedGift switch: we evaluate each
// reachable promise state and the spell is castable if any one is fully satisfiable.
bool spell_has_castable_targets(const Ability &primary, std::shared_ptr<Orderer> orderer,
                                Zone::Ownership caster, bool has_gift) {
    // Modal spell (CR 601.2b/601.2c): castable iff CharmNum$ DIFFERENT modes can be legally
    // chosen — a choosable mode needs no target, or has a legal target available. The modes
    // live in charm_choices (not subabilities), so the ordinary targeting walk below never
    // sees them; without this a Charm whose every mode lacked a target (Red Elemental Blast
    // with nothing blue anywhere) was offered and then hit an empty mode menu at cast.
    // Mirrors charm_mode_choosable, the filter announce_charm_modes applies at cast (X isn't
    // chosen yet at gate time, so an xPaid-driven target minimum counts as 0 here — X may
    // legally be 0 — matching effective_target_min).
    if (!primary.charm_choices.empty()) {
        int choosable = 0;
        int needed = primary.charm_num < 1 ? 1 : primary.charm_num;
        for (const Ability &mode : primary.charm_choices) {
            Ability probe = mode;
            probe.source = primary.source;
            probe.controller = caster;
            if (mode.valid_tgts == "N_A" ||
                effective_target_min(probe, caster, orderer) <= 0 ||
                !build_valid_targets(probe, orderer, caster).empty()) {
                if (++choosable >= needed) return true;
            }
        }
        return false;
    }

    std::vector<const Ability *> targeting = spell_targeting_abilities(primary);
    if (targeting.empty()) return true;  // no targets required

    if (gift_mode_satisfiable(targeting, orderer, caster, false)) return true;  // not-promised mode
    if (has_gift && gift_mode_satisfiable(targeting, orderer, caster, true)) return true;  // promised
    return false;
}

Ability cast_gate_probe(const Ability &tmpl, Entity card_entity, Zone::Ownership caster) {
    Ability probe = tmpl;
    probe.source = card_entity;
    probe.controller = caster;
    // Chained targeting sub-abilities (Into the Flood Maw's DBChangeZone, Cabal Therapy's
    // DB$ Discard) pick their own target as the spell is cast (CR 601.2c) and are probed via
    // spell_targeting_abilities, so they need the same source/controller. Charm modes are stamped
    // by spell_has_castable_targets from primary.source, so they inherit the stamp set here.
    for (auto &sub : probe.subabilities) {
        sub.source = card_entity;
        sub.controller = caster;
    }
    return probe;
}

// CR 601.2c: when a spell's REQUIRED target count is the X it is cast for (TargetMin$ X, e.g.
// Hide on the Ceiling — "exile X target artifacts and/or creatures", with TargetMin$ X =
// TargetMax$ X), the player can't announce an X larger than the number of legal targets, or
// targeting can't be completed and the spell would be cast/forced with too few targets. Returns
// the cap (legal-target count) the CHOOSE_X menu should clamp X to; returns SIZE_MAX when no
// targeting ability ties its mandatory minimum target count to X, i.e. no target-based cap applies.
static size_t spell_xpaid_target_cap(const CardData &card_data, Entity spell_entity,
                                     Zone::Ownership caster, std::shared_ptr<Orderer> orderer) {
    size_t cap = SIZE_MAX;
    for (const auto &ab : card_data.abilities) {
        if (ab.ability_type != Ability::SPELL) continue;
        for (const Ability *t : spell_targeting_abilities(ab)) {
            if (!t->target_min_from_xpaid) continue;  // only X-driven MANDATORY target counts
            Ability probe = *t;
            probe.source = spell_entity;
            probe.controller = caster;
            cap = std::min(cap, build_valid_targets(probe, orderer, caster).size());
        }
        break;  // single SPELL ability (matches the cast loop's one-ability assumption)
    }
    return cap;
}

// One target pick through `asker` (the shared machine's single-decision step).
// Returns the chosen index, or a negative value when the ask parked a pending
// query (the caller suspends; a resumed re-entry rebuilds the identical menu
// and the asker consumes the latched answer). The pending-decision context is
// the asking source, exactly the PendingDecisionScope the blocking path held.
static int select_single_target(Ability &ability, const std::vector<Entity> &valid_targets,
                                bool allow_done, TargetAsker &asker) {
    // Name the spell/ability asking for the target so the prompt is meaningful
    // before it resolves — otherwise the log only reveals what it was after the
    // target is chosen and the object goes on the stack. Arm-time only: a
    // resume already printed it when the query was armed.
    if (!asker.resuming()) {
        std::string src_name = ability.source != 0 ? entity_name(ability.source)
                                                   : std::string("this ability");
        game_log("Choose target for %s:\n", src_name.c_str());
    }
    std::vector<LegalAction> tgt_actions;
    if (ability.target_min == 0 || allow_done) {
        std::string label = allow_done ? "Done selecting targets" : "No target";
        LegalAction la(PASS_PRIORITY, label);
        la.category = ActionCategory::SELECT_TARGET;
        tgt_actions.push_back(la);
    }
    for (auto target : valid_targets) {
        std::string desc;
        if (global_coordinator.entity_has_component<Player>(target)) {
            auto &player = global_coordinator.GetComponent<Player>(target);
            desc = target_display_name(cur_game, target) + " (" +
                   std::to_string(player.life_total) + " life)";
        } else {
            desc = entity_name(target);
        }
        LegalAction la(PASS_PRIORITY, target, desc);
        la.category = ActionCategory::SELECT_TARGET;
        tgt_actions.push_back(la);
    }
    // INVARIANT (CR 601.2c): the engine must never generate a "choose target" decision with zero
    // options. A mandatory single/multi target (no "No target"/"Done" escape was added above) with
    // an empty candidate list means an upstream cast/activation-legality bug let a spell or ability
    // begin even though it has no legal target. Abort loudly with a diagnostic rather than spinning
    // on an empty menu. (A spell already on the stack whose targets become illegal by RESOLUTION is
    // handled separately — countered by game rules per CR 608.2b — and never reaches this path.)
    if (tgt_actions.empty()) {
        std::string src_name = ability.source != 0 ? entity_name(ability.source) : std::string("(unknown source)");
        fatal_error("Zero legal targets when choosing a required target for " + src_name +
                    " (ValidTgts$ " + ability.valid_tgts + ") — a targeted spell/ability with no "
                    "legal target was offered/forced (CR 601.2c violated upstream).");
    }
    int choice = asker.ask(tgt_actions, ability.source);
    if (choice < 0 && decision_suspended()) return choice;
    ability.target = tgt_actions[static_cast<size_t>(choice)].source_entity;
    game_log("Targeting choice %d\n", choice);
    return choice;
}

TargetStatus run_target_select(Ability &ability, TargetSelectRT &rt, TargetAsker &asker,
                               std::shared_ptr<Orderer> orderer, Zone::Ownership priority_player) {
    if (!rt.active) {
        // Resolve dynamic target counts up front (CR 601.2b: anything they depend on — X, the gift
        // promise — is already decided). Count$xPaid reads the X paid; a non-xPaid count-SVar
        // (Into the Flood Maw: Count$PromisedGift) is evaluated here. Stamp the results onto
        // target_min/target_max so resolution (is_target_valid) sees the same bounds. Stamped
        // ONCE — a resume re-enters with rt.active set and never re-evaluates.
        int effective_max = ability.target_max;
        if (ability.target_max_from_xpaid)
            effective_max = static_cast<int>(cur_game.x_paid);
        else if (!ability.target_max_count_expr.empty())
            effective_max = static_cast<int>(evaluate_dynamic_amount(
                ability.target_max_count_expr, priority_player, orderer, 0, ability.source));
        int effective_min = ability.target_min;
        if (ability.target_min_from_xpaid)
            effective_min = static_cast<int>(cur_game.x_paid);
        else if (!ability.target_min_count_expr.empty())
            effective_min = static_cast<int>(evaluate_dynamic_amount(
                ability.target_min_count_expr, priority_player, orderer, 0, ability.source));
        ability.target_min = effective_min;
        ability.target_max = effective_max;

        // Zero targets (Into the Flood Maw's unused mode when the gift promise switched the count
        // to 0): the ability targets nothing and does nothing on resolution. Choose no target.
        if (effective_max <= 0) {
            ability.target = 0;
            ability.targets.clear();
            return TargetStatus::DONE;
        }

        rt.active = true;
        rt.effective_min = effective_min;
        rt.effective_max = effective_max;
        rt.picked = 0;
        if (effective_max > 1) ability.targets.clear();
    }

    if (rt.effective_max <= 1) {
        std::vector<Entity> valid_targets = build_valid_targets(ability, orderer, priority_player);
        if (select_single_target(ability, valid_targets, false, asker) < 0)
            return TargetStatus::SUSPENDED;
        rt = TargetSelectRT{};
        return TargetStatus::DONE;
    }

    // Multi-target selection loop
    // "Up to X target ..." (Kozilek's Command): the cap is the X paid at cast time. When the
    // minimum is ALSO X (TargetMin$ X = TargetMax$ X), this becomes "exactly X target ..."
    // (Candelabra of Tawnos, Hide on the Ceiling): the loop neither offers "Done" nor stops
    // before X targets have been chosen, and clamps at X. X (x_paid) was chosen before targets
    // (CR 601.2b), so it is known here. The candidate pool is re-derived each pick as
    // build_valid_targets minus the already-chosen targets — byte-identical to the former
    // erase-chosen running pool, since nothing mutates between picks, and re-derivable on a
    // resume for free.
    for (int i = rt.picked; i < rt.effective_max; i++) {
        std::vector<Entity> valid_targets = build_valid_targets(ability, orderer, priority_player);
        for (Entity chosen : ability.targets)
            valid_targets.erase(
                std::remove(valid_targets.begin(), valid_targets.end(), chosen),
                valid_targets.end());
        if (valid_targets.empty()) break;
        bool can_stop = (i >= rt.effective_min);
        if (select_single_target(ability, valid_targets, can_stop, asker) < 0)
            return TargetStatus::SUSPENDED;
        if (ability.target == 0) break;  // chose "Done" or "No target"
        ability.targets.push_back(ability.target);
        rt.picked = i + 1;
    }
    // Set primary target to first chosen (for backward compat)
    if (!ability.targets.empty()) ability.target = ability.targets[0];
    rt = TargetSelectRT{};
    return TargetStatus::DONE;
}

namespace {
// Blocking asker — exactly the pre-suspension select_target convention: expose
// the asking source as the pending-decision context and read one choice inline
// (the caller has already seated priority at the choosing player; no repoint).
class BlockingTargetAsker final : public TargetAsker {
    public:
        int ask(const std::vector<LegalAction> &menu, Entity decision_source) override {
            PendingDecisionScope pending_scope(decision_source);
            return InputLogger::instance().get_input(menu);
        }
        bool resuming() const override { return false; }
};
}  // namespace

void select_target(Ability &ability, std::shared_ptr<Orderer> orderer, Zone::Ownership priority_player) {
    BlockingTargetAsker asker;
    TargetSelectRT rt;
    if (run_target_select(ability, rt, asker, orderer, priority_player) != TargetStatus::DONE)
        fatal_error("blocking select_target suspended — a blocking asker can never park a query");
}

// Can this charm mode be legally chosen right now (CR 601.2b/c)? A mode is unchoosable only
// when it REQUIRES a target and none exists. The required minimum mirrors select_target's
// bound resolution: an xPaid-driven min reads the X already paid (X is chosen before modes),
// a count-SVar min is evaluated against the current state, else the literal TargetMin$.
static bool charm_mode_choosable(Ability &candidate, std::shared_ptr<Orderer> orderer,
                                 Zone::Ownership caster) {
    if (candidate.valid_tgts == "N_A") return true;
    int min = candidate.target_min;
    if (candidate.target_min_from_xpaid)
        min = static_cast<int>(cur_game.x_paid);
    else if (!candidate.target_min_count_expr.empty())
        min = static_cast<int>(evaluate_dynamic_amount(
            candidate.target_min_count_expr, caster, orderer, 0, candidate.source));
    if (min <= 0) return true;
    return !build_valid_targets(candidate, orderer, caster).empty();
}

// The display label for one charm mode: its script description, else a positional fallback.
static std::string charm_mode_desc(const Ability &ability, size_t idx) {
    return (idx < ability.charm_choice_descriptions.size() &&
            !ability.charm_choice_descriptions[idx].empty())
               ? ability.charm_choice_descriptions[idx]
               : ("Mode " + std::to_string(idx + 1));
}

// One mode pick's menu (CR 601.2b): the not-yet-taken, currently-choosable modes, with
// mode_indices mapping action index -> charm_choices index. Pure ECS reads (choosability is
// re-evaluated per pick), so the suspended CHARM_MODE step re-derives the identical menu on
// resume. Shared by the blocking announce path (effect_choose_card's mini-cast) and the
// run_cast_flow CHARM_MODE step.
static std::vector<LegalAction> build_charm_mode_menu(Ability &ability,
                                                      std::shared_ptr<Orderer> orderer,
                                                      Zone::Ownership caster,
                                                      const std::vector<bool> &taken,
                                                      std::vector<size_t> &mode_indices) {
    std::vector<LegalAction> mode_actions;
    mode_indices.clear();
    for (size_t i = 0; i < ability.charm_choices.size(); i++) {
        if (taken[i]) continue;  // CR 601.2b: a mode can be chosen only once
        Ability &candidate = ability.charm_choices[i];
        candidate.source = ability.source;
        candidate.controller = caster;
        if (!charm_mode_choosable(candidate, orderer, caster)) continue;
        LegalAction la(PASS_PRIORITY, charm_mode_desc(ability, i));
        la.category = ActionCategory::CHOOSE_MODE;
        la.option_ordinal = static_cast<int>(i);  // mode index (into charm_choices)
        mode_actions.push_back(la);
        mode_indices.push_back(i);
    }
    return mode_actions;
}

// Modal spell announcement (CR 601.2b): as the spell is CAST, its controller chooses
// CharmNum$ different modes, then each chosen mode's targets (CR 601.2c) — all before any
// cost is paid and before the opponent gets priority, becoming public information. The picks
// are recorded in charm_chosen; effects::charm resolves exactly those modes, re-verifying
// target legality at resolution (CR 608.2b). BLOCKING form — kept for the cast-from-exile
// mini-cast (effect_choose_card); the CAST_SPELL action runs the same interleave through the
// suspendable CHARM_MODE / CHARM_TARGET steps of run_cast_flow.
static void announce_charm_modes(Ability &ability, std::shared_ptr<Orderer> orderer,
                                 Zone::Ownership caster) {
    PendingDecisionScope pending_scope(ability.source);
    int to_pick = ability.charm_num < 1 ? 1 : ability.charm_num;
    // Track which choice indices remain selectable.
    std::vector<bool> taken(ability.charm_choices.size(), false);

    for (int pick = 0; pick < to_pick; pick++) {
        game_log("Choose mode:\n");
        std::vector<size_t> mode_indices;  // map action index -> charm_choices index
        std::vector<LegalAction> mode_actions =
            build_charm_mode_menu(ability, orderer, caster, taken, mode_indices);
        if (mode_actions.empty()) {
            // No further legal mode (all taken or none with legal targets). The cast-legality
            // gate (spell_has_castable_targets) requires CharmNum$ choosable modes up front,
            // so this is only reachable when an earlier pick's target choice changed the
            // board — proceed with the modes picked so far rather than aborting the cast.
            game_log("No further legal mode — %d chosen\n", pick);
            break;
        }
        int choice = InputLogger::instance().get_input(mode_actions);
        size_t chosen_idx = mode_indices[static_cast<size_t>(choice)];
        taken[chosen_idx] = true;
        ability.charm_chosen.push_back(static_cast<int>(chosen_idx));
        Ability &chosen = ability.charm_choices[chosen_idx];
        game_log("%s chooses mode — %s\n", player_name(caster).c_str(),
                 charm_mode_desc(ability, chosen_idx).c_str());
        if (chosen.valid_tgts != "N_A") {
            select_target(chosen, orderer, caster);
        }
    }
}

// CR 601.2b/c: announce ALL of a spell's cast-time choices on its (already source/controller-
// stamped) primary ability — modal mode(s) first, then the primary target, then each targeting
// chained sub-ability's target. Shared by every path that puts a CAST spell on the stack: the
// CAST_SPELL action and the cast-from-exile mini-cast (effect_choose_card).
void announce_spell_targets(Ability &ability, std::shared_ptr<Orderer> orderer,
                            Zone::Ownership caster) {
    if (!ability.charm_choices.empty()) {
        announce_charm_modes(ability, orderer, caster);
    }

    if (ability.valid_tgts != "N_A") {
        select_target(ability, orderer, caster);
    }
    // A spell whose top-level effect doesn't itself target, but whose chained
    // sub-ability does, chooses that target as it's cast (CR 601.2c). Cabal
    // Therapy: SP$ NameCard (Defined$ You, no target) + DB$ Discard (ValidTgts$
    // Player). Select each targeting sub-ability's target now and store it on the
    // sub-ability template; resolution preserves it (see Ability::resolve).
    for (auto &sub : ability.subabilities) {
        if (sub.valid_tgts != "N_A") {
            sub.source = ability.source;
            sub.controller = caster;
            select_target(sub, orderer, caster);
        }
    }
}

// Ward (CR 702.21): "Whenever this permanent becomes the target of a spell or ability an
// opponent controls, counter that spell or ability unless that player pays {N}." Called right
// after a spell/ability with chosen targets is put on the stack. For each target that is a
// battlefield permanent with a Ward cost, controlled by an opponent of the targeting object's
// controller, push a Ward trigger onto the stack ABOVE the targeting object (so it resolves
// first). The Ward trigger is a Counter ability whose unless_generic_cost is the ward cost —
// reusing the existing "counter unless pay {N}" resolution. A permanent targeted multiple
// times (one spell, several targets) fires Ward once per time it became a target.
// Collect every Ward ability a permanent currently HAS (CR 702.21), honoring ward that is
// granted by a continuous effect (equipment/aura statics, Pump grants, keyword counters), not
// just the printed ward. Two storage forms, kept distinct so they are not double-counted:
//   - Printed ward: parse.cpp stores the numeric cost in CardData::ward_cost (with
//     ward_is_life) and pushes the BARE string "Ward" onto CardData::keywords.
//   - Granted ward: add_keywords_from_spec pushes the raw spec part "Ward:N" or
//     "Ward:PayLife<N>" (with a colon and cost arg) onto the effective keyword list — never
//     the bare "Ward".
// We therefore take the printed instance from ward_cost, and every granted instance from a
// "Ward:..." keyword string, deduping identical granted copies so a single granted Ward:1 fires
// exactly once. Distinct ward costs (e.g. printed Ward 2 plus granted Ward 1) each yield their
// own instance and each trigger, per CR 702.21h.
static std::vector<WardInstance> collect_ward_instances(Entity e) {
    std::vector<WardInstance> wards;
    // Printed ward.
    if (global_coordinator.entity_has_component<CardData>(e)) {
        const auto &cd = global_coordinator.GetComponent<CardData>(e);
        if (cd.ward_cost > 0) wards.push_back({cd.ward_cost, cd.ward_is_life});
    }
    // Granted ward(s) from the effective keyword list. Use the same effective-keyword view as
    // permanent_has_keyword: a creature's rebuilt Creature::keywords, else printed keywords.
    const std::vector<std::string> *kw_list = nullptr;
    if (global_coordinator.entity_has_component<Creature>(e))
        kw_list = &global_coordinator.GetComponent<Creature>(e).keywords;
    else if (global_coordinator.entity_has_component<CardData>(e))
        kw_list = &global_coordinator.GetComponent<CardData>(e).keywords;
    else if (global_coordinator.entity_has_component<Token>(e))
        kw_list = &global_coordinator.GetComponent<Token>(e).keywords;
    if (kw_list) {
        for (const std::string &kw : *kw_list) {
            // Only "Ward:N" (granted form). Bare "Ward" is the printed marker, already counted
            // via ward_cost above; skip it to avoid double-firing the printed ward.
            if (kw.rfind("Ward:", 0) != 0) continue;
            // Same cost grammar as the printed K:Ward parse — "Ward:N" is a generic-mana
            // cost, "Ward:PayLife<N>" a life payment (Hexing Squelcher's grant).
            WardInstance inst{1, false};
            parse_ward_cost(kw.substr(5), inst.cost, inst.is_life);
            // Dedupe identical granted copies (same source granting Ward:1 once must fire once).
            bool dup = false;
            for (const auto &w : wards)
                if (w.cost == inst.cost && w.is_life == inst.is_life) { dup = true; break; }
            if (!dup) wards.push_back(inst);
        }
    }
    return wards;
}

static void trigger_ward_for_targets(Entity targeting_entity, Zone::Ownership controller,
                                     const std::vector<Entity> &targets,
                                     std::shared_ptr<Orderer> orderer) {
    Zone::Ownership opp = (controller == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;
    for (Entity tgt : targets) {
        if (tgt == 0) continue;
        // The Ward permanent must be controlled by an opponent of the targeting player.
        if (!is_battlefield_permanent(tgt, opp)) continue;
        if (!global_coordinator.entity_has_component<CardData>(tgt)) continue;

        std::vector<WardInstance> wards = collect_ward_instances(tgt);
        std::string nm = entity_name(tgt);
        for (const WardInstance &w : wards) {
            if (w.cost <= 0) continue;
            Ability ward;
            ward.ability_type = Ability::TRIGGERED;
            ward.category = "Counter";
            ward.source = tgt;
            ward.controller = opp;            // the Ward permanent's controller
            ward.target = targeting_entity;   // counter the spell/ability that targeted it
            ward.unless_generic_cost = static_cast<size_t>(w.cost);
            ward.unless_cost_is_life = w.is_life;  // Ward—Pay N life pays life, not mana

            orderer->push_ability_onto_stack(ward, opp);
            game_log("Ward %s%d%s: %s's controller may pay to counter the spell or ability "
                     "targeting %s\n", w.is_life ? "—Pay " : "{", w.cost,
                     w.is_life ? " life" : "}", nm.c_str(), nm.c_str());
        }
    }
}

// Mode$ BecomesTarget (CR 603.2c): a permanent's "Whenever ~ becomes the target of a spell ..."
// triggered ability fires when a spell/ability with chosen targets is put on the stack. Called at
// the same point as the Ward hook (right after the targeting object is placed on the stack). For
// each target, emit a BECAME_TARGET event carrying the targeting object (ENTITY), its controller
// (PLAYER), and the targeted permanent (TARGET). The trigger scan (state_manager_triggers) drains
// these on the next SBA pass, matches each permanent's BecomesTarget trigger (ValidTarget$/
// ValidSource$ filters), and places the resulting trigger ABOVE the still-resolving spell so it
// resolves first. General: any becomes-target trigger reuses this; not special-cased to one card.
// A permanent targeted multiple times by one spell fires its trigger once per time it became a
// target (one event per (object, target) pair, matching the Ward "once per target" rule).
static void fire_became_target_events(Entity targeting_entity, Zone::Ownership controller,
                                      const std::vector<Entity> &targets) {
    Entity ctrl_entity = get_player_entity(controller);
    for (Entity tgt : targets) {
        if (tgt == 0) continue;
        // Only battlefield permanents can carry a BecomesTarget triggered ability (TriggerZones$
        // Battlefield). A player or a stack object that was targeted never fires one.
        if (!is_battlefield_permanent(tgt)) continue;
        Event ev(Events::BECAME_TARGET);
        ev.SetParam(Params::ENTITY, targeting_entity);
        ev.SetParam(Params::PLAYER, ctrl_entity);
        ev.SetParam(Params::TARGET, tgt);
        global_coordinator.SendEvent(ev);
    }
}

// Shared post-targeting hook point: once a targeting spell/ability entity is on the stack,
// fire the Ward triggers (CR 702.21) and BECAME_TARGET events (CR 603.2c) for its chosen
// targets. No-op for a non-targeting ability (ValidTgts$ absent → "N_A").
static void fire_targeting_hooks(Entity targeting_entity, Zone::Ownership controller,
                                 const Ability &targeting_ab, std::shared_ptr<Orderer> orderer) {
    if (targeting_ab.valid_tgts == "N_A") return;
    std::vector<Entity> tgts = targeting_ab.targets.empty()
        ? std::vector<Entity>{targeting_ab.target} : targeting_ab.targets;
    trigger_ward_for_targets(targeting_entity, controller, tgts, orderer);
    fire_became_target_events(targeting_entity, controller, tgts);
}

// ── The cast flow as a resumable state machine (Batch 9) ─────────────────────
//
// process_action's CAST_SPELL branch, extracted statement-for-statement into a
// persisted state machine (Game::PendingCast) so its LINEAR prompts — kicker /
// replicate y/n, the X ladder, phyrexian pips, variable-life X, the spell's own
// sacrifice cost, and the gift promise — suspend as loop-top pending decisions
// (pending_query tag CAST) instead of blocking mid-frame. Each converted prompt
// is an arm/apply pair: the arm builds EXACTLY the menu the blocking call asked
// with (pre-prompt narrative included) and returns out of the flow; the loop-top
// emitter reads the answer and resume_cast_flow re-enters with it latched. The
// announce stages (charm modes, select_target, aura) and the deferred payment
// (delve, mana, sac/exile, alt pitch/return) still pass through BLOCKING inside
// their steps, exactly as today — Batches 10-11 convert them.

// Build the exact two-option menu request_optional_yesno presents (Decline
// first, Accept second, OPTIONAL_YESNO ordinals 0/1), so a converted cast-time
// y/n prompt arms the byte-identical menu the blocking helper asked with.
// (request_optional_yesno itself stays blocking — Batch 8 finding g — so the
// cast prompts build their own menus here.)
static std::vector<LegalAction> optional_yesno_menu(const std::string &prompt) {
    std::vector<LegalAction> yn;
    LegalAction decline(PASS_PRIORITY, std::string("Decline: ") + prompt);
    decline.category = ActionCategory::OPTIONAL_YESNO;
    decline.option_ordinal = 0;  // 0 = decline
    yn.push_back(decline);
    LegalAction accept(PASS_PRIORITY, std::string("Accept: ") + prompt);
    accept.category = ActionCategory::OPTIONAL_YESNO;
    accept.option_ordinal = 1;  // 1 = accept
    yn.push_back(accept);
    return yn;
}

// Park a cast-time prompt (tag CAST) for the main loop to emit. Priority
// already sits with the caster at every cast-time prompt (the CAST_SPELL
// action was chosen at the caster's own priority window and nothing repoints
// before these prompts — request_optional_yesno's repoint was a no-op here),
// so like the combat sub-prompts nothing needs saving or restoring on resume.
// Byte-compat per site: the LINEAR cast prompts run with
// pending_decision_source == 0 (no PendingDecisionScope wraps the CAST_SPELL
// branch before the announce stages), so they arm with decision_source = 0
// (the default); the ANNOUNCE-stage prompts (charm modes and every target
// pick) ran under PendingDecisionScope(ability.source) in the blocking flow,
// so they arm with that same per-site source. The replay corpus decodes this
// state field into a "Pending:" transcript line, so the value is load-bearing.
static void arm_flow_query(Game &game, PendingQuery::Tag tag, std::vector<LegalAction> &&menu,
                           Zone::Ownership chooser, Entity decision_source) {
    PendingQuery &pq = game.pending_query;
    pq.tag = tag;
    pq.menu = std::move(menu);
    pq.chooser_is_a = (chooser == Zone::PLAYER_A);
    pq.decision_source = decision_source;
    pq.answered = false;
    pq.answer = -1;
    pq.active = true;
}

static void arm_cast_query(Game &game, std::vector<LegalAction> &&menu, Zone::Ownership chooser,
                           Entity decision_source) {
    arm_flow_query(game, PendingQuery::CAST, std::move(menu), chooser, decision_source);
}

namespace {
// TargetAsker for the cast announce stages (charm-mode / primary / sub-ability
// / aura targets), the replicate copy retargeting at FINISH, and the
// activation-time target selection (tag ACTIVATION): arms the pick as a
// loop-top pending decision with the constructing flow's family tag — directly
// on PendingQuery, since no resolution frame exists at cast/activation time
// (the TriggerPlaceTargetAsker model) — carrying the asking ability's source
// as the pending-decision context (the same PendingDecisionScope(ability.source)
// the blocking select_target held). Priority already sits with the caster/
// activator at every such prompt, so the asker never repoints the seat (the
// TargetAsker contract: the caller seats the chooser). The latched answer
// travels through the flow's resume_choice, consumed by the first ask the
// re-entered step reaches — which is exactly the ask that armed it, since
// arming always returns out of the flow and nothing runs in between.
class FlowTargetAsker final : public TargetAsker {
    public:
        FlowTargetAsker(Game &game, Zone::Ownership chooser, int &resume_choice,
                        PendingQuery::Tag tag = PendingQuery::CAST)
            : game(game), chooser(chooser), resume_choice(resume_choice), tag(tag) {}
        int ask(const std::vector<LegalAction> &menu, Entity decision_source) override {
            if (resume_choice >= 0) {
                int choice = resume_choice;
                resume_choice = -1;
                return choice;
            }
            arm_flow_query(game, tag, std::vector<LegalAction>(menu), chooser, decision_source);
            return -1;
        }
        bool resuming() const override { return resume_choice >= 0; }

    private:
        Game &game;
        Zone::Ownership chooser;
        int &resume_choice;
        PendingQuery::Tag tag;
};
}  // namespace

// Loop-top dispatcher entry (game_driver.cpp) for a parked cast prompt:
// consume the latched answer and re-enter the flow with it. The resume may arm
// the NEXT cast prompt (the caller loops back to the pending branch) or run
// the flow to completion/cancellation.
void resume_cast_flow(Game &game, std::shared_ptr<Orderer> orderer) {
    PendingQuery &pq = game.pending_query;
    if (!game.pending_cast.active || pq.tag != PendingQuery::CAST || !pq.answered)
        fatal_error("resume_cast_flow without a parked cast query");
    int answer = pq.answer;
    pq = PendingQuery{};
    run_cast_flow(game.pending_cast, game, orderer, answer);
}

// Loop-top dispatcher entry (game_driver.cpp) for a parked activation prompt:
// consume the latched answer and re-enter the flow with it. The resume may arm
// the NEXT activation prompt (the caller loops back to the pending branch) or
// run the flow to completion/cancellation.
void resume_activation_flow(Game &game, std::shared_ptr<Orderer> orderer) {
    PendingQuery &pq = game.pending_query;
    if (!game.pending_activation.active || pq.tag != PendingQuery::ACTIVATION || !pq.answered)
        fatal_error("resume_activation_flow without a parked activation query");
    int answer = pq.answer;
    pq = PendingQuery{};
    run_activation_flow(game.pending_activation, game, orderer, answer);
}

// Drive the persisted activation to its next prompt, cancellation, or
// completion. resume_choice < 0 = fresh entry from process_activate_ability;
// >= 0 = the latched answer for the prompt pa.step armed. Every statement keeps
// the blocking branch's exact order; a converted step distinguishes arm from
// apply by whether a latched answer is pending (the arm always returns, so
// re-entry lands on the very step that armed it, and its arm-time gates/menus
// re-derive identically — nothing runs between arm and resume). The interactive
// mana payment is the only prompt left blocking (a deliberate non-conversion —
// machine mode auto-pays with zero decisions), so every payment step is
// synchronous here; its cancel path fully rewinds (mana restore, untap, fail
// count) and clears pa, exactly the blocking early-return.
//
// Byte-compat per site (the corpus decodes pending_decision_source into a
// "Pending:" transcript line, so each prompt arms with TODAY's ambient value):
// the X_LADDER ladder ran under PendingDecisionScope(ability.source) and arms
// with that source; the target selections (ZONE_TARGET / TARGET) pass
// ability.source through the asker exactly as the blocking select_target's
// scope did; the loyalty-X ladder, the equip creature menu, the secondary
// sacrifice/return picks, and the ninjutsu return pick all ran source-less
// (no scope) and arm with 0.
static void run_activation_flow(Game::PendingActivation &pa, Game &game,
                                std::shared_ptr<Orderer> orderer, int resume_choice) {
    Entity permanent_entity = pa.source_entity;
    const Ability &ability = pa.ability;
    Zone::Ownership controller = pa.activator_is_a ? Zone::PLAYER_A : Zone::PLAYER_B;
    // InstantSpeed$ AddMana abilities (e.g. LED) are mana abilities too: they resolve off-stack.
    // The instant-speed timing restriction is enforced upstream (offered only at priority).
    bool is_mana_ability = ability_is_mana(ability);

    for (;;) switch (pa.step) {
        case Game::PendingActivation::NINJA_PAY: {
            // Ninjutsu (CR 702.49e): the source card is in its owner's hand. Pay the ninjutsu
            // mana cost and return one unblocked attacker the activator controls to its owner's
            // hand, then put the source card onto the battlefield tapped and attacking the
            // defender that returned attacker had been attacking. Legality (declare-blockers
            // step + an unblocked attacker exists + mana affordable) is gated in
            // determine_legal_actions. The candidate list is frozen BEFORE the payment (the
            // blocking flow computed it there), the pick suspends after.
            std::vector<Entity> choices = unblocked_attackers(orderer->mEntities, controller);
            if (choices.empty()) {
                game_log("No unblocked attacker to return for ninjutsu.\n");
                pa = Game::PendingActivation{};
                return;
            }

            // Pay the ninjutsu mana cost first (cancellable). The return-an-attacker cost
            // cannot fail once an unblocked attacker exists, so it is paid after the mana
            // commit.
            ManaValue cost = effective_activation_mana_cost(ability, controller, orderer);
            if (!cost.empty()) {
                auto mana_snap = snapshot_mana_state(controller, orderer);
                if (!prompt_mana_payment(controller, cost, permanent_entity, orderer)) {
                    restore_mana_state(controller, mana_snap, orderer);
                    cur_game.payment_fail_counts[permanent_entity]++;
                    game_log("Payment cancelled.\n");
                    pa = Game::PendingActivation{};
                    return;
                }
            }
            pa.frozen_choices = std::move(choices);
            pa.step = Game::PendingActivation::NINJA_RETURN;
            break;
        }

        case Game::PendingActivation::NINJA_RETURN: {
            // Return the chosen unblocked attacker to its owner's hand (the ninjutsu cost).
            // The menu labels are built at arm time (the blocking prompt built them after
            // the payment) from the pre-payment candidate list.
            if (resume_choice < 0) {
                arm_flow_query(game, PendingQuery::ACTIVATION,
                               permanent_choice_menu(pa.frozen_choices, "Return ",
                                                     " to hand (ninjutsu)",
                                                     ActionCategory::RETURN_PERMANENT),
                               controller);
                return;
            }
            Entity returned = pa.frozen_choices[static_cast<size_t>(resume_choice)];
            resume_choice = -1;
            Entity attack_target = global_coordinator.GetComponent<Creature>(returned).attack_target;
            std::string ret_name = entity_name(returned);
            orderer->add_to_zone(false, returned, Zone::HAND);
            game_log("%s returns %s to hand (ninjutsu)\n", player_name(controller).c_str(),
                     ret_name.c_str());

            // Put the ninja onto the battlefield from hand, tapped and attacking the same
            // defender.
            cur_game.pending_enters_tapped.insert(permanent_entity);
            if (attack_target != 0) cur_game.pending_enters_attacking[permanent_entity] = attack_target;
            std::string ninja_name = entity_name(permanent_entity);
            orderer->add_to_zone(false, permanent_entity, Zone::BATTLEFIELD);
            game_log("%s puts %s onto the battlefield tapped and attacking (ninjutsu)\n",
                     player_name(controller).c_str(), ninja_name.c_str());
            game.take_action();
            pa = Game::PendingActivation{};
            return;
        }

        case Game::PendingActivation::ZONE_TARGET: {
            // Select targets before paying costs
            if (pa.stack_ab.valid_tgts != "N_A") {
                FlowTargetAsker asker(game, controller, resume_choice, PendingQuery::ACTIVATION);
                if (run_target_select(pa.stack_ab, pa.tsel, asker, orderer, controller) !=
                    TargetStatus::DONE)
                    return;
            }
            pa.step = Game::PendingActivation::ZONE_PAY;
            break;
        }

        case Game::PendingActivation::ZONE_PAY: {
            // Pay mana cost (after ReduceCost$, e.g. Eiganjo's Channel cheaper per legendary
            // creature)
            ManaValue from_hand_cost = effective_activation_mana_cost(ability, controller, orderer);
            if (!from_hand_cost.empty()) {
                auto mana_snap = snapshot_mana_state(controller, orderer);
                if (!prompt_mana_payment(controller, from_hand_cost, permanent_entity, orderer)) {
                    restore_mana_state(controller, mana_snap, orderer);
                    cur_game.payment_fail_counts[permanent_entity]++;
                    game_log("Payment cancelled.\n");
                    pa = Game::PendingActivation{};
                    return;
                }
            }
            // Pay remaining costs (life, sacrifice, return-to-hand, discard)
            pa.step = Game::PendingActivation::SECONDARY_PRE;
            break;
        }

        case Game::PendingActivation::EQUIP_PAY: {
            auto &permanent = global_coordinator.GetComponent<Permanent>(permanent_entity);
            // Present list of creatures controlled by the equipment owner — frozen BEFORE the
            // cost is paid (the blocking flow built the menu here, then prompted with that
            // exact menu after the payment; a payment made by sacrificing a source for mana
            // must not change the offered menu or its P/T labels).
            std::vector<LegalAction> equip_targets;
            for (auto e : orderer->mEntities) {
                if (e == permanent_entity) continue;  // can't attach to itself (CR 301.5c / reconfigure)
                if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
                if (!global_coordinator.entity_has_component<Creature>(e)) continue;
                auto &ep = global_coordinator.GetComponent<Permanent>(e);
                if (ep.controller != controller) continue;
                std::string ename = ep.name;
                auto &ecr = global_coordinator.GetComponent<Creature>(e);
                LegalAction la(
                    PASS_PRIORITY, e, ename + " [" + std::to_string(ecr.power) + "/" + std::to_string(ecr.toughness) + "]");
                la.category = ActionCategory::SELECT_TARGET;
                equip_targets.push_back(la);
            }
            if (equip_targets.empty()) {
                game_log("No valid creatures to equip.\n");
                pa = Game::PendingActivation{};
                return;
            }
            // Pay equip cost
            if (ability.tap_cost) permanent.is_tapped = true;
            ManaValue equip_cost = effective_activation_mana_cost(ability, controller, orderer);
            if (!equip_cost.empty()) {
                auto mana_snap = snapshot_mana_state(controller, orderer);
                if (!prompt_mana_payment(controller, equip_cost, permanent_entity, orderer)) {
                    restore_mana_state(controller, mana_snap, orderer);
                    if (ability.tap_cost) permanent.is_tapped = false;
                    cur_game.payment_fail_counts[permanent_entity]++;
                    game_log("Payment cancelled.\n");
                    pa = Game::PendingActivation{};
                    return;
                }
            }
            pa.frozen_menu = std::move(equip_targets);
            pa.step = Game::PendingActivation::EQUIP_TARGET;
            break;
        }

        case Game::PendingActivation::EQUIP_TARGET: {
            if (resume_choice < 0) {
                game_log("Choose creature to equip:\n");
                arm_flow_query(game, PendingQuery::ACTIVATION,
                               std::vector<LegalAction>(pa.frozen_menu), controller);
                return;
            }
            Entity target_creature = pa.frozen_menu[static_cast<size_t>(resume_choice)].source_entity;
            resume_choice = -1;
            auto &permanent = global_coordinator.GetComponent<Permanent>(permanent_entity);

            // Detach from previous creature if any
            if (permanent.equipped_to != 0 && global_coordinator.entity_has_component<Permanent>(permanent.equipped_to)) {
                global_coordinator.GetComponent<Permanent>(permanent.equipped_to).equipped_by = 0;
            }
            permanent.equipped_to = target_creature;
            global_coordinator.GetComponent<Permanent>(target_creature).equipped_by = permanent_entity;
            std::string tname = global_coordinator.GetComponent<Permanent>(target_creature).name;
            game_log("%s equipped to %s.\n", permanent.name.c_str(), tname.c_str());
            game.take_action();
            pa = Game::PendingActivation{};
            return;
        }

        case Game::PendingActivation::X_LADDER: {
            // X ACTIVATION COST (Candelabra of Tawnos: Cost$ X T): X is part of the activation
            // cost, chosen during announcement BEFORE targets (CR 602.2b/601.2b) so an
            // exactly-X / up-to-X target count can read it. Prompt for X (bounded by the mana
            // available beyond the rest of the cost), record it as x_paid, and add X generic
            // to the mana cost paid at TAP_PAY. Arms with decision_source = ability.source —
            // the PendingDecisionScope the blocking prompt held, naming the ability making
            // this X choice (it isn't on the stack yet — X is chosen during announcement).
            if (!is_mana_ability && ability.activation_has_x) {
                if (resume_choice >= 0) {
                    pa.x_activation = static_cast<size_t>(resume_choice);
                    resume_choice = -1;
                    cur_game.x_paid = pa.x_activation;
                    game_log("%s chooses X = %zu\n", player_name(controller).c_str(),
                             pa.x_activation);
                } else {
                    ManaValue base_cost = effective_activation_mana_cost(ability, controller, orderer);
                    size_t max_x = max_available_mana(controller, base_cost, orderer);
                    game_log("Choose X value (0-%zu):\n", max_x);
                    std::vector<LegalAction> x_actions;
                    for (size_t xv = 0; xv <= max_x; xv++) {
                        LegalAction la(PASS_PRIORITY, std::string("X = " + std::to_string(xv)));
                        la.category = ActionCategory::CHOOSE_X;
                        la.option_ordinal = static_cast<int>(xv);  // the chosen X value
                        x_actions.push_back(la);
                    }
                    arm_flow_query(game, PendingQuery::ACTIVATION, std::move(x_actions),
                                   controller, ability.source);
                    return;
                }
            }
            pa.step = Game::PendingActivation::LOYALTY_X;
            break;
        }

        case Game::PendingActivation::LOYALTY_X: {
            // X LOYALTY COST (Chandra, Flamecaller's [-X]): choose X at announcement, bounded
            // by the planeswalker's current loyalty for a minus cost (you can't remove more
            // than it has, 606.5). Recorded as x_paid — set at APPLY, before the SECONDARY
            // steps run — so the loyalty cost (SECONDARY_PRE) and the effect's Count$xPaid
            // (NumDmg$ X) both read it. This is a loyalty cost, not mana — it never touches
            // the TAP_PAY cost. Ran source-less in the blocking flow (no scope), so it arms
            // with decision_source = 0.
            if (ability.loyalty_cost_is_x) {
                if (resume_choice >= 0) {
                    int x_choice = resume_choice;
                    resume_choice = -1;
                    cur_game.x_paid = static_cast<size_t>(x_choice);
                    game_log("%s chooses X = %d\n", player_name(controller).c_str(), x_choice);
                } else {
                    int max_x = (ability.loyalty_cost < 0) ? get_counters(permanent_entity, "LOYALTY") : 99;
                    if (max_x < 0) max_x = 0;
                    game_log("Choose X value (0-%d):\n", max_x);
                    std::vector<LegalAction> x_actions;
                    for (int xv = 0; xv <= max_x; xv++) {
                        LegalAction la(PASS_PRIORITY, std::string("X = " + std::to_string(xv)));
                        la.category = ActionCategory::CHOOSE_X;
                        la.option_ordinal = static_cast<int>(xv);  // the chosen X value
                        x_actions.push_back(la);
                    }
                    arm_flow_query(game, PendingQuery::ACTIVATION, std::move(x_actions),
                                   controller);
                    return;
                }
            }
            pa.step = Game::PendingActivation::TARGET;
            break;
        }

        case Game::PendingActivation::TARGET: {
            // SELECT TARGETS BEFORE PAYING COSTS
            if (!is_mana_ability && pa.stack_ab.valid_tgts != "N_A") {
                FlowTargetAsker asker(game, controller, resume_choice, PendingQuery::ACTIVATION);
                if (run_target_select(pa.stack_ab, pa.tsel, asker, orderer, controller) !=
                    TargetStatus::DONE)
                    return;
            }
            pa.step = Game::PendingActivation::TAP_PAY;
            break;
        }

        case Game::PendingActivation::TAP_PAY: {
            auto &permanent = global_coordinator.GetComponent<Permanent>(permanent_entity);
            // Tap cost
            if (ability.tap_cost) {
                permanent.is_tapped = true;
            }
            // Mana cost (after ReduceCost$ — CR 601.2f; reduces generic only). For an X-cost
            // ability the chosen X is added as generic mana on top of the base cost.
            ManaValue activate_cost = effective_activation_mana_cost(ability, controller, orderer);
            for (size_t i = 0; i < pa.x_activation; i++) activate_cost.insert(GENERIC);
            if (!activate_cost.empty()) {
                auto mana_snap = snapshot_mana_state(controller, orderer);
                if (!prompt_mana_payment(controller, activate_cost, permanent_entity, orderer)) {
                    restore_mana_state(controller, mana_snap, orderer);
                    if (ability.tap_cost) permanent.is_tapped = false;
                    cur_game.payment_fail_counts[permanent_entity]++;
                    game_log("Payment cancelled.\n");
                    pa = Game::PendingActivation{};
                    return;
                }
            }
            // Pay remaining costs (life, sacrifice, return-to-hand, discard).
            // life_cost legality is gated upstream in can_afford_alt / determine_legal_actions.
            pa.step = Game::PendingActivation::SECONDARY_PRE;
            break;
        }

        case Game::PendingActivation::SECONDARY_PRE: {
            // The non-mana, non-tap activation costs shared by hand- and battlefield-activated
            // abilities (the head of the old pay_secondary_activation_costs — everything
            // before its first pick). These costs cannot fail once legality has passed.
            // Loyalty cost (606.4/606.5): pay by adding/removing loyalty counters on the source
            // planeswalker, and mark the per-permanent once-per-turn gate (606.3). The ability
            // still goes on the stack and resolves later; the loyalty change is the cost, paid
            // now.
            if (ability.is_loyalty_ability &&
                global_coordinator.entity_has_component<Permanent>(permanent_entity)) {
                auto &perm = global_coordinator.GetComponent<Permanent>(permanent_entity);
                // An X loyalty cost (Chandra, Flamecaller's [-X]) removes/adds the X chosen at
                // activation (cur_game.x_paid); loyalty_cost carries only the sign. A fixed
                // cost uses loyalty_cost as-is.
                int loyalty_delta = ability.loyalty_cost_is_x
                    ? (ability.loyalty_cost < 0 ? -static_cast<int>(cur_game.x_paid)
                                                :  static_cast<int>(cur_game.x_paid))
                    : ability.loyalty_cost;
                int loyalty = add_counters(permanent_entity, "LOYALTY", loyalty_delta);
                perm.loyalty_ability_activated_this_turn = true;
                game_log("%s activates a loyalty ability (%+d, loyalty now %d)\n",
                         entity_name(permanent_entity).c_str(), loyalty_delta, loyalty);
            }
            // Life cost
            if (ability.life_cost > 0) {
                auto &activating_player =
                    global_coordinator.GetComponent<Player>(get_player_entity(controller));
                activating_player.life_total -= ability.life_cost;
                activating_player.life_lost_this_turn += ability.life_cost;  // CR 119.4: paying life is losing life
                game_log("%s pays %d life\n", player_name(controller).c_str(), ability.life_cost);
            }
            // Energy cost (PayEnergy<N>, CR 122.1c): affordability is gated in
            // determine_legal_actions, so the {E} is available to spend here.
            if (ability.energy_cost > 0) {
                auto &activating_player =
                    global_coordinator.GetComponent<Player>(get_player_entity(controller));
                pay_energy(activating_player, ability.energy_cost);
                game_log("%s pays %d energy\n", player_name(controller).c_str(), ability.energy_cost);
            }
            // Sacrifice self: move to graveyard; apply_permanent_components SBA removes
            // Permanent next pass
            if (ability.sac_self) {
                std::string sname = entity_name(permanent_entity);
                orderer->add_to_zone(false, permanent_entity, Zone::GRAVEYARD);
                game_log("%s sacrifices %s\n", player_name(controller).c_str(), sname.c_str());
            }
            pa.step = Game::PendingActivation::SECONDARY_SAC;
            break;
        }

        case Game::PendingActivation::SECONDARY_SAC: {
            // Type-based sacrifice cost (Cycling "Sac a land", Knight of the Reliquary). The
            // exact pool/menu the old blocking prompt_permanent_choice built, armed as a
            // pending decision instead; the pool re-derives identically at apply — nothing
            // runs between arm and resume.
            if (!ability.sac_cost_spec.empty()) {
                std::vector<Entity> choices = controlled_permanents_matching(
                    controller, ability.sac_cost_spec, orderer->mEntities, permanent_entity);
                if (!choices.empty()) {
                    if (resume_choice >= 0) {
                        Entity to_sac = choices[static_cast<size_t>(resume_choice)];
                        resume_choice = -1;
                        std::string sac_name = global_coordinator.GetComponent<Permanent>(to_sac).name;
                        orderer->add_to_zone(false, to_sac, Zone::GRAVEYARD);
                        game_log("%s sacrifices %s\n", player_name(controller).c_str(),
                                 sac_name.c_str());
                    } else {
                        arm_flow_query(game, PendingQuery::ACTIVATION,
                                       permanent_choice_menu(choices, "Sacrifice ", "",
                                                             ActionCategory::SACRIFICE_PERMANENT),
                                       controller);
                        return;
                    }
                }
            }
            pa.step = Game::PendingActivation::SECONDARY_RETURN;
            break;
        }

        case Game::PendingActivation::SECONDARY_RETURN: {
            // Return-to-hand cost (Scryb Ranger: return a Forest to hand) — same arm/apply
            // convention as the sacrifice pick above.
            if (!ability.return_cost_type.empty()) {
                std::vector<Entity> choices = controlled_permanents_matching(
                    controller, ability.return_cost_type, orderer->mEntities);
                if (!choices.empty()) {
                    if (resume_choice >= 0) {
                        Entity to_ret = choices[static_cast<size_t>(resume_choice)];
                        resume_choice = -1;
                        std::string ret_name = global_coordinator.GetComponent<Permanent>(to_ret).name;
                        orderer->add_to_zone(false, to_ret, Zone::HAND);
                        game_log("%s returns %s to hand\n", player_name(controller).c_str(),
                                 ret_name.c_str());
                    } else {
                        arm_flow_query(game, PendingQuery::ACTIVATION,
                                       permanent_choice_menu(choices, "Return ", " to hand",
                                                             ActionCategory::RETURN_PERMANENT),
                                       controller);
                        return;
                    }
                }
            }
            pa.step = Game::PendingActivation::SECONDARY_POST;
            break;
        }

        case Game::PendingActivation::SECONDARY_POST: {
            // Discard self from hand cost (Faerie Macabre)
            if (ability.discard_self_cost) {
                std::string cname = global_coordinator.entity_has_component<CardData>(permanent_entity)
                    ? global_coordinator.GetComponent<CardData>(permanent_entity).name : "card";
                orderer->add_to_zone(false, permanent_entity, Zone::GRAVEYARD);
                game_log("%s discards %s\n", player_name(controller).c_str(), cname.c_str());
            }
            // Discard hand cost (Lion's Eye Diamond)
            if (ability.discard_hand_cost) {
                for (auto card : orderer->get_hand(controller)) {
                    std::string cname = global_coordinator.entity_has_component<CardData>(card)
                        ? global_coordinator.GetComponent<CardData>(card).name : "card";
                    orderer->add_to_zone(false, card, Zone::GRAVEYARD);
                    game_log("%s discards %s\n", player_name(controller).c_str(), cname.c_str());
                }
            }
            pa.step = Game::PendingActivation::FINISH;
            break;
        }

        case Game::PendingActivation::FINISH: {
            if (pa.zone_path) {
                // Auto-consume the activated card to the graveyard, unless the ability
                // relocates it itself (Defined$ Self) or it was already discarded as an
                // explicit cost (discard_self_cost, paid above) — guarding against a double
                // move.
                if (!ability.defined_self && !ability.discard_self_cost) {
                    orderer->add_to_zone(false, permanent_entity, Zone::GRAVEYARD);
                }

                // Create standalone ability entity on the stack
                pa.stack_ab.source = permanent_entity;
                pa.stack_ab.controller = controller;
                Entity ability_stack_entity = orderer->push_ability_onto_stack(pa.stack_ab, controller);

                // Ward (702.21) + BecomesTarget (CR 603.2c) apply to hand/graveyard-activated
                // abilities too — CR 702.21b triggers on ANY spell or ability an opponent
                // controls that targets the warded permanent. (Graveyard/non-battlefield
                // targets are filtered inside the hooks.)
                fire_targeting_hooks(ability_stack_entity, controller, pa.stack_ab, orderer);

                auto &cd = global_coordinator.GetComponent<CardData>(permanent_entity);
                const char *from_zone = (ability.activation_zone == Zone::GRAVEYARD) ? "graveyard" : "hand";
                if (pa.stack_ab.target != 0) {
                    std::string tgt_names = chosen_targets_display(pa.stack_ab);
                    game_log("%s activates %s from %s targeting %s\n",
                        player_name(controller).c_str(), cd.name.c_str(), from_zone, tgt_names.c_str());
                } else {
                    game_log("%s activates %s from %s\n", player_name(controller).c_str(),
                             cd.name.c_str(), from_zone);
                }
                game.take_action();
                pa = Game::PendingActivation{};
                return;
            }
            // MANA ABILITY
            if (is_mana_ability) {
                // Costs were already paid above: tap at the generic tap-cost step, activation
                // mana via prompt_mana_payment, life/sacrifice/discard via the SECONDARY
                // steps. The shared production core handles the rest: amount eval (dynamic
                // amounts like Gaea's Cradle / Urza's Workshop), ProduceMana replacement, pool
                // insert, TapsForMana triggers, uncounterability flag, narrative, SubAbility$
                // riders (Ancient Tomb's damage), and the activation counter.
                auto &pl = global_coordinator.GetComponent<Player>(get_player_entity(controller));
                produce_mana_from_ability(permanent_entity, ability, controller, orderer, pl.mana,
                                          /*commit=*/true, ManaLogStyle::TAPPED_AMOUNT);
                // priority does not pass
                pa = Game::PendingActivation{};
                return;
            }
            // ACTIVATED ABILITY THAT IS NOT A MANA ABILITY - GOES ON STACK
            // puts on stack; we have targets from earlier
            auto &permanent = global_coordinator.GetComponent<Permanent>(permanent_entity);
            pa.stack_ab.source = permanent_entity;
            pa.stack_ab.controller = controller;
            Entity ability_stack_entity = orderer->push_ability_onto_stack(pa.stack_ab, controller);

            // Ward (702.21) + Mode$ BecomesTarget (CR 603.2c): abilities fire these too; the
            // per-trigger ValidSource$ filter (e.g. Reality Smasher's Spell.OppCtrl) gates out
            // ability sources for BecomesTarget.
            fire_targeting_hooks(ability_stack_entity, controller, pa.stack_ab, orderer);

            if (pa.stack_ab.target != 0) {
                std::string tgt_names = chosen_targets_display(pa.stack_ab);
                game_log("%s's %s ability targeting %s is on the stack\n",
                    player_name(controller).c_str(), permanent.name.c_str(), tgt_names.c_str());
            } else {
                game_log("%s's %s ability is on the stack\n", player_name(controller).c_str(),
                         permanent.name.c_str());
            }
            game.take_action();

            // Increment activation counter for limited abilities (e.g. Scryb Ranger)
            increment_activation_count(permanent, ability);
            // if target remains legal checked at resolution
            pa = Game::PendingActivation{};
            return;
        }

        default:
            fatal_error("run_activation_flow reached an unimplemented step " +
                        std::to_string(static_cast<int>(pa.step)));
    }
}

// Drive the persisted cast to its next prompt, cancellation, or completion.
// resume_choice < 0 = fresh entry from process_action; >= 0 = the latched
// answer for the prompt pc.step armed. Every statement keeps the blocking
// branch's exact order; a converted step distinguishes arm from apply by
// whether a latched answer is pending (the arm always returns, so re-entry
// with an answer lands on the very step/pip that armed it, and its
// arm-time gates/menus re-derive identically — nothing runs between arm and
// resume). Machine-mode auto-resolve paths (the hybrid branch, the deferred
// mana auto-payment) keep their inline calls; the interactive mana payment and
// interactive hybrid pips are the only prompts left blocking (deliberate
// non-conversions — machine mode resolves both with zero decisions).
static void run_cast_flow(Game::PendingCast &pc, Game &game, std::shared_ptr<Orderer> orderer,
                          int resume_choice) {
    Entity spell_entity = pc.spell_entity;
    auto &front_data = global_coordinator.GetComponent<CardData>(spell_entity);
    // Modal DFC cast as its NONLAND back face (CR 712.8): the spell has only the
    // BACK face's characteristics — same view the setup in process_action used.
    const CardData &card_data = (pc.cast_back_face && front_data.backside)
                                    ? *front_data.backside : front_data;
    Zone::Ownership caster = pc.caster_is_a ? Zone::PLAYER_A : Zone::PLAYER_B;

    for (;;) switch (pc.step) {
        case Game::PendingCast::COST: {
            // FLASHBACK COST — determined here (601.2f), but PAID after targets are
            // chosen (601.2c before 601.2g/h; see the deferred_* fields). Paying
            // the sacrifice first leaked information and changed the board before the
            // target was locked in (Cabal Therapy: Flashback—Sacrifice a creature).
            if (pc.use_flashback) {
                // Flashback mana cost — flashback is an alternative cost (CR 702.34a),
                // so an active SetCost floor (Trinisphere) pads it up to the floor (601.2f).
                pc.deferred_mana_cost =
                    floored_alt_mana_cost(card_data, card_data.flashback_mana_cost, caster);
                pc.deferred_mana_pending = true;
                pc.deferred_life_cost = card_data.flashback_alt_cost.life_cost;
                // Flashback sacrifice cost: the cast is only offered when a matching
                // permanent exists (cast legality), so there is always something to
                // sacrifice when the deferred payment runs.
                pc.deferred_sac_spec = card_data.flashback_alt_cost.sac_cost_spec;
                pc.step = Game::PendingCast::GIFT;

            // ESCAPE COST (CR 702.139): cast from the graveyard for the escape cost — the
            // escape mana cost plus the ExileFromGrave additional cost (exile other graveyard
            // cards covering >=N card types). The exile is a cost, paid as the spell is cast —
            // after targets are chosen, like every other cost (601.2c before 601.2g/h).
            } else if (pc.use_escape) {
                // Escape is an alternative cost (CR 702.139a): fold in any active SetCost floor.
                pc.deferred_mana_cost =
                    floored_alt_mana_cost(card_data, card_data.escape_mana_cost, caster);
                pc.deferred_mana_pending = true;
                pc.deferred_life_cost = card_data.escape_alt_cost.life_cost;
                pc.deferred_exile_min_types = card_data.escape_alt_cost.exile_grave_min_types;
                pc.deferred_exile_count = card_data.escape_alt_cost.exile_grave_count;
                pc.step = Game::PendingCast::GIFT;

            // IMPULSE CAST (Amped Raptor's DB$ Play): cast from exile under a one-shot
            // permission, paying its alternative RESOURCE cost (energy or life) instead of any
            // mana (CR 707 / 118.9). The permission carries the resolved amount. Consumed here
            // so it can't be reused. X spells cast this way count X = 0 (no X prompt).
            } else if (pc.impulse_cast) {
                Entity caster_entity = (caster == Zone::PLAYER_A)
                    ? cur_game.player_a_entity : cur_game.player_b_entity;
                auto &player = global_coordinator.GetComponent<Player>(caster_entity);
                auto it = cur_game.impulse_cast_permission.find(spell_entity);
                bool normal_play = false;
                if (it != cur_game.impulse_cast_permission.end()) {
                    const auto &grant = it->second;
                    if (grant.resource == Game::ImpulseCastPermission::FREE) {
                        // Ugin -11: cast without paying its mana cost (CR 118.9). No cost paid.
                        game_log("%s casts %s without paying its mana cost\n",
                                 player_name(caster).c_str(), card_data.name.c_str());
                    } else if (grant.resource == Game::ImpulseCastPermission::NORMAL) {
                        // Light Up the Stage: PLAY the exiled card for its NORMAL cost. Defer the
                        // full base cost (targets are chosen first, like every other cost); no
                        // alternative resource is paid.
                        normal_play = true;
                    } else if (grant.resource == Game::ImpulseCastPermission::ENERGY) {
                        pay_energy(player, grant.amount);
                        game_log("%s pays %d energy\n", player_name(caster).c_str(), grant.amount);
                    } else {
                        player.life_total -= grant.amount;
                        player.life_lost_this_turn += grant.amount;  // CR 119.4: paying life is losing life
                        game_log("%s pays %d life\n", player_name(caster).c_str(), grant.amount);
                    }
                    // ForgetOnMoved$ Exile / one-shot: the card leaves exile as it's cast, so the
                    // permission is consumed and can't be reused.
                    cur_game.impulse_cast_permission.erase(it);
                }

                if (card_data.has_x_cost) cur_game.x_paid = 0;

                if (normal_play) {
                    // Light Up the Stage: pay the normal mana cost (cost-increase-adjusted).
                    // X spells played this way resolve with X = 0 (no X prompt on this path).
                    pc.deferred_mana_cost = effective_base_cost(card_data, caster);
                    pc.deferred_mana_pending = true;
                } else {
                    // Cost-increase / SetCost-floor statics apply to alternative costs too
                    // (CR 118.9d / 601.2f): the impulse/free cast substitutes a {0} mana cost, but
                    // an active Trinisphere floor pads it up to its minimum ({3}) and Thalia adds
                    // its surcharge — paid ON TOP of the resource cost (energy/life) that was just
                    // paid. Deferred until after targets like every other cost. Empty (no floor /
                    // increase applies) leaves the cast free of mana, exactly as before.
                    ManaValue floor_mana = floored_alt_mana_cost(card_data, ManaValue{}, caster);
                    if (!floor_mana.empty()) {
                        pc.deferred_mana_cost = floor_mana;
                        pc.deferred_mana_pending = true;
                    }
                }
                pc.step = Game::PendingCast::GIFT;

            // ALTERNATE COST — determined here, paid in the payment phase like every
            // other cost branch (its pitch/return/sacrifice picks are the ALT_* steps).
            } else if (pc.use_alt_cost) {
                defer_alternate_cost(game, card_data, caster);
                pc.step = Game::PendingCast::GIFT;

            } else {  // REGULAR COST + DELVE
                // RaiseCost surcharge (NamedCard-aware) folded in; shared with legality.
                // caster passed so Affinity for artifacts reduces the generic cost (702.41).
                pc.cost_to_pay = effective_base_cost(card_data, caster);

                // Offspring (CR 702.171): additional cost paid on top of the spell's cost.
                if (pc.use_offspring)
                    for (Colors c : card_data.offspring_cost) pc.cost_to_pay.insert(c);

                if (!card_data.kicker_costs.empty())
                    pc.kicked_flags.assign(card_data.kicker_costs.size(), false);
                pc.step = Game::PendingCast::KICKER;
            }
            break;
        }

        case Game::PendingCast::ALT_PITCH: {
            // Alt-cost pitch (Force of Will: "exile a blue card from your hand"): one pick
            // per required card, the menu re-derived from the live hand each pass, minus
            // whatever earlier picks already claimed. pc.alt_pitch_done counts completed
            // picks. The three ALT_* steps belong to the ALTERNATIVE cost (CR 601.2b) and
            // are skipped entirely on a normal cast of the same card — Daze cast for {1}{U}
            // does not also return an Island.
            if (!pc.use_alt_cost) {
                pc.step = Game::PendingCast::SPELL_SAC;
                break;
            }
            Colors pitch_color = card_data.alt_cost.exile_from_hand_color;
            while (pc.alt_pitch_done < card_data.alt_cost.exile_from_hand_count) {
                std::vector<LegalAction> exile_actions;
                for (auto e : orderer->get_hand(caster)) {
                    if (e == spell_entity) continue;
                    if (already_chosen_as_cost(pc, e)) continue;
                    if (!global_coordinator.entity_has_component<ColorIdentity>(e)) continue;
                    if (pitch_color != NO_COLOR &&
                        !global_coordinator.GetComponent<ColorIdentity>(e).colors.count(pitch_color))
                        continue;
                    LegalAction la(PASS_PRIORITY, e,
                                   "Exile " + global_coordinator.GetComponent<CardData>(e).name);
                    la.category = ActionCategory::PAYING_COSTS;
                    exile_actions.push_back(la);
                }
                if (resume_choice >= 0) {
                    Entity exiled = exile_actions[static_cast<size_t>(resume_choice)].source_entity;
                    resume_choice = -1;
                    choose_cost_item(pc, exiled, Zone::EXILE,
                                     player_name(caster) + " exiles " +
                                         global_coordinator.GetComponent<CardData>(exiled).name);
                    pc.alt_pitch_done++;
                    continue;
                }
                arm_cast_query(game, std::move(exile_actions), caster);
                return;
            }
            pc.step = Game::PendingCast::ALT_RETURN;
            break;
        }

        case Game::PendingCast::ALT_RETURN: {
            // Alt-cost return-to-hand (Daze: "return an Island you control to its owner's
            // hand"): one pick per required permanent, menu re-derived from the live
            // battlefield each pass. pc.alt_return_done counts completed picks.
            while (pc.alt_return_done < card_data.alt_cost.return_to_hand_count) {
                std::vector<LegalAction> rth_actions;
                const std::string &type = card_data.alt_cost.return_to_hand_type;
                for (auto e : orderer->mEntities) {
                    if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
                    if (already_chosen_as_cost(pc, e)) continue;
                    auto &eperm = global_coordinator.GetComponent<Permanent>(e);
                    if (eperm.controller != caster) continue;
                    bool matches = false;
                    // can be subtype, type or supertype
                    for (auto &t : eperm.types) {
                        if (t.name == type) {
                            matches = true;
                            break;
                        }
                    }
                    if (!matches) continue;
                    LegalAction la(PASS_PRIORITY, e, "Return " + eperm.name);
                    la.category = ActionCategory::RETURN_PERMANENT;
                    rth_actions.push_back(la);
                }
                if (resume_choice >= 0) {
                    Entity returned = rth_actions[static_cast<size_t>(resume_choice)].source_entity;
                    resume_choice = -1;
                    choose_cost_item(pc, returned, Zone::HAND,
                                     player_name(caster) + " returns " +
                                         global_coordinator.GetComponent<Permanent>(returned).name +
                                         " to hand");
                    pc.alt_return_done++;
                    continue;
                }
                arm_cast_query(game, std::move(rth_actions), caster);
                return;
            }
            pc.step = Game::PendingCast::ALT_SAC;
            break;
        }

        case Game::PendingCast::ALT_SAC: {
            // Alt-cost sacrifice (Fireblast: "sacrifice two Mountains rather than pay this
            // spell's mana cost", CR 118.9): one pick per required permanent, the menu of
            // matching permanents the caster controls re-derived each pass (an earlier pick has
            // left the battlefield, so it drops from the next menu). pc.alt_sac_done counts
            // completed sacrifices. Cast legality (can_afford_alt) already guaranteed enough
            // matching permanents exist.
            while (pc.alt_sac_done < card_data.alt_cost.sac_cost_count) {
                std::vector<Entity> choices = controlled_permanents_matching(
                    caster, card_data.alt_cost.sac_cost_spec, orderer->mEntities, spell_entity);
                std::vector<LegalAction> sac_actions;
                for (auto e : choices) {
                    std::string nm = global_coordinator.GetComponent<Permanent>(e).name;
                    LegalAction la(PASS_PRIORITY, e, std::string("Sacrifice ") + nm);
                    la.category = ActionCategory::SACRIFICE_PERMANENT;
                    sac_actions.push_back(la);
                }
                drop_chosen_cost_items(pc, sac_actions);
                if (sac_actions.empty()) break;  // defensive: legality guaranteed enough permanents
                if (resume_choice >= 0) {
                    Entity to_sac = sac_actions[static_cast<size_t>(resume_choice)].source_entity;
                    resume_choice = -1;
                    choose_cost_item(pc, to_sac, Zone::GRAVEYARD,
                                     player_name(caster) + " sacrifices " +
                                         global_coordinator.GetComponent<Permanent>(to_sac).name);
                    pc.alt_sac_done++;
                    continue;
                }
                arm_cast_query(game, std::move(sac_actions), caster);
                return;
            }
            pc.step = Game::PendingCast::SPELL_SAC;
            break;
        }

        case Game::PendingCast::KICKER: {
            // KICKER (CR 702.33 / 601.2b): each kicker is an OPTIONAL ADDITIONAL cost
            // declared as the spell is cast. Offer one yes/no per kicker (only when its
            // extra mana is still affordable on top of everything chosen so far); an
            // accepted kicker's mana is folded into cost_to_pay and recorded in
            // kicked_flags so the spell becomes "kicked with its Nth kicker". General over
            // any number of independent kicker costs (multikicker-ready data model).
            while (pc.kicker_idx < card_data.kicker_costs.size()) {
                size_t ki = pc.kicker_idx;
                ManaValue with_kicker = pc.cost_to_pay;
                for (Colors c : card_data.kicker_costs[ki]) with_kicker.insert(c);
                if (!resolve_hybrid_cost(caster, with_kicker, card_data.hybrid_mana,
                                         spell_entity, orderer, card_data.has_delve,
                                         card_data.has_improvise)) {
                    pc.kicker_idx++;
                    continue;
                }
                if (resume_choice >= 0) {
                    // Apply the latched y/n for THIS pip (the arm below returned on it).
                    if (resume_choice == 1) {
                        pc.cost_to_pay = with_kicker;
                        pc.kicked_flags[ki] = true;
                        game_log("%s pays the kicker %zu cost for %s\n",
                                 player_name(caster).c_str(), ki + 1, card_data.name.c_str());
                    }
                    resume_choice = -1;
                    pc.kicker_idx++;
                    continue;
                }
                std::string prompt = "pay kicker " + std::to_string(ki + 1) +
                    " for " + card_data.name;
                arm_cast_query(game, optional_yesno_menu(prompt), caster);
                return;
            }
            pc.step = Game::PendingCast::REPLICATE;
            break;
        }

        case Game::PendingCast::REPLICATE: {
            // REPLICATE (CR 702.x / 601.2b): an OPTIONAL ADDITIONAL cost that may be paid
            // ANY NUMBER OF TIMES as the spell is cast. Offer a repeated yes/no — each "yes"
            // folds another replicate cost's mana into cost_to_pay and bumps the replicate
            // count — stopping once the next payment is unaffordable or declined. The count
            // drives the on-cast copy effect (Spell::replicate_count).
            if (card_data.has_replicate) {
                while (true) {
                    ManaValue with_replicate = pc.cost_to_pay;
                    for (Colors c : card_data.replicate_cost) with_replicate.insert(c);
                    if (!resolve_hybrid_cost(caster, with_replicate, card_data.hybrid_mana,
                                             spell_entity, orderer, card_data.has_delve,
                                             card_data.has_improvise))
                        break;
                    if (resume_choice >= 0) {
                        bool accepted = (resume_choice == 1);
                        resume_choice = -1;
                        if (!accepted) break;
                        pc.cost_to_pay = with_replicate;
                        pc.replicate_count++;
                        game_log("%s pays the replicate cost for %s (%d)\n",
                                 player_name(caster).c_str(), card_data.name.c_str(),
                                 pc.replicate_count);
                        continue;
                    }
                    std::string prompt = "pay replicate cost for " + card_data.name +
                        " (paid " + std::to_string(pc.replicate_count) + ")";
                    arm_cast_query(game, optional_yesno_menu(prompt), caster);
                    return;
                }
            }
            pc.step = Game::PendingCast::CHOOSE_X;
            break;
        }

        case Game::PendingCast::CHOOSE_X: {
            // X-COST: prompt player to choose X value, add X generic to cost
            if (card_data.has_x_cost) {
                if (resume_choice >= 0) {
                    size_t x_val = static_cast<size_t>(resume_choice);
                    resume_choice = -1;
                    cur_game.x_paid = x_val;
                    for (size_t i = 0; i < x_val; i++) pc.cost_to_pay.insert(GENERIC);
                    game_log("%s chooses X = %zu\n", player_name(caster).c_str(), x_val);
                } else {
                    size_t max_x = max_available_mana(caster, pc.cost_to_pay, orderer);
                    // For a spell whose required target count IS X (Hide on the Ceiling), X can't
                    // exceed the number of legal targets (CR 601.2c) — clamp so the agent can't
                    // pick an X that leaves a mandatory target choice with too few candidates.
                    max_x = std::min(max_x,
                                     spell_xpaid_target_cap(card_data, spell_entity, caster, orderer));

                    game_log("Choose X value (0-%zu):\n", max_x);
                    std::vector<LegalAction> x_actions;
                    for (size_t xv = 0; xv <= max_x; xv++) {
                        LegalAction la(PASS_PRIORITY, std::string("X = " + std::to_string(xv)));
                        la.category = ActionCategory::CHOOSE_X;
                        la.option_ordinal = static_cast<int>(xv);  // the chosen X value
                        x_actions.push_back(la);
                    }
                    arm_cast_query(game, std::move(x_actions), caster);
                    return;
                }
            }
            pc.step = Game::PendingCast::HYBRID;
            break;
        }

        case Game::PendingCast::HYBRID: {
            // HYBRID mana (CR 107.4): resolve each {W/U} / {2/W} pip to one concrete payment
            // before the colored/generic payment runs below. Machine/auto mode picks the first
            // payable assignment (shared with the cast-legality gate via resolve_hybrid_cost);
            // interactive mode prompts the player per pip, like the Phyrexian block below.
            // (Machine mode auto-resolves with ZERO decisions; the interactive per-pip menu is
            // a deliberate blocking non-conversion — snapshots can't be requested in CLI play.)
            if (!card_data.hybrid_mana.empty()) {
                if (InputLogger::instance().is_machine_schedule()) {
                    ManaValue resolved;
                    if (resolve_hybrid_cost(caster, pc.cost_to_pay, card_data.hybrid_mana,
                                            spell_entity, orderer, card_data.has_delve,
                                            card_data.has_improvise, &resolved)) {
                        pc.cost_to_pay = resolved;
                    } else {
                        // No payable assignment: add the colored option (else the generic
                        // alternative) so payment fails cleanly through the normal path.
                        for (const auto &pip : card_data.hybrid_mana) {
                            if (!pip.colors.empty()) pc.cost_to_pay.insert(pip.colors.front());
                            else for (int i = 0; i < pip.generic_alt; i++)
                                pc.cost_to_pay.insert(GENERIC);
                        }
                    }
                } else {
                    for (const auto &pip : card_data.hybrid_mana) {
                        std::vector<LegalAction> hybrid_actions;
                        for (Colors c : pip.colors) {
                            LegalAction a(PASS_PRIORITY,
                                std::string("Pay {") + mana_symbol_str(c) + "}");
                            a.category = ActionCategory::PAYING_COSTS;
                            hybrid_actions.push_back(a);
                        }
                        if (pip.generic_alt > 0) {
                            LegalAction a(PASS_PRIORITY,
                                "Pay {" + std::to_string(pip.generic_alt) + "} generic");
                            a.category = ActionCategory::PAYING_COSTS;
                            hybrid_actions.push_back(a);
                        }
                        int hc = InputLogger::instance().get_input(hybrid_actions);
                        size_t uc = static_cast<size_t>(hc);
                        if (uc < pip.colors.size()) {
                            pc.cost_to_pay.insert(pip.colors[uc]);
                        } else {
                            for (int i = 0; i < pip.generic_alt; i++)
                                pc.cost_to_pay.insert(GENERIC);
                        }
                    }
                }
            }
            pc.step = Game::PendingCast::PHYREXIAN_PIP;
            break;
        }

        case Game::PendingCast::PHYREXIAN_PIP: {
            // Phyrexian mana: for each symbol, choose to pay colored mana or 2 life
            if (!card_data.phyrexian_mana.empty()) {
                Entity caster_entity = (caster == Zone::PLAYER_A)
                    ? cur_game.player_a_entity : cur_game.player_b_entity;
                auto &phyrex_player = global_coordinator.GetComponent<Player>(caster_entity);
                while (pc.phyrexian_idx < card_data.phyrexian_mana.size()) {
                    Colors phyrex_color = card_data.phyrexian_mana[pc.phyrexian_idx];
                    std::string color_name = mana_symbol_str(phyrex_color);
                    // CR 119.4 / 118.3: a player can't pay 2 life they don't have. Only offer
                    // the life option when life >= 2 (paying down to exactly 0 is legal — they
                    // die to SBAs afterward). Re-checked per pip against the running life total,
                    // since an earlier pip's life payment lowers what's left for the next.
                    // (The life is paid at APPLY below, so the NEXT pip's arm re-derives
                    // can_pay_life against the updated total, exactly like the inline loop;
                    // between THIS pip's arm and its apply nothing runs, so the recompute at
                    // apply matches the armed menu's option layout.)
                    bool can_pay_life = phyrex_player.life_total >= 2;
                    if (resume_choice >= 0) {
                        int phyrex_choice = resume_choice;
                        resume_choice = -1;
                        // The life option occupies index 0 only when it was offered; with it
                        // suppressed the sole option is "Pay {color}", so fall through to mana.
                        if (can_pay_life && phyrex_choice == 0) {
                            phyrex_player.life_total -= 2;
                            phyrex_player.life_lost_this_turn += 2;  // CR 119.4: paying life is losing life
                            game_log("%s pays 2 life\n", player_name(caster).c_str());
                        } else {
                            pc.cost_to_pay.insert(phyrex_color);
                        }
                        pc.phyrexian_idx++;
                        continue;
                    }
                    std::vector<LegalAction> phyrex_actions;
                    if (can_pay_life) {
                        LegalAction pay_life(PASS_PRIORITY,
                            "Pay 2 life (instead of {" + color_name + "})");
                        pay_life.category = ActionCategory::PAYING_COSTS;
                        pay_life.option_ordinal = 0;  // Phyrexian pip: 0 = pay life
                        phyrex_actions.push_back(pay_life);
                    }
                    LegalAction pay_mana(PASS_PRIORITY, "Pay {" + color_name + "}");
                    pay_mana.category = ActionCategory::PAYING_COSTS;
                    pay_mana.option_ordinal = 1;  // Phyrexian pip: 1 = pay mana
                    phyrex_actions.push_back(pay_mana);
                    arm_cast_query(game, std::move(phyrex_actions), caster);
                    return;
                }
            }

            // Defer the actual mana payment until after targets are chosen (CR 601.2c before
            // 601.2g/h — see the deferred_mana_* fields), so a sacrifice-for-mana source spent
            // here can't remove a spell's only legal target before it is chosen. The cost is
            // fully resolved (base + offspring/kicker/replicate/X/hybrid/phyrexian), so the
            // deferred payment is a pure spend with no target-dependent choices left.
            // Untargeted spells take the same path — their (empty) target step makes "pay
            // after targets" identical to paying now, so a single unified order serves every
            // cast.
            pc.deferred_mana_cost = pc.cost_to_pay;
            pc.deferred_delve = card_data.has_delve;
            pc.deferred_improvise = card_data.has_improvise;
            pc.deferred_mana_pending = true;
            pc.step = Game::PendingCast::LIFE_X;
            break;
        }

        case Game::PendingCast::LIFE_X: {
            // VARIABLE LIFE X-COST (Toxic Deluge: "As an additional cost, pay X life").
            // The life paid IS the spell's X (Count$xPaid). CR 601.2b ANNOUNCES the value
            // of X here, before targets; the life itself is a cost, so it is deferred and
            // paid with everything else at PAY_APPLY — a cancelled mana payment then costs
            // no life. X may be 0..life (CR 119.4 lets a player pay up to their whole total).
            if (spell_has_variable_life_cost(card_data)) {
                Entity caster_entity = (caster == Zone::PLAYER_A)
                    ? cur_game.player_a_entity : cur_game.player_b_entity;
                auto &life_player = global_coordinator.GetComponent<Player>(caster_entity);
                if (resume_choice >= 0) {
                    size_t x_val = static_cast<size_t>(resume_choice);
                    resume_choice = -1;
                    cur_game.x_paid = x_val;
                    pc.life_x_announced = static_cast<int>(x_val);
                } else {
                    size_t max_x = static_cast<size_t>(std::max(0, life_player.life_total));
                    game_log("Choose X value (0-%zu):\n", max_x);
                    std::vector<LegalAction> x_actions;
                    for (size_t xv = 0; xv <= max_x; xv++) {
                        LegalAction la(PASS_PRIORITY, std::string("X = " + std::to_string(xv)));
                        la.category = ActionCategory::CHOOSE_X;
                        la.option_ordinal = static_cast<int>(xv);  // the chosen X value
                        x_actions.push_back(la);
                    }
                    arm_cast_query(game, std::move(x_actions), caster);
                    return;
                }
            }
            pc.step = Game::PendingCast::GIFT;
            break;
        }

        case Game::PendingCast::SPELL_SAC: {
            // ADDITIONAL SACRIFICE COST on the spell itself (CR 601.2f / 118.x):
            // Natural Order — "As an additional cost to cast this spell, sacrifice a
            // green creature." Paid here as part of casting (before the spell is on the
            // stack), using the same SACRIFICE_PERMANENT choice activated abilities use.
            // Cast legality already guaranteed a matching permanent exists. General to
            // any spell whose SPELL ability Cost$ carries a Sac<...> token; flashback /
            // alternate casts pay their own sac cost in their own branch above.
            std::string spell_sac_spec = spell_additional_sac_spec(card_data);
            if (!spell_sac_spec.empty()) {
                // The exact pool/menu pay_sacrifice_cost's prompt_permanent_choice built,
                // armed as a pending decision instead. The pool re-derives identically at
                // apply — nothing runs between arm and resume.
                std::vector<Entity> choices = controlled_permanents_matching(
                    caster, spell_sac_spec, orderer->mEntities, spell_entity);
                if (!choices.empty()) {
                    if (resume_choice >= 0) {
                        Entity to_sac = choices[static_cast<size_t>(resume_choice)];
                        resume_choice = -1;
                        choose_cost_item(pc, to_sac, Zone::GRAVEYARD,
                                         player_name(caster) + " sacrifices " +
                                             global_coordinator.GetComponent<Permanent>(to_sac).name);
                    } else {
                        std::vector<LegalAction> menu;
                        for (auto e : choices) {
                            std::string nm = global_coordinator.GetComponent<Permanent>(e).name;
                            LegalAction la(PASS_PRIORITY, e, std::string("Sacrifice ") + nm);
                            la.category = ActionCategory::SACRIFICE_PERMANENT;
                            menu.push_back(la);
                        }
                        arm_cast_query(game, std::move(menu), caster);
                        return;
                    }
                }
            }
            pc.step = Game::PendingCast::DELVE_COUNT;
            break;
        }

        case Game::PendingCast::GIFT: {
            // GIFT (CR 702.176b): as the spell is cast, its controller MAY promise the gift to an
            // opponent. The promise is not a cost — it is decided here (before targets are chosen,
            // CR 601.2c) and it both (a) switches a Count$PromisedGift-driven effect via the
            // pending flag while targets are selected and (b) makes the opponent receive the gift
            // on resolution (see Spell::gift_promised / Ability::resolve). Optional yes/no.
            if (card_data.has_gift) {
                if (resume_choice >= 0) {
                    bool accepted = (resume_choice == 1);
                    resume_choice = -1;
                    if (accepted) {
                        pc.gift_promised = true;
                        game_log("%s promises the gift to %s\n", player_name(caster).c_str(),
                                 player_name(caster == Zone::PLAYER_A ? Zone::PLAYER_B
                                                                      : Zone::PLAYER_A).c_str());
                    }
                } else {
                    std::string gname = card_data.gift_description.empty()
                                            ? std::string("the gift") : card_data.gift_description;
                    // CR 601.2c / 702.176b: the gift promise switches which target the spell requires
                    // (Into the Flood Maw: a creature when declined, a nonland permanent when promised).
                    // If the not-promised mode has no legal target — e.g. the opponent controls no
                    // (targetable) creature — declining is not a legal way to cast the spell, so don't
                    // offer the yes/no: force the promise rather than dropping into a zero-target mode
                    // that fizzles. The cast was only legal because the promised mode is satisfiable.
                    // Mirror case: if the PROMISED mode has no legal target — e.g. the opponent's
                    // only creature is Dryad Arbor, a Land Creature, so "nonland permanent" is empty
                    // while "creature" is not — promising is not a legal way to cast the spell
                    // either, so don't offer the yes/no at all (the cast was only legal because the
                    // not-promised mode is satisfiable).
                    bool must_promise = false;
                    bool can_promise = true;
                    for (const auto &t : card_data.abilities) {
                        if (t.ability_type != Ability::SPELL) continue;
                        // Probe with the real spell source/controller so a mode's target legality
                        // (protection from this spell's color, OppCtrl) matches select_target below —
                        // otherwise can_promise/must_promise can green-light a mode whose only target
                        // is protected (Scryb Ranger vs blue Into the Flood Maw), then select_target
                        // finds zero targets and aborts (CR 601.2c / 702.16e).
                        Ability probe = cast_gate_probe(t, spell_entity, caster);
                        std::vector<const Ability *> targeting = spell_targeting_abilities(probe);
                        if (!targeting.empty()) {
                            must_promise = !gift_mode_satisfiable(targeting, orderer, caster, false);
                            can_promise = gift_mode_satisfiable(targeting, orderer, caster, true);
                        }
                        break;
                    }
                    if (must_promise) {
                        // Forced promise: today's short-circuit never prompted here, so
                        // advance without arming — zero decisions, exactly as before.
                        pc.gift_promised = true;
                        game_log("%s promises the gift to %s\n", player_name(caster).c_str(),
                                 player_name(caster == Zone::PLAYER_A ? Zone::PLAYER_B
                                                                      : Zone::PLAYER_A).c_str());
                    } else if (can_promise) {
                        arm_cast_query(game,
                                       optional_yesno_menu("promise " + gname + " to your opponent"),
                                       caster);
                        return;
                    }
                }
            }
            cur_game.pending_gift_promised = pc.gift_promised;
            pc.step = Game::PendingCast::ANNOUNCE;
            break;
        }

        case Game::PendingCast::ANNOUNCE: {
            // Find the primary spell ability template and copy it into pc BY VALUE — the
            // ENTITY's Ability component is added only once every announce target is chosen
            // (end of SUB_TARGET), the blocking flow's exact position, so component state at
            // every announce prompt matches it (absent during announcement, present after).
            for (const auto &ability_template : card_data.abilities) {
                if (ability_template.ability_type != Ability::SPELL) continue;
                pc.ability = ability_template;
                pc.ability.source = spell_entity;
                pc.ability.controller = caster;
                // Carry the Gift keyword's gift effect onto the resolving spell's primary ability;
                // it fires at resolution only if the gift was promised (Ability::resolve).
                if (card_data.has_gift) pc.ability.gift_abilities = card_data.gift_abilities;
                pc.have_ability = true;
                break;  // TODO: support spells with multiple abilities
            }

            // Announce cast-time choices (CR 601.2b/c) across the steps below: modal mode(s)
            // interleaved with their targets (CHARM_MODE/CHARM_TARGET), the primary target
            // (PRIMARY_TARGET), targeting sub-abilities' targets (SUB_TARGET), and the aura
            // enchant target (AURA_TARGET) — the same interleave announce_spell_targets runs
            // for the blocking mini-cast. NOTE on ordering: strict CR 601.2b announces modes
            // before X, but X was chosen above in the cost branch — mode choosability can
            // depend on X (Kozilek's Command's "Creature.cmcLEX" exile mode), and both are
            // the caster's own announcements made atomically before any opponent priority,
            // so the swap is not opponent-observable.
            pc.charm_picks_done = 0;
            pc.sub_idx = 0;
            pc.tsel = TargetSelectRT{};
            if (!pc.have_ability)
                pc.step = Game::PendingCast::AURA_TARGET;
            else if (!pc.ability.charm_choices.empty())
                pc.step = Game::PendingCast::CHARM_MODE;
            else
                pc.step = Game::PendingCast::PRIMARY_TARGET;
            break;
        }

        case Game::PendingCast::CHARM_MODE: {
            // Modal spell announcement (CR 601.2b): one mode pick per pass; the just-picked
            // mode's targets are chosen (CHARM_TARGET) before the NEXT mode pick, exactly the
            // blocking announce_charm_modes interleave. The picked modes persist in
            // ability.charm_chosen — which also reconstructs the taken[] filter on resume —
            // and pc.charm_picks_done counts completed iterations.
            Ability &ability = pc.ability;
            int to_pick = ability.charm_num < 1 ? 1 : ability.charm_num;
            if (pc.charm_picks_done >= to_pick) {
                pc.step = Game::PendingCast::PRIMARY_TARGET;
                break;
            }
            std::vector<bool> taken(ability.charm_choices.size(), false);
            for (int ci : ability.charm_chosen)
                if (ci >= 0 && static_cast<size_t>(ci) < taken.size())
                    taken[static_cast<size_t>(ci)] = true;
            std::vector<size_t> mode_indices;  // map action index -> charm_choices index
            if (resume_choice >= 0) {
                // Apply the latched mode pick against the re-derived (identical) menu.
                std::vector<LegalAction> mode_actions =
                    build_charm_mode_menu(ability, orderer, caster, taken, mode_indices);
                size_t chosen_idx = mode_indices[static_cast<size_t>(resume_choice)];
                resume_choice = -1;
                ability.charm_chosen.push_back(static_cast<int>(chosen_idx));
                Ability &chosen = ability.charm_choices[chosen_idx];
                game_log("%s chooses mode — %s\n", player_name(caster).c_str(),
                         charm_mode_desc(ability, chosen_idx).c_str());
                if (chosen.valid_tgts != "N_A") {
                    pc.tsel = TargetSelectRT{};
                    pc.step = Game::PendingCast::CHARM_TARGET;
                } else {
                    pc.charm_picks_done++;  // no targets — straight to the next mode pick
                }
                break;
            }
            game_log("Choose mode:\n");
            std::vector<LegalAction> mode_actions =
                build_charm_mode_menu(ability, orderer, caster, taken, mode_indices);
            if (mode_actions.empty()) {
                // No further legal mode (all taken or none with legal targets). The
                // cast-legality gate (spell_has_castable_targets) requires CharmNum$
                // choosable modes up front, so this is only reachable when an earlier pick's
                // target choice changed the board — proceed with the modes picked so far
                // rather than aborting the cast. Re-evaluated at arm, like the blocking loop.
                game_log("No further legal mode — %d chosen\n", pc.charm_picks_done);
                pc.step = Game::PendingCast::PRIMARY_TARGET;
                break;
            }
            arm_cast_query(game, std::move(mode_actions), caster, ability.source);
            return;
        }

        case Game::PendingCast::CHARM_TARGET: {
            // The just-picked mode's targets (CR 601.2c), before the next mode pick.
            Ability &chosen = pc.ability.charm_choices[
                static_cast<size_t>(pc.ability.charm_chosen.back())];
            FlowTargetAsker asker(game, caster, resume_choice);
            if (run_target_select(chosen, pc.tsel, asker, orderer, caster) !=
                TargetStatus::DONE)
                return;
            pc.charm_picks_done++;
            pc.step = Game::PendingCast::CHARM_MODE;
            break;
        }

        case Game::PendingCast::PRIMARY_TARGET: {
            if (pc.ability.valid_tgts != "N_A") {
                FlowTargetAsker asker(game, caster, resume_choice);
                if (run_target_select(pc.ability, pc.tsel, asker, orderer, caster) !=
                    TargetStatus::DONE)
                    return;
            }
            pc.sub_idx = 0;
            pc.step = Game::PendingCast::SUB_TARGET;
            break;
        }

        case Game::PendingCast::SUB_TARGET: {
            // A spell whose top-level effect doesn't itself target, but whose chained
            // sub-ability does, chooses that target as it's cast (CR 601.2c). Cabal Therapy:
            // SP$ NameCard (Defined$ You, no target) + DB$ Discard (ValidTgts$ Player).
            // Select each targeting sub-ability's target now and store it on the sub-ability
            // template; resolution preserves it (see Ability::resolve).
            while (pc.sub_idx < pc.ability.subabilities.size()) {
                Ability &sub = pc.ability.subabilities[pc.sub_idx];
                if (sub.valid_tgts != "N_A") {
                    sub.source = pc.ability.source;
                    sub.controller = caster;
                    FlowTargetAsker asker(game, caster, resume_choice);
                    if (run_target_select(sub, pc.tsel, asker, orderer, caster) !=
                        TargetStatus::DONE)
                        return;
                }
                pc.sub_idx++;
            }
            // Announcement complete: add the fully-targeted Ability to the entity — the
            // blocking flow's exact position (right after announce_spell_targets returned).
            global_coordinator.AddComponent(spell_entity, pc.ability);
            pc.step = Game::PendingCast::AURA_TARGET;
            break;
        }

        case Game::PendingCast::AURA_TARGET: {
            // AURA cast (CR 303.4 / 601.2c): an Aura with no spell ability of its own still
            // targets the object it will enchant. Build a transient targeting ability from the
            // Enchant filter, choose the target now, and remember it so the resolved permanent
            // attaches to it (its equipped_to is set when its Permanent is created). The
            // transient ability persists in pc.enchant_ab; a suspended pick resumes it
            // (tsel.active guards the one-time construction).
            if (!card_data.enchant_filter.empty() &&
                !global_coordinator.entity_has_component<Ability>(spell_entity)) {
                if (!pc.tsel.active) {
                    pc.enchant_ab = Ability{};
                    pc.enchant_ab.source = spell_entity;
                    pc.enchant_ab.controller = caster;
                    pc.enchant_ab.valid_tgts = card_data.enchant_filter;
                    // "Enchant creature card in a graveyard" (Animate Dead): the enchant target is
                    // a creature card in a graveyard, so the pick searches graveyards (CR 303.4).
                    pc.enchant_ab.target_in_graveyard =
                        enchant_targets_graveyard(card_data.enchant_filter);
                }
                FlowTargetAsker asker(game, caster, resume_choice);
                if (run_target_select(pc.enchant_ab, pc.tsel, asker, orderer, caster) !=
                    TargetStatus::DONE)
                    return;
                if (pc.enchant_ab.target != 0) {
                    cur_game.pending_aura_target[spell_entity] = pc.enchant_ab.target;
                    game_log("%s casts %s enchanting %s\n", player_name(caster).c_str(),
                             card_data.name.c_str(),
                             target_display_name(cur_game, pc.enchant_ab.target).c_str());
                }
            }
            // Targets are locked in (CR 601.2c); everything from here is cost payment
            // (601.2f-h) — first the non-mana cost PICKS, then mana, then the moves.
            pc.step = Game::PendingCast::ALT_PITCH;
            break;
        }

        case Game::PendingCast::DELVE_COUNT: {
            // Delve (CR 702.66 / 601.2h): the caster chooses how many graveyard cards to
            // exile and which ones, one pick at a time; each exile pays one generic pip of
            // the deferred cost. Runs after targets are locked in (601.2c), like every
            // other cost. The remainder is then paid WITHOUT delve — the count menu is
            // constrained to counts whose remaining cost is payable. Stage 1 here: the
            // count menu. The count range [min_needed .. max_exiles] is computed at arm
            // and re-derived identically at apply (nothing runs between arm and resume):
            // each exile removes one GENERIC pip, so payability is monotone in the count
            // and the legal counts form a contiguous range; min_needed is found with the
            // shared simulate-mode payer (can_pay_mana, has_delve=false), so the offered
            // menu and the eventual payment can never disagree — cast legality (which
            // allowed the maximum delve) guarantees the range is non-empty.
            if (!pc.deferred_mana_pending || !pc.deferred_delve) {
                pc.step = Game::PendingCast::DEF_SAC;
                break;
            }
            size_t eligible_ct = 0;
            for (auto e : orderer->mEntities)
                if (is_delve_eligible(e, caster)) eligible_ct++;
            size_t max_exiles = std::min(eligible_ct, pc.deferred_mana_cost.count(GENERIC));
            if (max_exiles == 0) {
                pc.step = Game::PendingCast::DEF_SAC;
                break;
            }
            ManaValue reduced = pc.deferred_mana_cost;
            size_t min_needed = 0;
            while (min_needed < max_exiles &&
                   !can_pay_mana(caster, reduced, spell_entity, orderer, /*has_delve=*/false,
                                 pc.deferred_improvise)) {
                reduced.erase(reduced.find(GENERIC));
                min_needed++;
            }
            if (resume_choice >= 0) {
                // Apply the latched count against the re-derived (identical) range.
                pc.delve_exile_ct = min_needed + static_cast<size_t>(resume_choice);
                resume_choice = -1;
                pc.step = Game::PendingCast::DELVE_PICK;
                break;
            }
            // Seat both delve stages on the casting player — the blocking prompt's
            // save/repoint/restore becomes arm-time persistence: the prev seat lives in
            // pc and DELVE_PICK's completion restores it.
            pc.delve_prev_priority_a = cur_game.player_a_has_priority;
            cur_game.player_a_has_priority = (caster == Zone::PLAYER_A);
            // The count prompt is skipped when only one count is legal (the pre-collapse
            // condition, re-derived at arm). The actions carry the delve spell as their
            // source entity, so the machine protocol emits its card id (a plain X-cost
            // ladder emits the null sentinel — this is how observers tell the two
            // CHOOSE_X menus apart).
            if (max_exiles > min_needed) {
                std::vector<LegalAction> count_menu;
                for (size_t n = min_needed; n <= max_exiles; n++) {
                    LegalAction la(PASS_PRIORITY, spell_entity,
                                   "Exile " + std::to_string(n) + (n == 1 ? " card" : " cards") +
                                       " from graveyard (Delve)");
                    la.category = ActionCategory::CHOOSE_X;
                    la.option_ordinal = static_cast<int>(n);  // the delve exile count
                    la.card_is_public = true;  // the spell being cast is public
                    count_menu.push_back(la);
                }
                game_log("Choose how many cards to exile via Delve (%zu-%zu):\n", min_needed,
                         max_exiles);
                arm_cast_query(game, std::move(count_menu), caster);
                return;
            }
            pc.delve_exile_ct = min_needed;
            pc.step = Game::PendingCast::DELVE_PICK;
            break;
        }

        case Game::PendingCast::DELVE_PICK: {
            // Delve stage 2 — which cards, one pick at a time; the candidate menu is
            // re-derived from the live graveyard each pass (each exile drops that card
            // from the next menu). When the remaining candidates are all going to be
            // exiled anyway (candidates <= picks left, including the single-candidate
            // case) there is no real choice, so they are taken without a prompt — the
            // pre-collapse condition, re-checked per pick at arm. delve_exile_one mutates
            // pc.deferred_mana_cost IN PLACE (one GENERIC pip per exile) and records the
            // card in cur_game.delve_exiled; a cancelled payment's restore_mana_state
            // returns the exiles via its snapshot.
            while (pc.delve_picks_done < pc.delve_exile_ct) {
                std::vector<LegalAction> picks;
                for (auto e : orderer->mEntities) {
                    if (!is_delve_eligible(e, caster)) continue;
                    auto &ecd = global_coordinator.GetComponent<CardData>(e);
                    LegalAction la(PASS_PRIORITY, e, "Exile " + ecd.name + " (Delve)");
                    la.category = ActionCategory::CHOOSE_CARD;
                    la.card_is_public = true;  // graveyards are public zones
                    picks.push_back(la);
                }
                if (picks.empty()) break;  // defensive: eligibility was counted above
                if (picks.size() > pc.delve_exile_ct - pc.delve_picks_done) {
                    if (resume_choice >= 0) {
                        size_t pick = static_cast<size_t>(resume_choice);
                        resume_choice = -1;
                        delve_exile_one(picks[pick].source_entity, caster, orderer,
                                        pc.deferred_mana_cost);
                        pc.delve_picks_done++;
                        continue;
                    }
                    game_log("Choose a card to exile via Delve (%zu of %zu):\n",
                             pc.delve_picks_done + 1, pc.delve_exile_ct);
                    arm_cast_query(game, std::move(picks), caster);
                    return;
                }
                delve_exile_one(picks[0].source_entity, caster, orderer, pc.deferred_mana_cost);
                pc.delve_picks_done++;
            }
            // Restore the pre-delve seat (persisted at DELVE_COUNT's arm).
            cur_game.player_a_has_priority = pc.delve_prev_priority_a;
            pc.step = Game::PendingCast::DEF_SAC;
            break;
        }

        case Game::PendingCast::MANA_PAY: {
            // Pay the mana cost, last of the costs (CR 601.2h). Targets are locked in
            // (601.2c) and every non-mana cost item has been CHOSEN but not yet applied,
            // which is what makes this step safely failable: nothing irreversible has
            // happened, so the cancel path below is a complete rewind.
            // (The interactive payment stays BLOCKING inside this step — a deliberate
            // non-conversion; machine mode auto-pays with zero decisions.)
            //
            // CR 601.2g first: a permanent that is about to leave to pay one of those
            // costs is still on the battlefield right now, so tap it for mana on its way
            // out. Without this a Crop Rotation cast off a lone Savannah would sacrifice
            // the land and then be unable to pay {G}.
            if (pc.deferred_mana_pending)
                for (const auto &r : pc.cost_removals)
                    float_mana_before_cost_removal(r.entity, caster, orderer,
                                                   pc.deferred_mana_cost, spell_entity);
            // Record the mana actually spent (CR 106/601.2g) for a ValidSA$ Spell.ManaSpent
            // trigger to read at cast time. The deferred cost's pip count is the total mana paid,
            // after any delve/improvise reduction already removed pips from it. A no-mana
            // alternative cost leaves deferred_mana_pending false, and the 0 that
            // defer_alternate_cost recorded stands.
            if (pc.deferred_mana_pending)
                pc.mana_spent = static_cast<int>(pc.deferred_mana_cost.size());
            if (pc.deferred_mana_pending) {
                if (!prompt_mana_payment(caster, pc.deferred_mana_cost, spell_entity, orderer,
                                         /*has_delve=*/false, pc.deferred_improvise,
                                         &pc.mana_spent_colors)) {
                    // Payment cancelled (interactive only — machine mode pre-verifies
                    // affordability). Targets were already chosen but the spell never reached the
                    // stack, so rewind the half-finished cast: drop the targeting Ability / aura
                    // link, clear the pending gift flag, restore mana, and bump payment_fail_counts
                    // so the offer gate stops re-offering it (no scripted-agent payment loop).
                    // The parked cast state is cleared with it — and with it every CHOSEN but
                    // unapplied cost item (cost_removals) and the deferred life, so the failure
                    // costs nothing but the mana rewind (CR 733 / 601.2h). restore_mana_state
                    // also untaps whatever the float above tapped.
                    if (global_coordinator.entity_has_component<Ability>(spell_entity))
                        global_coordinator.RemoveComponent<Ability>(spell_entity);
                    cur_game.pending_aura_target.erase(spell_entity);
                    cur_game.pending_gift_promised = false;
                    restore_mana_state(caster, pc.mana_snap, orderer);
                    cur_game.payment_fail_counts[spell_entity]++;
                    game_log("Payment cancelled.\n");
                    pc = Game::PendingCast{};
                    return;
                }
            }

            pc.step = Game::PendingCast::PAY_APPLY;
            break;
        }

        case Game::PendingCast::PAY_APPLY: {
            // The mana committed, so the rest of the cost is now paid for real: the life,
            // then every cost item chosen during the pick steps (alt-cost pitch/bounce/
            // sacrifice, the spell's additional sacrifice, flashback's sacrifice, escape's
            // graveyard exiles). Nothing here can fail — the choices were made against the
            // live board and only this step moves anything — which is exactly why the
            // preceding payment is allowed to fail.
            //
            // A permanent leaving here may make a chosen target illegal; the spell then
            // fizzles at resolution (CR 608.2b), matching paper rules.
            if (pc.deferred_life_cost > 0 || pc.life_x_announced >= 0) {
                auto &player = global_coordinator.GetComponent<Player>(get_player_entity(caster));
                if (pc.deferred_life_cost > 0) {
                    player.life_total -= pc.deferred_life_cost;
                    player.life_lost_this_turn += pc.deferred_life_cost;  // CR 119.4: paying life is losing life
                    game_log("%s pays %d life\n", player_name(caster).c_str(), pc.deferred_life_cost);
                }
                if (pc.life_x_announced >= 0) {
                    player.life_total -= pc.life_x_announced;
                    player.life_lost_this_turn += pc.life_x_announced;
                    game_log("%s pays %d life (X = %d)\n", player_name(caster).c_str(),
                             pc.life_x_announced, pc.life_x_announced);
                }
            }
            for (const auto &r : pc.cost_removals) {
                game_log("%s\n", r.log.c_str());
                orderer->add_to_zone(false, r.entity, r.dest);
            }
            pc.cost_removals.clear();
            pc.step = Game::PendingCast::FINISH;
            break;
        }

        case Game::PendingCast::DEF_SAC: {
            // Deferred "sacrifice a <spec>" cost — the flashback alternate cost's
            // sacrifice (CR 702.34, Cabal Therapy). The exact pool/menu the old blocking
            // pay_sacrifice_cost built (via prompt_permanent_choice), armed as a pending
            // decision instead; the pool re-derives identically at apply. Cast legality
            // already guaranteed a matching permanent exists; the empty-choices guard is
            // a defensive no-op.
            if (!pc.deferred_sac_spec.empty()) {
                std::vector<Entity> choices = controlled_permanents_matching(
                    caster, pc.deferred_sac_spec, orderer->mEntities, spell_entity);
                if (!choices.empty()) {
                    if (resume_choice >= 0) {
                        Entity to_sac = choices[static_cast<size_t>(resume_choice)];
                        resume_choice = -1;
                        choose_cost_item(pc, to_sac, Zone::GRAVEYARD,
                                         player_name(caster) + " sacrifices " +
                                             global_coordinator.GetComponent<Permanent>(to_sac).name);
                    } else {
                        std::vector<LegalAction> menu;
                        for (auto e : choices) {
                            std::string nm = global_coordinator.GetComponent<Permanent>(e).name;
                            LegalAction la(PASS_PRIORITY, e, std::string("Sacrifice ") + nm);
                            la.category = ActionCategory::SACRIFICE_PERMANENT;
                            menu.push_back(la);
                        }
                        arm_cast_query(game, std::move(menu), caster);
                        return;
                    }
                }
            }
            pc.step = Game::PendingCast::DEF_EXILE_TYPES;
            break;
        }

        case Game::PendingCast::DEF_EXILE_TYPES: {
            // Deferred Escape ExileFromGrave cost (CR 702.139 / 601.2f): exile any number
            // of OTHER cards from the caster's graveyard until the exiled set collectively
            // has at least min_types distinct card types (CR 205.2). One pick per pass, the
            // candidate menu re-derived from the live graveyard at each arm; the loop is
            // mandatory (no "done" option) until the constraint — tracked across
            // suspensions in pc.escape_exiled_types — is met. Cast legality already
            // guaranteed enough types exist.
            if (pc.deferred_exile_min_types > 0) {
                while (static_cast<int>(pc.escape_exiled_types.size()) <
                       pc.deferred_exile_min_types) {
                    std::vector<LegalAction> menu =
                        escape_exile_menu(caster, spell_entity, orderer);
                    drop_chosen_cost_items(pc, menu);
                    if (menu.empty()) break;  // defensive: legality guaranteed enough cards
                    if (resume_choice >= 0) {
                        Entity to_exile = menu[static_cast<size_t>(resume_choice)].source_entity;
                        resume_choice = -1;
                        auto &cd = global_coordinator.GetComponent<CardData>(to_exile);
                        for (auto &t : cd.types)
                            if (t.kind == TYPE) pc.escape_exiled_types.insert(t.name);
                        choose_cost_item(pc, to_exile, Zone::EXILE,
                                         player_name(caster) + " exiles " + cd.name +
                                             " from their graveyard");
                        continue;
                    }
                    arm_cast_query(game, std::move(menu), caster);
                    return;
                }
            }
            pc.step = Game::PendingCast::DEF_EXILE_COUNT;
            break;
        }

        case Game::PendingCast::DEF_EXILE_COUNT: {
            // Deferred Escape ExileFromGrave cost in its literal-count form (CR 702.139 /
            // 601.2f): exile exactly `count` OTHER cards from the caster's graveyard (Uro:
            // "Exile five other cards from your graveyard"). One mandatory pick per pass
            // (no "done" option) until pc.escape_exiled_count reaches the count; the menu
            // re-derives from the live graveyard at each arm. Cast legality already
            // guaranteed enough cards exist.
            if (pc.deferred_exile_count > 0) {
                while (pc.escape_exiled_count < pc.deferred_exile_count) {
                    std::vector<LegalAction> menu =
                        escape_exile_menu(caster, spell_entity, orderer);
                    drop_chosen_cost_items(pc, menu);
                    if (menu.empty()) break;  // defensive: legality guaranteed enough cards
                    if (resume_choice >= 0) {
                        Entity to_exile = menu[static_cast<size_t>(resume_choice)].source_entity;
                        resume_choice = -1;
                        choose_cost_item(pc, to_exile, Zone::EXILE,
                                         player_name(caster) + " exiles " +
                                             global_coordinator.GetComponent<CardData>(to_exile).name +
                                             " from their graveyard");
                        pc.escape_exiled_count++;
                        continue;
                    }
                    arm_cast_query(game, std::move(menu), caster);
                    return;
                }
            }
            // Every non-mana cost item is chosen; the mana is the last thing paid.
            pc.step = Game::PendingCast::MANA_PAY;
            break;
        }

        case Game::PendingCast::FINISH: {
            // Log cast with target if applicable
            if (global_coordinator.entity_has_component<Ability>(spell_entity)) {
                Entity tgt = global_coordinator.GetComponent<Ability>(spell_entity).target;
                if (tgt != 0) {
                    std::string tgt_name = target_display_name(cur_game, tgt);
                    game_log("%s casts %s targeting %s\n", player_name(caster).c_str(),
                        card_data.name.c_str(), tgt_name.c_str());
                } else {
                    game_log("%s casts %s\n", player_name(caster).c_str(), card_data.name.c_str());
                }
            } else {
                game_log("%s casts %s\n", player_name(caster).c_str(), card_data.name.c_str());
            }

            // Add Spell component — present only while the entity is on the stack
            Spell spell;
            spell.caster = caster;
            spell.cast_with_flashback = pc.use_flashback;
            spell.cast_with_evoke = pc.use_alt_cost && card_data.alt_cost.is_evoke;
            spell.cast_with_escape = pc.use_escape;
            spell.cast_with_offspring = pc.use_offspring;
            spell.cast_with_impending = pc.use_alt_cost && card_data.alt_cost.is_impending;
            spell.cast_with_warp = pc.use_alt_cost && card_data.alt_cost.is_warp;
            spell.kicked = pc.kicked_flags;  // per-kicker "paid?" flags (empty for non-kicker spells)
            spell.replicate_count = pc.replicate_count;  // # of replicate payments (0 if none/no Replicate)
            spell.gift_promised = pc.gift_promised;  // Gift (CR 702.176): opponent gets the gift on resolution
            spell.mana_spent = pc.mana_spent;  // total mana paid (CR 106); read by ValidSA$ Spell.ManaSpent triggers
            // Converge (CR 702.90): the distinct real colors of mana spent (colorless is not a
            // color). Its size is the Converge count, restored into cur_game.converge at resolution
            // and read by a Count$Converge bound (Prismatic Ending's cmcLEY exile threshold).
            for (Colors c : pc.mana_spent_colors)
                if (c == WHITE || c == BLUE || c == BLACK || c == RED || c == GREEN)
                    spell.colors_spent.insert(c);
            cur_game.pending_gift_promised = false;  // consume the cast-time pending flag (targets chosen)
            // Record the X value paid so an "enters with X counters" replacement can read
            // it (Chalice of the Void: enters with X charge counters) and so the resolving
            // spell's Count$xPaid amount reads the right X (StackManager restores x_paid from
            // this). cur_game.x_paid is global and may be overwritten by a later cast before this
            // spell resolves. A variable-life X spell (Toxic Deluge) has no mana X, so also key
            // off its PayLife<X> cost.
            if (card_data.has_x_cost || spell_has_variable_life_cost(card_data))
                spell.x_paid = static_cast<int>(cur_game.x_paid);
            if (cur_game.pending_cant_be_countered) {
                spell.cant_be_countered = true;
                cur_game.pending_cant_be_countered = false;
            }
            // Check card's own replacement effects for "can't be countered" (Long Goodbye).
            // Only the UNCONDITIONAL SELF form ("This spell can't be countered") stamps the spell
            // at cast; the battlefield form (Hexing Squelcher's "Spells you control can't be
            // countered") is a continuous static consulted at counter-resolution time, and a
            // CONDITIONAL self form (Exquisite Firecraft's spell-mastery gate, cant_counter_present
            // non-empty) is likewise re-evaluated at counter time — not stamped now.
            for (const auto &r : card_data.replacement_effects) {
                if (r.kind == Effect::Replacement::CANT_BE_COUNTERED && !r.from_battlefield &&
                    r.cant_counter_present.empty()) {
                    spell.cant_be_countered = true;
                    break;
                }
            }
            global_coordinator.AddComponent(spell_entity, spell);

            // Fire NONCREATURE_SPELL_CAST event for non-creature spells
            {
                bool is_creature_spell = false;
                for (const auto &t : card_data.types)
                    if (t.kind == TYPE && t.name == "Creature") {
                        is_creature_spell = true;
                        break;
                    }
                if (!is_creature_spell) {
                    Event cast_ev(Events::NONCREATURE_SPELL_CAST);
                    Entity caster_entity =
                        (caster == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
                    cast_ev.SetParam(Params::ENTITY, spell_entity);
                    cast_ev.SetParam(Params::PLAYER, caster_entity);
                    global_coordinator.SendEvent(cast_ev);
                }
            }

            // Move to stack
            orderer->add_to_zone(false, spell_entity, Zone::STACK);  // Top of stack

            // Track spells cast and fire SPELL_CAST event
            {
                Entity caster_entity = (caster == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
                auto &caster_player = global_coordinator.GetComponent<Player>(caster_entity);
                caster_player.spells_cast_this_turn++;
                caster_player.spells_cast_this_game++;
                // Track noncreature spells for Deafening Silence, and instant/sorcery
                // spells for Arclight Phoenix's "cast three or more instant and sorcery
                // spells this turn" count.
                bool spell_is_creature = false;
                bool spell_is_instant_or_sorcery = false;
                for (auto &t : card_data.types)
                    if (t.kind == TYPE) {
                        if (t.name == "Creature") spell_is_creature = true;
                        if (t.name == "Instant" || t.name == "Sorcery") spell_is_instant_or_sorcery = true;
                    }
                if (!spell_is_creature) caster_player.noncreature_spells_cast_this_turn++;
                if (spell_is_instant_or_sorcery) caster_player.instant_sorcery_spells_cast_this_turn++;
                // Record the spell's colors so a "an opponent has cast a <color> spell this turn"
                // condition (Veil of Summer's Count$ThisTurnCast_Card.OppCtrl+Blue/Black) can be
                // evaluated. The spell entity carries the card's ColorIdentity.
                if (global_coordinator.entity_has_component<ColorIdentity>(spell_entity))
                    for (Colors c : global_coordinator.GetComponent<ColorIdentity>(spell_entity).colors)
                        caster_player.spell_colors_cast_this_turn.insert(c);
                Event spell_event(Events::SPELL_CAST);
                spell_event.SetParam(Params::PLAYER, caster_entity);
                spell_event.SetParam(Params::ENTITY, spell_entity);
                global_coordinator.SendEvent(spell_event);
            }

            // Ward (702.21): an opponent's permanent this spell targets may counter it. The
            // spell is already on the stack, so the Ward trigger pushed here lands above it and
            // resolves first. Read the chosen target(s) off the spell's Ability component.
            if (global_coordinator.entity_has_component<Ability>(spell_entity)) {
                auto &spell_ab = global_coordinator.GetComponent<Ability>(spell_entity);
                // Mode$ BecomesTarget triggers (Reality Smasher): a targeted permanent whose
                // becomes-target trigger matches fires it above this spell (CR 603.2c/603.3).
                fire_targeting_hooks(spell_entity, caster, spell_ab, orderer);
            }

            // REPLICATE (CR 702.x): "When you cast this spell, copy it for each time you paid
            // its replicate cost." The replicate count was recorded on the Spell as the cost
            // was paid; create that many copies of this spell on top of the stack now (the
            // copies resolve before the original and may choose new targets). Seed the shared
            // resumable copy machine and hand off to COPY_TARGETS — the copies' target picks
            // suspend as loop-top pending decisions, and a resume must not re-run this step's
            // cast events. A copy is not cast, so it replicates nothing.
            if (global_coordinator.entity_has_component<Spell>(spell_entity)) {
                int rc = global_coordinator.GetComponent<Spell>(spell_entity).replicate_count;
                if (rc > 0) copy_spell_begin(pc.copy_rt, spell_entity, rc, caster);
            }
            pc.step = Game::PendingCast::COPY_TARGETS;
            break;
        }

        case Game::PendingCast::COPY_TARGETS: {
            // Replicate copies choose their targets (CR 707.12). The machine tolerates a
            // no-legal-target copy being destroyed mid-loop and resumes with the remaining
            // count / the partially targeted copy persisted in pc.copy_rt.
            if (pc.copy_rt.active) {
                FlowTargetAsker asker(game, caster, resume_choice);
                if (run_copy_spell(pc.copy_rt, asker, orderer) != TargetStatus::DONE)
                    return;
            }
            // take_action stays LAST — after the copies, exactly the blocking order (the
            // cancel path never reaches here, like the old break).
            game.take_action();
            pc = Game::PendingCast{};
            return;
        }

        default:
            fatal_error("run_cast_flow reached an unimplemented step " +
                        std::to_string(static_cast<int>(pc.step)));
    }
}

void process_action(const LegalAction &action, Game &game, std::shared_ptr<Orderer> orderer) {
    switch (action.type) {
        case PASS_PRIORITY:
            game.pass_priority();
            break;

        case SPECIAL_ACTION: {
            // SUSPEND (CR 702.62a): pay the suspend cost and exile the card from hand with N time
            // counters on it. This is a special action (doesn't use the stack). The legal-action
            // gate already verified sorcery-speed timing, no cast prohibition, and affordability of
            // the suspend mana cost. Time counters are tracked in cur_game.suspend_time_counters
            // (an exiled card is not a permanent, so its counters can't live in Permanent::counters).
            if (action.suspend_action) {
                Entity card = action.source_entity;
                auto &zone = global_coordinator.GetComponent<Zone>(card);
                auto &cd = global_coordinator.GetComponent<CardData>(card);
                Zone::Ownership owner = zone.owner;
                auto mana_snap = snapshot_mana_state(owner, orderer);
                ManaValue cost = cd.suspend_cost;  // copy (prompt_mana_payment takes a mutable ref)
                if (!prompt_mana_payment(owner, cost, card, orderer)) {
                    restore_mana_state(owner, mana_snap, orderer);
                    game_log("Payment cancelled.\n");
                    break;
                }
                orderer->add_to_zone(false, card, Zone::EXILE);
                cur_game.suspend_time_counters[card] = cd.suspend_count;
                game_log("%s suspends %s (exiled with %d time counter(s)).\n",
                         player_name(owner).c_str(), cd.name.c_str(), cd.suspend_count);
                game.take_action();
                break;
            }

            // COMPANION (CR 702.139): pay {3} and put the chosen companion from the sideboard into
            // its owner's hand, once per game. The legal-action gate already verified the companion
            // is in the sideboard, unused this game, and that {3} is affordable.
            if (action.companion_to_hand) {
                Entity comp = action.source_entity;
                auto &zone = global_coordinator.GetComponent<Zone>(comp);
                Zone::Ownership owner = zone.owner;
                auto mana_snap = snapshot_mana_state(owner, orderer);
                ManaValue three = {GENERIC, GENERIC, GENERIC};
                if (!prompt_mana_payment(owner, three, comp, orderer)) {
                    restore_mana_state(owner, mana_snap, orderer);
                    game_log("Payment cancelled.\n");
                    break;
                }
                std::string cname = global_coordinator.GetComponent<CardData>(comp).name;
                orderer->add_to_zone(false, comp, Zone::HAND);
                auto &player = global_coordinator.GetComponent<Player>(get_player_entity(owner));
                player.companion_brought_to_hand = true;
                game_log("%s pays {3} and puts %s into their hand from outside the game\n",
                         player_name(owner).c_str(), cname.c_str());
                game.take_action();
                break;
            }

            // Play land
            Entity land_entity = action.source_entity;
            auto &zone = global_coordinator.GetComponent<Zone>(land_entity);
            auto &card_data = global_coordinator.GetComponent<CardData>(land_entity);

            // Modal DFC played as its back face (a land): the entity's CardData is the front
            // face, but it enters showing its back face. Reuse the transform machinery — mark it
            // pending_enters_transformed so apply_permanent_components flips it to the back face
            // at entry (suppressing the front-face ETBs). As a modal card it doesn't flip again.
            const CardData *played_face = &card_data;
            if (action.play_back_face && card_data.backside) {
                played_face = card_data.backside.get();
                cur_game.pending_enters_transformed.insert(land_entity);
            }

            // Move to battlefield
            orderer->add_to_zone(false, land_entity, Zone::BATTLEFIELD);
            zone.controller = zone.owner;
            // ForgetOnMoved$ Exile: a land played from exile under a Light Up the Stage play
            // permission consumes that permission as it leaves exile (harmless no-op otherwise).
            cur_game.impulse_cast_permission.erase(land_entity);

            // Permanent component added by apply_permanent_components on next SBA pass

            // Update player's lands played counter
            Entity player_entity = get_player_entity(zone.owner);
            auto &player = global_coordinator.GetComponent<Player>(player_entity);
            player.lands_played_this_turn++;

            game_log("%s played %s\n", player_name(zone.owner).c_str(), played_face->name.c_str());

            // Playing a land uses take_action() (resets pass tracking)
            game.take_action();
            break;
        }

        case ACTIVATE_ABILITY:
            process_activate_ability(action, game, orderer);
            break;

        case CAST_SPELL: {
            Entity spell_entity = action.source_entity;
            auto &zone = global_coordinator.GetComponent<Zone>(spell_entity);
            auto &front_data = global_coordinator.GetComponent<CardData>(spell_entity);
            // Modal DFC cast as its NONLAND back face (CR 712.8): the entity's CardData is the
            // front face, but the spell has only the BACK face's characteristics — pay the back's
            // mana cost, put the back's spell ability on the stack, and (if the back is a
            // permanent) enter as the back face. Source every cast-path read of card_data from the
            // back face for this cast. The land-back case is handled in the SPECIAL_ACTION path.
            const CardData &card_data = (action.cast_back_face && front_data.backside)
                                            ? *front_data.backside : front_data;
            Zone::Ownership caster = zone.owner;

            // If the chosen back face is a permanent, reuse the transform machinery so it enters
            // showing the back face (apply_permanent_components flips it at entry, suppressing the
            // front-face ETBs). Instant/sorcery backs resolve and leave the stack, so no flip is
            // needed and none is marked.
            if (action.cast_back_face && front_data.backside &&
                is_permanent_card(*front_data.backside))
                cur_game.pending_enters_transformed.insert(spell_entity);

            // Record whether this spell is being cast from its caster's own hand (a normal
            // CR 601 hand cast), so a permanent that later resolves onto the battlefield can
            // tell it "was cast from your hand by you" (Amped Raptor's dig gate). One-shot:
            // set here, consumed when the Permanent is created (state_manager_statics). Casts
            // from graveyard/exile (flashback, impulse) clear it so they don't count.
            if (zone.location == Zone::HAND && zone.owner == caster)
                cur_game.cast_from_hand.insert(spell_entity);
            else
                cur_game.cast_from_hand.erase(spell_entity);

            // A fresh delve cast must not inherit exiles recorded by a previous delve spell
            // (delve_exiled persists until an etbCounter replacement consumes it). Cleared
            // BEFORE the mana snapshot so a cancelled payment's rewind (restore_mana_state)
            // returns exactly this cast's exiles to the graveyard.
            if (card_data.has_delve) cur_game.delve_exiled.clear();

            // Snapshot mana state for rewind on payment failure
            auto mana_snap = snapshot_mana_state(caster, orderer);

            // Initialize the persisted cast state machine (Game::PendingCast) from the
            // consumed LegalAction and hand control to run_cast_flow — the extracted
            // CAST_SPELL body. The branch's former locals (cost accumulation, kicker
            // flags, replicate count, deferred payment pieces) live in pc; converted
            // prompts suspend as loop-top pending decisions (tag CAST) that the main
            // loop emits and resume_cast_flow re-enters with the latched answer.
            Game::PendingCast &pc = game.pending_cast;
            if (pc.active) fatal_error("CAST_SPELL with a cast flow already in flight");
            pc = Game::PendingCast{};
            pc.active = true;
            pc.spell_entity = spell_entity;
            pc.caster_is_a = (caster == Zone::PLAYER_A);
            pc.use_flashback = action.use_flashback;
            pc.use_escape = action.use_escape;
            pc.use_alt_cost = action.use_alt_cost;
            pc.use_offspring = action.use_offspring;
            pc.impulse_cast = action.impulse_cast;
            pc.cast_back_face = action.cast_back_face;
            pc.mana_snap = mana_snap;
            run_cast_flow(pc, game, orderer, -1);
            break;
        }
    }
}

// ── T3.10: prompted combat damage assignment among multiple blockers ──────────
// Collect an attacker's live blockers — the same set deal_combat_damage() iterates
// (a blocker killed in the first-strike step has already lost its Creature component).
static std::vector<Entity> collect_live_blockers(Entity attacker, std::shared_ptr<Orderer> orderer) {
    std::vector<Entity> blockers;
    for (auto b : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Creature>(b)) continue;
        auto &bcr = global_coordinator.GetComponent<Creature>(b);
        if (bcr.is_blocking && bcr.blocking_target == attacker) blockers.push_back(b);
    }
    return blockers;
}

// Does this attacker need its controller to choose how to divide combat damage this step?
// Only when it deals damage this step, is blocked by 2+ live blockers, and CANNOT assign lethal
// to all of them (power <= total lethal). When it can kill everything (power > total lethal) the
// choice is immaterial, so deal_combat_damage() auto-assigns instead (the ML simplification).
static bool attacker_needs_assignment(Entity attacker, std::shared_ptr<Orderer> orderer,
                                      bool first_strike_only) {
    if (!global_coordinator.entity_has_component<Creature>(attacker)) return false;
    auto &cr = global_coordinator.GetComponent<Creature>(attacker);
    if (!cr.is_attacking || !cr.is_blocked) return false;
    if (!should_deal_damage(cr, first_strike_only)) return false;
    auto blockers = collect_live_blockers(attacker, orderer);
    if (blockers.size() < 2) return false;
    uint32_t total_lethal = 0;
    for (auto b : blockers) total_lethal += lethal_needed_for_blocker(attacker, b);
    return cr.power <= total_lethal;
}

bool any_attacker_needs_damage_assignment(Game &game, std::shared_ptr<Orderer> orderer,
                                          bool first_strike_only) {
    for (auto entity : orderer->mEntities) {
        // Re-entrancy guard: the handler stores an entry for every attacker it prompts, so an
        // already-decided attacker is skipped and the step falls through to deal_combat_damage.
        if (game.combat_damage_assignment.count(entity)) continue;
        if (attacker_needs_assignment(entity, orderer, first_strike_only)) return true;
    }
    return false;
}

// Build and park the next lethal-order pick for the in-flight attacker
// (Game::pending_damage) as a loop-top pending decision (tag DAMAGE_ASSIGN):
// offer only blockers still killable with the remaining damage, plus the Done
// option — exactly the inner-loop menu the blocking get_input prompted with.
// Prints the same "--- Assign ... ---" header at arm time (it precedes the menu
// emission, as it preceded get_input before). Returns false without arming when
// no blocker is still killable (the inner loop's `offered.empty()` break).
static bool arm_damage_assign_query(Game &game) {
    auto &pd = game.pending_damage;
    std::string attacker_name = entity_name(pd.attacker);
    std::vector<LegalAction> actions;
    for (auto b : pd.pool) {
        uint32_t need = lethal_needed_for_blocker(pd.attacker, b);
        if (need == 0 || need > pd.remaining) continue;
        auto &bcr = global_coordinator.GetComponent<Creature>(b);
        LegalAction la(PASS_PRIORITY, b,
            entity_name(b) + " [" + std::to_string(bcr.power) + "/" +
                std::to_string(bcr.toughness) + "] (lethal " + std::to_string(need) + ")");
        la.category = ActionCategory::ASSIGN_DAMAGE;
        actions.push_back(la);
    }
    if (actions.empty()) return false;  // can't kill any more blockers
    LegalAction done(PASS_PRIORITY, std::string("Done assigning ") + attacker_name);
    done.category = ActionCategory::ASSIGN_DAMAGE;
    actions.push_back(done);

    game_log("\n--- Assign %s's combat damage (%u left) ---\n", attacker_name.c_str(), pd.remaining);
    PendingQuery &pq = game.pending_query;
    pq.tag = PendingQuery::DAMAGE_ASSIGN;
    pq.menu = std::move(actions);
    // The attacking (active) player divides the damage (510.1c); priority was
    // seated at them by run_damage_assignment and stays there between arms.
    pq.chooser_is_a = game.player_a_turn;
    // Byte-compat: this prompt runs with pending_decision_source == 0 today (no
    // PendingDecisionScope wraps it), and the corpus decodes that state field —
    // keep 0 (same convention as the combat target sub-prompts).
    pq.decision_source = 0;
    pq.answered = false;
    pq.answer = -1;
    pq.active = true;
    return true;
}

// Finish the in-flight attacker's division. 510.1a: all the attacker's power
// must be assigned among its blockers (no trample reaches the player in the
// prompt case, since power <= total lethal). Pour any leftover onto one blocker
// — harmless overkill on the last one chosen, or a non-lethal mark on the first
// blocker if none were killable (last_assigned == 0 implies no pick was made,
// so the untouched pool still IS the full blocker list).
static void finish_pending_attacker(Game &game) {
    auto &pd = game.pending_damage;
    if (pd.remaining > 0) {
        Entity dump = pd.last_assigned ? pd.last_assigned : pd.pool.front();
        game.combat_damage_assignment[pd.attacker][dump] += pd.remaining;
    }
    pd = Game::PendingDamageAssign{};
}

// Prompt the attacking player (rule 510.1c) to pick which blockers receive lethal damage, one
// at a time, until power runs out. Records the per-blocker assignment in
// game.combat_damage_assignment for deal_combat_damage() to apply.
//
// Resumable: each pick is parked as a loop-top pending decision (DAMAGE_ASSIGN)
// instead of blocking on get_input, so the whole multi-attacker division spreads
// over several main-loop iterations — arm a query, return; the loop top emits it
// and dispatches the answer back here (resume_choice >= 0), which applies the
// pick exactly as the inline post-get_input code did and arms the next query
// (same attacker, or the next one via the outer scan) or completes. In-flight
// state lives in Game::pending_damage; completed attackers are skipped by their
// combat_damage_assignment map entries, so the outer scan restarts from the top
// on every resume and lands on the first undecided attacker.
static void run_damage_assignment(Game &game, std::shared_ptr<Orderer> orderer, int resume_choice) {
    bool first_strike_only = (game.cur_step == FIRST_STRIKE_DAMAGE);
    // The attacking (active) player chooses the division — route input to them.
    game.player_a_has_priority = game.player_a_turn;
    auto &pd = game.pending_damage;

    if (resume_choice >= 0) {
        // Resume: apply the latched answer to the in-flight attacker's division.
        PendingQuery &pq = game.pending_query;
        bool is_done = (resume_choice == static_cast<int>(pq.menu.size()) - 1);
        Entity chosen = is_done ? 0 : pq.menu[static_cast<size_t>(resume_choice)].source_entity;
        pq = PendingQuery{};
        if (is_done) {
            finish_pending_attacker(game);
        } else {
            uint32_t need = lethal_needed_for_blocker(pd.attacker, chosen);
            game.combat_damage_assignment[pd.attacker][chosen] = need;
            pd.remaining -= need;
            pd.last_assigned = chosen;
            pd.pool.erase(std::remove(pd.pool.begin(), pd.pool.end(), chosen), pd.pool.end());
            game_log("  %s assigns %u (lethal) to %s\n", entity_name(pd.attacker).c_str(), need,
                     entity_name(chosen).c_str());
            // Next pick for the same attacker, or its division is finished.
            if (arm_damage_assign_query(game)) return;
            finish_pending_attacker(game);
        }
    } else {
        // Fresh entry (per strike step; survivors re-decide next step). The clear
        // must NOT run on a resume — mid-assignment the map already holds the
        // completed attackers' divisions (and the in-flight partial one).
        game.combat_damage_assignment.clear();
    }

    // Outer scan: first attacker still needing a division starts one. The map
    // entry (created when the division starts) doubles as the re-entrancy guard,
    // mirroring any_attacker_needs_damage_assignment().
    for (auto attacker : orderer->mEntities) {
        if (game.combat_damage_assignment.count(attacker)) continue;
        if (!attacker_needs_assignment(attacker, orderer, first_strike_only)) continue;
        auto &acr = global_coordinator.GetComponent<Creature>(attacker);
        game.combat_damage_assignment[attacker];  // creates the entry (also the guard)
        pd.active = true;
        pd.attacker = attacker;
        pd.remaining = acr.power;
        pd.pool = collect_live_blockers(attacker, orderer);
        pd.last_assigned = 0;
        if (arm_damage_assign_query(game)) return;
        // No blocker killable even at full power: dump everything, no prompt.
        finish_pending_attacker(game);
    }

    game.pending_choice = NONE;
}

// proc_mandatory_choice entry: a fresh ASSIGN_COMBAT_DAMAGE_CHOICE derivation.
static void assign_combat_damage(Game &game, std::shared_ptr<Orderer> orderer) {
    if (game.pending_damage.active || game.pending_query.active)
        fatal_error("assign_combat_damage entered with a damage-assignment query parked");
    run_damage_assignment(game, orderer, -1);
}

// Loop-top dispatcher entry (game_driver.cpp) for a parked DAMAGE_ASSIGN query.
void resume_damage_assignment(Game &game, std::shared_ptr<Orderer> orderer) {
    if (!game.pending_damage.active || game.pending_query.tag != PendingQuery::DAMAGE_ASSIGN
        || !game.pending_query.answered)
        fatal_error("resume_damage_assignment without a parked damage-assignment query");
    run_damage_assignment(game, orderer, game.pending_query.answer);
}

// Miracle (CR 702.94a) reveal decision. A first-of-turn miracle card was drawn and its owner may
// reveal it "as they draw it" — a PRIVATE special action taken off the stack (the opponent is not
// told a miracle card was drawn unless it is revealed). This forced yes/no is presented to the
// OWNER (who need not hold priority) before they proceed. On decline the card stays hidden in hand.
// On accept the card becomes public (belief state + log) and the linked "when you reveal this card
// this way, you may cast it" triggered ability is synthesized onto the stack (opponent now sees the
// revealed card and gets a response window); that trigger resolves in effect_miracle.cpp by opening
// the miracle-cast window, so the owner makes the actual cast decision at their following priority.
static void proc_miracle_reveal(Game &game, std::shared_ptr<Orderer> orderer) {
    Entity card = game.miracle_reveal_pending;
    game.miracle_reveal_pending = 0;  // consume up front — a single, one-shot decision
    // The card must still be in its owner's hand to be miracle-revealed (nothing runs between the
    // draw and this decision today, but guard against a vanished/moved entity regardless).
    if (!global_coordinator.entity_has_component<Zone>(card)) return;
    auto &z = global_coordinator.GetComponent<Zone>(card);
    if (z.location != Zone::HAND) return;
    Zone::Ownership owner = z.owner;
    if (owner != Zone::PLAYER_A && owner != Zone::PLAYER_B) return;
    const std::string nm = global_coordinator.entity_has_component<CardData>(card)
                               ? global_coordinator.GetComponent<CardData>(card).name
                               : "the card";

    std::vector<LegalAction> yn;
    LegalAction decline(PASS_PRIORITY, std::string("Don't reveal ") + nm + " (miracle)");
    decline.category = ActionCategory::OPTIONAL_YESNO;
    decline.option_ordinal = 0;  // 0 = decline
    yn.push_back(decline);
    LegalAction accept(PASS_PRIORITY, std::string("Reveal ") + nm + " for its miracle cost");
    accept.category = ActionCategory::OPTIONAL_YESNO;
    accept.option_ordinal = 1;  // 1 = accept (reveal)
    yn.push_back(accept);

    // Point the input query at the OWNER (the drawer), who need not be the priority holder — the
    // shared chooser-scope pattern (mirrors CLEANUP_DISCARD): machine mode then serializes the state
    // from the owner's perspective and routes the decision to them, keeping it hidden from the
    // opponent. Loop-safe: one decision derived from the pending card alone.
    bool prev_priority = game.player_a_has_priority;
    game.player_a_has_priority = (owner == Zone::PLAYER_A);
    search_set_loop_safe(true);
    int choice = InputLogger::instance().get_input(yn);
    search_set_loop_safe(false);
    game.player_a_has_priority = prev_priority;

    if (choice != 1) return;  // declined — the card stays hidden in hand, no cast opportunity

    // Revealed (CR 702.94a/702.94b): the card is now public. Record the reveal in the belief state
    // and log it, then put the linked "you may cast it" triggered ability on the stack.
    mark_card_revealed(card, owner);
    game_log("%s reveals %s for its miracle cost.\n", player_name(owner).c_str(), nm.c_str());
    Ability trig;
    trig.ability_type = Ability::TRIGGERED;
    trig.category = "MiracleCast";
    trig.source = card;
    trig.controller = owner;
    orderer->push_ability_onto_stack(trig, owner);
}

// Miracle (CR 702.94a) cast decision. The linked "you may cast it" triggered ability has resolved
// (effect_miracle.cpp armed Game::miracle_cast_pending), so the owner now makes a single immediate
// choice: cast the card for its miracle cost, or do not. Presented only while the card is still in
// the owner's hand; the "Cast" option is offered only when the miracle mana cost is affordable
// (CR 601.2f floor folded in), otherwise ONLY "do not cast" is offered. On "cast" the real cast is
// initiated immediately for the miracle alternate cost through the normal cast machinery
// (process_action -> run_cast_flow), so payment / targets / X behave exactly like any other cast.
static void proc_miracle_cast(Game &game, std::shared_ptr<Orderer> orderer) {
    Entity card = game.miracle_cast_pending;
    game.miracle_cast_pending = 0;  // consume up front — a single, one-shot decision
    if (!global_coordinator.entity_has_component<Zone>(card)) return;
    auto &z = global_coordinator.GetComponent<Zone>(card);
    if (z.location != Zone::HAND) return;  // left hand (e.g. countered reveal) — nothing to cast
    Zone::Ownership owner = z.owner;
    if (owner != Zone::PLAYER_A && owner != Zone::PLAYER_B) return;
    if (!global_coordinator.entity_has_component<CardData>(card)) return;
    const CardData &card_data = global_coordinator.GetComponent<CardData>(card);
    const std::string nm = card_data.name;

    // Affordability of the miracle mana cost, matching can_afford_alt's final mana check so the
    // offer and the actual payment agree.
    ManaValue alt_mana = floored_alt_mana_cost(card_data, card_data.alt_cost.mana_cost, owner);
    bool affordable = alt_mana.empty() || can_pay_mana(owner, alt_mana, card, orderer);

    std::vector<LegalAction> menu;
    LegalAction decline(PASS_PRIORITY, std::string("Do not cast ") + nm + " (miracle)");
    decline.category = ActionCategory::OPTIONAL_YESNO;
    decline.option_ordinal = 0;  // 0 = do not cast
    menu.push_back(decline);
    if (affordable) {
        LegalAction accept(PASS_PRIORITY, card, std::string("Cast ") + nm + " for its miracle cost");
        accept.category = ActionCategory::OPTIONAL_YESNO;
        accept.option_ordinal = 1;  // 1 = cast now for the miracle cost
        menu.push_back(accept);
    }

    // Present to the OWNER (seat repointed, loop-safe — same pattern as cleanup discard / the reveal).
    bool prev_priority = game.player_a_has_priority;
    game.player_a_has_priority = (owner == Zone::PLAYER_A);
    search_set_loop_safe(true);
    int choice = InputLogger::instance().get_input(menu);
    search_set_loop_safe(false);

    if (choice != 1) {
        game.player_a_has_priority = prev_priority;  // declined (or unaffordable) — restore priority
        return;
    }
    // Cast it now for the miracle alternate cost via the normal cast machinery. Priority stays at the
    // owner (the caster) so run_cast_flow's converted prompts (mana payment, targets, X) seat on them.
    LegalAction cast(CAST_SPELL, card, std::string("Cast ") + nm + " (miracle)");
    cast.category = ActionCategory::CAST_SPELL;
    cast.use_alt_cost = true;
    cast.option_ordinal = 1;
    process_action(cast, game, orderer);
}

void proc_mandatory_choice(Game &game, std::shared_ptr<Orderer> orderer) {
    // A pending miracle reveal or cast (CR 702.94) is a forced decision the drawing player makes
    // before proceeding; both ride this channel but are not pending_choice enum values.
    if (game.miracle_reveal_pending != 0) {
        proc_miracle_reveal(game, orderer);
        return;
    }
    if (game.miracle_cast_pending != 0) {
        proc_miracle_cast(game, orderer);
        return;
    }
    switch (game.pending_choice) {
        case DECLARE_ATTACKERS_CHOICE:
            declare_attackers(game, orderer);
            break;
        case DECLARE_BLOCKERS_CHOICE:
            declare_blockers(game, orderer);
            break;
        case ASSIGN_COMBAT_DAMAGE_CHOICE:
            assign_combat_damage(game, orderer);
            break;
        case CLEANUP_DISCARD: {
            Zone::Ownership active_player = game.player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B;
            auto hand = orderer->get_hand(active_player);

            game_log("\n--- Discard to hand size (%s) ---\n", player_name(active_player).c_str());
            game_log("Hand (%zu cards, must discard to 7):\n", hand.size());
            std::vector<LegalAction> discard_actions;
            for (auto card : hand) {
                auto &cd = global_coordinator.GetComponent<CardData>(card);
                LegalAction la(PASS_PRIORITY, card, cd.name);
                la.category = ActionCategory::DISCARD;
                discard_actions.push_back(la);
            }
            // The discarding player is the active player (CR 514.1), who is not
            // necessarily the current priority holder. Point the input query at
            // them (the shared chooser-scope pattern) so machine-mode serializes
            // the state from their perspective and routes the decision to them —
            // otherwise the opponent could be asked to choose the active player's
            // discard.
            bool prev_priority = game.player_a_has_priority;
            game.player_a_has_priority = (active_player == Zone::PLAYER_A);
            // Loop-safe: one discard per proc_mandatory_choice call, menu derived
            // from the hand alone.
            search_set_loop_safe(true);
            int choice = InputLogger::instance().get_input(discard_actions);
            search_set_loop_safe(false);
            game.player_a_has_priority = prev_priority;
            Entity card = discard_actions[static_cast<size_t>(choice)].source_entity;
            auto &cd = global_coordinator.GetComponent<CardData>(card);
            orderer->add_to_zone(false, card, Zone::GRAVEYARD);
            game_log("%s discards %s.\n", player_name(active_player).c_str(), cd.name.c_str());

            game.pending_choice = NONE;
            break;
        }
        case CHOOSE_ENTITY:
            game_log("TODO: Choose entity\n");
            game.pending_choice = NONE;
            break;
        case NONE:
            break;
    }
}
