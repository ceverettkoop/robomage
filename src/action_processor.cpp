#include "action_processor.h"

#include <algorithm>
#include <cstdio>

#include "classes/match_state.h"
#include "choice_labels.h"
#include "cli_output.h"
#include "components/ability.h"
#include "components/carddata.h"
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
#include "game_driver.h"
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
// The largest X a variable life cost (Toxic Deluge: "pay X life") may be announced as (CR 601.2b,
// 119.4): the life left after the cast's other life costs, lowered while the spell's mana could
// only be paid by a painful source spending the life X needs.
static size_t max_life_x(const Game::PendingCast &pc, Zone::Ownership caster, Entity spell_entity,
                         std::shared_ptr<Orderer> orderer);
static void process_activate_ability(const LegalAction &action, Game &game, std::shared_ptr<Orderer> orderer);
static void run_activation_flow(Game::PendingActivation &pa, Game &game,
                                std::shared_ptr<Orderer> orderer, int resume_choice);
// CR 602.2a: a non-mana activated ability (CR 605.3b) is created on the stack as its activation
// is proposed, before any choice or cost; it takes its final Ability (targets, X) once it becomes
// activated (finish_activated_ability).
static void begin_activation(Game::PendingActivation &pa, Zone::Ownership controller,
                             std::shared_ptr<Orderer> orderer);
static void finish_activated_ability(Game::PendingActivation &pa, Zone::Ownership controller,
                                     std::shared_ptr<Orderer> orderer);
// CR 733.1: reverse the activation in flight, at any step before PAY_APPLY applies a cost: the
// ability leaves the stack without a trace, every mana ability the payment activated (and the
// source's tap cost) is reversed (mana_snap), and the chosen-but-unapplied cost items are
// dropped. Clears pa; the player keeps priority (CR 733.2).
static void rewind_activation(Game::PendingActivation &pa, std::shared_ptr<Orderer> orderer);
// The activation's total cost can't be paid (CR 602.2b / 601.2h): count the failure for the
// offer gate's payment_blocked guard and reverse the activation.
static void fail_activation_payment(Game::PendingActivation &pa,
                                    std::shared_ptr<Orderer> orderer);
static std::vector<Entity> build_valid_targets(
    const Ability &ability, std::shared_ptr<Orderer> orderer, Zone::Ownership priority_player);
static void defer_alternate_cost(Game &game, const CardData &card_data, Zone::Ownership caster);
static void declare_attackers(Game &game, std::shared_ptr<Orderer> orderer);
static void park_combat_target_query(Game &game, PendingQuery::Tag tag,
                                     std::vector<LegalAction> &&menu, bool chooser_is_a,
                                     Entity chosen_creature);
static void resume_attack_target(Game &game);
static void resume_block_target(Game &game);
static bool player_controls_land_subtype(Zone::Ownership player, const std::string &subtype,
                                         const std::set<Entity> &entities);
static std::string landwalk_subtype(const std::string &kw);
static bool attacker_unblockable(Entity atk, Zone::Ownership defending_player,
                                 const std::set<Entity> &entities);
static std::vector<Entity> determine_blockable_attackers(Entity blocker, const std::vector<Entity> &attackers);
static void release_illegal_menace_blockers(const std::vector<Entity> &eligible,
                                            const std::vector<Entity> &attackers);
static void declare_blockers(Game &game, std::shared_ptr<Orderer> orderer);
static void finish_blocker_declaration(Game &game);
static bool attacker_needs_assignment(Entity attacker, std::shared_ptr<Orderer> orderer, bool first_strike_only);
static bool arm_damage_assign_query(Game &game);
static void finish_pending_attacker(Game &game);
static void run_damage_assignment(Game &game, std::shared_ptr<Orderer> orderer, int resume_choice);
static void assign_combat_damage(Game &game, std::shared_ptr<Orderer> orderer);
static void proc_miracle_reveal(Game &game, std::shared_ptr<Orderer> orderer);
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
                                     const std::vector<Entity> &targets);
static void fire_became_target_events(Entity targeting_entity, Zone::Ownership controller,
                                      const std::vector<Entity> &targets);
static void append_chosen_targets(const Ability &ab, std::vector<Entity> &out);
static std::vector<Entity> chosen_targets_of(Entity targeting_entity);
static std::vector<LegalAction> escape_exile_menu(Zone::Ownership caster, Entity spell_entity,
                                                  std::shared_ptr<Orderer> orderer);
static int effective_target_min(const Ability &ab, Zone::Ownership perspective,
                                std::shared_ptr<Orderer> orderer, bool x_announced);
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
static void arm_flow_query(Game &game, PendingQuery::Tag tag, std::vector<LegalAction> &&menu,
                           Zone::Ownership chooser, Entity decision_source);
static void arm_cast_query(Game &game, std::vector<LegalAction> &&menu, Zone::Ownership chooser,
                           Entity decision_source);
// Choose graveyard card `e` as one delve exile (CR 702.66a): it pays one GENERIC pip of the
// deferred mana cost now and is exiled — and recorded in cur_game.delve_exiled — at PAY_APPLY.
static void choose_delve_exile(Game::PendingCast &pc, Entity e, Zone::Ownership caster);
// CR 601.2a: the proposal begins. The card moves to the top of the stack as a spell `caster`
// controls (its Spell component names the caster), before any mode, target or cost is chosen.
static void begin_cast(Game::PendingCast &pc, Zone::Ownership caster,
                       std::shared_ptr<Orderer> orderer);
// CR 601.5 / 733.1: reverse the cast in flight, at any step before PAY_APPLY applies a cost. The
// card returns to where it was cast from as the same object, every mana ability the payment
// activated is reversed (mana_snap), the life paid for Phyrexian pips is given back, the
// chosen-but-unapplied cost items are dropped, the cast-time markers are cleared, and no event
// fires. Clears pc; the player keeps priority (CR 733.2).
static void rewind_cast(Game::PendingCast &pc, std::shared_ptr<Orderer> orderer);
// The cast's total cost can't be paid (CR 601.2h): count the failure for the offer gate's
// payment_blocked guard and reverse the cast.
static void fail_cast_payment(Game::PendingCast &pc, std::shared_ptr<Orderer> orderer);
static void run_cast_flow(Game::PendingCast &pc, Game &game, std::shared_ptr<Orderer> orderer,
                          int resume_choice);
static int ask_miracle_choice(Game &game, const std::vector<LegalAction> &menu, Entity card);

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

// Largest X the payer will actually accept. max_available_mana is a deliberately coarse
// upper bound — it counts ONE ability per source at face value — but the payer may take a
// SMALLER ability from the same permanent: with Yavimaya out, Ancient Tomb is also a Forest,
// and the painless {G} outranks its own painful {C}{C}. The bound then overshoots and the
// chosen X fails to pay. Walk it down with the shared simulate-mode predicate (payability is
// monotone in X, so the first accepted value is the maximum), exactly as DELVE_COUNT bounds
// its exile count — offered ladder and eventual payment can never disagree.
// `x_pips` is how many {X} the cost carries (2 for Blast Zone's {X}{X}); `exclude` is the
// ability's source when its own tap is part of the cost.
static size_t payable_max_x(Zone::Ownership payer, const ManaValue &base_cost, size_t coarse_max,
                            size_t x_pips, Entity paid_for, Entity exclude,
                            std::shared_ptr<Orderer> orderer, bool has_delve, bool has_improvise) {
    while (coarse_max > 0) {
        ManaValue trial = base_cost;
        for (size_t i = 0; i < coarse_max * x_pips; i++) trial.insert(GENERIC);
        if (can_pay_mana(payer, trial, paid_for, orderer, has_delve, has_improvise, exclude)) break;
        coarse_max--;
    }
    return coarse_max;
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

// See forward declaration at top of file.
static void choose_delve_exile(Game::PendingCast &pc, Entity e, Zone::Ownership caster) {
    auto git = pc.deferred_mana_cost.find(GENERIC);
    if (git != pc.deferred_mana_cost.end()) pc.deferred_mana_cost.erase(git);
    choose_cost_item(pc, e, Zone::EXILE,
                     player_name(caster) + " exiles " +
                         global_coordinator.GetComponent<CardData>(e).name + " via Delve.");
    pc.cost_removals.back().delve = true;
}

// See forward declaration at top of file.
static void begin_cast(Game::PendingCast &pc, Zone::Ownership caster,
                       std::shared_ptr<Orderer> orderer) {
    pc.x_paid_before = cur_game.x_paid;
    pc.cast_origin = orderer->begin_cast_move(pc.spell_entity, caster);
    // The spell has the characteristics of the face being cast from the moment it is on the
    // stack (CR 601.2a, 712.8f), so active_face reads them throughout the proposal.
    Spell spell;
    spell.caster = caster;
    spell.cast_back_face = pc.cast_back_face &&
                           global_coordinator.GetComponent<CardData>(pc.spell_entity).backside;
    global_coordinator.AddComponent(pc.spell_entity, spell);
}

// See forward declaration at top of file.
static void rewind_cast(Game::PendingCast &pc, std::shared_ptr<Orderer> orderer) {
    Entity spell_entity = pc.spell_entity;
    Zone::Ownership caster = pc.caster_is_a ? Zone::PLAYER_A : Zone::PLAYER_B;
    if (pc.mana_snap_taken) restore_mana_state(caster, pc.mana_snap, orderer);
    if (pc.phyrexian_life_paid > 0) {
        auto &player = global_coordinator.GetComponent<Player>(get_player_entity(caster));
        player.life_total += pc.phyrexian_life_paid;
        player.life_lost_this_turn -= pc.phyrexian_life_paid;
    }
    if (pc.delve_seat_held) cur_game.player_a_has_priority = pc.delve_prev_priority_a;
    if (global_coordinator.entity_has_component<Ability>(spell_entity))
        global_coordinator.RemoveComponent<Ability>(spell_entity);
    if (global_coordinator.entity_has_component<Spell>(spell_entity))
        global_coordinator.RemoveComponent<Spell>(spell_entity);
    orderer->rewind_cast_move(spell_entity, pc.cast_origin);
    cur_game.pending_aura_target.erase(spell_entity);
    cur_game.cast_from_hand.erase(spell_entity);
    cur_game.pending_enters_transformed.erase(spell_entity);
    cur_game.pending_gift_promised = false;
    cur_game.x_paid = pc.x_paid_before;
    pc = Game::PendingCast{};
}

// See forward declaration at top of file.
static void fail_cast_payment(Game::PendingCast &pc, std::shared_ptr<Orderer> orderer) {
    cur_game.payment_fail_counts[pc.spell_entity]++;
    game_log("Payment cancelled.\n");
    rewind_cast(pc, orderer);
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

// See forward declaration at top of file.
static size_t max_life_x(const Game::PendingCast &pc, Zone::Ownership caster, Entity spell_entity,
                         std::shared_ptr<Orderer> orderer) {
    const Player &player = global_coordinator.GetComponent<Player>(get_player_entity(caster));
    int max_x = std::max(0, player.life_total - pc.deferred_life_cost);
    if (!pc.deferred_mana_pending || pc.deferred_mana_cost.empty()) return static_cast<size_t>(max_x);
    while (max_x > 0 &&
           !can_pay_mana(caster, pc.deferred_mana_cost, spell_entity, orderer, pc.deferred_delve,
                         pc.deferred_improvise, /*exclude_entity=*/0,
                         /*life_reserve=*/pc.deferred_life_cost + max_x))
        --max_x;
    return static_cast<size_t>(max_x);
}

// Every chosen target of an ability, joined for the activation announcement. Targets are
// public information as soon as the ability is put on the stack (CR 601.2c), so a
// multi-target activation (e.g. Faerie Macabre's "up to two target cards") must announce
// all of its targets — naming only the first makes the transcript read as if the other
// cards were affected without ever being targeted.
static std::string chosen_targets_display(const Ability &ab) {
    if (ab.targets.empty()) return target_display_name(cur_game, ab.target.lki_entity());
    std::string out;
    for (size_t i = 0; i < ab.targets.size(); i++) {
        if (i > 0) out += (i + 1 == ab.targets.size()) ? " and " : ", ";
        out += target_display_name(cur_game, ab.targets[i].lki_entity());
    }
    return out;
}

// Put an activated ability's stack copy (pa.stack_ab) onto the stack under `controller`, with
// its source and the X announced for it (CR 107.3a; 0 when no X was announced).
// See forward declaration at top of file.
static void begin_activation(Game::PendingActivation &pa, Zone::Ownership controller,
                             std::shared_ptr<Orderer> orderer) {
    pa.x_paid_before = cur_game.x_paid;
    if (ability_is_mana(pa.ability)) return;
    Ability proposed = pa.stack_ab;
    proposed.source = ObjectRef::of(pa.source_entity);
    pa.stack_entity = orderer->push_ability_onto_stack(proposed, controller);
}

// See forward declaration at top of file.
static void finish_activated_ability(Game::PendingActivation &pa, Zone::Ownership controller,
                                     std::shared_ptr<Orderer> orderer) {
    pa.stack_ab.source = ObjectRef::of(pa.source_entity);
    pa.stack_ab.controller = controller;
    if (pa.stack_ab.x_paid < 0) pa.stack_ab.x_paid = 0;
    orderer->set_stack_ability(pa.stack_entity, pa.stack_ab, controller);
}

// See forward declaration at top of file.
static void rewind_activation(Game::PendingActivation &pa, std::shared_ptr<Orderer> orderer) {
    Zone::Ownership controller = pa.activator_is_a ? Zone::PLAYER_A : Zone::PLAYER_B;
    if (pa.mana_snap_taken) restore_mana_state(controller, pa.mana_snap, orderer);
    if (pa.stack_entity != 0) orderer->remove_from_stack(pa.stack_entity, Zone::GRAVEYARD);
    cur_game.x_paid = pa.x_paid_before;
    pa = Game::PendingActivation{};
}

// See forward declaration at top of file.
static void fail_activation_payment(Game::PendingActivation &pa,
                                    std::shared_ptr<Orderer> orderer) {
    cur_game.payment_fail_counts[pa.source_entity]++;
    game_log("Payment cancelled.\n");
    rewind_activation(pa, orderer);
}

static void process_activate_ability(const LegalAction &action, Game &game, std::shared_ptr<Orderer> orderer) {
    Entity permanent_entity = action.source_entity;
    const Ability &ability = action.ability;

    // Initialize the persisted activation state machine (Game::PendingActivation) from the
    // consumed LegalAction and hand control to run_activation_flow — the extracted
    // ACTIVATE_ABILITY body. The branch's former locals (the ability, the targeted
    // stack_ab copy, the chosen X) live in pa; converted prompts suspend as loop-top pending decisions (tag ACTIVATION)
    // that the main loop emits and resume_activation_flow re-enters with the answer.
    Game::PendingActivation &pa = game.pending_activation;
    if (pa.active) fatal_error("ACTIVATE_ABILITY with an activation flow already in flight");

    // ActivationZone$ Hand / Graveyard: card activated from a non-battlefield zone (no Permanent
    // component) — e.g. Cycling/Talon Gates from hand, or Unearth (CR 702.84) from the graveyard.
    // Same flow: select targets, choose and pay the costs, and the ability becomes activated (the
    // ZONE_TARGET step, converging on the shared cost steps).
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
        begin_activation(pa, ctrl, orderer);
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

    // Generic battlefield activation (mana abilities included — they skip the X ladder and
    // target steps inside the flow and resolve off-stack at FINISH).
    pa = Game::PendingActivation{};
    pa.active = true;
    pa.step = Game::PendingActivation::X_LADDER;
    pa.source_entity = permanent_entity;
    pa.activator_is_a = (controller == Zone::PLAYER_A);
    pa.ability = ability;
    pa.stack_ab = ability;  // not used for mana ability
    begin_activation(pa, controller, orderer);
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
            // A spell/ability can't target itself (CR 115.5) — a spell choosing its targets
            // as it is cast (CR 601.2a/c), a modal spell (Pyroblast/Hydroblast) that picks its
            // target at resolution, or an activated ability being activated (CR 602.2a).
            if (e == ability.source.get()) continue;
            if (cur_game.pending_activation.active && e == cur_game.pending_activation.stack_entity)
                continue;
            if (ability.is_legal_target(e, priority_player)) valid_targets.push_back(e);
        }
        return valid_targets;
    }

    Zone::Ownership opp = opponent_of(priority_player);

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

    // Players: opponent first, self second (is_legal_target applies the spec's player clause)
    if (target_spec_names_players(vt)) {
        for (Zone::Ownership seat : {opp, priority_player})
            if (ability.is_legal_target(get_player_entity(seat), priority_player))
                valid_targets.push_back(get_player_entity(seat));
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
    // The chosen attacker / blocker is the pending-decision source: the
    // observation names the creature whose attack / block target is being picked.
    pq.decision_source = chosen_creature;
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
    cr.attack_target = ObjectRef::of(pq.menu[static_cast<size_t>(pq.answer)].source_entity);
    game_log("%s attacking %s.\n", entity_name(chosen_attacker).c_str(),
        target_display_name(game, cr.attack_target.lki_entity()).c_str());
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
    const Entity attacker = pq.menu[static_cast<size_t>(pq.answer)].source_entity;
    cr.blocking_target = ObjectRef::of(attacker);
    // Mark the attacker as blocked. It stays blocked for the rest of combat even if this
    // (and every other) blocker later leaves combat (509.1h), so it assigns no damage to
    // the player unless it has trample.
    if (global_coordinator.entity_has_component<Creature>(attacker))
        global_coordinator.GetComponent<Creature>(attacker).is_blocked = true;
    game_log("%s blocking %s.\n", entity_name(chosen).c_str(), entity_name(attacker).c_str());
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
    Zone::Ownership active_player = active_seat();
    Entity defending_entity = get_player_entity(opponent_of(active_player));
    if (game.pending_attacker != 0)
        fatal_error("declare_attackers entered with an attack-target sub-prompt parked");

    // Collect eligible attackers with stable indices
    std::vector<Entity> eligible;
    for (auto entity : orderer->mEntities) {
        if (!is_battlefield_permanent(entity, active_player)) continue;
        if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
        if (global_coordinator.GetComponent<Permanent>(entity).is_tapped) continue;
        if (is_summoning_sick(entity)) continue;
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
    Zone::Ownership defending_owner = opponent_of(active_seat());
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
        cr.attack_target = ObjectRef::of(defending_entity);
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
                target_display_name(game, cr.attack_target.lki_entity()).c_str());
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
                Zone::Ownership t = seat_of_player(t_entity);
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
        cr.attack_target = ObjectRef::of(tgt_actions[0].source_entity);
        game_log("%s attacking %s.\n", chosen_name.c_str(),
            target_display_name(game, cr.attack_target.lki_entity()).c_str());
    }

    game_log("\nAttackers declared:\n");
    Entity actrl_entity = get_player_entity(active_player);
    int attacker_count = 0;
    Entity sole_attacker = 0;
    for (auto entity : eligible) {
        auto &cr = global_coordinator.GetComponent<Creature>(entity);
        if (!cr.is_attacking) continue;
        attacker_count++;
        sole_attacker = entity;
        game_log("  %s -> %s\n", entity_name(entity).c_str(),
                 target_display_name(game, cr.attack_target.lki_entity()).c_str());

        // Tap the attacker, unless it has vigilance (702.21).
        if (!creature_has_keyword(cr, "Vigilance"))
            global_coordinator.GetComponent<Permanent>(entity).is_tapped = true;

        // Fire a per-attacker "whenever this creature attacks" event (508.2 attack
        // declaration), so triggers like Mobilize go on the stack for each attacker.
        Event attacked_ev(Events::CREATURE_ATTACKED);
        attacked_ev.SetParam(Params::ENTITY, entity);
        attacked_ev.SetParam(Params::PLAYER, actrl_entity);
        global_coordinator.SendEvent(attacked_ev);
    }
    if (attacker_count == 0) game_log("  (none)\n");

    // "Whenever you attack" (Mode$ AttackersDeclared) — a player-level trigger that fires once
    // when one or more attackers are declared (508.2), independent of how many. Guide of Souls.
    if (attacker_count > 0) {
        Event declared_ev(Events::ATTACKERS_DECLARED);
        declared_ev.SetParam(Params::PLAYER, actrl_entity);
        global_coordinator.SendEvent(declared_ev);
    }

    // Exalted: if exactly one creature is attacking, fire the event so triggers go on the stack
    if (attacker_count == 1) {
        Event exalted_ev(Events::CREATURE_ATTACKED_ALONE);
        exalted_ev.SetParam(Params::ENTITY, sole_attacker);
        exalted_ev.SetParam(Params::PLAYER, actrl_entity);
        global_coordinator.SendEvent(exalted_ev);
    }

    game.attackers_declared = true;
    game.pending_choice = NONE;
}

static bool player_controls_land_subtype(Zone::Ownership player, const std::string &subtype,
                                         const std::set<Entity> &entities) {
    for (auto e : battlefield_permanents(entities, player))
        for (auto &t : global_coordinator.GetComponent<Permanent>(e).types)
            if (t.kind == SUBTYPE && t.name == subtype) return true;
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

// True if no creature `defending_player` controls can block `atk` (CR 509.1b): it can't be
// blocked this turn (Kappa Cannoneer), or it has landwalk and the defending player controls a
// land of that type (CR 702.14c).
static bool attacker_unblockable(Entity atk, Zone::Ownership defending_player,
                                 const std::set<Entity> &entities) {
    auto &acr = global_coordinator.GetComponent<Creature>(atk);
    if (acr.cant_be_blocked_this_turn) return true;
    for (auto &kw : acr.keywords) {
        std::string subtype = landwalk_subtype(kw);
        if (!subtype.empty() && player_controls_land_subtype(defending_player, subtype, entities))
            return true;
    }
    return false;
}

// The attackers among `attackers` (already filtered by attacker_unblockable) that `blocker`
// itself may block: flying/reach, shadow and protection are checked per blocker.
static std::vector<Entity> determine_blockable_attackers(Entity blocker, const std::vector<Entity> &attackers) {
    auto &bcr = global_coordinator.GetComponent<Creature>(blocker);
    bool blocker_can_fly = false;
    bool blocker_has_shadow = false;
    for (auto &kw : bcr.keywords) {
        if (kw == "Flying" || kw == "Reach") blocker_can_fly = true;
        if (kw == "Shadow") blocker_has_shadow = true;
    }

    std::vector<Entity> result;
    for (auto atk : attackers) {
        auto &acr = global_coordinator.GetComponent<Creature>(atk);
        bool atk_flying = creature_has_keyword(acr, "Flying");
        bool atk_has_shadow = creature_has_keyword(acr, "Shadow");

        // Shadow: creatures with shadow can only be blocked by shadow creatures,
        // and creatures without shadow cannot be blocked by shadow creatures (rule 702.28)
        if (atk_has_shadow != blocker_has_shadow) continue;

        if (atk_flying && !blocker_can_fly) continue;
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
            if (bcr.is_blocking && bcr.blocking_target.get() == atk) blockers.push_back(b);
        }
        if (blockers.size() == 1) {
            auto &bcr = global_coordinator.GetComponent<Creature>(blockers[0]);
            bcr.is_blocking = false;
            bcr.blocking_target = ObjectRef{};
            acr.is_blocked = false;  // no other blocker assigned this attacker
            game_log("%s cannot block %s alone (menace) — block released.\n",
                     entity_name(blockers[0]).c_str(), entity_name(atk).c_str());
        }
    }
}

static void declare_blockers(Game &game, std::shared_ptr<Orderer> orderer) {
    Zone::Ownership defending_player = opponent_of(active_seat());
    // defending player declares blockers — priority must be theirs for the input routing to work correctly
    game.player_a_has_priority = !game.player_a_turn;
    if (game.pending_blocker != 0)
        fatal_error("declare_blockers entered with a block-target sub-prompt parked");

    // Collect attackers, and the ones the defending player can block at all
    std::vector<Entity> attackers;
    std::vector<Entity> blockable;
    for (auto entity : orderer->mEntities) {
        if (!is_attacking_creature(entity)) continue;
        attackers.push_back(entity);
        if (!attacker_unblockable(entity, defending_player, orderer->mEntities))
            blockable.push_back(entity);
    }

    if (attackers.empty()) {
        game_log("No attackers — skipping declare blockers.\n");
        finish_blocker_declaration(game);
        return;
    }

    // Collect eligible blockers: defending player's untapped creatures that can block some attacker
    // (e.g. not non-flyers vs all-flying attackers)
    std::vector<Entity> eligible;
    for (auto entity : orderer->mEntities) {
        if (!is_battlefield_permanent(entity, defending_player)) continue;
        if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
        if (global_coordinator.GetComponent<Permanent>(entity).is_tapped) continue;
        if (determine_blockable_attackers(entity, blockable).empty()) continue;
        eligible.push_back(entity);
    }

    if (eligible.empty()) {
        game_log("No creatures eligible to block.\n");
        finish_blocker_declaration(game);
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
            std::string atk_name = entity_name(cr.blocking_target.lki_entity());
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

        auto legal_attackers = determine_blockable_attackers(chosen, blockable);
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
            game_log("  %s blocking %s\n", entity_name(entity).c_str(),
                     entity_name(cr.blocking_target.lki_entity()).c_str());
        }
    }
    if (!any) game_log("  (none)\n");

    finish_blocker_declaration(game);
}

// Close the declare-blockers turn-based action (CR 509.1). declare_blockers seats
// the defending player for the block prompts; once the declaration is complete the
// active player receives priority (CR 117.3a). Pass flags were reset when the step
// began and the declaration passes no priority, so they are left as they are.
static void finish_blocker_declaration(Game &game) {
    game.blockers_declared = true;
    game.pending_choice = NONE;
    game.player_a_has_priority = game.player_a_turn;
}

// Perspective player for an ability's target search. Ownership-restricted targets
// (.YouOwn/.YouCtrl/.OppOwn — e.g. Emry's "target artifact card in YOUR graveyard") are
// relative to the activating/controlling player, so the existence check must use that player
// rather than a hardcoded placeholder. Derive it from the ability's source: a battlefield
// permanent's controller, else its owning zone, else the ability's stored controller.
static Zone::Ownership ability_perspective_player(const Ability &ability) {
    Entity src = ability.source.get();
    if (src != 0) {
        if (global_coordinator.entity_has_component<Permanent>(src))
            return global_coordinator.GetComponent<Permanent>(src).controller;
        if (global_coordinator.entity_has_component<Zone>(src))
            return global_coordinator.GetComponent<Zone>(src).owner;
    }
    return ability.controller;
}

Ability enchant_target_ability(Entity aura, const CardData &cd, Zone::Ownership chooser) {
    Ability enchant_ab;
    enchant_ab.source = ObjectRef::of(aura);
    enchant_ab.controller = chooser;
    enchant_ab.valid_tgts = cd.enchant_filter;
    // "Enchant creature card in a graveyard" (Animate Dead): the legal objects are graveyard
    // cards, not battlefield permanents (CR 303.4).
    enchant_ab.target_in_graveyard = enchant_targets_graveyard(cd.enchant_filter);
    return enchant_ab;
}

bool pending_aura_target_legal(Entity aura, Zone::Ownership controller) {
    auto pat = cur_game.pending_aura_target.find(aura);
    if (pat == cur_game.pending_aura_target.end()) return false;
    if (!global_coordinator.entity_has_component<CardData>(aura)) return false;
    Entity tgt = pat->second.target.get();
    if (tgt == 0) return false;
    const auto &cd = global_coordinator.GetComponent<CardData>(aura);
    return enchant_target_ability(aura, cd, controller).is_legal_target(tgt, controller);
}

bool has_legal_targets(const Ability &ability, std::shared_ptr<Orderer> orderer) {
    if (ability.valid_tgts == "N_A") return true;
    // Ordering doesn't affect existence for symmetric targets, but ownership-restricted
    // targets must be evaluated from the controlling player's perspective (see above), or a
    // ".YouOwn" ability could be offered with no legal target and crash on an empty target menu.
    Zone::Ownership perspective = ability_perspective_player(ability);
    // optional targeting always has "legal targets"
    if (effective_target_min(ability, perspective, orderer, /*x_announced=*/false) <= 0) return true;
    return !build_valid_targets(ability, orderer, perspective).empty();
}

// The minimum number of targets `ab` requires (CR 601.2c), the one rule behind the cast- and
// activation-legality gates (has_legal_targets), the charm-mode filter and target selection. A
// static TargetMin$ is its literal value. A non-xPaid count-SVar min (Into the Flood Maw:
// TargetMin$ X = Count$PromisedGift.0.1) is evaluated now against the current game state (which
// reads the pending gift-promise flag).
// An xPaid-driven min ("exactly X targets", Hide on the Ceiling; "up to X", Kozilek's Command)
// reads the X announced for the spell when `x_announced`; before X is chosen (the cast-legality
// gate) it counts as 0 — X may legally be 0, so it must not gate castability.
static int effective_target_min(const Ability &ab, Zone::Ownership perspective,
                                std::shared_ptr<Orderer> orderer, bool x_announced) {
    if (ab.target_min_from_xpaid) return x_announced ? static_cast<int>(cur_game.x_paid) : 0;
    if (!ab.target_min_count_expr.empty())
        return static_cast<int>(evaluate_dynamic_amount(ab.target_min_count_expr, perspective,
                                                        orderer, 0, ab.source.lki_entity()));
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
        if (effective_target_min(*ab, caster, orderer, false) > 0 &&
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
    // Mirrors charm_mode_choosable, the filter the CHARM_MODE cast step applies (X isn't
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
                effective_target_min(probe, caster, orderer, false) <= 0 ||
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
    probe.source = ObjectRef::of(card_entity);
    probe.controller = caster;
    // Chained targeting sub-abilities (Into the Flood Maw's DBChangeZone, Cabal Therapy's
    // DB$ Discard) pick their own target as the spell is cast (CR 601.2c) and are probed via
    // spell_targeting_abilities, so they need the same source/controller. Charm modes are stamped
    // by spell_has_castable_targets from primary.source, so they inherit the stamp set here.
    for (auto &sub : probe.subabilities) {
        sub.source = probe.source;
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
            probe.source = ObjectRef::of(spell_entity);
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
        std::string src_name = !ability.source.empty() ? entity_name(ability.source.lki_entity())
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
        std::string src_name = !ability.source.empty() ? entity_name(ability.source.lki_entity())
                                                       : std::string("(unknown source)");
        fatal_error("Zero legal targets when choosing a required target for " + src_name +
                    " (ValidTgts$ " + ability.valid_tgts + ") — a targeted spell/ability with no "
                    "legal target was offered/forced (CR 601.2c violated upstream).");
    }
    int choice = asker.ask(tgt_actions, ability.source.lki_entity());
    if (choice < 0 && decision_suspended()) return choice;
    // Each pick names the object as it is now (CR 400.7); resolution re-checks it (CR 608.2b).
    ability.target = ObjectRef::of(tgt_actions[static_cast<size_t>(choice)].source_entity);
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
                ability.target_max_count_expr, priority_player, orderer, 0, ability.source.lki_entity()));
        int effective_min = effective_target_min(ability, priority_player, orderer, true);
        ability.target_min = effective_min;
        ability.target_max = effective_max;

        // Zero targets (Into the Flood Maw's unused mode when the gift promise switched the count
        // to 0): the ability targets nothing and does nothing on resolution. Choose no target.
        if (effective_max <= 0) {
            ability.target = ObjectRef{};
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
        for (const ObjectRef &chosen : ability.targets)
            valid_targets.erase(
                std::remove(valid_targets.begin(), valid_targets.end(), chosen.get()),
                valid_targets.end());
        if (valid_targets.empty()) break;
        bool can_stop = (i >= rt.effective_min);
        if (select_single_target(ability, valid_targets, can_stop, asker) < 0)
            return TargetStatus::SUSPENDED;
        if (ability.target.empty()) break;  // chose "Done" or "No target"
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
// when it REQUIRES a target and none exists. The required minimum is select_target's
// (effective_target_min, with the X already announced — X is chosen before modes).
static bool charm_mode_choosable(Ability &candidate, std::shared_ptr<Orderer> orderer,
                                 Zone::Ownership caster) {
    if (candidate.valid_tgts == "N_A") return true;
    if (effective_target_min(candidate, caster, orderer, true) <= 0) return true;
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
// resume. Used by the run_cast_flow CHARM_MODE step.
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
        // Ground every mode to the charm's source card so the serialized action
        // carries that card's id/zone/controller instead of the null-source
        // sentinel — otherwise each mode emits card_id -1 and the modes differ
        // only by the raw option_ordinal scalar, which reads as "all modes
        // identical" to the policy/search. The distinct option_ordinal still
        // separates the modes from one another.
        LegalAction la(PASS_PRIORITY, ability.source.lki_entity(), charm_mode_desc(ability, i));
        la.category = ActionCategory::CHOOSE_MODE;
        la.option_ordinal = static_cast<int>(i);  // mode index (into charm_choices)
        mode_actions.push_back(la);
        mode_indices.push_back(i);
    }
    return mode_actions;
}

// Ward (CR 702.21): "Whenever this permanent becomes the target of a spell or ability an
// opponent controls, counter that spell or ability unless that player pays {N}." Called right
// after a spell/ability with chosen targets is put on the stack. For each target that is a
// battlefield permanent with a Ward cost, controlled by an opponent of the targeting object's
// controller, push a Ward trigger onto the stack ABOVE the targeting object (so it resolves
// first). The Ward trigger is a Counter ability whose unless_generic_cost is the ward cost —
// reusing the existing "counter unless pay {N}" resolution. `targets` holds each object the
// spell/ability targets once (chosen_targets_of), so a permanent chosen by several of its
// "target" instances (two modes, a mode and a sub-ability) became its target once and fires
// each of its Ward abilities once.
// Collect every Ward ability a permanent currently HAS (CR 702.21), honoring ward that is
// granted by a continuous effect (equipment/aura statics, Pump grants, keyword counters), not
// just the printed ward. Two storage forms, kept distinct so they are not double-counted:
//   - Printed ward: parse.cpp stores the numeric cost in CardData::ward_cost (with
//     ward_is_life) and pushes the BARE string "Ward" onto CardData::keywords.
//   - Granted ward: add_keywords_from_spec pushes the raw spec part "Ward:N" or
//     "Ward:PayLife<N>" (with a colon and cost arg) onto the effective keyword list — never
//     the bare "Ward".
// We therefore take the printed instance from ward_cost, and every granted instance from a
// "Ward:..." keyword string. Each instance functions independently (CR 113.2c), so two
// Lavaspur Boots on one creature give two Ward {1} triggers; the per-pass keyword rebuild lists
// each static grant once per granting source. A permanent whose abilities are removed in layer 6
// (Humility, CR 613.1f — Permanent::abilities_removed) has no printed ward; the rebuilt keyword
// list already omits the grants the removal erased.
static std::vector<WardInstance> collect_ward_instances(Entity e) {
    std::vector<WardInstance> wards;
    bool abilities_removed = global_coordinator.entity_has_component<Permanent>(e) &&
                             global_coordinator.GetComponent<Permanent>(e).abilities_removed;
    // Printed ward.
    if (!abilities_removed && global_coordinator.entity_has_component<CardData>(e)) {
        const auto &cd = active_face(e, global_coordinator.GetComponent<CardData>(e));
        if (cd.ward_cost > 0) wards.push_back({cd.ward_cost, cd.ward_is_life});
    }
    // Granted ward(s) from the effective keyword list. Use the same effective-keyword view as
    // permanent_has_keyword: a creature's rebuilt Creature::keywords, else printed keywords.
    const std::vector<std::string> *kw_list = nullptr;
    if (global_coordinator.entity_has_component<Creature>(e))
        kw_list = &global_coordinator.GetComponent<Creature>(e).keywords;
    else if (global_coordinator.entity_has_component<CardData>(e))
        kw_list = &active_face(e, global_coordinator.GetComponent<CardData>(e)).keywords;
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
            wards.push_back(inst);
        }
    }
    return wards;
}

static void trigger_ward_for_targets(Entity targeting_entity, Zone::Ownership controller,
                                     const std::vector<Entity> &targets) {
    Zone::Ownership opp = opponent_of(controller);
    for (Entity tgt : targets) {
        if (tgt == 0) continue;
        // The Ward permanent must be controlled by an opponent of the targeting player.
        if (!is_battlefield_permanent(tgt, opp)) continue;

        std::vector<WardInstance> wards = collect_ward_instances(tgt);
        std::string nm = entity_name(tgt);
        for (const WardInstance &w : wards) {
            if (w.cost <= 0) continue;
            Ability ward;
            ward.ability_type = Ability::TRIGGERED;
            ward.category = "Counter";
            ward.source = ObjectRef::of(tgt);
            ward.controller = opp;            // the Ward permanent's controller
            ward.target = ObjectRef::of(targeting_entity);  // counter the spell/ability that targeted it
            ward.unless_generic_cost = static_cast<size_t>(w.cost);
            ward.unless_cost_is_life = w.is_life;  // Ward—Pay N life pays life, not mana

            // Ward is a triggered ability: it goes on the stack with everything else that
            // triggered before a player next receives priority (CR 603.3b), so a cast trigger
            // (prowess) is ordered with it in APNAP order.
            char line[256];
            snprintf(line, sizeof line,
                     "Ward %s%d%s: %s's controller may pay to counter the spell or ability "
                     "targeting %s", w.is_life ? "—Pay " : "{", w.cost, w.is_life ? " life" : "}",
                     nm.c_str(), nm.c_str());
            cur_game.queue_trigger(ward, line);
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
// One event per (targeting object, targeted permanent) pair — `targets` is de-duplicated.
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

// Append the targets chosen for one targeting instance of `ab` (CR 115.1) — the ability itself
// when it targets, every chosen mode (CR 700.2), and every chained sub-ability (an Execute$ body
// is a separate reflexive/delayed ability, targeted when it triggers) — to `out`,
// skipping an object already listed. A non-targeting ability (ValidTgts$ absent, "N_A") adds
// nothing of its own even if its `target` field carries a bound reference.
static void append_chosen_targets(const Ability &ab, std::vector<Entity> &out) {
    if (ab.valid_tgts != "N_A") {
        std::vector<Entity> mine =
            ab.targets.empty() ? std::vector<Entity>{ab.target.get()} : live_entities(ab.targets);
        for (Entity t : mine)
            if (t != 0 && std::find(out.begin(), out.end(), t) == out.end()) out.push_back(t);
    }
    for (int ci : ab.charm_chosen)
        if (ci >= 0 && static_cast<size_t>(ci) < ab.charm_choices.size())
            append_chosen_targets(ab.charm_choices[static_cast<size_t>(ci)], out);
    for (const Ability &sub : ab.subabilities)
        if (!sub.from_delayed_execute) append_chosen_targets(sub, out);  // not a later trigger's
}

// Every object or player the stack object `targeting_entity` targets, each listed once: the
// targets of its Ability (all modes and sub-abilities), and — for an Aura spell — the object
// its enchant ability targets (CR 115.1b, recorded in Game::pending_aura_target at cast).
static std::vector<Entity> chosen_targets_of(Entity targeting_entity) {
    std::vector<Entity> out;
    if (global_coordinator.entity_has_component<Ability>(targeting_entity))
        append_chosen_targets(global_coordinator.GetComponent<Ability>(targeting_entity), out);
    auto pat = cur_game.pending_aura_target.find(targeting_entity);
    const Entity aura_tgt =
        pat != cur_game.pending_aura_target.end() ? pat->second.target.get() : Entity{0};
    if (aura_tgt != 0 && std::find(out.begin(), out.end(), aura_tgt) == out.end())
        out.push_back(aura_tgt);
    return out;
}

void fire_targeting_hooks(Entity targeting_entity, Zone::Ownership controller) {
    std::vector<Entity> tgts = chosen_targets_of(targeting_entity);
    if (tgts.empty()) return;
    trigger_ward_for_targets(targeting_entity, controller, tgts);
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

// Park a cast-time prompt (tag CAST) for the main loop to emit. Priority
// already sits with the caster at every cast-time prompt (the CAST_SPELL
// action was chosen at the caster's own priority window and nothing repoints
// before these prompts — request_optional_yesno's repoint was a no-op here),
// so like the combat sub-prompts nothing needs saving or restoring on resume.
// Every armed prompt names its asking card as the pending-decision source: the
// cast flow's prompts (cost choices, X ladders, announce-stage modes and
// targets) pass the spell being cast, the activation flow's pass the activated
// card.
static void arm_flow_query(Game &game, PendingQuery::Tag tag, std::vector<LegalAction> &&menu,
                           Zone::Ownership chooser, Entity decision_source) {
    PendingQuery &pq = game.pending_query;
    pq.tag = tag;
    pq.menu = std::move(menu);
    if (offer_cast_cancel) {
        LegalAction cancel(PASS_PRIORITY, "Cancel");
        cancel.category = ActionCategory::PAYING_COSTS;
        cancel.cancel_proposal = true;
        pq.menu.push_back(cancel);
    }
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
// the blocking select_target held), or `source_override` when set (the
// activation flow's activated card). Priority already sits with the caster/
// activator at every such prompt, so the asker never repoints the seat (the
// TargetAsker contract: the caller seats the chooser). The latched answer
// travels through the flow's resume_choice, consumed by the first ask the
// re-entered step reaches — which is exactly the ask that armed it, since
// arming always returns out of the flow and nothing runs in between.
class FlowTargetAsker final : public TargetAsker {
    public:
        FlowTargetAsker(Game &game, Zone::Ownership chooser, int &resume_choice,
                        PendingQuery::Tag tag = PendingQuery::CAST, Entity source_override = 0)
            : game(game), chooser(chooser), resume_choice(resume_choice), tag(tag),
              source_override(source_override) {}
        int ask(const std::vector<LegalAction> &menu, Entity decision_source) override {
            if (resume_choice >= 0) {
                int choice = resume_choice;
                resume_choice = -1;
                return choice;
            }
            if (source_override != 0) decision_source = source_override;
            arm_flow_query(game, tag, std::vector<LegalAction>(menu), chooser, decision_source);
            return -1;
        }
        bool resuming() const override { return resume_choice >= 0; }

    private:
        Game &game;
        Zone::Ownership chooser;
        int &resume_choice;
        PendingQuery::Tag tag;
        Entity source_override;  // nonzero: the pending-decision source for every ask
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
    bool cancel = pq.menu[static_cast<size_t>(answer)].cancel_proposal;
    pq = PendingQuery{};
    if (cancel) {
        game_log("Casting cancelled.\n");
        rewind_cast(game.pending_cast, orderer);
        return;
    }
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
    bool cancel = pq.menu[static_cast<size_t>(answer)].cancel_proposal;
    pq = PendingQuery{};
    if (cancel) {
        game_log("Activation cancelled.\n");
        rewind_activation(game.pending_activation, orderer);
        return;
    }
    run_activation_flow(game.pending_activation, game, orderer, answer);
}

// Drive the persisted activation to its next prompt, cancellation, or
// completion. resume_choice < 0 = fresh entry from process_activate_ability;
// >= 0 = the latched answer for the prompt pa.step armed. A step distinguishes
// arm from apply by whether a latched answer is pending (the arm always returns,
// so re-entry lands on the very step that armed it, and its arm-time gates/menus
// re-derive identically — nothing runs between arm and resume). The interactive
// mana payment is the only prompt left blocking (a deliberate non-conversion —
// machine mode auto-pays with zero decisions), so every payment step is
// synchronous here; a failed payment reverses the activation (rewind_activation).
//
// Every prompt of the flow (the X ladders, the target selections, and the
// sacrifice/return cost picks, including ninjutsu's return) arms with the activated card (pa.source_entity) as the pending-decision
// source — also for a hand/graveyard-activated ability, whose template
// ability.source is not yet bound to the card.
static void run_activation_flow(Game::PendingActivation &pa, Game &game,
                                std::shared_ptr<Orderer> orderer, int resume_choice) {
    Entity permanent_entity = pa.source_entity;
    const Ability &ability = pa.ability;
    Zone::Ownership controller = pa.activator_is_a ? Zone::PLAYER_A : Zone::PLAYER_B;
    // InstantSpeed$ AddMana abilities (e.g. LED) are mana abilities too: they resolve off-stack.
    // The instant-speed timing restriction is enforced upstream (offered only at priority).
    bool is_mana_ability = ability_is_mana(ability);

    for (;;) switch (pa.step) {
        case Game::PendingActivation::ZONE_TARGET: {
            // Select targets before paying costs
            if (pa.stack_ab.valid_tgts != "N_A") {
                FlowTargetAsker asker(game, controller, resume_choice, PendingQuery::ACTIVATION,
                                      permanent_entity);
                if (run_target_select(pa.stack_ab, pa.tsel, asker, orderer, controller) !=
                    TargetStatus::DONE)
                    return;
            }
            pa.step = Game::PendingActivation::COST_SAC;
            break;
        }

        case Game::PendingActivation::X_LADDER: {
            // X ACTIVATION COST (Candelabra of Tawnos: Cost$ X T): X is part of the activation
            // cost, chosen during announcement BEFORE targets (CR 602.2b/601.2b) so an
            // exactly-X / up-to-X target count can read it. Prompt for X (bounded by the mana
            // available beyond the rest of the cost), record it as x_paid, and add X generic
            // to the mana cost paid at PAY. Arms with the activated card as the
            // pending-decision source.
            if (!is_mana_ability && ability.activation_has_x) {
                if (resume_choice >= 0) {
                    pa.x_activation = static_cast<size_t>(resume_choice);
                    resume_choice = -1;
                    cur_game.x_paid = pa.x_activation;
                    pa.stack_ab.x_paid = static_cast<int>(pa.x_activation);
                    game_log("%s chooses X = %zu\n", player_name(controller).c_str(),
                             pa.x_activation);
                } else {
                    ManaValue base_cost = effective_activation_mana_cost(ability, controller, orderer);
                    // The source's own tap is part of this ability's cost, so its mana
                    // ability can't also fund X (Blast Zone offered X=3 off 2 real mana by
                    // counting its own {C}). Each {X} in the cost is paid separately, so a
                    // {X}{X} ability can only afford half the budget (CR 601.2b).
                    Entity x_exclude = ability.tap_cost ? permanent_entity : 0;
                    size_t max_x = max_available_mana(controller, base_cost, orderer, x_exclude);
                    size_t x_pips = std::max<size_t>(1, ability.activation_x_count);
                    max_x /= x_pips;
                    max_x = payable_max_x(controller, base_cost, max_x, x_pips, permanent_entity,
                                          x_exclude, orderer, /*has_delve=*/false,
                                          /*has_improvise=*/false);
                    game_log("Choose X value (0-%zu):\n", max_x);
                    std::vector<LegalAction> x_actions;
                    for (size_t xv = 0; xv <= max_x; xv++) {
                        LegalAction la(PASS_PRIORITY, std::string("X = " + std::to_string(xv)));
                        la.category = ActionCategory::CHOOSE_X;
                        la.option_ordinal = static_cast<int>(xv);  // the chosen X value
                        x_actions.push_back(la);
                    }
                    arm_flow_query(game, PendingQuery::ACTIVATION, std::move(x_actions),
                                   controller, permanent_entity);
                    return;
                }
            }
            pa.step = Game::PendingActivation::LOYALTY_X;
            break;
        }

        case Game::PendingActivation::LOYALTY_X: {
            // X LOYALTY COST (Chandra, Flamecaller's [-X]): choose X at announcement, bounded
            // by the planeswalker's current loyalty for a minus cost (you can't remove more
            // than it has, 606.5). Recorded as x_paid, so the loyalty cost (PAY_APPLY) and the
            // effect's Count$xPaid (NumDmg$ X) both read it. This is a loyalty cost, not mana —
            // it never touches the PAY cost.
            if (ability.loyalty_cost_is_x) {
                if (resume_choice >= 0) {
                    int x_choice = resume_choice;
                    resume_choice = -1;
                    cur_game.x_paid = static_cast<size_t>(x_choice);
                    pa.stack_ab.x_paid = x_choice;
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
                                   controller, permanent_entity);
                    return;
                }
            }
            pa.step = Game::PendingActivation::TARGET;
            break;
        }

        case Game::PendingActivation::TARGET: {
            // SELECT TARGETS BEFORE PAYING COSTS
            if (!is_mana_ability && pa.stack_ab.valid_tgts != "N_A") {
                FlowTargetAsker asker(game, controller, resume_choice, PendingQuery::ACTIVATION,
                                      permanent_entity);
                if (run_target_select(pa.stack_ab, pa.tsel, asker, orderer, controller) !=
                    TargetStatus::DONE)
                    return;
            }
            pa.step = Game::PendingActivation::COST_SAC;
            break;
        }

        case Game::PendingActivation::COST_SAC: {
            // Type-based sacrifice cost (Cycling "Sac a land", Knight of the Reliquary): CHOOSE
            // the permanent now; it is sacrificed at PAY_APPLY, once the mana committed. The
            // pool re-derives identically at apply — nothing runs between arm and resume.
            if (!ability.sac_cost_spec.empty()) {
                std::vector<Entity> choices = controlled_permanents_matching(
                    controller, ability.sac_cost_spec, orderer->mEntities, permanent_entity);
                if (!choices.empty()) {
                    if (resume_choice < 0) {
                        arm_flow_query(game, PendingQuery::ACTIVATION,
                                       permanent_choice_menu(choices, "Sacrifice ", "",
                                                             ActionCategory::SACRIFICE_PERMANENT),
                                       controller, permanent_entity);
                        return;
                    }
                    pa.sac_choice = choices[static_cast<size_t>(resume_choice)];
                    resume_choice = -1;
                }
            }
            pa.step = Game::PendingActivation::COST_RETURN;
            break;
        }

        case Game::PendingActivation::COST_RETURN: {
            // Return-to-hand cost (Scryb Ranger: return a Forest to hand) — chosen now, returned
            // at PAY_APPLY, like the sacrifice pick above.
            if (!ability.return_cost_type.empty()) {
                std::vector<Entity> choices = controlled_permanents_matching(
                    controller, ability.return_cost_type, orderer->mEntities);
                if (!choices.empty()) {
                    if (resume_choice < 0) {
                        arm_flow_query(game, PendingQuery::ACTIVATION,
                                       permanent_choice_menu(choices, "Return ", " to hand",
                                                             ActionCategory::RETURN_PERMANENT),
                                       controller, permanent_entity);
                        return;
                    }
                    pa.return_choice = choices[static_cast<size_t>(resume_choice)];
                    resume_choice = -1;
                    // Ninjutsu's ninja attacks what the returned creature was attacking (CR
                    // 702.49c), read while the creature is still in combat.
                    if (ability.is_ninjutsu)
                        pa.stack_ab.ninjutsu_attack_target =
                            global_coordinator.GetComponent<Creature>(pa.return_choice).attack_target;
                }
            }
            pa.step = Game::PendingActivation::PAY;
            break;
        }

        case Game::PendingActivation::PAY: {
            // The first step that pays anything, so the snapshot a failed payment restores is
            // taken here (CR 733.1): it records the source untapped, so the rewind also undoes
            // the tap cost.
            pa.mana_snap = snapshot_mana_state(controller, orderer);
            pa.mana_snap_taken = true;
            // Tap cost
            if (ability.tap_cost && global_coordinator.entity_has_component<Permanent>(permanent_entity))
                global_coordinator.GetComponent<Permanent>(permanent_entity).is_tapped = true;
            // Mana cost (after ReduceCost$ — CR 601.2f; reduces generic only; Eiganjo's Channel
            // is cheaper per legendary creature). For an X-cost ability the chosen X is added as
            // generic mana on top of the base cost — once per {X} pip in the cost, so Blast
            // Zone's {X}{X} charges 2X, not X.
            ManaValue activate_cost = effective_activation_mana_cost(ability, controller, orderer);
            size_t x_pips = std::max<size_t>(1, ability.activation_x_count);
            for (size_t i = 0; i < pa.x_activation * x_pips; i++) activate_cost.insert(GENERIC);
            // CR 601.2g: a permanent about to be sacrificed or returned for this cost is still on
            // the battlefield, so it may be tapped for mana on its way out.
            for (Entity leaving : {pa.sac_choice, pa.return_choice})
                if (leaving != 0)
                    float_mana_before_cost_removal(leaving, controller, orderer, activate_cost,
                                                   permanent_entity);
            if (!activate_cost.empty() &&
                !prompt_mana_payment(controller, activate_cost, permanent_entity, orderer)) {
                fail_activation_payment(pa, orderer);
                return;
            }
            pa.step = Game::PendingActivation::PAY_APPLY;
            break;
        }

        case Game::PendingActivation::PAY_APPLY: {
            // The mana committed, so the rest of the cost is now paid (CR 602.2b / 601.2h): the
            // life and energy first — checked before anything of this step happens, since a
            // painful mana source can have left too little life (only for an activation the gate
            // wrongly offered), and then the whole cost can't be paid and the activation is
            // reversed — then the loyalty, sacrifice, return and discard costs.
            auto &activating_player =
                global_coordinator.GetComponent<Player>(get_player_entity(controller));
            if (!can_pay_life(activating_player, ability.life_cost) ||
                player_energy(activating_player) < ability.energy_cost) {
                fail_activation_payment(pa, orderer);
                return;
            }
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
                pay_life(activating_player, ability.life_cost);
                game_log("%s pays %d life\n", player_name(controller).c_str(), ability.life_cost);
            }
            // Energy cost (PayEnergy<N>, CR 122.1c).
            if (ability.energy_cost > 0) {
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
            if (pa.sac_choice != 0) {
                std::string sac_name = global_coordinator.GetComponent<Permanent>(pa.sac_choice).name;
                orderer->add_to_zone(false, pa.sac_choice, Zone::GRAVEYARD);
                game_log("%s sacrifices %s\n", player_name(controller).c_str(), sac_name.c_str());
            }
            if (pa.return_choice != 0) {
                std::string ret_name =
                    global_coordinator.GetComponent<Permanent>(pa.return_choice).name;
                orderer->add_to_zone(false, pa.return_choice, Zone::HAND);
                game_log("%s returns %s to hand\n", player_name(controller).c_str(),
                         ret_name.c_str());
            }
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

                // The ability, on the stack since its activation began, becomes activated.
                finish_activated_ability(pa, controller, orderer);

                // Ward (702.21) + BecomesTarget (CR 603.2c) apply to hand/graveyard-activated
                // abilities too — CR 702.21b triggers on ANY spell or ability an opponent
                // controls that targets the warded permanent. (Graveyard/non-battlefield
                // targets are filtered inside the hooks.)
                fire_targeting_hooks(pa.stack_entity, controller);

                auto &cd = global_coordinator.GetComponent<CardData>(permanent_entity);
                const char *from_zone = (ability.activation_zone == Zone::GRAVEYARD) ? "graveyard" : "hand";
                if (!pa.stack_ab.target.empty()) {
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
                // Costs were already paid above: tap and activation mana at PAY, life/sacrifice/
                // discard at PAY_APPLY. The shared production core handles the rest: amount eval
                // (dynamic amounts like Gaea's Cradle / Urza's Workshop), ProduceMana
                // replacement, pool insert, TapsForMana triggers, uncounterability flag,
                // narrative, SubAbility$ riders (Ancient Tomb's damage), and the activation
                // counter.
                auto &pl = global_coordinator.GetComponent<Player>(get_player_entity(controller));
                produce_mana_from_ability(permanent_entity, ability, controller, orderer, pl.mana,
                                          /*commit=*/true, ManaLogStyle::TAPPED_AMOUNT);
                // priority does not pass
                pa = Game::PendingActivation{};
                return;
            }
            // ACTIVATED ABILITY THAT IS NOT A MANA ABILITY — on the stack since its activation
            // began; it now becomes activated with the targets chosen earlier.
            auto &permanent = global_coordinator.GetComponent<Permanent>(permanent_entity);
            finish_activated_ability(pa, controller, orderer);

            // Ward (702.21) + Mode$ BecomesTarget (CR 603.2c): abilities fire these too; the
            // per-trigger ValidSource$ filter (e.g. Reality Smasher's Spell.OppCtrl) gates out
            // ability sources for BecomesTarget.
            fire_targeting_hooks(pa.stack_entity, controller);

            if (!pa.stack_ab.target.empty()) {
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
    // Modal DFC cast as its NONLAND back face (CR 712.8): the entity's CardData is the front
    // face, but the spell has only the BACK face's characteristics — pay the back's mana cost,
    // put the back's spell ability on the stack, and (if the back is a permanent) enter as the
    // back face. Every cast-path read of card_data comes from the back face for this cast.
    const CardData &card_data = (pc.cast_back_face && front_data.backside)
                                    ? *front_data.backside : front_data;
    Zone::Ownership caster = pc.caster_is_a ? Zone::PLAYER_A : Zone::PLAYER_B;

    for (;;) switch (pc.step) {
        case Game::PendingCast::COST: {
            // A NORMAL play-from-exile grant (Light Up the Stage, warp) casts the card for its
            // own costs, so it follows the regular branch: kicker, X, hybrid and Phyrexian pips
            // and additional costs all apply (CR 601.2b, 601.2f).
            auto grant_it = cur_game.impulse_cast_permission.find(spell_entity);
            const bool impulse_normal =
                pc.impulse_cast && grant_it != cur_game.impulse_cast_permission.end() &&
                grant_it->second.resource == Game::ImpulseCastPermission::NORMAL;
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

            // CAST FROM EXILE under a play permission, for the cost that replaces its mana cost
            // (CR 118.9): nothing (FREE: Ugin -11, Dauthi Voidwalker, suspend) or an alternative
            // resource (ENERGY / LIFE: Amped Raptor's DB$ Play) of the permission's resolved
            // amount, determined here (CR 601.2f) and paid with the other costs at PAY_APPLY
            // (CR 601.2h). The permission is consumed as the spell becomes cast (FINISH). X is 0
            // for a spell cast without paying its mana cost (CR 107.3b). A NORMAL grant (Light
            // Up the Stage, warp) pays the card's own costs, so it takes the regular branch below.
            } else if (pc.impulse_cast && !impulse_normal) {
                if (grant_it != cur_game.impulse_cast_permission.end()) {
                    const auto &grant = grant_it->second;
                    if (grant.resource == Game::ImpulseCastPermission::FREE)
                        game_log("%s casts %s without paying its mana cost\n",
                                 player_name(caster).c_str(), card_data.name.c_str());
                    else if (grant.resource == Game::ImpulseCastPermission::ENERGY)
                        pc.deferred_energy_cost = grant.amount;
                    else
                        pc.deferred_life_cost = grant.amount;
                }

                if (card_data.has_x_cost) cur_game.x_paid = 0;

                // Cost-increase / SetCost-floor statics apply to alternative costs too
                // (CR 118.9d / 601.2f): the cast substitutes a {0} mana cost, but an active
                // Trinisphere floor pads it up to its minimum ({3}) and Thalia adds its surcharge
                // — paid ON TOP of the resource cost (energy/life). Deferred until after targets
                // like every other cost. Empty (no floor / increase applies) leaves the cast free
                // of mana.
                ManaValue floor_mana = floored_alt_mana_cost(card_data, ManaValue{}, caster);
                if (!floor_mana.empty()) {
                    pc.deferred_mana_cost = floor_mana;
                    pc.deferred_mana_pending = true;
                }
                pc.step = Game::PendingCast::GIFT;

            // ALTERNATE COST — determined here, paid in the payment phase like every
            // other cost branch (its pitch/return/sacrifice picks are the ALT_* steps).
            } else if (pc.use_alt_cost) {
                defer_alternate_cost(game, card_data, caster);
                pc.step = Game::PendingCast::GIFT;

            } else {  // REGULAR COST + DELVE (also a NORMAL play-from-exile grant)
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
                    if (pitch_color != NO_COLOR && !effective_colors(e).count(pitch_color)) continue;
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
                arm_cast_query(game, std::move(exile_actions), caster, spell_entity);
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
                for (auto e : controlled_permanents_matching(caster, type, orderer->mEntities)) {
                    if (already_chosen_as_cost(pc, e)) continue;
                    LegalAction la(PASS_PRIORITY, e,
                                   "Return " + global_coordinator.GetComponent<Permanent>(e).name);
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
                arm_cast_query(game, std::move(rth_actions), caster, spell_entity);
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
                arm_cast_query(game, std::move(sac_actions), caster, spell_entity);
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
                std::string prompt = "pay kicker " + mana_value_text(card_data.kicker_costs[ki]) +
                    " for " + card_data.name;
                if (card_data.kicker_costs.size() > 1)
                    prompt += " (kicker " + std::to_string(ki + 1) + ")";
                arm_cast_query(game, optional_yesno_menu(prompt), caster, spell_entity);
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
                    std::string prompt = "pay replicate " + mana_value_text(card_data.replicate_cost) +
                        " for " + card_data.name + " (paid " + std::to_string(pc.replicate_count) + ")";
                    arm_cast_query(game, optional_yesno_menu(prompt), caster, spell_entity);
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
                    // Each {X} in the cost is paid with the one chosen value (CR 107.3a).
                    size_t x_pips = static_cast<size_t>(card_data.x_pip_count);
                    for (size_t i = 0; i < x_val * x_pips; i++) pc.cost_to_pay.insert(GENERIC);
                    game_log("%s chooses X = %zu\n", player_name(caster).c_str(), x_val);
                } else {
                    size_t x_pips = static_cast<size_t>(card_data.x_pip_count);
                    size_t max_x = max_available_mana(caster, pc.cost_to_pay, orderer) / x_pips;
                    max_x = payable_max_x(caster, pc.cost_to_pay, max_x, x_pips, spell_entity,
                                          /*exclude=*/0, orderer, card_data.has_delve,
                                          card_data.has_improvise);
                    // For a spell whose required target count IS X (Hide on the Ceiling), X can't
                    // exceed the number of legal targets (CR 601.2c) — clamp so the agent can't
                    // pick an X that leaves a mandatory target choice with too few candidates.
                    max_x = std::min(max_x,
                                     spell_xpaid_target_cap(card_data, spell_entity, caster, orderer));
                    // CR 601.2e: an X whose mana value a static prohibits (Lavinia) would make
                    // the proposed spell illegal, so it isn't offered.
                    max_x = std::min(max_x, static_cast<size_t>(rules_mod::max_castable_x(caster, card_data)));

                    game_log("Choose X value (0-%zu):\n", max_x);
                    std::vector<LegalAction> x_actions;
                    for (size_t xv = 0; xv <= max_x; xv++) {
                        LegalAction la(PASS_PRIORITY, std::string("X = " + std::to_string(xv)));
                        la.category = ActionCategory::CHOOSE_X;
                        la.option_ordinal = static_cast<int>(xv);  // the chosen X value
                        x_actions.push_back(la);
                    }
                    arm_cast_query(game, std::move(x_actions), caster, spell_entity);
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
                        int hc;
                        {
                            PendingDecisionScope pending(spell_entity);
                            hc = InputLogger::instance().get_input(hybrid_actions);
                        }
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
                Entity caster_entity = get_player_entity(caster);
                auto &phyrex_player = global_coordinator.GetComponent<Player>(caster_entity);
                while (pc.phyrexian_idx < card_data.phyrexian_mana.size()) {
                    Colors phyrex_color = card_data.phyrexian_mana[pc.phyrexian_idx];
                    std::string color_name = mana_symbol_str(phyrex_color);
                    // CR 119.4 / 118.3: a player can't pay 2 life they don't have. Only offer
                    // the life option when life >= 2 (paying down to exactly 0 is legal — they
                    // die to SBAs afterward). Re-checked per pip against the running life total,
                    // since an earlier pip's life payment lowers what's left for the next.
                    // The life is paid as the pip is announced rather than with the other
                    // costs (CR 601.2h lets costs be paid in any order), so the choice shows in
                    // the life total at the prompts that follow; a reversed cast gives it back
                    // (phyrexian_life_paid). Between this pip's arm and its apply nothing runs,
                    // so the recompute at apply matches the armed menu's option layout.
                    bool life_payable = can_pay_life(phyrex_player, 2);
                    if (resume_choice >= 0) {
                        int phyrex_choice = resume_choice;
                        resume_choice = -1;
                        // The life option occupies index 0 only when it was offered; with it
                        // suppressed the sole option is "Pay {color}", so fall through to mana.
                        // The mana option's own gate (below) never shifts this indexing — life
                        // is always index 0 when present — so only life_payable is recomputed
                        // here.
                        if (life_payable && phyrex_choice == 0) {
                            pay_life(phyrex_player, 2);
                            pc.phyrexian_life_paid += 2;
                            game_log("%s pays 2 life\n", player_name(caster).c_str());
                        } else {
                            pc.cost_to_pay.insert(phyrex_color);
                        }
                        pc.phyrexian_idx++;
                        continue;
                    }
                    // The colored half of the pip (CR 107.4f) is only a real choice when that
                    // mana is actually payable ON TOP of the cost accumulated so far. The
                    // cast-legality gate treats a Phyrexian pip as always payable — it is, via
                    // life — so without this check a {B/P} spell offered "Pay {B}" with no black
                    // source, and taking it dead-ended the whole cast in a payment failure.
                    // Checked against the same predicate the payer uses, so the option is
                    // offered exactly when it can be honoured.
                    ManaValue with_pip = pc.cost_to_pay;
                    with_pip.insert(phyrex_color);
                    bool can_pay_colored =
                        can_pay_mana(caster, with_pip, spell_entity, orderer,
                                     card_data.has_delve, card_data.has_improvise);
                    // Never arm an empty menu. Neither half payable means the cast should not
                    // have been offered (life < 2 AND no source); keep the mana option so the
                    // payment fails through the normal path instead of vanishing here.
                    if (!life_payable && !can_pay_colored) can_pay_colored = true;

                    std::vector<LegalAction> phyrex_actions;
                    if (life_payable) {
                        LegalAction pay_life_action(PASS_PRIORITY,
                            "Pay 2 life (instead of {" + color_name + "})");
                        pay_life_action.category = ActionCategory::PAYING_COSTS;
                        pay_life_action.option_ordinal = 0;  // Phyrexian pip: 0 = pay life
                        phyrex_actions.push_back(pay_life_action);
                    }
                    if (can_pay_colored) {
                        LegalAction pay_mana(PASS_PRIORITY, "Pay {" + color_name + "}");
                        pay_mana.category = ActionCategory::PAYING_COSTS;
                        pay_mana.option_ordinal = 1;  // Phyrexian pip: 1 = pay mana
                        phyrex_actions.push_back(pay_mana);
                    }
                    arm_cast_query(game, std::move(phyrex_actions), caster, spell_entity);
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
            // no life. X may be 0..life (CR 119.4 lets a player pay up to their whole total),
            // less whatever other life the cast costs and whatever life the mana payment must
            // take (max_life_x), so every offered X is payable.
            if (spell_has_variable_life_cost(card_data)) {
                if (resume_choice >= 0) {
                    size_t x_val = static_cast<size_t>(resume_choice);
                    resume_choice = -1;
                    cur_game.x_paid = x_val;
                    pc.life_x_announced = static_cast<int>(x_val);
                } else {
                    size_t max_x = max_life_x(pc, caster, spell_entity, orderer);
                    game_log("Choose X value (0-%zu):\n", max_x);
                    std::vector<LegalAction> x_actions;
                    for (size_t xv = 0; xv <= max_x; xv++) {
                        LegalAction la(PASS_PRIORITY, std::string("X = " + std::to_string(xv)));
                        la.category = ActionCategory::CHOOSE_X;
                        la.option_ordinal = static_cast<int>(xv);  // the chosen X value
                        x_actions.push_back(la);
                    }
                    arm_cast_query(game, std::move(x_actions), caster, spell_entity);
                    return;
                }
            }
            pc.step = Game::PendingCast::GIFT;
            break;
        }

        case Game::PendingCast::SPELL_SAC: {
            // ADDITIONAL SACRIFICE COST on the spell itself (CR 601.2f / 118.x):
            // Natural Order — "As an additional cost to cast this spell, sacrifice a
            // green creature." Chosen here and sacrificed at PAY_APPLY, using the same
            // SACRIFICE_PERMANENT choice activated abilities use.
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
                        arm_cast_query(game, std::move(menu), caster, spell_entity);
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
                                 player_name(opponent_of(caster)).c_str());
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
                                 player_name(opponent_of(caster)).c_str());
                    } else if (can_promise) {
                        arm_cast_query(game,
                                       optional_yesno_menu("promise " + gname + " to your opponent"),
                                       caster, spell_entity);
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
                pc.ability.source = ObjectRef::of(spell_entity);  // the spell on the stack (601.2a)
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
            // enchant target (AURA_TARGET). NOTE on ordering: strict CR 601.2b announces modes
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
            // mode's targets are chosen (CHARM_TARGET) before the NEXT mode pick. The picked
            // modes persist in
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
            arm_cast_query(game, std::move(mode_actions), caster, ability.source.lki_entity());
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
                    sub.targeted_player = pc.ability.player_target_for_subs();  // ParentTarget
                    FlowTargetAsker asker(game, caster, resume_choice);
                    if (run_target_select(sub, pc.tsel, asker, orderer, caster) !=
                        TargetStatus::DONE)
                        return;
                }
                pc.sub_idx++;
            }
            // Announcement complete: add the fully-targeted Ability to the entity.
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
                if (!pc.tsel.active)
                    pc.enchant_ab = enchant_target_ability(spell_entity, card_data, caster);
                FlowTargetAsker asker(game, caster, resume_choice);
                if (run_target_select(pc.enchant_ab, pc.tsel, asker, orderer, caster) !=
                    TargetStatus::DONE)
                    return;
                if (!pc.enchant_ab.target.empty()) {
                    cur_game.pending_aura_target[spell_entity] = PendingAuraTarget{pc.enchant_ab.target};
                    game_log("%s casts %s enchanting %s\n", player_name(caster).c_str(),
                             card_data.name.c_str(),
                             target_display_name(cur_game, pc.enchant_ab.target.lki_entity()).c_str());
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
            // pc and DELVE_PICK's completion (or a rewind) restores it.
            pc.delve_prev_priority_a = cur_game.player_a_has_priority;
            pc.delve_seat_held = true;
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
                arm_cast_query(game, std::move(count_menu), caster, spell_entity);
                return;
            }
            pc.delve_exile_ct = min_needed;
            pc.step = Game::PendingCast::DELVE_PICK;
            break;
        }

        case Game::PendingCast::DELVE_PICK: {
            // Delve stage 2 — which cards, one pick at a time; the candidate menu is
            // re-derived from the live graveyard each pass minus the cards already picked.
            // When the remaining candidates are all going to be exiled anyway (candidates
            // <= picks left, including the single-candidate case) there is no real choice,
            // so they are taken without a prompt — the pre-collapse condition, re-checked
            // per pick at arm. Each pick pays one GENERIC pip of pc.deferred_mana_cost
            // (CR 702.66a) and is a chosen cost item: the card is exiled, and recorded in
            // cur_game.delve_exiled, at PAY_APPLY.
            while (pc.delve_picks_done < pc.delve_exile_ct) {
                std::vector<LegalAction> picks;
                for (auto e : orderer->mEntities) {
                    if (!is_delve_eligible(e, caster) || already_chosen_as_cost(pc, e)) continue;
                    auto &ecd = global_coordinator.GetComponent<CardData>(e);
                    LegalAction la(PASS_PRIORITY, e, "Exile " + ecd.name + " (Delve)");
                    la.category = ActionCategory::CHOOSE_CARD;
                    la.card_is_public = true;  // graveyards are public zones
                    picks.push_back(la);
                }
                if (picks.empty()) break;  // defensive: eligibility was counted above
                size_t pick = 0;
                if (picks.size() > pc.delve_exile_ct - pc.delve_picks_done) {
                    if (resume_choice < 0) {
                        game_log("Choose a card to exile via Delve (%zu of %zu):\n",
                                 pc.delve_picks_done + 1, pc.delve_exile_ct);
                        arm_cast_query(game, std::move(picks), caster, spell_entity);
                        return;
                    }
                    pick = static_cast<size_t>(resume_choice);
                    resume_choice = -1;
                }
                choose_delve_exile(pc, picks[pick].source_entity, caster);
                pc.delve_picks_done++;
            }
            // Restore the pre-delve seat (persisted at DELVE_COUNT's arm).
            cur_game.player_a_has_priority = pc.delve_prev_priority_a;
            pc.delve_seat_held = false;
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
            // This is the first step that can activate a mana ability, so the snapshot a
            // reversed proposal restores them from is taken here (CR 733.1).
            pc.mana_snap = snapshot_mana_state(caster, orderer);
            pc.mana_snap_taken = true;
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
                    // Payment cancelled (interactive), or a machine-mode payment the gate
                    // wrongly offered: the proposal is reversed (CR 601.5 / 733.1).
                    fail_cast_payment(pc, orderer);
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
            //
            // The life and energy are checked first, while nothing of this step has
            // happened yet: a painful mana source the payment used can have left too little
            // life (the gate reserves it, so only a wrongly offered cast gets here), and
            // then the whole cost can't be paid and the proposal is reversed (CR 601.5).
            {
                auto &player = global_coordinator.GetComponent<Player>(get_player_entity(caster));
                const int life_x = pc.life_x_announced >= 0 ? pc.life_x_announced : 0;
                if (!can_pay_life(player, pc.deferred_life_cost + life_x) ||
                    player_energy(player) < pc.deferred_energy_cost) {
                    fail_cast_payment(pc, orderer);
                    return;
                }
                if (pc.deferred_life_cost > 0) {
                    pay_life(player, pc.deferred_life_cost);
                    game_log("%s pays %d life\n", player_name(caster).c_str(), pc.deferred_life_cost);
                }
                if (pc.life_x_announced >= 0) {
                    pay_life(player, pc.life_x_announced);
                    game_log("%s pays %d life (X = %d)\n", player_name(caster).c_str(),
                             pc.life_x_announced, pc.life_x_announced);
                }
                if (pc.deferred_energy_cost > 0) {
                    pay_energy(player, pc.deferred_energy_cost);
                    game_log("%s pays %d energy\n", player_name(caster).c_str(),
                             pc.deferred_energy_cost);
                }
            }
            // A delve cast's exiles replace any a previous delve spell recorded
            // (delve_exiled persists until an etbCounter replacement consumes it).
            if (card_data.has_delve) cur_game.delve_exiled.clear();
            for (const auto &r : pc.cost_removals) {
                game_log("%s\n", r.log.c_str());
                orderer->add_to_zone(false, r.entity, r.dest);
                if (r.delve) cur_game.delve_exiled.push_back(r.entity);
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
                        arm_cast_query(game, std::move(menu), caster, spell_entity);
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
                    arm_cast_query(game, std::move(menu), caster, spell_entity);
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
                    arm_cast_query(game, std::move(menu), caster, spell_entity);
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
                Entity tgt = global_coordinator.GetComponent<Ability>(spell_entity).target.lki_entity();
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

            // The spell becomes cast (CR 601.2i): its Spell component (added with the caster
            // as the card moved to the stack, begin_cast) now records how it was cast.
            Spell spell;
            spell.caster = caster;
            spell.cast_with_flashback = pc.use_flashback;
            spell.cast_with_evoke = pc.use_alt_cost && card_data.alt_cost.is_evoke;
            spell.cast_with_escape = pc.use_escape;
            spell.cast_with_offspring = pc.use_offspring;
            spell.cast_with_impending = pc.use_alt_cost && card_data.alt_cost.is_impending;
            spell.cast_with_warp = pc.use_alt_cost && card_data.alt_cost.is_warp;
            spell.cast_back_face = pc.cast_back_face && front_data.backside;
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
            global_coordinator.GetComponent<Spell>(spell_entity) = spell;
            // A play permission is consumed by the cast it allowed (it lapses once the card
            // has left exile).
            if (pc.impulse_cast) cur_game.impulse_cast_permission.erase(spell_entity);

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
                    Entity caster_entity = get_player_entity(caster);
                    cast_ev.SetParam(Params::ENTITY, spell_entity);
                    cast_ev.SetParam(Params::PLAYER, caster_entity);
                    global_coordinator.SendEvent(cast_ev);
                }
            }

            // The card's move to the stack (made as casting began, CR 601.2a) is reported now.
            orderer->complete_cast_move(spell_entity, pc.cast_origin);

            // Track spells cast and fire SPELL_CAST event
            {
                Entity caster_entity = get_player_entity(caster);
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
                // evaluated.
                for (Colors c : effective_colors(spell_entity))
                    caster_player.spell_colors_cast_this_turn.insert(c);
                Event spell_event(Events::SPELL_CAST);
                spell_event.SetParam(Params::PLAYER, caster_entity);
                spell_event.SetParam(Params::ENTITY, spell_entity);
                global_coordinator.SendEvent(spell_event);
            }

            // Ward (702.21): an opponent's permanent this spell targets may counter it. The
            // spell is already on the stack, so the Ward trigger pushed here lands above it and
            // resolves first. Mode$ BecomesTarget triggers (Reality Smasher): a targeted
            // permanent whose becomes-target trigger matches fires it above this spell (CR
            // 603.2c/603.3). Every target counts — each chosen mode's, each sub-ability's and an
            // Aura's enchant target (CR 115.1a/b, 601.2c).
            fire_targeting_hooks(spell_entity, caster);

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
            // cancel path never reaches here, like the old break). A spell cast during a
            // resolution is not a priority action: no player receives priority after it is
            // cast, and the resolving object's pass state stands (CR 608.2g).
            if (!game.resolution.active) game.take_action();
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

            // Move to battlefield. A land enters under the control of the player who played it
            // (CR 305.2 / 110.2a) — the priority holder, who need not own it (a land played from
            // exile under a play permission).
            Zone::Ownership land_player = priority_seat();
            orderer->add_to_zone(false, land_entity, Zone::BATTLEFIELD);
            zone.controller = land_player;
            // ForgetOnMoved$ Exile: a land played from exile under a Light Up the Stage play
            // permission consumes that permission as it leaves exile (harmless no-op otherwise).
            cur_game.impulse_cast_permission.erase(land_entity);

            // Permanent component added by apply_permanent_components on next SBA pass

            // Update player's lands played counter
            Entity player_entity = get_player_entity(land_player);
            auto &player = global_coordinator.GetComponent<Player>(player_entity);
            player.lands_played_this_turn++;

            game_log("%s played %s\n", player_name(land_player).c_str(), played_face->name.c_str());

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
            // The caster is the player taking this action (the priority holder, CR 601.2), who
            // need not own the card (a card cast from exile under a play permission).
            Zone::Ownership caster = priority_seat();

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
            begin_cast(pc, caster, orderer);
            run_cast_flow(pc, game, orderer, -1);
            break;
        }
    }
}

// ── T3.10: prompted combat damage assignment among multiple blockers ──────────

// Does this attacker need its controller to choose how to divide combat damage this step?
// Only when it deals damage this step, is blocked by 2+ live blockers, and CANNOT assign lethal
// to all of them (power <= total lethal). When it can kill everything (power > total lethal) the
// choice is immaterial, so deal_combat_damage() auto-assigns instead (the ML simplification).
static bool attacker_needs_assignment(Entity attacker, std::shared_ptr<Orderer> orderer,
                                      bool first_strike_only) {
    if (!is_attacking_creature(attacker)) return false;
    auto &cr = global_coordinator.GetComponent<Creature>(attacker);
    if (!cr.is_blocked) return false;
    if (!should_deal_damage(cr, first_strike_only)) return false;
    auto blockers = blockers_of(attacker, orderer->mEntities);
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
    // The attacking creature whose damage is being divided is the
    // pending-decision source.
    pq.decision_source = pd.attacker;
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
        pd.pool = blockers_of(attacker, orderer->mEntities);
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

// One loop-safe miracle yes/no read, with the miracle card as the pending-decision source.
// The baseline is reset to 0 first (a mandatory choice runs from the loop top with no ambient
// pending decision): a SNAPSHOT at this decision captures the scoped value, so a RESTORE
// re-enters with it still set, and without the reset the recreated scope would capture it as
// its prev and leak it past the answer.
static int ask_miracle_choice(Game &game, const std::vector<LegalAction> &menu, Entity card) {
    game.pending_decision_source = 0;
    PendingDecisionScope pending(card);
    search_set_loop_safe(true);
    int choice = InputLogger::instance().get_input(menu);
    search_set_loop_safe(false);
    return choice;
}

// Miracle (CR 702.94a) reveal decision. A first-of-turn miracle card was drawn and its owner may
// reveal it "as they draw it" — a PRIVATE special action taken off the stack (the opponent is not
// told a miracle card was drawn unless it is revealed). This forced yes/no is presented to the
// OWNER (who need not hold priority) before they proceed. On decline the card stays hidden in hand.
// On accept the card becomes public (belief state + log) and the linked "when you reveal this card
// this way, you may cast it" triggered ability is synthesized onto the stack (opponent now sees the
// revealed card and gets a response window); the owner decides whether to cast it as that trigger
// resolves (effect_miracle.cpp).
static void proc_miracle_reveal(Game &game, std::shared_ptr<Orderer> orderer) {
    Entity card = game.miracle_reveal_pending;
    // The pending flag is the ONLY state that lets a restored loop re-derive this
    // prompt (is_mandatory_choice_pending -> proc_mandatory_choice re-asks), so it
    // must stay SET across the get_input below: the ask is a loop-safe MCTS search
    // root, and a snapshot taken there with the flag already cleared restores into
    // a state that silently forgets the question — every post-restore simulation
    // then spends the root's action on an unrelated query (the az_mcts
    // world-consistency violation). Consume it AFTER the answer (or on the
    // validity bail-outs below).
    // The card must still be in its owner's hand to be miracle-revealed (nothing runs between the
    // draw and this decision today, but guard against a vanished/moved entity regardless).
    if (!global_coordinator.entity_has_component<Zone>(card)) {
        game.miracle_reveal_pending = 0;
        return;
    }
    auto &z = global_coordinator.GetComponent<Zone>(card);
    if (z.location != Zone::HAND || (z.owner != Zone::PLAYER_A && z.owner != Zone::PLAYER_B)) {
        game.miracle_reveal_pending = 0;
        return;
    }
    Zone::Ownership owner = z.owner;
    const std::string nm = global_coordinator.entity_has_component<CardData>(card)
                               ? global_coordinator.GetComponent<CardData>(card).name
                               : "the card";

    std::vector<LegalAction> yn =
        yesno_menu("Don't reveal " + nm + " (miracle)", "Reveal " + nm + " for its miracle cost");

    // Point the input query at the OWNER (the drawer), who need not be the priority holder — the
    // shared chooser-scope pattern (mirrors CLEANUP_DISCARD): machine mode then serializes the state
    // from the owner's perspective and routes the decision to them, keeping it hidden from the
    // opponent. Loop-safe: one decision derived from the pending card alone.
    bool prev_priority = game.player_a_has_priority;
    game.player_a_has_priority = (owner == Zone::PLAYER_A);
    int choice = ask_miracle_choice(game, yn, card);
    game.player_a_has_priority = prev_priority;
    // Answer consumed — NOW the one-shot decision is spent (see the flag note above).
    // On a search unwind the restore overwrites the flag from the snapshot (still
    // set), so the restored line re-derives this same prompt.
    game.miracle_reveal_pending = 0;

    if (choice != 1) return;  // declined — the card stays hidden in hand, no cast opportunity

    // Revealed (CR 702.94a/702.94b): the card is now public. Record the reveal in the belief state
    // and log it, then put the linked "you may cast it" triggered ability on the stack.
    mark_card_revealed(card, owner);
    game_log("%s reveals %s for its miracle cost.\n", player_name(owner).c_str(), nm.c_str());
    Ability trig;
    trig.ability_type = Ability::TRIGGERED;
    trig.category = "MiracleCast";
    trig.source = ObjectRef::of(card);
    trig.controller = owner;
    orderer->push_ability_onto_stack(trig, owner);
}

// See declaration in action_processor.h.
ResolutionCastStatus cast_during_resolution(const LegalAction &cast, Zone::Ownership caster,
                                            bool castable, const std::string &accept_label,
                                            ResolutionCastRt &rt, FrameCtx &ctx,
                                            std::shared_ptr<Orderer> orderer) {
    Entity card = cast.source_entity;
    if (rt.stage == ResolutionCastRt::OFFER) {
        if (!castable) {
            rt.stage = ResolutionCastRt::DONE;
            return ResolutionCastStatus::DECLINED;
        }
        int choice =
            ctx.ask(yesno_menu("Do not cast " + entity_name(card), accept_label, card), caster, card);
        if (choice < 0 && decision_suspended()) return ResolutionCastStatus::SUSPENDED;
        if (choice != 1) {
            rt.stage = ResolutionCastRt::DONE;
            return ResolutionCastStatus::DECLINED;
        }
        // The cast is made by `caster`, who holds the cast flow's prompts; priority returns to
        // the resolving ability's controller once it completes.
        rt.stage = ResolutionCastRt::CASTING;
        rt.prev_priority = cur_game.player_a_has_priority;
        cur_game.player_a_has_priority = (caster == Zone::PLAYER_A);
        process_action(cast, cur_game, orderer);
        if (decision_suspended()) return ResolutionCastStatus::SUSPENDED;
    }
    if (rt.stage == ResolutionCastRt::CASTING) {
        if (cur_game.pending_cast.active)
            fatal_error("cast_during_resolution re-entered with the cast still in flight");
        cur_game.player_a_has_priority = rt.prev_priority;
        rt.stage = ResolutionCastRt::DONE;
        // A cancelled cast leaves the card where it was.
        bool on_stack = global_coordinator.entity_has_component<Zone>(card) &&
                        global_coordinator.GetComponent<Zone>(card).location == Zone::STACK;
        return on_stack ? ResolutionCastStatus::CAST : ResolutionCastStatus::DECLINED;
    }
    return ResolutionCastStatus::DECLINED;
}

// See declaration in action_processor.h.
ResolutionCastStatus cast_during_resolution(Entity card, Zone::Ownership caster,
                                            Game::ImpulseCastPermission grant,
                                            ResolutionCastRt &rt, FrameCtx &ctx,
                                            std::shared_ptr<Orderer> orderer) {
    // The permission exists only while the offer is open or the cast is in flight: the cast
    // consumes it, and a declined or cancelled cast drops it.
    bool castable = false;
    if (rt.stage == ResolutionCastRt::OFFER) {
        grant.caster = caster;
        grant.during_resolution = true;
        cur_game.impulse_cast_permission[card] = grant;
        castable = exile_grant_castable(card, caster, /*sorcery_window=*/false, orderer);
    }
    const std::string nm = entity_name(card);
    LegalAction cast(CAST_SPELL, card, "Cast " + nm);
    cast.category = ActionCategory::CAST_SPELL;
    cast.impulse_cast = true;
    const char *how = grant.resource == Game::ImpulseCastPermission::FREE
                          ? " without paying its mana cost" : "";
    ResolutionCastStatus status =
        cast_during_resolution(cast, caster, castable, "Cast " + nm + how, rt, ctx, orderer);
    if (status != ResolutionCastStatus::SUSPENDED) cur_game.impulse_cast_permission.erase(card);
    return status;
}

void proc_mandatory_choice(Game &game, std::shared_ptr<Orderer> orderer) {
    // A pending miracle reveal (CR 702.94) is a forced decision the drawing player makes before
    // proceeding; it rides this channel but is not a pending_choice enum value.
    if (game.miracle_reveal_pending != 0) {
        proc_miracle_reveal(game, orderer);
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
            Zone::Ownership active_player = active_seat();
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
