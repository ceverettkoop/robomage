#include "action_processor.h"

#include <algorithm>
#include <cstdio>

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
#include "error.h"
#include "game_queries.h"
#include "input_logger.h"
#include "mana_system.h"
#include "systems/orderer.h"
#include "systems/state_manager.h"
#include "systems/state_manager_internal.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static Entity prompt_permanent_choice(const std::vector<Entity> &choices, const char *verb, const char *suffix,
                                      ActionCategory category);
static void pay_secondary_activation_costs(
    const Ability &ability, Entity source, Zone::Ownership controller, std::shared_ptr<Orderer> orderer);
static void select_single_target(Ability &ability, const std::vector<Entity> &valid_targets, bool allow_done);
static void process_activate_ability(const LegalAction &action, Game &game, std::shared_ptr<Orderer> orderer);
static std::vector<Entity> build_valid_targets(
    const Ability &ability, std::shared_ptr<Orderer> orderer, Zone::Ownership priority_player);
static void pay_alternate_cost(const LegalAction &action, Game &game, std::shared_ptr<Orderer> orderer,
    const CardData &card_data, Entity spell_entity, Zone zone);
static void declare_attackers(Game &game, std::shared_ptr<Orderer> orderer);
static bool player_controls_land_subtype(Zone::Ownership player, const std::string &subtype);
static std::string landwalk_subtype(const std::string &kw);
static std::vector<Entity> determine_blockable_attackers(Entity blocker, const std::vector<Entity> &attackers);
static void declare_blockers(Game &game, std::shared_ptr<Orderer> orderer);
static std::vector<Entity> collect_live_blockers(Entity attacker, std::shared_ptr<Orderer> orderer);
static bool attacker_needs_assignment(Entity attacker, std::shared_ptr<Orderer> orderer, bool first_strike_only);
static void assign_combat_damage(Game &game, std::shared_ptr<Orderer> orderer);

// entity_name() is shared from the StateManager TUs via state_manager_internal.h.
// mana_symbol_str() is the canonical const-char* color symbol from classes/colors.h.

// Present the player a menu of permanents and return the chosen entity. `verb`
// and `suffix` frame the label, e.g. ("Sacrifice ", "") or ("Return ", " to hand").
static Entity prompt_permanent_choice(const std::vector<Entity> &choices, const char *verb, const char *suffix,
                                      ActionCategory category) {
    std::vector<LegalAction> menu;
    for (auto e : choices) {
        std::string nm = global_coordinator.GetComponent<Permanent>(e).name;
        LegalAction la(PASS_PRIORITY, e, std::string(verb) + nm + suffix);
        la.category = category;
        menu.push_back(la);
    }
    int choice = InputLogger::instance().get_input(menu);
    return menu[static_cast<size_t>(choice)].source_entity;
}

// Pay the non-mana, non-tap activation costs shared by hand- and battlefield-
// activated abilities (life, sacrifice, return-to-hand, discard). Targets, tap,
// and mana are paid by the caller (those gate activation / can be cancelled);
// these costs cannot fail once legality has passed.
static void pay_secondary_activation_costs(
    const Ability &ability, Entity source, Zone::Ownership controller, std::shared_ptr<Orderer> orderer) {
    // Loyalty cost (606.4/606.5): pay by adding/removing loyalty counters on the source
    // planeswalker, and mark the per-permanent once-per-turn gate (606.3). The ability still
    // goes on the stack and resolves later; the loyalty change is the cost, paid now.
    if (ability.is_loyalty_ability && global_coordinator.entity_has_component<Permanent>(source)) {
        auto &perm = global_coordinator.GetComponent<Permanent>(source);
        int loyalty = add_counters(source, "LOYALTY", ability.loyalty_cost);
        perm.loyalty_ability_activated_this_turn = true;
        game_log("%s activates a loyalty ability (%+d, loyalty now %d)\n",
                 entity_name(source).c_str(), ability.loyalty_cost, loyalty);
    }
    // Life cost
    if (ability.life_cost > 0) {
        auto &activating_player = global_coordinator.GetComponent<Player>(get_player_entity(controller));
        activating_player.life_total -= ability.life_cost;
        game_log("%s pays %d life\n", player_name(controller).c_str(), ability.life_cost);
    }
    // Sacrifice self: move to graveyard; apply_permanent_components SBA removes Permanent next pass
    if (ability.sac_self) {
        std::string sname = entity_name(source);
        orderer->add_to_zone(false, source, Zone::GRAVEYARD);
        game_log("%s sacrifices %s\n", player_name(controller).c_str(), sname.c_str());
    }
    // Type-based sacrifice cost (Cycling "Sac a land", Knight of the Reliquary)
    if (!ability.sac_cost_spec.empty()) {
        std::vector<Entity> choices =
            controlled_permanents_matching(controller, ability.sac_cost_spec, orderer->mEntities);
        if (!choices.empty()) {
            Entity to_sac = prompt_permanent_choice(choices, "Sacrifice ", "", ActionCategory::SACRIFICE_PERMANENT);
            std::string sac_name = global_coordinator.GetComponent<Permanent>(to_sac).name;
            orderer->add_to_zone(false, to_sac, Zone::GRAVEYARD);
            game_log("%s sacrifices %s\n", player_name(controller).c_str(), sac_name.c_str());
        }
    }
    // Return-to-hand cost (Scryb Ranger: return a Forest to hand)
    if (!ability.return_cost_type.empty()) {
        std::vector<Entity> choices =
            controlled_permanents_matching(controller, ability.return_cost_type, orderer->mEntities);
        if (!choices.empty()) {
            Entity to_ret = prompt_permanent_choice(choices, "Return ", " to hand", ActionCategory::RETURN_PERMANENT);
            std::string ret_name = global_coordinator.GetComponent<Permanent>(to_ret).name;
            orderer->add_to_zone(false, to_ret, Zone::HAND);
            game_log("%s returns %s to hand\n", player_name(controller).c_str(), ret_name.c_str());
        }
    }
    // Discard self from hand cost (Faerie Macabre)
    if (ability.discard_self_cost) {
        std::string cname = global_coordinator.entity_has_component<CardData>(source)
            ? global_coordinator.GetComponent<CardData>(source).name : "card";
        orderer->add_to_zone(false, source, Zone::GRAVEYARD);
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
}

static void process_activate_ability(const LegalAction &action, Game &game, std::shared_ptr<Orderer> orderer) {
    Entity permanent_entity = action.source_entity;
    const Ability &ability = action.ability;

    // ActivationZone$ Hand: card in hand, no Permanent component
    if (ability.activation_zone == Zone::HAND &&
        !global_coordinator.entity_has_component<Permanent>(permanent_entity)) {
        auto &card_zone = global_coordinator.GetComponent<Zone>(permanent_entity);
        Zone::Ownership ctrl = card_zone.owner;
        Ability stack_ab = ability;

        // Select targets before paying costs
        if (stack_ab.valid_tgts != "N_A") {
            select_target(stack_ab, orderer, ctrl);
        }
        // Pay mana cost
        if (!ability.activation_mana_cost.empty()) {
            auto mana_snap = snapshot_mana_state(ctrl, orderer);
            if (!prompt_mana_payment(ctrl, ability.activation_mana_cost, permanent_entity, orderer)) {
                restore_mana_state(ctrl, mana_snap, orderer);
                cur_game.payment_fail_counts[permanent_entity]++;
                game_log("Payment cancelled.\n");
                return;
            }
        }
        // Pay remaining costs (life, sacrifice, return-to-hand, discard)
        pay_secondary_activation_costs(ability, permanent_entity, ctrl, orderer);
        // Auto-consume the activated card to the graveyard, unless the ability relocates
        // it itself (Defined$ Self) or it was already discarded as an explicit cost
        // (discard_self_cost, paid above) — guarding against a double move.
        if (!ability.defined_self && !ability.discard_self_cost) {
            orderer->add_to_zone(false, permanent_entity, Zone::GRAVEYARD);
        }

        // Create standalone ability entity on the stack
        stack_ab.source = permanent_entity;
        stack_ab.controller = ctrl;
        orderer->push_ability_onto_stack(stack_ab, ctrl);

        auto &cd = global_coordinator.GetComponent<CardData>(permanent_entity);
        if (stack_ab.target != 0) {
            std::string tgt_name = target_display_name(cur_game, stack_ab.target);
            game_log("%s activates %s from hand targeting %s\n",
                player_name(ctrl).c_str(), cd.name.c_str(), tgt_name.c_str());
        } else {
            game_log("%s activates %s from hand\n", player_name(ctrl).c_str(), cd.name.c_str());
        }
        game.take_action();
        return;
    }

    auto &permanent = global_coordinator.GetComponent<Permanent>(permanent_entity);
    Zone::Ownership controller = permanent.controller;

    // InstantSpeed$ AddMana abilities (e.g. LED) are mana abilities too: they resolve off-stack.
    // The instant-speed timing restriction is enforced upstream (offered only at priority).
    bool is_mana_ability = (ability.category == "AddMana");
    Ability stack_ab = ability;  // not used for mana ability

    // EQUIP: special activated ability — attach equipment to a creature
    if (ability.category == "Equip") {
        // Present list of creatures controlled by the equipment owner
        std::vector<LegalAction> equip_targets;
        for (auto e : orderer->mEntities) {
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
            return;
        }
        // Pay equip cost
        if (ability.tap_cost) permanent.is_tapped = true;
        if (!ability.activation_mana_cost.empty()) {
            auto mana_snap = snapshot_mana_state(controller, orderer);
            if (!prompt_mana_payment(controller, ability.activation_mana_cost, permanent_entity, orderer)) {
                restore_mana_state(controller, mana_snap, orderer);
                if (ability.tap_cost) permanent.is_tapped = false;
                cur_game.payment_fail_counts[permanent_entity]++;
                game_log("Payment cancelled.\n");
                return;
            }
        }

        game_log("Choose creature to equip:\n");
        int choice = InputLogger::instance().get_input(equip_targets);
        Entity target_creature = equip_targets[static_cast<size_t>(choice)].source_entity;

        // Detach from previous creature if any
        if (permanent.equipped_to != 0 && global_coordinator.entity_has_component<Permanent>(permanent.equipped_to)) {
            global_coordinator.GetComponent<Permanent>(permanent.equipped_to).equipped_by = 0;
        }
        permanent.equipped_to = target_creature;
        global_coordinator.GetComponent<Permanent>(target_creature).equipped_by = permanent_entity;
        std::string tname = global_coordinator.GetComponent<Permanent>(target_creature).name;
        game_log("%s equipped to %s.\n", permanent.name.c_str(), tname.c_str());
        game.take_action();
        return;
    }

    // SELECT TARGETS BEFORE PAYING COSTS
    if (!is_mana_ability) {
        if (stack_ab.valid_tgts != "N_A") {
            select_target(stack_ab, orderer, controller);
        }
    }
    // Tap cost
    if (ability.tap_cost) {
        permanent.is_tapped = true;
    }
    // Mana cost
    if (!ability.activation_mana_cost.empty()) {
        auto mana_snap = snapshot_mana_state(controller, orderer);
        if (!prompt_mana_payment(controller, ability.activation_mana_cost, permanent_entity, orderer)) {
            restore_mana_state(controller, mana_snap, orderer);
            if (ability.tap_cost) permanent.is_tapped = false;
            cur_game.payment_fail_counts[permanent_entity]++;
            game_log("Payment cancelled.\n");
            return;
        }
    }
    // Pay remaining costs (life, sacrifice, return-to-hand, discard).
    // life_cost legality is gated upstream in can_afford_alt / determine_legal_actions.
    pay_secondary_activation_costs(ability, permanent_entity, controller, orderer);
    // MANA ABILITY
    if (is_mana_ability) {
        // Evaluate dynamic amount (e.g. Gaea's Cradle: Count$Valid Creature.YouCtrl)
        size_t mana_amount = ability.amount;
        if (!ability.dynamic_amount_expr.empty() &&
            ability.dynamic_amount_expr.find("Count$Valid Creature.YouCtrl") != std::string::npos) {
            mana_amount = 0;
            for (auto e : orderer->mEntities) {
                if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
                if (!global_coordinator.entity_has_component<Creature>(e)) continue;
                auto &sz = global_coordinator.GetComponent<Zone>(e);
                if (sz.location != Zone::BATTLEFIELD) continue;
                if (global_coordinator.GetComponent<Permanent>(e).controller == controller)
                    mana_amount++;
            }
        }
        Colors mana_color = ability.color;
        add_mana(controller, mana_color, mana_amount);
        game_log("%s tapped %s for %zu(%s)\n", player_name(controller).c_str(), permanent.name.c_str(),
            mana_amount, mana_symbol_str(mana_color));
        // priority does not pass

        // A mana ability may carry a SubAbility$ rider that is part of the same mana ability
        // and resolves immediately (off-stack) as the ability resolves — e.g. Ancient Tomb's
        // "Ancient Tomb deals 2 damage to you" (CR 605.1a, 606.3). Resolve each sub-ability
        // here with the source/controller of the mana ability.
        for (auto sub_ab : ability.subabilities) {
            sub_ab.source = permanent_entity;
            sub_ab.controller = controller;
            sub_ab.resolve(orderer);
        }

        // Increment activation counter if this ability has a limit
        increment_activation_count(permanent, ability);
    } else {  // ACTIVATED ABILITY THAT IS NOT A MANA ABILITY - GOES ON STACK
        // puts on stack; we have targets from earlier
        stack_ab.source = permanent_entity;
        stack_ab.controller = controller;
        orderer->push_ability_onto_stack(stack_ab, controller);

        if (stack_ab.target != 0) {
            std::string tgt_name = target_display_name(cur_game, stack_ab.target);
            game_log("%s's %s ability targeting %s is on the stack\n",
                player_name(controller).c_str(), permanent.name.c_str(), tgt_name.c_str());
        } else {
            game_log("%s's %s ability is on the stack\n", player_name(controller).c_str(), permanent.name.c_str());
        }
        game.take_action();

        // Increment activation counter for limited abilities (e.g. Scryb Ranger)
        increment_activation_count(permanent, ability);
        // if target remains legal checked at resolution
    }
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
    // or Life from the Loam targeting Land.YouCtrl): opponent's graveyard first, then
    // own. is_legal_target applies the type/owner filter, so YouCtrl effects only keep
    // the caster's own cards.
    // target_in_graveyard covers spells that target a graveyard card via a non-ChangeZone
    // vehicle (Surgical Extraction's SP$ Pump with TgtZone$ Graveyard).
    if (ability.target_in_graveyard ||
        (ability.category == "ChangeZone" && ability.origin == Zone::GRAVEYARD &&
         ability.destination != Zone::BATTLEFIELD)) {
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
static void pay_alternate_cost(const LegalAction &action, Game &game, std::shared_ptr<Orderer> orderer,
    const CardData &card_data, Entity spell_entity, Zone zone) {
    Zone::Ownership caster = zone.owner;
    Entity caster_entity = (caster == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
    auto &player = global_coordinator.GetComponent<Player>(caster_entity);

    // Free alt cost (e.g. Once Upon a Time first spell)
    if (card_data.alt_cost.is_free) {
        game_log("%s casts for free (alternate cost)\n", player_name(caster).c_str());
        return;
    }

    // mana portion of the alt cost (e.g. Evoke:R). Affordability is pre-verified by
    // can_afford_alt, so in machine mode this always succeeds.
    if (!card_data.alt_cost.mana_cost.empty()) {
        prompt_mana_payment(caster, card_data.alt_cost.mana_cost, spell_entity, orderer);
    }

    // life
    if (card_data.alt_cost.life_cost != 0) {
        player.life_total -= card_data.alt_cost.life_cost;
        game_log("%s pays %d life\n", player_name(caster).c_str(), card_data.alt_cost.life_cost);
    }
    // pitch cards — exile a card of the required color from hand
    Colors pitch_color = card_data.alt_cost.exile_from_hand_color;
    for (int i = 0; i < card_data.alt_cost.exile_from_hand_count; i++) {
        std::vector<LegalAction> exile_actions;
        for (auto e : orderer->get_hand(caster)) {
            if (e == spell_entity) continue;
            if (!global_coordinator.entity_has_component<ColorIdentity>(e)) continue;
            if (pitch_color != NO_COLOR && !global_coordinator.GetComponent<ColorIdentity>(e).colors.count(pitch_color)) continue;
            LegalAction la(PASS_PRIORITY, e, "Exile " + global_coordinator.GetComponent<CardData>(e).name);
            la.category = ActionCategory::PAYING_COSTS;
            exile_actions.push_back(la);
        }
        int choice = InputLogger::instance().get_input(exile_actions);
        Entity exiled = exile_actions[static_cast<size_t>(choice)].source_entity;
        game_log("%s exiles %s\n", player_name(caster).c_str(),
            global_coordinator.GetComponent<CardData>(exiled).name.c_str());
        orderer->add_to_zone(false, exiled, Zone::EXILE);
    }
    // Is generalizable by type? I think
    for (int i = 0; i < card_data.alt_cost.return_to_hand_count; i++) {
        std::vector<LegalAction> rth_actions;
        const std::string &type = card_data.alt_cost.return_to_hand_type;
        for (auto e : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
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
        int choice = InputLogger::instance().get_input(rth_actions);
        Entity returned = rth_actions[static_cast<size_t>(choice)].source_entity;
        game_log("%s returns %s to hand\n", player_name(caster).c_str(),
            global_coordinator.GetComponent<Permanent>(returned).name.c_str());
        orderer->add_to_zone(false, returned, Zone::HAND);
    }
}

static void declare_attackers(Game &game, std::shared_ptr<Orderer> orderer) {
    Zone::Ownership active_player = game.player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B;
    Entity defending_entity = game.player_a_turn ? game.player_b_entity : game.player_a_entity;

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
        int creature_choice = InputLogger::instance().get_input(atk_actions);

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
        int target_choice = 0;
        if (tgt_actions.size() > 1) {
            game_log("Select target for %s:\n", chosen_name.c_str());
            target_choice = InputLogger::instance().get_input(tgt_actions);
        }
        cr.is_attacking = true;
        cr.attack_target = tgt_actions[static_cast<size_t>(target_choice)].source_entity;
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

static void declare_blockers(Game &game, std::shared_ptr<Orderer> orderer) {
    Zone::Ownership defending_player = game.player_a_turn ? Zone::PLAYER_B : Zone::PLAYER_A;
    // defending player declares blockers — priority must be theirs for the input routing to work correctly
    game.player_a_has_priority = !game.player_a_turn;

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
        int blocker_choice = InputLogger::instance().get_input(blk_actions);

        if (blocker_choice == static_cast<int>(blk_actions.size()) - 1) break;

        Entity chosen = unblocked[static_cast<size_t>(blocker_choice)];
        auto &cr = global_coordinator.GetComponent<Creature>(chosen);
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
        int attacker_choice = InputLogger::instance().get_input(blk_tgt_actions);
        cr.is_blocking = true;
        cr.blocking_target = blk_tgt_actions[static_cast<size_t>(attacker_choice)].source_entity;
        // Mark the attacker as blocked. It stays blocked for the rest of combat even if this
        // (and every other) blocker later leaves combat (509.1h), so it assigns no damage to
        // the player unless it has trample.
        if (global_coordinator.entity_has_component<Creature>(cr.blocking_target))
            global_coordinator.GetComponent<Creature>(cr.blocking_target).is_blocked = true;
        game_log("%s blocking %s.\n", chosen_name.c_str(), entity_name(cr.blocking_target).c_str());
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

bool has_legal_targets(const Ability &ability, std::shared_ptr<Orderer> orderer) {
    if (ability.valid_tgts == "N_A") return true;
    if (ability.target_min == 0) return true;  // optional targeting always has "legal targets"
    // Ordering doesn't matter for existence check; use PLAYER_A as a placeholder.
    return !build_valid_targets(ability, orderer, Zone::PLAYER_A).empty();
}

static void select_single_target(Ability &ability, const std::vector<Entity> &valid_targets,
                                  bool allow_done) {
    game_log("Choose target:\n");
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
    int choice = InputLogger::instance().get_input(tgt_actions);
    ability.target = tgt_actions[static_cast<size_t>(choice)].source_entity;
    game_log("Targeting choice %d\n", choice);
}

void select_target(Ability &ability, std::shared_ptr<Orderer> orderer, Zone::Ownership priority_player) {
    std::vector<Entity> valid_targets = build_valid_targets(ability, orderer, priority_player);

    if (ability.target_max <= 1) {
        select_single_target(ability, valid_targets, false);
        return;
    }

    // Multi-target selection loop
    ability.targets.clear();
    for (int i = 0; i < ability.target_max; i++) {
        if (valid_targets.empty()) break;
        bool can_stop = (i >= ability.target_min);
        select_single_target(ability, valid_targets, can_stop);
        if (ability.target == 0) break;  // chose "Done" or "No target"
        ability.targets.push_back(ability.target);
        // Remove chosen target from pool
        valid_targets.erase(
            std::remove(valid_targets.begin(), valid_targets.end(), ability.target),
            valid_targets.end());
    }
    // Set primary target to first chosen (for backward compat)
    if (!ability.targets.empty()) ability.target = ability.targets[0];
}

void process_action(const LegalAction &action, Game &game, std::shared_ptr<Orderer> orderer) {
    switch (action.type) {
        case PASS_PRIORITY:
            game.pass_priority();
            break;

        case SPECIAL_ACTION: {
            // Play land
            Entity land_entity = action.source_entity;
            auto &zone = global_coordinator.GetComponent<Zone>(land_entity);
            auto &card_data = global_coordinator.GetComponent<CardData>(land_entity);

            // Move to battlefield
            orderer->add_to_zone(false, land_entity, Zone::BATTLEFIELD);
            zone.controller = zone.owner;

            // Permanent component added by apply_permanent_components on next SBA pass

            // Update player's lands played counter
            Entity player_entity = get_player_entity(zone.owner);
            auto &player = global_coordinator.GetComponent<Player>(player_entity);
            player.lands_played_this_turn++;

            game_log("%s played %s\n", player_name(zone.owner).c_str(), card_data.name.c_str());

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
            auto &card_data = global_coordinator.GetComponent<CardData>(spell_entity);
            Zone::Ownership caster = zone.owner;

            // Snapshot mana state for rewind on payment failure
            auto mana_snap = snapshot_mana_state(caster, orderer);

            // FLASHBACK COST
            if (action.use_flashback) {
                // Pay flashback mana cost
                if (!card_data.flashback_mana_cost.empty()) {
                    if (!prompt_mana_payment(caster, card_data.flashback_mana_cost, spell_entity, orderer, false)) {
                        restore_mana_state(caster, mana_snap, orderer);
                        cur_game.payment_fail_counts[spell_entity]++;
                        game_log("Payment cancelled.\n");
                        break;
                    }
                }
                // Pay flashback life cost
                if (card_data.flashback_alt_cost.life_cost > 0) {
                    Entity caster_entity = (caster == Zone::PLAYER_A)
                        ? cur_game.player_a_entity : cur_game.player_b_entity;
                    auto &player = global_coordinator.GetComponent<Player>(caster_entity);
                    player.life_total -= card_data.flashback_alt_cost.life_cost;
                    game_log("%s pays %d life\n", player_name(caster).c_str(), card_data.flashback_alt_cost.life_cost);
                }

            // ALTERNATE COST
            } else if (action.use_alt_cost) {
                pay_alternate_cost(action, game, orderer, card_data, spell_entity, zone);

            } else {  // REGULAR COST + DELVE
                // RaiseCost surcharge (NamedCard-aware) folded in; shared with legality.
                ManaValue cost_to_pay = effective_base_cost(card_data);

                // Offspring (CR 702.171): additional cost paid on top of the spell's cost.
                if (action.use_offspring)
                    for (Colors c : card_data.offspring_cost) cost_to_pay.insert(c);

                // X-COST: prompt player to choose X value, add X generic to cost
                if (card_data.has_x_cost) {
                    size_t max_x = max_available_mana(caster, cost_to_pay, orderer);

                    game_log("Choose X value (0-%zu):\n", max_x);
                    std::vector<LegalAction> x_actions;
                    for (size_t xv = 0; xv <= max_x; xv++) {
                        LegalAction la(PASS_PRIORITY, std::string("X = " + std::to_string(xv)));
                        la.category = ActionCategory::CHOOSE_X;
                        x_actions.push_back(la);
                    }
                    int x_choice = InputLogger::instance().get_input(x_actions);
                    size_t x_val = static_cast<size_t>(x_choice);
                    cur_game.x_paid = x_val;
                    for (size_t i = 0; i < x_val; i++) cost_to_pay.insert(GENERIC);
                    game_log("%s chooses X = %zu\n", player_name(caster).c_str(), x_val);
                }

                // Phyrexian mana: for each symbol, choose to pay colored mana or 2 life
                if (!card_data.phyrexian_mana.empty()) {
                    Entity caster_entity = (caster == Zone::PLAYER_A)
                        ? cur_game.player_a_entity : cur_game.player_b_entity;
                    auto &phyrex_player = global_coordinator.GetComponent<Player>(caster_entity);
                    for (Colors phyrex_color : card_data.phyrexian_mana) {
                        std::string color_name = mana_symbol_str(phyrex_color);
                        std::vector<LegalAction> phyrex_actions;
                        LegalAction pay_life(PASS_PRIORITY, "Pay 2 life (instead of {" + color_name + "})");
                        pay_life.category = ActionCategory::PAYING_COSTS;
                        phyrex_actions.push_back(pay_life);
                        LegalAction pay_mana(PASS_PRIORITY, "Pay {" + color_name + "}");
                        pay_mana.category = ActionCategory::PAYING_COSTS;
                        phyrex_actions.push_back(pay_mana);
                        int phyrex_choice = InputLogger::instance().get_input(phyrex_actions);
                        if (phyrex_choice == 0) {
                            phyrex_player.life_total -= 2;
                            game_log("%s pays 2 life\n", player_name(caster).c_str());
                        } else {
                            cost_to_pay.insert(phyrex_color);
                        }
                    }
                }

                if (card_data.has_delve) cur_game.delve_exiled.clear();
                if (!prompt_mana_payment(caster, cost_to_pay, spell_entity, orderer, card_data.has_delve)) {
                    restore_mana_state(caster, mana_snap, orderer);
                    cur_game.payment_fail_counts[spell_entity]++;
                    game_log("Payment cancelled.\n");
                    break;
                }
            }

            // Find the primary spell ability template and copy it onto the entity
            for (const auto &ability_template : card_data.abilities) {
                if (ability_template.ability_type != Ability::SPELL) continue;

                Ability ability = ability_template;
                ability.source = spell_entity;
                ability.controller = caster;

                // Handle targeting
                if (ability.valid_tgts != "N_A") {
                    select_target(ability, orderer, caster);
                }

                global_coordinator.AddComponent(spell_entity, ability);
                break;  // TODO: support spells with multiple abilities
            }

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
            spell.cast_with_flashback = action.use_flashback;
            spell.cast_with_evoke = action.use_alt_cost && card_data.alt_cost.is_evoke;
            spell.cast_with_offspring = action.use_offspring;
            if (cur_game.pending_cant_be_countered) {
                spell.cant_be_countered = true;
                cur_game.pending_cant_be_countered = false;
            }
            // Check card's own replacement effects for "can't be countered" (Long Goodbye)
            for (const auto &r : card_data.replacement_effects) {
                if (r.kind == Effect::Replacement::CANT_BE_COUNTERED) {
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
                Event spell_event(Events::SPELL_CAST);
                spell_event.SetParam(Params::PLAYER, caster_entity);
                global_coordinator.SendEvent(spell_event);
            }

            game.take_action();
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

// Prompt the attacking player (rule 510.1c) to pick which blockers receive lethal damage, one
// at a time, until power runs out. Records the per-blocker assignment in
// game.combat_damage_assignment for deal_combat_damage() to apply.
static void assign_combat_damage(Game &game, std::shared_ptr<Orderer> orderer) {
    bool first_strike_only = (game.cur_step == FIRST_STRIKE_DAMAGE);
    // The attacking (active) player chooses the division — route input to them.
    game.player_a_has_priority = game.player_a_turn;
    game.combat_damage_assignment.clear();  // per strike step; survivors re-decide next step

    for (auto attacker : orderer->mEntities) {
        if (!attacker_needs_assignment(attacker, orderer, first_strike_only)) continue;
        auto &acr = global_coordinator.GetComponent<Creature>(attacker);
        std::string attacker_name = entity_name(attacker);

        std::vector<Entity> blockers = collect_live_blockers(attacker, orderer);
        auto &assign = game.combat_damage_assignment[attacker];  // creates the entry (also the guard)
        uint32_t remaining = acr.power;
        std::vector<Entity> pool = blockers;  // blockers not yet assigned lethal
        Entity last_assigned = 0;

        while (true) {
            // Offer only blockers we can still assign lethal to with the remaining damage.
            std::vector<LegalAction> actions;
            std::vector<Entity> offered;
            for (auto b : pool) {
                uint32_t need = lethal_needed_for_blocker(attacker, b);
                if (need == 0 || need > remaining) continue;
                auto &bcr = global_coordinator.GetComponent<Creature>(b);
                LegalAction la(PASS_PRIORITY, b,
                    entity_name(b) + " [" + std::to_string(bcr.power) + "/" +
                        std::to_string(bcr.toughness) + "] (lethal " + std::to_string(need) + ")");
                la.category = ActionCategory::ASSIGN_DAMAGE;
                actions.push_back(la);
                offered.push_back(b);
            }
            if (offered.empty()) break;  // can't kill any more blockers
            LegalAction done(PASS_PRIORITY, std::string("Done assigning ") + attacker_name);
            done.category = ActionCategory::ASSIGN_DAMAGE;
            actions.push_back(done);

            game_log("\n--- Assign %s's combat damage (%u left) ---\n", attacker_name.c_str(), remaining);
            int choice = InputLogger::instance().get_input(actions);
            if (choice == static_cast<int>(actions.size()) - 1) break;  // done

            Entity chosen = offered[static_cast<size_t>(choice)];
            uint32_t need = lethal_needed_for_blocker(attacker, chosen);
            assign[chosen] = need;
            remaining -= need;
            last_assigned = chosen;
            pool.erase(std::remove(pool.begin(), pool.end(), chosen), pool.end());
            game_log("  %s assigns %u (lethal) to %s\n", attacker_name.c_str(), need,
                     entity_name(chosen).c_str());
        }

        // 510.1a: all the attacker's power must be assigned among its blockers (no trample reaches
        // the player in the prompt case, since power <= total lethal). Pour any leftover onto one
        // blocker — harmless overkill on a chosen one, or a non-lethal mark if none were killable.
        if (remaining > 0) {
            Entity dump = last_assigned ? last_assigned : blockers.front();
            assign[dump] += remaining;
            remaining = 0;
        }
    }

    game.pending_choice = NONE;
}

void proc_mandatory_choice(Game &game, std::shared_ptr<Orderer> orderer) {
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
            int choice = InputLogger::instance().get_input(discard_actions);
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
