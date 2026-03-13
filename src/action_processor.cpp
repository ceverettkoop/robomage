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
#include "input_logger.h"
#include "mana_system.h"
#include "systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static std::string entity_name(Entity e);
static const char *mana_symbol_str(Colors color);
static void process_activate_ability(const LegalAction &action, Game &game, std::shared_ptr<Orderer> orderer);
static std::vector<Entity> build_valid_targets(
    const Ability &ability, std::shared_ptr<Orderer> orderer, Zone::Ownership priority_player);
static void pay_alternate_cost(const LegalAction &action, Game &game, std::shared_ptr<Orderer> orderer,
    const CardData &card_data, Entity spell_entity, Zone zone);
static void declare_attackers(Game &game, std::shared_ptr<Orderer> orderer);
static std::vector<Entity> determine_blockable_attackers(Entity blocker, const std::vector<Entity> &attackers);
static void declare_blockers(Game &game, std::shared_ptr<Orderer> orderer);

static std::string entity_name(Entity e) {
    if (global_coordinator.entity_has_component<Permanent>(e)) {
        auto &perm = global_coordinator.GetComponent<Permanent>(e);
        return perm.is_token ? perm.name + " token" : perm.name;
    }
    if (global_coordinator.entity_has_component<CardData>(e)) return global_coordinator.GetComponent<CardData>(e).name;
    if (global_coordinator.entity_has_component<Token>(e))
        return global_coordinator.GetComponent<Token>(e).name + " token";
    return "<unknown>";
}

static const char *mana_symbol_str(Colors color) {
    switch (color) {
        case WHITE:
            return "W";
        case BLUE:
            return "U";
        case BLACK:
            return "B";
        case RED:
            return "R";
        case GREEN:
            return "G";
        case COLORLESS:
            return "C";
        default:
            return "?";
    }
}

static void process_activate_ability(const LegalAction &action, Game &game, std::shared_ptr<Orderer> orderer) {
    Entity permanent_entity = action.source_entity;
    const Ability &ability = action.ability;

    auto &permanent = global_coordinator.GetComponent<Permanent>(permanent_entity);
    Zone::Ownership controller = permanent.controller;

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
        if (!ability.activation_mana_cost.empty()) spend_mana(controller, ability.activation_mana_cost);

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
        spend_mana(controller, ability.activation_mana_cost);
    }
    // Pay life cost //TODO VERIFY THIS WAS CHECKED AS POSSIBLE PRIOR TO HERE
    if (ability.life_cost > 0) {
        auto &activating_player = global_coordinator.GetComponent<Player>(get_player_entity(controller));
        activating_player.life_total -= ability.life_cost;
        game_log("%s pays %d life\n", player_name(controller).c_str(), ability.life_cost);
    }
    // Pay sacrifice cost: move to graveyard; apply_permanent_components SBA removes Permanent next pass
    if (ability.sac_self) {
        orderer->add_to_zone(false, permanent_entity, Zone::GRAVEYARD);
        game_log("%s sacrifices %s\n", player_name(controller).c_str(), permanent.name.c_str());
    }
    // MANA ABILITY
    if (is_mana_ability) {
        Colors mana_color = ability.color;
        add_mana(controller, mana_color, ability.amount);
        game_log("%s tapped %s for {%s}\n", player_name(controller).c_str(), permanent.name.c_str(),
            mana_symbol_str(mana_color));
        // priority does not pass

    } else {  // ACTIVATED ABILITY THAT IS NOT A MANA ABILITY - GOES ON STACK
        //  Initialize zone with HAND so add_to_zone removal of the origin zone is a no-op
        // lol that's hacky but OK
        Entity ability_entity = global_coordinator.CreateEntity();
        Zone ab_zone(Zone::HAND, controller, controller);
        global_coordinator.AddComponent(ability_entity, ab_zone);
        // puts on stack; we have targets from earlier
        orderer->add_to_zone(false, ability_entity, Zone::STACK);

        stack_ab.source = permanent_entity;
        stack_ab.controller = controller;
        global_coordinator.AddComponent(ability_entity, stack_ab);

        game_log("%s's %s ability is on the stack\n", player_name(controller).c_str(), permanent.name.c_str());
        game.take_action();
        // if target remains legal checked at resolution
    }
}

// TODO fix
// redundant with call in action processor and not generalizable!
//  Build the list of legal targets for an ability.
//  Targets are sorted from the caster's perspective: opponent entities first (opponent
//  player, then opponent's permanents in entity-ID order), followed by own entities
//  (own player, then own permanents in entity-ID order).  This keeps action index 0
//  pointing at the opponent player for burn spells regardless of which player is casting,
//  which makes the action space symmetric and simplifies self-play training.
static std::vector<Entity> build_valid_targets(
    const Ability &ability, std::shared_ptr<Orderer> orderer, Zone::Ownership priority_player) {
    std::vector<Entity> valid_targets;
    const std::string &vt = ability.valid_tgts;

    if (ability.target_type == "Spell") {
        for (auto e : orderer->get_stack()) {
            if (global_coordinator.entity_has_component<Spell>(e)) valid_targets.push_back(e);
        }
        return valid_targets;
    }

    bool any = (vt == "Any");
    bool inc_players = any || vt.find("Player") != std::string::npos;
    bool inc_creatures = any || vt.find("Creature") != std::string::npos;
    bool inc_lands = any || vt.find("Land") != std::string::npos;
    bool nonbasic_only = vt.find("nonBasic") != std::string::npos;
    // TODO: inc_planeswalker, inc_battle when those components exist

    Zone::Ownership opp = (priority_player == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;

    // Players: opponent first, self second
    if (inc_players) {
        valid_targets.push_back(get_player_entity(opp));
        valid_targets.push_back(get_player_entity(priority_player));
    }

    // Permanents: two passes — opponent's first, then own (entity-ID order within each group)
    for (int pass = 0; pass < 2; pass++) {
        Zone::Ownership slot_owner = (pass == 0) ? opp : priority_player;
        for (auto entity : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
            auto &tz = global_coordinator.GetComponent<Zone>(entity);
            if (tz.location != Zone::BATTLEFIELD) continue;
            if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
            if (global_coordinator.GetComponent<Permanent>(entity).controller != slot_owner) continue;

            if (inc_creatures && global_coordinator.entity_has_component<Creature>(entity)) {
                valid_targets.push_back(entity);
                continue;
            }
            if (inc_lands) {
                auto &tperm = global_coordinator.GetComponent<Permanent>(entity);
                bool is_land = false;
                bool is_basic = false;
                for (auto &t : tperm.types) {
                    if (t.kind == TYPE && t.name == "Land") is_land = true;
                    if (t.kind == SUPERTYPE && t.name == "Basic") is_basic = true;
                }
                if (is_land && (!nonbasic_only || !is_basic)) {
                    valid_targets.push_back(entity);
                    continue;
                }
            }
            // TODO: Planeswalker and Battle components
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

    // life
    if (card_data.alt_cost.life_cost != 0) {
        player.life_total -= card_data.alt_cost.life_cost;
        game_log("%s pays %d life\n", player_name(caster).c_str(), card_data.alt_cost.life_cost);
    }
    // pitch cards, currently just looks for blue, TODO make generalizable
    for (int i = 0; i < card_data.alt_cost.exile_blue_from_hand; i++) {
        std::vector<LegalAction> exile_actions;
        for (auto e : orderer->get_hand(caster)) {
            if (e == spell_entity) continue;
            if (!global_coordinator.entity_has_component<ColorIdentity>(e)) continue;
            if (!global_coordinator.GetComponent<ColorIdentity>(e).colors.count(BLUE)) continue;
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
            la.category = ActionCategory::OTHER_CHOICE;
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

    // Build targets: defending player first, then planeswalkers (TODO)
    std::vector<Entity> targets;
    targets.push_back(defending_entity);

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
            Zone::Ownership t = (cr.attack_target == game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
            game_log("  [attacking] %s [%d/%d] -> %s\n", ename.c_str(), cr.power, cr.toughness, player_name(t).c_str());
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

        game_log("Select target for %s:\n", chosen_name.c_str());
        std::vector<LegalAction> tgt_actions;
        for (auto t_entity : targets) {
            auto &player = global_coordinator.GetComponent<Player>(t_entity);
            Zone::Ownership t = (t_entity == game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
            LegalAction la(
                PASS_PRIORITY, t_entity, player_name(t) + " (" + std::to_string(player.life_total) + " life)");
            la.category = ActionCategory::OTHER_CHOICE;
            tgt_actions.push_back(la);
            // TODO: planeswalker entries here
        }
        int target_choice = InputLogger::instance().get_input(tgt_actions);
        cr.is_attacking = true;
        cr.attack_target = tgt_actions[static_cast<size_t>(target_choice)].source_entity;
        {
            Zone::Ownership t = (cr.attack_target == game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
            game_log("%s attacking %s.\n", chosen_name.c_str(), player_name(t).c_str());
        }
    }

    game_log("\nAttackers declared:\n");
    bool any = false;
    for (auto entity : eligible) {
        auto &cr = global_coordinator.GetComponent<Creature>(entity);
        if (cr.is_attacking) {
            any = true;
            std::string ename = entity_name(entity);
            Zone::Ownership t = (cr.attack_target == game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
            game_log("  %s -> %s\n", ename.c_str(), player_name(t).c_str());

            // Tap the attacker
            auto &permanent = global_coordinator.GetComponent<Permanent>(entity);
            permanent.is_tapped = true;
        }
    }
    if (!any) game_log("  (none)\n");

    game.attackers_declared = true;
    game.pending_choice = NONE;
}

static std::vector<Entity> determine_blockable_attackers(Entity blocker, const std::vector<Entity> &attackers) {
    auto &bcr = global_coordinator.GetComponent<Creature>(blocker);
    bool blocker_can_fly = false;
    for (auto &kw : bcr.keywords)
        if (kw == "Flying" || kw == "Reach") {
            blocker_can_fly = true;
            break;
        }

    std::vector<Entity> result;
    for (auto atk : attackers) {
        auto &acr = global_coordinator.GetComponent<Creature>(atk);
        bool atk_flying = false;
        for (auto &kw : acr.keywords)
            if (kw == "Flying") {
                atk_flying = true;
                break;
            }

        if (atk_flying && !blocker_can_fly) continue;
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
            la.category = ActionCategory::OTHER_CHOICE;
            blk_tgt_actions.push_back(la);
        }
        int attacker_choice = InputLogger::instance().get_input(blk_tgt_actions);
        cr.is_blocking = true;
        cr.blocking_target = blk_tgt_actions[static_cast<size_t>(attacker_choice)].source_entity;
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
    // Ordering doesn't matter for existence check; use PLAYER_A as a placeholder.
    return !build_valid_targets(ability, orderer, Zone::PLAYER_A).empty();
}

void select_target(Ability &ability, std::shared_ptr<Orderer> orderer, Zone::Ownership priority_player) {
    std::vector<Entity> valid_targets = build_valid_targets(ability, orderer, priority_player);
    game_log("Choose target:\n");
    std::vector<LegalAction> tgt_actions;
    for (auto target : valid_targets) {
        std::string desc;
        if (global_coordinator.entity_has_component<Player>(target)) {
            auto &player = global_coordinator.GetComponent<Player>(target);
            std::string name = (target == cur_game.player_a_entity) ? "Player A" : "Player B";
            desc = name + " (" + std::to_string(player.life_total) + " life)";
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

            // ALTERNATE COST
            if (action.use_alt_cost) {
                pay_alternate_cost(action, game, orderer, card_data, spell_entity, zone);

            } else {  // REGULAR COST + DELVE
                ManaValue cost_to_pay = card_data.mana_cost;

                // DELVE: exile instants/sorceries from graveyard to reduce generic cost
                if (card_data.has_delve && !can_afford(caster, cost_to_pay)) {
                    cur_game.delve_exiled.clear();
                    auto remaining_generic_to_pay = pay_partial(caster, cost_to_pay);
                    auto gen_ct = remaining_generic_to_pay.count(Colors::GENERIC);
                    game_log(
                        "%s: exile %zu cards from graveyard to pay generic cost (0=done):\n", player_name(caster).c_str(), gen_ct);
                    int gen_paid = 0;
                    while (gen_paid < gen_ct) {
                        // Build list of exilable instants/sorceries
                        std::vector<LegalAction> delve_actions;
                        //we can't be done until everything has been paid for, no early exit
                        //should not be allowed to get here unless we can afford it
                        for (auto e : orderer->mEntities) {
                            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
                            auto &ez = global_coordinator.GetComponent<Zone>(e);
                            if (ez.location != Zone::GRAVEYARD || ez.owner != caster) continue;
                            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
                            auto &ecd = global_coordinator.GetComponent<CardData>(e);
                            bool is_instant_or_sorcery = false;
                            for (auto &t : ecd.types)
                                if (t.kind == TYPE && (t.name == "Instant" || t.name == "Sorcery")) {
                                    is_instant_or_sorcery = true;
                                    break;
                                }
                            if (!is_instant_or_sorcery) continue;
                            LegalAction la(PASS_PRIORITY, e, ecd.name);
                            la.category = ActionCategory::PAYING_COSTS;
                            delve_actions.push_back(la);
                        }

                        if (delve_actions.size() < 1){
                            fatal_error("error paying delve costs, not enough delve cards");
                        }

                        int choice = InputLogger::instance().get_input(delve_actions);
                        Entity exiled = delve_actions[static_cast<size_t>(choice)].source_entity;
                        orderer->add_to_zone(false, exiled, Zone::EXILE);
                        cur_game.delve_exiled.push_back(exiled);
                        cost_to_pay.erase(cost_to_pay.find(GENERIC));
                        gen_paid++;
                        auto &ecd2 = global_coordinator.GetComponent<CardData>(exiled);
                        game_log("%s exiles %s via Delve.\n", player_name(caster).c_str(), ecd2.name.c_str());
                    }
                }

                spend_mana(caster, cost_to_pay);
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

            game_log("%s casts %s\n", player_name(caster).c_str(), card_data.name.c_str());

            // Add Spell component — present only while the entity is on the stack
            Spell spell;
            spell.caster = caster;
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
                Event spell_event(Events::SPELL_CAST);
                spell_event.SetParam(Params::PLAYER, caster_entity);
                global_coordinator.SendEvent(spell_event);
            }

            game.take_action();
            break;
        }
    }
}

void proc_mandatory_choice(Game &game, std::shared_ptr<Orderer> orderer) {
    switch (game.pending_choice) {
        case DECLARE_ATTACKERS_CHOICE:
            declare_attackers(game, orderer);
            break;
        case DECLARE_BLOCKERS_CHOICE:
            declare_blockers(game, orderer);
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
                la.category = ActionCategory::OTHER_CHOICE;
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
