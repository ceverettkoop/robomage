#include "action_processor.h"

#include <algorithm>
#include <cstdio>

#include "components/ability.h"
#include "components/carddata.h"
#include "components/color_identity.h"
#include "components/creature.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/spell.h"
#include "components/zone.h"
#include "debug.h"
#include "ecs/coordinator.h"
#include "ecs/entity.h"
#include "ecs/events.h"
#include "input_logger.h"
#include "mana_system.h"
#include "systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

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
    auto &card_data = global_coordinator.GetComponent<CardData>(permanent_entity);
    Zone::Ownership controller = permanent.controller;

    bool is_mana_ability = (ability.category == "AddMana");
    Ability stack_ab = ability;  // not used for manaabilty

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
        printf("%s pays %d life\n", player_name(controller).c_str(), ability.life_cost);
    }
    // Pay sacrifice cost: move to graveyard; apply_permanent_components SBA removes Permanent next pass
    if (ability.sac_self) {
        orderer->add_to_zone(false, permanent_entity, Zone::GRAVEYARD);
        printf("%s sacrifices %s\n", player_name(controller).c_str(), card_data.name.c_str());
    }
    // MANA ABILITY
    if (is_mana_ability) {
        Colors mana_color = ability.color;
        add_mana(controller, mana_color, ability.amount);
        printf("%s tapped %s for {%s}\n", player_name(controller).c_str(), card_data.name.c_str(),
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
        global_coordinator.AddComponent(ability_entity, stack_ab);

        printf("%s's %s ability is on the stack\n", player_name(controller).c_str(), card_data.name.c_str());
        game.take_action();
        // if target remains legal checked at resolution
    }
}

// Build the list of legal targets for an ability.
// Targets are sorted from the caster's perspective: opponent entities first (opponent
// player, then opponent's permanents in entity-ID order), followed by own entities
// (own player, then own permanents in entity-ID order).  This keeps action index 0
// pointing at the opponent player for burn spells regardless of which player is casting,
// which makes the action space symmetric and simplifies self-play training.
static std::vector<Entity> build_valid_targets(const Ability &ability, std::shared_ptr<Orderer> orderer,
                                               Zone::Ownership priority_player) {
    std::vector<Entity> valid_targets;
    const std::string &vt = ability.valid_tgts;

    if (ability.target_type == "Spell") {
        for (auto e : orderer->get_stack()) {
            if (global_coordinator.entity_has_component<Spell>(e)) valid_targets.push_back(e);
        }
        return valid_targets;
    }

    bool any = (vt == "Any");
    bool inc_players   = any || vt.find("Player")   != std::string::npos;
    bool inc_creatures = any || vt.find("Creature") != std::string::npos;
    bool inc_lands     = any || vt.find("Land")     != std::string::npos;
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
            if (inc_lands && global_coordinator.entity_has_component<CardData>(entity)) {
                auto &tcd = global_coordinator.GetComponent<CardData>(entity);
                bool is_land = false;
                bool is_basic = false;
                for (auto &t : tcd.types) {
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

bool has_legal_targets(const Ability &ability, std::shared_ptr<Orderer> orderer) {
    if (ability.valid_tgts == "N_A") return true;
    // Ordering doesn't matter for existence check; use PLAYER_A as a placeholder.
    return !build_valid_targets(ability, orderer, Zone::PLAYER_A).empty();
}

void select_target(Ability &ability, std::shared_ptr<Orderer> orderer, Zone::Ownership priority_player) {
    std::vector<Entity> valid_targets = build_valid_targets(ability, orderer, priority_player);
    while (true) {
        printf("Choose target:\n");
        for (size_t i = 0; i < valid_targets.size(); i++) {
            Entity target = valid_targets[i];
            if (global_coordinator.entity_has_component<Player>(target)) {
                auto &player = global_coordinator.GetComponent<Player>(target);
                std::string name = (target == cur_game.player_a_entity) ? "Player A" : "Player B";
                printf("  %zu: %s (%d life)\n", i, name.c_str(), player.life_total);
            } else {
                auto &card = global_coordinator.GetComponent<CardData>(target);
                printf("  %zu: %s\n", i, card.name.c_str());
            }
        }
        std::vector<ActionCategory> tgt_cats(valid_targets.size(), ActionCategory::SELECT_TARGET);
        int choice = InputLogger::instance().get_logged_input(cur_game.turn, tgt_cats, valid_targets);
        if (choice >= 0 && choice < static_cast<int>(valid_targets.size())) {
            ability.target = valid_targets[static_cast<size_t>(choice)];
            printf("Targeting choice %d\n", choice);
            return;
        }
        printf("Invalid target, must choose a legal target.\n");
    }
}

//TODO MAKE THIS GENERAL
static void pay_alternate_cost(const LegalAction &action, Game &game, std::shared_ptr<Orderer> orderer,
    const CardData &card_data, Entity spell_entity, Zone zone) {
    Zone::Ownership caster = zone.owner;
    Entity caster_entity = (caster == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
    auto &player = global_coordinator.GetComponent<Player>(caster_entity);

    player.life_total -= card_data.alt_cost.life_cost;
    printf("%s pays %d life\n", player_name(caster).c_str(), card_data.alt_cost.life_cost);
    for (int i = 0; i < card_data.alt_cost.exile_blue_from_hand; i++) {
        std::vector<ActionCategory> exile_cats;
        std::vector<Entity> exile_entities;
        for (auto e : orderer->get_hand(caster)) {
            if (e == spell_entity) continue;
            if (!global_coordinator.entity_has_component<ColorIdentity>(e)) continue;
            if (!global_coordinator.GetComponent<ColorIdentity>(e).colors.count(BLUE)) continue;
            printf("  %zu: %s\n", exile_entities.size(), global_coordinator.GetComponent<CardData>(e).name.c_str());
            exile_cats.push_back(ActionCategory::OTHER_CHOICE);
            exile_entities.push_back(e);
        }
        int choice = InputLogger::instance().get_logged_input(cur_game.turn, exile_cats, exile_entities);
        if (choice >= 0 && choice < static_cast<int>(exile_entities.size())) {
            Entity exiled = exile_entities[static_cast<size_t>(choice)];
            printf("%s exiles %s\n", player_name(caster).c_str(),
                global_coordinator.GetComponent<CardData>(exiled).name.c_str());
            orderer->add_to_zone(false, exiled, Zone::EXILE);
        }
    }
    for (int i = 0; i < card_data.alt_cost.return_to_hand_count; i++) {
        std::vector<ActionCategory> rth_cats;
        std::vector<Entity> rth_entities;
        const std::string &sub = card_data.alt_cost.return_to_hand_subtype;
        for (auto e : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
            auto &perm = global_coordinator.GetComponent<Permanent>(e);
            if (perm.controller != caster) continue;
            auto &ecd = global_coordinator.GetComponent<CardData>(e);
            bool matches = false;
            for (auto &t : ecd.types) {
                if (t.kind == SUBTYPE && t.name == sub) {
                    matches = true;
                    break;
                }
            }
            if (!matches) continue;
            printf("  %zu: %s\n", rth_entities.size(), ecd.name.c_str());
            rth_cats.push_back(ActionCategory::OTHER_CHOICE);
            rth_entities.push_back(e);
        }
        int choice = InputLogger::instance().get_logged_input(cur_game.turn, rth_cats, rth_entities);
        if (choice >= 0 && choice < static_cast<int>(rth_entities.size())) {
            Entity returned = rth_entities[static_cast<size_t>(choice)];
            printf("%s returns %s to hand\n", player_name(caster).c_str(),
                global_coordinator.GetComponent<CardData>(returned).name.c_str());
            orderer->add_to_zone(false, returned, Zone::HAND);
        }
    }
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

            printf("%s played %s\n", player_name(zone.owner).c_str(), card_data.name.c_str());

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

            } else {  // REGULAR COST
                spend_mana(caster, card_data.mana_cost);
            }

            // Find the primary spell ability template and copy it onto the entity
            for (const auto &ability_template : card_data.abilities) {
                if (ability_template.ability_type != Ability::SPELL) continue;

                Ability ability = ability_template;
                ability.source = spell_entity;

                // Handle targeting
                if (ability.valid_tgts != "N_A") {
                    select_target(ability, orderer, caster);
                }

                global_coordinator.AddComponent(spell_entity, ability);
                break;  // TODO: support spells with multiple abilities
            }

            printf("%s casts %s\n", player_name(caster).c_str(), card_data.name.c_str());

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

            game.take_action();
            break;
        }
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
        if (permanent.has_summoning_sickness) continue;
        eligible.push_back(entity);
    }

    if (eligible.empty()) {
        printf("No creatures eligible to attack.\n");
        game.attackers_declared = true;
        game.pending_choice = NONE;
        return;
    }

    // Build targets: defending player first, then planeswalkers (TODO)
    std::vector<Entity> targets;
    targets.push_back(defending_entity);

    // Selection loop — only un-declared creatures are offered each iteration.
    // Once a creature is declared as an attacker it cannot be removed.
    while (true) {
        // Build list of creatures not yet declared as attackers
        std::vector<Entity> not_yet_attacking;
        for (auto entity : eligible) {
            auto &cr = global_coordinator.GetComponent<Creature>(entity);
            if (!cr.is_attacking) not_yet_attacking.push_back(entity);
        }

        printf("\n--- Declare Attackers (%s) ---\n", player_name(active_player).c_str());
        // Show already-declared attackers
        for (auto entity : eligible) {
            auto &cr = global_coordinator.GetComponent<Creature>(entity);
            if (!cr.is_attacking) continue;
            auto &cd = global_coordinator.GetComponent<CardData>(entity);
            Zone::Ownership t = (cr.attack_target == game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
            printf("  [attacking] %s [%d/%d] -> %s\n", cd.name.c_str(), cr.power, cr.toughness, player_name(t).c_str());
        }
        // Show available choices (not-yet-attacking creatures)
        for (size_t i = 0; i < not_yet_attacking.size(); i++) {
            auto &cd = global_coordinator.GetComponent<CardData>(not_yet_attacking[i]);
            auto &cr = global_coordinator.GetComponent<Creature>(not_yet_attacking[i]);
            printf("  %zu: %s [%d/%d]\n", i, cd.name.c_str(), cr.power, cr.toughness);
        }
        printf("  -1: Confirm\n");

        // num_choices = not-yet-attacking creatures + 1 implicit confirm (-1)
        std::vector<ActionCategory> atk_cats(not_yet_attacking.size(), ActionCategory::SELECT_ATTACKER);
        atk_cats.push_back(ActionCategory::CONFIRM_ATTACKERS);
        std::vector<Entity> atk_ents(not_yet_attacking.begin(), not_yet_attacking.end());
        atk_ents.push_back(Entity(0));  // confirm slot — null sentinel
        int creature_choice = InputLogger::instance().get_logged_input(cur_game.turn, atk_cats, atk_ents);

        if (creature_choice == -1) break;

        if (creature_choice < 0 || creature_choice >= static_cast<int>(not_yet_attacking.size())) {
            printf("Invalid selection.\n");
            continue;
        }

        auto &cr = global_coordinator.GetComponent<Creature>(not_yet_attacking[static_cast<size_t>(creature_choice)]);
        auto &cd = global_coordinator.GetComponent<CardData>(not_yet_attacking[static_cast<size_t>(creature_choice)]);

        printf("Select target for %s:\n", cd.name.c_str());
        for (size_t i = 0; i < targets.size(); i++) {
            auto &player = global_coordinator.GetComponent<Player>(targets[i]);
            Zone::Ownership t = (targets[i] == game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
            printf("  %zu: %s (%d life)\n", i, player_name(t).c_str(), player.life_total);
            // TODO: planeswalker entries here
        }

        std::vector<ActionCategory> atk_tgt_cats(targets.size(), ActionCategory::OTHER_CHOICE);
        int target_choice = InputLogger::instance().get_logged_input(cur_game.turn, atk_tgt_cats, targets);

        if (target_choice >= 0 && target_choice < static_cast<int>(targets.size())) {
            cr.is_attacking = true;
            cr.attack_target = targets[static_cast<size_t>(target_choice)];
            Zone::Ownership t = (cr.attack_target == game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
            printf("%s attacking %s.\n", cd.name.c_str(), player_name(t).c_str());
        } else {
            printf("Invalid target.\n");
        }
    }

    printf("\nAttackers declared:\n");
    bool any = false;
    for (auto entity : eligible) {
        auto &cr = global_coordinator.GetComponent<Creature>(entity);
        if (cr.is_attacking) {
            any = true;
            auto &cd = global_coordinator.GetComponent<CardData>(entity);
            Zone::Ownership t = (cr.attack_target == game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
            printf("  %s -> %s\n", cd.name.c_str(), player_name(t).c_str());

            // Tap the attacker
            auto &permanent = global_coordinator.GetComponent<Permanent>(entity);
            permanent.is_tapped = true;
        }
    }
    if (!any) printf("  (none)\n");

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

    // Collect attackers
    std::vector<Entity> attackers;
    for (auto entity : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
        auto &cr = global_coordinator.GetComponent<Creature>(entity);
        if (cr.is_attacking) attackers.push_back(entity);
    }

    if (attackers.empty()) {
        printf("No attackers — skipping declare blockers.\n");
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
        printf("No creatures eligible to block.\n");
        game.blockers_declared = true;
        game.pending_choice = NONE;
        return;
    }

    // Selection loop
    while (true) {
        printf("\n--- Declare Blockers (%s) ---\n", player_name(defending_player).c_str());
        printf("Attackers:\n");
        for (size_t i = 0; i < attackers.size(); i++) {
            auto &cd = global_coordinator.GetComponent<CardData>(attackers[i]);
            auto &cr = global_coordinator.GetComponent<Creature>(attackers[i]);
            printf("  %zu: %s [%d/%d]\n", i, cd.name.c_str(), cr.power, cr.toughness);
        }
        printf("Your creatures:\n");
        for (size_t i = 0; i < eligible.size(); i++) {
            auto &cd = global_coordinator.GetComponent<CardData>(eligible[i]);
            auto &cr = global_coordinator.GetComponent<Creature>(eligible[i]);
            printf("  %zu: %s [%d/%d]", i, cd.name.c_str(), cr.power, cr.toughness);
            if (cr.is_blocking) {
                auto &attacker_cd = global_coordinator.GetComponent<CardData>(cr.blocking_target);
                printf(" blocking %s", attacker_cd.name.c_str());
            }
            printf("\n");
        }
        printf("  -1: Confirm\n");

        // num_choices = eligible blockers + 1 implicit confirm (-1)
        std::vector<ActionCategory> blk_cats(eligible.size(), ActionCategory::SELECT_BLOCKER);
        blk_cats.push_back(ActionCategory::CONFIRM_BLOCKERS);
        std::vector<Entity> blk_ents(eligible.begin(), eligible.end());
        blk_ents.push_back(Entity(0));  // confirm slot — null sentinel
        int blocker_choice = InputLogger::instance().get_logged_input(cur_game.turn, blk_cats, blk_ents);

        if (blocker_choice == -1) break;

        if (blocker_choice < 0 || blocker_choice >= static_cast<int>(eligible.size())) {
            printf("Invalid selection.\n");
            continue;
        }

        auto &cr = global_coordinator.GetComponent<Creature>(eligible[static_cast<size_t>(blocker_choice)]);
        auto &cd = global_coordinator.GetComponent<CardData>(eligible[static_cast<size_t>(blocker_choice)]);

        if (cr.is_blocking) {
            // Toggle off
            cr.is_blocking = false;
            cr.blocking_target = 0;
            printf("%s removed from blockers.\n", cd.name.c_str());
        } else {
            auto legal_attackers =
                determine_blockable_attackers(eligible[static_cast<size_t>(blocker_choice)], attackers);
            printf("Select attacker for %s to block:\n", cd.name.c_str());
            for (size_t i = 0; i < legal_attackers.size(); i++) {
                auto &acd = global_coordinator.GetComponent<CardData>(legal_attackers[i]);
                auto &acr = global_coordinator.GetComponent<Creature>(legal_attackers[i]);
                printf("  %zu: %s [%d/%d]\n", i, acd.name.c_str(), acr.power, acr.toughness);
            }

            std::vector<ActionCategory> blk_tgt_cats(legal_attackers.size(), ActionCategory::OTHER_CHOICE);
            int attacker_choice =
                InputLogger::instance().get_logged_input(cur_game.turn, blk_tgt_cats, legal_attackers);

            if (attacker_choice >= 0 && attacker_choice < static_cast<int>(legal_attackers.size())) {
                cr.is_blocking = true;
                cr.blocking_target = legal_attackers[static_cast<size_t>(attacker_choice)];
                auto &acd = global_coordinator.GetComponent<CardData>(cr.blocking_target);
                printf("%s blocking %s.\n", cd.name.c_str(), acd.name.c_str());
            } else {
                printf("Invalid attacker.\n");
            }
        }
    }

    printf("\nBlockers declared:\n");
    bool any = false;
    for (auto entity : eligible) {
        auto &cr = global_coordinator.GetComponent<Creature>(entity);
        if (cr.is_blocking) {
            any = true;
            auto &cd = global_coordinator.GetComponent<CardData>(entity);
            auto &acd = global_coordinator.GetComponent<CardData>(cr.blocking_target);
            printf("  %s blocking %s\n", cd.name.c_str(), acd.name.c_str());
        }
    }
    if (!any) printf("  (none)\n");

    game.blockers_declared = true;
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
        case CLEANUP_DISCARD: {
            Zone::Ownership active_player = game.player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B;
            auto hand = orderer->get_hand(active_player);

            while (true) {
                printf("\n--- Discard to hand size (%s) ---\n", player_name(active_player).c_str());
                printf("Hand (%zu cards, must discard to 7):\n", hand.size());
                for (size_t i = 0; i < hand.size(); i++) {
                    auto &cd = global_coordinator.GetComponent<CardData>(hand[i]);
                    printf("  %zu: %s\n", i, cd.name.c_str());
                }

                std::vector<ActionCategory> discard_cats(hand.size(), ActionCategory::OTHER_CHOICE);
                int choice = InputLogger::instance().get_logged_input(game.turn, discard_cats, hand);
                if (choice >= 0 && choice < static_cast<int>(hand.size())) {
                    Entity card = hand[static_cast<size_t>(choice)];
                    auto &cd = global_coordinator.GetComponent<CardData>(card);
                    orderer->add_to_zone(false, card, Zone::GRAVEYARD);
                    printf("%s discards %s.\n", player_name(active_player).c_str(), cd.name.c_str());
                    break;
                }
                printf("Invalid selection, must choose a card to discard.\n");
            }

            game.pending_choice = NONE;
            break;
        }
        case CHOOSE_ENTITY:
            printf("TODO: Choose entity\n");
            game.pending_choice = NONE;
            break;
        case NONE:
            break;
    }
}
