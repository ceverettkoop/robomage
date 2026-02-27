#include "action_processor.h"

#include <cstdio>

#include "components/ability.h"
#include "components/carddata.h"
#include "components/creature.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/spell.h"
#include "components/zone.h"
#include "debug.h"
#include "ecs/coordinator.h"
#include "input_logger.h"
#include "mana_system.h"
#include "systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static const char* mana_symbol_str(Colors color) {
    switch (color) {
        case WHITE:    return "W";
        case BLUE:     return "U";
        case BLACK:    return "B";
        case RED:      return "R";
        case GREEN:    return "G";
        case COLORLESS: return "C";
        default:       return "?";
    }
}

static void process_activate_ability(const LegalAction& action, Game& game, std::shared_ptr<Orderer> orderer) {
    (void)orderer;
    Entity permanent_entity = action.source_entity;
    const Ability& ability = action.ability;

    auto& permanent = global_coordinator.GetComponent<Permanent>(permanent_entity);
    auto& card_data = global_coordinator.GetComponent<CardData>(permanent_entity);
    Zone::Ownership controller = permanent.controller;

    bool is_mana_ability = (ability.category == "AddMana");

    // Pay costs
    if (ability.tap_cost) {
        permanent.is_tapped = true;
    }
    if (!ability.activation_mana_cost.empty()) {
        spend_mana(controller, ability.activation_mana_cost);
    }

    if (is_mana_ability) {
        Colors mana_color = ability.color;
        add_mana(controller, mana_color, ability.amount);
        printf("%s tapped %s for {%s}\n",
               player_name(controller).c_str(),
               card_data.name.c_str(),
               mana_symbol_str(mana_color));
        // Mana abilities don't use the stack; priority does not pass
    } else {
        // Non-mana activated ability
        // TODO: put on stack for opponents to respond; for now resolve immediately
        printf("%s activates %s\n", player_name(controller).c_str(), card_data.name.c_str());
        game.take_action();
    }
}

void process_action(const LegalAction& action, Game& game, std::shared_ptr<Orderer> orderer) {

    switch (action.type) {
        case PASS_PRIORITY:
            game.pass_priority();
            break;

        case SPECIAL_ACTION: {
            // Play land
            Entity land_entity = action.source_entity;
            auto& zone = global_coordinator.GetComponent<Zone>(land_entity);
            auto& card_data = global_coordinator.GetComponent<CardData>(land_entity);

            // Move to battlefield
            orderer->add_to_zone(false, land_entity, Zone::BATTLEFIELD);
            zone.controller = zone.owner;

            // Add Permanent component
            Permanent permanent;
            permanent.controller = zone.owner;
            permanent.has_summoning_sickness = false;  // Lands don't have summoning sickness
            permanent.is_tapped = false;
            permanent.timestamp_entered_battlefield = game.timestamp++;
            global_coordinator.AddComponent(land_entity, permanent);

            // Update player's lands played counter
            Entity player_entity = get_player_entity(zone.owner);
            auto& player = global_coordinator.GetComponent<Player>(player_entity);
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
            auto& zone = global_coordinator.GetComponent<Zone>(spell_entity);
            auto& card_data = global_coordinator.GetComponent<CardData>(spell_entity);

            Zone::Ownership caster = zone.owner;

            // Pay mana cost
            spend_mana(caster, card_data.mana_cost);

            // Find the primary spell ability template and copy it onto the entity
            for (const auto& ability_template : card_data.abilities) {
                if (ability_template.ability_type != Ability::SPELL) continue;

                Ability ability = ability_template;
                ability.source = spell_entity;

                // Handle targeting
                if (ability.valid_tgts != "N_A") {
                    // Build target list: players, creatures, TODO: planeswalkers, battles
                    std::vector<Entity> valid_targets;

                    valid_targets.push_back(cur_game.player_a_entity);
                    valid_targets.push_back(cur_game.player_b_entity);

                    for (auto entity : orderer->mEntities) {
                        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
                        auto& target_zone = global_coordinator.GetComponent<Zone>(entity);
                        if (target_zone.location != Zone::BATTLEFIELD) continue;

                        if (global_coordinator.entity_has_component<Creature>(entity)) {
                            valid_targets.push_back(entity);
                            continue;
                        }
                        // TODO: check for Planeswalker component when implemented
                        // TODO: check for Battle component when implemented
                    }

                    while (true) {
                        printf("Choose target:\n");
                        for (size_t i = 0; i < valid_targets.size(); i++) {
                            Entity target = valid_targets[i];
                            if (global_coordinator.entity_has_component<Player>(target)) {
                                auto& player = global_coordinator.GetComponent<Player>(target);
                                std::string name = (target == cur_game.player_a_entity) ? "Player A" : "Player B";
                                printf("  %zu: %s (%d life)\n", i, name.c_str(), player.life_total);
                            } else {
                                auto& target_card = global_coordinator.GetComponent<CardData>(target);
                                printf("  %zu: %s\n", i, target_card.name.c_str());
                            }
                        }

                        std::vector<ActionCategory> tgt_cats(valid_targets.size(), ActionCategory::SELECT_TARGET);
                        int target_choice = InputLogger::instance().get_logged_input(cur_game.turn, tgt_cats);
                        if (target_choice >= 0 && target_choice < static_cast<int>(valid_targets.size())) {
                            ability.target = valid_targets[static_cast<size_t>(target_choice)];
                            printf("Targeting choice %d\n", target_choice);
                            break;
                        }
                        printf("Invalid target, must choose a legal target.\n");
                    }
                }

                global_coordinator.AddComponent(spell_entity, ability);
                break; // TODO: support spells with multiple abilities
            }

            printf("%s casts %s\n", player_name(caster).c_str(), card_data.name.c_str());

            // Add Spell component — present only while the entity is on the stack
            Spell spell;
            spell.caster = caster;
            global_coordinator.AddComponent(spell_entity, spell);

            // Move to stack
            orderer->add_to_zone(false, spell_entity, Zone::STACK);  // Top of stack

            game.take_action();
            break;
        }
    }
}

static void declare_attackers(Game& game, std::shared_ptr<Orderer> orderer) {
    Zone::Ownership active_player = game.player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B;
    Entity defending_entity = game.player_a_turn ? game.player_b_entity : game.player_a_entity;

    // Collect eligible attackers with stable indices
    std::vector<Entity> eligible;
    for (auto entity : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
        auto& permanent = global_coordinator.GetComponent<Permanent>(entity);
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
            auto& cr = global_coordinator.GetComponent<Creature>(entity);
            if (!cr.is_attacking) not_yet_attacking.push_back(entity);
        }

        printf("\n--- Declare Attackers (%s) ---\n", player_name(active_player).c_str());
        // Show already-declared attackers
        for (auto entity : eligible) {
            auto& cr = global_coordinator.GetComponent<Creature>(entity);
            if (!cr.is_attacking) continue;
            auto& cd = global_coordinator.GetComponent<CardData>(entity);
            Zone::Ownership t = (cr.attack_target == game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
            printf("  [attacking] %s [%d/%d] -> %s\n", cd.name.c_str(), cr.power, cr.toughness, player_name(t).c_str());
        }
        // Show available choices (not-yet-attacking creatures)
        for (size_t i = 0; i < not_yet_attacking.size(); i++) {
            auto& cd = global_coordinator.GetComponent<CardData>(not_yet_attacking[i]);
            auto& cr = global_coordinator.GetComponent<Creature>(not_yet_attacking[i]);
            printf("  %zu: %s [%d/%d]\n", i, cd.name.c_str(), cr.power, cr.toughness);
        }
        printf("  -1: Confirm\n");

        // num_choices = not-yet-attacking creatures + 1 implicit confirm (-1)
        std::vector<ActionCategory> atk_cats(not_yet_attacking.size(), ActionCategory::SELECT_ATTACKER);
        atk_cats.push_back(ActionCategory::CONFIRM_ATTACKERS);
        int creature_choice = InputLogger::instance().get_logged_input(cur_game.turn, atk_cats);

        if (creature_choice == -1) break;

        if (creature_choice < 0 || creature_choice >= static_cast<int>(not_yet_attacking.size())) {
            printf("Invalid selection.\n");
            continue;
        }

        auto& cr = global_coordinator.GetComponent<Creature>(not_yet_attacking[static_cast<size_t>(creature_choice)]);
        auto& cd = global_coordinator.GetComponent<CardData>(not_yet_attacking[static_cast<size_t>(creature_choice)]);

        printf("Select target for %s:\n", cd.name.c_str());
        for (size_t i = 0; i < targets.size(); i++) {
            auto& player = global_coordinator.GetComponent<Player>(targets[i]);
            Zone::Ownership t = (targets[i] == game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
            printf("  %zu: %s (%d life)\n", i, player_name(t).c_str(), player.life_total);
            // TODO: planeswalker entries here
        }

        std::vector<ActionCategory> atk_tgt_cats(targets.size(), ActionCategory::OTHER_CHOICE);
        int target_choice = InputLogger::instance().get_logged_input(cur_game.turn, atk_tgt_cats);

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
        auto& cr = global_coordinator.GetComponent<Creature>(entity);
        if (cr.is_attacking) {
            any = true;
            auto& cd = global_coordinator.GetComponent<CardData>(entity);
            Zone::Ownership t = (cr.attack_target == game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
            printf("  %s -> %s\n", cd.name.c_str(), player_name(t).c_str());

            // Tap the attacker
            auto& permanent = global_coordinator.GetComponent<Permanent>(entity);
            permanent.is_tapped = true;
        }
    }
    if (!any) printf("  (none)\n");

    game.attackers_declared = true;
    game.pending_choice = NONE;
}

static void declare_blockers(Game& game, std::shared_ptr<Orderer> orderer) {
    Zone::Ownership defending_player = game.player_a_turn ? Zone::PLAYER_B : Zone::PLAYER_A;

    // Collect attackers
    std::vector<Entity> attackers;
    for (auto entity : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
        auto& cr = global_coordinator.GetComponent<Creature>(entity);
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
        auto& permanent = global_coordinator.GetComponent<Permanent>(entity);
        if (permanent.controller != defending_player) continue;
        if (permanent.is_tapped) continue;
        eligible.push_back(entity);
    }

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
            auto& cd = global_coordinator.GetComponent<CardData>(attackers[i]);
            auto& cr = global_coordinator.GetComponent<Creature>(attackers[i]);
            printf("  %zu: %s [%d/%d]\n", i, cd.name.c_str(), cr.power, cr.toughness);
        }
        printf("Your creatures:\n");
        for (size_t i = 0; i < eligible.size(); i++) {
            auto& cd = global_coordinator.GetComponent<CardData>(eligible[i]);
            auto& cr = global_coordinator.GetComponent<Creature>(eligible[i]);
            printf("  %zu: %s [%d/%d]", i, cd.name.c_str(), cr.power, cr.toughness);
            if (cr.is_blocking) {
                auto& attacker_cd = global_coordinator.GetComponent<CardData>(cr.blocking_target);
                printf(" blocking %s", attacker_cd.name.c_str());
            }
            printf("\n");
        }
        printf("  -1: Confirm\n");

        // num_choices = eligible blockers + 1 implicit confirm (-1)
        std::vector<ActionCategory> blk_cats(eligible.size(), ActionCategory::SELECT_BLOCKER);
        blk_cats.push_back(ActionCategory::CONFIRM_BLOCKERS);
        int blocker_choice = InputLogger::instance().get_logged_input(cur_game.turn, blk_cats);

        if (blocker_choice == -1) break;

        if (blocker_choice < 0 || blocker_choice >= static_cast<int>(eligible.size())) {
            printf("Invalid selection.\n");
            continue;
        }

        auto& cr = global_coordinator.GetComponent<Creature>(eligible[static_cast<size_t>(blocker_choice)]);
        auto& cd = global_coordinator.GetComponent<CardData>(eligible[static_cast<size_t>(blocker_choice)]);

        if (cr.is_blocking) {
            // Toggle off
            cr.is_blocking = false;
            cr.blocking_target = 0;
            printf("%s removed from blockers.\n", cd.name.c_str());
        } else {
            printf("Select attacker for %s to block:\n", cd.name.c_str());
            for (size_t i = 0; i < attackers.size(); i++) {
                auto& acd = global_coordinator.GetComponent<CardData>(attackers[i]);
                auto& acr = global_coordinator.GetComponent<Creature>(attackers[i]);
                printf("  %zu: %s [%d/%d]\n", i, acd.name.c_str(), acr.power, acr.toughness);
            }

            std::vector<ActionCategory> blk_tgt_cats(attackers.size(), ActionCategory::OTHER_CHOICE);
            int attacker_choice = InputLogger::instance().get_logged_input(cur_game.turn, blk_tgt_cats);

            if (attacker_choice >= 0 && attacker_choice < static_cast<int>(attackers.size())) {
                cr.is_blocking = true;
                cr.blocking_target = attackers[static_cast<size_t>(attacker_choice)];
                auto& acd = global_coordinator.GetComponent<CardData>(cr.blocking_target);
                printf("%s blocking %s.\n", cd.name.c_str(), acd.name.c_str());
            } else {
                printf("Invalid attacker.\n");
            }
        }
    }

    printf("\nBlockers declared:\n");
    bool any = false;
    for (auto entity : eligible) {
        auto& cr = global_coordinator.GetComponent<Creature>(entity);
        if (cr.is_blocking) {
            any = true;
            auto& cd = global_coordinator.GetComponent<CardData>(entity);
            auto& acd = global_coordinator.GetComponent<CardData>(cr.blocking_target);
            printf("  %s blocking %s\n", cd.name.c_str(), acd.name.c_str());
        }
    }
    if (!any) printf("  (none)\n");

    game.blockers_declared = true;
    game.pending_choice = NONE;
}

void proc_mandatory_choice(Game& game, std::shared_ptr<Orderer> orderer) {
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
                    auto& cd = global_coordinator.GetComponent<CardData>(hand[i]);
                    printf("  %zu: %s\n", i, cd.name.c_str());
                }

                std::vector<ActionCategory> discard_cats(hand.size(), ActionCategory::OTHER_CHOICE);
                int choice = InputLogger::instance().get_logged_input(game.turn, discard_cats);
                if (choice >= 0 && choice < static_cast<int>(hand.size())) {
                    Entity card = hand[static_cast<size_t>(choice)];
                    auto& cd = global_coordinator.GetComponent<CardData>(card);
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
