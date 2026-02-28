#include "debug.h"

#include <algorithm>
#include "components/carddata.h"
#include "input_logger.h"
#include "components/creature.h"
#include "components/damage.h"
#include "components/permanent.h"
#include "components/player.h"
#include "ecs/coordinator.h"
#include "error.h"
#include "mana_system.h"
#include "systems/orderer.h"
#include "systems/state_manager.h"

const char *step_to_string(Step in_step) {
    switch (in_step) {
        case UNTAP:
            return "Untap";
            break;
        case UPKEEP:
            return "Upkeep";
            break;
        case DRAW:
            return "Draw";
            break;
        case FIRST_MAIN:
            return "First Main";
            break;
        case BEGIN_COMBAT:
            return "Begin Combat";
            break;
        case DECLARE_ATTACKERS:
            return "Declare Attackers";
            break;
        case DECLARE_BLOCKERS:
            return "Declare Blockers";
            break;
        case COMBAT_DAMAGE:
            return "Combat Damage";
            break;
        case END_OF_COMBAT:
            return "End of Combat";
            break;
        case SECOND_MAIN:
            return "Second Main";
            break;
        case END_STEP:
            return "End Step";
            break;
        case CLEANUP:
            return "Cleanup";
            break;
        default:
            return "ERROR UNREACHABLE";
            break;
    }
}

void print_step(const Game& cur_game) {
    if (InputLogger::instance().is_machine_mode()) return;
    Zone::Ownership active_player = cur_game.player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B;

    // Get life totals
    auto& player_a = global_coordinator.GetComponent<Player>(cur_game.player_a_entity);
    auto& player_b = global_coordinator.GetComponent<Player>(cur_game.player_b_entity);

    printf("\n=== Turn %zu - %s's %s ===\n",
           cur_game.turn,
           player_name(active_player).c_str(),
           step_to_string(cur_game.cur_step));
    printf("Life: Player A=%d, Player B=%d\n", player_a.life_total, player_b.life_total);
}

void print_library(std::shared_ptr<Orderer> orderer, Zone::Ownership owner) {
    if (InputLogger::instance().is_machine_mode()) return;
    auto library = orderer->get_library_contents(owner);
    size_t n = library.size();

    for (size_t i = 0; i < n; i++) {
        bool found = false;
        for (auto &&card_id : library) {
            auto &card = global_coordinator.GetComponent<CardData>(card_id);
            auto &zone = global_coordinator.GetComponent<Zone>(card_id);
            if (zone.distance_from_top == i) {
                printf("Card %s is %zu from top \n", card.uid.c_str(), i);
                found = true;
                break;
            }
        }
        if (!found) fatal_error("Missing card in library at index " + std::to_string(i));
    }
}

void print_hand(std::shared_ptr<Orderer> orderer, Zone::Ownership owner) {
    if (InputLogger::instance().is_machine_mode()) return;
    auto hand = orderer->get_hand(owner);

    printf("%s hand:\n", player_name(owner).c_str());
    for (auto &&card : hand) {
        auto &data = global_coordinator.GetComponent<CardData>(card);
        printf("%s\n", data.name.c_str());
    }
}

void print_stack(std::shared_ptr<Orderer> orderer) {
    if (InputLogger::instance().is_machine_mode()) return;
    auto stack = orderer->get_stack();
    if(stack.size() > 0){
        printf("STACK:\n");
    }else{
        return;
    }
    for (size_t i = 0; i < stack.size(); i++){
        auto entity = stack.at(i);
        if(global_coordinator.entity_has_component<CardData>(entity)){
            printf("%zu: %s\n", i, global_coordinator.GetComponent<CardData>(entity).name.c_str());
        }else{
            printf("%zu: %s\n", i, "TODO: Generate descriptive names for abilities");
        }


    }
    return;
}

std::string player_name(Zone::Ownership owner) {
    if (owner == Zone::PLAYER_A)
        return "Player A";
    else
        return "Player B";
}

void print_draw(Zone::Ownership player, const std::vector<Entity>& cards) {
    if (InputLogger::instance().is_machine_mode()) return;
    std::vector<Entity> sorted = cards;
    std::sort(sorted.begin(), sorted.end(), [](Entity a, Entity b) {
        return global_coordinator.GetComponent<Zone>(a).distance_from_top <
               global_coordinator.GetComponent<Zone>(b).distance_from_top;
    });
    for (auto card : sorted) {
        printf("%s draws %s\n", player_name(player).c_str(),
               global_coordinator.GetComponent<CardData>(card).name.c_str());
    }
}

void print_legal_actions(const Game& cur_game, std::vector<LegalAction> legal_actions) {
    if (InputLogger::instance().is_machine_mode()) return;
    Zone::Ownership priority_player = cur_game.player_a_has_priority ? Zone::PLAYER_A : Zone::PLAYER_B;
    printf("\n%s has priority. Legal actions:\n", player_name(priority_player).c_str());
    for (size_t i = 0; i < legal_actions.size(); i++) {
        printf("  %zu: %s\n", i, legal_actions[i].description.c_str());
    }
}

void print_mandatory_choice_description(const Game& cur_game) {
    if (InputLogger::instance().is_machine_mode()) return;
    Zone::Ownership active_player = cur_game.player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B;

    printf("\n");
    switch (cur_game.pending_choice) {
        case DECLARE_ATTACKERS_CHOICE:
            printf("%s must declare attackers.\n", player_name(active_player).c_str());
            printf("TODO: List creatures that can attack\n");
            break;

        case DECLARE_BLOCKERS_CHOICE:
            printf("%s must declare blockers.\n", player_name(active_player).c_str());
            printf("TODO: List creatures that can block and valid attackers to block\n");
            break;

        case CLEANUP_DISCARD:
            printf("%s must discard to hand size.\n", player_name(active_player).c_str());
            printf("TODO: Show current hand size and maximum hand size\n");
            break;

        case CHOOSE_ENTITY:
            printf("%s must choose an entity.\n", player_name(active_player).c_str());
            printf("TODO: Show available choices (legend rule, replacement effects, etc.)\n");
            break;

        case NONE:
            // Should not reach here
            break;
    }
}

void print_battlefield(std::shared_ptr<Orderer> orderer) {
    if (InputLogger::instance().is_machine_mode()) return;
    printf("\n--- BATTLEFIELD ---\n");
    for (auto owner : {Zone::PLAYER_A, Zone::PLAYER_B}) {
        printf("%s:\n", player_name(owner).c_str());

        bool found_any = false;
        for (auto entity : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
            auto& zone = global_coordinator.GetComponent<Zone>(entity);

            if (zone.location == Zone::BATTLEFIELD && zone.owner == owner) {
                found_any = true;
                auto& card_data = global_coordinator.GetComponent<CardData>(entity);
                auto& permanent = global_coordinator.GetComponent<Permanent>(entity);

                printf("  %s", card_data.name.c_str());
                if (permanent.is_tapped) printf(" (TAPPED)");
                if (permanent.has_summoning_sickness) printf(" (SICK)");

                // Print P/T for creatures
                if (global_coordinator.entity_has_component<Creature>(entity)) {
                    auto& creature = global_coordinator.GetComponent<Creature>(entity);
                    printf(" [%d/%d]", creature.power, creature.toughness);
                    if (global_coordinator.entity_has_component<Damage>(entity)) {
                        auto& damage = global_coordinator.GetComponent<Damage>(entity);
                        if (damage.damage_counters > 0) {
                            printf(" (%u damage)", damage.damage_counters);
                        }
                    }
                }
                printf("\n");
            }
        }
        if (!found_any) {
            printf("  (no permanents)\n");
        }
    }
}

void print_mana_pools() {
    if (InputLogger::instance().is_machine_mode()) return;
    bool header_printed = false;

    for (auto owner : {Zone::PLAYER_A, Zone::PLAYER_B}) {
        Entity player_entity = get_player_entity(owner);
        auto& player = global_coordinator.GetComponent<Player>(player_entity);

        if (player.mana.empty()) {
            //nothing
        } else {
            if(!header_printed){
                printf("\n--- MANA POOLS ---\n");
                header_printed = true;
            }
            for (auto color : player.mana) {
                printf("%s: ", player_name(owner).c_str());
                switch (color) {
                    case WHITE:
                        printf("{W} ");
                        break;
                    case BLUE:
                        printf("{U} ");
                        break;
                    case BLACK:
                        printf("{B} ");
                        break;
                    case RED:
                        printf("{R} ");
                        break;
                    case GREEN:
                        printf("{G} ");
                        break;
                    case COLORLESS:
                        printf("{C} ");
                        break;
                    case GENERIC:
                        printf("{1} ");
                        break;
                }
            }
        }
        printf("\n");
    }
}
