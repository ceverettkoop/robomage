#include "orderer.h"

#include <algorithm>
#include <numeric>

#include "../card_db.h"
#include "../classes/deck.h"
#include "../classes/game.h"
#include "../classes/action.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/color_identity.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"

// orderer cares about anything that has a zone
void Orderer::init() {
    Signature signature;
    signature.set(global_coordinator.GetComponentType<Zone>());
    global_coordinator.SetSystemSignature<Orderer>(signature);
}

void Orderer::add_to_zone(bool on_bottom, Entity target, Zone::ZoneValue destination) {
    size_t back = 0;
    auto &target_zone = global_coordinator.GetComponent<Zone>(target);

    // If the entity is leaving an ordered zone, close the gap it leaves behind.
    // LIBRARY, STACK, and GRAVEYARD are ordered zones where distance_from_top is meaningful.
    Zone::ZoneValue origin = target_zone.location;
    if (origin == Zone::LIBRARY || origin == Zone::STACK || origin == Zone::GRAVEYARD) {
        size_t departing_pos = target_zone.distance_from_top;
        Zone::Ownership owner = target_zone.owner;
        for (auto &&card : mEntities) {
            if (card == target) continue;
            auto &cmp_zone = global_coordinator.GetComponent<Zone>(card);
            if (cmp_zone.location != origin) continue;
            // Library and graveyard are per-player; stack is shared
            if ((origin == Zone::LIBRARY || origin == Zone::GRAVEYARD) && cmp_zone.owner != owner) continue;
            if (cmp_zone.distance_from_top > departing_pos) {
                cmp_zone.distance_from_top--;
            }
        }
    }

    if (!on_bottom) {
        target_zone.distance_from_top = 0;
    }

    for (auto &&card : mEntities) {
        if (card == target) continue;
        auto &cmp_zone = global_coordinator.GetComponent<Zone>(card);
        if (cmp_zone.location != destination) continue;
        // Library and graveyard are per-player; only shift cards belonging to the same owner
        if ((destination == Zone::LIBRARY || destination == Zone::GRAVEYARD) && cmp_zone.owner != target_zone.owner)
            continue;
        if (!on_bottom) {
            // placing on top: shift everything else down one
            cmp_zone.distance_from_top++;
        } else {
            // placing on bottom: find the current bottom position
            if (cmp_zone.distance_from_top > back) back = cmp_zone.distance_from_top;
        }
    }

    if (on_bottom) target_zone.distance_from_top = back + 1;
    target_zone.location = destination;
}

// TODO MERGE THESE INTO A GENERIC GETTER
std::vector<Entity> Orderer::get_library_contents(Zone::Ownership owner) {
    std::vector<Entity> contents;

    for (auto &&card : mEntities) {
        auto &card_zone = global_coordinator.GetComponent<Zone>(card);
        if ((card_zone.location == Zone::LIBRARY) && (card_zone.owner == owner)) {
            contents.push_back(card);
        }
    }
    return contents;
}

std::vector<Entity> Orderer::get_hand(Zone::Ownership owner) {
    std::vector<Entity> contents;

    for (auto &&card : mEntities) {
        auto &card_zone = global_coordinator.GetComponent<Zone>(card);
        if ((card_zone.location == Zone::HAND) && (card_zone.owner == owner)) {
            contents.push_back(card);
        }
    }
    return contents;
}

void Orderer::shuffle_library(Zone::Ownership owner) {
    auto contents = get_library_contents(owner);
    size_t n = contents.size();
    std::vector<int> placements(n);
    std::iota(placements.begin(), placements.end(), 0);
    std::shuffle(placements.begin(), placements.end(), cur_game.gen);

    size_t i = 0;
    for (auto &&card : contents) {
        auto &card_zone = global_coordinator.GetComponent<Zone>(card);
        card_zone.distance_from_top = placements[i];
        i++;
    }
}

void Orderer::generate_libraries(const Deck &deck_a, const Deck &deck_b) {
    Zone::Ownership owner = Zone::PLAYER_A;
    auto target_deck = deck_a;
    Coordinator &coordinator = Coordinator::global();

    // outer loop to assign each deck to proper player
    for (size_t i = 0; i < 2; i++) {
        if (i == 1) {
            owner = Zone::PLAYER_B;
            target_deck = deck_b;
        }
        // loop through each card and create an entity in appropriate library per qty
        for (auto &&card_name : target_deck.main_deck) {
            for (size_t i = 0; i < card_name.first; i++) {  // qty
                // TODO this will probably need to be made a function for when it is repeated in the case of token
                // creation
                Entity card_id = coordinator.CreateEntity();
                auto card_data_id = load_card(card_name.second);
                coordinator.AddComponent(card_id, coordinator.GetComponent<CardData>(card_data_id));
                coordinator.AddComponent(card_id, Zone(Zone::LIBRARY, owner, owner));
                ColorIdentity ci;
                auto &cd = coordinator.GetComponent<CardData>(card_id);
                for (auto c : cd.mana_cost) {
                    if (c != GENERIC && c != COLORLESS && c != NO_COLOR) ci.colors.insert(c);
                }
                coordinator.AddComponent(card_id, ci);
            }
        }
    }

    shuffle_library(Zone::PLAYER_A);
    shuffle_library(Zone::PLAYER_B);
}

void Orderer::draw_hands() {
    draw(Zone::PLAYER_A, 7);
    draw(Zone::PLAYER_B, 7);
}

// actual effect here, not considering triggers or replacement
void Orderer::draw(Zone::Ownership player, size_t ct) {
    std::vector<Entity> cards_to_draw;
    for (auto &&card : mEntities) {
        auto &card_zone = global_coordinator.GetComponent<Zone>(card);
        if (card_zone.location == Zone::LIBRARY && card_zone.owner == player) {
            if (card_zone.distance_from_top < ct) {
                cards_to_draw.push_back(card);
            } else {
                card_zone.distance_from_top -= ct;
            }
        }
    }
    // TODO: first card drawn here is subject to replacement effects (e.g. Miracle, Leyline of Anticipation)
    // Inline print_draw: sort by distance_from_top then log each draw
    {
        std::vector<Entity> sorted = cards_to_draw;
        std::sort(sorted.begin(), sorted.end(), [](Entity a, Entity b) {
            return global_coordinator.GetComponent<Zone>(a).distance_from_top <
                   global_coordinator.GetComponent<Zone>(b).distance_from_top;
        });
        for (auto card : sorted) {
            game_log_private(player, "%s draws %s\n", player_name(player).c_str(),
                     global_coordinator.GetComponent<CardData>(card).name.c_str());
        }
    }
    for (auto &&card : cards_to_draw) {
        auto &card_zone = global_coordinator.GetComponent<Zone>(card);
        card_zone.location = Zone::HAND;
    }
    if (cards_to_draw.size() < ct) {
        if (player == Zone::PLAYER_A) {
            printf("\nPlayer A decked - Player B wins!\n");
        } else {
            printf("\nPlayer B decked - Player A wins!\n");
        }
        cur_game.ended = true;
    }
}

// ordered stack, top first
std::vector<Entity> Orderer::get_stack() {
    std::vector<Entity> on_stack;
    for (auto &&card : mEntities) {
        auto &card_zone = global_coordinator.GetComponent<Zone>(card);
        if (card_zone.location == Zone::STACK) {
            on_stack.push_back(card);
        }
    }
    std::sort(on_stack.begin(), on_stack.end(), [](Entity const &a, Entity const &b) {
        return global_coordinator.GetComponent<Zone>(a).distance_from_top <
               global_coordinator.GetComponent<Zone>(a).distance_from_top;
    });

    return on_stack;
}


void Orderer::do_london_mulligan() {
    int mulligans_a = 0;
    int mulligans_b = 0;
    bool a_kept = false;
    bool b_kept = false;

    auto do_bottom_deck = [&](Zone::Ownership owner, int count) {
        std::string pname = player_name(owner);
        for (int i = 0; i < count; i++) {
            auto hand = this->get_hand(owner);
            if (hand.empty()) break;
            game_log("%s: Choose card to put on library bottom (%d remaining):\n", pname.c_str(), count - i);
            std::vector<LegalAction> btm_actions;
            for (auto card : hand) {
                auto &cd = global_coordinator.GetComponent<CardData>(card);
                LegalAction la(PASS_PRIORITY, card, cd.name);
                la.category = ActionCategory::BOTTOM_DECK_CARD;
                btm_actions.push_back(la);
            }
            int choice = InputLogger::instance().get_input(btm_actions);
            this->add_to_zone(true, hand[static_cast<size_t>(choice)], Zone::LIBRARY);
        }
    };

    while (!a_kept || !b_kept) {
        // Player A decides
        if (!a_kept) {
            cur_game.player_a_has_priority = true;
            {
                auto hand_display = this->get_hand(Zone::PLAYER_A);
                game_log_private(Zone::PLAYER_A, "%s hand:\n", player_name(Zone::PLAYER_A).c_str());
                for (auto card : hand_display) {
                    auto &data = global_coordinator.GetComponent<CardData>(card);
                    game_log_private(Zone::PLAYER_A, "%s\n", data.name.c_str());
                }
            }
            std::vector<LegalAction> mull_actions = {
                LegalAction(PASS_PRIORITY, std::string("Keep")),
                LegalAction(PASS_PRIORITY, std::string("Mulligan")),
            };
            mull_actions[0].category = ActionCategory::MULLIGAN;
            mull_actions[1].category = ActionCategory::MULLIGAN;
            int choice = InputLogger::instance().get_input(mull_actions);
            if (choice == 0) {
                a_kept = true;
                do_bottom_deck(Zone::PLAYER_A, mulligans_a);
            } else {
                mulligans_a++;
                if (mulligans_a == 7) {
                    a_kept = true;
                    do_bottom_deck(Zone::PLAYER_A, mulligans_a);
                } else {
                    if (mulligans_a >= 3 && InputLogger::instance().is_machine_mode()) {
                        printf("MULLIGAN_PENALTY: A\n");
                        fflush(stdout);
                    }
                    auto hand = this->get_hand(Zone::PLAYER_A);
                    for (auto card : hand) {
                        this->add_to_zone(false, card, Zone::LIBRARY);
                    }
                    this->shuffle_library(Zone::PLAYER_A);
                    this->draw(Zone::PLAYER_A, 7);
                }
            }
        }

        // Player B decides
        if (!b_kept) {
            cur_game.player_a_has_priority = false;
            {
                auto hand_display = this->get_hand(Zone::PLAYER_B);
                game_log_private(Zone::PLAYER_B, "%s hand:\n", player_name(Zone::PLAYER_B).c_str());
                for (auto card : hand_display) {
                    auto &data = global_coordinator.GetComponent<CardData>(card);
                    game_log_private(Zone::PLAYER_B, "%s\n", data.name.c_str());
                }
            }
            std::vector<LegalAction> mull_actions = {
                LegalAction(PASS_PRIORITY, std::string("Keep")),
                LegalAction(PASS_PRIORITY, std::string("Mulligan")),
            };
            mull_actions[0].category = ActionCategory::MULLIGAN;
            mull_actions[1].category = ActionCategory::MULLIGAN;
            int choice = InputLogger::instance().get_input(mull_actions);
            if (choice == 0) {
                b_kept = true;
                do_bottom_deck(Zone::PLAYER_B, mulligans_b);
            } else {
                mulligans_b++;
                if (mulligans_b == 7) {
                    b_kept = true;
                    do_bottom_deck(Zone::PLAYER_B, mulligans_b);
                } else {
                    if (mulligans_b >= 3 && InputLogger::instance().is_machine_mode()) {
                        printf("MULLIGAN_PENALTY: B\n");
                        fflush(stdout);
                    }
                    auto hand = this->get_hand(Zone::PLAYER_B);
                    for (auto card : hand) {
                        this->add_to_zone(false, card, Zone::LIBRARY);
                    }
                    this->shuffle_library(Zone::PLAYER_B);
                    this->draw(Zone::PLAYER_B, 7);
                }
            }
        }
    }
}
