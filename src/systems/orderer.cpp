#include "orderer.h"

#include "../card_db.h"
#include "../classes/deck.h"
#include "../components/carddata.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../classes/game.h"

#include <numeric>
#include <algorithm>

// orderer cares about anything that has a zone
void Orderer::init() {
    Signature signature;
    signature.set(global_coordinator.GetComponentType<Zone>());
    global_coordinator.SetSystemSignature<Orderer>(signature);
}

void Orderer::add_to_zone(bool on_bottom, Entity target, Zone::ZoneValue destination) {
    size_t back = 0;
    auto &target_zone = global_coordinator.GetComponent<Zone>(target);

    if (!on_bottom) {
        target_zone.distance_from_top = 0;
    }

    for (auto &&card : mEntities) {
        auto &cmp_zone = global_coordinator.GetComponent<Zone>(card);
        if (cmp_zone.location == destination) {
            if (!on_bottom) {
                // if we are placing on top, move everything else down one
                cmp_zone.distance_from_top++;
            } else {
                // if we are going on the bottom, determine where the bottom is
                if (cmp_zone.distance_from_top > back) back = cmp_zone.distance_from_top;
            }
        }
    }

    if (on_bottom) target_zone.distance_from_top = back + 1;
}

std::vector<Entity> Orderer::get_library_contents(Zone::Ownership owner) {
    std::vector<Entity> contents;

    for (auto &&card : mEntities){
        auto &card_zone = global_coordinator.GetComponent<Zone>(card);
        if((card_zone.location == Zone::LIBRARY) && (card_zone.owner == owner)){
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
    for (auto &&card : contents){
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
        for (auto &&card_name : deck_a.main_deck) {
            for (size_t i = 0; i < card_name.first; i++) {  // qty
                // TODO this will probably need to be made a function for when it is repeated in the case of token
                // creation
                Entity card_id = coordinator.CreateEntity();
                auto card_data_id = load_card(card_name.second);
                coordinator.AddComponent(card_id, coordinator.GetComponent<CardData>(card_data_id));
                coordinator.AddComponent(card_id, Zone(Zone::LIBRARY, owner, owner));
            }
        }
    }

    shuffle_library(Zone::PLAYER_A);
    shuffle_library(Zone::PLAYER_B);
}