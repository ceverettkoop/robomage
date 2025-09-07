#include "game.h"
#include "deck.h"
#include "../card_db.h"
#include "../ecs/coordinator.h"
#include "../components/carddata.h"
#include "../components/zone.h"

void Game::generate_players(const Deck &deck_a, const Deck &deck_b) {
    player_a_entity = gen_player(deck_a);
    player_b_entity = gen_player(deck_b);
}

void Game::generate_libraries(const Deck &deck_a, const Deck &deck_b) {
    Zone::Ownership owner = Zone::PLAYER_A;
    auto target_deck = deck_a;
    Coordinator &coordinator = Coordinator::global();

//outer loop to assign each deck to proper player
    for (size_t i = 0; i < 2; i++) {
        if (i == 1) {
            owner = Zone::PLAYER_B;
            target_deck = deck_b;
        }
        //loop through each card and create an entity in appropriate library per qty
        for (auto &&card_name : deck_a.main_deck) {
            for (size_t i = 0; i < card_name.first; i++) { //qty
                Entity card_id = coordinator.CreateEntity();
                coordinator.AddComponent(card_id, coordinator.GetComponent<CardData>(load_card(card_name.second)));
                coordinator.AddComponent(card_id, Zone(Zone::LIBRARY, owner, owner));
            }
        }
    }
}

Entity Game::gen_player(const Deck &deck) {
    return Entity();
}
