#include "game.h"
#include "deck.h"


void Game::generate_players(const Deck &deck_a, const Deck &deck_b) {
    player_a_entity = gen_player(deck_a);
    player_b_entity = gen_player(deck_b);
}

Entity Game::gen_player(const Deck &deck) {
    return Entity();
}
