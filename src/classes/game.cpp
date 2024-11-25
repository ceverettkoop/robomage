#include "game.h"
#include "deck.h"

void Game::generate_players(const Deck& otp_deck, const Deck& otd_deck) {

    player_otp = gen_player(otp_deck);
    player_otp = gen_player(otd_deck);

}

Entity Game::gen_player(const Deck &deck) {
    return Entity();
}
