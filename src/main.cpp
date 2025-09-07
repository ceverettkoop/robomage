#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <time.h>
#include "card_db.h"
#include "classes/game.h"
#include "classes/deck.h"
#include "ecs/coordinator.h"
#include "components/damage.h"
#include "components/ability.h"
#include "components/carddata.h"
#include "components/player.h"
#include "components/zone.h"

#ifndef VERSION_NUMBER
#define VERSION_NUMBER "0.001"
#endif

/*
extern "C" {
#include "gui.h"
}
*/

std::string RESOURCE_DIR;
Coordinator global_coordinator = Coordinator();
Deck DEFAULT_DECK_ONE;
Deck DEFAULT_DECK_TWO;

int main(int argc, char const *argv[]) {
    char buf[FILENAME_MAX];
    RESOURCE_DIR = getcwd(buf, FILENAME_MAX);
    RESOURCE_DIR += "/resources";

    DEFAULT_DECK_ONE = Deck(RESOURCE_DIR + "/decks/deck_one.dk");
    DEFAULT_DECK_TWO = Deck(RESOURCE_DIR + "/decks/deck_two.dk");

    printf("robomage %s\n", VERSION_NUMBER);
    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    std::srand(seed);

    global_coordinator.Init();
    global_coordinator.RegisterComponent<Ability>();
    global_coordinator.RegisterComponent<CardData>();
    global_coordinator.RegisterComponent<Damage>();
    global_coordinator.RegisterComponent<Player>();
    global_coordinator.RegisterComponent<Zone>();

    
    //one time setup for this game
    Game cur_game(seed);
    cur_game.generate_players(DEFAULT_DECK_ONE,DEFAULT_DECK_TWO);
    cur_game.generate_libraries(DEFAULT_DECK_ONE, DEFAULT_DECK_TWO);

//game loop

    //if something resolves bc of priority passing, resolve that now

    //state based effects / triggers

    //active player can take action or pass priority (pass means skip to end)

    //special game actions resolve immediately

    //state based effects

    //repeat

    return 0;
}
