#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <time.h>
#include "card_db.h"
#include "classes/game.h"
#include "classes/player.h"
#include "classes/deck.h"
#include "ecs/coordinator.h"
#include "components/damage.h"
#include "components/player.h"
#include "components/spell_ability.h"

#ifndef VERSION_NUMBER
#define VERSION_NUMBER "0.001"
#endif

/*
extern "C" {
#include "gui.h"
}
*/

std::string RESOURCE_DIR;
Coordinator global_coordinator;

int main(int argc, char const *argv[]) {
    char buf[FILENAME_MAX];
    RESOURCE_DIR = getcwd(buf, FILENAME_MAX);
    RESOURCE_DIR += "/resources";

    printf("robomage %s\n", VERSION_NUMBER);
    unsigned int seed = time(nullptr);
    std::srand(seed);

    global_coordinator.Init();
    global_coordinator.RegisterComponent<Damage>();
    global_coordinator.RegisterComponent<Player>();
    global_coordinator.RegisterComponent<SpellAbility>();

    Game cur_game(seed);
    Game.generate_players(DEFAULT_DECK_ONE,DEFAULT_DECK_TWO);
    Player player_otp(true);
    Player player_otd(false);
    Deck deck_one = DEFAULT_DECK_ONE;
    Deck deck_two = DEFAULT_DECK_TWO;


//game loop

    //if something resolves bc of priority passing, resolve that now

    //state based effects / triggers

    //active player can take action or pass priority (pass means skip to end)

    //special game actions resolve immediately

    //state based effects

    //repeat


    return 0;
}
