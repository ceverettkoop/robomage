#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include "card_db.h"
#include "game.h"
#include "player.h"

#ifndef VERSION_NUMBER
#define VERSION_NUMBER "0.001"
#endif

/*
extern "C" {
#include "gui.h"
}
*/
std::string RESOURCE_DIR;

int main(int argc, char const *argv[]) {
    char buf[FILENAME_MAX];
    RESOURCE_DIR = getcwd(buf, FILENAME_MAX);
    RESOURCE_DIR += "/resources";

    printf("robomage %s\n", VERSION_NUMBER);

    //init state based on decks and shuffle per seed
    Game cur_game;
    Player player_otp;
    Player player_otd;


//game loop

    //if something resolves bc of priority passing, resolve that now

    //state based effects / triggers

    //active player can take action or pass priority (pass means skip to end)

    //special game actions resolve immediately

    //state based effects

    //repeat


    return 0;
}
