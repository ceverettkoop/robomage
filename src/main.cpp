#include <cstdlib>
#include <cstdio>
#include "ecs/coordinator.h"

#ifndef VERSION_NUMBER
#define VERSION_NUMBER "0.001"
#endif

/*
extern "C" {
#include "gui.h"
}
*/

Coordinator game_coordinator;

int main(int argc, char const *argv[]) {
    printf("robomage %s\n", VERSION_NUMBER);

    //init state based on decks and shuffle per seed


//game loop

    //if something resolves bc of priority passing, resolve that now

    //state based effects / triggers

    //active player can take action or pass priority (pass means skip to end)

    //special game actions resolve immediately

    //state based effects

    //repeat


    return 0;
}
