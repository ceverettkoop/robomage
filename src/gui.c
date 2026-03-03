#ifdef GUI

#include "gui.h"
#include <stdlib.h>
#include <stdio.h>
#include "pthread.h"
#include "raylib.h"

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

extern pthread_t gui_thread;


static void *gui_loop(void *arg){

    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "robomage");
    while(!WindowShouldClose()){
        BeginDrawing();
        ClearBackground(WHITE);
        DrawText("HELLO", 200, 200, 24, BLACK);
        EndDrawing();
    }
    return NULL;
}


void init_gui() {
    if (pthread_create(&gui_thread, NULL, gui_loop, NULL) != 0){
        perror("pthread_create");
        exit(1);
    }
}

#endif