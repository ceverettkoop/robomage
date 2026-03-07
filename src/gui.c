#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#include "gui.h"
#include "pthread.h"
#include "raylib.h"
#include "stdbool.h"
#include "string.h"
#include "classes/gamestate.h"

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

extern pthread_t gui_thread;
extern const GameState *gs_ptr;
extern const Query *query_ptr;
pthread_mutex_t input_mutex = PTHREAD_MUTEX_INITIALIZER;
char gui_input[GUI_INPUT_MAX] = {'\0'};
bool gui_input_requested = false;
bool gui_input_sent = false;
bool gui_killed = false;
int gui_cmd = 0;

#ifdef GUI
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

static void render_gs(){



}

//scrollable window that displays everything that would be propogated to the CLI
static void render_info_log(){

}

//scrollable window that displays specifically and only the last query
static void render_query(){

}

static void render_choices(){
    
}

static void *gui_loop(void *arg) {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "robomage");

    while (!WindowShouldClose()) {
        BeginDrawing();

        ClearBackground(WHITE);
        DrawText("HELLO", 200, 200, 24, BLACK);

        // INPUT TEXT BOX DRAW AND UPDATE; this could be a function
        if (GuiTextInputBox((Rectangle){SCREEN_WIDTH * .2, SCREEN_HEIGHT * .7, SCREEN_WIDTH * .6, SCREEN_HEIGHT * .2},
                "Input Command", "", "OK", gui_input, GUI_INPUT_MAX, NULL) == true) {
            if (gui_input_requested) {
                // validate input
                for (size_t i = 0; i < GUI_INPUT_MAX; ++i) {
                    if (!isdigit(gui_input[i]) && gui_input[i] != '\0') {
                        memset(gui_input, '\0', GUI_INPUT_MAX);
                        break;
                    }
                }
                if (gui_input[0] == '\0') goto INPUT_END;
                // valid input, send it and clear
                pthread_mutex_lock(&input_mutex);
                int parsed = atoi(gui_input);
                gui_cmd = parsed;
                gui_input_sent = true;
                pthread_mutex_unlock(&input_mutex);
                memset(gui_input, '\0', GUI_INPUT_MAX);
            }
        }
    INPUT_END:
        //display game state
        render_gs();
        //display info log
        render_info_log();
        //display choices available in query
        render_choices();
    
    
        EndDrawing();
    
    
    
    }
    gui_killed = true;
    return NULL;
}

void init_gui() {
    if (pthread_create(&gui_thread, NULL, gui_loop, NULL) != 0) {
        perror("pthread_create");
        exit(1);
    }
}

#endif