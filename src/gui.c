#include <ctype.h>
#ifdef GUI

#include <stdio.h>
#include <stdlib.h>

#include "gui.h"
#include "pthread.h"
#include "raylib.h"
#include "stdbool.h"
#include "string.h"
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

extern pthread_t gui_thread;
pthread_mutex_t input_mutex = PTHREAD_MUTEX_INITIALIZER;
char gui_input[GUI_INPUT_MAX] = {'\0'};
bool gui_input_requested = false;
bool gui_input_sent = false;
bool gui_killed = false;
int gui_cmd = 0;

static void *gui_loop(void *arg) {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "robomage");

    while (!WindowShouldClose()) {
        BeginDrawing();

        ClearBackground(WHITE);
        DrawText("HELLO", 200, 200, 24, BLACK);

        // INPUT TEXT BOX DRAW AND UPDATE
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