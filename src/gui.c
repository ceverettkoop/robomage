#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#include "gui.h"
#include "pthread.h"
#include "raylib.h"
#include "stdbool.h"
#include "string.h"
#include "classes/gamestate.h"

#define SCREEN_WIDTH 1920
#define SCREEN_HEIGHT 1080

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

//todo later
static void render_gs(){

}

//scrollable box that displays everything that would be propogated to the CLI
static void render_info_log(){
    int line_count = gui_log_line_count();
    int font_size = 16;
    int line_height = 18;
    int area_top = 10;
    int area_bottom = (int)(SCREEN_HEIGHT * 0.60);
    int lines_to_show = (area_bottom - area_top) / line_height;
    int start_idx = (line_count > lines_to_show) ? line_count - lines_to_show : 0;
    int y = area_top;
    for (int i = start_idx; i < line_count; i++) {
        const char* line = gui_log_get_line(i);
        DrawText(line, 10, y, font_size, BLACK);
        y += line_height;
    }
}

//scrollable box that displays specifically and only the last query
static void render_query(){
    render_info_log();
}


static void render_choices(){
    int line_count = gui_query_line_count();
    int font_size = 16;
    int line_height = 18;
    int y = (int)(SCREEN_HEIGHT * 0.62);
    int y_max = (int)(SCREEN_HEIGHT * 0.68);
    for (int i = 0; i < line_count && y < y_max; i++) {
        const char* line = gui_query_get_line(i);
        DrawText(line, 10, y, font_size, DARKBLUE);
        y += line_height;
    }
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
    gui_loop(NULL);
}

#endif