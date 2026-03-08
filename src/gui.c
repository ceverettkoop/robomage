#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#include "gui.h"
#include "pthread.h"
#include "stdbool.h"
#include "string.h"
#include "classes/gamestate.h"

#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720

extern pthread_t game_loop_thread;
extern const GameState *gs_ptr;
extern const Query *query_ptr;
pthread_mutex_t input_mutex = PTHREAD_MUTEX_INITIALIZER;
char gui_input[GUI_INPUT_MAX] = {'\0'};
bool gui_input_requested = false;
bool gui_input_sent = false;
bool gui_killed = false;
int gui_cmd = 0;
static char gui_resource_dir[512] = "resources";

void gui_set_resource_dir(const char *path) {
    snprintf(gui_resource_dir, sizeof(gui_resource_dir), "%s", path);
}

#ifdef GUI
#include "raylib.h"
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

static Font g_font;

//todo later
static void render_gs(){

}

//scrollable box that displays everything that would be propogated to the CLI
static void render_info_log(){
    static Vector2 scroll = {0, 0};
    static int last_line_count = 0;

    int line_count = gui_log_line_count();
    float font_size = 16.0f;
    float line_height = 18.0f;

    Rectangle bounds = {10, 10, SCREEN_WIDTH  * 0.4, SCREEN_HEIGHT * 0.70f - 10};
    float content_height = line_count * line_height;
    if (content_height < bounds.height) content_height = bounds.height;
    Rectangle content = {bounds.x, bounds.y, bounds.width - 12, content_height};

    // auto-scroll to bottom when new lines arrive
    if (line_count != last_line_count) {
        scroll.y = -(content_height - bounds.height) - line_height;
        if (scroll.y > 0) scroll.y = 0;
        last_line_count = line_count;
    }

    Rectangle view = {0};
    GuiScrollPanel(bounds, NULL, content, &scroll, &view);

    BeginScissorMode((int)view.x, (int)view.y, (int)view.width, (int)view.height);
    float y = bounds.y + scroll.y;
    for (int i = 0; i < line_count; i++) {
        if (y + line_height >= view.y && y <= view.y + view.height) {
            const char* line = gui_log_get_line(i);
            DrawTextEx(g_font, line, (Vector2){bounds.x + 4 + scroll.x, y}, font_size, 1.0f, BLACK);
        }
        y += line_height;
    }
    EndScissorMode();
}

//scrollable box that displays specifically and only the last query
static void render_query(){
//
}

static void render_choices(){
    int line_count = gui_query_line_count();
    float font_size = 16.0f;
    float line_height = 18.0f;
    float y = (float)(int)(SCREEN_HEIGHT * 0.74);
    float y_max = (float)(int)(SCREEN_HEIGHT * 0.82);
    for (int i = 0; i < line_count && y < y_max; i++) {
        const char* line = gui_query_get_line(i);
        DrawTextEx(g_font, line, (Vector2){10, y}, font_size, 1.0f, DARKBLUE);
        y += line_height;
    }
}

static void *gui_loop(void *arg) {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "robomage");

    char font_path[512];
    snprintf(font_path, sizeof(font_path), "%s/Magicmedieval-pRV1.ttf", gui_resource_dir);
    g_font = LoadFontEx(font_path, 32, NULL, 0);

    while (!WindowShouldClose()) {
        BeginDrawing();

        ClearBackground(WHITE);

        // INPUT TEXT BOX DRAW AND UPDATE; this could be a function
        if (GuiTextBox((Rectangle){SCREEN_WIDTH * .3, SCREEN_HEIGHT * .8, SCREEN_WIDTH * .4, SCREEN_HEIGHT * .05},
                 gui_input, GUI_INPUT_MAX, true) == true) {
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
    UnloadFont(g_font);
    gui_killed = true;
    return NULL;
}

void init_gui() {
    gui_loop(NULL);
}

#endif