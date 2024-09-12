#include "gui.h"

#include "raylib.h"

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

static bool first_run = true;

bool gui_loop() {
    if (first_run) InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "robomage");
    first_run = false;

    BeginDrawing();
    ClearBackground(WHITE);
    DrawText("HELLO", 200, 200, 24, BLACK);
    EndDrawing();

    if (WindowShouldClose()) return 1;
    return 0;
}