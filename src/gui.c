#include "gui.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#include "classes/gamestate.h"
#include "pthread.h"
#include "stdbool.h"
#include "string.h"

extern pthread_t game_loop_thread;
extern const GameState *gs_ptr;
pthread_mutex_t input_mutex = PTHREAD_MUTEX_INITIALIZER;
char gui_input[GUI_INPUT_MAX] = {'\0'};
volatile bool gui_input_requested = false;
volatile bool gui_input_sent = false;
volatile bool gui_killed = false;
volatile int gui_cmd = 0;
static char gui_resource_dir[512] = "resources";
int SCREEN_WIDTH;
int SCREEN_HEIGHT;

void gui_set_resource_dir(const char *path) {
    snprintf(gui_resource_dir, sizeof(gui_resource_dir), "%s", path);
}

#ifdef GUI
#include "raylib.h"
#define RAYGUI_IMPLEMENTATION
#include <math.h>

#include "raygui.h"

static Font g_font;

// Hover state — reset each frame, set by draw_perm_chip / hand rendering
static int gs_hover_vocab_idx = -1;
static float gs_hover_tx = 0.0f;
static float gs_hover_ty = 0.0f;
static int gs_hover_perm_power = -1;  // -1 = not a perm hover
static int gs_hover_perm_toughness = -1;
static int gs_hover_perm_damage = 0;

// Forward declarations
static int count_gy(const int *gy, int max);
static void draw_wrapped_text(const char *text, float x, float y, float max_w, float font_size, Color color);
static void draw_perm_chip(float sx, float sy, float sw, float sh, const PermanentState *perm, Vector2 mouse);
static void render_battlefield(float px, float py, float pw, float ph, const PermanentState *perms, int max_slots,
    Vector2 mouse, const char *label);
static void render_gs(void);

static int count_gy(const int *gy, int max) {
    int n = 0;
    for (int i = 0; i < max; i++) {
        if (gy[i] >= 0) n++;
    }
    return n;
}

static void determine_screen_size() {
    InitWindow(10, 10, "robomage");
    int monitor = GetCurrentMonitor();
    SCREEN_WIDTH = GetMonitorWidth(monitor) * 0.8;
    SCREEN_HEIGHT = GetMonitorHeight(monitor) * 0.8;
    SetWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
    SetWindowPosition((GetMonitorWidth(monitor) - SCREEN_WIDTH) / 2, (GetMonitorHeight(monitor) - SCREEN_HEIGHT) / 2);
}

// Simple word-wrap renderer; stops drawing if it would exceed max_lines lines.
static void draw_wrapped_text(const char *text, float x, float y, float max_w, float font_size, Color color) {
    char word[256];
    char line[512];
    int line_pos = 0;
    float line_y = y;
    const char *p = text;
    line[0] = '\0';

    while (*p) {
        // Collect next word
        int wlen = 0;
        while (*p && *p != ' ' && *p != '\n' && wlen < 254) word[wlen++] = *p++;
        word[wlen] = '\0';

        if (wlen == 0) {
            if (*p == '\n') {
                DrawTextEx(g_font, line, (Vector2){x, line_y}, font_size, 1.0f, color);
                line_y += font_size + 2.0f;
                line[0] = '\0';
                line_pos = 0;
                p++;
            }
            continue;
        }

        // Try appending word to current line
        char test[512];
        if (line_pos == 0)
            snprintf(test, sizeof(test), "%s", word);
        else
            snprintf(test, sizeof(test), "%s %s", line, word);

        float tw = MeasureTextEx(g_font, test, font_size, 1.0f).x;
        if (tw > max_w && line_pos > 0) {
            DrawTextEx(g_font, line, (Vector2){x, line_y}, font_size, 1.0f, color);
            line_y += font_size + 2.0f;
            snprintf(line, sizeof(line), "%s", word);
            line_pos = wlen;
        } else {
            snprintf(line, sizeof(line), "%s", test);
            line_pos = (int)strlen(test);
        }

        if (*p == '\n') {
            DrawTextEx(g_font, line, (Vector2){x, line_y}, font_size, 1.0f, color);
            line_y += font_size + 2.0f;
            line[0] = '\0';
            line_pos = 0;
            p++;
        } else if (*p == ' ') {
            p++;
        }
    }
    if (line_pos > 0) DrawTextEx(g_font, line, (Vector2){x, line_y}, font_size, 1.0f, color);
}

static void draw_perm_chip(float sx, float sy, float sw, float sh, const PermanentState *perm, Vector2 mouse) {
    if (perm->card_vocab_idx < 0) {
        // Empty slot — faint outline only
        DrawRectangleLinesEx((Rectangle){sx + 1, sy + 1, sw - 2, sh - 2}, 0.5f, (Color){190, 190, 190, 60});
        return;
    }

    // Pick fill and border colours
    Color bg, border_col;
    if (perm->is_attacking) {
        bg = (Color){255, 175, 175, 255};
        border_col = RED;
    } else if (perm->is_blocking) {
        bg = (Color){175, 175, 255, 255};
        border_col = BLUE;
    } else if (perm->has_summoning_sickness) {
        bg = (Color){255, 255, 175, 255};
        border_col = ORANGE;
    } else if (perm->is_tapped) {
        bg = (Color){195, 195, 195, 255};
        border_col = DARKGRAY;
    } else {
        bg = (Color){238, 238, 238, 255};
        border_col = GRAY;
    }
    // Opponent cards: slightly cooler tint
    if (!perm->controller_is_self) {
        bg.r = (unsigned char)((int)bg.r * 85 / 100);
        bg.g = (unsigned char)((int)bg.g * 90 / 100);
    }

    float cx = sx + sw * 0.5f;
    float cy = sy + sh * 0.5f;
    float cw = sw - 4.0f;
    float ch = sh - 4.0f;
    float rot = perm->is_tapped ? 90.0f : 0.0f;

    // Filled rotated rectangle
    DrawRectanglePro((Rectangle){cx, cy, cw, ch}, (Vector2){cw * 0.5f, ch * 0.5f}, rot, bg);

    // Border: 4 rotated lines
    float cos_r = cosf(rot * DEG2RAD);
    float sin_r = sinf(rot * DEG2RAD);
    float hw = cw * 0.5f, hh = ch * 0.5f;
    float lc[4][2] = {{-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh}};
    Vector2 sc[4];
    for (int i = 0; i < 4; i++) {
        sc[i].x = lc[i][0] * cos_r - lc[i][1] * sin_r + cx;
        sc[i].y = lc[i][0] * sin_r + lc[i][1] * cos_r + cy;
    }
    for (int i = 0; i < 4; i++) DrawLineEx(sc[i], sc[(i + 1) % 4], 1.5f, border_col);

    // Text
    const char *name = gui_card_name(perm->card_vocab_idx);
    float font_sz = 10.0f;

    if (!perm->is_tapped) {
        if (sw >= 30.0f) {
            DrawTextEx(g_font, name, (Vector2){sx + 3.0f, sy + 3.0f}, font_sz, 1.0f, BLACK);
            if (perm->is_creature && sw >= 40.0f) {
                char pt[32];
                if (perm->damage > 0)
                    snprintf(pt, sizeof(pt), "%d/%d (%ddmg)", perm->power, perm->toughness, perm->damage);
                else
                    snprintf(pt, sizeof(pt), "%d/%d", perm->power, perm->toughness);
                DrawTextEx(g_font, pt, (Vector2){sx + 3.0f, sy + sh - font_sz - 3.0f}, font_sz, 1.0f, DARKGRAY);
            }
        }
    } else {
        // Tapped 90° CW: card's local "up" = screen right (+x).
        // Local point (0, -ch*0.25) maps to screen (cx - ch*0.25, cy).
        // Draw text with -90° rotation so it reads when you tilt head right.
        if (ch >= 30.0f) {
            float name_w = MeasureTextEx(g_font, name, font_sz, 1.0f).x;
            DrawTextPro(g_font, name, (Vector2){cx - ch * 0.25f, cy}, (Vector2){name_w * 0.5f, font_sz * 0.5f}, -90.0f,
                font_sz, 1.0f, BLACK);
            if (perm->is_creature) {
                char pt[16];
                snprintf(pt, sizeof(pt), "%d/%d", perm->power, perm->toughness);
                float pt_w = MeasureTextEx(g_font, pt, font_sz, 1.0f).x;
                DrawTextPro(g_font, pt, (Vector2){cx + ch * 0.25f, cy}, (Vector2){pt_w * 0.5f, font_sz * 0.5f}, -90.0f,
                    font_sz, 1.0f, DARKGRAY);
            }
        }
    }

    // Hover detection
    bool hovered;
    if (!perm->is_tapped) {
        hovered = (mouse.x >= sx && mouse.x < sx + sw && mouse.y >= sy && mouse.y < sy + sh);
    } else {
        // Inverse-rotate mouse into card local frame (90° CCW = reverse of 90° CW)
        float dx = mouse.x - cx;
        float dy = mouse.y - cy;
        float local_x = dy;   // cos(-90)*dx - sin(-90)*dy = dy
        float local_y = -dx;  // sin(-90)*dx + cos(-90)*dy = -dx
        hovered = (local_x > -hw && local_x < hw && local_y > -hh && local_y < hh);
    }
    if (hovered) {
        gs_hover_vocab_idx = perm->card_vocab_idx;
        gs_hover_tx = mouse.x;
        gs_hover_ty = mouse.y;
        gs_hover_perm_power = perm->is_creature ? perm->power : -1;
        gs_hover_perm_toughness = perm->is_creature ? perm->toughness : -1;
        gs_hover_perm_damage = perm->damage;
    }
}

static void render_battlefield(float px, float py, float pw, float ph, const PermanentState *perms, int max_slots,
    Vector2 mouse, const char *label) {
    // Count active permanents
    int active_idx[MAX_BATTLEFIELD_SLOTS];
    int active = 0;
    for (int i = 0; i < max_slots; i++) {
        if (perms[i].card_vocab_idx >= 0) active_idx[active++] = i;
    }

    int display_slots = (active < 4) ? 4 : active;
    float slot_w = pw / (float)display_slots;

    // Background
    DrawRectangle((int)px, (int)py, (int)pw, (int)ph, (Color){245, 245, 235, 255});
    DrawRectangleLinesEx((Rectangle){px, py, pw, ph}, 1.0f, LIGHTGRAY);
    DrawTextEx(g_font, label, (Vector2){px + 3.0f, py + 2.0f}, 9.0f, 1.0f, GRAY);

    // Active permanents
    for (int d = 0; d < active; d++) {
        float sx = px + (float)d * slot_w;
        draw_perm_chip(sx, py, slot_w, ph, &perms[active_idx[d]], mouse);
    }
    // Empty filler slots
    for (int d = active; d < display_slots; d++) {
        float sx = px + (float)d * slot_w;
        DrawRectangleLinesEx(
            (Rectangle){sx + 1.0f, py + 1.0f, slot_w - 2.0f, ph - 2.0f}, 0.5f, (Color){190, 190, 190, 60});
    }
}

static void render_gs(void) {
    if (!gs_ptr) return;
    const GameState *gs = gs_ptr;

    // Reset hover state
    gs_hover_vocab_idx = -1;
    gs_hover_perm_power = -1;
    gs_hover_perm_toughness = -1;
    gs_hover_perm_damage = 0;

    Vector2 mouse = GetMousePosition();
    bool q_held = IsKeyDown(KEY_Q);

    float px = SCREEN_WIDTH * 0.25f + 20.0f;
    float pw = (float)SCREEN_WIDTH - px - 10.0f;
    float y = 2.0f;
    float small_sz = 10.0f;

    // ── 1. Turn / Step bar ──────────────────────────────────────────────
    {
        char bar[256];
        snprintf(bar, sizeof(bar), "Turn %d  |  %s  |  %s  |  Priority: %s", gs->turn, gui_step_name((int)gs->cur_step),
            gs->is_active_player ? "Your turn" : "Opp turn", gs->is_active_player ? "You" : "Opp");
        DrawRectangle((int)px, (int)y, (int)pw, 18, (Color){55, 55, 75, 255});
        DrawTextEx(g_font, bar, (Vector2){px + 4.0f, y + 3.0f}, small_sz, 1.0f, WHITE);
        y += 20.0f;
    }

    // ── 2. Opponent info bar ────────────────────────────────────────────
    {
        char bar[256];
        snprintf(bar, sizeof(bar), "OPP  Life:%d  Poison:%d  W:%d U:%d B:%d R:%d G:%d C:%d  Hand:%d  Lib:%d",
            gs->opponent.life, gs->opponent.poison_counters, gs->opponent.mana[0], gs->opponent.mana[1],
            gs->opponent.mana[2], gs->opponent.mana[3], gs->opponent.mana[4], gs->opponent.mana[5],
            gs->opponent.hand_ct, gs->opp_library_ct);
        DrawRectangle((int)px, (int)y, (int)pw, 18, (Color){185, 65, 65, 230});
        DrawTextEx(g_font, bar, (Vector2){px + 4.0f, y + 3.0f}, small_sz, 1.0f, WHITE);
        y += 20.0f;
    }

    // ── 3. Opponent hand (face-down) ────────────────────────────────────
    {
        float hand_h = 50.0f;
        DrawRectangle((int)px, (int)y, (int)pw, (int)hand_h, (Color){220, 200, 200, 180});
        int n = gs->opponent.hand_ct;
        if (n > MAX_HAND_SLOTS) n = MAX_HAND_SLOTS;
        float card_w = 58.0f;
        float gap = 5.0f;
        float start_x = px + (pw - (float)n * (card_w + gap)) * 0.5f;
        for (int i = 0; i < n; i++) {
            float cx_i = start_x + (float)i * (card_w + gap);
            DrawRectangle((int)cx_i, (int)(y + 4.0f), (int)card_w, (int)(hand_h - 8.0f), (Color){75, 35, 115, 210});
            DrawRectangleLinesEx((Rectangle){cx_i, y + 4.0f, card_w, hand_h - 8.0f}, 1.0f, (Color){40, 20, 75, 255});
        }
        y += hand_h + 4.0f;
    }

    // ── 4. Opponent battlefield ─────────────────────────────────────────
    {
        float bf_h = 175.0f;
        render_battlefield(px, y, pw, bf_h, gs->opp_permanents, MAX_BATTLEFIELD_SLOTS, mouse, "Opponent");
        y += bf_h + 4.0f;
    }

    // ── 5. Stack ────────────────────────────────────────────────────────
    {
        float stack_h = 42.0f;
        DrawRectangle((int)px, (int)y, (int)pw, (int)stack_h, (Color){215, 215, 170, 210});
        char stack_label[32];
        snprintf(stack_label, sizeof(stack_label), "Stack (%d)", gs->stack_size);
        DrawTextEx(g_font, stack_label, (Vector2){px + 4.0f, y + 3.0f}, small_sz, 1.0f, DARKGRAY);
        float ex = px + 85.0f;
        for (int i = 0; i < gs->stack_size && i < MAX_STACK_DISPLAY; i++) {
            const StackEntry *se = &gs->stack[i];
            const char *sname = (se->card_vocab_idx >= 0) ? gui_card_name(se->card_vocab_idx) : "?";
            char chip[80];
            snprintf(chip, sizeof(chip), "[%s - %s]", sname, se->controller_is_self ? "You" : "Opp");
            float chip_w = MeasureTextEx(g_font, chip, small_sz, 1.0f).x + 8.0f;
            Color chip_col = se->controller_is_self ? (Color){90, 150, 240, 220} : (Color){240, 110, 90, 220};
            DrawRectangle((int)ex, (int)(y + 12.0f), (int)chip_w, 22, chip_col);
            DrawTextEx(g_font, chip, (Vector2){ex + 4.0f, y + 15.0f}, small_sz, 1.0f, WHITE);
            ex += chip_w + 4.0f;
        }
        y += stack_h + 4.0f;
    }

    // ── 6. Self battlefield ─────────────────────────────────────────────
    {
        float bf_h = 175.0f;
        render_battlefield(px, y, pw, bf_h, gs->self_permanents, MAX_BATTLEFIELD_SLOTS, mouse, "You");
        y += bf_h + 4.0f;
    }

    // ── 7. Self hand ────────────────────────────────────────────────────
    {
        float hand_h = 85.0f;
        DrawRectangle((int)px, (int)y, (int)pw, (int)hand_h, (Color){200, 225, 200, 190});

        int n = 0;
        for (int i = 0; i < MAX_HAND_SLOTS; i++) {
            if (gs->self_hand[i] >= 0) n++;
        }
        int display_n = (n < 4) ? 4 : n;
        float card_w = pw / (float)display_n;

        int col = 0;
        for (int i = 0; i < MAX_HAND_SLOTS; i++) {
            int vi = gs->self_hand[i];
            if (col >= display_n) break;
            float hx = px + (float)col * card_w;
            float hy = y;
            if (vi >= 0) {
                const char *cname = gui_card_name(vi);
                const char *ctype = gui_card_type_line(vi);
                const char *coracle = gui_card_oracle(vi);

                DrawRectangle((int)(hx + 2.0f), (int)(hy + 2.0f), (int)(card_w - 4.0f), (int)(hand_h - 4.0f),
                    (Color){195, 225, 195, 255});
                DrawRectangleLinesEx((Rectangle){hx + 2.0f, hy + 2.0f, card_w - 4.0f, hand_h - 4.0f}, 1.0f, DARKGREEN);

                DrawTextEx(g_font, cname, (Vector2){hx + 4.0f, hy + 4.0f}, small_sz, 1.0f, BLACK);
                DrawTextEx(g_font, ctype, (Vector2){hx + 4.0f, hy + 16.0f}, 9.0f, 1.0f, DARKGRAY);
                // Oracle text: first ~3 lines
                BeginScissorMode((int)(hx + 2.0f), (int)(hy + 26.0f), (int)(card_w - 4.0f), (int)(hand_h - 30.0f));
                draw_wrapped_text(coracle, hx + 4.0f, hy + 27.0f, card_w - 8.0f, 8.5f, DARKGRAY);
                EndScissorMode();

                // Hover check for tooltip (hand cards use base stats)
                if (mouse.x >= hx && mouse.x < hx + card_w && mouse.y >= hy && mouse.y < hy + hand_h) {
                    gs_hover_vocab_idx = vi;
                    gs_hover_tx = mouse.x;
                    gs_hover_ty = mouse.y;
                    gs_hover_perm_power = -1;  // signal: use base stats
                    gs_hover_perm_toughness = -1;
                    gs_hover_perm_damage = 0;
                }
            } else {
                DrawRectangleLinesEx(
                    (Rectangle){hx + 2.0f, hy + 2.0f, card_w - 4.0f, hand_h - 4.0f}, 0.5f, (Color){170, 210, 170, 90});
            }
            col++;
        }
        y += hand_h + 4.0f;
    }

    // ── 8. Self info bar ────────────────────────────────────────────────
    {
        char bar[256];
        int self_gy = count_gy(gs->self_graveyard, MAX_GY_SLOTS);
        snprintf(bar, sizeof(bar), "YOU  Life:%d  Poison:%d  W:%d U:%d B:%d R:%d G:%d C:%d  Lands:%d  Lib:%d  GY:%d",
            gs->self.life, gs->self.poison_counters, gs->self.mana[0], gs->self.mana[1], gs->self.mana[2],
            gs->self.mana[3], gs->self.mana[4], gs->self.mana[5], gs->self.lands_played_this_turn, gs->self_library_ct,
            self_gy);
        DrawRectangle((int)px, (int)y, (int)pw, 18, (Color){50, 95, 50, 255});
        DrawTextEx(g_font, bar, (Vector2){px + 4.0f, y + 3.0f}, small_sz, 1.0f, WHITE);
        y += 20.0f;
    }

    // ── 9. Graveyard / exile summary ────────────────────────────────────
    {
        // Self graveyard
        char line[640];
        int self_gy_ct = count_gy(gs->self_graveyard, MAX_GY_SLOTS);
        snprintf(line, sizeof(line), "Your GY (%d):", self_gy_ct);
        for (int i = 0; i < MAX_GY_SLOTS; i++) {
            if (gs->self_graveyard[i] >= 0) {
                strncat(line, " ", sizeof(line) - strlen(line) - 1);
                strncat(line, gui_card_name(gs->self_graveyard[i]), sizeof(line) - strlen(line) - 1);
                strncat(line, ",", sizeof(line) - strlen(line) - 1);
            }
        }
        DrawTextEx(g_font, line, (Vector2){px + 4.0f, y}, small_sz, 1.0f, DARKBROWN);
        y += 13.0f;

        // Opponent graveyard
        int opp_gy_ct = count_gy(gs->opp_graveyard, MAX_GY_SLOTS);
        snprintf(line, sizeof(line), "Opp GY  (%d):", opp_gy_ct);
        for (int i = 0; i < MAX_GY_SLOTS; i++) {
            if (gs->opp_graveyard[i] >= 0) {
                strncat(line, " ", sizeof(line) - strlen(line) - 1);
                strncat(line, gui_card_name(gs->opp_graveyard[i]), sizeof(line) - strlen(line) - 1);
                strncat(line, ",", sizeof(line) - strlen(line) - 1);
            }
        }
        DrawTextEx(g_font, line, (Vector2){px + 4.0f, y}, small_sz, 1.0f, DARKBROWN);
    }

    // ── Tooltip (Q + hover) ─────────────────────────────────────────────
    if (q_held && gs_hover_vocab_idx >= 0) {
        const char *tname = gui_card_name(gs_hover_vocab_idx);
        const char *toracle = gui_card_oracle(gs_hover_vocab_idx);
        const char *ttype = gui_card_type_line(gs_hover_vocab_idx);

        float tp_w = 300.0f, tp_h = 210.0f;
        float tp_x = gs_hover_tx + 14.0f;
        float tp_y = gs_hover_ty - tp_h * 0.5f;
        // Clamp to screen
        if (tp_x + tp_w > (float)SCREEN_WIDTH - 5.0f) tp_x = gs_hover_tx - tp_w - 14.0f;
        if (tp_y < 5.0f) tp_y = 5.0f;
        if (tp_y + tp_h > (float)SCREEN_HEIGHT - 5.0f) tp_y = (float)SCREEN_HEIGHT - tp_h - 5.0f;

        DrawRectangle((int)tp_x, (int)tp_y, (int)tp_w, (int)tp_h, (Color){252, 248, 218, 252});
        DrawRectangleLinesEx((Rectangle){tp_x, tp_y, tp_w, tp_h}, 2.0f, (Color){100, 80, 20, 255});

        float ty = tp_y + 6.0f;
        DrawTextEx(g_font, tname, (Vector2){tp_x + 6.0f, ty}, 13.0f, 1.0f, BLACK);
        ty += 16.0f;
        DrawTextEx(g_font, ttype, (Vector2){tp_x + 6.0f, ty}, small_sz, 1.0f, DARKGRAY);
        ty += 14.0f;

        // P/T line: perm-hover uses actual stats; hand-hover uses base stats
        bool show_pt = false;
        int disp_p = 0, disp_t = 0, disp_d = 0;
        if (gs_hover_perm_power >= 0) {
            show_pt = true;
            disp_p = gs_hover_perm_power;
            disp_t = gs_hover_perm_toughness;
            disp_d = gs_hover_perm_damage;
        } else {
            int bp = gui_card_base_power(gs_hover_vocab_idx);
            int bt = gui_card_base_toughness(gs_hover_vocab_idx);
            if (bp > 0 || bt > 0) {
                show_pt = true;
                disp_p = bp;
                disp_t = bt;
            }
        }
        if (show_pt) {
            char pt[40];
            if (disp_d > 0)
                snprintf(pt, sizeof(pt), "%d/%d  (%d damage)", disp_p, disp_t, disp_d);
            else
                snprintf(pt, sizeof(pt), "%d/%d", disp_p, disp_t);
            DrawTextEx(g_font, pt, (Vector2){tp_x + 6.0f, ty}, small_sz, 1.0f, BLACK);
            ty += 14.0f;
        }

        // Oracle text clipped to tooltip
        BeginScissorMode((int)(tp_x + 4.0f), (int)ty, (int)(tp_w - 8.0f), (int)(tp_y + tp_h - ty - 4.0f));
        draw_wrapped_text(toracle, tp_x + 6.0f, ty, tp_w - 12.0f, small_sz, BLACK);
        EndScissorMode();
    }
}

// scrollable box that displays everything that would be propogated to the CLI
static void render_info_log() {
    static Vector2 scroll = {0, 0};
    static int last_line_count = 0;
    static float max_content_width = 0;

    int line_count = gui_log_line_count();
    float font_size = 16.0f;
    float line_height = 18.0f;

    Rectangle bounds = {10, 10, SCREEN_WIDTH * 0.25, SCREEN_HEIGHT * 0.65f - 10};

    // recompute max line width and auto-scroll when new lines arrive
    if (line_count != last_line_count) {
        max_content_width = 0;
        for (int i = 0; i < line_count; i++) {
            const char *line = gui_log_get_line(i);
            float w = MeasureTextEx(g_font, line, font_size, 1.0f).x;
            if (w > max_content_width) max_content_width = w;
        }
        float content_height_new = line_count * line_height;
        scroll.y = -(content_height_new - bounds.height) - line_height;
        if (scroll.y > 0) scroll.y = 0;
        last_line_count = line_count;
    }

    float content_height = line_count * line_height;
    if (content_height < bounds.height) content_height = bounds.height;
    float content_width = max_content_width + 8;
    if (content_width < bounds.width - 12) content_width = bounds.width - 12;
    Rectangle content = {bounds.x, bounds.y, content_width, content_height};

    Rectangle view = {0};
    GuiScrollPanel(bounds, NULL, content, &scroll, &view);

    BeginScissorMode((int)view.x, (int)view.y, (int)view.width, (int)view.height);
    float y = bounds.y + scroll.y;
    for (int i = 0; i < line_count; i++) {
        if (y + line_height >= view.y && y <= view.y + view.height) {
            const char *line = gui_log_get_line(i);
            DrawTextEx(g_font, line, (Vector2){bounds.x + 4 + scroll.x, y}, font_size, 1.0f, BLACK);
        }
        y += line_height;
    }
    EndScissorMode();
}

static void render_choices() {
    int line_count = gui_query_line_count();
    float font_size = 16.0f;
    float line_height = 18.0f;
    float y = (SCREEN_HEIGHT * 0.7);
    for (int i = 0; i < line_count; i++) {
        const char *line = gui_query_get_line(i);
        DrawTextEx(g_font, line, (Vector2){10, y}, font_size, 1.0f, DARKBLUE);
        y += line_height;
    }
}

static void *gui_loop(void *arg) {
    determine_screen_size();
    char font_path[512];
    snprintf(font_path, sizeof(font_path), "%s/Magicmedieval-pRV1.ttf", gui_resource_dir);
    g_font = LoadFontEx(font_path, 32, NULL, 0);

    while (!WindowShouldClose()) {
        BeginDrawing();

        ClearBackground(WHITE);

        // INPUT TEXT BOX DRAW AND UPDATE; this could be a functions
        if (GuiTextBox((Rectangle){SCREEN_WIDTH * .4, SCREEN_HEIGHT * .9, SCREEN_WIDTH * .4, SCREEN_HEIGHT * .05},
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
        // display game state
        render_gs();
        // display info log
        render_info_log();
        // display choices available in query
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