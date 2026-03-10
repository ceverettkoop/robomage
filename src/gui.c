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
volatile bool quit_gui = false;
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

// ──────────────────────────────────────────────────────────────────────────
// Layout & visual constants — edit these to adjust GUI proportions
// ──────────────────────────────────────────────────────────────────────────

// Left sidebar (log + choices): fraction of screen width
#define LAYOUT_SIDEBAR_W_RATIO      0.2f
// Padding / gaps in pixels (small fixed values, intentionally not scaled)
#define LAYOUT_SIDEBAR_PAD          10.0f   // padding inside sidebar
#define LAYOUT_RIGHT_PAD            10.0f   // right edge padding
#define LAYOUT_MAIN_LEFT_PAD        20.0f   // gap between sidebar right edge and main area
#define LAYOUT_SECTION_GAP          4.0f    // vertical gap between sections
#define LAYOUT_CARD_GAP             4.0f    // gap between cards in hand / battlefield

// Main game area row heights — fractions of screen height
#define LAYOUT_STATUS_BAR_H         0.046f  // turn bar / player info bars
#define LAYOUT_OPP_BF_H             0.155f  // opponent battlefield
#define LAYOUT_STACK_H              0.038f  // stack display
#define LAYOUT_SELF_BF_H            0.155f  // self battlefield
#define LAYOUT_SELF_HAND_H          0.3f  // self hand cards

// Sidebar vertical splits — fractions of screen height
#define LAYOUT_LOG_BOTTOM_RATIO     0.62f   // log panel bottom edge
#define LAYOUT_CHOICES_Y_RATIO      0.63f   // choices list starts here

// Input text box — width/height as fractions of screen; anchored to bottom-right corner
#define LAYOUT_INPUT_W_RATIO        0.35f
#define LAYOUT_INPUT_H_RATIO        0.050f
#define LAYOUT_INPUT_MARGIN         10.0f   // gap from right and bottom edges

// Button bar — sits between sidebar right edge and input box, same height as input box
#define LAYOUT_N_BUTTONS            6       // total button slots (fill left-to-right as needed)
#define LAYOUT_BUTTON_GAP           6.0f    // gap between buttons

// Font rendering sizes — fractions of screen height
#define FONT_SIZE_MAIN_RATIO        0.017f  // log text (do not increase — user preference)
#define FONT_SIZE_CHOICES_RATIO     0.020f  // clickable action list
#define FONT_SIZE_BAR_RATIO         0.026f  // turn/player info bar text
#define FONT_SIZE_LABEL_RATIO       0.019f  // battlefield/stack labels
#define FONT_SIZE_CARD_RATIO        0.015f  // card name / P/T text on cards
#define FONT_SIZE_TINY_RATIO        0.012f  // oracle text inside hand cards

// Card proportions
#define CARD_ASPECT_RATIO           1.36f   // height / width for all rendered cards

// Tooltip dimensions — fractions of screen
#define LAYOUT_TOOLTIP_W_RATIO      0.21f
#define LAYOUT_TOOLTIP_H_RATIO      0.27f

// Font texture rasterisation: load size = GetMonitorHeight() / FONT_LOAD_DIVISOR
// Smaller divisor = larger texture = crisper text when scaled up
#define FONT_LOAD_DIVISOR           28

// ──────────────────────────────────────────────────────────────────────────

static Font g_font;
static Font g_font_oracle;  // serif font (Merriweather) for oracle text

// Hover state — reset each frame, set by draw_perm_card / hand rendering
static int gs_hover_vocab_idx = -1;
static float gs_hover_tx = 0.0f;
static float gs_hover_ty = 0.0f;
static int gs_hover_perm_power = -1;  // -1 = not a perm hover
static int gs_hover_perm_toughness = -1;
static int gs_hover_perm_damage = 0;

// Forward declarations
static int count_gy(const int *gy, int max);
static void draw_wrapped_text(Font font, const char *text, float x, float y, float max_w, float font_size, Color color);
static void draw_perm_card(float cx, float cy, float cw, float ch, const PermanentState *perm, Vector2 mouse);
static void render_battlefield(float px, float py, float pw, float ph, const PermanentState *perms, int max_slots,
    Vector2 mouse, const char *label);
static void render_gs(void);
static void render_info_log(void);
static void render_choices(void);
static void render_button_bar(Rectangle bar);

static int count_gy(const int *gy, int max) {
    int n = 0;
    for (int i = 0; i < max; i++) {
        if (gy[i] >= 0) n++;
    }
    return n;
}

static void determine_screen_size(void) {
    InitWindow(10, 10, "robomage");
    SetWindowState(FLAG_WINDOW_RESIZABLE);
    int monitor = GetCurrentMonitor();
    SCREEN_WIDTH = (int)(GetMonitorWidth(monitor) * 0.8f);
    SCREEN_HEIGHT = (int)(GetMonitorHeight(monitor) * 0.8f);
    SetWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
    SetWindowPosition((GetMonitorWidth(monitor) - SCREEN_WIDTH) / 2, (GetMonitorHeight(monitor) - SCREEN_HEIGHT) / 2);
}

// Simple word-wrap renderer
static void draw_wrapped_text(Font font, const char *text, float x, float y, float max_w, float font_size, Color color) {
    char word[256];
    char line[512];
    int line_pos = 0;
    float line_y = y;
    const char *p = text;
    line[0] = '\0';

    while (*p) {
        int wlen = 0;
        while (*p && *p != ' ' && *p != '\n' && wlen < 254) word[wlen++] = *p++;
        word[wlen] = '\0';

        if (wlen == 0) {
            if (*p == '\n') {
                DrawTextEx(font, line, (Vector2){x, line_y}, font_size, 1.0f, color);
                line_y += font_size + 2.0f;
                line[0] = '\0';
                line_pos = 0;
                p++;
            }
            continue;
        }

        char test[512];
        if (line_pos == 0)
            snprintf(test, sizeof(test), "%s", word);
        else
            snprintf(test, sizeof(test), "%s %s", line, word);

        float tw = MeasureTextEx(font, test, font_size, 1.0f).x;
        if (tw > max_w && line_pos > 0) {
            DrawTextEx(font, line, (Vector2){x, line_y}, font_size, 1.0f, color);
            line_y += font_size + 2.0f;
            snprintf(line, sizeof(line), "%s", word);
            line_pos = wlen;
        } else {
            snprintf(line, sizeof(line), "%s", test);
            line_pos = (int)strlen(test);
        }

        if (*p == '\n') {
            DrawTextEx(font, line, (Vector2){x, line_y}, font_size, 1.0f, color);
            line_y += font_size + 2.0f;
            line[0] = '\0';
            line_pos = 0;
            p++;
        } else if (*p == ' ') {
            p++;
        }
    }
    if (line_pos > 0) DrawTextEx(font, line, (Vector2){x, line_y}, font_size, 1.0f, color);
}

// Draw a single card as a portrait rectangle (CARD_ASPECT_RATIO height/width).
// cx/cy = top-left of the card rect, cw/ch = card dimensions.
static void draw_perm_card(float cx, float cy, float cw, float ch, const PermanentState *perm, Vector2 mouse) {
    float font_sz = (float)GetScreenHeight() * FONT_SIZE_CARD_RATIO;

    if (perm->card_vocab_idx < 0) {
        // Empty slot — faint outline only
        DrawRectangleLinesEx((Rectangle){cx + 1, cy + 1, cw - 2, ch - 2}, 0.5f, (Color){190, 190, 190, 60});
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
    } else if (perm->is_tapped) {
        bg = (Color){185, 195, 210, 255};
        border_col = (Color){90, 110, 140, 255};
    } else if (perm->has_summoning_sickness) {
        bg = (Color){255, 255, 175, 255};
        border_col = ORANGE;
    } else {
        bg = (Color){238, 238, 238, 255};
        border_col = GRAY;
    }
    if (!perm->controller_is_self) {
        bg.r = (unsigned char)((int)bg.r * 85 / 100);
        bg.g = (unsigned char)((int)bg.g * 90 / 100);
    }

    // Card centre in screen space — constant regardless of tap state
    float card_cx = cx + cw * 0.5f;
    float card_cy = cy + ch * 0.5f;
    float rot = perm->is_tapped ? 90.0f : 0.0f;
    float hw = cw * 0.5f, hh = ch * 0.5f;

    // Filled rotated rectangle
    DrawRectanglePro((Rectangle){card_cx, card_cy, cw, ch}, (Vector2){hw, hh}, rot, bg);

    // Rotated border
    float cos_r = cosf(rot * DEG2RAD);
    float sin_r = sinf(rot * DEG2RAD);
    float lc[4][2] = {{-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh}};
    Vector2 sc[4];
    for (int i = 0; i < 4; i++) {
        sc[i].x = lc[i][0] * cos_r - lc[i][1] * sin_r + card_cx;
        sc[i].y = lc[i][0] * sin_r + lc[i][1] * cos_r + card_cy;
    }
    for (int i = 0; i < 4; i++) DrawLineEx(sc[i], sc[(i + 1) % 4], 1.5f, border_col);

    const char *name = gui_card_name(perm->card_vocab_idx);
    const char *oracle = gui_card_oracle(perm->card_vocab_idx);
    float oracle_sz = font_sz * 0.85f;

    if (!perm->is_tapped) {
        // Upright: scissor to card bounds, draw name / oracle / P/T normally
        BeginScissorMode((int)cx, (int)cy, (int)cw, (int)ch);

        DrawTextEx(g_font, name, (Vector2){cx + 3.0f, cy + 3.0f}, font_sz, 1.0f, BLACK);

        float oracle_y = cy + font_sz + 5.0f;
        float pt_reserve = perm->is_creature ? (font_sz + 5.0f) : 0.0f;
        float oracle_clip_h = ch - (oracle_y - cy) - pt_reserve - 3.0f;
        if (oracle_clip_h > oracle_sz && oracle[0] != '\0') {
            draw_wrapped_text(g_font_oracle, oracle, cx + 3.0f, oracle_y, cw - 6.0f, oracle_sz, DARKGRAY);
        }

        if (perm->is_creature) {
            char pt[32];
            if (perm->damage > 0)
                snprintf(pt, sizeof(pt), "%d/%d (%d)", perm->power, perm->toughness, perm->damage);
            else
                snprintf(pt, sizeof(pt), "%d/%d", perm->power, perm->toughness);
            DrawTextEx(g_font, pt, (Vector2){cx + 3.0f, cy + ch - font_sz - 3.0f}, font_sz, 1.0f, DARKGRAY);
        }
        EndScissorMode();
    } else {
        // Tapped 90° CW: card's local "up" points screen-right.
        // Draw text at -90° so it reads when the viewer tilts their head right.
        // Local offset (lx, ly) → screen: (card_cx - ly, card_cy + lx)
        float name_w = MeasureTextEx(g_font, name, font_sz, 1.0f).x;
        // Place name near the "top" of the card (local top = screen left)
        DrawTextPro(g_font, name,
            (Vector2){card_cx - hh + 3.0f, card_cy},
            (Vector2){0.0f, font_sz * 0.5f},
            -90.0f, font_sz, 1.0f, BLACK);

        if (perm->is_creature) {
            char pt[32];
            if (perm->damage > 0)
                snprintf(pt, sizeof(pt), "%d/%d (%d)", perm->power, perm->toughness, perm->damage);
            else
                snprintf(pt, sizeof(pt), "%d/%d", perm->power, perm->toughness);
            float pt_w = MeasureTextEx(g_font, pt, font_sz, 1.0f).x;
            // Place P/T near the "bottom" of the card (local bottom = screen right)
            DrawTextPro(g_font, pt,
                (Vector2){card_cx + hh - 3.0f, card_cy},
                (Vector2){pt_w, font_sz * 0.5f},
                -90.0f, font_sz, 1.0f, DARKGRAY);
        }
        (void)name_w;
    }

    // Hover detection — inverse-rotate mouse for tapped cards
    bool hovered;
    if (!perm->is_tapped) {
        hovered = (mouse.x >= cx && mouse.x < cx + cw && mouse.y >= cy && mouse.y < cy + ch);
    } else {
        float dx = mouse.x - card_cx;
        float dy = mouse.y - card_cy;
        float local_x = dy;   // inverse of 90° CW rotation
        float local_y = -dx;
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
    float label_sz = (float)GetScreenHeight() * FONT_SIZE_LABEL_RATIO;

    // Background
    DrawRectangle((int)px, (int)py, (int)pw, (int)ph, (Color){245, 245, 235, 255});
    DrawRectangleLinesEx((Rectangle){px, py, pw, ph}, 1.0f, LIGHTGRAY);
    DrawTextEx(g_font, label, (Vector2){px + 3.0f, py + 2.0f}, label_sz, 1.0f, GRAY);

    // Card dimensions from area height and aspect ratio
    float card_h = ph - 4.0f;
    float card_w = card_h / CARD_ASPECT_RATIO;
    float gap = LAYOUT_CARD_GAP;

    // Collect active permanents
    int active_idx[MAX_BATTLEFIELD_SLOTS];
    int active = 0;
    for (int i = 0; i < max_slots; i++) {
        if (perms[i].card_vocab_idx >= 0) active_idx[active++] = i;
    }

    // Draw active permanents left-to-right
    for (int d = 0; d < active; d++) {
        float card_x = px + gap + (float)d * (card_w + gap);
        draw_perm_card(card_x, py + 2.0f, card_w, card_h, &perms[active_idx[d]], mouse);
    }
    // Empty placeholder slots (up to 4 visible)
    int empties = (active < 4) ? (4 - active) : 0;
    for (int d = 0; d < empties; d++) {
        float card_x = px + gap + (float)(active + d) * (card_w + gap);
        DrawRectangleLinesEx(
            (Rectangle){card_x, py + 2.0f, card_w, card_h}, 0.5f, (Color){190, 190, 190, 60});
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

    float sw = (float)GetScreenWidth();
    float sh = (float)GetScreenHeight();
    Vector2 mouse = GetMousePosition();
    bool q_held = IsKeyDown(KEY_Q);

    // Derived layout values
    float px = sw * LAYOUT_SIDEBAR_W_RATIO + LAYOUT_MAIN_LEFT_PAD;
    float pw = sw - px - LAYOUT_RIGHT_PAD;
    float bar_sz = sh * FONT_SIZE_BAR_RATIO;
    float small_sz = sh * FONT_SIZE_LABEL_RATIO;

    float bar_slot = roundf(sh * LAYOUT_STATUS_BAR_H);
    float bar_h = bar_slot - 2.0f;
    float opp_bf_h = roundf(sh * LAYOUT_OPP_BF_H);
    float stack_h = roundf(sh * LAYOUT_STACK_H);
    float self_bf_h = roundf(sh * LAYOUT_SELF_BF_H);
    float self_hand_h = roundf(sh * LAYOUT_SELF_HAND_H);

    float y = 2.0f;

    // ── 1. Turn / Step bar ──────────────────────────────────────────────
    {
        char bar[256];
        snprintf(bar, sizeof(bar), "Turn %d  |  %s  |  %s  |  Priority: %s", gs->turn, gui_step_name((int)gs->cur_step),
            gs->is_active_player ? "Your turn" : "Opp turn", gs->viewer_has_priority ? "You" : "Opp");
        DrawRectangle((int)px, (int)y, (int)pw, (int)bar_h, (Color){55, 55, 75, 255});
        DrawTextEx(g_font, bar, (Vector2){px + 4.0f, y + (bar_h - bar_sz) * 0.5f}, bar_sz, 1.0f, WHITE);
        y += bar_slot;
    }

    // ── 2. Opponent info bar ────────────────────────────────────────────
    {
        char bar[256];
        snprintf(bar, sizeof(bar), "OPP  Life:%d  Poison:%d  W:%d U:%d B:%d R:%d G:%d C:%d  Hand:%d  Lib:%d",
            gs->opponent.life, gs->opponent.poison_counters, gs->opponent.mana[0], gs->opponent.mana[1],
            gs->opponent.mana[2], gs->opponent.mana[3], gs->opponent.mana[4], gs->opponent.mana[5],
            gs->opponent.hand_ct, gs->opp_library_ct);
        DrawRectangle((int)px, (int)y, (int)pw, (int)bar_h, (Color){185, 65, 65, 230});
        DrawTextEx(g_font, bar, (Vector2){px + 4.0f, y + (bar_h - bar_sz) * 0.5f}, bar_sz, 1.0f, WHITE);
        y += bar_slot;
    }

    // ── 3. Opponent battlefield ─────────────────────────────────────────
    {
        render_battlefield(px, y, pw, opp_bf_h, gs->opp_permanents, MAX_BATTLEFIELD_SLOTS, mouse, "Opponent");
        y += opp_bf_h + LAYOUT_SECTION_GAP;
    }

    // ── 4. Stack ────────────────────────────────────────────────────────
    {
        DrawRectangle((int)px, (int)y, (int)pw, (int)stack_h, (Color){215, 215, 170, 210});
        char stack_label[32];
        snprintf(stack_label, sizeof(stack_label), "Stack (%d)", gs->stack_size);
        DrawTextEx(g_font, stack_label, (Vector2){px + 4.0f, y + 3.0f}, small_sz, 1.0f, DARKGRAY);
        float ex = px + MeasureTextEx(g_font, stack_label, small_sz, 1.0f).x + small_sz * 2.0f;
        for (int i = 0; i < gs->stack_size && i < MAX_STACK_DISPLAY; i++) {
            const StackEntry *se = &gs->stack[i];
            const char *sname = (se->card_vocab_idx >= 0) ? gui_card_name(se->card_vocab_idx) : "?";
            char chip[80];
            snprintf(chip, sizeof(chip), "[%s (%s) - %s]", sname,
                se->is_spell ? "Spell" : "Trigger",
                se->controller_is_self ? "You" : "Opp");
            float chip_w = MeasureTextEx(g_font, chip, small_sz, 1.0f).x + 8.0f;
            float chip_h = small_sz * 1.8f;
            float chip_y = y + (stack_h - chip_h) * 0.5f;
            Color chip_col = se->controller_is_self ? (Color){90, 150, 240, 220} : (Color){240, 110, 90, 220};
            DrawRectangle((int)ex, (int)chip_y, (int)chip_w, (int)chip_h, chip_col);
            DrawTextEx(g_font, chip, (Vector2){ex + 4.0f, chip_y + (chip_h - small_sz) * 0.5f}, small_sz, 1.0f, WHITE);
            ex += chip_w + 4.0f;
        }
        y += stack_h + LAYOUT_SECTION_GAP;
    }

    // ── 5. Self battlefield ─────────────────────────────────────────────
    {
        render_battlefield(px, y, pw, self_bf_h, gs->self_permanents, MAX_BATTLEFIELD_SLOTS, mouse, "You");
        y += self_bf_h + LAYOUT_SECTION_GAP;
    }

    // ── 6. Self hand ────────────────────────────────────────────────────
    {
        float card_h = self_hand_h - 4.0f;
        float card_w = card_h / CARD_ASPECT_RATIO;
        float gap = LAYOUT_CARD_GAP;
        float font_card = sh * FONT_SIZE_CARD_RATIO;
        float font_tiny = sh * FONT_SIZE_TINY_RATIO;

        DrawRectangle((int)px, (int)y, (int)pw, (int)self_hand_h, (Color){200, 225, 200, 190});

        int col = 0;
        for (int i = 0; i < MAX_HAND_SLOTS; i++) {
            int vi = gs->self_hand[i];
            if (vi < 0) continue;

            float card_x = px + gap + (float)col * (card_w + gap);
            float card_y = y + 2.0f;

            const char *cname = gui_card_name(vi);
            const char *ctype = gui_card_type_line(vi);
            const char *coracle = gui_card_oracle(vi);

            DrawRectangle((int)card_x, (int)card_y, (int)card_w, (int)card_h, (Color){195, 225, 195, 255});
            DrawRectangleLinesEx((Rectangle){card_x, card_y, card_w, card_h}, 1.5f, DARKGREEN);

            float name_y = card_y + 3.0f;
            float type_y = card_y + font_card + 5.0f;
            float oracle_y = card_y + font_card + font_tiny + 8.0f;
            float oracle_clip_h = card_h - (oracle_y - card_y) - 3.0f;

            BeginScissorMode((int)card_x, (int)card_y, (int)card_w, (int)card_h);
            DrawTextEx(g_font, cname, (Vector2){card_x + 3.0f, name_y}, font_card, 1.0f, BLACK);
            DrawTextEx(g_font_oracle, ctype, (Vector2){card_x + 3.0f, type_y}, font_tiny, 1.0f, DARKGRAY);
            if (oracle_clip_h > font_tiny && coracle[0] != '\0') {
                draw_wrapped_text(g_font_oracle, coracle, card_x + 3.0f, oracle_y, card_w - 6.0f, font_tiny, DARKGRAY);
            }
            EndScissorMode();

            // Hover check for tooltip
            if (mouse.x >= card_x && mouse.x < card_x + card_w && mouse.y >= card_y && mouse.y < card_y + card_h) {
                gs_hover_vocab_idx = vi;
                gs_hover_tx = mouse.x;
                gs_hover_ty = mouse.y;
                gs_hover_perm_power = -1;
                gs_hover_perm_toughness = -1;
                gs_hover_perm_damage = 0;
            }
            col++;
        }
        y += self_hand_h + LAYOUT_SECTION_GAP;
    }

    // ── 7. Self info bar ────────────────────────────────────────────────
    {
        char bar[256];
        int self_gy = count_gy(gs->self_graveyard, MAX_GY_SLOTS);
        snprintf(bar, sizeof(bar), "YOU  Life:%d  Poison:%d  W:%d U:%d B:%d R:%d G:%d C:%d  Lands:%d  Lib:%d  GY:%d",
            gs->self.life, gs->self.poison_counters, gs->self.mana[0], gs->self.mana[1], gs->self.mana[2],
            gs->self.mana[3], gs->self.mana[4], gs->self.mana[5], gs->self.lands_played_this_turn, gs->self_library_ct,
            self_gy);
        DrawRectangle((int)px, (int)y, (int)pw, (int)bar_h, (Color){50, 95, 50, 255});
        DrawTextEx(g_font, bar, (Vector2){px + 4.0f, y + (bar_h - bar_sz) * 0.5f}, bar_sz, 1.0f, WHITE);
        y += bar_slot;
    }

    // ── 8. Graveyard summary ────────────────────────────────────────────
    {
        float gy_line_h = small_sz * 1.3f;
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
        y += gy_line_h;

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

        float tp_w = sw * LAYOUT_TOOLTIP_W_RATIO;
        float tp_h = sh * LAYOUT_TOOLTIP_H_RATIO;
        float tp_x = gs_hover_tx + 14.0f;
        float tp_y = gs_hover_ty - tp_h * 0.5f;
        if (tp_x + tp_w > sw - 5.0f) tp_x = gs_hover_tx - tp_w - 14.0f;
        if (tp_y < 5.0f) tp_y = 5.0f;
        if (tp_y + tp_h > sh - 5.0f) tp_y = sh - tp_h - 5.0f;

        float tt_title_sz = sh * FONT_SIZE_LABEL_RATIO * 1.2f;
        float tt_body_sz = sh * FONT_SIZE_CARD_RATIO;
        float tt_line_h = tt_body_sz * 1.4f;

        DrawRectangle((int)tp_x, (int)tp_y, (int)tp_w, (int)tp_h, (Color){252, 248, 218, 252});
        DrawRectangleLinesEx((Rectangle){tp_x, tp_y, tp_w, tp_h}, 2.0f, (Color){100, 80, 20, 255});

        float ty = tp_y + tp_h * 0.03f;
        DrawTextEx(g_font, tname, (Vector2){tp_x + 6.0f, ty}, tt_title_sz, 1.0f, BLACK);
        ty += tt_title_sz + tt_line_h * 0.3f;
        DrawTextEx(g_font, ttype, (Vector2){tp_x + 6.0f, ty}, tt_body_sz, 1.0f, DARKGRAY);
        ty += tt_line_h;

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
            DrawTextEx(g_font, pt, (Vector2){tp_x + 6.0f, ty}, tt_body_sz, 1.0f, BLACK);
            ty += tt_line_h;
        }

        float clip_h = (tp_y + tp_h) - ty - 4.0f;
        BeginScissorMode((int)(tp_x + 4.0f), (int)ty, (int)(tp_w - 8.0f), (int)clip_h);
        draw_wrapped_text(g_font_oracle, toracle, tp_x + 6.0f, ty, tp_w - 12.0f, tt_body_sz, BLACK);
        EndScissorMode();
    }
}

// Scrollable box that displays everything that would be propagated to the CLI
static void render_info_log(void) {
    static Vector2 scroll = {0, 0};
    static int last_line_count = 0;
    static float max_content_width = 0;

    float sh = (float)GetScreenHeight();
    float sw = (float)GetScreenWidth();
    float font_size = sh * FONT_SIZE_MAIN_RATIO;
    float line_height = font_size * 1.15f;

    Rectangle bounds = {
        LAYOUT_SIDEBAR_PAD,
        LAYOUT_SIDEBAR_PAD,
        sw * LAYOUT_SIDEBAR_W_RATIO - LAYOUT_SIDEBAR_PAD * 2.0f,
        sh * LAYOUT_LOG_BOTTOM_RATIO - LAYOUT_SIDEBAR_PAD,
    };

    int line_count = gui_log_line_count();

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

static void render_choices(void) {
    float sh = (float)GetScreenHeight();
    float sw = (float)GetScreenWidth();
    float font_size = sh * FONT_SIZE_CHOICES_RATIO;
    float line_height = font_size * 1.55f;
    float panel_x = LAYOUT_SIDEBAR_PAD;
    float panel_w = sw * LAYOUT_SIDEBAR_W_RATIO - LAYOUT_SIDEBAR_PAD * 2.0f;
    float y = sh * LAYOUT_CHOICES_Y_RATIO;
    Vector2 mouse = GetMousePosition();
    bool clicked = IsMouseButtonPressed(MOUSE_LEFT_BUTTON);

    int line_count = gui_query_line_count();
    //first line is always prompt
    for (int i = 0; i < line_count; i++) {
        const char *line = gui_query_get_line(i);
        //ASSUMES FIRST LINE IS ALWAYS PROMPT
        if(i != 0){
            Rectangle row = {panel_x, y, panel_w, line_height - 2.0f};
            bool hovered = CheckCollisionPointRec(mouse, row);
            Color bg = hovered ? (Color){180, 210, 255, 220} : (Color){220, 235, 255, 180};
            DrawRectangleRec(row, bg);
            DrawRectangleLinesEx(row, 1.0f, (Color){90, 130, 200, 160});
            DrawTextEx(g_font, line, (Vector2){panel_x + 6.0f, y + (line_height - font_size) * 0.4f},
                font_size, 1.0f, (Color){20, 40, 160, 255});
            if (hovered && clicked && gui_input_requested) {
                pthread_mutex_lock(&input_mutex);
                //ASSUMES CHOICES ALWAYS STARTING WITH 0 - TO VERIFY
                gui_cmd = i - 1;
                gui_input_sent = true;
                pthread_mutex_unlock(&input_mutex);
            }
        } else {
            // Header / label line
            DrawTextEx(g_font, line, (Vector2){panel_x, y + (line_height - font_size) * 0.4f},
                font_size, 1.0f, (Color){40, 40, 160, 255});
        }
        y += line_height;
    }
}

// Button bar: bar.x/y/width/height define the full available strip.
// Buttons fill left-to-right; empty slots are left blank.
static void render_button_bar(Rectangle bar) {
    float btn_w = (bar.width - (LAYOUT_N_BUTTONS - 1) * LAYOUT_BUTTON_GAP) / LAYOUT_N_BUTTONS;
    float btn_h = bar.height;

    // Slot 0: QUIT
    Rectangle quit_rect = {bar.x, bar.y, btn_w, btn_h};
    if (GuiButton(quit_rect, "QUIT")) {
        pthread_mutex_lock(&input_mutex);
        gui_cmd = FLAG_QUIT;
        gui_input_sent = true;
        pthread_mutex_unlock(&input_mutex);
    }

    // Slots 1-5: reserved for future buttons
}

static void *gui_loop(void *arg) {
    determine_screen_size();

    int monitor = GetCurrentMonitor();
    int font_load_size = GetMonitorHeight(monitor) / FONT_LOAD_DIVISOR;
    char font_path[512];
    snprintf(font_path, sizeof(font_path), "%s/Magicmedieval-pRV1.ttf", gui_resource_dir);
    g_font = LoadFontEx(font_path, font_load_size, NULL, 0);
    char oracle_font_path[512];
    snprintf(oracle_font_path, sizeof(oracle_font_path), "%s/Merriweather-VariableFont_opsz,wdth,wght.ttf",
        gui_resource_dir);
    g_font_oracle = LoadFontEx(oracle_font_path, font_load_size, NULL, 0);

    bool input_focused = false;

    while (!WindowShouldClose() && !quit_gui) {
        SCREEN_WIDTH = GetScreenWidth();
        SCREEN_HEIGHT = GetScreenHeight();

        BeginDrawing();
        ClearBackground(WHITE);

        float sh = (float)SCREEN_HEIGHT;
        float sw = (float)SCREEN_WIDTH;
        float input_w = sw * LAYOUT_INPUT_W_RATIO;
        float input_h = sh * LAYOUT_INPUT_H_RATIO;
        float bottom_y = sh - input_h - LAYOUT_INPUT_MARGIN;
        Rectangle input_rect = {
            sw - input_w - LAYOUT_INPUT_MARGIN,
            bottom_y,
            input_w,
            input_h,
        };

        // Button bar: from sidebar right edge to input box left edge
        float bar_x = sw * LAYOUT_SIDEBAR_W_RATIO + LAYOUT_SIDEBAR_PAD;
        float bar_right = input_rect.x - LAYOUT_BUTTON_GAP;
        Rectangle button_bar = {bar_x, bottom_y, bar_right - bar_x, input_h};
        render_button_bar(button_bar);

        // Click outside the box → lose focus; click inside or press Enter → gain focus
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            Vector2 m = GetMousePosition();
            input_focused = CheckCollisionPointRec(m, input_rect);
        } 

        if (GuiTextBox(input_rect, gui_input, GUI_INPUT_MAX, input_focused) == 1) {
            if (gui_input_requested) {
                for (size_t i = 0; i < GUI_INPUT_MAX; ++i) {
                    if (!isdigit(gui_input[i]) && gui_input[i] != '\0') {
                        memset(gui_input, '\0', GUI_INPUT_MAX);
                        break;
                    }
                }
                if (gui_input[0] == '\0') goto INPUT_END;
                pthread_mutex_lock(&input_mutex);
                gui_cmd = atoi(gui_input);
                gui_input_sent = true;
                pthread_mutex_unlock(&input_mutex);
                memset(gui_input, '\0', GUI_INPUT_MAX);
            }
        }
    INPUT_END:
        render_gs();
        render_info_log();
        render_choices();
        EndDrawing();
    }
    UnloadFont(g_font);
    UnloadFont(g_font_oracle);
    gui_killed = true;
    return NULL;
}

void init_gui() {
    gui_loop(NULL);
}

#endif
