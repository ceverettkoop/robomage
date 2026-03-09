#include "cli_output.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>

#include "card_vocab.h"
#include "input_logger.h"
#include "machine_io.h"

extern bool gui_mode;

// ── GUI log ring buffer ───────────────────────────────────────────────────────

#define GUI_LOG_CAPACITY 512
#define GUI_LOG_LINE_LEN 512

static char g_gui_log[GUI_LOG_CAPACITY][GUI_LOG_LINE_LEN];
static int  g_gui_log_head  = 0;
static int  g_gui_log_count = 0;

#define GUI_QUERY_CAPACITY 64

static char g_query_buf[GUI_QUERY_CAPACITY][GUI_LOG_LINE_LEN];
static int  g_query_count = 0;

static pthread_mutex_t g_buf_mutex = PTHREAD_MUTEX_INITIALIZER;

// ── Utility ───────────────────────────────────────────────────────────────────

const char* step_to_string(Step in_step) {
    switch (in_step) {
        case UNTAP:             return "Untap";
        case UPKEEP:            return "Upkeep";
        case DRAW:              return "Draw";
        case FIRST_MAIN:        return "First Main";
        case BEGIN_COMBAT:      return "Begin Combat";
        case DECLARE_ATTACKERS: return "Declare Attackers";
        case DECLARE_BLOCKERS:  return "Declare Blockers";
        case COMBAT_DAMAGE:     return "Combat Damage";
        case END_OF_COMBAT:     return "End of Combat";
        case SECOND_MAIN:       return "Second Main";
        case END_STEP:          return "End Step";
        case CLEANUP:           return "Cleanup";
        default:                return "ERROR UNREACHABLE";
    }
}

std::string player_name(Zone::Ownership owner) {
    return (owner == Zone::PLAYER_A) ? "Player A" : "Player B";
}

static const char* mana_symbol(int idx) {
    switch (idx) {
        case 0: return "{W}";
        case 1: return "{U}";
        case 2: return "{B}";
        case 3: return "{R}";
        case 4: return "{G}";
        case 5: return "{C}";
        default: return "{?}";
    }
}

// ── game_log ──────────────────────────────────────────────────────────────────

static void game_log_va(const char* fmt, va_list ap) {
    if (gui_mode) {
        char buf[GUI_LOG_LINE_LEN];
        vsnprintf(buf, sizeof(buf), fmt, ap);
        pthread_mutex_lock(&g_buf_mutex);
        char* p = buf;
        while (true) {
            char* nl = strchr(p, '\n');
            if (nl) *nl = '\0';
            // store segment if it has content, or if there's a following newline (blank line)
            if (nl != NULL || *p != '\0') {
                strncpy(g_gui_log[g_gui_log_head], p, GUI_LOG_LINE_LEN - 1);
                g_gui_log[g_gui_log_head][GUI_LOG_LINE_LEN - 1] = '\0';
                g_gui_log_head = (g_gui_log_head + 1) % GUI_LOG_CAPACITY;
                if (g_gui_log_count < GUI_LOG_CAPACITY) g_gui_log_count++;
            }
            if (!nl) break;
            p = nl + 1;
        }
        pthread_mutex_unlock(&g_buf_mutex);
    } else {
        vprintf(fmt, ap);
    }
}

void game_log(const char* fmt, ...) {
    if (InputLogger::instance().is_machine_mode() && !gui_mode) return;
    va_list ap;
    va_start(ap, fmt);
    game_log_va(fmt, ap);
    va_end(ap);
}

void game_log_private(Zone::Ownership private_to, const char* fmt, ...) {
    if (InputLogger::instance().is_machine_mode() && !gui_mode) return;
    extern bool has_human_player;
    extern bool human_player_is_a;
    if (has_human_player) {
        Zone::Ownership human_owner = human_player_is_a ? Zone::PLAYER_A : Zone::PLAYER_B;
        if (private_to != human_owner) return;
    }
    va_list ap;
    va_start(ap, fmt);
    game_log_va(fmt, ap);
    va_end(ap);
}

// ── C-API buffer accessors ────────────────────────────────────────────────────

extern "C" {

int gui_log_line_count(void) {
    pthread_mutex_lock(&g_buf_mutex);
    int c = g_gui_log_count;
    pthread_mutex_unlock(&g_buf_mutex);
    return c;
}

const char* gui_log_get_line(int idx) {
    pthread_mutex_lock(&g_buf_mutex);
    int actual;
    if (g_gui_log_count < GUI_LOG_CAPACITY) {
        actual = idx;
    } else {
        actual = (g_gui_log_head + idx) % GUI_LOG_CAPACITY;
    }
    const char* result = (idx >= 0 && idx < g_gui_log_count) ? g_gui_log[actual] : "";
    pthread_mutex_unlock(&g_buf_mutex);
    return result;
}

void gui_query_clear(void) {
    pthread_mutex_lock(&g_buf_mutex);
    g_query_count = 0;
    pthread_mutex_unlock(&g_buf_mutex);
}

int gui_query_line_count(void) {
    pthread_mutex_lock(&g_buf_mutex);
    int c = g_query_count;
    pthread_mutex_unlock(&g_buf_mutex);
    return c;
}

const char* gui_query_get_line(int idx) {
    pthread_mutex_lock(&g_buf_mutex);
    const char* result = (idx >= 0 && idx < g_query_count) ? g_query_buf[idx] : "";
    pthread_mutex_unlock(&g_buf_mutex);
    return result;
}

} // extern "C"

// ── Machine query emitter ─────────────────────────────────────────────────────

void cli_emit_machine_query(const Query* q, const GameState* gs) {
    auto state_vec = serialize_state(gs);
    printf("QUERY: %d", q->num_choices);
    for (float f : state_vec) printf(" %.4f", f);
    for (int i = 0; i < q->num_choices; i++) printf(" %d", q->choices[i].category);
    const float id_null = -1.0f / static_cast<float>(N_CARD_TYPES);
    for (int i = 0; i < q->num_choices; i++) {
        float id_f = q->choices[i].card_vocab_idx >= 0
            ? static_cast<float>(q->choices[i].card_vocab_idx) / static_cast<float>(N_CARD_TYPES)
            : id_null;
        printf(" %.4f", id_f);
    }
    const float ctrl_null = -1.0f / static_cast<float>(N_CARD_TYPES);
    for (int i = 0; i < q->num_choices; i++) {
        float ctrl = (q->choices[i].zone_ref != REF_NONE)
            ? (q->choices[i].controller_is_self ? 1.0f : 0.0f)
            : ctrl_null;
        printf(" %.4f", ctrl);
    }
    printf("\n");
    fflush(stdout);
}

// ── Startup / meta ────────────────────────────────────────────────────────────

void cli_print_version(const char* version) {
    game_log("robomage %s\n", version);
}

void cli_print_help(const char* program, const char* version) {
    printf("robomage %s\n", version);
    printf("Usage: %s [options]\n", program);
    printf("Options:\n");
    printf("  --replay <logfile>  Replay a previously logged game\n");
    printf("  --machine           Machine mode: emit QUERY lines for AI input\n");
    printf("  --help, -h          Show this help message\n");
    printf("  --gui               Launch with GUI\n");
}

void cli_print_seed(unsigned int seed) {
    game_log("Using seed: %u\n", seed);
}

void cli_print_turn_header(size_t turn, bool player_a_turn) {
    const char* name = player_a_turn ? "Player A" : "Player B";
    game_log("\n======== TURN %zu (%s) ========\n", turn, name);
}

void cli_print_invalid_action() {
    game_log("Invalid action\n");
}

void cli_print_gui_exit() {
    printf("User exited GUI, quitting\n");
}

// ── Game state display ────────────────────────────────────────────────────────

void print_game_state(const GameState* gs) {
    if (InputLogger::instance().is_machine_mode() && !gui_mode) return;

    const char* self_name   = gs->self_is_player_a ? "Player A" : "Player B";
    const char* opp_name    = gs->self_is_player_a ? "Player B" : "Player A";
    const char* active_name = gs->is_active_player  ? self_name : opp_name;

    game_log("\n=== Turn %d - %s's %s ===\n", gs->turn, active_name, step_to_string(gs->cur_step));
    game_log("Life: Player A=%d, Player B=%d\n",
             gs->self_is_player_a ? gs->self.life : gs->opponent.life,
             gs->self_is_player_a ? gs->opponent.life : gs->self.life);

    if (gs->stack_size > 0) {
        game_log("STACK:\n");
        int display = gs->stack_size < MAX_STACK_DISPLAY ? gs->stack_size : MAX_STACK_DISPLAY;
        for (int i = 0; i < display; i++) {
            const StackEntry* se = &gs->stack[i];
            const char* cname = se->card_vocab_idx >= 0 ? card_index_to_name(se->card_vocab_idx) : "Unknown";
            game_log("  %d: %s\n", i, cname);
        }
    }

    game_log("\n--- BATTLEFIELD ---\n");

    const PermanentState* perm_arrays[2] = { gs->self_permanents, gs->opp_permanents };
    const char* perm_names[2] = { self_name, opp_name };

    for (int p = 0; p < 2; p++) {
        game_log("%s:\n", perm_names[p]);
        bool found_any = false;
        for (int i = 0; i < MAX_BATTLEFIELD_SLOTS; i++) {
            const PermanentState* slot = &perm_arrays[p][i];
            if (slot->card_vocab_idx < 0) continue;
            found_any = true;
            char line[512];
            int len = snprintf(line, sizeof(line), "  %s", card_index_to_name(slot->card_vocab_idx));
            if (slot->is_creature && len < (int)sizeof(line))
                len += snprintf(line + len, (size_t)((int)sizeof(line) - len), " [%d/%d]", slot->power, slot->toughness);
            if (slot->is_tapped && len < (int)sizeof(line))
                len += snprintf(line + len, (size_t)((int)sizeof(line) - len), " (TAPPED)");
            if (slot->has_summoning_sickness && len < (int)sizeof(line))
                len += snprintf(line + len, (size_t)((int)sizeof(line) - len), " (SICK)");
            if (slot->damage > 0 && len < (int)sizeof(line))
                snprintf(line + len, (size_t)((int)sizeof(line) - len), " (%d damage)", slot->damage);
            game_log("%s\n", line);
        }
        if (!found_any) game_log("  (no permanents)\n");
    }

    bool header_printed = false;
    const PlayerState* players[2] = { &gs->self, &gs->opponent };
    const char* player_names_arr[2] = { self_name, opp_name };
    for (int p = 0; p < 2; p++) {
        bool has_mana = false;
        for (int i = 0; i < 6; i++) {
            if (players[p]->mana[i] > 0) { has_mana = true; break; }
        }
        if (!has_mana) continue;
        if (!header_printed) {
            game_log("\n--- MANA POOLS ---\n");
            header_printed = true;
        }
        char line[256];
        int len = snprintf(line, sizeof(line), "%s:", player_names_arr[p]);
        for (int i = 0; i < 6; i++) {
            for (int n = 0; n < players[p]->mana[i]; n++) {
                if (len < (int)sizeof(line))
                    len += snprintf(line + len, (size_t)((int)sizeof(line) - len), " %s", mana_symbol(i));
            }
        }
        game_log("%s\n", line);
    }

    extern bool has_human_player;
    extern bool human_player_is_a;
    bool show_hand = !has_human_player || (human_player_is_a == gs->self_is_player_a);
    if (show_hand) {
        game_log("\n%s hand:\n", self_name);
        bool hand_empty = true;
        for (int i = 0; i < MAX_HAND_SLOTS; i++) {
            if (gs->self_hand[i] < 0) continue;
            hand_empty = false;
            game_log("  %s\n", card_index_to_name(gs->self_hand[i]));
        }
        if (hand_empty) game_log("  (empty)\n");
    }
}

// ── Choice display ────────────────────────────────────────────────────────────

void print_query(const Query* q, bool player_a_has_priority) {
    if (InputLogger::instance().is_machine_mode() && !gui_mode) return;
    const char* pname = player_a_has_priority ? "Player A" : "Player B";
    if (gui_mode) {
        pthread_mutex_lock(&g_buf_mutex);
        g_query_count = 0;
        char header[256];
        snprintf(header, sizeof(header), "%s has priority. Legal actions:", pname);
        strncpy(g_query_buf[0], header, GUI_LOG_LINE_LEN - 1);
        g_query_buf[0][GUI_LOG_LINE_LEN - 1] = '\0';
        g_query_count = 1;
        for (int i = 0; i < q->num_choices && g_query_count < GUI_QUERY_CAPACITY; i++) {
            char line[512];
            snprintf(line, sizeof(line), "  %d: %s", i, q->choices[i].description);
            strncpy(g_query_buf[g_query_count], line, GUI_LOG_LINE_LEN - 1);
            g_query_buf[g_query_count][GUI_LOG_LINE_LEN - 1] = '\0';
            g_query_count++;
        }
        pthread_mutex_unlock(&g_buf_mutex);
    } else {
        game_log("\n%s has priority. Legal actions:\n", pname);
        for (int i = 0; i < q->num_choices; i++) {
            game_log("  %d: %s\n", i, q->choices[i].description);
        }
    }
}

// ── Errors ────────────────────────────────────────────────────────────────────

void cli_error(const std::string& msg) {
    printf("%s\n", msg.c_str());
}

[[noreturn]] void cli_fatal_error(const std::string& msg) {
    printf("Fatal error: %s\n", msg.c_str());
    exit(1);
}
