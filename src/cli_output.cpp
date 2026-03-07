#include "cli_output.h"

#include <cstdio>
#include <cstdlib>
#include "card_vocab.h"
#include "input_logger.h"

// ── Utility ──────────────────────────────────────────────────────────────────

const char* step_to_string(Step in_step) {
    switch (in_step) {
        case UNTAP:            return "Untap";
        case UPKEEP:           return "Upkeep";
        case DRAW:             return "Draw";
        case FIRST_MAIN:       return "First Main";
        case BEGIN_COMBAT:     return "Begin Combat";
        case DECLARE_ATTACKERS: return "Declare Attackers";
        case DECLARE_BLOCKERS: return "Declare Blockers";
        case COMBAT_DAMAGE:    return "Combat Damage";
        case END_OF_COMBAT:    return "End of Combat";
        case SECOND_MAIN:      return "Second Main";
        case END_STEP:         return "End Step";
        case CLEANUP:          return "Cleanup";
        default:               return "ERROR UNREACHABLE";
    }
}

static const char* mana_symbol(int idx) {
    // 0=W, 1=U, 2=B, 3=R, 4=G, 5=C
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

// ── Startup / meta ───────────────────────────────────────────────────────────

void cli_print_version(const char* version) {
    if (InputLogger::instance().is_machine_mode()) return;
    printf("robomage %s\n", version);
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
    if (InputLogger::instance().is_machine_mode()) return;
    printf("Using seed: %u\n", seed);
}

void cli_print_turn_header(size_t turn, bool player_a_turn) {
    if (InputLogger::instance().is_machine_mode()) return;
    const char* name = player_a_turn ? "Player A" : "Player B";
    printf("\n======== TURN %zu (%s) ========\n", turn, name);
}

void cli_print_invalid_action() {
    if (InputLogger::instance().is_machine_mode()) return;
    printf("Invalid action\n");
}

void cli_print_gui_exit() {
    printf("User exited GUI, quitting\n");
}

// ── Game state display ────────────────────────────────────────────────────────

void print_game_state(const GameState* gs) {
    if (InputLogger::instance().is_machine_mode()) return;

    // Determine display names
    const char* self_name = gs->self_is_player_a ? "Player A" : "Player B";
    const char* opp_name  = gs->self_is_player_a ? "Player B" : "Player A";
    const char* active_name = gs->is_active_player ? self_name : opp_name;

    // Turn / step header + life totals
    printf("\n=== Turn %d - %s's %s ===\n", gs->turn, active_name, step_to_string(gs->cur_step));
    printf("Life: Player A=%d, Player B=%d\n",
           gs->self_is_player_a ? gs->self.life : gs->opponent.life,
           gs->self_is_player_a ? gs->opponent.life : gs->self.life);

    // Stack
    if (gs->stack_size > 0) {
        printf("STACK:\n");
        int display = gs->stack_size < MAX_STACK_DISPLAY ? gs->stack_size : MAX_STACK_DISPLAY;
        for (int i = 0; i < display; i++) {
            const StackEntry* se = &gs->stack[i];
            const char* cname = se->card_vocab_idx >= 0 ? card_index_to_name(se->card_vocab_idx) : "Unknown";
            printf("  %d: %s\n", i, cname);
        }
    }

    // Battlefield
    printf("\n--- BATTLEFIELD ---\n");

    // Print both players; self first, then opp
    const PermanentState* perm_arrays[2] = { gs->self_permanents, gs->opp_permanents };
    const char* perm_names[2] = { self_name, opp_name };

    for (int p = 0; p < 2; p++) {
        printf("%s:\n", perm_names[p]);
        bool found_any = false;
        for (int i = 0; i < MAX_BATTLEFIELD_SLOTS; i++) {
            const PermanentState* slot = &perm_arrays[p][i];
            if (slot->card_vocab_idx < 0) continue;
            found_any = true;
            printf("  %s", card_index_to_name(slot->card_vocab_idx));
            if (slot->is_creature)
                printf(" [%d/%d]", slot->power, slot->toughness);
            if (slot->is_tapped) printf(" (TAPPED)");
            if (slot->has_summoning_sickness) printf(" (SICK)");
            if (slot->damage > 0) printf(" (%d damage)", slot->damage);
            printf("\n");
        }
        if (!found_any) printf("  (no permanents)\n");
    }

    // Mana pools
    bool header_printed = false;
    const PlayerState* players[2] = { &gs->self, &gs->opponent };
    const char* player_names[2] = { self_name, opp_name };
    for (int p = 0; p < 2; p++) {
        bool has_mana = false;
        for (int i = 0; i < 6; i++) {
            if (players[p]->mana[i] > 0) { has_mana = true; break; }
        }
        if (!has_mana) continue;
        if (!header_printed) {
            printf("\n--- MANA POOLS ---\n");
            header_printed = true;
        }
        printf("%s:", player_names[p]);
        for (int i = 0; i < 6; i++) {
            for (int n = 0; n < players[p]->mana[i]; n++)
                printf(" %s", mana_symbol(i));
        }
        printf("\n");
    }

    // Self hand (priority player's hand)
    printf("\n%s hand:\n", self_name);
    bool hand_empty = true;
    for (int i = 0; i < MAX_HAND_SLOTS; i++) {
        if (gs->self_hand[i] < 0) continue;
        hand_empty = false;
        printf("  %s\n", card_index_to_name(gs->self_hand[i]));
    }
    if (hand_empty) printf("  (empty)\n");
}

// ── Choice display ────────────────────────────────────────────────────────────

void print_query(const Query* q, bool player_a_has_priority) {
    if (InputLogger::instance().is_machine_mode()) return;
    const char* pname = player_a_has_priority ? "Player A" : "Player B";
    printf("\n%s has priority. Legal actions:\n", pname);
    for (int i = 0; i < q->num_choices; i++) {
        printf("  %d: %s\n", i, q->choices[i].description);
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
