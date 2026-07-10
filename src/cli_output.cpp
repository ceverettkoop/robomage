#include "cli_output.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "card_vocab.h"
#include "classes/colors.h"
#include "input_logger.h"
#include "machine_io.h"

// Formats a player's non-mana resource counters (poison, energy) into `buf` as a
// " poison N, energy N" suffix, listing only the counters that are nonzero. Writes
// an empty string when the player has neither. Display only.
static void format_player_resources(const PlayerState& ps, char* buf, size_t buf_len);

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
        case FIRST_STRIKE_DAMAGE: return "First Strike Damage";
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

// ── game_log ──────────────────────────────────────────────────────────────────

static void game_log_va(const char* fmt, va_list ap) {
    vprintf(fmt, ap);
}

extern bool narrative_mode;

void game_log(const char* fmt, ...) {
    if (InputLogger::instance().is_machine_mode() && !narrative_mode) return;
    va_list ap;
    va_start(ap, fmt);
    game_log_va(fmt, ap);
    va_end(ap);
}

// Resolve the perspective that one-sided narrative output should be limited to.
// Returns true and sets *owner when output is restricted: a human is at the
// keyboard (--player) or a spectator view is pinned (--log-viewer, used by the
// TUI which drives both seats over the machine protocol but still wants
// one-sided narrative). Returns false for full perfect-information narrative.
bool resolve_narrative_viewer(Zone::Ownership* owner) {
    extern bool has_human_player;
    extern bool human_player_is_a;
    extern bool log_viewer_set;
    extern Zone::Ownership log_viewer_owner;
    if (has_human_player) {
        *owner = human_player_is_a ? Zone::PLAYER_A : Zone::PLAYER_B;
        return true;
    }
    if (log_viewer_set) {
        *owner = log_viewer_owner;
        return true;
    }
    return false;
}

void game_log_private(Zone::Ownership private_to, const char* fmt, ...) {
    if (InputLogger::instance().is_machine_mode() && !narrative_mode) return;
    Zone::Ownership viewer;
    if (resolve_narrative_viewer(&viewer) && private_to != viewer) return;
    va_list ap;
    va_start(ap, fmt);
    game_log_va(fmt, ap);
    va_end(ap);
}

// Complement of game_log_private: emits only to a pinned viewer who is NOT
// `owner` — the redacted/generic line an opponent sees in place of hidden info
// (e.g. "draws a card" instead of "draws Ponder"). Silent when narrative is
// unrestricted or the viewer IS the owner, so pairing it with game_log_private
// on the same event yields exactly one line in every perspective.
void game_log_redacted(Zone::Ownership owner, const char* fmt, ...) {
    if (InputLogger::instance().is_machine_mode() && !narrative_mode) return;
    Zone::Ownership viewer;
    if (!resolve_narrative_viewer(&viewer) || viewer == owner) return;
    va_list ap;
    va_start(ap, fmt);
    game_log_va(fmt, ap);
    va_end(ap);
}

// ── Machine query emitter ─────────────────────────────────────────────────────

void cli_emit_machine_query(const Query* q, const GameState* gs) {
    const auto& state_vec = serialize_state(gs);

    // Text header line: "BQUERY: N STATE_SIZE MAX_ACTIONS\n"
    // The two trailing sizes are a runtime layout handshake: the Python driver
    // asserts them against its own imported STATE_SIZE / MAX_ACTIONS so a C++
    // layout change without regenerated Python constants fails loudly instead of
    // silently misframing the binary payload.
    // Followed immediately by binary payload (no text parsing needed on Python side).
    printf("BQUERY: %d %d %d\n", q->num_choices, STATE_SIZE, MAX_ACTIONS);

    // Binary state vector (STATE_SIZE float32s)
    fwrite(state_vec.data(), sizeof(float), static_cast<size_t>(STATE_SIZE), stdout);

    // Per-action metadata padded to MAX_ACTIONS for fixed-size reads.
    const float id_null   = -1.0f / static_cast<float>(N_CARD_TYPES);
    const float ctrl_null = -1.0f / static_cast<float>(N_CARD_TYPES);
    int32_t cats[MAX_ACTIONS]  = {};
    float   ids [MAX_ACTIONS];
    float   ctrl[MAX_ACTIONS];
    float   pub [MAX_ACTIONS]  = {};  // 1.0 = card identity publicly known (revealed), else 0.0
    int32_t zone[MAX_ACTIONS]  = {};  // ActionRefZone of the choice's entity (REF_NONE = none)
    int32_t refs[MAX_ACTIONS];        // entity-reference slot of the choice's source (-1 = none)
    std::fill(ids,  ids  + MAX_ACTIONS, id_null);
    std::fill(ctrl, ctrl + MAX_ACTIONS, ctrl_null);
    std::fill(refs, refs + MAX_ACTIONS, static_cast<int32_t>(-1));

    for (int i = 0; i < q->num_choices; i++) {
        cats[i] = q->choices[i].category;
        ids[i]  = q->choices[i].card_vocab_idx >= 0
            ? static_cast<float>(q->choices[i].card_vocab_idx) / static_cast<float>(N_CARD_TYPES)
            : id_null;
        ctrl[i] = (q->choices[i].zone_ref != REF_NONE)
            ? (q->choices[i].controller_is_self ? 1.0f : 0.0f)
            : ctrl_null;
        pub[i]  = q->choices[i].card_is_public ? 1.0f : 0.0f;
        zone[i] = static_cast<int32_t>(q->choices[i].zone_ref);
        refs[i] = static_cast<int32_t>(q->choices[i].slot_ref);
    }

    fwrite(cats, sizeof(int32_t), MAX_ACTIONS, stdout);
    fwrite(ids,  sizeof(float),   MAX_ACTIONS, stdout);
    fwrite(ctrl, sizeof(float),   MAX_ACTIONS, stdout);
    fwrite(pub,  sizeof(float),   MAX_ACTIONS, stdout);
    fwrite(zone, sizeof(int32_t), MAX_ACTIONS, stdout);
    fwrite(refs, sizeof(int32_t), MAX_ACTIONS, stdout);

    // Human-readable per-action descriptions (e.g. "Target Player B (20 life)",
    // "Pay 4 life", "Put on top"), only under --narrative so the ML training path
    // stays binary-only and OBS-unaffected. Observers/test harness read this
    // block to label choices the (category, card_id, controller) metadata alone
    // can't — player targets, Sylvan Library pay-vs-return, charm modes, etc.
    // Fixed-size [MAX_ACTIONS][MAX_CHOICE_DESC] for simple framing; NUL-padded.
    if (narrative_mode) {
        char descs[MAX_ACTIONS][MAX_CHOICE_DESC] = {};
        for (int i = 0; i < q->num_choices; i++)
            snprintf(descs[i], MAX_CHOICE_DESC, "%s", q->choices[i].description);
        fwrite(descs, MAX_CHOICE_DESC, MAX_ACTIONS, stdout);

        // Per-permanent typed-counter summaries ("+1/+1:2, time:3, loyalty:4"),
        // narrative-only like the descriptions above — display info for observers
        // (TUI/harness board labels); the ML training path stays binary-only and
        // OBS-unaffected. Slot order matches the state vector's permanent blocks
        // (48 self slots, then 48 opp slots), so slot i's summary labels the
        // permanent serialized in slot i. Fixed-size NUL-padded char block.
        char ctrs[2 * MAX_BATTLEFIELD_SLOTS][PERM_COUNTERS_LEN] = {};
        for (int i = 0; i < MAX_BATTLEFIELD_SLOTS; i++) {
            snprintf(ctrs[i], PERM_COUNTERS_LEN, "%s", gs->self_permanents[i].counters);
            snprintf(ctrs[MAX_BATTLEFIELD_SLOTS + i], PERM_COUNTERS_LEN, "%s",
                     gs->opp_permanents[i].counters);
        }
        fwrite(ctrs, PERM_COUNTERS_LEN, 2 * MAX_BATTLEFIELD_SLOTS, stdout);

        // Per-permanent token names — the state vector only carries the generic
        // TOKEN_SENTINEL card id for every token, so observers can't tell a Clue
        // from a Human Knight without this. Narrative-only side-channel, slot-aligned
        // with the permanent blocks like the counters above; empty for non-tokens.
        char toks[2 * MAX_BATTLEFIELD_SLOTS][PERM_TOKEN_NAME_LEN] = {};
        for (int i = 0; i < MAX_BATTLEFIELD_SLOTS; i++) {
            snprintf(toks[i], PERM_TOKEN_NAME_LEN, "%s", gs->self_permanents[i].token_name);
            snprintf(toks[MAX_BATTLEFIELD_SLOTS + i], PERM_TOKEN_NAME_LEN, "%s",
                     gs->opp_permanents[i].token_name);
        }
        fwrite(toks, PERM_TOKEN_NAME_LEN, 2 * MAX_BATTLEFIELD_SLOTS, stdout);
    }
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
    printf("  --deck <name>       Load named deck for both players (default: test_minimal)\n");
    printf("  --deck-a <name>     Load named deck for player A\n");
    printf("  --deck-b <name>     Load named deck for player B\n");
    printf("  --player <A|B>      Designate a human player (A or B)\n");
    printf("  --replay <logfile>  Replay a previously logged game (self-contained: seed,\n");
    printf("                      flags and decks come from the log's RMLOG v2 header)\n");
    printf("  --machine           Machine mode: emit BQUERY lines for AI input\n");
    printf("  --log-decisions     Write the decision log in machine mode (off by default\n");
    printf("                      there; CLI/interactive games always log)\n");
    printf("  --bo3               Best-of-three match mode\n");
    printf("  --help, -h          Show this help message\n");
    printf("\nDeck names are filenames without .dk, relative to resources/decks/.\n");
}

void cli_print_seed(unsigned int seed) {
    game_log("Using seed: %u\n", seed);
}

// Header for the pre-game decisions (mulligans, CR 103.6b opening-hand actions) so they
// don't read as part of a numbered turn.
void cli_print_pregame_header() {
    game_log("\n-------- PREGAME --------\n");
}

void cli_print_turn_header(size_t turn, bool player_a_turn) {
    const char* name = player_a_turn ? "Player A" : "Player B";
    // Game::turn counts from 0; display 1-based so the first header reads "TURN 1"
    // (never "TURN 0"). Display-only — the 0-based counter itself is untouched.
    game_log("\n-------- TURN %zu (%s) --------\n", turn + 1, name);
}

void cli_print_invalid_action() {
    game_log("Invalid action\n");
}

// ── Game state display ────────────────────────────────────────────────────────

static void format_player_resources(const PlayerState& ps, char* buf, size_t buf_len) {
    buf[0] = '\0';
    size_t off = 0;
    const char* sep = " ";
    if (ps.poison_counters > 0) {
        off += snprintf(buf + off, buf_len - off, "%spoison %d", sep, ps.poison_counters);
        sep = ", ";
    }
    if (ps.energy > 0) {
        off += snprintf(buf + off, buf_len - off, "%senergy %d", sep, ps.energy);
        sep = ", ";
    }
}

void print_game_state(const GameState* gs) {
    if (InputLogger::instance().is_machine_mode()) return;

    const char* self_name   = gs->self_is_player_a ? "Player A" : "Player B";
    const char* opp_name    = gs->self_is_player_a ? "Player B" : "Player A";
    const char* active_name = gs->is_active_player  ? self_name : opp_name;

    game_log("\n--- Turn %d - %s's %s ---\n", gs->turn, active_name, step_to_string(gs->cur_step));
    game_log("Life: Player A=%d, Player B=%d\n",
             gs->self_is_player_a ? gs->self.life : gs->opponent.life,
             gs->self_is_player_a ? gs->opponent.life : gs->self.life);

    // Non-mana resources (poison, energy) shown only when a player actually has some.
    const PlayerState& player_a = gs->self_is_player_a ? gs->self : gs->opponent;
    const PlayerState& player_b = gs->self_is_player_a ? gs->opponent : gs->self;
    char res_buf[64];
    format_player_resources(player_a, res_buf, sizeof(res_buf));
    if (res_buf[0]) game_log("  Player A:%s\n", res_buf);
    format_player_resources(player_b, res_buf, sizeof(res_buf));
    if (res_buf[0]) game_log("  Player B:%s\n", res_buf);

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
                len += snprintf(line + len, (size_t)((int)sizeof(line) - len), " (%d damage)", slot->damage);
            if (slot->counters[0] != '\0' && len < (int)sizeof(line))
                snprintf(line + len, (size_t)((int)sizeof(line) - len), " {%s}", slot->counters);
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
                    len += snprintf(line + len, (size_t)((int)sizeof(line) - len), " (%s)",
                                    mana_symbol_str((Colors)i));
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
    if (InputLogger::instance().is_machine_mode()) return;
    const char* pname = player_a_has_priority ? "Player A" : "Player B";
    game_log("\n%s has priority. Legal actions:\n", pname);
    for (int i = 0; i < q->num_choices; i++) {
        game_log("  %d: %s\n", i, q->choices[i].description);
    }
}

// ── Errors ────────────────────────────────────────────────────────────────────

void cli_warning(const std::string& msg) {
    fprintf(stderr, "WARNING: %s\n", msg.c_str());
}

void cli_error(const std::string& msg) {
    fprintf(stderr, "ERROR: %s\n", msg.c_str());
}

[[noreturn]] void cli_fatal_error(const std::string& msg) {
    fprintf(stderr, "FATAL: %s\n", msg.c_str());
    exit(1);
}
