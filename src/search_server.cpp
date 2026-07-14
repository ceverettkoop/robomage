#include "search_server.h"

#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "classes/game.h"
#include "components/player.h"
#include "ecs/coordinator.h"
#include "error.h"
#include "mana_system.h"
#include "snapshot.h"
#include "stable_rng.h"

extern Game cur_game;
extern Coordinator global_coordinator;
extern int match_game_number;  // -1 = single game, 0-2 = bo3 game index (main.cpp)
extern bool sideboard_phase;   // true while a between-game sideboard prompt is live

bool search_server_mode = false;
unsigned int g_sim_seed_salt = 0;

static bool any_slot_live();
static bool read_command_line(char *buf, size_t len);
static int parse_slot_arg(const char *line, const char *cmd);

static bool g_loop_safe = false;
static int g_pending_restore_slot = -1;
static int g_unwind_count = 0;
// Unset by default; only the Phase D in-process actor installs it (see header).
static std::function<bool(int)> g_game_end_hook;
// A runaway unwind (a menu family that never terminates under repeated
// first-choice selection) would silently auto-play forever; cap it loudly.
static const int UNWIND_CAP = 10000;

void search_set_loop_safe(bool safe) { g_loop_safe = safe; }
bool search_loop_safe() { return g_loop_safe; }

bool search_restore_pending() { return g_pending_restore_slot >= 0; }

void search_request_restore(int slot) {
    if (slot < 0 || slot >= N_SNAPSHOT_SLOTS || !snapshot_slot_live(slot)) {
        fatal_error("search-server: RESTORE from empty snapshot slot " + std::to_string(slot));
    }
    g_pending_restore_slot = slot;
}

void search_set_game_end_hook(std::function<bool(int)> hook) { g_game_end_hook = std::move(hook); }
void search_clear_game_end_hook() { g_game_end_hook = nullptr; }

int search_unwind_choice() {
    if (++g_unwind_count > UNWIND_CAP) {
        fatal_error("search-server: cooperative unwind exceeded " +
                    std::to_string(UNWIND_CAP) + " decisions without reaching the main loop");
    }
    return 0;
}

void search_apply_pending_restore() {
    if (g_pending_restore_slot < 0) return;
    // A MATCH-scoped restore targets a sideboard root one dispatcher level up;
    // leave it latched for search_apply_pending_match_restore() to apply there.
    if (snapshot_slot_is_match(g_pending_restore_slot)) return;
    int slot = g_pending_restore_slot;
    g_pending_restore_slot = -1;
    g_unwind_count = 0;
    if (!snapshot_restore(slot)) {
        fatal_error("search-server: RESTORE from empty snapshot slot " + std::to_string(slot));
    }
}

bool search_match_restore_pending() {
    return g_pending_restore_slot >= 0 && snapshot_slot_is_match(g_pending_restore_slot);
}

void search_apply_pending_match_restore() {
    if (!search_match_restore_pending()) return;
    int slot = g_pending_restore_slot;
    g_pending_restore_slot = -1;
    g_unwind_count = 0;
    if (!snapshot_restore(slot)) {
        fatal_error("search-server: RESTORE from empty snapshot slot " + std::to_string(slot));
    }
}

static bool any_slot_live() {
    for (int i = 0; i < N_SNAPSHOT_SLOTS; i++) {
        if (snapshot_slot_live(i)) return true;
    }
    return false;
}

// Read one stdin line for the search protocol; EOF here is always fatal (the
// driver owns the process and must not vanish mid-search).
static bool read_command_line(char *buf, size_t len) {
    if (!fgets(buf, static_cast<int>(len), stdin)) {
        fatal_error("search-server: stdin closed while waiting for a command");
        return false;
    }
    if (!strchr(buf, '\n')) {
        int c;
        while ((c = getchar()) != '\n' && c != EOF);
    }
    return true;
}

// Parse "<cmd> <n>" and validate the slot range; fatal on a malformed line.
static int parse_slot_arg(const char *line, const char *cmd) {
    long slot = -1;
    if (sscanf(line, "%*s %ld", &slot) != 1 || slot < 0 || slot >= N_SNAPSHOT_SLOTS) {
        fatal_error("search-server: " + std::string(cmd) + " needs a slot in [0, " +
                    std::to_string(N_SNAPSHOT_SLOTS - 1) + "], got: " + line);
    }
    return static_cast<int>(slot);
}

bool search_handle_command(const char *line) {
    char tok[32] = {};
    if (sscanf(line, "%31s", tok) != 1) {
        fatal_error("search-server: empty command line");
    }
    if (strcmp(tok, "SNAPSHOT") == 0) {
        if (!g_loop_safe) fatal_error("search-server: SNAPSHOT at a non-loop-safe decision");
        int slot = parse_slot_arg(line, "SNAPSHOT");
        snapshot_save(slot);
        printf("SNAPSHOT_OK %d\n", slot);
        fflush(stdout);
        return true;
    }
    if (strcmp(tok, "RESTORE") == 0) {
        int slot = parse_slot_arg(line, "RESTORE");
        if (!snapshot_slot_live(slot)) {
            fatal_error("search-server: RESTORE from empty snapshot slot " + std::to_string(slot));
        }
        g_pending_restore_slot = slot;
        return true;
    }
    if (strcmp(tok, "DETERMINIZE") == 0) {
        if (!g_loop_safe) fatal_error("search-server: DETERMINIZE at a non-loop-safe decision");
        unsigned long seed = 0;
        if (sscanf(line, "%*s %lu", &seed) != 1) {
            fatal_error("search-server: DETERMINIZE needs a seed, got: " + std::string(line));
        }
        determinize_hidden_state(static_cast<unsigned int>(seed));
        printf("DETERMINIZE_OK\n");
        fflush(stdout);
        return true;
    }
    if (strcmp(tok, "RELEASE") == 0) {
        snapshot_release_all();
        printf("RELEASE_OK\n");
        fflush(stdout);
        return true;
    }
    fatal_error("search-server: unknown command '" + std::string(tok) + "'");
    return false;
}

bool search_intercept_game_end() {
    // Game ended while auto-playing toward a pending restore: apply it here --
    // we are at loop level with no simulation frames below, so a direct restore
    // is safe -- and keep the loop running. No SIM_RESULT: the driver already
    // sent RESTORE and is waiting for the restored decision's query.
    if (search_restore_pending()) {
        search_apply_pending_restore();
        return true;
    }
    // In-process actor hook (Phase D): a simulated line reached game over. When
    // the hook is installed and a snapshot is live to roll back to, hand the
    // verdict to it instead of the stdio SIM_RESULT/stdin protocol below. The
    // hook records the result and latches a restore via search_request_restore();
    // returning true keeps the loop running so the pending restore unwinds at the
    // loop top (exactly as a stdio RESTORE would). With no snapshot live, don't
    // call it — fall through to the existing behavior. Unset in bin/robomage, so
    // the stdio branch stays reachable and byte-identical.
    if (g_game_end_hook && any_slot_live()) {
        return g_game_end_hook(cur_game.winner);
    }
    if (!search_server_mode || !any_slot_live()) return false;

    // A simulated line reached game over. Announce it and hold the process for
    // the driver's verdict; only RESTORE (continue searching) or RELEASE (let
    // the real game end) are meaningful here.
    const char *result = (cur_game.winner == Zone::PLAYER_A)   ? "A"
                         : (cur_game.winner == Zone::PLAYER_B) ? "B"
                                                               : "DRAW";
    printf("SIM_RESULT: %s\n", result);
    fflush(stdout);
    while (true) {
        char line[128];
        read_command_line(line, sizeof(line));
        char tok[32] = {};
        if (sscanf(line, "%31s", tok) != 1) continue;
        if (strcmp(tok, "RESTORE") == 0) {
            int slot = parse_slot_arg(line, "RESTORE");
            if (!snapshot_slot_live(slot)) {
                fatal_error("search-server: RESTORE from empty snapshot slot " +
                            std::to_string(slot));
            }
            // A GAME-scoped restore is safe to apply in place here (loop level, no
            // simulation frames below). A MATCH-scoped one rewinds the between-game
            // MatchContext/stage that only play_bo3_match's dispatcher owns: latch
            // it and return true so the unwind exits play_single_game via its
            // loop-top match check and the dispatcher applies the rollback.
            if (snapshot_slot_is_match(slot)) {
                g_pending_restore_slot = slot;
                g_unwind_count = 0;
                return true;
            }
            if (!snapshot_restore(slot)) {
                fatal_error("search-server: RESTORE from empty snapshot slot " +
                            std::to_string(slot));
            }
            g_unwind_count = 0;
            return true;
        }
        if (strcmp(tok, "RELEASE") == 0) {
            snapshot_release_all();
            printf("RELEASE_OK\n");
            fflush(stdout);
            return false;
        }
        fatal_error("search-server: expected RESTORE or RELEASE after SIM_RESULT, got '" +
                    std::string(tok) + "'");
    }
}

void determinize_hidden_state(unsigned int world_seed) {
    // At a sideboard root the only hidden information is which next-game deal this
    // world gets: both 75s are public, and the opponent's in/out choices are
    // interior tree nodes. The just-ended game's ECS is irrelevant to the future,
    // so skip all zone mixing and simply record this world's seed salt — the next
    // game's per-game seed (match_game_seed in game_driver.cpp) folds it in, so
    // each world samples a distinct shuffle/deal. Gate on the sideboard_phase
    // global, NOT match_game_number: at a game-1->2 sideboard root that index is
    // still the ENDED game's (0). g_sim_seed_salt is captured (as 0) at the real
    // sideboard prompt and restored on every RESTORE, so each DETERMINIZE sets a
    // fresh salt from a clean baseline and the final restore returns the real line
    // to salt 0.
    if (sideboard_phase) {
        g_sim_seed_salt = world_seed;
        return;
    }
    std::mt19937 world_gen(world_seed);
    Zone::Ownership p = cur_game.player_a_has_priority ? Zone::PLAYER_A : Zone::PLAYER_B;
    Zone::Ownership opp = (p == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;
    const int *known_a = cur_game.known_top_library_a;
    const int *known_b = cur_game.known_top_library_b;

    // A library position is pinned when its occupant's identity is tracked in
    // the owner's known-top array: the priority player's pins are cards they
    // genuinely know; pinning the opponent's known-top as well keeps the
    // opponent's own belief state (their Brainstorm knowledge) internally
    // consistent at opponent nodes inside the sampled world -- a mild
    // more-informed-than-P approximation, accepted for v1.
    std::vector<Entity> own_pool;          // priority player's unpinned library cards
    std::vector<size_t> own_positions;     // the positions they vacate
    std::vector<Entity> opp_pool;          // opp unknown hand cards, then unpinned library cards
    std::vector<size_t> opp_lib_positions; // vacated opp library positions
    size_t opp_hand_slots = 0;
    size_t opp_lib_slots = 0;

    // Post-board games of a bo3 (game index 1+, matching machine_io's
    // is_post_board): the opponent's sideboard SWAPS are hidden information, so
    // P only knows the combined 75 — which 60 are in the deck is unknowable.
    // Model that by adding the opponent's SIDEBOARD cards to the exchange pool:
    // every sampled world re-deals which cards are in hand/library and which
    // sat out in the sideboard. Game 1 (and single games) keep the sideboard
    // out of the pool — the pre-board 60 is exactly the known decklist. A
    // declared companion is public (CR 702.139 reveal) and stays pinned.
    bool opp_sideboard_hidden = match_game_number > 0;
    Entity opp_companion = 0;
    if (opp_sideboard_hidden) {
        Entity opp_pe = get_player_entity(opp);
        if (global_coordinator.entity_has_component<Player>(opp_pe))
            opp_companion = global_coordinator.GetComponent<Player>(opp_pe).chosen_companion;
    }

    Entity max_e = global_coordinator.GetMaxIssuedEntity();
    for (Entity e = 0; e < max_e; e++) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location == Zone::LIBRARY && (z.owner == p || z.owner == opp)) {
            const int *known = (z.owner == Zone::PLAYER_A) ? known_a : known_b;
            bool pinned = z.distance_from_top < KNOWN_TOP_LIBRARY_SIZE &&
                          known[z.distance_from_top] != -1;
            if (pinned) continue;
            if (z.owner == p) {
                own_pool.push_back(e);
                own_positions.push_back(z.distance_from_top);
            } else {
                opp_pool.push_back(e);
                opp_lib_positions.push_back(z.distance_from_top);
                opp_lib_slots++;
            }
        } else if (z.location == Zone::HAND && z.owner == opp && !z.identity_known) {
            // Revealed hand cards (Duress, revealed tutor targets) stay put;
            // only the genuinely unknown ones join the exchange pool.
            opp_pool.push_back(e);
            opp_hand_slots++;
        } else if (opp_sideboard_hidden && z.location == Zone::SIDEBOARD &&
                   z.owner == opp && e != opp_companion) {
            opp_pool.push_back(e);
        }
    }

    // Priority player's own library: permute the unpinned cards among the
    // positions they already occupy (P cannot know this order).
    stable_shuffle(own_pool, world_gen);
    for (size_t i = 0; i < own_pool.size(); i++) {
        global_coordinator.GetComponent<Zone>(own_pool[i]).distance_from_top = own_positions[i];
    }

    // Opponent's unknown cards: hand, library and (post-board) sideboard are
    // exchangeable from P's perspective, so shuffle the combined pool and deal
    // it back -- the first opp_hand_slots entries become the hand, the next
    // opp_lib_slots fill the vacated library positions, and any remainder (the
    // sideboard cards' slots) returns to the sideboard. Zone sizes are
    // preserved exactly. Direct Zone writes, no orderer add_to_zone: this is a
    // belief-state resample, not a game event, and must fire no triggers.
    stable_shuffle(opp_pool, world_gen);
    size_t li = 0;
    size_t sb = 0;
    for (size_t i = 0; i < opp_pool.size(); i++) {
        auto &z = global_coordinator.GetComponent<Zone>(opp_pool[i]);
        if (i < opp_hand_slots) {
            z.location = Zone::HAND;
            z.distance_from_top = 0;  // not meaningful in hand
        } else if (li < opp_lib_slots) {
            z.location = Zone::LIBRARY;
            z.distance_from_top = opp_lib_positions[li++];
        } else {
            z.location = Zone::SIDEBOARD;
            z.distance_from_top = sb++;  // stable, arbitrary sideboard order
        }
        z.identity_known = false;
    }
}
