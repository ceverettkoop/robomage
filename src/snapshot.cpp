#include "snapshot.h"

#include <cstdint>
#include <string>
#include <unordered_map>

#include "card_db.h"
#include "classes/deck_state.h"
#include "classes/game.h"
#include "classes/match_context.h"
#include "classes/match_state.h"
#include "components/player.h"
#include "components/zone.h"
#include "ecs/coordinator.h"
#include "ecs/entity.h"
#include "game_driver.h"
#include "search_server.h"

extern Game cur_game;
extern Coordinator global_coordinator;
// card_db is declared in card_db.h. The match globals (g_match_ctx,
// g_card_db_generation, match_game_number/wins/sideboard_phase*) live in
// game_driver.{h,cpp}; g_sim_seed_salt in search_server.{h,cpp}.

namespace {

enum class SnapScope { GAME, MATCH };

// One saved game state. Holds a deep copy of every mutable per-game store: the
// value-typed Game, the ECS (via EcsSnapshot), the card name->entity map, and the
// match-scoped revealed arrays. card_db MUST be part of the snapshot: a card first
// loaded during a simulated line adds a prototype entity to the ECS and a
// name->entity mapping to card_db, so restoring the ECS without also restoring
// card_db would leave that mapping pointing at an entity that no longer exists —
// a later load_card/GetData would then abort.
struct GameSnapshot {
    bool live = false;
    SnapScope scope = SnapScope::GAME;
    Game game{};
    EcsSnapshot ecs;
    std::unordered_map<std::string, Entity> card_db_copy;
    // Bumped every init_ecs(); a mismatch on restore forces the card_db copy even
    // when sizes coincidentally match (see snapshot_restore).
    uint64_t card_db_generation = 0;
    unsigned char revealed_a[REVEALED_SIZE] = {};
    unsigned char revealed_b[REVEALED_SIZE] = {};

    // ── Match section (captured whenever a match is active; restored wholesale).
    // These are the between-game stores that survive an init_ecs() teardown, so a
    // MATCH-scoped (sideboard-root) snapshot can roll forward into the next game
    // and restore the sideboard-node state exactly. Harmless to carry for in-game
    // bo3 snapshots too — they restore to the identical live values.
    bool match_captured = false;
    MatchContext match_ctx;
    DeckStateSnapshot deck_state;
    int match_game_number = -1;
    int match_wins_a = 0;
    int match_wins_b = 0;
    bool sideboard_phase = false;
    Zone::Ownership sideboard_phase_player = Zone::UNKNOWN;
    // Placeholder captured/restored now, consumed in Stage 3 (determinization
    // seed salt); always 0 until then so the plumbing is complete and inert.
    unsigned int sim_seed_salt = 0;
};

GameSnapshot g_slots[N_SNAPSHOT_SLOTS];

// See snapshot_set_post_restore_hook (snapshot.h): re-derives the rule-613
// derived state after every restore, before a resumed suspended flow reads it.
void (*g_post_restore_hook)() = nullptr;

static bool valid_slot(int slot);

static bool valid_slot(int slot) { return slot >= 0 && slot < N_SNAPSHOT_SLOTS; }

}  // namespace

bool snapshot_save(int slot) {
    if (!valid_slot(slot)) return false;
    GameSnapshot &s = g_slots[slot];
    // A snapshot taken at a sideboard prompt is a valid MCTS search root only if
    // it captures the match level (it must roll forward across init_ecs()); mark
    // it MATCH so restore/release treat it as such.
    s.scope = sideboard_phase ? SnapScope::MATCH : SnapScope::GAME;
    s.game = cur_game;
    global_coordinator.snapshot_to(s.ecs);
    s.card_db_copy = card_db;
    s.card_db_generation = g_card_db_generation;
    for (int i = 0; i < REVEALED_SIZE; ++i) {
        s.revealed_a[i] = g_revealed_by_a[i];
        s.revealed_b[i] = g_revealed_by_b[i];
    }
    // Match section — captured whenever a match is in progress (see struct note).
    s.match_captured = g_match_ctx.active;
    if (s.match_captured) s.match_ctx = g_match_ctx;
    deck_state_capture(s.deck_state);
    s.match_game_number = match_game_number;
    s.match_wins_a = match_wins_a;
    s.match_wins_b = match_wins_b;
    s.sideboard_phase = sideboard_phase;
    s.sideboard_phase_player = sideboard_phase_player;
    s.sim_seed_salt = g_sim_seed_salt;
    s.live = true;
#ifndef NDEBUG
    // Counterpart of the [restore] trace: every save, so an overwritten search
    // slot (a save between a root's sims) is visible in the log.
    fprintf(stderr, "[save] slot=%d scope=%s step=%d lifeA=%d lifeB=%d mir=%u/%u\n",
            slot, s.scope == SnapScope::MATCH ? "MATCH" : "GAME",
            static_cast<int>(cur_game.cur_step),
            global_coordinator.entity_has_component<Player>(cur_game.player_a_entity)
                ? global_coordinator.GetComponent<Player>(cur_game.player_a_entity).life_total : -99,
            global_coordinator.entity_has_component<Player>(cur_game.player_b_entity)
                ? global_coordinator.GetComponent<Player>(cur_game.player_b_entity).life_total : -99,
            cur_game.miracle_reveal_pending, cur_game.miracle_cast_pending);
#endif
    return true;
}

bool snapshot_restore(int slot) {
    if (!valid_slot(slot)) return false;
    GameSnapshot &s = g_slots[slot];
    if (!s.live) return false;
    cur_game = s.game;
    global_coordinator.restore_from(s.ecs);
    // card_db is add-only WITHIN a game (load_card only inserts), so within one
    // generation a size match means nothing was loaded since the save and the
    // maps are identical — skip the copy (the in-game hot path). But an init_ecs()
    // between save and restore does card_db.clear()+reload: sizes can coincide
    // with different contents, so a generation change forces the copy.
    if (g_card_db_generation != s.card_db_generation ||
        card_db.size() != s.card_db_copy.size()) {
        card_db = s.card_db_copy;
    }
    g_card_db_generation = s.card_db_generation;
    for (int i = 0; i < REVEALED_SIZE; ++i) {
        g_revealed_by_a[i] = s.revealed_a[i];
        g_revealed_by_b[i] = s.revealed_b[i];
    }
    // Match section — restored wholesale (the between-game stores that outlive the
    // ECS). For a GAME-scoped in-match snapshot these equal the live values.
    if (s.match_captured) g_match_ctx = s.match_ctx;
    deck_state_restore(s.deck_state);
    match_game_number = s.match_game_number;
    match_wins_a = s.match_wins_a;
    match_wins_b = s.match_wins_b;
    sideboard_phase = s.sideboard_phase;
    sideboard_phase_player = s.sideboard_phase_player;
    g_sim_seed_salt = s.sim_seed_salt;
    if (g_post_restore_hook) g_post_restore_hook();
#ifndef NDEBUG
    // Debug trace for the az_mcts world-consistency hunt: what suspended-flow
    // state each restore hands back to the loop, plus enough authoritative
    // fields (lives, step, miracle flags) to see whether the slot's CONTENT
    // changed between two restores of the same search (an overwritten slot).
    fprintf(stderr,
            "[restore] slot=%d pq(tag=%d act=%d ans=%d) pc=%d pa=%d pd=%d res=%d "
            "step=%d lifeA=%d lifeB=%d mir=%u/%u\n",
            slot, static_cast<int>(cur_game.pending_query.tag),
            cur_game.pending_query.active ? 1 : 0,
            cur_game.pending_query.answered ? 1 : 0,
            cur_game.pending_cast.active ? 1 : 0,
            cur_game.pending_activation.active ? 1 : 0,
            cur_game.pending_draw.active ? 1 : 0,
            cur_game.resolution.active ? 1 : 0,
            static_cast<int>(cur_game.cur_step),
            global_coordinator.entity_has_component<Player>(cur_game.player_a_entity)
                ? global_coordinator.GetComponent<Player>(cur_game.player_a_entity).life_total : -99,
            global_coordinator.entity_has_component<Player>(cur_game.player_b_entity)
                ? global_coordinator.GetComponent<Player>(cur_game.player_b_entity).life_total : -99,
            cur_game.miracle_reveal_pending, cur_game.miracle_cast_pending);
#endif
    return true;
}

void snapshot_set_post_restore_hook(void (*hook)()) { g_post_restore_hook = hook; }

bool snapshot_slot_live(int slot) { return valid_slot(slot) && g_slots[slot].live; }

bool snapshot_slot_is_match(int slot) {
    return valid_slot(slot) && g_slots[slot].live && g_slots[slot].scope == SnapScope::MATCH;
}

bool snapshot_any_match_scope_live() {
    for (int i = 0; i < N_SNAPSHOT_SLOTS; ++i) {
        if (g_slots[i].live && g_slots[i].scope == SnapScope::MATCH) return true;
    }
    return false;
}

void snapshot_release_all() {
    for (int i = 0; i < N_SNAPSHOT_SLOTS; ++i) {
        g_slots[i] = GameSnapshot{};  // drops the held copies and clears the live flag
    }
}
